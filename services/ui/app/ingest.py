"""Resolve, confirm, fetch -- and never in a different order.

    POST /ui/resolve   guard -> MeTube /add auto_start:false -> probe
    POST /ui/commit    MeTube /start (or re-add, when a clip range was chosen)
    POST /ui/abandon   MeTube /delete from BOTH queues, then verify
    GET  /ui/progress  MeTube /history, as percent / speed / eta
    POST /ui/fetch     MeTube's file -> multipart -> gateway -> stt-stack

NOTHING IS DOWNLOADED BEFORE THE USER SAYS SO, which is the requirement the
user pressed hardest on. `auto_start:false` on /add resolves the URL and parks
it; only /start moves bytes; /delete abandons it. All three verified against
MeTube's source and live against the running instance.

WHY /add COMES BEFORE THE PROBE, when probing first would be tidier. Because
POST /add is where MeTube's url_guard runs, and that guard is better than ours:
ingress validation plus a connect-time getaddrinfo hook inside the download
subprocess, so it also covers redirects and DNS rebinding. Putting our
subprocess first would make our own forty-line pre-filter the only thing that
had run before a URL was handed to yt-dlp.

THE COST OF THAT ORDERING, NAMED RATHER THAN DISCOVERED LATER. Every link the
user declines has already left a pending record in MeTube. So /ui/abandon is
not a nicety, it is the other half of /ui/resolve, and it is tested -- see
tests/test_ingest.py::test_abandon_reaps_a_declined_link. MeTube's /delete
answers {"status":"ok"} when it deletes nothing at all, so abandoning re-reads
/history and reports whether the record actually went.

THE ESTIMATES ARE NOT COMPUTED HERE, deliberately. This module returns FACTS --
title, duration, the size of the audio-only stream, is_live, whether real
subtitles exist -- and the page does the arithmetic with the realtime factor it
has measured on this box. Two reasons. The rate is a moving number the browser
keeps an EMA of from every transcription it runs, and a server-side estimate
would be a second, staler copy of it. And the download half and the transcribe
half are NEVER blended into one figure: for long media the download is the slow
half, and one merged number hides which half to blame when it drags.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from voice_common.errors import error_response

from . import config, guard, metube, probe

log = logging.getLogger("voice-ui.ingest")

# What /v1/audio/transcriptions accepts. Kept here rather than imported from
# the gateway because this service must not depend on that one's internals,
# and kept as a set rather than a regex because the whole vocabulary is five
# words -- see the interpolation guard in fetch() for why it is checked at all.
RESPONSE_FORMATS = frozenset({"json", "text", "srt", "vtt", "verbose_json"})

router = APIRouter()

# Per-caller budget for /ui/resolve. That route spawns a process which makes an
# outbound request to a host the caller chose; unmetered, it is a port and host
# scanner with a nice JSON interface. Keyed on the presented API key, falling
# back to the peer address when authentication is off.
_recent: dict[str, list[float]] = {}


def _rate_limited(key: str) -> bool:
    now = time.monotonic()
    window = [t for t in _recent.get(key, ()) if now - t < 60.0]
    if len(window) >= config.RESOLVE_PER_MINUTE:
        _recent[key] = window
        return True
    window.append(now)
    _recent[key] = window
    if len(_recent) > 512:
        # Unbounded growth is the only way this dict misbehaves. Drop the
        # entries whose window is empty rather than keeping a second structure.
        for name in [k for k, v in _recent.items()
                     if not v or now - v[-1] > 120.0]:
            _recent.pop(name, None)
    return False


class ResolveRequest(BaseModel):
    url: str


class CommitRequest(BaseModel):
    token: str
    # Promoted out of any expert panel and onto the confirm card itself,
    # because for long media this is what turns "no, too much" into "yes, but
    # only this bit" -- and for a live stream, which has no end, it is the only
    # answer there is.
    clip_start: float | None = Field(default=None, ge=0)
    clip_end: float | None = Field(default=None, ge=0)
    # A video with real (non-ASR) captions already has a human transcript.
    # MeTube's captions type sets skip_download and returns it in about two
    # seconds for near-zero cost, which beats transcribing it however fast
    # Parakeet is.
    captions: bool = False


class TokenRequest(BaseModel):
    token: str


def _client(request: Request) -> metube.MeTube:
    return metube.MeTube(request.app.state.client)


def _unavailable(exc: metube.MeTubeError) -> Response:
    """Render a MeTube failure as the thing that actually went wrong.

    A refusal is the caller's link; anything else is our dependency. Getting
    this backwards sent someone to debug a healthy MeTube because their own
    URL was rejected.
    """
    if isinstance(exc, metube.MeTubeRefused):
        return error_response(400, str(exc), code="refused_url", param="url")
    return error_response(502, str(exc), type_="server_error",
                          code="ingestion_unavailable")


@router.post("/ui/resolve")
async def resolve(request: Request, body: ResolveRequest) -> Response:
    if not metube.configured():
        return error_response(
            501,
            "Link ingestion is not configured: set UI_METUBE_URL to the "
            "MeTube instance on this host. File upload still works.",
            code="ingestion_not_configured")

    who = request.headers.get("authorization") or (
        request.client.host if request.client else "-")
    if _rate_limited(who):
        return error_response(
            429, f"Too many links resolved; the limit is "
            f"{config.RESOLVE_PER_MINUTE} a minute.",
            code="rate_limited", headers={"Retry-After": "30"})

    # Layer one. Our own stdlib pre-filter, so a URL we hate never becomes a
    # log line in MeTube either. See app/guard.py for what is blocked and why.
    try:
        url = guard.check(body.url)
    except guard.GuardError as exc:
        return error_response(400, str(exc), code="refused_url", param="url")

    # Layer two, and the one that decides. MeTube's url_guard runs inside this
    # call; if it refuses we return its message verbatim and stop.
    client = _client(request)
    try:
        await client.add(url, auto_start=False)
    except metube.MeTubeError as exc:
        return _unavailable(exc)

    try:
        found = await client.find(url)
    except metube.MeTubeError as exc:
        return _unavailable(exc)
    if found is None:
        return error_response(
            502, "MeTube accepted the link and then had no record of it.",
            type_="server_error", code="ingestion_unavailable")

    where, entry = found
    if where == "done" and entry.get("status") != metube.FINISHED:
        # A rejected add still creates a record, in `done`, with the reason in
        # `msg` or `error`. Surface it and clear it rather than leaving a dead
        # row behind.
        await client.abandon(url)
        return error_response(
            400, str(entry.get("error") or entry.get("msg")
                     or "MeTube could not resolve that link."),
            code="refused_url", param="url")

    # Layer three, and only now: a URL both guards have already accepted.
    facts = await probe.run(url) if probe.available() else None

    title = (facts or {}).get("title") or entry.get("title") or url
    duration = (facts or {}).get("duration")
    size = (facts or {}).get("bytes")
    live = bool((facts or {}).get("is_live")) or entry.get("live_status") == "is_live"

    # WHEN TO NAG. Below both thresholds and not live, the page skips the
    # dialog entirely -- see config.CONFIRM_SECONDS for the defence of the
    # numbers. An unknown duration always confirms: not knowing is exactly the
    # case the dialog exists for.
    confirm = (live or duration is None
               or duration > config.CONFIRM_SECONDS
               or (size or 0) > config.CONFIRM_BYTES)

    return Response(media_type="application/json", content=_json({
        "token": url,
        "title": title,
        "uploader": (facts or {}).get("uploader"),
        "duration": duration,
        "bytes": size,
        "is_live": live,
        "has_subtitles": bool((facts or {}).get("has_subtitles")),
        "probed": facts is not None,
        "confirm": confirm,
        # So the card can say "duration unknown -- yt-dlp is not installed in
        # this image" rather than silently showing less than it should.
        "probe_enabled": probe.available(),
    }))


@router.post("/ui/commit")
async def commit(request: Request, body: CommitRequest) -> Response:
    client = _client(request)
    try:
        url = guard.check(body.token)
    except guard.GuardError as exc:
        return error_response(400, str(exc), code="refused_url", param="token")

    # THE GATE IS ENFORCED HERE, NOT ONLY IN THE PAGE. Without this, POST
    # /ui/commit succeeded on a token that had never been resolved, and with
    # clip_start or captions set it went straight to /add {auto_start:true} --
    # a download starting with no resolve step and no dialog. The page never
    # takes that path, and the caller is already authenticated, so it was a UX
    # gate rather than a boundary; "nothing is fetched until the user agrees"
    # was a property of the page and not of the service. Requiring the pending
    # record that /ui/resolve leaves behind makes it a property of both.
    try:
        parked = await client.find(url)
    except metube.MeTubeError as exc:
        return _unavailable(exc)
    if parked is None:
        return error_response(
            409,
            "That link was never resolved, so there is nothing to confirm. "
            "POST /ui/resolve first and commit the token it returns.",
            code="not_resolved", param="token")

    trimmed = body.clip_start is not None or body.clip_end is not None
    try:
        if trimmed or body.captions:
            # /start promotes a pending item with the options it was ADDED
            # with; there is no route that edits them. So a clip range or a
            # switch to captions means dropping the pending record and adding
            # it again with the final options and auto_start:true. That costs a
            # second extract_info on MeTube's side and is the only correct way
            # to apply a trim.
            await client.abandon(url)
            await client.add(url, auto_start=True,
                             download_type="captions" if body.captions else "audio",
                             clip_start=body.clip_start,
                             clip_end=body.clip_end)
        else:
            await client.start(url)
    except metube.MeTubeError as exc:
        return _unavailable(exc)
    return Response(media_type="application/json",
                    content=_json({"token": url, "status": "started"}))


@router.post("/ui/abandon")
async def abandon(request: Request, body: TokenRequest) -> Response:
    client = _client(request)
    try:
        reaped = await client.abandon(body.token)
    except metube.MeTubeError as exc:
        return _unavailable(exc)
    # Reported rather than assumed. A false here means MeTube still holds the
    # record and someone should look, which is strictly better than a green
    # tick over an orphan.
    return Response(media_type="application/json",
                    content=_json({"token": body.token, "reaped": reaped}))


@router.get("/ui/progress")
async def progress(request: Request, token: str) -> Response:
    client = _client(request)
    try:
        found = await client.find(token)
    except metube.MeTubeError as exc:
        return _unavailable(exc)
    if found is None:
        return error_response(404, "MeTube has no record of that link.",
                              code="unknown_token", param="token")

    where, entry = found
    status = entry.get("status")
    # Terminal success is `finished` and nothing else: anything else in `done`
    # means _post_download_cleanup rewrote the status to "error" and nulled the
    # filename, so treating "in done" as "ready" hands the next step a null.
    ready = where == "done" and status == metube.FINISHED and entry.get("filename")
    return Response(media_type="application/json", content=_json({
        "token": token,
        "where": where,
        "status": status,
        "ready": bool(ready),
        # MeTube's own numbers, which are real, unlike anything we could
        # predict about someone else's bandwidth.
        "percent": entry.get("percent"),
        "speed": entry.get("speed"),
        "eta": entry.get("eta"),
        "filename": entry.get("filename"),
        "error": entry.get("error") or entry.get("msg"),
    }))


def _multipart(boundary: str, fields: dict[str, str], *, filename: str,
               chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """A multipart body streamed from a remote response, never buffered.

    Written by hand rather than handed to httpx's `files=`, which wants a
    file-like object it can size. The alternative was buffering the whole
    ingest -- 131 MB for the brief's own 2h14m example -- into a container with
    a 512 MB limit, or spooling it to a disk this service otherwise never
    touches. Multipart is four lines of framing; neither of those is worth it.
    """
    async def body() -> AsyncIterator[bytes]:
        for name, value in fields.items():
            yield (f"--{boundary}\r\n"
                   f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                   f"{value}\r\n").encode()
        yield (f"--{boundary}\r\n"
               f'Content-Disposition: form-data; name="file"; '
               f'filename="{filename}"\r\n'
               f"Content-Type: application/octet-stream\r\n\r\n").encode()
        async for chunk in chunks:
            yield chunk
        yield f"\r\n--{boundary}--\r\n".encode()
    return body()


@router.post("/ui/fetch")
async def fetch(request: Request, body: TokenRequest) -> Response:
    """Stream MeTube's finished file into the gateway's transcription route.

    SERVER-SIDE, and that is the point: the browser never downloads the media.
    The 131 MB of a two-hour podcast goes MeTube -> here -> gateway ->
    stt-stack over the NAS's own network and the laptop sees only the
    transcript. It is also the only place it CAN happen: MeTube's
    CORS_ALLOWED_ORIGINS is empty, `on_prepare` returns early and emits no CORS
    headers, and its socket.io server was constructed with
    cors_allowed_origins=[], so a browser page cannot call MeTube at all.
    """
    client = _client(request)
    try:
        found = await client.find(body.token)
    except metube.MeTubeError as exc:
        return _unavailable(exc)
    if found is None:
        return error_response(404, "MeTube has no record of that link.",
                              code="unknown_token", param="token")
    where, entry = found
    filename = entry.get("filename")
    if where != "done" or entry.get("status") != metube.FINISHED or not filename:
        return error_response(
            409, f"that download is {entry.get('status') or where}, not finished",
            code="not_ready", param="token")

    # NEVER PREDICTED. `filename` is OUTPUT_TEMPLATE sanitised and byte-trimmed
    # by MeTube; reconstructing it from the title is wrong for every title with
    # a slash, a colon or a non-BMP character in it.
    source = client.audio_url(str(filename), str(entry.get("folder") or ""))
    http: httpx.AsyncClient = request.app.state.client
    query = dict(request.query_params)
    # ALLOWLISTED, because this string is interpolated into a multipart frame
    # this module builds by hand. Unvalidated, a value carrying CRLF and a
    # boundary marker closes the part and opens another: a crafted
    # ?response_format= put a SECOND `name="model"` part on the wire, and the
    # gateway received model=parakeet followed by model=whisper. No privilege
    # is gained -- the same caller can POST arbitrary multipart straight at
    # /v1/audio/transcriptions -- but a hand-built protocol frame must not take
    # an unvalidated string, and the set of valid values is this short.
    wanted = query.get("response_format", "json")
    if wanted not in RESPONSE_FORMATS:
        return error_response(
            400, f"response_format must be one of "
                 f"{', '.join(sorted(RESPONSE_FORMATS))}, not {wanted!r}",
            code="invalid_response_format", param="response_format")
    fields = {
        # Required by /v1 validation, and it does NOT choose an engine --
        # Parakeet runs regardless and says so in x-stt-engine.
        "model": "parakeet",
        "response_format": wanted,
    }

    async def upstream() -> AsyncIterator[bytes]:
        seen = 0
        async with http.stream("GET", source, timeout=60.0) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(65536):
                seen += len(chunk)
                if seen > config.MAX_UPLOAD_BYTES:
                    # A cap on what OUR code pulls, because the only other
                    # ceiling in this chain is stt-stack's memory limit and it
                    # meets it as an OOM kill rather than as an error.
                    raise httpx.ReadError(
                        f"ingested file exceeded {config.MAX_UPLOAD_BYTES} bytes")
                yield chunk

    boundary = "----ai-voice-ingest-boundary-9f2c1a"
    try:
        upstream_response = await http.send(
            http.build_request(
                "POST", f"{config.GATEWAY_URL}/v1/audio/transcriptions",
                headers={
                    "content-type": f"multipart/form-data; boundary={boundary}",
                    # The caller's key, forwarded. The gateway is the auth
                    # boundary for this stack and this request is made on the
                    # caller's behalf, so it carries the caller's credential
                    # and nothing of ours.
                    **({"authorization": request.headers["authorization"]}
                       if "authorization" in request.headers else {}),
                },
                content=_multipart(boundary, fields, filename=str(filename),
                                   chunks=upstream()),
                timeout=httpx.Timeout(960.0, connect=5.0),
            ),
            stream=True,
        )
    except httpx.RequestError as exc:
        log.warning("ingest fetch failed: %s", exc)
        return error_response(
            502, f"could not hand the downloaded audio to the gateway: "
                 f"{type(exc).__name__}",
            type_="server_error", code="ingestion_unavailable")

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_response.aiter_raw():
                yield chunk
        finally:
            await upstream_response.aclose()

    return StreamingResponse(
        relay(), status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get("content-type",
                                                 "application/json"))


def _json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode()
