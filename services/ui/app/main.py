"""The page, and the three things a browser cannot do for itself.

    GET  /ui                       the page: one HTML file, no build step
    GET  /ui/config                what the page needs to boot, no secrets
    GET  /ui/clips  POST  DELETE   the reference-clip store (voice cloning)
    POST /ui/resolve /commit
         /abandon  /fetch          MeTube ingestion -- see app/ingest.py
    GET  /ui/progress
    everything else in the table   forwarded verbatim to the gateway

WHY THIS IS A FIFTH CONTAINER AND NOT THREE ROUTES ON THE GATEWAY. The survey
argued for the latter and the argument was good: the page is 8 KB of static
markup, the gateway is already the only published port, and the whole estate
is organised around not adding a container. It is a fifth container because
the brief asked for `services/ui` with a Containerfile, a compose entry, a
workflow matrix row and a README like its four siblings -- and because the
gateway's Containerfile makes a specific promise this feature would break:
283 MB, "the cheap check on this file", no dependency that is not fastapi,
httpx or uvicorn. Ingestion needs yt-dlp, and yt-dlp in the gateway makes the
process that holds every API key also the process that spawns a subprocess on
a URL a browser chose. Separate images keep that blast radius where it is.

WHAT IT COSTS, HONESTLY: a second published port. compose.yaml's own comment
says "if a second `ports` entry ever appears here, this file has stopped doing
its job", and it is right about backend ports -- 8000, 8001 and 8002 are what
that sentence was written about, and they stay closed. This one is a browser
origin, not a bypass: it reaches the gateway and nothing else, it forwards the
caller's Authorization untouched, and it can answer nothing the gateway would
not have answered. The boundary is where it was.

WHY THE PAGE'S XHRs COME BACK HERE RATHER THAN GOING STRAIGHT TO :30080. One
origin means no CORS on the gateway, no preflight on every upload, and no
second base URL for someone to get wrong. The forwarding table below is a
FIXED allowlist -- there is no wildcard and no catch-all, for the same reason
the gateway has none: a wildcard would proxy /docs and /openapi.json to
services that deliberately do not publish them.

AUTHENTICATION IS THE GATEWAY'S, AND ONLY THE GATEWAY'S. This service holds no
key list and compares no token. Proxied routes carry the caller's header
through and the gateway answers 401 itself. The routes that are ours --
ingestion, clips -- ask the gateway whether the presented key is good by making
the cheapest authenticated call it has, `GET /v1/models`, which it answers from
a static table without touching a backend. So there is exactly one place a key
is configured, and when GATEWAY_API_KEYS is unset this service is exactly as
open as the stack already is: no more, and not accidentally less.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, UploadFile
from fastapi import File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response, StreamingResponse
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from voice_common.errors import error_response, http_error_response, v1_path

from . import clips, config, ingest, metube, probe

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voice-ui")
# One access line per request is this service's whole observability budget, as
# it is the gateway's. httpx logging one of its own would double every line and
# say less.
logging.getLogger("httpx").setLevel(logging.WARNING)

PAGE = Path(__file__).with_name("static") / "ui.html"

# THE ALLOWLIST. Every path the page is allowed to reach through this process,
# with the method it may use. Nothing else is forwarded; an unlisted path is a
# 404 here exactly as it is at the gateway.
#
# /v1/audio/translations is absent because the gateway does not route it and
# Parakeet refuses translation anyway. DELETE /jobs/{id} is present because
# tts-long has had that route all along (main.py:580) and only the gateway's
# route table was missing it -- which is why the Jobs tab can offer "stop and
# keep what's done" rather than a job that cannot be called off.
PROXIED: tuple[tuple[str, str], ...] = (
    ("POST", "/v1/audio/transcriptions"),
    ("POST", "/transcribe"),
    ("POST", "/v1/audio/speech"),
    ("POST", "/speak"),
    ("GET", "/voices"),
    ("POST", "/jobs"),
    ("GET", "/jobs"),
    ("GET", "/jobs/{job_id}"),
    ("DELETE", "/jobs/{job_id}"),
    ("GET", "/jobs/{job_id}/audio"),
    ("GET", "/v1/models"),
)

# Routes whose request body is an upload and must therefore never be buffered
# here, and whose Content-Length is checked before a byte is forwarded.
UPLOAD_PATHS = frozenset({"/v1/audio/transcriptions", "/transcribe"})

HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade",
})
# `host` because httpx sets the gateway's own. `authorization` is NOT dropped,
# unlike in the gateway: there the next hop is a backend running with its keys
# unset, and here the next hop is the process that checks the key.
DROP_FROM_REQUEST = HOP_BY_HOP | {"host"}


def new_client() -> httpx.AsyncClient:
    """The one connection pool, kept open for the process lifetime.

    A module-level function rather than an inline constructor for the same
    reason services/gateway has one: it is the seam the tests hand a transport
    through, so the whole forwarding path runs for real against a mock gateway
    and a mock MeTube with no socket anywhere.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        # A redirect from the gateway means something about the gateway's
        # routing and following it here would hide it.
        follow_redirects=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.client = new_client()
    app.state.authorised = {}
    log.info("ready: gateway=%s metube=%s probe=%s voices=%s",
             config.GATEWAY_URL, config.METUBE_URL or "(unset)",
             "on" if probe.available() else "off", config.VOICE_DIR)
    if not metube.configured():
        log.warning("UI_METUBE_URL is unset: the link box is hidden and only "
                    "file upload is offered. Set it to the MeTube on this "
                    "host, by LAN address -- it is a separate TrueNAS app and "
                    "shares no DNS with this one.")
    if config.PROBE and not probe.available():
        log.warning("UI_PROBE is on but yt-dlp is not on PATH: the confirm "
                    "card will show a title and no duration or size.")
    yield
    await app.state.client.aclose()


app = FastAPI(
    title="voice-ui",
    description="One page in front of the gateway.",
    lifespan=lifespan,
    # Same reasoning as the gateway's: publishing a map of a service that
    # deliberately does not publish one undoes that decision from a new port.
    openapi_url=None, docs_url=None, redoc_url=None,
)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
    if v1_path(request.url.path):
        return http_error_response(request, exc)
    if exc.status_code == 404:
        return error_response(
            404, f"Invalid URL ({request.method} {request.url.path}). This UI "
                 "forwards a fixed set of paths to the gateway.",
            code="unknown_url")
    return error_response(exc.status_code, str(exc.detail),
                          code="invalid_request_error")


# ------------------------------------------------------------------- auth --


def _cache_key(header: str | None) -> str:
    # Hashed, so a key never sits in this process's memory in a form a heap
    # dump hands over, and never reaches a log line by accident.
    return hashlib.sha256((header or "").encode("utf-8", "replace")).hexdigest()


async def authorised(request: Request) -> Response | None:
    """None if the caller may use our own routes, or the refusal to return.

    Asked of the gateway rather than answered here. GET /v1/models is the
    cheapest authenticated call in the stack -- a static table, no backend
    contacted -- so the check costs one loopback request a minute per key and
    the answer is always the CURRENT key list rather than a copy of it that
    drifts.
    """
    header = request.headers.get("authorization")
    cache: dict[str, float] = request.app.state.authorised
    key = _cache_key(header)
    now = time.monotonic()
    if cache.get(key, 0.0) > now:
        return None

    client: httpx.AsyncClient = request.app.state.client
    try:
        response = await client.get(
            config.GATEWAY_URL + "/v1/models",
            headers={"authorization": header} if header else {},
            timeout=10.0)
    except httpx.RequestError as exc:
        # FAIL CLOSED. These routes spawn a subprocess and write files; running
        # them unchecked because the thing that checks is down is the wrong way
        # round. The proxied routes are unaffected -- the gateway refuses them
        # itself, or is unreachable and says so.
        return error_response(
            503, f"the gateway is not reachable, so the API key cannot be "
                 f"checked ({type(exc).__name__}).",
            type_="server_error", code="backend_unavailable",
            headers={"Retry-After": "30"})

    if response.status_code == 401:
        return Response(content=response.content, status_code=401,
                        media_type=response.headers.get("content-type",
                                                        "application/json"),
                        headers={"WWW-Authenticate": "Bearer"})
    if response.status_code != 200:
        return error_response(
            502, f"the gateway answered {response.status_code} when asked to "
                 "check the API key",
            type_="server_error", code="backend_error")

    if len(cache) > 256:
        cache.clear()
    # 60 s. Long enough that a page polling /jobs is not re-checked on every
    # tick, short enough that a revoked key stops working within a minute.
    cache[key] = now + 60.0
    return None


# ------------------------------------------------------------------ proxy --


def _request_headers(request: Request) -> list[tuple[str, str]]:
    return [(k, v) for k, v in request.headers.items()
            if k.lower() not in DROP_FROM_REQUEST]


def _response_headers(upstream: httpx.Response) -> MutableHeaders:
    # A list of pairs, never a dict: HTTP allows a field name to repeat and
    # collapsing the repeats corrupts the message. Same reasoning, and the same
    # measurement, as services/gateway/app/main.py.
    return MutableHeaders(raw=[
        (k.encode("latin-1"), v.encode("latin-1"))
        for k, v in upstream.headers.multi_items()
        if k.lower() not in HOP_BY_HOP])


async def _relay(upstream: httpx.Response, *, path: str,
                 started: float) -> AsyncIterator[bytes]:
    status: object = upstream.status_code
    try:
        # aiter_raw, not aiter_bytes: the body goes out exactly as it came in,
        # which is what keeps content-encoding and content-length honest.
        async for chunk in upstream.aiter_raw():
            yield chunk
    except httpx.TimeoutException:
        status = f"{upstream.status_code}+read-timeout"
    finally:
        await upstream.aclose()
        log.info("proxy=%s status=%s duration=%.3f", path, status,
                 time.monotonic() - started)


async def _forward(request: Request) -> Response:
    """Send this request to the gateway and stream the answer back."""
    started = time.monotonic()
    path = request.url.path

    if path in UPLOAD_PATHS:
        # THE CEILING THAT DOES NOT EXIST ANYWHERE ELSE IN THE CHAIN.
        # services/stt/app/main.py:138 is a bare `file.file.read()` on an
        # UploadFile: no Content-Length check, no cap, no streaming, so a 4 GB
        # MKV is buffered whole into a container with a 6 GB limit and the
        # failure is an OOM kill rather than a message. Rejecting here costs
        # one comparison and happens before a byte is forwarded.
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > config.MAX_UPLOAD_BYTES:
            return error_response(
                413,
                f"that upload is {int(declared) / 1024**3:.1f} GB; the ceiling "
                f"is {config.MAX_UPLOAD_BYTES / 1024**3:.1f} GB. The page "
                "normally extracts the audio in your browser first, which "
                "turns a 2 GB video into about 15 MB.",
                code="upload_too_large")

    client: httpx.AsyncClient = request.app.state.client
    url = config.GATEWAY_URL + path
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = request.stream() if request.method in {"POST", "PUT", "PATCH"} else None
    try:
        upstream = await client.send(
            client.build_request(
                request.method, url, headers=_request_headers(request),
                content=body,
                # Above the gateway's own longest read timeout (900 s for STT)
                # so its honest 504 always wins the race. A shorter one here
                # would report a timeout for a transcription that is still
                # running and about to succeed.
                timeout=httpx.Timeout(960.0, connect=5.0)),
            stream=True)
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return error_response(
            503, "the gateway is not reachable from the UI container; it may "
                 "be restarting.",
            type_="server_error", code="backend_unavailable",
            headers={"Retry-After": "30"})
    except ClientDisconnect:
        return Response(status_code=499)
    except httpx.RequestError as exc:
        return error_response(
            503, f"the gateway could not be reached: {type(exc).__name__}.",
            type_="server_error", code="backend_unavailable",
            headers={"Retry-After": "30"})

    return StreamingResponse(
        _relay(upstream, path=path, started=started),
        status_code=upstream.status_code,
        headers=_response_headers(upstream))


# Registered from the table in a loop rather than as eleven near-identical
# decorated functions. It is still an explicit allowlist -- FastAPI is handed
# one route per entry and no wildcard exists -- and the loop is what keeps the
# list readable as a list.
for _method, _path in PROXIED:
    app.add_api_route(_path, _forward, methods=[_method], include_in_schema=False)


# ------------------------------------------------------------------- page --


@app.get("/", include_in_schema=False)
async def root() -> Response:
    return RedirectResponse("/ui", status_code=307)


@app.get("/ui", include_in_schema=False)
async def page() -> Response:
    """The whole UI: one file, inline CSS and JS, no build step.

    Read from disk on every request rather than cached at import. It is tens of
    kilobytes off a local filesystem, this process serves one user, and being
    able to edit the file in a running container and reload is worth more than
    the syscall.
    """
    return HTMLResponse(PAGE.read_text(encoding="utf-8"), headers={
        # The page makes requests to its own origin and nowhere else -- no CDN,
        # no external font, no analytics -- so the policy can say exactly that,
        # and it keeps working on a NAS with no internet.
        "Content-Security-Policy":
            "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
            "script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; frame-ancestors 'none'",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    })


@app.get("/ui/config", include_in_schema=False)
async def ui_config() -> Response:
    """What the page needs before it can draw itself. Deliberately no secrets.

    Unauthenticated, like the page itself: it names which FEATURES are
    configured, never how to reach them. There is no MeTube URL in here -- the
    browser could not use it anyway, since MeTube emits no CORS headers at all
    (CORS_ALLOWED_ORIGINS empty, on_prepare returns early, AsyncServer built
    with cors_allowed_origins=[]), which is the other half of why ingestion is
    server-side.
    """
    return Response(media_type="application/json", content=json.dumps({
        "ingestion": metube.configured(),
        "probe": probe.available(),
        "cloning": clips.writable(),
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
        "max_clip_seconds": config.MAX_CLIP_SECONDS,
        "confirm_seconds": config.CONFIRM_SECONDS,
        "confirm_bytes": config.CONFIRM_BYTES,
        # The seed only. The page keeps its own EMA in localStorage from the
        # realtime_factor every native transcription returns, because the
        # repository states three different rates for Parakeet -- 47-63x in the
        # root README, 8.5-10.4x in the gateway (which is what its 900 s
        # timeout and 504 text were built on) and about 5x in the stt README.
        # A measured number beats all three.
        "stt_rtf_seed": config.STT_RTF_SEED,
        "stt_budget_seconds": config.STT_BUDGET_SECONDS,
    }))


# ------------------------------------------------------------------ clips --


@app.get("/ui/clips", include_in_schema=False)
async def list_clips(request: Request) -> Response:
    return Response(media_type="application/json", content=json.dumps({
        "voices": clips.listing(), "writable": clips.writable(),
        "max_seconds": config.MAX_CLIP_SECONDS}))


@app.post("/ui/clips", include_in_schema=False)
async def add_clip(request: Request, name: str = Form(...),
                   replace: bool = Form(default=False),
                   file: UploadFile = File(...)) -> Response:
    # read() is bounded by the size cap the store applies immediately after --
    # a clip is 10-30 s of 24 kHz mono, about 1.4 MB, and the ceiling is 25 MB.
    # This is the one upload in the service small enough to hold.
    data = await file.read()
    try:
        saved = clips.save(name, data, replace=replace)
    except clips.ClipError as exc:
        return error_response(400, str(exc), code="invalid_clip", param="file")
    return Response(media_type="application/json", status_code=201,
                    content=json.dumps({"voice": saved,
                                        "voices": clips.listing()}))


@app.delete("/ui/clips/{name}", include_in_schema=False)
async def delete_clip(request: Request, name: str) -> Response:
    try:
        gone = clips.remove(name)
    except clips.ClipError as exc:
        return error_response(400, str(exc), code="invalid_clip", param="name")
    if not gone:
        return error_response(404, f"no voice called {name!r}",
                              code="unknown_voice", param="name")
    return Response(media_type="application/json",
                    content=json.dumps({"deleted": name,
                                        "voices": clips.listing()}))


# -------------------------------------------------------------- ingestion --


@app.middleware("http")
async def guard_ingestion(request: Request, call_next):
    """Our own routes are key-checked; the proxied ones are the gateway's job.

    Middleware rather than a dependency on each of the five, for the reason the
    gateway's auth module gives: a route added later is protected by default,
    and the alternative is remembering.
    """
    path = request.url.path
    # /ui itself and /ui/config are public: the page is static markup with no
    # data in it, and the config route names which features exist without
    # saying how to reach any of them. Every other /ui/* route either spawns a
    # process or writes a file.
    if path.startswith("/ui/") and path != "/ui/config":
        refused = await authorised(request)
        if refused is not None:
            return refused
    return await call_next(request)


app.include_router(ingest.router)


# ----------------------------------------------------------------- health --


@app.get("/health", include_in_schema=False)
async def health(request: Request) -> Response:
    """This container's own probe. Never a key, and always 200.

    Reports what the gateway said, but does not adopt its verdict: a UI that
    reported itself unhealthy because tts-long was restarting would be killed
    by the orchestrator at exactly the moment someone wanted to look at the
    page and find out why. Read `status`, as everywhere else in this stack.
    """
    client: httpx.AsyncClient = request.app.state.client
    gateway: dict[str, object]
    try:
        response = await client.get(config.GATEWAY_URL + "/health", timeout=5.0)
        gateway = {"reachable": True, "http_status": response.status_code,
                   "health": response.json()}
    except (httpx.RequestError, ValueError) as exc:
        gateway = {"reachable": False, "error": type(exc).__name__}
    return Response(media_type="application/json", content=json.dumps({
        "status": "ok" if gateway.get("reachable") else "degraded",
        "ui": "ok",
        "features": {"ingestion": metube.configured(),
                     "probe": probe.available(),
                     "cloning": clips.writable()},
        "gateway": gateway,
    }))
