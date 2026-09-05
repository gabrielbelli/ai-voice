"""Self-hosted speech-to-text. One container, one model.

    audio -> VAD -> recogniser -> glossary repair -> text

Parakeet by default, Whisper by request. There is no LLM cleanup stage and no
second recogniser; both were tried and measured, and both made the transcript
worse. The reasoning is in asr.py and in the README.

Three routes reach the same pipeline. /transcribe is native and returns
everything the run measured; /v1/audio/transcriptions and
/v1/audio/translations are OpenAI-compatible and return the subset that
specification has fields for, so existing clients work unchanged. Prefer the
native one where you control the client — see openai_api.py for what the
compatible shape has to drop, and for the one rule that surface is built
around: every field is honoured or refused by name, never accepted and
dropped.

The wire contract around those routes — the API keys, the 401, OpenAI's error
envelope, the health route and the log configuration — comes from voice_common,
which this service shares with tts-stack and tts-long. Three hand-vendored
copies of that code had drifted into three different defects; the package
docstrings carry the detail. app/errors.py is gone with them: it completed the
envelope with `param` and the 404/405 handlers while voice-common was pinned by
tarball SHA and could not be changed from here, and all of it now lives in
voice_common.errors. Everything below this line is what is genuinely particular
to speech-to-text.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from voice_common import auth, errors, health
from voice_common import logging as voice_logging

from . import asr, openai_api, pipeline, profiles

# STT_LOG_LEVEL comes with this: until now the only way to get DEBUG out of a
# running container was to edit the source and rebuild the image.
log = voice_logging.setup("stt-stack", "STT")


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    started = time.monotonic()
    pipeline.start()

    log.info("ready in %.1fs, model=%s, %d threads",
             time.monotonic() - started, pipeline.MODEL, pipeline.THREADS)
    yield
    pipeline.stop()


app = FastAPI(
    title="stt-stack",
    description="VAD, one recogniser, glossary repair. Parakeet by default.",
    lifespan=lifespan,
)

# ApiError, the /v1 validation handler, the 404/405 handler and the unhandled
# 500, all rendered in OpenAI's four-field envelope.
errors.install_errors(app)


def _health_details() -> dict[str, object]:
    """The per-service half of /health. Must not block: it runs on the loop.

    Every value here is a module constant or a dict lookup, so it does not.
    `status` is overridden while the model is still loading — that is the one
    field voice_common fills in itself, and the one this service has to
    contradict.
    """
    loaded = pipeline.loaded()
    return {
        "status": "ok" if loaded else "loading",
        "model": pipeline.MODEL,
        "model_id": os.getenv("STT_MODEL_ID", ""),
        "accepts_vocabulary": bool(getattr(loaded, "accepts_vocabulary", False)),
        # What the compatibility surface will and will not do on this
        # deployment, so a client can find out without spending a request on a
        # refusal. Both are properties of the engine, not of the service.
        "translations": bool(getattr(loaded, "can_translate", False)),
        "streaming": bool(getattr(loaded, "can_stream", False)),
        "hotwords": pipeline.HOTWORDS_ENABLED,
        # Which profiles this process has loaded, so a client can see the set
        # without spending a request on a 400 for a name that is not there.
        #
        # Read straight out of the pipeline's state rather than through
        # pipeline.registry(), and NOT refreshed: both of those stat() the
        # directories, and this function runs on the event loop, where a stat
        # against a hung NFS mount would block every request including the
        # container's own healthcheck. Whether writes are possible is a
        # question for GET /glossaries, which is allowed to touch the disk.
        "glossaries": sorted(getattr(pipeline.state.get("glossaries"),
                                     "profiles", {})),
        "vad": pipeline.VAD_ENABLED,
        "threads": pipeline.THREADS,
        "max_concurrent": pipeline.MAX_CONCURRENT,
    }


# Registers GET /health AND exempts exactly that path from the key check, so
# the route and the exemption can never come to name different strings.
# Container healthchecks call it and have no key; requiring one would turn a
# working service into a restart loop.
health.install_health(app, details=_health_details)

# Everything else needs the key, /openapi.json, /docs and /redoc included: a
# schema dump is a free map of the service. This is middleware rather than a
# per-route dependency, which is what lets FastAPI's own three pages be used
# again — they are plain Starlette routes that no router dependency reaches,
# so this file used to re-register all three by hand just to get the check in
# front of them. Middleware also cannot be forgotten on a route added later.
auth.install(app, "STT_API_KEYS")

app.include_router(openai_api.router)


class Transcript(BaseModel):
    text: str
    raw: str
    repaired: list[str]
    model: str
    audio_seconds: float
    speech_seconds: float
    compute_seconds: float
    realtime_factor: float


# Deliberately `def`, not `async def`. pipeline.run is blocking CPU work;
# declared async it would run ON the event loop and starve every other
# request, /health included, so a container healthcheck fails under load and
# the orchestrator restarts a service that is working correctly.
@app.post("/transcribe", response_model=Transcript)
def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    # The same selector /v1 takes, spelled the same way. The native route is
    # not bound by ADR 0001 and could have used any name; using a different one
    # would mean the two routes on one service disagree about what a glossary
    # profile is called, which is the sort of difference a client discovers by
    # being wrong.
    glossary: str | None = Form(default=None),
) -> Transcript:
    # Built field by field rather than from asdict(): the pipeline's Result now
    # also carries segments, words and logprobs for the /v1 shapes, and this
    # body is a contract that already has clients. Widening it because another
    # route needed the data would be the same mistake as narrowing it.
    result = pipeline.run(
        file.file.read(),
        asr.Options(language=language),
        rules=_select(glossary).rules,
        # 16 kHz only, still. See pipeline.decode: /v1 resamples because no
        # OpenAI client expects otherwise, and this route does not because
        # telling a client its audio is the wrong rate is the documented
        # behaviour it was built with.
        allow_resample=False,
    )
    return Transcript(
        text=result.text,
        raw=result.raw,
        repaired=result.repaired,
        model=result.model,
        audio_seconds=result.audio_seconds,
        speech_seconds=result.speech_seconds,
        compute_seconds=result.compute_seconds,
        realtime_factor=result.realtime_factor,
    )


# ── glossary profiles ─────────────────────────────────────────────────────────
#
# NATIVE ROUTES, NOT /v1, and the reason is ADR 0001 rather than convenience:
# OpenAI has no concept of a glossary profile, so there is nothing here to be
# 1:1 with. Claiming /v1/glossaries would be taking specification territory
# that does not exist, and would collide the day OpenAI uses that path. Native
# routes are explicitly out of that ADR's scope, so these keep FastAPI's
# {"detail": ...} bodies like /transcribe does.
#
# AUTHENTICATION IS STT_API_KEYS, unchanged, and it is worth saying plainly
# what that means rather than leaving it in a comment nobody reads: with the
# keys unset — which is how this stack is deployed today — a write API is an
# UNAUTHENTICATED write API. Set them before mounting the volume. The README
# says so where an operator will actually meet it.
#
# Writability follows the volume. This is not a permission system and calling
# it one would be dishonest: a deployment that mounted nowhere to persist has
# said, by omission, that it does not want run-time profiles, and accepting a
# PUT that evaporates on the next restart would be worse than refusing it. Same
# reasoning as the UI's clips.writable().


def _select(names: str | None) -> profiles.Selection:
    """Resolve a native request's `glossary=` into compiled rules.

    Mirrors openai_api._glossary, including the 400 on an unknown name, but
    raises HTTPException so the native body stays {"detail": ...}. An unknown
    profile is named, never ignored: a caller who believes their vocabulary was
    applied when it was not has no way to discover the difference.
    """
    wanted = profiles.split_selection(names)
    if not wanted:
        return profiles.Selection(rules=pipeline.default_rules())
    registry = pipeline.registry()
    registry.refresh()
    try:
        return registry.select(wanted)
    except profiles.UnknownProfile as exc:
        raise HTTPException(
            400,
            f"unknown glossary profile {exc.name!r}; this deployment has: "
            f"{', '.join(exc.known) or 'none'}",
        ) from exc


def _registry() -> profiles.Registry:
    registry = pipeline.registry()
    registry.refresh()
    return registry


def _profile(registry: profiles.Registry, name: str) -> profiles.Profile:
    if not profiles.valid_name(name):
        raise HTTPException(
            400,
            f"{name!r} is not a usable profile name; it becomes a filename, so "
            f"it must match {profiles.NAME_PATTERN.pattern}")
    try:
        return registry.get(name)
    except profiles.UnknownProfile as exc:
        raise HTTPException(404, f"no glossary profile named {exc.name!r}") from exc


def _require_writable(registry: profiles.Registry) -> None:
    writable, reason = registry.writability()
    if not writable:
        # 503 rather than 403: nothing about the CALLER is being refused. The
        # deployment has no writable volume, and the message says which one and
        # what to do about it, because "forbidden" would send an operator
        # looking for a permission they never configured.
        raise HTTPException(503, f"glossary profiles are read-only here: {reason}")


def _refuse_shadowing(profile: profiles.Profile) -> None:
    """A built-in name is a 409, not a silent shadow.

    A profile whose contents depend on which directory won is a profile nobody
    can reason about, and "why is `tech` different on that box" is not a
    question worth creating. Copy it under another name instead.
    """
    if not profile.writable:
        raise HTTPException(
            409,
            f"{profile.name!r} is a {profile.source} profile and is read-only. "
            f"It ships in the image at {profile.path}. Copy it to a new name "
            "and edit that: GET /glossaries/"
            f"{profile.name} gives you its text.")


@app.get("/glossaries")
def list_glossaries() -> dict[str, object]:
    """Every profile, where it came from, and how many terms it carries."""
    registry = _registry()
    writable, reason = registry.writability()
    body: dict[str, object] = {
        "glossaries": [registry.profiles[name].summary()
                       for name in registry.names],
        "writable": writable,
        # The profiles a request gets when it selects none. Empty on a default
        # deployment, and that is the point: an irrelevant glossary raised WER
        # by 28% on Whisper, and nothing measurable on Parakeet across 25 cells, so
        # always-on is opted into by name rather than inherited.
        "default": profiles.split_selection(pipeline.DEFAULT_PROFILES),
        "builtin_dir": str(registry.builtin_dir),
        "custom_dir": str(registry.custom_dir),
    }
    if not writable:
        body["reason"] = reason
    return body


@app.get("/glossaries/{name}")
def get_glossary(name: str) -> dict[str, object]:
    """One profile's terms, and the file text they were parsed from.

    `text` is returned as well as the parsed halves so that editing a profile
    is a round trip — GET, change a line, PUT — rather than a reconstruction
    from two JSON objects that would drop every comment in the file. The
    comments are where a glossary explains why a rule is a hotword rather than
    a replacement, which is exactly the knowledge worth not losing.
    """
    registry = _registry()
    profile = _profile(registry, name)
    return {
        **profile.summary(),
        "replacements": profile.parsed.replacements,
        "hotwords": list(profile.parsed.hotwords),
        "text": profile.text,
        "path": str(profile.path),
    }


@app.put("/glossaries/{name}")
async def put_glossary(name: str, request: Request) -> Response:
    """Create or replace a custom profile.

    Two body shapes, because both callers are real: `application/json` with
    {"text": ..., "force": ...}, and a raw `text/plain` body for
    `curl -X PUT --data-binary @mine.txt`, which is a perfectly good client for
    a text file. `?force=true` works with either.

    `async def` only because the body has to be awaited. Everything after that
    stat()s and writes a mounted volume, which is exactly the work that must
    not happen on the event loop — a hung NFS mount would otherwise take
    /health down with it and have the orchestrator restart a service that is
    working — so it runs in a worker thread.
    """
    text, force = await _body(request)
    return await run_in_threadpool(_write_profile, name, text, force)


def _write_profile(name: str, text: str, force: bool) -> Response:
    """The blocking half of PUT. See put_glossary for why it is split off.

    NOTHING IS WRITTEN IF ANYTHING WAS REJECTED. The rejected lines come back
    with their line numbers and reasons and the file on disk is untouched.

    That is a deliberate reading of "the response says how many terms were
    accepted and lists every line that was not": a 200 carrying a `rejected`
    array is trivially ignored by a script, and a profile that silently lost
    three of its rules is precisely the half-succeeded write this repository
    has already been bitten by three times in other forms. An error status
    cannot be ignored by accident, and re-sending a corrected file costs one
    request.
    """
    registry = _registry()
    if not profiles.valid_name(name):
        raise HTTPException(
            400,
            f"{name!r} is not a usable profile name; it becomes a filename, so "
            f"it must match {profiles.NAME_PATTERN.pattern}")

    # A name already taken by a built-in is a 409 BEFORE the writability check,
    # so an operator on an unmounted box is not told to mount a volume and then
    # told, one deploy later, that the name was never available anyway.
    existing = registry.profiles.get(name.strip().lower())
    if existing is not None:
        _refuse_shadowing(existing)
    _require_writable(registry)

    try:
        parsed = profiles.check(text, force=force)
    except profiles.TooLarge as exc:
        # 413, not 400: the payload is the problem, and a client that reads
        # status codes rather than messages should still learn the right thing.
        raise HTTPException(413, str(exc)) from exc

    if parsed.rejected:
        raise HTTPException(400, {
            "message": (
                f"{len(parsed.rejected)} line(s) rejected; nothing was written. "
                f"{parsed.terms} term(s) would have been accepted."),
            "accepted": parsed.terms,
            "rejected": [r.as_dict() for r in parsed.rejected],
        })

    path = registry.write(name, text)
    profile = registry.get(name)
    log.info("glossary profile %r written: %d terms (%s)", profile.name,
             profile.parsed.terms, "forced" if force else "validated")
    return JSONResponse(
        status_code=200 if existing is not None else 201,
        content={**profile.summary(), "path": str(path), "forced": force,
                 "created": existing is None},
    )


@app.delete("/glossaries/{name}")
def delete_glossary(name: str) -> dict[str, object]:
    registry = _registry()
    profile = _profile(registry, name)
    _refuse_shadowing(profile)
    _require_writable(registry)
    registry.remove(profile.name)
    log.info("glossary profile %r deleted", profile.name)
    return {"name": profile.name, "deleted": True}


async def _body(request: Request) -> tuple[str, bool]:
    """The proposed file text, and whether the caller forced it.

    `force` is read from the query string first so that it works for both body
    shapes; a JSON body may also carry it. Anything but JSON is taken as the
    file itself, which is what makes `--data-binary @mine.txt` work with no
    content type set at all.
    """
    force = request.query_params.get("force", "").strip().lower() in {
        "1", "true", "yes", "on"}
    raw = await request.body()
    media = (request.headers.get("content-type") or "").split(";")[0].strip()

    if media == "application/json":
        try:
            payload = json.loads(raw or b"{}")
        except ValueError as exc:
            raise HTTPException(400, f"body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(
                payload.get("text"), str):
            raise HTTPException(
                400, "a JSON body must be an object with a string 'text' field; "
                     "send the file as text/plain to skip the wrapper")
        return payload["text"], force or bool(payload.get("force"))

    try:
        return raw.decode("utf-8"), force
    except UnicodeDecodeError as exc:
        raise HTTPException(
            400, "glossary files are UTF-8 text; this body is not") from exc
