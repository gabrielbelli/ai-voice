"""The OpenAI shape: the error envelope, and the model list the router advertises.

Two things live here, and they are the same thing seen twice.

The envelope is the shape every error leaves this service in, `/v1` or not.
openai-python reads `error.message` and shows it to the caller; FastAPI's own
`{"detail": ...}` reaches it as an unparsed body and surfaces as a bare status
code. The three backends already emit this envelope under `/v1`, so a gateway
that wrapped their answers in a different one would be a second, lying source
of truth — see the governing rule in main.py.

`MODELS` is the only content this service invents. It exists because the
`model` string is the routing key (main.py: `LONG_MODELS`) and a client needs
somewhere to learn the names, rather than being told them by whoever set the
service up. It is answered from this table with no backend call: the names are
a property of the routing contract, not of any backend's state, and a service
that could not tell you its own routing table while a backend was restarting
would be answering the wrong question.

Kept next to the envelope so the one thing that must not drift — an advertised
name that routes somewhere the table does not claim — drifts in one file.
tests/test_gateway.py asserts the two agree.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

# OpenAI's model object requires `created`, and nothing here has a creation
# date: the entries are routing keys, not artefacts. This is the date the
# table was written. A client that sorts by it gets a stable order; a client
# that renders it gets a date that is at least not 1970.
_CREATED = 1767225600  # 2026-01-01T00:00:00Z

# `owned_by` carries the backend name on purpose. It is the one field in
# OpenAI's model object with room for it, so `GET /v1/models` doubles as the
# routing table: the answer to "why did that take nine minutes" is visible in
# the same response that told the client the name existed.
MODELS: tuple[dict[str, object], ...] = (
    # Fast path, tts-stack. `kokoro` is the honest name; the three OpenAI
    # names are here because clients arrive with them already configured and
    # tts-stack ignores the field anyway.
    {"id": "kokoro", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    {"id": "tts-1", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    {"id": "tts-1-hd", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    {"id": "gpt-4o-mini-tts", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    # Long path, tts-long. Strictly opt-in: these two names are the only
    # values that reach Chatterbox, and they are the only ones that can turn a
    # 17-second call into a job. See LONG_MODELS in main.py.
    {"id": "chatterbox", "object": "model", "created": _CREATED, "owned_by": "tts-long"},
    {"id": "tts-long", "object": "model", "created": _CREATED, "owned_by": "tts-long"},
    # Speech-to-text. There is one STT backend and no decision to make, so
    # these names are documentation rather than routing keys — every
    # transcription request reaches stt-stack whatever `model` says.
    {"id": "parakeet", "object": "model", "created": _CREATED, "owned_by": "stt-stack"},
    {"id": "whisper-1", "object": "model", "created": _CREATED, "owned_by": "stt-stack"},
)

MODEL_LIST: dict[str, object] = {"object": "list", "data": list(MODELS)}


def error_response(status: int, message: str, type_: str, code: str,
                   headers: dict[str, str] | None = None) -> JSONResponse:
    """An error in OpenAI's envelope.

    Used on the native routes too, unlike the backends, which keep FastAPI's
    `{"detail": ...}` there. A gateway-side failure happens before or instead
    of routing, so there is no backend whose conventions it could follow, and
    the client most likely to be turned away is the one that reads only
    `error.message`.
    """
    return JSONResponse(status_code=status,
                        content={"error": {"message": message,
                                           "type": type_,
                                           "code": code}},
                        headers=headers)
