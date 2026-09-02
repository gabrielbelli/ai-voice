"""The model list this router advertises, and nothing else.

The error envelope used to live here too, as a fourth hand-written copy of a
shape three other services already shared. It is `voice_common.errors` now.
Two things were wrong with the copy and both were on the wire: it built three
keys where OpenAI's schema requires four — `param` is required-but-nullable and
was absent from every error this gateway had ever emitted — and its
`error_response(status, message, type_, code)` took the two strings
POSITIONALLY, which is the exact mistake the shared function is keyword-only to
prevent. Two sibling repos had already swapped that pair; this was the last
place in the estate where the swap could still be written.

The envelope remains the shape every error leaves this service in, `/v1` or
not, which is this service's own older decision and not the backends': a
gateway-side failure happens before or instead of routing, so there is no
backend whose conventions it could follow. main.py carries the boundary the
shared handlers draw and the one place this service stays on its own side of it.

`MODELS` is the only content this service invents. It exists because the
`model` string is the routing key (main.py: `LONG_MODELS`) and a client needs
somewhere to learn the names, rather than being told them by whoever set the
service up. It is answered from this table with no backend call: the names are
a property of the routing contract, not of any backend's state, and a service
that could not tell you its own routing table while a backend was restarting
would be answering the wrong question.

The one thing that must not drift — an advertised name that routes somewhere
the table does not claim — is asserted by tests/test_gateway.py.
"""

from __future__ import annotations

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
