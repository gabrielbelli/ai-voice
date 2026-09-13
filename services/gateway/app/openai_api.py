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

The model list is the only content this service invents. It exists because the
`model` string is the routing key (main.py: `LONG_MODELS`) and a client needs
somewhere to learn the names, rather than being told them by whoever set the
service up. It is answered from this table with no backend call: the names are
a property of the routing contract, not of any backend's state, and a service
that could not tell you its own routing table while a backend was restarting
would be answering the wrong question.

THE LONG-FORM ROWS ARE GENERATED, NOT LISTED, and that is the whole reason
`model_list` is a function rather than the tuple it replaces. The engine is now
the model string, so which long-form names exist is a property of the
deployment (`GATEWAY_LONG_MODELS`) rather than of this file. A hand-written row
beside a configured routing set is two tables, and the one failure that matters
here is exactly the drift between them: a name advertised in `GET /v1/models`
and routed somewhere else, or routed and never advertised. Deriving both from
one frozenset makes that unrepresentable rather than merely tested.

The one thing that must not drift — an advertised name that routes somewhere
the table does not claim — is asserted by tests/test_gateway.py.
"""

from __future__ import annotations

from typing import Iterable

# OpenAI's model object requires `created`, and nothing here has a creation
# date: the entries are routing keys, not artefacts. This is the date the
# table was written. A client that sorts by it gets a stable order; a client
# that renders it gets a date that is at least not 1970.
_CREATED = 1767225600  # 2026-01-01T00:00:00Z

# `owned_by` carries the backend name on purpose. It is the one field in
# OpenAI's model object with room for it, so `GET /v1/models` doubles as the
# routing table: the answer to "why did that take nine minutes" is visible in
# the same response that told the client the name existed.
#
# These are the rows no deployment decides. The fast path is one backend with
# no engine selection in it, and there is one STT backend, so nothing here can
# move.
FIXED: tuple[dict[str, object], ...] = (
    # Fast path, tts-stack. `kokoro` is the honest name; the three OpenAI
    # names are here because clients arrive with them already configured and
    # tts-stack ignores the field anyway.
    {"id": "kokoro", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    {"id": "tts-1", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    {"id": "tts-1-hd", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    {"id": "gpt-4o-mini-tts", "object": "model", "created": _CREATED, "owned_by": "tts-stack"},
    # Speech-to-text. There is one STT backend and no decision to make, so
    # these names are documentation rather than routing keys — every
    # transcription request reaches stt-stack whatever `model` says.
    {"id": "parakeet", "object": "model", "created": _CREATED, "owned_by": "stt-stack"},
    {"id": "whisper-1", "object": "model", "created": _CREATED, "owned_by": "stt-stack"},
)

# Where the generated long-form rows are spliced in, so the published order
# stays fast path, long path, speech-to-text — the order a reader of the old
# hand-written tuple would recognise, and the order the README prints.
_LONG_AT = 4


def model_list(long_models: Iterable[str]) -> dict[str, object]:
    """`GET /v1/models`, built from the set that actually routes.

    THE ARGUMENT IS THE ROUTING SET ITSELF, passed in rather than read here.
    main.py owns `LONG_MODELS` because main.py is where the branch on it lives,
    and a second `os.getenv("GATEWAY_LONG_MODELS")` in this module would be a
    second reading of one variable — which is how a table advertises a name the
    router does not take. There is one read, one set, and both answers come off
    it.

    Sorted so the advertised order does not depend on how the operator happened
    to spell the variable, which would otherwise make an innocuous compose edit
    reorder a client's model picker.
    """
    long_rows = tuple(
        # Strictly opt-in: these names are the only values that reach a
        # long-form engine, and they are the only ones that can turn a
        # 17-second call into a job. See LONG_MODELS in main.py.
        #
        # `owned_by` is the literal rather than a catalogue lookup because it
        # is TRUE BY CONSTRUCTION HERE and a lookup would be the weaker claim:
        # this argument IS the router's `if key in LONG_MODELS: backend = LONG`
        # set, so every row below goes to tts-long whatever any table says
        # about the checkpoint. Reading an owner off the catalogue would let
        # the published row disagree with where the request lands — which is
        # the one failure this module's docstring exists to rule out. The
        # catalogue's owner decides which names are REFUSED here rather than
        # sent fast (LONG_KNOWN in main.py); it does not decide where an
        # operator's own routing set goes.
        {"id": name, "object": "model", "created": _CREATED, "owned_by": "tts-long"}
        for name in sorted(long_models))
    rows = FIXED[:_LONG_AT] + long_rows + FIXED[_LONG_AT:]
    return {"object": "list", "data": list(rows)}
