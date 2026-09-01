"""The OpenAI shape, translated onto Kokoro.

    POST /v1/audio/speech   model, input, voice, response_format, speed

This exists so that anything already speaking OpenAI — openai-python, Open
WebUI, a shell script written against api.openai.com — can be pointed at this
service by changing a base URL. It is a translation layer, not the interface:
OpenAI's body has no field for segments with pauses, and its response has no
field for the realtime factor, so `/speak` stays the route worth preferring.

The interesting decision here is the voice table. Kokoro ships 54 voices and
borrowed OpenAI's names for five of them — af_alloy, am_echo, am_onyx, af_nova
and the British bm_fable — so five of the six names map by name alone and are
not a matter of taste. `shimmer` has no counterpart at all, so it is mapped by
ear, and that one row is a judgement call rather than a fact.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

# The six names an OpenAI client is allowed to send. Native Kokoro names are
# accepted too and take precedence, so nothing is shadowed: `af_nova` reaches
# af_nova whether the alias table exists or not.
VOICE_ALIASES = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    # No af_shimmer exists. af_bella is the closest American female by ear and
    # is one of the few voices Kokoro grades highly, so it carries the name.
    "shimmer": "af_bella",
}

# Kokoro encodes locale in the first letter of the voice name, and every code
# below is one kokoro-onnx accepts. The OpenAI body has no language field, so
# the voice is the only thing left to infer it from — asking for `fable` and
# getting a British voice phonemised as American would be worse than guessing.
LANGUAGE_BY_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


def error_response(status: int, message: str, type_: str, code: str) -> JSONResponse:
    """An error in OpenAI's envelope.

    openai-python reads `error.message` and shows it to the caller; FastAPI's
    own `{"detail": ...}` reaches it as an unparsed body and surfaces as a bare
    status code. Only the /v1 routes use this — changing the native routes'
    error shape would break clients already written against them.
    """
    return JSONResponse(status_code=status,
                        content={"error": {"message": message,
                                           "type": type_,
                                           "code": code}})


def resolve_voice(requested: str | None, available: list[str],
                  default: str) -> str | None:
    """Map a requested voice onto a loaded Kokoro voice, or None if unknown.

    Native names are tried first, so `af_nova` reaches af_nova whatever the
    alias table says.
    """
    name = requested or default
    if name in available:
        return name
    # The table is written here rather than read from the model, so it can name
    # a voice a different voices.bin does not carry. Startup warns about that;
    # at request time it is simply an unknown voice, and saying so is better
    # than substituting one the caller did not ask for.
    target = VOICE_ALIASES.get(name.lower())
    return target if target and target in available else None


def language_for_voice(voice: str, fallback: str) -> str:
    """Infer the phonemiser language from a Kokoro voice name."""
    # Only for names in Kokoro's own <locale><gender>_<name> form. Anything
    # else keeps the configured default rather than being guessed at.
    if len(voice) > 3 and voice[1] in "fm" and voice[2] == "_":
        return LANGUAGE_BY_PREFIX.get(voice[0], fallback)
    return fallback


def unmapped_aliases(available: list[str]) -> list[str]:
    """Alias targets missing from the loaded voice set, for the startup log."""
    return sorted(name for name, target in VOICE_ALIASES.items()
                  if target not in available)
