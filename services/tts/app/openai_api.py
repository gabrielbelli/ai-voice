"""The OpenAI shape, translated onto Kokoro.

    POST /v1/audio/speech   model, input, voice, response_format, speed

This exists so that anything already speaking OpenAI — openai-python, Open
WebUI, a shell script written against api.openai.com — can be pointed at this
service by changing a base URL. It is a translation layer, not the interface:
OpenAI's body has no field for segments with pauses, and its response has no
field for the realtime factor, so `/speak` stays the route worth preferring.

The interesting decision here is the voice table. OpenAI publishes thirteen
built-in names. Kokoro ships 54 voices and borrowed OpenAI's names for five of
them — af_alloy, am_echo, am_onyx, af_nova and the British bm_fable — so five
rows map by name alone and are not a matter of taste. The other eight have no
counterpart in the model and are mapped by ear against OpenAI's own description
of each voice, which makes them judgement calls rather than facts. They are
marked as such in the table and in the README.

Six of the thirteen used to be here and the other seven came back 400. That was
never a limitation: with 54 voices loaded, the missing names were a table that
had not been extended, and a client written against the published enum had no
way to know which half of it this service would take.

The error envelope this route answers in is no longer written here. It lives in
voice_common.errors, which every service in the estate now shares — four
incompatible copies of the same four-key JSON object, two of them passing
`code` and `type_` positionally in opposite orders, is what that move was for.
"""

from __future__ import annotations

# The thirteen names an OpenAI client is allowed to send, in the order the
# schema lists them. Native Kokoro names are accepted too and take precedence,
# so nothing is shadowed: `af_nova` reaches af_nova whether the alias table
# exists or not.
#
# Five rows are Kokoro's own borrowings and are exact. The eight marked below
# are mapped by ear from OpenAI's published description of the voice, weighted
# towards the voices Kokoro itself grades highly — af_heart is its only A.
VOICE_ALIASES = {
    "alloy": "af_alloy",
    "ash": "am_fenrir",        # by ear: firm, darker male
    "ballad": "bm_lewis",      # by ear: British male, the expressive one
    "coral": "af_sarah",       # by ear: bright, warm female
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "sage": "af_heart",        # by ear: calm and measured, and Kokoro's best
    "shimmer": "af_bella",     # by ear: warm American female
    "verse": "am_puck",        # by ear: expressive male
    "marin": "af_aoede",       # by ear: the newer, plainer female
    "cedar": "am_michael",     # by ear: the newer, plainer male
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


def custom_voice_id(value: object) -> object:
    """Unwrap OpenAI's `{"id": "voice_1234"}` voice object to its string.

    `VoiceIdsOrCustomVoice` is `anyOf[string, {id: string}]`, and the reference
    client sends the object form on its minimal call — which this service used
    to answer with `400 voice: Input should be a valid string`. Normalising
    before validation rather than typing the field as a union keeps every
    error's `loc` flat, so the envelope's `param` stays "voice" instead of
    "voice.str".

    Anything else is passed through untouched for pydantic to judge.
    """
    if isinstance(value, dict) and "id" in value:
        return value["id"]
    return value


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
