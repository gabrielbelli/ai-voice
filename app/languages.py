"""ISO-639-1 code to the English name of the language.

`verbose_json.language` is "the language of the input audio", and the
specification's own example spells it `"english"` rather than `"en"` — that is
what whisper-1 returns, and a client that renders the field puts the word in
front of a person. faster-whisper reports the code, so the map lives here.

These are Whisper's own ninety-nine, in Whisper's own spelling, because the
value has to round-trip: a client may send `language=en` and read `english`
back, and inventing a nicer name for either end would break that pair.
"""

from __future__ import annotations

NAMES: dict[str, str] = {
    "af": "afrikaans", "am": "amharic", "ar": "arabic", "as": "assamese",
    "az": "azerbaijani", "ba": "bashkir", "be": "belarusian", "bg": "bulgarian",
    "bn": "bengali", "bo": "tibetan", "br": "breton", "bs": "bosnian",
    "ca": "catalan", "cs": "czech", "cy": "welsh", "da": "danish",
    "de": "german", "el": "greek", "en": "english", "es": "spanish",
    "et": "estonian", "eu": "basque", "fa": "persian", "fi": "finnish",
    "fo": "faroese", "fr": "french", "gl": "galician", "gu": "gujarati",
    "ha": "hausa", "haw": "hawaiian", "he": "hebrew", "hi": "hindi",
    "hr": "croatian", "ht": "haitian creole", "hu": "hungarian",
    "hy": "armenian", "id": "indonesian", "is": "icelandic", "it": "italian",
    "ja": "japanese", "jw": "javanese", "ka": "georgian", "kk": "kazakh",
    "km": "khmer", "kn": "kannada", "ko": "korean", "la": "latin",
    "lb": "luxembourgish", "ln": "lingala", "lo": "lao", "lt": "lithuanian",
    "lv": "latvian", "mg": "malagasy", "mi": "maori", "mk": "macedonian",
    "ml": "malayalam", "mn": "mongolian", "mr": "marathi", "ms": "malay",
    "mt": "maltese", "my": "myanmar", "ne": "nepali", "nl": "dutch",
    "nn": "nynorsk", "no": "norwegian", "oc": "occitan", "pa": "punjabi",
    "pl": "polish", "ps": "pashto", "pt": "portuguese", "ro": "romanian",
    "ru": "russian", "sa": "sanskrit", "sd": "sindhi", "si": "sinhala",
    "sk": "slovak", "sl": "slovenian", "sn": "shona", "so": "somali",
    "sq": "albanian", "sr": "serbian", "su": "sundanese", "sv": "swedish",
    "sw": "swahili", "ta": "tamil", "te": "telugu", "tg": "tajik",
    "th": "thai", "tk": "turkmen", "tl": "tagalog", "tr": "turkish",
    "tt": "tatar", "uk": "ukrainian", "ur": "urdu", "uz": "uzbek",
    "vi": "vietnamese", "yi": "yiddish", "yo": "yoruba", "yue": "cantonese",
    "zh": "chinese",
}

# What verbose_json reports when the engine that ran detects no language at
# all. Parakeet v3 takes no language hint and onnx-asr surfaces no language
# from it, so under the default engine there is nothing to report — and
# `language` is a REQUIRED field of the verbose_json body, so something has to
# be said. "unknown" is the one answer that is neither a guess nor an empty
# string a client would read as "the server forgot".
UNKNOWN = "unknown"


def name(code: str | None) -> str:
    """The English name for an ISO-639-1 code, or UNKNOWN."""
    if not code:
        return UNKNOWN
    return NAMES.get(code.lower(), code.lower())


def known(code: str) -> bool:
    """Is this a language code the Whisper path can be pinned to?"""
    return code.lower() in NAMES
