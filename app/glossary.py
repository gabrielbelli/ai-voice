"""Post-transcription term repair.

Parakeet is a CTC/TDT model. Unlike Whisper it takes no `hotwords` and no
`initial_prompt`, so there is no way to bias the decoder toward a vocabulary.
The repair therefore happens after decoding, on the text.

This is cruder than decoder biasing — it cannot recover a term the acoustic
model never got close to — but it costs nothing and it fixes the failure that
actually matters: a correctly-heard word mapped to the wrong spelling.
"""

from __future__ import annotations

import re
from pathlib import Path

# Matched case-insensitively against whole words only, longest first, so that
# "text to speak" wins over a "text" rule. Replacement preserves nothing about
# the original casing: these are proper nouns and identifiers with one correct
# spelling each.
DEFAULT_GLOSSARY: dict[str, str] = {
    "cloud code": "Claude Code",
    "todd code": "Claude Code",
    "clode": "Claude Code",
    "entropic": "Anthropic",
    "ghost paper": "Ghost Pepper",
    "comet": "commit",
    "theory dashboard": "Theoria dashboard",
    "ldr": "TLDR",
    "dts": "STT",
    "tex to speak": "text-to-speech",
    "text to speak": "text-to-speech",
    "open sauce": "open source",
    "11 labs": "ElevenLabs",
    "eleven labs": "ElevenLabs",
    "ghosty": "Ghostty",
}


def load(path: str | Path | None) -> dict[str, str]:
    """Merge a `heard = intended` file over the defaults.

    Lines that are blank or start with # are ignored. A missing file is not an
    error — the defaults are a usable glossary on their own.
    """
    terms = dict(DEFAULT_GLOSSARY)
    if not path:
        return terms
    p = Path(path)
    if not p.is_file():
        return terms
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        heard, intended = line.split("=", 1)
        terms[heard.strip().lower()] = intended.strip()
    return terms


def compile_rules(terms: dict[str, str]) -> list[tuple[re.Pattern[str], str]]:
    """Longest key first, so multi-word entries are not pre-empted by their
    own first word."""
    return [
        (re.compile(rf"\b{re.escape(heard)}\b", re.IGNORECASE), intended)
        for heard, intended in sorted(terms.items(), key=lambda kv: -len(kv[0]))
    ]


def apply(text: str, rules: list[tuple[re.Pattern[str], str]]) -> tuple[str, list[str]]:
    """Return the repaired text and the list of terms that actually fired.

    The caller surfaces that list. A silent substitution is worse than no
    substitution: if a rule is wrong, you want to see it named.
    """
    fired: list[str] = []
    for pattern, intended in rules:
        text, count = pattern.subn(intended, text)
        if count:
            fired.append(intended)
    return text, fired
