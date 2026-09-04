"""Post-transcription term repair: the mechanism, not the vocabulary.

Parakeet is a CTC/TDT model. Unlike Whisper it takes no `hotwords` and no
`initial_prompt`, so there is no way to bias the decoder toward a vocabulary.
The repair therefore happens after decoding, on the text.

This is cruder than decoder biasing — it cannot recover a term the acoustic
model never got close to — but it costs nothing and it fixes the failure that
actually matters: a correctly-heard word mapped to the wrong spelling.

WHAT USED TO BE HERE AND IS NOT ANY MORE
----------------------------------------
A module-level `DEFAULT_GLOSSARY` dict, applied to every transcript whether or
not a glossary file existed, holding `ghost paper = Ghost Pepper`,
`theory dashboard = Theoria dashboard`, `clode = Claude Code` and `ghosty =
Ghostty`. It was the same defect as the shipped glossary.txt — one person's
vocabulary inside a public image — with the extra property that no operator
could turn it off, because it was not in a file to delete. Its general entries
now live in the built-in `dictation` profile and are applied only when a
request asks for them; its project names moved to a deployment-supplied
profile (examples/glossaries/personal.txt).

Which terms exist, where they come from and how they are selected is
profiles.py's job. This file only knows how to compile a mapping and apply it.
"""

from __future__ import annotations

import re


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
