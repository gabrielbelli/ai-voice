"""Post-transcription term repair: the mechanism, not the vocabulary.

Parakeet is a CTC/TDT model. Unlike Whisper it takes no `hotwords` and no
`initial_prompt`, so nothing here can steer the decoder. The repair happens
after decoding, on the text.

This is cruder than decoder biasing — it cannot recover a term the acoustic
model never got close to — but it costs nothing and it fixes the failure that
actually matters: a correctly-heard word mapped to the wrong spelling.

TWO KINDS OF RULE, FROM TWO KINDS OF LINE
-----------------------------------------
A glossary profile writes `heard = intended` for a replacement and a bare term
for a hotword, and only the first has ever compiled to a rule here. A bare term
has no `heard` side, so there is nothing to rewrite FROM — which is why a
profile's hotword-only section does nothing on an engine that takes no
vocabulary, and why dictation.txt says so in its own header.

A request's `prompt` and `keywords[]` are bare terms with no other expression
available: a caller cannot write `heard = intended` in a `prompt`, and refusing
the field left the engine this service actually deploys with no way to honour
it at all. `term_rules` is what a bare term CAN do to a finished transcript —
repair its own spelling — and it is deliberately the weaker half. See its
docstring for what it recovers and what it cannot.

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
from collections.abc import Iterable

# A bare term shorter than this is not compiled into a repair rule. It is a
# length floor, not a dictionary — the same honest limitation
# profiles._could_occur_innocently already writes down — but it is the floor
# that keeps `\bus\b -> "US"` and `\bit\b -> "IT"` from rewriting ordinary
# sentences on a two-letter prompt. NVIDIA's context-biasing work lands on the
# same floor from the other end — terms under three characters "lead to an
# excessive number of false accepts and, therefore, should not be included in
# the context-biasing list" (arXiv:2406.07096 §4) — which is the same
# observation about the same shape, made about a decoder rather than a regex.
MIN_TERM_CHARS = 3


def compile_rules(terms: dict[str, str]) -> list[tuple[re.Pattern[str], str]]:
    """Longest key first, so multi-word entries are not pre-empted by their
    own first word."""
    return [
        (re.compile(rf"\b{re.escape(heard)}\b", re.IGNORECASE), intended)
        for heard, intended in sorted(terms.items(), key=lambda kv: -len(kv[0]))
    ]


def term_rules(terms: Iterable[str]) -> list[tuple[re.Pattern[str], str]]:
    """Case repair for BARE terms — the only rewrite a hotword can express.

    A bare term spells its own intended form and names no wrong one, so the
    only rule derivable from it is over its own letters: wherever the
    transcript already contains the term, spell it the way the term was
    written. That recovers the failure the default engine really has on a
    proper noun it DID hear — Parakeet returns "ghost pepper" and "theoria" in
    lower case, every phoneme right and only the capitalisation wrong.

    It cannot recover a word the acoustic model never approached. "Antropic"
    stays "Antropic", because no rule here names "Antropic". That is the same
    limit the module header states, and the route's own comments say it rather
    than implying a prompt does on Parakeet what it does on Whisper.

    Two shapes are skipped, and both skips are the rule rather than tidying:

      * a term that is already entirely lower case. Its rule is either a no-op
        (`kubectl` -> `kubectl`) or actively harmful: `\\bsync\\b -> "sync"`
        LOWERCASES a correct sentence-initial "Sync". The shipped profiles are
        full of these — `commit`, `nginx`, `kubectl` — which is what makes the
        skip worth naming rather than assuming.
      * a term shorter than MIN_TERM_CHARS.

    A term in a script with no case — `日本語` — is skipped by the first of
    those, and correctly: there is no capitalisation to repair, so its rule
    would rewrite the term to itself. What such a term needs is decoder
    biasing, which this engine does not have and this function does not
    pretend to be.

    Skipping is silent because a term that produced no rule still reached the
    decoder on Whisper and is still not a field that was accepted and dropped:
    what a term did is reported by X-Glossary-Repaired, which names the terms
    that actually rewrote something.
    """
    return compile_rules({
        term.lower(): term
        for term in terms
        if len(term) >= MIN_TERM_CHARS and term != term.lower()
    })


def apply(text: str, rules: list[tuple[re.Pattern[str], str]]) -> tuple[str, list[str]]:
    """Return the repaired text and the list of terms that actually fired.

    The caller surfaces that list. A silent substitution is worse than no
    substitution: if a rule is wrong, you want to see it named.
    """
    fired: list[str] = []
    for pattern, intended in rules:
        repaired, count = pattern.subn(intended, text)
        # A match is not a change. Every rule matches case-insensitively, so a
        # rule fires against text that was already right — and a term_rules
        # rule is BUILT from its own output, so it matches every time the
        # decoder spelled the term correctly. Counting matches would make
        # X-Glossary-Repaired name terms nothing happened to, which is the
        # same false report as a silent substitution with the sign flipped.
        if count and repaired != text:
            fired.append(intended)
        text = repaired
    return text, fired
