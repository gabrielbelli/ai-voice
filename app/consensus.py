"""Word-level comparison of two transcripts.

Where two models from different families agree, the word is almost certainly
right. Where they disagree, it is almost certainly a proper noun, an acronym
or a piece of jargon — the exact words worth doubting, and the exact words a
transcript consumer can resolve from context.

The output marks those spans inline rather than silently choosing:

    I need to make a <comet|commit> on the dashboard

This is what replaces an LLM cleanup stage. It flags uncertainty instead of
inventing confidence, which is the failure mode a small local model cannot
avoid.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_WORD = re.compile(r"\S+")


def _norm(word: str) -> str:
    """Compare on letters and digits only. A disagreement about a comma is not
    a disagreement worth surfacing."""
    return re.sub(r"[^\w]", "", word).casefold()


def merge(primary: str, secondary: str, marker: str = "<{a}|{b}>") -> tuple[str, list[dict[str, str]]]:
    """Return the primary transcript with disagreements marked, and the list
    of disagreements.

    The primary text is authoritative throughout — the secondary model never
    replaces a word, it only casts doubt on one. Its transcript is the second
    opinion, not a vote with equal weight.
    """
    a_words = _WORD.findall(primary)
    b_words = _WORD.findall(secondary)
    if not a_words or not b_words:
        return primary, []

    matcher = SequenceMatcher(
        None, [_norm(w) for w in a_words], [_norm(w) for w in b_words], autojunk=False
    )

    out: list[str] = []
    disagreements: list[dict[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            out.extend(a_words[i1:i2])
            continue
        mine = " ".join(a_words[i1:i2])
        theirs = " ".join(b_words[j1:j2])
        if not mine:
            # The secondary heard something the primary dropped entirely.
            # Worth flagging, but never inserted — the primary stays the text.
            disagreements.append({"primary": "", "secondary": theirs})
            continue
        if not theirs:
            out.append(mine)
            disagreements.append({"primary": mine, "secondary": ""})
            continue
        out.append(marker.format(a=mine, b=theirs))
        disagreements.append({"primary": mine, "secondary": theirs})

    return " ".join(out), disagreements


def agreement(primary: str, secondary: str) -> float:
    """Fraction of primary words the secondary model also produced. A low
    score means one of the models is having a bad time; both transcripts
    deserve suspicion, not just the marked spans."""
    a = [_norm(w) for w in _WORD.findall(primary)]
    b = [_norm(w) for w in _WORD.findall(secondary)]
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()
