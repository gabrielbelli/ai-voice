"""Split text into pieces one generate() call can actually finish.

This exists for two reasons, and the first one is a bug rather than a feature.

**Chatterbox truncates.** `ChatterboxMultilingualTTS.generate` calls
`t3.inference(..., max_new_tokens=1000)` (chatterbox-tts 0.1.7,
chatterbox/mtl_tts.py:297) and S3 speech tokens run at 25 Hz
(chatterbox/models/s3tokenizer/s3tokenizer.py:18, `S3_TOKEN_RATE = 25`), so a
single call cannot emit more than **40 seconds** of audio no matter how much
text it is handed. Measured on the deployed instance on 2026-09-01: 1690
characters of ordinary prose, which is around 170 seconds of speech, came back
as `audio_seconds: 40.0` after 184.6 s of compute. No error, no warning — the
caller got the first quarter of their text read out and a wav file that ends
mid-sentence. Everything past that point was paid for in CPU and thrown away.

Splitting the input and splicing the pieces is the fix, and it is the only
fix: the ceiling is inside the model's inference loop.

**And it is what makes streaming possible.** generate() is autoregressive over
the whole string it is given and returns nothing until it finishes, so the
first byte of a stream cannot precede the first COMPLETED piece. Chunk size is
therefore the floor on time-to-first-audio, which is why the target below is a
sentence rather than the largest thing that fits.

**The same text, chunked, on the same instance.** 1690 characters submitted as
one string produced 40.0 s of audio in 184.6 s. Submitted as twenty segments it
produced **100.2 s of audio** in 338.1 s — two and a half times as much speech
from the same words, because none of it was thrown away.

Measured speech rate, all four samples:

    65 chars   ->   6.6 s    9.8 chars/s   one short sentence
    336 chars  ->  17.4 s   19.3 chars/s   one passage
    1690 chars -> 100.2 s   16.9 chars/s   twenty segments
    1690 chars ->  40.0 s          n/a     truncated, so it measures nothing

The spread is real: a short utterance carries a fixed half-second or so of
silence at each end, which dominates it. CHARS_PER_SECOND below is the middle
of that range and it is only ever used for estimates and for sizing chunks, so
being wrong in either direction costs an inaccurate `estimated_seconds` rather
than audio. The chunk ceiling is checked against the SLOWEST of the four.
"""

from __future__ import annotations

import os
import re

__all__ = ["CHARS_PER_SECOND", "MAX_CHARS", "TARGET_CHARS", "chunk_text",
           "speech_seconds"]

# Characters of ordinary prose per second of speech. Measured, not assumed —
# see the table above, which spans 9.8 to 19.3 depending on how much of the
# sample is the silence around a single utterance.
#
# 12.0, NOT THE 15 THIS HELD BEFORE. 15 was the middle of that spread, chosen
# when the spread was all there was; a job run against the deployed stack put
# 449 characters of ordinary prose at 37.4 seconds of Chatterbox audio, which
# is 12.0 chars/s. 15 under-predicted the AUDIO by a fifth, and the audio is
# then divided by a realtime factor near 0.27 to get the wait — so the error
# arrives at the reader multiplied by about four. Kokoro measured 16.3 on the
# same host and the same day, which is why the page now keeps one figure per
# engine rather than sharing this one; this file only ever describes
# Chatterbox, because only tts-long imports it.
CHARS_PER_SECOND = float(os.getenv("TTS_CHARS_PER_SECOND", "12"))

# Hard ceiling on a single generate() call. 40 s of audio is the model's own
# limit (see the module docstring). 280 characters is 18.7 s at the rate above
# and 28.6 s at the SLOWEST rate ever measured here — under the ceiling either
# way, which is the property that matters. Sizing it to the mean would put the
# slow case over the edge, and over the edge is silent truncation.
MAX_CHARS = int(os.getenv("TTS_CHUNK_MAX_CHARS", "280"))
# Short sentences are merged up to this, because one generate() per five-word
# sentence restarts the prosody five times as often and costs a fixed overhead
# each time. Above it, sentences stand alone: the smaller the chunk, the sooner
# a stream produces its first sound.
TARGET_CHARS = int(os.getenv("TTS_CHUNK_TARGET_CHARS", "160"))

# A sentence end: terminal punctuation, optional closing quotes or brackets,
# then whitespace. The lookbehind for a lone capital keeps "J. Random" and
# "e.g." from becoming two chunks — an audible break in the middle of a name is
# worse than a chunk twenty characters longer than intended.
_SENTENCE_END = re.compile(
    r"""(?<![A-Z])(?<!\b[Ee]\.g)(?<!\b[Ii]\.e)(?<!\bMr)(?<!\bMrs)(?<!\bMs)
        (?<!\bDr)(?<!\bSt)(?<!\bProf)(?<!\bvs)(?<!\betc)
        ([.!?…。！？]+["'”’)\]]*)\s+""",
    re.VERBOSE)

# Where an over-long sentence may be broken. Clause punctuation first, because
# a break at a comma is a break a listener already expects.
_CLAUSE = re.compile(r"([,;:—–]+)\s+")


def speech_seconds(chars: int) -> float:
    """Estimated seconds of speech for `chars` characters. See CHARS_PER_SECOND.

    Characters rather than words, and that is the fix rather than a taste:
    the estimate this replaced counted whitespace-separated words, so a
    5000-character string with no spaces in it was estimated at two seconds.
    """
    return chars / CHARS_PER_SECOND if CHARS_PER_SECOND > 0 else 0.0


def _split(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split on `pattern`, keeping its captured punctuation on the left part."""
    parts: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        parts.append(text[last:match.end(1)])
        last = match.end()
    tail = text[last:]
    if tail:
        parts.append(tail)
    return [p.strip() for p in parts if p.strip()]


def _hard_wrap(piece: str, limit: int) -> list[str]:
    """Break a single over-long sentence, at clauses first, then at words.

    Reached by a caller who wrote 300 characters without a full stop. Something
    has to give: breaking at a comma is audible but survivable, and the
    alternative — handing the whole thing to generate() — is silent truncation.
    """
    out: list[str] = []
    for clause in _split(piece, _CLAUSE) or [piece]:
        if len(clause) <= limit:
            out.append(clause)
            continue
        words, current = clause.split(), ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > limit:
                out.append(current)
                current = word
            else:
                current = candidate
        # A single word longer than the limit is cut mid-word, which is the
        # one case with no good answer. It is also not a real sentence.
        while len(current) > limit:
            out.append(current[:limit])
            current = current[limit:]
        if current:
            out.append(current)
    return out


def chunk_text(text: str, *, max_chars: int | None = None,
               target_chars: int | None = None) -> list[str]:
    """Split `text` into pieces generate() can finish, longest sensible first.

    Sentences are the unit. Consecutive short ones are merged up to
    `target_chars`; anything still over `max_chars` is broken at clause
    boundaries and then at word boundaries.

    Empty or whitespace-only input returns an empty list rather than a list
    holding one empty string: a caller asking for nothing gets nothing, and
    generate() has an unhelpful opinion about empty text (it substitutes "You
    need to add some text for me to talk", chatterbox/mtl_tts.py:54).
    """
    limit = max_chars or MAX_CHARS
    target = min(target_chars or TARGET_CHARS, limit)

    text = " ".join(text.split())
    if not text:
        return []

    sentences: list[str] = []
    for sentence in _split(text, _SENTENCE_END) or [text]:
        sentences.extend(_hard_wrap(sentence, limit) if len(sentence) > limit
                         else [sentence])

    chunks: list[str] = []
    for sentence in sentences:
        if chunks and len(chunks[-1]) + 1 + len(sentence) <= target:
            chunks[-1] = f"{chunks[-1]} {sentence}"
        else:
            chunks.append(sentence)
    return chunks
