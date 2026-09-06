"""Kokoro speech synthesis.

One model, 82M parameters, 310 MB of shared weights. A voice is a separate
510 KB embedding tensor — so switching voices costs nothing after load, and a
request may use a different voice per segment if it wants. Only swapping the
model itself is expensive, and there is only one model.

**The chunking is this module's, not kokoro-onnx's, and that is a fix rather
than a preference.** Upstream `_split_phonemes` bounds the phoneme STRING and
breaks only on `[.,!?;]`, so a long run between two full stops arrives at
`_create_audio` as one oversized batch, is truncated to exactly 510 phonemes,
tokenises to exactly 510 tokens, and then indexes a voice tensor of shape
(510, 1, 256) at row 510. Measured: 400 characters of unpunctuated English
phonemise to 518 symbols and return HTTP 500, "index 510 is out of bounds for
axis 0 with size 510", from ordinary prose well inside the 4096-character
input the schema allows. `plan()` below bounds every chunk at 509 and splits a
punctuation-free run at a space, so the row index can never reach 510.

Splitting here rather than inside `create()` is also what makes streaming
possible at all: the route gets a list of chunks it can synthesise one at a
time and send as it goes, instead of one call that returns when the last word
is finished.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
from voice_common.audio import SAMPLE_RATE, check_rate, splice

log = logging.getLogger("tts-stack.synth")

# Re-exported so app.main keeps importing the rate from the module that
# produces the audio, rather than reaching past it into the shared package.
__all__ = ["SAMPLE_RATE", "MAX_CHUNK_PHONEMES", "Synth", "chunk_phonemes",
           "ramp_chunks"]

# kokoro_onnx.config.MAX_PHONEME_LENGTH is 510 and a voice tensor has 510 rows,
# so 510 tokens index row 510 and raise. 509 is the largest count that cannot.
MAX_CHUNK_PHONEMES = 509

# The model emits audio in 600-sample frames at 24 kHz — 25 ms. Measured, not
# assumed: the greatest common divisor of five untrimmed outputs of different
# lengths is exactly 600. It is the only defensible unit for the `output_tokens`
# the SSE done event has to report; see app/main.py.
FRAME_SAMPLES = 600

# Kokoro splits on these and so does this module, because a seam at a mark the
# model was going to pause on is the one place a seam is inaudible.
_BREAKS = ".,!?;"


def _wire_espeak() -> None:
    """Point the phonemiser at the system espeak-ng.

    The espeakng-loader wheel hardcodes a path from its own build machine
    (/Users/runner/work/...), which of course does not exist anywhere else.
    Left alone it fails at first synthesis, not at import, so the service
    starts healthy and then breaks on the first request.
    """
    data = os.getenv("ESPEAK_DATA_PATH", "/usr/lib/x86_64-linux-gnu/espeak-ng-data")
    lib = os.getenv("ESPEAK_LIBRARY", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
    if not Path(data).is_dir():
        for candidate in ("/usr/share/espeak-ng-data",
                          "/usr/lib/aarch64-linux-gnu/espeak-ng-data",
                          "/opt/homebrew/opt/espeak-ng/share/espeak-ng-data"):
            if Path(candidate).is_dir():
                data = candidate
                break
    if not Path(lib).is_file():
        for candidate in ("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
                          "/usr/lib/libespeak-ng.so.1",
                          "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib"):
            if Path(candidate).is_file():
                lib = candidate
                break

    import espeakng_loader
    espeakng_loader.get_data_path = lambda: data
    espeakng_loader.get_library_path = lambda: lib
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_data_path(data)
    EspeakWrapper.set_library(lib)
    log.info("espeak-ng: %s", lib)


def _split_long(piece: str, cap: int) -> list[str]:
    """Break a run with no punctuation in it, at spaces where there are any."""
    out: list[str] = []
    while len(piece) > cap:
        cut = piece.rfind(" ", 0, cap + 1)
        # No space either: a single unbroken run of symbols longer than the
        # context window. Cut it. Nothing else fits.
        cut = cut if cut > 0 else cap
        out.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        out.append(piece)
    return out


def chunk_phonemes(phonemes: str, target: int = MAX_CHUNK_PHONEMES,
                   cap: int = MAX_CHUNK_PHONEMES) -> list[str]:
    """Group phonemes into chunks the model can take in one pass.

    The greedy fill is upstream's, deliberately and line for line, because the
    native route's audio must not move: given the same text this returns the
    same batches `kokoro_onnx` would have, so /speak and a buffered
    /v1/audio/speech return the samples they always returned. Verified against
    `Kokoro._split_phonemes` on 400 randomly generated English texts of 3 to
    400 words: identical batches on all 400, and the route's audio is
    bit-identical to what `create()` returned for the same text.

    Two things are added, and only two. Batches are bounded at `cap`, which
    upstream's are not: it appends a punctuation mark to a full batch without
    re-checking the length, and it starts a new batch from a run it has already
    decided is too long, so a long stretch between two full stops arrives at
    the model as 510 tokens and indexes a 510-row voice tensor at row 510. And
    an empty batch is dropped, which upstream emits whenever the first piece is
    over the limit — a model call with nothing in it.

    `target` below `cap` buys latency and costs length. Measured on a
    4096-character input: 509 gives 7 chunks, 154.5 s of audio and 6.21 s to
    the first; 200 gives 17 chunks, 174.5 s and 2.25 s; 100 gives 39 chunks,
    180.8 s and 1.29 s. The 17% growth is the duration predictor seeing less
    context, not silence at the seams, so it is a real change to the speech.
    """
    batches: list[str] = []
    current = ""
    for part in re.split(f"([{re.escape(_BREAKS)}])", phonemes):
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) + 1 > target:
            batches.append(current.strip())
            current = part
        elif part in _BREAKS:
            # No space before a mark, and no length check — upstream's rule,
            # which is why `cap` is enforced below rather than here.
            current += part
        elif current:
            current = f"{current} {part}"
        else:
            current = part
    if current:
        batches.append(current.strip())

    bounded: list[str] = []
    for batch in batches:
        bounded.extend(_split_long(batch, cap) if len(batch) > cap
                       else ([batch] if batch else []))
    return bounded


# A LAST CHUNK THIS SMALL IS A MODEL CALL TO SAY A FULL STOP. One call costs
# about 0.40 s on orko before it has generated anything, and 24 phonemes is
# roughly a second of speech, so a ramped request folds a tail that short back
# into the chunk before it. Ramped requests only: the buffered path must keep
# returning the samples it always returned, and merging two chunks changes
# what the duration predictor sees.
TAIL_MERGE_PHONEMES = 24


def _mark(phonemes: str, mark: str, start: int = 0, end: int | None = None,
          forwards: bool = False) -> int:
    """Where `mark` ENDS A WORD inside phonemes[start:end], or -1.

    A mark with another symbol straight after it is INSIDE a token rather than
    between two words. espeak renders "12,400" as `twˈɛlv,fˈɔːhˈʌndɹɪd`, so a
    chunk that ends at that comma gives "twelve" the falling contour of a
    clause ending and opens the next chunk on "four hundred". That is ordinary
    prose with a number in it, not a corner case, and it is why the mark alone
    is not enough to cut on.

    THE RANGE IS A PAIR OF BOUNDS AND NOT A SLICE, which is the whole reason
    this takes `start` and `end` rather than a substring. What follows the
    mark is read from the WHOLE string: on a slice, a mark sitting at the last
    position looks like the end of the text and is accepted, and the one
    budget where that happened cut `twˈɛlv,` off `fˈɔːhˈʌndɹɪd` exactly as if
    the rule were not there.
    """
    stop = len(phonemes) if end is None else end
    at = start - 1 if forwards else stop
    while True:
        found = (phonemes.find(mark, at + 1, stop) if forwards
                 else phonemes.rfind(mark, start, at))
        if found < 0:
            return -1
        if found + 1 >= len(phonemes) or phonemes[found + 1] == " ":
            return found
        at = found


def _take(phonemes: str, budget: int, cap: int = MAX_CHUNK_PHONEMES,
          stretch: float = 1.5) -> tuple[str, str]:
    """The first `budget`-ish phonemes, and the literal remainder.

    BOTH HALVES ARE SLICES OF THE INPUT, which is the property the ramp turns
    on: head plus rest is the input, always, so nothing is doubled at the seam
    and nothing is eaten. Rebuilding the head with `chunk_phonemes` and then
    slicing the source by its length does NOT have that property, because
    `chunk_phonemes` strips every piece and rejoins a mark with no space
    before it, so the offset is wrong on any input whose phonemes carried
    whitespace the rewrite removed. Measured on six ordinary sentences, three
    of them came back with a phoneme duplicated or a letter eaten at the seam.

    Where the cut falls, in order:

      1. the last word-ending mark inside the budget, provided it is at or
         past half of it. The floor is what stops a full stop becoming a chunk
         of its own;
      2. no usable mark: the first one within `stretch` times the budget, and
         never past `cap`. A seam at a mark is free and the extra wait is a
         fraction of a second. The cap is not a tidy-up: it is the 510th row
         of the voice tensor, and a head that reached past it would be the
         crash this module was written to remove;
      3. still none: the last space. Measured on the deployed service, a cut
         at a space inserts 0.09 to 0.15 s of silence, which is the same order
         as the silence Kokoro already leaves at the end of every utterance,
         and 0.3% of total duration. This is what lets a long unpunctuated
         opening sentence be ramped at all;
      4. an unbroken run of symbols: the budget. Nothing else fits, and this
         is the case `_split_long` already handles the same way.
    """
    if len(phonemes) <= budget:
        return phonemes, ""

    at = max((found for found in
              (_mark(phonemes, mark, end=budget) for mark in _BREAKS)
              if found >= 0), default=-1)
    if at >= budget // 2:
        return phonemes[:at + 1], phonemes[at + 1:].lstrip()

    reach = min(int(budget * stretch), cap)
    hit = min((found for found in
               (_mark(phonemes, mark, start=budget, end=reach, forwards=True)
                for mark in _BREAKS) if found >= 0), default=-1)
    if hit >= 0:
        return phonemes[:hit + 1], phonemes[hit + 1:].lstrip()

    cut = phonemes.rfind(" ", 0, budget + 1)
    if cut <= 0:
        cut = budget
    return phonemes[:cut].rstrip(), phonemes[cut:].lstrip()


def _merge_tail(chunks: list[str], cap: int) -> list[str]:
    """Fold a very short last chunk into the one before it. See above."""
    if len(chunks) < 2 or len(chunks[-1]) > TAIL_MERGE_PHONEMES:
        return chunks
    # No space before a mark, which is the rule `chunk_phonemes` already uses.
    separator = "" if chunks[-1][0] in _BREAKS else " "
    joined = chunks[-2] + separator + chunks[-1]
    return chunks if len(joined) > cap else chunks[:-2] + [joined]


def ramp_chunks(phonemes: str, schedule: list[int],
                cap: int = MAX_CHUNK_PHONEMES) -> list[str]:
    """Leading chunks at the sizes `schedule` asks for, then the tail as usual.

    The schedule comes from the caller rather than from here, because how fast
    the chunks may grow is a question about the machine and about the client
    waiting for them, and this module knows neither. What lives here is the
    cutting: where a chunk of a given size ends without a listener hearing it.

    The remainder goes to the unchanged `chunk_phonemes` at the unchanged
    window, so the bulk of a long request is batched exactly as it is today
    and only the opening seconds are affected.
    """
    chunks: list[str] = []
    rest = phonemes.strip()
    for budget in schedule:
        if len(rest) <= budget:
            break
        head, rest = _take(rest, min(budget, cap), cap)
        if not head:
            break
        chunks.append(head)
    chunks.extend(chunk_phonemes(rest, target=cap) if rest else [])
    return _merge_tail(chunks, cap)


class Synth:
    def __init__(self, model_path: str, voices_path: str) -> None:
        _wire_espeak()
        from kokoro_onnx import Kokoro
        self._k = Kokoro(model_path, voices_path)
        self.voices = sorted(self._k.get_voices())
        log.info("kokoro ready, %d voices", len(self.voices))

    def plan(self, text: str, language: str,
             target: int = MAX_CHUNK_PHONEMES, *,
             ramp: Callable[[int], list[int] | None] | None = None
             ) -> list[str]:
        """Phonemise `text` and split it into chunks the model can take whole.

        Returned rather than synthesised so the caller can count tokens before
        any audio exists — the SSE done event needs that number — and so it can
        decide how much of the work to do before answering.

        Empty or whitespace-only input returns no chunks rather than raising.
        The schema sets no minLength, so `""` is a legal request; it used to
        reach numpy as an empty concatenate and come back 500.

        `ramp` is asked, with the phoneme count, for the sizes of the leading
        chunks, and may answer None for the batching this always did. It is
        KEYWORD-ONLY and defaults to None on purpose: `speak` and through it
        `speak_segments` call this positionally, and a ramp that reached them
        would change /speak's audio, which this module's docstring promises is
        what `create()` returned.
        """
        if not text.strip():
            return []
        phonemes = self._k.tokenizer.phonemize(text, language)
        schedule = ramp(len(phonemes)) if ramp else None
        if schedule:
            return ramp_chunks(phonemes, schedule, cap=target)
        return chunk_phonemes(phonemes, target=target)

    def token_count(self, chunks: list[str]) -> int:
        """Model input tokens across chunks — the vocabulary's own count."""
        return sum(len(self._k.tokenizer.tokenize(chunk)) for chunk in chunks)

    def speak_chunk(self, phonemes: str, voice: str, language: str,
                    speed: float) -> np.ndarray:
        """One planned chunk, synthesised. `is_phonemes` skips a second pass."""
        audio, rate = self._k.create(phonemes, voice=voice, speed=speed,
                                     lang=language, is_phonemes=True)
        check_rate(rate)
        return audio.astype(np.float32)

    def stream(self, text: str, voice: str, language: str, speed: float,
               target: int = MAX_CHUNK_PHONEMES) -> Iterator[np.ndarray]:
        """Chunks of audio, yielded as each is generated rather than at the end."""
        for phonemes in self.plan(text, language, target):
            yield self.speak_chunk(phonemes, voice, language, speed)

    def speak(self, text: str, voice: str, language: str,
              speed: float) -> np.ndarray:
        """The whole utterance, through the same chunking `stream` uses.

        It goes through `plan` rather than straight to `create` so that the
        native route cannot hit the out-of-bounds crash either — it is the same
        synthesiser and the same defect, and a 500 on 400 characters of
        unpunctuated prose is no more acceptable on /speak than on /v1. At the
        default chunk size the batches are the ones upstream would have made,
        so the audio is unchanged.

        The rate check that used to live here now sits in `speak_chunk`, which
        is the call that reaches the model: kokoro is 24 kHz, and a model that
        changed it would ship every file with the wrong rate in its header and
        play at the wrong pitch.
        """
        parts = [self.speak_chunk(phonemes, voice, language, speed)
                 for phonemes in self.plan(text, language)]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def speak_segments(self, segments: list[tuple[str, float, str]],
                       language: str, speed: float) -> np.ndarray:
        """Synthesise each (text, pause_after, voice) and splice in silence.

        Pauses are generated here rather than asked of the model, because no
        TTS model reliably produces a beat you can act inside. Punctuation
        buys a breath; an instruction needs a gap. Measured by ear on the same
        voice and words, inserted silence is what separates audio that sounds
        like instructions from audio that sounds like narration.

        The voice arrives per segment, already resolved by the caller. It used
        to be one voice for the whole call, which quietly made a documented
        per-segment `voice` a lie; there was never a cost to honouring it, a
        voice being a 510 KB embedding over weights that are already resident.

        The splicing itself is voice_common.audio.splice: an empty segment
        contributes its pause and no audio, and a request of nothing but
        pauses returns zeros(0) rather than raising, exactly as this did.
        """
        pieces = [splice([(self.speak(text, voice, language, speed)
                           if text.strip() else None, pause_after)])
                  for text, pause_after, voice in segments]
        return (np.concatenate(pieces) if pieces
                else np.zeros(0, dtype=np.float32)), _offsets(pieces)


def _offsets(pieces: list[np.ndarray]) -> list[float]:
    """Where each segment STARTS, in seconds, from the samples it produced.

    Exact, not estimated. Each segment is synthesised and spliced on its own
    above, so its length is known at the moment it is made -- the running total
    is the boundary. This is the number a client needs to follow the text as it
    plays, and it was being computed and thrown away.

    duration x (characters so far / characters total) is the alternative and it
    is wrong from the first sentence: the inserted pause after a segment is a
    fixed number of seconds regardless of its length, and speech rate moves
    with punctuation.
    """
    out, running = [], 0
    for piece in pieces:
        out.append(round(running / SAMPLE_RATE, 3))
        running += piece.size
    return out
