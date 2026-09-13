"""Chatterbox synthesis, CPU.

Slow by nature. Measured on an M2 Max at 4, 8 and 16 threads it held around
0.21x realtime with under 5% spread — it is not thread-bound, because
autoregressive token generation is sequential and cores cannot parallelise it.
Peak RSS was 6.5-6.8 GB. The deployed instance re-measured at 0.217x on
2026-09-01, which is the same number a year later.

So this is a batch service, not an interactive one. A ten-minute recording
takes roughly three quarters of an hour to produce. It exists because Kokoro,
which is twenty times faster and twenty times lighter, is a small model and
sounds like one on long-form material.

The model is loaded lazily and unloaded after an idle timeout, because 6.5 GB
resident is not something to leave sitting on a shared host between jobs.

**Nothing here is handed more than one chunk of text.** generate() stops after
1000 speech tokens, which is 40 seconds of audio, and says nothing when it
does — see app/chunking.py for the measurement. Splitting is the caller's job;
this module's job is one piece at a time, plus the silence between them.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from voice_common.audio import SAMPLE_RATE, check_rate, splice

from .engines import (DEFAULT_ENGINE, ENGINES, assert_named_checkpoint,
                      assert_runtime, generate_kwargs)

log = logging.getLogger("tts-long.synth")

# Re-exported so app.main keeps importing it from here. The constant itself is
# voice_common.audio's: 24 kHz is OpenAI's headerless `pcm` rate and the native
# rate of both Chatterbox and Kokoro, so it is a wire fact rather than a
# property of this model file.
__all__ = ["SAMPLE_RATE", "SPEECH_TOKEN_RATE", "SUPPORTED_LANGUAGES", "Spoken",
           "Synth", "speech_tokens"]

# chatterbox/models/s3tokenizer/s3tokenizer.py:18 — S3_TOKEN_RATE = 25. One
# speech token is 40 ms of audio, which is what makes `output_tokens` in the
# SSE `speech.audio.done` event a count rather than a guess, and what puts
# generate()'s max_new_tokens=1000 exactly 40 seconds from the start.
SPEECH_TOKEN_RATE = 25

# THE MULTILINGUAL MODEL'S LANGUAGES. voice_common.engines.CATALOGUE is what
# requests are validated against now, per engine, because a second checkpoint
# speaks one language and validating it against this list would accept
# twenty-two that raise TypeError. This copy stays because it is what the
# cross-check below compares the loaded model against, and
# test_engines.py::test_the_two_language_lists_agree fails the day the two
# drift.
#
# chatterbox/mtl_tts.py:24, SUPPORTED_LANGUAGES. Copied rather than imported
# because importing it drags in torch, and this list is needed to answer a
# request BEFORE anything is loaded: generate() raises ValueError on an
# unsupported language_id, and finding that out in the worker means the caller
# waited in a queue to be told about a typo. Cross-checked against the model
# when it loads, so a chatterbox upgrade that adds a language is a log line
# rather than a mystery 400.
SUPPORTED_LANGUAGES = (
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh")


def speech_tokens(samples: int) -> int:
    """Speech tokens behind `samples` of audio, at the model's own token rate.

    s3gen turns one 25 Hz speech token into 40 ms of 24 kHz audio, so the
    length of what came back is the count of what was generated.
    """
    return int(round(samples / SAMPLE_RATE * SPEECH_TOKEN_RATE))


@dataclass
class Spoken:
    """Audio, and what it cost in the model's own units."""

    audio: np.ndarray
    input_tokens: int


def _pin_reference_to_float32(module) -> None:
    """NEP 50 KILLS EVERY CLONED VOICE ON A NUMPY 2 IMAGE, AND THIS IS THE SEAM.

    `norm_loudness` multiplies a float32 clip by an `np.float64` gain that
    pyloudnorm returns. Under NumPy 1.x the scalar was demoted and the clip
    stayed float32; under NumPy >= 2 the value-based rules are gone, so the
    WHOLE CLIP promotes to float64 -- and s3tokenizer's mel filters are
    float32, so `prepare_conditionals` raises "expected m1 and m2 to have the
    same dtype, but got: float != double" before a single token is generated.

    It is device-independent and it is not hypothetical: spring escapes it only
    by running NumPy 1.x, and chatterbox-tts pins numpy>=2.0.0 for Python 3.13,
    so a fresh image walks straight into it. WITHOUT THIS THERE IS NO LOCAL
    FLOOR, and "every advertised engine works with the runner switched off" is
    a claim rather than a design.

    Re-casting a float32 array is free and a no-op, so the wrapper is correct
    on both NumPy generations rather than conditional on either. `_stub_watermarker`
    below is the same manoeuvre for the same reason: this service does not get
    to choose what is inside the wheel it depends on, only what it does about it.

    NAMED WHEN IT APPLIES AND NAMED WHEN IT DOES NOT. A guard that silently
    fails to attach is the class of defect this whole release is about.
    """
    fn = getattr(module, "norm_loudness", None)
    if fn is None or getattr(fn, "_float32_pinned", False):
        return
    def norm_loudness(*args, **kwargs):  # noqa: ANN001, ANN202
        out = fn(*args, **kwargs)
        if isinstance(out, np.ndarray) and out.dtype != np.float32:
            return out.astype("float32", copy=False)
        return out

    norm_loudness._float32_pinned = True  # type: ignore[attr-defined]
    module.norm_loudness = norm_loudness
    log.info("%s.norm_loudness is pinned to float32; without it a NumPy 2 "
             "image raises a dtype mismatch in prepare_conditionals and every "
             "cloned voice fails", module.__name__)


def _pin_every_loaded_module() -> None:
    """Every chatterbox module that has the name bound, not a guessed path.

    The function is a module-level global at every call site inside the
    package, so rebinding it on the module is what a call actually resolves.
    Which modules exist is a fact about the installed wheel rather than
    something this file may assume, so the loaded ones are walked instead.
    """
    found = False
    for name, module in list(sys.modules.items()):
        if not (name == "chatterbox" or name.startswith("chatterbox.")):
            continue
        if hasattr(module, "norm_loudness"):
            _pin_reference_to_float32(module)
            found = True
    if found or int(np.__version__.split(".")[0]) < 2:
        return
    # NAMED, NOT SILENT, AND ONLY WHERE IT MATTERS. On NumPy 1.x the promotion
    # cannot happen and there is nothing to say. On NumPy 2 a guard that did
    # not attach is the one state worth hearing about: either this build does
    # not normalise the reference clip -- fine -- or the function has moved and
    # every cloned voice is about to fail with a dtype mismatch nobody will
    # connect to this. A guard that quietly fails to attach is the class of
    # defect this whole release is about.
    log.warning("running on NumPy %s and no chatterbox module exports "
                "norm_loudness, so the float32 guard did not attach. If cloned "
                "voices fail with \"expected m1 and m2 to have the same dtype\", "
                "the reference-clip normalisation has moved and "
                "_pin_reference_to_float32 needs to follow it.", np.__version__)


def _stub_watermarker() -> None:
    """Neutralise resemble-perth.

    It imports pkg_resources, removed in Python 3.14, and it only stamps an
    inaudible watermark. Stubbing keeps the dependency from deciding which
    interpreter this image runs.
    """
    import perth

    class _NoWatermark:
        def apply_watermark(self, wav, sample_rate=None, **kw):  # noqa: ANN001
            return wav

    perth.PerthImplicitWatermarker = _NoWatermark


class Synth:
    """Loads on first use, unloads after `idle_timeout` seconds of quiet."""

    def __init__(self, idle_timeout: float = 600.0, threads: int = 8,
                 spec=None) -> None:
        self.idle_timeout = idle_timeout
        self.threads = threads
        # WHICH CHECKPOINT THIS ONE IS. Default rather than required, because
        # every existing caller and every test builds a Synth without one and
        # means the engine there has always been.
        self.spec = spec if spec is not None else ENGINES[DEFAULT_ENGINE]
        self._model = None
        self._last_used = 0.0
        # HOW LONG THE LAST LOAD TOOK, MEASURED HERE RATHER THAN ASSUMED.
        # Turbo's cold load is 67.5 s against the multilingual model's 22.2 s,
        # and that difference lands inside whatever a caller is waiting on.
        self.load_seconds = 0.0
        # HOW MANY TIMES THIS CHECKPOINT HAS COME OFF DISK IN THIS PROCESS. A
        # job charges a cold load only when it caused one, and comparing the
        # DURATION would call two identical loads one load.
        self.loads = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._reaper, daemon=True).start()

    # ---------------------------------------------------------------- load --

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _stub_watermarker()
        os.environ.setdefault("OMP_NUM_THREADS", str(self.threads))
        import torch

        torch.set_num_threads(self.threads)
        # THE CLASS COMES OFF THE CATALOGUE, not out of a branch on a name.
        # "module:Class", imported here and nowhere else, so a third engine is
        # a row in voice_common.engines and no code in this file.
        module_name, _, class_name = self.spec.local_class.partition(":")
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        # THE CHECKPOINT THAT LOADS IS THE ONE THE ROW NAMED, ASKED BEFORE THE
        # 68 SECONDS AND THE 6.5 GB ARE SPENT ON IT. The line above is the last
        # of the four single-token edits that answer a turbo request with
        # baseline audio while the header, the record and engine_reason all go
        # on saying turbo -- and it is the one no test can see from outside,
        # because every test in this tree fakes `Synth._speak` and nothing is
        # ever loaded. See engines.assert_named_checkpoint for why this one
        # raises where assert_runtime below only warns.
        assert_named_checkpoint(self.spec, cls)
        _pin_every_loaded_module()

        t = time.monotonic()
        log.info("loading %s on cpu, %d threads", self.spec.id, self.threads)
        self._model = cls.from_pretrained(device="cpu")
        self.load_seconds = round(time.monotonic() - t, 1)
        self.loads += 1
        log.info("loaded %s in %.0fs", self.spec.id, self.load_seconds)

        # WHAT THE INSTALLED PACKAGE ACTUALLY DOES, against what the catalogue
        # claims, checked once here rather than one wrong job at a time.
        assert_runtime(self.spec, cls)

        # The list this service validates requests against is a copy, so say
        # so the moment the real one disagrees rather than answering 400 to a
        # language the model would have accepted. Only for an engine that has
        # a language list to disagree with: a single-language checkpoint
        # publishes none, and comparing against an absent one would report
        # twenty-three languages of drift on a model that has no language
        # conditioning at all.
        model_languages = getattr(module, "SUPPORTED_LANGUAGES", None)
        if model_languages is not None and len(self.spec.languages) > 1:
            drift = set(model_languages) ^ set(self.spec.languages)
            if drift:
                log.warning("%s language list has moved: %s. Requests are "
                            "validated against voice_common.engines.CATALOGUE, "
                            "which needs updating.",
                            self.spec.id, ", ".join(sorted(drift)))
        # 24 kHz is asserted in every wav header this service writes and in the
        # `pcm` contract. A model update that changed it would ship every file
        # at the wrong pitch, playable and wrong, with nothing reporting an
        # error. voice_common.audio.check_rate exists for exactly this and had
        # never been called from here.
        #
        # AGAINST THE CATALOGUE'S FIGURE FOR THIS ENGINE, not against the
        # module constant. They are the same 24000 for everything shipped and
        # the difference is what stops the next engine being compared with the
        # last one's rate -- which is the exact shape of the bug in the
        # voxtral-int4 repo, where post-processing resamples to 48000 and three
        # of the four writers then label the result 24000.
        check_rate(int(getattr(self._model, "sr", SAMPLE_RATE)),
                   self.spec.facts.native_sample_rate)

    def _reaper(self) -> None:
        while not self._stop.wait(30.0):
            with self._lock:
                idle = time.monotonic() - self._last_used
                if self._model is not None and self._last_used and idle > self.idle_timeout:
                    log.info("unloading after %.0fs idle", idle)
                    self._model = None
                    import gc

                    gc.collect()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def close(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- generate --

    def _count_tokens(self, text: str, language: str) -> int:
        """Text tokens for `text`, from the model's own tokeniser.

        Called with the lock held and the model loaded. `usage.input_tokens` in
        the SSE done event is required to be an integer, and the only honest
        integer is the one the model actually tokenised — generate() runs
        punc_norm first, so this does too.

        The fallback is four characters to a token, the ratio every OpenAI
        tokeniser lands near on English prose. It is an approximation and it is
        logged as one; it is reached only if a chatterbox upgrade moves the
        tokeniser, and reporting a slightly wrong count beats failing a
        synthesis that already succeeded.
        """
        try:
            from chatterbox.mtl_tts import punc_norm

            kwargs = ({"language_id": language.lower() if language else None}
                      if (len(self.spec.languages) > 1
                          and not self.spec.facts.language_from_voice) else {})
            tokens = self._model.tokenizer.text_to_tokens(  # type: ignore[union-attr]
                punc_norm(text), **kwargs)
            return int(tokens.shape[-1])
        except Exception:  # noqa: BLE001 - usage must not fail a synthesis
            log.warning("tokeniser unavailable; input_tokens is approximated "
                        "from character count", exc_info=True)
            return max(1, math.ceil(len(text) / 4))

    def speak(self, text: str, language: str, controls: dict,
              reference: str | None = None) -> np.ndarray:
        """One chunk. Anything over 40 seconds of speech is truncated: chunk it."""
        return self._speak(text, language, controls, reference).audio

    def _speak(self, text: str, language: str, controls: dict,
               reference: str | None) -> Spoken:
        with self._lock:
            self._ensure_loaded()
            tokens = self._count_tokens(text, language)
            # ONLY THE FIELDS THIS CHECKPOINT READS, and absent rather than
            # None for the rest. Turbo accepts `exaggeration` and `cfg_weight`
            # as keyword arguments, logs a warning and discards them -- so
            # passing them would be this service quietly delivering audio that
            # ignored two fields, which is the one thing it must never do.
            wav = self._model.generate(  # type: ignore[union-attr]
                text,
                audio_prompt_path=reference,
                **generate_kwargs(self.spec, language=language,
                                  controls=controls),
            )
            self._last_used = time.monotonic()
        audio = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        return Spoken(audio=audio, input_tokens=tokens)

    def speak_segments(self, segments: list[tuple[str, float]], language: str,
                       controls: dict,
                       reference: str | None = None,
                       on_chunk: Callable[[np.ndarray], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> Spoken:
        """Synthesise each segment, inserting real silence between them.

        `controls` IS ONE MAPPING AND IT USED TO BE THREE POSITIONAL FLOATS.
        The three were the three fields Chatterbox reads, spelled into this
        signature, into RemoteSynth's identical one, into `_run`'s positional
        call and into `_vendor_fields` -- four places to widen for a fourth
        field, and `_run` calls this positionally on whichever backend it was
        handed, so a signature that drifts between the two is a defect nothing
        would catch until a job ran on the lane nobody tested. A dict is the
        same house rule one level up: which keys exist is read off the engine,
        and an engine with no such control contributes no key.

        Pauses are generated here rather than asked of the model. No TTS model
        reliably produces a beat you can act inside — punctuation buys a
        breath, an instruction needs a gap.

        The splicing itself is voice_common.audio.splice, which tts-stack also
        calls: a segment whose text is empty contributes its pause and nothing
        else, and a request of nothing but pauses returns silence rather than
        raising. It is applied per segment here so the piece handed to
        `on_chunk` is exactly the piece that lands in the finished array —
        concatenating what a stream emitted and splicing the whole list are the
        same bytes.

        `on_chunk` is called from this thread, in order, as each segment
        finishes: that call is the only reason a stream can start before the
        whole job does. `cancelled` is polled between segments, which is as
        fine-grained as cancellation gets — generate() has no interruption
        point inside it.
        """
        parts: list[np.ndarray] = []
        total_tokens = 0
        for text, pause_after in segments:
            if cancelled is not None and cancelled():
                log.info("cancelled after %d of %d segments",
                         len(parts), len(segments))
                break
            if text.strip():
                spoken = self._speak(text, language, controls, reference)
                total_tokens += spoken.input_tokens
                piece = splice([(spoken.audio, pause_after)])
            else:
                piece = splice([(None, pause_after)])
            if piece.size:
                parts.append(piece)
                if on_chunk is not None:
                    on_chunk(piece)
        audio = (np.concatenate(parts) if parts
                 else np.zeros(0, dtype=np.float32))
        return Spoken(audio=audio, input_tokens=total_tokens)
