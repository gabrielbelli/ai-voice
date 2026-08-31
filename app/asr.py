"""The two recognisers.

Two models, deliberately from different families, because the whole point is
that they fail differently. An ensemble of two Whispers agrees on its own
mistakes; Whisper and Parakeet do not.

  primary    Whisper large-v3 via CTranslate2. Encoder-decoder, slower, and
             the only one of the two that accepts a vocabulary at decode time
             (`hotwords`). This is where the glossary does real work rather
             than string replacement after the fact.

  secondary  Parakeet TDT 0.6B v3 via ONNX Runtime. CTC/TDT, roughly an order
             of magnitude cheaper, no vocabulary biasing. Its job is not to be
             right; its job is to disagree in the places worth doubting.

Both are optional. With `STT_SECONDARY` empty the service returns the primary
transcript with no consensus pass, which is the right configuration on a host
too small to hold both.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("stt-stack.asr")

SAMPLE_RATE = 16_000


class Primary:
    """Whisper via faster-whisper. Accepts hotwords."""

    def __init__(self, model_id: str, compute_type: str, threads: int,
                 language: str | None, hotwords: str | None) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self.language = language or None
        self.hotwords = hotwords or None
        self._model = WhisperModel(
            model_id, device="cpu", compute_type=compute_type, cpu_threads=threads
        )

    def transcribe(self, samples: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            samples,
            language=self.language,
            beam_size=5,
            # The default temperature ladder, kept on purpose. Pinning
            # temperature=0 disables Whisper's retry on low-confidence output,
            # which is one of the few things that rescues a bad segment.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=False,
            hotwords=self.hotwords,
            vad_filter=False,  # already done, upstream, once, for both models
        )
        return " ".join(s.text.strip() for s in segments).strip()


class Secondary:
    """Parakeet via onnx-asr. No hotwords; not expected to have any."""

    def __init__(self, model_id: str, quantisation: str) -> None:
        import onnx_asr  # noqa: PLC0415

        self._model = onnx_asr.load_model(model_id, quantization=quantisation)

    def transcribe(self, samples: np.ndarray) -> str:
        return self._model.recognize(samples, sample_rate=SAMPLE_RATE).strip()


def build(threads: int, hotwords: str | None) -> tuple[Primary, Secondary | None]:
    primary = Primary(
        model_id=os.getenv("STT_PRIMARY", "large-v3"),
        compute_type=os.getenv("STT_PRIMARY_COMPUTE", "int8"),
        threads=threads,
        language=os.getenv("STT_LANGUAGE") or None,
        hotwords=hotwords,
    )
    log.info("primary ready: %s", os.getenv("STT_PRIMARY", "large-v3"))

    secondary_id = os.getenv("STT_SECONDARY", "istupakov/parakeet-tdt-0.6b-v3-onnx")
    if not secondary_id:
        log.info("secondary disabled — no consensus pass")
        return primary, None

    secondary = Secondary(secondary_id, os.getenv("STT_SECONDARY_QUANT", "int8"))
    log.info("secondary ready: %s", secondary_id)
    return primary, secondary
