"""The recogniser, chosen at startup.

Parakeet is the default. Measured across 25 conditions and five Brazilian
Portuguese corpora on identical audio, it beat Whisper large-v3 on 21 of them
at roughly seventy times the speed, in a third of the memory:

    Parakeet TDT 0.6B v3   WER 0.144 pt-BR / 0.121 en   47-63x realtime
    Whisper large-v3       WER 0.250 pt-BR / 0.131 en   0.5-0.9x realtime

It also degrades far more gracefully. Band-limiting the audio to 4 kHz — a
cheap or distant microphone — cost Whisper +206% WER on CORAA and Parakeet
+41%. That collapse was a property of Whisper's autoregressive decoder, not a
physical limit.

Whisper remains available, because it genuinely wins on clean read speech and
it is the only one of the two that accepts a vocabulary at decode time.

There is deliberately no second recogniser. A consensus pass was tried and
removed: across every disagreement observed, the second model was the wrong
one, so its dissent carried no information and cost roughly 40% of throughput
— worst on the short clips that dictation actually consists of.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("stt-stack.asr")

SAMPLE_RATE = 16_000

PARAKEET_DEFAULT = "istupakov/parakeet-tdt-0.6b-v3-onnx"
WHISPER_DEFAULT = "large-v3"


class Parakeet:
    """Parakeet TDT via ONNX Runtime. CTC/TDT, so no decode-time vocabulary.

    Terms are repaired after decoding instead (see glossary.py). That is
    weaker than biasing — it cannot recover a word the acoustic model never
    approached — but it is what this runtime offers. FluidAudio implements
    real CTC boosting for the same model, and is CoreML-only.
    """

    accepts_vocabulary = False

    def __init__(self, model_id: str, quantisation: str) -> None:
        import onnx_asr  # noqa: PLC0415

        self._model = onnx_asr.load_model(model_id, quantization=quantisation)

    def transcribe(self, samples: np.ndarray, language: str | None = None,
                   hotwords: str | None = None) -> str:
        # Parakeet v3 detects language itself and takes no hint, and its TDT
        # decoder has no vocabulary argument at all — a per-request prompt
        # arriving from the OpenAI-compatible route cannot be honoured here,
        # only documented.
        del language, hotwords
        return self._model.recognize(samples, sample_rate=SAMPLE_RATE).strip()


class Whisper:
    """Whisper via CTranslate2. Accepts hotwords at decode time."""

    accepts_vocabulary = True

    def __init__(self, model_id: str, compute_type: str, threads: int,
                 language: str | None, hotwords: str | None) -> None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        self.language = language
        self.hotwords = hotwords
        self._model = WhisperModel(
            model_id, device="cpu", compute_type=compute_type, cpu_threads=threads
        )

    def _vocabulary(self, extra: str | None) -> str | None:
        """The configured glossary, plus whatever this request added.

        A request prompt extends the deployment's vocabulary rather than
        replacing it. The glossary is the list of terms this box is known to
        mishear, measured; dropping it because a client named one extra proper
        noun would trade a measured win for a guess.
        """
        return ", ".join(part for part in (self.hotwords, extra) if part) or None

    def transcribe(self, samples: np.ndarray, language: str | None = None,
                   hotwords: str | None = None) -> str:
        segments, _ = self._model.transcribe(
            samples,
            # None means autodetect, which is right for a speaker who
            # code-switches. Pinning the wrong language does not degrade the
            # transcript, it TRANSLATES it: English speech under language="pt"
            # returns fluent Portuguese that reads like a working transcript.
            language=language or self.language,
            beam_size=5,
            # The default temperature ladder, kept deliberately. Pinning
            # temperature=0 disables Whisper's retry on low-confidence output.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=False,
            hotwords=self._vocabulary(hotwords),
            vad_filter=False,  # done once upstream, for whichever model runs
        )
        return " ".join(s.text.strip() for s in segments).strip()


def build(threads: int, hotwords: str | None) -> Parakeet | Whisper:
    """Load the model named by STT_MODEL. Parakeet unless asked otherwise."""
    choice = os.getenv("STT_MODEL", "parakeet").strip().lower()

    if choice in {"parakeet", "parakeet-v3"}:
        model_id = os.getenv("STT_MODEL_ID", PARAKEET_DEFAULT)
        model = Parakeet(model_id, os.getenv("STT_QUANTISATION", "int8"))
        log.info("parakeet ready: %s (no decode-time vocabulary)", model_id)
        return model

    if choice == "whisper":
        model_id = os.getenv("STT_MODEL_ID", WHISPER_DEFAULT)
        model = Whisper(
            model_id=model_id,
            compute_type=os.getenv("STT_QUANTISATION", "int8"),
            threads=threads,
            language=os.getenv("STT_LANGUAGE") or None,
            hotwords=hotwords,
        )
        log.info("whisper ready: %s (hotwords %s)", model_id,
                 "on" if hotwords else "off")
        return model

    raise ValueError(
        f"STT_MODEL={choice!r} is not recognised; expected 'parakeet' or 'whisper'"
    )
