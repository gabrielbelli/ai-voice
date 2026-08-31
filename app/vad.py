"""Voice activity detection.

Silero VAD, ~2 MB, ONNX. It runs before either ASR model and earns its place
twice over: it removes the silence that makes Whisper hallucinate text out of
nothing, and it removes the silence both models would otherwise be paid to
transcribe. On a long clip with pauses, dropping non-speech is the single
largest speed win available on CPU.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
_WINDOW = 512  # Silero's required frame size at 16 kHz


class Vad:
    def __init__(self, threshold: float = 0.5, min_silence_ms: int = 300,
                 speech_pad_ms: int = 100) -> None:
        import onnxruntime  # noqa: PLC0415 - keep import cost off module load
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download("onnx-community/silero-vad", "onnx/model.onnx")
        opts = onnxruntime.SessionOptions()
        # One thread. The VAD is trivially cheap and a second thread costs more
        # in coordination than it returns.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            path, opts, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self.min_silence = int(min_silence_ms * SAMPLE_RATE / 1000)
        self.pad = int(speech_pad_ms * SAMPLE_RATE / 1000)

    def _probabilities(self, samples: np.ndarray) -> np.ndarray:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        out = []
        for start in range(0, len(samples) - _WINDOW + 1, _WINDOW):
            frame = samples[start:start + _WINDOW].reshape(1, -1).astype(np.float32)
            prob, state = self._session.run(
                None,
                {"input": frame, "state": state,
                 "sr": np.array(SAMPLE_RATE, dtype=np.int64)},
            )
            out.append(float(prob.item()))
        return np.asarray(out, dtype=np.float32)

    def speech_only(self, samples: np.ndarray) -> tuple[np.ndarray, float]:
        """Return speech-only audio and the fraction of the original kept.

        Falls back to the untouched input when nothing crosses the threshold —
        a clip that VAD believes is entirely silent is far more likely to be a
        quiet microphone than genuine silence, and returning nothing to
        transcribe would be the wrong failure.
        """
        if len(samples) < _WINDOW:
            return samples, 1.0

        speech = self._probabilities(samples) >= self.threshold
        if not speech.any():
            return samples, 1.0

        keep = np.zeros(len(samples), dtype=bool)
        run_start: int | None = None
        gap = 0
        for i, is_speech in enumerate(speech):
            if is_speech:
                if run_start is None:
                    run_start = i
                gap = 0
            elif run_start is not None:
                gap += 1
                if gap * _WINDOW >= self.min_silence:
                    lo = max(0, run_start * _WINDOW - self.pad)
                    hi = min(len(samples), (i - gap + 1) * _WINDOW + self.pad)
                    keep[lo:hi] = True
                    run_start = None
        if run_start is not None:
            lo = max(0, run_start * _WINDOW - self.pad)
            keep[lo:] = True

        kept = samples[keep]
        if kept.size == 0:
            return samples, 1.0
        return kept, kept.size / samples.size
