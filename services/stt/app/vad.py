"""Voice activity detection.

Silero VAD, ~2 MB, ONNX. It runs before either ASR model and earns its place
twice over: it removes the silence that makes Whisper hallucinate text out of
nothing, and it removes the silence both models would otherwise be paid to
transcribe. On a long clip with pauses, dropping non-speech is the single
largest speed win available on CPU.

It also decides the timeline. The recogniser never sees the original clip — it
sees the speech runs, concatenated — so every timestamp it reports is in a
compacted timeline that is shorter than the audio the client sent. This module
used to return only the *fraction* it kept, which threw away the one thing
needed to undo that, and the result was a subtitle track whose cues drifted
further out of step the more silence the recording contained. `Speech.spans`
carries the offsets, and `Speech.original` maps a recogniser time back.

The thresholds are settable per request because the specification has a field
for them: `chunking_strategy[type]=server_vad` with threshold,
prefix_padding_ms and silence_duration_ms maps one-to-one onto the three
arguments below. A client tuning a VAD on a service that visibly runs one and
having the values dropped is the worst of both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16_000
_WINDOW = 512  # Silero's required frame size at 16 kHz

# Silero's own defaults, and the ones this service has always run with.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_SILENCE_MS = 300
DEFAULT_SPEECH_PAD_MS = 100


@dataclass(frozen=True)
class Speech:
    """Speech-only audio, and where each piece of it came from."""

    samples: np.ndarray
    # (start, end) sample offsets into the ORIGINAL clip, in order, one per
    # kept run. Concatenating the clip over these spans reproduces `samples`.
    spans: tuple[tuple[int, int], ...]
    kept: float

    def original(self, seconds: float) -> float:
        """Map a time in the compacted timeline back to the original clip.

        The recogniser reports times against `samples`; a caller writing a
        subtitle needs them against what the client sent. Walks the spans,
        which is O(runs) per call and never more than a few dozen.
        """
        want = max(seconds, 0.0) * SAMPLE_RATE
        consumed = 0.0
        for start, end in self.spans:
            length = end - start
            if want <= consumed + length:
                return (start + (want - consumed)) / SAMPLE_RATE
            consumed += length
        # Past the end of the speech: report the end of the last run rather
        # than a time the audio does not reach.
        return (self.spans[-1][1] / SAMPLE_RATE) if self.spans else 0.0


def whole(samples: np.ndarray) -> Speech:
    """The identity timeline, for when the VAD is switched off."""
    return Speech(samples=samples, spans=((0, len(samples)),), kept=1.0)


class Vad:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
                 speech_pad_ms: int = DEFAULT_SPEECH_PAD_MS) -> None:
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
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms

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

    def speech_only(self, samples: np.ndarray, *,
                    threshold: float | None = None,
                    min_silence_ms: int | None = None,
                    speech_pad_ms: int | None = None) -> Speech:
        """Speech-only audio, the spans it came from, and the fraction kept.

        Falls back to the untouched input when nothing crosses the threshold —
        a clip that VAD believes is entirely silent is far more likely to be a
        quiet microphone than genuine silence, and returning nothing to
        transcribe would be the wrong failure.
        """
        if len(samples) < _WINDOW:
            return whole(samples)

        level = self.threshold if threshold is None else threshold
        silence = int((self.min_silence_ms if min_silence_ms is None
                       else min_silence_ms) * SAMPLE_RATE / 1000)
        pad = int((self.speech_pad_ms if speech_pad_ms is None
                   else speech_pad_ms) * SAMPLE_RATE / 1000)

        speech = self._probabilities(samples) >= level
        if not speech.any():
            return whole(samples)

        spans: list[tuple[int, int]] = []
        run_start: int | None = None
        gap = 0
        for i, is_speech in enumerate(speech):
            if is_speech:
                if run_start is None:
                    run_start = i
                gap = 0
            elif run_start is not None:
                gap += 1
                if gap * _WINDOW >= silence:
                    lo = max(0, run_start * _WINDOW - pad)
                    hi = min(len(samples), (i - gap + 1) * _WINDOW + pad)
                    spans.append((lo, hi))
                    run_start = None
        if run_start is not None:
            spans.append((max(0, run_start * _WINDOW - pad), len(samples)))

        # Padding can push one run's end past the next run's start. Merging is
        # what makes `spans` a partition, which is what `original` walks — and
        # what stops a word landing in two segments at once.
        merged: list[tuple[int, int]] = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        kept = np.concatenate([samples[lo:hi] for lo, hi in merged]) if merged \
            else np.zeros(0, dtype=samples.dtype)
        if kept.size == 0:
            return whole(samples)
        return Speech(samples=kept, spans=tuple(merged),
                      kept=kept.size / samples.size)
