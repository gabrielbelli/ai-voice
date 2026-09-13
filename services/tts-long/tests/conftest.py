"""One app, one fake model, no torch.

Every test below drives the REAL routes, the real queue, the real chunker and
the real encoders. The only thing replaced is `Synth._speak`, the single method
that touches chatterbox — so nothing here downloads 3 GB or allocates 6.5 GB,
and everything here would still have caught the defects it guards.

The fake is deterministic: the same text always produces the same samples, so
"the streamed bytes are the buffered bytes" is a comparison of two encodes of
identical audio rather than of two rolls of a sampler.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time

import numpy as np
import pytest


# ONE FREQUENCY BAND PER ENGINE, far enough apart that an FFT peak names the
# engine without ambiguity and low enough that nothing lands near Nyquist.
# 2000 Hz apart against a 220 Hz spread within a band, so a peak can only ever
# be read as the engine that made it.
_ENGINE_BAND = 2000.0


def engine_tone(engine: str, seed: int) -> float:
    """The frequency this engine's fake speaks at. Deterministic, and unique."""
    from voice_common.engines import CATALOGUE

    index = sorted(CATALOGUE).index(engine) if engine in CATALOGUE else 0
    return 110.0 + _ENGINE_BAND * (index + 1) + seed % 220


def engine_of(audio, rate=24000) -> str:
    """Which engine's fake made this audio, read back out of the samples.

    THE POINT IS THAT IT IS READ OUT OF THE AUDIO AND NOT OFF A LABEL. Every
    label in this stack -- the header, the record, `engine_reason` -- is
    written by the same code path that chose the engine, so all of them agree
    with each other whether or not any of them is true.
    """
    import numpy as np

    from voice_common.engines import CATALOGUE

    spectrum = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
    peak = float(np.fft.rfftfreq(len(audio), 1.0 / rate)[int(np.argmax(spectrum))])
    best, distance = None, None
    for index, engine in enumerate(sorted(CATALOGUE)):
        centre = 110.0 + _ENGINE_BAND * (index + 1) + 110.0
        gap = abs(peak - centre)
        if distance is None or gap < distance:
            best, distance = engine, gap
    return best


def _build(tmp_path, monkeypatch):
    """Import a fresh app with synthesis faked, and return it.

    Imported after the environment is set, because app.main reads its
    configuration once, at import — which is also how the service behaves:
    rotating a key or a limit is a restart.
    """

    monkeypatch.setenv("TTS_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("TTS_VOICE_DIR", str(tmp_path / "voices"))
    monkeypatch.delenv("TTS_API_KEYS", raising=False)
    # The model is faked, so it is never "loaded" and the synchronous budget
    # would otherwise be charged a cold start that is not happening.
    monkeypatch.setenv("TTS_COLD_LOAD_SECONDS", "0")
    monkeypatch.setenv("TTS_SSE_KEEPALIVE", "0.2")
    for name in [n for n in sys.modules if n == "app" or n.startswith("app.")]:
        del sys.modules[name]

    from app import synth as synth_module

    delay = float(os.getenv("TTS_TEST_DELAY", "0"))

    def _fake_speak(self, text, language, controls, reference):
        # THE SIGNATURE IS Synth._speak's, EXACTLY. `_run` calls
        # `speak_segments` positionally on whichever backend it was handed, so
        # a fake whose parameters have drifted from the real one is a fake that
        # passes while production would raise -- which is the shape of defect
        # this whole suite exists to catch, planted in the suite itself.
        # `controls` is one mapping now, and which keys are in it is the
        # engine's business rather than this fake's.
        # A tone whose length and content are a pure function of the text, so
        # two runs are byte-identical. Deliberately over 1.0 in places, which
        # is what exercises the clip-before-scale in pcm_bytes.
        if delay:
            time.sleep(delay)
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        samples = int(len(text) / 10 * synth_module.SAMPLE_RATE)
        t = np.arange(samples, dtype=np.float32) / synth_module.SAMPLE_RATE
        # THE ENGINE IS IN THE AUDIO, and that is the whole reason this fake
        # takes a frequency rather than a constant one.
        #
        # An audit found FOUR single-token edits that produce one engine's
        # audio from another engine's request while every label -- the
        # response header, the job record, engine_reason -- still says what was
        # asked for, and all four kept the suite green. Nothing in this
        # codebase asserted that THE ENGINE THAT RAN IS THE ENGINE THAT WAS
        # ASKED FOR, because the fake's output was a function of the text
        # alone: swap the checkpoint and the bytes are identical.
        #
        # `engine_tone` READS THE SPEC, SO IT WITNESSES ONE LEG AND NOT THE
        # OTHER, and knowing which is the difference between a guard and a
        # comfort. It proves that the job reached the Synth carrying the
        # catalogue row that was asked for. It CANNOT prove that the row then
        # loaded the checkpoint it names, because `_speak` is this fake and
        # `_ensure_loaded` therefore never runs -- so a row pointed at the
        # other engine's `local_class` leaves this tone correct too. That leg
        # is test_engine_identity.py's
        # `test_the_checkpoint_that_loaded_is_the_one_the_row_named`, which
        # needs no weights and runs on every commit beside this one.
        audio = (1.4 * np.sin(2 * np.pi * engine_tone(self.spec.id, seed) * t)
                 ).astype(np.float32)
        return synth_module.Spoken(audio=audio, input_tokens=len(text.split()))

    monkeypatch.setattr(synth_module.Synth, "_speak", _fake_speak)

    from app.main import app

    return app


@pytest.fixture
def build(tmp_path, monkeypatch):
    """Build the app with extra environment set, for one test.

    The service reads its configuration once, at import, exactly as it behaves
    in production -- so a test that wants a different engine set, a different
    default or a different local lane has to set the variables BEFORE the
    import, and `_build` already deletes every `app.*` module for that reason.
    Anything a test sets here survives, because `_build` only sets its own keys.
    """

    def make(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return _build(tmp_path, monkeypatch)

    return make


@pytest.fixture
def voice_dir(tmp_path):
    """TTS_VOICE_DIR, with a helper that writes a clip of a chosen length.

    A REAL FILE WITH A REAL HEADER. `Registry` reads the duration with
    soundfile.info, which is a seek and a struct rather than a decode, so a
    hand-written stub with no header would answer None and quietly disarm every
    assertion about a reference clip being too short.
    """
    directory = tmp_path / "voices"
    directory.mkdir(parents=True, exist_ok=True)

    def clip(name: str, seconds: float) -> None:
        import soundfile

        rate = 24000
        soundfile.write(str(directory / f"{name}.wav"),
                        np.zeros(int(rate * seconds), dtype=np.float32), rate)

    clip.directory = directory  # type: ignore[attr-defined]
    return clip


@pytest.fixture
def speech(tmp_path, monkeypatch):
    """A TestClient on that app. Everything but the model is real."""
    from starlette.testclient import TestClient

    with TestClient(_build(tmp_path, monkeypatch)) as client:
        yield client


@pytest.fixture
def live(tmp_path, monkeypatch):
    """The same app behind a real uvicorn, on an ephemeral port.

    Needed for exactly one thing, and it is the important one: TestClient runs
    the application to completion before it hands back a response, so
    time-to-first-byte through it is always the total time. Measuring whether
    a stream is genuinely incremental therefore needs a socket.
    """
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(
        _build(tmp_path, monkeypatch), host="127.0.0.1", port=0,
        log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)
