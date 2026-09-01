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

    def _fake_speak(self, text, language, exaggeration, cfg_weight,
                    temperature, reference):
        # A tone whose length and content are a pure function of the text, so
        # two runs are byte-identical. Deliberately over 1.0 in places, which
        # is what exercises the clip-before-scale in pcm_bytes.
        if delay:
            time.sleep(delay)
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        samples = int(len(text) / 10 * synth_module.SAMPLE_RATE)
        t = np.arange(samples, dtype=np.float32) / synth_module.SAMPLE_RATE
        audio = (1.4 * np.sin(2 * np.pi * (110 + seed % 220) * t)).astype(np.float32)
        return synth_module.Spoken(audio=audio, input_tokens=len(text.split()))

    monkeypatch.setattr(synth_module.Synth, "_speak", _fake_speak)

    from app.main import app

    return app


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
