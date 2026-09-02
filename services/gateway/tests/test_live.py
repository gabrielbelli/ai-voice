"""The gateway against the three real services on orko, or nothing at all.

The mocked suite proves the routing rules. It cannot prove the two things that
only a real backend can disagree about: that these paths exist on the far side
with these shapes, and that streaming a real multipart upload and a real audio
response through httpx produces the bytes the client asked for.

So this runs the actual app in-process with its backend URLs pointed at the
live containers, over real sockets. It is SKIPPED, not failed, when orko is
unreachable — the usual case on a CI runner, on a train, or when the NAS is
asleep — because a test suite that fails when someone's house is offline stops
being read.

Nothing here queues a Chatterbox job. tts-long runs one job at a time on a
6.5 GB model that takes minutes to load, and a test suite that enqueued work on
a shared machine would be a test suite people learn to avoid running. The long
path is exercised read-only, through GET /jobs.

    GATEWAY_LIVE_HOST=orko.gabrielbelli.com   the host the three run on
    GATEWAY_LIVE=0                            skip even when it is reachable
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from conftest import gateway  # noqa: F401  (imported for the fixture's module)

HOST = os.getenv("GATEWAY_LIVE_HOST", "orko.gabrielbelli.com")
BACKENDS = {"stt": f"http://{HOST}:8000",
            "tts": f"http://{HOST}:8001",
            "tts_long": f"http://{HOST}:8002"}


def _reachable() -> str | None:
    """None if all three answer /health, otherwise why not."""
    if os.getenv("GATEWAY_LIVE") == "0":
        return "GATEWAY_LIVE=0"
    for name, url in BACKENDS.items():
        try:
            response = httpx.get(f"{url}/health", timeout=3.0)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - any failure is a skip
            return f"{name} at {url} is not reachable: {type(exc).__name__}"
    return None


_why = _reachable()
pytestmark = pytest.mark.skipif(_why is not None, reason=f"live backends: {_why}")


@pytest.fixture
async def live(monkeypatch):
    """The real app, real httpx, real sockets, pointed at the live containers."""
    import importlib

    monkeypatch.setenv("GATEWAY_STT_URL", BACKENDS["stt"])
    monkeypatch.setenv("GATEWAY_TTS_URL", BACKENDS["tts"])
    monkeypatch.setenv("GATEWAY_TTS_LONG_URL", BACKENDS["tts_long"])
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    main = importlib.reload(importlib.import_module("app.main"))

    async with main.app.router.lifespan_context(main.app):
        async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://gateway.test", timeout=60.0) as client:
            yield client


async def test_health_reports_all_three_live_backends(live):
    """The call this component exists to make possible: one poll, three answers."""
    response = await live.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok", json.dumps(payload, indent=2)
    for name in ("stt", "tts", "tts_long"):
        assert payload["backends"][name]["reachable"] is True
        assert payload["backends"][name]["health"]["status"] in {"ok", "loading"}


async def test_the_advertised_names_match_what_is_running(live):
    """/v1/models is answered here, so nothing but this test compares it to reality."""
    ids = {m["id"] for m in (await live.get("/v1/models")).json()["data"]}
    assert {"kokoro", "chatterbox", "parakeet"} <= ids

    health = (await live.get("/health")).json()["backends"]
    assert health["stt"]["health"]["model"] == "parakeet"


async def test_voices_are_proxied_from_tts_stack(live):
    response = await live.get("/voices")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["voices"]) > 40
    assert "bm_george" in payload["voices"]


async def test_a_default_model_synthesises_real_audio(live):
    """The whole fast path, end to end: buffer the body, route on `model`, stream back."""
    response = await live.post("/v1/audio/speech", json={
        "model": "kokoro", "input": "Here is the change to make.",
        "voice": "bm_george", "response_format": "mp3"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
    # tts-stack's own measurement, forwarded rather than recomputed.
    assert float(response.headers["x-realtime-factor"]) > 0


async def test_an_unknown_model_gets_audio_rather_than_a_400(live):
    """The asymmetry, against the real backend: unknown goes fast and still works."""
    response = await live.post("/v1/audio/speech", json={
        "model": "whatever-the-ui-was-holding", "input": "Short.",
        "response_format": "wav"})

    assert response.status_code == 200
    assert response.content[:4] == b"RIFF"


async def test_a_real_upload_streams_through_to_stt(live):
    """A multipart body, forwarded chunk by chunk, transcribed by the real model.

    One second of 16 kHz silence: enough to prove the pipeline answers, small
    enough not to occupy a shared box.
    """
    silence = (b"RIFF" + (36 + 32000).to_bytes(4, "little") + b"WAVEfmt "
               + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
               + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little")
               + (32000).to_bytes(4, "little") + (2).to_bytes(2, "little")
               + (16).to_bytes(2, "little") + b"data"
               + (32000).to_bytes(4, "little") + b"\x00" * 32000)

    response = await live.post("/v1/audio/transcriptions",
                               files={"file": ("silence.wav", silence, "audio/wav")},
                               data={"model": "whisper-1"})

    assert response.status_code == 200
    assert "text" in response.json()


async def test_the_long_backends_job_list_is_reachable_flat(live):
    """Read-only: GET /jobs proves the unprefixed mount without queueing work."""
    response = await live.get("/jobs")

    assert response.status_code == 200
    assert "jobs" in response.json()


async def test_the_schema_is_not_published_through_the_gateway(live):
    """stt-stack put its own behind a key; a wildcard here would undo that."""
    for path in ("/docs", "/openapi.json"):
        response = await live.get(path)
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "unknown_url"
