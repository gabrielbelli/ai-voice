"""Fixtures for the deployment-and-documentation tests.

These tests belong to `compose.yaml` and to the prose, not to any one service,
which is why they live under `docs/`. They import nothing from a service. The
only package they touch is `packages/common/voice_common`, and only to read the
engine catalogue — the single table both `tts-long` and the gateway derive their
model lists from.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]

# The catalogue lives in packages/common. Put it on the path here rather than
# demanding a PYTHONPATH, so `pytest docs/tests` works from a clean checkout.
sys.path.insert(0, str(ROOT / "packages" / "common"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def compose() -> dict:
    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def compose_text() -> str:
    return (ROOT / "compose.yaml").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def env_of(compose):
    def _env(service: str) -> dict[str, str]:
        return {k: str(v) for k, v
                in (compose["services"][service].get("environment") or {}).items()}
    return _env


@pytest.fixture(scope="session")
def catalogue():
    """The engine catalogue, or a message naming who owes it.

    NOT `importorskip`. A skip is how a dead test hides, and this test exists
    because the deployment is about to advertise a second engine: if the
    catalogue is missing, the half of the slice that makes the deployment true
    has not landed and that is the finding, not a reason to stay quiet.
    """
    try:
        from voice_common import engines  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - the message IS the result
        pytest.fail(
            "packages/common/voice_common/engines.py is missing, so nothing can "
            "check compose.yaml's engine names against the one table both "
            f"tts-long and the gateway read ({exc}). Either that module has not "
            "landed yet, or TTS_ENGINES in compose.yaml names engines this code "
            "base has no catalogue for.")
    return engines


# Every line of application source in the repository, as one string, so a test
# can ask whether an environment key is read anywhere at all.
def _source() -> str:
    parts: list[str] = []
    for pattern in ("services/*/app/**/*.py",
                    "packages/common/voice_common/**/*.py",
                    "packages/common/*.sh"):
        for path in ROOT.glob(pattern):
            if ".venv" in path.parts or "__pycache__" in path.parts:
                continue
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


@pytest.fixture(scope="session")
def source() -> str:
    return _source()


KEY = re.compile(r"\b(?:TTS|GATEWAY|STT|UI|AIV|RUNLOG)_[A-Z0-9_]+\b")


def keys_read_by_code(text: str, source_text: str) -> set[str]:
    """Which environment keys named in `text` are actually read by the code."""
    return {k for k in KEY.findall(text)
            if f'"{k}"' in source_text or f"'{k}'" in source_text}
