"""Six lines, and the one thing none of the three services can do today."""

from __future__ import annotations

import logging

import pytest

from voice_common.logging import setup


@pytest.fixture(autouse=True)
def _restore_root_level():
    """basicConfig mutates the root logger, so put it back."""
    level = logging.root.level
    handlers = list(logging.root.handlers)
    yield
    logging.root.setLevel(level)
    logging.root.handlers = handlers


def test_the_level_applies_even_when_something_already_configured_logging(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """basicConfig is a no-op once root has a handler, and uvicorn installs one.

    Reading the variable and then ignoring it is the same as not having it.
    """
    logging.root.addHandler(logging.NullHandler())
    monkeypatch.setenv("TTS_LOG_LEVEL", "ERROR")
    setup("svc", "TTS")
    assert logging.root.level == logging.ERROR


def test_the_level_can_be_raised_without_editing_code_and_rebuilding_an_image(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The only reason this is a module rather than a copied line.

    A running container that has started misbehaving is exactly when DEBUG is
    wanted and exactly when none of the three can produce it.
    """
    monkeypatch.setenv("TTS_LOG_LEVEL", "debug")
    setup("svc", "TTS")
    assert logging.root.level == logging.DEBUG


def test_the_prefix_keeps_each_service_naming_its_own_variable(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STT_LOG_LEVEL", "WARNING")
    monkeypatch.delenv("TTS_LOG_LEVEL", raising=False)
    setup("svc", "STT")
    assert logging.root.level == logging.WARNING


def test_the_default_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TTS_LOG_LEVEL", raising=False)
    setup("svc", "TTS")
    assert logging.root.level == logging.INFO


def test_a_typo_in_the_level_never_stops_the_service_starting(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Falling back to INFO and complaining beats refusing to boot over a
    logging preference."""
    monkeypatch.setenv("TTS_LOG_LEVEL", "VERBOSE")
    with caplog.at_level(logging.WARNING):
        log = setup("svc", "TTS")
        # Asserted inside the block: at_level restores the root level on exit.
        assert logging.root.level == logging.INFO
    assert log.name == "svc"
    assert any("not a level name" in message for message in caplog.messages)
