"""The voice registry, and the one thing about it that changed.

app/voices.py imports nothing but os and pathlib, so this file runs without
torch, without chatterbox and without the 6.5 GB of weights the rest of the
service needs — which is the point: the behaviour under test is a directory
listing and a stat(), and it should be assertable in milliseconds.
"""

from __future__ import annotations

import os
import time

import pytest

from app import voices


def clip(directory, name, suffix=".wav"):
    path = directory / (name + suffix)
    path.write_bytes(b"not really audio, and nothing here decodes it")
    return path


def test_an_empty_directory_still_has_the_builtin(tmp_path):
    registry = voices.load_registry(tmp_path)
    assert registry.names == ["default"]
    assert registry.resolve("default") == ("default", None)


def test_the_thirteen_openai_names_alias_to_the_builtin(tmp_path):
    registry = voices.load_registry(tmp_path)
    # A deviation, declared: with no clips installed — the state every existing
    # deployment is in — refusing `alloy` would break every unmodified OpenAI
    # client at the moment of an upgrade.
    assert registry.resolve("alloy") == ("default", None)
    assert "alloy" in registry.aliased


def test_a_clip_wins_over_an_alias(tmp_path):
    clip(tmp_path, "alloy")
    registry = voices.load_registry(tmp_path)
    name, reference = registry.resolve("alloy")
    assert name == "alloy" and reference.endswith("alloy.wav")


def test_a_clip_added_after_startup_is_found_on_the_next_request(tmp_path):
    """THE CHANGE. The registry used to scan once, at startup, so a clip
    arriving by ANY route — services/ui writing into the shared volume, a copy
    over SMB, an scp — was invisible until someone restarted a container that
    holds 6.5 GB of Chatterbox and pays a ~60 s model load on its next job."""
    registry = voices.load_registry(tmp_path)
    assert registry.resolve("gabriel") is None

    clip(tmp_path, "gabriel")
    # st_mtime_ns has nanosecond resolution but some filesystems do not, so
    # the stamp is nudged rather than the test being made flaky by a fast disk.
    os.utime(tmp_path, (time.time() + 1, time.time() + 1))

    registry.refresh()
    name, reference = registry.resolve("gabriel")
    assert name == "gabriel" and reference.endswith("gabriel.wav")
    assert "gabriel" in registry.names


def test_refresh_does_nothing_when_the_directory_has_not_changed(tmp_path):
    registry = voices.load_registry(tmp_path)
    stamp = registry.stamp
    registry.refresh()
    assert registry.stamp == stamp    # one stat(), no listing


def test_a_registry_built_by_hand_never_refreshes(tmp_path):
    # Every Registry in the rest of the suite is constructed directly, and one
    # of those quietly rescanning a directory it was never given would be a
    # surprising kind of wrong.
    registry = voices.Registry({}, strict=False)
    registry.refresh()
    assert registry.names == ["default"]


def test_a_vanished_volume_keeps_what_it_had(tmp_path, monkeypatch):
    clip(tmp_path, "gabriel")
    registry = voices.load_registry(tmp_path)

    def gone(*args, **kwargs):
        raise OSError("volume went away")
    monkeypatch.setattr(voices.Path, "stat", gone)
    registry.refresh()
    # Emptying the registry on a mount hiccup would turn every voice into
    # "unknown voice" at once, which is a worse story than one decode error.
    assert "gabriel" in registry.names


def test_removal_is_seen_too(tmp_path):
    path = clip(tmp_path, "gabriel")
    registry = voices.load_registry(tmp_path)
    assert "gabriel" in registry.names
    path.unlink()
    os.utime(tmp_path, (time.time() + 1, time.time() + 1))
    registry.refresh()
    assert registry.names == ["default"]


def test_first_spelling_wins_deterministically(tmp_path):
    clip(tmp_path, "alloy", ".flac")
    clip(tmp_path, "alloy", ".wav")
    registry = voices.load_registry(tmp_path)
    assert str(registry.clips["alloy"]).endswith(".flac")


def test_strict_turns_the_aliases_off(tmp_path):
    registry = voices.load_registry(tmp_path, strict=True)
    assert registry.aliased == []
    assert registry.resolve("alloy") is None
