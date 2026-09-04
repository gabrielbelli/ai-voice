"""The reference-clip store: the write path, and everything it refuses.

The browser is the transcoder, so what arrives here is always a WAV — which is
what lets the validation be the stdlib `wave` module rather than librosa, and
what lets the allowlist be one extension rather than six. Server-side checks
still run: a client-side check is a courtesy, not a boundary.
"""

from __future__ import annotations

import io
import wave

import pytest

from app import clips, config


def wav_bytes(seconds: float, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


@pytest.fixture
def store(monkeypatch, tmp_path):
    directory = tmp_path / "voices"
    directory.mkdir()
    monkeypatch.setattr(clips.config, "VOICE_DIR", str(directory))
    return directory


def test_a_clip_is_written_and_listed(store):
    saved = clips.save("Gabriel", wav_bytes(12))
    assert saved["name"] == "gabriel"
    assert (store / "gabriel.wav").exists()
    assert [c["name"] for c in clips.listing()] == ["gabriel"]
    assert 11.9 < clips.listing()[0]["seconds"] < 12.1


@pytest.mark.parametrize("given,expected", [
    ("Gabriel Belli", "gabriel-belli"),
    ("  spaced  ", "spaced"),
    # Lowercased so that a case-insensitive filesystem (macOS, SMB) cannot make
    # Gabriel and gabriel one voice on one host and two on another.
    ("MiXeD_Case-9", "mixed_case-9"),
    ("Ünïcodé", "n-cod"),
])
def test_names_are_built_from_a_whitelist(given, expected):
    # The filename is derived from [a-z0-9_-] and NOTHING else — not from a
    # sanitiser applied to the uploaded filename. A whitelist that cannot
    # express "/", "..", a NUL or a leading dot cannot be tricked into
    # expressing them, which is the difference between this and upstream's
    # sanitize_filename being "the only thing" in the way.
    stem = clips.slug(given)
    assert "/" not in stem and ".." not in stem
    assert stem == expected


@pytest.mark.parametrize("attack", [
    "../../etc/passwd", "..", ".", "/etc/shadow", "a/../../b", "\x00nul",
])
def test_path_traversal_cannot_be_expressed(store, attack):
    try:
        stem = clips.slug(attack)
    except clips.ClipError:
        return                       # refused outright, which is also fine
    assert "/" not in stem and stem not in {"", ".", ".."}
    clips.save(attack, wav_bytes(12))
    written = [p for p in store.iterdir() if p.is_file()]
    assert all(p.parent == store for p in written)


def test_default_is_reserved_because_it_is_chatterboxs_own_speaker(store):
    # A clip wins over an alias in Registry.resolve, which is deliberate — but
    # a clip called `default` would shadow the built-in itself.
    with pytest.raises(clips.ClipError, match="built-in"):
        clips.save("default", wav_bytes(12))


def test_a_clip_over_the_duration_ceiling_is_trimmed_and_leaves_one_file(store):
    """This asserted a REFUSAL until yt-dlp's keyframe seeking made the ceiling
    unreachable from the link path -- see the trim tests at the end of this
    file. What still matters is the half it was really guarding: exactly one
    file in the directory afterwards, and no `.part` left behind.
    """
    saved = clips.save("long", wav_bytes(45))
    assert saved["seconds"] <= config.MAX_CLIP_SECONDS + 0.01
    files = sorted(p.name for p in store.iterdir())
    assert files == ["long.wav"], f"expected one finished clip, got {files}"


def test_bytes_claiming_to_be_a_wav_and_failing_are_refused(store):
    """The suffix is the claim; this is the check on it.

    Non-WAV formats are stored as they arrive now -- tts-long reads them with
    librosa and this image has no decoder -- but a file SAVED as .wav must
    still be one, because that is the one format the store measures.
    """
    with pytest.raises(clips.ClipError, match="cannot be read as one"):
        clips.save("junk", b"\x1a\x45\xdf\xa3 this is webm, not wav")
    assert list(store.iterdir()) == []


def test_a_format_the_voice_service_cannot_read_is_refused(store):
    with pytest.raises(clips.ClipError, match="not a format"):
        clips.save("junk", b"anything", suffix=".aiff")
    assert list(store.iterdir()) == []


def test_an_opus_voice_note_is_stored_as_it_arrives(store):
    """The case that prompted this: a WhatsApp voice note is ogg/opus, carries
    no duration in its container, and decodeAudioData returned ONE SECOND of
    audio with no error -- a useless reference clip that looked like a
    successful upload. The browser is no longer required to convert it."""
    saved = clips.save("note", b"OggS\x00\x02 not really opus, but named so",
                       suffix=".opus")
    assert saved["seconds"] is None, "duration is not knowable without a decoder"
    assert (store / "note.opus").exists()
    assert [c["name"] for c in clips.listing()] == ["note"]


def test_one_voice_keeps_one_file_across_formats(store):
    """Saving note.mp3 over note.opus must not leave two clips claiming the
    name, with the registry picking by directory order."""
    clips.save("note", b"OggS\x00\x02 x", suffix=".opus")
    clips.save("note", b"ID3 x", suffix=".mp3", replace=True)
    files = sorted(p.name for p in store.iterdir())
    assert files == ["note.mp3"], files


def test_the_accepted_formats_match_what_tts_long_reads(store):
    """A format accepted here that tts-long cannot load would be a clip that
    saves and then fails at generation time."""
    import re as _re
    from pathlib import Path as _Path

    voices = (_Path(__file__).resolve().parents[3]
              / "services" / "tts-long" / "app" / "voices.py").read_text()
    block = voices[voices.index("_SUFFIXES = ("):]
    backend = set(_re.findall(r'"(\.\w+)"', block[:block.index(")")]))
    assert set(clips.SUFFIXES) == backend, (
        f"ui-only {set(clips.SUFFIXES) - backend}, "
        f"tts-long-only {backend - set(clips.SUFFIXES)}")


def test_a_clip_over_the_size_cap_is_refused(store, monkeypatch):
    monkeypatch.setattr(clips.config, "MAX_CLIP_BYTES", 1024)
    with pytest.raises(clips.ClipError, match="ceiling"):
        clips.save("big", wav_bytes(12))


def test_a_collision_needs_replace(store):
    clips.save("gabriel", wav_bytes(12))
    with pytest.raises(clips.ClipError, match="already exists"):
        clips.save("gabriel", wav_bytes(12))
    assert clips.save("gabriel", wav_bytes(14), replace=True)["name"] == "gabriel"


def test_an_unmounted_directory_says_so_rather_than_pretending(monkeypatch, tmp_path):
    monkeypatch.setattr(clips.config, "VOICE_DIR", str(tmp_path / "nope"))
    # Listing is a configuration state, not an error: no volume means no
    # voices, and the Speak tab keeps working.
    assert clips.listing() == []
    assert clips.writable() is False
    # Saving is where it matters, because a clip written into a container's
    # own filesystem dies on the next restart, which is worse than refusing.
    with pytest.raises(clips.ClipError, match="not mounted"):
        clips.save("gabriel", wav_bytes(12))


def test_removing(store):
    clips.save("gabriel", wav_bytes(12))
    assert clips.remove("gabriel") is True
    assert clips.remove("gabriel") is False


def test_the_route_returns_the_refreshed_list(client, tmp_path):
    api, _, _ = client(UI_VOICE_DIR=str(tmp_path / "v"))
    (tmp_path / "v").mkdir()
    response = api.post("/ui/clips", data={"name": "gabriel"},
                        files={"file": ("x.wav", wav_bytes(12), "audio/wav")})
    assert response.status_code == 201
    assert [v["name"] for v in response.json()["voices"]] == ["gabriel"]
    assert api.get("/ui/clips").json()["writable"] is True
    assert api.delete("/ui/clips/gabriel").json()["voices"] == []


# ------------------------------------------ the ceiling is a trim, not a wall --


def test_a_clip_over_the_ceiling_is_trimmed_not_refused(store):
    """yt-dlp seeks to KEYFRAMES, so a requested window overshoots.

    Measured against the deployed instance: twenty seconds requested came back
    as 25.97, thirty came back as 36. That made the ceiling unreachable from
    the link path -- asking for the maximum guaranteed exceeding it -- and the
    error told the user their clip was too long for a length they had never
    chosen. The browser's toWav has always trimmed rather than rejected; the
    server refusing was the odd one out.
    """
    saved = clips.save("overshoot", wav_bytes(36))
    assert saved["seconds"] <= config.MAX_CLIP_SECONDS + 0.01, saved


def test_a_clip_under_the_ceiling_is_left_alone(store):
    saved = clips.save("short", wav_bytes(12))
    assert abs(float(saved["seconds"]) - 12) < 0.01


def test_a_clip_that_is_too_short_is_still_refused(store):
    """Trimming cannot invent audio, so the lower bound stays a refusal."""
    with pytest.raises(clips.ClipError):
        clips.save("tiny", wav_bytes(0.4))
