"""What `voice` means here, and what it cannot mean.

The audit's finding was blunt and correct: `alloy`, `nova` and
`custom_voice_xyz` all returned 200 and all returned the same audio. Unlike
`speed`, which is refused outright, this one handed back a voice the caller did
not ask for and said nothing.

Chatterbox has no named voices. It clones from a reference clip, which is a
real mechanism rather than a missing one — `generate(audio_prompt_path=...)` is
in the signature — so the honest shape is a registry of clips:

    TTS_VOICE_DIR/alloy.wav   ->  voice "alloy"
    TTS_VOICE_DIR/gabriel.wav ->  voice "gabriel"

with `default`, the model's own built-in speaker, always present. A request
naming a voice this service does not have is a **400 naming `voice`**, which is
what tts-stack already does for an unknown Kokoro name.

**The one place this is a deviation rather than a fix.** With no clips
installed — the state every existing deployment is in — there is exactly one
voice, and refusing `alloy` would break every unmodified OpenAI client at the
moment of an upgrade. So OpenAI's thirteen documented names resolve to the
built-in voice unless a clip of that name exists, and the response says which
voice it actually used in an `X-Voice` header. Declared, visible on the wire,
listed in the README as a deviation, and closable by dropping thirteen files in
a directory. `TTS_VOICE_STRICT=1` turns the aliases off and makes an unbacked
name a 400 like any other.

The schema's `VoiceIdsOrCustomVoice` also admits an object, `{"id": "voice_1234"}`,
and openai-python's `Voice` alias includes it. It resolves through exactly the
same table: the id is a name like any other.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["BUILTIN", "OPENAI_VOICES", "Registry", "load_registry"]

# The model's own speaker, the one every request has been getting. Named so it
# can be asked for deliberately.
BUILTIN = "default"

# The thirteen names OpenAI's schema documents, in its order. They are aliases
# here, not voices: see the module docstring.
OPENAI_VOICES = ("alloy", "ash", "ballad", "coral", "echo", "fable", "onyx",
                 "nova", "sage", "shimmer", "verse", "marin", "cedar")

# What librosa can load as a reference clip. Chatterbox resamples it itself
# (chatterbox/mtl_tts.py:224 loads the prompt at S3GEN_SR), so the rate of the
# file does not matter — only that it decodes.
_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


class Registry:
    """The voices this process can actually produce."""

    def __init__(self, clips: dict[str, Path], strict: bool = False,
                 directory: Path | None = None,
                 stamp: tuple[float, int] | None = None,
                 seconds: dict[str, float | None] | None = None) -> None:
        self.clips = clips
        self.strict = strict
        # HOW LONG EACH CLIP IS, read at scan time and never per request.
        # Chatterbox Turbo asserts a reference longer than five seconds, and
        # that is a property of the PAIR (engine, voice) rather than of either
        # -- so it has nowhere else to live, and the only alternative is an
        # AssertionError inside somebody's job after a job id already exists.
        #
        # None WHERE IT COULD NOT BE READ CHEAPLY, and a None never refuses
        # anything: a header this side cannot parse is not evidence the clip is
        # short. Turbo's own assert catches that case, loudly and rarely.
        self.seconds: dict[str, float | None] = dict(seconds or {})
        # Where the clips came from, and what the directory looked like when
        # they were read. Both optional so a hand-built Registry — every one in
        # the test suite — still works and simply never refreshes.
        self.directory = directory
        self.stamp = stamp

    def refresh(self) -> None:
        """Rescan if, and only if, the directory has changed since last time.

        THE ORIGINAL DECISION WAS "SCAN ONCE, AT STARTUP", and the argument for
        it was sound as far as it went: a directory listing on every /v1 call
        is a syscall per request to answer a question that hardly ever changes.
        What it also meant, though, was that a clip arriving by ANY route — a
        copy over SMB, an scp, or the UI service writing into the shared
        volume — was invisible until someone restarted a container that holds
        6.5 GB of Chatterbox and pays a ~60 s model load on its next job. That
        is a heavy price for a file appearing in a directory.

        So the listing is still not done per request: st_mtime_ns and st_ino
        are one stat() and change only when an entry is created, renamed or
        removed in the directory. A rescan happens on the request AFTER a clip
        lands and never again until the next one. The cost the original comment
        was avoiding is avoided; the restart is not required.

        st_ino is in the stamp because a volume can be swapped underneath a
        running container — remounted, or replaced by a deploy — and the new
        directory's mtime can plausibly be older than the one we recorded.
        """
        if self.directory is None:
            return
        try:
            info = self.directory.stat()
        except OSError:
            # The volume went away. Keep what we had rather than emptying the
            # registry: a request for a voice whose file is gone fails with a
            # decode error, which is a better story than every voice silently
            # becoming "unknown voice" the moment a mount hiccups.
            return
        stamp = (info.st_mtime_ns, info.st_ino)
        if stamp == self.stamp:
            return
        fresh = load_registry(self.directory, self.strict)
        self.clips = fresh.clips
        self.seconds = fresh.seconds
        self.stamp = fresh.stamp

    @property
    def names(self) -> list[str]:
        return [BUILTIN, *sorted(self.clips)]

    @property
    def aliased(self) -> list[str]:
        """OpenAI names with no clip behind them, so answered by the built-in."""
        if self.strict:
            return []
        return [name for name in OPENAI_VOICES if name not in self.clips]

    def seconds_for(self, name: str) -> float | None:
        """How long this voice's reference clip is, or None if it is unknown.

        The built-in speaker has no clip at all, which is not a short clip: it
        is the model's own voice and no engine's minimum applies to it.
        """
        return self.seconds.get(name)

    def resolve(self, requested: str | None) -> tuple[str, str | None] | None:
        """(name, reference clip path or None), or None if the voice is unknown.

        A clip wins over an alias, so `alloy.wav` in the directory is reached
        by `alloy` and nothing is shadowed by the table.
        """
        name = (requested or BUILTIN).strip()
        if not name:
            return None
        clip = self.clips.get(name)
        if clip is not None:
            return name, str(clip)
        if name == BUILTIN:
            return BUILTIN, None
        if not self.strict and name in OPENAI_VOICES:
            # Documented alias, reported back in X-Voice. See the docstring.
            return BUILTIN, None
        return None


def load_registry(directory: str | os.PathLike[str] | None = None,
                  strict: bool | None = None) -> Registry:
    """Scan the voice directory, and record what it looked like.

    Called once at startup and then only from Registry.refresh, which fires
    when the directory's stat() says something in it has changed. See that
    method for why "once, at startup" was narrowed rather than kept.
    """
    path = Path(directory if directory is not None
                else os.getenv("TTS_VOICE_DIR", "/voices"))
    if strict is None:
        strict = os.getenv("TTS_VOICE_STRICT", "").strip().lower() in {
            "1", "true", "yes", "on"}
    clips: dict[str, Path] = {}
    stamp: tuple[float, int] | None = None
    if path.is_dir():
        for entry in sorted(path.iterdir()):
            if entry.is_file() and entry.suffix.lower() in _SUFFIXES:
                # First spelling wins, deterministically: sorted() above means
                # alloy.flac beats alloy.wav every start rather than whichever
                # the filesystem happened to hand over first.
                clips.setdefault(entry.stem, entry)
        try:
            info = path.stat()
            stamp = (info.st_mtime_ns, info.st_ino)
        except OSError:
            stamp = None
    return Registry(clips, strict=strict, directory=path, stamp=stamp,
                    seconds={name: _clip_seconds(clip)
                             for name, clip in clips.items()})


def _clip_seconds(clip: Path) -> float | None:
    """The duration of a reference clip, from its HEADER. Never a decode.

    A HEADER READ AND NOTHING MORE. This runs for every clip in the directory
    on every rescan, and a rescan happens on the request after a file lands, so
    decoding a few minutes of audio here would put that cost on somebody's next
    request. soundfile.info is a seek and a struct.

    NEVER FATAL, and the answer to "I could not read it" is None rather than
    zero. Zero is a length, and a length shorter than every engine's minimum
    would refuse a voice that is probably fine -- for an mp3 this build cannot
    parse, or a file being written as the scan walked past it.
    """
    try:
        import soundfile
    except ImportError:  # pragma: no cover - the image installs it
        return None
    try:
        return float(soundfile.info(str(clip)).duration)
    except Exception:  # noqa: BLE001 - the failure IS "unknown", see above
        return None
