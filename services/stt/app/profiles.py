"""Named glossary profiles: what is loaded, from where, and who may write it.

A glossary is a **named profile**, selected per request. Nothing is applied
unless a request asks for it, which is the opposite of what this service did
until now — `pipeline.start` read one file at boot, compiled it into
`state["rules"]`, and applied it to every transcript for the life of the
container.

WHY THE SET IS NOT ONE LIST ANY MORE
------------------------------------
Two separate failures, and only one of them is obvious.

**The list was one person's vocabulary inside a public artefact.**
`services/stt/glossary.txt` shipped `catalaxy = Catallaxy`,
`theory dashboard = Theoria dashboard` and `ghost paper = Ghost Pepper` into an
image published to a public registry under a BSD licence. Anyone who pulled it
got those rewrites applied to their own audio, naming projects they have never
heard of, discoverable only by reading the image.

**And an irrelevant term is not free.** Measured across 25 cells: a
glossary whose terms do NOT occur in the audio raised WER by **28% on Whisper
and 28% on Whisper**. That measurement is the whole argument for small opt-in
profiles. If an unused term cost nothing, one big always-on list would be fine
— and it is also why selecting several profiles at once is discouraged in the
README for a measured reason rather than a tidiness one.

WHERE PROFILES COME FROM
------------------------
Three sources, and which one a profile came from is visible in the API rather
than implied:

    builtin   STT_GLOSSARY_BUILTIN, /etc/ai-voice/glossaries in the image.
              Read-only. PUT or DELETE on one of these names is a 409, never a
              silent shadow: a profile whose contents depend on which directory
              won is a profile nobody can reason about.
    custom    STT_GLOSSARY_DIR, /glossaries, a mounted volume. Writable over
              the API. If nothing is mounted there the write routes answer 503
              naming the reason and the built-ins still serve — a deployment
              that mounted nowhere to persist has said, by omission, that it
              does not want run-time profiles, and accepting a PUT that
              evaporates on restart would be worse than refusing it. Same
              reasoning as the UI's clips.writable().
    env       STT_GLOSSARY, one file named by the operator, registered under
              its own stem. It is the pre-profile variable, kept working
              because a deployment that set it should not silently lose its
              vocabulary — but it is a profile now like any other, opted into
              per request rather than applied to everything.

RELOADING WITHOUT A RESTART
---------------------------
Per-request selection is meaningless while the set is frozen at boot, so the
registry rescans when the files change. The stamping is `voices.py`'s, with one
deliberate difference: that file stamps only the DIRECTORY (st_mtime_ns and
st_ino), because a voice clip arrives as a whole new file and a directory's
mtime changes when an entry is created, renamed or removed. A glossary is a
text file somebody edits IN PLACE with an editor, which does not touch the
directory's mtime at all, so each known file is stat()ed too. That is a couple
of extra stat() calls against a request that spends seconds in a recogniser.

st_ino is in the stamp for `voices.py`'s reason: a volume can be swapped under a
running container — remounted, or replaced by a deploy — and the new
directory's mtime can plausibly be older than the one we recorded.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import glossary

log = logging.getLogger("stt-stack.profiles")

BUILTIN = "builtin"
CUSTOM = "custom"
ENV = "env"

# Defaults, not the values in use: load_registry() reads the environment when
# it is CALLED, not when this module is imported. Import-time capture is what
# makes a setting untestable without a subprocess, and it is the reason the two
# directories below are the only place these paths appear.
DEFAULT_BUILTIN_DIR = "/etc/ai-voice/glossaries"
DEFAULT_CUSTOM_DIR = "/glossaries"

SUFFIX = ".txt"

# A profile name is a filename, so it is validated as one rather than trusted.
# `../../etc/passwd` and `foo/bar` are the reason this is an allowlist and not
# a blocklist: the name goes on to be joined onto a directory and written to.
NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

# Ceilings, because these are matched with compiled regexes against every word
# of every transcript. They are generous — the shipped `tech` profile is under
# a hundred entries — and they exist so that a paste accident cannot make every
# subsequent request slow in a way nobody connects to the paste.
MAX_BYTES = 64 * 1024
MAX_ENTRIES = 500
MAX_LINE = 200


class UnknownProfile(LookupError):
    """A request named a profile that does not exist.

    Raised rather than ignored: a caller who believes their vocabulary was
    applied when it was not is the same silence the whole /v1 surface is built
    to prevent.
    """

    def __init__(self, name: str, known: Sequence[str]) -> None:
        super().__init__(name)
        self.name = name
        self.known = list(known)


@dataclass(frozen=True)
class Rejection:
    """One line a PUT refused, and why. Rendered straight onto the wire."""

    line: int
    text: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"line": self.line, "text": self.text, "reason": self.reason}


@dataclass(frozen=True)
class Parsed:
    """What one glossary file says, and what it was not allowed to say."""

    replacements: dict[str, str] = field(default_factory=dict)
    hotwords: tuple[str, ...] = ()
    rejected: tuple[Rejection, ...] = ()

    @property
    def terms(self) -> int:
        return len(self.replacements) + len(self.hotwords)


@dataclass(frozen=True)
class Profile:
    """One named set of terms, as loaded."""

    name: str
    source: str
    path: Path
    parsed: Parsed
    text: str

    @property
    def writable(self) -> bool:
        return self.source == CUSTOM

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "terms": self.parsed.terms,
            "replacements": len(self.parsed.replacements),
            "hotwords": len(self.parsed.hotwords),
            "writable": self.writable,
        }


@dataclass(frozen=True)
class Selection:
    """The compiled result of one request's profile choice."""

    names: tuple[str, ...] = ()
    rules: list[tuple[re.Pattern[str], str]] = field(default_factory=list)
    # Decode-time vocabulary, in the comma-separated form faster-whisper takes.
    # None when nothing was selected, so the absent case reaches the engine as
    # the absent case rather than as an empty string.
    hotwords: str | None = None
    # THE SAME VOCABULARY, UNJOINED. Both fields are built from one list in
    # `select`, and the duplication is the point rather than an oversight: the
    # two engines want different shapes and re-splitting the joined string on
    # ", " would corrupt any term containing a comma.
    #
    # faster-whisper takes one string. Parakeet's decode-time biasing takes a
    # list of phrases, because boosting.compile_automaton tokenises each phrase
    # separately against the model's own piece inventory — see boosting.py.
    # Until that landed there was no second consumer and this field would have
    # been unused, which is why it is new.
    terms: tuple[str, ...] = ()


# ── parsing ───────────────────────────────────────────────────────────────────


def _could_occur_innocently(heard: str) -> bool:
    """Is this left-hand side a shape that can eat a correct sentence?

    The rule is: **a single-word left-hand side needs `force`.** That is the
    `belly` case, which the shipped glossary's own header has argued since long
    before this module — "Belli" is heard as "belly", but a `belly = Belli`
    rule corrupts any sentence that genuinely says belly.

    WHAT THIS IS NOT, stated plainly because the gap matters. It is not a
    dictionary lookup. There is no word list in a python:3.13-slim image, and
    the two ways of getting one were both worse than this:

      * an apt package (`wamerican`, ~1 MB) to answer a question asked a few
        times a day, in an image whose Containerfile installs exactly one apt
        package and says why;
      * an embedded list of the commonest few thousand words, which would not
        contain "belly" — it sits around rank 4000 — and so would confidently
        pass the one case the check exists for. A check that misses its own
        motivating example is worse than a blunt one, because it is trusted.

    So the check over-refuses instead: `catalaxy = Catallaxy` is not an English
    word and is refused anyway, and `force` accepts it in one flag with the
    reason printed. Over-refusal is recoverable; a corrupted transcript is
    discovered weeks later, if ever.

    It also under-refuses, and the rejection message says so. A multi-word
    left-hand side is NOT automatically safe — "my sequel = MySQL" and
    "red is = Redis" would both eat ordinary sentences — but nothing local to
    the rule can see that, and pretending otherwise would be the same false
    confidence as the word list.
    """
    return len(heard.split()) == 1


def parse(text: str, *, force: bool = False) -> Parsed:
    """Read a glossary file. Line numbers are 1-based, as an editor shows them.

    Two line forms, because they are two different jobs:

        catalaxy = Catallaxy    a replacement AND a hotword
        Catallaxy               a hotword only

    `force` switches off the single-word-left-hand-side refusal only. Nothing
    forces a duplicate: two rules with the same left-hand side are a conflict,
    and last-one-wins would silently pick for the operator.
    """
    replacements: dict[str, str] = {}
    hotwords: list[str] = []
    seen_hotwords: dict[str, int] = {}
    seen_replacements: dict[str, int] = {}
    rejected: list[Rejection] = []

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if len(line) > MAX_LINE:
            rejected.append(Rejection(
                number, line[:60] + "…",
                f"longer than {MAX_LINE} characters"))
            continue

        if "=" in line:
            heard, intended = line.split("=", 1)
            heard = heard.strip().lower()
            intended = intended.strip()
            if not heard:
                rejected.append(Rejection(
                    number, line, "empty left-hand side"))
                continue
            if not intended:
                rejected.append(Rejection(
                    number, line,
                    "empty replacement — use the bare form if you meant a "
                    "hotword"))
                continue
            if heard in seen_replacements:
                rejected.append(Rejection(
                    number, line,
                    f"duplicate left-hand side {heard!r}, already defined on "
                    f"line {seen_replacements[heard]}; two rules for one "
                    "heard term is a conflict, not last-one-wins"))
                continue
            if not force and _could_occur_innocently(heard):
                rejected.append(Rejection(
                    number, line,
                    f"{heard!r} is a single word, so this rule would rewrite "
                    "any sentence that says it correctly. Use the bare form "
                    f"({intended!r} on a line of its own) to bias the decoder "
                    "without rewriting, or send force to accept it. Note the "
                    "check cannot see a multi-word phrase that occurs "
                    "innocently — 'my sequel' — so forcing is not the only "
                    "risk here"))
                continue
            seen_replacements[heard] = number
            replacements[heard] = intended
        else:
            key = line.lower()
            if key in seen_hotwords:
                rejected.append(Rejection(
                    number, line,
                    f"duplicate hotword, already on line {seen_hotwords[key]}"))
                continue
            seen_hotwords[key] = number
            hotwords.append(line)

    return Parsed(replacements=replacements, hotwords=tuple(hotwords),
                  rejected=tuple(rejected))


class TooLarge(ValueError):
    """The whole payload is refused, rather than half of it accepted.

    A size problem is a problem with the request, not with three of its lines,
    so it is not reported line by line: half a glossary is not what anybody
    asked for.
    """


def check(text: str, *, force: bool = False) -> Parsed:
    """Validate a proposed profile. Raises TooLarge; never touches disk."""
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise TooLarge(
            f"glossary is {len(encoded)} bytes, over the {MAX_BYTES}-byte "
            "ceiling")
    parsed = parse(text, force=force)
    if parsed.terms > MAX_ENTRIES:
        raise TooLarge(
            f"glossary has {parsed.terms} terms, over the {MAX_ENTRIES}-term "
            "ceiling. Every one of them is a compiled regex matched against "
            "every word of every transcript that selects this profile")
    return parsed


# ── the registry ──────────────────────────────────────────────────────────────


def _stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_ino, info.st_size)


class Registry:
    """Every profile this process can see, and the rules compiled from them."""

    def __init__(self, builtin_dir: Path | None = None,
                 custom_dir: Path | None = None,
                 env_file: str | Path | None = None) -> None:
        self.builtin_dir = Path(builtin_dir) if builtin_dir else None
        self.custom_dir = Path(custom_dir) if custom_dir else None
        self.env_file = Path(env_file) if env_file else None
        self.profiles: dict[str, Profile] = {}
        self._stamps: dict[Path, tuple[int, int, int] | None] = {}
        # Compiled rules, keyed by the exact selection that produced them. The
        # cost of a profile is the regex compilation, not the read, so this is
        # the thing worth keeping. Cleared wholesale on any rescan or write —
        # a half-invalidated cache is how a deployment ends up serving a rule
        # that no file on disk contains.
        self._compiled: dict[tuple[str, ...], Selection] = {}
        self.reload()

    # -- scanning --

    def _watched(self) -> list[Path]:
        """Directories and files whose stat() decides whether to rescan."""
        paths: list[Path] = []
        for directory in (self.builtin_dir, self.custom_dir):
            if directory is not None:
                paths.append(directory)
        if self.env_file is not None:
            paths.append(self.env_file)
        paths.extend(profile.path for profile in self.profiles.values())
        return paths

    def refresh(self) -> None:
        """Rescan if, and only if, something on disk has changed."""
        stamps = {path: _stamp(path) for path in self._watched()}
        if stamps == self._stamps:
            return
        self.reload()

    def reload(self) -> None:
        profiles: dict[str, Profile] = {}
        # Order matters and is deliberate: the env file cannot shadow a
        # built-in and a custom file cannot either. The write routes answer 409
        # on a built-in name for the same reason — a profile whose contents
        # depend on which directory won is a profile nobody can reason about.
        for name, path in self._scan(self.custom_dir):
            profiles[name] = self._read(name, CUSTOM, path)
        if self.env_file is not None and self.env_file.is_file():
            name = _normalise(self.env_file.stem)
            if name and NAME_PATTERN.match(name):
                if name in profiles:
                    log.warning(
                        "STT_GLOSSARY=%s would shadow the custom profile %r; "
                        "the file in %s wins and the env file is ignored",
                        self.env_file, name, self.custom_dir)
                else:
                    profiles[name] = self._read(name, ENV, self.env_file)
            else:
                log.warning(
                    "STT_GLOSSARY=%s has no usable profile name in its "
                    "filename; expected something matching %s",
                    self.env_file, NAME_PATTERN.pattern)
        for name, path in self._scan(self.builtin_dir):
            if name in profiles:
                log.warning(
                    "%s would shadow the built-in profile %r and is ignored; "
                    "built-ins are read-only", path, name)
            profiles[name] = self._read(name, BUILTIN, path)

        self.profiles = profiles
        self._compiled.clear()
        self._stamps = {path: _stamp(path) for path in self._watched()}

    def _scan(self, directory: Path | None) -> list[tuple[str, Path]]:
        if directory is None or not directory.is_dir():
            return []
        found: list[tuple[str, Path]] = []
        for entry in sorted(directory.iterdir()):
            if not entry.is_file() or entry.suffix.lower() != SUFFIX:
                continue
            name = _normalise(entry.stem)
            if not NAME_PATTERN.match(name):
                log.warning("ignoring %s: %r is not a usable profile name",
                            entry, entry.stem)
                continue
            found.append((name, entry))
        return found

    def _read(self, name: str, source: str, path: Path) -> Profile:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("could not read %s: %s", path, exc)
            text = ""
        # force=True: a file already on disk is not a proposal to validate, it
        # is state to load, and dropping its rules at boot because one of them
        # is single-word would change what a deployment transcribes without
        # anyone having asked for it. Structural problems are still named in
        # the log rather than swallowed.
        parsed = parse(text, force=True)
        for rejection in parsed.rejected:
            log.warning("%s line %d rejected: %s — %s", path, rejection.line,
                        rejection.text, rejection.reason)
        return Profile(name=name, source=source, path=path, parsed=parsed,
                       text=text)

    # -- reading --

    @property
    def names(self) -> list[str]:
        return sorted(self.profiles)

    def get(self, name: str) -> Profile:
        profile = self.profiles.get(_normalise(name))
        if profile is None:
            raise UnknownProfile(name, self.names)
        return profile

    def select(self, names: Iterable[str]) -> Selection:
        """Compile the rules for one request's choice of profiles.

        Merged in the order given, so a later profile's rule for the same heard
        term wins. Selecting several at once is discouraged in the README for
        the measured reason above; this is what happens when it is done anyway.
        """
        wanted = tuple(_normalise(name) for name in names if _normalise(name))
        if not wanted:
            return Selection()
        cached = self._compiled.get(wanted)
        if cached is not None:
            return cached

        replacements: dict[str, str] = {}
        hotwords: list[str] = []
        for name in wanted:
            profile = self.get(name)
            replacements.update(profile.parsed.replacements)
            hotwords.extend(profile.parsed.hotwords)

        # Every replacement's INTENDED spelling is a hotword too: the term the
        # decoder should have produced is exactly the term to bias it towards.
        vocabulary = sorted(set(list(replacements.values()) + hotwords))
        selection = Selection(
            names=wanted,
            rules=glossary.compile_rules(replacements),
            hotwords=", ".join(vocabulary) or None,
            terms=tuple(vocabulary),
        )
        self._compiled[wanted] = selection
        return selection

    # -- writing --

    def writability(self) -> tuple[bool, str]:
        """Whether custom profiles can be written, and why not when they cannot.

        This is not a permission system and calling it one would be dishonest.
        It reports whether a deployment mounted somewhere to persist.
        """
        directory = self.custom_dir
        if directory is None:
            return False, "this process has no custom glossary directory"
        if not directory.exists():
            return False, (
                f"nothing is mounted at {directory}: this deployment has no "
                "writable glossary volume, so a profile written here would be "
                "gone on the next restart. Mount a volume there and restart")
        if not directory.is_dir():
            return False, f"{directory} exists but is not a directory"
        if not os.access(directory, os.W_OK | os.X_OK):
            return False, (
                f"{directory} is not writable by uid {os.getuid()}: a bind "
                "mount arrives with the host directory's ownership. Either "
                "chown it to 1000, or set `user:` in compose to a uid that "
                "owns it")
        return True, ""

    def write(self, name: str, text: str) -> Path:
        """Replace a custom profile's file, atomically, and invalidate.

        os.replace rather than a plain open-and-write: a request arriving while
        a half-written file is on disk would compile a truncated glossary and
        cache it, and nothing about the resulting transcript would look wrong.
        """
        directory = self.custom_dir
        assert directory is not None  # the route checks writability first
        path = directory / f"{_normalise(name)}{SUFFIX}"
        handle, temporary = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(text)
            os.replace(temporary, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise
        self.reload()
        return path

    def remove(self, name: str) -> None:
        profile = self.get(name)
        profile.path.unlink()
        self.reload()


def _normalise(name: str) -> str:
    """Profile names are case-insensitive and lowercase on disk.

    Not cosmetic: macOS and SMB volumes are case-insensitive, so `Tech.txt` and
    `tech.txt` are one file there and two on Linux. A name that means different
    things depending on the filesystem under the mount is the "why is `tech`
    different on that box" question this design exists to avoid.
    """
    return name.strip().lower()


def valid_name(name: str) -> bool:
    return bool(NAME_PATTERN.match(_normalise(name)))


def split_selection(raw: str | None) -> list[str]:
    """`glossary=tech,dictation` — comma-separated, as ADR 0002 writes it."""
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_registry() -> Registry:
    """The registry this process serves, from the environment.

    STT_GLOSSARY is the pre-profile variable and is deliberately unset in the
    image now: it used to point at the file the image shipped, which is exactly
    the always-on personal glossary this module exists to stop. A deployment
    that still sets it keeps its vocabulary, as a profile like any other, opted
    into per request rather than applied to everything.
    """
    return Registry(
        builtin_dir=os.getenv("STT_GLOSSARY_BUILTIN", DEFAULT_BUILTIN_DIR),
        custom_dir=os.getenv("STT_GLOSSARY_DIR", DEFAULT_CUSTOM_DIR),
        env_file=os.getenv("STT_GLOSSARY", "").strip() or None,
    )
