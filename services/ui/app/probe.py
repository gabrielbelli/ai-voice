"""Five scalars out of a URL, and not one byte of media.

WHY THIS EXISTS AT ALL, given that the whole point of this service is to
delegate downloading to MeTube rather than embed a downloader.

MeTube resolves the metadata and then throws it away. `__extract_info`
(ytdl.py:1570) calls `extract_info(url, download=False)` on every POST /add
regardless of auto_start, so the full yt-dlp info-dict -- duration,
filesize_approx, per-format audio sizes, all of it -- exists inside MeTube for
a moment. It is stored as `self.entry` and then deliberately stripped on the
way out: `_PUBLIC_EXCLUDED_FIELDS = ("entry", "subtitle_files")` (ytdl.py:521).
`DownloadInfo` (ytdl.py:461-501) has no duration field anywhere in MeTube's
source, and `self.size` stays None until os.path.getsize fills it AFTER the
download finishes (ytdl.py:1065).

So a pending MeTube entry gives you a title and a live_status. The brief's
whole requirement -- "title, duration and size, with the estimated
transcription time" -- is computed from the one field MeTube never exposes.

A PROBE IS NOT A DOWNLOADER, and the difference is enumerable rather than
rhetorical. This has no output template, no writable directory, no format
selection, no post-processing chain, no cookie jar, no concurrency, no retry
policy, no resume, no progress reporting and no disk. It runs once, for at
most PROBE_TIMEOUT seconds, and produces at most a few hundred bytes of JSON
that are parsed into five values. Everything MeTube absorbed, it keeps.

THE HONEST COST, because it is a real one. This is a second yt-dlp to keep
current and it rots on the same weekly schedule as MeTube's. Three things
blunt it:

  * It runs THIRD, only on a URL that our own guard and then MeTube's
    url_guard have both already accepted. It is never the thing that decides
    whether a URL is fetchable.
  * It runs in a subprocess with a hard kill, cwd on a directory it cannot
    write, `--no-config` and `--no-cache-dir` so nothing on disk influences it
    and nothing is left behind.
  * Its failure mode is a title-only confirm card, never a blocked or broken
    fetch. A stale extractor here costs an estimate, not a download.

If the operator would rather not carry it at all, UI_PROBE=0 removes it and
the card degrades to MeTube's title. The upstream fix is three lines in
MeTube's `to_public_dict` to carry duration and filesize_approx, which is
worth sending and is not worth blocking on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

from . import config

log = logging.getLogger("voice-ui.probe")

__all__ = ["Probe", "available", "run"]

# argv, fixed. Nothing from the user reaches it except the URL, in the last
# position, passed as a list element so no shell ever sees it.
#
#   -J                    one JSON object on stdout, nothing else
#   --skip-download       belt to --no-download's braces; nothing is fetched
#   --no-playlist         a playlist URL resolves as the single video it names,
#                         so pasting a "watch?v=x&list=y" link does not enqueue
#                         someone's 400-video mix
#   --no-config           no /etc/yt-dlp.conf, no ~/.config/yt-dlp, no
#                         .netrc: nothing on disk can change what this does
#   --no-cache-dir        writes nothing
#   --socket-timeout 10   a hung TLS handshake does not eat the whole budget
#   --no-warnings         warnings on stderr would otherwise be interleaved
ARGV = ["-J", "--skip-download", "--no-playlist", "--no-config",
        "--no-cache-dir", "--no-warnings", "--socket-timeout", "10"]


class Probe(dict):
    """A plain dict; named so the type is readable at the call site."""


def available() -> bool:
    return config.PROBE and shutil.which("yt-dlp") is not None


def _best_audio_bytes(info: dict[str, Any]) -> int | None:
    """The size of the audio we would actually pull, not of the video.

    MeTube requests `bestaudio[ext=opus]/bestaudio/best`, so the number worth
    showing is the largest audio-only format's size, not filesize_approx --
    which on a 4K video is the whole video and would make the card warn about
    tens of gigabytes we are never going to fetch.
    """
    best: int | None = None
    for fmt in info.get("formats") or []:
        if not isinstance(fmt, dict):
            continue
        if fmt.get("vcodec") not in (None, "none"):
            continue
        if fmt.get("acodec") in (None, "none"):
            continue
        size = fmt.get("filesize") or fmt.get("filesize_approx")
        if isinstance(size, (int, float)) and size > 0:
            best = max(best or 0, int(size))
    if best is not None:
        return best

    # No per-format sizes (many extractors give none). Fall back to the
    # measured rule of thumb rather than to filesize_approx: opus at MeTube's
    # settings is about 1 MB per minute, and a number the card can label
    # "approximate" beats a number that is honestly the video's size.
    duration = info.get("duration")
    if isinstance(duration, (int, float)) and duration > 0:
        return int(duration / 60 * 1024 * 1024)
    return None


def _real_subtitles(info: dict[str, Any]) -> bool:
    """Human captions, as distinct from the machine ones we are replacing.

    `automatic_captions` is YouTube's own ASR and is exactly what this stack
    exists to do better. `subtitles` is what a person uploaded, and when it is
    present the honest answer is to offer it: MeTube's
    download_type:"captions" sets skip_download and returns a real transcript
    in about two seconds for near-zero cost.
    """
    subs = info.get("subtitles")
    return isinstance(subs, dict) and bool(subs)


async def run(url: str) -> Probe | None:
    """Probe `url`, or return None if we could not, for any reason.

    None is a first-class answer, not an error path: every caller degrades to
    a title-only card, so there is nothing here worth raising over.
    """
    if not available():
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", *ARGV, "--", url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # A directory this process cannot write. yt-dlp writes nothing with
            # the flags above; cwd is the second lock on that, so a future flag
            # change cannot silently start dropping files next to the app.
            cwd="/",
        )
    except (OSError, ValueError) as exc:
        log.warning("probe could not start: %s", exc)
        return None

    try:
        out, err = await asyncio.wait_for(proc.communicate(),
                                          timeout=config.PROBE_TIMEOUT)
    except asyncio.TimeoutError:
        # kill(), not terminate(): the point of the timeout is a hard ceiling,
        # and a process that ignores SIGTERM would otherwise sit here holding
        # the budget it was given to bound.
        proc.kill()
        await proc.wait()
        log.warning("probe timed out after %.0f s", config.PROBE_TIMEOUT)
        return None

    if proc.returncode != 0:
        # Truncated: yt-dlp's stderr on a private video is one useful line, and
        # on a site that changed its player is a page of them.
        log.info("probe exited %s: %s", proc.returncode,
                 err.decode("utf-8", "replace")[:300].strip())
        return None

    try:
        info = json.loads(out)
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None

    # FIVE SCALARS AND NOTHING ELSE. The info-dict is a large object built from
    # a page this container fetched on a user's instruction, and the only
    # defensible thing to do with it is take the few numbers the dialog needs
    # and drop the rest on the floor. Nothing here is a response body, a URL,
    # or anything a caller could steer.
    duration = info.get("duration")
    return Probe(
        title=str(info.get("title") or "")[:300] or None,
        uploader=str(info.get("uploader") or info.get("channel") or "")[:200] or None,
        duration=float(duration) if isinstance(duration, (int, float)) else None,
        bytes=_best_audio_bytes(info),
        is_live=bool(info.get("is_live")) or info.get("live_status") == "is_live",
        has_subtitles=_real_subtitles(info),
    )
