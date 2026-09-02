"""A narrow client for the MeTube already running on this NAS.

    ix-metube-metube-1, host port 30097, version 2026.08.28, yt-dlp 2026.08.19

WHY DELEGATE RATHER THAN EMBED A DOWNLOADER. It is running, it is maintained,
and it updates its own yt-dlp, so there is no extractor rot for this repository
to chase. `download_type:"audio"` means a two-hour 4K video never has its video
stream pulled at all -- roughly 1 MB/min at opus, so ~131 MB for a 2h14m
podcast instead of tens of gigabytes. And `/audio_download/` serves the
finished file over HTTP with `Accept-Ranges: bytes`, which is the single
biggest reason this is delegation and not integration: there is NO shared
volume to arrange between two separate TrueNAS apps.

LICENCE. MeTube is AGPL-3.0. Calling its HTTP API puts no obligation on this
repository -- a network API client is not a derivative work -- and §13 binds
the OPERATOR, who is the user, running alexta69's published image unmodified on
his own box. Forking it or vendoring its code would be a different answer.
That is why this file is a client and contains none of MeTube's code.

FIVE THINGS VERIFIED AGAINST ITS SOURCE AND LIVE, each of which is a bug if
assumed the other way.

 1. `auto_start:false` genuinely resolves WITHOUT downloading. __extract_info
    runs on every /add; with auto_start false __add_download routes to
    `self.pending.put(download)` (ytdl.py:1648-1655) and no bytes move.
    auto_start defaults to TRUE when the field is None, so it is sent
    explicitly on every call below and never omitted.
 2. GET /history returns THREE lists -- done, queue AND pending -- and an
    auto_start:false item lands in **pending**, not queue. Reading `queue`
    alone finds nothing and looks like a failed resolve.
 3. `ids` in /start and /delete are the URLs, NOT the short `id` field.
    PersistentQueue.put keys on `value.info.url` (ytdl.py:1251); deleting by
    the short id returns {"status":"ok"} and SILENTLY DOES NOTHING. Every id
    passed from this module is a URL, and `abandon` re-reads /history to prove
    the record actually went.
 4. `filename` is OUTPUT_TEMPLATE sanitised and byte-trimmed. It is never
    predicted here -- it is read back from /history and percent-encoded.
 5. A REJECTED add still creates a record, in `done`, with status "error".
    Abandoning has to clear both queues, not just the one we put it in.

AND ONE THING THAT IS NOT OURS TO FIX. MeTube has NO AUTHENTICATION of any
kind: Config._DEFAULTS carries no auth, user, password or token key, and an
unauthenticated GET /history from off-NAS returns 200. This service does not
widen that -- our ingestion routes sit behind the gateway's key check, so we
are a strictly narrower client of something already open to the LAN. But
shipping a UI that makes MeTube load-bearing is the moment to close it:
unpublish 30097 or firewall it to the NAS and reach it by LAN IP. That is a
deployment change, it is in this service's README, and it must not be allowed
to hide behind this feature.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from . import config

log = logging.getLogger("voice-ui.metube")

__all__ = ["MeTubeError", "MeTube", "configured"]

# Terminal success. Anything else in `done` means _post_download_cleanup
# rewrote the status to "error" and nulled filename.
FINISHED = "finished"


class MeTubeError(RuntimeError):
    """MeTube refused, or could not be reached. The message is user-facing."""


def configured() -> bool:
    return bool(config.METUBE_URL)


class MeTube:
    """One MeTube, one httpx client. Constructed once at startup."""

    def __init__(self, client: httpx.AsyncClient, base: str | None = None) -> None:
        self.client = client
        self.base = (base if base is not None else config.METUBE_URL).rstrip("/")

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(self.base + path, json=body,
                                              timeout=30.0)
        except httpx.RequestError as exc:
            # Never a stuck spinner. The page turns this into "ingestion
            # unavailable -- file upload still works" and every other feature
            # keeps working, which is the whole degradation stance.
            raise MeTubeError(
                f"ingestion unavailable: MeTube did not answer "
                f"({type(exc).__name__}). File upload still works.") from None

        try:
            payload = response.json()
        except ValueError:
            raise MeTubeError(
                f"MeTube answered {response.status_code} with a body that is "
                f"not JSON") from None

        if response.status_code >= 400 or payload.get("status") == "error":
            # MeTube's own message, verbatim and untranslated. Its url_guard
            # says things like 'Refusing to fetch internal host "localhost"',
            # which is more useful and more true than anything we would write
            # over the top of it.
            raise MeTubeError(str(payload.get("msg")
                                  or f"MeTube refused with {response.status_code}"))
        return payload

    async def history(self) -> dict[str, list[dict[str, Any]]]:
        try:
            response = await self.client.get(self.base + "/history", timeout=15.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            raise MeTubeError(
                f"could not read MeTube's queue ({type(exc).__name__})") from None
        # Three lists, and `pending` is the one an auto_start:false item is in.
        return {key: list(payload.get(key) or [])
                for key in ("pending", "queue", "done")}

    async def find(self, url: str) -> tuple[str, dict[str, Any]] | None:
        """(which list, the record) for `url`, or None if MeTube has forgotten it."""
        history = await self.history()
        for where, entries in history.items():
            for entry in entries:
                if entry.get("url") == url:
                    return where, entry
        return None

    async def add(self, url: str, *, auto_start: bool,
                  download_type: str = "audio",
                  clip_start: float | None = None,
                  clip_end: float | None = None) -> None:
        body: dict[str, Any] = {
            "url": url,
            "download_type": download_type,
            # MeTube 400s on any quality but "best" for opus and the m4a-shaped
            # choices, so this is not a knob and is not exposed as one.
            "quality": "best",
            "format": config.METUBE_FORMAT,
            # See config.METUBE_FOLDER: /download/ and /audio_download/ are the
            # same directory on this deployment, so without a folder our ingest
            # lands in the user's music library.
            "folder": config.METUBE_FOLDER,
            "auto_start": auto_start,
        }
        # Free server-side trim: a three-hour stream is transcribed as a
        # ten-minute excerpt without fetching the rest. This is why the clip
        # fields are on the confirm card itself rather than buried in an expert
        # panel -- it turns "no, too much" into "yes, but only this bit".
        if clip_start is not None:
            body["clip_start"] = clip_start
        if clip_end is not None:
            body["clip_end"] = clip_end
        await self._post("/add", body)

    async def start(self, url: str) -> None:
        await self._post("/start", {"ids": [url]})

    async def delete(self, url: str, where: str) -> None:
        await self._post("/delete", {"ids": [url], "where": where})

    async def abandon(self, url: str) -> bool:
        """Reap every trace of `url`. True if MeTube no longer knows it.

        THIS IS THE PATH THE ORDERING CREATES AND THEREFORE THE ONE THAT MUST
        BE TESTED. Resolving calls POST /add BEFORE probing -- on purpose, so
        that MeTube's url_guard is the thing that decides whether a URL may be
        fetched at all -- which means every link the user then declines has
        left a pending record behind. Probing first would leave nothing, and
        would also mean our own guard was the only one that ran.

        Both queues are cleared because a REJECTED add lands in `done` with
        status "error" rather than in `pending`. And the result is verified by
        re-reading /history rather than trusting the answer, because deleting
        by the wrong id returns {"status":"ok"} and silently does nothing --
        an unverified abandon is indistinguishable from a working one right up
        until the queue is full of orphans.
        """
        for where in ("queue", "done"):
            try:
                await self.delete(url, where)
            except MeTubeError as exc:
                # Not fatal on its own: the item is in one of the two lists,
                # so one of these two calls is expected to be a no-op.
                log.info("delete from %s said: %s", where, exc)
        try:
            return await self.find(url) is None
        except MeTubeError:
            return False

    def audio_url(self, filename: str) -> str:
        """The HTTP URL of a finished file.

        `filename` comes from /history and is never constructed here. quote()
        with a permissive safe set because MeTube's own static route serves the
        path as written and a title with a slash in it has already been
        sanitised on its side; percent-encoding the rest is what keeps a title
        containing "#", "?" or a space from truncating the URL.
        """
        return f"{self.base}/audio_download/{quote(filename, safe='/')}"
