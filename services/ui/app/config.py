"""Every knob this service has, read once at import.

The process is the unit of configuration here, exactly as it is in the four
siblings: a container is restarted to change a setting, so reading the
environment at import means `docker inspect` and the startup log agree with
what the code will do for the whole life of the process.

EVERYTHING BELOW IS OPTIONAL AND EVERY DEFAULT DEGRADES RATHER THAN FAILS.
That is the stance for this service specifically: it is a convenience in front
of a stack that already works without it, and the worst outcome would be a UI
that refuses to start — or worse, starts and hangs — because MeTube is down or
yt-dlp is missing. Uploads and TTS must never depend on ingestion being
available.
"""

from __future__ import annotations

import os

__all__ = [
    "GATEWAY_URL", "GATEWAY_VERIFY", "GATEWAY_API_KEY", "gateway_authorization",
    "METUBE_URL", "METUBE_FOLDER", "METUBE_FORMAT", "METUBE_VIDEO_FORMAT",
    "PROBE", "PROBE_TIMEOUT", "MAX_UPLOAD_BYTES", "MAX_CAPTION_BYTES",
    "MAX_MEDIA_BYTES",
    "CONFIRM_SECONDS", "CONFIRM_BYTES", "STT_RTF_SEED", "STT_BUDGET_SECONDS",
    "VOICE_DIR", "MAX_CLIP_BYTES", "MAX_CLIP_SECONDS", "RESOLVE_PER_MINUTE",
    "flag",
]


def flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# The ONLY address this service speaks HTTP to for speech work. Not
# stt-stack:8000, not tts-stack:8001, not tts-long:8002 -- those three have no
# `ports` entry in compose.yaml, so a UI that addressed them would work on a
# laptop with all five containers and fail on the NAS, which is the exact bug
# the port deletion was made to prevent.
GATEWAY_URL = os.getenv("UI_GATEWAY_URL", "http://voice-gateway:8080").rstrip("/")

# WHETHER TO VERIFY THE GATEWAY'S CERTIFICATE ON THE INTERNAL HOP.
#
# Default true, and it must stay true for anything crossing a network. It is
# set to 0 in compose for one specific reason: once the gateway serves HTTPS it
# serves ONLY HTTPS, including to this container, and this container reaches it
# as `https://voice-gateway:8080` -- a compose service name, on the app's own
# bridge network. No certificate for a public hostname can match that name, and
# no certificate authority will issue one that does.
#
# The alternatives were considered and are worse: a second listener on plain
# HTTP would be the second door this whole change removed, and a private CA
# with `voice-gateway` in its SAN is a certificate authority to maintain,
# renew and distribute for one hop between two processes on one host.
#
# What this does NOT do is weaken anything a client sees. The published port
# still presents the real wildcard certificate and still validates. This is the
# hop that never leaves the box.
GATEWAY_VERIFY = flag("UI_GATEWAY_VERIFY", True)

# THE KEY THIS SERVICE PRESENTS TO THE GATEWAY, and the reason the page no
# longer has a box for one.
#
# It used to be the browser's: the page kept a key in localStorage, put it on
# every XHR, and this service forwarded it untouched. That made :30080 the
# trust boundary and this container a pipe. The user's decision is that this is
# not a bring-your-own-key tool, so the credential moved here.
#
# WHAT THAT COSTS, AND IT IS NOT SMALL: the trust boundary moves from :30080 to
# :30081. Anyone who can open the page is authenticated by it, because this
# process signs their requests for them. That is a defensible trade for a tool
# on a LAN behind a firewall and it is not defensible for anything reachable
# from outside one. Publishing 30081 more widely than 30080 now means MORE
# access, not less.
#
# UNSET BY DEFAULT, deliberately, exactly like GATEWAY_API_KEYS in
# compose.yaml: a key invented in a file that gets deployed is how a
# placeholder becomes production credentials. Unset means no header is added
# and any inbound one is forwarded as before, so a deployment with
# GATEWAY_API_KEYS also unset behaves precisely as it did.
GATEWAY_API_KEY = (os.getenv("UI_GATEWAY_API_KEY") or "").strip()


def gateway_authorization(inbound: str | None) -> str | None:
    """The Authorization to put on a request to the gateway, or None.

    One function rather than the same conditional at each of the four call
    sites -- the proxy, the key probe, the ingest hand-off and the page's own
    calls -- because "which credential goes on this hop" is exactly the kind of
    question that gets answered three ways and then disagrees in production.

    Ours WINS over an inbound header when it is configured. A caller who sends
    their own must not be able to make this service present a different key
    than the one it was given, and there is no case where a browser on this
    page sends one at all.
    """
    if GATEWAY_API_KEY:
        return f"Bearer {GATEWAY_API_KEY}"
    return inbound

# MeTube. Unset means the URL box is not rendered at all and the page says
# "link ingestion not configured" -- not a broken button, not a spinner.
#
# A HOST ADDRESS, never a compose service name. MeTube is a separate TrueNAS
# app (ix-metube-metube-1) with host_network:false and its web port published
# at 30097, so it shares no DNS with this app's internal network. There is no
# `metube` name to resolve and `host.docker.internal` is not present either
# without an explicit extra_hosts entry. Use the NAS's LAN IP.
METUBE_URL = (os.getenv("UI_METUBE_URL") or "").strip().rstrip("/")

# Mandatory in effect, not merely tidy. MeTube's AUDIO_DOWNLOAD_DIR defaults to
# "%%DOWNLOAD_DIR" and is unset on this deployment, so /download/ and
# /audio_download/ are the SAME directory -- verified. Without a folder every
# file we ingest lands in the middle of the user's music library. CUSTOM_DIRS
# and CREATE_CUSTOM_DIRS both default true and are unset, so MeTube creates it.
METUBE_FOLDER = os.getenv("UI_METUBE_FOLDER", "stt-ingest")

# opus at ~1 MB/min. A 2h14m podcast is ~131 MB of audio rather than tens of
# gigabytes of 4K video, because download_type:"audio" never fetches the video
# stream. MeTube 400s on any `quality` but "best" for opus, so quality is not
# a knob here.
METUBE_FORMAT = os.getenv("UI_METUBE_FORMAT", "opus")

# THE CONTAINER ASKED FOR WHEN THE USER TICKS "keep the video", and only then.
# Verified live against this deployment's MeTube: POST /add with
# download_type:"video", format:"mp4", quality:"best" answers {"status":"ok"}
# and writes "Me at the zoo.mp4" into the stt-ingest folder.
#
# mp4 rather than the source container, because the file is served straight
# back to a <video> element: Matroska is what canPlayType answers "maybe" to
# and then declines, and a picture that will not render is the one thing this
# choice exists to produce. yt-dlp remuxes rather than re-encodes for mp4, so
# it costs no compute on MeTube's side.
METUBE_VIDEO_FORMAT = os.getenv("UI_METUBE_VIDEO_FORMAT", "mp4")

# The metadata probe. On by default; UI_PROBE=0 gives a title-only confirm card
# and never spawns yt-dlp. See app/probe.py for why a probe is not a downloader
# and what it is still on the hook for.
PROBE = flag("UI_PROBE", True)
PROBE_TIMEOUT = float(os.getenv("UI_PROBE_TIMEOUT", "20"))

# 2 GiB. services/stt/app/main.py:138 is a bare `file.file.read()` on an
# UploadFile -- no Content-Length check, no cap, no streaming -- so a 4 GB MKV
# is buffered whole into the stt container's 6 GB memory limit. Rejecting on
# Content-Length here costs one comparison and happens before a byte is
# forwarded. The page also extracts audio in the browser above ~50 MB, which
# is the real fix; this is the boundary behind it, because a client-side check
# is a courtesy and not a boundary.
MAX_UPLOAD_BYTES = int(os.getenv("UI_MAX_UPLOAD_BYTES", str(2 * 1024**3)))

# 8 MiB, and it is a sanity bound rather than a real limit. A captions download
# is a subtitle track and nothing else -- yt-dlp sets skip_download, so no media
# stream is fetched at all -- and an hour of dense dialogue is on the order of
# 100 KB of WebVTT. /ui/captions reads the whole file into memory to parse it,
# which is the right call for something that size and the wrong one for
# anything that is not, so the ceiling exists to catch the case where MeTube
# hands back something that is NOT a subtitle file. Without it, a filename that
# got past the suffix check would be buffered whole into a container with
# mem_limit: 384m -- the same shape of bug as the clip route's, which is
# documented at MAX_CLIP_BYTES and was an OOM kill rather than a message.
MAX_CAPTION_BYTES = int(os.getenv("UI_MAX_CAPTION_BYTES", str(8 * 1024**2)))

# THE CEILING ON WHAT /ui/media WILL RELAY, and it is deliberately NOT
# MAX_UPLOAD_BYTES. That one bounds what is pushed INTO the stack -- a body
# services/stt reads whole into a container with a 6 GB limit, which is why it
# is a memory question. This bounds what is pulled OUT of it, down a domestic
# connection, into a browser tab: nothing here is buffered, so it is not about
# memory at all, and answering "how big a file may stt decode" with "how big a
# file may this laptop stream" would tie two unrelated decisions together.
#
# 4 GiB is about two hours of 1080p. Past that the honest answer is to fetch
# the audio, keep the transcript and open the file some other way -- and the
# confirm card says what the video costs before anything is downloaded.
MAX_MEDIA_BYTES = int(os.getenv("UI_MAX_MEDIA_BYTES", str(4 * 1024**3)))

# WHEN TO NAG, and why these two numbers.
#
# Below BOTH thresholds the confirm dialog is skipped and the fetch starts. Ten
# minutes of audio is ~10 MB at opus and, at the conservative 8.5x figure the
# gateway's own timeout was built on, ~71 s of transcription -- an order of
# magnitude inside GATEWAY_STT_TIMEOUT (900 s) and well inside anyone's
# patience. A dialog there is pure friction, and a dialog that fires on
# everything is a dialog people learn to dismiss without reading, which is
# exactly how the three-hour stream gets through.
#
# The size threshold is the second gate rather than an alternative: a short
# video with an enormous audio stream is still a real download.
CONFIRM_SECONDS = float(os.getenv("UI_CONFIRM_SECONDS", "600"))
CONFIRM_BYTES = int(os.getenv("UI_CONFIRM_BYTES", str(50 * 1024**2)))

# THE NUMBER THIS REPOSITORY CONTRADICTS ITSELF ABOUT, seeded at the
# conservative end on purpose.
#
#   root README.md:95                 47-63x realtime
#   gateway/app/main.py:117-123       8.5-10.4x   <- the 900 s timeout and the
#                                                    504 help text are built
#                                                    on THIS one
#   stt/README.md:590                 ~5x on four cores
#
# A factor of twelve apart. The dialog that matters most -- "this is a 2h14m
# podcast" -- is wrong by 5x if the optimistic figure is quoted and the
# pessimistic one is true, and at 8.5x that file needs 946 s of compute, which
# EXCEEDS the gateway's own 900 s ceiling. So the seed is 8.5, the page
# labels it an estimate, the page keeps its own EMA from the realtime_factor
# the native /transcribe route returns on every real transcription, and the
# page warns when duration/rtf crosses the budget below. Someone must
# re-measure on orko and correct main.py:121's help text in the same change,
# or the UI and the gateway's own 504 will disagree in front of one user.
STT_RTF_SEED = float(os.getenv("UI_STT_RTF", "8.5"))
STT_BUDGET_SECONDS = float(os.getenv("UI_STT_BUDGET", "900"))

# The reference-clip store. Shared with tts-long as a named volume: this
# service is the only WRITER, tts-long mounts it read-only and rescans when the
# directory changes. See app/clips.py and services/tts-long/app/voices.py.
VOICE_DIR = os.getenv("UI_VOICE_DIR", "/voices")

# Resemble's own guidance is 10-30 s of one speaker. 30 s of 24 kHz mono WAV is
# 1.4 MB; 25 MB is room for a phone recording that has not been transcoded yet
# and is still nowhere near a memory problem.
MAX_CLIP_BYTES = int(os.getenv("UI_MAX_CLIP_BYTES", str(25 * 1024**2)))
MAX_CLIP_SECONDS = float(os.getenv("UI_MAX_CLIP_SECONDS", "30"))

# /ui/resolve spawns a process that makes an outbound request on a URL a user
# chose. That is a scanning primitive if it is free, so it is not free.
RESOLVE_PER_MINUTE = int(os.getenv("UI_RESOLVE_PER_MINUTE", "12"))
