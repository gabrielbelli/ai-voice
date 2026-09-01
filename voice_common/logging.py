"""One basicConfig line, plus the level switch none of the three services has.

The configuration below is byte-identical in tts-stack/app/main.py,
tts-long/app/main.py and stt-stack/app/main.py. On its own that is a copied
line, not a module, and copying it again would be cheaper than importing it.

What makes it worth a module is the second line: today there is no way to get
DEBUG output from any of the three services without editing the source and
rebuilding an image. A running container that has started misbehaving is
exactly when that is wanted and exactly when it is impossible.

Six lines. If this grows a formatter class or JSON output, that belongs to
whatever is consuming the logs, not to three services that write to stderr.
"""

from __future__ import annotations

import logging
import os

__all__ = ["setup"]

FORMAT = "%(asctime)s %(levelname)s %(message)s"


def setup(name: str, prefix: str) -> logging.Logger:
    """Configure root logging and return the service's logger.

    `prefix` is the service's env var prefix — STT or TTS — so the level is
    read from STT_LOG_LEVEL or TTS_LOG_LEVEL and no new operator-visible
    naming convention is invented for it. An unrecognised value falls back to
    INFO rather than raising: a typo in a log level must never be the reason a
    service does not start.
    """
    wanted = os.getenv(f"{prefix}_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, wanted, None)
    effective = level if isinstance(level, int) else logging.INFO
    logging.basicConfig(level=effective, format=FORMAT)
    # basicConfig is a no-op once root has a handler, and something upstream —
    # uvicorn's own logging config, a test runner, a wrapper script — may well
    # have installed one before this runs. Without the explicit setLevel the
    # variable would be read, parsed, and then quietly ignored, which is the
    # same failure as not having the switch at all.
    logging.getLogger().setLevel(effective)
    log = logging.getLogger(name)
    if not isinstance(level, int):
        log.warning("%s_LOG_LEVEL=%r is not a level name; using INFO",
                    prefix, wanted)
    return log
