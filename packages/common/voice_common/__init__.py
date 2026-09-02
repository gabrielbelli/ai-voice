"""Shared runtime for stt-stack, tts-stack and tts-long.

Three services, three hand-vendored copies of the same auth, the same error
envelope, the same health route and the same TLS entrypoint. The copies drifted
— 197, 187 and 170 differing lines between the three app/auth.py files — and
one adversarial review round found three DIFFERENT defects, one per copy,
because each had drifted separately. That is what this package exists to stop,
and the argument is measured rather than predicted. voice_common.auth carries
the detail.

What belongs here: the wire contract. What a client sees, what an operator
configures, what a healthcheck probes. What does not: anything that is a
property of one model file, one wheel, one image or one service's reason for
existing. README.md draws the line in full, with the list of things that were
considered and deliberately left behind.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
