# Third-party notices

This repository is BSD-2-Clause (see `LICENSE`). This file records what came
from elsewhere, under what licence, and in what form — because "we learned it
from X" and "we copied X" carry different obligations, and the difference is
easy to lose once the code is in the tree.

Three categories, deliberately kept apart:

| Form | What it means | What is owed |
|---|---|---|
| **Depended on** | Called over the network or installed by pip. Not a derivative work. | Nothing beyond honouring the dependency's own terms |
| **Copied** | Their source, verbatim or lightly edited, in our files | Their copyright and permission notice must travel with it |
| **Learned from** | An idea, a layout, a parameter range, a sequence of steps | Nothing legally. Credited here anyway |

---

## Depended on

### MeTube — `alexta69/metube` — **AGPL-3.0**

Used for URL ingestion. **`services/ui/app/metube.py` is an HTTP client and
contains none of MeTube's code**, and that is a deliberate licence decision
rather than a style one.

AGPL-3.0's §13 obliges an operator who *modifies* the program and lets users
interact with it over a network to offer them the modified source. Calling a
separate program's network API is not a derivative work, and the operator here
runs the published image unmodified on his own machine. Forking or vendoring
MeTube would have been a materially different answer, and would have pulled
this repository's UI service into AGPL territory.

**If anyone later vendors, patches or embeds MeTube, that reasoning stops
holding.** Re-read this section first.

### pip dependencies

| Package | Version | Licence |
|---|---|---|
| yt-dlp | 2026.8.19 | Unlicense (public domain) |
| fastapi | 0.121.2 | MIT |
| uvicorn[standard] | 0.38.0 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| python-multipart | 0.0.20 | Apache-2.0 |

yt-dlp is used **as a metadata probe only** — `extract_info(download=False)`,
to resolve a pasted link to a title, duration and size so the user can confirm
before anything is fetched. The fetching itself is MeTube's job.

---

## Learned from

No code from these was copied verbatim. Credited because the ideas were load
bearing, and because a reader deserves to know where to look for the original.

### `devnen/Chatterbox-TTS-Server` — MIT — Copyright (c) 2025 devnen

The reference-audio upload **sequence** was transplanted in shape: sanitise the
filename, enforce an extension allowlist, write, validate the duration, unlink
on failure, return the refreshed list with per-file errors. Their
`server.py:670-753` is the original. Worth reading before changing ours.

### `resemble-ai/chatterbox` — MIT — Copyright (c) 2025 Resemble AI

Their Gradio demo supplied the **parameter table**, not code: exaggeration,
`cfg_weight`, temperature, `min_p`, `top_p`, `repetition_penalty` and their
ranges, plus the advice to set `cfg_weight` to 0 for cross-language transfer.

Note that our `services/tts-long` clamps exaggeration and `cfg_weight` to
`0.0–1.0`, so the demo's `0.25–2.0` exaggeration range is **rejected by our own
backend**. The ranges had to be reconciled, not copied.

### `speaches-ai/speaches` — MIT — Copyright (c) 2024 Fedir Zadniprovskyi

The closest prior art to this whole stack, and the reason `ai-voice` is scoped
the way it is: it already does OpenAI-compatible STT and TTS over
faster-whisper and Kokoro. What it does *not* do — Parakeet, post-decode
glossary repair, a long-form job queue — is what justifies this repository
existing at all.

### `jamiepine/voicebox` — MIT — Copyright (c) 2026 Voicebox Contributors

A local-first voice studio: seven TTS engines, cloning, global dictation. It
hosts its own models and its OpenAI-compatible API is planned rather than
shipped, so it is an alternative to this stack rather than a client of it — but
its UX is worth studying, and it is MIT, so its code may be copied into this
repository provided the notice above travels with it.

Its `RESPONSIBLE_USE.md` is a policy document, not an additional licence
condition. It does not restrict the MIT grant. It is still the right position
on cloning a voice you do not own, and this project offers voice cloning.

---

## If you copy code in

MIT and BSD-2-Clause are compatible: both are permissive and both ask only that
the notice be preserved. So copying is allowed — silently dropping the notice
is not.

1. Put the origin's copyright line and permission notice at the top of the file
   holding the copied code, or in a clearly-labelled block around it.
2. Move that project from **Learned from** to **Copied** in this file, naming
   the file and the upstream path.
3. Say in the commit message what was taken and from where.
