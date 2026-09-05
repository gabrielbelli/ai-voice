# ADR 0002 — Glossaries are named profiles, chosen per request, and none ships personal terms

**Status:** accepted
**Date:** 2026-09-03

## Decision

A glossary becomes a **named profile**, selected per request. The image ships
no personal vocabulary. One deployment-owned profile carries the operator's own
terms and is supplied at run time, not baked in.

## The problem, stated as it actually is

`services/stt/glossary.txt` is in the repository, is copied into the image, and
is loaded once at startup from `STT_GLOSSARY`. Its contents today:

```
ghost paper = Ghost Pepper
catalaxy    = Catallaxy
theory dashboard = Theoria dashboard
entropic    = Anthropic
comet       = commit
open sauce  = open source
```

Two different things are wrong with that list, and only one of them is obvious.

**It is one person's vocabulary inside a public artefact.** `ai-voice-stt` is
published to a public registry under a BSD licence. Anyone who pulls it gets
`catalaxy = Catallaxy` applied to their audio — a rewrite naming a project they
have never heard of, that they cannot discover without reading the image, and
that they did not ask for.

**And it is not free to carry.** Measured across 25 cells: a glossary
whose terms do not occur in the audio raises WER by **28% on Whisper and 28%
on Whisper**. Irrelevant terms are not inert — they actively cost accuracy. So
a single always-on list is the worst shape available: it is simultaneously too
personal for other people and too broad for any one recording.

That measurement is the whole argument for profiles. If an unused term were
free, one big list would be fine.

## The shape

Profiles are separate, small, and opted into per request:

| profile | contents | who owns it |
|---|---|---|
| *(none)* | empty — the default | — |
| `dictation` | mishearings of ordinary speech that any dictating user hits: `ldr = TLDR`, `dts = STT`, `tex to speak = text-to-speech` | the repository |
| `tech` | **general** technical vocabulary only: kubernetes, nginx, PostgreSQL, ONNX. Vendor and tool names anyone in the field would say | the repository |
| `<operator's own>` | project names, colleagues, internal systems | **the deployment** |

**`tech` does not contain anyone's project names.** That is the point of the
split, not a stylistic preference: a term in `tech` is paid for by every request
that selects `tech`, so a name only one person ever says makes the profile worse
for everybody else who uses it.

The operator's profile is supplied at run time — a mounted directory or an
environment variable naming a file — and is never committed to this repository
or copied into the image.

## Reaching it from the API

Per ADR 0001, one spec field and one extension:

```
prompt:   "Theoria, Catallaxy"    # spec field: this request's one-off terms
glossary: "tech,dictation"        # extension: named profiles, via extra_body
```

`prompt` is defined by the specification as text guiding the model, which is
exactly this. It needs no extension and is what a one-off should use. The named
profile is the extension, allowlisted beside `keywords` and `languages`.

Absent both, behaviour is the specification's: no glossary, no biasing.

## Consequences

- **`services/stt/glossary.txt` stops shipping personal terms.** What remains
  in the repository is `dictation` and `tech`; the current file's project names
  move to a deployment-supplied profile on orko.
- Profiles must be loadable **without a restart**. Today `pipeline.py:108`
  reads the file once at startup, so changing a term needs a new container.
  Per-request selection is meaningless if the set is frozen at boot.
- A request naming an unknown profile is a 400 that names it. Silently
  ignoring it would mean a caller believing their vocabulary was applied when
  it was not — the same silence ADR 0001 exists to prevent.
- Selecting many profiles at once should be discouraged in the documentation
  for the measured reason above, not merely on grounds of tidiness.
