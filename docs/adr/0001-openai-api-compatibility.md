# ADR 0001 — The API is OpenAI's, and extensions use OpenAI's own mechanisms

**Status:** accepted
**Date:** 2026-09-03

## Decision

`/v1` is OpenAI's API, not an API that resembles it. Where the specification
says something, this stack does that thing, including where the specification
is inconvenient.

Anything this stack offers that OpenAI does not is an **extension**, and an
extension must travel by a channel the OpenAI client libraries already provide
— `extra_body`, `extra_query`, `extra_headers` — never by changing the meaning
or the shape of something the specification already defines.

## The test this has to pass

> **A client that knows nothing about our extensions must behave exactly as it
> would against OpenAI.**

Three rules follow from it, and they are the whole of the policy:

1. **Absence is spec behaviour.** Every extension defaults to off. A request
   that sets none of them gets what OpenAI would return.
2. **An extension never changes the response *shape*.** It may change the
   content of a field; it may not add a required key, remove one, or alter a
   type. `verbose_json` stays `verbose_json`.
3. **No extension is ever required.** There is no operation reachable only
   through a non-standard field.

## Why not "close enough"

Because the failures are silent and land on the client's side of the wire.

- **`param` was missing from every error this stack produced.** The schema
  requires all four of `type`, `message`, `param` and `code`, with `param` and
  `code` required-but-NULLABLE — present as JSON `null`, not absent. Three
  services had each noticed and written their own fix; the gateway, the one an
  SDK actually talks to, had none. A client reading `err.param` to find out
  which field it got wrong read `None` whether or not the server knew.
- **404 and 405 under `/v1` escaped the envelope entirely**, returning
  FastAPI's `{"detail": "Not Found"}`. `openai-python` reads no message off
  that shape and reports a bare "unknown error".
- **A non-ASCII API key could never authenticate.** Starlette decodes header
  bytes as latin-1; the code compared UTF-8, so the *correct* key was rejected
  with a message saying it was wrong.

None of these produced a stack trace. Each was found by reading the
specification against the wire, which is the argument for treating the
specification as binding rather than aspirational.

## What this permits, concretely

**Use a spec field for its documented purpose before inventing one.**
`prompt` is defined as text that guides the model, and vocabulary biasing is
what it is used for in practice. Feeding a request's own glossary terms through
`prompt` is more compliant than refusing `prompt` — not less.

**When a genuinely new axis is needed, add a field and allowlist it.** Two
already exist and are the pattern to copy:

| field | status | why it exists |
|---|---|---|
| `keywords[]` | extension | list-shaped vocabulary, reaching the same decoder argument as `prompt` |
| `languages[]` | extension | a code-switching speaker; the spec's `language` is singular |

Both sit in `TRANSCRIPTION_FIELDS` beside the specification's own names, and an
SDK caller reaches them with `extra_body={"keywords": [...]}`.

**Unknown fields stay refused.** `CreateTranscriptionRequest` sets
`additionalProperties: false`, so a field this service does not know is a 400
naming it, not a silent no-op. Leniency here is what turned every unhonoured
field into silence in the first place.

**Say no by name.** Where a backend genuinely cannot honour a spec field, the
answer is an error that names the field and the reason — as Parakeet does today
for `prompt` — never acceptance followed by nothing happening. A field that is
accepted and ignored is indistinguishable, from the client's side, from a field
that worked.

## Consequences

- Every `/v1` change is measured against the published schema, not against what
  seems reasonable. `packages/common`'s conformance suite runs in every
  consumer's CI for this reason.
- Native routes (`/transcribe`, `/speak`, `/voices`, `/jobs`) are NOT bound by
  this ADR. They keep FastAPI's `{"detail": ...}` and its 422, they have
  clients, and reshaping them to tidy up a compatibility layer those clients
  never touch is how a working deployment breaks during a refactor. The
  boundary is the `/v1` prefix and it is load-bearing.
- An extension is a maintenance commitment: it has to keep working when the
  specification moves, and it has to not collide with a name OpenAI later
  takes. Prefer a spec field, then a header, then a body field, in that order.
