# ADR 0005 — Parakeet is biased at decode time, by subclassing onnx-asr's greedy loop, and it is off unless a request asks

**Status:** accepted
**Date:** 2026-09-04

## Decision

Parakeet's TDT decoder now takes a vocabulary. A request's terms are compiled
into a boosting automaton and shallow-fused into onnx-asr's greedy transducer
loop, one frame before the argmax.

| | |
|---|---|
| how | a subclass of `NemoConformerTdt` overriding the private `_decoding`, constructed through onnx-asr's own resolver |
| when | only when the request sends `boost=true` (or the deployment sets `STT_BOOST=1`) |
| off switch | `STT_HOTWORDS=0`, which now covers both engines |
| new module | `services/stt/app/boosting.py` |

`Parakeet.accepts_vocabulary` flips from `false` to `true`, which changes what
`/health` reports on the default deployment.

## The claim this overturns, because it is the point of the ADR

Every docstring in this service said Parakeet's decoder took no vocabulary.
`app/glossary.py` offered post-decode repair as the consolation prize for it.
`app/openai_api.py` refused `prompt` and `keywords[]` with a 400 citing it. The
README told operators to deploy Whisper if they needed decode-time vocabulary.

The sentence behind all of that was **"onnx-asr exposes no biasing argument"**,
and it is true. The conclusion drawn from it — that the decoder could not be
biased — was not. onnx-asr's transducer decoding loop is plain Python with the
joint's per-frame logits in a local variable:

```python
logits, step, state = self._decode(tokens, prev_state, encodings[t])
token = logits.argmax()
```

What was missing was a parameter, not a capability. The distance between those
two was about forty lines, and the belief that they were the same thing cost
this project a feature and cost a specification field a 4xx on the engine it
actually deploys.

**The general lesson, which is why this is an ADR and not a commit message: a
refusal has to name a mechanism that genuinely does not exist.** `language` and
`temperature` are refused on Parakeet because a TDT decoder has no language
conditioning and no sampling temperature — facts about the model. "The library
we use exposes no argument for it" is a fact about the library, and it is not
grounds for telling a caller their request is impossible.

## Why a subclass rather than a patch or a fork

The seam is a private method of a pinned dependency, so all three options are
uncomfortable. The deciding property is **what happens when someone bumps the
pin**, because upstream's `_decoding` reads its options with `kwargs.get()` and
ignores every key it does not recognise:

| approach | on a version bump |
|---|---|
| monkey-patch the method | patches a method that may not exist; silent no-op or AttributeError depending on how it is written |
| vendor a fork of onnx-asr | never breaks, never gets upstream's fixes either, and the copy rots invisibly |
| **subclass, verify at startup** | the model still loads, biasing is withdrawn, `/health` says so, and `boost` is refused by name with the version in the message |

The failure to design against is not a crash. It is `boost=[...]` flowing down
a stock code path, being discarded with no error, and every transcript
afterwards looking plausible and being slightly worse. `boosting.verify_seam`
is the entire defence: an allowlist of onnx-asr versions, a check that the
resolver built our class, a check of the vocabulary and blank index, a read-back
of the joint's output width, and a probe that sends a sentinel keyword through
the **public** adapter and asserts it arrived in our `_decoding`.

A pin in `requirements.txt` does not detect that failure. It only delays it
until someone changes the pin, which is exactly when nobody is looking for it.

With no boost list the override calls `super()._decoding`, so an unboosted
decode is upstream's code running upstream's arithmetic — byte-identical by
construction rather than by resemblance. Verified anyway on sixteen real corpus
clips, across text, tokens, timestamps and logprobs.

## Why it is off by default

ADR 0002 removed an always-on glossary because a glossary whose terms do **not**
occur in the audio raised WER by **12% on Parakeet and 28% on Whisper** across
250 conditions. That measurement was taken through decoder biasing, so it is a
measurement of this feature, and turning this feature on for every request would
re-create the exact defect ADR 0002 was written to remove — this time inside the
decode, where post-decode repair cannot undo it.

So biasing is a bet on knowing what is in the audio, and the caller who knows is
the one who places it. A deployment that wants it everywhere sets `STT_BOOST=1`
and pays that cost knowingly, the same shape as `STT_GLOSSARY_DEFAULT`.

Every default in `boosting.py` was chosen against the terms-**absent** axis
rather than the terms-present one. `STT_BOOST_START_WEIGHT` is 0 for that
reason: raising it recovered more terms on a probe *and* inserted a spurious
word and capitalised two innocent ones.

**No WER number for this feature exists yet.** A 6.6 s synthetic clip
establishes mechanism and scale, not accuracy. `bench/bench.py` on both axes is
what has to decide the defaults before any deployment turns this on, and the
terms-absent run is the one that decides.

## Consequences

- `/health` reports `accepts_vocabulary: true` on a Parakeet deployment. Anything
  that hides a vocabulary field on that flag will start showing it.
- A profile's hotword-only lines do something on the default engine for the
  first time. `glossaries/dictation.txt` documented that cost in its own header
  and no longer needs to.
- A response carries `x-boost-applied` naming the phrases that reached the
  decoder, because a term can be dropped by two ceilings that are not 400s.
- The weights are calibrated against this model's raw logit scale at int8.
  Changing `STT_MODEL_ID` or `STT_QUANTISATION` invalidates them, and
  `verify_seam` reads the joint's output width back so a mis-shaped model is
  caught rather than mis-sliced.
- Snapshot rollback — rewinding the decoder when a partial match dies — is
  designed and deliberately **not** implemented. It only earns its keep at
  `STT_BOOST_START_WEIGHT > 0`, its failure mode is a subtly wrong transcript
  rather than a crash, and no measurement justifies that setting yet. If it is
  ever raised in anger, that is the thing to build first.
