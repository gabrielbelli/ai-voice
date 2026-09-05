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

## Why it is off by default — and why the original reason was wrong

This ADR shipped saying the cost of an unused term was **12% WER on Parakeet**,
citing ADR 0002, and adding: *"That measurement was taken through decoder
biasing, so it is a measurement of this feature."* It has since been measured
directly, and **that sentence was false**.

The 12% comes from `stt-stack/bench/runs/parakeet-vocab.json` against
`parakeet-plain.json` — 25 conditions, mean WER 0.1443 → 0.1620. Both runs
report a realtime factor of **63x**, which is `bench.py --engine parakeet`:
`fluidaudiocli --custom-vocab`, FluidAudio's CoreML implementation on the Neural
Engine. It is a measurement of *a* decoder biasing, on a different
implementation and a different runtime from the ONNX-on-CPU path this service
deploys. Quoting it as the cost of `app/boosting.py` was borrowing a number
across a boundary it does not cross.

The **Whisper** half of that citation does not have this problem and is not
withdrawn: 28% is `baseline` 0.3208 against `whisper-nohotwords-orko` 0.2499,
both at rtf 0.8–0.9, which is CTranslate2 `hotwords` — the code path this
service still uses on Whisper. Absent vocabulary really does cost 28% there.
Two engines, two biasing implementations, two very different answers, and that
is the substantive finding: **the cost of an unused term is a property of the
implementation, not of the idea.**

One more correction while in here: both figures come from a **25-cell**
(source × condition) comparison in those run files. "Across 250 conditions",
repeated in ADR 0002, the README and both glossary headers, is not supported by
the surviving data and should be read as 25 until someone produces the run that
justifies it.

**Measured directly** (`bench/boost_bench.py`, 145 clips, 942 s, four corpora,
95% paired bootstrap over clips):

| the list sent to the decoder | change vs plain | 95% CI |
|---|---|---|
| the shipped `tech`+`dictation` profile, audio containing none of it | **byte-identical**, 0 false fires | — |
| 200 unrelated phrases (`STT_BOOST_MAX_PHRASES` exactly) | +0.4% | [−0.8, +1.7] |
| the words that ARE in the audio | −5.2% | [−9.2, −1.3] |
| those same words, padded to 200 phrases with another language's | −5.2% | [−9.5, −1.4] |

The cost of irrelevant vocabulary here is **zero**, not 12%, and the mechanism
is the one the design already relied on: at `START_WEIGHT = 0` a phrase must be
entered on acoustics, and a word that is not in the audio is never entered. The
last two rows being the same number to four decimal places is the whole point —
padding a clip's real terms out to the 200-phrase ceiling is free.

**So off-by-default survives, for a better reason.** Not "it costs 12%" but
**"it buys fourteen words in 2,378, so there is no case for paying anything for
it by default, and only the caller knows whether they are about to say any of
the words."** The −5.2% is real — its interval excludes zero — and it is also
term recall 0.908 → 0.922, about a sixth of what the model already missed. That
is worth having when you ask for it and is not worth a global switch.

Two defaults are now vindicated by measurement rather than by argument:

- **`STT_BOOST_WEIGHT = 3.0` is the optimum.** The curve peaks there and decays
  either side (1.0 → −2.6%, 2.0 → −4.1%, 3.0 → −5.2%, 4.5 → −3.7%, 6.0 →
  −2.2%), and 3.0 is the largest weight whose interval still excludes zero.
- **`STT_BOOST_START_WEIGHT = 0.0` was the setting that mattered.** Raising it
  to 1.5 does lift recall, to 0.944 — and on the realistic list shape it costs
  **+81% WER** [+49, +122]. At 3.0 with the shipped profile, CORAA goes 0.2195 →
  0.8005. The recall win is an oracle's, available only to a caller who already
  knows the transcript.

### What is still not measured, and it is the case the feature was built for

**There is no jargon corpus on this machine.** The terms-present axis is a
proxy — rare words lifted from public corpora, not `Catallaxy` and `Theoria` in
the voice that says them. And **CORAA moved by nothing at any weight**: 3.4 s
clips, 10 words, 14 of 40 with no word distinctive enough to boost. The corpus
closest to real dictation returned a null result for want of testable terms,
which is a limit of the method, not a finding about the feature.

Recording a jargon suite is the measurement that would actually settle this.
Until it exists, the honest summary is: **free, mildly useful, and unproven on
the audio it exists for.**

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
  `STT_BOOST_START_WEIGHT > 0`, and that setting is now **measured harmful**:
  +81% WER on a realistic list at 1.5, CORAA 0.2195 → 0.8005 at 3.0. What was
  "no measurement justifies it yet" is now "the measurement is against it". If
  it is ever raised in anger, rollback is still the thing to build first — but
  the case for raising it has got weaker, not stronger.
