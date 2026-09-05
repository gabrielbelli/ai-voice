# Benchmark

Measures WER as a function of **recording condition**, not as a single number
on studio audio.

```bash
cd bench
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python bench.py fetch                      # cache samples once
./.venv/bin/python bench.py run --url http://orko:8000 --label whisper-only
./.venv/bin/python bench.py run --url http://orko:8000 --label whisper+parakeet
./.venv/bin/python bench.py report
```

The venv is deliberate: `datasets` pulls a large dependency tree that has no
business in a system interpreter, and pinning it here keeps a benchmark run
reproducible independently of whatever else the machine has installed.
`.venv/`, `cache/` and `runs/` are all git-ignored.

Fetch is separate from run so every configuration sees byte-identical audio.
A benchmark whose inputs move cannot tell you whether a change helped.

## Locales and sources

Several sources per locale, tagged by style. **Sources are not comparable
across styles** — a model can be excellent on read speech and fall apart on
spontaneous speech at the same WER. Compare a source against its own history.

| Locale | Sources | Styles |
|---|---|---|
| en-US | FLEURS, LibriSpeech clean, Common Voice (US), Earnings-22 | read, crowd, spontaneous |
| en-UK | Common Voice (England English), EdAcc | crowd, spontaneous |
| pt-BR | FLEURS, MLS Portuguese, CORAA, Common Voice pt | read, spontaneous, crowd |

Two caveats that change how you read the numbers:

- **FLEURS has no `en_uk`.** British English is crowd and spontaneous only, so
  a gap against en-US is partly a recording-condition gap, not only accent.
- **Common Voice `pt` is majority European Portuguese.** It is a contrast
  source, not a pt-BR measurement. Weight CORAA highest for pt-BR — it is
  unscripted Brazilian speech, the closest thing here to real dictation.

## Conditions

Every condition runs over the same clips. `clean` is the control.

| Group | Conditions |
|---|---|
| Noise | `pink-20db` `pink-10db` `pink-5db` `babble-10db` `babble-5db` |
| Microphone | `mic-good` `mic-cheap` `mic-phoneline` `mic-closed-lid` |
| Room | `room-reverb` `hot-gain` |
| Codec | `opus-32k` `opus-16k` `mp3-32k` `amr-nb` |

Babble noise is built from the corpus itself, so nothing extra is downloaded.
It is the hardest noise for ASR because it *is* speech — Whisper will happily
transcribe the wrong voice.

Codec conditions need `ffmpeg` and are skipped with a message if it is absent.

## Why conditions rather than one number

The single largest accuracy loss in this project came from a laptop microphone
inside a closed lid. No model recovers audio that muffled, and no clean-speech
benchmark predicts it.

**Where the curve bends is the useful part.** If WER is flat down to 10 dB SNR
and collapses at 5, buy a better microphone rather than a bigger model. If
`mic-cheap` alone doubles WER, the hardware is the bottleneck and no amount of
model tuning will move it.

`mic-closed-lid` reproduces that specific failure: band-limited to 150–3000 Hz
with reverb.

## Suites with no dataset

- **silence** — expects empty output. Whisper's signature failure is inventing
  fluent text from nothing, and it does so at *high* confidence, so
  `avg_logprob` cannot flag it. This is the one failure a second model catches
  and confidence scoring does not.
- **jargon** — your own recordings with hand-corrected references. No public
  corpus contains Catallaxy or Theoria, so the measurement that matters most
  cannot come from a dataset.

## Normalisation

Both sides are lowercased, stripped of punctuation and whitespace-collapsed
before scoring. Whisper emits cased, punctuated text and most references are
neither; skipping this measures formatting rather than recognition.

**Accents are kept for pt-BR.** Stripping them would hide a real class of
error.

## Decode-time boosting — `boost_bench.py`

A second harness, because boosting is not a configuration. It is a per-request
argument whose effect depends on a term list that has to be built **per clip
from that clip's own reference**, and it has to sweep a weight — twenty-odd
decodes of the same audio, which is the shape `bench.py`'s HTTP client is worst
at. So this drives `app.asr.Parakeet` in process and imports `bench.normalise`,
so both harnesses score identically and the numbers stay comparable.

```bash
./.venv/bin/pip install -r ../requirements.txt    # it drives the model in process
./.venv/bin/python bench.py fetch                 # populates cache/ (once)
./.venv/bin/python boost_bench.py run --cache cache --condition clean
./.venv/bin/python boost_bench.py report          # the sweep, one line per config
./.venv/bin/python boost_bench.py ci              # paired bootstrap on the deltas
```

It starts no server — it drives the model in process. `--cache` takes any
`bench.py fetch` cache, which is why it is a flag rather than a constant: the
recorded results were produced against the already-fetched cache in the
predecessor repository, `../../../../stt-stack/bench/cache`, so that the
clips are byte-identical to the ones the 25-cell Parakeet-vs-Whisper runs used.
Re-fetching here would give different clips and a table that cannot be compared
with those.

Model: Parakeet TDT 0.6B v3, ONNX **int8, CPU**, this laptop — 145 clips, 942 s
over `pt-BR/coraa` (40), `pt-BR/fleurs-pt` (25), `en-US/librispeech-clean` (40)
and `en-US/earnings22` (40).

### The five term lists, and why five

| list | what is in it | which axis |
|---|---|---|
| `present` | the clip's OWN distinctive words, from its reference | does it help |
| `mixed` | those, padded to 200 with the other locale's words | **the realistic shape** |
| `absent-small` | 12 project names, in none of the audio | what it costs |
| `absent-large` | 200 foreign words — `STT_BOOST_MAX_PHRASES` exactly | the ceiling's cost |
| `profile` | `glossaries/tech.txt` + `dictation.txt` through `profiles.Registry.select` | **what an operator actually sends** |

`mixed` matters most: it is the shape a saved glossary profile has — the right
terms are in the list, and so are ninety wrong ones. A feature that wins on
`present` and loses on `mixed` helps an oracle, not a deployment.

### Result: it is close to free, and it helps a little

Pooled over all 145 clips, relative WER change against no boosting, with a 95%
paired bootstrap over clips (`ci`):

| config | WER | change | 95% CI | errors |
|---|---|---|---|---|
| plain | 0.1135 | — | — | — |
| `present@w1` | 0.1106 | −2.6% | [−5.5, −0.3] | −7 |
| `present@w2` | 0.1089 | −4.1% | [−7.7, −1.1] | −11 |
| **`present@w3`** (shipped) | 0.1077 | **−5.2%** | **[−9.2, −1.3]** | −14 |
| `present@w4.5` | 0.1093 | −3.7% | [−8.0, +0.3] | −10 |
| `present@w6` | 0.1110 | −2.2% | [−6.6, +2.1] | −6 |
| `mixed@w3` | 0.1077 | −5.2% | [−9.5, −1.4] | −14 |
| `absent-small@w3` | 0.1135 | +0.0% | [+0.0, +0.0] | 0 |
| `absent-large@w3` | 0.1140 | +0.4% | [−0.8, +1.7] | +1 |

Four things that table says, in order of how much they matter.

**The absent axis costs nothing.** `absent-small` is byte-identical — not
"small", identical. `absent-large` is 200 phrases at the ceiling and moves one
word in 2,378, an interval that contains zero. The **shipped profile**, 79
terms through the real `select()` call, is byte-identical on all four corpora at
weights 1, 2, 3 and 4.5 with zero false fires; one clip in CORAA moves at 6.0.
This is the number that decides, and it is a zero.

**`mixed@w3` equals `present@w3` to four decimal places.** A clip's own terms
padded out to the full 200-phrase ceiling with another language's words score
the same as those terms alone. Irrelevant vocabulary in this decoder is not
merely cheap, it is inert — because at `start_weight = 0` a phrase must be
entered on acoustics, and a word that is not in the audio is never entered.

**The shipped weight is the measured optimum.** The curve peaks at 3.0 and
decays either side, and 3.0 is the largest weight whose interval still excludes
zero. That was chosen before this ran.

**The win is real but small.** −5.2% relative is fourteen words out of 2,378.
Term recall pooled goes 0.9079 → 0.9223, so it recovers about a sixth of the
in-audio terms plain missed. Anyone quoting "−5%" without "fourteen words" is
overselling it.

### `STT_BOOST_START_WEIGHT` is confirmed dangerous

Raising it is the one change that recovers more terms — recall 0.9079 → 0.9439
— and on the realistic list it is a catastrophe:

| config | WER | change | 95% CI |
|---|---|---|---|
| `present@w3+s1.5` (oracle list) | 0.1077 | −5.2% | [−14.9, **+5.5**] |
| `mixed@w3+s1.5` (realistic list) | 0.2056 | **+81.1%** | [+49.4, +122.4] |
| `profile@w3+s3` | — | CORAA 0.2195 → **0.8005** | — |

The oracle list's apparent win no longer excludes zero, and the same setting on
a list with ninety wrong terms in it nearly doubles WER. Shipping `0.0` was
right, and this is the measurement that says so rather than the argument.

### Realtime factor: no measurable cost

A Python trie in the decode loop is exactly the kind of thing that quietly
halves throughput. It does not.

| automaton | rtf | sd |
|---|---|---|
| none (plain) | 17.82 | 1.51 |
| 12 phrases | 17.93 | 1.48 |
| 200 phrases (the ceiling) | 18.01 | 1.36 |

All three are the same number. The spread across repeats is ±8%, so that is the
resolution of this measurement, and the automaton's cost is under it. **This
18x is int8-on-CPU in process on a laptop; it is not the 47–63x this project
quotes for the deployment, and only the comparison between rows transfers.**

### What this does NOT measure, stated plainly

- **There is no jargon corpus on this machine.** The `jargon` suite above — the
  user's own recordings, hand-corrected — does not exist, so the terms-present
  axis is a **proxy**: rare words lifted from public corpora, not `Catallaxy`
  and `Theoria` in the voice that says them. The case this feature was built
  for is still unmeasured.
- **CORAA returned a null result, and it is the corpus that counts most.** Its
  clips average 3.4 s and 10 words, 14 of 40 contain no word distinctive enough
  to be a candidate, and mean 1.25 terms per clip. WER there is unchanged at
  every weight. That is a limit of this method on short spontaneous speech, not
  evidence that boosting is inert on it.
- One condition beyond `clean` (`babble-10db`, a coarser ladder). The curve
  shifts by less than its own interval there; see `results/boost.jsonl`.
