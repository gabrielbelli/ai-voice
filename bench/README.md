# Benchmark

Measures WER as a function of **recording condition**, not as a single number
on studio audio.

```bash
pip install -r requirements.txt
python bench.py fetch                                  # cache samples once
python bench.py run --url http://orko:8000 --label whisper-only
python bench.py run --url http://orko:8000 --label whisper+parakeet
python bench.py report
```

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
