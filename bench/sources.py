"""Where each locale's audio comes from.

Several sources per locale on purpose. A single corpus measures one recording
condition and one speaking style, and a model can be excellent on read speech
while falling apart on spontaneous speech at the same WER budget. Reading one
number off one dataset hides exactly the differences worth knowing about.

Sources are tagged by STYLE (read / spontaneous / crowd) and are not
comparable across tags. Compare a source against its own history across runs.
"""

from __future__ import annotations

# style: read = studio or audiobook, scripted
#        crowd = volunteers on consumer microphones, scripted
#        spontaneous = unscripted speech, disfluent, the closest to dictation
SOURCES: dict[str, list[dict]] = {
    "en-US": [
        {"id": "fleurs-en", "dataset": "google/fleurs", "config": "en_us",
         "split": "test", "text_key": "transcription", "style": "read"},
        {"id": "librispeech-clean", "dataset": "openslr/librispeech_asr",
         "config": "clean", "split": "test", "text_key": "text", "style": "read"},
        {"id": "cv-en-us", "dataset": "mozilla-foundation/common_voice_17_0",
         "config": "en", "split": "test", "text_key": "sentence", "style": "crowd",
         "accent": "United States English"},
        # Accented, real meetings, dense domain jargon. The closest public
        # proxy for the failure this project exists to fix.
        {"id": "earnings22", "dataset": "distil-whisper/earnings22",
         "config": "chunked", "split": "test", "text_key": "transcription",
         "style": "spontaneous"},
    ],
    "en-UK": [
        # FLEURS has no en_uk config — no British studio-read source exists in
        # it, so British English is crowd and spontaneous only. That is a real
        # asymmetry, not an oversight: a WER gap against en-US is partly a
        # recording-condition gap.
        {"id": "cv-en-gb", "dataset": "mozilla-foundation/common_voice_17_0",
         "config": "en", "split": "test", "text_key": "sentence", "style": "crowd",
         "accent": "England English"},
        {"id": "edacc", "dataset": "edinburghcstr/edacc", "config": "default",
         "split": "test", "text_key": "text", "style": "spontaneous"},
        # Common Voice is gated, so British read speech comes from VCTK
        # instead — Edinburgh's corpus, filtered on its own accent field.
        # Without it en-UK would be spontaneous-only and not comparable to
        # the read sources the other locales have.
        {"id": "vctk-gb", "dataset": "sanchit-gandhi/vctk", "config": "default",
         "split": "train", "text_key": "text", "style": "read",
         "accent": "English"},
    ],
    "pt-BR": [
        {"id": "fleurs-pt", "dataset": "google/fleurs", "config": "pt_br",
         "split": "test", "text_key": "transcription", "style": "read"},
        {"id": "mls-pt", "dataset": "facebook/multilingual_librispeech",
         "config": "portuguese", "split": "test", "text_key": "transcript",
         "style": "read"},
        # Unscripted interviews and conversation, Brazilian varieties, with
        # per-clip annotations for hesitation, filled pauses and background
        # noise. The closest public corpus to real dictation and the one whose
        # result should carry the most weight.
        {"id": "coraa", "dataset": "Racoci/CORAA-v1.1", "config": "default",
         "split": "test", "text_key": "text", "style": "spontaneous"},
        # Oral-history interviews, elderly speakers, low formality. Harder
        # than CORAA and a good check that the pipeline degrades gracefully
        # rather than hallucinating over disfluency.
        {"id": "coraa-mupe", "dataset": "nilc-nlp/CORAA-MUPE-ASR",
         "config": "default", "split": "test", "text_key": "normalized_text",
         "style": "spontaneous"},
        # TEDx talks: prepared but delivered live, so fluent yet unscripted.
        # Sits between read and conversational speech.
        {"id": "mtedx-pt", "dataset": "dominguesm/mTEDx-ptbr", "config": "default",
         "split": "test", "text_key": "transcription", "style": "spontaneous"},
    ],
}

# Suites that need no dataset, measuring failures WER cannot see.
SUITES = {
    # Whisper's signature failure is inventing fluent text from nothing, and
    # it does so with HIGH confidence, so avg_logprob will not flag it. The
    # only correct output is an empty string.
    "silence": "expect empty output; any text is a hallucination",
    # No public corpus contains Catallaxy or Theoria, so the measurement that
    # matters most cannot come from a dataset.
    "jargon": "your own recordings with hand-corrected references",
}
