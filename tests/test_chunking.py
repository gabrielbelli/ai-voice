"""The chunker, and the truncation it exists to stop.

Measured on the deployed instance on 2026-09-01: 1690 characters of prose came
back as `audio_seconds: 40.0`, because generate() stops at 1000 speech tokens
and 1000 tokens at 25 Hz is forty seconds. Nothing reported it. Every
assertion here is about a piece of text that would have been silently cut.
"""

from __future__ import annotations

from app.chunking import MAX_CHARS, chunk_text, speech_seconds


def test_nothing_is_longer_than_one_generate_call_can_finish():
    """The whole point: no chunk may exceed the model's 40-second ceiling."""
    text = " ".join(f"Sentence number {n} explains one more step of the setup "
                    f"in a plain and unhurried way." for n in range(60))
    chunks = chunk_text(text)
    assert chunks
    assert max(len(c) for c in chunks) <= MAX_CHARS
    # 40 s is the hard limit; the chunk ceiling has to sit under it with room
    # for a passage read more slowly than the average.
    assert speech_seconds(MAX_CHARS) < 40


def test_nothing_is_lost_or_invented():
    """Splitting must be lossless: same words, same order."""
    text = ("Open your configuration file. Find the section marked network! "
            "Is that clear? Then save it.")
    assert " ".join(chunk_text(text)).split() == text.split()


def test_a_sentence_without_a_full_stop_is_still_split():
    """A caller who writes 400 characters of comma-spliced prose gets audio.

    Before, that was one generate() call and therefore truncated. Breaking at
    a comma is audible; being cut off mid-word is worse.
    """
    text = ", ".join(["one more clause that keeps going"] * 30)
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert max(len(c) for c in chunks) <= MAX_CHARS


def test_a_single_unbroken_word_is_cut_rather_than_dropped():
    """The one case with no good answer, and it must not be an exception."""
    chunks = chunk_text("x" * (MAX_CHARS * 3))
    assert "".join(chunks) == "x" * (MAX_CHARS * 3)
    assert max(len(c) for c in chunks) <= MAX_CHARS


def test_short_sentences_are_merged():
    """One generate() per five-word sentence restarts the prosody each time."""
    assert chunk_text("One. Two. Three. Four.") == ["One. Two. Three. Four."]


def test_an_initial_does_not_end_a_sentence():
    """"J. Random Hacker" split into two chunks is an audible break in a name."""
    assert chunk_text("Ask J. Random Hacker about it.") == [
        "Ask J. Random Hacker about it."]


def test_empty_input_produces_no_chunks():
    """generate() substitutes its own sentence for empty text. Never send it."""
    assert chunk_text("   \n  ") == []


def test_the_estimate_counts_characters_not_words():
    """A 5000-character string with no spaces was estimated at two seconds."""
    assert speech_seconds(5000) == speech_seconds(len("x" * 5000))
    assert speech_seconds(5000) > 100
