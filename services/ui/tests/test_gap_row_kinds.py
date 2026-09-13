"""What a job row says now that three engines write to it.

Static, for the reason test_jobs.py and test_playback.py are: every claim here
is a property of the bytes in ui.html -- which word a count is printed in, how
many decimals survive, which press opens a dialog -- and starting a browser to
find that out would add a dependency to a service whose whole claim is that it
has none.

THE ROW WAS WRITTEN WHEN EVERY JOB WAS A CHATTERBOX CLONE. One kind, one
engine, one shape: text cut into a dozen segments, forty minutes of compute,
megabytes of wav on the disk at the end. It now draws three kinds. Kokoro
answers a sentence in a fraction of a second and keeps no file; Parakeet reads
a whole clip in one window and its artefact is the text. Every test in this
file is named after a place where the old shape's assumptions print something
false about one of the two new ones.
"""

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "ui.html"
HTML = PAGE.read_text()


def body_of(name: str, ends: str) -> str:
    """The source of one function, from its declaration to `ends`."""
    start = HTML.index(name)
    return HTML[start:HTML.index(ends, start)]


def code(source: str) -> str:
    """The same slice with its comments removed.

    Every comment in this page names the failure it prevents, so it quotes the
    very string a negative assertion here forbids -- "read \"1 chunks\"", "was
    toFixed(0)". Matching the raw text would assert against the subject's own
    documentation and pass or fail on prose. The sibling files strip for the
    same reason.
    """
    return re.sub(r"/\*.*?\*/|<!--.*?-->|//[^\n]*", "", source, flags=re.S)


ROW = code(HTML[HTML.index('$("joblist").innerHTML = list.map'):
                HTML.index('}).join("")')])
RENDER = code(body_of("function renderJobs()", "let jobUrl = null;"))


# ------------------------------------------------- the pieces a run has --


def test_a_run_made_of_one_piece_is_not_reported_as_1_chunks():
    """THE COMMONEST ROW ON THE TAB, GETTING ITS PLURAL WRONG.

    The header hint was `job.chunks + " chunks"`, which is true of a clone:
    the text is cut into segments and there are a dozen of them. Parakeet
    windows the AUDIO, and every clip under about thirty seconds is one
    window, so the transcription row -- the one there are most of -- read
    "1 chunks".
    """
    assert '" chunks"' not in ROW and "' chunks'" not in ROW, \
        "the row still prints a bare, unpluralised 'chunks'"
    # The count is pluralised where it is built, not spelled twice.
    assert re.search(r'=== 1 \? "" : "s"', ROW), \
        "nothing in the row pluralises the piece count"


def test_the_pieces_are_named_in_the_word_the_engine_that_made_them_uses():
    """A CLONE'S CHUNKS AND A TRANSCRIPTION'S ARE NOT THE SAME THING.

    tts-long splits the TEXT and calls the parts segments -- the live bar two
    lines below this hint already says "n of 14 segments" about this very
    field. stt splits the AUDIO into windows and reports how many it needed.
    One word over both makes the row say "chunks" in the header and
    "segments" about the same number underneath, and tells a reader of a
    transcription that their recording was cut into pieces of text.
    """
    assert 'kind === "transcribe" ? "window"' in ROW, \
        "a transcription's windows are still called something else"
    assert '"segment"' in ROW, "the clone's own word for this field is not used"
    # And the bar underneath still counts the same field the same way.
    assert "madeCount} of ${job.chunks} segments" in ROW


def test_the_queue_position_never_leads_with_a_separator():
    """The separator was glued to the front of the queue position, so a row
    with a place in the queue and no piece count yet -- which is exactly what
    the OpenAI route's 202 hands back -- opened with a bare "· 2 ahead".
    """
    assert '" · " + job.queued_ahead' not in ROW, \
        "the separator is still concatenated onto the queue position"
    assert 'join(" · ")' in ROW, "the header hint is still built by concatenation"


# ------------------------------------------------ the numbers under a row --


def test_a_measurement_under_a_second_is_not_rounded_away_to_zero():
    """A ROUNDING WRITTEN FOR A JOB MEASURED IN MINUTES.

    Every number on the row was toFixed(0), which is right for a clone: forty
    minutes of compute for five of audio. Parakeet transcribes a five second
    clip in about 0.29 s and Kokoro answers a sentence in about the same, so
    the row printed "0 s of compute" -- and then printed a realtime factor
    beside it that could only have come from dividing by the number it had
    just rounded to nothing.
    """
    for field in ("audio_seconds", "compute_seconds", "speech_seconds"):
        assert f"job.{field}.toFixed(0)" not in ROW, \
            f"job.{field} is still rounded to whole seconds on the row"
    said = code(body_of("function secondsSaid(seconds)", "\n\nconst JOBTEXT"))
    assert "toPrecision(2)" in said, \
        "the small end has no significant figures, so it can still print 0 s"
    assert "Math.round" in said, "a forty minute job still carries a decimal"
    # Every seconds figure on the row goes through it, so the two cannot drift.
    assert ROW.count("secondsSaid(") >= 3


def test_the_second_number_a_transcription_measures_is_not_dropped():
    """`speech_seconds` EXISTS ON ONE KIND AND WAS RENDERED BY NOBODY.

    stt reports the clip's length and the speech inside it as two different
    numbers, on purpose: `audio_seconds` means the same thing on all three
    kinds -- how much audio the run is about -- and the VAD's answer is the
    one that says why a twelve second clip was six seconds of work. The row
    read the first and threw the second away, so the only place in the stack
    that could show it showed nothing.
    """
    assert 'typeof job.speech_seconds === "number"' in ROW, \
        "the row never looks at speech_seconds"
    assert "of speech" in ROW
    # Guarded like its neighbours: one record carrying a field without the
    # ones under it must not take list.map, and with it the whole tab, down.
    for hit in re.finditer(r"job\.speech_seconds\b(?!\s*===)", ROW):
        assert 'typeof job.speech_seconds === "number"' in ROW[:hit.start()], \
            "speech_seconds is used before it is checked"


def test_a_one_character_transcript_is_not_1_characters():
    """The same missing plural as the piece count, on the field that is the
    whole point of a transcription row."""
    assert "} characters" not in ROW, "the character count is not pluralised"
    assert re.search(r'character\$\{job\.chars === 1 \? "" : "s"\}', ROW), \
        "nothing pluralises the character count"


# ------------------------------------------- what a press can destroy now --


def test_deleting_the_only_copy_of_a_transcript_asks_first():
    """THE "NOTHING LEFT TO LOSE" RULE, MET BY A KIND THAT HAS EVERYTHING.

    forgetJob asks before it deletes only while there is audio on the server,
    and a row with none was judged not worth a dialog. That was written when
    a record was a few hundred bytes of index pointing at a wav file. A
    transcription has no audio and never had -- the clip belonged to whoever
    supplied it -- so its record is not the index, it IS the artefact, and one
    press removed it from the server and therefore from every device with no
    dialog, no undo and nothing anywhere that could make it again.
    """
    body = code(body_of("async function forgetJob(id)", "\nasync function stopJob"))
    assert 'job.kind || "clone"' in body or 'kind === "transcribe"' in body, \
        "forgetJob does not know what kind of record it is deleting"
    assert body.count("confirm(") == 2, \
        "there is still one dialog, so one of the two kinds of loss is silent"
    assert "transcript" in body, "the dialog does not name what is being lost"
    # The audio dialog is unchanged and still comes first: a row WITH audio
    # must not be asked twice, and must not be told its wav was a transcript.
    assert "hasAudio(job)" in body
    first = body.index("confirm(")
    assert "hasAudio(job)" in body[:first]


def test_a_transcription_row_says_which_button_takes_the_text_away():
    """"Delete the record" beside "Copy the transcript" reads as tidying a
    list. It is the same press either way; the label is the only warning the
    reader gets before the dialog."""
    assert 'kind === "transcribe" ? "Delete the transcript"' in ROW, \
        "the destructive button on a transcription row does not name the text"
    # The other two labels are untouched: a clone with a wav still says the
    # record goes TOO, which is the half that was already right.
    assert '"Delete the record too"' in ROW and '"Delete the record"' in ROW
