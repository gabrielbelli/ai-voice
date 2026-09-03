"""Playback speed and karaoke highlighting: the parts decidable from source.

Static and parser-based, for the same reason test_escaping.py is. What these
assert -- which element gets a rate control, which format is asked for, what
the highlight is drawn with, whether a colour is the only cue -- is a property
of the bytes in ui.html, and running a headless browser to discover it would
add a dependency to a service whose whole claim is that it has none.

The two features share a file because they share a failure: both of them are
about a NUMBER on screen agreeing with the audio. A rate that silently reverts
and a highlight that silently drifts are the same bug wearing different
clothes, and the tests that stop them read the same source.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "ui.html"
HTML = PAGE.read_text()


def body_of(name: str, ends: str) -> str:
    """The source of one function, from its declaration to `ends`."""
    start = HTML.index(name)
    return HTML[start:HTML.index(ends, start)]


def code(source: str) -> str:
    """The same slice with its comments removed.

    Every comment in this file's subject explains the failure it prevents, so
    it names the thing it was written to stop -- `<audio>` inside renderJobs,
    scrollIntoView in follow(). A negative assertion over the raw text matches
    the prose and asserts against its own documentation. test_escaping.py
    strips comments for exactly this reason.
    """
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)


# ------------------------------------------------------- playback speed --


def test_the_two_speed_controls_are_not_both_called_speed():
    """They do different things and one of them is sent to the server.

    `#speed` is Kokoro's synthesis rate: a request field, baked into the
    samples, changing the file that comes off the Download button, and a 400 on
    Chatterbox. The playback rate is none of those -- it is
    HTMLMediaElement.playbackRate and it never leaves the browser. Two controls
    labelled "Speed" on one tab is how someone concludes the download will come
    back faster, so the label carries the distinction.
    """
    assert ">Synthesis speed " in HTML, "the Kokoro slider no longer says which speed it is"
    labels = re.findall(r'<label for="\w*rate">Playback speed</label>', HTML)
    assert len(labels) == 3, (
        f"three players, {len(labels)} controls named the same way")
    assert not re.search(r"<label for=\"speed\">Speed\b", HTML), (
        "the synthesis slider is called plain 'Speed' again")


def test_pitch_is_preserved_or_a_voice_at_1_5x_is_a_chipmunk():
    """The default is true; setting it is how the decision stays written down.

    Resampling instead of time-stretching moves the formants, and formants are
    what intelligibility rides on -- so the one thing the control exists for,
    getting through a recording faster while still following it, is exactly
    what dropping this would destroy. The prefixed spelling is Safari before
    17, where the unprefixed property does nothing at all.
    """
    wire = body_of("function wirePlayer(", "\nwirePlayer(")
    assert "player.preservesPitch = true;" in wire
    assert "player.webkitPreservesPitch = true;" in wire


def test_every_player_on_the_page_gets_a_speed_control():
    """A new <audio> must not quietly ship without one.

    `clippreview` is the exception and is meant to be: it is a 10-30 second
    reference clip being checked for noise and level before it is uploaded, and
    playing that back at 1.5x tells you nothing about whether Chatterbox can
    use it.
    """
    ids = set(re.findall(r'<audio id="([a-z]+)"', HTML))
    wired = set(re.findall(r'wirePlayer\(\$\("[a-z]+"\), \$\("([a-z]+)"\)\)', HTML))
    assert ids - wired == {"clippreview"}, (
        f"these players have no speed control: {sorted(ids - wired - {'clippreview'})}")


def test_one_rate_is_shared_by_every_player_and_remembered():
    """Setting it once per tab per session is how a feature gets called useless."""
    assert 'store.set("rate.playback", value)' in HTML
    assert 'store.get("rate.playback", 1)' in HTML
    setter = body_of("function setPlaybackRate(", "\nfunction wirePlayer")
    assert "for (const entry of players)" in setter, (
        "the other players are not brought into step")


def test_the_rate_is_reapplied_when_a_new_clip_loads():
    """playbackRate is a property of the ELEMENT, not of the media.

    A new src is a new load, and a player that quietly went back to 1x on the
    second clip is the sort of bug that gets lived with rather than reported.
    """
    wire = body_of("function wirePlayer(", "\nwirePlayer(")
    assert 'player.addEventListener("loadedmetadata"' in wire


def test_an_offered_rate_is_one_the_stored_preference_can_be():
    """A value out of the list would leave the select showing nothing.

    localStorage is shared with whatever the last version of this page wrote,
    so the stored number is not necessarily one of today's options.
    """
    assert "RATES.includes(stored) ? stored : 1" in HTML


# ------------------------------------------------- the jobs tab's player --


def test_the_jobs_player_is_not_inside_the_list_rebuilt_on_a_timer():
    """renderJobs() assigns innerHTML, on a two-second tick while a job runs.

    An <audio> inside that markup is destroyed and recreated on every tick, so
    it restarts from zero every two seconds. That is not a layout detail, it is
    the feature not working, and the only fix is one persistent element outside
    the list.
    """
    render = code(body_of("function renderJobs()", "async function downloadJob"))
    assert "<audio" not in render, "a player is being rebuilt by the job list"
    assert '<audio id="jobplayer"' in HTML, "the jobs tab has no player at all"


def test_the_job_audio_is_fetched_with_the_key_and_not_put_in_a_src():
    """GET /jobs/{id}/audio needs Authorization and <audio src> carries none.

    Pointing the element straight at the route is a 401 on any deployment with
    GATEWAY_API_KEYS set, which is the deployment this is for.
    """
    play = body_of("async function playJob(", "\n$(\"jobclose\")")
    assert "await api(" in play, "the job audio is not fetched through api()"
    assert "URL.createObjectURL" in play


# ------------------------------------------------------- where cues come --


def test_word_timings_are_asked_for_only_when_there_is_audio_to_play():
    """Otherwise it is pure cost for numbers nothing on screen can use.

    Timestamps are a second decoder pass per segment on Whisper, and stt's
    asr.py measures about 5% on Parakeet -- 5.34 s against 5.07 s on a 14.2 s
    clip. A link is transcribed server-side and its audio never reaches this
    browser, so on that path there is nothing to follow along with.
    """
    chooser = body_of("function formatForUpload(", "\n\nasync function transcribeToken")
    assert "return sttAudio() ? \"verbose_json\" : \"text\";" in chooser


def test_an_expert_response_format_is_never_silently_overridden():
    """Someone who selected srt in the panel gets srt, not verbose_json."""
    chooser = body_of("function formatForUpload(", "\n\nasync function transcribeToken")
    first = chooser.index("if ($(\"x-rf\").value) return wanted;")
    assert first < chooser.index("verbose_json"), (
        "the expert choice is checked after the swap, so it can be overridden")


def test_both_granularities_are_requested_so_there_is_a_fallback():
    """`word` is what the highlight follows; `segment` is what saves it.

    A glossary rule spanning two words fires in the segment text and cannot
    fire in either word, so on that transcript the words stop reconstructing
    the line and locate() refuses them. Without segments in the same response
    the highlight would simply vanish for those files.
    """
    submit = HTML[HTML.index('form.append("response_format", format);'):]
    head = submit[:submit.index("x-logprobs")]
    assert 'form.append("timestamp_granularities[]", "word");' in head
    assert 'form.append("timestamp_granularities[]", "segment");' in head


def test_the_download_still_writes_the_format_the_user_asked_for():
    """The upload path can send verbose_json for a transcript chosen as text.

    The bytes are identical -- openai_api.py's _body() returns result.text for
    response_format=text and puts that same string in verbose_json's `text` --
    but the file name must not follow the request the page made behind the
    scenes.
    """
    render = body_of("async function render(response, format, wanted)", "$(\"copy\")")
    assert "const kind = wanted || format;" in render
    assert 'name: "transcript." + (kind ===' in render


def test_the_subtitle_cue_pattern_matches_what_this_stack_writes():
    """The page's parser and services/stt's _clock() must agree about a cue.

    SubRip separates the fraction with a comma and WebVTT with a full stop, and
    stt writes both from the same function. A pattern that took only one of
    them would silently return no cues for half the formats this page offers,
    which looks exactly like "the file had no timings in it".
    """
    found = re.search(r"const CUE_LINE = /\^(.*?)/;\n", HTML)
    assert found, "ui.html has no CUE_LINE pattern"
    pattern = re.compile(found.group(1))
    assert pattern.match("00:00:04,099 --> 00:00:07,500"), "SubRip cue not matched"
    assert pattern.match("00:00:04.099 --> 00:00:07.500"), "WebVTT cue not matched"
    assert not pattern.match("1"), "a SubRip index line is being read as a cue"
    assert not pattern.match("WEBVTT"), "the WebVTT header is being read as a cue"


# --------------------------------------------------- how it is rendered --


class Sniff(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.handlers: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.handlers += [k for k, _ in attrs if k.startswith("on")]


def test_the_highlighted_transcript_is_built_from_nodes_not_from_a_string():
    """This is the change that would reintroduce the hole fixed two commits ago.

    A transcript is remote data, it is the largest piece of remote data this
    page renders, and per-word spans are the only place it is rendered as
    hundreds of elements. Built with createElement and textContent there is no
    string for an injection to live in at all, which is a stronger property
    than "esc() was remembered on every one of them".
    """
    paint = body_of("function paint(host, display, cues)", "\n/* The cue whose")
    assert "innerHTML" not in paint, "the karaoke renderer reaches innerHTML"
    assert "createElement" in paint and "createTextNode" in paint
    assert "span.textContent = display.slice(" in paint, (
        "the word is not being set as text")


def test_a_hostile_transcript_would_be_text_even_if_it_were_interpolated():
    """The failing half of the test above, so it cannot pass vacuously.

    textContent takes this string as five-and-a-bit words. innerHTML takes it
    as an element with an event handler, on a page whose CSP allows inline
    script because its own script block is inline, and whose API key is in
    localStorage beside it.
    """
    payload = '<img src=x onerror="fetch(\'/x\'+localStorage.key)">'
    sniff = Sniff()
    sniff.feed(f"<span>{payload}</span>")
    assert "img" in sniff.tags and "onerror" in sniff.handlers


def test_a_timing_that_cannot_be_placed_exactly_is_refused():
    """A highlight the page cannot place is not shown at all.

    Cues are ranges of the displayed string, never copies of it, so the pane is
    painted from the response's own bytes and cannot disagree with what the
    Download button writes. locate() returning null is the guard, and it is
    meant to fire rather than being defensive decoration.
    """
    locate = body_of("function locate(display, pieces)", "\nfunction timedFromJson")
    assert "if (at < 0) return null;" in locate
    assert "cues.push({ start: piece.start, end: piece.end, at, length:" in locate
    assert '(piece.text || "").trim()' in locate, (
        "faster-whisper's leading space is not stripped, so no segment will be found")


def test_the_highlight_is_never_carried_by_colour_alone():
    """Colour is gone for a red-green deficiency, gone on a badly set
    projector, and gone in forced-colours mode where the system palette
    overrides `background` and only the underline survives."""
    rule = HTML[HTML.index(".cue.on{"):]
    rule = rule[:rule.index("}")]
    assert "font-weight:700" in rule
    assert "text-decoration:underline" in rule
    assert "background:var(--mark-bg)" in rule


def test_the_highlight_has_a_colour_in_both_themes():
    """The page follows prefers-color-scheme, and a token defined only in the
    light block renders as nothing at all in the dark one."""
    light = HTML[HTML.index(":root{"):HTML.index("@media (prefers-color-scheme:dark)")]
    dark = HTML[HTML.index("@media (prefers-color-scheme:dark)"):HTML.index("*{box-sizing")]
    for token in ("--mark-bg", "--mark-ink"):
        assert token in light, f"{token} is not defined for the light theme"
        assert token in dark, f"{token} is not defined for the dark theme"


def test_reduced_motion_removes_the_motion_and_not_the_information():
    """The highlight is information; the transition and the scroll are motion.

    Switching the highlight off under prefers-reduced-motion would remove the
    feature rather than calm it, so only the two moving parts respond to it.
    """
    assert "@media (prefers-reduced-motion:reduce){ .cue.on{transition:none} }" in HTML
    follow = body_of("function follow()", "/* requestAnimationFrame, not")
    assert 'matchMedia("(prefers-reduced-motion: reduce)").matches' in follow
    assert '? "auto" : "smooth"' in follow


def test_the_highlight_scrolls_the_transcript_box_and_not_the_page():
    """scrollIntoView drags the whole document, taking the audio controls off
    screen at the exact moment they are wanted."""
    follow = code(body_of("function follow()", "/* requestAnimationFrame, not"))
    assert "scrollIntoView" not in follow
    assert "host.scrollTo({" in follow


def test_the_follow_loop_is_not_driven_by_timeupdate_alone():
    """Every engine fires timeupdate about four times a second.

    A word lasting 300 ms would be lit up to 250 ms late, which is close enough
    to look like the drift this feature exists not to have.
    """
    assert "requestAnimationFrame(tickKaraoke)" in HTML
    assert "cancelAnimationFrame(karaoke.frame)" in HTML, "the loop is never stopped"


# ------------------------------------------- what is deliberately absent --


def test_the_speech_direction_offers_no_highlight_and_says_why():
    """Nothing in the speech direction returns a timing, so nothing is faked.

    /speak answers audio and a usage count, /v1/audio/speech the same, and
    tts-long's _public() strips `segments` from every job it reports -- leaving
    chunks, audio_seconds and compute_seconds, and no chunk boundaries. The
    only thing constructible from that is duration x (chars so far / chars
    total), and it is wrong from the first sentence: Chatterbox inserts
    per-segment pauses, chunk_text() splits where the server decides, and
    speech rate moves with punctuation.
    """
    hint = HTML[HTML.index('id="playhint"'):]
    hint = hint[:hint.index("</div>")]
    assert "no word-by-word highlight here" in hint
    assert "neither speech engine returns" in hint


def test_the_link_path_says_why_there_is_nothing_to_play():
    """Not a disabled button, and not a player that 404s.

    /ui/fetch streams MeTube's file straight into the gateway server-side and
    the browser never receives a byte. That is the design -- it is what makes a
    two-hour podcast cost this laptop a transcript rather than 131 MB -- so the
    absence is explained rather than worked around.
    """
    playback = body_of("function sttPlayback(display, cues)", "/* A file the browser")
    assert "$(\"sttwhy\").textContent = stt.token" in playback
    assert "never reaches this browser" in playback


def test_a_file_the_browser_cannot_decode_does_not_look_like_a_failure():
    """The server decoded it, the transcript is real, and only the
    follow-along is missing -- stt decodes with libav and handles far more than
    a browser does."""
    handler = HTML[HTML.index('$("sttplayer").addEventListener("error"'):]
    handler = handler[:handler.index("\nlet lastTranscript")]
    assert "The transcript above is unaffected" in handler
    assert '$("sttplay").hidden = true' in handler
