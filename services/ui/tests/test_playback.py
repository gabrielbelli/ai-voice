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
    # <video> as well as <audio>, since the transcribe tab now has one. A
    # video element that silently shipped without a rate control is the same
    # gap this test was written for, wearing a different tag.
    ids = set(re.findall(r'<(?:audio|video) id="([a-z]+)"', HTML))
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


def test_word_timings_are_asked_for_only_when_something_can_follow_them():
    """Otherwise it is pure cost for numbers nothing on screen can use.

    Timestamps are a second decoder pass per segment on Whisper, and stt's
    asr.py measures about 5% on Parakeet -- 5.34 s against 5.07 s on a 14.2 s
    clip.

    THE ANSWER USED TO BE "is there a file in this browser", and that is
    exactly why a link never had a highlight: its media is streamed into the
    gateway server-side, so sttAudio() is null and the page asked for plain
    text. A finished link is playable through /ui/media now, so it qualifies --
    and a captions download still does not, because skip_download means no
    media was fetched at all.
    """
    chooser = body_of("function formatForTimings(", "\n\n/* ONE ANSWER FOR BOTH")
    assert "return willFollowAlong() ? \"verbose_json\" : \"text\";" in chooser
    predicate = code(body_of("function willFollowAlong()", "\n\n/* WHY THIS ASKS"))
    assert "if (sttAudio()) return true;" in predicate
    assert "return !!stt.token && !stt.captions;" in predicate


def test_an_expert_response_format_is_never_silently_overridden():
    """Someone who selected srt in the panel gets srt, not verbose_json."""
    chooser = body_of("function formatForTimings(", "\n\n/* ONE ANSWER FOR BOTH")
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
    chooser = code(body_of("function granularities(format, wanted)",
                           "\n\nasync function transcribeToken"))
    assert 'return ["word", "segment"];' in chooser
    # A value left in a control the user cannot reach must not go on deciding
    # what is sent -- the same rule chosenFormat() follows.
    assert '$("x-gran").value && !$("x-gran").disabled' in chooser


def test_the_link_and_the_upload_ask_for_the_same_cues():
    """Two copies of this rule would be two things that must agree about a cue,
    with only one of them the one these tests read.

    The upload path puts them in a multipart form and the link path in a query
    string /ui/fetch turns into the same fields, so the SHAPE differs and the
    decision must not.
    """
    upload = HTML[HTML.index('form.append("response_format", format);'):]
    upload = upload[:upload.index("x-logprobs")]
    assert "for (const grain of granularities(format, wanted))" in upload
    link = body_of("async function transcribeToken()", "\n\n/* THE CAPTIONS PATH")
    assert "for (const grain of granularities(format, wanted))" in link


def test_a_link_asks_for_the_verbose_body_the_cues_live_in():
    """The other half of why a link had no highlight, and the half that is not
    in this file: /ui/fetch forwarded `model` and `response_format` and nothing
    else, so even asking would not have reached stt. Both had to move."""
    link = code(body_of("async function transcribeToken()", "\n\n/* THE CAPTIONS PATH"))
    assert "const format = formatForTimings(wanted);" in link
    assert 'query.append("timestamp_granularities", grain);' in link
    # `wanted`, so a transcript chosen as text still downloads as .txt even
    # though verbose_json was asked for behind the scenes.
    assert "await render(response, format, wanted);" in link


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

    THIS IS ABOUT THE CODE, NOT THE COPY. The explanation used to be three
    sentences under the player and was cut as interface clutter: why a backend
    cannot return timings is a fact for a commit message, not for someone
    mid-task. What must stay true is that nothing FAKES a highlight here.
    """
    speak = HTML[HTML.index("function play(blob"):]
    speak = speak[:speak.index("\n}\n")]
    for faked in ("karaoke", "cues", "paint("):
        assert faked not in speak, (
            f"play() references {faked!r}: the speech direction has no timings "
            f"to highlight with, and an estimate drifts from the first sentence")


# ------------------------------------------------- a link, played back --
#
# EVERY SCREENSHOT THIS USER SENDS IS A PASTED LINK, and until /ui/media the
# player, the caption band and the karaoke highlight all worked only for a
# dropped file. Three separate things had to be true for a link and none of
# them were: the response had to carry timings (formatForTimings and the
# granularities above), the media had to reach the browser at all, and the
# element had to be able to seek in it.


def _playback() -> str:
    return body_of("function sttPlayback(display, cues, lines)",
                   "/* A file the browser")


def test_a_link_has_a_player_now_and_not_an_explanation_of_its_absence():
    """The sentence #sttwhy used to always print stopped being true.

    It said a link's audio never reaches this browser -- correct until
    /ui/media, and afterwards the page explaining why a feature is missing
    directly above that feature working.
    """
    source = code(body_of("function sttSource()", "\nlet sttUrl"))
    assert "if (!stt.media) return null;" in source
    assert "return { url: stt.media.url" in source
    body = code(_playback())
    assert "never reaches this browser" not in body, (
        "the page still claims a link cannot be played back")


def test_the_only_link_left_without_a_player_is_the_captions_one():
    """skip_download means no media stream was pulled at all, so that path
    genuinely has nothing to play -- and it is the one case where this page is
    fastest, which is worth its own sentence rather than a shortened version of
    somebody else's."""
    body = _playback()
    assert '$("sttwhy").textContent = stt.token' in body
    assert "the subtitles were taken and no media was downloaded" in body


def test_the_element_is_given_the_url_and_does_the_ranges_itself():
    """THE WHOLE REASON SEEKING WORKS. A <video> seeks by asking for a byte
    range; fetch()ing the file here into a blob first would download the entire
    thing before the first frame and put a two-hour podcast in this tab, which
    is the exact cost the server-side design exists to avoid."""
    body = code(_playback())
    assert "player.src = source.url;" in body
    assert "createObjectURL(source.file)" in body, (
        "a dropped file must still play from bytes already in this browser")


def test_nothing_is_fetched_until_somebody_presses_play():
    """preload="auto" on an element pointed at /ui/media would pull the whole
    file off the NAS the moment a transcript appeared, silently undoing the
    thing that makes a link affordable."""
    for element in ('<video id="sttvideo"', '<audio id="sttplayer"'):
        tag = HTML[HTML.index(element):]
        tag = tag[:tag.index(">")]
        assert 'preload="metadata"' in tag, f"{element} may preload the lot"


def test_a_stale_element_is_not_left_pulling_ranges_off_the_nas():
    """A link's media is a plain URL rather than an object URL, so the old
    "only clear it if there was a blob" guard left an element streaming behind
    a transcript that had already been replaced."""
    body = code(_playback())
    clear = body.index('$(id).removeAttribute("src");')
    revoke = body.index("URL.revokeObjectURL(sttUrl)")
    assert clear < revoke, (
        "the src must be dropped before the revoke, or the element raises an "
        "error event over a transcript that is fine")
    assert "if (sttUrl) { URL.revokeObjectURL" in body


def test_which_element_is_decided_by_the_file_that_arrived():
    """Not by what was asked for. A video commit that came back as audio -- an
    extractor with no muxed format -- would otherwise be a black rectangle with
    nothing on screen saying why."""
    watch = code(body_of("async function watchDownload()", "\n/* ------"))
    assert "video: VIDEO_FILE.test(state.filename" in watch
    assert 'stt.media = { url: "/ui/media?token=" + encodeURIComponent(stt.token)' in watch


def test_the_video_is_off_until_it_is_ticked_and_stays_off_next_time():
    """Audio-only is what makes a link affordable and it is the default. A
    checkbox that remembered its last value would make the expensive answer the
    one already selected when the card opens."""
    confirm = code(body_of("function showConfirm(facts)", "\n$(\"c-cancel\")"))
    assert '$("c-video").checked = false;' in confirm
    go = body_of('$("c-go").addEventListener', "\n\nasync function startFetch")
    assert 'video: $("c-video").checked' in go
    assert 'type="checkbox" id="c-video"' in HTML


def test_the_card_says_what_keeping_the_video_costs_before_it_is_fetched():
    """And says it in the row it changes. A tick that turns 131 MB into
    gigabytes while the Download row goes on quoting 131 MB is a cost
    discovered afterwards rather than decided beforehand."""
    facts = body_of("function paintFacts(facts, compute)", "\n\n/* Bound once")
    assert "gigabytes, not the ${human(facts.bytes)} of audio" in facts
    # And the row is repainted when the box changes, or it says the old number.
    assert '$("c-video").addEventListener("change"' in HTML


def test_the_new_copy_stays_under_the_line_length_this_page_holds_to():
    """The user has twice asked for interface copy to be CUT. Every string this
    feature adds is one sentence, and this is the bar it was written to."""
    strings = [
        "Keep the video — a much bigger download",
        "video — gigabytes rather than megabytes",
        "No player: that download is not something this page can play back.",
        " Streamed from the server as you play.",
    ]
    for line in strings:
        assert line in HTML, f"the page no longer says {line!r}"
        assert len(line) < 110, f"{len(line)} characters: {line!r}"


def test_a_file_the_browser_cannot_decode_does_not_look_like_a_failure():
    """The server decoded it, the transcript is real, and only the
    follow-along is missing -- stt decodes with libav and handles far more than
    a browser does."""
    handler = HTML[HTML.index('$("sttplayer").addEventListener("error"'):]
    handler = handler[:handler.index("\nlet lastTranscript")]
    assert "The transcript above is unaffected" in handler
    assert '$("sttplay").hidden = true' in handler


# ------------------------------------- following the text on the Speak tab --


def test_a_link_that_will_not_load_is_not_blamed_on_the_browser():
    """Two faults, two places to look. A dropped file that will not decode is
    the browser's limitation; a link that will not load is /ui/media answering
    something other than media, and the gateway missing the route is the first
    thing to check. Telling someone their browser cannot play a file it never
    received sends them to fix the wrong thing."""
    handler = HTML[HTML.index('$("sttplayer").addEventListener("error"'):]
    handler = handler[:handler.index("\n/* A CONTAINER THE BROWSER")]
    assert "$(\"sttwhy\").textContent = sttStreaming" in handler
    assert "the server would not serve that media back" in handler
    body = code(_playback())
    assert "sttStreaming = !source.file;" in body


def test_the_speak_tab_follows_the_text_now_that_offsets_exist():
    """This was refused when the feature was built, and correctly: neither
    engine returned timings, and duration x (chars so far / chars total) drifts
    from the first sentence.

    That is no longer true. Both synthesisers make one segment at a time and
    know each one's length as they splice it, so the boundaries are exact --
    they were simply computed and thrown away. tts-stack returns them in
    X-Segment-Offsets and tts-long records them on the job.
    """
    assert "function speakCues(" in HTML
    body = HTML[HTML.index("function speakCues("):]
    body = body[:body.index("\n}\n")]
    assert "x-segment-offsets" in HTML.lower()
    assert 'karaoke.player = $("player")' in body


def test_the_speak_highlight_is_segment_level_and_says_so():
    """A segment boundary is a fact the synthesiser measured. A word boundary
    inside one is not -- it would need forced alignment, and dividing a
    segment's duration by its word count is the drifting estimate this avoids.
    """
    body = HTML[HTML.index("function speakCues("):]
    head = HTML[:HTML.index("function speakCues(")]
    comment = head[head.rindex("/*"):]
    assert "forced alignment" in comment or "NOT word level" in comment


def test_paragraphs_survive_into_the_highlight():
    """Segments are joined the way they read, not the way they were sent."""
    body = HTML[HTML.index("function speakCues("):]
    body = body[:body.index("\n}\n")]
    assert 'const sep = "\\n\\n"' in body


def test_a_mismatched_offset_count_is_ignored_rather_than_guessed():
    """Fewer offsets than segments would silently misalign every highlight
    after the gap, which is worse than showing none."""
    body = HTML[HTML.index("function speakCues("):]
    body = body[:body.index("\n}\n")]
    assert "offsets.length !== texts.length" in body


# ================================================ subtitles as transcript ==
#
# "This has real subtitles already" on the confirm card asks MeTube for
# download_type "captions", which sets yt-dlp's skip_download and produces a
# .vtt or .srt and no media. The page then sent that file to /ui/fetch, which
# streams into /v1/audio/transcriptions -- so stt-stack was handed a text file
# and asked to decode it as media. The button could not work as written, and
# the failure surfaced two services away as a decode error.


def test_a_captions_download_is_never_sent_to_the_transcriber():
    """THE BUG, on the page side. watchDownload() called transcribeToken() for
    every finished download, including the one that is already a transcript."""
    watch = body_of("async function watchDownload()", "/* ------------------")
    ready = watch[watch.index("if (state.ready)"):]
    branch = ready[:ready.index("transcribeToken()")]
    assert "if (stt.captions)" in branch, "there is no captions branch at all"
    assert "captionsToken()" in branch, (
        "the captions branch does not reach the route that reads subtitles")


def test_the_captions_route_transcribes_nothing():
    """The whole value of the button is that it costs about two seconds and no
    compute. A captions path that still called stt would be slower than the
    ordinary one, for a worse transcript."""
    fn = code(body_of("async function captionsToken()", "\nfunction renderCaptions"))
    assert '"/ui/captions"' in fn
    for transcriber in ("/ui/fetch", "/v1/audio/transcriptions", "/transcribe"):
        assert transcriber not in fn, f"the captions path reaches {transcriber}"


def test_the_captions_flag_is_cleared_wherever_the_token_is():
    """A stale `true` sends the NEXT link to /ui/captions, which answers 409
    not_captions about a download that is perfectly fine."""
    resets = HTML.count("stt.captions = false")
    assert resets >= 3, (
        f"only {resets} places clear it; pick(), abandon() and resolveLink() "
        f"each set or clear the token and must each clear this")


def test_text_was_asked_for_so_timecodes_are_not_what_comes_back():
    """Text is the default and it is why most people press that button.

    Handing back a WebVTT file with its cue numbers and arrows in it, because
    that happens to be what yt-dlp wrote, is the page showing its plumbing.
    """
    fn = body_of("function renderCaptions(payload)", "\n$(\"go-stt\")")
    assert 'lines.map(line => line.text).join("\\n")' in fn, (
        "the prose branch is missing; the raw subtitle file reaches the pane")
    assert 'const kind = !lines.length ? payload.format : verbatim ? wanted : "txt";' in fn


def test_a_subtitle_format_that_was_asked_for_is_the_one_written():
    """yt-dlp writes WebVTT unless told otherwise, so someone who chose SubRip
    would otherwise get a file named .srt with WebVTT inside it."""
    fn = body_of("function renderCaptions(payload)", "\n$(\"go-stt\")")
    assert 'toSubtitles(lines, wanted === "vtt")' in fn


def test_an_unparsable_caption_file_is_shown_rather_than_swallowed():
    """A format this page's one pattern does not read is not a reason to draw
    an empty pane over a file that plainly has words in it."""
    fn = body_of("function renderCaptions(payload)", "\n$(\"go-stt\")")
    assert "!lines.length ? payload.text" in fn


def test_there_is_exactly_one_subtitle_parser():
    """Three callers now want cues -- the highlight, the band and the sidecar.

    A second implementation would be two parsers that must agree about what a
    cue is, in the same file, with only one of them read by these tests.
    """
    uses = [n for n, line in enumerate(HTML.splitlines(), 1) if "CUE_LINE" in line]
    # One declaration, one use inside parseSubtitles.
    assert len(uses) == 2, f"CUE_LINE is read in {len(uses)} places: {uses}"
    assert "CUE_LINE.exec" in body_of("function parseSubtitles(raw)",
                                      "\nfunction timedFromSubtitles")


# =========================================================== the overlay ==


def test_a_video_upload_gets_a_video_element_and_not_only_sound():
    """The ask was a player for the video with the transcript over it. An
    <audio> cannot become a <video>, so both exist and one is shown."""
    assert '<video id="sttvideo"' in HTML
    playback = body_of("function sttPlayback(display, cues, lines)",
                       "/* A file the browser")
    assert "const video = source.video;" in playback
    assert '$("sttstage").hidden = !video;' in playback
    assert '$("sttplayer").hidden = !!video;' in playback


def test_the_video_plays_the_original_file_and_not_the_decoded_wav():
    """`prepared` is the 16 kHz mono WAV this page made for the upload and it
    has no picture in it. The timings still line up because toWav is called
    from pick() with no maxSeconds and no startAt, so both are a whole-file
    decode on one timeline -- which is what makes drawing one over the other
    correct rather than approximately correct."""
    fn = body_of("function sttVideo()", "\n/* WHAT THIS PLAYER IS POINTED AT")
    assert "const file = stt.file;" in fn
    assert "stt.prepared" not in fn, "the video element is being given the WAV"
    assert "canPlayType" in fn, (
        "decodeAudioData reads containers the browser will not render, so the "
        "MIME prefix alone is not the question being asked")


def test_the_band_is_segment_level_and_the_highlight_stays_word_level():
    """One word at a time over a picture is unreadable, and a subtitle is the
    unit a viewer's eye is trained on. They are two scales of one cue list, not
    two sources -- linesFromJson reads the segments verbose_json already
    carries, so the band costs nothing extra on the wire."""
    follow = code(body_of("function follow()", "/* requestAnimationFrame, not"))
    assert "band(karaoke.player.currentTime);" in follow
    assert "cueAt(karaoke.player.currentTime, karaoke.cues, 2)" in follow
    band = code(body_of("function band(seconds)", "\nfunction follow()"))
    assert "cueAt(seconds, karaoke.lines, 0.3)" in band, (
        "the band uses the word-level tolerance, so a caption hangs into "
        "silence over a picture that shows nothing is being said")


def test_the_band_is_written_only_when_it_changes():
    """follow() runs once a frame. Assigning textContent sixty times a second
    for a line that turns over about once a sentence is a layout the browser
    does not need to do."""
    band = code(body_of("function band(seconds)", "\nfunction follow()"))
    assert "if (index === karaoke.line) return;" in band


def test_the_band_is_text_and_never_markup():
    """It carries the transcript, which is the largest piece of remote data on
    the page and the one with a real XSS in its history."""
    band = code(body_of("function band(seconds)", "\nfunction follow()"))
    assert '$("sttbandtext").textContent =' in band
    assert "innerHTML" not in band


def test_the_band_is_drawn_only_over_a_picture():
    """Over an audio element it is a caption floating on nothing, saying what
    the highlighted word three centimetres below it already says."""
    playback = body_of("function sttPlayback(display, cues, lines)",
                       "/* A file the browser")
    assert "karaoke.lines = video && lines && lines.length ? lines : null;" in playback


def test_stopping_the_highlight_also_clears_the_band():
    """Otherwise the last caption of the previous file sits over the next one."""
    stop = body_of("function stopKaraoke()", "/* BOTH PLAYERS DRIVE")
    assert "karaoke.lines = null" in stop
    assert '$("sttband").hidden = true;' in stop


def test_the_video_element_drives_the_same_highlight_as_the_audio_one():
    """karaoke.player is whichever element got the src, so nothing downstream
    needs to know which of the two is playing."""
    assert '["sttplayer", "sttvideo", "player"].forEach(id => {' in HTML
    playback = body_of("function sttPlayback(display, cues, lines)",
                       "/* A file the browser")
    assert "karaoke.player = player;" in playback


def test_a_video_the_browser_will_not_render_keeps_the_audio_and_the_words():
    """canPlayType answers "maybe" for a great deal it then declines. Losing
    the transcript and the follow-along as well as the picture would be three
    things broken by one missing decoder."""
    handler = body_of('$("sttvideo").addEventListener("error"', "\nlet lastTranscript")
    assert "stt.noVideo = true;" in handler, "the fallback can loop"
    assert "sttPlayback(lastPlayback.display" in handler


def test_the_overlay_has_its_own_contrast_and_does_not_borrow_the_page_tokens():
    """It sits over a picture, so the page's background token says nothing
    about what it needs to be readable against. The plate is opaque for the
    same reason."""
    rule = HTML[HTML.index(".band span{"):]
    rule = rule[:rule.index("}")]
    assert "background:rgba(0,0,0,.74)" in rule and "color:#fff" in rule
    assert "var(--" not in rule, "the caption is following the page theme"


# ============================================================ the sidecar ==
#
# Burn-in was decided against: it means ffmpeg in this image and a full
# re-encode to produce a caption track every player already reads. A .srt or
# .vtt beside the file is what makes that decision livable.


def _cue_pattern() -> re.Pattern:
    found = re.search(r"const CUE_LINE = /\^(.*?)/;\n", HTML)
    assert found, "ui.html has no CUE_LINE pattern"
    return re.compile(found.group(1))


def _stamp(seconds: float, comma: bool) -> str:
    """The page's stamp(), transcribed. Kept in step by the test below."""
    ms = round(max(0.0, seconds) * 1000)
    return (f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}"
            f"{',' if comma else '.'}{ms % 1000:03d}")


def test_the_page_stamp_matches_the_one_this_file_asserts_with():
    """If ui.html's stamp() is changed, the parity check below must stop
    agreeing with it rather than quietly testing a copy of the old one."""
    fn = body_of("function stamp(seconds, comma)", "\nfunction toSubtitles")
    assert "Math.round(Math.max(0, Number(seconds) || 0) * 1000)" in fn
    assert 'comma ? "," : "."' in fn
    for part in ("ms / 3600000, 2", "ms / 60000 % 60, 2",
                 "ms / 1000 % 60, 2", "ms % 1000, 3"):
        assert part in fn, f"stamp() no longer builds {part}"


def test_a_written_cue_line_is_one_this_page_can_read_back():
    """The sidecar and the parser are the two halves of one round trip: a file
    this page writes and cannot then parse would break the karaoke highlight
    for anyone who saved a transcript and dropped it back in."""
    pattern = _cue_pattern()
    for comma in (True, False):
        line = f"{_stamp(3661.5, comma)} --> {_stamp(3663.25, comma)}"
        assert pattern.match(line), f"{line!r} does not match CUE_LINE"
    assert _stamp(3661.5, True).startswith("01:01:01,500")


def test_subrip_is_numbered_and_webvtt_carries_its_header():
    """A .vtt without WEBVTT on the first line is rejected outright by every
    browser that loads it, and a .srt without cue numbers by several players."""
    fn = body_of("function toSubtitles(lines, vtt)", "\nfunction saveText")
    assert '(vtt ? "" : (i + 1) + "\\n")' in fn
    assert '(vtt ? "WEBVTT\\n\\n" : "")' in fn


def test_the_sidecar_is_hidden_rather_than_disabled_when_there_are_no_timings():
    """On the native route and on a plain json response there are no timings at
    all, and there is nothing the user could do about it -- so a disabled
    button would be a promise the page cannot keep."""
    fn = body_of("function offerSidecar(lines)", "\nfunction sidecarName")
    assert '$("dl-srt").hidden = $("dl-vtt").hidden = !lastLines.length;' in fn
    assert '<button class="small tight" id="dl-srt" type="button" hidden>' in HTML


def test_every_path_that_draws_a_transcript_also_offers_the_sidecar():
    """A transcript with timings and no way to save them as subtitles is the
    feature half-built."""
    for fn, ends in (("async function render(response, format, wanted)", '$("copy")'),
                     ("function renderCaptions(payload)", '\n$("go-stt")')):
        assert "offerSidecar(" in body_of(fn, ends), f"{fn} does not offer it"


def test_the_sidecar_and_the_download_button_write_through_one_helper():
    """Two object-URL lifetimes to get right rather than one is how the second
    one leaks."""
    assert HTML.count("function saveText(text, name, type)") == 1
    assert HTML.count("URL.revokeObjectURL(url), 5000") == 1


def test_no_burn_in_arrived_by_the_back_door():
    """The overlay exists BECAUSE burn-in was rejected: it would mean ffmpeg in
    this image and a re-encode of the whole file. The Containerfile says so in
    two places and the requirements in one."""
    root = Path(__file__).resolve().parents[1]
    assert "ffmpeg" not in (root / "requirements.txt").read_text().replace(
        "no ffmpeg", "").replace("its own ffmpeg", "")
    container = (root / "Containerfile").read_text()
    assert "install -y --no-install-recommends util-linux" in container, (
        "the image installs something new; if it is a codec library the "
        "no-burn-in decision has been undone")
