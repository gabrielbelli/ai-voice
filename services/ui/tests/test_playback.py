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


def test_the_cues_are_asked_for_on_every_path_that_can_carry_them():
    """No state of this page may produce a transcript with no cues in it.

    THE ORIGINAL DEFECT, which this test was written for and still guards: a
    pasted link never had a highlight. The ask was gated on "is there a file in
    this browser", and a link's media is streamed into the gateway server-side,
    so sttAudio() was null and the page asked for plain text. The gate was
    fixed once to let a link through -- /ui/media makes it playable -- and this
    now removes the gate instead, which cannot be got wrong a third time.

    WHY THE GATE WENT RATHER THAN BEING WIDENED AGAIN. It bought about 5%
    (asr.py: 5.34 s against 5.07 s on a 14.2 s clip; a second decoder pass per
    segment on Whisper) by dropping to plain text when nothing could play the
    timings back. That was affordable only while the Transcript format control
    existed, because someone who wanted a subtitle file and no playback could
    select SRT and get one. With that control retired, verbose_json is the only
    thing that writes a subtitle file at all, so the gate would now withhold
    the feature that replaced the control -- from exactly the people who had
    been using it. The 5% buys the sidecar.
    """
    chooser = code(body_of("function formatForTimings()", "\n\n/* ONE ANSWER FOR BOTH"))
    assert 'return expertFormat() || "verbose_json";' in chooser
    # Comments stripped: the one above formatForTimings names the retired gate
    # on purpose, because a reader who does not know it existed cannot know
    # what the 5% is being spent on.
    assert "willFollowAlong" not in re.sub(r"/\*.*?\*/", "", HTML, flags=re.S), (
        "the cost gate is back; a run with no player loses its subtitle files")


def test_an_expert_response_format_is_never_silently_overridden():
    """Someone who selected srt in the panel gets srt, not verbose_json."""
    chooser = code(body_of("function formatForTimings()", "\n\n/* ONE ANSWER FOR BOTH"))
    assert chooser.index("expertFormat()") < chooser.index("verbose_json"), (
        "the expert choice is checked after the swap, so it can be overridden")


def test_both_granularities_are_requested_so_there_is_a_fallback():
    """`word` is what the highlight follows; `segment` is what saves it.

    A glossary rule spanning two words fires in the segment text and cannot
    fire in either word, so on that transcript the words stop reconstructing
    the line and locate() refuses them. Without segments in the same response
    the highlight would simply vanish for those files.
    """
    chooser = code(body_of("function granularities(format)",
                           "\n\nasync function transcribeToken"))
    assert 'return ["word", "segment"];' in chooser
    # A value left in a control the user cannot reach must not go on deciding
    # what is sent -- the same rule expertFormat() follows.
    assert '$("x-gran").value && !$("x-gran").disabled' in chooser
    # AND THE ESCAPE HATCH STAYS EXACT. The guard used to read
    # `format !== wanted`, meaning "only when the page did the upgrading":
    # someone who typed verbose_json into the panel and left the granularities
    # alone got verbose_json with none, which is what they asked for. `wanted`
    # went with the format control, so the same rule is written as what it
    # always meant -- not when the format came out of the escape hatch.
    assert 'format === "verbose_json" && !expertFormat()' in chooser


def test_the_link_and_the_upload_ask_for_the_same_cues():
    """Two copies of this rule would be two things that must agree about a cue,
    with only one of them the one these tests read.

    The upload path puts them in a multipart form and the link path in a query
    string /ui/fetch turns into the same fields, so the SHAPE differs and the
    decision must not.
    """
    upload = HTML[HTML.index('form.append("response_format", format);'):]
    upload = upload[:upload.index("x-logprobs")]
    assert "for (const grain of granularities(format))" in upload
    link = body_of("async function transcribeToken()", "\n\n/* THE CAPTIONS PATH")
    assert "for (const grain of granularities(format))" in link


def test_a_link_asks_for_the_verbose_body_the_cues_live_in():
    """The other half of why a link had no highlight, and the half that is not
    in this file: /ui/fetch forwarded `model` and `response_format` and nothing
    else, so even asking would not have reached stt. Both had to move."""
    link = code(body_of("async function transcribeToken()", "\n\n/* THE CAPTIONS PATH"))
    assert "const format = formatForTimings();" in link
    assert 'query.append("timestamp_granularities", grain);' in link
    assert "await render(response, format);" in link


def test_the_download_button_names_the_file_it_writes():
    """THE DEFECT THIS PINNED: the upload path sends verbose_json for what the
    user chose as Text, and the extension followed the request the page made
    behind the scenes rather than the choice. The bytes are identical --
    openai_api.py's _body() returns result.text for response_format=text and
    puts that same string in verbose_json's `text` -- so it was only ever the
    name that could be wrong, and it was.

    WHAT IT ASSERTS NOW. There is no choice to disagree with: the extension
    follows the format actually sent. text, json, verbose_json and native all
    put prose in the pane and all write .txt; only an expert response_format of
    srt or vtt puts a subtitle file there, and that writes it. The .srt and
    .vtt buttons come off the cues instead and never reach this line, which is
    how one run offers all three.

    AND THE LABEL SAYS WHICH. The button read the bare word "Download" beside
    two buttons reading ".srt" and ".vtt", so the row looked like one download
    and two somethings -- while the file it wrote silently followed a control
    somewhere above it. It is derived from lastTranscript.name, so it cannot
    drift from the bytes.
    """
    render = body_of("async function render(response, format)", "$(\"copy\")")
    assert 'const kind = format === "srt" ? "srt" : format === "vtt" ? "vtt" : "txt";' in render
    assert 'name: "transcript." + kind' in render
    assert "nameDownload();" in render
    label = body_of("function nameDownload()", "\nfunction sidecarName")
    assert "lastTranscript.name" in label, "the label is not read off the file"
    assert '<button class="small tight" id="download" type="button">Download .txt<' in HTML

    # THE ASSERTION ABOVE WAS SATISFIED BY THE WRONG LINE. nameDownload() reads
    # lastTranscript.name twice -- once to find the dot, once to take the
    # extension -- so replacing the whole assignment with the bare string
    # "Download" left the first read in place and the test went on passing. It
    # is the ASSIGNMENT that has to carry the extension, so that is what this
    # reads: the textContent the button ends up with, not merely a mention of
    # the file somewhere in the same function.
    written = code(label[label.index('$("download").textContent'):])
    assert "lastTranscript.name.slice(dot)" in written, \
        "the label no longer takes the extension off the file it writes"


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


def test_the_cues_are_actually_painted_and_not_merely_parsed():
    """PARSING THE TIMINGS AND DRAWING THEM ARE TWO THINGS, and only the first
    was pinned. Every test above this one asks whether the cues were requested,
    returned and located; none asked whether sttPlayback then turned them into
    the spans the highlight moves between. Emptying that one assignment --
    `karaoke.spans = []` instead of the paint() call -- left the whole suite
    green while the transcript rendered as one flat block of text and no word
    ever lit up, which is the entire feature gone with nothing to catch it.

    The three lines together are what make a highlight: the spans exist, the
    player they follow is known, and the pane is put into the mode the CSS
    styles the lit word from.
    """
    playback = code(body_of("function sttPlayback(display, cues, lines)",
                            "\n/* A file the browser cannot decode"))
    assert 'karaoke.spans = paint($("transcript"), display, cues);' in playback, \
        "the cues are located but never drawn"
    assert "karaoke.cues = cues;" in playback
    assert "karaoke.player = player;" in playback
    assert '$("transcript").className = "out karaoke";' in playback


def test_the_transcript_is_painted_before_it_is_told_to_look_painted():
    """ORDER, because the class is what the CSS draws the lit word with and the
    spans are what carries it. Setting the class first and painting after would
    look identical in every static assertion and flicker an unstyled transcript
    on every run.
    """
    playback = code(body_of("function sttPlayback(display, cues, lines)",
                            "\n/* A file the browser cannot decode"))
    assert playback.index("karaoke.spans = paint(") \
        < playback.index('$("transcript").className = "out karaoke"')


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
    """Nothing play() can see returns a timing, so play() fakes nothing.

    THE DOCSTRING CHANGED AND THE ASSERTION DID NOT, which is the right way
    round. It used to say tts-long's _public() strips every chunk boundary from
    the jobs it reports. That stopped being true twice over: `offsets` is in
    SIDECAR_KEYS, and tts-long now publishes it PER SEGMENT while the job runs,
    so a live clone job states exact boundaries. tts-stack returns them in
    X-Segment-Offsets on the buffered route.

    None of that reaches play(). play() is handed a finished blob and a
    filename; the boundaries belong to speakCues, which has the segments this
    page sent and can map them, and to the job row, which has a count. The one
    route with neither is /v1 over SSE: synth.plan splits by CHUNK_PHONEMES
    rather than by the caller's sentences, so the page cannot reconstruct the
    text of a delta, and counting deltas to index segments would be exactly the
    drifting estimate this ban exists to prevent.

    THIS IS ABOUT THE CODE, NOT THE COPY. Why a backend cannot return timings
    is a fact for a commit message, not for someone mid-task. What must stay
    true is that nothing FAKES a highlight here.
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
    somebody else's.

    THE SENTENCE IS SHORTER NOW. It read "the subtitles were taken and no media
    was downloaded", which says one thing twice: the reader is looking at a
    transcript, so where it came from is on the screen already. What is missing
    is the player, and the reason is that no media exists to play."""
    body = _playback()
    assert '$("sttwhy").textContent = stt.token' in body
    assert "No player: no media was downloaded." in body


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
    """The user has now asked THREE times for interface copy to be CUT. Every
    string this feature adds is one sentence, and this is the bar.

    THE FOURTH STRING IS GONE RATHER THAN SHORTENED. " Streamed from the server
    as you play." was true and did nothing: the scrub bar behaves the same
    either way, so it told the reader a fact with no action attached to it. It
    is asserted absent here so that a later pass cannot quietly restore it.

    The other three lost their em dashes. A dash is the punctuation that hides
    a second clause inside a first one, which is the construction this page's
    writing standard replaces with brackets, a colon or a full stop.
    """
    strings = [
        "Keep the video (a much bigger download)",
        "gigabytes, not megabytes",
        "No player: this page cannot play that download.",
    ]
    for line in strings:
        assert line in HTML, f"the page no longer says {line!r}"
        assert len(line) < 110, f"{len(line)} characters: {line!r}"
        assert "—" not in line, f"an em dash is back in {line!r}"
    assert "Streamed from the server as you play" not in code(_playback()), \
        "the sentence that did nothing is back on the player"


def test_a_file_the_browser_cannot_decode_does_not_look_like_a_failure():
    """The server decoded it, the transcript is real, and only the
    follow-along is missing -- stt decodes with libav and handles far more than
    a browser does."""
    handler = HTML[HTML.index('$("sttplayer").addEventListener("error"'):]
    handler = handler[:handler.index("\nlet lastTranscript")]
    # "above" is a direction, and a reader on a phone, or one who has scrolled,
    # is not looking at whatever the writer was. The transcript is named; where
    # it sits is not. The libav clause went with it: the comment over this
    # handler already carries the reason, and the reader cannot act on it.
    assert "The transcript is unaffected" in handler
    assert "above" not in handler, "the handler points at a position again"
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
    # Active voice with the actor named: the server did or did not do a thing.
    assert "the server did not return the media" in handler
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
    """Not by rebuilding them -- by never taking them apart.

    This asserted a separator the page inserted between sentences. That was
    the bug: the passage was reconstructed, so the writer's own line breaks
    were replaced by the page's. The spans are now offsets into the text as
    typed, and the whitespace between them is whatever was already there.
    """
    body = HTML[HTML.index("function speakCues("):]
    body = body[:body.index("\n}\n")]
    code = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "const sep =" not in code, "the page is still choosing the separator"
    assert "const display = plain" in code, "the original text should be shown"


def test_a_mismatched_offset_count_is_ignored_rather_than_guessed():
    """Fewer offsets than segments would silently misalign every highlight
    after the gap, which is worse than showing none.

    THE GUARD IS UNCHANGED AND MUST STAY UNCHANGED. speakCues is handed a
    FINISHED response, so a count that does not match is a disagreement and not
    a partial: padding it is guessing.

    A growing list is a different thing and it lives somewhere else. A running
    clone job publishes offsets per segment, so offsets.length <= chunks is a
    legal state on the jobs tab -- and there it is used as a COUNT and never as
    a highlight, because the passage would have to be rebuilt on every poll to
    light it and that is the defect the in-place fix removed. The two rules do
    not meet, and this test is what keeps them apart.
    """
    body = HTML[HTML.index("function speakCues("):]
    body = body[:body.index("\n}\n")]
    assert "offsets.length !== texts.length" in body
    # The jobs tab counts what it was given and interpolates nothing between.
    rows = code(body_of("function renderJobs()", "let jobUrl = null;"))
    assert "job.offsets.length" in rows, "the live boundaries are not read"
    assert "chars" not in rows, "a character-count estimate is back on the row"


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


def test_a_subtitle_file_taken_from_a_link_reaches_the_pane_as_prose():
    """Handing back a WebVTT file with its cue numbers and arrows in it,
    because that happens to be what yt-dlp wrote, is the page showing its
    plumbing. A transcript is what someone pressing that button came for.

    IT USED TO BE CONDITIONAL on Text being selected, which it was by default
    and which is why most people pressed the button. There is nothing to select
    now, so the good branch is the only branch -- and the two formats the other
    branches used to produce are on the sidecar buttons beside it, off the same
    parse.
    """
    fn = code(body_of("function renderCaptions(payload)", "\n$(\"go-stt\")"))
    assert 'lines.map(line => line.text).join("\\n")' in fn, (
        "the prose branch is missing; the raw subtitle file reaches the pane")
    assert 'const kind = lines.length ? "txt" : payload.format;' in fn
    assert "toSubtitles" not in fn, "renderCaptions writes timecodes into the pane again"
    # /ui/captions reads a file off disk and sends no response_format anywhere,
    # so an expert override has no business deciding how it is displayed.
    assert "expertFormat" not in fn and "x-rf" not in fn


def test_a_file_named_srt_can_never_contain_webvtt():
    """yt-dlp writes WebVTT unless told otherwise, so a caption download passed
    through as it arrived is a file named .srt with WebVTT inside it.

    IT USED TO BE renderCaptions' job, writing the download into whichever
    format was selected. Nothing is selected now and nothing is passed through:
    both sidecar buttons are written by toSubtitles() from the parsed cues, on
    the captions path exactly as on the transcription path, so the extension
    and the bytes come from the same place by construction.
    """
    fn = code(body_of("function renderCaptions(payload)", "\n$(\"go-stt\")"))
    assert "offerSidecar(lines)" in fn, "the captions path offers no subtitle files"
    assert "payload.text" in fn
    srt = body_of('$("dl-srt").addEventListener', '$("dl-vtt").addEventListener')
    assert "toSubtitles(lastLines, false)" in srt and 'sidecarName("srt")' in srt


def test_an_unparsable_caption_file_is_shown_rather_than_swallowed():
    """A format this page's one pattern does not read is not a reason to draw
    an empty pane over a file that plainly has words in it."""
    fn = code(body_of("function renderCaptions(payload)", "\n$(\"go-stt\")"))
    assert ": payload.text;" in fn


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
    for fn, ends in (("async function render(response, format)", '$("copy")'),
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


def test_the_spoken_text_is_shown_as_typed_not_rebuilt():
    """It used to join the sentences back together with a blank line, which is
    a DIFFERENT DOCUMENT: every paragraph break the writer made was replaced by
    one the page chose, and a line break inside a sentence vanished. The point
    of following along is reading your own text, not a reflowed copy.
    """
    body = HTML[HTML.index("function speakCues("):]
    body = body[:body.index("\n}\n")]
    code = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert 'texts.join("\\n\\n")' not in code, "the text is still being rebuilt"
    assert "locate(display" in code, \
        "spans should be offsets into the original, as the subtitle path does"


def test_an_unlocatable_split_shows_no_highlight_rather_than_a_wrong_one():
    """If the split and the original disagree, locate() returns null. A
    highlight sliding off the words is worse than none."""
    body = HTML[HTML.index("function speakCues("):]
    body = body[:body.index("\n}\n")]
    assert "if (!spans) return;" in body


def test_the_rendered_text_preserves_whitespace():
    """Without pre-wrap the browser collapses the newlines the spans were
    measured against, so the highlight would land in the wrong place."""
    assert "white-space:pre-wrap" in HTML[HTML.index(".out{"):
                                          HTML.index(".out{") + 120]


# ============================================== playing as it is made ======
#
# THE RULE THAT OUTRANKS THE FEATURE: the player must never run dry. An honest
# progress bar beats audio that stutters, and a player that runs dry teaches
# people to distrust every one they meet afterwards.
#
# Everything below is that rule, in the four places it can be broken: the
# arithmetic that decides, the format that makes the arithmetic possible, the
# scheduling that makes the joins exact, and the engine that must never be
# allowed to stream because the numbers say it cannot sustain.


def _rule() -> str:
    return HTML[HTML.index("const STREAM = {"):HTML.index("/* ========================================================== the tabs")]


def test_one_function_owns_the_lead_time_rule():
    """A second opinion about whether it is safe to play is how the two
    disagree at the one moment it matters.

    safe(headroom, remaining, R) is the whole decision, and it is the brief's
    t0 >= T * (1/R - 1) written in the quantity the page can measure. Headroom
    falls at (1-R) for the (T-arrived)/R seconds the generator still has to
    run, so its minimum is headroom - (1-R)(T-arrived)/R, and requiring that to
    be non-negative gives the predicate. With arrived = R*t0 at the start it
    reduces back to the brief's line.
    """
    rule = _rule()
    assert "function safe(headroom, remaining, R)" in rule
    assert "remaining * (STREAM.safety / R - 1)" in rule
    # A margin, a floor and a minimum rate, each with its reason beside it.
    for named in ("safety:", "floor:", "minRate:", "maxLead:"):
        assert named in rule, f"{named} is not a named constant any more"
    # No second decision anywhere: nothing else may gate the sound.
    script = code(HTML)
    assert script.count("function safe(") == 1


def test_nothing_starts_playing_on_a_fixed_delay():
    """"wait a second and hope" is the design this replaces. Every serious
    player has a named pre-roll and one second is the usual default for a
    source known to outrun playback; this source is not known to, and on one
    engine it provably does not."""
    sink = code(body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND"))
    assert "safe(headroom(), remaining, R)" in sink, "the sink decides for itself"
    assert "arrived >= STREAM.floor" in sink
    # Not a byte count and not a delta count.
    assert "deltas >=" not in sink.split("function wireRate")[1].split("}")[1] \
        if "function wireRate" in sink else True


def test_the_rate_is_measured_on_the_wire_and_health_is_only_the_prior():
    """tts-long's realtime_factor is an EMA seeded with RTF_SEED, so the field
    is never absent and a seed reads exactly like a measurement. A rule that
    trusts health blindly trusts a constant and will authorise a stream that
    stalls.

    pcm makes arrival self-describing -- bytes over 2 over 24000 is the seconds
    of audio that actually landed -- so the measured figure includes the
    network and both proxies rather than the model alone.
    """
    sink = body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND")
    assert "function wireRate()" in sink
    assert "arrived / elapsed" in code(sink), "the wire rate is not measured"
    # TWO DELTAS, NOT ONE. One delta measures model warm-up, not a rate.
    assert "deltas >= 2" in code(sink)
    # And the measurement outranks the prior once it exists.
    assert "const w = wireRate(); return w > 0 ? w : plan.R;" in code(sink)


def test_the_streamed_format_is_pcm_and_only_pcm():
    """Streamed wav writes a RIFF size of 8 and a data size of 0, so a
    concatenation of wav deltas decodes to nothing. mp3, opus, aac and flac
    carry encoder priming and padding, and their deltas lag the segments by up
    to TTS_ENCODER_FLUSH_WAIT with a tail delta after the last one, so a delta
    boundary is not a segment boundary there.

    For pcm, _RawEncoder.write returns the piece whole and close() returns
    nothing, so one delta is one synthesis chunk and its byte length over
    2 x 24000 is that chunk's exact duration. The sink requires it rather than
    degrading quietly into a format whose boundaries are a guess.
    """
    speak = code(body_of("async function speakNow(voice, text)", "async function collectSSE"))
    assert 'const format = ctx ? "pcm"' in speak, "the streamed format is not forced"
    assert "const PCM_RATE = 24000;" in HTML


def test_buffer_start_times_are_absolute_and_never_the_current_time():
    """THE CLASSIC DEFECT, and it is inaudible in a test that does not record
    the `when` argument. ctx.currentTime advances while the scheduling loop
    runs, so start times computed from it land in the past, where a buffer
    plays immediately and overlaps the one before it.

    pcm has no priming and no padding, so consecutive buffers abut
    sample-exactly if, and only if, the timeline is absolute.
    """
    sink = code(body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND"))
    assert "src.start(Math.max(when, floorAt)" in sink
    assert "const when = origin + piece.at;" in sink
    # Seeded once per start, from the context clock, and never inside the loop.
    assert sink.count("ctx.currentTime + 0.05") == 1


def test_the_underrun_is_pre_empted_by_a_timer_that_a_hidden_tab_cannot_stop():
    """requestAnimationFrame is throttled to a stop in a background tab and the
    audio goes on playing there, so the watchdog would stop watching exactly
    where nobody is looking.

    The graph is stopped while it still has audio in hand: a stated pause is
    recoverable, a stutter is not.
    """
    sink = code(body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND"))
    assert "setInterval(tick, 200)" in sink
    assert "requestAnimationFrame" not in sink, "the watchdog is throttled in a hidden tab"
    # AND NOT AGAINST THE DELTA SIZE. At the moment playback starts, headroom
    # IS one delta, so comparing the two paused a hundred milliseconds after
    # starting on the fast engine -- the stutter the rule exists to prevent.
    # The second reading is the gap since the last delta: if nothing has
    # arrived for longer than the audio still in hand, the next one is already
    # too late, whatever the mean rate says.
    assert "headroom() <= gapSeconds()" in sink, "no pre-emptive pause at all"
    assert "biggest" not in sink, "the delta size is back as a threshold"
    # A suspended context is the browser pausing, not the model falling behind.
    assert 'ctx.state !== "running"' in sink
    assert 'say("flat", "Paused.")' in sink


def test_the_clone_path_never_streams_audio_and_the_warning_stays():
    """0.230x on this server's CPU means a lead of 3.35 x T: a minute of speech
    would cost three minutes twenty of silence first, which is strictly worse
    than waiting for the file. The GPU runner at 0.644x to 0.746x is still
    under realtime.

    The arithmetic refuses it, so the player contains no engine name. What
    keeps the clone on 202-and-poll is a second, separate fact: tts-long's
    _sse_events finally marks a queued or running job cancelled when the reader
    hangs up, so closing a laptop lid would kill a thirty-six minute job.
    """
    queue = body_of("async function queueJob(voice, text)", "function hasAudio(job)")
    assert "stream_format" not in code(queue), "the clone path can open a stream"
    # Whitespace collapsed: a comment wraps at whatever column it wraps at, and
    # matching the raw text would pin the line breaks rather than the warning.
    flat = re.sub(r"\s+", " ", queue)
    assert "closing the stream CANCELS the job" in flat
    # And the submit path says the same thing where it chooses the route.
    speak = re.sub(r"\s+", " ", body_of("async function speakNow(voice, text)",
                                        "async function collectSSE"))
    assert "closing that stream cancels the job" in speak
    # And the decision is arithmetic, not a name.
    sink = code(body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND"))
    for name in ("chatterbox", "clone", "kokoro"):
        assert name not in sink.lower(), f"the player knows about {name!r}"


def test_a_stream_that_dies_half_way_still_hands_over_what_it_made():
    """A short file that looks complete is the failure to avoid, so the bytes
    are kept AND the reader is told the file is short.

    X-Job-Id is on every streamed response and survives both proxies, so work
    already written to disk is reachable from the Jobs tab rather than being
    run again from the start.
    """
    collect = code(body_of("async function collectSSE", "/* Builds the highlight"))
    assert "err.chunks = chunks;" in collect, "the bytes die with the error"
    speak = code(body_of("async function speakNow(voice, text)", "async function collectSSE"))
    assert "The stream ended early." in speak
    assert 'response.headers.get("x-job-id")' in speak
    assert "adopt(jobId)" in speak


def test_the_bytes_that_reach_the_file_are_the_bytes_that_arrived():
    """A tts-long test pins the streamed deltas byte-equal to the buffered
    body, and tts-stack's audio_out.py records that its hand-written 44 byte
    header is byte-identical to libsndfile's. Both properties are what make it
    safe to write the header here, and both die the moment this page adds
    anything to the audio."""
    collect = code(body_of("async function collectSSE", "/* Builds the highlight"))
    assert "chunks.push(bytes);" in collect
    wav = code(body_of("function wavFrom(chunks, hz)", "\n/* The live sink"))
    # A header in front of a copy, and nothing else: no gain, no resample, no
    # normalisation. Normalising the audio is a defect this stack has had.
    assert "new Blob([header, ...chunks]" in wav
    assert "44" in body_of("function wavFrom(chunks, hz)", "\n/* The live sink")


def test_two_players_cannot_speak_at_once():
    """The <audio> element with the assembled file and the Web Audio graph can
    both make a sound. This is the whole guard, and forgetting it is the most
    likely audible bug in the design."""
    assert '$("player").addEventListener("play", () => { if (speakSink) speakSink.silence(); });' in HTML


def test_stopping_the_sound_does_not_throw_the_work_away():
    """Abandoning the reader trips tts-long's _sse_events finally, which
    cancels a queued or running job. Stop means stop the sound; the body is
    still read to the end and the reader still gets the whole file.

    The same rule governs visibilitychange, which must never abort a reader.
    """
    sink = body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND")
    silence = sink[sink.index("silence()"):]
    silence = silence[:silence.index("\n")]
    assert "abort" not in silence.lower(), "Stop cancels the work"
    assert "quiet = true" in silence
    body = code(HTML)
    assert "visibilitychange" in body
    assert 'visibilitychange", () => { schedule(0); }' in body.replace(
        'visibilitychange", () => schedule(0)', 'visibilitychange", () => { schedule(0); }')


def test_a_failure_inside_the_sink_cannot_take_the_route_down():
    """Playing as it is made is an enhancement over a route that already works.
    A throw in the audio path stops the sound and lets the buffered path finish
    and play the file."""
    sink = code(body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND"))
    commit = sink[sink.index("commit(bytes)"):sink.index("close()")]
    assert "try {" in commit and "catch" in commit
    assert "failed = true;" in commit
    assert "stopScheduled();" in commit


def test_a_browser_with_no_audio_context_is_never_promised_anything():
    """Decided BEFORE the request is built, so response_format is never forced
    to pcm on a browser that cannot decode it, and the format the reader chose
    is honoured. No message, because nothing was promised."""
    speak = code(body_of("async function speakNow(voice, text)", "async function collectSSE"))
    assert "const ctx = plan && plan.mode === \"audio\" ? openAudio() : null;" in speak
    opener = code(body_of("function openAudio()", "\n/* 44 bytes of RIFF"))
    assert "return null" in opener
    # And it is resumed inside the gesture, before the first await: a context
    # made outside one starts suspended in Chrome and Safari, and the first
    # delta then plays into silence while the arithmetic says all is well.
    assert "speakCtx.resume()" in opener
    assert speak.index("openAudio()") < speak.index("await api(")


def test_the_odd_byte_survives_a_proxy_that_splits_a_frame():
    """For pcm a delta is a whole segment and therefore an even number of
    bytes, but a proxy is free to split a frame, and half a sample
    desynchronises every sample after it."""
    sink = code(body_of("function audioSink(ctx, plan)", "\n/*\n  THE SINK THAT MAKES NO SOUND"))
    assert "bytes.length & 1" in sink
    assert "carry = bytes.slice(-1)" in sink
    # DataView and not Int16Array: a subarray can start on an odd byte offset.
    assert "new DataView(bytes.buffer" in sink
    assert "new Int16Array(" not in sink


def test_the_page_makes_one_audio_context_and_not_one_per_press():
    """Chrome allows about six AudioContexts per document and refuses the
    seventh, so a context made per run turns the seventh Speak of a session
    into a silent failure with the headroom arithmetic still reporting that
    everything is fine."""
    opener = code(body_of("function openAudio()", "\n/* 44 bytes of RIFF"))
    assert "if (!speakCtx || speakCtx.state === \"closed\")" in opener
    assert "if (speakCtx.resume) speakCtx.resume();" in opener
