"""The page must not execute a title chosen by whoever owns the media.

These are static and parser-based rather than browser tests: the assertion is
about what BYTES reach innerHTML, which is decidable without a DOM, and the one
thing a real browser would add here is confirmation that html.parser and Blink
agree about `<`.

The hole was real. `$("picked").innerHTML = ...${facts.title}...` took the
title of the page behind a pasted link -- yt-dlp's probe or MeTube's record --
and interpolated it raw. The CSP this page sets is script-src 'self'
'unsafe-inline', because the page's own script block is inline, so an injected
event handler EXECUTES; connect-src 'self' stops an XHR exfiltrating, but not a
navigation or a form POST, and the API key sits in localStorage. It needs only
that the user paste a link to media somebody else named, which is the ordinary
use of that box.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "ui.html"
HTML = PAGE.read_text()

PAYLOAD = '<img src=x onerror="fetch(\'/x\'+localStorage.key)">'


def esc(value: str) -> str:
    """The page's esc(), transcribed. Kept in step by test_the_page_escaper_matches."""
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;"))


class Sniff(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.handlers: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.handlers += [k for k, _ in attrs if k.startswith("on")]


def parse(markup: str) -> Sniff:
    sniff = Sniff()
    sniff.feed(markup)
    return sniff


def test_the_page_escaper_matches_the_one_these_tests_assert_with():
    """If ui.html's esc() is weakened, this file must stop agreeing with it."""
    found = re.search(r"const esc = \(s\) =>(.*?);\n", HTML, re.S)
    assert found, "ui.html has no esc() helper"
    for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&#39;"):
        assert entity in found.group(1), f"esc() no longer emits {entity}"


def test_a_hostile_media_title_renders_as_text_not_as_an_element():
    rendered = f'<span class="filechip">{esc(PAYLOAD)}<span class="hint">x</span></span>'
    sniff = parse(rendered)
    assert "img" not in sniff.tags, "remote data created an element"
    assert sniff.handlers == [], f"an event handler survived: {sniff.handlers}"


def test_the_same_payload_unescaped_really_does_execute():
    """Proves the test above can fail, rather than passing vacuously."""
    sniff = parse(f"<span>{PAYLOAD}</span>")
    assert "img" in sniff.tags
    assert "onerror" in sniff.handlers


@pytest.mark.parametrize("expression", [
    "facts.title",    # remote: whoever owns the media chooses it
    "file.name",      # local, but still not this file's text
    "err.message",    # server-derived
    "job.voice",      # a clone name, which came from an uploaded clip
    "job.id",
    "job.error",
    "c.name",
])
def test_no_sink_interpolates_this_without_escaping_it(expression):
    """A new template literal must not become the next unescaped sink."""
    raw = "${" + expression + "}"
    offenders = [n for n, line in enumerate(HTML.splitlines(), 1) if raw in line]
    assert not offenders, (
        f"{raw} reaches innerHTML unescaped at line(s) {offenders}; "
        f"wrap it in esc()")


def test_attribute_context_is_escaped_too():
    """Escaping < and & alone is not enough where the value lands in an attribute.

    A segment's text goes into value="...", and a job id into data-get="...".
    Without quote escaping a value can close its own attribute and open another.
    """
    hostile = '" onmouseover="alert(1)'
    sniff = parse(f'<input type="text" value="{esc(hostile)}">')
    assert sniff.handlers == [], "a value broke out of its attribute"


# --------------------------------------------------------- paste handling --
#
# Not escaping, but the same kind of check: a property of ui.html that is
# decidable from the source and was got wrong once already.


def _paste_handler() -> str:
    """The body of the document-level paste listener."""
    start = HTML.index('document.addEventListener("paste"')
    return HTML[start:start + 1800]


def test_a_paste_into_the_url_box_does_not_also_assign_the_value():
    """It used to do both, and the browser pasted the text twice.

    The handler assigned `$("url").value = text` and returned without
    preventDefault(), so the default paste then inserted the same string again
    at the caret. A pasted Instagram link became
    ".../reel/DCH9NdDpXis/https://www.instagram.com/reel/DCH9NdDpXis/", which
    MeTube accepted and then had no record of -- an error message about a URL
    the user never typed.
    """
    body = _paste_handler()
    assign = body.index('$("url").value = text')
    prevent = body.index("ev.preventDefault()")
    assert prevent < assign, (
        "the value is assigned without the default paste being cancelled first")


def test_the_url_box_path_lets_the_browser_do_the_paste():
    body = _paste_handler()
    assert "intoUrlBox" in body, "no branch for a paste landing in the url box"
    branch = body[body.index("if (intoUrlBox)"):]
    head = branch[:branch.index("return;")]
    assert '$("url").value =' not in head, (
        "the url-box branch must not set the value; the browser is doing it")


def test_pasting_a_link_does_not_resolve_it():
    """It used to. resolveLink spawns yt-dlp against a host the CLIPBOARD
    chose, before the user had clicked anything -- and on a paste they may
    have been aiming at another field entirely. Resolve is one click away and
    is the button that says so.
    """
    body = _paste_handler()
    code = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    assert "resolveLink()" not in code, \
        "pasting must fill the box and stop"


def test_a_paste_into_another_field_is_left_alone():
    """This is a document-level listener, so it sees every paste on the tab.

    A link pasted into the glossary box or an expert field was being swallowed
    and resolved as if it had been meant for the url box.
    """
    body = _paste_handler()
    assert "intoSomeOtherField" in body
    assert 'target.tagName === "TEXTAREA"' in body
    assert "isContentEditable" in body


# ------------------------------------------------------- the voice picker --


def _load_voices() -> str:
    start = HTML.index("async function loadVoices()")
    return HTML[start:HTML.index("function currentVoice()")]


def test_chatterbox_appears_even_with_nothing_cloned():
    """It used to be built only `if (clones.length)`.

    On a fresh deployment the second engine was absent from the picker
    entirely, and the only sign it existed was "+ Use my own voice…", which
    reads like an upload button rather than a model. tts-long has shipped its
    own speaker all along -- voices.py BUILTIN = "default", no clip needed --
    and POST /jobs with voice:"default" has always worked.
    """
    body = _load_voices()
    group = body[body.index("Chatterbox, slow"):]
    head = group[:group.index("</optgroup>")]
    assert 'value="c:default"' in head, "the built-in voice is not offered"
    # Comments are stripped first: this file's own comment explains the old
    # `if (clones.length)` behaviour, and matching prose would make the test
    # assert against its own documentation rather than against the code.
    code = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "if (clones.length)" not in code, (
        "the Chatterbox group is still conditional on having a clone")


def test_both_engines_are_named_in_the_picker():
    """Labels said "Instant" and "Cloned", naming the experience not the model.

    The two expert panels below ARE named after the engines, so they appeared
    to swap for no visible reason.
    """
    body = _load_voices()
    assert "Kokoro, instant" in body
    assert "Chatterbox, slow" in body


def test_the_builtin_voice_offers_no_delete_button():
    """There is no file behind it; the delete would 404 on the one voice the
    user cannot break."""
    start = HTML.index("function onVoiceChange()")
    body = HTML[start:HTML.index('$("voice").addEventListener')]
    assert 'voice.name === "default"' in body
    assert '$("delvoice").hidden = !clone || builtin;' in body


def test_the_engine_is_named_where_the_choice_is_made():
    """The running hint under the picker is gone as interface clutter, so the
    optgroup labels are now the only place the engine is named. They were
    always the better place: the cost is visible before the click rather than
    after it."""
    body = _load_voices()
    assert "Kokoro, instant" in body
    assert "Chatterbox, slow" in body


# ---------------------------------------------- language before voice --


def test_language_comes_before_voice_in_the_markup():
    """You know what language you want before you know which of 54 voices.

    Asking in the other order means scrolling past every Portuguese voice to
    reach an English one.
    """
    assert HTML.index('for="lang"') < HTML.index('for="voice"')


def test_the_language_list_no_longer_depends_on_the_voice():
    """It used to be built FROM the chosen voice's engine, so it was rebuilt on
    every voice change -- which reset a deliberate choice back to auto whenever
    the engine changed, and rebuilt a <select> the user might have open."""
    start = HTML.index("function populateLanguages()")
    body = HTML[start:HTML.index("function voicesForLanguage")]
    assert "currentVoice()" not in body, "the language list still reads the voice"
    assert "KOKORO_LANGS" in body and "CHATTERBOX_LANGS" in body, \
        "the list should be the union of both engines"

    change = HTML.index("function onVoiceChange()")
    tail = HTML[change:HTML.index('$("voice").addEventListener')]
    code = re.sub(r"/\*.*?\*/", "", tail, flags=re.S)
    assert "populateLanguages()" not in code, \
        "onVoiceChange must not rebuild a list that no longer depends on it"


def test_choosing_a_language_rebuilds_the_voice_list():
    handler = HTML[HTML.index('$("lang").addEventListener'):]
    assert "loadVoices" in handler[:200], \
        "a language change must re-filter the voices below it"


def test_chatterbox_voices_are_not_filtered_by_language():
    """Its voices are not per-language the way Kokoro's are: the language is a
    parameter it takes, so `default` and every clone can read any of them.
    Filtering them out would hide voices that can do the job."""
    start = HTML.index("async function loadVoices()")
    body = HTML[start:HTML.index("function currentVoice()")]
    chatterbox = body[body.index("Chatterbox, slow"):]
    assert "forLang(" not in chatterbox[:400]


def test_populate_languages_runs_before_the_first_voice_load():
    """The voice list is filtered by $("lang"), so the options must exist
    before the first filter runs.

    THE ANCHORS CHANGED WITH THE CONCURRENT BOOT, and the property did not.
    This used to slice from the literal `await poll();` and compare against
    `await loadVoices()`. boot() no longer awaits either: poll() is started and
    left to land, and loadVoices runs inside a Promise.all with the glossaries
    and the job listing, because awaiting a cold /ui/health put a 2,517 ms
    status read in front of the voice picker. The ordering rule that matters is
    unchanged and is what is asserted: populateLanguages is synchronous and has
    to have run before anything can list a voice.
    """
    boot = HTML[HTML.index("(async function boot()"):]
    assert boot.index("populateLanguages()") < boot.index("loadVoices()")
    # And it is still not awaited into the group, which would reorder it.
    assert "await Promise.all([loadGlossaries(), loadVoices()," in boot


# --------------------------------------------------------- auto-detect --


def _stopwords() -> dict[str, set[str]]:
    """The lists as they exist in the page, parsed rather than duplicated."""
    start = HTML.index("const STOPWORDS = {")
    body = HTML[start:HTML.index("};", start)]
    out = {}
    for code, words in re.findall(r'(\w+):"([^"]*)"', body):
        out[code] = set(words.split())
    return out


def test_an_ordinary_english_sentence_reaches_the_two_hit_threshold():
    """It did not, and auto-detect said "not enough text to tell yet".

    Each list held sixteen words, and detectLanguage needs two hits to commit.
    "hello there, are u fine?" scored ONE -- only `are` was in the English list.
    The gate was never the length of the text; it was the size of the list.
    """
    stop = _stopwords()
    for sentence in ("hello there are u fine",
                     "how are you doing today my friend",
                     "can you tell me what time it is"):
        hits = sum(1 for w in sentence.split() if w in stop["en"])
        assert hits >= 2, f"{sentence!r} scores {hits}, needs 2"


def test_portuguese_still_wins_on_portuguese():
    stop = _stopwords()
    for sentence in ("oi tudo bem com você hoje",
                     "eu preciso que você faça isso agora"):
        pt = sum(1 for w in sentence.split() if w in stop["pt"])
        en = sum(1 for w in sentence.split() if w in stop["en"])
        assert pt >= 2 and pt > en, f"{sentence!r}: pt={pt} en={en}"


def test_no_word_appears_in_both_english_and_portuguese():
    """A shared word adds a hit to both and moves nothing, and the near-tie
    rule then refuses to answer at all -- so it is worse than no word. These
    two are the pair that matters here: the user code-switches between them."""
    stop = _stopwords()
    shared = stop["en"] & stop["pt"]
    assert not shared, f"shared between en and pt: {sorted(shared)}"


def test_the_scandinavian_lists_stay_indistinguishable_on_purpose():
    """Danish, Norwegian and Swedish share most function words. The tie rule
    returning null for a short sample is the correct answer, not a gap."""
    stop = _stopwords()
    assert len(stop["da"] & stop["no"]) > 5, \
        "these were deliberately similar; a rewrite that separated them is " \
        "claiming a distinction a short sample cannot support"


# ------------------------------------------ language without the region --


def test_the_language_list_carries_no_region():
    """It offered "English (US)", "English (UK)" and "Portuguese (Brazil)".

    That put the accent in the wrong control: the region is a property of the
    VOICE, and Kokoro already encodes it in the name (a->en-us, b->en-gb,
    p->pt-br). Asking twice also made "English (US)" with a British voice a
    state the page allowed and could not honour.
    """
    body = HTML[HTML.index("function populateLanguages()"):
                HTML.index("function voicesForLanguage")]
    assert "name.replace(" in body, "the region is not being stripped"
    assert r"\([^)]*\)" in body, "the trailing parenthetical is not matched"
    # And the source lists it strips from still carry the regions, so the test
    # is asserting a transformation rather than an already-clean input.
    assert '"English (US)"' in HTML and '"Portuguese (Brazil)"' in HTML


def test_a_chosen_language_goes_through_toenginelang():
    """The select holds ISO stems now, so sending the value verbatim would put
    "en" on the wire where Kokoro wants "en-gb" -- and "en-us" where Chatterbox
    wants "en", which is the 400 a user actually hit:

        unsupported language 'en-us'; chatterbox has ar, da, de, el, en, ...
    """
    body = HTML[HTML.index("function resolvedLanguage()"):]
    body = body[:body.index("\n}\n")]
    chosen = body[body.index('chosen !== "auto"'):]
    assert "toEngineLang(chosen" in chosen, \
        "a hand-picked language must resolve the same way auto-detect does"


def test_the_two_engines_disagree_about_spelling_and_that_is_handled():
    """Kokoro wants pt-br, en-us and cmn; Chatterbox wants pt, en and zh. One
    control, two vocabularies, and toEngineLang is the only translator."""
    fn = HTML[HTML.index("function toEngineLang"):]
    fn = fn[:fn.index("\n}\n")]
    assert 'kind === "clone"' in fn, "the clone path must not take Kokoro codes"
    assert "pt:\"pt-br\"" in fn.replace(" ", "") or 'pt:"pt-br"' in fn


def test_the_page_and_the_backend_agree_on_chatterbox_languages():
    """The page's list is a copy of services/tts-long's, which is itself a copy
    of the model's, cross-checked at load. A third copy drifting means a 400
    for a language the model would have accepted."""
    page = set(re.findall(r'\["(\w+)","[^"]+"\]',
                          HTML[HTML.index("const CHATTERBOX_LANGS"):
                               HTML.index("];", HTML.index("const CHATTERBOX_LANGS"))]))
    synth = (Path(__file__).resolve().parents[3]
             / "services" / "tts-long" / "app" / "synth.py").read_text()
    block = synth[synth.index("SUPPORTED_LANGUAGES = ("):]
    backend = set(re.findall(r'"(\w+)"', block[:block.index(")")]))
    assert page == backend, f"page-only {page - backend}, backend-only {backend - page}"


# ------------------------------------------------- the microphone, twice --


def test_transcribe_has_its_own_recorder():
    """The clone sheet's is a FIFTEEN SECOND one that hands its result to
    prepareClip -- a reference clip, capped at the clone ceiling and resampled
    to 24 kHz. Dictation is a different job with a different length, so sharing
    it would have meant a flag deciding which of two behaviours applied."""
    assert 'id="sttrec"' in HTML
    assert "sttRecorder" in HTML
    body = HTML[HTML.index("$(\"sttrec\").addEventListener"):]
    body = body[:body.index('$("sttrecstop")')]
    assert "prepareClip" not in body, "that is the clone path, not this one"
    assert "await pick(" in body, \
        "a recording should become a file like any other"


def test_the_dictation_recorder_has_no_countdown():
    """There is no natural length for "say the thing you wanted transcribed",
    and cutting someone off mid-sentence is worse than a long file. The only
    stop is the ceiling the upload has anyway."""
    body = HTML[HTML.index("$(\"sttrec\").addEventListener"):]
    body = body[:body.index('$("sttrecstop")')]
    assert "secs > 3600" in body
    assert "left -= 0.1" not in body, "that is the clone sheet's countdown"


# ------------------------------------------ uploads the browser cannot read --


def test_an_undecodable_upload_is_sent_as_it_arrives():
    """decodeAudioData throws on some containers and, worse, SUCCEEDS
    uselessly on others: a WhatsApp voice note is ogg/opus with no duration in
    its container and came back as ONE SECOND of audio with no error -- a dead
    reference clip that looked like a successful upload."""
    body = HTML[HTML.index("async function prepareClip"):]
    body = body[:body.index("\n}\n")]
    assert "tooShortToBeReal" in body
    assert "clipSuffix" in body, "the original format has to reach the server"


def test_the_upload_carries_its_real_extension():
    """clips.save reads the suffix to decide what it is storing, so opus bytes
    named .wav would put a file in the voice directory that says one thing and
    is another."""
    assert 'name + clipSuffix' in HTML
    assert 'name + ".wav"' not in HTML


def test_forget_deletes_on_the_server_not_just_in_the_page():
    """It removed the job from this page's map and localStorage and stopped
    there, so the next poll listed it again three seconds later. The button had
    done exactly what it said and nothing anyone wanted.

    It matters more since tts-long started recovering finished jobs from
    /output at startup: a forgotten job came back from disk on every restart,
    permanently.
    """
    body = HTML[HTML.index("async function forgetJob(id)"):]
    body = body[:body.index("\n}\n")]
    assert 'method: "DELETE"' in body, "forget must reach the server"
    assert "jobs.delete(id)" in body, "and still clear the local copy"
    order = body.index('method: "DELETE"') < body.index("jobs.delete(id)")
    assert order, "delete on the server before forgetting locally"


def test_forget_treats_a_404_as_success():
    """The service not having it either -- swept, or lost to a restart this
    page still remembers -- is the case where local cleanup IS the whole job.
    Alerting there would make the button look broken for the one state it
    handles perfectly."""
    body = HTML[HTML.index("async function forgetJob(id)"):]
    body = body[:body.index("\n}\n")]
    assert "err.status !== 404" in body



def test_the_button_that_deletes_the_audio_says_so():
    """It said "Forget", sitting next to "Download the audio".

    That reads as "take this row off my list", and it is a DELETE that removes
    the file from the server -- so it goes from every other device too, and
    nothing in this stack can bring back forty minutes of CPU. The label now
    names the consequence wherever there is one. A row with nothing left to
    lose -- failed, or expired, or already swept -- really is only forgotten,
    and is still called Forget.
    """
    row = HTML[HTML.index("$(\"joblist\").innerHTML = list.map"):]
    row = row[:row.index("}).join(\"\")")]
    # NOW TWO BUTTONS, because they are two different acts. Deleting used to
    # take the audio AND the record together, so freeing a gigabyte also threw
    # away what was said, which voice said it and which machine did the work --
    # a few hundred bytes that are the only evidence any of it happened.
    #
    # The rule the old assertion protected is unchanged and now applies twice:
    # every label names what it destroys, and neither of them says a bare
    # "Forget" beside a Download button.
    assert ">Delete the audio</button>" in row
    assert 'audio ? "Delete the record too" : "Delete the record"' in row
    assert ">Forget</button>" not in row, "a label that hides the consequence is back"
    # The audio button only exists while there is audio to delete.
    assert 'audio ? `<button class="small" data-delaudio=' in row


def test_one_predicate_decides_whether_a_row_still_has_audio():
    """Play, Download, the destructive label and the confirmation all have to
    agree about it. Written out twice, the two copies drift -- and a button
    that offers to download a file that is gone is the same defect as one that
    silently deletes a file it never mentioned."""
    row = HTML[HTML.index("$(\"joblist\").innerHTML = list.map"):]
    row = row[:row.index("}).join(\"\")")]
    assert "const audio = hasAudio(job);" in row
    assert '(job.status === "done" || job.status === "cancelled") && !expired' \
        not in row, "the predicate is spelled out again instead of shared"
    assert "function hasAudio(job)" in HTML


def test_forget_asks_first_while_there_is_still_audio_to_lose():
    """One press deleted the recording, from the server, with no undo anywhere.
    The dialog is the second half of the relabel: a mistaken press costs a
    click, not the audio. It has to come BEFORE the DELETE, and it has to say
    that the file goes from the server rather than from this page."""
    body = HTML[HTML.index("async function forgetJob(id)"):]
    body = body[:body.index("\n}\n")]
    stripped = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "confirm(" in stripped
    assert stripped.index("confirm(") < stripped.index('method: "DELETE"'), \
        "it asks after it has already deleted the file"
    assert "server" in stripped, "the dialog must say where the file goes from"


def test_a_row_with_no_audio_left_is_not_worth_a_dialog():
    """A failed, expired or already-swept job has nothing to lose, and a
    confirmation for a free action teaches people to click through the one that
    is not free."""
    body = HTML[HTML.index("async function forgetJob(id)"):]
    body = body[:body.index("\n}\n")]
    stripped = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "hasAudio(job)" in stripped, "the dialog is not gated on anything"


# --------------------------------------------------- telling jobs apart --


def test_a_recovered_job_does_not_claim_the_default_voice():
    """`default` is a REAL voice on this service -- Chatterbox's own speaker.

    A recovered job's voice is unknown, not default: the record is rebuilt from
    a filename and the voice lived only in the dict that the restart cleared.
    Printing `job.voice || "default"` asserted a fact nobody knew, and a list
    of jobs made with different cloned voices all read "default".
    """
    row = HTML[HTML.index("$(\"joblist\").innerHTML = list.map"):]
    row = row[:row.index("}).join(\"\")")]
    assert 'job.recovered ? "voice unknown" : "default"' in row


def test_the_full_text_is_fetched_on_open_not_with_the_listing():
    """`text` is capped at 4096 characters and the listing holds fifty jobs, so
    sending it with every poll would be a couple of hundred kilobytes every few
    seconds for something nobody is reading."""
    body = HTML[HTML.index('details.jobtext"'):]
    assert '"toggle"' in body[:900], "the text should load when a row opens"
    assert "body.dataset.loaded" in body[:1400], "and be fetched once, not per tick"


def test_the_job_text_is_rendered_as_text():
    """It is the largest piece of user-supplied data on this tab."""
    start = HTML.index('d.addEventListener("toggle"')
    section = HTML[start:HTML.index("}));", start)]
    assert "body.textContent = said" in section
    # Comments stripped first: the line above this one in ui.html says
    # "textContent, not innerHTML", and matching prose would make the test
    # assert against its own documentation. Same trap as the language test.
    code = re.sub(r"/\*.*?\*/", "", section, flags=re.S)
    assert "innerHTML" not in code, \
        "the job text must never be assigned as markup"


def test_an_open_row_survives_the_two_second_re_render():
    """renderJobs reassigns #joblist wholesale on a timer. Without this, a row
    someone had opened to read closed itself a moment later."""
    assert "openRows" in HTML
    assert 'openRows.has(job.id) ? " open" : ""' in HTML


# ------------------------------------ a control does what its label says --


def test_resolve_only_resolves():
    """It called startFetch() whenever the server said the media was short
    enough not to warrant a dialog, so a button labelled "Resolve" downloaded
    the file and transcribed it -- and finding out what a link WAS cost you the
    transcription. "Do not nag" is a reason to skip the modal, never a reason
    to do the work unasked.
    """
    body = HTML[HTML.index("async function resolveLink"):]
    body = body[:body.index("\n}\n")]
    code = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    assert "startFetch(" not in code, \
        "resolve must not start the download itself"
    assert "showConfirm(facts)" in code, "it should offer the choice instead"


def test_the_button_stops_saying_transcribe_once_the_subtitles_were_taken():
    """Taking a link's existing subtitles left the button reading "Transcribe",
    and pressing it re-read the subtitle file.

    The ACTION was already honest -- a captions commit goes to /ui/captions and
    nothing on this page transcribes anything -- but the label was not, and it
    is the label that is read. Someone who saw "Transcribe" had every reason to
    believe the audio was about to go through Parakeet: minutes of compute they
    did not ask for, producing a second and worse transcript of something a
    person had already written.

    The press is kept, but not for the reason it used to be. renderCaptions
    wrote the cues into whichever of Text, SRT or VTT was selected, so a second
    press was how the same subtitles came back in another format; all three
    come off one press now. What a second press is for is getting the subtitles
    back into the pane after something else has replaced them, without
    re-resolving the link -- still two seconds and no compute.
    """
    body = _fn("function syncActionLabel()")
    assert '"Read the subtitles"' in body
    assert '"Transcribe"' in body
    assert "stt.captions" in body, "the label has to follow the mode"


def test_every_path_that_changes_the_mode_changes_the_label():
    """Four places set stt.captions -- a file, a link, abandoning one, and the
    commit itself. One of them forgetting to relabel is exactly the bug above,
    coming back in a state nobody tests by hand."""
    lines = HTML.splitlines()
    for number, line in enumerate(lines):
        if not re.search(r"\bstt\.captions\s*=(?!=)", line):
            continue
        window = "\n".join(lines[number:number + 3])
        assert "syncActionLabel()" in window, (
            f"line {number + 1} changes the mode and not the label: {line.strip()}")


def test_a_hidden_segments_editor_does_not_apply():
    """It lives in the Kokoro expert panel, which onVoiceChange hides when a
    Chatterbox voice is chosen -- but segments() read the rows regardless, so
    segments added on the fast path were sent to a clone job from a panel the
    user could not see and had no way to clear.
    """
    body = HTML[HTML.index("function segments()"):]
    body = body[:body.index("\n}\n")]
    assert '$("tts-expert-fast").hidden' in body


def test_visibility_is_read_rather_than_the_voice():
    """Whatever decides which panel is shown stays the single source of truth,
    so this cannot drift from it."""
    body = HTML[HTML.index("function segments()"):]
    body = body[:body.index("\n}\n")]
    assert "currentVoice()" not in body


# ------------------------------------------------------------------------
# THE SECOND PASS OVER THE SAME QUESTION. Three controls above were found to
# act without being asked, or to be read where they could not be seen. These
# are the rest of that sweep: every interactive element on the three tabs and
# the clone sheet was walked against the code that builds each request, and
# each test below is named after the thing the user saw.


def _fn(name: str) -> str:
    """A top-level function's body, comments and all."""
    start = HTML.index(name)
    return HTML[start:HTML.index("\n}\n", start)]


def _code(text: str) -> str:
    """The same, with the comments taken out -- this file's comments quote the
    very code they are about, so a naive `in` check passes on the prose."""
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", "", text, flags=re.S))


def test_a_dropped_link_is_not_resolved_by_the_drop():
    """Dropping a FILE prepared it and waited for Transcribe; dropping a link
    spawned yt-dlp against whatever host the drag came from. The same gesture
    on the same target meant "get ready" for one payload and "go" for the
    other, which is the paste bug wearing a different hat.
    """
    start = HTML.index('$("drop").addEventListener("drop"')
    body = _code(HTML[start:HTML.index("});", start)])
    assert "resolveLink()" not in body, "a drop must fill the box and stop"
    assert '$("url").value = text' in body


def test_a_start_offset_from_a_hidden_row_does_not_trim_an_upload():
    """#clipstart lives in #cliprange, which is hidden until a link resolves.
    prepareClip read it anyway, so resolving a link, setting "start at 90" and
    then changing your mind and uploading a file instead trimmed that file from
    ninety seconds in, from a control that was no longer on the screen.
    """
    body = _fn("async function prepareClip(")
    assert '$("cliprange").hidden ? 0 :' in body


def test_the_native_route_greys_out_the_fields_it_cannot_carry():
    """/transcribe takes a file and nothing else -- no response_format, no
    timestamp_granularities, no include[], no chunking_strategy -- but every
    one of those controls stayed live in the panel beside it, and the request
    builder simply never looked at them.
    """
    body = _code(_fn("function syncTranscribeControls()"))
    assert '$("x-rf").disabled = native;' in body
    # native only: the link path carries granularities now, and greying a
    # control out for a reason that has stopped being true is the very thing
    # syncTranscribeControls exists to prevent.
    assert '$("x-gran").disabled = native;' in body
    assert '$("x-logprobs").disabled = !form;' in body
    assert 'x-vad-t' in body and 'disabled = !form' in body


def test_a_link_greys_out_what_ui_fetch_cannot_carry():
    """/ui/fetch streams MeTube's file into /v1 with `model` and
    `response_format` and no other field (ingest.py), so on that path the route
    select and everything under it was decoration.
    """
    body = _code(_fn("function syncTranscribeControls()"))
    assert "const link = !!stt.token && !stt.file;" in body
    assert '$("x-route").disabled = link;' in body


def test_a_disabled_response_format_is_not_a_choice():
    """It is greyed out on the native route, and a value left in a control the
    user can no longer reach must not go on deciding which route is offered.

    WHAT THIS ASSERTS NOW. The rule is the same and it has one reader instead
    of two. It used to live in chosenFormat(), which the retired Transcript
    format control also fed; formatForTimings() read `$("x-rf").value` raw and
    so skipped the check entirely, which is the same defect in a second copy.
    Both go through expertFormat() now, so a greyed field cannot mean one thing
    to the code that picks the route and another to the code that builds the
    request.
    """
    assert '$("x-rf").disabled ? "" : $("x-rf").value' in _fn("function expertFormat()")
    for reader in ("function formatForTimings()", "function granularities(format)",
                   "function updateRouteAvailability()"):
        assert '$("x-rf").value' not in _code(_fn(reader)), \
            f"{reader} reads the field raw and can act on a greyed choice"


def test_the_native_route_is_not_offered_for_subtitles():
    """It answers its own JSON shape with no cue timings in it, so choosing SRT
    and then the native route quietly handed back a .txt.

    WHAT THIS ASSERTS NOW. The refusal survived the removal of the Transcript
    format control rather than going with it. It used to read that control;
    the conflict it prevents is still reachable through the one setting that
    can still ask for a subtitle file, so it hangs off the expert
    response_format instead of being deleted as unreachable.
    """
    body = _code(_fn("function updateRouteAvailability()"))
    assert "const expert = expertFormat();" in body
    assert 'const subtitles = expert === "srt" || expert === "vtt";' in body
    assert "const ok = sixteen && !subtitles;" in body
    assert "cannot write subtitles" in body


def test_the_route_that_has_no_timings_says_it_gives_up_the_sidecar():
    """The quieter half of the same fact, and the half that has no refusal to
    carry it.

    A native run has no highlight and no .srt or .vtt, because /transcribe
    carries no timings at all. That was one of three ways to end up without
    subtitles while a format control existed. It is the only one left now that
    every /v1 run offers them, so the hint under the route select has to say
    so -- a route that silently drops two buttons the last run had is the page
    taking something away without a word.
    """
    hint = HTML[HTML.index('id="x-route-hint"'):]
    hint = hint[:hint.index("</div>")]
    assert "no timings" in hint
    assert ".srt" in hint and ".vtt" in hint


def test_stream_format_is_off_on_the_route_that_has_none():
    """It was read inside the /v1 branch only, so on /speak -- the DEFAULT
    route -- choosing SSE did nothing whatsoever.

    THE FORMAT CONTROL JOINED IT, for the same reason and by the same idiom. A
    run that will be played as it is made is forced to pcm: streamed wav writes
    a RIFF size of 8 and a data size of 0, and mp3, opus, aac and flac carry
    encoder priming and padding, so a delta boundary is not a segment boundary.
    A control the request cannot honour is greyed, never hidden.

    It is asked of speechPlan and not of the stream control alone, because a
    stream the arithmetic sends down the silent path DOES honour the format:
    nothing is decoded in the browser there.
    """
    body = _fn("function syncSpeakControls()")
    assert '$("x-stream").disabled = !v1;' in body
    assert '$("x-tfmt").disabled = v1 && $("x-stream").value === "sse"' in body
    assert 'speechPlan("kokoro", $("text").value).mode === "audio"' in body


def test_segments_are_off_on_the_route_that_has_none():
    """segments() ran on both branches and was attached on one, so rows typed
    into the visible editor were thrown away by /v1 along with their
    per-segment voice."""
    body = _code(_fn("function syncSpeakControls()"))
    assert '$("addseg").disabled = v1;' in body
    assert '$("segments").querySelectorAll' in body
    # And the request builder reads the same greying rather than the route.
    assert '$("addseg").disabled' in _fn("function segments()")


def test_the_language_control_is_off_where_it_is_not_sent():
    """/v1/audio/speech has no language field: tts's openai_api.py infers the
    phonemiser from the VOICE's first letter. So on that route the Language
    control -- which exists precisely so the voice would stop deciding the
    language -- was inert, and said nothing about it.
    """
    assert '$("lang").disabled = v1;' in _fn("function syncSpeakControls()")
    assert "no segments, no language" in HTML, \
        "the route option should name what it drops"


def test_the_speak_route_is_the_one_that_reads_the_segments():
    """Read inside the branch that sends them, so this cannot come back."""
    body = _code(_fn("async function speakNow("))
    assert "useV1 ? null : (segments() || autoSegments(text))" in body, \
        "segments must still be read only on the route that carries them"
    assert "language: resolvedLanguage().code" in body


def test_queue_it_anyway_queues_it():
    """It cleared the warning and stopped, so a button that said "Queue it
    anyway" was a Dismiss with someone else's label on it -- and the job it
    named still had to be started from the button the warning was covering.
    """
    start = HTML.index('$("anyway").addEventListener')
    body = _code(HTML[start:start + 600])
    # AND IT PRESSES THE BUTTON THAT IS ON SCREEN. #go-tts is hidden for a
    # cloned voice -- it offers to play a stream this backend never produces --
    # so clicking it would be the same do-nothing press under a new name.
    assert '$("go-tts-quiet").click();' in body


def test_the_quick_voice_offered_is_one_the_picker_actually_has():
    """nearestKokoro read the unfiltered list from /voices while the picker is
    filtered by the Language control, so "Use af_heart instead" could assign a
    value the <select> does not have -- which leaves it holding "" and sent
    voice:"" to /speak.
    """
    body = _code(_fn("function nearestKokoro()"))
    assert '$("voice").options' in body
    assert "VOICES.kokoro" not in body


def test_the_speak_button_is_off_when_it_would_do_nothing():
    """The click handler opened with `if (voice.kind === "new") return;` and
    `if (!text.trim() && !segments()) return;`, so pressing Speak with the
    clone sheet open, or with an empty box, produced no audio, no error and no
    sign that anything had been read at all.
    """
    body = _code(_fn("function estimate()"))
    assert 'voice.kind === "new"' in body
    assert "!(text.trim() || segments())" in body
    # Not re-enabled under a request that is still running.
    assert "speaking ||" in body


def test_the_code_switch_note_points_at_a_control_that_is_there():
    """It sends the reader to the segments editor, which is inside the Kokoro
    panel and greyed out on /v1 -- so for a cloned voice, and on that route, it
    was advice pointing at a control that is not on the screen."""
    body = _code(_fn("function codeSwitchNote(text)"))
    assert 'const editor = !$("tts-expert-fast").hidden && !$("addseg").disabled;' in body
    assert "pt && en && editor" in body


def test_cancelling_the_clone_sheet_drops_the_resolved_link():
    """/ui/resolve calls MeTube's /add before it probes, so every link resolved
    and then dropped leaves a pending record behind -- which is why the
    transcribe tab has abandon(). Cancel closed the sheet and left the token,
    the record and the filled-in box exactly where they were.
    """
    start = HTML.index('$("cancelclip").addEventListener')
    body = _code(HTML[start:HTML.index("});", start)])
    assert "forgetClipLink()" in body
    assert '$("cliplink").value = "";' in body
    assert '"/ui/abandon"' in _fn("async function forgetClipLink()")


def test_plain_text_is_split_into_sentences_so_there_is_something_to_follow():
    """A plain `text` request makes one piece of audio and therefore one
    offset, so the highlight covered the whole passage and karaoke on this tab
    did nothing unless you had opened the expert panel and built segments by
    hand. Which nobody does.
    """
    body = _code(_fn("function autoSegments("))
    assert "split(/(?<=[.!?" in body, "no sentence split"
    assert "out.length > 1 ? out : null" in body, \
        "one sentence is one highlight over everything -- worse than none"


def test_the_split_keeps_paragraphs():
    body = _code(_fn("function autoSegments("))
    assert 'split(/\\n\\s*\\n/)' in body


def test_the_split_inserts_no_pauses():
    """speak() already chunks at MAX_CHUNK_PHONEMES, so this moves a break from
    a token budget to a full stop. Adding silence would change the audio."""
    body = _code(_fn("function autoSegments("))
    assert "pause_after" not in body


# ------------------------------------------ vocabulary, not an expert setting --


def test_the_vocabulary_control_is_outside_the_expert_panel():
    """Which words a recording is likely to contain is the one thing the person
    who made it knows and nobody else does. Behind a disclosure triangle, the
    only control they could usefully set was where they would not find it."""
    box = HTML.index('id="glossbox"')
    panel = HTML.index('id="stt-expert"')
    assert box < panel, "the vocabulary control is inside or after the expert panel"


def test_the_vocabulary_row_still_says_what_it_is():
    """THE FAILURE A PROSE PASS MAKES: the element survives and the string that
    named it does not. #glossbox holds one <label> and a container the buttons
    are written into, so an emptied label leaves a row of bare profile names
    with nothing saying what choosing one does -- and every other assertion
    about this control goes on passing, because the control is still there.

    The row is also the only field left on its line, so there is no neighbour
    to borrow a heading from.
    """
    box = HTML.index('id="glossbox"')
    row = HTML[box:HTML.index('id="gloss"', box)]
    found = re.search(r"<label[^>]*>([^<]*)</label>", row)
    assert found, "the vocabulary row has no label at all"
    assert found.group(1).strip(), "the vocabulary label lost its text"


def test_nothing_is_selected_by_default():
    """Measured across 25 cells: a glossary whose terms do NOT occur in
    the audio raises WER by 28% on Whisper, and nothing measurable on Parakeet. An always-on
    list is the worst available shape."""
    body = _code(_fn("async function loadGlossaries("))
    assert 'aria-pressed="false"' in body
    assert 'aria-pressed="true"' not in body


def test_an_empty_selection_sends_no_field_at_all():
    """Per ADR 0001 an extension is absent by default -- not sent empty, which
    is a value the service would have to interpret."""
    for fn in ("async function transcribeToken(", "async function transcribeFile("):
        try:
            body = _code(_fn(fn))
        except ValueError:
            continue
        if "chosenGlossaries()" not in body:
            continue
        assert "chosen.length" in body or "nativeChosen.length" in body, \
            f"{fn} sends the field unconditionally"


def test_the_control_is_hidden_when_the_service_offers_none():
    """A control with nothing in it is a question the page cannot answer."""
    body = _code(_fn("async function loadGlossaries("))
    assert '$("glossbox").hidden = !GLOSSARIES.length' in body


def test_the_page_can_reach_the_glossary_listing():
    """It is a gateway route, so a bare relative call would land on the gateway
    and skip the UI's key. /ui/api is what brings it back through."""
    assert "glossaries" in HTML[HTML.index("const GATEWAY_ROUTES"):
                                HTML.index("const GATEWAY_ROUTES") + 200]


# ------------------------------------------------ managing the profiles --
#
# The page could list the profile names and nothing else: no way to see what a
# profile contains, and no way to add, change or remove one. That work went
# through curl, so the terms a transcript depends on were invisible to the
# person depending on them.
#
# Every test here is named after a refusal the panel has to handle honestly.
# The service is the authority on all of them; what these pin is that the page
# does not offer a write it can already see will fail, and does not swallow one
# it could not.


def _guards() -> str:
    return _code(_fn("function glossGuards("))


def test_a_builtin_is_never_offered_an_edit_that_would_409():
    """`dictation` and `tech` ship in the image and PUT on either is a 409.

    Offering a Save that is certain to be refused is the same defect as a
    control the current route cannot carry staying live: the page knows, from
    `writable` in the listing it already has, and says so instead. Greyed and
    not hidden, because a control that vanishes takes its reason with it.
    """
    body = _guards()
    assert "profile.writable === false" in body, "the listing's own flag is not read"
    assert 'is a built-in profile' in body, "nothing says why Save is off"
    assert '$("glosssave").disabled' in body
    assert '$("glossdelete").disabled' in body
    # Readable, because copying a built-in is the documented way past the 409,
    # and a disabled textarea cannot be selected.
    assert '$("glosstext").readOnly = builtin' in body


def test_an_unwritable_deployment_is_said_once_in_the_servers_own_words():
    """A deployment with nothing mounted answers 503 on every write, and the
    listing says so up front with a `reason` naming the directory and the fix.

    The page must not invent its own wording for that: the server knows which
    of four things is wrong (no directory configured, nothing mounted, not a
    directory, wrong uid) and this file knows none of them.
    """
    load = _code(_fn("async function loadGlossaries("))
    assert "GLOSS_WRITABLE = !!(listing && listing.writable)" in load
    assert "GLOSS_REASON = (listing && listing.reason)" in load
    body = _guards()
    assert "if (!GLOSS_WRITABLE) why = GLOSS_REASON" in body
    assert '$("glossnew").disabled = !GLOSS_WRITABLE' in body


def test_the_editor_round_trips_the_file_text_and_not_the_parsed_halves():
    """GET /glossaries/{name} returns `text` beside `replacements` and
    `hotwords` precisely so that GET, edit, PUT is a round trip.

    Rebuilding the file from the two parsed objects would drop every comment,
    and a glossary's comments are where it says why a rule is a hotword rather
    than a rewrite.
    """
    body = _code(_fn("async function glossOpen("))
    assert '$("glosstext").value = profile.text' in body


def test_the_two_glossary_routes_disagree_about_two_field_names():
    """`replacements` and `hotwords` are INTEGER COUNTS on GET /glossaries and
    the full object and array on GET /glossaries/{name}. Same keys, two types.

    One renderer over both is a "13" where a dictionary belongs, or the other
    way round, with nothing raising. The panel reads neither field from either
    route: the counts it shows are `terms`, which is an integer on both.
    """
    script = _code(HTML[HTML.index("<script>"):])
    for field in (".replacements", ".hotwords"):
        assert field not in script, \
            f"{field} is read from a route where its type is not what is assumed"


def test_deleting_a_profile_asks_before_the_file_goes():
    """It removes a file from the volume, and the terms with it. Same shape as
    the clone voice, which is the other destructive control on this page."""
    body = _code(_fn("async function glossDelete("))
    assert "confirm(" in body
    assert "${target.name}" in body, "the question does not name what it deletes"
    assert 'method: "DELETE"' in body


def test_a_write_refreshes_the_row_of_toggles_without_a_reload():
    """The Vocabulary row on the Transcribe tab is drawn from the listing. A
    profile created or deleted in the panel and not in that row is a control
    lying about what the service holds, until somebody reloads the page."""
    for name in ("async function glossSave(", "async function glossDelete("):
        assert "await loadGlossaries();" in _code(_fn(name)), \
            f"{name} leaves the selector stale"


def test_a_ticked_profile_survives_the_refresh_a_write_causes():
    """loadGlossaries() rebuilds the row by assigning innerHTML, so calling it
    after a save wiped a selection the reader had made before they ever opened
    the panel -- silently, on a control whose whole subject is which terms this
    recording contains.

    THIS DOES NOT MAKE ANYTHING A DEFAULT, which is what
    test_nothing_is_selected_by_default is for: a first load reads an empty row
    and restores nothing.
    """
    body = _code(_fn("async function loadGlossaries("))
    assert "const kept = chosenGlossaries();" in body
    assert 'b.setAttribute("aria-pressed", "true")' in body
    assert "kept.indexOf(b.dataset.gloss) >= 0" in body


def test_force_is_offered_only_for_the_refusal_it_can_fix():
    """`force` switches off the single-word left-hand-side rule and nothing
    else, so a body that also has an empty left-hand side or an over-long line
    in it is refused again with the flag set.

    The service marks the forceable refusals itself, in the reason it prints,
    and reading that is more honest than this file keeping a second list of
    which refusals are which and being wrong on the day they diverge.
    """
    body = _code(_fn("function glossRefused("))
    assert 'indexOf("send force")' in body
    assert "lines.every(" in body, "one forceable line must not offer force for all of them"
    assert '$("glossforce").hidden = !forceable' in body


def test_a_refused_body_is_not_cleared_from_the_editor():
    """The service writes NOTHING when any line is refused, so the text in the
    editor is the only copy of it left. A panel that cleared on failure would
    delete the work the failure was about."""
    body = _code(_fn("async function glossSave("))
    assert '$("glosstext").value = ""' not in body


def test_the_refused_lines_reach_the_reader_at_all():
    """`detail` is a STRING for every other failure in this stack and an OBJECT
    for this one. reason() read the two known shapes, returned the fallback for
    the third, and the one failure a person can actually fix printed "400
    Error" and nothing else."""
    body = _code(_fn("function reason(payload, fallback)"))
    assert 'typeof payload.detail.message === "string"' in body
    # And the array beside that message, which reason() cannot return.
    assert "err.payload = payload;" in HTML, "the body is thrown away"


def test_no_profile_name_reaches_innerhtml_unescaped():
    """A custom profile is named by whoever wrote it, which is not this file.

    Scoped to the innerHTML sinks on purpose. The other two places a name is
    drawn are #glosswhy, which is written with textContent, and the delete
    question, which is a confirm() -- both are text, and escaping there would
    put a literal &quot; on the screen. So the assertion is per sink rather
    than per name.
    """
    for raw in ("${g.name}", "${written.name}", "${r.text}", "${r.reason}",
                "${r.line}"):
        offenders = [n for n, line in enumerate(HTML.splitlines(), 1)
                     if raw in line and "esc(" not in line]
        assert not offenders, f"{raw} reaches innerHTML unescaped at line(s) {offenders}"
    # Deleted <strong>name</strong> is markup, so that one is escaped too.
    assert "Deleted <strong>${esc(target.name)}</strong>" in HTML
    # And the two text sinks are still text sinks.
    assert '$("glosswhy").textContent =' in HTML
    assert 'confirm(`Delete the vocabulary profile' in HTML
