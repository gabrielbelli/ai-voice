"""The engine control: how the reader picks one, and what they are told.

Static, for the reason test_interface.py, test_jobs.py and test_playback.py
are: what these assert is a property of the bytes in ui.html -- which document
the option labels are rendered from, which fields reach the wire, which control
is greyed rather than removed -- and starting a browser to find that out would
add a dependency to a service whose whole claim is that it has none. The
behaviour that only a running page can show (the sliders leaving, the language
collapsing, the body that Generate actually posts) is driven on the fake clock
in the node harness beside this suite.

WHY THIS FILE EXISTS. A second engine is not a second voice and not a second
route: it is one field on a request the page already makes. Every way it could
go wrong is a way the page could say one thing and send another --

  * an engine id written down HERE, in a browser nobody can redeploy, drifting
    from the catalogue the service ships;
  * a slider left on screen for a field the engine has no control for, so the
    reader sets a value this stack refuses BY NAME;
  * `model` added to the submit path and not to the retry path, so pressing
    Retry on a turbo run silently produces baseline audio;
  * an option that is HIDDEN rather than disabled when it cannot be used, which
    is how chatterbox-cpu stayed invisible for its entire life.

Each of those is one test below, named after it.
"""

import ast
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "ui.html"
HTML = PAGE.read_text()


def code(source: str) -> str:
    """The same source with its comments removed.

    Every comment in this page names the failure it prevents, so the prose
    quotes the very strings the assertions below forbid -- `r.service ===
    "chatterbox"`, "the fifth model table". Scanning the raw text would make
    each negative assertion argue with the file's own documentation. The four
    sibling files strip for exactly this reason.
    """
    return re.sub(r"/\*.*?\*/|<!--.*?-->|(?<!:)//[^\n]*", "", source, flags=re.S)


SCRIPT = code(HTML)


def body_of(name: str, ends: str) -> str:
    start = SCRIPT.index(name)
    return SCRIPT[start:SCRIPT.index(ends, start)]


# ------------------------------------------------- nothing is written down --


def test_no_engine_is_named_anywhere_in_this_page():
    """THE ONE PROPERTY EVERYTHING ELSE HERE RESTS ON.

    This page already holds four model tables and every one of them has been
    wrong at least once: the language lists, the CHARS_PER_SECOND pair, the
    optgroup labels, and `r.service === "chatterbox"`, which decided which
    machine's rate to show by comparing a service id to a literal and read
    "local" for every id it did not recognise.

    A fifth would be the worst of them, because it would be the only one that
    cannot be corrected without shipping a new page: the engines a deployment
    offers are a property of that deployment. So every label, every language,
    every control, every minimum clip length and both lanes' readiness are read
    out of /health.engines, and an id appears in this file only as data that
    came from the wire.

    `chatterbox` survives in the optgroup labels, in CHARS_PER_SECOND and in
    the group headings, which name the FAMILY of voices in the picker and
    predate any of this. What must not appear is a second engine's id, because
    there is no honest reason for this page to know one.
    """
    assert "chatterbox-turbo" not in SCRIPT, (
        "an engine id is spelled in the page. Every engine-shaped fact belongs "
        "in /health.engines, which the deployment can change and this file "
        "cannot.")
    assert "turbo" not in SCRIPT.lower(), (
        "the page reasons about an engine by name rather than by capability")


def test_the_picker_asks_no_route_of_its_own():
    """The engine block is drawn from a document the page already fetches.

    A second request would be a second thing to allowlist in the UI proxy, a
    second thing to go stale, and a second answer that can disagree with the
    rates and the runner panel drawn from /ui/health beside it. It is the same
    document; there is no new path, which is also why the PROXIED table in
    services/ui/app/main.py needs no entry for any of this.
    """
    module = SCRIPT[SCRIPT.index("function engines()"):SCRIPT.index("function kokoroRate()")]
    for caller in ("fetch(", "json(", "api("):
        assert caller not in module, f"the engine readers issue their own {caller}"
    assert "ttsLongHealth()" in module, "they no longer read tts-long's health"


# ----------------------------------------------------- honoured or refused --


def test_a_field_the_engine_cannot_honour_is_never_sent():
    """THE HOUSE RULE, ON THE PAGE'S SIDE OF IT.

    services/stt/app/openai_api.py states it and this whole surface is built on
    it: every field is either honoured or refused by name, and none is accepted
    and dropped. tts-long refuses exaggeration and cfg_weight on an engine with
    no conditioning layer, which is the right answer to a caller that asked for
    them -- and the wrong thing for this page to make a reader collect, because
    the sliders that would have sent them are not on the screen.

    So the body is built from the engine's own `controls`. There is no branch
    on an engine id here and there cannot be one: a third engine is a row in
    the service's catalogue and no page change at all.
    """
    body = body_of("async function queueJob(", "const parts = segments();")
    assert "spec.controls" in body, "the request body is not built from the capabilities"
    for field in ("exaggeration", "cfg_weight", "temperature"):
        assert f'controls.includes("{field}")' in body, \
            f"{field} is sent without asking whether the engine has it"
    # AND THE DEFAULT IS TODAY'S THREE FIELDS, so a tts-long that publishes no
    # engine block gets exactly the request it gets now.
    assert '["exaggeration", "cfg_weight", "temperature"]' in body, \
        "a deployment with no engine block lost its vendor fields"


def test_the_submit_and_the_retry_carry_the_engine_together():
    """TWO HALVES OF ONE FEATURE, AND ONE OF THEM SILENT, IS THE SHAPE OF EVERY
    BUG THIS STACK SHIPPED THIS WEEK.

    Retry does not read the controls: it rebuilds a body from a stored record,
    on a tab that may be open on a different device from the one that made the
    run. An engine left out of THAT path is a turbo job retried as baseline --
    different languages, different loudness, no error anywhere in the stack,
    under a button labelled Retry.

    `model` is what the caller asked for and `engine` is what actually ran;
    either pins it, and a record from before any of this has neither and
    resolves to the deployment's default, which is what it ran on.
    """
    queue = body_of("async function queueJob(", "const parts = segments();")
    assert "body.model = engine" in queue, "the submit path sends no engine"

    retry = body_of("async function retryJob(", "const payload = await json")
    assert "record.model || record.engine" in retry, \
        "the retry path rebuilds a body with no engine in it"
    # AND IT DOES NOT INVENT ONE EITHER. A record from an engine with no
    # exaggeration control never carried the field, and putting the key back
    # with an empty value is the shape this stack refuses by name.
    assert 'typeof record[field] === "number"' in retry, \
        "a field the record does not carry is sent back as null"


def test_a_deployment_that_names_no_engine_is_asked_for_none():
    """NULL IS A REAL ANSWER HERE.

    Before /health lands, and on a tts-long that predates the engine block, the
    page does not know that a choice exists. A name guessed at that point is a
    request for an engine the deployment may never have enabled, and the reader
    collects a 400 for a word nobody typed.
    """
    body = body_of("function currentEngine()", "\n}")
    assert "defaultEngine()" in body
    assert "engineIds()[0] || null" in body_of("function defaultEngine()", "\n}"), \
        "an empty catalogue resolves to something other than null"
    queue = body_of("async function queueJob(", "const parts = segments();")
    assert "if (engine) body.model = engine;" in queue, \
        "a null engine still puts a `model` on the wire"


# ------------------------------------------------- refusals stay on screen --


def test_an_engine_that_cannot_be_used_is_disabled_and_not_hidden():
    """Disabled-with-a-reason is this page's version of refused-by-name.

    chatterbox-cpu was configured on the server, wired to no lane, and drawn
    nowhere: it was invisible for its entire life and nothing ever said so. An
    engine this deployment cannot run, or that this voice's clip is too short
    for, is greyed WITH THE REASON BESIDE IT -- the way out of both is a
    sentence long, and a reader cannot take it if the control is not there.
    """
    body = body_of("function renderEngines()", "\n}")
    assert "disabled" in body, "a blocked engine is not disabled"
    assert "engineBlocked(id)" in body, "the reason is not computed per option"
    assert '" off"' in body, "there is no greyed state at all"
    # The reason is rendered INTO the option, beside the label, not into a log.
    assert 'class="why"' in body, "the reason is computed and never drawn"
    # NOT filtered out of the list.
    assert "filter(id => !engineBlocked" not in body, "a blocked engine is dropped from the list"


def test_the_reason_an_engine_is_unavailable_is_the_runners_own_word():
    """The agent publishes an exhaustive set -- not_installed, not_enabled,
    no_such_service, gpu_busy, machine_busy: <its own words>, unreachable:
    <ExcName>, manifest_mismatch -- and the two halves of it mean completely
    different things to a reader: one command on that machine fixes the first
    group for ever, and the second group clears on its own in a minute.

    This page turns into a sentence only the words it knows, and PASSES
    ANYTHING ELSE THROUGH VERBATIM. A reason this file has not met is still the
    truest thing anybody can say about why an engine is not running; a blank
    where it should have been is not.
    """
    table = SCRIPT[SCRIPT.index("const ENGINE_WHY = {"):]
    table = table[:table.index("};")]
    for word in ("not_installed", "not_enabled", "no_such_service", "manifest_mismatch"):
        assert word in table, f"{word} reaches the reader as a machine word"
    said = body_of("function whySaid(", "\n")
    assert "ENGINE_WHY[why] || why" in said, \
        "a reason this table has not met is swallowed"


def test_a_clip_shorter_than_the_engine_needs_refuses_that_engine_only():
    """The five-second minimum is a property of the PAIR, not of the voice and
    not of the engine: the same clip is fine on one engine and refused by the
    other, and the engine asserts it itself rather than failing gently.

    A LENGTH NOBODY READ REFUSES NOTHING. /ui/clips measures every clip it
    lists, but the built-in speaker has no file in that store at all, and a
    clip this side could not measure answers null. Turning that null into a
    zero would take an engine away from a voice that can use it perfectly well;
    the engine's own assert is the backstop for the rare case.
    """
    body = body_of("function engineBlocked(", "\n}")
    assert "spec.min_reference_seconds" in body, "the minimum is not read from /health"
    assert "have !== null" in body, "an unmeasured clip refuses an engine"
    seconds = body_of("function clipSeconds(", "\n}")
    assert 'typeof seconds === "number"' in seconds, "a missing length is not a state"


# --------------------------------------------------------- the whole claim --


def test_the_speed_claim_comes_from_the_rates_the_server_publishes():
    """2.36x was measured on one 3070 with both models on the same card.

    It is NOT the reader's answer when turbo is running on this server's CPU
    because the card is busy, and printing it there would put the picker out of
    step with the ETA in the button two controls away -- which is computed from
    cloneRate for the engine actually chosen. One speed claim, one source.
    """
    body = body_of("function speedWords(", "\n}")
    assert "cloneRate(id)" in body and "cloneRate(base)" in body, \
        "the comparison is not made from the published rates"
    assert "2.36" not in SCRIPT, "a measured figure from one machine is hard-coded"
    assert "2.4" not in body, "the speed-up is written down rather than computed"


def test_the_speech_rate_is_not_split_per_engine():
    """CHARS_PER_SECOND measures how long the speech IS, not how long it takes
    to make. Turbo renders faster; it does not talk faster.

    Splitting it would double-count the speed-up in every estimate on the tab,
    because the realtime factor beside it already carries all of it.
    """
    assert "const CHARS_PER_SECOND = { kokoro: 16.3, clone: 12.0 };" in HTML
    assert "CHARS_PER_SECOND[engine]" in SCRIPT
    table = SCRIPT[SCRIPT.index("const CHARS_PER_SECOND"):]
    assert "engineSpec" not in table[:200], \
        "the speech rate has been made a property of the engine"


def test_the_engine_control_belongs_to_the_voices_that_do_not_name_one():
    """WHAT THE OLD TEST HERE PROTECTED, AND WHY ITS PREMISE STOPPED HOLDING.

    It asserted that the voice list has no engine dimension at all -- one
    Chatterbox optgroup, `c:` in front of every clone -- on the reasoning that
    "the same clip is read by either engine", which is true of two clone
    engines and false of a checkpoint whose speakers are embeddings baked into
    its own weights. A clip is a file either model can open; `pt_male` is a
    tensor that exists in exactly one of them.

    So the property it was really protecting is asserted directly instead: the
    engine is a REQUEST FIELD, the control that picks it is drawn for the
    voices that leave it open, and a voice that already names its engine is not
    asked the question twice.
    """
    assert '<fieldset id="enginerow"' in HTML, "the engine control is not on the page"
    assert HTML.index('for="voice"') < HTML.index('id="enginerow"'), \
        "the engine is asked before the voice it applies to"
    body = body_of("function renderEngines()", "\n}")
    # ONLY THE ENGINES THAT CAN READ THE VOICE IN THE PICKER. An engine with no
    # speaker encoder cannot read a reference clip at all, so a radio for it
    # beside an uploaded voice is an option whose only outcome is a 400.
    assert "engineIds().filter(id => !presetVoices(id))" in body, \
        "the radio offers every engine, including ones that cannot read a clip"
    assert 'voice.kind === "preset"' in body, \
        "a voice that carries its engine is still asked which engine"


def test_the_page_holds_no_voice_list_of_its_own():
    """TWENTY NAMES IN NINE LANGUAGES WOULD BE THE FIFTH MODEL TABLE.

    Every engine-shaped fact on this page comes out of /health.engines for one
    reason: a table written here is the only one nobody can correct without
    shipping a new page. A preset engine's voices are the strongest case of it
    -- they are not a list somebody chose, they are which tensors are in a
    checkpoint, and the deployment that loads the checkpoint is the only thing
    that knows.

    The language is read the same way, off the voice's own row. Deriving it
    from the name is how PREFIX["d"] answers en-us for a German voice: one of
    twenty right by coincidence and the rest silently mislabelled.
    """
    voices = SCRIPT[SCRIPT.index("function renderVoices()"):
                    SCRIPT.index("function currentVoice()")]
    assert "presetVoicesForLanguage(id, chosenLang)" in voices, \
        "the preset groups are not built from the published rows"
    assert "presetVoices(id)" in voices, "the group is drawn without asking /health"
    filt = body_of("function presetVoicesForLanguage(", "\n}")
    assert "spec.language_from_voice" in filt, \
        "the language filter is not asked of the engine"
    assert "v.language" in filt, "the language is not read off the voice's own row"
    assert "PREFIX" not in filt, \
        "a preset voice's language is still guessed from its first letter"


def test_no_catalogue_engine_is_spelled_in_this_page():
    """THE JS TWIN OF tts-long's test_no_branch_on_an_engine_name.

    The page's fifth model table was one character wide --
    `value.slice(0,1) === "c" ? "clone" : "kokoro"` -- and the fence is what
    stops a sixth. An id that reaches this file comes off the wire and is used
    as data: interpolated into an option value, looked up in engines(), never
    compared to a word spelled here.

    `Kokoro` and `Chatterbox` survive as PROSE -- optgroup labels, the
    CHARS_PER_SECOND keys, the two plan keys speechPlan takes -- and those
    predate all of this. What must not appear is an id in a quoted literal,
    because there is no honest reason for this page to know one.
    """
    assert "voxtral" not in SCRIPT.lower(), \
        "the third engine is named in the page rather than read from /health"
    assert "turbo" not in SCRIPT.lower(), \
        "the page reasons about an engine by name rather than by capability"
    assert re.search(r"""[\"']chatterbox[\"']""", SCRIPT, re.I) is None, \
        "an engine id is spelled as a literal and can be compared against"
    # AND THE OPTION VALUE'S PREFIX IS INTERPOLATED, never typed. `k:` is the
    # instant path, which has no catalogue row to read an id from yet.
    voices = SCRIPT[SCRIPT.index("function renderVoices()"):
                    SCRIPT.index("function currentVoice()")]
    for literal in ('value="c:', 'value="chatterbox'):
        assert literal not in voices, f"{literal} is written down rather than read"


def test_the_picker_is_absent_where_there_is_nothing_to_pick():
    """One engine is not a choice, and a Kokoro voice is not this question:
    the control belongs to the request that goes to tts-long, and the fast path
    carries no model of its own.

    A control that is always on screen and never has two options is furniture,
    and this page removed a whole expert gate for being exactly that.
    """
    body = body_of("function renderEngines()", "\n}")
    # ON WHERE IT RUNS AND ON WHETHER THE VOICE LEAVES THE QUESTION OPEN. It
    # was `currentVoice().kind !== "clone"`, which answered both at once only
    # while every tts-long voice was a clone.
    assert "!isJob(voice)" in body
    assert 'voice.kind === "preset"' in body
    assert "ids.length < 2" in body
    # AND IT STILL CLEARS UP AFTER ITSELF. Hiding the control must not leave a
    # language disabled or a slider gone from the last engine that was chosen.
    assert "syncEngineControls();" in body[body.index("row.hidden"):], \
        "hiding the picker leaves its consequences on the other controls"


# ---------------------------------------- a voice that carries its engine --
#
# A THIRD ENGINE WHOSE VOICES ARE NOT CLIPS. It sits beside Kokoro
# conceptually -- a fixed list of speakers baked into the checkpoint -- and
# beside Chatterbox operationally: minutes of GPU on another machine, a job,
# a file at the end. The page had exactly one axis for both of those questions,
# `kind === "clone"`, and it answered them the same way. Each test below is one
# place where the two answers had to come apart.


def test_a_preset_voice_never_gets_the_instant_path():
    """FOUR SILENT WRONGS FROM ONE CHARACTER, and this is the fence over all of
    them.

    `value.slice(0,1) === "c" ? "clone" : "kokoro"` reported a preset voice as
    Kokoro, and then, with no error anywhere: the button kept "Generate &
    listen", a promise of instant audio for three and a half minutes of GPU;
    the ETA was skipped; streamDefaults planned it against Kokoro's 2.79x and
    set the route to SSE; and runSpeak sent it to speakNow -- a synchronous
    /speak against a model that has never heard of the voice.

    All four now ask isJob(), which is where it runs, not what the voice is.
    """
    est = code(body_of("function estimate()", "\nfunction nearestKokoro()"))
    assert "const job = isJob(voice);" in est
    assert '$("go-tts").hidden = job;' in est, "the Listen button is offered for a job"
    assert "if (!job) {" in est, "the ETA branch still tests the voice's kind"

    stream = body_of("function streamDefaults()", "\n}")
    assert "&& !isJob()" in stream, "a job can still be planned onto the audio path"

    run = body_of("async function runSpeak(listen)", "\n}")
    assert "if (isJob(voice)) await queueJob(voice.name, text);" in run, \
        "a preset voice still reaches the synchronous route"

    # AND THE KIND IS DERIVED, so a <select> that outlives a deployment change
    # cannot hold a stale one.
    voice = body_of("function currentVoice()", "\n}")
    assert "engineKind(engine)" in voice, "the kind is stored in the option value"
    assert "value.slice(0, 1)" not in voice, "the one-character prefix is back"


def test_the_legacy_prefixes_still_parse():
    """A REMEMBERED SELECTION MUST SURVIVE THE DEPLOY.

    `k:` and `c:` are in every stored job, every bookmark and both node
    harnesses. Neither is an engine id -- engineSpec settles that before the
    map is consulted -- and `c:` on a deployment with no engine block at all is
    still a clone, which is the one case where this page must behave byte for
    byte as it did before any of this.
    """
    voice = body_of("function currentVoice()", "\n}")
    assert 'head === "c" ? defaultEngine() : null' in voice, \
        "the legacy clone prefix no longer resolves to an engine"
    assert 'head === "c" ? "clone" : "kokoro"' in voice, \
        "a deployment with no engine block lost the kind it has always had"
    assert "engineSpec(head) ? head" in voice, \
        "a one-letter prefix could shadow an engine id"


def test_a_preset_voice_group_is_disabled_with_the_runners_reason():
    """chatterbox-cpu was configured, unreachable and invisible for its entire
    life, and the reason it stayed invisible is that nothing ever drew it.

    An engine that needs the runner and cannot have it is the same case with a
    9-language voice list attached: the group is drawn, greyed, with the
    runner's own word in the label -- because the way out is one command on
    that machine and a reader cannot take it if the control is not there.
    """
    voices = SCRIPT[SCRIPT.index("function renderVoices()"):
                    SCRIPT.index("function currentVoice()")]
    assert "engineOffline(id)" in voices, "the group is drawn without asking"
    assert 'off ? " disabled" : ""' in voices, "an unreachable group is not greyed"
    assert 'off ? " · " + off : ""' in voices, "the reason is computed and never drawn"
    assert "if (!presetVoices(id) || !listed.length) continue;" in voices, \
        "the group is dropped for a reason other than having no voices in it"
    # AND THE LANE HALF IS ASKED WITHOUT A VOICE. engineBlocked also measures
    # the SELECTED voice's clip, which is a different voice from the ones in
    # the group being drawn.
    off = body_of("function engineOffline(", "\n}")
    assert "currentVoice()" not in off, \
        "the group's reason depends on whichever voice happens to be selected"


def test_retry_copies_the_controls_the_engine_declares():
    """THE HALF OF THE PAIR THAT COULD NOT CARRY WHAT THE OTHER HALF SENT.

    queueJob stopped hard-coding ["exaggeration","cfg_weight","temperature"] a
    release ago and builds the body from spec.controls; Retry still held the
    literal three. So an engine tuned by two fields that list does not name was
    resubmitted without them -- different audio, no error anywhere, under a
    button labelled Retry. That is the identical defect its own surrounding
    comment was written about, one field along, and fixing it here fixes turbo
    at the same time.
    """
    retry = body_of("async function retryJob(id, button)", "const payload = await json")
    assert "engineSpec(model)" in retry, "the retry path does not ask the catalogue"
    assert "Array.isArray(spec.controls) ? spec.controls" in retry, \
        "the fields are still written down in this file"
    assert "for (const field of fields)" in retry
    # AND THE THREE SURVIVE ONLY AS THE NO-ENGINE-BLOCK FALLBACK, which is the
    # deployment that has always sent exactly those.
    assert ': ["exaggeration", "cfg_weight", "temperature"];' in retry, \
        "a deployment with no engine block lost its vendor fields on retry"
    # AND A LANGUAGE THE ENGINE CARRIES ON ITS VOICE IS NOT SENT BACK.
    assert "!(spec && spec.language_from_voice)" in retry, \
        "retry puts a language on the wire for an engine that takes no language"


def test_retry_and_speak_again_ask_which_service_made_the_run():
    """`kind` says what a run PRODUCED and it stopped deciding this.

    A preset-voice run on tts-long is kind "speech" -- minutes of GPU, a file
    on the server, a Retry that resubmits and means something. A Kokoro run is
    kind "speech" too and keeps no file at all, so "get it back" there means
    saying it again. On the kind test the first got Speak again, which posts to
    /speak with a voice Kokoro has never heard of, and got no Retry at all.
    """
    row = SCRIPT[SCRIPT.index('$("joblist").innerHTML = list.map'):
                 SCRIPT.index('}).join("")')]
    assert 'const queued = job.service ? job.service === "tts-long" : kind === "clone";' \
        in row, "the row does not read which service made the run"
    assert 'kind === "speech" && !queued ?' in row, \
        "Speak again is offered for a run the instant route cannot make"
    assert 'queued && job.status === "failed" ?' in row, \
        "Retry is not offered for every failed run tts-long made"


def test_a_run_that_never_ran_names_no_machine():
    """An engine with no lane on this server fails without ever being
    dispatched, and the record still carries `host` -- the box whose process
    WROTE it. The last line of ranOn printed that beside the engine, so work
    that happened nowhere read as "it ran here".

    And "after the runner gave up" must never appear for it either: `fell_back`
    is false for an engine that has nowhere to fall back TO.
    """
    body = body_of("function ranOn(job)", "\n}")
    assert 'if (job.status === "failed" && !job.backend) return' in body, \
        "a job no lane ever took still names a machine"
    assert body.index('!job.backend') < body.index("if (job.fell_back)"), \
        "the fall-back sentence is reached first for a job that never ran"


def test_the_speech_kind_is_no_longer_labelled_instant():
    """tts-long writes "speech" for an engine whose voices are presets, so the
    filter labelled "Instant speech" now covers runs that take minutes. The
    kind is still the right axis; the label was describing the engine."""
    assert '<option value="speech">Speech</option>' in HTML
    assert "Instant speech" not in HTML, "the filter still promises instant"
    assert 'speech: "Speech"' in SCRIPT, "the two lists disagree about the label"


def test_the_quick_alternative_is_offered_only_where_it_can_read_the_text():
    """`|| all.find(v => v[0] === "b") || all[0]` meant that for any language
    Kokoro does not speak the offer was a BRITISH voice: "Use bm_george
    instead" over German text, under a warning about the German job's cost.

    Two engines speaking nearly the same set hid it. Nine languages on one
    engine and Kokoro speaking six of them shows it immediately, and a button
    that is not there is a smaller failure than one that reads the text in the
    wrong language.
    """
    body = body_of("function nearestKokoro()", "\n}")
    assert 'return all.find(v => v[0] === want) || null;' in body, \
        "the offer falls back to a voice that cannot read the text"
    # AND THE WARNING STILL DRAWS WITHOUT IT: the Use-instead button is
    # conditional, the Queue-it-anyway button is not.
    est = body_of("function estimate()", "\nfunction nearestKokoro()")
    assert "${quick ? `<button" in est, "the offer is drawn even when there is none"
    assert 'id="anyway"' in est, "the other way out went with it"


def test_the_language_control_is_not_asked_of_an_engine_that_carries_it():
    """There is no language field on that request: the server derives it from
    which embedding was named and refuses by name any code that contradicts it.
    A control left live is that refusal arriving minutes later -- and a second
    derivation in this browser could only agree with the first or turn a valid
    request into a 400.
    """
    sync = body_of("function syncLanguageForEngine()", "\n}")
    assert "if (languageCarried()) {" in sync
    assert '$("langsaid").textContent' in sync, "the control is greyed with no reason"
    assert 'id="langsaid"' in HTML, "there is nowhere to put the reason"

    queue = body_of("async function queueJob(", "const parts = segments();")
    assert "if (!languageCarried()) body.language" in queue, \
        "a language is sent for an engine that takes no language field"

    resolved = body_of("function resolvedLanguage()", "\n}")
    assert "presetLanguage(voice)" in resolved, \
        "the resolved language is not read off the voice's own row"


def test_an_engine_with_no_lane_has_no_rate_rather_than_this_servers():
    """`local.ready` is "is it in TTS_LOCAL_ENGINES" -- a deployment fact, not a
    warm one -- so an engine left out of that list has NO lane on this server.

    cloneBackend read "local" for anything the runner could not take, which was
    true while every engine had a local lane. For one that does not, it keys
    the rate on a lane that will never run it, and the figure that comes back
    was measured on a completely different model on this server's CPU: an ETA
    and a group heading both confidently four times wrong. Null is the answer
    cloneRate already knows how to hold, and this page already treats a null
    rate as a real answer and not a zero.
    """
    body = body_of("function cloneBackend(engine)", "\n}")
    assert 'if (spec.local && spec.local.ready) return "local";' in body, \
        "the local lane's readiness is not read at all"
    assert "spec.local || spec.runner ? null : \"local\"" in body, \
        "an engine with no lane still reads as local"
    # AND THE COST HEADING IS DRAWN FROM cloneRate, not from rate.clone(),
    # whose 0.275 fallback is a Chatterbox CPU figure.
    cost = body_of("function costWords(", "\n}")
    assert "cloneRate(id)" in cost and "rate.clone" not in cost, \
        "a group with no measured lane still claims a number"
    assert 'f > 0 ? `about' in cost, "a null rate becomes a figure"


def test_the_cost_on_a_group_is_dropped_for_the_groups_own_ambiguity():
    """It was dropped the moment `engineIds().length > 1`, which is the
    deployment's ambiguity and not the group's.

    That rule is right for the clip group -- either Chatterbox reads the same
    wav, so a figure there would be one of two models printed over both -- and
    wrong twice over otherwise: it hides the figure on a group whose voices
    belong to exactly one engine, and it hides it on the clip group of a
    deployment that has exactly one thing reading clips.
    """
    voices = SCRIPT[SCRIPT.index("function renderVoices()"):
                    SCRIPT.index("function currentVoice()")]
    assert "const clipEngines = engineIds().filter(id => !presetVoices(id));" in voices, \
        "the rule does not count the engines that read the group's voices"
    assert "clipEngines.length === 1 ? costWords(clipEngines[0])" in voices, \
        "one clip engine no longer gets its figure back"
    assert "engineIds().length > 1 ?" not in voices, \
        "the deployment-wide suppression is back"
    # AND A PRESET GROUP CARRIES ITS OWN, unconditionally: its voices are one
    # engine's by construction.
    assert "const cost = costWords(id);" in voices


def test_every_control_the_engine_lacks_can_leave_the_panel():
    """Temperature sat OUTSIDE the group that can leave, on the argument that
    it survives on both engines -- true of the two that existed.

    An engine that reads no temperature at all would have been left showing
    that one slider with its value refused by name at the server, which is
    worse than an absent one: it is a promise the request cannot keep.
    """
    body = body_of("function syncEngineControls()", "\n}")
    assert 'const gone = ["exaggeration", "cfg_weight", "temperature"]' in body, \
        "a control the engine lacks can still be left on screen"
    assert 'controls && !controls.includes(field)' in body
    assert '$("clone-temp").hidden = !warm;' in body, "temperature cannot leave"
    assert '$("resetclone").hidden = !expressive && !warm;' in body, \
        "Reset to defaults outlives the sliders it resets"
    assert 'id="clone-temp"' in HTML, "there is no group for it to leave in"
    # AND THE SENTENCE IS BUILT FROM THE SAME LIST, so the panel and the wire
    # cannot disagree about what is settable.
    assert "gone[gone.length - 1]" in body, "the sentence is written out by hand"
    assert "controls.join(" in body, "it does not name what the engine does take"


# ----------------------------------- the two halves, read against each other --
#
# THE BUG THIS STACK KEEPS SHIPPING IS ALWAYS THE SAME SHAPE: two halves of one
# feature that disagree while both suites stay green, because each half was
# tested against a fixture the other half never wrote. A page asking for a count
# name the server never emitted. A filter whose count meant a different set from
# the rows it labelled. Four single-token edits that produced the wrong engine's
# audio under the right engine's label.
#
# The engine block is the widest surface of that shape on this page: eleven
# fields, published by one service, read by a browser nobody can redeploy, and
# every test on either side driven by a hand-written health document. A field
# the page reads and the service does not send costs NO error anywhere -- the
# read is `undefined`, the sentence it feeds is "", and the reader simply never
# sees a thing that was built for them.
#
# So this reads the real halves against each other: the fields the page takes
# off an engine row, against the keys the service actually puts in one. No
# fixture on either side.

# Reached the way test_gap_counts_contract.py reaches the same service, and
# read the same way: as source, never imported.
TTS_LONG = Path(__file__).resolve().parents[2] / "tts-long" / "app"


def _rest_of_block(start: int) -> str:
    """The source from `start` to the end of the block it sits in.

    Scoped rather than global ON PURPOSE. `mine` is bound to an engine row in
    one function and to a job record in another, and a whole-file scan for
    `mine.` charges this page with reading an engine field called `params`.
    """
    depth = 0
    for i in range(start, len(SCRIPT)):
        if SCRIPT[i] == "{":
            depth += 1
        elif SCRIPT[i] == "}":
            if depth == 0:
                return SCRIPT[start:i]
            depth -= 1
    return SCRIPT[start:]


def _fields_the_page_reads() -> set[str]:
    """Every key this page takes off one row of /health.engines."""
    found: set[str] = set()
    # `const spec = engineSpec(id)`, `const mine = engineSpec(e), base = ...`
    for match in re.finditer(r"\b(\w+)\s*=\s*engineSpec\(", SCRIPT):
        scope = _rest_of_block(match.end())
        found |= {m.group(1) for m
                  in re.finditer(rf"\b{match.group(1)}\.(\w+)", scope)}
    # `engineSpec(currentEngine()).languages`, read without a binding.
    found |= {m.group(1) for m in re.finditer(r"engineSpec\([^()]*\)\.(\w+)", SCRIPT)}
    # `const all = engines(); ... all[id].default` -- the row reached through
    # the map rather than through engineSpec, which is how defaultEngine reads
    # the flag that decides what an absent `model` resolves to.
    for match in re.finditer(r"\b(\w+)\s*=\s*engines\(\)\s*;", SCRIPT):
        scope = _rest_of_block(match.end())
        found |= {m.group(1) for m
                  in re.finditer(rf"\b{match.group(1)}\[[^\]]+\]\.(\w+)", scope)}
    return found


def _fields_the_service_publishes() -> set[str]:
    """Every key tts-long actually puts in one row of /health.engines.

    READ FROM THE SOURCE, NOT IMPORTED. Importing tts-long's app package from
    this suite would drag its dependencies into a service whose whole claim is
    that it has none, and the keys are literals in two dict displays -- which
    `ast` reads exactly and a regex reads approximately.
    """
    def _dict_keys(node: ast.Dict) -> set[str]:
        return {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}

    # The catalogue half: engine_rows() in app/engines.py.
    rows = ast.parse((TTS_LONG / "engines.py").read_text())
    fn = next(n for n in ast.walk(rows)
              if isinstance(n, ast.FunctionDef) and n.name == "engine_rows")
    comp = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Return))
    assert isinstance(comp, ast.DictComp), \
        "engine_rows no longer returns one dict display per engine"
    published = _dict_keys(comp.value)

    # The lane half: _health() in app/main.py merges `**row` with the two lane
    # rows, and the page reads `.ready` and `.why` off both.
    health = ast.parse((TTS_LONG / "main.py").read_text())
    fn = next(n for n in ast.walk(health)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name == "_health")
    body = next(n.value for n in ast.walk(fn)
                if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
    engines = next(v for k, v in zip(body.keys, body.values)
                   if isinstance(k, ast.Constant) and k.value == "engines")
    lanes = {k.value: _dict_keys(v) for k, v in zip(engines.value.keys,
                                                    engines.value.values)
             if isinstance(k, ast.Constant) and isinstance(v, ast.Dict)}
    for lane in ("local", "runner"):
        assert {"ready", "why"} <= lanes.get(lane, set()), (
            f"the {lane} row no longer carries ready and why, which is what "
            f"engineOffline turns into the reason beside a disabled option")
    return published | set(lanes)


# THE ONE FIELD THIS PAGE READS THAT NOTHING PUBLISHES, and it is named here
# rather than quietly excluded so that finishing it FAILS this test.
#
# loudnessLine() is the reader's warning that a quieter engine is quiet by
# design: turbo normalises to -27 LUFS against the current model's -22,
# measured, and the difference is audible enough to be heard as a fault. The
# browser half of it shipped; `loudness_lufs` is in no EngineFacts field, no
# engine_rows() key and no health body on this stack, so the note is a sentence
# that cannot be reached on any deployment that exists.
#
# TO FINISH IT: add `loudness_lufs: float | None` to EngineFacts in
# packages/common/voice_common/engines.py with the measured value on each row,
# publish it from engine_rows() in services/tts-long/app/engines.py, and delete
# this set. The assertion below then holds with nothing excluded.
UNPUBLISHED = frozenset({"loudness_lufs"})


def test_every_engine_field_the_page_reads_is_one_the_service_publishes():
    """THE FENCE ACROSS THE DEFECT CLASS, IN THE DIRECTION IT ACTUALLY RUNS.

    A page that reads a key the service does not send raises nothing: no 404,
    no console error, no line in any log. The value is `undefined`, the guard
    above it returns "", and the feature is simply absent for everybody. Both
    suites stay green because each was written against a document its own side
    made up.

    Nothing here is made up. The left side is every member read off an engine
    row in ui.html; the right side is every key tts-long writes into one. They
    have to be the same set, and the only difference allowed is the one named
    in UNPUBLISHED above -- which is itself pinned, so the day somebody
    publishes that field this test fails and points at the note that then has
    to be switched on.
    """
    reads = _fields_the_page_reads()
    published = _fields_the_service_publishes()
    missing = reads - published

    landed = UNPUBLISHED - missing
    assert not landed, (
        f"{', '.join(sorted(landed))} is published now, so the page half that "
        f"was waiting for it works: delete it from UNPUBLISHED in this file "
        f"and the assertion below holds with nothing excluded.")
    assert missing == set(UNPUBLISHED), (
        f"the page reads {', '.join(sorted(missing - UNPUBLISHED))} off an "
        f"engine row and services/tts-long publishes no such key. It is not an "
        f"error anywhere -- the read is undefined and whatever it feeds goes "
        f"silent -- so either add the field to engine_rows() or stop reading "
        f"it. Published today: {', '.join(sorted(published))}.")


def test_the_reasons_an_engine_cannot_run_are_the_runners_own_words():
    """The other half of the same pair, and the one that fails SAFE.

    ENGINE_WHY turns the codes the runner publishes into a sentence, and every
    code it does not know is passed through verbatim rather than swallowed --
    "machine_busy: somebody is at the keyboard" reads perfectly well, and a
    blank where the reason should have been does not. So a new code on the
    service costs nothing here.

    What it must not do is name a code that no longer exists, because that is a
    translation nobody can reach and a reader left with the raw word the page
    was built to explain.
    """
    table = body_of("const ENGINE_WHY = {", "};")
    known = set(re.findall(r"^\s*(\w+):", table, re.M))
    assert known, "the reason table is empty"
    remote = (TTS_LONG / "remote.py").read_text() + (TTS_LONG / "dispatch.py").read_text()
    for code_ in sorted(known):
        assert f'"{code_}"' in remote, (
            f"the page translates {code_!r}, which the runner lane no longer "
            f"publishes: the sentence is unreachable and a code the service "
            f"DOES send now reaches the reader raw")
