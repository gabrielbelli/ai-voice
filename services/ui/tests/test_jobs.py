"""The Jobs tab as a run log over all three kinds of run.

Static, for the reason test_interface.py and test_playback.py are: what these
assert is a property of the bytes in ui.html -- which query the filter sends,
which fields a row dereferences, what a control says when it has no number --
and starting a browser to find that out would add a dependency to a service
whose whole claim is that it has none.

Every test here is named after a defect that was in the file or after one this
change could introduce. The two that matter most are the two silent ones: a
single record with a realtime factor and no audio_seconds took the WHOLE tab
into "The job list could not be drawn", and a filter applied on this side of
the wire silently hides rows that only exist past the server's fifty-row cap.
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

    Every comment in this file's subject names the failure it prevents, so it
    quotes the very thing a negative assertion forbids -- "was `now -
    job.finished_at > TTL`", "it used to answer ''". Matching the raw text
    would assert against the subject's own documentation. The three sibling
    files strip for exactly this reason.
    """
    return re.sub(r"/\*.*?\*/|<!--.*?-->|//[^\n]*", "", source, flags=re.S)


BARE = code(HTML)
ROW = code(HTML[HTML.index('$("joblist").innerHTML = list.map'):
                HTML.index('}).join("")')])
RENDER = code(body_of("function renderJobs()", "let jobUrl = null;"))
REFRESH = code(body_of("async function refreshJobsInner()", "\nfunction ping(job)"))


# ------------------------------------------------------------- the filters --


def test_the_filter_for_the_jobs_whose_audio_is_gone_exists_by_name():
    """THE THING THAT WAS ASKED FOR AND WAS NOT THERE.

    The tab had exactly one filter -- "only jobs whose audio still exists" --
    and no way at all to ask the opposite question: which of the things I made
    has the server thrown away. Deleted and expired are separate options
    because they are separate facts: somebody pressed a button, or a sweeper
    ran, and the second is not anybody's decision.
    """
    assert "const JOB_FILTERS" in HTML
    assert 'query: { audio: "deleted" }' in HTML
    assert 'query: { audio: "expired" }' in HTML
    assert 'id="jobfilter"' in HTML
    for value in ("playable", "all", "deleted", "expired", "failed"):
        assert f'<option value="{value}">' in HTML, f"no {value} option on the control"


def test_every_kind_of_run_can_be_asked_for_on_its_own():
    """Instant speech and transcriptions land in the same listing as a clone
    and there are far more of them, so without this the tab is one kind of run
    buried under two others."""
    for value in ("all", "clone", "speech", "transcribe"):
        assert f'<option value="{value}">' in HTML, f"no {value} option on the kind control"
    assert 'if (kind !== "all") params.set("kind", kind);' in REFRESH


def test_the_filter_is_a_query_and_never_a_pass_over_the_answer():
    """THE ARITHMETIC, not the taste. tts-long caps a listing at fifty rows, so
    filtering after the fetch asks for the fifty most recent runs of ANY kind
    and then shows whichever of them happen to match -- and one afternoon of
    instant speech pushes this morning's clone off the end before anything on
    this side has run.
    """
    assert "new URLSearchParams()" in REFRESH, "the filter is not sent as a query"
    assert 'json("/jobs" + (query ? "?" + query : ""))' in REFRESH
    assert "const list = all;" in RENDER, "the answer is being filtered a second time"
    assert "onlyAudio" not in BARE, "the old client-side filter is still here"


def test_a_live_row_and_a_failed_row_are_never_hidden_on_this_side():
    """Both rules are the service's, and repeating them here would make this
    page the second place they can drift -- but re-filtering the answer would
    silently undo them, which is the reading of a filter that looks like data
    loss. The thing you are waiting for and the thing that has just gone wrong
    are the two rows nobody ever means to hide.
    """
    body = RENDER[:RENDER.index('$("joblist").innerHTML = list.map')]
    # `all.filter` for the live COUNT in the tab chip is fine and stays; what
    # must not exist is a second definition of what gets drawn.
    assert re.findall(r"\blist\s*=", body) == ["list ="], \
        "the rows drawn are decided more than once"
    assert "list.filter(" not in RENDER, "renderJobs is filtering the rows again"
    # And the query for the default never names a status, so the service's own
    # two rules are the only thing deciding those rows.
    playable = HTML[HTML.index("playable: {"):HTML.index("\n  all:")]
    assert "status" not in playable, "the default filter is asking for a status"


def test_the_filter_controls_are_always_on_screen():
    """A DELIBERATE REVERSAL of the rule that governed the checkbox they
    replace. That control appeared only when it would actually hide something,
    which was right while it defaulted to OFF. These default to something other
    than everything, so the same rule inverted removes the only explanation on
    the page for why a row somebody remembers is missing.
    """
    assert "onlyaudiowrap" not in HTML, "the only-when-it-hides wrapper is back"
    header = HTML[HTML.index('<h2 class="tight">Jobs</h2>'):HTML.index('id="jobplay"')]
    for control in ('id="jobkind"', 'id="jobfilter"'):
        assert control in header, f"{control} is not in the card header"
    selects = re.findall(r"<select[^>]*id=\"job(?:kind|filter)\"[^>]*>", header)
    assert len(selects) == 2
    for tag in selects:
        assert "hidden" not in tag, "a filter control can hide itself"
        assert "aria-label=" in tag, "an unlabelled select on a phone is a mystery box"


def test_the_choice_survives_a_reload():
    """A filter that resets on every reload has to be set again every time
    somebody comes back to look for the thing they were filtering for."""
    assert 'store.get("jobfilter", "playable")' in HTML
    assert 'store.get("jobkind", "all")' in HTML
    assert 'store.set(control, $(control).value);' in HTML
    # Validated on the way out: a value from an older build, or one typed into
    # localStorage, must not become a query nothing answers.
    assert "JOB_FILTERS[value] ? value : \"playable\"" in HTML
    assert "JOB_KINDS[value] ? value : \"all\"" in HTML


def test_changing_a_filter_asks_the_server_again():
    """This line used to say "filtering is a question about the rows on this
    page, not a question for the server", and that was true of a checkbox over
    an unfiltered fifty. It is false of a filter whose whole purpose is
    reaching rows the fifty do not contain."""
    wiring = code(HTML[HTML.index('for (const control of ["jobfilter", "jobkind"])'):])
    wiring = wiring[:wiring.index("\n}")]
    assert "schedule(0)" in wiring, "the new filter is never sent"
    assert "renderJobs()" not in wiring, "it redraws the old answer instead of asking"


def test_a_count_that_was_not_sent_prints_no_number():
    """The counts label the options -- "Everything (412)" is legible on the
    closed select, which is what stops a default filter reading as data loss.
    A count worked out on this side, on the control that decides what the
    reader can see, would be worse than none: a missing number reads as "not
    said" and a wrong one reads as fact.
    """
    body = code(body_of("function jobCount(group, name)", "\nconst FILTER_COUNT"))
    assert "if (!jobCounts) return null;" in body
    assert "return null;" in body.rsplit("\n", 3)[-2] or body.count("return null") >= 2
    sync = code(body_of("function syncJobFilters()", "\nfunction renderJobs()"))
    assert 'n === null ? "" : ` (${n})`' in sync
    assert sync.count('n === null ? "" : ` (${n})`') == 2, \
        "one of the two controls invents a count"
    assert "jobCounts = payload.counts || null;" in REFRESH


def test_the_empty_card_says_which_filter_is_holding_rows_back():
    """An empty card under a default that hides things is indistinguishable
    from an empty card under a stack that has lost everything, and the second
    reading is the one people act on."""
    empty = RENDER[RENDER.index("if (!list.length)"):]
    empty = empty[:empty.index('document.title = "ai-voice"')]
    assert "JOB_FILTERS[filter].empty" in empty, "the empty state does not name the filter"
    # A phrase per filter, not the control's own label: "No Failures runs" is
    # what happens when the two are made to be one string.
    for value in ("playable", "deleted", "expired", "failed"):
        assert f"{value}:" in HTML and "empty:" in HTML
    assert "No runs whose audio was deleted" in HTML
    assert "No runs whose audio was swept" in HTML
    assert "hidden by this filter" in empty
    assert "No jobs. Cloned voices queue here." not in HTML, \
        "the empty state still says the tab is only for clones"


def test_a_truncated_listing_says_so():
    """The service caps a listing. Silence about the cap is how "where did my
    job go" becomes a bug report about deletion."""
    assert "jobTruncated = Boolean(payload.truncated);" in REFRESH
    assert "jobTruncated ?" in RENDER


def test_a_filtered_listing_never_declares_a_remembered_job_lost():
    """THE TRAP THE QUERY PARAMETER OPENS. refreshJobsInner concludes things
    from an id being absent from the answer -- that the job died with a
    restart, that it was swept -- and every one of those conclusions needs the
    listing to have been of EVERYTHING. Under a filter the server was never
    asked about that job, so its absence says nothing at all, and without the
    guard the first press of a select marks half this browser's history
    "lost when the service restarted".
    """
    assert 'const unfiltered = query === "";' in REFRESH
    assert "if (unfiltered && now - seen.at > TTL) continue;" in REFRESH
    assert "} else if (unfiltered) {" in REFRESH, \
        "a finished job hidden by the filter is restored anyway"


# ------------------------------------------------------------ what a row says --


def test_a_rate_with_no_seconds_behind_it_does_not_take_the_whole_tab_down():
    """THE SILENT ONE. The line was guarded on job.realtime_factor alone and
    then dereferenced job.audio_seconds.toFixed(0) and
    job.compute_seconds.toFixed(0). One record carrying a rate without the two
    numbers under it -- a transcription that failed after the clip was
    measured, any sender that reports a rate on its own -- threw inside
    list.map, and list.map is inside the try that writes "The job list could
    not be drawn". One undefined field on one row, and NO ROWS AT ALL.
    """
    assert "job.realtime_factor ? `<div" not in ROW, "the one-field guard is back"
    assert 'typeof job.audio_seconds === "number"' in ROW
    assert 'typeof job.compute_seconds === "number"' in ROW
    assert 'typeof job.realtime_factor === "number"' in ROW
    # Nothing is dereferenced outside its own guard.
    for field in ("audio_seconds", "compute_seconds", "realtime_factor"):
        for hit in re.finditer(rf"job\.{field}\.toFixed", ROW):
            before = ROW[:hit.start()]
            assert f'typeof job.{field} === "number"' in before, \
                f"job.{field} is used before it is checked"


def test_a_rate_above_realtime_does_not_read_like_one_below_it():
    """Parakeet measures 17.14x and Chatterbox 0.23x -- one seventeen times
    faster than the clock, the other four times slower -- and both printed
    "${rtf}x realtime", the same sentence with opposite meanings, on the one
    tab whose purpose is comparing them."""
    body = code(body_of("function rateWords(rtf)", "\nfunction statusClass"))
    assert "rtf >= 1" in body
    assert "faster than realtime" in body
    assert "toFixed(1)" in body and "toFixed(2)" in body
    # And every row that prints a rate goes through it, so the two cannot drift.
    assert "× realtime" not in ROW, "a row still spells the rate out for itself"
    assert "rateWords(" in ROW


def test_whether_a_row_still_has_audio_is_read_and_not_guessed():
    """It ended with `!(job.finished_at && now - job.finished_at > TTL)` --
    this page's arithmetic against its own copy of the sweep interval. tts-long
    sweeps the audio and the record on two different clocks now, so that guess
    is wrong in both directions: a player drawn over a file swept an hour ago,
    and a player greyed out on a record kept deliberately for a month.
    """
    body = code(body_of("function hasAudio(job)", "\n/* WHERE IT RAN"))
    assert "TTL" not in body, "the age guess is still in the predicate"
    assert "finished_at" not in body
    assert 'job.audio.state === "present"' in body
    # And the row does not re-derive it either.
    assert "now - job.finished_at > TTL" not in ROW
    assert "TTL / 3600" not in HTML, "the row still quotes this page's own TTL at the reader"


def test_a_record_with_no_kind_is_a_clone():
    """Every sidecar tts-long wrote before this release has no `kind`, and the
    default is what makes all of them valid rows rather than a migration."""
    assert 'const kind = job.kind || "clone";' in ROW


def test_every_row_says_where_it_ran():
    """ranOn answered "" for anything that was not a runner job or a
    fall-back, on the argument that a stack with one machine would otherwise
    label every job "local". The record carries `host` now, so this names WHICH
    machine rather than restating that there was one -- and the tab lists three
    engines whose rates are four orders apart. A row that does not say where it
    ran is the row that cannot be compared.
    """
    body = code(body_of("function ranOn(job)", "\n/* AN RTF ABOVE 1"))
    assert 'if (!job.backend) return "";' not in body, \
        "a row with no backend still says nothing"
    assert "job.host" in body, "the host is never read"
    assert "job.engine" in body, "the engine is never read"
    assert "`${host} · ${engine}${why}`" in body
    # The fall-back is still the one case worth a sentence.
    assert "gave up" in body


def test_a_row_says_why_that_engine_and_not_the_other_one():
    """With two engines offered, the engine's NAME on a row answers half the
    question. "chatterbox" means one thing when somebody typed it and another
    when the deployment chose it for a caller who named no model at all, and
    the difference is the whole of "why did I not get turbo".

    engine_reason is the service's own word for it, and a word this table has
    not met is printed as the service spelled it rather than dropped -- the
    same rule the runner's unavailable_reason follows.
    """
    body = code(body_of("function ranOn(job)", "\n/* AN RTF ABOVE 1"))
    assert "job.engine_reason" in body, "the row never says why that engine"
    assert "ENGINE_REASON[job.engine_reason] || job.engine_reason" in body, \
        "an unknown reason is dropped instead of being printed verbatim"
    table = HTML[HTML.index("const ENGINE_REASON = {"):]
    table = table[:table.index("};")]
    for reason in ("pinned", "default", "alias:tts-long", "alias:openai"):
        assert reason in table, f"{reason} is not one of the answers"


def test_the_audio_states_are_not_collapsed_into_one_sentence():
    """`never` is the load-bearing one: an instant run's audio was not lost, it
    was never kept, and telling a reader "deleted" about a file that never
    existed is the lie the enum exists to prevent. So the deleted line is
    gated on the state and not on "there is no file"."""
    assert 'job.audio.state === "deleted"' in ROW
    assert 'job.audio.state === "expired"' in ROW
    assert "The audio was deleted. This record was kept." in ROW
    # ...and neither sentence may be printed for a kind that never had audio.
    deleted = ROW[ROW.index("The audio was deleted."):]
    assert "audio_deleted ?" not in ROW.split("The audio was deleted.")[0][-200:], \
        "the deleted line is back on the bare boolean"
    assert deleted  # the slice above must exist at all


# ---------------------------------------------------- getting the thing back --


def test_a_transcription_can_get_its_transcript_back():
    """A transcription has no audio on this row and never had -- the clip
    belonged to whoever supplied it -- so a tab whose stated job is "get the
    thing back" is inert for that kind of run unless the text can leave it."""
    assert 'kind === "transcribe" ? `<button class="small" data-copy=' in ROW
    assert "Copy the transcript" in ROW
    assert "copyTranscript(b.dataset.copy, b)" in HTML
    body = code(body_of("async function copyTranscript(id, button)", "\n\nasync function refreshJobs"))
    assert "navigator.clipboard.writeText" in body
    # The same cache the open row fills, so opening it and pressing this is one
    # request between them rather than two.
    assert "JOBTEXT.get(id)" in body and "JOBTEXT.set(id" in body


def test_an_instant_run_offers_the_only_way_it_can_come_back():
    """Kokoro keeps no file: the audio was streamed or downloaded and the
    record is all that is left, so "get it back" here means making it again.

    AND IT IS OFFERED ON THE SERVICE, NOT ON THE KIND. `kind` says what a run
    produced and it stopped answering this the day tts-long grew an engine
    whose voices are presets: that run is kind "speech" too, it is minutes of
    GPU, and its audio is a file on the server -- so Speak again would have
    posted its voice name to /speak, an instant route on a different service
    whose model has never heard of it.
    """
    assert 'kind === "speech" && !queued ? `<button class="small" data-again=' in ROW
    assert "Speak again" in ROW
    body = code(body_of("async function speakAgain(id, button)", "\n\n/* THE TRANSCRIPT IS"))
    assert 'api("/speak"' in body, "it does not reach the instant route"
    assert "recordFor(id)" in body, "the parameters do not come from the record"
    assert "playBlob(" in body, "the answer is never played"
    # /speak is extra="forbid", so nothing speculative is sent.
    for field in ("voice", "text", "format"):
        assert f"{field}:" in body or f"body.{field} =" in body


def test_the_parameters_for_a_repeat_come_from_the_record():
    """Retry read localStorage, so it existed only on the machine that made the
    job: the phone drew the button and then answered "the parameters for that
    job were not kept", which is a control offered where it cannot work. The
    record has held all of it on the server the whole time.
    """
    body = code(body_of("async function retryJob(id, button)", "\n\n/* SAY IT AGAIN"))
    assert "recordFor(id)" in body
    assert "remembered().find" not in body, \
        "retry still reads this browser's memory as its source"
    # adopt() survives only as the seed for a job the server has not listed yet.
    assert "adopt(payload.id, params)" in body
    fetcher = code(body_of("async function recordFor(id)", "\nasync function retryJob"))
    assert "json(`/jobs/${encodeURIComponent(id)}`)" in fetcher
    assert "remembered().find" in fetcher, "there is no fallback when the record is gone"


def test_retry_belongs_to_the_kind_that_can_be_retried():
    """A transcription cannot be run again from here -- the clip is not kept --
    and offering the button anyway is the phone-Retry defect in a new place."""
    assert 'queued && job.status === "failed" ?' in ROW
    # AND ON THE SERVICE THAT CAN TAKE IT BACK. Retry resubmits to POST /jobs,
    # so what it needs is a run tts-long made -- which is every clone AND every
    # run of an engine whose voices are presets, filed under kind "speech". On
    # the kind test that second one drew no Retry at all: a job that failed
    # because the runner was away, with the one button that would run it again
    # missing from the row.
    assert 'const queued = job.service ? job.service === "tts-long" : kind === "clone";' \
        in ROW, "a record with no service is no longer read as a clone"
    # Every Retry button on the row is behind that gate, not just one of them:
    # the old markup offered it on status alone.
    for hit in re.finditer(r"data-retry=", ROW):
        assert "queued &&" in ROW[:hit.start()][-80:], \
            "a Retry button is offered on status alone"
