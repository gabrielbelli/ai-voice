"""The visual and structural pass: hierarchy, materials, motion, wayfinding.

Static, for the same reason test_playback.py is. What these assert is a
property of the bytes in ui.html -- which selector carries which surface, which
element announces, what a progress bar writes -- and starting a browser to
discover it would add a dependency to a service whose whole claim is that it
has none.

Every test here is named after a defect that was actually in the file, or after
one the change that fixed it could reintroduce. The two that matter most are
the two silent ones: a progress bar that stops moving with no error, and a
control the current route cannot carry disappearing instead of greying.
"""

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "ui.html"
HTML = PAGE.read_text()
CSS = HTML[HTML.index("<style>"):HTML.index("</style>")]
SCRIPT = HTML[HTML.index("<script>"):HTML.index("</script>", HTML.index("<script>"))]


def bare(source: str) -> str:
    """The same text with its comments removed.

    Every comment in this file's subject names the failure it prevents, so it
    quotes the very thing the assertion below forbids -- "was max-width:920px",
    "used to write style.width". A negative assertion over the raw text matches
    the prose and asserts against its own documentation. test_playback.py and
    test_escaping.py strip for exactly this reason.
    """
    return re.sub(r"/\*.*?\*/|<!--.*?-->|//[^\n]*", "", source, flags=re.S)


BARE_CSS = bare(CSS)


def visible(html: str = HTML) -> str:
    r"""The page with its comments gone and its markup intact.

    THE OBVIOUS VERSION OF THIS IS WRONG, and it was wrong here first. Running
    `/\*.*?\*/` over the whole document opens a comment on the `/*` inside
    `accept="audio/*,video/*"`, then closes it on the next `*/` hundreds of
    lines later -- which silently deleted the entire transcription Expert panel
    from what the assertions below were reading. Every negative assertion over
    that region passed because the region was not there.

    So `/* */` is stripped only inside <style> and <script>, where it is a
    comment, and `<!-- -->` is stripped everywhere, where it always is.
    """
    out = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    for opener, closer in (("<style>", "</style>"), ("<script>", "</script>")):
        start = out.index(opener)
        end = out.index(closer, start)
        out = (out[:start]
               + re.sub(r"/\*.*?\*/", "", out[start:end], flags=re.S)
               + out[end:])
    return out


def test_the_comment_strip_does_not_swallow_the_page_it_is_reading():
    """The fence the four tests over `visible()` stand on.

    A stripper that ate the Expert panels would make every "this string is
    gone" assertion pass by deleting the place the string would have been.
    These four markers sit either side of the accept attribute that caused it.
    """
    for marker in ("<summary>Expert: transcription</summary>",
                   "<summary>Expert: Kokoro voices</summary>",
                   "<summary>Expert: Chatterbox voices</summary>",
                   '<label for="x-route">Route</label>'):
        assert marker in visible(), f"the strip ate {marker!r}"


def rule(selector: str) -> str:
    """One rule, from its selector to the first `}`.

    Anchored at a line start, because `details{` is a substring of
    `.card > details{` and the first match would otherwise be the wrong rule.
    """
    start = BARE_CSS.index("\n" + selector) + 1
    return BARE_CSS[start:BARE_CSS.index("}", start)]


# --------------------------------------------------------------- hierarchy --


def test_the_transcript_is_not_the_smallest_text_on_the_page():
    """It was 13px while the drop-zone prompt above it was 14px bold, which is
    the hierarchy inverted: the entire product of the Transcribe tab rendered
    smaller than the control asking for the file. It is also the longest
    continuously-read text here, and the only one measured in thousands of
    words."""
    out = rule(".out{")
    assert "font-size:var(--t-body)" in out, "the transcript is off the type scale again"
    assert "line-height:1.7" in out


def test_the_transcript_leading_clears_the_karaoke_underline():
    """.cue.on carries text-decoration-thickness:2px at text-underline-offset:2px.
    At the body's 1.55 that underline sat in the descenders of the line below --
    on the one surface where losing your place costs a five-thousand-word
    transcript."""
    highlight = rule(".cue.on{")
    assert "text-decoration-thickness:2px" in highlight
    assert "text-underline-offset:2px" in highlight
    leading = float(re.search(r"line-height:([\d.]+)", rule(".out{")).group(1))
    assert leading >= 1.65, f"{leading} leaves the underline in the next line"


def test_the_transcript_pane_holds_a_line_count_and_not_a_pixel_count():
    """max-height:420px shows fewer lines the moment someone raises their text
    size, which is backwards: a bigger setting should not mean less text on
    screen. details.jobtext .body already used em; this follows it."""
    assert "max-height:26em" in rule(".out{")
    assert "max-height:14em" in HTML, "the jobtext precedent is gone"


def test_tracking_is_size_specific_rather_than_declared_twice():
    """The file used to set letter-spacing exactly twice in 211 lines of CSS,
    which means it was wrong at four of its five sizes: large text reads too
    loose as it grows and small text too tight."""
    tracking = re.findall(r"letter-spacing:(-?[\d.]+)em", BARE_CSS)
    assert len(tracking) >= 8, f"only {len(tracking)} tracking declarations"
    values = [float(v) for v in tracking]
    assert min(values) < 0, "nothing on the page tightens; display type is untracked"
    assert max(values) > 0, "nothing on the page loosens; small type is untracked"


def test_the_page_scales_with_the_readers_text_setting():
    """Every dimension used to be px, so a larger OS text size overflowed the
    layout instead of growing it. The measure is the one that matters: it was
    max-width:920px written out twice, on .wrap and on .bar."""
    assert "--measure:57.5rem" in CSS
    assert "max-width:920px" not in BARE_CSS, "the measure is pinned to pixels again"
    assert rule(".wrap{").count("var(--measure)") == 1
    assert rule(".bar{").count("var(--measure)") == 1


# ---------------------------------------------------------------- materials --


def test_the_dark_card_is_not_carried_by_a_shadow_it_cannot_show():
    """A drop shadow does not read on #1b1f26. One .card rule with two
    materials behind a token: a shadow and no border in light, a border and an
    inset top highlight in dark."""
    light = HTML[HTML.index(":root{"):HTML.index("@media (prefers-color-scheme:dark)")]
    dark = HTML[HTML.index("@media (prefers-color-scheme:dark)"):HTML.index("*{box-sizing")]
    for token in ("--lift", "--card-line"):
        assert token in light, f"{token} is not defined for the light theme"
        assert token in dark, f"{token} is not defined for the dark theme"
    assert "inset" in dark[dark.index("--lift"):dark.index("--lift") + 60], (
        "the dark card is back on a drop shadow")


def test_the_translucent_header_falls_back_to_an_opaque_one():
    """ORDER IS THE WHOLE FIX. An engine without color-mix drops that
    declaration and keeps whatever came before it. With the two the other way
    round, the same engine renders a fully transparent sticky header over
    scrolling text -- unreadable, and with nothing on screen saying why."""
    head = rule("header{")
    solid = head.index("background:var(--panel)")
    mixed = head.index("background:color-mix(")
    assert solid < mixed, "the fallback is declared after the thing it backs up"


def test_only_one_surface_on_the_page_is_translucent():
    """Stacked translucency is where legibility collapses, and the header is
    the only element here with content genuinely scrolling under it."""
    declared = re.findall(r"([-\w.#\[\]=>: ]+)\{[^}]*backdrop-filter:blur", BARE_CSS)
    subjects = {s.strip().split("{")[0] for s in declared}
    # @starting-style wraps the backdrop's own opening frame, so it appears as
    # a subject of the same rule rather than as a second surface.
    subjects -= {"starting-style"}
    assert subjects <= {"header", "dialog::backdrop", "dialog[open]::backdrop"}, subjects


def test_the_dialog_is_the_only_thing_that_dims_the_page():
    """It is the one blocking task here. The clone sheet is a parallel,
    non-blocking one and gets a recess instead -- a scrim over it would say
    "you cannot do anything else now", which is false."""
    assert "::backdrop" in BARE_CSS
    assert HTML.count("::backdrop") == CSS.count("::backdrop")
    assert 'id="clone"' in HTML and "<dialog id=\"clone\"" not in HTML


def test_the_clone_sheet_is_a_recess_and_not_a_card_inside_a_card():
    """It used to fake the difference with an inline background:var(--bg) that
    the stylesheet did not express, so the class and the appearance disagreed
    and only one of them was greppable."""
    assert '<div class="card well" id="clone" hidden>' in HTML
    assert 'id="clone" hidden style=' not in HTML, "the inline background is back"
    assert "box-shadow:none" in rule(".card.well{")


def test_the_expert_panels_do_not_look_like_the_primary_path():
    """Panel background, border and radius gave the advanced path the same
    visual weight as the thing the tab is for. Same information, same one click
    away -- it just no longer reads as the point of the screen."""
    panel = rule("details{")
    assert "background:none" in panel
    assert "border:0" in panel and "border-top:1px solid var(--line)" in panel


def test_the_disclosure_marker_turns_rather_than_being_swapped():
    """It was two different characters, which is two states with nothing
    between them. A rotation is an affordance: it says the thing moves."""
    assert 'content:"\\203A"' in BARE_CSS, "the chevron is not one glyph"
    assert "transform:rotate(90deg)" in rule("details[open]>summary::before{")
    assert 'content:"▸ "' not in bare(HTML)


# ------------------------------------------------------------------ motion --


def test_a_press_is_answered_before_the_button_is_released():
    """The page had no press feedback at all: every button was visually inert
    until its handler finished. :active fires on pointerdown, which is the
    whole of the response rule for three lines of CSS."""
    assert "button:active:not(:disabled){transform:scale(" in BARE_CSS
    # .seg{overflow:hidden} clips a scaling child, and a .link is inline text
    # inside a sentence -- scaling it reads as a wobble.
    assert "overflow:hidden" in rule(".seg{")
    assert "transform:none" in rule(".seg button:active:not(:disabled){")
    assert "transform:none" in rule("button.link:active:not(:disabled){")


def test_nothing_on_this_page_overshoots():
    """No interaction here is a drag, a flick or a swipe, so no gesture carries
    momentum to project forward. Overshoot on a control that was merely clicked
    is the misuse the reference names by name."""
    spring = HTML[HTML.index("--spring:"):]
    spring = spring[:spring.index(";")]
    stops = [float(v) for v in re.findall(r"[\d.]+", spring.split("linear(")[1])]
    assert stops == sorted(stops), "the spring curve is not monotonic"
    assert max(stops) <= 1, "the spring overshoots its target"


def test_the_tab_change_never_makes_a_click_wait():
    """An exit animation means the second of two quick tab presses queues
    behind the first. There is deliberately none, and the enter animation runs
    over a state change that has already completed."""
    handler = SCRIPT[SCRIPT.index("const TABS = Array.from"):]
    handler = handler[:handler.index("const scrollEdge")]
    assert 'hidden = name !== button.dataset.tab' in handler
    assert handler.index("hidden = name !==") < handler.index(".animate("), (
        "the panel is animated before it is shown, so the click waits on it")
    assert "finished" not in handler and "await" not in handler


def test_the_tab_change_is_calmer_under_reduced_motion_rather_than_gone():
    """Gentler, not none. The fade aids comprehension and stays; the translate
    is the vestibular part and goes. Read at call time, as follow() does, so
    changing the OS setting mid-session takes effect."""
    handler = SCRIPT[SCRIPT.index("const TABS = Array.from"):]
    handler = handler[:handler.index("const scrollEdge")]
    assert 'matchMedia("(prefers-reduced-motion: reduce)").matches' in handler
    assert "calm ? 0 : 4 * (1 - from)" in handler, "the translate survived"
    assert "opacity: from" in handler, "the fade went with the slide"
    # `from` is 0 on an uninterrupted change, so the fade is unchanged there;
    # it is the LIVE value when a tab is grabbed mid-flight. The literal
    # `opacity: 0` this used to assert was the defect, not the property --
    # see test_the_tab_animation_starts_from_the_presentation_value.


def test_the_tab_animation_starts_from_the_presentation_value():
    """It animated from the TARGET, which the reference calls the single most
    important rule to get right.

    The first keyframe was the literal `opacity: 0` -- not a live read -- so a
    tab grabbed mid-flight jumped. Measured under a stub DOM: a second click at
    t=60ms restarted from 0 while 0.333 was on screen. Two reachable paths, A
    to B to A inside 180ms, and a double-click on one tab, which had no guard
    at all.
    """
    handler = SCRIPT[SCRIPT.index("const TABS = Array.from"):]
    handler = handler[:handler.index("const scrollEdge")]
    assert "panel.getAnimations()" in handler, "no live value is read"
    assert "getComputedTiming()" in handler
    assert "running.cancel()" in handler, "the old animation is left running"
    assert "opacity: from" in handler
    assert "(1 - from)" in handler, "the remaining duration is not shortened"


def test_re_selecting_the_selected_tab_animates_nothing():
    """Without this a double-click on one tab flashed it: there is no state
    change to decorate."""
    handler = SCRIPT[SCRIPT.index("const TABS = Array.from"):]
    handler = handler[:handler.index("const scrollEdge")]
    assert 'button.getAttribute("aria-selected") !== "true"' in handler
    assert "&& changed" in handler


def test_the_asserted_reduced_motion_rule_is_still_one_line_by_itself():
    """test_reduced_motion_removes_the_motion_and_not_the_information reads
    this literal, spaces included. Every rule this pass added lives in a
    SECOND block later in the sheet so that one is never reformatted."""
    literal = "@media (prefers-reduced-motion:reduce){ .cue.on{transition:none} }"
    assert literal in HTML
    blocks = [m.start() for m in re.finditer(r"@media \(prefers-reduced-motion", BARE_CSS)]
    assert len(blocks) == 2, f"{len(blocks)} reduced-motion blocks; the fence needs two"
    assert BARE_CSS.index(literal) < blocks[1], "the new rules were folded into the old block"


def test_a_progress_bar_moves_a_transform_and_not_a_width():
    """THE SILENT ONE. Both write sites are template literals inside an
    innerHTML string -- the MeTube download bar and renderJobs -- so missing one
    leaves a bar frozen at zero with no error anywhere. width also animated
    layout on a two-second timer, and its transition was dead code besides:
    renderJobs reassigns #joblist.innerHTML, so a transition on an element
    created this tick never fires."""
    assert HTML.count('class="bar-fill" style="transform:scaleX(') == 2, (
        "one of the two progress bars still writes a width")
    assert 'class="bar-fill" style="width:' not in HTML
    fill = rule(".bar-fill{")
    assert "transform-origin:left" in fill and "width:100%" in fill


def test_the_progress_cap_was_divided_and_not_dropped():
    """Math.min(95, ...) is a percentage; scaleX takes a fraction. Removing the
    cap instead of scaling it would let a job that overran its estimate report
    itself finished."""
    assert "Math.min(95, elapsed / budget * 100)" in HTML
    assert "(percent / 100).toFixed(4)" in HTML


def test_the_mic_meter_is_painted_from_frames_and_never_overshoots():
    """It wrote style.width from a 100 ms interval: a layout property, sampled
    below the rate of the signal it reports. And it must not be a spring -- a
    meter that overshoots draws a peak that was never in the audio, which is
    the control lying about the microphone it exists to vouch for."""
    body = HTML[HTML.index("function meter(bar, analyser, data)"):]
    body = body[:body.index("\n}\n")]
    assert "requestAnimationFrame(paint)" in body
    assert "cancelAnimationFrame(frame)" in body, "the loop is never stopped"
    assert "target > shown ? target : shown + (target - shown) * 0.25" in body, (
        "instant attack and exponential release are gone")
    assert 'bar.style.transform = "scaleX(' in body
    assert "style.width" not in bare(HTML), "a meter or a bar is back on layout"


def test_both_recorders_stop_their_own_meter():
    """A frame loop left running after the stream's tracks are stopped reads a
    dead analyser sixty times a second for as long as the tab is open."""
    for handle in ("levelStop", "sttLevelStop"):
        assert f"if ({handle}) {{ {handle}(); {handle} = null; }}" in HTML, handle


def test_the_drag_state_changes_all_of_itself():
    """.drop.over swaps the background as well as the border, but only the
    border was in the transition list, so the drag feedback half-changed."""
    drop = rule(".drop{")
    assert "border-color" in drop and "background-color" in drop
    assert "background:var(--bg)" in rule(".drop.over{")


# -------------------------------------------------------------- wayfinding --


def test_where_am_i_does_not_scroll_off_the_screen():
    """The tabs sat in .wrap under a sticky header holding a wordmark and two
    HTML comments. On the Transcribe tab, which is one long scroll, the answer
    to "where am I" left the viewport immediately."""
    header = HTML[HTML.index("<header>"):HTML.index("</header>")]
    assert '<nav class="tabs"' in header, "the tabs are out of the sticky header"
    assert "flex-wrap:nowrap" in rule(".bar{"), (
        "the header can wrap to two rows on a phone and eat the space this "
        "move exists to give back")


def test_the_tablist_points_at_the_panels_it_claims_to_control():
    """role="tablist" was declared with no panels to point at, which is a
    control lying about its own structure -- the attribute promises a
    relationship a screen reader then cannot follow."""
    for name in ("transcribe", "speak", "jobs"):
        assert f'id="tab-btn-{name}"' in HTML
        assert f'aria-controls="tab-{name}"' in HTML
        assert (f'<section id="tab-{name}" role="tabpanel" '
                f'aria-labelledby="tab-btn-{name}"') in HTML


def test_the_tabs_are_one_stop_and_the_arrows_move_between_them():
    """Three stops in front of the panel is what the tablist pattern exists to
    avoid. next.click() rather than a second copy of what a tab does, so
    keyboard and mouse cannot diverge."""
    handler = SCRIPT[SCRIPT.index("const TABS = Array.from"):]
    handler = handler[:handler.index("const scrollEdge")]
    assert "b.tabIndex = b === button ? 0 : -1" in handler
    for key in ("ArrowRight", "ArrowLeft", "Home", "End"):
        assert key in handler, f"{key} does nothing on the tablist"
    assert "TABS[to].click();" in handler, "the keyboard has its own idea of a tab"
    assert 'tabindex="-1"' in HTML, "every tab is in the page's tab order at boot"


def test_every_status_host_announces_what_it_writes():
    """The page had zero aria-live attributes on an interface whose entire
    subject is work that takes minutes. note() already writes innerHTML into
    these, so marking the host announces on mutation for the cost of an
    attribute. polite and not assertive: they are status, and #stt-note
    sometimes contains the Stop button."""
    hosts = ("stt-note", "speak-note", "clipnote", "cliplinkhint",
             "linkhint", "jobplaying", "result-meta")
    for host in hosts:
        tag = HTML[HTML.index(f'id="{host}"'):]
        tag = tag[:tag.index(">")]
        assert 'aria-live="polite"' in tag, f"#{host} changes silently"
    assert 'aria-live="assertive"' not in HTML


def test_the_scroll_edge_cannot_hold_up_a_scroll():
    """A non-passive scroll listener lets the page block the compositor, which
    is the one way a five-line decoration becomes a performance bug."""
    assert 'addEventListener("scroll", scrollEdge, { passive: true })' in HTML
    assert "body.scrolled header{box-shadow:" in BARE_CSS


# ------------------------------------------------------------ the contract --


def test_no_selector_hides_a_disabled_control():
    """THE STANDING RULE, and the reason the greying pass is CSS and not JS.
    expertFormat(), granularities() and segments() read .disabled and .hidden
    as the authority for what to send -- chosenFormat() was the third reader
    until the Transcript format control it served was retired, and the rule
    outlived it. A control the current route cannot carry
    is greyed WITH THE REASON beside it; a control that vanishes takes its
    explanation with it, and fifteen defects of that shape were fixed once."""
    for offender in re.finditer(r"([^{}]*:disabled[^{}]*)\{([^}]*)\}", BARE_CSS):
        selector, body = offender.group(1), offender.group(2)
        assert "display:none" not in body, selector
        assert "visibility:hidden" not in body, selector


def test_a_greyed_field_greys_as_one_thing():
    """The control was disabled and its label and hint left at full strength
    above it, which reads as "the label is live and the control is broken"
    rather than as one unavailable field."""
    assert ".grid2 > div:has(:disabled) > label" in BARE_CSS
    # And a floor underneath it, because the :has() rule is coupled to that
    # nesting and stops matching silently if the markup is ever restructured.
    assert "input:disabled,select:disabled,textarea:disabled{opacity:.5" in BARE_CSS


def test_the_focus_ring_cannot_reach_the_transcript_spans():
    """.cue spans are deliberately not focusable -- a five-thousand-word
    transcript is five thousand tab stops. The ring is scoped by tag for
    exactly that reason, and :where() so it costs no specificity."""
    assert (":where(button,select,input,textarea,summary,[role=tab]):focus-visible"
            in BARE_CSS)
    assert ".cue:focus" not in BARE_CSS


def test_the_owl_rule_does_not_open_a_gap_where_nothing_is():
    """#stt-note, #speak-note, #codeswitch and #clipnote are empty most of the
    time. Without the guard the rule that replaced their inline margins spaces
    the page around elements with nothing in them."""
    assert ".card > * + *{margin-top:var(--s3)}" in BARE_CSS
    assert ".card > *:empty{margin-top:0}" in BARE_CSS
    # And a label stays welded to the control it names.
    assert ".card > label + *{margin-top:0}" in BARE_CSS


def test_the_interface_gained_no_new_prose():
    """The user has twice asked, in those words, for interface copy to be cut.
    This pass shortens one string and makes one more specific; it adds none."""
    # BOTH NAMES ARE GONE, not just the bad one. "Output" was the
    # generic-and-safe failure and was renamed to "Transcript format"; the
    # control under it has since been retired outright, because one
    # verbose_json run already produces the transcript, the .srt and the .vtt,
    # so the three buttons were a choice between a default and two worse
    # versions of it. Prose removed is prose removed.
    assert "<label>Output</label>" not in HTML
    assert "<label>Transcript format</label>" not in HTML
    assert 'id="fmt"' not in HTML, "the retired format control is back"
    assert "data-fmt" not in HTML
    assert '"(" + list.filter' not in HTML, "the job count is a sentence again"

def test_the_institutional_memory_moved_into_comments_and_was_not_lost():
    """THIS ASSERTION IS THE INVERSE OF THE ONE IT REPLACES, on purpose.

    Four .note flat paragraphs sat in the expert panels holding why denoise is
    absent, why there is no language control, why Chatterbox's sliders are not
    at Resemble's values, and why submission is always POST /jobs. They were
    kept on screen so nobody would add denoise back or copy Resemble's ranges,
    and the previous version of this test pinned them word for word.

    The user has now asked a third time for interface copy to be cut, and named
    those paragraphs: they are commit messages that escaped onto the screen.
    Every one of them explains WHY the software is built this way, cites a
    measurement, or argues with a hypothetical reader. None of them is
    something the person transcribing a file needs at that moment.

    So the reasoning moved to the comment above the element it described, which
    is where the next reader looks, and this test is what stops the move from
    being a deletion. Both halves are asserted: gone from the rendered page,
    still in the file.
    """
    screen = visible()
    # Gone from what a person reads.
    for cut in ("This note exists so nobody adds it back.",
                "it is the only correct behaviour",
                "reconciled, not copied",
                "The sliders are not broken",
                "a finding no slider can express",
                "a naive client writes the third into a .wav",
                "They are placeholders, not measurements"):
        assert cut not in screen, f"{cut!r} is still on the page"
    # And every measured figure in them survived the move. Whitespace is
    # collapsed first: a comment wraps at a different column than the markup it
    # replaced, so matching the raw text would pin the line breaks rather than
    # the facts.
    file = re.sub(r"\s+", " ", HTML)
    for kept in ("+26% mean WER", "worse in 9 of 13 conditions",
                 "WER 1.0 from hallucination",
                 "agreement collapsed to 0.017",
                 "accepts_language = False",
                 "/etc/stt-stack/glossary.txt",
                 "0.5 / 0.5 / 0.8", "ge=0.0, le=1.0",
                 "reconciled, not copied",
                 "closing that stream cancels the job"):
        assert kept in file, f"{kept!r} was lost rather than moved"


def test_no_user_facing_string_carries_an_em_dash():
    """The em dash was the loudest tell in this file: 27 of them, and every one
    in a sentence a person reads.

    IT IS BANNED FOR WHAT IT DOES, not for how it looks. A dash hides a second
    clause inside a first one, so the reader meets the qualification after they
    have already acted on the main verb -- "Keep the video -- a much bigger
    download" is a decision and its cost welded into one breath. The
    replacements are a full stop, a comma, a colon or brackets, and choosing
    between them forces the writer to say which relationship the halves are in.

    COMMENTS ARE STRIPPED FIRST and are exempt. They are not user-facing, they
    hold the reasoning this pass moved off the screen, and rewriting the record
    to match a style rule about the interface would be the wrong trade.
    """
    assert "\u2014" not in visible(), "an em dash is back in interface copy"


def test_no_control_is_explained_by_a_paragraph_beside_it():
    """The user asked three times for interface copy to be cut, and the third
    time named the failure: paragraphs of explanation next to the controls.

    Each string here was on the page. Each one explains WHY the software is
    built this way, cites a measurement, names a file path, or argues with a
    reader who has not complained yet. None of them is something the person
    transcribing a file needs at the moment they are transcribing it. They are
    in comments now, beside the code that made them true.
    """
    screen = visible()
    for paragraph in (
            "There is no preprocessing stage in this codebase",
            "The glossary is a startup file",
            "No language control, in either mode",
            "This deployment is",
            "Submission is always",
            "diarized_json is a 400 on this stack",
            "Chatterbox runs at about",
            "Glossary changes are not reported",
            "time-to-first-audio",
            "X-Ignored-Parameters",
            "Three sliders at non-stock values",
            "Links work, but there will be no length or size",
            "speech in, speech out",
    ):
        assert paragraph not in screen, f"{paragraph!r} is back on the page"


def test_a_hint_under_a_control_stays_one_sentence_long():
    """A ceiling with the reason for it attached.

    Every static hint and note in the markup is measured with its tags removed
    and its whitespace collapsed. The longest survivor of this pass is 78
    characters. The longest string it replaced was 423, and four more ran past
    200. A hundred is the fence: it passes everything the page says now, and it
    fails the shortest of the paragraphs that were cut.
    """
    markup = HTML[HTML.index("<body>"):HTML.index("<script>")]
    markup = re.sub(r"<!--.*?-->", "", markup, flags=re.S)
    pattern = r'<(div|span)[^>]*class="(?:note[^"]*|hint[^"]*)"[^>]*>(.*?)</\1>'
    seen = 0
    for match in re.finditer(pattern, markup, re.S):
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", match.group(2))).strip()
        if not text:
            continue
        seen += 1
        assert len(text) <= 100, f"{len(text)} characters of hint: {text!r}"
    assert seen > 8, "the scan stopped finding hints, so it proves nothing"


def test_nothing_on_screen_sends_the_reader_in_a_direction():
    """"Trim it below", "the transcript above", "follow the control above".

    A direction is only true for the writer's own screen. A reader on a phone,
    or one who has scrolled, or one using a screen reader, is not looking at
    what the writer was looking at. Naming the control costs the same number of
    words and survives every layout: "Set Start at and Stop at to trim it".
    """
    screen = visible()
    for direction in (r"\babove\b", r"\bbelow\b", r"[Cc]lick here"):
        found = re.search(direction, screen)
        assert not found, f"the page points at {found.group(0)!r} rather than a control"


def test_every_pressable_surface_acknowledges_the_press():
    """`:active` reached <button> and nothing else.

    A transcript word is the most-clicked surface on the Transcribe tab -- it
    seeks the player -- and it committed on release with no feedback on press.
    A disclosure summary was the same. Both take cursor:pointer, which is the
    promise this was failing to keep.
    """
    assert ".cue:active{" in BARE_CSS
    assert "details>summary:active{" in BARE_CSS


def test_the_cue_press_does_not_move_its_neighbours():
    """Opacity, not scale: a word inside a line of text cannot shrink without
    shifting the words beside it."""
    rule = BARE_CSS[BARE_CSS.index(".cue:active{"):]
    rule = rule[:rule.index("}")]
    assert "transform" not in rule and "scale" not in rule


def test_the_caption_plate_answers_the_transparency_queries():
    """It is the one translucent surface both queries missed, and the one over
    moving video -- which is exactly where raising opacity matters."""
    for query in ("prefers-reduced-transparency:reduce", "prefers-contrast:more"):
        block = BARE_CSS[BARE_CSS.index("@media (" + query + ")"):]
        block = block[:block.index("\n}")]
        assert ".band span" in block, f"{query} does not reach the caption plate"


def test_no_control_refers_to_a_label_that_no_longer_exists():
    """An <option> read "follow the Output control above" after that control
    had been renamed Transcript format. A pointer to a name nothing carries is
    worse than no pointer."""
    import re as _re

    labels = set(_re.findall(r"<label[^>]*>([^<]+)</label>", HTML))
    labels = {l.strip() for l in labels}
    for referenced in _re.findall(r"follow (?:the )?([A-Z][A-Za-z ]+?)(?: control)? above", HTML):
        assert referenced.strip() in labels, \
            f"an option points at {referenced!r}, which is not a label on this page"


def test_one_term_per_concept_survives_into_the_result_lines():
    """The control is Vocabulary and the button is Delete. The lines that
    report what happened used to say Glossary and Removed.

    Both leaked the same way: the internal name is glossary everywhere (the
    query parameter, the form field, loadGlossaries), and an earlier pass
    renamed the voice button to Delete for exactly this reason and then missed
    its own success line. One concept, one word, on the way in and on the way
    out.
    """
    assert "Glossary rewrote" not in HTML
    assert "Vocabulary rewrote" in HTML
    assert "Removed <strong>${voice.name}" not in HTML
    assert "Deleted <strong>${voice.name}" in HTML


def test_the_page_says_once_that_a_job_survives_the_tab_closing():
    """It used to be inside the job row, so three running jobs said it three
    times. It is one fact about this page, not a property of any one job.

    It now lives above the list and is shown only while something is running,
    which is the only time it means anything.
    """
    assert HTML.count("Close this page if you want") == 1
    assert "You can close this page. The job runs" not in HTML
    assert 'id="jobsafe"' in HTML
    # Above the list, not inside the markup renderJobs assigns.
    assert HTML.index('id="jobsafe"') < HTML.index('<div id="joblist">')
    assert '$("jobsafe").hidden = !running;' in HTML


def test_no_interaction_word_assumes_a_mouse():
    """ASD-STE100 bans click, swipe and the directional words with it: the
    person following the caption highlight may be on a keyboard or a screen
    reader, and there is nothing to click there.
    """
    # visible() leaves JS *code* intact, and addEventListener("click") is not
    # copy. So this looks for the word where a reader would meet it: opening a
    # string literal, or opening a text node.
    # visible() leaves JS *code* intact, and addEventListener("click") is not
    # copy. The word only becomes an instruction when prose follows it, which
    # is what this looks for.
    screen = visible()
    for word in ("Click", "click", "Swipe", "swipe"):
        found = re.search(rf"\b{word}\s+[a-z]", screen)
        assert not found, f"{word!r} is back on the page: {found.group(0)!r}"


def test_no_string_explains_a_deployment_state_to_the_reader():
    """"the metadata probe is off in this image" named nothing the reader can
    see, used container jargon, and described something they cannot change.
    The sentence without it says the same useful half: there is no estimate.
    """
    screen = visible()
    assert "metadata probe" not in screen
    assert "in this image" not in screen, "container jargon is back on the page"
    assert "No length or size for this link. The estimate is unavailable." in HTML


def test_every_rate_on_this_page_is_one_that_was_measured():
    """Three of the four rates here were quoted from a README rather than
    measured, and three of them were wrong.

    Against the deployed stack, same host, same day: STT 8.81x on a 297 s clip
    (the 8.5 seed is 4% out and stays), Kokoro 1.83x where the page said 1.3,
    Chatterbox 0.275x where the page fell back to 0.138. The last two were 41%
    and 100% out, and both were quoted from a timeout comment in the gateway.
    """
    # Scoped to the rate table. Both old figures still appear elsewhere in the
    # file, quoted inside comments that record what they were and why they
    # were wrong -- which is the point of keeping them.
    table = HTML[HTML.index("const rate = {"):HTML.index("const CHARS_PER_SECOND")]
    assert "kokoro: () => 1.83" in table, "the measured Kokoro rate"
    assert "1.3," not in table, "the quoted 1.3 is back in the rate table"
    assert "return f > 0 ? f : 0.275;" in table, "the measured clone fallback"
    assert "0.138" not in table, "the old clone fallback is back"


def test_the_two_engines_do_not_share_one_speech_rate():
    """One constant for both was 36% out on one of them: Kokoro measured 16.3
    chars/s and Chatterbox 12.0 on the same host on the same day.

    It matters most for the clone, where the audio length is then divided by a
    realtime factor near 0.27 -- so the error reaches the reader multiplied.
    """
    assert "const CHARS_PER_SECOND = { kokoro: 16.3, clone: 12.0 };" in HTML
    # Both call sites name their engine; a bare call would silently take the
    # Kokoro rate for a Chatterbox job.
    assert 'speechSeconds(text, "clone")' in HTML
    assert 'speechSeconds(text, "kokoro")' in HTML
    assert "speechSeconds(text)" not in HTML, "an unnamed engine takes the wrong rate"


def test_the_page_does_not_pretend_to_learn_a_rate_it_cannot_learn():
    """rate.learn() had exactly one caller, inside `if (format === "native")`.

    The page stopped taking the native route when the Transcript format
    control was retired, so the average corrected nothing and rate.stt()
    returned the seed for ever. A learner that never learns is worse than a
    constant, because the constant does not invite trust it has not earned.
    """
    # visible() rather than HTML: the comment on `rate` quotes both names to
    # record what was removed, and a fence that cannot tell a live call from
    # its own obituary is not a fence.
    live = visible()
    assert "rate.learn" not in live, "the dead learner is back on the page"
    assert 'store.get("rtf.stt"' not in live
    assert 'store.set("rtf.stt"' not in live
    assert "stt: () => CONFIG.stt_rtf_seed," in HTML
