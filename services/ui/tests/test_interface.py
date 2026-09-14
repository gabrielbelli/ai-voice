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


def bare_css(source: str) -> str:
    r"""The same, for a STYLESHEET, where `//` is not a comment.

    A REAL DEFECT IN THIS HARNESS, found by the structural test below. bare()
    also strips `//` to end of line, because it is shared with the script --
    but CSS has no line comment, and `//` occurs constantly inside a url():
    every `https://`, and every third character of a base64 payload.

    The display face is a 15 KB woff2 inlined as a data: URI, so the first
    `//` in its base64 deleted the remaining 19 KB of that line -- including
    the `}` that closes @font-face. Every rule after it was then read as
    nested one block deep, and every negative assertion over BARE_CSS below
    that point ("x is not in the sheet") passed because the text it was
    scanning had been truncated. A test that passes because its input is empty
    is worse than no test.
    """
    return re.sub(r"/\*.*?\*/", "", source, flags=re.S)


BARE_CSS = bare_css(CSS)
# The guard, because the failure mode above is silent by construction: if the
# stripper ever eats the sheet again, this says so at import rather than
# letting forty assertions quietly stop scanning anything.
assert len(BARE_CSS) > len(CSS) * 0.5, "the comment stripper ate the stylesheet"
assert BARE_CSS.count("{") == BARE_CSS.count("}"), "the stripped sheet is unbalanced"


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
    assert "--measure:46rem" in CSS
    assert "max-width:920px" not in BARE_CSS, "the measure is pinned to pixels again"
    assert rule(".wrap{").count("var(--measure)") == 1
    # THE HEAD IS A MASTHEAD NOW, not a bar: the navigation left the top of the
    # page for a dock at the foot of the viewport, and what is left up there is
    # the mark and the name of the section. It still has to share the column's
    # measure or the <h1> would not line up with the cards under it.
    assert rule(".masthead{").count("var(--measure)") == 1
    # The dock is the one thing on the page deliberately NOT on the measure: it
    # is a fixed object sized to the thumb, not to the text column.
    assert "width:min(410px," in rule(".dock{")


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


def test_the_dock_falls_back_to_a_plain_bar_without_its_script():
    """ORDER IS THE WHOLE FIX, AND IT OUTLIVED THE HEADER IT WAS WRITTEN FOR.
    The bar's real outline is an SVG path cut by JS, so the element carries a
    plain rounded background first and the script removes it by putting .js on
    the root. Declared the other way round -- or gated on a class that is added
    before the path exists -- an engine that never runs the script renders four
    icons floating on the page with no bar under them at all."""
    assert "background:var(--plate)" in rule(".dock{"), \
        "the dock has no fallback if its script never runs"
    assert "border-radius:var(--dock-r)" in rule(".dock{")
    plain = CSS.index(".dock{")
    stripped = CSS.index(".js .dock{background:none}")
    assert plain < stripped, "the fallback is declared after the thing it backs up"
    # And the class that strips it is added by the script itself, not printed
    # into the markup, so "the script ran" is the only thing that can remove it.
    assert 'document.documentElement.classList.add("js")' in SCRIPT
    assert '<html lang="en">' in HTML and 'class="js"' not in HTML[:200]


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
    handler = handler[:handler.index("async function poll()")]
    # The hide/show used to be `hidden = name !== button.dataset.tab` over a
    # hand-written list of tab names; it derives the set from TABS now, because
    # adding a fourth tab to that list was forgotten and the button emptied the
    # page. The ORDER is what this test is about and it is unchanged.
    assert "panel.hidden = b !== button" in handler
    assert handler.index("panel.hidden = b !== button") < handler.index(".animate("), (
        "the panel is animated before it is shown, so the click waits on it")
    assert "finished" not in handler and "await" not in handler


def test_the_tab_change_is_calmer_under_reduced_motion_rather_than_gone():
    """Gentler, not none. The fade aids comprehension and stays; the translate
    is the vestibular part and goes. Read at call time, as follow() does, so
    changing the OS setting mid-session takes effect."""
    handler = SCRIPT[SCRIPT.index("const TABS = Array.from"):]
    handler = handler[:handler.index("async function poll()")]
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
    handler = handler[:handler.index("async function poll()")]
    assert "panel.getAnimations()" in handler, "no live value is read"
    assert "getComputedTiming()" in handler
    assert "running.cancel()" in handler, "the old animation is left running"
    assert "opacity: from" in handler
    assert "(1 - from)" in handler, "the remaining duration is not shortened"


def test_re_selecting_the_selected_tab_animates_nothing():
    """Without this a double-click on one tab flashed it: there is no state
    change to decorate."""
    handler = SCRIPT[SCRIPT.index("const TABS = Array.from"):]
    handler = handler[:handler.index("async function poll()")]
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
    created this tick never fires.

    THE COUNT IS ALSO A CEILING, and that half is new. The jobs tab lists
    instant speech and transcriptions now, and both are terminal by
    construction -- an imported record has already finished, so there is
    nothing left for a bar to be a fraction of. A third bar here would be one
    drawn at 100% the moment its row appeared, which teaches a reader to
    distrust the two that mean something."""
    assert HTML.count('class="bar-fill" style="transform:scaleX(') == 2, (
        "one of the two progress bars still writes a width, "
        "or a finished row has grown a bar of its own")
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
    to "where am I" left the viewport immediately.

    IT IS A FIXED DOCK NOW RATHER THAN A STICKY STRIP, which is the stronger
    form of the same fix: sticky still moves within its own containing block
    and still shares a row with whatever else is up there. The dock is pinned
    to the viewport at every width and every scroll position."""
    rail = rule(".rail{")
    assert "position:fixed" in rail, "the navigation can scroll away again"
    assert "bottom:calc(22px + env(safe-area-inset-bottom))" in rail, \
        "the dock is not held off the bottom edge, or ignores the home indicator"
    # The tablist is inside the dock, and the dock is inside the rail.
    dock = HTML[HTML.index('<div class="rail">'):HTML.index("<!-- ==================================================== /the dock ====")]
    assert '<div class="tabs" role="tablist"' in dock, "the tabs are out of the dock"
    # THE RAIL IS TRANSPARENT TO THE POINTER AND THE DOCK IS NOT. A full-width
    # fixed strip that swallowed clicks would make the bottom of every panel
    # unusable, which is the cost of centring this way.
    assert "pointer-events:none" in rail
    assert "pointer-events:auto" in rule(".dock{")


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
    handler = handler[:handler.index("async function poll()")]
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
             "linkhint", "jobplaying", "result-meta", "speak-meta")
    for host in hosts:
        tag = HTML[HTML.index(f'id="{host}"'):]
        tag = tag[:tag.index(">")]
        assert 'aria-live="polite"' in tag, f"#{host} changes silently"
    assert 'aria-live="assertive"' not in HTML


def test_nothing_on_this_page_listens_to_a_scroll():
    """A non-passive scroll listener lets the page block the compositor, which
    is the one way a five-line decoration becomes a performance bug. The page
    used to have exactly one, marked passive, toggling body.scrolled so a
    translucent sticky header could grow a shadow only while something was
    genuinely underneath it.

    THE HEADER IS GONE AND SO IS THE LISTENER. The navigation is an opaque
    plate fixed at the foot of the viewport, nothing scrolls behind anything,
    and the edge was drawing a line on nothing. No listener at all is the
    strongest form of the rule, so the test asserts that rather than the flag."""
    assert "scrollEdge" not in SCRIPT, "the dead scroll edge came back"
    assert "body.scrolled" not in BARE_CSS
    assert 'addEventListener("scroll"' not in SCRIPT, \
        "a scroll listener is back; if it must exist it has to be passive"


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

    THE KOKORO FIGURE MOVED, AND THAT IS WHY THIS DOCSTRING CHANGED RATHER THAN
    THE TEST BEING WEAKENED. 1.83x was measured at TTS_THREADS=4. The same
    weights and the same 899-character line measured 2.79x at 8, so a rate here
    is a claim about a machine's configuration and has to name the machine and
    the thread count that produced it. Both are true; neither is the rate.

    So the constant is a fallback now and not the only figure: tts-stack
    reports its own realtime factor, absent until it has actually synthesised
    something, and the page prefers what it has seen over what it assumed. The
    property this test defends is unchanged -- every number in this table is
    one somebody measured, and each says where.
    """
    # Scoped to the rate table. The old figures still appear elsewhere in the
    # file, quoted inside comments that record what they were and why they
    # were wrong -- which is the point of keeping them.
    table = HTML[HTML.index("const rate = {"):HTML.index("const CHARS_PER_SECOND")]
    assert "return f > 0 ? f : 2.79;" in table, "the measured Kokoro fallback"
    assert "TTS_THREADS=8" in table, "a rate constant that does not name its machine"
    assert "1.3," not in table, "the quoted 1.3 is back in the rate table"
    assert "return f > 0 ? f : 0.275;" in table, "the measured clone fallback"
    assert "0.138" not in table, "the old clone fallback is back"
    # AND THE PER-BACKEND READ, which is the whole reason the table moved. The
    # merged realtime_factor averages a NAS CPU at 0.21 with a GPU at 0.7 and
    # describes no machine that exists, so the rate the rule reads is keyed on
    # the backend the next job will actually run on.
    assert "realtime_factor_by_backend" in HTML
    assert "function cloneBackend(engine)" in HTML
    # AND IT IS KEYED PER ENGINE AS WELL AS PER LANE, because one lane running
    # two engines 2.36x apart has two rates and an average of them describes
    # neither. The lane-only map is still read, for a tts-long that publishes
    # only that one.
    assert "realtime_factor_by_engine" in HTML
    # A backend with no key has never been measured, and that is a state rather
    # than a zero: the live payload carries {"local": 0.21} and no "runner" key
    # at all while the runner sits there able to run.
    body = HTML[HTML.index("function cloneRate(engine)"):]
    body = body[:body.index("\n}\n")]
    assert "f > 0 ? f : null" in body, "an unmeasured backend gets a default number"


def test_the_lane_is_never_decided_by_a_service_id_this_file_spells():
    """`r.service === "chatterbox"` was the fourth model table on this stack.

    It failed in the worst direction. An unrecognised runner service id -- a
    deployment that named it chatterbox-gpu, or the second engine's own service
    -- read as "local", so the page showed the NAS rate for a job about to run
    on the card: confidently, with no sign anything had been assumed, on the
    number every Chatterbox estimate here is divided by.

    The engine block answers it per engine now, because with two engines the
    runner can serve one and not the other; without that block there is one
    engine and the runner's own can_run is the whole answer.
    """
    body = bare(HTML[HTML.index("function cloneBackend(engine)"):])
    body = body[:body.index("\n}\n")]
    assert '"chatterbox"' not in body, "a service id is spelled in the page again"
    assert "spec.runner" in body, "the per-engine readiness is not read"
    assert "r.can_run" in body, "the one-engine deployment lost its answer"


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


def test_auto_detect_says_what_it_detected():
    """resolvedLanguage() has always returned `why` beside the code, and until
    now nothing read it.

    The running commentary that used to sit under the picker was cut as slop,
    and rightly: it was a sentence restating a control. The answer is not a
    sentence somewhere else, it is the control reporting what it resolved to,
    the way a translator does. Only the Auto-detect option is rewritten, and
    only while it is selected.
    """
    assert "function showDetected()" in HTML
    assert '`Auto-detect (${name})`' in HTML
    # Fed from the same list the options are built from, so the name in the
    # label and the name in the list cannot drift apart.
    assert "LANG_NAMES = Object.fromEntries(merged);" in HTML
    # Only a real reading is named. `unsupported` recognised a language this
    # stack cannot speak and `undetected` could not tell; naming the fallback
    # for either would report a guess as a reading.
    assert 'resolved.why === "detected"' in HTML
    # Driven from estimate(), which is already wired to every input, segment
    # edit and voice change -- a second listener is a second chance to
    # disagree with the code actually sent.
    assert "  showDetected();" in HTML


def test_the_runner_panel_asks_the_three_questions_separately():
    """Is it there, will it take work, and if not why not.

    Those are three different questions and a single status word answers none
    of them well. The state is the headline, the runner's own reason is its own
    line, and "not answering" is a state rather than an error, because the
    machine is somebody's desktop and switched off is ordinary.
    """
    assert 'id="runnerbox"' in HTML
    assert "function paintRunner()" in HTML
    # Drawn from the health poll the page already runs. A second request would
    # be a second thing to fail and a second thing to disagree with.
    assert "  paintRunner();" in HTML
    # THE PATH, PINNED. /ui/health is the UI's own document and it WRAPS the
    # gateway's, so tts-long sits at gateway.health.backends.tts_long.health.
    # The panel was written against the gateway's /health, read one level too
    # shallow, found undefined and hid itself on a machine with a runner
    # answering beside it. Both readers go through one accessor now.
    assert "function ttsLongHealth()" in HTML
    assert "HEALTH.gateway.health" in HTML.replace("HEALTH && HEALTH.gateway && HEALTH.gateway.health",
                                                   "HEALTH.gateway.health")
    assert "const r = ttsLongHealth().runner;" in HTML
    assert "ttsLongHealth().realtime_factor" in HTML, "the rate reader drifted off the accessor"
    # visible(), not HTML: the comment above the accessor quotes the wrong
    # path deliberately, to record what it was.
    assert "HEALTH.backends" not in visible(), "the shallow path is back"
    # Hidden completely when no runner is configured: a panel that always says
    # "none" is furniture.
    assert "if (!r) { box.hidden = true; return; }" in HTML
    # The reason is the runner's words, not a guess from the state name.
    assert "r.reason" in HTML


def test_every_tab_button_has_a_panel_and_the_switch_finds_it():
    """THE DEFECT THIS PREVENTS: a tab that empties the page.

    The switch hid and showed panels from a list written out in the script,
    ["transcribe","speak","jobs"]. Adding Vocabulary as a fourth tab gave a
    button that set aria-selected, hid the panel you were on, and never showed
    the new one, because its name was not in that list. The button worked; the
    page went blank.

    Every tab is in TABS by definition and its panel id is "tab-" plus its
    data-tab, so the switch derives the set instead of being told it.
    """
    # visible(), because the comment on the fix quotes the old list on purpose.
    assert '["transcribe","speak","jobs"]' not in visible(), \
        "the hand-written list is back"
    assert 'const panel = $("tab-" + b.dataset.tab);' in HTML

    # And the markup keeps its half of the bargain: one panel per button.
    buttons = re.findall(r'role="tab"[^>]*data-tab="([a-z]+)"', HTML)
    assert len(buttons) >= 4, buttons
    for name in buttons:
        assert f'id="tab-{name}"' in HTML, f"the {name} tab has no panel"
        assert f'aria-controls="tab-{name}"' in HTML


def test_managing_vocabulary_is_a_tab_and_choosing_it_is_not():
    """The user asked for a tab after it shipped as a disclosure on Transcribe.

    The two controls are deliberately not together: PICKING a profile happens
    while you are setting up a transcription, and EDITING one does not.
    """
    assert 'id="tab-btn-vocab"' in HTML
    assert 'id="tab-vocab"' in HTML
    # The manager moved; the chooser did not.
    assert HTML.index('id="glossbox"') < HTML.index('id="tab-vocab"'), \
        "the chooser left the Transcribe panel"
    assert HTML.index('id="tab-vocab"') < HTML.index('id="glossman"'), \
        "the manager is not inside its own tab"


# ================================= a wait that can be watched, and left =====


def _tag(source: str, element_id: str) -> str:
    """The one opening tag that carries this id, whole.

    Sliced backwards from the id to its own "<": counting characters instead
    lands in the middle of the tag before it, which is how the first version of
    this test read a row's style attribute and passed on nothing.
    """
    at = source.index(f'id="{element_id}"')
    return source[source.rindex("<", 0, at):source.index(">", at) + 1]


def test_a_ticking_figure_never_lands_in_a_live_region():
    """THE aria-live TICKER TRAP, and it is the one accessibility defect this
    whole feature was most likely to ship.

    A bar and a counter redrawn four times a second inside aria-live="polite"
    announce four times a second and make the page unusable with a screen
    reader. The split is: the announcing host says the state once
    ("Transcribing...", then the result), and the bar and the moving figure sit
    in an aria-hidden span where they are for eyes only.

    The Stop button is in neither: it is a control, it must stay reachable, and
    it is outside the hidden span for exactly that reason.
    """
    for host, ticker in (("stt-progress", "stt-elapsed"), ("speak-progress", "speak-lead")):
        block = HTML[HTML.index(f'id="{host}"'):]
        block = block[:block.index("</div>\n      </div>")]
        assert 'aria-live' not in block, f"#{host} announces on every tick"
        assert 'aria-hidden="true"' in _tag(block, ticker), \
            f"#{ticker} is announced as it counts"
        assert 'class="bar-track" aria-hidden="true"' in block, \
            f"#{host}'s bar is announced as it moves"
    for control in ("stt-cancel", "speak-stop"):
        assert "aria-hidden" not in _tag(HTML, control), \
            f"#{control} is hidden from a screen reader"


def test_the_transcription_wait_is_a_bar_and_a_way_out():
    """A ten minute recording is about 68 seconds of compute at the measured
    8.81x, and for all of it the page showed four words and a disabled button.

    Parakeet cannot stream: asr.py sets can_stream False, stream() raises, and
    openai_api.py refuses stream=true by name with the measurement behind it,
    5.07 s to the first and only output on a 14.2 s clip. So this is a progress
    problem and not a partials problem, and cutting a finished transcript into
    timed fake deltas is the one thing the service already refuses to do.

    The bar is driven by duration / rate.stt(), which is the same expression
    paintFacts already quotes on the confirm card. Two opinions about the same
    wait would be worse than none.
    """
    assert 'id="stt-progress"' in HTML
    body = HTML[HTML.index("function sttProgress()"):]
    body = body[:body.index("\n}\n")]
    assert "sttSeconds() / rate.stt()" in body, "the bar and the card disagree"
    # The same two sentences the jobs bar uses, so one reading serves both.
    assert "past the " in body and " estimate" in body
    # No measured duration means no bar rather than a bar moving at a rate
    # nobody measured.
    assert "!budget ? clock(elapsed)" in body
    # AND A WAY OUT. Sixty-eight seconds with no escape is half of what makes a
    # wait feel long. One controller over the single await, on both routes.
    assert "sttAbort = new AbortController();" in HTML
    assert HTML.count("body: form, signal }") == 1
    assert 'err.name === "AbortError"' in HTML


def test_the_page_invents_no_transcription_partials():
    """The frames exist and Parakeet refuses them by name. Nothing here may
    manufacture the deltas the service will not produce."""
    live = visible()
    assert "transcript.text.delta" not in live
    # And the request never asks for a stream the loaded model cannot give.
    form = HTML[HTML.index('$("go-stt").addEventListener'):]
    form = form[:form.index("/* ================================================== karaoke")]
    assert 'append("stream"' not in form


def test_detecting_a_language_narrows_the_voices():
    """THE DEFECT THIS PREVENTS, reported from a screenshot: the picker said
    "Auto-detect (Portuguese)" above a list of every voice on the machine.

    Narrowing the list is what detecting a language is FOR, and two things
    stopped it. The filter read the control, which says the literal "auto", so
    it never applied for anybody who left the default alone, which is
    everybody. And nothing rebuilt the picker when the detection changed:
    loadVoices is the only thing that builds it and it ran on a language
    change, which auto-detect never causes.

    Re-fetching per keystroke would have been two network calls per character.
    The list is already in VOICES, so rendering is free and only the fetch
    stays rare.
    """
    assert "function renderVoices()" in HTML, "the render half was never split out"
    assert 'resolved.why === "detected" && resolved.iso ? resolved.iso : "auto"' in HTML
    # Only when the answer changes: this runs on every keystroke, and
    # rebuilding a 54-option select each time fights the person using it.
    assert "if (stem !== _detectedStem) {" in HTML
    assert "renderVoices();" in HTML


def test_the_speed_slider_and_the_delete_button_share_one_slot():
    """They overlapped on screen, and they can never both be meaningful.

    Delete exists only for a cloned voice, and `speed` is a Kokoro request
    field that Chatterbox answers with a 400. So a clone could never use the
    slider and a Kokoro voice could never use the button: a clone was being
    shown a control that could not work even before it collided with the one
    next to it.
    """
    assert 'id="speedslot"' in HTML
    assert 'id="speedwrap"' in HTML
    assert '$("speedwrap").hidden = !$("delvoice").hidden;' in HTML


def test_streaming_is_the_default_when_the_arithmetic_allows_it():
    """THE DEFECT THIS PREVENTS, reported as "I still have to wait for the
    whole speak to finish": streaming was built and never happened.

    Both controls that turn it on defaulted to the other option, and both sit
    inside a collapsed Expert panel. A streaming player behind two dropdowns
    nobody opens is the same as no streaming player.

    The controls WRITE their values rather than being read around. A first
    attempt read past them and streamed anyway, which was worse: the panel
    would have said "audio" while the page streamed, and a control that
    disagrees with the behaviour is a bug rather than a default.
    """
    assert "function streamDefaults()" in HTML
    assert 'route.value = can ? "v1" : "speak";' in HTML
    assert 'stream.value = can ? "sse" : "audio";' in HTML
    # An explicit choice survives. dataset.touched is set by the change
    # handlers, and streamDefaults returns early when either is set.
    assert 'route.dataset.touched === "1" || stream.dataset.touched === "1"' in HTML
    # speechPlan is the gate, not a guess, and NOTHING THAT QUEUES ever
    # streams: at 0.23x Chatterbox needs more lead than the audio is long, and
    # the engine below it is four times slower again. This tested for a clone,
    # which was the same question only while a clone was the only job -- a
    # preset voice fell through and was planned against Kokoro's 2.79x, so the
    # route was set to /v1 and the stream to sse for a request about to be
    # posted to /jobs.
    assert "&& !isJob()" in HTML, \
        "the stream default asks what kind of voice it is rather than where it runs"
    assert 'speechPlan("kokoro", text).mode === "audio"' in HTML


# ------------------------------------------------- the press, answered --
#
# MEASURED UNDER THE STUB HARNESS with every route delayed 300 ms, so nothing
# passes by being answered on a microtask. Before this change: Refresh,
# Download the audio and Retry changed nothing at all, Stop and keep what's
# done took 605 ms and the row still said "running", and a vocabulary name took
# 305 ms. RAIL gives a press 100 ms to register; Doherty puts the flow of work
# at 400 ms.


def test_every_press_that_waits_on_the_network_answers_before_it_waits():
    """Five presses on this page changed nothing until the answer came back.

    The helper is the whole fix, so the assertion is that all five reach it and
    that it cannot itself be asynchronous: it touches only the element that was
    pressed, which is why it cannot race the render that follows.
    """
    assert "function busy(button, label) {" in HTML
    body = HTML[HTML.index("function busy(button, label) {"):]
    body = body[:body.index("\n}\n")]
    assert "await" not in body, "the acknowledgement itself waits on something"
    assert 'button.setAttribute("aria-busy", "true")' in body, "drawn but not announced"
    # It hands back the undo rather than keeping state, so a caller cannot
    # forget which button it greyed.
    assert "return () => {" in body

    script = bare(SCRIPT)
    # Refresh: the worst of the five, because renderJobs writes the same markup
    # back when the listing has not moved, so the press was invisible for ever.
    assert 'const done = busy($("refresh"), "Refreshing…");' in script
    assert 'busy(button, "Fetching…")' in script      # downloadJob
    assert 'busy(button, "Retrying…")' in script      # retryJob
    assert 'busy(button, "Stopping…")' in script      # stopJob
    # The listeners hand the pressed element over, which is what makes the
    # acknowledgement possible at all.
    for attr, fn in (("data-get", "downloadJob"), ("data-stop", "stopJob"),
                     ("data-retry", "retryJob")):
        assert f"{fn}(b.dataset." in script and f"[{attr}]" in script


def test_stop_says_stopping_and_never_says_stopped():
    """The row went on saying "running" across a whole round trip and only
    corrected on the poll after it: 605 ms to any change at all.

    IT WRITES "stopping" AND NOT "stopped". An optimistic terminal state has to
    be taken back when the server refuses, and taking a word back is worse than
    the wait it saved. A state that only says the request is in flight cannot
    be wrong.
    """
    script = bare(SCRIPT)
    assert "const STOPPING = new Set();" in script
    body = script[script.index("async function stopJob(id, button)"):]
    body = body[:body.index("\n}\n")]
    # Written BEFORE the await, which is the point.
    assert body.index("STOPPING.add(id)") < body.index("await json(")
    assert body.index("renderJobs()") < body.index("await json(")
    # And taken back when the server refuses, rather than left standing.
    assert "STOPPING.delete(id); renderJobs();" in body
    assert "stopped" not in body
    row = HTML[HTML.index("function renderJobs()"):]
    row = row[:row.index("\n}\n")]
    assert 'stopping ? "stopping…" : job.status' in row
    # And the button goes, so the same request cannot be sent twice.
    assert "live && !stopping ?" in row


def test_the_jobs_tab_asks_as_soon_as_somebody_looks_at_it():
    """The poller runs on a backoff ladder and only polls while the panel is
    open, so opening it showed rows as old as the last rung: measured on the
    fake clock, a job 400 s old was next polled in 26,600 ms.

    visibilitychange already did exactly this line. The tab did not.
    """
    assert 'if (changed && button.dataset.tab === "jobs") schedule(0);' in HTML
    assert 'document.addEventListener("visibilitychange", () => schedule(0));' in HTML


def test_the_first_paint_does_not_wait_on_a_cold_health_read():
    """/ui/health is proxied to the gateway at timeout 5, the gateway fans out
    to three backends at GATEWAY_HEALTH_TIMEOUT 5, and one of those reaches
    over the network to a GPU runner that is somebody's desktop. Measured
    2,517 ms on the first read of a session.

    boot() awaited it and then made five more requests one after another, so
    the voice picker was 2,690 ms away on a cold load. Nothing needs the answer
    except one optgroup label, which is redrawn when it lands.
    """
    boot = HTML[HTML.index("(async function boot()"):]
    assert "const health = poll();" in boot, "health is on the critical path again"
    assert "await poll();" not in boot
    # Nor awaited under its new name, which is the same defect renamed.
    assert "await health" not in boot
    assert "await Promise.all([loadGlossaries(), loadVoices()," in boot
    assert "health.then(renderVoices);" in boot, "the one label that needs it"
    # loadVoices was itself two serial reads that do not depend on each other.
    voices = HTML[HTML.index("async function loadVoices()"):]
    voices = voices[:voices.index("\n}\n")]
    assert "Promise.allSettled(" in voices
    # allSettled and not all: a failed clip listing is an empty list and must
    # not take the Kokoro voices down with it.
    assert "Promise.all(" not in bare(voices)


def test_the_words_left_do_not_stand_still_for_forty_four_seconds():
    """roughly()'s bottom rung was everything under 45 s, so a ten minute
    transcription said "about a minute left" for 23 s and then "a few seconds
    left" for 44 s. Measured over the whole 68 s wait, the phrase changed once.

    A reading that does not move is read as a reading that has stopped,
    whatever the clock beside it is doing. Five second steps are inside what
    the estimate can support: the seed is 8.5x, Parakeet measures 8.81x, and
    that is 2.7 s over a 68 s job.
    """
    body = HTML[HTML.index("function roughly(seconds)"):]
    body = body[:body.index("\n}\n")]
    assert 'if (seconds < 5) return "a few seconds";' in body
    assert 'Math.round(seconds / 5) * 5 + " seconds"' in body
    # Above a minute it stays deliberately vague: a confirm dialog that says
    # "1m 47s" invites somebody to time it.
    assert '"about a minute"' in body


def test_a_file_this_browser_cannot_decode_still_has_a_length():
    """sttSeconds() was 0 for a file over CAN_DECODE and for any container
    decodeAudioData refuses, so the progress bar had no budget: a 68 s wait
    painted a moving elapsed clock beside a bar frozen at scaleX(0). That was
    the one genuinely featureless wait left on this page, and it is the path a
    two hour MKV takes.

    A media element reads the container header and stops, which is a different
    code path from decodeAudioData and answers for a good deal of what it
    refuses. Failing is an ordinary answer and it is 0, so the bar then stays
    the honest elapsed clock rather than moving at a rate nobody measured.
    """
    assert "function probeSeconds(file)" in HTML
    body = HTML[HTML.index("function probeSeconds(file)"):]
    body = body[:body.index("\n}\n")]
    assert 'media.preload = "metadata"' in body, "it decodes rather than reads"
    assert 'media.addEventListener("error", () => finish(0))' in body
    assert "resolve(isFinite(seconds) && seconds > 0 ? seconds : 0)" in body
    # A media element handed a container it cannot parse is free to fire
    # neither event, so the wait needs an end.
    assert "setTimeout(() => finish(0), PROBE_MS)" in body
    assert "URL.revokeObjectURL(url)" in body, "one object URL leaked per file"
    pick = HTML[HTML.index("async function pick(file)"):]
    pick = pick[:pick.index("\nasync function")]
    assert pick.count("await probeSeconds(file)") == 2, (
        "the two paths that reach Transcribe with no measured length")
    # stt.seconds was the one field pick() did not clear, so a large file
    # picked after a small one inherited the small one's length.
    assert "stt.seconds = null;" in pick


def js_between(start: str, end: str) -> str:
    """The source of one function, from its declaration to the next."""
    a = HTML.index(start)
    return HTML[a:HTML.index(end, a)]


# ------------------------------------------------------------- meniscus ----
#
# The identity. A meniscus is the curve a liquid makes where it meets its
# container; the tab strip floats on one, and the trough leans while it
# travels. Adapted from hasib41/meniscus-liquid-nav, MIT.


def test_the_selected_tab_is_not_also_a_filled_pill():
    """The bead says which tab is selected -- it sits in a socket cut for it,
    the icon rides up into it, and the label appears underneath. Leaving a
    filled pill behind that means two mechanisms for one job, and the pill is
    the louder of the two: it competes with every --accent control on the page.

    THERE IS NO aria-selected FILL RULE AT ALL NOW, which is the point. The
    only thing the attribute selects for is the label's opacity."""
    fills = [r for r in BARE_CSS.split("\n")
             if "[aria-selected=true]" in r and "background:" in r
             and "forced-colors" not in r]
    # The forced-colours fallback is allowed to fill, and must: see
    # test_forced_colours_gets_the_pill_back.
    fc = CSS[CSS.index("@media (forced-colors:active)"):]
    fc = fc[:fc.index("}\n}") + 3]
    fills = [r for r in fills if r.strip() not in
             {ln.strip() for ln in fc.split("\n")}]
    assert not fills, f"the pill survived the rebrand: {fills}"
    assert ".tabs button[aria-selected=true] .label," in BARE_CSS, \
        "the selected tab does not name itself"
    assert "opacity:var(--t)" in rule(".tabs button[aria-selected=true] .label,")


def test_the_surface_is_decorative_and_the_state_is_not_in_it():
    """aria-selected and the label already carry the selection, so a screen
    reader gains nothing from the plate, the bead or the ground shadow and
    would only have to skip past three of them."""
    start = HTML.index('class="skin"')
    tag = HTML[start:HTML.index(">", start)]
    assert 'aria-hidden="true"' in tag
    assert 'focusable="false"' in tag, "SVG is focusable in IE/Edge legacy trees"
    for decoration in ('<span class="cast"', '<span class="bead"',
                       '<div class="bloom"'):
        at = HTML.index(decoration)
        assert 'aria-hidden="true"' in HTML[at:HTML.index(">", at)], decoration
    # The icons too: each button already has a label beside it, so an icon that
    # announced itself would say the name of the tab twice.
    icons = HTML[HTML.index('<div class="tabs"'):HTML.index("<!-- ==================================================== /the dock ====")]
    assert icons.count("<svg viewBox") == 4
    assert icons.count('aria-hidden="true" focusable="false"') == 4


def test_the_trough_is_one_path_rather_than_assembled_shapes():
    """A socket built from a border-radius and two overlapping boxes leaves
    seams, and the seams appear exactly when the shape is asymmetric -- which
    is the whole point of the lean.

    THREE TANGENT ARCS, SOLVED RATHER THAN EYEBALLED: a convex shoulder that
    turns the top edge down, a concave bowl that wraps the bead, and a second
    shoulder back up. Because they are solved for tangency the bowl hugs the
    bead exactly and the joins are invisible at any size."""
    body = js_between("function dockTrough(", "3 one rAF, one spring")
    assert body.count('return "M0 "') == 1, "the outline is not one path"
    assert body[:body.index("\n}\n")].rstrip().endswith('+ "Z";'), \
        "the outline is not closed, so the plate has no inside to fill"
    assert body.count('"A"') >= 6, "a socket in a rounded bar needs six arcs"
    # The tangency solve itself, which is what makes it one shape rather than
    # three drawn next to each other.
    assert "external tangency" in body
    assert "const dreach = (s, rb, by) => Math.sqrt(" in SCRIPT


def test_the_lean_is_clamped_and_the_two_shoulders_disagree():
    """THE LIQUID IS ENTIRELY IN THE SHOULDERS. The trailing radius draws out
    long and the leading one tightens, so the socket smears behind the weight
    moving through it; give both the same radius and the bar is a notch that
    slides. The signed travel has to be clamped or a fast drag inverts the
    arithmetic and the socket turns inside out."""
    body = js_between("function dockPaint(", "function dockLoop(")
    assert "dclamp(dv / 1100, -1, 1)" in body, "the lean is neither normalised nor clamped"
    assert "+ 0.40 * q" in body and "- 0.40 * q" in body, \
        "both shoulders lean the same way, so the surface slides instead of dragging"
    assert "G.S * 0.55, G.S * 2.1" in body, "a shoulder can collapse or run away"
    # Volume-preserving squash: a bead that only widened would gain area as it
    # travelled, which reads as it inflating rather than being thrown.
    assert "(1 / sx).toFixed(3)" in body, "the squash is not volume preserving"


def test_the_frame_loop_cancels_itself_once_the_surface_settles():
    """A requestAnimationFrame loop that runs for the life of the page to
    animate nothing is the usual cost of this pattern, and it is the reason a
    decorative flourish shows up in a battery trace."""
    body = js_between("function dockLoop(", "function dockRun(")
    assert "draf = 0;" in body, "the loop never stops"
    assert "else { dx = dtarget; dv = 0; dockPaint(); }" in body, \
        "the loop reschedules unconditionally"
    # And dockRun is the only thing that arms it, so there is exactly one place
    # a second loop could be started from.
    assert SCRIPT.count("requestAnimationFrame(dockLoop)") == 1


def test_the_surface_still_moves_under_reduced_motion_it_just_arrives():
    """Reduced motion means a gentler equivalent, not the removal of the one
    cue that says which tab is selected."""
    body = js_between("function meniscusTo(", "4 the masthead")
    assert "MENISCUS.calm.matches" in body, "reduced motion is not consulted"
    assert "dx = dtarget; dv = 0; dockPaint(); return;" in body, \
        "the bead fails to arrive, so the selection has no surface cue at all"
    # The icon still rises and the label still appears: both answer "which one
    # is selected", and neither is vestibular. What goes is the travel.
    calm = CSS[CSS.index("@media (prefers-reduced-motion:reduce){\n  /*"):]
    calm = calm[:calm.index("\n}\n") + 3]
    assert "opacity" not in rule(".tabs button svg{"), \
        "the icon fades rather than moving, so there is nothing to reduce"
    assert ".tabs button svg{transition:none}" in calm


def test_forced_colours_gets_the_pill_back():
    """A gradient stroke is painted over by the system palette, so under forced
    colours the selection would rest on font-weight alone -- the cue those
    users are most likely to have lost already."""
    block = CSS[CSS.index("@media (forced-colors:active)"):]
    block = block[:block.index("}\n}") + 3]
    assert ".dock .skin,.dock .bead,.dock .cast,.bloom{display:none}" in block
    assert "background:Highlight" in block
    # AND THE RISE HAS TO BE UNDONE WITH THEM. With the bead painted over, an
    # icon left translated 38px up sits outside a bar that is now a plain
    # rectangle -- the selected tab would be the one whose icon has vanished.
    assert ".tabs button svg{transform:translate(-50%,-50%)}" in block
    assert ".tabs .label{opacity:1;color:ButtonText}" in block, \
        "only the selected tab is named, and its name is the cue being lost"


def test_the_trough_is_measured_rather_than_positioned_by_a_constant():
    """Tab widths move with the reader's text size, with translation, and with
    the jobs chip appearing. A number written into the source is wrong the
    first time any of those changes."""
    body = js_between("function dockMeasure(", "2 the skin --")
    assert "getBoundingClientRect()" in body
    assert "G.slots = DOCKTABS.map" in body, "the tab centres are written down"
    assert "new ResizeObserver(() => dockLayout(false)).observe(DOCK)" in SCRIPT, \
        "the badge changes the bar without changing the window"
    # AND AGAIN ONCE THE DISPLAY FACE ARRIVES. The face is a data: URI so it
    # does not cross the network, but it still decodes asynchronously, and the
    # masthead reflowing is exactly the kind of thing that moves the dock.
    assert "document.fonts.ready.then(() => dockLayout(false))" in SCRIPT


def test_the_borrowed_component_is_attributed_where_it_is_used():
    """MIT asks for attribution and the licence text says 'Use it in anything'.
    The obligation is cheap and the provenance is worth more than the licence:
    somebody reading this geometry should be able to find the original."""
    assert HTML.count("hasib41/meniscus-liquid-nav") >= 2, \
        "attributed in fewer than both places it was adapted"
    assert "MIT" in HTML


# --------------------------------------------------------------- identity ---
#
# The palette read as AI, and it was right to: a cyan accent lit by a #22d3ee
# bead over a blue-grey ground is the house style of every AI product shipped
# since 2023, which is to say it is the default, and a default is what makes a
# thing look generated. What replaced it is an instrument: a warm faceplate, an
# achromatic control colour with a cool cast, and ONE saturated colour that
# means "this is happening right now".
#
# The tests below are the fence around that decision. Every one of them is
# named after the way it comes undone.


def _oklch(hexcolour: str):
    """Perceptual lightness, chroma and hue for a #rrggbb string.

    IN THE FILE AND NOT IN A DEPENDENCY, which is the same rule the page
    itself lives under: this service's whole claim is that it has none, and a
    colour-space conversion is thirty lines of arithmetic. Chroma is the
    number that matters here -- it is what separates "a grey with a cast" from
    "an accent hue", and no WCAG ratio can see the difference.
    """
    import math
    value = hexcolour.lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
               for c in channels]
    long = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    med = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    short = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = [x ** (1 / 3) if x >= 0 else -((-x) ** (1 / 3))
                  for x in (long, med, short)]
    lightness = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return lightness, math.hypot(a, bb), math.degrees(math.atan2(bb, a)) % 360


def _ratio(fore: str, back: str) -> float:
    """The WCAG 2.1 contrast ratio between two #rrggbb strings."""
    def luminance(value):
        value = value.lstrip("#")
        channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                   for c in channels]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    high, low = sorted((luminance(fore), luminance(back)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def tokens(theme: str) -> dict:
    """Every --token: #hex declared in one theme block.

    The light block is :root up to the dark media query; the dark block is
    that query up to the reset. Both are read from BARE_CSS so a hex quoted in
    a comment -- and the comments here quote a great many -- cannot be mistaken
    for a declaration.
    """
    light = BARE_CSS[BARE_CSS.index(":root{"):
                     BARE_CSS.index("@media (prefers-color-scheme:dark)")]
    dark = BARE_CSS[BARE_CSS.index("@media (prefers-color-scheme:dark)"):
                    BARE_CSS.index("*{box-sizing")]
    block = light if theme == "light" else dark
    return {name: value for name, value
            in re.findall(r"(--[\w-]+):\s*(#[0-9a-fA-F]{6})", block)}


def test_the_page_has_no_interactive_hue():
    """THE FIRST TELL, AND THE ONE THE OWNER NAMED. #0c7c8a with a #22d3ee
    highlight is cyan-on-dark with a gradient bead, which is what every
    generated interface looks like. The replacement is a graphite with a cool
    cast: OKLCh chroma .027 in light and .020 in dark, which is a MATERIAL
    difference against a warm face rather than a second colour competing with
    the lamp. A ratio cannot catch this coming back -- chroma can."""
    for theme in ("light", "dark"):
        _, chroma, _ = _oklch(tokens(theme)[
            "--accent" if theme == "light" else "--accent"])
        assert chroma < 0.05, (
            f"the {theme} accent has chroma {chroma:.3f}: it is a hue again, "
            "and the page has two colours competing to mean 'important'")


def test_the_ground_is_warm():
    """THE SECOND TELL, which nobody flagged for the whole life of this file.
    #f6f7f9 is a blue grey, and a blue grey under a cyan bead is the entire
    look. A warm ground is also what makes a red lamp read as LIT rather than
    as stuck on: red on blue-black is a Christmas pair."""
    for theme in ("light", "dark"):
        for name in ("--bg", "--panel"):
            _, chroma, hue = _oklch(tokens(theme)[name])
            assert 40 <= hue <= 110, f"{theme} {name} sits at hue {hue:.0f}"
            assert chroma > 0.004, f"{theme} {name} is dead neutral"


def test_every_pair_a_reader_reads_clears_AA():
    """ACCESSIBILITY IS NOT A LATER PASS, so the ratios are asserted rather
    than stated. Body text needs 4.5, a bounded graphic needs 3.

    --live on --bg is the floor at 4.56, and that is deliberate: it is a
    tighter margin than anything else here, it still beats the 4.59 the old
    accent managed against its own ground, and it is the price of a red that
    is dark enough to be a filament rather than a highlighter.
    """
    text = [("--ink", "--panel"), ("--ink", "--bg"), ("--ink", "--sunk"),
            ("--dim", "--panel"), ("--dim", "--bg"), ("--dim", "--sunk"),
            ("--accent", "--panel"), ("--accent", "--bg"), ("--accent", "--sunk"),
            ("--accent-ink", "--accent"), ("--accent-ink", "--live"),
            ("--live", "--panel"), ("--live", "--bg"),
            ("--bad", "--bad-bg"), ("--bad", "--panel"), ("--bad", "--bg"),
            ("--warn", "--warn-bg"), ("--good", "--good-bg"),
            ("--mark-ink", "--mark-bg")]
    graphic = [("--accent", "--line"), ("--live", "--line"),
               ("--lamp-off", "--panel"), ("--lamp-core", "--lamp-off"),
               ("--lamp-core", "--lamp-rim")]
    for theme in ("light", "dark"):
        palette = tokens(theme)
        missing = {n for pair in text + graphic for n in pair} - set(palette)
        assert not missing, f"{theme} does not define {sorted(missing)}"
        for fore, back in text:
            got = _ratio(palette[fore], palette[back])
            assert got >= 4.5, f"{theme} {fore} on {back} is {got:.2f}"
        for fore, back in graphic:
            got = _ratio(palette[fore], palette[back])
            assert got >= 3.0, f"{theme} {fore} on {back} is {got:.2f}"
    # And the one pair that is deliberately NOT separated by contrast.
    for theme in ("light", "dark"):
        palette = tokens(theme)
        assert _ratio(palette["--live"], palette["--bad"]) < 2, (
            "the two reds now differ by luminance, which is an accident: the "
            "separation is form, and a test that passes for the wrong reason "
            "is worse than no test")


def test_the_lamp_is_never_a_ground_behind_words():
    """HALF OF THE ANSWER TO THE RED CONFLICT, and the half a stylesheet can
    lose quietly. --live and --bad are 1.48:1 apart and share a hue by design,
    so a deuteranope sees one red; the moment --live becomes a filled chip with
    a word in it, "this is running" and "this went wrong" are the same object
    on a tab that shows both at once.

    Red that is a small filled OBJECT means the signal is moving. Red that is
    ink resting in a pale wash beside a word means it already went wrong.
    """
    filled = {s.strip() for s
              in re.findall(r"\n([^\n{}]+)\{[^}]*background:var\(--live\)", BARE_CSS)}
    allowed = {".bar-fill", ".meter div", ".scrub .played",
               ".status.running::before", ".brand.lit .lamp"}
    assert filled <= allowed, f"the lamp became a ground: {filled - allowed}"
    # AND THE SCAN MUST FIND SOMETHING, or a page with no lamp at all passes
    # this and the fence proves nothing. These four are the lamp: a progress
    # bar, a level meter, a playhead, and the dot beside the word Running.
    assert {".bar-fill", ".meter div", ".scrub .played",
            ".status.running::before"} <= filled, f"the lamp went out: {filled}"
    # The other direction: --live as ink is allowed on exactly one thing, and
    # it is a countdown during a capture rather than a label about one.
    inked = re.findall(r"\n([^\n{}]+)\{[^}]*color:var\(--live\)", BARE_CSS)
    assert {s.strip() for s in inked} <= {".ring"}, \
        f"--live is being used as text colour: {inked}"


def test_a_failure_never_glows_and_never_moves():
    """THE OTHER HALF. If --bad ever gains a transition, an animation or a
    glow, the two reds are separated by nothing at all: a pulsing failure and a
    pulsing lamp are one thing seen twice. A failed chip is printed, not lit --
    it carries a 2px left rule instead, which is the cue that survives having
    no colour at all."""
    for selector, body in re.findall(r"\n([^\n{}]+)\{([^}]*var\(--bad[^}]*)\}", BARE_CSS):
        for motion in ("animation", "box-shadow"):
            assert motion not in body, f"{selector.strip()} makes a failure glow"
        if "background:var(--bad" in body:
            assert "transition" not in body, f"{selector.strip()} makes a failure move"
    assert "border-left:2px solid var(--bad)" in rule(".status.failed{"), \
        "the non-colour cue on a failed job is gone"
    assert "border-left:2px solid var(--bad)" in rule(".note.bad{")


def test_the_primary_action_is_not_the_same_object_as_a_secondary_one():
    """.primary had EIGHT call sites and no rule, and .small had thirty-three.
    Every tab's main action rendered pixel-identical to the secondary button
    beside it, which is the largest hierarchy gap left in this file, and it was
    invisible because the classes were already there and already meaningless.

    The filled object in a row is achromatic on purpose: at 9.75:1 it is the
    heaviest thing present and it still cannot be confused with the lamp."""
    primary = rule(".primary{")
    assert "background:var(--accent)" in primary
    assert "color:var(--accent-ink)" in primary
    assert re.search(r"font-size:", rule(".small{")), "the small rank has no size"
    # .link was the third class with a comment describing it and no rule to
    # make the description true. Four ranks, four appearances, or the classes
    # are documentation of an intention nobody implemented.
    link = rule("button.link{")
    assert "background:none" in link and "border-color:transparent" in link


def test_a_destructive_button_is_outlined_and_never_filled():
    """"Delete the record too" with a red fill would be the loudest object on
    the Jobs tab, and red would then mean "press this" -- which collapses the
    whole argument for a red accent in the first place. Fill means go, outline
    means careful, wash means past tense."""
    danger = rule(".danger,")
    assert "color:var(--bad)" in danger and "border-color:var(--bad)" in danger
    assert "background:var(--bad" not in danger, "a destructive button is filled"
    assert "background:var(--panel)" in danger
    # And the two job-row deletes reach it, which they cannot do by class:
    # test_escaping.py pins `class="small" data-delaudio=` byte for byte.
    assert ".job [data-delaudio]" in danger and ".job [data-forget]" in danger


def test_the_navigation_is_one_object_and_the_labels_are_inside_it():
    """THE NOTE WAS "a bold menu with text inside the form", and twice it was
    read as a strip with a line under it. The bar is a solid plate; the icons,
    the label and the bead all live inside its bounds, and the bead is a thing
    the plate's own top edge dips beneath rather than a marker drawn on top.

    THE MECHANISM MOVED AND THE INTENT DID NOT. This used to assert a full-width
    slab spanning the text measure; the navigation is a fixed dock sized to the
    thumb now, so what must be true is that it is ONE shape with everything
    inside it -- not that it is as wide as the column."""
    assert "background:var(--panel)" in rule("body{"), "the page lost its faceplate"
    # One path, one fill, one stroke: the plate is the shape, not a box with a
    # dip drawn under it.
    assert "fill:url(#dock-plate)" in rule(".dock .skin path{")
    assert "stroke:url(#dock-rim)" in rule(".dock .skin path{"), \
        "the plate has no silhouette, so the socket has no edge to be cut into"
    skin = rule(".dock .skin{")
    assert "position:absolute" in skin and "inset:0" in skin, \
        "the plate does not span the dock"
    # The controls sit INSIDE the plate's bounds and fill it edge to edge.
    tabs = rule(".tabs{")
    assert "position:absolute" in tabs and "inset:0" in tabs
    assert "flex:1" in rule(".tabs button{"), \
        "the labels do not fill the bar they sit inside"
    # And the content column is not glued to it any more -- it clears it, which
    # is what a fixed object at the foot of the viewport requires.
    assert "98px" in rule(".wrap{"), "the last control on every tab is under the dock"

def test_the_biggest_thing_on_the_page_is_where_you_are():
    """It was the other way round twice. First the brand was --t-display at
    weight 700 over --t-label tabs, so the largest, heaviest string in the head
    was the one thing nobody ever presses. Then the mark and the wordmark were
    put INSIDE the navigation, which is two identities in one object and was
    rejected in those words.

    The product name is now the smallest type on the page and the section you
    are on is the largest. "I rather have things more bold" -- this is it."""
    brand, word = rule(".brand{"), rule(".word{")
    assert "font-size:var(--t-micro)" in brand and "text-transform:uppercase" in brand
    assert "clamp(2.75rem,9vw,6rem)" in word, "the headline is not display size"
    assert "font-weight:700" in word
    # A DISPLAY FACE, NOT THE UI FONT AT A LARGER SIZE, and self-hosted: this
    # service serves exactly one file, so the face is a data: URI in the sheet
    # rather than a sibling asset needing a route of its own.
    assert '@font-face{font-family:"Calliope Display"' in CSS
    assert "src:url(data:font/woff2;base64," in CSS
    assert 'font-family:"Calliope Display",system-ui' in word, \
        "the headline has no fallback if the face fails to decode"
    # Tracking is size-specific: it goes negative because the size went up.
    assert "letter-spacing:-.04em" in word
    # And the brand is NOT in the navigation, which was the whole note.
    dock = HTML[HTML.index('<div class="rail">'):HTML.index("<!-- ==================================================== /the dock ====")]
    assert "Calliope" not in dock, "the wordmark is back inside the menu"
    # The top of the type scale did not leave the page with the old wordmark.
    assert "font-size:var(--t-display)" in rule(".card:not(.well) > h2,")


def test_the_page_tells_you_which_tab_you_are_on_from_across_the_room():
    """THE DEFLECTION SIGN IS GONE AND SOMETHING LOUDER REPLACED IT. The old
    surface rose on Speak and dipped everywhere else -- one bit, carried by a
    7px swing in a 24px box, on a strip 920px wide. It was the only liquid
    thing on the page that said anything, and it is not portable to a socket
    that always dips: inverting the reference's bowl for one tab is the kind of
    deviation that got the last two attempts rejected.

    What carries it now is four bits, not one, and it is the size of the
    window: every tab has its own accent, and the bead, its halo, the label and
    the wash behind the entire page are all lerped to it as the bead travels.
    You do not read which tab you are on; the room changes colour."""
    dock = HTML[HTML.index('<div class="rail">'):HTML.index("<!-- ==================================================== /the dock ====")]
    accents = re.findall(r'--acc:(#[0-9A-Fa-f]{6})', dock)
    assert len(accents) == 4, f"not every tab carries an accent: {accents}"
    assert len(set(accents)) == 4, f"two tabs share an accent: {accents}"
    # READ FROM THE STYLESHEET, NOT WRITTEN TWICE. A palette with a second copy
    # in the script is a palette that drifts the first time one of them moves.
    body = js_between("const DOCKACC", "1 geometry --")
    assert 'getComputedStyle(t).getPropertyValue("--acc")' in SCRIPT
    # And the wash is driven by the SAME channels, so the colour of the room is
    # the colour of the bead by construction rather than by coincidence.
    paint = js_between("function dockPaint(", "function dockLoop(")
    assert 'document.documentElement.style.setProperty("--glow-rgb"' in paint
    assert "dmix(DOCKACC[near], DOCKACC[other], t)" in paint, \
        "the accent steps between tabs instead of travelling with the bead"
    assert "rgb(var(--glow-rgb) / var(--bloom-a))" in rule(".bloom{")


def test_the_level_does_not_start_a_second_frame_loop():
    """meniscusStep cancels itself the moment the surface settles, which is the
    one thing that keeps a decorative flourish out of a battery trace. A
    waterline that armed its own rAF would defeat that and leave a loop running
    for the life of the page. meter() already runs a display-synced loop with
    the PPM ballistic on it, so the level borrows that one."""
    body = js_between("function meniscusLevel(", "function meniscusBusy(")
    assert "meniscusLamp()" in body
    assert "requestAnimationFrame" not in body, "the lamp started its own loop"
    meter = HTML[HTML.index("function meter(bar, analyser, data)"):]
    meter = meter[:meter.index("\n}\n")]
    assert "meniscusLevel(shown, true)" in meter, "the seam reads a second analyser"
    assert "meniscusLevel(0, false)" in meter, "the lamp stays lit after the stop"


def test_the_lamp_has_a_state_it_can_be_seen_to_be_in():
    """A bead that is permanently red spends the page's one colour before
    anything has happened, which turns "this is being captured" into "this is
    the brand". Unlit it is dark red glass at 7.52:1 on --panel, so you learn
    where the indicator is BEFORE it lights, which is what makes you notice
    when it does."""
    for theme in ("light", "dark"):
        assert "--lamp-off" in tokens(theme), f"--lamp-off is undefined for {theme}"
    # THE LAMP MOVED OUT OF THE NAVIGATION AND BECAME THE MARK. It was a bead
    # in the tab strip; the bead in the dock is the SELECTION now and carries
    # the tab's accent at all times, so it cannot also mean "something is
    # running" without meaning both at once. The disc beside the wordmark is
    # the one object on the page whose only job is that fact.
    assert "background:var(--lamp-off)" in rule(".brand .lamp{"), \
        "the lamp has no unlit state, so there is nothing to notice lighting"
    assert "background:var(--live)" in rule(".brand.lit .lamp{")
    assert "box-shadow:0 0 0 4px color-mix(in srgb, var(--live)" in rule(".brand.lit .lamp{"), \
        "lit and unlit differ by fill alone, which is 3.2:1 at 8px across"
    # No pulse. A half-hertz oscillation in the optical centre of the viewport,
    # over a transcript somebody is reading, would be there for the life of the
    # page -- and lit versus unlit is already carried by fill and a halo.
    assert "animation" not in rule(".brand .lamp{")
    assert "animation" not in rule(".brand.lit .lamp{")
    # Two independent reasons can light it and neither may dark the other.
    busy = js_between("function meniscusBusy(", "the drag --")
    assert "MENISCUS.busy.add(source)" in busy and "MENISCUS.busy.delete(source)" in busy
    assert 'b.classList.toggle("lit", MENISCUS.mic || MENISCUS.busy.size > 0)' in SCRIPT


def test_where_am_i_is_reachable_by_thumb_on_a_phone():
    """Four labels at --t-title plus a two-digit chip is about 340px of strip
    that flex-wrap:nowrap is forbidden to break, on a 375px viewport. It goes
    to the bottom edge, which is also where the borrowed component actually
    lives -- hasib41/meniscus-liquid-nav is a bottom navigation -- so "where am
    I" is now on screen at EVERY scroll position rather than only at the top.
    """
    # IT IS THERE AT EVERY WIDTH NOW, so this is no longer a phone override --
    # there is no layout to switch, only proportions to tighten. The desktop
    # page and the phone page are the same page.
    rail = rule(".rail{")
    assert "position:fixed" in rail
    assert "env(safe-area-inset-bottom)" in rail, "the dock sits under the home indicator"
    assert "padding-inline:clamp(26px,10.5%,54px)" in rule(".tabs{"), \
        "the tabs run to the plate's corners, where the socket cannot clear them"
    # And the column still has to end above it, or the last control on every
    # tab is behind the bar.
    assert "padding:var(--s5) var(--s4) calc(var(--s6) + 98px + env(safe-area-inset-bottom))" \
        in rule(".wrap{"), "the dock covers the last rows of a scroll"
    block = CSS[CSS.index("@media (max-width:30rem)"):]
    block = block[:block.index("\n}\n") + 3]
    assert "width:22px" in block, "the icons do not tighten on a narrow screen"


def test_the_live_state_survives_the_surface_being_switched_off():
    """The bead is display:none under forced colours, so if it were the only
    carrier of "something is running" those readers would lose it entirely.
    Every lamp on this page has a string beside it: the chip says Running, the
    tab carries a numeral, the clone clock counts down. Nothing here depends on
    red alone, in any mode."""
    block = CSS[CSS.index("@media (forced-colors:active)"):]
    block = block[:block.index("}\n}") + 3]
    assert ".dock .skin,.dock .bead,.dock .cast,.bloom{display:none}" in block
    assert "background:Highlight" in block
    assert "ButtonFace" in block, "the filled primary loses its edge"
    assert '<span class="count" id="jobcount"></span>' in HTML
    assert "stopping…" in HTML and 'class="status ${statusClass(job.status)}"' in HTML
    # THE BADGE IS NEVER RED, which is the same fence drawn one step earlier:
    # --live and --bad share a hue, so a count sitting in a --live ground on
    # the one tab that shows both running and failed jobs is the collision this
    # palette exists to avoid. The numeral is on ink; .live rings it.
    assert "background:var(--ink)" in rule(".tabs .count{")
    assert "box-shadow:0 0 0 2px var(--live)" in rule(".tabs .count.live{")


def test_the_recess_is_carried_by_two_cues_and_not_by_a_1_14_step():
    """--panel over --bg is 1.14:1. On a poorly calibrated display, or in
    sunlight, one 1.14:1 fill step is nothing at all, so the card would simply
    stop existing. The hairline is the second channel and it is not optional --
    which is also why --line is darker than the #dfe3e8 it replaced: it carries
    a card edge now rather than only a divider."""
    card = rule(".card{")
    assert "background:var(--bg)" in card, "the card is a raised slab again"
    assert "border:1px solid var(--card-line)" in card
    for theme in ("light", "dark"):
        block = BARE_CSS[BARE_CSS.index(":root{"):BARE_CSS.index("*{box-sizing")]
        assert block.count("--card-line:var(--line)") == 2, \
            "a theme lost the hairline and kept only the fill step"
    assert "inset" in rule(":root{")[rule(":root{").index("--lift"):], \
        "the light card is back on a drop shadow, which is the third tell"


def test_no_control_is_left_wearing_the_operating_system():
    """A REAL DEFECT THE OWNER FOUND BEFORE THIS SUITE DID. Eight range inputs
    had exactly one rule between them -- `width:100%` -- so they rendered in
    the system's own blue, which is the single loudest colour on a page whose
    palette is warm neutrals plus one red. Nineteen <select>s had this page's
    fill and border and then let macOS draw its own control on top, arrow and
    intrinsic height included, which is why a select never quite lined up with
    the input beside it.

    THE RULE: every control the browser paints itself must be claimed by the
    palette -- either taken over outright with appearance:none, or tinted with
    accent-color. There is no third option and no exemption, because the whole
    failure mode here is a control that looks fine to whoever wrote the markup
    and looks foreign to whoever opens the page.
    """
    claimed = {
        "select:not([multiple]){": ("-webkit-appearance:none", "appearance:none"),
        "input[type=range]{": ("accent-color:var(--accent)",),
        "input[type=checkbox],input[type=radio]{": ("accent-color:var(--accent)",),
    }
    for selector, needles in claimed.items():
        body = rule(selector)
        for needle in needles:
            assert needle in body, \
                f"{selector[:-1]} is still painted by the user agent ({needle} missing)"

    # AND THE ARROW IS REPLACED, not merely removed. appearance:none takes away
    # the one thing that says a select opens a list, so something has to put it
    # back -- and it has to follow the ink into dark mode, which is why it is
    # currentColor in a gradient rather than an SVG with a colour written in.
    arrow = rule("select:not([multiple]){")
    assert "currentColor 50%" in arrow, "the dropdown has no affordance left"
    assert "data:image" not in arrow, \
        "the chevron is a fixed-colour asset, so it needs one copy per theme"
    assert "padding-right" in arrow, "a long option runs underneath the chevron"

    # The audio scrubber is the exception that proves it: it paints its own
    # track, so accent-color would do nothing and it must not be relied on.
    assert "appearance:none" in rule(".seekbar{"), \
        "the scrubber stopped painting its own track"


def test_the_bead_cannot_be_left_under_the_wrong_tab():
    """A REAL DEFECT, found by rendering the page rather than by reading it.
    The bead was placed by `requestAnimationFrame(() => meniscusTo(true))`
    fired BEFORE the loop that sets aria-selected -- so the one cue that says
    which tab you are on depended on a frame callback landing, and it read the
    attributes it was racing. Rendered headless, where frames are scheduled on
    demand, the panel and the heading switched to Jobs while the bead stayed
    sitting under Transcribe.

    A frame that never comes is not hypothetical here: a backgrounded tab still
    runs timers and still finishes jobs. There is nothing to defer for either
    -- meniscusTo reads G.slots, which dockMeasure has already filled in, so it
    touches no layout at all.
    """
    # bare(), NOT SCRIPT: the comment above the change quotes the very call it
    # removed, which is the trap bare()'s own docstring is about. A negative
    # assertion over the raw text would match the documentation of the fix.
    handler = bare(SCRIPT)
    handler = handler[handler.index("const TABS = Array.from"):]
    handler = handler[:handler.index("async function poll()")]
    assert "requestAnimationFrame(() => meniscusTo" not in handler, \
        "the bead is behind a frame callback again"
    assert "meniscusTo(changed);" in handler, "the click no longer places the bead"
    # AND IT IS PLACED AFTER THE ATTRIBUTE IT READS, which is the other half:
    # called first, it finds the tab that was selected a moment ago.
    sets = handler.index('b.setAttribute("aria-selected"')
    places = handler.index("meniscusTo(changed);")
    assert sets < places, "the bead is placed from the previous selection"


def test_the_stylesheet_has_no_declaration_outside_a_block():
    """A REAL DEFECT, AND THE ONE THIS SUITE WAS BLIND TO. An edit left a
    single orphan line -- `padding:...}` with no selector and no opening brace
    -- directly after the rule it used to belong to. CSS error recovery does
    not skip it: the parser reads it as the start of a qualified rule and
    consumes forward looking for `{`, swallowing the next comment and the whole
    of the following rule into an invalid selector. The rule after it was
    .masthead, so the page's <h1> rendered full-bleed at x=0 with no measure
    and no padding, and every other test in this file still passed, because
    every one of them asserts that a STRING is present in the sheet rather
    than that the sheet parses.

    Comments are stripped first: a `{` or `}` inside prose is not structure.
    """
    depth, line_no = 0, 0
    for line_no, line in enumerate(BARE_CSS.split("\n"), 1):
        stripped = line.strip()
        if depth == 0 and ":" in stripped and "{" not in stripped:
            # At the top level the only things allowed are at-rules, selectors
            # and the closing brace of the rule above.
            head = stripped.split(":", 1)[0].strip()
            assert stripped.startswith("@") or stripped.startswith("}") or not head \
                or " " in head or "." in head or "#" in head or "[" in head, \
                f"line {line_no} is a declaration outside any block: {stripped!r}"
        depth += line.count("{") - line.count("}")
        assert depth >= 0, f"line {line_no} closes a block that was never opened"
    assert depth == 0, f"the stylesheet ends {depth} block(s) deep"


def test_the_head_rule_actually_lands_on_the_masthead():
    """THE SAME DEFECT FROM THE OTHER SIDE, and the cheaper half to check. The
    structural test above catches an orphan declaration; this one catches a
    rule that is present, well formed, and still not applying -- because the
    parser reached it inside a selector it had already given up on. If
    .masthead ever stops carrying the measure, the <h1> goes full-bleed and
    stops lining up with the cards underneath it."""
    head = rule(".masthead{")
    assert "max-width:var(--measure)" in head and "margin:0 auto" in head, \
        "the masthead does not share the column's measure"
    assert "var(--s4)" in head, "the heading runs to the viewport edge"
    # And it is a top-level rule, not one nested inside a media query, or it
    # would only apply at one width.
    at = BARE_CSS.index(".masthead{")
    before = BARE_CSS[:at]
    assert before.count("{") == before.count("}"), \
        "the masthead rule is nested inside a block it should not be in"


def test_the_page_has_a_heading_and_a_landmark():
    """It had neither. No <h1> at all, so a screen reader had nothing to
    announce as the name of the thing it landed on and every heading below was
    an <h2> under nothing; and <div class="wrap"> meant the four tabpanels sat
    in no region, so "skip to content" had nothing to skip to."""
    # The wordmark is a link inside the heading now -- the mark is the one
    # thing on the page that should take you home -- so the h1 is the wrapper
    # and the anchor carries the accessible name.
    # THE HEADING NAMES THE SECTION, NOT THE PRODUCT. It was the wordmark,
    # which is a string nobody navigates by; it is the tab you are on now, and
    # the four panels write it between them from one element rather than
    # carrying an <h1> each, so the document never has two.
    assert '<h1 class="word" id="word">' in HTML
    assert visible().count("<h1") == 1, "a second first-level heading is two subjects"
    for name in ("Transcribe", "Speak", "Jobs", "Vocabulary"):
        assert f'"{name}"' in SCRIPT, f"the heading never says {name}"
    assert 'const w = $("word")' in SCRIPT and "w.textContent = said[0]" in SCRIPT
    assert '<main class="wrap">' in HTML and "</main>" in HTML
    # visible(), not HTML: the comment above the change quotes the thing the
    # change removed, which is the trap bare()'s own docstring is about.
    assert '<div class="wrap">' not in visible()


def test_the_vocabulary_panel_closes_the_elements_it_opens():
    """A REAL DEFECT, found by the restructure rather than by a reader. The
    panel carried one <div> more closing than opening, plus a column-zero
    indent on #glossman that hid it, so the section leaked its close into the
    element after it. It parsed by accident.
    """
    from html.parser import HTMLParser
    void = {"meta", "link", "br", "hr", "img", "input", "source", "track",
            "stop", "path", "circle", "area", "base", "col", "embed", "wbr"}

    class Balance(HTMLParser):
        def __init__(self):
            super().__init__()
            self.open, self.bad = [], []

        def handle_starttag(self, tag, attrs):
            if tag not in void:
                self.open.append(tag)

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.open or self.open[-1] != tag:
                self.bad.append(tag)
                if tag in self.open:
                    del self.open[self.open.index(tag):]
                return
            self.open.pop()

    markup = re.sub(r"<!--.*?-->", "",
                    HTML[HTML.index("<body>"):HTML.index("<script>")], flags=re.S)
    check = Balance()
    check.feed(markup)
    assert check.bad == [], f"{check.bad} closes an element that is not open"
    assert check.open == ["body"], f"left open: {check.open}"
