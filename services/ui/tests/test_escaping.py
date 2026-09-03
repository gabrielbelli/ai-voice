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
    assert "setTimeout" in head, (
        "the value must be read after the default action, not before it")


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


def test_the_engine_is_named_on_both_paths():
    start = HTML.index("function onVoiceChange()")
    body = HTML[start:HTML.index('$("voice").addEventListener')]
    hint = body[body.index('$("voicehint")'):body.index('$("tts-expert-fast")')]
    assert "<strong>Chatterbox</strong>" in hint
    assert "<strong>Kokoro</strong>" in hint, (
        "the fast path never said Kokoro while its expert panel was titled so")


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
    assert "loadVoices()" in handler[:200], \
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
    before the first filter runs."""
    boot = HTML[HTML.index("  await poll();"):]
    assert boot.index("populateLanguages()") < boot.index("await loadVoices()")
