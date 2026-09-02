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
