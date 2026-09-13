"""The counts on the two filter controls: one shape, owned by two services.

Static and parser-based, for the reason test_jobs.py and test_playback.py are:
what this asserts is a property of the bytes in ui.html and of the bytes in
tts-long's main.py, decidable without a browser, without a server and without
the network. NOTHING HERE IMPORTS tts-long. It is read as source through `ast`,
because importing another service's app would drag that service's dependencies
and its import-time configuration into this suite, and the thing under test is
a set of NAMES, which is in the source either way.

WHY THIS FILE EXISTS AT ALL. `GET /v1/jobs` answers `counts`, and the page puts
those numbers in the option text of the two selects, so the closed control says
how much it is hiding. The two halves were built against one written contract
and diverged anyway: the server emits `counts["all"]`, the page asks for
"total". Every test on the server side passed, every test on the page side
passed, and the "Everything" option shipped with no number on it, because each
side tested its own half against its own idea of the name.

So the assertion here is deliberately NOT "the page says all" and NOT "the
server says all". It is the SHARED shape: every name the page looks up is a
name the server emits. Both lists are read out of the two files rather than
written down here. A test that restated either list would be the same bug in a
new place -- a third copy of the contract, free to drift from both.
"""

import ast
import re
from html.parser import HTMLParser
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "app" / "static" / "ui.html"
SERVER = Path(__file__).resolve().parents[2] / "tts-long" / "app" / "main.py"
HTML = PAGE.read_text()
SOURCE = SERVER.read_text()

# The page's own vocabulary, and the only three strings this file is allowed to
# know: the lookup helper and the ids of the two controls whose option text
# carries a count. Everything else -- every count NAME on either side -- is
# read out of the two files below.
LOOKUP = "jobCount"
CONTROLS = ("jobfilter", "jobkind")


def code(source: str) -> str:
    """The same source with its comments removed.

    Every comment in this page's script names the failure it prevents, so the
    prose quotes the very calls this file scans for -- "jobCount() then prints
    no number". Scanning the raw text would collect a name that appears only in
    a sentence about the name. The three sibling files strip for this reason.
    The `//` rule refuses a match preceded by a colon so that a URL in the
    markup does not swallow the rest of its line.
    """
    return re.sub(r"/\*.*?\*/|<!--.*?-->|(?<!:)//[^\n]*", "", source, flags=re.S)


SCRIPT = code(HTML)


# ------------------------------------------------- what the server emits --


def _strings(node: ast.AST, constants: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Every string in a tuple expression, resolving the module's own names.

    The initial dict is built from `("all",) + KINDS + AUDIO_STATES + (...)`,
    so an evaluator that only understood literals would read three names and
    call the other nine absent -- and a contract test that quietly loses most
    of one side's list passes for the wrong reason.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (_strings(node.left, constants) + _strings(node.right, constants))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return tuple(item.value for item in node.elts
                     if isinstance(item, ast.Constant) and isinstance(item.value, str))
    if isinstance(node, ast.Name):
        return constants.get(node.id, ())
    return ()


def server_counts() -> set[str]:
    """The key set of the `counts` dict tts-long builds for every listing."""
    tree = ast.parse(SOURCE)
    constants: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            constants[target.id] = _strings(node.value, {})

    found: list[set[str]] = []
    for node in ast.walk(tree):
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if not any(isinstance(t, ast.Name) and t.id == "counts" for t in targets):
            continue
        value = node.value
        if isinstance(value, ast.DictComp) and len(value.generators) == 1:
            found.append(set(_strings(value.generators[0].iter, constants)))
    assert len(found) == 1, (
        f"{SERVER.name} no longer builds exactly one `counts` dict "
        f"comprehension ({len(found)} found); this parser has gone blind and "
        "would let any page name through")
    return found[0]


# ------------------------------------------------- what the page asks for --


class Options(HTMLParser):
    """The option values of the two selects, in markup order."""

    def __init__(self) -> None:
        super().__init__()
        self.found: dict[str, list[str]] = {}
        self._select: str | None = None

    def handle_starttag(self, tag, attrs):
        got = dict(attrs)
        if tag == "select":
            self._select = got.get("id")
            if self._select in CONTROLS:
                self.found.setdefault(self._select, [])
        elif tag == "option" and self._select in CONTROLS:
            self.found[self._select].append(got.get("value", ""))

    def handle_endtag(self, tag):
        if tag == "select":
            self._select = None


def filter_count() -> dict[str, tuple[str, str] | None]:
    """FILTER_COUNT, read as the [group, name] pairs the page really writes.

    `null` IS AN ENTRY AND NOT AN ABSENCE, which is the whole reason this
    reader knows two shapes. An option whose set has no single count on the
    server -- "Playable" is audio=present,pending, two states -- can either ask
    for a name nothing emits, which is the defect this file exists for, or say
    in the table that it asks for nothing. Read as a missing key instead, the
    option would fall through to the fallback below and be recorded as asking
    for its own value, and this file would fail a page that had done the right
    thing.
    """
    start = SCRIPT.index("const FILTER_COUNT")
    literal = SCRIPT[start:SCRIPT.index("};", start)]
    table: dict[str, tuple[str, str] | None] = {
        key: None for key in re.findall(r"(\w+)\s*:\s*null\b", literal)}
    table.update({key: (group, name) for key, group, name in re.findall(
        r"""(\w+)\s*:\s*\[\s*["']([^"']*)["']\s*,\s*["']([^"']*)["']\s*\]""",
        literal)})
    return table


def query_of(key: str) -> dict[str, str]:
    """One filter option's own query, as the parameter names and values it sends.

    Read from JOB_FILTERS rather than written down here, for the reason
    everything else in this file is: a copy of the query beside a copy of the
    count is two copies free to agree with each other and with neither half of
    the page.
    """
    filters = SCRIPT[SCRIPT.index("const JOB_FILTERS"):SCRIPT.index("const JOB_KINDS")]
    entry = filters[filters.index(f"{key}:"):]
    found = re.search(r"query:\s*\{([^}]*)\}", entry)
    assert found, f"the {key} option has no query at all"
    return dict(re.findall(r"""(\w+)\s*:\s*["']([^"']*)["']""", found.group(1)))


def page_counts() -> set[str]:
    """Every count name the page looks up, from all three places it does it.

    THE NAME IS WHAT HAS TO MATCH, not the group beside it. jobCount(group,
    name) reads the flat key first and only falls back to `counts[group][name]`
    for a grouped answer this service does not send -- so a name the server
    never emits is null under both readings, and null prints nothing at all.

    The three places are the literal calls, the FILTER_COUNT table used by the
    "which runs" select, and the kind select, whose option values are passed
    straight in as names. The kind select's aggregate option is the one that
    takes a literal call instead; including its value here costs nothing and
    keeps this reader from having to know which option that is.
    """
    options = Options()
    options.feed(HTML)
    assert set(options.found) == set(CONTROLS), (
        f"the two counted selects are no longer {CONTROLS}: found "
        f"{sorted(options.found)}")

    table = filter_count()
    names = {name for _, name in re.findall(
        rf"""\b{LOOKUP}\(\s*["'][^"']*["']\s*,\s*(["'])([^"']*)\1\s*\)""", SCRIPT)}
    for value in options.found["jobfilter"]:
        # The fallback is real code -- an option that is not a key of
        # FILTER_COUNT is looked up as ["", option.value] -- so an option with
        # no entry asks for its own value. An entry of null asks for nothing.
        if value in table:
            if table[value] is not None:
                names.add(table[value][1])  # type: ignore[index]
        else:
            names.add(value)
    names |= set(options.found["jobkind"])
    names |= {pair[1] for pair in table.values() if pair is not None}
    return names


# ------------------------------------------------------------ the tests --


def test_neither_reader_has_gone_blind():
    """A contract test with an empty side passes while asserting nothing.

    This is the failure mode this workflow has already shipped three times: a
    stub that invented a shape and agreed with the bug, a harness that read a
    stale copy of the page, and eighty-five tests that each checked one half.
    If a rewrite moves either list somewhere these parsers cannot see, THIS
    fails first and says so, rather than the subset test going quietly green.
    """
    emitted, asked = server_counts(), page_counts()
    assert len(emitted) >= 5, f"only {sorted(emitted)} read out of {SERVER.name}"
    assert len(asked) >= 5, f"only {sorted(asked)} read out of {PAGE.name}"
    assert emitted & asked, (
        "the two sides now have NO name in common, which is not a rename, it "
        f"is a different feature: server {sorted(emitted)} page {sorted(asked)}")


def test_an_option_may_only_give_up_its_count_when_no_count_could_mean_it():
    """THE ESCAPE HATCH, KEPT NARROW.

    An entry of null is how the page says "this option has no number", and it
    is the honest answer for "Playable", whose query is audio=present,pending:
    two states, a count per state, and no one number over them. Summing the two
    here would be arithmetic on the control that decides what the reader can
    see, and wrong as well as invented -- every filter is answered with the
    live and the failed rows too, so the sum would be smaller than the list
    under it.

    It is also, unguarded, a way to make this file green by taking the number
    off whichever option has drifted. So a null is allowed ONLY where the
    option's own query asks for more than one state; an option that asks for
    exactly one has a count that means it, and must name it.
    """
    table = filter_count()
    silent = sorted(key for key, spec in table.items() if spec is None)
    for key in silent:
        wanted = list(query_of(key).values())
        assert any("," in value for value in wanted), (
            f"the {key} option has given up its count while asking for "
            f"{wanted or 'everything'} -- one state has a count that means it, "
            "and the number belongs on the option")


def test_a_count_may_only_label_the_set_ITS_OWN_FILTER_ASKS_FOR():
    """THE OTHER DIRECTION, AND THE ONE THAT SHIPPED TWICE.

    The rule above catches an option that has dropped a number it could have
    had. It says nothing at all about an option that KEEPS a number meaning
    something else, and that is exactly what went out: "Failures" sends
    status=failed,cancelled and this table asked for `failed` alone, so the
    option printed "Failures (1)" over three rows. Measured against the real
    route. The page's own comment forbade it in the paragraph directly above
    the line, for the option one line up.

    A number smaller than the list under it is worse than no number: the rows
    it leaves out are the ones somebody opened this tab to find, and the count
    is the control telling them they are not there.

    So the shape is asserted rather than the spelling. An option that names one
    state must ask for THAT state, in the group it named it under; an option
    that names two has no single count that means it and must say nothing; and
    the option that asks for everything is the aggregate. Both sides are read
    out of the file, so renaming a state breaks nothing here and mismatching
    one fails immediately.
    """
    for key, spec in filter_count().items():
        wanted = query_of(key)
        if spec is None:
            continue  # the test above governs the silent ones
        group, name = spec
        if not wanted:
            assert (group, name) == ("", "all"), (
                f"the {key} option filters nothing and is counted as "
                f"{group or 'flat'}/{name}, which is a subset labelling the whole")
            continue
        assert len(wanted) == 1, (
            f"the {key} option sends {wanted} and carries a single count "
            f"{group}/{name}: one number cannot mean two parameters")
        [(param, value)] = wanted.items()
        assert group == param, (
            f"the {key} option filters on `{param}` and is counted under "
            f"`{group}`, so the number is drawn from a different question")
        assert "," not in value, (
            f"the {key} option asks for the {value.count(',') + 1} states "
            f"{value!r} and prints the count of one of them. This is the "
            "Failures bug: the number means a SMALLER set than the list under "
            "it. Either the count goes (null, as playable does) or the query "
            "narrows to the one state the count means.")
        assert name == value, (
            f"the {key} option asks for {value!r} and prints the count of "
            f"{name!r} -- a number that means a different set from its filter")


def test_the_page_only_asks_for_counts_the_server_actually_emits():
    """THE CONTRACT. The key is "all"; the server owns it and does not move.

    A name the page asks for and the server does not send is not an error
    anywhere: jobCount returns null, the option keeps its plain label, and the
    control silently stops saying how many runs it is hiding. Nothing logs, no
    request fails, and both suites stay green -- which is exactly how "total"
    survived a fifteen-agent build.
    """
    emitted, asked = server_counts(), page_counts()
    missing = sorted(asked - emitted)
    assert not missing, (
        f"{PAGE.name} asks for {missing}, and {SERVER.name} emits none of "
        f"them. It emits {sorted(emitted)}. Every option keyed to a missing "
        "name shows no count at all.")
