"""The prose, checked against the deployment it describes.

A README that describes a surface the service no longer has is not a cosmetic
problem here. `services/tts-long/README.md` said in two places that there was
one model and that refusing `model` "would be theatre", and every one of the
three sentences it justified stopped being true the day a second engine was
reachable over the API.
"""

from __future__ import annotations

import pytest

TTS_LONG_README = "services/tts-long/README.md"
GATEWAY_README = "services/gateway/README.md"
UI_README = "services/ui/README.md"
ADR = "docs/adr/0008-two-engines-and-both-stay-jobs.md"
ADR_VOXTRAL = "docs/adr/0009-a-third-engine-that-cannot-run-here.md"


def _read(root, rel: str) -> str:
    return (root / rel).read_text(encoding="utf-8")


def _csv(value: str) -> list[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def test_the_readme_no_longer_says_there_is_one_model_here(root):
    """Two sentences, both justifications, both false the moment turbo ships.

    "Accepted, ignored — there is one model here" described `model`, which is
    now the engine selector, and "refusing the field on the strength of a name
    would be theatre" was the reason given for not doing what this design does.
    """
    text = _read(root, TTS_LONG_README).lower()
    for dead in ("there is one model here",
                 "there is one model here and refusing the field"):
        assert dead not in text, (
            f"{TTS_LONG_README} still says {dead!r}, and it is not true any "
            f"more: `model` selects the engine.")
    assert "accepted, ignored" not in text, (
        f"{TTS_LONG_README} still describes a field as accepted and ignored. "
        f"Every field on this service is honoured or refused by name.")


def test_the_readme_documents_every_engine_the_deployment_advertises(
        root, env_of):
    """An engine on the API and absent from the README is an undocumented API.

    It is also how somebody finds out about `chatterbox-turbo` by reading a
    400.
    """
    text = _read(root, TTS_LONG_README)
    for engine in _csv(env_of("tts-long").get("TTS_ENGINES", "chatterbox")):
        assert engine in text, (
            f"compose.yaml advertises the engine {engine!r} and "
            f"{TTS_LONG_README} never names it.")


def test_the_gateway_readme_documents_every_model_it_routes(root, env_of):
    text = _read(root, GATEWAY_README)
    routed = _csv(env_of("voice-gateway").get("GATEWAY_LONG_MODELS",
                                              "chatterbox,tts-long"))
    for model in routed:
        assert model in text, (
            f"the gateway routes model {model!r} and {GATEWAY_README} never "
            f"names it.")


def test_the_ui_readme_no_longer_says_the_voice_picks_the_engine(root):
    """It picks the BACKEND. Two engines share every voice on this stack.

    Leaving the old sentence in is how a reader concludes the engine control
    they are looking at cannot exist, or that registering a second clip is the
    way to reach turbo.
    """
    text = _read(root, UI_README)
    assert "**picking the voice picks the\nengine**" not in text
    assert "picking the voice picks the engine" not in " ".join(text.split()), (
        f"{UI_README} still says the voice picks the engine. The engine is a "
        f"request field; the voice picks the backend.")


@pytest.mark.parametrize("cost", [
    "1.5426",      # the measured rate, not a rounded 1.5
    "0.6531",      # what it is 2.36x faster than
    "67.5",        # cold load, against baseline's 22.2 s
    "22.2",
    "27 LUFS",     # quieter by design
    "hp.emotion_adv",   # why exaggeration and cfg_weight cannot be honoured
    "language_id",      # why there is no language
    "RTX 3070",         # the hardware every number above was taken on
])
def test_the_adr_records_what_turbo_costs_and_not_only_what_it_buys(root, cost):
    """A decision record that lists only the upside is a sales page.

    Each of these is a number or a name somebody would otherwise re-derive, and
    two of them are the reason a field is a 400 rather than a shrug.
    """
    assert cost in _read(root, ADR), (
        f"{ADR} does not mention {cost!r}.")


def test_every_link_to_an_adr_resolves(root):
    """A decision record nobody can open is a decision record nobody reads.

    Four files now point at ADR 0008 for the reason turbo stays a job. Renaming
    it without fixing them would leave four dead ends at the exact question
    somebody has when they see 1.5426x.
    """
    import re
    broken = []
    for path in [root / "compose.yaml", *(root / "services").glob("*/README.md"),
                 *(root / "docs" / "adr").glob("*.md"), root / "README.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for ref in re.findall(r"\b(?:docs/adr/)?(\d{4}-[a-z0-9-]+\.md)", text):
            if not (root / "docs" / "adr" / ref).exists():
                broken.append(f"{path.relative_to(root)} -> {ref}")
    assert not broken, f"links to ADRs that do not exist: {sorted(set(broken))}"


def test_the_adr_records_that_turbo_stays_a_job_and_why(root):
    """The owner's decision, and the reason, in the document that outlives us.

    Turbo crossed realtime, so the obvious next move is a synchronous route.
    The record has to say out loud that this was considered and refused, or the
    next person reads 1.54x as an oversight.
    """
    flat = " ".join(_read(root, ADR).split()).lower()
    assert "stays a job" in flat or "stay jobs" in flat
    assert "synchronous" in flat, (
        f"{ADR} must say that a synchronous turbo route was refused, not merely "
        f"that turbo is a job.")


# -- the third engine, and the settings its owner intends to change ---------

@pytest.mark.parametrize("cost", [
    "0.104",              # the rate, at the settings that ship
    "0.354",              # and the rate at the wrapper's own default, so that
                          # nobody reads 0.104 as "a faster setting exists"
    "8265",               # the load peak
    "8192",               # ...on a card this big. Both numbers or neither
    "63",                 # the load, paid every time
    "28.8",               # of which this much is the quantisation alone
    "7.49",               # the download
    "flow_steps",         # the knob the owner tuned by ear
    "cfg_alpha",          # and the one a caller will confuse with cfg_weight
    "speaker encoder",    # why it cannot clone
    "24000",              # the sample-rate trap, both halves
    "48000",
    "RTX 3070",           # the hardware every number above was taken on
    "torchao",            # why there is no processor lane
    "--quantized",        # the flag that looks like the fix and is not
])
def test_the_adr_records_what_voxtral_costs_and_not_only_what_it_buys(
        root, cost):
    """A decision record that lists only the upside is a sales page.

    Voxtral's upside is one sentence -- twenty voices in nine languages,
    Portuguese included. Everything else about it is a bill, and each item here
    is a number or a name somebody would otherwise spend an afternoon
    re-deriving on a card that is busy. Two of them are traps rather than
    costs: the 24000/48000 pair is a bug in the upstream repository that makes
    audio play at half speed, and `--quantized` is a flag that looks like it
    removes the 63-second load and does not.
    """
    assert cost in _read(root, ADR_VOXTRAL), (
        f"{ADR_VOXTRAL} does not mention {cost!r}.")


def test_the_adr_records_that_voxtral_cannot_clone_and_cannot_run_here(root):
    """The two facts that decide every other line of the design.

    Get either wrong and the rest follows wrongly: a reader who thinks it
    clones goes looking for the reference-audio parameter, and a reader who
    thinks it merely runs SLOWLY here goes looking for the CPU seed to tune.
    Neither thing exists, and the record has to say so rather than imply it.
    """
    flat = " ".join(_read(root, ADR_VOXTRAL).split()).lower()
    assert "not a cloning engine" in flat or "cannot clone" in flat, (
        f"{ADR_VOXTRAL} must say plainly that this checkpoint cannot clone.")
    assert "no processor path" in flat or "cannot run" in flat, (
        f"{ADR_VOXTRAL} must say plainly that it has no local lane, not that "
        f"it is slow here.")


def test_a_runner_only_engine_is_documented_as_one(root, env_of, catalogue):
    """AN ENGINE THAT CAN 503 AND A README THAT DOES NOT SAY SO.

    `TTS_ALLOW_RUNNER_ONLY_ENGINES=1` buys a real, visible failure mode: a
    request refused because a machine in somebody's house is switched off. If
    the deployment takes that trade, the document a caller reads has to name the
    engine it applies to and the status code they will get -- otherwise the
    first person to meet it finds out from a 503 with no context, which is the
    same class of surprise as finding out about an engine by reading a 400.
    """
    env = env_of("tts-long")
    engines = _csv(env.get("TTS_ENGINES", "chatterbox"))
    local = set(_csv(env.get("TTS_LOCAL_ENGINES", ",".join(engines))))
    text = _read(root, TTS_LONG_README)
    for engine in [e for e in engines if e not in local]:
        assert engine in text, (
            f"{engine!r} is advertised and cannot run locally, and "
            f"{TTS_LONG_README} never names it.")
        assert "engine_unavailable" in text, (
            f"{engine!r} can be refused with 503 engine_unavailable and "
            f"{TTS_LONG_README} never mentions that code.")
        assert "TTS_ALLOW_RUNNER_ONLY_ENGINES" in text, (
            f"{TTS_LONG_README} must document the key that permits "
            f"{engine!r} to be advertised without a local lane.")


def test_every_tunable_this_deployment_sets_has_a_documented_default(
        root, env_of, catalogue):
    """"I WILL TUNE IT LATER" IS A REQUIREMENT AND THIS IS IT, AS A TEST.

    Every quality setting on this engine is a configuration key precisely so
    that changing it never means editing code. That promise is only kept if the
    reader can FIND the key: a value set in compose.yaml and absent from the
    service's own key table is a setting somebody has to read a YAML file to
    discover, and the first thing they will do instead is edit the source.

    A per-LANE key may be documented in its templated form
    (`TTS_REALTIME_FACTOR_<LANE>_<ENGINE>`), because what it means does not
    change with the engine and its default is "the catalogue seed" either way.

    A per-ENGINE CONTROL key may NOT. `TTS_<ENGINE>_<FIELD>` is a row in that
    table already, and accepting it here made this test pass while
    `TTS_VOXTRAL_FLOW_STEPS` was spelled wrong everywhere else in the file --
    caught by mutating the README and watching this test stay green. A catch-all
    that matches every possible per-engine key documents none of them: it cannot
    tell the reader that the default is 32, or that the upstream default is 8,
    which is the entire reason the key exists.
    """
    env = env_of("tts-long")
    text = _read(root, TTS_LONG_README)
    slugs = sorted((catalogue.slug(e) for e in catalogue.CATALOGUE_IDS),
                   key=len, reverse=True)
    lanes = ("LOCAL", "RUNNER")
    undocumented = []
    for key in env:
        if not key.startswith("TTS_") or key in text:
            continue
        forms = set()
        for slug in slugs:
            if key.endswith("_" + slug):
                stem = key[: -len(slug)]
                forms.add(stem + "<ENGINE>")
                for lane in lanes:
                    forms.add(stem.replace("_" + lane + "_", "_<LANE>_")
                              + "<ENGINE>")
            if key.startswith("TTS_" + slug + "_"):
                # THE FIELD STAYS SPELLED OUT. `TTS_<ENGINE>_<FIELD>` is
                # deliberately not accepted -- see the docstring.
                forms.add("TTS_<ENGINE>_" + key[len("TTS_" + slug + "_"):])
        if not any(form in text for form in forms):
            undocumented.append(key)
    assert not undocumented, (
        f"compose.yaml sets {sorted(undocumented)} on tts-long and "
        f"{TTS_LONG_README} documents neither the key nor its templated form. "
        f"A knob the owner is expected to tune has to be findable in the "
        f"document, not only in the deployment file.")


# -- an engine the deployment retired, and the prose that has to follow it ---

ADR_RETIRED = "docs/adr/0010-the-third-engine-was-measured-and-retired.md"

# EVERY DOCUMENT A READER MEETS BEFORE THEY MEET THE API. The decision records
# are deliberately NOT here: an ADR is a dated account of what was decided, and
# a record that has to be re-edited every time the deployment changes is a
# repository that rewrites its own history. What a superseded record owes the
# reader is a Status line pointing forward, and the test after this one is
# where that is checked instead.
def _reader_facing(root):
    return ["compose.yaml", "README.md",
            *(str(p.relative_to(root))
              for p in sorted((root / "services").glob("*/README.md")))]


# Said of the engine, not of a knob: each one is a claim about whether a caller
# can type this name HERE. "switched off" is deliberately absent -- every one of
# these documents says "with the runner switched off" about something else.
RETIRED_PHRASES = ("not enabled", "does not enable", "has not enabled",
                   "retired")

# How far from the first mention the disclaimer may sit. A paragraph, not a
# file: a README that names an engine forty times and disowns it once in an
# appendix is the document this test exists to fail.
NEAR = 400


def test_no_reader_facing_document_offers_an_engine_this_deployment_retired(
        root, env_of, catalogue):
    """TWO HALVES OF ONE RETIREMENT, AND ONLY ONE OF THEM DONE.

    Taking an engine out of TTS_ENGINES and GATEWAY_LONG_MODELS is two lines.
    Every other document in the tree goes on offering it: a table listing three
    engines, a README telling somebody to send `model: "voxtral"`, a compose
    comment recommending a runner service id for it. That is this project's
    recurring defect -- two halves of one feature that disagree while every
    suite stays green -- and the previous round of it left compose.yaml
    recommending a lane that had already been deleted from the code.

    The rule: a document may discuss an engine this deployment has not enabled,
    and it often should, because the code is still there and a bigger card can
    run it. What it may not do is introduce it without saying so. THE FIRST
    MENTION IN EACH DOCUMENT IS WHAT IS CHECKED, within a paragraph either
    side, because that is the sentence a reader forms their belief from.

    Not every mention: a section that documents the engine properly -- its
    voices, its refusals, its settings -- would then need the disclaimer in
    every paragraph, which is the kind of repetition a reader learns to skip
    past, and skipping past it is the failure this test is for.
    """
    enabled = set(_csv(env_of("tts-long").get("TTS_ENGINES", "chatterbox")))
    retired = sorted(set(catalogue.CATALOGUE_IDS) - enabled)
    if not retired:
        # NOT A GAP. Every engine in the catalogue is enabled here, so there
        # is no disowning for any document to do.
        pytest.skip("this deployment enables every engine in the catalogue")
    silent = []
    for rel in _reader_facing(root):
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").lower()
        for engine in retired:
            first = text.find(engine)
            if first < 0:
                continue
            window = text[max(0, first - NEAR):first + NEAR]
            if any(phrase in window for phrase in RETIRED_PHRASES):
                continue
            silent.append(f"{rel}, first at character {first}")
    assert not silent, (
        f"these documents introduce {retired}, which TTS_ENGINES does not "
        f"offer, without saying so within a paragraph of the first mention: "
        f"{silent}. Say which it is -- 'not enabled' or 'retired' -- where the "
        f"name first appears, and point at {ADR_RETIRED}.")


def test_a_superseded_decision_record_says_so_and_its_successor_agrees(root):
    """A DECISION THAT TWO RECORDS TELL DIFFERENTLY IS WORSE THAN NEITHER.

    ADR 0009 read "**Status:** accepted" over a decision to ship a third engine
    for as long as it took somebody to notice that the engine had been retired
    in compose.yaml. A reader who opens the record first -- which is what the
    records are FOR -- is told the deployment offers something it does not, by
    the one file in the tree whose whole job is to be the account of record.

    Both directions are checked, because one direction is how the pair drifts:
    a Status line pointing at a successor that never claims it, or a successor
    claiming a record that still reads accepted.
    """
    import re
    adrs = sorted((root / "docs" / "adr").glob("*.md"))
    texts = {p.name: p.read_text(encoding="utf-8") for p in adrs}
    status = re.compile(
        r"^\*\*Status:\*\*\s*superseded by \[[^\]]+\]\(([^)]+)\)", re.M)
    supersedes = re.compile(r"^\*\*Supersedes:\*\*\s*\[[^\]]+\]\(([^)]+)\)",
                            re.M)
    faults = []
    forward = {}
    for name, text in texts.items():
        m = status.search(text)
        if not m:
            continue
        target = m.group(1)
        forward[name] = target
        if target not in texts:
            faults.append(f"{name} is superseded by {target}, which does not "
                          f"exist")
            continue
        back = supersedes.search(texts[target])
        if back is None or back.group(1) != name:
            faults.append(f"{name} says {target} supersedes it and {target} "
                          f"does not say so in its own header")
    for name, text in texts.items():
        m = supersedes.search(text)
        if m is None:
            continue
        superseded = m.group(1)
        if superseded not in texts:
            faults.append(f"{name} supersedes {superseded}, which does not "
                          f"exist")
            continue
        if forward.get(superseded) != name:
            faults.append(
                f"{name} supersedes {superseded}, and {superseded} still "
                f"reads "
                f"{texts[superseded].splitlines()[2].strip()!r}")
    assert not faults, "; ".join(faults)


@pytest.mark.parametrize("evidence", [
    "8265",          # the load peak, measured
    "8288",          # ...and the top of the range repeated runs found
    "8192",          # ...on a card this big. Every one of the three or none
    "segfault",      # what happens past the margin, which no handler catches
    "1.000",         # the best transcript match against the script
    "0.682",         # and the worst, on the same script at the same settings
    "0.104",         # the rate, the slowest figure in this repository
    "speaker encoder",   # what it has none of, which is why 0.104x buys nothing
    "GAB-634",       # where the full run log is
    "1.594",         # turbo's rate: what SHIPS, so the record is not read as
    "0.635",         # a retreat from the whole round
])
def test_the_retirement_record_carries_the_measurements_that_decided_it(
        root, evidence):
    """A RETIREMENT WITH NO NUMBERS IN IT IS A CHANGE OF MIND.

    This engine was built, reviewed and documented over a full round. The only
    thing that can outrank that record is measurement, so each item here is a
    figure somebody would otherwise re-derive on a card that is busy -- or, in
    the last two rows, the figure that says which parts of the round survived.
    """
    # FLATTENED, BECAUSE A LINE BREAK IS NOT A DELETION. "speaker encoder" wraps
    # across two lines in the record and a raw substring test would report it
    # missing -- a failure that teaches the next person to reflow the prose
    # rather than to keep the evidence.
    flat = " ".join(_read(root, ADR_RETIRED).split())
    assert evidence in flat, (
        f"{ADR_RETIRED} does not mention {evidence!r}.")
