"""What `compose.yaml` promises, checked against what the code can deliver.

Two failures this repository has already shipped are what these tests are for.

The first: `compose.yaml` documented `TTS_RUNNER_CPU_MIN_PCT` and
`TTS_RUNNER_CPU_WHEN_BACKLOG_S` as live knobs and recommended an order naming a
lane the code did not have, so **an operator following this repository's own
advice configured a lane that silently did no work.**

The second is the shape this file mostly guards: two halves of one feature, one
of them silent. A `model` string the gateway routes and tts-long has not enabled;
an engine advertised on the API that only this deployment's runner can render; a
per-engine knob for a control the engine does not have. Every one of those is
audible here and inaudible in production until somebody reads a waveform.
"""

from __future__ import annotations

import re

import pytest

from conftest import KEY, keys_read_by_code

TTS_LONG = "tts-long"
GATEWAY = "voice-gateway"


def _csv(value: str) -> list[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


# -- the engine set ---------------------------------------------------------

def test_compose_never_advertises_an_engine_the_code_has_no_catalogue_for(
        env_of, catalogue):
    """TTS_ENGINES is checked against the one table, not against a memory.

    A name here that the catalogue does not know is a deployment offering a
    checkpoint nothing can load. tts-long refuses to start on it; this says so
    before anybody deploys.
    """
    named = set(_csv(env_of(TTS_LONG).get("TTS_ENGINES", "chatterbox")))
    unknown = named - set(catalogue.CATALOGUE_IDS)
    assert not unknown, (
        f"compose.yaml offers {sorted(unknown)} in TTS_ENGINES, and "
        f"voice_common.engines.CATALOGUE knows only "
        f"{sorted(catalogue.CATALOGUE_IDS)}.")


def test_the_gateway_and_tts_long_agree_on_the_engine_set(env_of):
    """THE ONE PLACE TWO SERVICES' MODEL TABLES CAN DRIFT.

    GATEWAY_LONG_MODELS decides what the gateway routes to tts-long AND what
    `GET /v1/models` advertises. TTS_ENGINES decides what tts-long will actually
    accept. A name in the first and not the second is a 400 from a model the
    gateway told the client it had; a name in the second and not the first
    reaches Kokoro or a 404 instead of the engine it names.
    """
    long_models = set(_csv(env_of(GATEWAY).get("GATEWAY_LONG_MODELS",
                                               "chatterbox,tts-long")))
    engines = set(_csv(env_of(TTS_LONG).get("TTS_ENGINES", "chatterbox")))
    assert long_models - {"tts-long"} == engines, (
        f"GATEWAY_LONG_MODELS is {sorted(long_models)} and TTS_ENGINES is "
        f"{sorted(engines)}; they must match once the `tts-long` alias is "
        f"removed.")


def test_the_default_engine_is_one_this_deployment_offers(env_of):
    """TTS_DEFAULT_ENGINE is what `tts-1` and an absent `model` resolve to.

    Pointing it at an engine TTS_ENGINES does not name would 400 every
    unmodified OpenAI client on the stack at once.
    """
    env = env_of(TTS_LONG)
    engines = _csv(env.get("TTS_ENGINES", "chatterbox"))
    default = (env.get("TTS_DEFAULT_ENGINE") or engines[0]).strip().lower()
    assert default in engines, (
        f"TTS_DEFAULT_ENGINE is {default!r} and TTS_ENGINES offers "
        f"{engines}.")


def _opted_out(env: dict) -> bool:
    value = (env.get("TTS_ALLOW_RUNNER_ONLY_ENGINES") or "0").strip().lower()
    return value not in {"0", "", "false", "no"}


def test_every_advertised_engine_can_run_with_the_runner_switched_off(
        env_of, catalogue):
    """SPRING IS SPEED, NEVER AVAILABILITY.

    An engine that only exists while somebody's gaming PC is switched on is a
    spring-shaped hole in the API, and the first Friday evening that hole eats a
    job. tts-long refuses to start in this state; this catches it in the file
    that creates it, which is where it is cheap.

    THIS USED TO SKIP ENTIRELY THE MOMENT THE OPT-OUT WAS SET, and setting it is
    exactly what shipping `voxtral` did. A test that answers "not checking" as
    soon as the risky configuration is the live one is the shape of dead test
    this directory exists to prevent, so the opt-out now NARROWS the assertion
    instead of switching it off: an orphan is allowed only where the catalogue
    itself says the checkpoint has no local class. A typo in TTS_LOCAL_ENGINES
    is still caught, and so is an engine that could have run here and was left
    out by accident.
    """
    env = env_of(TTS_LONG)
    engines = set(_csv(env.get("TTS_ENGINES", "chatterbox")))
    local = set(_csv(env.get("TTS_LOCAL_ENGINES", env.get("TTS_ENGINES",
                                                          "chatterbox"))))
    orphans = engines - local
    if not _opted_out(env):
        assert not orphans, (
            f"TTS_ENGINES offers {sorted(orphans)} but TTS_LOCAL_ENGINES "
            f"cannot run it, so it would exist only while the runner is up.")
        return
    avoidable = sorted(
        e for e in orphans
        if getattr(catalogue.CATALOGUE.get(e), "local_class", None) is not None)
    assert not avoidable, (
        f"TTS_ALLOW_RUNNER_ONLY_ENGINES is set, but {avoidable} could run on "
        f"this container -- the catalogue gives each of them a local_class. "
        f"The opt-out is for engines with NO local path at all; using it to "
        f"cover an engine that has one hides a missing name in "
        f"TTS_LOCAL_ENGINES.")


def test_the_default_engine_can_run_with_the_runner_switched_off(
        env_of, catalogue):
    """THE NARROWED INVARIANT, AND THE ONE THE OPT-OUT MAY NEVER TOUCH.

    ADR 0008 said every advertised engine runs locally. ADR 0009 narrows that to
    the default engine, because `voxtral` cannot satisfy the old rule. What must
    survive is this: a caller that does not TYPE a runner-only engine's name
    cannot be failed by somebody switching a gaming PC off. `tts-long`, an
    absent `model` and OpenAI's three names all resolve to TTS_DEFAULT_ENGINE,
    so if that one has no local lane the whole API goes away with the runner --
    which is the failure the opt-out looks like it permits and must not.
    """
    env = env_of(TTS_LONG)
    engines = _csv(env.get("TTS_ENGINES", "chatterbox"))
    local = set(_csv(env.get("TTS_LOCAL_ENGINES", ",".join(engines))))
    default = (env.get("TTS_DEFAULT_ENGINE") or engines[0]).strip().lower()
    assert default in local, (
        f"TTS_DEFAULT_ENGINE is {default!r} and TTS_LOCAL_ENGINES is "
        f"{sorted(local)}. Every alias resolves to the default engine, so a "
        f"default without a local lane makes every unmodified OpenAI client "
        f"fail whenever the runner is away.")
    facts = catalogue.CATALOGUE.get(default)
    assert getattr(facts, "local_class", "unset") is not None, (
        f"TTS_DEFAULT_ENGINE is {default!r} and the catalogue gives it no "
        f"local_class, so naming it in TTS_LOCAL_ENGINES cannot make it run "
        f"here.")


def test_no_engine_is_promised_a_local_lane_the_catalogue_cannot_give_it(
        env_of, catalogue):
    """THE `chatterbox-cpu` TRAP, IN THE OTHER DIRECTION.

    That rung was a lane named in configuration that no code could route work
    to. This is the same mistake made from the deployment side: adding
    `voxtral` to TTS_LOCAL_ENGINES because it looks like the other two. There is
    no processor path through an int4 tile-packed checkpoint -- torchao calls
    torch.cuda.get_device_capability() before any device dispatch -- so the lane
    would be configured, published on /health, and unable to do work.
    """
    env = env_of(TTS_LONG)
    local = _csv(env.get("TTS_LOCAL_ENGINES", env.get("TTS_ENGINES",
                                                      "chatterbox")))
    impossible = sorted(
        e for e in local
        if e in catalogue.CATALOGUE
        and getattr(catalogue.CATALOGUE[e], "local_class", "unset") is None)
    assert not impossible, (
        f"TTS_LOCAL_ENGINES names {impossible}, and the catalogue says each of "
        f"them has no local class at all. That is a lane configured and unable "
        f"to run anything, which is what the deleted `chatterbox-cpu` rung was.")


def test_no_per_engine_key_names_a_control_that_engine_lacks(env_of, catalogue):
    """A knob that does nothing is the house rule broken with a longer fuse.

    `TTS_CHATTERBOX_TURBO_EXAGGERATION` would be read by nothing, because
    chatterbox-turbo has no exaggeration control at all — hp.emotion_adv is
    False, so the conditioning layer is never built. tts-long makes that fatal
    at boot; this makes it fatal before the deploy.
    """
    # LONGEST SLUG FIRST. `CHATTERBOX` is a prefix of `CHATTERBOX_TURBO`, so
    # matching in catalogue order would read TTS_CHATTERBOX_TURBO_TEMPERATURE as
    # a `chatterbox` key for a control called `turbo_temperature` and report a
    # fault that is not there.
    slugs = sorted(((catalogue.slug(e), e) for e in catalogue.CATALOGUE_IDS),
                   key=lambda pair: -len(pair[0]))
    offenders = []
    for key in env_of(TTS_LONG):
        if not key.startswith("TTS_"):
            continue
        rest = key[len("TTS_"):]
        for slug, engine in slugs:
            if not rest.startswith(slug + "_"):
                continue
            field = rest[len(slug) + 1:].lower()
            controls = catalogue.CATALOGUE[engine].controls
            if field not in controls:
                offenders.append((key, engine, sorted(controls)))
            break
    assert not offenders, (
        "per-engine keys naming a control that engine does not have: "
        + "; ".join(f"{k} ({e} has {c})" for k, e, c in offenders))


# -- nothing is advertised that no longer exists ----------------------------

# Phrases that claim a NAMED key is dead. "a knob that is read by no code is
# worse than no knob" is deliberately not here: it is an aphorism about knobs in
# general, and matching it would flag every key that happens to share a
# paragraph with it.
DEAD_CLAIM = re.compile(
    r"read by nothing|gone from the code|are gone from it",
    re.IGNORECASE)

DOCS = ("compose.yaml",
        "services/tts-long/README.md",
        "services/gateway/README.md",
        "services/ui/README.md",
        "services/tts/README.md",
        "services/stt/README.md",
        "docs/adr/0007-two-lanes-not-three-rungs.md",
        "docs/adr/0008-two-engines-and-both-stay-jobs.md")


def _sentences(text: str) -> list[str]:
    flat = " ".join(line.lstrip().lstrip("#").strip() for line in text.splitlines())
    flat = re.sub(r"\s+", " ", flat)
    return flat.split(". ")


@pytest.mark.parametrize("doc", DOCS)
def test_a_document_never_lists_a_live_key_among_the_dead_ones(doc, root, source):
    """Rounding a deletion up is the same falsehood as the dead knob was.

    `compose.yaml` and the tts-long README both said `TTS_RUNNER_CPU_MAX_WAIT`
    and `TTS_REALTIME_FACTOR_RUNNER_CPU` were "read by nothing" while
    app/remote.py and app/main.py were still parsing both of them. A reader who
    trusts that sentence stops looking, and the second RunnerClient those keys
    still build goes on being constructed at every startup.
    """
    path = root / doc
    if not path.exists():
        pytest.skip(f"{doc} is not in this tree")
    for sentence in _sentences(path.read_text(encoding="utf-8")):
        if not DEAD_CLAIM.search(sentence):
            continue
        alive = keys_read_by_code(sentence, source)
        assert not alive, (
            f"{doc} says of {sorted(alive)}: {sentence.strip()!r} — but the "
            f"code still reads every one of them.")


def test_no_commented_out_knob_in_compose_is_read_by_nothing(compose_text, source):
    """A recommendation is advice, and advice for a key nothing reads is a trap.

    Every `# KEY: "value"` line in this file reads as a supported setting
    somebody can uncomment. That is exactly how `TTS_RUNNER_CPU_MIN_PCT` and
    `TTS_RUNNER_CPU_WHEN_BACKLOG_S` were configured into a lane that did no
    work: they were presented here, at their documented defaults, for months.
    """
    suggested = set()
    for line in compose_text.splitlines():
        m = re.match(r'^\s*#\s*((?:TTS|GATEWAY|STT|UI|AIV|RUNLOG)_[A-Z0-9_]+):\s*"',
                     line)
        if m:
            suggested.add(m.group(1))
    dead = {k for k in suggested
            if f'"{k}"' not in source and f"'{k}'" not in source}
    assert not dead, (
        f"compose.yaml presents {sorted(dead)} as commented-out settings and "
        f"no code reads them.")


def test_every_environment_key_this_deployment_sets_is_read_by_its_service(
        compose, source, catalogue):
    """A key that is set and read by nothing is a belief with no effect.

    Per-engine keys (`TTS_<ENGINE>_<FIELD>`) are built from the catalogue at
    runtime and never spelled literally in the source, which is the point of the
    slug: no key spells `turbo` by hand. They are checked by the test above
    instead.
    """
    slugs = tuple(catalogue.slug(e) for e in catalogue.CATALOGUE_IDS)
    templated = re.compile(
        r"^TTS_(?:RUNNER_SERVICE|COLD_LOAD_SECONDS|REALTIME_FACTOR_(?:LOCAL|RUNNER))_(?:"
        + "|".join(re.escape(s) for s in slugs) + r")$|^TTS_(?:"
        + "|".join(re.escape(s) for s in slugs) + r")_[A-Z0-9_]+$")
    dead: list[str] = []
    for name, svc in compose["services"].items():
        for key in (svc.get("environment") or {}):
            if not KEY.fullmatch(key) or templated.match(key):
                continue
            if f'"{key}"' in source or f"'{key}'" in source:
                continue
            # ASSEMBLED FROM A PREFIX AT RUNTIME, so no source file spells it.
            # voice-entrypoint.sh reads ${VOICE_TLS_PREFIX}_TLS_CERT, which is
            # how GATEWAY_TLS_CERT reaches uvicorn without ever appearing as a
            # literal anywhere.
            if f"_{key.split('_', 1)[1]}" in source:
                continue
            dead.append(f"{name}:{key}")
    assert not dead, (
        f"set in compose.yaml and read by no code: {sorted(dead)}")


def test_the_runner_only_opt_out_is_set_only_while_an_engine_needs_it(env_of):
    """A SAFETY CATCH LEFT OFF AFTER THE THING IT EXCUSED HAS GONE.

    TTS_ALLOW_RUNNER_ONLY_ENGINES=1 turns the boot refusal above into a 503 at
    submit. That trade is right for a deployment that deliberately offers an
    engine with no local lane, and it was right while `voxtral` was enabled
    here. Left set after the engine is retired it protects nothing and disarms
    everything: the next engine added to TTS_ENGINES without a local lane boots
    cleanly, is advertised on GET /v1/models, and disappears the first evening
    somebody switches a gaming PC on.

    Nothing at runtime can catch that, and that is the point. The service sees
    a flag set and no orphan to excuse, which is a legal state and a quiet one.
    Only this file knows it is also a pointless one.
    """
    env = env_of(TTS_LONG)
    engines = set(_csv(env.get("TTS_ENGINES", "chatterbox")))
    local = set(_csv(env.get("TTS_LOCAL_ENGINES", ",".join(sorted(engines)))))
    if not _opted_out(env):
        return
    assert engines - local, (
        "TTS_ALLOW_RUNNER_ONLY_ENGINES is set and every engine in TTS_ENGINES "
        "is named in TTS_LOCAL_ENGINES, so it excuses nothing and silently "
        "removes the refusal that stops the next runner-only engine being "
        "advertised. Unset it, or name the engine that needs it.")


def test_no_per_engine_key_is_set_for_an_engine_this_deployment_does_not_offer(
        env_of, catalogue):
    """A KEY THAT LOOKS LIVE, PARSES FOR NOBODY, AND CHANGES NO SOUND.

    Every per-engine key -- TTS_<ENGINE>_<FIELD>, TTS_RUNNER_SERVICE_<ENGINE>,
    TTS_COLD_LOAD_SECONDS_<ENGINE>, TTS_REALTIME_FACTOR_<LANE>_<ENGINE> -- is
    read by walking the ENABLED engines and building the key name from each
    one's slug. So a key for an engine TTS_ENGINES does not name is never
    looked up at all: it sits in this file at a carefully chosen value, reads
    like configuration, and does nothing. That is `TTS_RUNNER_CPU_MIN_PCT`
    again, arriving the other way round -- not a lane documented after the code
    went, but a key left behind after the engine went.

    NOTHING ELSE IN THIS DIRECTORY CATCHES IT. The test that asks whether every
    key is read by some code exempts the templated per-engine forms on purpose,
    because no source file spells them; the one that checks a key names a real
    control reads the CATALOGUE, which still carries the retired engine's row.
    Both stay green while this file configures an engine nobody can reach.
    """
    env = env_of(TTS_LONG)
    enabled = set(_csv(env.get("TTS_ENGINES", "chatterbox")))
    # LONGEST SLUG FIRST, for the same reason as the control test above:
    # CHATTERBOX is a prefix of CHATTERBOX_TURBO.
    slugs = sorted(((catalogue.slug(e), e) for e in catalogue.CATALOGUE_IDS),
                   key=lambda pair: -len(pair[0]))
    orphaned = []
    for key in env:
        if not key.startswith("TTS_"):
            continue
        for slug, engine in slugs:
            if engine in enabled:
                continue
            if key == f"TTS_{slug}" or key.startswith(f"TTS_{slug}_") \
                    or key.endswith(f"_{slug}"):
                orphaned.append(f"{key} ({engine})")
                break
    assert not orphaned, (
        f"compose.yaml sets {sorted(orphaned)} on tts-long, and TTS_ENGINES "
        f"offers {sorted(enabled)}. A per-engine key is built from an enabled "
        f"engine's slug, so these are never read. Remove them, or enable the "
        f"engine they belong to.")
