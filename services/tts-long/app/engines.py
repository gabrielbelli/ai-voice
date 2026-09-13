"""Which engine speaks a job, and every field that engine cannot honour.

WHY THIS MODULE EXISTS AT ALL. There is a second Chatterbox checkpoint now --
`chatterbox-turbo`, 2.36x the current model on the owner's 3070 -- and it is
not the same model with a speed knob. It speaks one language instead of
twenty-three, its emotion conditioning layer is never built, and it has no
classifier-free-guidance path. All three of those are things it ACCEPTS AND
DISCARDS: `generate()` takes `exaggeration` and `cfg_weight` as keyword
arguments, logs a warning and ignores them, and the deployment sets both from
compose.yaml, so a request that merely reached the wrong checkpoint would come
back sounding wrong with nothing anywhere reporting a fault. Measured: the
warning fired on all 21 segments of a test run and no caller could have seen it.

services/stt/app/openai_api.py states the house rule this file implements:
EVERY FIELD IS EITHER HONOURED OR REFUSED BY NAME. None is accepted and
dropped. A dropped field is a client that believes something false about the
audio it just received.

NO `if engine == "chatterbox-turbo"` IN THIS PACKAGE, HERE INCLUDED. Every
decision below walks `spec.controls`, `spec.languages` and the rest of the
catalogue row. A third engine is a row in that table and no code anywhere --
and `test_no_branch_on_an_engine_name` parses the AST for the branch that
would undo it.

THE THIRD ENGINE ARRIVED AND IT BROKE TWO PROXIES rather than the design. A
proxy is a test that happens to be right about every engine there has been so
far: `len(spec.languages) > 1` stood in for "takes a language_id", and a voice
being a file in TTS_VOICE_DIR stood in for "a voice". Voxtral speaks nine
languages WITHOUT taking a parameter for them, and carries its twenty speakers
as embeddings inside the weights, so both proxies now have a fact of their own
on the row -- `language_from_voice` and `voices`. That is the shape of every
change here: a proxy replaced by the fact it was standing in for.

CONFIGURATION IS READ ONCE, AT IMPORT, exactly as app.main reads its own: this
service's configuration changes on a restart, and a value that can be re-read
mid-process is a value two requests can disagree about.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Iterable

from voice_common.engines import (CATALOGUE, CATALOGUE_IDS, CONTROL_RANGES,
                                  WIRE_CONTROLS, EngineFacts, slug)

__all__ = ["CONTROL_RANGES", "DEFAULT_ENGINE", "ENGINES", "EngineSpec",
           "LOCAL_RESIDENT_MAX", "OPENAI_MODEL_NAMES", "PRESET_VOICE_ENGINE",
           "Refusal", "SERVICE_ALIAS", "WIRE_CONTROLS", "WITNESS_FIELDS",
           "assert_named_checkpoint", "assert_runtime", "catalogue_witness",
           "check_config", "checkpoint_witness", "defaults_for",
           "generate_kwargs", "refuse", "refuse_unavailable", "spec_for",
           "warn_voice_collisions"]

# The names OpenAI documents for this endpoint. They are ALIASES for this
# deployment's default engine, which is what they have always meant here, and
# the response says which engine actually ran in x-tts-engine. Honoured as
# aliases is not the same as accepted and dropped: there is no other engine
# they could name.
OPENAI_MODEL_NAMES = frozenset({"tts-1", "tts-1-hd", "gpt-4o-mini-tts"})
# This service's own name, for a caller that wants "whatever the long-form
# service is" without pinning a checkpoint.
SERVICE_ALIAS = "tts-long"

# WIRE_CONTROLS AND CONTROL_RANGES ARE voice_common's NOW, re-exported here so
# every existing importer is unchanged. They moved for the reason the catalogue
# moved: the wire's bounds and the config loader's bounds were two literals in
# two files, so TTS_EXAGGERATION=9 was accepted at boot and refused on every
# request it was then written onto. `language` stays separate from both: it is
# refused against spec.languages rather than against spec.controls, because a
# model can condition on a language without offering a dial for it.


@dataclass(frozen=True)
class Refusal:
    """A 4xx that has not happened yet, carried rather than raised.

    TWO ROUTES RENDER THIS AND THEY MUST NOT BE UNIFIED. /v1/audio/speech owes
    OpenAI's four-field envelope, which is what `code` and `param` are for;
    /jobs has answered `{"detail": ...}` since before any of that existed and
    something out there parses it. One function decides WHAT is refused; each
    route decides how its own callers are told.
    """

    status: int
    message: str
    code: str
    param: str


@dataclass(frozen=True)
class EngineSpec:
    """One engine as this deployment has it: the facts, plus what was configured.

    The facts are voice_common's and cannot be configured away. The rest --
    which runner service id to ask for, what a cold load costs here, what the
    sliders default to, whether the local lane may run it -- is this machine's,
    with a documented default for every key.
    """

    facts: EngineFacts
    runner_service: str
    cold_load_seconds: float
    # {field: value} for fields in `controls` ONLY, plus "language". An engine
    # with no such control has no entry, no environment key and no slider.
    defaults: dict[str, float | str]
    # Whether this machine's own CPU may speak it. See the startup invariant
    # at the bottom of this file: with no opt-out, every advertised engine is
    # local, because an engine that exists only while somebody's gaming PC is
    # on is an engine that disappears.
    local: bool

    @property
    def id(self) -> str:
        return self.facts.id

    @property
    def label(self) -> str:
        return self.facts.label

    @property
    def languages(self) -> tuple[str, ...]:
        return self.facts.languages

    @property
    def controls(self) -> frozenset[str]:
        return self.facts.controls

    @property
    def min_reference_seconds(self) -> float:
        return self.facts.min_reference_seconds

    @property
    def local_class(self) -> str | None:
        return self.facts.local_class

    @property
    def voices(self) -> tuple | None:
        """The fixed speakers this checkpoint ships, or None for a clip engine."""
        return self.facts.voices

    @property
    def voice_names(self) -> tuple[str, ...]:
        """Just the names, for a message. Empty for a clip-cloning engine."""
        return tuple(v.name for v in (self.facts.voices or ()))

    def language_of(self, voice: str) -> str | None:
        """Which language a preset voice speaks, or None if it is not one of them."""
        for preset in self.facts.voices or ():
            if preset.name == voice:
                return preset.language
        return None


def _split(raw: str) -> list[str]:
    """A comma-separated list, order kept, blanks dropped, duplicates dropped."""
    out: list[str] = []
    for part in raw.split(","):
        name = part.strip().lower()
        if name and name not in out:
            out.append(name)
    return out


def _flag(env: dict[str, str] | os._Environ, key: str, default: str) -> bool:
    return (env.get(key) or default).strip().lower() not in {"0", "false", "no", ""}


def _fatal(message: str) -> None:
    """Refuse to start, named, with the way out in the message.

    SystemExit AND NOT A WARNING, and the distinction is the whole reason this
    function exists. A warning at startup is read once, by whoever happened to
    be watching that terminal, and then scrolls away for the life of the
    deployment; the misconfiguration it described is still there a month later
    when somebody wonders why a slider does nothing. Every refusal below is a
    state the operator can fix in one line, so the container that will not
    start IS the bug report.
    """
    raise SystemExit(f"tts-long: {message}")


# ------------------------------------------------------------ the config --


def _engine_ids(env) -> list[str]:
    names = _split(env.get("TTS_ENGINES") or "chatterbox")
    if not names:
        _fatal("TTS_ENGINES is empty. It names the engines this deployment "
               "offers; the shipped default is 'chatterbox'.")
    unknown = [n for n in names if n not in CATALOGUE_IDS]
    if unknown:
        _fatal(f"TTS_ENGINES names {', '.join(unknown)}, which "
               f"{'are' if len(unknown) > 1 else 'is'} not in the engine "
               f"catalogue. Known engines: {', '.join(sorted(CATALOGUE_IDS))}.")
    return names


def _runner_service_for(env, facts: EngineFacts) -> str:
    """Which idlegpu service id carries this engine, per engine.

    TTS_RUNNER_SERVICE IS STILL HONOURED and only for `chatterbox`. It is set
    in compose files and in people's shells, it has only ever meant the one
    engine there was, and a variable somebody set that silently stops applying
    is the same class of failure as a field accepted and dropped.
    """
    specific = (env.get(f"TTS_RUNNER_SERVICE_{slug(facts.id)}") or "").strip()
    legacy = (env.get("TTS_RUNNER_SERVICE") or "").strip()
    if facts.id == "chatterbox" and legacy and specific and legacy != specific:
        # Named, both of them. Silently preferring one is how somebody spends
        # an afternoon wondering which of two variables the service read.
        import logging
        logging.getLogger("tts-long.engines").warning(
            "TTS_RUNNER_SERVICE=%s and TTS_RUNNER_SERVICE_%s=%s are both set; "
            "the per-engine one wins. Remove the legacy key.",
            legacy, slug(facts.id), specific)
    if specific:
        return specific
    if facts.id == "chatterbox" and legacy:
        return legacy
    return facts.runner_service


def _defaults_from_env(env, facts: EngineFacts) -> dict[str, float | str]:
    """The per-engine defaults, refusing any key the engine cannot honour.

    THE FAILURE THIS PREVENTS HAS A LONGER FUSE THAN A DROPPED REQUEST FIELD.
    A config key that does nothing is exactly the house-rule failure moved into
    a file nobody reads: somebody sets TTS_CHATTERBOX_TURBO_EXAGGERATION, hears
    no difference, and concludes the model is broken. The engine that has no
    such control has no such key, and saying so at boot is the only moment
    anybody is looking.
    """
    out: dict[str, float | str] = {}
    for field in WIRE_CONTROLS:
        key = f"TTS_{slug(facts.id)}_{field.upper()}"
        raw = env.get(key)
        if raw is None:
            continue
        if field not in facts.controls:
            _fatal(
                f"{key} is set, and {facts.id} has no {field} control: "
                f"{_why_no_control(facts, field)} Remove it, or set "
                f"TTS_{field.upper()}, which applies to every engine that has "
                f"the control.")
        out[field] = _control_value(key, field, raw)
    # The global keys, for the fields this engine does have. An engine with no
    # such control gets no default at all -- not the global one, not zero --
    # because a default is a value the caller did not choose, and a value the
    # engine discards is worse than no value.
    for field in WIRE_CONTROLS:
        if field in out or field not in facts.controls:
            continue
        raw = env.get(f"TTS_{field.upper()}")
        if raw is not None:
            out[field] = _control_value(f"TTS_{field.upper()}", field, raw)
    # NO LANGUAGE DEFAULT AT ALL FOR AN ENGINE THAT CARRIES IT ON THE VOICE,
    # AND THAT IS A BUG FIX RATHER THAN A NEW CASE. `language` used to be
    # written here unconditionally, so TTS_LANGUAGE=en -- which every
    # deployment sets -- would stamp `language: "en"` onto a pt_male job and
    # put `en` on the record beside a Portuguese speaker. `_choose` derives the
    # language from the chosen voice and records THAT.
    chosen = _default_language(env, facts)
    if chosen is not None:
        out["language"] = chosen
    return out


def _control_value(key: str, field: str, raw: str) -> float:
    """One control off the environment, typed and bounded by CONTROL_RANGES.

    THE FAILURE THIS PREVENTS IS A DEPLOYMENT DEFAULT THAT EVERY REQUEST IS
    THEN REFUSED FOR. `float(raw)` accepted TTS_FLOW_STEPS=200 at boot and the
    Pydantic field refused it on the wire at le=64 -- so the service started,
    looked well, and answered 422 to a body that had never mentioned the field.
    One table, read here and by the request models, is what makes that
    impossible rather than merely unlikely.

    The type matters as much as the range. flow_steps is a loop count:
    `range(32.0)` on the runner is a TypeError three and a half minutes into
    somebody's job.
    """
    kind, low, high = CONTROL_RANGES[field]
    try:
        value = kind(raw)
    except ValueError:
        _fatal(f"{key} is {raw!r}, which is not {'a whole number' if kind is int else 'a number'}. "
               f"{field} must be {'an integer' if kind is int else 'a number'} "
               f"between {low} and {high}.")
    if not (low <= value <= high):
        _fatal(f"{key} is {raw}, and {field} must be between {low} and {high}. "
               f"The wire refuses anything outside that range, so this "
               f"deployment would start and then answer 422 to requests that "
               f"never named the field.")
    return value


def _default_language(env, facts: EngineFacts) -> str | None:
    """This engine's default language, or None where it has no such thing."""
    key = f"TTS_{slug(facts.id)}_LANGUAGE"
    specific = (env.get(key) or "").strip().lower()
    if facts.language_from_voice:
        if not specific:
            return None
        # A deployment-wide default here could only ever agree with the voice
        # or contradict it, and the second is a record that says `en` beside a
        # Portuguese speaker.
        _fatal(f"{key} is set, and {facts.id}'s language is carried by the "
               f"voice embedding rather than passed to generate(). A "
               f"deployment-wide default would either agree with the voice or "
               f"contradict it. Pick a voice instead: its language is a "
               f"property of which tensor it is.")
    if specific:
        if specific not in facts.languages:
            _fatal(f"{key} is {specific!r} and {facts.id} does not speak it: "
                   f"{', '.join(facts.languages)}.")
        return specific
    shared = (env.get("TTS_LANGUAGE") or "en").strip().lower()
    if shared in facts.languages:
        return shared
    if len(facts.languages) == 1:
        # NOT A SUBSTITUTION, and the difference matters. This engine speaks
        # exactly one language, so a caller who named none is not having a
        # choice made for them -- there was no choice. A caller who NAMES a
        # language this engine cannot speak is still refused by R5.
        return facts.languages[0]
    _fatal(f"TTS_LANGUAGE is {shared!r} and {facts.id} does not speak it: "
           f"{', '.join(facts.languages)}. Set TTS_{slug(facts.id)}_LANGUAGE "
           f"to one it does, or remove {facts.id} from TTS_ENGINES.")
    return shared  # unreachable; _fatal raises


def _why_no_control(facts: EngineFacts, field: str) -> str:
    """The model-level reason a control is absent, in the model's own terms.

    "unsupported" is what an error message says when nobody looked. These are
    what was actually found in each package, and each one tells the reader
    whether waiting for a newer build could ever help.

    PER ENGINE NOW, OFF THE CATALOGUE ROW, and the move is what a third engine
    made necessary. This was one table shared by every engine, so the sentence
    about `exaggeration` described turbo's checkpoint -- "hp.emotion_adv is
    False" -- and would have been handed verbatim to somebody asking Voxtral
    for it, which is a Mistral flow-matching model with no `hp` at all. A
    reason that is wrong about the model is worse than "unsupported": it sends
    the reader looking at the wrong package.
    """
    reason = facts.control_absent_because.get(field)
    if reason:
        return reason
    return "this checkpoint's generate() does not read it."


def _check_facts(facts: EngineFacts) -> None:
    """A catalogue row that cannot be honoured, refused where it is written.

    THREE WAYS A ROW CAN BE WRONG IN A WAY NO REQUEST WOULD REVEAL, and every
    one of them was reachable while EngineFacts was six fields:

    * a `controls` entry outside WIRE_CONTROLS is a dial the model reads and
      the wire has no field for -- honoured by nothing, refused by nothing,
      invisible;
    * a wire control that is neither in `controls` nor in
      `control_absent_because` gets the generic "generate() does not read it",
      which is the "unsupported" this whole design exists to stop;
    * `voices=()` is a closed voice list with nothing in it, so every request
      naming any voice is refused and the engine can never speak.

    Fatal at import, because a catalogue row is code and the person who added
    it is the person looking.
    """
    stray = sorted(set(facts.controls) - set(WIRE_CONTROLS))
    if stray:
        _fatal(f"the catalogue says {facts.id} has {', '.join(stray)}, which "
               f"this service's wire does not carry. Add it to "
               f"voice_common.engines.CONTROL_RANGES with its bounds, or the "
               f"model reads a field no caller can ever set.")
    unexplained = sorted(set(WIRE_CONTROLS) - set(facts.controls)
                         - set(facts.control_absent_because))
    if unexplained:
        _fatal(f"{facts.id} has no {', '.join(unexplained)} control and the "
               f"catalogue does not say why. Add the model's own reason to "
               f"control_absent_because; a caller refused with 'unsupported' "
               f"cannot tell whether a newer build would help.")
    both = sorted(set(facts.controls) & set(facts.control_absent_because))
    if both:
        _fatal(f"the catalogue says {facts.id} both has and does not have "
               f"{', '.join(both)}.")
    if facts.voices is not None and not facts.voices:
        _fatal(f"{facts.id} has an empty preset voice list. None means 'the "
               f"voices are an open set named elsewhere'; an empty tuple means "
               f"a closed list with nothing in it, which no request could name.")


def _build(env) -> tuple[dict[str, EngineSpec], str, int]:
    ids = _engine_ids(env)
    # THE DEFAULT IS "EVERY ENGINE THAT HAS A LOCAL LANE TO BE PUT IN", not
    # "every engine". It was the second, which was true while every catalogue
    # row had a local_class and silently wrong the moment one did not: adding
    # voxtral to TTS_ENGINES would have made it local by omission, and the
    # first job would have died importing a class that is None. An operator who
    # adds it and changes nothing else now meets the runner-only invariant
    # below, which names the variable that turns it on.
    local_ids = _split(env.get("TTS_LOCAL_ENGINES")
                       or ",".join(e for e in ids
                                   if CATALOGUE[e].local_class is not None))
    unknown_local = [n for n in local_ids if n not in ids]
    if unknown_local:
        _fatal(f"TTS_LOCAL_ENGINES names {', '.join(unknown_local)}, which "
               f"TTS_ENGINES does not offer. Enabled here: {', '.join(ids)}.")
    # AN ENGINE WITH NO LOCAL CLASS CANNOT BE PUT ON THE LOCAL LANE BY ASKING.
    # `local_class` is None where there is no CPU path and no prospect of one,
    # so this is not "not installed here" -- it is a lane that would accept the
    # job, import None, and fail it after a job id already existed and a
    # progress bar had started.
    impossible = [n for n in local_ids if CATALOGUE[n].local_class is None]
    if impossible:
        _fatal(f"TTS_LOCAL_ENGINES names {', '.join(impossible)}, which "
               f"{'have' if len(impossible) > 1 else 'has'} no local "
               f"implementation at all -- not one that is missing from this "
               f"image, one that does not exist. Remove "
               f"{'them' if len(impossible) > 1 else 'it'} from "
               f"TTS_LOCAL_ENGINES and set TTS_ALLOW_RUNNER_ONLY_ENGINES=1 to "
               f"serve {'them' if len(impossible) > 1 else 'it'} from the "
               f"runner alone.")

    specs: dict[str, EngineSpec] = {}
    for engine_id in ids:
        facts = CATALOGUE[engine_id]
        _check_facts(facts)
        cold = env.get(f"TTS_COLD_LOAD_SECONDS_{slug(engine_id)}")
        specs[engine_id] = EngineSpec(
            facts=facts,
            runner_service=_runner_service_for(env, facts),
            cold_load_seconds=(float(cold) if cold is not None
                               else facts.cold_load_seconds),
            defaults=_defaults_from_env(env, facts),
            local=engine_id in local_ids)

    default = (env.get("TTS_DEFAULT_ENGINE") or ids[0]).strip().lower()
    if default not in specs:
        _fatal(f"TTS_DEFAULT_ENGINE is {default!r}, which TTS_ENGINES does not "
               f"offer. Enabled here: {', '.join(ids)}.")

    # SO-1. THE DEFAULT ENGINE MUST HAVE A LOCAL LANE, and this is the
    # invariant that makes "the server does not rely on the runner" provable
    # rather than believed. Every alias resolves to the default -- `tts-long`,
    # OpenAI's three names, and an absent `model` -- so a runner-only default
    # would mean an OpenAI client that has never heard of Voxtral gets a 503
    # because somebody sat down at a gaming PC. No caller may reach a
    # runner-only engine without typing its name.
    if not specs[default].local:
        _fatal(f"TTS_DEFAULT_ENGINE is {default!r} and TTS_LOCAL_ENGINES "
               f"cannot run it, so every request that names no model -- and "
               f"every OpenAI alias, which all resolve to the default -- would "
               f"fail whenever the runner is away. Local here: "
               f"{', '.join(e for e in ids if e in local_ids) or 'nothing'}.")

    # A GLOBAL CONTROL KEY THAT REACHES NO ENABLED ENGINE IS THE PER-ENGINE
    # LIE RELOCATED. TTS_EXAGGERATION on a box running only voxtral is set,
    # read, narrowed away by `_defaults_from_env` and honoured by nothing --
    # the same "I set it and heard no difference" with one more indirection in
    # front of it. Named at boot, which is the one moment anybody is looking.
    for field in WIRE_CONTROLS:
        key = f"TTS_{field.upper()}"
        if env.get(key) is None:
            continue
        if any(field in specs[e].controls for e in ids):
            continue
        why = "; ".join(f"{e}: {_why_no_control(CATALOGUE[e], field)}"
                        for e in ids)
        elsewhere = sorted(e for e, f in CATALOGUE.items()
                           if field in f.controls and e not in ids)
        _fatal(f"{key} is set and no engine enabled here has {field} ({why}) "
               f"Remove it"
               + (f", or add {elsewhere[0]} to TTS_ENGINES." if elsewhere
                  else "."))

    # AN ENGINE THAT ONLY EXISTS WHILE SOMEBODY'S GAMING PC IS ON IS AN ENGINE
    # THAT DISAPPEARS. The rule this whole stack is built on is that the server
    # must not rely on the runner: it has to work with that machine switched
    # off, unplugged, or lying about itself. An option that is runner-shaped is
    # a runner-shaped hole in the API, and the first Friday evening that hole
    # eats a job. Refused at boot rather than discovered at submit, because the
    # state is created by two variables and cannot be created by a request.
    orphans = [e for e in ids if e not in local_ids]
    if orphans and not _flag(env, "TTS_ALLOW_RUNNER_ONLY_ENGINES", "0"):
        _fatal(
            f"TTS_ENGINES offers {', '.join(orphans)} but TTS_LOCAL_ENGINES "
            f"cannot run {'them' if len(orphans) > 1 else 'it'}, so "
            f"{'they' if len(orphans) > 1 else 'it'} would exist only while "
            f"the runner is up. Add {'them' if len(orphans) > 1 else 'it'} to "
            f"TTS_LOCAL_ENGINES, remove {'them' if len(orphans) > 1 else 'it'} "
            f"from TTS_ENGINES, or set TTS_ALLOW_RUNNER_ONLY_ENGINES=1 and "
            f"accept a 503 at submit whenever the runner is away.")

    resident = int(env.get("TTS_LOCAL_RESIDENT_MAX") or 1)
    if resident < 1:
        _fatal("TTS_LOCAL_RESIDENT_MAX must be at least 1: zero resident "
               "checkpoints means the local lane cannot speak anything.")
    return specs, default, resident


def check_config(env=None) -> tuple[dict[str, EngineSpec], str, int]:
    """Build the engine table, or refuse to start. Named, fatal, never a warning."""
    return _build(env if env is not None else os.environ)


ENGINES, DEFAULT_ENGINE, LOCAL_RESIDENT_MAX = check_config()

# {engine id: idlegpu service id}, so remote.py can name a service without
# knowing what an engine is.
RUNNER_SERVICE_FOR = {e: s.runner_service for e, s in ENGINES.items()}
LOCAL_ENGINES = [e for e, s in ENGINES.items() if s.local]

# {preset voice name: the engine that ships it}, over the ENABLED engines only.
#
# VOICE NAMES ARE SCOPED BY ENGINE AND THE PAIR IS THE KEY -- which is already
# the reasoning behind `min_reference_seconds` being a property of the pair. So
# `pt_male` is not a name this service has; it is a name voxtral has. This map
# is what lets a request that sends it to Chatterbox be refused with the
# engine that does have it rather than with "unknown voice", and it is empty on
# a deployment that enables no preset engine, which is what makes every
# sentence below unreachable there rather than merely unused.
PRESET_VOICE_ENGINE: dict[str, str] = {
    preset.name: engine
    for engine, spec in ENGINES.items()
    for preset in (spec.facts.voices or ())
}


def warn_voice_collisions(clip_names, log=None) -> list[str]:
    """A clip whose name shadows a preset: named at boot, and nothing else.

    NOT FATAL, AND THE RESTRAINT IS THE DECISION. A service that refuses to
    start because somebody dropped a file called `pt_male.wav` in a shared
    volume is worse than the collision it is refusing: the collision costs
    nothing, because the PAIR is the key. `pt_male` on voxtral is the preset
    embedding and `pt_male` on chatterbox is the clip, and both are reachable
    -- the picker's option values are `engine:name`, so a person can tell them
    apart too.

    Said out loud anyway, once, because the person who dropped the file
    probably expected one of the two and should learn which.
    """
    shadowed = sorted(set(clip_names) & set(PRESET_VOICE_ENGINE))
    if shadowed and log is not None:
        for name in shadowed:
            log.warning(
                "the clip '%s' in TTS_VOICE_DIR has the same name as one of "
                "%s's preset voices. Both stay reachable -- a voice is a pair "
                "(engine, name) here -- so model='%s' gets the checkpoint's "
                "embedding and every other engine gets the file.",
                name, PRESET_VOICE_ENGINE[name], PRESET_VOICE_ENGINE[name])
    return shadowed


# --------------------------------------------------------- resolving names --


def spec_for(model: str | None) -> tuple[EngineSpec, str] | Refusal:
    """A wire `model` string to (engine, why that engine).

    `engine_reason` goes on the record and is the only thing that can later
    distinguish "the caller named this engine" from "the caller named nothing
    and the deployment's default happened to be this". Those two rows look
    identical without it, which is how a default change becomes invisible in
    six months of history.

    `model.strip().lower()`, matching gateway/app/main.py's own comparison, so
    the two cannot disagree about whether ' Chatterbox ' is a name.
    """
    if model is None or not model.strip():
        return ENGINES[DEFAULT_ENGINE], "default"
    key = model.strip().lower()
    spec = ENGINES.get(key)
    if spec is not None:
        return spec, "pinned"
    if key == SERVICE_ALIAS:
        return ENGINES[DEFAULT_ENGINE], f"alias:{SERVICE_ALIAS}"
    if key in OPENAI_MODEL_NAMES:
        return ENGINES[DEFAULT_ENGINE], "alias:openai"
    if key in CATALOGUE_IDS:
        # R2. KNOWN BUT NOT ENABLED, AND IT IS A DIFFERENT SENTENCE FROM R1.
        # "this service has never heard of that" sends somebody looking for a
        # typo; "that exists and this box has not turned it on" names the
        # variable that turns it on.
        return Refusal(
            400,
            f"model '{key}' is not enabled on this deployment. It is enabled "
            f"by adding it to TTS_ENGINES. Enabled here: "
            f"{', '.join(ENGINES)}.",
            "model_not_found", "model")
    return Refusal(
        400,
        f"model '{model}' is not one this service has: "
        f"{', '.join([*ENGINES, SERVICE_ALIAS])}. Names OpenAI documents "
        f"({', '.join(sorted(OPENAI_MODEL_NAMES))}) resolve to this service's "
        f"default engine, {DEFAULT_ENGINE}.",
        "model_not_found", "model")


def defaults_for(spec: EngineSpec) -> dict[str, float | str]:
    """What the caller did not say, for the engine they are about to get.

    READ AFTER THE ENGINE IS KNOWN, and that ordering is the whole point.
    compose.yaml sets TTS_EXAGGERATION=0.3, so a default resolved before the
    engine would put a value turbo cannot honour on EVERY turbo request --
    written by a config file rather than by a person, and then either refused
    for something nobody asked for or silently discarded by the model. An
    engine with no such control gets no default here, so the field arrives as
    None and never reaches generate() at all.
    """
    return dict(spec.defaults)


# ------------------------------------------------------------- refusals ----


def refuse(spec: EngineSpec, *, language: str | None,
           controls: dict[str, float | None],
           voice_seconds: float | None,
           voice_name: str | None = None,
           voice_is_object: bool = False,
           voice_is_alias: bool = False,
           voice_resolved: bool = True) -> Refusal | None:
    """The whole honest surface of an engine, decided before a job id exists.

    FIXED EVALUATION ORDER -- the voice, the language, the voice-and-language
    pair, then every control in WIRE_CONTROLS order, then the reference clip --
    so a client fixing one field at a time converges instead of being told
    about a different field on every attempt.

    THE VOICE MOVED TO THE FRONT AND IT IS THE ONE REORDERING HERE. It used to
    be resolved before this function ran, against one directory of clips, which
    was the whole truth while every engine cloned. A preset engine's voices are
    a property of the CHECKPOINT, so the name can only be judged once the
    engine is known -- and until this moved, a Voxtral request naming `pt_male`
    died with a sentence about TTS_VOICE_DIR.

    EQUALITY WITH A DEPLOYMENT DEFAULT IS NOT CONSENT. `exaggeration=0.3` on
    turbo is refused even though 0.3 is exactly what compose.yaml sets for the
    other engine, because the caller TYPED the field and believes it did
    something. A value that arrives as None is the caller saying nothing; any
    other value is a claim about the audio that this engine cannot make true.

    `controls` IS ONE DICT RATHER THAN ONE PARAMETER PER FIELD, and that is
    what makes a fourth engine no code at all: the loop below walks
    WIRE_CONTROLS over the mapping, so a control added to the catalogue is
    refused by name here the same instant, with no signature to widen and no
    call site to find.
    """
    refusal = _refuse_voice(spec, voice_name, voice_is_object, voice_is_alias,
                            voice_resolved)
    if refusal is not None:
        return refusal
    if language is not None and language.strip().lower() not in spec.languages:
        return _refuse_language(spec, language)
    refusal = _refuse_voice_language(spec, voice_name, language)
    if refusal is not None:
        return refusal
    for field in WIRE_CONTROLS:
        value = controls.get(field)
        if value is None or field in spec.controls:
            continue
        alternatives = sorted(e for e, s in ENGINES.items()
                              if field in s.controls)
        way_out = (f" Send model='{alternatives[0]}' if you need it."
                   if alternatives else "")
        return Refusal(
            400,
            f"{field} is not supported by {spec.id}: "
            f"{_why_no_control(spec.facts, field)}"
            + (f" Delivery on this engine is controlled with "
               f"{', '.join(sorted(spec.controls))} alone."
               if spec.controls else "")
            + way_out,
            "unsupported_value", field)
    if (voice_seconds is not None and spec.min_reference_seconds > 0
            and voice_seconds < spec.min_reference_seconds):
        # THE FIVE-SECOND MINIMUM IS A PROPERTY OF THE PAIR, not of the engine
        # and not of the voice, so this is the only place it can be checked.
        # Turbo asserts it itself, inside the worker, where it is an
        # AssertionError on somebody's job rather than a sentence they can act
        # on -- and by then a job id exists and a progress bar has started.
        alternatives = sorted(e for e, s in ENGINES.items()
                              if s.min_reference_seconds <= voice_seconds)
        way_out = (f" Send model='{alternatives[0]}', or clone the voice again "
                   f"from a longer clip." if alternatives
                   else " Clone the voice again from a longer clip.")
        return Refusal(
            400,
            f"voice '{voice_name}' cannot be used with {spec.id}: it asserts a "
            f"reference clip longer than {spec.min_reference_seconds:.1f} s "
            f"and this one is {voice_seconds:.1f} s." + way_out,
            "unsupported_value", "voice")
    return None


def _clone_engine() -> str | None:
    """An enabled engine that can clone from a clip, for a way out. None if
    this deployment has none, in which case the sentence simply stops."""
    return next((e for e, s in ENGINES.items() if s.facts.reference_audio), None)


def _refuse_voice(spec: EngineSpec, voice_name: str | None,
                  is_object: bool, is_alias: bool,
                  resolved: bool = True) -> Refusal | None:
    """R0. A voice this engine has no way to be.

    THE PAIR IS THE KEY, so this is four sentences and not one. "unknown voice"
    is true of none of them: the name is usually perfectly good somewhere in
    this service, and telling somebody their voice does not exist when it
    exists on the other engine sends them looking for a typo.
    """
    presets = spec.voice_names
    if presets:
        if voice_name in presets:
            return None
        listed = ", ".join(presets)
        clone = _clone_engine()
        if is_object:
            # R0b. OpenAI's custom-voice object is a CLONED voice by
            # definition. There is no source file in this checkpoint that takes
            # a reference-audio parameter, so a clip cannot be honoured even in
            # part -- not resampled, not approximated, not "closest preset".
            return Refusal(
                400,
                f"{spec.id} cannot clone: no source file in this checkpoint "
                f"takes a reference-audio parameter, and it carries no speaker "
                f"encoder, so a clip cannot be honoured even in part. Its "
                f"{len(presets)} voices are fixed: {listed}."
                + (f" Send model='{clone}'." if clone else ""),
                "unsupported_value", "voice")
        if is_alias:
            # R0c. AND IT IS NOT MAPPED ONTO A PRESET BY EAR. Answering
            # `alloy` with `neutral_female` would be a field accepted and
            # quietly turned into something else, which is the house rule
            # inverted rather than bent.
            return Refusal(
                400,
                f"voice '{voice_name}' is one of this service's OpenAI aliases "
                f"and resolves to a reference clip, which {spec.id} cannot "
                f"read. It is not mapped onto a preset by ear. {spec.id}'s "
                f"{len(presets)} voices are: {listed}."
                + (f" Send model='{clone}' for the aliases." if clone else ""),
                "unsupported_value", "voice")
        # R0. The plain case, with the voices of the caller's own language
        # first if the name told us one, because twenty names is a wall.
        return Refusal(
            400,
            f"voice '{voice_name}' is not one of {spec.id}'s {len(presets)}: "
            f"this checkpoint carries its speakers as embeddings baked into "
            f"the weights and has no speaker encoder of any kind, so a clip "
            f"cannot become a voice here. All {len(presets)}: {listed}."
            + (f" Send model='{clone}' to clone '{voice_name}' from its "
               f"reference clip." if clone else ""),
            "unsupported_value", "voice")
    # R0d. A CLIP ENGINE HANDED ANOTHER ENGINE'S PRESET NAME, AND ONLY WHERE
    # NO CLIP OF THAT NAME EXISTS. `pt_male` is a real voice on this service
    # and this is not the engine that has it, so "unknown voice: this service
    # has default, gabriel" is a lie by omission.
    #
    # `resolved` IS WHAT KEEPS THE COLLISION HARMLESS. A file called
    # pt_male.wav in TTS_VOICE_DIR is a perfectly good clip for a cloning
    # engine -- the pair is the key -- and refusing it because another
    # checkpoint ships an embedding by the same name would make one filename
    # able to take a voice away from an engine that can read it.
    owner = PRESET_VOICE_ENGINE.get(voice_name or "")
    if not resolved and owner is not None and owner != spec.id:
        return Refusal(
            400,
            f"voice '{voice_name}' cannot be used with {spec.id}: it is one of "
            f"{owner}'s preset speaker embeddings, not a reference clip in "
            f"TTS_VOICE_DIR. Send model='{owner}'.",
            "unsupported_value", "voice")
    return None


def _refuse_voice_language(spec: EngineSpec, voice_name: str | None,
                           language: str | None) -> Refusal | None:
    """R7. The language the caller named is not the one this voice speaks.

    ONLY FOR AN ENGINE THAT CARRIES THE LANGUAGE ON THE VOICE. Honouring both
    is impossible -- there is no language_id to pass, the embedding is the
    language -- so the choice is refuse or ignore one of them, and ignoring
    either is a caller who believes something false about the audio.
    """
    if not spec.facts.language_from_voice or language is None:
        return None
    spoken = spec.language_of(voice_name or "")
    wanted = language.strip().lower()
    if spoken is None or spoken == wanted:
        return None
    others = [v.name for v in (spec.facts.voices or ())
              if v.language == wanted]
    way_out = (f" Send voice={others[0]!r}"
               + (f" or {others[1]!r}" if len(others) > 1 else "")
               + ", or omit language and let the voice decide."
               if others else " Omit language and let the voice decide.")
    return Refusal(
        400,
        f"language '{wanted}' cannot be used with voice '{voice_name}': "
        f"{spec.id}'s voices are per-language and this one is {spoken}."
        + way_out,
        "unsupported_value", "language")


def _refuse_language(spec: EngineSpec, language: str) -> Refusal:
    if len(spec.languages) > 1:
        # R4: today's message, unchanged, because it was already right.
        return Refusal(
            400,
            f"language '{language}' is not one {spec.id} speaks: "
            f"{', '.join(spec.languages)}.",
            "unsupported_value", "language")
    # R5. ONE LANGUAGE IS NOT A SHORT LIST, and a message that reads like one
    # invites somebody to wait for the other twenty-two to be added. This
    # checkpoint has no language conditioning of any kind.
    multilingual = sorted(e for e, s in ENGINES.items() if len(s.languages) > 1)
    way_out = (f" Send model='{multilingual[0]}' for the "
               f"{len(ENGINES[multilingual[0]].languages)}-language model, or "
               f"language='{spec.languages[0]}'." if multilingual
               else f" This engine speaks {spec.languages[0]}.")
    return Refusal(
        400,
        f"language '{language}' is not supported by {spec.id}: this model has "
        f"no language conditioning of any kind -- its generate() takes no "
        f"language_id and raises TypeError if given one." + way_out,
        "unsupported_value", "language")


def refuse_unavailable(spec: EngineSpec, probe) -> Refusal | None:
    """Nothing here can run this engine right now, so say so BEFORE a job id.

    A 202 THAT CANNOT BE SERVED IS WORSE THAN A 503 THAT CAN BE READ. The
    dispatcher's `_pick` returns None and loops with no deadline, and `_sweep`
    skips anything with `finished_at is None`, so a job admitted for an engine
    no lane can run sits `queued` for ever: a progress bar that never moves,
    and thirty-two of them make `_full()` answer 429 to every BASELINE caller
    on a completely idle lane.

    Unreachable unless TTS_ALLOW_RUNNER_ONLY_ENGINES=1, because the startup
    invariant refuses to create the state otherwise. It is here anyway: the
    opt-out exists, and the operator who took it is owed a sentence rather than
    a silence.
    """
    if spec.local:
        return None
    ready = probe is not None and probe.ok_for(spec.id)
    if ready:
        return None
    why = (probe.why_for(spec.id) if probe is not None
           else "no runner is configured")
    return Refusal(
        503,
        f"{spec.id} is enabled but nothing here can run it right now: it is "
        f"not in TTS_LOCAL_ENGINES, and the runner reports '{why}' for service "
        f"'{spec.runner_service}'. Install it with 'idlegpu service install "
        f"{spec.runner_service}' on that machine and restart the agent, or "
        f"send model='{DEFAULT_ENGINE}'.",
        "engine_unavailable", "model")


# ------------------------------------------------- what the package really is --


# THE FOUR FACTS A CHECKPOINT WILL ADMIT TO WITHOUT SPEAKING A WORD. Read off
# the loaded object rather than out of the row beside it, because every label
# in this stack -- the x-tts-engine header, the job record's `engine`,
# `engine_reason` -- is written by the code that CHOSE the engine, so all three
# agree with each other whether or not any of them is true. These four are not
# written by that code. They are properties of the weights.
WITNESS_FIELDS: tuple[str, ...] = ("reference_audio", "language_id",
                                   "emotion_adv", "preset_voices")


def checkpoint_witness(target) -> dict[str, object | None]:
    """What the thing that loaded says about itself. None means it did not say.

    `target` IS THE CLASS OR THE LOADED MODEL, DELIBERATELY EITHER. `Synth` has
    the class before it spends 68 seconds and 6.5 GB and the instance
    afterwards, and every fact below reads the same on both, so a caller never
    has to hold the wrong one to ask.

    None IS NOT False, AND THE DIFFERENCE IS THE WHOLE VALUE OF THIS FUNCTION.
    A generate() this interpreter cannot introspect and a generate() that has
    no language_id are different states; reading the first as the second is how
    a check reports drift against a package it simply could not read, and how
    the reader downstream concludes it is looking at the other engine.
    """
    out: dict[str, object | None] = dict.fromkeys(WITNESS_FIELDS)
    try:
        params = inspect.signature(target.generate).parameters
    except (AttributeError, TypeError, ValueError):
        # A C-implemented or wrapped generate(), or no generate() at all.
        params = None
    if params is not None:
        # CAN THIS PACKAGE BE GIVEN A CLIP AT ALL. Three spellings because
        # three packages in this space spell the same parameter three ways.
        out["reference_audio"] = any(
            name in params for name in
            ("audio_prompt_path", "reference_audio", "speaker_wav"))
        # TURBO RAISES TypeError ON language_id AND THE MULTILINGUAL MODEL
        # TAKES IT. One parameter, and it is the sharpest difference between
        # the two checkpoints this deployment ships.
        out["language_id"] = "language_id" in params
    # hp.emotion_adv IS False IN TURBO'S CHECKPOINT: the emotion conditioning
    # layer is never built, so `exaggeration` is accepted by generate() and
    # discarded with a warning no caller can see.
    emotion = getattr(getattr(target, "hp", None), "emotion_adv", None)
    if emotion is not None:
        out["emotion_adv"] = bool(emotion)
    published = getattr(target, "AVAILABLE_VOICES", None)
    if published is not None:
        out["preset_voices"] = frozenset(published)
    return out


def catalogue_witness(spec: EngineSpec) -> dict[str, object | None]:
    """The same four facts, as the catalogue row states them.

    ONE READER FOR THE WEIGHTS AND ONE FOR THE ROW, AND NOTHING ELSE MAY
    COMPUTE EITHER. The checkpoint test used to re-implement this comparison
    with `assert` where this module implemented it with `log.warning` -- two
    halves of one feature, each of them tested, neither of them reading the
    other. That is the defect class this package is named after, and it had
    been planted inside the file that exists to prevent it.
    """
    return {
        "reference_audio": spec.facts.reference_audio,
        "language_id": (len(spec.languages) > 1
                        and not spec.facts.language_from_voice),
        "emotion_adv": "exaggeration" in spec.controls,
        "preset_voices": (frozenset(v.name for v in spec.facts.voices)
                          if spec.facts.voices is not None else None),
    }


def assert_named_checkpoint(
        spec: EngineSpec, cls,
        enabled: dict[str, EngineSpec] | None = None) -> None:
    """THE CHECKPOINT THAT LOADS IS THE ONE THE ROW NAMED, OR NOTHING SPEAKS.

    THE FAILURE THIS PREVENTS IS THE ONE THE WHOLE PACKAGE IS NAMED FOR. An
    audit found four single-token edits that answer a turbo request with
    baseline audio while the response header, the job record and
    `engine_reason` all still say turbo -- and every suite stayed green,
    because all three of those labels are written by the code that chose the
    engine. The last of the four lives one line above this call: swap
    `self.spec.local_class` for another row's and the wrong weights come off
    disk under the right name, which no test that fakes `Synth._speak` can
    see, because nothing was ever loaded.

    RAISED, NOT LOGGED, AND THE DISTINCTION IS NOT THE ONE assert_runtime
    MAKES. Drift is a fact going stale under a checkpoint that is still the
    right checkpoint, and killing a working service over a keyword argument
    would be the worse trade -- so that stays a warning. This is a DIFFERENT
    model answering to this engine's name, and it cannot be served at all:
    every label would keep saying what was asked for while the audio was
    somebody else's, which is precisely the state nobody can see from outside.

    TWO CONDITIONS, AND BOTH ARE NEEDED TO AVOID REFUSING A WORKING SERVICE.
    A package that re-exports its class from a shorter path spells
    `local_class` differently and is still the right weights, so a name
    mismatch alone is not enough; a point release that gains or loses a
    keyword is drift, so a capability mismatch alone is not enough either.
    Refused only where the object is BOTH under a name the row does not spell
    AND answering to another enabled engine's row -- a coincidence the two
    shipped checkpoints cannot produce between them.
    """
    named = spec.local_class
    if named is None:  # a runner-only engine never reaches a local class
        return
    where = getattr(cls, "__module__", "") or ""
    what = getattr(cls, "__qualname__", "") or ""
    # AN OBJECT THAT CANNOT SAY WHERE IT CAME FROM IS NOT THEREFORE INNOCENT.
    # Named as unnamed rather than as ":", which reads like a path and sends
    # the reader looking for a module called nothing.
    loaded = f"{where}:{what}" if where and what else ""
    if loaded == named:
        return
    witness = checkpoint_witness(cls)
    known = {field: value for field, value in witness.items()
             if value is not None}
    if len(known) < 2:
        # NOT ENOUGH OF THE OBJECT IS READABLE TO ACCUSE IT OF ANYTHING. One
        # fact is a coincidence; refusing on it would take a service down over
        # an interpreter that could not read a signature.
        return
    mine = catalogue_witness(spec)
    if all(known[field] == mine[field] for field in known):
        # It says what its own row says. The path is spelled differently and
        # that is a re-export, not another model.
        return
    for other in (ENGINES if enabled is None else enabled).values():
        if other.id == spec.id:
            continue
        theirs = catalogue_witness(other)
        if all(known[field] == theirs[field] for field in known):
            differs = ", ".join(
                f"{field}={known[field]}" for field in sorted(known)
                if known[field] != mine[field])
            _fatal_checkpoint(
                f"{spec.id} was asked for and {loaded or 'an unnamed object'} "
                f"was loaded, which answers to {other.id}'s row instead "
                f"({differs}). Nothing here will speak it: the response "
                f"header, the job record and engine_reason would all still "
                f"say {spec.id} while the audio was {other.id}'s, and no "
                f"caller could see the difference. "
                f"voice_common.engines.CATALOGUE names "
                f"{named} for {spec.id}.")


def _fatal_checkpoint(message: str) -> None:
    """Refuse the load, named, where the operator will meet it.

    SEPARATE FROM `_fatal` BECAUSE THIS ONE IS NOT AT BOOT. `_fatal` raises
    SystemExit so a misconfigured container does not start; this happens on a
    worker thread, minutes into somebody's queue, so it has to end ONE JOB with
    a sentence on the record rather than take the service down under whichever
    request happened to be first.
    """
    import logging
    logging.getLogger("tts-long.engines").error("%s", message)
    raise RuntimeError(message)


def assert_runtime(spec: EngineSpec, cls) -> None:
    """What the INSTALLED PACKAGE does, checked against what the catalogue claims.

    THE CATALOGUE IS A FACT ABOUT A CHECKPOINT AND FACTS GO STALE. A turbo
    point release that adds `language_id`, or a build where `hp.emotion_adv` is
    True, would leave this service refusing fields the model would now honour --
    and, far worse the other way, a checkpoint that LOSES a control would leave
    it forwarding fields that are silently discarded, which is the exact
    failure the whole design exists to prevent.

    Checked once, when the model loads, so a drift is one loud sentence in the
    log rather than a wrong answer per job for ever. NOT FATAL: the model is
    loaded, the audio is real, and killing a working service over a keyword
    argument would be a worse trade than saying so clearly.
    """
    import logging
    log = logging.getLogger("tts-long.engines")
    witness = checkpoint_witness(cls)
    if witness["reference_audio"] is None:
        # A C-implemented or wrapped generate(). Nothing below can be read, and
        # guessing at it would be drift reported against a package this
        # interpreter never saw.
        log.debug("%s: generate() has no readable signature", spec.id)
        return
    claimed = catalogue_witness(spec)
    # CAN THIS PACKAGE BE GIVEN A CLIP AT ALL, checked against the row that
    # says whether the wire may accept one. The catalogue's `reference_audio`
    # is what turns a `voice` into a 400 rather than into a clone, so a
    # checkpoint that gained or lost a speaker encoder under a service that
    # believes otherwise is either refusing clips it could read or accepting
    # ones it will silently ignore.
    if witness["reference_audio"] != claimed["reference_audio"]:
        log.warning(
            "%s: the catalogue says this engine %s clone from a reference "
            "clip and the installed generate() %s one. Requests are validated "
            "against voice_common.engines.CATALOGUE, which needs updating.",
            spec.id, "can" if claimed["reference_audio"] else "cannot",
            "takes" if witness["reference_audio"] else "does not take")
    # A PRESET LIST THAT HAS DRIFTED FROM THE WEIGHTS refuses nineteen good
    # names and accepts one that no longer exists -- and the one that no longer
    # exists fails inside the worker, after a job id, on a KeyError.
    if witness["preset_voices"] is not None and claimed["preset_voices"] is not None:
        drift = witness["preset_voices"] ^ claimed["preset_voices"]
        if drift:
            log.warning(
                "%s preset voice list has moved: %s. Requests are validated "
                "against voice_common.engines.CATALOGUE, which needs updating.",
                spec.id, ", ".join(sorted(drift)))
    if witness["language_id"] != claimed["language_id"]:
        log.warning(
            "%s: the catalogue says this engine %s language conditioning and "
            "the installed generate() %s a language_id parameter. Requests are "
            "validated against voice_common.engines.CATALOGUE, which needs "
            "updating.", spec.id,
            "has" if claimed["language_id"] else "has no",
            "takes" if witness["language_id"] else "does not take")
    if (witness["emotion_adv"] is not None
            and witness["emotion_adv"] != claimed["emotion_adv"]):
        log.warning(
            "%s: hp.emotion_adv is %s and the catalogue %s an exaggeration "
            "control. One of the two is wrong and the audio follows the model.",
            spec.id, witness["emotion_adv"],
            "declares" if claimed["emotion_adv"] else "declares no")


def generate_kwargs(spec: EngineSpec, *, language: str | None,
                    controls: dict[str, float | None]) -> dict[str, object]:
    """The keyword arguments this engine's generate() actually reads.

    ABSENT, NOT NONE. `generate(exaggeration=None)` on turbo is the same
    silently-discarded keyword as `generate(exaggeration=0.3)`; the only way
    not to send a field is not to build the key.

    `language_id` IS BUILT FROM `language_from_voice`, NOT FROM THE LANGUAGE
    COUNT, and that swap is a defect fix rather than a tidy. The old test was
    `len(spec.languages) > 1`, which is a proxy that happens to be right about
    both Chatterbox rows and wrong about the first engine that speaks nine
    languages WITHOUT taking a parameter for them: it would have handed
    `language_id=` to a generate_speech_fast() that has no such argument, which
    is a TypeError four frames into somebody's job, three and a half minutes
    after they submitted it. A model whose language is an embedding does not
    take the parameter at all.
    """
    kwargs: dict[str, object] = {}
    if (not spec.facts.language_from_voice and len(spec.languages) > 1
            and language is not None):
        kwargs["language_id"] = language
    for field in WIRE_CONTROLS:
        value = controls.get(field)
        if field in spec.controls and value is not None:
            kwargs[field] = value
    return kwargs


def engine_rows(names: Iterable[str] | None = None) -> dict[str, dict]:
    """The catalogue half of /health.engines. The lane half is main.py's."""
    return {
        e: {"label": ENGINES[e].label,
            "default": e == DEFAULT_ENGINE,
            "languages": list(ENGINES[e].languages),
            "controls": sorted(ENGINES[e].controls),
            "min_reference_seconds": ENGINES[e].min_reference_seconds,
            "cold_load_seconds": ENGINES[e].cold_load_seconds,
            # WHETHER THERE IS A SPEAKER ENCODER AT ALL, as a fact of its own.
            # `min_reference_seconds: 0.0` already means "this engine imposes
            # no minimum" for Chatterbox baseline, so it cannot also be read as
            # "there is no way to give this model a clip" -- and a page that
            # read it that way would draw a clone button on an engine that
            # cannot clone.
            "reference_audio": ENGINES[e].facts.reference_audio,
            # Whether the language is a parameter or a property of the voice.
            # It is what tells a picker to disable the Language control rather
            # than draw one that contradicts the voice underneath it.
            "language_from_voice": ENGINES[e].facts.language_from_voice,
            # THE ONLY HONEST HOME FOR A FIXED VOICE LIST. `null` for an engine
            # whose voices are an open set of clips -- those are in /voices,
            # which is the endpoint that owns the directory. Anything else puts
            # a fifth model table in the browser, and the page's own comment
            # says what the last four cost.
            "voices": ([{"name": v.name, "language": v.language}
                        for v in ENGINES[e].facts.voices]
                       if ENGINES[e].facts.voices is not None else None),
            # THE 24k/48k TRAP, PUBLISHED. The wrapper this engine runs through
            # resamples to 48000 in post-processing and then writes the result
            # at 24000 in three of its four writers. One assertion at load, one
            # on the wire, one in the record -- and this is the number all
            # three of them compare against.
            "native_sample_rate": ENGINES[e].facts.native_sample_rate}
        for e in (names if names is not None else ENGINES)
    }
