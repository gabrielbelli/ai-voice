"""What each speech checkpoint can actually do, as one table.

FACTS ABOUT CHECKPOINTS, NOT CHOICES ABOUT DEPLOYMENTS. Nothing in here says
which engines a machine offers, which lane may run them or what a slider
defaults to; those are configuration and they live in the service. What lives
here is the part that cannot be configured because it was decided by whoever
trained the weights: which languages the model conditions on, which generation
controls its `generate()` actually reads, and how long the checkpoint takes to
come off disk.

WHY IT IS IN voice-common AND NOT IN tts-long. Two services have to agree about
the set of model strings that exist: tts-long resolves one to an engine, and
the gateway decides whether a name is long-form work or Kokoro's. They agreed
by having two literals in two files, which is the arrangement that produced
`chatterbox-cpu` -- a service id configured on one side, never registered on
the other, and nothing anywhere able to say so. One imported table is the only
version of "these two must agree" that a test can hold.

THE HOUSE RULE THIS TABLE EXISTS TO SERVE. services/stt/app/openai_api.py
states it and this stack is built on it: EVERY FIELD IS EITHER HONOURED OR
REFUSED BY NAME, never accepted and dropped. Chatterbox Turbo accepts
`exaggeration`, `cfg_weight` and `language_id`-shaped keyword arguments and
throws them away with a logged warning -- measured, on all 21 segments of a
test run -- so a service that passed a caller's request straight through would
be handing back audio that silently ignored three fields the caller set.
`controls` and `languages` are what make that refusable before a job id exists.

NOTHING IN THIS MODULE MAY IMPORT FROM A SERVICE. It is the interface two of
them share, so a dependency in that direction is a cycle waiting for the first
person who tries to import it from the gateway.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CATALOGUE", "CATALOGUE_IDS", "CONTROL_RANGES", "EngineFacts",
           "PresetVoice", "WIRE_CONTROLS", "slug"]


# EVERY GENERATION FIELD THIS ESTATE'S WIRE CARRIES, in the order a refusal
# walks them, with the type and the bounds each one is legal within.
#
# ONE TABLE, READ BY THE PYDANTIC FIELDS AND BY THE CONFIG LOADER, so the wire
# and compose.yaml cannot disagree about what a legal value is. It was two
# literals: `Field(ge=0.0, le=1.0)` in main.py and a bare `float(raw)` in
# engines.py, which accepted TTS_EXAGGERATION=9 at boot and refused it on the
# wire -- a deployment default that every request is then refused for.
#
# `int` FOR flow_steps AND IT IS NOT COSMETIC. It is a loop count in the
# flow-matching solver; 32.0 through a config key would reach
# `range(flow_steps)` on the runner as a TypeError four frames into somebody's
# job, three and a half minutes after they submitted it.
CONTROL_RANGES: dict[str, tuple[type, float, float]] = {
    # Chatterbox's three, unchanged. The bounds are the ones main.py's
    # JobRequest has always declared.
    "exaggeration": (float, 0.0, 1.0),
    "cfg_weight": (float, 0.0, 1.0),
    "temperature": (float, 0.1, 1.5),
    # Voxtral's two. THE MAIN QUALITY KNOB FIRST: 32 was judged clearly best by
    # ear on the owner's 3070, 16 close behind, 8 and 4 audibly worse. The
    # upstream repo defaults to 8. 64 is the ceiling because nothing above it
    # has ever been listened to and the cost is linear in the step count.
    "flow_steps": (int, 1, 64),
    # 1.2 is full classifier-free guidance on that solver. 1.0 is documented
    # upstream as faster and garbled, which is why the floor is 1.0 rather
    # than 0: a value below it is not a weaker effect, it is a broken one.
    "cfg_alpha": (float, 1.0, 3.0),
}

# The same names as a tuple, because membership and ORDER are what the refusal
# loop and the config loader walk. Derived rather than repeated: a control
# added to the table above is on the wire the same instant, with its bounds.
WIRE_CONTROLS: tuple[str, ...] = tuple(CONTROL_RANGES)


@dataclass(frozen=True)
class PresetVoice:
    """One fixed speaker a checkpoint ships, and the language it speaks.

    THE LANGUAGE IS ON THE VOICE, NOT BESIDE IT, and that is the whole reason
    this is a pair rather than a bare string. `de_female` is German because of
    which tensor it is, not because of what its name starts with -- and the
    moment a browser parses `de_` to find out, it is back on Kokoro's
    first-letter convention, where `pt_male` resolves correctly by coincidence
    and `de_female` resolves to en-us. The fact travels with the voice, from
    the checkpoint, through /health, to the picker.

    `casual_female` and `cheerful_female` are the proof it cannot be derived:
    nothing in either name says English.
    """

    name: str
    language: str


@dataclass(frozen=True)
class EngineFacts:
    """One checkpoint, and everything about it that a deployment cannot change.

    `controls` is the set of generation fields that REACH `generate()` and
    change the audio. A field outside it is not "unsupported" in the vague
    sense -- it is a field the model's own code discards, so honouring it is
    impossible and accepting it is a lie. That is why this is a set of names
    and not a boolean per field: a third engine is a row here and no code
    anywhere else.

    `local_seed` and `runner_seed` are realtime factors, seeds for the moving
    average each lane keeps. They are measurements and each one names the
    machine it came from in the comments below. `chars_per_second` is
    deliberately NOT a field: it measures how long the speech IS, not how long
    it takes to make, and turbo renders faster without talking faster. Putting
    a speed-up in both places double-counts it in every estimate.
    """

    id: str
    # What a picker prints. Not derived from the id: "Chatterbox Turbo" is a
    # product name and title-casing an id is how a UI ends up saying
    # "Chatterbox-Turbo".
    label: str
    # The gateway's routing key: which service in this stack owns the name.
    owned_by: str
    # "module:ClassName", imported lazily by the local lane. A string rather
    # than the class, because importing it drags torch into every process that
    # merely wants to know a language list.
    #
    # None MEANS THERE IS NO LOCAL LANE AND THERE IS NO PROSPECT OF ONE. Not
    # "not installed yet": Voxtral's int4 path calls
    # torch.cuda.get_device_capability() inside torchao's from_hp before any
    # device dispatch, its BF16 escape hatch calls torch.cuda.synchronize()
    # unconditionally on the decode path, and its flow_steps/cfg_alpha exist
    # only on a fast path the standard path does not share. A CPU class here
    # would be a lie a boot-time check could not catch.
    local_class: str | None
    # The idlegpu service id this engine's runner lane asks for by default.
    runner_service: str
    languages: tuple[str, ...]
    # The generation fields that reach generate(). See the class docstring.
    controls: frozenset[str]
    # WHY A CONTROL IS ABSENT, IN THE MODEL'S OWN TERMS, per field name.
    # "unsupported" is what an error message says when nobody looked, and it
    # tells a caller nothing about whether waiting for a newer build could help.
    # A field named here and also in `controls` is a construction error.
    control_absent_because: dict[str, str]
    # THE TWENTY FIXED SPEAKERS, or None for a checkpoint whose voices are an
    # open set named somewhere else (a directory of reference clips). An EMPTY
    # tuple is neither and is refused at boot: it would mean an engine that has
    # a closed voice list with nothing in it, which no request could ever name.
    voices: tuple[PresetVoice, ...] | None
    # Whether this checkpoint has a speaker encoder at all. SEPARATE FROM
    # min_reference_seconds, and it has to be: 0.0 already means "this engine
    # imposes no minimum" for Chatterbox baseline, so it cannot also be made to
    # mean "there is no way to give this model a clip".
    reference_audio: bool
    # Whether the language is a property of the chosen voice rather than a
    # parameter the caller passes. True here replaces the `len(languages) > 1`
    # proxy, which would hand `language_id` to a generate_speech_fast() that
    # has no such parameter -- a TypeError four frames into somebody's job.
    language_from_voice: bool
    # The shortest reference clip this checkpoint will clone from. A property
    # of the PAIR (engine, voice) and so checkable only where both are known.
    min_reference_seconds: float
    # WHAT RATE THIS CHECKPOINT'S CODEC ACTUALLY PRODUCES. Asserted at load and
    # again on the wire, because getting it wrong is the failure that ships
    # audio which plays at half speed with nothing anywhere reporting an error.
    # The voxtral-int4 repo does exactly that: postprocess_audio resamples
    # 24000 -> 48000 and three of its four writers then write 24000.
    native_sample_rate: int
    # Measured, cold, off a warm page cache.
    cold_load_seconds: float
    # PEAK RESIDENT VRAM DURING THE LOAD, not while running, or None where
    # nobody has measured it. Voxtral peaks at 8265 MiB on an 8192 MiB card --
    # it survived once, with 73 MiB to spare, because the desktop was idle --
    # so it is an operational fact a runner has to preflight rather than a
    # footnote.
    load_peak_vram_mib: int | None
    # Realtime factors: seconds of speech per second of compute.
    #
    # local_seed is None WHERE NO CPU FIGURE HAS EVER BEEN MEASURED ON ANY
    # MACHINE IN THIS STACK, which is the honest value and not a pessimistic
    # guess. It is what makes the synchronous branch structurally unreachable
    # for such an engine rather than arithmetically lucky: an invented number
    # would be the `runner_cpu` trap a second time -- a rung nothing could ever
    # measure, permanent because it was never run.
    local_seed: float | None
    runner_seed: float


# THE TWENTY, WITH THE LANGUAGE EACH ONE ACTUALLY SPEAKS. Shipped as .pt
# tensors inside the checkpoint, so this list is a fact about the weights and
# not a configuration: a deployment cannot add a twenty-first.
#
# `casual_*`, `cheerful_*` and `neutral_*` ARE ENGLISH AND NOTHING IN THEIR
# NAMES SAYS SO. That is the entire argument for PresetVoice being a pair. Five
# of the twenty carry no language prefix at all, and of the fifteen that do,
# `pt_male` happens to resolve correctly under Kokoro's first-letter convention
# while `de_female` resolves to en-us -- one coincidence and one silent wrong
# answer, from the same rule.
VOXTRAL_VOICES: tuple[PresetVoice, ...] = (
    PresetVoice("ar_male", "ar"),
    PresetVoice("casual_female", "en"),
    PresetVoice("casual_male", "en"),
    PresetVoice("cheerful_female", "en"),
    PresetVoice("de_female", "de"),
    PresetVoice("de_male", "de"),
    PresetVoice("es_female", "es"),
    PresetVoice("es_male", "es"),
    PresetVoice("fr_female", "fr"),
    PresetVoice("fr_male", "fr"),
    PresetVoice("hi_female", "hi"),
    PresetVoice("hi_male", "hi"),
    PresetVoice("it_female", "it"),
    PresetVoice("it_male", "it"),
    PresetVoice("neutral_female", "en"),
    PresetVoice("neutral_male", "en"),
    PresetVoice("nl_female", "nl"),
    PresetVoice("nl_male", "nl"),
    PresetVoice("pt_female", "pt"),
    PresetVoice("pt_male", "pt"),
)

# WHY NEITHER CHATTERBOX HAS VOXTRAL'S TWO DIALS. Shared because it is the same
# sentence about both checkpoints and a copy would be a copy that drifts.
_FLOW_MATCHING_ABSENT = {
    "flow_steps":
        "it is the number of steps the flow-matching solver takes in "
        "Voxtral's acoustic decoder, and this checkpoint has no "
        "flow-matching decoder at all.",
    "cfg_alpha":
        "it is the guidance scale of Voxtral's flow-matching solver. This "
        "checkpoint's guidance dial, where it has one, is cfg_weight, on a "
        "different scale.",
}


CATALOGUE: dict[str, EngineFacts] = {
    "chatterbox": EngineFacts(
        id="chatterbox",
        label="Chatterbox",
        owned_by="tts-long",
        local_class="chatterbox.mtl_tts:ChatterboxMultilingualTTS",
        runner_service="chatterbox",
        # chatterbox/mtl_tts.py:24, SUPPORTED_LANGUAGES. app/synth.py holds the
        # same list for the same reason -- answering a request before torch is
        # imported -- and cross-checks itself against the model at load.
        languages=("ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
                   "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
                   "sw", "tr", "zh"),
        controls=frozenset({"exaggeration", "cfg_weight", "temperature"}),
        control_absent_because=_FLOW_MATCHING_ABSENT,
        # AN OPEN SET, NAMED SOMEWHERE ELSE. This engine clones, so its voices
        # are whatever is in TTS_VOICE_DIR and the service that owns that
        # directory is the only thing that can list them.
        voices=None,
        reference_audio=True,
        language_from_voice=False,
        min_reference_seconds=0.0,
        native_sample_rate=24000,
        # Measured: 22.2 s on the deployed instance.
        cold_load_seconds=22.0,
        load_peak_vram_mib=None,
        # local: the NAS, Xeon E5-2697 v4, eight threads, measured 0.230x.
        # runner: a desktop RTX 3070, midpoint of a measured 0.644-0.746.
        local_seed=0.21,
        runner_seed=0.70),
    "chatterbox-turbo": EngineFacts(
        id="chatterbox-turbo",
        label="Chatterbox Turbo",
        owned_by="tts-long",
        local_class="chatterbox.tts_turbo:ChatterboxTurboTTS",
        runner_service="chatterbox-turbo",
        # ONE LANGUAGE, AND IT IS STRUCTURAL RATHER THAN A GAP. This
        # checkpoint's generate() has no language_id parameter at all: passing
        # one raises TypeError immediately, so the 23-language surface is not
        # degraded here, it is absent.
        languages=("en",),
        # temperature, top_k, top_p and repetition_penalty survive; only
        # temperature is on this service's wire. exaggeration and cfg_weight
        # are STRUCTURALLY ABSENT: hp.emotion_adv is False, so the conditioning
        # layer is never built, and inference_turbo has no CFG path. Both are
        # accepted as keyword arguments and discarded with a logged warning.
        controls=frozenset({"temperature"}),
        control_absent_because={
            "exaggeration":
                "the emotion conditioning layer is not built in this "
                "checkpoint (hp.emotion_adv is False), so the value would be "
                "accepted by generate(), logged as a warning and discarded.",
            "cfg_weight":
                "this checkpoint has no classifier-free-guidance path at all, "
                "so the value would be accepted and discarded.",
            **_FLOW_MATCHING_ABSENT,
        },
        voices=None,
        reference_audio=True,
        language_from_voice=False,
        # Turbo asserts this itself, so a shorter clip is an AssertionError
        # inside somebody's job rather than a 400 they can read.
        min_reference_seconds=5.0,
        native_sample_rate=24000,
        # Measured 67.5 s: a 1.83 GB checkpoint plus an AutoTokenizer. Three
        # times the current model's, which is why a service that yields the
        # card and reloads pays a real price rather than a footnote.
        cold_load_seconds=68.0,
        # runner: 1.5426x measured on the 3070, fp32. local: NOT MEASURED --
        # turbo has never been run on either CPU in this stack. Seeded at the
        # bottom of a 0.5-0.7 estimate on purpose, because a pessimistic rate
        # defers to a 202 rather than promising what the box cannot do, and
        # `engine_observations` publishes the count of zero beside it.
        load_peak_vram_mib=None,
        local_seed=0.45,
        runner_seed=1.54),
    "voxtral": EngineFacts(
        id="voxtral",
        label="Voxtral",
        owned_by="tts-long",
        # NO -int4 SUFFIX ON THE NAME A CALLER PINS. The quantisation is a
        # deployment fact -- it is what the runner's manifest publishes -- and
        # putting it in the model string would make a client's pinned name go
        # stale the day somebody serves the same checkpoint another way.
        #
        # NOT A CLONING ENGINE, AND THAT IS THE FACT THE WHOLE ROW TURNS ON.
        # Twenty fixed speaker embeddings ship as .pt tensors inside the
        # checkpoint; there is no speaker encoder anywhere in it and no
        # reference-audio parameter in any of the wrapper's nine source files.
        # It sits beside Kokoro conceptually, not beside Chatterbox.
        #
        # local_class IS None AND IT IS NOT A GAP TO BE FILLED. See the field's
        # comment: three independent walls in torchao and a CUDA-only decode
        # path, plus a different ODE solver on the only path that has
        # flow_steps at all. Estimated 0.002-0.004x if any of that were
        # removed, which is a 20-second clip in one and a half to three hours.
        local_class=None,
        runner_service="voxtral",
        languages=("ar", "de", "en", "es", "fr", "hi", "it", "nl", "pt"),
        controls=frozenset({"flow_steps", "cfg_alpha"}),
        control_absent_because={
            "exaggeration":
                "this is a Mistral flow-matching checkpoint with no emotion "
                "conditioning of any kind -- generate_speech_fast() has no "
                "such parameter and would raise TypeError.",
            "cfg_weight":
                "its guidance dial is cfg_alpha, on a different scale and a "
                "different solver -- 1.2 is full CFG there and 1.0 is "
                "documented upstream as faster but garbled.",
            "temperature":
                "this checkpoint's sampler does not read it.",
        },
        voices=VOXTRAL_VOICES,
        reference_audio=False,
        language_from_voice=True,
        # NOT 0.0-MEANING-ANYTHING-GOES. There is no clip at all, which
        # `reference_audio=False` is what says; the floor beside it is unread.
        min_reference_seconds=0.0,
        # 24 kHz RAW, AND IT IS A WIRE CONTRACT. The codec produces 24 kHz
        # natively. The 48 kHz that appears upstream exists only because
        # audio_postprocess.postprocess_audio resamples on its way past -- and
        # generate.py, generate_fast.py and benchmark_all.py then write the
        # result back at 24000, which is how a file ends up playing at half
        # speed with nothing reporting an error. Only serve.py gets it right.
        cold_load_seconds=63.0,
        # 8265 MiB PEAK ON AN 8192 MiB CARD. load_model_int4 builds on CPU,
        # load_file()s 8.0 GB of BF16, moves the whole thing to the device and
        # only then quantises to int4 -- 28.8 s of the 63 is the quantisation.
        # There is no pre-quantised checkpoint shipped and --quantized is not
        # one: it routes to an unrelated TurboQuant experiment whose package is
        # not in the repository, not on PyPI and not installed.
        load_peak_vram_mib=8265,
        native_sample_rate=24000,
        # NO LOCAL FIGURE, BECAUSE THERE IS NO LOCAL LANE. See the field.
        local_seed=None,
        # Measured on the owner's 3070 at flow_steps=32, cfg_alpha=1.2 -- the
        # settings this deployment ships. At the repo's default of 8 steps it
        # is 0.354x, so this number is a statement about the quality settings
        # and not about the card. A 20-second clip is about three and a half
        # minutes of GPU. THAT IS WHY IT IS ALWAYS A JOB.
        runner_seed=0.104),
}

CATALOGUE_IDS = frozenset(CATALOGUE)


def slug(engine_id: str) -> str:
    """An engine id as an environment-variable fragment.

    NO KEY ANYWHERE SPELLS AN ENGINE NAME BY HAND. `TTS_REALTIME_FACTOR_RUNNER_`
    + slug("chatterbox-turbo") is TTS_REALTIME_FACTOR_RUNNER_CHATTERBOX_TURBO,
    and the day a third engine lands there is no file to go and edit -- which
    is the difference between a registry row and a branch.
    """
    return engine_id.upper().replace("-", "_")
