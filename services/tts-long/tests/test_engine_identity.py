"""THE ENGINE THAT RAN IS THE ENGINE THAT WAS ASKED FOR.

THE HOLE THIS FILE CLOSES. An audit of the turbo round found FOUR single-token
edits that produce one engine's audio from another engine's request while every
label -- the `x-tts-engine` response header, the job record's `engine`, the
`engine_reason` beside it -- still says the engine that was asked for. All four
kept every suite green, and a third engine widens the hole rather than
narrowing it.

WHY NOTHING CAUGHT THEM. Every one of those labels is written by the same code
path that chose the engine, so they agree with each other whether or not any of
them is true. The question underneath all four is one question -- DID THE
CHECKPOINT THAT RAN MATCH THE ONE THAT WAS NAMED? -- and it has two halves,
because the journey from a model string to a set of weights has two legs.

  THE REQUEST TO THE SPEC. Which `Synth`, carrying which catalogue row, was
  handed the job. The only witness here that is not written by the choosing
  code is THE AUDIO, and the fake `Synth._speak` used to be a function of the
  text alone -- swap the spec and the bytes are identical. Its tone is a
  function of (text, engine) now, so
  `test_the_engine_that_ran_is_the_engine_that_was_asked_for` and the two
  beside it read the engine back out of the samples.

  THE SPEC TO THE WEIGHTS. Which class `Synth._ensure_loaded` actually imported
  out of `spec.local_class`. NO AUDIO TEST IN THIS TREE CAN SEE THIS LEG, and
  the reason is structural rather than an oversight: every test here fakes
  `_speak`, which is the one method that touches a checkpoint, so nothing is
  ever loaded and the fake's tone comes from the spec it was handed rather than
  from any weights. Swapping one row's `local_class` for another's is therefore
  the one edit of the four that leaves the audio right too.

  `test_the_checkpoint_that_loaded_is_the_one_the_row_named` closes that leg,
  and it RUNS BY DEFAULT. It needs no weights and no card, because the two
  facts that separate the shipped checkpoints are properties of the object:
  Chatterbox's generate() takes `language_id` and its `hp.emotion_adv` is True;
  Turbo's raises TypeError on `language_id` and its `hp.emotion_adv` is False.
  A stand-in class asserting exactly those facts, installed where the catalogue
  says the weights live, drives the real `_ensure_loaded` end to end.

  `test_the_installed_weights_are_the_ones_the_catalogue_names` is the same
  assertion against the real package, marked `checkpoint` and deselected
  because importing it drags in torch. Both halves call
  `engines.checkpoint_witness` and `engines.catalogue_witness` -- THE SERVICE'S
  OWN READERS, not a second copy of them. The version of this file that shipped
  re-implemented the comparison here with `assert` where the service
  implemented it with `log.warning`, which is two halves of one feature that
  can disagree, planted in the file that exists to prevent exactly that.

THE VOXTRAL CASE IS GONE RATHER THAN FIXED. The engine was measured and
rejected this week -- 0.104x, a segfault on an 8 GB card, transcript match down
to 0.682 -- and its install has been deleted from spring. See ADR 0009 and
GAB-634. Its parametrisation here failed for a third reason: `voxtral` is not
in the default TTS_ENGINES, so all three cases failed on the first line of the
body and the deselected half had never run anywhere at all.

Every test is named after the mistake it prevents.
"""

from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest
from starlette.testclient import TestClient

from conftest import engine_of

TURBO = "chatterbox-turbo"
VOXTRAL = "voxtral"
LOCAL_BOTH = "chatterbox,chatterbox-turbo"


def _wait(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never finished; it is still {job['status']}")


def _samples(client, job_id) -> np.ndarray:
    """The audio this job actually produced, off disk, as float samples.

    Collected through the route rather than out of the job dict, because the
    route is what a caller gets and the file on disk is what survives a
    restart. wav here: it is the one format in this image that is written
    without an encoder, so nothing between the model and this array can have
    changed a frequency.
    """
    raw = client.get(f"/jobs/{job_id}/audio").content
    import io
    import wave

    with wave.open(io.BytesIO(raw)) as f:
        assert f.getframerate() == 24000, "the file's own header disagrees"
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return pcm.astype(np.float64) / 32767.0


@pytest.mark.parametrize("engine", ["chatterbox", TURBO])
def test_the_engine_that_ran_is_the_engine_that_was_asked_for(
        engine, build, voice_dir):
    """FOUR SINGLE-TOKEN EDITS PRODUCED THE WRONG AUDIO WITH EVERY LABEL RIGHT.

    The audio is the only witness that is not written by the code that chose
    the engine. This asserts the witness and the labels agree -- and it fails
    if the request is answered by the other checkpoint no matter how many
    labels say otherwise.
    """
    voice_dir("gabriel", 61.0)
    with TestClient(build(TTS_ENGINES=LOCAL_BOTH)) as client:
        posted = client.post("/jobs", json={"model": engine, "voice": "gabriel",
                                            "text": "One short line, spoken once."})
        assert posted.status_code == 202, posted.text
        assert posted.json()["engine"] == engine

        row = _wait(client, posted.json()["id"])
        assert row["status"] == "done", row.get("error")
        assert row["engine"] == engine, "the record names the wrong engine"

        produced = engine_of(_samples(client, row["id"]))
        assert produced == engine, (
            f"the row, the 202 and the header all say {engine}, and the audio "
            f"was made by {produced}")


def test_the_header_the_record_and_the_audio_cannot_disagree(build, voice_dir):
    """THREE LABELS AND ONE WITNESS, ON ONE REQUEST.

    /v1/audio/speech is the route that hands back a file, so it is the one
    where a client holding audio has nothing but `x-tts-engine` to go on. The
    labels agreeing with each other proves nothing; they are written by the
    same function.
    """
    voice_dir("gabriel", 61.0)
    # A SYNCHRONOUS ANSWER IS THE POINT OF THIS TEST: it is the only route
    # that puts bytes in a caller's hand with nothing but a header beside them.
    # Turbo's cold load is 68 s off disk and the model here is faked, so the
    # budget has to stop being charged for a load that is not happening.
    with TestClient(build(TTS_ENGINES=LOCAL_BOTH,
                          TTS_OPENAI_SYNC_TIMEOUT="60",
                          TTS_COLD_LOAD_SECONDS_CHATTERBOX_TURBO="0")) as client:
        answered = client.post("/v1/audio/speech",
                               json={"model": TURBO, "voice": "gabriel",
                                     "input": "One short line.",
                                     "response_format": "wav"})
        assert answered.status_code == 200, answered.text
        assert answered.headers["x-tts-engine"] == TURBO

        import io
        import wave
        with wave.open(io.BytesIO(answered.content)) as f:
            pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
        produced = engine_of(pcm.astype(np.float64) / 32767.0)
        assert produced == TURBO, (
            f"x-tts-engine says {TURBO} and the bytes in the caller's hand "
            f"were made by {produced}")


def test_swapping_the_synth_for_the_other_engines_is_caught(build, voice_dir):
    """THE TEST ABOVE IS ONLY WORTH HAVING IF IT CAN FAIL.

    This IS one of the four single-token edits, performed deliberately:
    `_Synths.get` handed the pool the wrong spec. Everything downstream --
    `engine_reason`, the record, the header -- is written from the REQUEST and
    goes on saying turbo, so this is precisely the state that kept every suite
    green. The assertion has to notice.
    """
    voice_dir("gabriel", 61.0)
    with TestClient(build(TTS_ENGINES=LOCAL_BOTH)) as client:
        import app.main as main

        original = main._Synths.get
        # The edit: whatever engine was asked for, load the default one.
        main._Synths.get = lambda self, spec: original(
            self, main.ENGINES[main.DEFAULT_ENGINE])
        try:
            posted = client.post("/jobs", json={"model": TURBO,
                                                "voice": "gabriel",
                                                "text": "One short line."})
            row = _wait(client, posted.json()["id"])
        finally:
            main._Synths.get = original

        assert row["engine"] == TURBO, \
            "the mutation was supposed to leave every LABEL saying turbo"
        assert engine_of(_samples(client, row["id"])) == "chatterbox", \
            "the mutation did not actually swap the checkpoint"


def test_a_runner_only_engine_never_reaches_a_local_synth_at_all(build, voice_dir):
    """THE THIRD ENGINE'S VERSION OF THE SAME QUESTION.

    It has no CPU implementation, so "which checkpoint answered" has a third
    answer here -- none, and the request is refused before a job id. The
    failure this prevents is the local lane accepting it and loading whichever
    checkpoint it happened to have resident, which would be audio in the wrong
    voice with a row that says voxtral.
    """
    voice_dir("gabriel", 61.0)
    with TestClient(build(TTS_ENGINES="chatterbox,chatterbox-turbo,voxtral",
                          TTS_LOCAL_ENGINES=LOCAL_BOTH,
                          TTS_ALLOW_RUNNER_ONLY_ENGINES="1")) as client:
        import app.main as main

        loaded: list[str] = []
        original = main._Synths.get
        main._Synths.get = lambda self, spec: (loaded.append(spec.id)
                                               or original(self, spec))
        try:
            refused = client.post("/jobs", json={"model": VOXTRAL,
                                                 "voice": "pt_male",
                                                 "text": "olá"})
        finally:
            main._Synths.get = original

        assert refused.status_code == 503, refused.text
        assert VOXTRAL not in loaded, \
            "the local pool was asked for a checkpoint that has no local class"



# --------------------------------------------- the spec to the weights ---


class _MultilingualLike:
    """`chatterbox.mtl_tts:ChatterboxMultilingualTTS` as the loader meets it.

    A STAND-IN FOR THE WEIGHTS AND NOT FOR THE CATALOGUE, which is the whole
    difference between this and the fake `Synth._speak`. That fake reads
    `self.spec.id` -- it IS the catalogue's claim, so it cannot witness against
    it. These two classes state, independently of any row, the facts the real
    checkpoints state: this one takes `language_id` and was built with the
    emotion conditioning layer.

    `__module__` and `__qualname__` ARE SET ON PURPOSE. They are how the loader
    says which checkpoint it has, so a stand-in that did not carry the real
    one's name would be a different checkpoint, and the guard would be right to
    refuse it.
    """

    __module__ = "chatterbox.mtl_tts"
    __qualname__ = "ChatterboxMultilingualTTS"
    # Asserted against the catalogue's native_sample_rate at the end of the
    # load; see synth._ensure_loaded and voice_common.audio.check_rate.
    sr = 24000

    class hp:
        emotion_adv = True

    @staticmethod
    def generate(text, audio_prompt_path=None, language_id=None,
                 exaggeration=0.5, cfg_weight=0.5, temperature=0.8):
        """The real signature, and NO AUDIO. Identity is the only question."""
        return None

    @classmethod
    def from_pretrained(cls, device="cpu"):
        return cls()


class _TurboLike:
    """`chatterbox.tts_turbo:ChatterboxTurboTTS` as the loader meets it.

    NO `language_id` AND NO `**kwargs`, because that is the actual shape: the
    catalogue row records that `exaggeration` and `cfg_weight` are accepted and
    discarded with a logged warning while `language_id` raises TypeError
    immediately. A stand-in with `**kwargs` would swallow the one call that
    tells the two checkpoints apart.
    """

    __module__ = "chatterbox.tts_turbo"
    __qualname__ = "ChatterboxTurboTTS"
    sr = 24000

    class hp:
        emotion_adv = False

    @staticmethod
    def generate(text, audio_prompt_path=None, exaggeration=0.5,
                 cfg_weight=0.5, temperature=0.8):
        """The real signature, and NO AUDIO. Identity is the only question."""
        return None

    @classmethod
    def from_pretrained(cls, device="cpu"):
        return cls()


CHECKPOINTS = {"chatterbox": _MultilingualLike, TURBO: _TurboLike}


@pytest.fixture
def weights(monkeypatch):
    """The two shipped checkpoints, stood in where the catalogue says they are.

    WHAT IS STUBBED AND WHY IT IS ONLY THIS. `torch` and `perth` are imported
    by `_ensure_loaded` before it resolves anything, and neither has the
    faintest bearing on which checkpoint answered -- one sets a thread count
    and the other is a watermarker this service already neutralises. Everything
    between the catalogue row and the loaded object is the real code: the
    partition of `local_class`, the import, the attribute lookup, the identity
    guard, the float32 pin, `from_pretrained` and the sample-rate assertion.

    Returns the installer, so a test can re-point one row at the other's class
    -- which is the fourth single-token edit, performed deliberately.
    """
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda count: None
    perth = types.ModuleType("perth")
    perth.PerthImplicitWatermarker = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "perth", perth)
    monkeypatch.setitem(sys.modules, "chatterbox",
                        types.ModuleType("chatterbox"))

    def install(engine: str, double=None):
        """Put `double` where CATALOGUE says `engine`'s weights are."""
        from voice_common.engines import CATALOGUE

        local = CATALOGUE[engine].local_class
        module_name, _, class_name = local.partition(":")
        module = sys.modules.get(module_name) or types.ModuleType(module_name)
        setattr(module, class_name, double or CHECKPOINTS[engine])
        # THE NAME synth._pin_every_loaded_module WALKS EVERY LOADED chatterbox
        # MODULE LOOKING FOR. Without it that guard reports, correctly, that it
        # could not attach -- so a stand-in without it would make the load path
        # log a warning about a wheel that is not there.
        module.norm_loudness = lambda wav, *args, **kwargs: wav
        monkeypatch.setitem(sys.modules, module_name, module)
        return module

    for engine in CHECKPOINTS:
        install(engine)
    return install


def _loaded(engine: str):
    """Run the real `_ensure_loaded` for an engine, hand back what it got."""
    from app.engines import ENGINES
    from app.synth import Synth

    synth = Synth(spec=ENGINES[engine])
    try:
        synth._ensure_loaded()  # noqa: SLF001 - the method under test
        return synth, synth._model  # noqa: SLF001
    finally:
        synth.close()


@pytest.mark.parametrize("engine", ["chatterbox", TURBO])
def test_the_checkpoint_that_loaded_is_the_one_the_row_named(
        engine, weights, build):
    """THE LEG NO AUDIO TEST IN THIS TREE CAN SEE.

    `_ensure_loaded` turns `spec.local_class` into a class, and every other
    test here fakes `Synth._speak`, so that line never runs and the fake's tone
    comes from the spec rather than from any weights. One row's `local_class`
    swapped for another's is therefore the single-token edit that leaves the
    header, the record, `engine_reason` AND the audio all saying the right
    thing.

    NO WEIGHTS AND NO CARD, WHICH IS WHY THIS RUNS ON EVERY COMMIT. The two
    facts that separate the shipped checkpoints are properties of the loaded
    object and cost nothing to read: `language_id` is a parameter of one
    generate() and a TypeError on the other, and `hp.emotion_adv` is True in
    one set of weights and False in the other. A check that needs a GPU is a
    check that is deselected, and a check that is deselected is the one nobody
    notices has never run -- which is what happened to the version of this test
    that shipped.
    """
    build(TTS_ENGINES=LOCAL_BOTH)
    from app.engines import ENGINES, catalogue_witness, checkpoint_witness

    synth, model = _loaded(engine)
    spec = ENGINES[engine]

    assert f"{type(model).__module__}:{type(model).__qualname__}" == \
        spec.local_class, "the row named one checkpoint and another loaded"
    # THE OBJECT AGAINST THE ROW, THROUGH THE SERVICE'S OWN TWO READERS. A
    # comparison written out again here would be a second implementation that
    # could agree with the row while the service's disagreed with it.
    assert checkpoint_witness(model) == catalogue_witness(spec)
    # AND THE SAME FACT EXERCISED RATHER THAN INSPECTED, because a signature
    # read and a call made are two different claims about one parameter.
    if catalogue_witness(spec)["language_id"]:
        model.generate("one short line.", language_id="en")
    else:
        with pytest.raises(TypeError):
            model.generate("one short line.", language_id="en")
    assert synth.loads == 1, "the load this test is about did not happen"


def test_loading_the_other_engines_checkpoint_is_refused_before_a_sample(
        weights, build):
    """THE FOURTH SINGLE-TOKEN EDIT, PERFORMED DELIBERATELY.

    Turbo's row resolved to the multilingual checkpoint. Everything downstream
    is written from the REQUEST -- the 202, the record's `engine`,
    `engine_reason`, `x-tts-engine` -- so all four go on saying turbo, and so
    does the fake's tone, because the fake reads the spec. This is the state in
    which the entire suite stays green while the audio is the wrong model's.

    REFUSED AND NOT LOGGED, AND BEFORE `from_pretrained`. A warning here would
    scroll past whoever was watching that terminal and the job would be served
    anyway; sixty-eight seconds and several gigabytes would also have been
    spent proving something already known. See engines.assert_named_checkpoint
    for why this raises where the drift check beside it only warns.
    """
    build(TTS_ENGINES=LOCAL_BOTH)
    from app.engines import ENGINES, catalogue_witness, checkpoint_witness

    weights(TURBO, _MultilingualLike)
    # The premise, stated rather than assumed: what is now behind turbo's row
    # really is the OTHER shipped engine and not merely a broken object.
    assert (checkpoint_witness(_MultilingualLike)
            == catalogue_witness(ENGINES["chatterbox"]))

    with pytest.raises(RuntimeError) as refused:
        _loaded(TURBO)

    said = str(refused.value)
    assert TURBO in said and "chatterbox" in said, \
        f"a refusal that names neither engine is unactionable: {said}"
    assert ENGINES[TURBO].local_class in said, \
        "nobody was told which catalogue entry to go and read"


def test_a_checkpoint_under_a_shorter_import_path_is_not_refused(
        weights, build):
    """THE GUARD ABOVE MUST NOT TAKE A WORKING SERVICE DOWN.

    A package that re-exports its class from `chatterbox` rather than from
    `chatterbox.mtl_tts` spells `local_class` differently and is still the same
    weights. Refusing on the NAME alone would turn a packaging change upstream
    into a dead engine, which is why the guard wants both a name it does not
    recognise and a set of facts belonging to another enabled engine.
    """
    build(TTS_ENGINES=LOCAL_BOTH)

    class _Reexported(_MultilingualLike):
        """The same checkpoint, reached by a name the row does not spell."""

        __module__ = "chatterbox"
        __qualname__ = "ChatterboxMultilingualTTS"

    weights("chatterbox", _Reexported)
    synth, model = _loaded("chatterbox")
    assert isinstance(model, _Reexported), \
        "the same weights under a different import path were refused"
    assert synth.loads == 1


# ----------------------------------------------- the real weights, on demand ---


@pytest.mark.checkpoint
@pytest.mark.parametrize("engine,witness", [
    ("chatterbox",
     "generate() takes language_id and hp.emotion_adv is True"),
    (TURBO,
     "generate(language_id=...) raises TypeError and hp.emotion_adv is False"),
])
def test_the_installed_weights_are_the_ones_the_catalogue_names(
        engine, witness, build):
    """THE SAME QUESTION, ASKED OF THE PACKAGE THAT IS ACTUALLY INSTALLED.

    DESELECTED BY DEFAULT AND THE REASON IS NARROW NOW. The test above runs
    everywhere and proves the SERVICE reads a checkpoint correctly and refuses
    the wrong one; what it cannot prove is that the wheel on this machine is
    what the catalogue describes, because importing it drags in torch and
    several gigabytes of weights. That is the only thing left here, and it is
    run deliberately -- `pytest -m checkpoint` -- where the package is.

    Not a skipif: "is chatterbox installed" would make this silently absent on
    the one machine it matters on, which is how `chatterbox-cpu` became a rung
    nothing ever offered a job. Deselected and named is a state somebody can
    see in the run summary.

    THE COMPARISON IS THE SERVICE'S OWN, through `checkpoint_witness` and
    `catalogue_witness`. This test used to re-implement it with `assert` where
    engines.assert_runtime implemented it with `log.warning` -- two halves of
    one feature that could disagree, in the file about halves that disagree.
    """
    import importlib

    build(TTS_ENGINES=LOCAL_BOTH)
    from app.engines import ENGINES, catalogue_witness, checkpoint_witness

    spec = ENGINES[engine]
    module_name, _, class_name = spec.local_class.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)

    assert checkpoint_witness(cls) == catalogue_witness(spec), witness
    assert f"{cls.__module__}:{cls.__qualname__}" == spec.local_class, \
        "the installed class is not the one the row names"
