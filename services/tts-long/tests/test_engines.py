"""Two engines on one API, and every way that could quietly ruin an answer.

The house rule this file exists to hold: EVERY FIELD IS EITHER HONOURED OR
REFUSED BY NAME. Chatterbox Turbo accepts `exaggeration`, `cfg_weight` and a
language and throws all three away with a logged warning nobody sees, so a
service that passed a request straight through would return audio that silently
ignored three fields the caller set. Every test below is named after the
mistake it prevents.

NOTHING HERE LOADS A MODEL OR OPENS A SOCKET. conftest fakes `Synth._speak`,
which is the one method that touches chatterbox, and no test in this file
configures a runner unless it is testing what happens when one is there.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

TURBO = "chatterbox-turbo"
BOTH = "chatterbox,chatterbox-turbo"


def _client(build, voice_dir, **env):
    env.setdefault("TTS_ENGINES", BOTH)
    return TestClient(build(**env))


def _wait(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"{job_id} never finished; it is still {job['status']}")


def _error(response):
    return response.json()["error"]


# ------------------------------------------------- the runner is optional ---


@pytest.mark.parametrize("engine", ["chatterbox", TURBO])
def test_every_advertised_engine_finishes_with_no_runner_at_all(
        engine, build, voice_dir):
    """SPRING OFF, UNPLUGGED, OR LYING.

    The rule this whole stack is built on is that the server must not rely on
    the runner. An engine that only exists while somebody's gaming PC is on is
    an engine that disappears, and an option that is runner-shaped is a
    runner-shaped hole in the API. Every engine this deployment advertises must
    produce audio here, on this CPU, as an ordinary job -- never a 503, never a
    202 that sits `queued` for ever.
    """
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir) as client:
        import app.main as main
        assert main.state.get("runner") is None, "this test needs no runner"

        posted = client.post("/jobs", json={"model": engine, "voice": "gabriel",
                                            "text": "hello there"})
        assert posted.status_code == 202, posted.text
        assert posted.json()["engine"] == engine
        job = _wait(client, posted.json()["id"])
        assert job["status"] == "done", job.get("error")
        assert job["backend"] == "local"
        assert job["engine"] == engine


def test_a_runner_only_engine_is_refused_at_submit_and_never_queued(
        build, voice_dir):
    """An option honestly, visibly absent is fine. A 202 nothing can serve is a
    progress bar that never moves.

    `_pick` returns None and loops with no deadline, and `_sweep` skips anything
    with `finished_at is None` -- so a job admitted for an engine no lane can
    run sits queued until the process dies, and thirty-two of them make `_full()`
    answer 429 to every caller of the OTHER engine on a completely idle lane.
    """
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir,
                 TTS_LOCAL_ENGINES="chatterbox",
                 TTS_ALLOW_RUNNER_ONLY_ENGINES="1") as client:
        import app.main as main
        before = len(main.jobs)
        refused = client.post("/jobs", json={"model": TURBO, "voice": "gabriel",
                                             "text": "hello there"})
        assert refused.status_code == 503, refused.text
        assert refused.headers["Retry-After"]
        assert "TTS_LOCAL_ENGINES" in refused.json()["detail"]
        assert len(main.jobs) == before, "a job was created that nothing can run"

        # And the engine that IS local is untouched by its neighbour's absence.
        assert client.post("/jobs", json={"model": "chatterbox",
                                          "voice": "gabriel",
                                          "text": "hello"}).status_code == 202


def test_an_engine_with_no_local_lane_refuses_to_start_at_all(build, voice_dir):
    """The state is created by two variables and cannot be created by a request,
    so it is refused where it is created rather than discovered at submit.

    A warning here would be read once, by whoever happened to be watching that
    terminal, and the runner-shaped hole would still be there a month later.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=BOTH, TTS_LOCAL_ENGINES="chatterbox")
    said = str(refused.value)
    assert TURBO in said
    assert "TTS_LOCAL_ENGINES" in said
    assert "TTS_ALLOW_RUNNER_ONLY_ENGINES" in said, "no way out was named"


# --------------------------------------------------- honoured or refused ----


def test_a_default_that_the_engine_cannot_honour_is_never_sent(build, voice_dir):
    """THE DEFECT THIS PREVENTS: compose.yaml speaking for the caller.

    TTS_EXAGGERATION=0.3 is set in the deployment. Resolving defaults before the
    engine is known would put that value on EVERY turbo request -- written by a
    config file rather than by a person -- where the model accepts it, logs a
    warning nobody sees and discards it. The field must not exist at all.
    """
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir, TTS_EXAGGERATION="0.3",
                 TTS_CFG_WEIGHT="0.3", TTS_TEMPERATURE="0.6") as client:
        import app.main as main
        seen: list[dict] = []
        original = main.Synth.speak_segments

        def record(self, segments, language, controls, reference=None,
                   on_chunk=None, cancelled=None):
            seen.append({"engine": self.spec.id, "language": language,
                         **controls})
            return original(self, segments, language, controls, reference,
                            on_chunk=on_chunk, cancelled=cancelled)

        main.Synth.speak_segments = record
        try:
            job = client.post("/jobs", json={"model": TURBO, "voice": "gabriel",
                                             "text": "hello there"}).json()
            _wait(client, job["id"])
        finally:
            main.Synth.speak_segments = original

        assert seen, "the job never reached a synth"
        got = seen[-1]
        assert got["engine"] == TURBO
        # ABSENT, NOT None, and the mapping is what makes that expressible. A
        # key that is present and None is the same silently-discarded keyword
        # as a key that is present and 0.3.
        assert "exaggeration" not in got, "the deployment default reached turbo"
        assert "cfg_weight" not in got, "the deployment default reached turbo"
        # temperature survives on this checkpoint, so it is honoured normally.
        assert got["temperature"] == 0.6

        # And the fields turbo cannot honour never reach generate() either.
        from app.engines import ENGINES, generate_kwargs
        kwargs = generate_kwargs(ENGINES[TURBO], language="en",
                                 controls={"exaggeration": 0.3,
                                           "cfg_weight": 0.3,
                                           "temperature": 0.6})
        assert set(kwargs) == {"temperature"}


def test_a_value_equal_to_the_deployment_default_is_still_refused(build,
                                                                  voice_dir):
    """EQUALITY WITH A DEPLOYMENT DEFAULT IS NOT CONSENT.

    0.3 is exactly what compose.yaml sets for the other engine. The caller still
    TYPED the field and believes it did something, so handing back audio that
    ignored it is the failure whether or not the number was a surprise.
    """
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir, TTS_EXAGGERATION="0.3") as client:
        refused = client.post("/jobs", json={"model": TURBO, "voice": "gabriel",
                                             "text": "hi", "exaggeration": 0.3})
        assert refused.status_code == 400, refused.text
        assert "exaggeration" in refused.json()["detail"]


@pytest.mark.parametrize("body,param,expected", [
    ({"model": "nonsense"}, "model", "is not one this service has"),
    ({"model": TURBO, "language": "pt"}, "language",
     "no language conditioning"),
    ({"model": TURBO, "exaggeration": 0.7}, "exaggeration", "emotion"),
    ({"model": TURBO, "cfg_weight": 0.7}, "cfg_weight", "guidance"),
    ({"model": "chatterbox", "language": "xx"}, "language",
     "is not one chatterbox speaks"),
])
def test_every_refusal_names_the_field_and_the_way_out(body, param, expected,
                                                       build, voice_dir):
    """Named, in the model's own terms, with something to do about it.

    "unsupported" is what an error says when nobody looked. A caller told that
    `exaggeration` is unsupported does not know whether to wait for a newer
    build or send a different model; one told the conditioning layer is not
    built in this checkpoint knows there is nothing to wait for.
    """
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir) as client:
        payload = {"voice": "gabriel", "input": "hello there", **body}
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 400, response.text
        error = _error(response)
        assert error["param"] == param
        assert expected in error["message"], error["message"]
        # A WAY OUT, ALWAYS. Every one of these is fixable by the caller in one
        # edit, and a refusal that does not say which edit is a dead end.
        assert "model=" in error["message"] or "chatterbox" in error["message"]


def test_a_short_reference_clip_is_refused_before_a_job_id_exists(build,
                                                                  voice_dir):
    """THE FIVE-SECOND MINIMUM IS A PROPERTY OF THE PAIR.

    Turbo asserts it itself, inside the worker -- an AssertionError on somebody's
    job, after a job id exists and a progress bar has started. Nothing but the
    server knows both the engine and the clip, so this is the only place it can
    be an error a person can read.
    """
    voice_dir("snippet", 3.8)
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir) as client:
        refused = client.post("/jobs", json={"model": TURBO, "voice": "snippet",
                                             "text": "hello there"})
        assert refused.status_code == 400, refused.text
        said = refused.json()["detail"]
        assert "3.8" in said and "5.0" in said
        # The SAME clip on the engine that has no minimum is fine, which is what
        # makes this a fact about the pair rather than about the voice.
        assert client.post("/jobs", json={"model": "chatterbox",
                                          "voice": "snippet",
                                          "text": "hello"}).status_code == 202
        # And the long clip is fine on both.
        assert client.post("/jobs", json={"model": TURBO, "voice": "gabriel",
                                          "text": "hello"}).status_code == 202


def test_the_voice_list_says_which_engines_can_use_each_voice(build, voice_dir):
    """DISABLED WITH A REASON, NEVER HIDDEN.

    A page that quietly drops the turbo option for a short-clip voice is how
    `chatterbox-cpu` stayed invisible for its whole life. `detail` is additive:
    `voices` keeps its exact shape, because a picker is reading it right now.
    """
    voice_dir("snippet", 3.8)
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir) as client:
        doc = client.get("/voices").json()
        assert doc["voices"] == ["default", "gabriel", "snippet"]
        detail = {row["name"]: row for row in doc["detail"]}
        assert detail["gabriel"]["engines"] == ["chatterbox", TURBO]
        assert detail["snippet"]["engines"] == ["chatterbox"]
        assert "3.8" in detail["snippet"]["excluded"][TURBO]
        # The built-in speaker has no clip at all, which is not a short clip.
        assert detail["default"]["reference_seconds"] is None
        assert detail["default"]["engines"] == ["chatterbox", TURBO]


def test_a_per_engine_key_for_an_absent_control_fails_at_start(build):
    """A config key that does nothing is the house-rule failure with a longer fuse.

    Somebody sets it, hears no difference, and concludes the model is broken.
    The container that will not start IS the bug report, and boot is the one
    moment anybody is looking.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=BOTH, TTS_CHATTERBOX_TURBO_EXAGGERATION="0.5")
    said = str(refused.value)
    assert "TTS_CHATTERBOX_TURBO_EXAGGERATION" in said
    assert "emotion_adv" in said, "the reason was not the model's own"
    assert "TTS_EXAGGERATION" in said, "no way out was named"


def test_an_engine_that_is_not_in_the_catalogue_fails_at_start(build):
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES="chatterbox,chatterbox-ultra")
    assert "chatterbox-ultra" in str(refused.value)


def test_no_branch_on_an_engine_name(build):
    """A THIRD ENGINE MUST BE A CATALOGUE ROW AND NO CODE.

    The moment `if engine == "chatterbox-turbo"` appears in a route, a
    dispatcher or a synth, the next engine is a search through this package for
    every place the last one was special-cased -- and the one that gets missed
    is the one that silently drops a field.

    THE AST, NOT A GREP. Prose names the engine constantly and must be allowed
    to; what may not exist is the NAME AS A VALUE in running code. Docstrings
    are the module's own prose and are excluded by position, exactly as a
    documentation tool would find them.

    `chatterbox` itself is exempt and the exemption is named: it is the id this
    service had before there were two, so it survives as RunnerConfig's
    documented legacy default for TTS_RUNNER_SERVICE and as the true engine of
    every record written before this release. Every id added after it must be
    reachable only through the catalogue.
    """
    import ast

    from voice_common.engines import CATALOGUE_IDS

    watched = CATALOGUE_IDS - {"chatterbox"}
    assert watched, "the catalogue has only the legacy engine; nothing to check"

    package = Path(__file__).resolve().parent.parent / "app"
    offenders = []
    for source in sorted(package.glob("*.py")):
        if source.name == "engines.py":
            # The one module allowed to know engine names, and even it reaches
            # them through voice_common.engines rather than by spelling them.
            continue
        tree = ast.parse(source.read_text())
        prose = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                first = (node.body or [None])[0]
                if (isinstance(first, ast.Expr)
                        and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    prose.add(id(first.value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value in watched and id(node) not in prose):
                offenders.append(f"{source.name}:{node.lineno}: {node.value!r}")
    assert not offenders, ("an engine is named as a value outside engines.py:\n"
                           + "\n".join(offenders))


# ------------------------------------------------------- what came back -----


def test_the_response_says_which_engine_made_the_audio(build, voice_dir):
    """It is the only way a client holding a file can tell which of two engines
    produced it. Mirrors x-stt-engine, which the transcription side publishes
    for exactly the same reason."""
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir) as client:
        accepted = client.post("/v1/audio/speech",
                               json={"model": TURBO, "voice": "gabriel",
                                     "input": "x" * 3000})
        assert accepted.status_code == 202
        assert accepted.headers["x-tts-engine"] == TURBO

        streamed = client.post("/v1/audio/speech",
                               json={"model": "chatterbox", "voice": "gabriel",
                                     "input": "hello", "stream_format": "sse"})
        assert streamed.headers["x-tts-engine"] == "chatterbox"


def test_the_record_says_which_engine_and_why_that_one(build, voice_dir):
    """`chatterbox` on a row cannot distinguish "the caller named it" from "the
    caller named nothing" -- which is how a change of TTS_DEFAULT_ENGINE becomes
    invisible in six months of history."""
    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir) as client:
        pinned = client.post("/jobs", json={"model": TURBO, "voice": "gabriel",
                                            "text": "hello"}).json()["id"]
        silent = client.post("/jobs", json={"voice": "gabriel",
                                            "text": "hello"}).json()["id"]
        assert _wait(client, pinned)["engine_reason"] == "pinned"
        row = _wait(client, silent)
        assert row["engine_reason"] == "default"
        assert row["engine"] == "chatterbox"

        aliased = client.post("/v1/audio/speech",
                              json={"model": "tts-1", "voice": "gabriel",
                                    "input": "x" * 3000}).json()["id"]
        assert _wait(client, aliased)["engine_reason"] == "alias:openai"


def test_health_publishes_each_engines_honest_surface(build, voice_dir):
    """`controls` and `languages` are what let a page remove a slider that would
    do nothing rather than draw one that lies."""
    with _client(build, voice_dir) as client:
        doc = client.get("/health").json()
        assert doc["default_engine"] == "chatterbox"
        turbo = doc["engines"][TURBO]
        assert turbo["languages"] == ["en"]
        assert turbo["controls"] == ["temperature"]
        assert turbo["min_reference_seconds"] == 5.0
        assert turbo["local"]["ready"] is True
        assert turbo["runner"]["service"] == TURBO
        assert turbo["runner"]["ready"] is False
        assert doc["engines"]["chatterbox"]["default"] is True


# --------------------------------------------------------------- rates ------


def test_two_engines_on_one_lane_do_not_share_an_average(build, voice_dir):
    """THE SAME ARGUMENT AS THE PER-BACKEND SPLIT, ONE LEVEL DOWN.

    The two checkpoints are 2.36x apart on the SAME card. One average over both
    accepts a synchronous request the slow one can never finish, and it does it
    at exactly the worst moment.
    """
    with _client(build, voice_dir) as client:
        import app.main as main
        baseline = main.rate_for("runner", "chatterbox").value
        main.rate_for("runner", TURBO).observe(audio_seconds=154.0,
                                               compute_seconds=100.0)
        assert main.rate_for("runner", "chatterbox").value == baseline, \
            "a turbo measurement moved the baseline estimate"
        assert main.rate_for("runner", TURBO).value != baseline

        doc = client.get("/health").json()
        assert doc["engine_observations"][f"runner/{TURBO}"] == 1
        assert doc["engine_observations"]["runner/chatterbox"] == 0
        assert doc["realtime_factor_by_engine"][f"runner/{TURBO}"] > \
            doc["realtime_factor_by_engine"]["runner/chatterbox"]


def test_local_chatterbox_is_the_same_object_as_rate(build, voice_dir):
    """`rate` is read directly in several places and rate_for() in others. If
    they ever stopped being the same object, half this service's arithmetic
    would silently use a different number from the other half."""
    with _client(build, voice_dir):
        import app.main as main
        assert main.rate_for("local") is main.rate
        assert main.rate_for("local", "chatterbox") is main.rate
        assert main.rate_for("local", TURBO) is not main.rate


def test_realtime_factor_by_backend_still_answers(build, voice_dir):
    """THE PAGE IS NOT PART OF THIS SLICE AND MUST NOT BREAK.

    ui.html and test_interface.py read `realtime_factor_by_backend` and
    `backend_observations`. Narrowed to the default engine rather than left as
    an average of two things 2.36x apart, they stay true -- which is what lets
    the server half ship before the page half.
    """
    with _client(build, voice_dir) as client:
        import app.main as main
        main.rate_for("runner", TURBO).observe(audio_seconds=154.0,
                                               compute_seconds=100.0)
        doc = client.get("/health").json()
        assert set(doc["realtime_factor_by_backend"]) == {"local", "runner"}
        assert doc["realtime_factor_by_backend"]["local"] == \
            round(main.rate.value, 3)
        assert doc["backend_observations"]["runner"] == 0, \
            "a turbo measurement leaked into the figure the page draws"


def test_the_seed_for_a_second_engine_is_not_the_first_ones(build, voice_dir):
    """TTS_REALTIME_FACTOR_RUNNER is a measurement of the multilingual model on
    a 3070. Handing it to turbo would understate it by 2.36x, and an understated
    rate is the number that decides whether a caller is answered synchronously."""
    with _client(build, voice_dir, TTS_REALTIME_FACTOR_RUNNER="0.70"):
        import app.main as main
        assert main.rate_for("runner", "chatterbox").value == 0.70
        assert main.rate_for("runner", TURBO).value == 1.54
    # And the specific key wins outright over both.


def test_a_pair_key_overrides_the_catalogue(build, voice_dir):
    with _client(build, voice_dir,
                 TTS_REALTIME_FACTOR_LOCAL_CHATTERBOX_TURBO="0.55"):
        import app.main as main
        assert main.rate_for("local", TURBO).value == 0.55


# ----------------------------------------------------------- the lanes ------


def test_a_turbo_job_is_never_dispatched_at_a_runner_that_has_no_turbo(
        build, voice_dir):
    """THE LANE IS OPEN AND THIS ENGINE IS NOT NECESSARILY ON IT.

    `Lane.free()` asks whether the machine will take work at all. A job names
    ONE engine, and the runner carries one speech service per engine. Without
    the per-engine filter a turbo job is handed to a runner with no turbo
    service, comes back `no_such_service` and bounces -- once per job, for ever,
    on a lane that was working perfectly for the other engine.
    """
    from test_remote import _client_answering, _wait as wait_remote, runner

    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir) as client:
        offered = _client_answering({
            "gpu_available": True,
            "services": [{"id": "chatterbox", "device": "gpu", "installed": True,
                          "enabled": True, "available": True}]})
        with runner(offered) as main:
            # The runner is free, and it is free for the engine it carries.
            lane = main.dispatch.lanes["runner"]
            assert lane.probe.ok() is True
            assert lane.probe.ok_for("chatterbox") is True
            assert lane.probe.ok_for(TURBO) is False

            eligible = main.dispatch._eligible(
                {"id": "x", "engine": TURBO}, "x", 0.0)
            assert [l.name for l in eligible] == ["local"], \
                "a turbo job was offered a runner with no turbo service"
            eligible = main.dispatch._eligible(
                {"id": "y", "engine": "chatterbox"}, "y", 0.0)
            assert "runner" in [l.name for l in eligible], \
                "the runner lost the engine it does carry"

            posted = client.post("/jobs", json={"model": TURBO,
                                                "voice": "gabriel",
                                                "text": "One short line."})
            finished = wait_remote(client, posted.json()["id"])
            assert finished["status"] == "done"
            assert finished["backend"] == "local"
            assert not finished.get("fell_back"), \
                "the job bounced off the runner instead of never being offered"


def test_the_two_language_lists_agree():
    """TWO COPIES OF THE SAME FACT, AND THIS IS THE ONLY THING HOLDING THEM.

    app/synth.py's SUPPORTED_LANGUAGES is what the loaded model is cross-checked
    against, and services/ui reads it out of the source file; the catalogue is
    what requests are validated against. Neither can import the other -- one
    would drag torch into the gateway, the other is read as TEXT by a test in a
    different package -- so a test is the only place the two can be made to
    agree, and drift here means a 400 for a language the model speaks.
    """
    from app.synth import SUPPORTED_LANGUAGES
    from voice_common.engines import CATALOGUE

    assert CATALOGUE["chatterbox"].languages == SUPPORTED_LANGUAGES


# ------------------------------------------------------- the NumPy floor ----


def test_the_reference_clip_stays_float32_under_numpy_two(caplog):
    """NEP 50 KILLS EVERY CLONED VOICE ON A NUMPY 2 IMAGE, AND THAT IS THE FLOOR.

    `norm_loudness` multiplies a float32 clip by an `np.float64` gain from
    pyloudnorm. Under NumPy 1.x the scalar was demoted; under NumPy 2 the
    value-based rules are gone, so the whole clip promotes to float64 and
    s3tokenizer's float32 mel filters raise "expected m1 and m2 to have the
    same dtype, but got: float != double" in prepare_conditionals -- before a
    single token. Device-independent: spring escapes it only by running NumPy
    1.x, and chatterbox-tts pins numpy>=2.0.0 for Python >= 3.13.

    WITHOUT THIS THERE IS NO LOCAL FLOOR, and "every advertised engine works
    with the runner switched off" is a claim rather than a design.
    """
    import sys
    import types

    import numpy as np

    from app.synth import _pin_every_loaded_module

    fake = types.ModuleType("chatterbox.pretend")

    def norm_loudness(clip):
        # Exactly what pyloudnorm does to it: a float64 scalar times a float32
        # array, which under NEP 50 promotes the array.
        return clip * np.float64(0.5)

    fake.norm_loudness = norm_loudness
    sys.modules["chatterbox.pretend"] = fake
    try:
        clip = np.ones(8, dtype=np.float32)
        assert fake.norm_loudness(clip).dtype == np.float64, \
            "this NumPy does not promote, so the guard is untested here"
        _pin_every_loaded_module()
        assert fake.norm_loudness(clip).dtype == np.float32
        # And it does not stack: pinning twice must not wrap twice.
        _pin_every_loaded_module()
        assert fake.norm_loudness(clip).dtype == np.float32
    finally:
        del sys.modules["chatterbox.pretend"]


def test_a_guard_that_could_not_attach_says_so_on_numpy_two(caplog):
    """A GUARD THAT SILENTLY FAILS TO ATTACH IS THE CLASS OF DEFECT THIS WHOLE
    RELEASE IS ABOUT.

    If the function moves upstream, every cloned voice fails with a dtype
    mismatch nobody would connect to a NumPy release note. Silent on NumPy 1.x,
    where the promotion cannot happen and there is nothing to say.
    """
    import numpy as np

    from app.synth import _pin_every_loaded_module

    if int(np.__version__.split(".")[0]) < 2:
        pytest.skip("nothing to warn about on NumPy 1.x")
    with caplog.at_level("WARNING"):
        _pin_every_loaded_module()
    assert any("norm_loudness" in r.getMessage() for r in caplog.records), \
        "the guard did not attach and did not say so"


def test_a_turbo_job_on_the_runner_asks_for_the_turbo_service(build, voice_dir):
    """THE ENGINE RIDES ON THE JOB AND RESOLVES TO A SERVICE ID AT SUBMIT.

    One machine, one card, one lane -- and two speech services on it. A turbo
    job that asked for `chatterbox` would be answered by the baseline
    controller: audio that is real, playable, in the wrong voice-model, with
    nothing anywhere reporting a fault. It is the same failure the manifest-id
    assertion catches from the runner's side, reached from this one.
    """
    import json

    from test_remote import _wait as wait_remote, runner

    voice_dir("gabriel", 61.0)
    with _client(build, voice_dir, TTS_RUNNER_HOP_S="0") as client:
        from app.remote import RunnerClient, RunnerConfig

        asked: list[str] = []
        base = RunnerClient(RunnerConfig(host="runner.invalid",
                                         service="chatterbox", poll=0.0))

        def transport(method, path, body=None, content_type=None, extra=None,
                      timeout=None):
            asked.append(f"{method} {path}")
            if path == "/v1/services":
                return 200, {}, json.dumps({
                    "gpu_available": True,
                    "services": [
                        {"id": "chatterbox", "device": "gpu", "installed": True,
                         "enabled": True, "available": True},
                        {"id": TURBO, "device": "gpu", "installed": True,
                         "enabled": True, "available": True}]}).encode()
            if method == "POST" and path.endswith("/jobs"):
                return 200, {}, json.dumps({"job_id": "r1"}).encode()
            if path.endswith("/jobs/r1"):
                return 200, {}, json.dumps({
                    "status": "done", "artefacts": ["r1.0.f32"],
                    "record": {"input_tokens": 3}}).encode()
            if "result" in path:
                import numpy as np
                return 200, {}, np.zeros(2400, dtype="<f4").tobytes()
            return 200, {}, b"{}"

        base._request = transport

        with runner(base) as main:
            posted = client.post("/jobs", json={"model": TURBO,
                                                "voice": "gabriel",
                                                "text": "One short line."})
            finished = wait_remote(client, posted.json()["id"])

    assert finished["status"] == "done", finished.get("error")
    assert finished["backend"] == "runner", "the job never crossed the network"
    assert finished["runner_service"] == TURBO, \
        "a turbo job was submitted to the baseline controller"
    assert any(f"/v1/services/{TURBO}/jobs" in call for call in asked), \
        f"nothing was submitted to the turbo service; asked: {asked}"
    assert not any("/v1/services/chatterbox/jobs" in call for call in asked), \
        "the baseline service was asked to speak a turbo job"


# ================================================================= VOXTRAL ==
#
# A THIRD ENGINE, AND THE FIRST ONE THIS HOST CANNOT RUN AT ALL. Everything
# below is either "an engine with no local lane does not become a hole in the
# API" or "a field this checkpoint has no way to honour is refused by name".
#
# It is a preset-voice engine: twenty speaker embeddings baked into the
# weights, no speaker encoder anywhere in the checkpoint, no reference-audio
# parameter in any source file. So it sits beside Kokoro conceptually and not
# beside Chatterbox, and every assumption this service made about a "voice"
# being a file in a directory is tested here.

VOXTRAL = "voxtral"
THREE = "chatterbox,chatterbox-turbo,voxtral"


def _runner_is_up(main, *engines):
    """Tell the runner lane's probe that it carries these engines, and nothing
    else about it.

    A REFUSAL BEFORE A JOB ID NEEDS A LANE TO ASK, and `refuse_unavailable` is
    the last question `_choose` puts. Every test that wants to see what comes
    AFTER it -- which language landed on the record, which controls survived --
    needs the probe to say yes, and standing up a whole fake runner to answer
    one boolean would be testing the transport instead.
    """
    lane = main.dispatch.lanes["runner"]

    class _Up:
        def ok(self):
            return True

        def ok_for(self, engine):
            return engine in engines

        def why_for(self, engine):
            return "" if engine in engines else "no_such_service"

        def start(self):
            pass

        def stop(self):
            pass

        def snapshot(self):
            return {"ready": True}

    lane.probe = _Up()
    return lane


def _three(build, voice_dir, **env):
    """All three engines, with the one that has no CPU path served by the runner."""
    env.setdefault("TTS_ENGINES", THREE)
    env.setdefault("TTS_LOCAL_ENGINES", "chatterbox,chatterbox-turbo")
    env.setdefault("TTS_ALLOW_RUNNER_ONLY_ENGINES", "1")
    return TestClient(build(**env))


# ------------------------------------------------- SO: the spring-off rule ---


def test_the_default_engine_must_have_a_local_lane(build):
    """SO-1. THE INVARIANT THAT MAKES "THE SERVER DOES NOT RELY ON SPRING"
    PROVABLE RATHER THAN BELIEVED.

    Every alias resolves to the default -- `tts-long`, OpenAI's three names and
    an absent `model` -- so a runner-only default means an OpenAI client that
    has never heard of this engine gets a 503 because somebody sat down at a
    gaming PC. The state is created by two variables and cannot be created by a
    request, so it is refused where it is created.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE, TTS_LOCAL_ENGINES="chatterbox",
              TTS_ALLOW_RUNNER_ONLY_ENGINES="1",
              TTS_DEFAULT_ENGINE=VOXTRAL)
    said = str(refused.value)
    assert "TTS_DEFAULT_ENGINE" in said and "TTS_LOCAL_ENGINES" in said
    assert "alias" in said, "the reason -- every alias resolves here -- was not given"


def test_no_alias_can_reach_a_runner_only_engine(build, voice_dir):
    """SO-2. NO CALLER REACHES A RUNNER-ONLY ENGINE WITHOUT TYPING ITS NAME.

    The runner is away for the whole of this test. Every one of these bodies
    has to be answered by an engine this host can run, because none of them
    names the one it cannot -- and a 503 for `model` absent would be the
    runner-shaped hole in the API that the whole design refuses to have.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        import app.main as main
        assert main.state.get("runner") is None, "this test needs no runner"
        for model in (None, "tts-long", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"):
            body = {"voice": "gabriel", "text": "hello there"}
            if model is not None:
                body["model"] = model
            posted = client.post("/jobs", json=body)
            assert posted.status_code == 202, f"{model}: {posted.text}"
            assert posted.json()["engine"] != VOXTRAL, \
                f"model={model!r} reached an engine nothing here can run"


def test_a_runner_only_engine_is_refused_before_a_job_id(build, voice_dir):
    """SO-3. 503 AT SUBMIT, BEFORE A JOB ID AND BEFORE A QUEUE SLOT.

    A 202 nothing can serve is a progress bar that never moves, and thirty-two
    of them make `_full()` answer 429 to every caller of an engine that works.
    Both routes, because /jobs and /v1/audio/speech are two renderings of one
    decision and used to ask two different subsets of it.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        import app.main as main
        before = len(main.jobs)

        native = client.post("/jobs", json={"model": VOXTRAL,
                                            "voice": "pt_male",
                                            "text": "olá"})
        assert native.status_code == 503, native.text
        assert native.headers["Retry-After"]
        assert VOXTRAL in native.json()["detail"]

        openai = client.post("/v1/audio/speech", json={"model": VOXTRAL,
                                                       "voice": "pt_male",
                                                       "input": "olá"})
        assert openai.status_code == 503, openai.text
        assert _error(openai)["code"] == "engine_unavailable"

        assert len(main.jobs) == before, "a job was created that nothing can run"
        assert not list((Path(main.OUT_DIR) / "runs").glob("*.json")), \
            "a record was written for a job that never existed"


def test_the_service_survives_the_runner_being_away(build, voice_dir):
    """SO-4. THE SERVICE DOES NOT RELY ON SPRING; ONE ENGINE DOES.

    In exactly the state that refuses a Voxtral request, both local engines
    have to produce real audio on this CPU as ordinary jobs.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        for engine in ("chatterbox", TURBO):
            posted = client.post("/jobs", json={"model": engine,
                                                "voice": "gabriel",
                                                "text": "hello there"})
            assert posted.status_code == 202, posted.text
            job = _wait(client, posted.json()["id"])
            assert job["status"] == "done", job.get("error")
            assert job["backend"] == "local"


def test_a_runner_only_engine_still_needs_the_opt_out(build):
    """SO-6. The escape hatch stays opt-in for everybody who did not ask for it."""
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE, TTS_LOCAL_ENGINES="chatterbox,chatterbox-turbo")
    said = str(refused.value)
    assert VOXTRAL in said
    assert "TTS_ALLOW_RUNNER_ONLY_ENGINES" in said, "no way out was named"


def test_an_engine_with_no_cpu_path_cannot_be_made_local_by_asking(build):
    """`local_class` IS None AND THAT IS NOT "NOT INSTALLED HERE".

    There is no CPU implementation and no prospect of one: torchao's int4 path
    calls torch.cuda.get_device_capability() before any device dispatch, the
    BF16 escape hatch calls torch.cuda.synchronize() unconditionally, and the
    two dials the audio was tuned on exist only on a CUDA-only fast path. A
    lane that accepted this engine would import None INSIDE somebody's job,
    after a job id and a progress bar already existed.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE,
              TTS_LOCAL_ENGINES="chatterbox,chatterbox-turbo,voxtral")
    said = str(refused.value)
    assert VOXTRAL in said and "TTS_LOCAL_ENGINES" in said
    assert "does not exist" in said, \
        "it reads as a missing install rather than an absent implementation"


def test_adding_it_to_TTS_ENGINES_alone_does_not_make_it_local_by_omission(build):
    """TTS_LOCAL_ENGINES DEFAULTS TO "EVERY ENGINE THAT HAS A LANE TO BE PUT IN".

    It defaulted to "every engine", which was true while every catalogue row
    had a local class and silently wrong the moment one did not. An operator
    who adds this engine and changes nothing else must meet the runner-only
    invariant -- which names the variable that turns it on -- rather than a
    lane that accepts the job and fails it.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE)
    said = str(refused.value)
    assert "TTS_ALLOW_RUNNER_ONLY_ENGINES" in said
    # AND IT IS THE RUNNER-ONLY INVARIANT THAT SPEAKS, not the one about a
    # lane that has no class. The two have different fixes: one says "accept a
    # 503 when the runner is away", the other says "you asked for something
    # that does not exist". Landing on the second would mean the default had
    # quietly put the engine on the local lane first.
    assert "TTS_ENGINES offers" in said, said


def test_a_preset_engine_is_never_answered_synchronously(build, voice_dir):
    """0.104x REALTIME IS A JOB. ALWAYS. AND NOT BY A BRANCH ON A NAME.

    A 20-second clip is three and a half minutes of GPU at the settings this
    deployment ships. The arithmetic already refuses it, and agreeing is not
    the same as being unable to disagree: `_compute_seconds` divides by a LOCAL
    rate, and no local rate for this engine has ever been measured on any
    machine in this stack. None is the honest value, `inf` is what it produces,
    and `inf` cannot pass on somebody else's faster card or the day
    TTS_OPENAI_SYNC_TIMEOUT is raised.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        import app.main as main
        import math

        assert main._compute_seconds(10, VOXTRAL) == math.inf
        assert main._compute_seconds(10, "chatterbox") < math.inf
        # And no `local/voxtral` figure is published, because publishing one
        # would put a number nobody measured beside two that were.
        doc = client.get("/health").json()
        assert f"local/{VOXTRAL}" not in doc["realtime_factor_by_engine"], \
            "a local rate was published for an engine with no local lane"
        assert doc["realtime_factor_by_engine"][f"runner/{VOXTRAL}"] == 0.104


# ------------------------------------------- honoured or refused, by name ---


@pytest.mark.parametrize("body,param,expected", [
    ({"model": VOXTRAL, "voice": "pt_male", "exaggeration": 0.7},
     "exaggeration", "no emotion conditioning"),
    ({"model": VOXTRAL, "voice": "pt_male", "cfg_weight": 0.7},
     "cfg_weight", "its guidance dial is cfg_alpha"),
    ({"model": VOXTRAL, "voice": "pt_male", "temperature": 0.7},
     "temperature", "sampler does not read it"),
    ({"model": VOXTRAL, "voice": "gabriel"},
     "voice", "no speaker encoder"),
    ({"model": VOXTRAL, "voice": "alloy"},
     "voice", "OpenAI aliases"),
    ({"model": VOXTRAL, "voice": "pt_male", "language": "ja"},
     "language", "is not one voxtral speaks"),
    ({"model": VOXTRAL, "voice": "pt_male", "language": "de"},
     "language", "per-language and this one is pt"),
    ({"model": TURBO, "voice": "gabriel", "flow_steps": 32},
     "flow_steps", "no flow-matching decoder"),
    ({"model": "chatterbox", "voice": "gabriel", "cfg_alpha": 1.2},
     "cfg_alpha", "flow-matching solver"),
    ({"model": "chatterbox", "voice": "pt_male"},
     "voice", "preset speaker embeddings"),
])
def test_every_voxtral_refusal_names_the_field_and_the_way_out(
        body, param, expected, build, voice_dir):
    """EVERY FIELD IS EITHER HONOURED OR REFUSED BY NAME. None is dropped.

    This checkpoint has no reference clip, no exaggeration, no cfg_weight, no
    temperature and only twenty fixed voices, and each of those is a different
    sentence because each has a different way out. "unsupported" is what an
    error says when nobody looked: a caller told cfg_weight is unsupported
    cannot tell whether to wait for a newer build or send cfg_alpha.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        payload = {"input": "hello there", **body}
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 400, response.text
        error = _error(response)
        assert error["param"] == param
        assert error["code"] == "unsupported_value"
        assert expected in error["message"], error["message"]
        # A WAY OUT, ALWAYS, and for a language it is the list of the nine this
        # checkpoint speaks rather than another engine -- `ja` is not somewhere
        # else on this service, it is nowhere.
        assert (param == "language"
                or "model=" in error["message"]
                or "voice=" in error["message"]), error["message"]


def test_a_clip_cannot_become_a_voice_on_a_checkpoint_with_no_encoder(
        build, voice_dir):
    """OpenAI's CUSTOM-VOICE OBJECT IS A CLONED VOICE BY DEFINITION.

    `{"id": "voice_1234"}` on a checkpoint whose nine source files contain no
    reference-audio parameter cannot be honoured even in part -- not
    resampled, not approximated, and above all not mapped onto the nearest
    preset by ear, which would be a field accepted and quietly turned into
    something else.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        refused = client.post("/v1/audio/speech",
                              json={"model": VOXTRAL, "input": "hi",
                                    "voice": {"id": "voice_1234"}})
        assert refused.status_code == 400, refused.text
        message = _error(refused)["message"]
        assert "cannot clone" in message
        assert "reference-audio parameter" in message
        assert "chatterbox" in message, "no way out was named"


def test_the_language_comes_off_the_voice_and_never_off_a_deployment_default(
        build, voice_dir):
    """THE BUG NOBODY CAUGHT, AND IT IS ONE LINE.

    `_defaults_from_env` wrote `out["language"]` unconditionally, so
    TTS_LANGUAGE=en -- which every deployment sets -- would stamp `en` onto a
    pt_male job and put it on the record beside a Portuguese speaker. This
    checkpoint's language IS the voice embedding; there is nothing for a
    deployment-wide default to mean.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir, TTS_LANGUAGE="en") as client:
        from app.engines import ENGINES
        assert "language" not in ENGINES[VOXTRAL].defaults, \
            "a deployment default was resolved for a language nobody can set"
        assert ENGINES["chatterbox"].defaults["language"] == "en", \
            "the engines that DO take a language lost their default"

        import app.main as main
        _runner_is_up(main, VOXTRAL)
        chosen = main._choose(model=VOXTRAL, voice="pt_male", language=None,
                              controls={})
        assert chosen.params["language"] == "pt", \
            "the record would carry no language beside a Portuguese voice"

        german = main._choose(model=VOXTRAL, voice="de_female", language=None,
                              controls={})
        assert german.params["language"] == "de", \
            "the language was derived from the name rather than from the tensor"

        # And the two that carry no language prefix at all, which is the whole
        # argument for the language being ON the voice rather than parsed off it.
        casual = main._choose(model=VOXTRAL, voice="cheerful_female",
                              language=None, controls={})
        assert casual.params["language"] == "en"


def test_a_deployment_wide_language_for_a_preset_engine_fails_at_start(build):
    """It could only ever agree with the voice or contradict it, and the second
    puts `en` on a row beside a Portuguese speaker."""
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE, TTS_LOCAL_ENGINES="chatterbox,chatterbox-turbo",
              TTS_ALLOW_RUNNER_ONLY_ENGINES="1", TTS_VOXTRAL_LANGUAGE="pt")
    said = str(refused.value)
    assert "TTS_VOXTRAL_LANGUAGE" in said
    assert "voice embedding" in said, "the reason was not the model's own"


# --------------------------------------------------- the tuning contract ----


def test_the_quality_settings_are_config_keys_with_documented_defaults(
        build, voice_dir):
    """THE OWNER SAID "I WILL TUNE IT LATER" AND THAT IS A REQUIREMENT.

    32 steps was arrived at by ear -- 16 close behind, 8 and 4 audibly worse --
    and the upstream repo defaults to 8, so this key is the difference between
    three and a half minutes and one. It has to be changeable from
    compose.yaml, per engine, without editing a line of code, and the value
    that produced a given recording has to be on that recording's row.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir, TTS_VOXTRAL_FLOW_STEPS="16",
                TTS_VOXTRAL_CFG_ALPHA="1.4") as client:
        from app.engines import ENGINES
        assert ENGINES[VOXTRAL].defaults["flow_steps"] == 16
        assert ENGINES[VOXTRAL].defaults["cfg_alpha"] == 1.4
        # AN INTEGER, AND NOT COSMETICALLY. flow_steps is a loop count on the
        # runner: `range(16.0)` is a TypeError three and a half minutes into
        # somebody's job.
        assert isinstance(ENGINES[VOXTRAL].defaults["flow_steps"], int)


def test_a_deployment_default_outside_the_wires_range_fails_at_start(build):
    """THE FAILURE THIS PREVENTS IS A SERVICE THAT STARTS AND THEN REFUSES
    EVERYTHING.

    The bounds were two literals: `Field(ge=..., le=...)` on the request models
    against a bare `float(raw)` in the config loader. So a deployment could set
    a value at boot that every request was then answered 422 for -- a body
    that had never mentioned the field.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE, TTS_LOCAL_ENGINES="chatterbox,chatterbox-turbo",
              TTS_ALLOW_RUNNER_ONLY_ENGINES="1", TTS_VOXTRAL_FLOW_STEPS="200")
    said = str(refused.value)
    assert "TTS_VOXTRAL_FLOW_STEPS" in said and "64" in said

    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE, TTS_LOCAL_ENGINES="chatterbox,chatterbox-turbo",
              TTS_ALLOW_RUNNER_ONLY_ENGINES="1", TTS_VOXTRAL_FLOW_STEPS="16.5")
    assert "whole number" in str(refused.value)


def test_a_global_key_that_reaches_no_enabled_engine_fails_at_start(build):
    """GATE 1's LIE, RELOCATED. TTS_EXAGGERATION on a box running only this
    engine is set, read, narrowed away and honoured by nothing -- the same "I
    set it and heard no difference" with one more indirection in front of it.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES="chatterbox-turbo,voxtral",
              TTS_LOCAL_ENGINES=TURBO,
              TTS_ALLOW_RUNNER_ONLY_ENGINES="1", TTS_DEFAULT_ENGINE=TURBO,
              TTS_EXAGGERATION="0.3")
    said = str(refused.value)
    assert "TTS_EXAGGERATION" in said
    assert "emotion conditioning" in said, "the reason was not the model's own"


def test_a_per_engine_key_names_this_checkpoints_own_reason(build):
    """`_why_no_control` WAS ONE TABLE FOR EVERY ENGINE AND THAT WENT WRONG THE
    MOMENT A THIRD ARRIVED.

    The shared sentence about `exaggeration` described turbo -- "hp.emotion_adv
    is False" -- and would have been handed verbatim to somebody asking a
    Mistral flow-matching checkpoint that has no `hp` at all. A reason that is
    wrong about the model is worse than "unsupported": it sends the reader to
    the wrong package.
    """
    with pytest.raises(SystemExit) as refused:
        build(TTS_ENGINES=THREE, TTS_LOCAL_ENGINES="chatterbox,chatterbox-turbo",
              TTS_ALLOW_RUNNER_ONLY_ENGINES="1", TTS_VOXTRAL_EXAGGERATION="0.5")
    said = str(refused.value)
    assert "TTS_VOXTRAL_EXAGGERATION" in said
    assert "generate_speech_fast" in said, "turbo's reason was reused"
    assert "emotion_adv" not in said, "turbo's reason was reused"


# ------------------------------------------- what the row and /health say ---


def test_health_publishes_the_preset_voices_and_the_facts_a_picker_needs(
        build, voice_dir):
    """THE ONLY HONEST HOME FOR A FIXED VOICE LIST.

    Anything else is a fifth model table in the browser -- and the last four
    are why `de_female` would resolve to en-us under a first-letter rule while
    `pt_male` resolved correctly by coincidence. The language is ON the voice
    because it is a property of which tensor it is.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        row = client.get("/health").json()["engines"][VOXTRAL]
        assert row["label"] == "Voxtral"
        assert row["default"] is False
        assert row["controls"] == ["cfg_alpha", "flow_steps"]
        assert row["reference_audio"] is False, \
            "a page would draw a clone button on an engine that cannot clone"
        assert row["language_from_voice"] is True
        assert row["native_sample_rate"] == 24000
        assert row["cold_load_seconds"] == 63.0
        assert row["local"]["ready"] is False
        assert row["runner"]["service"] == VOXTRAL

        voices = {v["name"]: v["language"] for v in row["voices"]}
        assert len(voices) == 20
        assert voices["pt_male"] == "pt" and voices["de_female"] == "de"
        # THE FIVE WITH NO LANGUAGE IN THEIR NAMES, which is the whole reason
        # this is a pair and not a parsing rule.
        assert voices["cheerful_female"] == "en"
        assert voices["casual_male"] == "en"
        assert voices["neutral_female"] == "en"
        assert sorted(set(voices.values())) == row["languages"]

        # A CLONING ENGINE PUBLISHES null AND NOT AN EMPTY LIST. Its voices are
        # an open set that /voices owns, and `[]` would read as "this engine
        # has no voices at all".
        assert client.get("/health").json()["engines"]["chatterbox"]["voices"] is None
        assert client.get("/health").json()["engines"]["chatterbox"]["reference_audio"] is True


def test_the_voice_list_says_a_preset_engine_cannot_read_a_clip_at_all(
        build, voice_dir):
    """`reference_seconds: null` MEANS "THE LENGTH COULD NOT BE READ", which a
    page turns into "probably fine". A checkpoint with no speaker encoder is a
    different sentence, and it is about the checkpoint rather than the file.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        detail = {row["name"]: row
                  for row in client.get("/voices").json()["detail"]}
        assert VOXTRAL not in detail["gabriel"]["engines"]
        assert "no speaker encoder" in detail["gabriel"]["excluded"][VOXTRAL]
        assert "chatterbox" in detail["gabriel"]["engines"]


def test_the_record_carries_the_exact_settings_that_produced_the_sound(
        build, voice_dir):
    """"I WILL TUNE IT LATER" MEANS THE ROWS HAVE TO BE COMPARABLE.

    A tuning log whose rows cannot say which flow_steps made which audio is
    not a log. The columns are flat and additive: RECORD_KEYS is an allow-list
    and run_records.json is a pinned wire shape both halves of POST /runs
    read, so a nested `controls` object would be a migration and six columns
    are not.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir, TTS_VOXTRAL_FLOW_STEPS="16") as client:
        import app.main as main
        _runner_is_up(main, VOXTRAL)
        chosen = main._choose(model=VOXTRAL, voice="pt_female", language=None,
                              controls={"cfg_alpha": 1.5})
        job_id = main._enqueue(segments=[("olá", 0.0)], language=chosen.params.get("language"),
                               controls=chosen.params, voice=chosen.voice,
                               reference=None, spec=chosen.spec,
                               model_requested=VOXTRAL,
                               engine_reason=chosen.engine_reason)
        row = main.jobs[job_id]
        assert row["flow_steps"] == 16, "the deployment default is not on the row"
        assert row["cfg_alpha"] == 1.5, "the caller's value is not on the row"
        assert row["language"] == "pt"
        assert row["sample_rate"] == 24000
        # AND THE FIELDS THIS ENGINE CANNOT HONOUR ARE None RATHER THAN
        # SOMEBODY ELSE'S DEFAULT, so `_write_record` drops them entirely.
        assert row["exaggeration"] is None and row["temperature"] is None
        assert "flow_steps" in main.RECORD_KEYS and "cfg_alpha" in main.RECORD_KEYS
        assert "runner_settings" in main.RECORD_KEYS


def test_a_preset_job_is_speech_and_not_a_fourth_kind(build, voice_dir):
    """`kind` IS ONE EXPRESSION OFF THE CATALOGUE AND NOT A FOURTH VALUE.

    `KINDS` is wire-visible and three services post to /runs, so a fourth
    value is a fourth table that has to agree -- for a distinction the
    `engine` column already makes. An engine with no speaker encoder is not
    cloning anything, which is the same word Kokoro's rows already use.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        import app.main as main
        _runner_is_up(main, VOXTRAL)
        chosen = main._choose(model=VOXTRAL, voice="pt_male", language=None,
                              controls={})
        job_id = main._enqueue(segments=[("olá", 0.0)], language="pt",
                               controls=chosen.params, voice=chosen.voice,
                               reference=None, spec=chosen.spec)
        assert main.jobs[job_id]["kind"] == "speech"
        assert main.KINDS == ("clone", "speech", "transcribe"), \
            "a fourth kind arrived and three services have to agree about it"

        clip = main._enqueue(segments=[("hi", 0.0)], language="en", controls={},
                             voice="gabriel", reference=None,
                             spec=main.ENGINES["chatterbox"])
        assert main.jobs[clip]["kind"] == "clone"


def test_a_clip_that_shadows_a_preset_name_is_a_warning_and_never_fatal(
        build, voice_dir, caplog):
    """A SERVICE THAT WOULD NOT START OVER A FILENAME IS WORSE THAN THE
    COLLISION IT IS REFUSING.

    The pair is the key -- which is already the reasoning behind
    `min_reference_seconds` being a property of the pair -- so `pt_male` on
    the preset engine is the embedding and `pt_male` in TTS_VOICE_DIR is the
    clip, and both stay reachable.
    """
    voice_dir("pt_male", 61.0)
    with caplog.at_level("WARNING"):
        with _three(build, voice_dir) as client:
            said = " ".join(r.getMessage() for r in caplog.records)
            assert "pt_male" in said and VOXTRAL in said

            import app.main as main
            _runner_is_up(main, VOXTRAL)
            preset = main._choose(model=VOXTRAL, voice="pt_male",
                                  language=None, controls={})
            assert preset.reference is None, \
                "the clip shadowed the checkpoint's own embedding"
            clip = main._choose(model="chatterbox", voice="pt_male",
                                language=None, controls={})
            assert clip.reference is not None, \
                "the preset name shadowed a real file in TTS_VOICE_DIR"


def test_the_estimate_for_a_runner_only_engine_is_never_this_hosts_rate(
        build, voice_dir):
    """THAT NUMBER GOES OUT IN A 202 AND SIZES A PROGRESS BAR.

    `estimate_for` walked every lane and `local` has no probe, so it was
    always a candidate -- quoting a rate nobody has ever measured for an
    engine it cannot load.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        import app.main as main
        _runner_is_up(main, VOXTRAL, "chatterbox")
        lane, _ = main.dispatch.estimate_for(60.0, VOXTRAL)
        assert lane == "runner", "a lane with no implementation was quoted"
        # 60 s of speech at the measured 0.104x is about ten minutes, so the
        # estimate has to be minutes rather than the 4 minutes this CPU would
        # have been quoted at its own 0.21x.
        assert main._estimate(60 * 16, VOXTRAL) > 400


# --------------------------- the audio is the length it says it is ----------


def test_a_lane_that_resampled_and_relabelled_the_audio_is_caught(build, voice_dir):
    """THE 24k/48k TRAP, WHICH IS IN THE UPSTREAM REPO ITSELF.

    `audio_postprocess.postprocess_audio` resamples 24000 -> 48000 and
    generate.py, generate_fast.py and benchmark_all.py all write the result
    back at 24000; only serve.py gets it right. A file like that is playable,
    is exactly half the length it should be, and nothing anywhere raises.

    `frames / frame_rate` against the audio's real length is arithmetic this
    service can do on its own evidence. A factor of two is not a rounding
    error, and the frame grid is a property of the checkpoint rather than of
    the wire, so it is also the one number that says WHICH generator made this.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir):
        import app.main as main

        class _Lane:
            frames, frame_rate, runner_settings = 280, 12.5, {"group_size": 32}

        # 280 frames on a 12.5 Hz grid is 22.4 s, and that is what arrived.
        job = {"segments": [("olá", 0.0)]}
        main._record_frames(job, _Lane(), 22.4)
        assert job["frames"] == 280 and job["frame_rate"] == 12.5
        assert job["runner_settings"] == {"group_size": 32}

        # THE PAUSES ARE SUBTRACTED RATHER THAN ABSORBED INTO A WIDE
        # TOLERANCE. Silence between segments is spliced here and the frame
        # count covers only what the model generated, so the audio is longer
        # by exactly the pauses -- a tolerance big enough to swallow them
        # would be big enough to swallow a real disagreement.
        spaced = {"segments": [("olá", 1.0), ("bom dia", 0.0)]}
        main._record_frames(spaced, _Lane(), 23.4)
        assert spaced["frames"] == 280

        # And the failure it exists for: the same frames, half the audio.
        with pytest.raises(RuntimeError) as caught:
            main._record_frames({"segments": [("olá", 0.0)]}, _Lane(), 11.2)
        assert "wrong speed" in str(caught.value)

        # A lane that reports nothing writes no columns rather than wrong ones.
        class _Silent:
            frames = frame_rate = runner_settings = None

        quiet: dict = {"segments": []}
        main._record_frames(quiet, _Silent(), 22.4)
        assert "frames" not in quiet and "runner_settings" not in quiet


def test_a_stranded_job_ends_terminally_and_never_says_it_fell_back(
        build, voice_dir):
    """SO-5. THE MID-RUN HALF OF THE SPRING-OFF RULE, AS A ROW SOMEBODY READS.

    `fell_back` means a machine took this job and gave it up. Nothing took
    this one -- it was accepted while the runner was up and the runner went
    home -- and the page renders "after spring gave up" off that flag, which
    would be a story that did not happen.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        import app.main as main
        # THE JOB IS BUILT RATHER THAN SUBMITTED, and deliberately: the unit
        # here is what `_expire_stranded` LEAVES BEHIND, and putting a real job
        # in the queue would have the real chooser racing to run it -- the
        # dispatcher's own half of this is on a fake clock in test_dispatch.py.
        job_id = "3f2a1c00-0000-4000-8000-000000000000"
        main.jobs[job_id] = {
            "id": job_id, "status": "queued", "created_at": time.time(),
            "kind": "speech", "service": "tts-long", "engine": VOXTRAL,
            "host": "test", "route": "/jobs", "voice": "pt_male",
            "language": "pt", "segments": [("olá", 0.0)], "text": "olá",
            "format": "wav", "reference": None, "cancelled": False,
            "stream": None, "chunks": 1,
        }
        try:
            main._expire_stranded(job_id, 900.0)
            row = main.jobs[job_id]
        finally:
            main.jobs.pop(job_id, None)
        assert row["status"] == "failed"
        assert row["finished_at"] is not None, \
            "_sweep skips anything unfinished, so this would never be collected"
        assert not row.get("fell_back"), \
            "a machine that never had this job was blamed for giving it up"
        assert "15 minute" in row["error"], row["error"]
        assert VOXTRAL in row["error"] and "nothing here can run it" in row["error"]
        # THE ROW IS THE INDEX AND A FAILED RUN HAS NOTHING ELSE.
        assert (Path(main.OUT_DIR) / "runs" / f"{job_id}.json").is_file(), \
            "the one run worth keeping is the one that went wrong"


def test_the_installed_package_is_checked_against_the_row_that_describes_it(
        build, voice_dir):
    """THE CATALOGUE IS A FACT ABOUT A CHECKPOINT AND FACTS GO STALE.

    A build that GAINED a speaker encoder would leave this service refusing
    clips the model could now read; one that LOST a control would leave it
    forwarding fields that are silently discarded -- the exact failure the
    whole design exists to prevent, reached from the other direction.

    Said out loud and not fatal, at load, once. The model is loaded and the
    audio is real; killing a working service over a keyword argument is a worse
    trade than saying so clearly.

    THE PROMOTION HAS SINCE BEEN DECIDED, AND IT WENT THE OTHER WAY ROUND FROM
    A SEVERITY DIAL. `engines.assert_named_checkpoint` refuses the load
    outright, but only where the object is under a name the row does not spell
    AND answers to another enabled engine's row -- which is a different model
    wearing this one's name rather than this one's facts going stale. Drift,
    which is what this test plants, stays a warning.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir):
        import logging

        from app.engines import ENGINES, assert_runtime

        class _Drifted:
            """Weights that lost their speaker encoder and gained a language."""

            @staticmethod
            def generate(text, language_id=None):
                return None

        log = logging.getLogger("tts-long.engines")
        records: list[str] = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r.getMessage())
        log.addHandler(handler)
        try:
            assert_runtime(ENGINES["chatterbox"], _Drifted)
        finally:
            log.removeHandler(handler)

        said = " ".join(records)
        assert "reference clip" in said, \
            "a checkpoint that stopped taking a clip was not reported"
        assert "CATALOGUE" in said, "nobody was told which table to correct"


def test_the_language_id_is_built_from_the_fact_and_not_from_the_count(
        build, voice_dir):
    """THE PROXY THAT WOULD HAVE FAILED A JOB FOUR FRAMES DOWN.

    `len(spec.languages) > 1` was a stand-in for "generate() takes a
    language_id", and it is right about both Chatterbox rows. The first engine
    that speaks nine languages WITHOUT taking a parameter for them turns it
    into `generate_speech_fast(language_id=...)`, which is a TypeError three
    and a half minutes into somebody's job.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir):
        from app.engines import ENGINES, generate_kwargs

        assert len(ENGINES[VOXTRAL].languages) > 1, \
            "the proxy this test is about is only wrong for a multilingual row"
        kwargs = generate_kwargs(ENGINES[VOXTRAL], language="pt",
                                 controls={"flow_steps": 32, "cfg_alpha": 1.2})
        assert "language_id" not in kwargs, \
            "a parameter this checkpoint does not take was about to be passed"
        assert kwargs == {"flow_steps": 32, "cfg_alpha": 1.2}

        # And the engine that DOES take one still gets it.
        assert generate_kwargs(ENGINES["chatterbox"], language="pt",
                               controls={})["language_id"] == "pt"


def test_the_runner_is_told_the_rate_and_never_the_modules_constant(
        build, voice_dir):
    """`check_rate(SAMPLE_RATE)` WAS A CONSTANT COMPARED WITH ITSELF.

    It can only ever pass. The number that matters is the one THIS ENGINE'S
    codec produces, off the catalogue, sent to the runner in the job body and
    asserted against every segment that comes back -- because raw PCM carries
    no header to read it out of, so the assertion is the only thing that would
    catch a runner answering at another rate.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir):
        import dataclasses

        from app.engines import ENGINES
        from app.remote import RemoteSynth

        assert RemoteSynth(None, "j", spec=ENGINES[VOXTRAL])._rate == 24000
        assert RemoteSynth(None, "j")._rate == 24000, \
            "the engine that existed before there were two changed rate"

        # A ROW THAT SAYS 48000, which is EXACTLY the state the upstream
        # wrapper produces: postprocess_audio resamples 24000 -> 48000 and
        # three of the four writers then label the result 24000. The whole
        # point of reading the rate off the row is that a row saying something
        # else changes what crosses the wire and what is asserted on the way
        # back -- with the module constant, the two can never disagree and the
        # comparison can only ever pass.
        resampled = dataclasses.replace(
            ENGINES[VOXTRAL],
            facts=dataclasses.replace(ENGINES[VOXTRAL].facts,
                                      native_sample_rate=48000))
        synth = RemoteSynth(None, "j", spec=resampled)
        assert synth._rate == 48000, \
            "the rate did not come off the row at all"
        with pytest.raises(RuntimeError) as caught:
            synth._decode(b"\x00\x00\x00\x00")
        assert "48000" in str(caught.value) and "wrong pitch" in str(caught.value)


def test_the_refusals_for_instructions_and_speed_name_no_engine(build, voice_dir):
    """THEY ARE CHECKED BEFORE THE ENGINE IS KNOWN AND THAT IS DELIBERATE.

    Moving them after `_choose` would change which refusal a doubly-invalid
    request gets first, and tests assert that ordering. Deriving the sentence
    from the catalogue instead costs nothing and cannot go stale: an engine
    that HAD instruction conditioning would put it in `controls`, and no
    engine in the catalogue has a rate control at all.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        for field, body in (("instructions", {"instructions": "be cheerful"}),
                            ("speed", {"speed": 1.1})):
            refused = client.post("/v1/audio/speech",
                                  json={"voice": "gabriel", "input": "hi",
                                        **body})
            assert refused.status_code == 400, refused.text
            message = _error(refused)["message"]
            assert _error(refused)["param"] == field
            assert "Chatterbox" not in message, message
            assert "no engine on this service" in message, message
        # AND THE CONTROLS IT NAMES ARE EVERY ENGINE'S, not one engine's.
        refused = client.post("/v1/audio/speech",
                              json={"voice": "gabriel", "input": "hi",
                                    "instructions": "be cheerful"})
        message = _error(refused)["message"]
        for control in ("cfg_alpha", "flow_steps", "exaggeration"):
            assert control in message, message


def test_a_preset_engine_with_no_voice_named_is_refused_not_guessed(
        build, voice_dir):
    """NOT A FIELD DROPPED -- A FIELD INVENTED, WHICH IS THE SAME RULE.

    A cloning engine has a built-in speaker, so `default` is a real answer to
    an unasked question. Twenty equal embeddings have no such thing, and
    answering with whichever sorts first -- `ar_male` -- would hand somebody
    Arabic because they left a field out.
    """
    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir) as client:
        refused = client.post("/jobs", json={"model": VOXTRAL, "text": "hello"})
        assert refused.status_code == 400, refused.text
        said = refused.json()["detail"]
        assert "voice is required" in said
        assert "ar_male" in said and "pt_male" in said, "the list was not given"
        # And the engine that DOES have a built-in speaker still answers.
        assert client.post("/jobs", json={"model": "chatterbox",
                                          "text": "hello"}).status_code == 202


def test_a_voxtral_job_crosses_the_wire_with_exactly_its_own_fields(
        build, voice_dir):
    """THE WHOLE PATH, ONCE, WITH A RUNNER THAT ANSWERS.

    Every other test here checks one seam. This one follows a job from a
    submit that names a preset voice to a record that says which settings
    produced the sound, and asserts the three things that would each be
    invisible on their own:

      * the job body carries flow_steps and cfg_alpha and NOT `language`,
        `exaggeration`, `cfg_weight` or `temperature` -- absent, never null,
        because a null is a value to a controller that reads the key;
      * it is submitted to this engine's own service id, not the baseline
        controller, which would answer in a real voice with nothing reporting
        a fault;
      * `frames / frame_rate` from the runner agrees with the audio that
        arrived, which is the arithmetic that catches a resample labelled with
        the old rate.
    """
    import json

    from test_remote import _wait as wait_remote, runner

    voice_dir("gabriel", 61.0)
    with _three(build, voice_dir, TTS_RUNNER_HOP_S="0",
                TTS_VOXTRAL_FLOW_STEPS="32") as client:
        from app.remote import RunnerClient, RunnerConfig

        asked: list[str] = []
        submitted: list[dict] = []
        base = RunnerClient(RunnerConfig(host="runner.invalid",
                                         service="chatterbox", poll=0.0))

        def transport(method, path, body=None, content_type=None, extra=None,
                      timeout=None):
            asked.append(f"{method} {path}")
            if path == "/v1/services":
                return 200, {}, json.dumps({
                    "gpu_available": True,
                    "services": [
                        {"id": "chatterbox", "device": "gpu", "installed": True,
                         "enabled": True, "available": True},
                        {"id": VOXTRAL, "device": "gpu", "installed": True,
                         "enabled": True, "available": True,
                         "settings": {"group_size": 32, "max_frames": 2000,
                                      "fade_ms": 120, "low_pass_hz": 0}}]}).encode()
            if method == "POST" and path.endswith("/jobs"):
                submitted.append(json.loads(body))
                return 200, {}, json.dumps({"job_id": "r1"}).encode()
            if path.endswith("/jobs/r1"):
                return 200, {}, json.dumps({
                    "status": "done", "artefacts": ["r1.0.f32"],
                    # 0.1 s of audio is 1.25 frames on the 12.5 Hz grid, and
                    # the tolerance is what the grid itself costs.
                    "record": {"input_tokens": 3, "frames": 1,
                               "frame_rate": 12.5,
                               "settings": {"group_size": 32,
                                            "max_frames": 2000}}}).encode()
            if "result" in path:
                import numpy as np
                return 200, {}, np.zeros(2400, dtype="<f4").tobytes()
            return 200, {}, b"{}"

        base._request = transport

        with runner(base):
            posted = client.post("/jobs", json={"model": VOXTRAL,
                                                "voice": "pt_male",
                                                "text": "Uma linha curta.",
                                                "cfg_alpha": 1.4})
            assert posted.status_code == 202, posted.text
            assert posted.json()["engine"] == VOXTRAL
            finished = wait_remote(client, posted.json()["id"])

    assert finished["status"] == "done", finished.get("error")
    assert finished["backend"] == "runner", "there is no other lane for it"
    assert finished["runner_service"] == VOXTRAL, \
        "a preset job was submitted to the baseline controller"

    params = submitted[-1]
    assert params["flow_steps"] == 32, "the deployment default never left"
    assert params["cfg_alpha"] == 1.4, "the caller's value never left"
    assert params["sample_rate"] == 24000
    for absent in ("language", "exaggeration", "cfg_weight", "temperature"):
        assert absent not in params, \
            f"{absent} crossed the wire to a checkpoint that cannot read it"

    assert finished["flow_steps"] == 32 and finished["cfg_alpha"] == 1.4
    assert finished["language"] == "pt", "the voice's own language is not on the row"
    assert finished["kind"] == "speech"
    assert finished["frames"] == 1 and finished["frame_rate"] == 12.5
    assert finished["runner_settings"]["group_size"] == 32
