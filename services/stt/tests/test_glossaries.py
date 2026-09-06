"""Named glossary profiles, asserted against the failures that motivated them.

Every test here is named after the thing that goes wrong without it, because
two of these failures are silent by construction: a profile that ships one
person's project names into a public image is invisible to the person it is
applied to, and a PUT that half-succeeded leaves a file that looks fine.

Nothing in this module loads a model. The pipeline's state is filled in by
hand, exactly as test_parity does it, so the suite runs in CI where the real
recognisers (460 MB and 2.9 GB) are not present.
"""

from __future__ import annotations

import io
import re
import struct
import wave
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest
from starlette.testclient import TestClient

from app import asr, boosting, openai_api, pipeline, profiles
from app.main import app

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "glossaries"


def wav(seconds: float = 0.5, rate: int = 16_000) -> bytes:
    frames = int(seconds * rate)
    tone = [int(8000 * np.sin(2 * np.pi * 220 * n / rate)) for n in range(frames)]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"".join(struct.pack("<h", s) for s in tone))
    return buffer.getvalue()


# A stand-in piece inventory: every ASCII character this file's terms are made
# of, and nothing else. It is what makes `vocabulary_problems` below a real
# answer rather than a hard-coded one — a term outside this alphabet has no
# pieces, exactly as "café ☕" has none in the shipped model, and the same
# boosting.compile_automaton decides so.
FAKE_VOCAB = {i: c for i, c in enumerate(
    " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")}
FAKE_VOCAB[len(FAKE_VOCAB)] = "<blk>"


class FakeEngine:
    """Parakeet's capability profile, INCLUDING decode-time biasing.

    The default here on purpose: Parakeet is what this service deploys. Its
    profile changed when boosting.py landed and this fake changed with it —
    `accepts_vocabulary` was False here for as long as the codebase believed a
    TDT decoder could not be biased, and a fake left behind at that value would
    have gone on asserting the old behaviour was correct.

    Both halves of a glossary now run on this engine, so "the profile was
    applied" is visible in two places: the repaired text, and the terms that
    reached the decoder.
    """

    name = "parakeet"
    accepts_vocabulary = True
    accepts_boost = True
    vocabulary_unavailable = None
    accepts_language = False
    accepts_temperature = False
    can_translate = False
    can_stream = False
    reports_language = False
    reports_segments = False
    reports_token_logprobs = True
    reports_token_ids = False

    def __init__(self, text: str = "I made a comet on the theory dashboard") -> None:
        self.text = text
        self.seen: asr.Options | None = None

    def vocabulary_problems(self, terms):  # noqa: ANN001, ANN201
        return boosting.compile_automaton(FAKE_VOCAB, terms).untokenisable

    def transcribe(self, samples, opts):  # noqa: ANN001, ANN201
        del samples
        self.seen = opts
        boosted = ()
        if opts.boost and opts.vocabulary:
            boosted = boosting.compile_automaton(
                FAKE_VOCAB, opts.vocabulary).phrases
        return asr.Recognition(text=self.text, words=(), boosted=boosted)

    def stream(self, samples, opts):  # noqa: ANN001, ANN201
        raise NotImplementedError(self.name)


class FakeWhisper(FakeEngine):
    """The engine that DOES take a vocabulary at decode time."""

    name = "whisper"
    accepts_vocabulary = True
    accepts_language = True
    accepts_temperature = True
    can_translate = True
    reports_segments = False


@pytest.fixture
def builtin(tmp_path: Path) -> Path:
    directory = tmp_path / "builtin"
    directory.mkdir()
    (directory / "tech.txt").write_text(
        "# general vocabulary\nkuber netes = Kubernetes\nPostgreSQL\n",
        encoding="utf-8")
    (directory / "dictation.txt").write_text(
        "theory dashboard = Theoria dashboard\n", encoding="utf-8")
    return directory


def serve(builtin: Path, custom: Path | None = None,
          engine: FakeEngine | None = None) -> TestClient:
    """A client over the real app, with the registry and engine injected.

    TestClient WITHOUT its context manager, which is what keeps the lifespan
    from running: `with TestClient(app)` starts it, and the lifespan loads a
    real 460 MB model.
    """
    pipeline.state.clear()
    pipeline.state["asr"] = engine or FakeEngine()
    pipeline.state["glossaries"] = profiles.Registry(
        builtin_dir=builtin, custom_dir=custom)
    pipeline.state["rules"] = []
    return TestClient(app)


@pytest.fixture
def client(builtin: Path, tmp_path: Path):  # noqa: ANN201
    """A deployment with the built-ins and NO volume mounted for custom ones.

    The custom directory is named and absent rather than unset, because that is
    what an unmounted /glossaries actually looks like from inside a container.
    """
    served = serve(builtin, tmp_path / "not-mounted")
    yield served
    pipeline.state.clear()


@pytest.fixture
def writable(builtin: Path, tmp_path: Path):  # noqa: ANN201
    custom = tmp_path / "custom"
    custom.mkdir()
    served = serve(builtin, custom)
    served.custom = custom  # type: ignore[attr-defined]
    yield served
    pipeline.state.clear()


# ── the reason this feature exists ────────────────────────────────────────────


def test_the_shipped_profiles_carry_no_personal_vocabulary() -> None:
    """glossary.txt shipped one person's project names in a PUBLIC image.

    `catalaxy = Catallaxy`, `theory dashboard = Theoria dashboard` and
    `ghost paper = Ghost Pepper` were copied into ai-voice-stt and applied to
    every request, so anyone who pulled the image got rewrites naming projects
    they have never heard of. This test is the thing that stops them coming
    back one convenient commit at a time.
    """
    personal = ("catallaxy", "catalaxy", "theoria", "theory dashboard",
                "ghost pepper", "ghost paper", "belli", "gabriel")
    for path in sorted(SHIPPED.glob("*.txt")):
        parsed = profiles.parse(path.read_text(encoding="utf-8"), force=True)
        # Terms only, not the raw file: both headers quote the `belly = Belli`
        # example, which is the argument for the rule rather than a rule.
        terms = [*parsed.replacements, *parsed.replacements.values(),
                 *parsed.hotwords]
        for term in terms:
            assert term.lower() not in personal, (
                f"{path.name} contains {term!r}: personal vocabulary belongs "
                "in a deployment-supplied profile, not in the image")


def test_the_shipped_profiles_parse_with_nothing_rejected() -> None:
    """A built-in with a bad line would load short and say so only in the log."""
    for path in sorted(SHIPPED.glob("*.txt")):
        parsed = profiles.parse(path.read_text(encoding="utf-8"), force=True)
        assert not parsed.rejected, (path.name, parsed.rejected)
        assert parsed.terms


def test_no_profile_is_applied_unless_a_request_asks(client: TestClient) -> None:
    """The old shape applied one list to everything, at a measured cost.

    A glossary whose terms do NOT occur in the audio raised WER by 12% on
    Parakeet and 28% on Whisper across 25 cells, so the default has to be
    nothing at all.
    """
    response = client.post("/transcribe",
                           files={"file": ("clip.wav", wav(), "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == body["raw"]
    assert body["repaired"] == []


def test_selecting_a_profile_repairs_the_transcript(client: TestClient) -> None:
    response = client.post("/transcribe",
                           files={"file": ("clip.wav", wav(), "audio/wav")},
                           data={"glossary": "dictation"})
    assert response.status_code == 200
    body = response.json()
    assert "Theoria dashboard" in body["text"]
    assert body["repaired"] == ["Theoria dashboard"]


def test_an_unknown_profile_is_refused_by_name(client: TestClient) -> None:
    """Ignoring it would leave a caller believing their vocabulary applied."""
    response = client.post("/transcribe",
                           files={"file": ("clip.wav", wav(), "audio/wav")},
                           data={"glossary": "nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]

    v1 = client.post("/v1/audio/transcriptions",
                     files={"file": ("clip.wav", wav(), "audio/wav")},
                     data={"model": "whisper-1", "glossary": "nope"})
    assert v1.status_code == 400
    error = v1.json()["error"]
    assert error["param"] == "glossary"
    assert "nope" in error["message"]
    assert set(error) == {"message", "type", "param", "code"}


def test_glossary_is_allowlisted_rather_than_an_unknown_field(
        client: TestClient) -> None:
    """Every extension travels beside `keywords` and `languages` or not at all."""
    assert "glossary" in openai_api.TRANSCRIPTION_FIELDS
    assert "glossary" in openai_api.TRANSLATION_FIELDS
    response = client.post("/v1/audio/transcriptions",
                           files={"file": ("clip.wav", wav(), "audio/wav")},
                           data={"model": "whisper-1", "glossary": "dictation"})
    assert response.status_code == 200
    assert "Theoria dashboard" in response.json()["text"]


def test_a_profile_reaches_both_halves_on_parakeet(client: TestClient) -> None:
    """The repair half, and — only when asked — the decoder half.

    THIS TEST USED TO ASSERT THE OPPOSITE. It was called
    `test_a_profile_is_repair_only_on_an_engine_with_no_vocabulary` and it
    pinned `seen.hotwords is None` on the grounds that a TDT decoder had
    nowhere to put a vocabulary. That was true of onnx-asr's argument list and
    false about the decoder, and an assertion is exactly how a wrong belief
    outlives the comment that explained it.

    The default is still repair only, because biasing is opt-in — so the
    profile's terms are CARRIED on the options either way and only reach the
    decoder when boost said so. Both are checked here: a request that did not
    ask gets no boosting, and the one that did gets both halves.
    """
    files = {"file": ("clip.wav", wav(), "audio/wav")}
    response = client.post("/v1/audio/transcriptions", files=files,
                           data={"model": "whisper-1", "glossary": "dictation"})
    assert response.status_code == 200
    assert "Theoria dashboard" in response.json()["text"]
    assert pipeline.state["asr"].seen.boost is False
    assert "x-boost-applied" not in response.headers

    response = client.post("/v1/audio/transcriptions", files=files,
                           data={"model": "whisper-1", "glossary": "dictation",
                                 "boost": "true"})
    assert response.status_code == 200
    assert pipeline.state["asr"].seen.boost is True
    assert "Theoria dashboard" in pipeline.state["asr"].seen.vocabulary
    assert "Theoria dashboard" in response.headers["x-boost-applied"]
    assert "Theoria dashboard" in response.json()["text"]


def test_a_profile_reaches_the_decoder_on_an_engine_that_takes_one(
        builtin: Path) -> None:
    client = serve(builtin, engine=FakeWhisper())
    try:
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("clip.wav", wav(), "audio/wav")},
            data={"model": "whisper-1", "glossary": "tech", "prompt": "Rackula"})
        assert response.status_code == 200
        hotwords = pipeline.state["asr"].seen.hotwords
        assert "Kubernetes" in hotwords
        assert "PostgreSQL" in hotwords
        # The request's own one-off term comes last and is never dropped in
        # favour of a server-side profile.
        assert hotwords.endswith("Rackula")
    finally:
        pipeline.state.clear()


# ── the four routes ───────────────────────────────────────────────────────────


def test_listing_says_where_each_profile_came_from(client: TestClient) -> None:
    body = client.get("/glossaries").json()
    names = {entry["name"]: entry for entry in body["glossaries"]}
    assert set(names) == {"tech", "dictation"}
    assert names["tech"]["source"] == "builtin"
    assert names["tech"]["writable"] is False
    assert names["tech"]["terms"] == 2
    assert body["writable"] is False
    assert body["default"] == []


def test_a_profile_can_be_read_back_as_the_text_it_was_written_from(
        writable: TestClient) -> None:
    """Editing is GET, change a line, PUT. Without `text` the comments are lost.

    A glossary's comments are where it explains why a term is a hotword rather
    than a replacement, which is exactly the knowledge that must survive a
    round trip.
    """
    source = "# why this rule exists\nghost paper = Ghost Pepper\nBelli\n"
    assert writable.put("/glossaries/mine", content=source).status_code == 201
    body = writable.get("/glossaries/mine").json()
    assert body["text"] == source
    assert body["replacements"] == {"ghost paper": "Ghost Pepper"}
    assert body["hotwords"] == ["Belli"]


def test_a_written_profile_applies_without_a_restart(
        writable: TestClient) -> None:
    """pipeline.py:108 read the glossary ONCE at startup.

    Per-request selection is meaningless while the set is frozen at boot:
    changing a term needed a new container. The registry rescans when a stat()
    says a file changed, so the request AFTER a write sees it.
    """
    assert writable.post(
        "/transcribe", files={"file": ("clip.wav", wav(), "audio/wav")},
        data={"glossary": "mine"}).status_code == 400

    writable.put("/glossaries/mine",
                 content="theory dashboard = Written Dashboard\n")
    response = writable.post(
        "/transcribe", files={"file": ("clip.wav", wav(), "audio/wav")},
        data={"glossary": "mine"})
    assert response.status_code == 200
    assert "Written Dashboard" in response.json()["text"]


def test_a_profile_edited_in_place_is_noticed(writable: TestClient) -> None:
    """A directory's mtime does not change when a file inside it is edited.

    voices.py stamps the directory only, which is right for clips that arrive
    whole. A glossary is a text file somebody opens in an editor, so the files
    are stat()ed too — without that, `vi /glossaries/mine.txt` would need a
    restart to take effect, which is the failure this whole change removes.
    """
    writable.put("/glossaries/mine", content="ghost paper = Ghost Pepper\n")
    (writable.custom / "mine.txt").write_text(  # type: ignore[attr-defined]
        "ghost paper = Something Else\n", encoding="utf-8")
    body = writable.get("/glossaries/mine").json()
    assert body["replacements"] == {"ghost paper": "Something Else"}


def test_a_deleted_profile_stops_being_selectable(writable: TestClient) -> None:
    writable.put("/glossaries/mine", content="ghost paper = Ghost Pepper\n")
    assert writable.delete("/glossaries/mine").status_code == 200
    assert writable.get("/glossaries/mine").status_code == 404
    assert writable.post(
        "/transcribe", files={"file": ("clip.wav", wav(), "audio/wav")},
        data={"glossary": "mine"}).status_code == 400


def test_deleting_a_profile_that_is_not_there_is_a_404(
        writable: TestClient) -> None:
    assert writable.delete("/glossaries/absent").status_code == 404


# ── built-ins are read-only ───────────────────────────────────────────────────


def test_a_put_over_a_built_in_is_a_conflict_not_a_silent_shadow(
        writable: TestClient) -> None:
    """A profile whose contents depend on which directory won is unreasonable-about.

    "Why is `tech` different on that box" is not a question worth creating.
    """
    response = writable.put("/glossaries/tech", content="a b = C\n")
    assert response.status_code == 409
    assert "read-only" in response.json()["detail"]
    # And the built-in is untouched.
    assert writable.get("/glossaries/tech").json()["source"] == "builtin"


def test_a_delete_of_a_built_in_is_a_conflict(writable: TestClient) -> None:
    assert writable.delete("/glossaries/dictation").status_code == 409
    assert writable.get("/glossaries/dictation").status_code == 200


def test_a_built_in_name_conflicts_before_the_volume_is_blamed(
        client: TestClient) -> None:
    """409 outranks 503, so an operator is not sent to mount a pointless volume."""
    response = client.put("/glossaries/tech", content="a b = C\n")
    assert response.status_code == 409


def test_a_file_in_the_custom_directory_cannot_shadow_a_built_in(
        builtin: Path, tmp_path: Path) -> None:
    """The 409 would be theatre if dropping a file in the volume worked."""
    custom = tmp_path / "custom"
    custom.mkdir()
    (custom / "tech.txt").write_text("a b = Shadowed\n", encoding="utf-8")
    registry = profiles.Registry(builtin_dir=builtin, custom_dir=custom)
    assert registry.get("tech").source == "builtin"
    assert "a b" not in registry.get("tech").parsed.replacements


# ── writability follows the volume ────────────────────────────────────────────


def test_writes_are_refused_when_no_volume_is_mounted(
        client: TestClient) -> None:
    """Accepting a PUT that evaporates on restart is worse than refusing it.

    503 rather than 403: nothing about the caller is being refused, and
    "forbidden" would send an operator looking for a permission they never
    configured.
    """
    response = client.put("/glossaries/mine", content="ghost paper = X\n")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "mounted" in detail

    listing = client.get("/glossaries").json()
    assert listing["writable"] is False
    assert "reason" in listing


def test_the_built_ins_still_serve_with_no_volume(client: TestClient) -> None:
    """A read-only deployment is a working deployment, not a broken one."""
    assert client.get("/glossaries/tech").status_code == 200
    response = client.post("/transcribe",
                           files={"file": ("clip.wav", wav(), "audio/wav")},
                           data={"glossary": "dictation"})
    assert response.status_code == 200


# ── the volume the write routes need ──────────────────────────────────────────
#
# The four routes above were complete, tested and unusable on the deployed
# stack for as long as compose.yaml mounted nothing at /glossaries: `grep -n
# glossar compose.yaml` returned nothing, so the directory did not exist in the
# container, writability() reported false and every PUT and DELETE answered 503
# telling the operator to mount a volume. Everything above this line passes
# with that volume missing, which is exactly why these four tests read the
# deployment files rather than the app.


def _compose() -> dict:
    # PyYAML rather than a regex over the file: this asserts the shape of a
    # mapping, and a regex would pass on a `/glossaries` that appears in a
    # comment. It is installed by services/stt/requirements.txt, which names
    # uvicorn[standard], and CI installs that file before running this suite.
    import yaml  # noqa: PLC0415

    return yaml.safe_load((REPO.parents[1] / "compose.yaml").read_text(
        encoding="utf-8"))


def _containerfile_env() -> dict[str, str]:
    """The image's ENV block, as a mapping. One ENV, backslash-continued."""
    text = (REPO / "Containerfile").read_text(encoding="utf-8")
    settings: dict[str, str] = {}
    for line in text.replace("\\\n", " ").splitlines():
        if not line.startswith("ENV "):
            continue
        for token in re.findall(r'(\w+)=("[^"]*"|\S+)', line[4:]):
            settings[token[0]] = token[1].strip('"')
    return settings


def test_the_deployed_compose_mounts_the_volume_the_write_routes_need() -> None:
    """Without this mount the whole write API answers 503 and nothing else fails.

    A write surface that refuses every write is not a failing test anywhere: it
    is a service behaving exactly as designed on a deployment that never asked
    for run-time profiles. This test is the one thing that can tell the two
    apart.
    """
    compose = _compose()
    mounts = compose["services"]["stt-stack"]["volumes"]
    targets = {entry.split(":")[1]: entry.split(":")[0] for entry in mounts}
    assert profiles.DEFAULT_CUSTOM_DIR in targets, (
        f"nothing is mounted at {profiles.DEFAULT_CUSTOM_DIR}: every PUT and "
        "DELETE on /glossaries answers 503 on this deployment")

    source = targets[profiles.DEFAULT_CUSTOM_DIR]
    # A named volume, like every other mount in that file bar the two
    # read-only host paths. A bind mount would put a host path into a
    # published file and arrive with the host directory's ownership.
    assert not source.startswith((".", "/")), (
        f"{source} is a host path; the other volumes in this file are named")
    assert source in compose["volumes"], (
        f"{source} is mounted but never declared, so compose creates an "
        "anonymous volume that a redeploy orphans")


def test_the_mounted_path_is_the_one_the_image_actually_writes_to() -> None:
    """Two files have to agree and neither imports the other.

    STT_GLOSSARY_DIR decides where custom profiles are read and written;
    compose decides where the volume lands. Drift between them is silent: the
    volume mounts, the container starts, and the write routes go on answering
    503 about a directory the operator can see in `docker inspect`.
    """
    compose = _compose()
    targets = {entry.split(":")[1] for entry
               in compose["services"]["stt-stack"]["volumes"]}
    assert _containerfile_env()["STT_GLOSSARY_DIR"] in targets


def test_the_entrypoint_takes_ownership_of_the_glossary_directory() -> None:
    """A named volume's directory is created owned by root; uvicorn is uid 1000.

    The image has no /glossaries for docker to copy ownership from, and the
    Containerfile says why it must not, so the mount arrives root-owned and
    writability()'s os.access check answers 503 "not writable by uid 1000".
    That is a mounted volume that still refuses writes, which is the confusing
    failure rather than the clear one. voice-entrypoint.sh chowns everything in
    VOICE_CHOWN_DIRS while it is still root.
    """
    chown = _containerfile_env()["VOICE_CHOWN_DIRS"].split()
    assert profiles.DEFAULT_CUSTOM_DIR in chown


def test_the_whole_write_path_round_trips(writable: TestClient) -> None:
    """Create, list, read, replace, delete, on a volume that is really there.

    The tests above each assert one refusal or one step. This is the sequence
    an operator actually performs, in order, and it is the one thing that was
    never possible on the deployed stack.
    """
    created = writable.put("/glossaries/mine",
                           content="# mine\nghost paper = Ghost Pepper\n")
    assert created.status_code == 201
    assert created.json()["created"] is True

    listing = writable.get("/glossaries").json()
    assert listing["writable"] is True
    assert "reason" not in listing
    mine = {entry["name"]: entry for entry in listing["glossaries"]}["mine"]
    # One term, not two: the comment is not a term and the intended spelling
    # becomes a hotword only when a request selects the profile.
    assert (mine["source"], mine["writable"], mine["terms"]) == ("custom", True, 1)

    read = writable.get("/glossaries/mine").json()
    assert read["text"] == "# mine\nghost paper = Ghost Pepper\n"

    replaced = writable.put("/glossaries/mine",
                            content="ghost paper = Ghost Pepper\nBelli\n")
    assert replaced.status_code == 200
    assert replaced.json()["created"] is False
    assert writable.get("/glossaries/mine").json()["hotwords"] == ["Belli"]

    assert writable.delete("/glossaries/mine").status_code == 200
    assert writable.get("/glossaries/mine").status_code == 404
    assert [entry["name"] for entry in
            writable.get("/glossaries").json()["glossaries"]] == [
        "dictation", "tech"]


def test_a_replace_is_a_200_and_a_create_is_a_201(writable: TestClient) -> None:
    """A client that cannot tell the two apart cannot warn before overwriting.

    Both answers carry `created` as well, because a status code is the half of
    the answer a proxy is allowed to rewrite.
    """
    first = writable.put("/glossaries/mine", content="ghost paper = A\n")
    second = writable.put("/glossaries/mine", content="ghost paper = B\n")
    assert (first.status_code, second.status_code) == (201, 200)
    assert (first.json()["created"], second.json()["created"]) == (True, False)


# ── writes are validated ──────────────────────────────────────────────────────


def test_an_ordinary_word_left_hand_side_is_refused_without_force(
        writable: TestClient) -> None:
    """"Belli" is heard as "belly", and `belly = Belli` eats real sentences.

    The one failure mode that damages sentences the glossary was never meant to
    touch, and the reason the shipped file's own header argues for the bare
    hotword form.
    """
    response = writable.put("/glossaries/mine", content="belly = Belli\n")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["rejected"][0]["line"] == 1
    assert "single word" in detail["rejected"][0]["reason"]
    assert not (writable.custom / "mine.txt").exists()  # type: ignore[attr-defined]


def test_force_accepts_the_rule_the_operator_meant(
        writable: TestClient) -> None:
    response = writable.put("/glossaries/mine?force=true",
                            content="belly = Belli\n")
    assert response.status_code == 201
    assert response.json()["forced"] is True
    assert writable.get("/glossaries/mine").json()["replacements"] == {
        "belly": "Belli"}


def test_a_multi_word_left_hand_side_needs_no_force(
        writable: TestClient) -> None:
    """The rule is about whether a phrase can occur innocently, not word count.

    "ghost paper" has no innocent reading; "belly" has nothing but. The check
    cannot see the middle of that range — "my sequel = MySQL" would be accepted
    and would eat a sentence — and its rejection message says so rather than
    implying a guarantee it does not offer.
    """
    assert writable.put("/glossaries/mine",
                        content="ghost paper = Ghost Pepper\n").status_code == 201


def test_a_duplicate_left_hand_side_is_a_conflict_not_last_one_wins(
        writable: TestClient) -> None:
    """Last-one-wins picks for the operator and never says which one it picked."""
    response = writable.put(
        "/glossaries/mine",
        content="ghost paper = Ghost Pepper\nghost paper = Ghostwriter\n")
    assert response.status_code == 400
    rejected = response.json()["detail"]["rejected"]
    assert rejected[0]["line"] == 2
    assert "duplicate" in rejected[0]["reason"]
    assert "line 1" in rejected[0]["reason"]


def test_nothing_is_written_when_any_line_was_rejected(
        writable: TestClient) -> None:
    """A PUT that half-succeeded is the failure this repo has hit three times.

    A 200 with a `rejected` array is trivially ignored by a script, and the
    profile that results has silently lost rules. So the good lines are not
    written either: the file on disk keeps its previous contents, or does not
    appear at all.
    """
    writable.put("/glossaries/mine", content="ghost paper = Ghost Pepper\n")
    response = writable.put(
        "/glossaries/mine",
        content="cloud code = Claude Code\nbelly = Belli\n")
    assert response.status_code == 400
    assert response.json()["detail"]["accepted"] == 1
    kept = writable.get("/glossaries/mine").json()
    assert kept["replacements"] == {"ghost paper": "Ghost Pepper"}


def test_an_oversized_glossary_is_refused_whole(writable: TestClient) -> None:
    """Every entry is a compiled regex run against every word of every transcript."""
    payload = "".join(f"heard word {n} = Term{n}\n"
                      for n in range(profiles.MAX_ENTRIES + 10))
    response = writable.put("/glossaries/mine", content=payload)
    assert response.status_code == 413
    assert str(profiles.MAX_ENTRIES) in response.json()["detail"]

    big = "# " + "x" * (profiles.MAX_BYTES + 1) + "\n"
    assert writable.put("/glossaries/big", content=big).status_code == 413


def test_a_profile_name_cannot_escape_the_directory(
        writable: TestClient) -> None:
    """The name becomes a filename, so it is validated as one rather than trusted."""
    for name in ("../etc/passwd", "a/b", ".hidden", "UPPER CASE", "", "-x"):
        assert not profiles.valid_name(name), name
        response = writable.put(f"/glossaries/{name}", content="a b = C\n")
        assert response.status_code in {400, 404, 405, 307}, (
            name, response.status_code)
    assert sorted(p.name for p in writable.custom.iterdir()) == []  # type: ignore[attr-defined]


def test_a_json_body_is_accepted_beside_a_raw_one(
        writable: TestClient) -> None:
    """curl is a fine client for a text file; an SDK would rather send JSON."""
    response = writable.put("/glossaries/mine",
                            json={"text": "belly = Belli\n", "force": True})
    assert response.status_code == 201
    assert writable.get("/glossaries/mine").json()["replacements"] == {
        "belly": "Belli"}


def test_a_json_body_without_text_is_refused_by_name(
        writable: TestClient) -> None:
    response = writable.put("/glossaries/mine", json={"terms": ["a"]})
    assert response.status_code == 400
    assert "text" in response.json()["detail"]


# ── the registry itself ───────────────────────────────────────────────────────


def test_an_env_named_file_becomes_a_profile_rather_than_a_global(
        builtin: Path, tmp_path: Path) -> None:
    """STT_GLOSSARY used to be applied to everything. It is a profile now.

    A deployment that set it keeps its vocabulary and has to opt into it per
    request, which is the whole point: the file's terms stop being charged to
    every recording that does not contain them.
    """
    operator = tmp_path / "personal.txt"
    operator.write_text("catalaxy = Catallaxy\n", encoding="utf-8")
    registry = profiles.Registry(builtin_dir=builtin, env_file=operator)
    assert registry.get("personal").source == "env"
    assert registry.select([]).rules == []
    assert registry.select(["personal"]).rules


def test_compiled_rules_are_cached_per_selection(builtin: Path) -> None:
    """The cost of a profile is the regex compilation, not the read."""
    registry = profiles.Registry(builtin_dir=builtin)
    first = registry.select(["tech"])
    assert registry.select(["tech"]) is first
    registry.reload()
    assert registry.select(["tech"]) is not first


def test_a_hotword_is_never_turned_into_a_replacement(builtin: Path) -> None:
    """The bare form biases the decoder and must never rewrite the text.

    Biasing toward "Belli" is safe; rewriting "belly" is not, and a bug that
    quietly promoted one form to the other would be invisible in the diff.
    """
    registry = profiles.Registry(builtin_dir=builtin)
    selection = registry.select(["tech"])
    assert "PostgreSQL" in (selection.hotwords or "")
    assert all(pattern.pattern != r"\bPostgreSQL\b"
               for pattern, _ in selection.rules)


# ── a request's own vocabulary ────────────────────────────────────────────────
#
# `prompt` and `keywords[]` used to be a 400 on Parakeet, which is the engine
# this service actually deploys. Every test below is named after what that
# refusal cost, or after the way honouring it could go wrong instead.


def v1(client: TestClient, **data: str):  # noqa: ANN201
    return client.post("/v1/audio/transcriptions",
                       files={"file": ("clip.wav", wav(), "audio/wav")},
                       data={"model": "whisper-1", **data})


def test_a_prompt_is_honoured_on_parakeet_rather_than_refused(
        builtin: Path) -> None:
    """The 400 this replaces was a spec field answered with an error.

    `prompt` is defined as text that guides the model and vocabulary is what it
    carries in practice, so refusing it was less compliant than honouring it,
    not more — ADR 0001 says exactly that and then cited this refusal as its
    example of saying no by name. The terms reach the repair stage — and, since
    boosting.py, the decoder too, but only when the request asks for it.
    """
    client = serve(builtin, engine=FakeEngine("I opened the theoria dashboard"))
    try:
        response = v1(client, prompt="Theoria")
        assert response.status_code == 200
        assert "Theoria dashboard" in response.json()["text"]
        # Carried, but NOT boosted: the decoder half is opt-in, so a prompt
        # that says nothing about boosting must not quietly acquire it.
        assert pipeline.state["asr"].seen.vocabulary == ("Theoria",)
        assert pipeline.state["asr"].seen.boost is False
    finally:
        pipeline.state.clear()


def test_a_prompt_term_reaches_both_halves_on_whisper(builtin: Path) -> None:
    """Whisper must not lose the decoder half to gain the repair half."""
    client = serve(builtin, engine=FakeWhisper("I opened the theoria dashboard"))
    try:
        response = v1(client, prompt="Theoria")
        assert response.status_code == 200
        assert pipeline.state["asr"].seen.hotwords == "Theoria"
        assert "Theoria dashboard" in response.json()["text"]
    finally:
        pipeline.state.clear()


def test_a_prompt_cannot_recover_a_word_the_model_never_approached(
        builtin: Path) -> None:
    """The honest limit, pinned so no comment can drift into claiming more.

    A bare term names its own spelling and no wrong one, so "entropic" is not
    reachable from a prompt saying "Anthropic". Only a `heard = intended` rule
    in a profile, or real decoder biasing, recovers that — and this engine has
    neither.
    """
    client = serve(builtin, engine=FakeEngine("entropic released a model"))
    try:
        response = v1(client, prompt="Anthropic")
        assert response.status_code == 200
        assert response.json()["text"] == "entropic released a model"
        assert "x-glossary-repaired" not in response.headers
    finally:
        pipeline.state.clear()


def test_an_all_lowercase_term_does_not_lowercase_correct_text(
        builtin: Path) -> None:
    """`sync` as a term would compile to a rule that BREAKS a right sentence.

    Every rule matches case-insensitively, so a lower-case term rewrites a
    correct sentence-initial capital down to lower case. The shipped profiles
    are full of these terms — `commit`, `nginx`, `kubectl` — so this is the
    ordinary case rather than an exotic one.
    """
    client = serve(builtin, engine=FakeEngine("Sync the files, then commit"))
    try:
        response = v1(client, prompt="sync, commit")
        assert response.status_code == 200
        assert response.json()["text"] == "Sync the files, then commit"
    finally:
        pipeline.state.clear()


def test_a_two_letter_term_does_not_rewrite_an_ordinary_word(
        builtin: Path) -> None:
    """`US` would turn "he told us" into "he told US" on every request."""
    client = serve(builtin, engine=FakeEngine("he told us the plan"))
    try:
        response = v1(client, prompt="US")
        assert response.status_code == 200
        assert response.json()["text"] == "he told us the plan"
    finally:
        pipeline.state.clear()


def test_a_profile_and_a_prompt_compose_with_the_request_last(
        builtin: Path) -> None:
    """Both halves, both sources, and the caller's own spelling winning.

    The profile rewrites "theory dashboard" to "Theoria dashboard"; the
    request's term then normalises a term the profile never mentions. A
    request that names one term must not displace the profile it also asked
    for, and must not be displaced by it.
    """
    client = serve(builtin, engine=FakeWhisper(
        "the theory dashboard runs on postgresql"))
    try:
        response = v1(client, glossary="dictation,tech", prompt="PostgreSQL")
        assert response.status_code == 200
        text = response.json()["text"]
        assert "Theoria dashboard" in text
        assert "PostgreSQL" in text
        # The profiles' terms first, the request's own last, in the decoder
        # half too — a one-off term is never dropped for a server-side profile.
        assert pipeline.state["asr"].seen.hotwords.endswith("PostgreSQL")
    finally:
        pipeline.state.clear()


def test_a_profiles_bare_hotword_still_never_rewrites_the_text(
        builtin: Path) -> None:
    """The asymmetry with a prompt is deliberate and has to stay deliberate.

    Both shipped profiles promise in their own headers that a bare term
    "biases the decoder, never rewrites the text", and deployments have read
    that. A profile's author can write `heard = intended` when they want a
    rewrite; a `prompt` cannot express one at all, which is the whole reason
    its terms get the weaker repair instead of nothing.
    """
    client = serve(builtin, engine=FakeEngine("we deployed postgresql today"))
    try:
        response = v1(client, glossary="tech")
        assert response.status_code == 200
        assert response.json()["text"] == "we deployed postgresql today"
    finally:
        pipeline.state.clear()


def test_keywords_are_read_as_the_same_list_as_a_prompt(builtin: Path) -> None:
    """The list-shaped spelling of the same field, bracketed and bare."""
    for key in ("keywords[]", "keywords"):
        client = serve(builtin, engine=FakeEngine("I opened the theoria dashboard"))
        try:
            response = v1(client, **{key: "Theoria"})
            assert response.status_code == 200, key
            assert "Theoria dashboard" in response.json()["text"], key
        finally:
            pipeline.state.clear()


def test_too_many_terms_are_refused_by_name(builtin: Path) -> None:
    """Every term is a regex run over every word; a paste is not a vocabulary.

    profiles.MAX_ENTRIES is the same ceiling a glossary file is held to, for
    the same reason, rather than a second number to keep in step with it.
    """
    client = serve(builtin)
    try:
        response = v1(client, prompt=", ".join(
            f"Term{n}" for n in range(profiles.MAX_ENTRIES + 1)))
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["param"] == "prompt"
        assert str(profiles.MAX_ENTRIES) in error["message"]
    finally:
        pipeline.state.clear()


def test_hotwords_off_drops_the_decoder_half_and_keeps_the_repair(
        builtin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """STT_HOTWORDS=0 measures the model, not the text repair.

    It exists so a benchmark can separate what the vocabulary contributes from
    what the model does, and a rewrite applied to the model's finished output
    changes neither. Dropping the repair too would make the switch mean
    something it has never claimed — and the /v1 route used to answer a prompt
    with a 400 under it, which denied a client the half that was never at
    stake.
    """
    monkeypatch.setattr(pipeline, "HOTWORDS_ENABLED", False)
    client = serve(builtin, engine=FakeWhisper("I opened the theoria dashboard"))
    try:
        response = v1(client, prompt="Theoria", glossary="tech")
        assert response.status_code == 200
        assert pipeline.state["asr"].seen.hotwords is None
        assert "Theoria dashboard" in response.json()["text"]
    finally:
        pipeline.state.clear()


# ── X-Glossary-Repaired ───────────────────────────────────────────────────────
#
# /transcribe has always returned `repaired`, and this surface could not: a
# client here could not tell a transcript the glossary had rewritten from one
# it had not. A header rather than a body key, because ADR 0001 forbids an
# extension changing the response shape and because `text`, `srt` and `vtt`
# have nowhere to put a key at all.


def test_the_repaired_header_names_what_was_rewritten(builtin: Path) -> None:
    client = serve(builtin, engine=FakeEngine("I opened the theory dashboard"))
    try:
        response = v1(client, glossary="dictation")
        assert response.status_code == 200
        assert response.headers["x-glossary-repaired"] == "Theoria dashboard"
    finally:
        pipeline.state.clear()


def test_the_repaired_header_is_absent_when_nothing_fired(
        builtin: Path) -> None:
    """Its presence has to mean something, so it is not sent empty."""
    client = serve(builtin, engine=FakeEngine("nothing here matches"))
    try:
        response = v1(client, glossary="dictation")
        assert response.status_code == 200
        assert "x-glossary-repaired" not in response.headers
    finally:
        pipeline.state.clear()


def test_a_term_the_decoder_already_spelled_right_is_not_reported(
        builtin: Path) -> None:
    """A match is not a change, and the header must report changes.

    Rules match case-insensitively, and a term rule is built from its own
    output — so it matches every time the decoder got the term right. Counting
    matches would make this header name terms nothing happened to, which is
    the same false report as a silent substitution with the sign flipped.
    """
    client = serve(builtin, engine=FakeEngine("Theoria shipped today"))
    try:
        response = v1(client, prompt="Theoria")
        assert response.status_code == 200
        assert response.json()["text"] == "Theoria shipped today"
        assert "x-glossary-repaired" not in response.headers
    finally:
        pipeline.state.clear()


def test_a_non_latin1_repaired_term_does_not_become_a_500(
        builtin: Path) -> None:
    """Starlette encodes a header value as latin-1, and terms are not latin-1.

    A Cyrillic or CJK vendor name is a perfectly ordinary glossary entry, and
    putting it in a header raw turns a working transcription into an unhandled
    UnicodeEncodeError after the work is done. Percent-encoded UTF-8 instead.
    """
    client = serve(builtin, engine=FakeEngine("we index with яндекс"))
    try:
        response = v1(client, prompt="Яндекс")
        assert response.status_code == 200
        assert "Яндекс" in response.json()["text"]
        header = response.headers["x-glossary-repaired"]
        assert header.isascii()
        assert unquote(header) == "Яндекс"
    finally:
        pipeline.state.clear()


def test_a_comma_inside_a_term_does_not_split_the_header(
        writable: TestClient) -> None:
    """The header is comma-separated, so a term's own comma has to escape.

    A reader splitting on ", " would otherwise see one term as two, and the
    second half would name a rewrite that never happened.
    """
    (writable.custom / "legal.txt").write_text(  # type: ignore[attr-defined]
        "acme inc = Acme, Inc.\n", encoding="utf-8")
    pipeline.state["asr"] = FakeEngine("filed by acme inc yesterday")
    response = v1(writable, glossary="legal")
    assert response.status_code == 200
    header = response.headers["x-glossary-repaired"]
    assert "," not in header.replace("%2C", "")
    assert unquote(header) == "Acme, Inc."


def test_the_requests_own_spelling_wins_over_the_profiles(builtin: Path) -> None:
    """Request rules run LAST, and that is what decides a disagreement.

    The `dictation` profile produces "Theoria dashboard"; this request asks for
    "Theoria Dashboard". Run the request's rules first and its term never
    matches, because the profile has not produced the term yet — the caller's
    own spelling is silently dropped in favour of a server-side profile, which
    is the one ordering rule _decode_vocabulary has always had on the other
    half.
    """
    client = serve(builtin, engine=FakeEngine("I opened the theory dashboard"))
    try:
        response = v1(client, glossary="dictation", prompt="Theoria Dashboard")
        assert response.status_code == 200
        assert "Theoria Dashboard" in response.json()["text"]
    finally:
        pipeline.state.clear()


def test_the_repaired_header_reaches_the_formats_with_no_body_key(
        builtin: Path) -> None:
    """`text`, `srt` and `vtt` have nowhere to put a key, which is the point.

    A body key would also have changed the response shape, which ADR 0001
    forbids an extension from doing. A header does neither and works on all
    five formats.
    """
    for response_format in ("json", "text", "verbose_json", "srt", "vtt"):
        client = serve(builtin, engine=FakeEngine("I opened the theory dashboard"))
        try:
            response = v1(client, glossary="dictation",
                          response_format=response_format)
            assert response.status_code == 200, response_format
            assert response.headers["x-glossary-repaired"] == "Theoria dashboard", (
                response_format)
        finally:
            pipeline.state.clear()


# ── decode-time biasing, at the route ─────────────────────────────────────────


def test_boosting_is_off_unless_the_request_asks(builtin: Path) -> None:
    """The failure: an always-on boost list, which is what the +12% rules out.

    A glossary whose terms do NOT occur in the audio raised WER by 12% on
    Parakeet across 25 cells, so a vocabulary the caller did not ask to
    have BIASED must not acquire it by arriving. Every other way of sending
    terms — prompt, keywords[], a named profile — is checked here, because the
    default has to hold on all of them and not merely on the one that was
    written first.
    """
    for data in ({"prompt": "Theoria"},
                 {"keywords[]": "Theoria"},
                 {"glossary": "dictation"}):
        client = serve(builtin, engine=FakeEngine("the theory dashboard"))
        try:
            response = v1(client, **data)
            assert response.status_code == 200
            assert pipeline.state["asr"].seen.boost is False, data
            assert "x-boost-applied" not in response.headers, data
        finally:
            pipeline.state.clear()


def test_untokenisable_phrase_is_a_400_naming_the_character(builtin: Path) -> None:
    """The failure: a caller believing their vocabulary reached the decoder.

    Same argument as profiles.UnknownProfile. A phrase whose characters have no
    pieces cannot be biased towards at all, and a caller who asked for biasing
    and silently did not get it is indistinguishable from one who did — right
    up until a transcript is wrong. Named with the offending character, because
    "one of your terms" is not something anybody can act on.
    """
    client = serve(builtin, engine=FakeEngine())
    try:
        response = v1(client, prompt="日本語", boost="true")
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["param"] == "boost"
        assert "日本語" in error["message"]
        assert "日" in error["message"]

        # …and the SAME term is accepted without boost=true, because
        # post-decode repair does not care whether the model can spell it.
        assert v1(client, prompt="日本語").status_code == 200
    finally:
        pipeline.state.clear()


def test_hotwords_off_refuses_boost_rather_than_dropping_it(
        builtin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure: STT_HOTWORDS=0 becoming a half-open door on the default engine.

    That switch exists so a benchmark can measure the model rather than the
    vocabulary. It used to be Whisper-only because Parakeet had no decoder to
    switch off; now it covers both, and a request that asked to bias must be
    told no rather than answered with an unbiased transcript that looks
    biased. pipeline.run holds the second lock on the same door.
    """
    monkeypatch.setattr(pipeline, "HOTWORDS_ENABLED", False)
    client = serve(builtin, engine=FakeEngine())
    try:
        response = v1(client, prompt="Theoria", boost="true")
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "boost"
        assert "STT_HOTWORDS" in response.json()["error"]["message"]
        # The repair half is untouched by the switch, as it always was.
        assert v1(client, prompt="Theoria").status_code == 200
    finally:
        pipeline.state.clear()


def test_hotwords_off_strips_the_vocabulary_inside_the_pipeline_too(
        builtin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The second lock, tested at the second lock.

    The route's refusal is reachable only through /v1. pipeline.run is what
    every other caller goes through, and if it cleared only `hotwords` a
    benchmark would believe it had measured the model while the default engine
    was still being biased.
    """
    monkeypatch.setattr(pipeline, "HOTWORDS_ENABLED", False)
    client = serve(builtin, engine=FakeEngine())
    try:
        pipeline.run(wav(), asr.Options(hotwords="Theoria",
                                        vocabulary=("Theoria",), boost=True))
        seen = pipeline.state["asr"].seen
        assert seen.hotwords is None
        assert seen.vocabulary == ()
        assert seen.boost is False
    finally:
        pipeline.state.clear()


def test_boost_reports_which_phrases_reached_the_decoder(builtin: Path) -> None:
    """The failure: a term dropped by a ceiling with nobody told.

    Only ONE of the three ways a term can fail to reach the decoder is a 400 —
    the one that is a fact about the model's vocabulary. The other two are
    policy ceilings, and a caller whose term went over one has to be able to
    find out. x-boost-applied names what actually got there; its absence means
    nothing did.
    """
    client = serve(builtin, engine=FakeEngine())
    try:
        response = v1(client, prompt="Anthropic, US, Theoria", boost="true")
        assert response.status_code == 200
        applied = unquote(response.headers["x-boost-applied"])
        assert "Anthropic" in applied
        assert "Theoria" in applied
        assert "US" not in applied.split(", "), (
            "a two-character term reached the decoder")
    finally:
        pipeline.state.clear()


def test_a_malformed_boost_is_refused_rather_than_read_as_false(
        builtin: Path) -> None:
    """The failure: boost=yes-please quietly meaning off.

    A value this route cannot parse is a request whose intent it does not know,
    and guessing "off" would be the accepted-and-dropped defect wearing a
    different hat.
    """
    client = serve(builtin, engine=FakeEngine())
    try:
        response = v1(client, prompt="Theoria", boost="yes-please")
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "boost"
    finally:
        pipeline.state.clear()


# ------------------------------------ an all-capitals term repairs nothing --


def test_an_uppercase_term_does_not_shout_ordinary_words():
    """`ARM` passed both filters and compiled \\barm\\b -> "ARM".

    An ordinary sentence came back "I SET the ARM on the BUS ... ALL of the
    tags are NEW". Measured over 100 real clips, a plain infrastructure
    acronym list corrupted NINE of them.
    """
    from app.glossary import apply, term_rules

    text = ("I set the arm on the bus, and then I put more RAM in it, "
            "all of the tags are new.")
    out, fired = apply(text, term_rules(["ARM", "RAM", "BUS", "SET", "ALL", "NEW"]))
    assert out == text, out
    assert fired == []


def test_an_uppercase_term_does_not_rewrite_another_language():
    """The sharpest case, in the locale this deployment exists for: `NAS`
    rewrote the Portuguese preposition "nas"."""
    from app.glossary import apply, term_rules

    text = "do primeiro-ministro nas novas notas de 100 dolares canadenses."
    out, _ = apply(text, term_rules(["NAS", "ZFS", "TrueNAS"]))
    assert out == text, out


def test_a_mixed_case_term_still_repairs():
    """The skip is about case CONTRAST, not about capitals. A term that shows
    which letters are capitals against which are not is still well defined,
    and that is the failure the default engine really has."""
    from app.glossary import apply, term_rules

    out, fired = apply("the theoria dashboard uses ghost pepper",
                       term_rules(["Theoria", "Ghost Pepper"]))
    assert out == "the Theoria dashboard uses Ghost Pepper"
    assert sorted(fired) == ["Ghost Pepper", "Theoria"]


def test_a_mixed_case_acronym_still_repairs():
    from app.glossary import apply, term_rules

    out, _ = apply("i run truenas at home", term_rules(["TrueNAS"]))
    assert out == "i run TrueNAS at home"
