"""The error envelope, and every drift that made it worth sharing.

The tests below the keyword-only one arrived with the code they guard. Three
services each vendored their own completion of this module — `param`, a 404/405
handler, an unhandled-500 handler — while voice-common was pinned by tarball
SHA, and the three copies had already disagreed with each other on the wire.
Each assertion here names the copy that got it wrong.
"""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from starlette.testclient import TestClient

from voice_common.errors import (ApiError, error_response, install_errors,
                                 v1_path)
from voice_common.errors import _union_branches


class Voice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str


class Body(BaseModel):
    # extra="forbid" so `extra_forbidden` is reachable: OpenAI's own schema
    # sets additionalProperties: false and answers "Unrecognized request
    # argument supplied", and only one of the three vendored copies said so.
    model_config = ConfigDict(extra="forbid")

    input: str
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    # A union, because pydantic reports one error PER BRANCH and the naive
    # reading of the first one named a parameter `voice.str`.
    voice: str | Voice | None = None


class Segment(BaseModel):
    text: str


class Segments(BaseModel):
    segments: list[Segment]


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    install_errors(app)

    @app.post("/v1/audio/speech")
    async def speech(req: Body) -> dict[str, str]:
        return {"input": req.input}

    @app.post("/speak")
    async def speak(req: Body) -> dict[str, str]:
        return {"input": req.input}

    @app.get("/v1/boom")
    async def boom() -> None:
        raise ApiError(401, "no key", code="invalid_api_key")

    @app.get("/v1/busy")
    async def busy() -> None:
        raise ApiError(429, "too many", code="rate_limit_exceeded",
                       headers={"Retry-After": "1"})

    @app.get("/v1/crash")
    async def crash() -> None:
        raise RuntimeError("kaboom")

    @app.get("/crash")
    async def native_crash() -> None:
        raise RuntimeError("kaboom")

    @app.get("/v1/teapot")
    async def teapot() -> None:
        raise HTTPException(status_code=503, detail="still loading")

    return TestClient(app, raise_server_exceptions=False)


def test_type_and_code_are_keyword_only_so_they_cannot_be_swapped() -> None:
    """tts-stack takes (status, message, type_, code); tts-long takes them swapped.

    Both produce a valid-looking envelope, both are strings, and nothing
    catches the mistake — the only symptom is a client reading a code out of
    the field that should name a category. Positional calls are refused here
    so a line copied from either repo fails at the first test instead.
    """
    signature = inspect.signature(error_response)
    for name in ("type_", "code", "param"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError):
        error_response(400, "x", "invalid_request_error", "invalid_value")  # type: ignore[misc]


def test_a_bad_v1_body_is_400_not_422_because_that_is_what_openai_answers(
        client: TestClient) -> None:
    """stt-stack answers 422 where the other two and the real API answer 400.

    A client written against api.openai.com branches on the status, so the one
    service claiming to imitate it most closely is the one it mishandles.
    """
    response = client.post("/v1/audio/speech", json={})
    assert response.status_code == 400


def test_the_envelope_carries_the_message_openai_python_reads(
        client: TestClient) -> None:
    """openai-python reports a useless "unknown error" off any other shape."""
    body = client.post("/v1/audio/speech", json={}).json()
    assert body["error"]["message"] == "input: Field required"
    assert body["error"]["type"] == "invalid_request_error"


def test_a_missing_field_gets_its_own_code_so_a_client_need_not_parse_prose(
        client: TestClient) -> None:
    assert client.post("/v1/audio/speech", json={}).json()["error"]["code"] \
        == "missing_required_parameter"


def test_a_bad_value_is_distinguishable_from_a_missing_one(
        client: TestClient) -> None:
    body = client.post("/v1/audio/speech",
                       json={"input": "hi", "speed": 9}).json()
    assert body["error"]["code"] == "invalid_value"
    assert body["error"]["message"].startswith("speed: ")


def test_malformed_json_names_no_field_because_loc_holds_a_byte_offset(
        client: TestClient) -> None:
    """loc is ("body", 12) for json_invalid, and naming a field "12" is worse
    than naming none."""
    response = client.post("/v1/audio/speech", content=b"{not json",
                           headers={"content-type": "application/json"})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "request body is not valid JSON"


def test_a_list_index_survives_into_the_field_path() -> None:
    """segments.0.text, not "text" — the caller has to know which one."""
    app = FastAPI()
    install_errors(app)

    @app.post("/v1/speak")
    async def speak(req: Segments) -> dict[str, int]:
        return {"n": len(req.segments)}

    client = TestClient(app, raise_server_exceptions=False)
    body = client.post("/v1/speak", json={"segments": [{}]}).json()
    assert body["error"]["message"].startswith("segments.0.text: ")


def test_native_routes_keep_fastapis_own_body(client: TestClient) -> None:
    """Something out there already parses `detail` on the non-/v1 routes.

    /v1 is a compatibility boundary and the only place the shape is reshaped;
    changing the native routes to tidy up would break clients that did nothing.
    """
    response = client.post("/speak", json={})
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(("path", "expected"),
                         [("/v1", True), ("/v1/", True),
                          ("/v1/audio/speech", True),
                          ("/v1beta/thing", False), ("/speak", False),
                          ("/health", False)])
def test_the_v1_guard_does_not_capture_a_neighbouring_prefix(
        path: str, expected: bool) -> None:
    """startswith("/v1") also matches /v1beta, and reshaping an unrelated
    route's errors is not noticed until a client notices it."""
    assert v1_path(path) is expected


def test_api_error_renders_through_the_same_function_as_the_middleware(
        client: TestClient) -> None:
    """The raise path and the return path cannot produce different bodies.

    Four incompatible copies of this envelope exist today, one of them a
    hardcoded 401 in tts-long/app/auth.py. One function, two entry points.
    """
    response = client.get("/v1/boom")
    assert response.status_code == 401
    assert response.json() == {"error": {"message": "no key",
                                         "type": "invalid_request_error",
                                         "param": None,
                                         "code": "invalid_api_key"}}


def test_a_401_carries_the_challenge_rfc_9110_asks_for(client: TestClient) -> None:
    assert client.get("/v1/boom").headers["WWW-Authenticate"] == "Bearer"


def test_a_non_401_api_error_carries_no_challenge() -> None:
    response = error_response(400, "x")
    assert "WWW-Authenticate" not in response.headers


# ── `param`, the field this module used to omit ───────────────────────────────

def test_every_v1_error_carries_param_it_used_to_be_absent_entirely(
        client: TestClient) -> None:
    """OpenAI's `Error` requires all four, with `param` required-but-NULLABLE.

    This module built three keys, so `param` was missing from every error every
    service had ever produced. A client reading `err.param` to learn which
    field it got wrong read None whether or not the server knew the answer —
    and it usually did.
    """
    for response in (client.post("/v1/audio/speech", json={}),
                     client.post("/v1/audio/speech", json={"input": "hi",
                                                           "speed": 9}),
                     client.post("/v1/audio/speech", content=b"{not json",
                                 headers={"content-type": "application/json"}),
                     client.post("/v1/nonexistent", json={}),
                     client.get("/v1/audio/speech"),
                     client.get("/v1/boom"),
                     client.get("/v1/crash")):
        assert response.status_code >= 400, response.text
        assert set(response.json()["error"]) == {"message", "type", "param",
                                                 "code"}, response.text


def test_param_names_the_field_the_message_already_knew_about(
        client: TestClient) -> None:
    """The information was available and thrown away.

    The validation handler computed a field name to put in the message string
    and then built an envelope without it.
    """
    body = client.post("/v1/audio/speech", json={}).json()
    assert body["error"]["param"] == "input"
    body = client.post("/v1/audio/speech",
                       json={"input": "hi", "speed": 9}).json()
    assert body["error"]["param"] == "speed"


def test_param_is_null_and_not_absent_when_no_single_field_is_at_fault(
        client: TestClient) -> None:
    """Required-but-nullable: a generated client may read the key rather than
    test for its presence."""
    body = client.post("/v1/audio/speech", content=b"{not json",
                       headers={"content-type": "application/json"}).json()
    assert "param" in body["error"] and body["error"]["param"] is None


def test_an_unknown_field_is_named_rather_than_dropped(
        client: TestClient) -> None:
    """`{"stream": true}` and `{"totally_unknown_field": 1}` both returned 200.

    `stream` is the TRANSCRIPTION-side switch, so a client that sent it
    expecting a stream got a buffered file and no way to tell. Only one of the
    three vendored copies mapped `extra_forbidden`; the message is OpenAI's own
    wording, because a client that greps its logs for that sentence has met the
    real API.
    """
    body = client.post("/v1/audio/speech",
                       json={"input": "hi", "wibble": 1}).json()
    assert body["error"]["code"] == "unknown_parameter"
    assert body["error"]["param"] == "wibble"
    assert body["error"]["message"] == \
        "Unrecognized request argument supplied: wibble"


# ── 404 and 405 under /v1, which used to escape the envelope ─────────────────

def test_an_unknown_v1_url_is_an_envelope_it_used_to_leak_fastapis_detail(
        client: TestClient) -> None:
    """`{"detail":"Not Found"}` is a shape openai-python reads no message off,
    so it reported a bare "unknown error" for a mistyped path."""
    response = client.post("/v1/nonexistent", json={})
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["message"] == "Invalid URL (POST /v1/nonexistent)"
    assert error["code"] == "unknown_url"


def test_a_wrong_method_under_v1_is_an_envelope_and_says_so_in_code(
        client: TestClient) -> None:
    """`GET /v1/audio/speech` answered `{"detail":"Method Not Allowed"}`.

    The message is OpenAI's own wording for an unrouted request, which one of
    the three copies spelt "Invalid method (…)" instead. Because 404 and 405
    then share a message SHAPE, `code` is the only machine-readable thing
    telling them apart, and it has to differ.
    """
    response = client.get("/v1/audio/speech")
    assert response.status_code == 405
    error = response.json()["error"]
    assert error["message"] == "Invalid URL (GET /v1/audio/speech)"
    assert error["code"] == "method_not_allowed"
    assert error["code"] != client.post("/v1/nope",
                                        json={}).json()["error"]["code"]


def test_a_405_keeps_the_allow_header_one_vendored_copy_dropped_it(
        client: TestClient) -> None:
    """RFC 9110 §15.5.6 requires Allow on every 405.

    Starlette sets it and one of the three copies rebuilt the response without
    `exc.headers`, so its 405 arrived with no Allow at all.
    """
    assert client.get("/v1/audio/speech").headers["allow"] == "POST"


def test_a_5xx_http_exception_is_typed_server_error_not_invalid_request(
        client: TestClient) -> None:
    """Two of the three copies left it `invalid_request_error`.

    That tells a client to go and fix a request that was fine, at the moment
    the server is the thing that broke.
    """
    response = client.get("/v1/teapot")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
    assert response.json()["error"]["message"] == "still loading"


def test_a_native_404_still_carries_fastapis_detail(client: TestClient) -> None:
    """The 404 handler stops at /v1 like every other one here."""
    assert client.post("/nonexistent", json={}).json() == {"detail": "Not Found"}


# ── the unhandled 500, which used to be plain text ───────────────────────────

def test_an_unhandled_v1_error_is_json_it_used_to_be_a_plain_text_500(
        client: TestClient) -> None:
    """Same silence as the 404, at the moment the caller most needs a sentence.

    Only one of the three vendored copies registered this handler at all.
    """
    response = client.get("/v1/crash")
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "server_error"
    assert error["code"] == "internal_error"
    assert "kaboom" in error["message"]


def test_a_native_unhandled_error_is_still_starlettes_plain_text(
        client: TestClient) -> None:
    """Byte for byte what Starlette's own default sends, because something on
    the native side may already be matching on it."""
    response = client.get("/crash")
    assert response.status_code == 500
    assert response.text == "Internal Server Error"
    assert response.headers["content-type"].startswith("text/plain")


# ── headers on a raised ApiError ─────────────────────────────────────────────

def test_a_raised_api_error_carries_its_retry_after_header(
        client: TestClient) -> None:
    """Retry-After on a 429 is the header an OpenAI client's backoff reads.

    Only one of the three copies gave ApiError a `headers` argument, and
    services/stt raises exactly this from `_busy()`.
    """
    response = client.get("/v1/busy")
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.json()["error"]["code"] == "rate_limit_exceeded"


# ── pydantic unions, which only one copy read correctly ──────────────────────

def test_a_malformed_union_names_the_field_not_the_branch_tag(
        client: TestClient) -> None:
    """`param` has to be a parameter a client can act on.

    pydantic reports a union one error PER BRANCH, tagging each with the
    branch's type name, so reading the first error the way every other field is
    read produced `param: "voice.str"` — a name that appears nowhere in
    OpenAI's schema or in the caller's request. Both branches' complaints are
    kept: a caller who mistyped a key inside the object needs to hear about the
    object form, not only about the string one.
    """
    response = client.post("/v1/audio/speech",
                           json={"input": "hi",
                                 "voice": {"id": "nova", "typo": 1}})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "voice"
    assert "typo" in error["message"]
    assert "valid string" in error["message"]


def test_a_union_collapse_does_not_swallow_two_ordinary_field_errors() -> None:
    """Two errors on two ordinary sub-fields are not a union and stay apart.

    The guard on the collapse above. `segments.0.text` and
    `segments.0.pause_after` share a prefix and diverge at the same depth that
    union branches do; only the branch TAG tells the two cases apart, so this
    asserts the tag test rather than the shape.

    Moved here from tts-long/tests/test_openai_api.py with the function it
    tests.
    """
    union = [{"loc": ("body", "voice", "str"), "msg": "Input should be a "
                                                      "valid string"},
             {"loc": ("body", "voice", "CustomVoice", "typo"),
              "msg": "Extra inputs are not permitted"}]
    assert _union_branches(union) == (
        "voice", ["Input should be a valid string",
                  "typo: Extra inputs are not permitted"])

    ordinary = [{"loc": ("body", "segments", 0, "text"), "msg": "a"},
                {"loc": ("body", "segments", 0, "pause_after"), "msg": "b"}]
    assert _union_branches(ordinary) is None
    assert _union_branches([{"loc": ("body", "input"), "msg": "c"}]) is None
