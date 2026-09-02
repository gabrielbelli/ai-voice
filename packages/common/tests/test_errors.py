"""The error envelope, and the two drifts that made it worth sharing."""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.testclient import TestClient

from voice_common.errors import ApiError, error_response, install_errors, v1_path


class Body(BaseModel):
    input: str
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


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

    return TestClient(app, raise_server_exceptions=False)


def test_type_and_code_are_keyword_only_so_they_cannot_be_swapped() -> None:
    """tts-stack takes (status, message, type_, code); tts-long takes them swapped.

    Both produce a valid-looking envelope, both are strings, and nothing
    catches the mistake — the only symptom is a client reading a code out of
    the field that should name a category. Positional calls are refused here
    so a line copied from either repo fails at the first test instead.
    """
    signature = inspect.signature(error_response)
    for name in ("type_", "code"):
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
                                         "code": "invalid_api_key"}}


def test_a_401_carries_the_challenge_rfc_9110_asks_for(client: TestClient) -> None:
    assert client.get("/v1/boom").headers["WWW-Authenticate"] == "Bearer"


def test_a_non_401_api_error_carries_no_challenge() -> None:
    response = error_response(400, "x")
    assert "WWW-Authenticate" not in response.headers
