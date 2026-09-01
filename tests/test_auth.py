"""Every test here is named after the failure it prevents.

The three defects at the top were found in one adversarial review round, in
three copies of the same file that had drifted apart. Two of the three are
still live in the copies that never got the fix. If one of these tests ever
goes red, the corresponding service is back to the behaviour that was already
paid for once.
"""

from __future__ import annotations

import hmac
import logging

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from voice_common import auth
from voice_common.auth import ConfigurationError

EXOTIC_KEY = "clé-très-secrète"


def build(keys: str | None, monkeypatch: pytest.MonkeyPatch,
          **kwargs: object) -> TestClient:
    """An app with one authenticated route, plus whatever install() defaults to."""
    if keys is None:
        monkeypatch.delenv("TTS_API_KEYS", raising=False)
    else:
        monkeypatch.setenv("TTS_API_KEYS", keys)

    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/thing")
    async def thing() -> dict[str, bool]:
        return {"ok": True}

    auth.install(app, "TTS_API_KEYS", **kwargs)  # type: ignore[arg-type]
    return TestClient(app, raise_server_exceptions=False)


# --------------------------------------------------------------------------
# Defect 1: the correct key was rejected as the wrong one (tts-stack)
# --------------------------------------------------------------------------

def test_a_non_ascii_key_authenticates_it_used_to_be_rejected_as_incorrect(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Starlette decodes header bytes latin-1; the old code compared UTF-8.

    The bytes therefore never matched for any key containing an accent, and
    the CORRECT key came back as "Incorrect API key provided". Still live at
    stt-stack/app/auth.py:115 and tts-long/app/auth.py:68.
    """
    client = build(EXOTIC_KEY, monkeypatch)
    # Sent as bytes, because that is what reaches the socket. httpx refuses to
    # guess an encoding for a non-ASCII header value, and so does every other
    # careful client: UTF-8 is what they choose when they choose. Starlette
    # then decodes those bytes latin-1, which is the gap the defect lived in.
    response = client.get(
        "/v1/thing",
        headers={"Authorization": f"Bearer {EXOTIC_KEY}".encode("utf-8")})
    assert response.status_code == 200


def test_a_non_ascii_key_is_compared_on_wire_bytes_not_on_the_decoded_string(
        ) -> None:
    """The unit form of the same thing, without a client in the way."""
    # What starlette hands the middleware for a client that sent UTF-8.
    on_the_wire = EXOTIC_KEY.encode("utf-8").decode("latin-1")
    assert on_the_wire != EXOTIC_KEY  # the bug lived in this gap
    assert auth.authorised(f"Bearer {on_the_wire}", [EXOTIC_KEY])


def test_a_key_that_cannot_be_latin_1_encoded_is_a_401_not_a_500() -> None:
    """Unreachable from a header, reachable from a direct call. Not a traceback."""
    assert not auth.authorised("Bearer ☃", ["snowman"])


# --------------------------------------------------------------------------
# Defect 2: GET /health/ returned 401 once keys were on (tts-stack)
# --------------------------------------------------------------------------

def test_health_with_a_trailing_slash_is_exempt_it_used_to_return_401(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The check runs before routing, so FastAPI's 307 to /health never happens.

    A probe written with the trailing slash worked until keys were configured,
    then failed permanently and the orchestrator restarted a healthy service.
    Still live at tts-long/app/auth.py:103, which matches OPEN_PATHS exactly.
    """
    client = build("k1", monkeypatch)
    assert client.get("/health/").status_code == 200
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(("given", "expected"),
                         [("/health", "/health"), ("/health/", "/health"),
                          ("/health//", "/health"), ("/", "/"), ("//", "/")])
def test_path_normalisation_strips_trailing_slashes_but_keeps_root(
        given: str, expected: str) -> None:
    assert auth.normalise_path(given) == expected


# --------------------------------------------------------------------------
# Defect 3: a set-but-degenerate value disabled auth and logged it as unset
# --------------------------------------------------------------------------

@pytest.mark.parametrize("degenerate", ["", ",", "  ", ",,", ",, ,", " , "])
def test_a_set_but_keyless_value_refuses_to_start_it_used_to_disable_auth(
        degenerate: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`-e TTS_API_KEYS=$SECRET` with SECRET unset is how this is reached.

    The operator asked for authentication and got a service open to anyone.
    Present in tts-stack and tts-long; stt-stack alone got this right.
    """
    monkeypatch.setenv("TTS_API_KEYS", degenerate)
    with pytest.raises(ConfigurationError):
        auth.load_keys("TTS_API_KEYS")


def test_the_refusal_is_a_systemexit_so_nothing_can_swallow_it(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A broad `except Exception` in a startup path must not catch this."""
    monkeypatch.setenv("TTS_API_KEYS", ",")
    try:
        auth.load_keys("TTS_API_KEYS")
    except Exception:  # noqa: BLE001 - that is exactly what is being tested
        pytest.fail("a refusal to enforce authentication was caught as Exception")
    except SystemExit as exc:
        assert "names no key" in str(exc)


def test_a_degenerate_value_is_never_reported_as_unset(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The second half of the same defect, and the nastier half.

    The old code logged "TTS_API_KEYS is unset" for a variable that was set to
    ',', so an operator chasing an open service was told their configuration
    was not there. It was there and being ignored.
    """
    monkeypatch.setenv("TTS_API_KEYS", ",")
    with caplog.at_level(logging.WARNING, logger="voice_common.auth"):
        with pytest.raises(ConfigurationError) as caught:
            auth.load_keys("TTS_API_KEYS")
    assert "unset" not in "".join(caplog.messages).lower()
    assert repr(",") in str(caught.value)  # the value is quoted back


def test_an_unset_variable_disables_auth_and_says_so_loudly(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Open is allowed because someone chose it. Quietly open is not.

    These services already run on a LAN with callers that have no key, so
    refusing to boot would break the deployment it was meant to protect.
    """
    with caplog.at_level(logging.WARNING, logger="voice_common.auth"):
        client = build(None, monkeypatch)
    assert client.get("/v1/thing").status_code == 200
    assert any("DISABLED" in message for message in caplog.messages)


# --------------------------------------------------------------------------
# The rest of the contract
# --------------------------------------------------------------------------

def test_no_key_is_a_401_carrying_the_challenge_and_the_openai_envelope(
        monkeypatch: pytest.MonkeyPatch) -> None:
    client = build("k1", monkeypatch)
    response = client.get("/v1/thing")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    body = response.json()["error"]
    assert body["code"] == "invalid_api_key"
    assert "Authorization: Bearer" in body["message"]


@pytest.mark.parametrize("header", ["", "Bearer", "Bearer ", "Basic k1",
                                    "k1", "bearer  ", "Token k1"])
def test_a_header_that_is_not_a_bearer_key_is_rejected(
        header: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = build("k1", monkeypatch)
    assert client.get("/v1/thing",
                      headers={"Authorization": header}).status_code == 401


def test_the_scheme_is_matched_case_insensitively(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC 9110 says the scheme is case-insensitive, and clients vary."""
    client = build("k1", monkeypatch)
    assert client.get("/v1/thing",
                      headers={"Authorization": "bearer k1"}).status_code == 200


def test_docs_and_openapi_need_a_key_a_schema_dump_is_a_map_of_the_service(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI's own doc routes are Starlette routes no router dependency reaches.

    That is one of the two reasons this is middleware rather than a
    per-route dependency; the other is the route somebody adds next year.
    """
    client = build("k1", monkeypatch)
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 401, path


def test_a_route_added_after_install_is_covered_without_being_asked(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole argument for middleware over Depends(require_key)."""
    monkeypatch.setenv("TTS_API_KEYS", "k1")
    app = FastAPI()
    auth.install(app, "TTS_API_KEYS")

    @app.get("/added/later")
    async def later() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/added/later").status_code == 401
    assert client.get("/added/later",
                      headers={"Authorization": "Bearer k1"}).status_code == 200


def test_every_key_is_compared_even_after_one_has_matched(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuiting times how far down the list a valid key sits.

    That is a slower version of the leak compare_digest is here to close, so
    the loop accumulates with |= and never breaks.
    """
    calls = []
    real = hmac.compare_digest

    def counting(a: bytes, b: bytes) -> bool:
        calls.append(b)
        return real(a, b)

    monkeypatch.setattr(auth.hmac, "compare_digest", counting)
    assert auth.authorised("Bearer first", ["first", "second", "third"])
    assert len(calls) == 3


def test_compare_digest_is_used_rather_than_equality() -> None:
    """== returns at the first differing byte and leaks the matching prefix."""
    source = auth.authorised.__code__.co_names
    assert "compare_digest" in source


def test_keys_are_stripped_so_a_spaced_list_works_as_written(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP strips a field value's trailing whitespace anyway.

    A key with a space on the end could not be presented even if it were
    configured, so stripping loses nothing and makes `k1, k2` mean what it
    looks like.
    """
    monkeypatch.setenv("TTS_API_KEYS", " k1 , k2,,  k3  ")
    assert auth.load_keys("TTS_API_KEYS") == ("k1", "k2", "k3")


def test_any_of_the_configured_keys_is_accepted(
        monkeypatch: pytest.MonkeyPatch) -> None:
    client = build("k1,k2,k3", monkeypatch)
    for key in ("k1", "k2", "k3"):
        assert client.get("/v1/thing",
                          headers={"Authorization": f"Bearer {key}"}).status_code == 200
    assert client.get("/v1/thing",
                      headers={"Authorization": "Bearer k4"}).status_code == 401


def test_a_prefix_of_a_valid_key_is_rejected(
        monkeypatch: pytest.MonkeyPatch) -> None:
    client = build("supersecret", monkeypatch)
    assert client.get("/v1/thing",
                      headers={"Authorization": "Bearer super"}).status_code == 401


def test_the_env_var_name_is_the_only_thing_that_varies_between_services(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """stt-stack keeps STT_API_KEYS and tts-* keep TTS_API_KEYS.

    Renaming an operator-visible variable to adopt a shared package would be a
    breaking change for a deployment that has done nothing wrong.
    """
    monkeypatch.setenv("STT_API_KEYS", "k1")
    monkeypatch.delenv("TTS_API_KEYS", raising=False)
    assert auth.load_keys("STT_API_KEYS") == ("k1",)
    assert auth.load_keys("TTS_API_KEYS") == ()


def test_a_non_ascii_key_warns_at_startup_because_http_does_not_promise_utf_8(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """It authenticates now, but only for clients that send UTF-8.

    Complained about at load rather than at the request that fails, which is
    where an operator can still act on it.
    """
    with caplog.at_level(logging.WARNING, logger="voice_common.auth"):
        build(EXOTIC_KEY, monkeypatch)
    assert any("non-ASCII" in message for message in caplog.messages)


def test_exemptions_may_be_added_after_install(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """install() and install_health() must work in either order.

    The middleware holds the set itself rather than a copy, which is what
    makes the ordering irrelevant.
    """
    monkeypatch.setenv("TTS_API_KEYS", "k1")
    app = FastAPI()
    auth.install(app, "TTS_API_KEYS", extra_public_paths=())

    @app.get("/probe")
    async def probe() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    assert client.get("/probe").status_code == 401
    auth.exempt(app, "/probe/")
    assert client.get("/probe").status_code == 200
