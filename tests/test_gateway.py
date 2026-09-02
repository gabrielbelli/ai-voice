"""What the gateway promises, asserted against mock backends.

One test per rule in the contract, named for the rule rather than for the
function, because the rules are the thing that must not drift. Where a test
exists to prevent a specific failure, the failure is in the docstring.
"""

from __future__ import annotations

import json

import httpx
import pytest
from conftest import MockBackend, Slow, Unreachable, gateway, reload_gateway

SPEECH = "/v1/audio/speech"


def body(**kwargs) -> bytes:
    return json.dumps({"input": "Here is the change to make.", **kwargs}).encode()


# ------------------------------------------------------------------ routing --


@pytest.mark.parametrize("model", ["kokoro", "tts-1", "tts-1-hd",
                                   "gpt-4o-mini-tts", None, "banana",
                                   "gpt-9-turbo-audio", ""])
async def test_everything_but_the_two_long_names_goes_fast(monkeypatch, backends, model):
    """An unknown model goes fast, and is not a 400.

    The two wrong answers are asymmetric: Kokoro on long-form costs some
    quality, Chatterbox on an ordinary request turns a 17-second call into a
    ten-minute job nobody asked for. Rejecting unknown names would also break
    a client that sends whatever string its UI was left holding.
    """
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        payload = body() if model is None else body(model=model)
        response = await client.post(SPEECH, content=payload)

    assert response.status_code == 200
    assert len(tts.seen) == 1 and not long.seen


@pytest.mark.parametrize("model", ["chatterbox", "tts-long", "Chatterbox",
                                   " chatterbox "])
async def test_only_the_two_opt_in_names_reach_the_long_backend(monkeypatch, backends, model):
    """Case and surrounding whitespace do not decide a nine-minute difference."""
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body(model=model))

    assert response.status_code == 200
    assert len(long.seen) == 1 and not tts.seen


async def test_transcription_routes_reach_the_only_stt_backend(monkeypatch, backends):
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        for path in ("/v1/audio/transcriptions", "/transcribe"):
            assert (await client.post(path, content=b"RIFF....")).status_code == 200

    assert [r["path"] for r in stt.seen] == ["/v1/audio/transcriptions", "/transcribe"]
    assert not tts.seen and not long.seen


async def test_native_fast_routes_reach_tts_stack(monkeypatch, backends):
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        assert (await client.post("/speak", content=body())).status_code == 200
        assert (await client.get("/voices")).status_code == 200

    assert [r["path"] for r in tts.seen] == ["/speak", "/voices"]


async def test_job_routes_mount_flat_and_are_not_rewritten(monkeypatch, backends):
    """The path reaches tts-long verbatim, which is what keeps its own URLs valid.

    tts-long answers a 202 with `Location: /jobs/{id}` and a body field
    `audio_url: /jobs/{id}/audio`, both relative to its own root. Any prefix
    here would force the gateway to rewrite a header and a JSON field, and
    that rule rots the first time the backend adds a field.
    """
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        await client.post("/jobs", content=body())
        await client.get("/jobs")
        await client.get("/jobs/abc-123")
        await client.get("/jobs/abc-123/audio?format=wav")

    assert [r["path"] for r in long.seen] == [
        "/jobs", "/jobs", "/jobs/abc-123", "/jobs/abc-123/audio"]
    assert long.seen[-1]["query"] == "format=wav"


async def test_advertised_models_route_where_the_list_says_they_do(monkeypatch, backends):
    """GET /v1/models is the routing table, so it must not drift from it.

    The list names an `owned_by` per model. This sends every advertised TTS
    name through the router and checks it lands on the backend the list
    claims — the one failure that would make the discoverable contract a lie.
    """
    stt, tts, long = backends
    from app.openai_api import MODELS

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        for entry in MODELS:
            if entry["owned_by"] == "stt-stack":
                continue  # no decision to make: one STT backend
            await client.post(SPEECH, content=body(model=entry["id"]))
            landed = "tts-stack" if tts.seen else "tts-long"
            assert landed == entry["owned_by"], entry["id"]
            tts.seen.clear()
            long.seen.clear()


async def test_models_is_answered_without_touching_a_backend(monkeypatch, backends):
    """It must answer while a backend is restarting — that is when it is needed."""
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.get("/v1/models")

    assert response.status_code == 200
    ids = [m["id"] for m in response.json()["data"]]
    assert {"kokoro", "chatterbox", "whisper-1"} <= set(ids)
    assert not stt.seen and not tts.seen and not long.seen


# ------------------------------------------------------------- pass-through --


async def test_the_202_deviation_passes_through_untouched(monkeypatch, backends):
    """tts-long's honest 202 is forwarded, header and body byte-for-byte.

    The gateway does not invent this and must not re-implement it: the backend
    already decides when a wait is honest. Rewriting the body would break the
    audio_url the caller needs.
    """
    stt, tts, long = backends
    payload = json.dumps({"id": "job-1", "status": "queued", "queued_ahead": 0,
                          "estimated_seconds": 580,
                          "audio_url": "/jobs/job-1/audio"}).encode()
    long.reply = lambda record: (202, {"content-type": "application/json",
                                       "location": "/jobs/job-1"}, payload)

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body(model="chatterbox"))

    assert response.status_code == 202
    assert response.headers["location"] == "/jobs/job-1"
    assert response.content == payload
    assert response.headers["content-type"] == "application/json"


@pytest.mark.parametrize("status,code", [(400, "invalid_value"),
                                         (422, "missing_required_parameter"),
                                         (503, "model_loading")])
async def test_a_backend_envelope_is_never_rewrapped(monkeypatch, backends, status, code):
    """stt-stack answers 422 where the TTS services answer 400. Both pass as-is.

    Re-wrapping destroys the `code` the client switches on, and normalising
    the difference would make the gateway a second, lying source of truth.
    """
    stt, tts, long = backends
    payload = json.dumps({"error": {"message": "no", "type": "invalid_request_error",
                                    "code": code}}).encode()
    tts.reply = lambda record: (status, {"content-type": "application/json"}, payload)

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.status_code == status
    assert response.content == payload
    assert response.json()["error"]["code"] == code


async def test_the_clients_key_is_stripped_and_not_replaced(monkeypatch, backends):
    """Forwarding it would copy the secret into three more log streams.

    The backends run with their own keys unset, so a relayed token buys
    nothing at all.
    """
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long,
                       api_keys="sk-test") as (client, _):
        response = await client.post(SPEECH, content=body(),
                                     headers={"authorization": "Bearer sk-test"})

    assert response.status_code == 200
    assert "authorization" not in tts.last["headers"]


async def test_response_headers_survive_except_the_hop_by_hop_ones(monkeypatch, backends):
    """X-Realtime-Factor is how the operator sees the box keeping up.

    `connection` and `transfer-encoding` describe one connection and must not
    be copied onto the next.
    """
    stt, tts, long = backends
    tts.reply = lambda record: (200, {"content-type": "audio/mpeg",
                                      "x-realtime-factor": "1.4",
                                      "connection": "keep-alive",
                                      "transfer-encoding": "chunked"}, b"ID3audio")

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.content == b"ID3audio"
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["x-realtime-factor"] == "1.4"
    assert "connection" not in response.headers
    assert "transfer-encoding" not in response.headers


# ----------------------------------------------------------------- streaming --


async def test_an_upload_is_forwarded_chunk_by_chunk(monkeypatch, backends):
    """An hour of wav is 100 MB+; buffering it here doubles resident memory.

    The mock counts one ASGI message per chunk it received, so more than one
    chunk out of a five-chunk upload proves the body was passed through rather
    than collected first.
    """
    stt, tts, long = backends

    async def upload():
        for _ in range(5):
            yield b"x" * 65536

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post("/transcribe", content=upload())

    assert response.status_code == 200
    assert stt.last["chunks"] > 1
    assert stt.last["body"] == b"x" * 65536 * 5


async def test_the_speech_body_is_buffered_but_forwarded_unchanged(monkeypatch, backends):
    """This is the one body the gateway reads: `model` has to come out of it.

    It is text measured in kilobytes, unlike the audio uploads. What reaches
    the backend is still the client's own bytes, not a re-serialisation —
    re-encoding would drop any field the gateway does not know about.
    """
    stt, tts, long = backends
    payload = b'{"model":"kokoro","input":"hi","instructions":"whisper it"}'

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        await client.post(SPEECH, content=payload)

    assert tts.last["body"] == payload
    assert tts.last["headers"]["content-length"] == str(len(payload))


# ------------------------------------------------------------------ failures --


async def test_a_container_that_is_down_is_503_and_names_itself(monkeypatch, backends):
    """503, not 502: 502 claims the upstream answered badly and it never answered.

    With three backends behind one URL, "upstream failed" is unactionable.
    """
    stt, tts, _ = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=Unreachable()) as (client, _):
        response = await client.post(SPEECH, content=body(model="chatterbox"))

    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    error = response.json()["error"]
    assert error["code"] == "backend_unavailable"
    assert "tts-long" in error["message"]


async def test_a_timeout_names_the_rate_and_the_way_out(monkeypatch, backends):
    """A 504 whose body does not name the alternative just gets retried."""
    stt, _, long = backends
    async with gateway(monkeypatch, stt=stt, tts=Slow(), long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.status_code == 504
    error = response.json()["error"]
    assert error["code"] == "backend_timeout"
    assert "1.2-1.5x realtime" in error["message"]
    assert "chatterbox" in error["message"]
    assert "/jobs/{id}/audio" in error["message"]


async def test_a_long_path_timeout_points_at_the_queue(monkeypatch, backends):
    """The job may still be running; a 504 that hid the id would be a leak."""
    stt, tts, _ = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=Slow()) as (client, _):
        response = await client.post(SPEECH, content=body(model="chatterbox"))

    assert response.status_code == 504
    assert "GET /jobs" in response.json()["error"]["message"]


async def test_the_long_read_timeout_sits_above_the_backends_own(monkeypatch):
    """240 s vs tts-long's SYNC_TIMEOUT of 180 s, so its honest 202 wins the race.

    A gateway timing out first would return 504 for a job that is still
    running and will produce audio, and would throw the job id away.
    """
    main = reload_gateway(monkeypatch)
    assert main.LONG.read_timeout > 180
    assert main.STT.read_timeout == 900
    assert main.TTS.read_timeout == 300
    assert main.CONNECT_TIMEOUT == 2


async def test_a_non_json_5xx_is_wrapped_and_truncated(monkeypatch, backends):
    """An HTML error page or a bare uvicorn 500 is not an envelope.

    Truncated because a stack trace reaches a client that cannot use it, and
    200 bytes is enough to tell an HTML page from a Python exception. The
    whole body goes to the log.
    """
    stt, tts, long = backends
    page = b"<html><body>" + b"z" * 4000 + b"</body></html>"
    tts.reply = lambda record: (500, {"content-type": "text/html"}, page)

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "backend_error"
    assert "tts-stack" in error["message"]
    assert "z" * 100 in error["message"]
    assert len(error["message"]) < 400  # not the whole 4 KB page


async def test_a_json_5xx_is_left_alone(monkeypatch, backends):
    """It is already an envelope; wrapping it would hide the backend's own code."""
    stt, tts, long = backends
    payload = json.dumps({"error": {"message": "synthesis failed: ffmpeg",
                                    "type": "server_error",
                                    "code": "synthesis_failed"}}).encode()
    tts.reply = lambda record: (500, {"content-type": "application/json"}, payload)

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "synthesis_failed"


async def test_a_loading_backend_gets_a_retry_after_and_nothing_else(monkeypatch, backends):
    """tts-stack's own 503 model_loading passes through; the gateway adds a header.

    Kokoro is 330 MB and always resident, so the window is seconds at
    container start, not minutes.
    """
    stt, tts, long = backends
    payload = json.dumps({"error": {"message": "model still loading",
                                    "type": "server_error",
                                    "code": "model_loading"}}).encode()
    tts.reply = lambda record: (503, {"content-type": "application/json"}, payload)

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.status_code == 503
    assert response.headers["retry-after"] == "10"
    assert response.content == payload


async def test_a_backends_own_retry_after_is_not_overwritten(monkeypatch, backends):
    stt, tts, long = backends
    tts.reply = lambda record: (503, {"content-type": "application/json",
                                      "retry-after": "120"}, b"{}")

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.headers["retry-after"] == "120"


async def test_unparseable_json_is_the_only_body_validation(monkeypatch, backends):
    """The gateway cannot route what it cannot parse — and nothing else is its business.

    Empty input, a bad voice, an unsupported response_format: the backend has
    better messages for all of them, and never sees this one.
    """
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=b"{not json")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_value"
    assert not tts.seen and not long.seen


async def test_a_json_body_that_is_not_an_object_still_reaches_the_backend(monkeypatch, backends):
    """It has no `model`, so it goes fast and the backend says what is wrong."""
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=b'["input"]')

    assert response.status_code == 200
    assert tts.last["body"] == b'["input"]'


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/redoc",
                                  "/v1/audio/translations", "/anything"])
async def test_everything_outside_the_table_is_404(monkeypatch, backends, path):
    """No catch-all pass-through.

    stt-stack deliberately put /docs, /redoc and /openapi.json behind its key
    because FastAPI's defaults were handing out a free map of the service. A
    wildcard route here would quietly undo that decision.
    """
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.get(path)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_url"
    assert not stt.seen and not tts.seen and not long.seen


async def test_a_wrong_method_is_an_envelope_too(monkeypatch, backends):
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.get("/transcribe")

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_supported"


# --------------------------------------------------------------------- auth --


async def test_unset_keys_means_open(monkeypatch, backends):
    """Loudly open, not refusing to boot: the LAN it runs on has no keys today."""
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        assert (await client.get("/voices")).status_code == 200


async def test_a_missing_key_is_a_401_envelope_with_a_challenge(monkeypatch, backends):
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long,
                       api_keys="sk-one,sk-two") as (client, _):
        response = await client.get("/voices")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "invalid_api_key"
    # Rejected before any backend was contacted.
    assert not tts.seen


@pytest.mark.parametrize("key", ["sk-one", "sk-two"])
async def test_every_configured_key_is_accepted(monkeypatch, backends, key):
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long,
                       api_keys="sk-one, sk-two") as (client, _):
        response = await client.get("/voices",
                                    headers={"authorization": f"Bearer {key}"})

    assert response.status_code == 200


async def test_a_non_ascii_key_authenticates(monkeypatch, backends):
    """The bug tts-stack paid for once: latin-1 wire bytes against a UTF-8 key.

    Re-encoding starlette's latin-1-decoded header as UTF-8 produced different
    bytes for every non-ASCII key, so the correct key was rejected as
    "Incorrect API key".
    """
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long,
                       api_keys="sk-café") as (client, _):
        # Bytes, not a str: this is what a client puts on the wire, and httpx
        # refuses to guess an encoding for a non-ASCII header value — which is
        # the same ambiguity the startup warning is about.
        response = await client.get(
            "/voices", headers={"authorization": "Bearer sk-café".encode()})

    assert response.status_code == 200


@pytest.mark.parametrize("path", ["/health", "/health/"])
async def test_health_needs_no_key_with_or_without_the_slash(monkeypatch, backends, path):
    """The other bug tts-stack paid for.

    Middleware runs before routing, so /health/ never reached the 307 that
    would have normalised it, and a probe written that way went permanently
    unhealthy the moment keys were set.
    """
    stt, tts, long = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=long,
                       api_keys="sk-one") as (client, _):
        response = await client.get(path, follow_redirects=True)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("value", ["", ",", "  ", ",,  ,"])
def test_keys_set_but_naming_none_refuses_to_start(monkeypatch, value):
    """`-e GATEWAY_API_KEYS=$SECRET` with SECRET unset must not mean "open".

    It matters more here than in a backend: this process is the only thing
    checking a token for three services, so the accident opens all of them.
    """
    with pytest.raises(SystemExit):
        reload_gateway(monkeypatch, api_keys=value)


# ------------------------------------------------------------------- health --


async def test_health_reports_all_three_without_proxying_auth(monkeypatch, backends):
    """One call, three answers, no key — the main justification beyond routing."""
    stt, tts, long = backends
    stt.reply = lambda r: (200, {"content-type": "application/json"},
                           b'{"status":"ok","model":"parakeet"}')
    tts.reply = lambda r: (200, {"content-type": "application/json"},
                           b'{"status":"ok","voices":54}')
    long.reply = lambda r: (200, {"content-type": "application/json"},
                            b'{"status":"ok","model_loaded":false,"queued":0}')

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long,
                       api_keys="sk-one") as (client, _):
        response = await client.get("/health")

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["gateway"] == "ok"
    # Each backend's own body, inlined rather than summarised.
    assert payload["backends"]["stt"]["health"]["model"] == "parakeet"
    assert payload["backends"]["tts"]["health"]["voices"] == 54
    assert payload["backends"]["tts_long"]["health"]["model_loaded"] is False
    # No key was forwarded, because the backends' /health want none either.
    assert all("authorization" not in b.last["headers"] for b in (stt, tts, long))
    assert [b.last["path"] for b in (stt, tts, long)] == ["/health"] * 3


async def test_health_answers_200_while_a_sibling_is_down(monkeypatch, backends):
    """A container must not be restarted because a sibling is restarting.

    The TrueNAS healthcheck for this container calls this endpoint. A 503 here
    for tts-long's cold start would have the orchestrator kill the gateway.
    Read `status`, not the code.
    """
    stt, tts, _ = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=Unreachable()) as (client, _):
        response = await client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["backends"]["tts_long"]["reachable"] is False
    assert "ConnectError" in payload["backends"]["tts_long"]["error"]
    # The two that are up are still reported in full.
    assert payload["backends"]["stt"]["reachable"] is True
    assert payload["backends"]["tts"]["health"]["backend"] == "tts-stack"


async def test_a_backend_that_is_loading_still_counts_as_answering(monkeypatch, backends):
    """It answered, and its own body says "loading" for anyone reading past the first field."""
    stt, tts, long = backends
    long.reply = lambda r: (200, {"content-type": "application/json"},
                            b'{"status":"ok","model_loaded":false}')
    tts.reply = lambda r: (200, {"content-type": "application/json"},
                           b'{"status":"loading","voices":0}')

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        payload = (await client.get("/health")).json()

    assert payload["status"] == "ok"
    assert payload["backends"]["tts"]["health"]["status"] == "loading"


async def test_health_survives_a_backend_answering_html(monkeypatch, backends):
    """Something else on the port, or a proxy in the way. Say so, do not raise."""
    stt, tts, long = backends
    long.reply = lambda r: (502, {"content-type": "text/html"}, b"<html>bad gateway")

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        payload = (await client.get("/health")).json()

    assert payload["status"] == "degraded"
    assert payload["backends"]["tts_long"]["http_status"] == 502
    assert "bad gateway" in payload["backends"]["tts_long"]["health"]["body"]


async def test_health_does_not_wait_forever_on_a_wedged_backend(monkeypatch, backends):
    """One call must answer even when a container has stopped talking."""
    stt, tts, _ = backends
    async with gateway(monkeypatch, stt=stt, tts=tts, long=Slow()) as (client, _):
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["backends"]["tts_long"]["reachable"] is False


# --------------------------------------------------------------------- logs --


async def test_one_log_line_per_request_carries_the_model_and_the_rate(monkeypatch, backends, caplog):
    """The entire observability budget. `grep` has to be able to answer

    "why did that take nine minutes", so the line names the route, the backend
    it chose, the model string as the client sent it, the upstream status, the
    duration this process observed, and the backend's own realtime factor.
    """
    stt, tts, long = backends
    tts.reply = lambda r: (200, {"content-type": "audio/mpeg",
                                 "x-realtime-factor": "1.4"}, b"ID3")

    with caplog.at_level("INFO", logger="voice-gateway"):
        async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
            await client.post(SPEECH, content=body(model="tts-1"))

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("route=")]
    assert len(lines) == 1
    for fragment in ("route=/v1/audio/speech", "backend=tts-stack", "model=tts-1",
                     "status=200", "duration=", "rtf=1.4"):
        assert fragment in lines[0]


async def test_a_client_that_hangs_up_mid_upload_still_writes_its_line(monkeypatch, backends, caplog):
    """A 499 used to return silently, which made "one line per request" false.

    /v1/audio/speech is the one route that reads the whole body before it can
    route, so it is the one route that can lose the client before a backend is
    ever chosen. It answered 499 and logged nothing, so the disconnects were
    invisible to the grep that is this service's entire observability budget.

    Driven as raw ASGI rather than through httpx: the disconnect has to arrive
    as an `http.disconnect` message while the handler is reading the body, and
    a transport that delivers a complete request cannot produce that.
    """
    stt, tts, long = backends
    sent: list[dict] = []

    async def receive():
        # The client vanished before any body arrived. Starlette turns this
        # message into ClientDisconnect inside request.body().
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "path": SPEECH, "raw_path": SPEECH.encode(),
        "query_string": b"", "root_path": "", "scheme": "http",
        "headers": [(b"host", b"gateway.test"),
                    (b"content-type", b"application/json"),
                    (b"content-length", b"64")],
        "client": ("10.0.0.9", 51234), "server": ("gateway.test", 8080),
    }

    with caplog.at_level("INFO", logger="voice-gateway"):
        async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (_, main):
            await main.app(scope, receive, send)

    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [499]
    # Nothing was forwarded: the request died before a backend was picked.
    assert not tts.seen and not long.seen

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("route=")]
    assert len(lines) == 1
    assert "status=client-disconnect" in lines[0]
    assert "route=/v1/audio/speech" in lines[0]


async def test_a_repeated_response_header_is_not_collapsed_into_one(monkeypatch, backends):
    """Two Set-Cookies must stay two, not become `a=1, b=2`.

    The proxy built its response headers with a dict comprehension over httpx's
    Headers.items(), which is a Mapping view: it joins duplicates with a comma.
    Measured — httpx.Headers([("set-cookie","a=1"),("set-cookie","b=2")])
    .items() yields 'a=1, b=2', which is one malformed cookie rather than two
    good ones.

    No backend in this stack sends a duplicate header today, so this never bit
    anyone. It is asserted because the day one starts to, the symptom is a
    broken login somewhere else entirely and nothing points back here.
    """
    stt, tts, long = backends
    tts.reply = lambda record: (200, [("content-type", "audio/mpeg"),
                                      ("set-cookie", "a=1"),
                                      ("set-cookie", "b=2"),
                                      ("vary", "accept"),
                                      ("vary", "origin")], b"ID3audio")

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(SPEECH, content=body())

    assert response.status_code == 200
    assert response.content == b"ID3audio"
    pairs = response.headers.multi_items()
    assert ("set-cookie", "a=1") in pairs and ("set-cookie", "b=2") in pairs
    assert ("vary", "accept") in pairs and ("vary", "origin") in pairs
    # The failure this guards against, stated as the thing that must not appear.
    assert not any(", " in v for k, v in pairs if k in ("set-cookie", "vary"))


async def test_a_repeated_request_header_reaches_the_backend_intact(monkeypatch, backends):
    """The same collapsing, in the other direction and with the other cause.

    Starlette's Headers.items() yields every pair, so a dict comprehension did
    not comma-join here — it let the LAST duplicate overwrite the first and
    dropped one silently. Two Cookie headers must arrive as two.
    """
    stt, tts, long = backends

    async with gateway(monkeypatch, stt=stt, tts=tts, long=long) as (client, _):
        response = await client.post(
            SPEECH, content=body(),
            headers=[("content-type", "application/json"),
                     ("cookie", "a=1"), ("cookie", "b=2")])

    assert response.status_code == 200
    assert len(tts.seen) == 1
    cookies = [v for k, v in tts.last["raw_headers"] if k == "cookie"]
    assert cookies == ["a=1", "b=2"]
