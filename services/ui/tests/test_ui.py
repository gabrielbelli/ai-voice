"""The page, the forwarding table, the key check and the upload ceiling."""

from __future__ import annotations

import json

from voice_common.conformance import assert_four_field_envelope


def test_the_page_is_served_and_needs_no_key(client):
    api, gateway, _ = client()
    gateway.keys = ("sk-real",)
    response = api.get("/ui")
    assert response.status_code == 200
    assert "<title>ai-voice</title>" in response.text
    # The page is static markup with no data in it, so it is public. Every XHR
    # it makes carries the key from the box on it.
    assert "connect-src 'self'" in response.headers["content-security-policy"]


def test_root_redirects_to_the_page(client):
    api, _, _ = client()
    assert api.get("/", follow_redirects=False).status_code == 307


def test_the_page_has_no_external_reference_of_any_kind(client):
    """No CDN, no font, no analytics: it works on a NAS with no internet."""
    api, _, _ = client()
    page = api.get("/ui").text
    for marker in ("http://", "https://cdn", "googleapis", "unpkg", "jsdelivr"):
        assert marker not in page.replace("http://127.0.0.1", ""), marker


def test_a_forwarded_route_reaches_the_gateway_with_the_key_intact(client):
    api, gateway, _ = client()
    response = api.get("/voices", headers={"Authorization": "Bearer sk-x"})
    assert response.status_code == 200
    forwarded = gateway.seen[-1]
    assert forwarded.url.path == "/voices"
    # NOT stripped, unlike at the gateway: there the next hop runs with its
    # keys unset, here the next hop is the thing that checks the key.
    assert forwarded.headers["authorization"] == "Bearer sk-x"


def test_delete_jobs_is_forwarded(client):
    """The route the gateway used to answer 405 for, which is what made a
    36-minute Chatterbox job impossible to call off through the front door."""
    api, gateway, _ = client()
    assert api.delete("/jobs/abc").status_code == 200
    assert (gateway.seen[-1].method, gateway.seen[-1].url.path) == ("DELETE", "/jobs/abc")


def test_an_unlisted_path_is_a_404_here_not_a_wildcard_proxy(client):
    api, gateway, _ = client()
    before = len(gateway.seen)
    response = api.get("/openapi.json")
    assert response.status_code == 404
    assert len(gateway.seen) == before, "nothing may be forwarded off-table"
    assert_four_field_envelope(response)


def test_an_oversized_upload_is_refused_before_a_byte_is_forwarded(client):
    api, gateway, _ = client(UI_MAX_UPLOAD_BYTES="1024")
    before = len(gateway.seen)
    response = api.post("/v1/audio/transcriptions", content=b"x" * 4096,
                        headers={"content-type": "audio/wav"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"
    # services/stt reads an UploadFile whole into RAM with no cap of its own,
    # so "before a byte is forwarded" is the entire point of this test.
    assert len(gateway.seen) == before


def test_the_key_is_checked_against_the_gateway_and_nowhere_else(client):
    api, gateway, _ = client()
    gateway.keys = ("sk-real",)
    refused = api.post("/ui/abandon", json={"token": "https://example.com/x"})
    assert refused.status_code == 401
    assert refused.headers["www-authenticate"] == "Bearer"
    assert refused.json()["error"]["code"] == "invalid_api_key"


def test_a_good_key_is_cached_so_polling_does_not_hammer_the_gateway(client):
    api, gateway, tube = client()
    gateway.keys = ("sk-real",)
    head = {"Authorization": "Bearer sk-real"}
    api.post("/ui/abandon", json={"token": "https://example.com/x"}, headers=head)
    checks = sum(1 for r in gateway.seen if r.url.path == "/v1/models")
    api.post("/ui/abandon", json={"token": "https://example.com/x"}, headers=head)
    assert sum(1 for r in gateway.seen if r.url.path == "/v1/models") == checks


def test_config_names_features_and_never_an_address(client):
    api, _, _ = client()
    payload = api.get("/ui/config").json()
    assert payload["ingestion"] is True
    assert "metube" not in json.dumps(payload).lower()
    # The seed is the conservative figure the gateway's own 900 s timeout was
    # built on, not the root README's optimistic one.
    assert payload["stt_rtf_seed"] == 8.5


def test_config_says_ingestion_is_off_when_metube_is_unset(client):
    api, _, _ = client(UI_METUBE_URL="")
    assert api.get("/ui/config").json()["ingestion"] is False


def test_resolve_is_a_clean_501_rather_than_a_hang_when_unconfigured(client):
    api, _, _ = client(UI_METUBE_URL="")
    response = api.post("/ui/resolve", json={"url": "https://example.com/v"})
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "ingestion_not_configured"


def test_health_is_unauthenticated_and_200_even_when_the_gateway_is_down(client):
    api, _, _ = client(UI_GATEWAY_URL="http://nothing.test")
    response = api.get("/health")
    assert response.status_code == 200
    body = response.json()
    # A UI reported unhealthy because a backend is restarting would be killed
    # by the orchestrator exactly when someone wants to open it and find out
    # why. Read `status`, not the code.
    assert body["status"] == "degraded"
    assert body["ui"] == "ok"
