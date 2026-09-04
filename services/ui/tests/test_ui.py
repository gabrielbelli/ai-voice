"""The page, the forwarding table, the key check and the upload ceiling.

On the key: this service used to forward a credential the BROWSER held and add
none of its own. It now adds UI_GATEWAY_API_KEY when that is set, and the tests
below are split accordingly — the ones with no `UI_GATEWAY_API_KEY` in their
environment are there to prove the old behaviour is untouched when it is unset,
which is the deployment this repository ships.
"""

from __future__ import annotations

import json

from voice_common.conformance import assert_four_field_envelope


def test_the_page_is_served_and_needs_no_key(client):
    api, gateway, _ = client()
    gateway.keys = ("sk-real",)
    response = api.get("/ui")
    assert response.status_code == 200
    assert "<title>ai-voice</title>" in response.text
    # The page is static markup with no data and no credential in it, so it is
    # public. Every XHR it makes goes back to this origin, and this process is
    # what puts a key on the ones that leave it.
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


# ------------------------------------------ the key this container holds --
#
# The page has no key box any more. These cover the header this service adds in
# its place, and the property that matters most: with UI_GATEWAY_API_KEY unset,
# nothing above this line changes.


def test_the_container_key_is_added_to_a_proxied_request(client):
    """The browser sends no Authorization at all now, so this is the only one.

    Without it every proxied route would be an anonymous request and the whole
    page would be 401 on any deployment with GATEWAY_API_KEYS set.
    """
    api, gateway, _ = client(UI_GATEWAY_API_KEY="sk-container")
    gateway.keys = ("sk-container",)
    assert api.get("/voices").status_code == 200
    assert gateway.seen[-1].headers["authorization"] == "Bearer sk-container"


def test_a_caller_cannot_substitute_their_own_key(client):
    """Ours replaces theirs; it does not join it.

    HTTP allows a field name to repeat, so appending would put two
    Authorization headers on the wire and leave the gateway authenticating
    whichever one it read first — which is the caller's.
    """
    api, gateway, _ = client(UI_GATEWAY_API_KEY="sk-container")
    gateway.keys = ("sk-container",)
    response = api.get("/voices", headers={"Authorization": "Bearer sk-theirs"})
    assert response.status_code == 200
    sent = gateway.seen[-1].headers.get_list("authorization")
    assert sent == ["Bearer sk-container"], sent


def test_nothing_is_added_when_the_variable_is_unset(client):
    """The shipped deployment, and it must behave exactly as it always has."""
    api, gateway, _ = client()
    assert api.get("/voices").status_code == 200
    assert "authorization" not in gateway.seen[-1].headers


def test_our_own_routes_are_checked_with_the_container_key(client):
    """/ui/* spawns yt-dlp and writes files, and the caller presents nothing.

    The check is still the gateway's answer to GET /v1/models; what changed is
    whose credential is on it.
    """
    api, gateway, _ = client(UI_GATEWAY_API_KEY="sk-container")
    gateway.keys = ("sk-container",)
    response = api.post("/ui/abandon", json={"token": "https://example.com/x"})
    assert response.status_code == 200
    probe = [r for r in gateway.seen if r.url.path == "/v1/models"][-1]
    assert probe.headers["authorization"] == "Bearer sk-container"


def test_a_refused_container_key_is_a_503_and_never_a_401(client):
    """A 401 tells the user to fix a key box that no longer exists.

    UI_GATEWAY_API_KEY not matching GATEWAY_API_KEYS is a deployment fault, and
    passing the gateway's "Incorrect API key provided" through would send
    somebody looking for a field to correct it in.
    """
    api, gateway, _ = client(UI_GATEWAY_API_KEY="sk-wrong")
    gateway.keys = ("sk-real",)
    response = api.post("/ui/abandon", json={"token": "https://example.com/x"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "misconfigured_api_key"
    assert "www-authenticate" not in response.headers
    assert_four_field_envelope(response)


def test_an_ingested_file_is_handed_over_with_the_container_key(client):
    """The hand-off to the gateway is a request this service builds by hand.

    It copied the caller's header verbatim, so with the key box gone it would
    have carried nothing at all — and the failure would land as a 401 AFTER the
    download had already been paid for, which is the worst place in the flow to
    discover an auth problem.
    """
    url = "https://media.example/watch?v=abcdef"
    api, gateway, tube = client(UI_GATEWAY_API_KEY="sk-container")
    gateway.keys = ("sk-container",)
    api.post("/ui/resolve", json={"url": url})
    api.post("/ui/commit", json={"token": url})
    tube.finish(url, "A Title #1.opus")

    assert api.post("/ui/fetch", json={"token": url}).status_code == 200
    sent = [r for r in gateway.seen
            if r.url.path == "/v1/audio/transcriptions"][-1]
    assert sent.headers["authorization"] == "Bearer sk-container"


# ------------------------------------------- what was taken off the page --


def test_the_page_has_no_key_box(client):
    """It is not a bring-your-own-key tool; the container holds the key."""
    page = client()[0].get("/ui").text
    assert 'id="key"' not in page
    assert 'id="keyshow"' not in page
    assert 'store.get("key"' not in page, "a key is still kept in localStorage"


def test_the_page_never_attaches_an_authorization_header(client):
    """Every XHR goes through api(), which is now a bare fetch.

    A header set here would be a credential typed into a browser, which is the
    thing that was removed.
    """
    page = client()[0].get("/ui").text
    assert "Bearer" not in page.replace("WWW-Authenticate", ""), (
        "the page is building an Authorization header again")


def test_the_page_has_no_global_expert_gate(client):
    """One control per thing hidden. The <details> panels are the control.

    The checkbox was a second, global way to hide the same three panels, so a
    setting could be out of sight for two unrelated reasons and finding it
    meant reasoning about both.
    """
    page = client()[0].get("/ui").text
    assert 'id="expert"' not in page
    assert 'class="expert-only"' not in page
    assert "body:not(.expert)" not in page
    assert 'store.get("expert"' not in page


def test_the_three_expert_panels_are_still_on_the_page_and_openable(client):
    """Removing the gate must not remove the panels behind it."""
    page = client()[0].get("/ui").text
    for panel in ("stt-expert", "tts-expert-fast", "tts-expert-clone"):
        assert f'<details id="{panel}">' in page, f"{panel} is gone"


def test_the_two_engine_panels_still_swap_with_the_chosen_voice(client):
    """This is per-ENGINE, not per-expertise, and it is not what was removed.

    onVoiceChange() shows Kokoro's panel for a Kokoro voice and Chatterbox's
    for a clone. Deleting the global gate must leave that alone, or both panels
    appear at once and half the controls on screen belong to a model that is
    not going to run.
    """
    page = client()[0].get("/ui").text
    assert '$("tts-expert-fast").hidden = clone;' in page
    assert '$("tts-expert-clone").hidden = !clone;' in page


# ------------------------------------------- the prefixed mount (one door) --
#
# The gateway now fronts this service on a single published port, so the page
# is served from the gateway's origin. A bare relative call to /v1/... from
# there lands on the GATEWAY and never reaches this process -- skipping
# UI_GATEWAY_API_KEY, the only key a browser has since the key box was removed.
# /ui/api/... comes back here first. These pin that both mounts exist and that
# the prefixed one is not a way past the allowlist.


def test_the_prefixed_mount_reaches_the_same_route(client):
    api, gateway, _ = client()
    gateway.json = {"text": "hello"}
    assert api.post("/ui/api/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"RIFF", "audio/wav")},
                    data={"model": "parakeet"}).status_code == 200
    assert gateway.seen[-1].url.path == "/v1/audio/transcriptions", \
        "the prefix must be stripped before the request leaves this service"


def test_the_bare_mount_still_works_for_a_direct_caller(client):
    """Both mounts are live: this service's own port is still a valid door."""
    api, gateway, _ = client()
    gateway.json = {"text": "hello"}
    assert api.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", b"RIFF", "audio/wav")},
                    data={"model": "parakeet"}).status_code == 200
    assert gateway.seen[-1].url.path == "/v1/audio/transcriptions"


def test_the_prefix_is_not_a_way_past_the_allowlist(client):
    """PROXIED is matched against the STRIPPED path, so a prefixed request
    cannot reach a route the unprefixed one could not. /docs is the case that
    matters: the gateway's own 404 handler exists because a wildcard would
    proxy it to a service that deliberately does not publish it."""
    api, _, _ = client()
    for path in ("/ui/api/docs", "/ui/api/openapi.json", "/ui/api/health"):
        assert api.get(path).status_code == 404, f"{path} should not be routed"


def test_the_page_has_its_own_health_because_the_shapes_differ(client):
    """The page reads HEALTH.gateway.health. The gateway's own /health has no
    `gateway` key, so a page served from its origin asking for /health would
    read undefined for every status pill."""
    api, _, _ = client()
    body = api.get("/ui/health").json()
    assert "gateway" in body and "features" in body
    assert api.get("/health").json() == body, \
        "both paths are the same handler; only the URL differs"


def test_the_media_route_is_behind_the_same_key_check_as_the_rest(client):
    """It is a <video src>, so no XHR wrapper and no header is involved -- the
    browser fetches it the way it fetches an image, on the page's own origin,
    and the container's credential is added here as it is for every other
    /ui/* route. A route that serves bytes off a download directory must not be
    the one that opted out of the middleware."""
    api, gateway, _ = client(UI_GATEWAY_API_KEY="sk-container")
    gateway.keys = ("sk-container",)
    response = api.get("/ui/media", params={"token": "https://example.com/x"})
    # 404 rather than 401: the key was accepted and MeTube simply has no such
    # record. What matters is that the check ran at all.
    assert response.status_code == 404
    probe = [r for r in gateway.seen if r.url.path == "/v1/models"][-1]
    assert probe.headers["authorization"] == "Bearer sk-container"


def test_the_page_may_load_media_from_its_own_origin(client):
    """/ui/media is same-origin, so `media-src 'self'` covers it -- but a CSP
    tightened to `media-src blob:` alone would block every link's playback with
    a console message and no visible cause."""
    api, _, _ = client()
    policy = api.get("/ui").headers["Content-Security-Policy"]
    directive = [d.strip() for d in policy.split(";") if d.strip().startswith("media-src")]
    assert directive, "there is no media-src at all, so default-src decides"
    assert "'self'" in directive[0], directive


def test_the_page_is_never_cached(client):
    """It carried no Cache-Control at all, so browsers applied their own
    heuristic and served a stale copy. A control that had been added and
    deployed was reported as missing, and the diagnosis went through the
    markup, the boot order and the route table before reaching the cache.

    There is nothing to gain by caching it: one file from a local disk on a
    LAN, re-read per request by design, and its whole content changes on every
    deploy.
    """
    api, _, _ = client()
    for path in ("/ui", "/"):
        response = api.get(path, follow_redirects=True)
        assert response.status_code == 200
        cache = response.headers.get("cache-control", "")
        assert "no-cache" in cache, f"{path} served with cache-control={cache!r}"
