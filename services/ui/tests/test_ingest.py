"""Resolve, confirm, fetch — and abandon, which the ordering makes mandatory."""

from __future__ import annotations

URL = "https://media.example/watch?v=abcdef"


def probe_returning(main, monkeypatch, **facts):
    """Stand in for the yt-dlp subprocess. No process is ever spawned here."""
    async def fake(url):
        return main.probe.Probe(facts) if facts else None
    monkeypatch.setattr(main.probe, "run", fake)
    monkeypatch.setattr(main.probe, "available", lambda: bool(facts))
    monkeypatch.setattr(main.ingest.probe, "run", fake)
    monkeypatch.setattr(main.ingest.probe, "available", lambda: bool(facts))


def test_resolving_queues_without_downloading(client, monkeypatch):
    api, _, tube = client()
    main = __import__("app.main", fromlist=["main"])
    probe_returning(main, monkeypatch, title="A Title", duration=634.0,
                    bytes=10_271_496, is_live=False, has_subtitles=False,
                    uploader="Blender")
    body = api.post("/ui/resolve", json={"url": URL}).json()

    # auto_start MUST be sent explicitly false — it defaults to TRUE when the
    # field is None, so omitting it downloads immediately.
    path, sent = tube.calls[0]
    assert (path, sent["auto_start"]) == ("/add", False)
    assert sent["download_type"] == "audio", "a 4K video must not be pulled whole"
    assert sent["folder"], "without a folder the ingest lands in the music library"
    # And the item is in `pending`, not `queue`. A client that reads `queue`
    # finds nothing and reports a failed resolve.
    assert URL in tube.pending and not tube.queue

    assert body["duration"] == 634.0
    assert body["bytes"] == 10_271_496
    assert body["confirm"] is True   # over the size threshold


def test_short_obvious_media_does_not_nag(client, monkeypatch):
    api, _, _ = client()
    main = __import__("app.main", fromlist=["main"])
    # Four minutes, 4 MB: inside both thresholds, so no dialog. At the
    # conservative 8.5x that is under half a minute of transcription.
    probe_returning(main, monkeypatch, title="Short", duration=240.0,
                    bytes=4 * 1024 * 1024, is_live=False, has_subtitles=False,
                    uploader=None)
    assert api.post("/ui/resolve", json={"url": URL}).json()["confirm"] is False


def test_a_live_stream_always_confirms(client, monkeypatch):
    api, _, _ = client()
    main = __import__("app.main", fromlist=["main"])
    probe_returning(main, monkeypatch, title="Live", duration=30.0, bytes=1000,
                    is_live=True, has_subtitles=False, uploader=None)
    body = api.post("/ui/resolve", json={"url": URL}).json()
    assert body["is_live"] is True and body["confirm"] is True


def test_an_unknown_duration_confirms_because_not_knowing_is_the_point(client):
    api, _, _ = client(UI_PROBE="0")
    body = api.post("/ui/resolve", json={"url": URL}).json()
    assert body["probed"] is False and body["confirm"] is True
    assert body["title"] == "A Title"   # MeTube still gives a title


def test_our_guard_refuses_before_metube_is_even_told(client):
    api, _, tube = client()
    response = api.post("/ui/resolve", json={"url": "http://10.0.0.5:8080/x"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "refused_url"
    assert tube.calls == [], "a URL our guard hates must not become MeTube's problem"


def test_a_refused_link_is_a_400_about_the_link_not_a_502_about_metube(client):
    """It used to be a 502, and that sent people to debug a healthy MeTube.

    Every MeTubeError funnelled through _unavailable() and came back as
    ingestion_unavailable, so someone who pasted a link MeTube's own url_guard
    rejected was told the downloader was broken. The refusal is a fact about
    their input; only an unreachable or unusable MeTube is a 502.
    """
    api, _, tube = client()
    tube.refuse = 'Refusing to fetch internal host "localhost"'
    response = api.post("/ui/resolve", json={"url": URL})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "refused_url"
    # MeTube's url_guard says something more useful and more true than
    # anything we would write over the top of it.
    assert 'internal host "localhost"' in response.json()["error"]["message"]


def test_an_unreachable_metube_is_still_a_502(client):
    """The other half of the split above: an outage must not become a 400.

    If a refusal is a 400, the risk is that everything becomes a 400 and a
    genuine outage is reported as the user's fault.
    """
    api, _, tube = client()
    tube.down = True
    response = api.post("/ui/resolve", json={"url": URL})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ingestion_unavailable"


def test_commit_refuses_a_token_that_was_never_resolved(client):
    """The confirm gate is the server's, not only the page's.

    POST /ui/commit used to succeed on any URL, and with clip_start set it went
    straight to /add auto_start:true -- a download starting with no resolve step
    and no dialog. The page never took that path, so "nothing is fetched until
    the user agrees" was a property of the page rather than of the service.
    """
    api, _, tube = client()
    response = api.post("/ui/commit",
                        json={"token": URL, "clip_start": 0, "clip_end": 600})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_resolved"
    assert not any(path == "/add" for path, _ in tube.calls), \
        "a commit without a resolve must not reach MeTube's /add at all"


def test_committing_starts_the_pending_item_by_url_not_by_id(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    assert api.post("/ui/commit", json={"token": URL}).status_code == 200
    path, sent = tube.calls[-1]
    assert path == "/start"
    # PersistentQueue keys on info.url; the short `id` field is a silent no-op.
    assert sent["ids"] == [URL]
    assert URL in tube.queue and URL not in tube.pending


def test_a_clip_range_is_re_added_because_start_cannot_change_options(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "clip_start": 0, "clip_end": 600})
    paths = [p for p, _ in tube.calls]
    assert "/start" not in paths
    _, last = tube.calls[-1]
    assert last["clip_start"] == 0 and last["clip_end"] == 600
    assert last["auto_start"] is True


def test_existing_subtitles_are_fetched_as_captions_not_transcribed(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "captions": True})
    assert tube.calls[-1][1]["download_type"] == "captions"


def test_abandon_reaps_a_declined_link(client):
    """THE PATH THE ORDERING CREATES.

    /ui/resolve calls POST /add before probing, so MeTube's url_guard is what
    decides whether a URL may be fetched at all. Every declined link therefore
    leaves a pending record behind, and this is the test that says it does not
    stay there.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    assert URL in tube.pending
    body = api.post("/ui/abandon", json={"token": URL}).json()
    assert body["reaped"] is True
    assert URL not in tube.pending and URL not in tube.queue and URL not in tube.done


def test_abandon_reports_honestly_when_the_record_survives(client, monkeypatch):
    """MeTube answers {"status":"ok"} whether or not it deleted anything, so an
    unverified abandon is indistinguishable from a working one until the queue
    is full of orphans."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})

    def deaf(request):
        import httpx
        if request.url.path == "/delete":
            return httpx.Response(200, json={"status": "ok"})   # and does nothing
        return original(request)
    original = tube.handle
    tube.handle = deaf
    assert api.post("/ui/abandon", json={"token": URL}).json()["reaped"] is False


def test_progress_only_calls_a_download_ready_when_metube_says_finished(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL})
    assert api.get("/ui/progress", params={"token": URL}).json()["ready"] is False
    tube.finish(URL)
    body = api.get("/ui/progress", params={"token": URL}).json()
    assert body["ready"] is True and body["filename"] == "A Title.opus"


def test_a_failed_download_in_done_is_not_mistaken_for_ready(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    # _post_download_cleanup rewrites the status and nulls filename, so "it is
    # in done" is not the same as "it is ready".
    tube.pending.pop(URL)
    tube.done[URL] = {"url": URL, "title": "A Title", "status": "error",
                      "filename": None, "error": "unavailable"}
    body = api.get("/ui/progress", params={"token": URL}).json()
    assert body["ready"] is False and body["error"] == "unavailable"


def test_fetch_streams_the_file_into_the_gateway_as_multipart(client):
    api, gateway, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL})
    tube.finish(URL, "A Title #1.opus")

    response = api.post("/ui/fetch", params={"response_format": "srt"},
                        json={"token": URL})
    assert response.status_code == 200
    sent = [r for r in gateway.seen if r.url.path == "/v1/audio/transcriptions"][-1]
    body = sent.content
    assert b"multipart/form-data" in sent.headers["content-type"].encode()
    assert b'name="response_format"\r\n\r\nsrt' in body
    # `model` is required by /v1 validation and does not choose an engine.
    assert b'name="model"\r\n\r\nparakeet' in body
    # The filename came from /history and was percent-encoded, never predicted.
    assert b"RIFFfake-audio-bytes" in body


def test_fetch_refuses_a_download_that_has_not_finished(client):
    api, _, _ = client()
    api.post("/ui/resolve", json={"url": URL})
    response = api.post("/ui/fetch", json={"token": URL})
    assert response.status_code == 409


def test_resolve_is_rate_limited_so_it_is_not_a_scanner(client):
    api, _, _ = client(UI_RESOLVE_PER_MINUTE="2")
    for _ in range(2):
        api.post("/ui/resolve", json={"url": URL})
    response = api.post("/ui/resolve", json={"url": URL})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"


def test_a_job_audio_path_is_forwarded_whole(client):
    """/jobs/{id} and /jobs/{id}/audio are separate table entries and a path
    parameter does not match a slash, so the more specific one is not shadowed
    by the shorter."""
    api, gateway, _ = client()
    api.get("/jobs/abc-123/audio")
    assert gateway.seen[-1].url.path == "/jobs/abc-123/audio"
    api.get("/jobs/abc-123")
    assert gateway.seen[-1].url.path == "/jobs/abc-123"


def test_the_download_url_carries_the_folder_metube_put_the_file_in():
    """Omitting it 404'd every real download while the suite stayed green.

    compose.yaml sets a folder so ingest files land in one place rather than in
    the middle of the user's music library. MeTube writes them there and records
    it in the entry's `folder`; the URL built from that entry dropped it, so the
    file was fetched from /audio_download/<name> when it was at
    /audio_download/stt-ingest/<name>. The user saw "Transcription failed: 500
    Internal Server Error" and the cause was an httpx 404 three frames down.
    """
    from app.metube import MeTube

    client = MeTube(None, base="http://metube.test")
    assert client.audio_url("A Title.opus", "stt-ingest") == (
        "http://metube.test/audio_download/stt-ingest/A%20Title.opus")


def test_no_folder_still_produces_the_bare_path():
    """The default deployment writes to the root, and must keep working."""
    from app.metube import MeTube

    client = MeTube(None, base="http://metube.test")
    assert client.audio_url("A Title.opus", "") == (
        "http://metube.test/audio_download/A%20Title.opus")
    assert client.audio_url("A Title.opus") == (
        "http://metube.test/audio_download/A%20Title.opus")


def test_a_folder_with_odd_characters_is_encoded_not_broken():
    from app.metube import MeTube

    client = MeTube(None, base="http://metube.test")
    got = client.audio_url("x.opus", "my ingest")
    assert got == "http://metube.test/audio_download/my%20ingest/x.opus"
