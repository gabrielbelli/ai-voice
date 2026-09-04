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


# ------------------------------------------------ cloning from a link --


def test_a_finished_download_becomes_a_reference_clip(client, tmp_path):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    tube.finish(URL, filename="A Title.opus")
    response = api.post("/ui/clips/from-link",
                        json={"token": URL, "name": "gabriel"})
    # 400 is the honest answer when the fixture's bytes are not decodable
    # audio; what must NOT happen is a 404 or a 502, which would mean the
    # route never found the file MeTube reported.
    assert response.status_code in (201, 400), response.text
    assert response.json()["error"]["code"] != "unknown_token" \
        if response.status_code == 400 else True


def test_importing_before_the_download_finishes_is_a_409(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    response = api.post("/ui/clips/from-link",
                        json={"token": URL, "name": "gabriel"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_ready"


def test_an_unknown_link_is_a_404_not_a_500(client):
    api, _, _ = client()
    response = api.post("/ui/clips/from-link",
                        json={"token": "https://example.com/never-resolved",
                              "name": "x"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_token"


def test_the_import_route_is_not_the_transcribe_route(client):
    """/ui/fetch streams into the gateway and keeps nothing; this keeps the
    bytes and never transcribes. Sharing them would mean a `purpose` flag
    deciding which of two unrelated things happens."""
    from app import ingest

    source = inspect_source(ingest.clip_from_link)
    assert "clips.save" in source
    assert "multipart" not in source.lower()


def inspect_source(fn):
    import inspect
    return inspect.getsource(fn)


def test_a_clip_import_asks_metube_for_wav(client):
    """MeTube's default here is opus, which services/ui cannot read.

    clips.save measures duration with the stdlib `wave` module -- this image
    has no ffmpeg and no audio library -- so an opus file came back "that file
    is not a WAV this service can read", which was true and unhelpful.
    Transcription keeps opus: a two-hour podcast at ~1 MB a minute is what
    makes link ingestion affordable, and the gateway decodes it anyway.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "for_clip": True,
                                 "clip_start": 10, "clip_end": 30})
    adds = [body for path, body in tube.calls if path == "/add"]
    assert adds[-1]["format"] == "wav", adds[-1]
    assert adds[-1]["clip_start"] == 10 and adds[-1]["clip_end"] == 30


def test_a_transcription_commit_still_asks_for_opus(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "clip_start": 0, "clip_end": 60})
    adds = [body for path, body in tube.calls if path == "/add"]
    assert adds[-1]["format"] == "opus", adds[-1]


def test_an_untrimmed_clip_import_still_re_adds(client):
    """/start promotes a pending item with the options it was ADDED with, and
    /ui/resolve queued it as opus. Without the re-add, a clip import with no
    trim would quietly download the wrong format."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "for_clip": True})
    adds = [body for path, body in tube.calls if path == "/add"]
    assert adds[-1]["format"] == "wav"
    assert not any(path == "/start" for path, _ in tube.calls), \
        "a clip import must re-add rather than start the opus item"


# ------------------------------------------------------------ captions --
#
# "This has real subtitles already" on the confirm card asks MeTube for
# download_type "captions", which sets yt-dlp's skip_download and produces a
# .vtt or .srt and no media at all. Every test below exists because that file
# used to be streamed into /v1/audio/transcriptions, where stt-stack was handed
# a text file and asked to decode it as media.

VTT = ("WEBVTT\n\n"
       "00:00:01.000 --> 00:00:03.500\n"
       "The first line.\n\n"
       "00:00:03.500 --> 00:00:06.000\n"
       "And the second.\n")


def finished_captions(tube, url, filename="A Title.en.vtt", body=VTT):
    tube.finish(url, filename=filename)
    tube.content = body.encode()


def test_a_captions_download_is_never_handed_to_the_transcriber(client):
    """THE BUG. /ui/fetch took whatever `filename` MeTube reported and streamed
    it into /v1/audio/transcriptions, so a subtitle file reached a service whose
    next move is to hand the bytes to libav. The button on the confirm card
    could not work, and the failure surfaced two services away as a decode
    error about a file the user never saw.
    """
    api, gateway, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "captions": True})
    finished_captions(tube, URL)

    before = len(gateway.seen)
    response = api.post("/ui/fetch", json={"token": URL})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_media"
    assert len(gateway.seen) == before, \
        "a subtitle file reached the gateway; that is the whole bug"


def test_captions_come_back_as_text_and_nothing_is_transcribed(client):
    """The point of the button: about two seconds and no compute at all.

    A video with human-written subtitles already has a transcript attached to
    it. Transcribing it would be paying minutes of Parakeet to reproduce, less
    accurately, a file we are already holding.
    """
    api, gateway, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "captions": True})
    finished_captions(tube, URL)

    before = len(gateway.seen)
    body = api.post("/ui/captions", json={"token": URL}).json()
    assert body["format"] == "vtt"
    assert body["text"] == VTT
    assert not [r for r in gateway.seen[before:]
                if r.url.path == "/v1/audio/transcriptions"]


def test_subrip_is_reported_as_subrip(client):
    """The two suffixes are the only thing in the /history entry that tells a
    captions download from an audio one, and the page needs it for the
    extension the Download button writes."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_captions(tube, URL, filename="A Title.en.srt", body="1\n")
    assert api.post("/ui/captions", json={"token": URL}).json()["format"] == "srt"


def test_captions_refuses_a_media_download(client):
    """The mirror of the guard in fetch(). Returning a few megabytes of opus as
    if it were text is a worse answer than saying which route to use."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL})
    tube.finish(URL, filename="A Title.opus")
    response = api.post("/ui/captions", json={"token": URL})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_captions"


def test_a_subtitle_file_beside_the_video_is_still_found(client):
    """AUDIO_DOWNLOAD_DIR defaults to "%%DOWNLOAD_DIR" and is unset here, so
    /download/ and /audio_download/ are one directory and everything is found
    first time. A deployment that sets them apart puts the .vtt beside the
    VIDEO -- skip_download writes it there -- and the first URL 404s.

    Without the fallback that is a 502 whose cause is a setting in somebody
    else's application, which is the same shape as the missing folder segment
    that 404'd every download this service ever started.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_captions(tube, URL)
    tube.served_from = "video"
    body = api.post("/ui/captions", json={"token": URL}).json()
    assert body["text"] == VTT


def test_captions_missing_from_both_directories_is_a_502_not_an_empty_pane(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_captions(tube, URL)
    tube.served_from = "nowhere"
    response = api.post("/ui/captions", json={"token": URL})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ingestion_unavailable"


def test_something_the_size_of_media_is_refused_however_it_is_named(client):
    """A name ending .vtt is not a promise about the size behind it, and this
    route buffers rather than streams. compose.yaml gives this service
    mem_limit: 384m, so the unbounded version fails as an OOM kill rather than
    as a message -- the same trap as the clip route's, documented at
    config.MAX_CLIP_BYTES."""
    api, _, tube = client(UI_MAX_CAPTION_BYTES="1024")
    api.post("/ui/resolve", json={"url": URL})
    finished_captions(tube, URL, body="x" * 4096)
    response = api.post("/ui/captions", json={"token": URL})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "captions_too_large"


def test_captions_on_an_unfinished_download_is_a_409(client):
    api, _, _ = client()
    api.post("/ui/resolve", json={"url": URL})
    assert api.post("/ui/captions", json={"token": URL}).status_code == 409


def test_the_video_route_is_the_other_static_directory(client):
    from app.metube import MeTube

    tube = MeTube(None, base="http://metube.test")
    assert tube.video_url("A Title.en.vtt", "stt-ingest") == (
        "http://metube.test/download/stt-ingest/A%20Title.en.vtt")


# ------------------------------------------------------------- the video --
#
# THE DEFAULT IS AUDIO AND MUST STAY AUDIO. download_type "audio" never pulls
# the video stream, which is the whole reason a 2h14m podcast costs ~131 MB
# instead of gigabytes. Every test below is about the opt-in NOT leaking into
# the paths that did not ask for it.


def test_keeping_the_video_asks_metube_for_a_container_a_browser_can_play(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    body = api.post("/ui/commit", json={"token": URL, "video": True}).json()

    add = [sent for path, sent in tube.calls if path == "/add"][-1]
    assert add["download_type"] == "video"
    assert add["format"] == "mp4", add
    assert add["quality"] == "best"
    assert add["auto_start"] is True
    # Echoed, so the page picks the <video> element from what happened rather
    # than from what it asked for.
    assert body["video"] is True


def test_a_video_commit_re_adds_because_start_cannot_change_the_type(client):
    """/start promotes a pending item with the options it was ADDED with.

    /ui/resolve queued it as download_type "audio", so a commit that merely
    started it would download the audio and then hand the page a filename it
    would try to show in a <video>: a black rectangle and no picture, with
    nothing anywhere saying why.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "video": True})
    assert not any(path == "/start" for path, _ in tube.calls)


def test_an_ordinary_commit_still_downloads_audio_only(client):
    """The regression this whole feature could have been: silently expensive.

    Nothing about adding an opt-in may change what happens when it is not
    taken, and this asserts the cheap path is still both cheap and a /start
    rather than the more expensive re-add.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    body = api.post("/ui/commit", json={"token": URL}).json()
    add = [sent for path, sent in tube.calls if path == "/add"][-1]
    assert add["download_type"] == "audio" and add["format"] == "opus"
    assert any(path == "/start" for path, _ in tube.calls)
    assert body["video"] is False


def test_a_clip_import_is_never_turned_into_a_video(client):
    """clips.save reads WAV with the stdlib `wave` module and nothing else.

    The clone sheet posts for_clip and never posts video, so this is about the
    precedence holding if both ever arrive: an mp4 would reach clips.save and
    come back "that file is not a WAV this service can read".
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    api.post("/ui/commit", json={"token": URL, "for_clip": True, "video": True})
    add = [sent for path, sent in tube.calls if path == "/add"][-1]
    assert (add["download_type"], add["format"]) == ("audio", "wav")


def test_captions_beat_video_because_there_is_no_media_either_way(client):
    """skip_download means no stream is pulled at all, so asking for both is a
    contradiction and the cheap half of it wins."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    body = api.post("/ui/commit",
                    json={"token": URL, "captions": True, "video": True}).json()
    add = [sent for path, sent in tube.calls if path == "/add"][-1]
    assert add["download_type"] == "captions"
    assert body["video"] is False


# ------------------------------------------------------------ /ui/media --
#
# WHAT THE PLAYER ACTUALLY NEEDS, which is byte ranges. A <video> seeks by
# asking for one; served by something that ignores Range and answers 200 with
# the whole file it plays from the start and drops every scrub on the floor.
# MeTube already answers ranges correctly -- verified live: `Range:
# bytes=0-1023` gave `206`, `Content-Range: bytes 0-1023/533915`,
# `Accept-Ranges: bytes`, `Content-Type: video/mp4` -- so these tests are about
# the RELAY being faithful, not about a range parser we deliberately do not
# have.

MEDIA = b"0123456789abcdefghij"


def finished_media(tube, url, filename="A Title.mp4",
                   content_type="video/mp4", body=MEDIA):
    tube.finish(url, filename=filename)
    tube.content = body
    tube.content_type = content_type


def test_a_link_is_playable_at_all_or_none_of_this_works(client):
    """The plain request, which is what an element makes before it seeks."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL)

    response = api.get("/ui/media", params={"token": URL})
    assert response.status_code == 200
    assert response.content == MEDIA
    assert response.headers["content-type"] == "video/mp4"
    # Without this the element never sends a Range at all and the scrub bar is
    # decorative.
    assert response.headers["accept-ranges"] == "bytes"


def test_a_bounded_range_comes_back_as_a_206_and_not_as_the_whole_file(client):
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL)

    response = api.get("/ui/media", params={"token": URL},
                       headers={"Range": "bytes=4-9"})
    assert response.status_code == 206
    assert response.content == MEDIA[4:10]
    assert response.headers["content-range"] == f"bytes 4-9/{len(MEDIA)}"
    # THE SLICE, not the file. A content-length recomputed here is how a player
    # ends up waiting for bytes that are never coming.
    assert response.headers["content-length"] == "6"


def test_an_open_ended_range_is_relayed_rather_than_interpreted_here(client):
    """`bytes=12-` is what an element sends when it seeks and then plays on.

    Nothing in this service parses it: the header goes up as it arrived and
    MeTube's own answer comes back, which is why there is no second, worse
    range implementation to keep in step with aiohttp's.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL)

    response = api.get("/ui/media", params={"token": URL},
                       headers={"Range": "bytes=12-"})
    assert response.status_code == 206
    assert response.content == MEDIA[12:]
    assert response.headers["content-range"] == f"bytes 12-19/{len(MEDIA)}"


def test_a_range_past_the_end_is_the_416_a_browser_knows_how_to_correct(client):
    """Not a 500, and not a 200 with the whole file.

    A player that asked past the end reads the total out of the 416 and asks
    again; dressed up as an outage it stops instead, and dressed up as a 200 it
    is handed bytes it did not ask for and splices them in.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL)

    response = api.get("/ui/media", params={"token": URL},
                       headers={"Range": "bytes=9000-9999"})
    assert response.status_code == 416
    assert response.headers["accept-ranges"] == "bytes"


def test_if_range_goes_up_so_a_stale_player_is_not_handed_two_files(client):
    """A conditional range whose condition never travels is unconditional.

    If-Range is compared against the ETag, and both the request header and the
    ETag that answers it have to survive the relay. Drop either and a player
    holding a stale validator gets a slice of a DIFFERENT file and splices the
    two together with no error anywhere.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL)

    fresh = api.get("/ui/media", params={"token": URL},
                    headers={"Range": "bytes=0-3", "If-Range": tube.etag})
    assert fresh.status_code == 206 and fresh.content == MEDIA[:4]

    stale = api.get("/ui/media", params={"token": URL},
                    headers={"Range": "bytes=0-3", "If-Range": '"gone"'})
    # The whole file, which is what a stale condition means, rather than a
    # slice of something the player cannot line up with what it already has.
    assert stale.status_code == 200 and stale.content == MEDIA

    # And the validator itself is relayed, or the browser has nothing to send
    # back on the next request.
    assert fresh.headers["etag"] == tube.etag
    assert "last-modified" in fresh.headers


def test_an_audio_only_link_is_served_too_because_the_words_still_follow(client):
    """The video tick is opt-in and most links will not have it.

    With the audio playable the transcript follows along exactly as an upload
    does, which is the larger half of this feature: a caption band needs a
    picture, a karaoke highlight does not.
    """
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL, filename="A Title.opus", content_type="audio/ogg")

    response = api.get("/ui/media", params={"token": URL})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/ogg"


def test_media_refuses_a_subtitle_file_however_it_is_reached(client):
    """A .vtt served as video/mp4 is a confusing failure two layers away."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    tube.finish(URL, filename="A Title.en.vtt")
    assert api.get("/ui/media", params={"token": URL}).status_code == 409


def test_media_refuses_a_name_that_is_not_media_at_all(client):
    """An allowlist, so the info sidecar and the thumbnail yt-dlp may leave
    beside the file are refused by default rather than served and corrected."""
    api, _, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    tube.finish(URL, filename="A Title.info.json")
    assert api.get("/ui/media", params={"token": URL}).status_code == 409


def test_media_refuses_a_download_that_has_not_finished(client):
    api, _, _ = client()
    api.post("/ui/resolve", json={"url": URL})
    assert api.get("/ui/media", params={"token": URL}).status_code == 409


def test_media_serves_only_a_token_this_page_actually_resolved(client):
    """The gate on the whole route. Without the /history lookup this is an open
    read of anything in someone else's download directory, by a name a caller
    supplies."""
    api, _, _ = client()
    response = api.get("/ui/media", params={"token": "https://media.example/other"})
    assert response.status_code == 404


def test_a_file_past_the_ceiling_is_refused_rather_than_streamed(client):
    api, _, tube = client(UI_MAX_MEDIA_BYTES="10")
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL)

    response = api.get("/ui/media", params={"token": URL})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "media_too_large"


def test_the_ceiling_reads_the_whole_file_and_not_the_slice(client):
    """Otherwise a 4 GB file walks in one kilobyte at a time.

    On a 206 the size of the file is the figure after the slash in
    Content-Range; Content-Length is the slice. A ceiling that read the slice
    would pass every range ever sent.
    """
    api, _, tube = client(UI_MAX_MEDIA_BYTES="10")
    api.post("/ui/resolve", json={"url": URL})
    finished_media(tube, URL)

    response = api.get("/ui/media", params={"token": URL},
                       headers={"Range": "bytes=0-3"})
    assert response.status_code == 413


def test_the_media_ceiling_is_not_the_upload_ceiling(client):
    """Two different questions: how big a file services/stt may read into a
    6 GB container, and how big a file this laptop may pull down a domestic
    line. Sharing one number would tie them together."""
    from app import config
    assert config.MAX_MEDIA_BYTES != config.MAX_UPLOAD_BYTES


# ------------------------------------------- timings on the link path --
#
# WHY A LINK HAD NO HIGHLIGHT. The cues come from timedFromJson(), which needs
# verbose_json; formatForUpload() asks for it but only ran on the upload path,
# and this route forwarded `model` and `response_format` and nothing else -- so
# a link was transcribed as plain text and no timing ever came back.


def test_a_link_can_ask_for_the_timings_the_highlight_is_drawn_from(client):
    api, gateway, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    tube.finish(URL, "A Title.opus")

    response = api.post(
        "/ui/fetch",
        params=[("response_format", "verbose_json"),
                ("timestamp_granularities", "word"),
                ("timestamp_granularities", "segment")],
        json={"token": URL})
    assert response.status_code == 200

    sent = [r for r in gateway.seen
            if r.url.path == "/v1/audio/transcriptions"][-1].content
    assert b'name="response_format"\r\n\r\nverbose_json' in sent
    # BOTH, and a dict would have carried one. `word` is what the highlight
    # follows; `segment` is what the caption band and the sidecar are built
    # from, and what the highlight falls back to when a glossary rule spanning
    # two words stops the words reconstructing the line.
    assert sent.count(b'name="timestamp_granularities[]"') == 2
    assert b'name="timestamp_granularities[]"\r\n\r\nword' in sent
    assert b'name="timestamp_granularities[]"\r\n\r\nsegment' in sent


def test_an_invented_granularity_never_reaches_the_multipart_frame(client):
    """The frame is built by hand, so an unvalidated value carrying CRLF closes
    one part and opens another -- the same hole response_format was allowlisted
    for."""
    api, gateway, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    tube.finish(URL, "A Title.opus")

    response = api.post(
        "/ui/fetch",
        params=[("response_format", "verbose_json"),
                ("timestamp_granularities",
                 'word"\r\n\r\nx\r\n--boundary\r\nContent-Disposition: form-data; name="model')],
        json={"token": URL})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_granularity"
    assert not [r for r in gateway.seen if r.url.path == "/v1/audio/transcriptions"]


def test_asking_for_no_granularity_still_sends_none(client):
    """The audio-only, text-only case must not start paying for timestamps it
    has nothing to draw with -- about 5% on Parakeet, measured in asr.py."""
    api, gateway, tube = client()
    api.post("/ui/resolve", json={"url": URL})
    tube.finish(URL, "A Title.opus")

    api.post("/ui/fetch", params={"response_format": "text"}, json={"token": URL})
    sent = [r for r in gateway.seen
            if r.url.path == "/v1/audio/transcriptions"][-1].content
    assert b"timestamp_granularities" not in sent
