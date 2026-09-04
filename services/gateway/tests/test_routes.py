

# ------------------------------------------- the routes the agents needed --


def test_the_captions_route_reaches_the_ui():
    """A captions download is already a transcript and never touches stt.
    Absent from UI_PATHS the page 404s on it from the published port, which is
    how DELETE /jobs/{id} stayed unreachable while tts-long had it all along."""
    from app.main import UI_PATHS

    assert ("POST", "/ui/captions") in UI_PATHS


def test_every_glossary_route_is_routed():
    from app.main import app

    routes = {(m, r.path) for r in app.routes
              for m in getattr(r, "methods", ()) or ()}
    for method, path in (("GET", "/glossaries"), ("GET", "/glossaries/{name}"),
                         ("PUT", "/glossaries/{name}"),
                         ("DELETE", "/glossaries/{name}")):
        assert (method, path) in routes, f"{method} {path} is not routed"


def test_the_query_string_survives_the_proxy():
    """?force=true is what lets a single-word left-hand side through. Dropped
    silently, a `belly = Belli` rule becomes unenterable through the front door
    while appearing to work."""
    import inspect

    from app import main

    source = inspect.getsource(main._proxy)
    assert "request.url.query" in source


def test_the_media_relay_is_routed():
    """Byte ranges are what let <video> seek. Unrouted here, playback 404s from
    the published port -- the DELETE /jobs/{id} failure again."""
    from app.main import UI_PATHS

    assert ("GET", "/ui/media") in UI_PATHS


def test_proxy_does_not_strip_range_headers():
    """The relay parses no ranges of its own; it depends on this hop leaving
    Range and Content-Range alone."""
    from app.main import DROP_FROM_REQUEST, HOP_BY_HOP

    for header in ("range", "if-range", "content-range", "accept-ranges"):
        assert header not in DROP_FROM_REQUEST, f"{header} is dropped"
        assert header not in HOP_BY_HOP, f"{header} is treated as hop-by-hop"
