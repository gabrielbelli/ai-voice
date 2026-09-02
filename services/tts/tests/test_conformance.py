"""Run voice-common's shipped conformance suite against this service's own app.

Sharing app/auth.py stops three copies of app/auth.py from drifting. It does
nothing about the parts this repo still writes itself — its routes, its error
paths, its health payload — and that is where the same class of defect
reappears. So the package ships the assertions too and every consumer runs them
against the app object it actually builds: a bad voice-common bump fails here,
at this repo's build, rather than on orko port 8001.

Every test in that suite is named after a defect that was really found. Three
of them were found in THIS repo:

  * a non-ASCII TTS_API_KEYS value could never authenticate
  * GET /health/ came back 401 the moment keys were configured
  * TTS_API_KEYS=',' disabled authentication and logged it as unset

The star import is deliberate. It puts those test functions in a module inside
this repo's own tree, so this rootdir and any conftest here apply normally;
`pytest --pyargs voice_common.conformance` would collect them out of
site-packages instead.

Nothing here loads Kokoro. The suite never enters the app's lifespan, on
purpose — a conformance run that pulled 340 MB of weights into CI would be
switched off within a week, which is worse than any defect it guards.
"""

from __future__ import annotations

import pytest
from voice_common.conformance import *  # noqa: F401,F403
from voice_common.conformance import Service, module_app


@pytest.fixture
def voice_service() -> Service:
    return Service(env_var="TTS_API_KEYS",
                   build=module_app("app.main"),
                   v1_path="/v1/audio/speech")
