# One door in front of stt-stack, tts-stack and tts-long.
#
#   :8080  ->  /v1/audio/transcriptions  /transcribe   -> stt-stack:8000
#              /v1/audio/speech (by model) /speak /voices -> tts-stack:8001
#              /v1/audio/speech (chatterbox) /jobs/*   -> tts-long:8002
#              /v1/models  /health                     -> answered here
#
# This is the only port the TrueNAS app publishes. 8000, 8001 and 8002 stay on
# the app-internal container network, which is what makes the single auth
# boundary real rather than aspirational — and it is free, because the three
# containers already share a network: they are one app.
#
# No torch, no numpy, no model, no volume. The image is fastapi + httpx on the
# slim base and measures 283 MB (`docker images voice-gateway`). 215 MB of
# that is python:3.13-slim-trixie itself, so everything this repository adds
# is the remaining ~68 MB: fastapi, httpx, uvicorn[standard] and their
# dependencies. The figure is written down because it is the cheap check on
# this file — an audio or model library arriving by accident moves it by
# gigabytes, not megabytes. Nothing here decodes audio; bytes are streamed
# between two sockets and never held.
#
# Build:
#   docker build -t voice-gateway .
#
# Run (compose or a TrueNAS app, on the network the three backends share):
#   docker run -p 8080:8080 \
#     -e GATEWAY_STT_URL=http://stt-stack:8000 \
#     -e GATEWAY_TTS_URL=http://tts-stack:8001 \
#     -e GATEWAY_TTS_LONG_URL=http://tts-long:8002 \
#     -e GATEWAY_API_KEYS=sk-workstation,sk-laptop \
#     voice-gateway
#
# GATEWAY_API_KEYS unset means authentication is off and every request is
# accepted — for all three backends, since this is the only thing checking a
# token. The startup log says so at WARNING. Set but naming no key refuses to
# start; see app/auth.py.

FROM python:3.13-slim-trixie

# util-linux supplies setpriv for the entrypoint's privilege drop. It is
# normally already in the slim base; naming it means a base image change
# cannot silently remove it, and `command -v` fails the build rather than the
# container if it ever does.
#
# Nothing else is installed. No ffmpeg, no libsndfile, no curl: this process
# never opens an audio file, and the healthcheck below uses the interpreter it
# already has.
RUN apt-get update \
 && apt-get install -y --no-install-recommends util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && command -v setpriv

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# The backend URLs default to the service names the compose file uses. They are
# named here rather than left to the application's defaults so that `docker
# inspect` shows what this container will talk to.
#
# GATEWAY_API_KEYS is deliberately absent: keys do not belong baked into a
# published image, and unset is the documented "authentication off" state.
ENV GATEWAY_STT_URL=http://stt-stack:8000 \
    GATEWAY_TTS_URL=http://tts-stack:8001 \
    GATEWAY_TTS_LONG_URL=http://tts-long:8002 \
    PYTHONUNBUFFERED=1

# uid 1000, created rather than reused: the slim base has no non-root account.
# There is no USER instruction, for the same reason as in the siblings — the
# entrypoint starts as root so it can take ownership of anything bind-mounted,
# then drops to uid 1000 before uvicorn runs. Nothing here ever serves as root.
RUN useradd --create-home --uid 1000 gateway

# No VOLUME. This service is stateless by design: no model, no cache, no job
# state. A unified job abstraction over the two TTS backends would have needed
# storage and restart survival, and was rejected precisely because tts-long
# already has both.
EXPOSE 8080

# /health is exempt from GATEWAY_API_KEYS so this needs no key, and it answers
# 200 even when a backend is down — a container must not be restarted because a
# sibling is restarting. It reports which one is down in the body; this probe
# only asks whether the gateway itself is still answering.
#
# python rather than curl, which is not in this image and is not worth a layer.
# A HEALTHCHECK is not part of the OCI image spec, so a `--format oci` build
# drops it silently; the CI here builds in the default docker format, and a
# deployment that needs the probe regardless should pass it at run time.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/health', timeout=8).read()"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
