# Kokoro text-to-speech, CPU only.
#
#   text or segments -> phonemise -> Kokoro -> wav, opus, mp3, aac, flac, pcm
#
# No CUDA and no torch. Kokoro is 82M parameters on ONNX Runtime, so the whole
# image lands around 400 MB — a torch wheel alone would be several times that,
# and nothing in the inference path needs it.
#
# The model is NOT baked in. It is ~310 MB plus 28 MB of voice embeddings, and
# it downloads into the volume at /models on first start.
#
# Build:
#   docker build -t tts-stack .
#
# Run:
#   docker run -p 8001:8001 -v tts-models:/models --cpus 4 -e TTS_THREADS=4 tts-stack
#
# Run with authentication and TLS:
#   docker run -p 8001:8001 -v tts-models:/models -v /etc/tts-certs:/certs:ro \
#     -e TTS_API_KEYS=key-one,key-two \
#     -e TTS_TLS_CERT=/certs/fullchain.pem -e TTS_TLS_KEY=/certs/privkey.pem \
#     tts-stack
#
# Both are off when their variables are unset, which is how the service shipped
# and how it stays on upgrade. The key file must be readable by uid 1000; the
# entrypoint checks that and refuses to start rather than falling back to HTTP.
#
# That entrypoint is no longer in this repo. It ships in voice-common and pip
# installs it to /usr/local/bin, so the TLS logic and the auth logic ride the
# same pin and cannot be at different versions inside one image — which is
# exactly how three copies of the script ended up with three different holes.

FROM python:3.13-slim-trixie

# espeak-ng does the phonemisation Kokoro needs; libsndfile backs soundfile;
# ffmpeg is here only for opus output, which is optional but small.
# util-linux supplies setpriv for the entrypoint's privilege drop — named
# explicitly so a base image change cannot silently remove it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      espeak-ng libsndfile1 ffmpeg curl util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && command -v setpriv

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# VOICE_* are read by the shared entrypoint and are image settings, not
# operator settings: they only tell it which prefix this service's TLS
# variables use and which volume to take ownership of. Every variable an
# operator sets is still spelled TTS_.
ENV TTS_MODEL_DIR=/models \
    TTS_VOICE=bm_george \
    TTS_LANGUAGE=en-us \
    TTS_THREADS=4 \
    VOICE_TLS_PREFIX=TTS \
    VOICE_CHOWN_DIRS=/models \
    PYTHONUNBUFFERED=1

# uid 1000, created rather than reused: the slim base has no non-root account.
# There is no USER instruction — the entrypoint starts as root purely to take
# ownership of a bind-mounted /models, which arrives with the host directory's
# ownership regardless of what is set here, then drops to uid 1000 before
# anything else runs.
RUN useradd --create-home --uid 1000 tts \
 && mkdir -p /models \
 && chown tts:tts /models

VOLUME ["/models"]
EXPOSE 8001

# No HEALTHCHECK instruction — the OCI image spec has no field for one, so a
# --format oci build drops it silently. Pass it at run time.
ENTRYPOINT ["/usr/local/bin/voice-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
