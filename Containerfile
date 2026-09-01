# Self-hosted speech-to-text, whole stack, CPU only.
#
#   audio -> VAD -> recogniser -> glossary repair
#
# One model, Parakeet by default. No CUDA and no torch: ONNX Runtime carries
# Parakeet and Silero, CTranslate2 carries Whisper for anyone who selects it.
# Together they cost a fraction of a single torch wheel.
#
# Models are NOT baked in. They are the piece most likely to be swapped, and
# baking one would mean rebuilding the image to change it. Whichever model is
# selected downloads into the volume at /models on first start — Parakeet is
# ~460 MB, Whisper large-v3 ~2.9 GB.
#
# Build:
#   docker build -t stt-stack .
#
# Run:
#   docker run -p 8000:8000 -v stt-models:/models --cpus 4 -e STT_THREADS=4 stt-stack

FROM python:3.13-slim-trixie

# ffmpeg is deliberately absent: the service takes 16 kHz mono and rejects
# anything else rather than resampling, so a client sending 44.1 kHz is told,
# not quietly degraded.
# libsndfile is soundfile's only non-Python dependency. util-linux supplies
# setpriv, which the entrypoint uses to drop privileges; it is normally already
# present in the slim base, and naming it here means a base change cannot
# silently remove it. The `command -v` line fails the build rather than the
# container if it ever goes missing.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && command -v setpriv

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY glossary.txt /etc/stt-stack/glossary.txt
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# HF_HOME points at the volume so models survive container replacement.
# Without it every restart re-downloads more than a gigabyte.
ENV HF_HOME=/models \
    STT_MODEL=parakeet \
    STT_QUANTISATION=int8 \
    STT_GLOSSARY=/etc/stt-stack/glossary.txt \
    STT_THREADS=4 \
    STT_VAD=1 \
    PYTHONUNBUFFERED=1

# uid 1000, created rather than reused: the slim base has no non-root account
# and nothing here wants root.
#
# There is no USER instruction. The container starts as root so the entrypoint
# can take ownership of a bind-mounted /models — a bind mount arrives with the
# host directory's ownership and overrides anything set here — and then drops
# to uid 1000 before exec'ing uvicorn. Nothing in the service ever runs as
# root. Set `user:` in compose to skip the chown entirely if you would rather
# manage ownership on the host.
RUN useradd --create-home --uid 1000 stt \
 && mkdir -p /models \
 && chown stt:stt /models

VOLUME ["/models"]
EXPOSE 8000

# No HEALTHCHECK instruction — the OCI image spec has no field for one, so a
# --format oci build drops it silently and the published config would not
# contain it. Pass it at run time:
#   docker run --health-cmd "python -c \
#     \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')\"" ...
#
# /health is exempt from STT_API_KEYS, so the probe needs no key. Under
# STT_TLS_CERT it needs the https:// URL and a certificate it can verify.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
