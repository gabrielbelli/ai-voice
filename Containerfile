# Self-hosted speech-to-text, whole stack, CPU only.
#
#   audio -> VAD -> primary ASR -> secondary ASR -> consensus -> glossary
#
# No CUDA and no torch. CTranslate2 carries Whisper, ONNX Runtime carries
# Parakeet and Silero; together they cost a fraction of a single torch wheel,
# and nothing in the inference path needs it.
#
# Models are NOT baked in. Together they are well over a gigabyte, they are
# the pieces most likely to be swapped, and baking them would mean rebuilding
# the image to change one. They download into the volume at /models on first
# start.
#
# Build:
#   docker build -t stt-stack .
#
# Run:
#   docker run -p 8000:8000 -v stt-models:/models --cpus 4 -e STT_THREADS=4 stt-stack

FROM python:3.13-slim-trixie

# libsndfile is soundfile's only non-Python dependency. ffmpeg is deliberately
# absent: the service takes 16 kHz mono and rejects anything else rather than
# resampling, so a client sending 44.1 kHz is told, not quietly degraded.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY glossary.txt /etc/stt-stack/glossary.txt

# HF_HOME points at the volume so models survive container replacement.
# Without it every restart re-downloads more than a gigabyte.
ENV HF_HOME=/models \
    STT_PRIMARY=large-v3 \
    STT_PRIMARY_COMPUTE=int8 \
    STT_SECONDARY=istupakov/parakeet-tdt-0.6b-v3-onnx \
    STT_SECONDARY_QUANT=int8 \
    STT_GLOSSARY=/etc/stt-stack/glossary.txt \
    STT_THREADS=4 \
    STT_VAD=1 \
    PYTHONUNBUFFERED=1

# uid 1000, created rather than reused: the slim base has no non-root account
# and nothing here wants root. /models must be writable or the first-run
# download fails.
RUN useradd --create-home --uid 1000 stt \
 && mkdir -p /models \
 && chown stt:stt /models
USER stt

VOLUME ["/models"]
EXPOSE 8000

# No HEALTHCHECK instruction — the OCI image spec has no field for one, so a
# --format oci build drops it silently and the published config would not
# contain it. Pass it at run time:
#   docker run --health-cmd "python -c \
#     \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')\"" ...
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
