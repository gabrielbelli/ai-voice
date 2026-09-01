# Chatterbox long-form speech, CPU only.
#
#   POST /jobs -> queued -> Chatterbox -> wav on disk
#
# POST /v1/audio/speech is the same queue in OpenAI's shape, blocking only for
# input short enough that blocking is honest. /jobs remains the richer route.
#
# This is a job queue, not a request/response service. Measured on an M2 Max
# CPU it runs at roughly 0.21x realtime, so a ten-minute recording takes about
# three quarters of an hour. An HTTP request waiting for that would time out.
#
# Torch is unavoidable here — Chatterbox has no ONNX build — but the CPU wheel
# is used deliberately. The CUDA wheels add several gigabytes, and the GPU this
# would otherwise target is a GTX 1060: Pascal, whose FP16 runs at 1/64 rate,
# so it would not help even where one exists.
#
# Build:
#   docker build -t tts-long .
#
# Run:
#   docker run -p 8002:8002 -v tts-long-models:/models -v tts-long-out:/output \
#              --cpus 8 -e TTS_THREADS=8 tts-long

# 3.12, not 3.13+: resemble-perth imports pkg_resources, which 3.14 removed.
# The watermarker is stubbed in app/synth.py as well, but pinning the
# interpreter keeps a transitive import from deciding this for us.
FROM python:3.12-slim-trixie

# libgomp1 supplies libgomp.so.1, the OpenMP runtime torch links against. The
# amd64 wheels vendor their own copy and the arm64 ones do not, so without this
# the image builds cleanly on both and then fails to import torch on arm64.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libgomp1 util-linux \
 && rm -rf /var/lib/apt/lists/* \
 && command -v setpriv

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# HF_HOME on the volume so the ~3 GB of weights survive container replacement.
# librosa pulls numba, which JIT-compiles on first use and caches the result
# next to its own installed source — read-only for a non-root user, so the
# first job dies with "no locator available for file .../librosa/...". Point
# the cache somewhere writable instead.
# HOME is set explicitly because the entrypoint starts as root and drops
# privileges with setpriv, which changes the uid but leaves HOME=/root.
# chatterbox pulls spacy-pkuseg, which downloads a model into $HOME/.pkuseg
# on first use and dies with EACCES when that is root's home.
ENV HOME=/home/tts \
    NUMBA_CACHE_DIR=/tmp/numba \
    HF_HOME=/models \
    TTS_OUTPUT_DIR=/output \
    TTS_THREADS=8 \
    TTS_IDLE_TIMEOUT=600 \
    TTS_LANGUAGE=en \
    TTS_EXAGGERATION=0.3 \
    TTS_CFG_WEIGHT=0.3 \
    TTS_TEMPERATURE=0.6 \
    TTS_OPENAI_SYNC_MAX_CHARS=300 \
    TTS_OPENAI_SYNC_TIMEOUT=180 \
    PYTHONUNBUFFERED=1

# TTS_API_KEYS, TTS_TLS_CERT and TTS_TLS_KEY are deliberately absent. Keys do
# not belong baked into a published image, and a certificate path is only
# meaningful once something is mounted at it. Unset means no auth and plain
# HTTP, which is what this already did — the startup log says so at WARNING.

RUN useradd --create-home --uid 1000 tts \
 && mkdir -p /models /output \
 && chown tts:tts /models /output

VOLUME ["/models", "/output"]
EXPOSE 8002

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
