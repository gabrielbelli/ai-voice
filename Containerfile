# Kokoro text-to-speech, CPU only.
#
#   text or segments -> phonemise -> Kokoro -> wav/opus
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
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

ENV TTS_MODEL_DIR=/models \
    TTS_VOICE=bm_george \
    TTS_LANGUAGE=en-us \
    TTS_THREADS=4 \
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
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
