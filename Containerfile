# CPU-only speech-to-text. Parakeet TDT 0.6B v3 on ONNX Runtime.
#
# There is no CUDA here and no torch. onnx-asr[cpu] pulls onnxruntime alone,
# which keeps the image around 400 MB rather than the 3 GB a torch-based NeMo
# install would cost — and torch would buy nothing, because inference runs
# entirely in ONNX Runtime.
#
# The model is NOT baked in. It is ~600 MB, it is the thing most likely to be
# swapped (a Portuguese fine-tune, a different size), and baking it would mean
# rebuilding the image to change it. It downloads on first start into the
# volume mounted at /models.
#
# Build:
#   buildah bud --platform linux/amd64 -t parakeet-stt .
#
# Run:
#   podman run -p 8000:8000 -v parakeet-models:/models parakeet-stt

FROM python:3.13-slim-trixie

# libsndfile is soundfile's only non-Python dependency. ffmpeg is deliberately
# absent: the service accepts WAV/FLAC/OGG at 16 kHz and rejects anything else
# rather than resampling, so there is nothing for ffmpeg to do.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY glossary.txt /etc/parakeet-stt/glossary.txt

# HF_HOME points at the volume so the model survives container replacement.
# Without this every restart re-downloads 600 MB.
ENV HF_HOME=/models \
    STT_MODEL=istupakov/parakeet-tdt-0.6b-v3-onnx \
    STT_QUANTISATION=int8 \
    STT_GLOSSARY=/etc/parakeet-stt/glossary.txt \
    STT_THREADS=4 \
    PYTHONUNBUFFERED=1

# uid 1000, created rather than reused: the slim base has no non-root account
# and the process has no reason to be root. /models must be writable by it or
# the first-run download fails.
RUN useradd --create-home --uid 1000 stt \
 && mkdir -p /models \
 && chown stt:stt /models
USER stt

VOLUME ["/models"]
EXPOSE 8000

# No HEALTHCHECK instruction — the OCI image spec has no field for one, so a
# --format oci build drops it silently. Pass it at run time instead:
#   podman run --health-cmd "python -c \
#     \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')\"" ...
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
