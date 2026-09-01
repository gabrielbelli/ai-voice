#!/bin/sh
# Fix ownership of the model volume, add TLS if asked, then drop privileges.
#
# A bind mount arrives with the HOST directory's ownership, which overrides
# whatever the image set. On a NAS that usually means root, and a container
# running as uid 1000 then cannot write the model it downloads on first start.
# The image cannot fix this at build time — the mount does not exist yet.
set -eu

# TLS lives here rather than in the application because uvicorn takes the
# certificate as command-line flags, and CMD is what the deployment overrides.
# Both halves are required: a certificate without its key does not fail
# usefully, it fails at bind time with the container already reported healthy.
#
# Nothing generates a certificate if none is supplied. A cert that appears by
# magic is one nobody validates, and it would teach every client on this LAN to
# pass --insecure permanently. Mount a real one, or terminate in front of this.
if [ -n "${TTS_TLS_CERT:-}" ] && [ -n "${TTS_TLS_KEY:-}" ]; then
    if [ "${1:-}" = "uvicorn" ]; then
        set -- "$@" --ssl-certfile "$TTS_TLS_CERT" --ssl-keyfile "$TTS_TLS_KEY"
        echo "entrypoint: serving HTTPS with $TTS_TLS_CERT"
    else
        echo "entrypoint: TLS is configured but the command is not uvicorn; ignoring"
    fi
elif [ -n "${TTS_TLS_CERT:-}" ] || [ -n "${TTS_TLS_KEY:-}" ]; then
    echo "entrypoint: only one of TTS_TLS_CERT/TTS_TLS_KEY is set; serving plain HTTP"
fi

if [ "$(id -u)" = "0" ]; then
    for d in /models /output; do
        if [ ! -w "$d" ] || [ "$(stat -c %u "$d")" != "1000" ]; then
            echo "entrypoint: taking ownership of $d for uid 1000"
            chown -R 1000:1000 "$d"
        fi
    done
    exec setpriv --reuid=1000 --regid=1000 --init-groups --inh-caps=-all "$@"
fi

exec "$@"
