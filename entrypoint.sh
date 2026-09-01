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
#
# Every path out of a partial TLS configuration is `exit 1`, because the only
# job this feature has is keeping bearer tokens off the LAN. Both of the older
# fallbacks printed a line and then served plain HTTP: a cert with no key, and
# a CMD that was not uvicorn. Neither is visible to the operator, who reads
# "TLS is configured" from their own compose file and believes it — while a
# key travels the network in the clear on every request. Failing to start is
# the loud failure; serving plaintext is the one that hides.
#
# Nothing generates a certificate if none is supplied. A cert that appears by
# magic is one nobody validates, and it would teach every client on this LAN to
# pass --insecure permanently. Mount a real one, or terminate in front of this.
if [ -n "${TTS_TLS_CERT:-}" ] || [ -n "${TTS_TLS_KEY:-}" ]; then
    if [ -z "${TTS_TLS_CERT:-}" ] || [ -z "${TTS_TLS_KEY:-}" ]; then
        echo "entrypoint: TTS_TLS_CERT and TTS_TLS_KEY must both be set" >&2
        exit 1
    fi
    if [ "${1:-}" != "uvicorn" ]; then
        # The flags below are uvicorn's. Anything else cannot be given a
        # certificate here, so refuse rather than run it in the clear.
        echo "entrypoint: TLS is configured but the command is '${1:-}', not" \
             "uvicorn; refusing to serve plain HTTP" >&2
        exit 1
    fi
    set -- "$@" --ssl-certfile "$TTS_TLS_CERT" --ssl-keyfile "$TTS_TLS_KEY"
    echo "entrypoint: serving HTTPS with $TTS_TLS_CERT"
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
