#!/bin/sh
# Fix ownership of the model volume, then drop privileges.
#
# A bind mount arrives with the HOST directory's ownership, which overrides
# whatever the image set. On TrueNAS and most NAS platforms that means root,
# and a container running as uid 1000 then cannot write the models it is
# supposed to download on first start. The image cannot fix this at build
# time — the mount does not exist yet.
#
# So: start as root, chown only if needed, and immediately become `stt`. If
# the container was already started as a non-root user (user: in compose, or
# a platform that forces one), skip straight to exec — there is nothing to
# fix and nothing to drop.
set -eu

# TLS is opt-in and never invented. Point STT_TLS_CERT and STT_TLS_KEY at real
# PEM files and uvicorn serves HTTPS; leave them unset and it serves plain HTTP
# as before. Nothing here generates a self-signed certificate: one that appears
# by magic is one every client is told to stop validating, which is worse than
# the plain HTTP it replaced.
#
# Only uvicorn is given the flags. The image is also run with other commands —
# the CI import check runs `python -c` against it — and appending unknown
# arguments to those would break them.
if [ "${1:-}" = "uvicorn" ]; then
    if [ -n "${STT_TLS_CERT:-}" ] && [ -n "${STT_TLS_KEY:-}" ]; then
        echo "entrypoint: serving HTTPS with $STT_TLS_CERT"
        set -- "$@" --ssl-certfile "$STT_TLS_CERT" --ssl-keyfile "$STT_TLS_KEY"
    elif [ -n "${STT_TLS_CERT:-}" ] || [ -n "${STT_TLS_KEY:-}" ]; then
        # Half-configured TLS silently serving HTTP is the failure worth
        # shouting about: everything works, and nothing is encrypted.
        echo "entrypoint: STT_TLS_CERT and STT_TLS_KEY must BOTH be set;" \
             "serving plain HTTP" >&2
    fi
fi

if [ "$(id -u)" = "0" ]; then
    if [ ! -w /models ] || [ "$(stat -c %u /models)" != "1000" ]; then
        echo "entrypoint: taking ownership of /models for uid 1000"
        chown -R 1000:1000 /models
    fi
    exec setpriv --reuid=1000 --regid=1000 --init-groups --inh-caps=-all "$@"
fi

exec "$@"
