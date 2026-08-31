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

if [ "$(id -u)" = "0" ]; then
    if [ ! -w /models ] || [ "$(stat -c %u /models)" != "1000" ]; then
        echo "entrypoint: taking ownership of /models for uid 1000"
        chown -R 1000:1000 /models
    fi
    exec setpriv --reuid=1000 --regid=1000 --init-groups --inh-caps=-all "$@"
fi

exec "$@"
