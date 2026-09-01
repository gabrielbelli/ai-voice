#!/bin/sh
# Fix ownership of the model volume, then drop privileges.
#
# A bind mount arrives with the HOST directory's ownership, which overrides
# whatever the image set. On a NAS that usually means root, and a container
# running as uid 1000 then cannot write the model it downloads on first start.
# The image cannot fix this at build time — the mount does not exist yet.
set -eu

if [ "$(id -u)" = "0" ]; then
    if [ ! -w /models ] || [ "$(stat -c %u /models)" != "1000" ]; then
        echo "entrypoint: taking ownership of /models for uid 1000"
        chown -R 1000:1000 /models
    fi
    exec setpriv --reuid=1000 --regid=1000 --init-groups --inh-caps=-all "$@"
fi

exec "$@"
