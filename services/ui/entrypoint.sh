#!/bin/sh
# Take ownership of any mounted directory, then drop privileges.
#
# The shape is the gateway's, and for the same reason: there is no TLS here and
# no ${PREFIX}_TLS_CERT to honour, so voice-entrypoint.sh's TLS half would be
# dead code in this image. What is NOT dead here, unlike in the gateway, is the
# loop: this service is the only writer of the reference-clip volume, and a
# bind mount arrives with the HOST directory's ownership, which overrides
# whatever the image set. On a NAS that usually means root, and a container
# running as uid 1000 then cannot write the clip someone just recorded — with
# the failure landing on the user as "could not write the clip: Permission
# denied" after they have already spoken into a microphone.
#
# The image cannot fix this at build time, because the mount does not exist
# yet.
set -eu

if [ "$(id -u)" = "0" ]; then
    for d in ${UI_VOLUMES:-/voices}; do
        [ -d "$d" ] || continue
        if [ ! -w "$d" ] || [ "$(stat -c %u "$d")" != "1000" ]; then
            echo "entrypoint: taking ownership of $d for uid 1000"
            chown -R 1000:1000 "$d"
        fi
    done
    # --inh-caps=-all so nothing downstream can regain a capability the drop
    # was meant to remove. That matters more here than in the siblings: this is
    # the one image that spawns a subprocess on a URL a browser chose.
    exec setpriv --reuid=1000 --regid=1000 --init-groups --inh-caps=-all "$@"
fi

# Already unprivileged: `user: 1000` in compose, or a rootless runtime. Nothing
# to drop, and the chown above would have failed anyway.
exec "$@"
