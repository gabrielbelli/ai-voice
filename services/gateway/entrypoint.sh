#!/bin/sh
# Take ownership of any mounted directory, then drop privileges.
#
# The shape is the siblings': a bind mount arrives with the HOST directory's
# ownership, which overrides whatever the image set. On a NAS that usually
# means root, and a container running as uid 1000 then cannot write into it.
# The image cannot fix this at build time — the mount does not exist yet.
#
# This service has no volume, so GATEWAY_VOLUMES is empty and the loop below
# does nothing today. It is here rather than deleted because the alternative to
# an empty loop is a fourth entrypoint that looks different from the other
# three, and the day something does get mounted here — a socket directory, a
# key file — the difference is where the bug goes.
#
# There is no TLS handling, unlike the siblings. It is LAN traffic on one host;
# if this is ever exposed beyond the LAN that belongs to whatever reverse proxy
# already terminates TLS for the NAS, not to a bespoke implementation here.
set -eu

if [ "$(id -u)" = "0" ]; then
    for d in ${GATEWAY_VOLUMES:-}; do
        [ -d "$d" ] || continue
        if [ ! -w "$d" ] || [ "$(stat -c %u "$d")" != "1000" ]; then
            echo "entrypoint: taking ownership of $d for uid 1000"
            chown -R 1000:1000 "$d"
        fi
    done
    # --inh-caps=-all so nothing downstream can regain a capability the drop
    # was meant to remove.
    exec setpriv --reuid=1000 --regid=1000 --init-groups --inh-caps=-all "$@"
fi

# Already unprivileged: `user: 1000` in compose, or a rootless runtime. Nothing
# to drop, and the chown above would have failed anyway.
exec "$@"
