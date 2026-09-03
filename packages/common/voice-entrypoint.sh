#!/bin/sh
# Fix ownership of the mounted volumes, wire up TLS if asked, drop privileges.
#
# Shipped by voice-common and installed to /usr/local/bin by setuptools
# `scripts`, so it rides the same pin as the Python code. TLS logic and auth
# logic can then never be at different versions inside one image, which is the
# specific way three copies of this file ended up with three different holes:
#
#   stt-stack   warned and served plain HTTP when only one of CERT/KEY was set
#   tts-stack   had no command check at all, and appended --ssl-certfile to
#               whatever CMD it was given
#   tts-stack   alone tested the key as uid 1000 rather than as root, and
#               alone distinguished a setpriv failure from an unreadable file
#
# This script is the union of the strictest rule from each.
#
# Parameterised by two variables, both read from the image's ENV rather than
# from the operator's:
#   VOICE_TLS_PREFIX   STT or TTS. Chooses ${PREFIX}_TLS_CERT and _TLS_KEY, so
#                      no operator-visible variable had to be renamed for a
#                      service to adopt this.
#   VOICE_CHOWN_DIRS   space-separated list, e.g. "/models" or "/models /output"
set -eu

PREFIX="${VOICE_TLS_PREFIX:-TTS}"
CHOWN_DIRS="${VOICE_CHOWN_DIRS:-/models}"
UID_TARGET="${VOICE_UID:-1000}"
GID_TARGET="${VOICE_GID:-1000}"

# eval rather than an indirect expansion, which is a bashism; this runs under
# /bin/sh in a slim image.
eval "CERT=\${${PREFIX}_TLS_CERT:-}"
eval "KEY=\${${PREFIX}_TLS_KEY:-}"

# TLS is opt-in and never invented. Nothing here generates a self-signed
# certificate: one that appears by magic is one every client is taught to stop
# validating, and a client taught to skip verification keeps skipping it
# against the real certificate too. Mount a real one, or terminate at a proxy.
#
# Every path out of a partial configuration is `exit 1`. The only job this
# feature has is keeping bearer tokens off the LAN, and both of the older
# fallbacks printed a line and then served plain HTTP anyway. That is invisible
# to the operator, who reads "TLS is configured" from their own compose file
# and believes it — while a key crosses the network in the clear on every
# request. Failing to start is the loud failure; serving plaintext is the one
# that hides.
if [ -n "$CERT" ] || [ -n "$KEY" ]; then
    if [ -z "$CERT" ] || [ -z "$KEY" ]; then
        echo "entrypoint: ${PREFIX}_TLS_CERT and ${PREFIX}_TLS_KEY must both" \
             "be set; refusing to serve plain HTTP with TLS half configured" >&2
        exit 1
    fi

    # The flags below are uvicorn's. The image is also run with other commands
    # — CI runs `python -c` against it — and appending unknown arguments to
    # those would break them. But the check is INSIDE this block, not around
    # it: gating the whole block on the command meant a compose `command:`
    # override, `python -m uvicorn` included, skipped every line of it, with
    # fully configured TLS, plain HTTP, and not one word on stderr.
    if [ "${1:-}" != "uvicorn" ]; then
        echo "entrypoint: TLS is configured but the command is '${1:-}', not" \
             "uvicorn. Only uvicorn is given the certificate, so this would" \
             "serve plain HTTP. Drop the command override, or unset" \
             "${PREFIX}_TLS_CERT and ${PREFIX}_TLS_KEY to serve HTTP on" \
             "purpose." >&2
        exit 1
    fi

    # A KEY THE CONTAINER CANNOT READ IS THE NORMAL CASE, not an error to
    # report. TrueNAS keeps its certificates in /etc/certificates with the key
    # 0400 root:root, which is correct: loosening it so a container can read it
    # would weaken the file every service on the host shares.
    #
    # So when we are root and the target uid cannot read the key, copy the pair
    # somewhere private and hand uvicorn that instead. Copied on every start
    # rather than once by hand, so a renewed certificate is picked up by a
    # restart -- a copy made manually goes stale silently, and the failure
    # arrives as an expired certificate months later.
    #
    # Only when the ORIGINAL is unreadable, so a deployment that already mounts
    # a readable key keeps using it untouched.
    if [ "$(id -u)" = "0" ] && [ -r "$KEY" ] \
       && ! setpriv "--reuid=$UID_TARGET" "--regid=$GID_TARGET" \
                    --init-groups test -r "$KEY" 2>/dev/null; then
        private="${VOICE_TLS_PRIVATE_DIR:-/run/voice-tls}"
        mkdir -p "$private" || exit 1
        cp "$CERT" "$private/tls.crt" || exit 1
        cp "$KEY" "$private/tls.key" || exit 1
        chown "$UID_TARGET:$GID_TARGET" "$private/tls.crt" "$private/tls.key" || exit 1
        chmod 0400 "$private/tls.key" || exit 1
        chmod 0444 "$private/tls.crt" || exit 1
        chmod 0500 "$private" || exit 1
        chown "$UID_TARGET:$GID_TARGET" "$private" || exit 1
        echo "entrypoint: $KEY is root-only; using a private copy in $private"
        CERT="$private/tls.crt"
        KEY="$private/tls.key"
    fi

    for f in "$CERT" "$KEY"; do
        if [ "$(id -u)" = "0" ]; then
            # Tested as the target uid, not as root: a key mounted 0600
            # root:root reads fine here as root and not at all in the process
            # that will open it, and uvicorn's failure at that point is a
            # traceback rather than a sentence.
            #
            # setpriv's own failures — missing binary, no CAP_SETUID, no such
            # uid in this image — also exit non-zero, and reporting one of
            # those as "not readable" sends an operator to inspect a file
            # whose permissions were fine all along. `test -r` says nothing on
            # failure and setpriv always explains itself, so its output is the
            # thing that tells the two apart.
            why=$(setpriv "--reuid=$UID_TARGET" "--regid=$GID_TARGET" \
                          --init-groups test -r "$f" 2>&1) || {
                if [ -n "$why" ]; then
                    echo "entrypoint: could not check $f as uid" \
                         "$UID_TARGET: $why" >&2
                else
                    echo "entrypoint: $f is not readable by uid $UID_TARGET" >&2
                fi
                exit 1
            }
        elif [ ! -r "$f" ]; then
            echo "entrypoint: $f is not readable" >&2
            exit 1
        fi
    done

    echo "entrypoint: serving HTTPS with $CERT"
    set -- "$@" --ssl-certfile "$CERT" --ssl-keyfile "$KEY"
fi

# A bind mount arrives with the HOST directory's ownership, which overrides
# whatever the image set. On a NAS that usually means root, and a container
# running as uid 1000 then cannot write the model it downloads on first start.
# The image cannot fix this at build time — the mount does not exist yet.
#
# If the container was started as a non-root user already (user: in compose,
# or a platform that forces one), there is nothing to fix and nothing to drop:
# skip straight to exec.
if [ "$(id -u)" = "0" ]; then
    for d in $CHOWN_DIRS; do
        # A DIRECTORY THAT IS NOT THERE IS NOT AN ERROR, it is a volume nobody
        # mounted. Without this line the container dies at boot instead of
        # starting: `[ ! -w "$d" ]` is true for a path that does not exist, so
        # the chown runs, fails, and `set -e` takes the process with it. A
        # service whose volume is optional -- voice-ui's /voices, which only
        # exists once someone clones a voice -- could not start at all.
        #
        # Found in the two copies of this file that had drifted; it was the one
        # thing they did better than the original.
        [ -d "$d" ] || continue
        if [ ! -w "$d" ] || [ "$(stat -c %u "$d")" != "$UID_TARGET" ]; then
            echo "entrypoint: taking ownership of $d for uid $UID_TARGET"
            chown -R "$UID_TARGET:$GID_TARGET" "$d"
        fi
    done
    exec setpriv "--reuid=$UID_TARGET" "--regid=$GID_TARGET" \
                 --init-groups --inh-caps=-all "$@"
fi

exec "$@"
