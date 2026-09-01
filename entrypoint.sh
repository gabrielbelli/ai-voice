#!/bin/sh
# Fix ownership of the model volume, wire up TLS if asked, then drop privileges.
#
# A bind mount arrives with the HOST directory's ownership, which overrides
# whatever the image set. On a NAS that usually means root, and a container
# running as uid 1000 then cannot write the model it downloads on first start.
# The image cannot fix this at build time — the mount does not exist yet.
set -eu

# TLS is off unless both paths are given, and there is deliberately no
# self-signed fallback: a certificate that appears by magic is one nobody
# validates, and a client taught to skip verification keeps skipping it against
# the real certificate too. Use a real one, or terminate at a reverse proxy.
#
# The paths become uvicorn flags rather than being read in Python, so the CMD
# baked into the image stays exactly what it was for anyone already running it.
# Half a configuration exits rather than carrying on in the clear: a silent
# downgrade to plain HTTP is precisely the failure nobody notices until it is
# quoted back at them.
if [ -n "${TTS_TLS_CERT:-}" ] || [ -n "${TTS_TLS_KEY:-}" ]; then
    if [ -z "${TTS_TLS_CERT:-}" ] || [ -z "${TTS_TLS_KEY:-}" ]; then
        echo "entrypoint: TTS_TLS_CERT and TTS_TLS_KEY must both be set" >&2
        exit 1
    fi
    for f in "$TTS_TLS_CERT" "$TTS_TLS_KEY"; do
        # Tested as uid 1000, not as root: a key mounted 0600 root:root reads
        # fine here and not at all in the process that will open it, and
        # uvicorn's failure at that point is a traceback rather than a sentence.
        if [ "$(id -u)" = "0" ]; then
            # setpriv's own failures — missing binary, no CAP_SETUID, no uid
            # 1000 in this image — also exit non-zero, and reporting one of
            # those as "not readable" sends an operator to inspect a file
            # whose permissions were fine all along. `test -r` says nothing on
            # failure and setpriv always explains itself, so its output is the
            # thing that tells the two apart.
            why=$(setpriv --reuid=1000 --regid=1000 --init-groups test -r "$f" 2>&1) || {
                if [ -n "$why" ]; then
                    echo "entrypoint: could not check $f as uid 1000: $why" >&2
                else
                    echo "entrypoint: $f is not readable by uid 1000" >&2
                fi
                exit 1
            }
        elif [ ! -r "$f" ]; then
            echo "entrypoint: $f is not readable" >&2
            exit 1
        fi
    done
    echo "entrypoint: serving HTTPS with $TTS_TLS_CERT"
    set -- "$@" --ssl-certfile "$TTS_TLS_CERT" --ssl-keyfile "$TTS_TLS_KEY"
fi

if [ "$(id -u)" = "0" ]; then
    if [ ! -w /models ] || [ "$(stat -c %u /models)" != "1000" ]; then
        echo "entrypoint: taking ownership of /models for uid 1000"
        chown -R 1000:1000 /models
    fi
    exec setpriv --reuid=1000 --regid=1000 --init-groups --inh-caps=-all "$@"
fi

exec "$@"
