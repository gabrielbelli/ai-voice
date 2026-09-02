"""What a pasted URL has to survive before anything is done with it.

READ THIS BEFORE RELAXING ANYTHING BELOW.

This box hosts the user's other services. MeTube on 30097, Gitea on 30008,
UniFi on 30072-75, rackula on 30323, the TrueNAS middleware itself on 443, and
a handful of things bound to 127.0.0.1 that were never meant to leave the
machine. A route that takes a URL from a browser and makes an outbound request
with it is, unmodified, a way to ask this container to fetch any of those and
tell the caller what came back. That is the whole of SSRF, and on a NAS the
interesting targets are all on the near side of the firewall.

THREE LAYERS, IN THIS ORDER, AND THE ORDER IS THE DESIGN.

  1. This module. Stdlib only, no network, ~40 lines: scheme, userinfo, port,
     then getaddrinfo and a check of EVERY address the name resolves to.
  2. MeTube's own url_guard.validate_url, which runs inside its POST /add. It
     is a serious two-layer guard -- ingress validation plus a connect-time
     getaddrinfo hook installed in the download subprocess, so it also covers
     redirects and DNS rebinding during the download itself. Verified live:
     `http://localhost:8080/secret` comes back "Refusing to fetch internal
     host", `http://10.0.0.5/x` comes back "Refusing to fetch internal
     address". ALLOW_PRIVATE_ADDRESSES defaults false and is unset here.
  3. Only then the yt-dlp probe (app/probe.py), and only on a URL MeTube has
     already accepted. It never runs on a URL the first two layers rejected.

Layer 1 exists even though layer 2 is better, because layer 2 lives in another
app that the operator can reconfigure without touching this repository, and
because a URL our own guard hates should never become a log line in someone
else's service either.

WHAT IS BLOCKED, AND WHY EACH ONE.

  scheme other than http/https   file:, gopher:, dict: and ftp: are all
                                 fetchable by some client in this chain and
                                 none of them is a media URL.
  userinfo (user:pass@host)      the classic parser split: several URL
                                 libraries disagree about where the host
                                 starts when an @ is present, and the whole
                                 attack is making two parsers disagree.
  ports other than 80 and 443    an internal service is almost never on 80 or
                                 443 on this box; every app listed above is on
                                 a 300xx port. This single rule removes most
                                 of the LAN as a target.
  loopback         127/8, ::1    the container's own listeners
  private          10/8, 172.16/12, 192.168/16, fc00::/7   the LAN, the NAS,
                                 and every other container on it
  link-local       169.254/16, fe80::/10   which includes 169.254.169.254,
                                 the cloud metadata address -- not applicable
                                 on a NAS today, and the day this moves it is
                                 the first thing anyone tries
  CGNAT            100.64/10      carrier NAT, and Tailscale's range
  multicast, reserved, unspecified, and IPv4-mapped IPv6 of any of the above

  by name          localhost, *.localhost, metadata.google.internal, and the
                   TrueNAS .local mDNS suffix -- belt and braces over the
                   address check, which already covers them, for the case
                   where resolution is the thing that is lying.

WHAT IS STILL OPEN, stated rather than hidden. yt-dlp's extraction follows
redirects, and neither MeTube's ingress guard nor this one covers a redirect
DURING extraction to an internal host -- MeTube documents that exact limitation
in url_guard.py's own docstring, and notes that curl_cffi's native resolver
bypasses even its socket guard. The impact is BLIND SSRF: the probe's output is
parsed into five scalars, none of which is a response body, nothing is written
to disk, and nothing is returned to the caller but a title, a duration, a size
and two booleans. The real backstop is not code -- the gateway and this
container have no business reaching the NAS's other services at all, and an
egress rule on the ai-voice app is the fix. That belongs in the deployment, and
it is written down in this service's README so it cannot be assumed.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

__all__ = ["GuardError", "check"]

ALLOWED_SCHEMES = frozenset({"http", "https"})
ALLOWED_PORTS = frozenset({80, 443})
BLOCKED_NAMES = ("localhost", "metadata.google.internal")
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")


class GuardError(ValueError):
    """The URL is refused. The message is shown to the user verbatim."""


def _forbidden(address: str) -> str | None:
    """Why this literal address may not be fetched, or None if it may."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:  # pragma: no cover - getaddrinfo returns literals
        return "it did not resolve to an IP address"

    # An IPv4 address wrapped in IPv6 (::ffff:127.0.0.1) is the same host and
    # is_private on the wrapper is not the same question as is_private on the
    # address inside it. Unwrap first, then ask once.
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped  # type: ignore[assignment]

    if ip.is_loopback:
        return f"{ip} is loopback"
    if ip.is_link_local:
        # Covers 169.254.169.254, the cloud metadata address.
        return f"{ip} is link-local"
    if ip.is_private:
        # ipaddress folds ULA (fc00::/7), 10/8, 172.16/12 and 192.168/16 into
        # this one predicate, and also 100.64/10 since Python 3.13.
        return f"{ip} is a private address"
    if ip.is_multicast:
        return f"{ip} is multicast"
    if ip.is_reserved or ip.is_unspecified:
        return f"{ip} is reserved"
    if not ip.is_global:
        # The catch-all, and the reason it is last rather than first: the
        # named predicates above produce a message that says WHICH rule fired,
        # which is what someone debugging a refused link needs. This one
        # closes what they miss -- 100.64/10 in particular, whose is_private
        # answer has changed between CPython releases, plus 192.0.0.0/24,
        # 198.18/15, the documentation ranges and 2001:db8::/32.
        return f"{ip} is not a globally routable address"
    return None


def check(url: str, *, resolve=None) -> str:
    """Return the URL if it may be handed onward, or raise GuardError.

    `resolve` is injectable so the tests can assert the address rules without
    depending on what DNS says today, and so a test can prove that a name
    resolving to a private address is refused without needing such a name.

    It defaults to None and is looked up below rather than being written as
    `resolve=socket.getaddrinfo`: a default argument is evaluated once, at
    definition, so the latter captures the original function and no later
    patch of socket.getaddrinfo — a test's, or a runtime's — is ever seen.
    """
    if resolve is None:
        resolve = socket.getaddrinfo
    url = (url or "").strip()
    if not url:
        raise GuardError("no URL given")
    if len(url) > 2048:
        # Not a security boundary, a sanity one: nothing legitimate is longer,
        # and the value ends up in a subprocess argument list.
        raise GuardError("that URL is implausibly long")

    parts = urlsplit(url)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise GuardError(
            f"only http and https links can be fetched, not {parts.scheme or 'that'}:")
    if parts.username or parts.password:
        raise GuardError(
            "links carrying a username or password are refused: URL parsers "
            "disagree about where the host starts once an @ is present")

    host = (parts.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise GuardError("that URL has no host in it")
    if host in BLOCKED_NAMES or host.endswith(BLOCKED_SUFFIXES):
        raise GuardError(f"refusing to fetch the internal host {host!r}")

    try:
        port = parts.port
    except ValueError as exc:
        raise GuardError(f"that URL has an unusable port: {exc}") from None
    port = port if port is not None else (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise GuardError(
            f"only ports 80 and 443 are fetched, not {port}. Every service on "
            "this host that is not meant to be reachable from here is on some "
            "other port, and this rule is what keeps them that way.")

    # Resolved BEFORE anything fetches it, and every answer is checked rather
    # than the first: a name with one public A record and one 10.x A record is
    # a rebinding attack with the work already done for it.
    try:
        infos = resolve(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise GuardError(f"{host} does not resolve ({exc.strerror or exc})") from None
    if not infos:
        raise GuardError(f"{host} does not resolve to any address")

    for info in infos:
        address = info[4][0]
        why = _forbidden(address)
        if why is not None:
            raise GuardError(f"refusing to fetch {host}: {why}")

    return url
