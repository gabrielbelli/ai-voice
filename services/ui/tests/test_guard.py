"""The SSRF pre-filter, which is the one thing here that must not be wrong.

Every address rule is asserted against an injected resolver rather than against
whatever DNS says today: the point is to prove that a NAME resolving to a
private address is refused, and there is no public name that reliably does
that on someone else's network.
"""

from __future__ import annotations

import socket

import pytest

from app import guard


def resolver(*addresses):
    def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                 (address, port)) for address in addresses]
    return resolve


PUBLIC = resolver("93.184.216.34")


def refuse(url, resolve=PUBLIC):
    with pytest.raises(guard.GuardError) as caught:
        guard.check(url, resolve=resolve)
    return str(caught.value)


def test_a_public_https_url_passes():
    assert guard.check("https://example.com/watch?v=x", resolve=PUBLIC)


def test_only_http_and_https():
    assert "http and https" in refuse("file:///etc/passwd")
    assert "http and https" in refuse("gopher://example.com/")


def test_userinfo_is_refused_because_parsers_disagree_about_it():
    assert "username or password" in refuse("https://user:pw@example.com/")


def test_only_ports_80_and_443():
    # Every service on this NAS that is not meant to be reachable from the UI
    # container is on a 300xx port, which is what this rule is for.
    assert "ports 80 and 443" in refuse("http://example.com:30097/history")
    assert "ports 80 and 443" in refuse("http://example.com:8080/secret")


@pytest.mark.parametrize("address,why", [
    ("127.0.0.1", "loopback"),
    ("10.0.0.5", "private"),
    ("172.16.4.4", "private"),
    ("192.168.1.9", "private"),
    ("169.254.169.254", "link-local"),      # the cloud metadata address
    ("100.64.3.3", "not a globally routable"),  # CGNAT, and Tailscale's range
    ("::1", "loopback"),
    ("fc00::1", "private"),
    ("224.0.0.1", "multicast"),
    # 0.0.0.0/8 is inside CPython's private set rather than its reserved
    # one; what matters is that it is refused, not which clause caught it.
    ("0.0.0.0", "private"),
    ("240.1.2.3", "private"),
])
def test_every_internal_range_is_refused(address, why):
    assert why in refuse("https://anything.example/", resolver(address))


def test_an_ipv4_mapped_ipv6_address_is_unwrapped_first():
    # ::ffff:127.0.0.1 is the same host as 127.0.0.1, and is_loopback on the
    # wrapper is not the same question as is_loopback on what is inside it.
    assert "loopback" in refuse("https://anything.example/",
                                resolver("::ffff:127.0.0.1"))


def test_every_answer_is_checked_not_just_the_first():
    # One public A record and one 10.x A record is a rebinding attack with the
    # work already done for it.
    assert "private" in refuse("https://anything.example/",
                               resolver("93.184.216.34", "10.1.2.3"))


def test_internal_names_are_refused_before_resolution_is_even_asked():
    def explode(*args, **kwargs):
        raise AssertionError("resolution should not have been attempted")
    for host in ("http://localhost/", "http://x.localhost/",
                 "http://orko.local/", "http://metadata.google.internal/"):
        assert "internal host" in refuse(host, explode)


def test_a_name_that_does_not_resolve_says_so():
    def fail(*args, **kwargs):
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    assert "does not resolve" in refuse("https://nope.example/", fail)


def test_nothing_and_absurd_lengths_are_refused():
    assert "no URL" in refuse("")
    assert "implausibly long" in refuse("https://example.com/" + "a" * 2100)
