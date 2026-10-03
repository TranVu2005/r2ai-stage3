from r2ai.paths import ROOT

import socket

from vicrawl import dns_cache


def test_getaddrinfo_cached_until_ttl_or_clear(monkeypatch):
    calls = []

    def fake(host, port, *a, **k):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('1.2.3.4', port or 0))]

    now = [0.0]
    monkeypatch.setattr(socket, 'getaddrinfo', fake)
    monkeypatch.setattr(dns_cache, '_orig', None)
    dns_cache.install(ttl=300, clock=lambda: now[0])
    try:
        socket.getaddrinfo('a.vn', 443)
        socket.getaddrinfo('a.vn', 443)
        assert calls == ['a.vn']
        now[0] = 301
        socket.getaddrinfo('a.vn', 443)
        assert calls == ['a.vn', 'a.vn']
        dns_cache.clear()
        socket.getaddrinfo('a.vn', 443)
        assert len(calls) == 3
    finally:
        dns_cache.uninstall()


def test_failures_are_not_cached(monkeypatch):
    n = [0]

    def fake(host, port, *a, **k):
        n[0] += 1
        raise socket.gaierror('nope')

    monkeypatch.setattr(socket, 'getaddrinfo', fake)
    monkeypatch.setattr(dns_cache, '_orig', None)
    dns_cache.install(ttl=300)
    try:
        for _ in range(2):
            try:
                socket.getaddrinfo('b.vn', 80)
            except socket.gaierror:
                pass
        assert n[0] == 2
    finally:
        dns_cache.uninstall()
