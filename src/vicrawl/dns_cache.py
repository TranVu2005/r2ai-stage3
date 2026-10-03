"""Process-wide TTL cache for socket.getaddrinfo (httpx/asyncio resolve through it)."""
from __future__ import annotations

from r2ai.paths import ROOT

import socket
import threading
import time

_orig = None
_cache: dict = {}
_lock = threading.Lock()


def install(ttl: float = 300.0, clock=time.monotonic):
    global _orig
    if _orig is not None:
        return
    _orig = socket.getaddrinfo
    real = _orig

    def cached(host, port, family=0, type=0, proto=0, flags=0):
        key = (host, port, family, type, proto, flags)
        now = clock()
        with _lock:
            hit = _cache.get(key)
            if hit and now - hit[0] < ttl:
                return list(hit[1])
        res = real(host, port, family, type, proto, flags)   # failures propagate, never cached
        with _lock:
            _cache[key] = (now, res)
        return list(res)

    socket.getaddrinfo = cached


def uninstall():
    global _orig
    if _orig is not None:
        socket.getaddrinfo = _orig
        _orig = None
    clear()


def clear():
    with _lock:
        _cache.clear()
