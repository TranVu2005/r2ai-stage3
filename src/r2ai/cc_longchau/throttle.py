"""Self-adjusting request rate for the CC fetch (one worker).

Throttle events = HTTP 503/429 answers. Rules (counted over attempts since the last rate change):
  - more than DOWN_RATE of the attempts of the last DOWN_WINDOW_S seconds were throttled (and >= MIN_EVENTS attempts): rps / 2, floor FLOOR;
  - less than UP_RATE throttled over a full UP_WINDOW_S seconds since the last change (and >= MIN_UP_EVENTS attempts): rps * UP_FACTOR, ceiling CEIL.
The rate survives restarts (state/cc_longchau/throttle.json); the window does not.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

FLOOR, CEIL = 0.1, 0.6
DOWN_WINDOW_S, DOWN_RATE, MIN_EVENTS = 600, 0.30, 5
UP_WINDOW_S, UP_RATE, MIN_UP_EVENTS, UP_FACTOR = 1800, 0.05, 10, 1.25
THROTTLED = (429, 503)


class AutoRate:
    def __init__(self, rps=CEIL, *, floor=FLOOR, ceil=CEIL, path=None, clock=time.time):
        self.floor, self.ceil, self.clock = floor, ceil, clock
        self.path = Path(path) if path else None
        self.rps = min(max(rps, floor), ceil)
        saved = self._load()
        if saved is not None:
            self.rps = min(max(saved, floor), min(ceil, max(rps, floor)))   # never above the rps asked for on the command line
        self.changed_at = clock()
        self.events: deque = deque()                                        # (time, throttled)
        self.changes: list = []                                             # (time, old, new, why) of this run

    def _load(self):
        try:
            return float(json.loads(self.path.read_text('utf-8'))['rps']) if self.path and self.path.exists() else None
        except (OSError, ValueError, KeyError):
            return None

    def _save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'rps': self.rps, 'changed_at': self.changed_at}), encoding='utf-8')
        os.replace(tmp, self.path)

    def window(self, seconds):
        """(attempts, throttled share) of the last `seconds`, never reaching before the last rate change."""
        since = max(self.clock() - seconds, self.changed_at)
        ev = [t for ts, t in self.events if ts >= since]
        return len(ev), (sum(ev) / len(ev) if ev else 0.0)

    def recent(self, seconds=DOWN_WINDOW_S):
        """Like window() but ignoring the last change (what the heartbeat reports)."""
        since = self.clock() - seconds
        ev = [t for ts, t in self.events if ts >= since]
        return len(ev), (sum(ev) / len(ev) if ev else 0.0)

    def _set(self, new, why):
        self.changes.append((self.clock(), self.rps, new, why))
        self.rps, self.changed_at = new, self.clock()
        self._save()

    def record(self, status):
        now = self.clock()
        self.events.append((now, status in THROTTLED))
        while self.events and self.events[0][0] < now - UP_WINDOW_S - DOWN_WINDOW_S:
            self.events.popleft()
        n, r = self.window(DOWN_WINDOW_S)
        if n >= MIN_EVENTS and r > DOWN_RATE and self.rps > self.floor:
            self._set(max(self.floor, self.rps / 2), f'throttled {r:.0%} of {n} attempts in {DOWN_WINDOW_S // 60} min')
            return
        if now - self.changed_at >= UP_WINDOW_S and self.rps < self.ceil:
            n, r = self.window(UP_WINDOW_S)
            if n >= MIN_UP_EVENTS and r < UP_RATE:
                self._set(min(self.ceil, round(self.rps * UP_FACTOR, 4)), f'throttled {r:.0%} of {n} attempts in {UP_WINDOW_S // 60} min')
