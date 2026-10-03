"""Per-domain adaptive rate / concurrency. Pure logic, no I/O (state persisted by the engine)."""
from __future__ import annotations

from r2ai.paths import ROOT

import math
import time
from collections import deque

MIN_RATE = 0.25
STEP = 0.5
WINDOW = 500
BASELINE_N = 100
ERR_WINDOW = 200
ERR_MIN_SAMPLES = 50
ERR_THRESHOLD = 0.20
PAUSE_S = 60.0
SLOW_P50 = 3.0
SLOW_MAX_CONNS = 4
MAX_CONSECUTIVE_ERRORS = 12
RATE_LIMIT_CODES = (428, 429, 503)     # 428: Varnish "Too Many Requests" (dantri.com.vn)


def start_rate(learned: float | None, cap: float) -> float:
    if not learned:
        return min(1.0, cap)
    return max(MIN_RATE, min(cap, 0.75 * learned))


def _pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)] if s else 0.0


class Tuner:
    def __init__(self, cap: float, max_conns: int, rate: float = 1.0, baseline_p95: float | None = None, crawl_delay: float | None = None, clock=time.monotonic, pause_s: float = PAUSE_S):
        self.cap, self.max_conns, self.clock = cap, max(1, max_conns), clock
        self.rate = max(MIN_RATE, min(cap, rate))
        self.conns = 1
        self.baseline_p95 = baseline_p95
        self.crawl_delay = crawl_delay
        self.paused_until = 0.0
        self.pause_s = pause_s
        self.consecutive_errors = 0
        self.halted = False
        self.halt_reason = ''
        self._win_n = 0
        self._win_lat: list[float] = []
        self._win_bad = False
        self._base_lat: list[float] = []
        self._recent = deque(maxlen=100)
        self._errs = deque(maxlen=ERR_WINDOW)
        self._since_conn_check = 0

    def interval(self) -> float:
        return max(1.0 / self.rate, self.crawl_delay or 0.0)

    @property
    def p50(self) -> float:
        return _pct(self._recent, 0.5) if self._recent else 0.0

    @staticmethod
    def is_error(http_status, error: bool) -> bool:
        if error or http_status is None:
            return True
        return http_status >= 500 or (http_status >= 400 and http_status not in (404, 410))

    def record(self, latency: float, http_status, error: bool) -> list[dict]:
        events: list[dict] = []
        is_err = self.is_error(http_status, error)
        self._errs.append(is_err)
        self.consecutive_errors = self.consecutive_errors + 1 if is_err else 0
        self._win_n += 1
        if not error and http_status is not None:
            self._recent.append(latency)
            self._win_lat.append(latency)
            if self.baseline_p95 is None:
                self._base_lat.append(latency)
                if len(self._base_lat) >= BASELINE_N:
                    self.baseline_p95 = _pct(self._base_lat, 0.95)
                    events.append({'event': 'baseline', 'p95': self.baseline_p95})
        if http_status in RATE_LIMIT_CODES:
            self._win_bad = True
            if self.clock() >= self.paused_until:
                old, self.rate = self.rate, max(MIN_RATE, self.rate / 2)
                self.paused_until = self.clock() + self.pause_s
                self._win_n, self._win_lat = 0, []
                events.append({'event': 'rate_down', 'old': old, 'new': self.rate, 'http': http_status})
        self._since_conn_check += 1
        if self._since_conn_check >= 25:
            self._since_conn_check = 0
            events += self._adapt_conns()
        if self._win_n >= WINDOW:
            events += self._end_window()
        if not self.halted:
            if len(self._errs) >= ERR_MIN_SAMPLES and sum(self._errs) / len(self._errs) > ERR_THRESHOLD:
                self.halted, self.halt_reason = True, f'error_rate {sum(self._errs)}/{len(self._errs)} in last {len(self._errs)} requests'
            elif self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS and sum(self._errs) / len(self._errs) > 0.5:      # dead from the start / hard outage
                self.halted, self.halt_reason = True, f'error_rate: {self.consecutive_errors} consecutive failures'
            if self.halted:
                events.append({'event': 'halt', 'reason': self.halt_reason})
        return events

    def _adapt_conns(self) -> list[dict]:
        if not self._recent:
            return []
        need = min(self.max_conns, SLOW_MAX_CONNS if self.p50 > SLOW_P50 else self.max_conns, math.ceil(self.rate * self.p50 * 1.25))
        if need > self.conns:
            self.conns = need
            return [{'event': 'conns_up', 'conns': need}]
        return []

    def _end_window(self) -> list[dict]:
        ev: list[dict] = []
        bad, lat = self._win_bad, self._win_lat
        self._win_n, self._win_lat, self._win_bad = 0, [], False
        if bad or not lat or self.baseline_p95 is None:
            return ev
        if self.p50 > SLOW_P50:           # slow server: widen the pipe, keep the rate
            if self.conns < min(self.max_conns, SLOW_MAX_CONNS):
                self.conns += 1
                ev.append({'event': 'conns_up', 'conns': self.conns})
            return ev
        if _pct(lat, 0.95) <= 1.5 * self.baseline_p95 and self.rate < self.cap:
            old, self.rate = self.rate, min(self.cap, self.rate + STEP)
            ev.append({'event': 'rate_up', 'old': old, 'new': self.rate})
        return ev
