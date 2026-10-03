from r2ai.paths import ROOT

import time

from vicrawl.engine import compute_deadline


def test_hours():
    assert compute_deadline(3, None, 1000.0) == 1000.0 + 3 * 3600


def test_until_today_when_still_ahead_else_tomorrow():
    now = time.mktime((2026, 10, 2, 20, 0, 0, 0, 0, -1))
    assert compute_deadline(None, '23:30', now) - now == 3.5 * 3600
    later = compute_deadline(None, '08:15', now)
    assert later - now == 12.25 * 3600


def test_no_limit():
    assert compute_deadline(None, None, 5.0) is None
