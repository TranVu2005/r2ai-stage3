from r2ai.paths import ROOT

from vicrawl.tuner import Tuner, start_rate


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def feed(t, n, latency=0.2, status=200, error=False):
    ev = []
    for _ in range(n):
        ev += t.record(latency, status, error)
    return ev


def test_start_rate_is_75_percent_of_learned_clamped():
    assert start_rate(None, cap=4) == 1.0
    assert start_rate(4.0, cap=4) == 3.0
    assert start_rate(0.1, cap=4) == 0.25      # floor
    assert start_rate(8.0, cap=4) == 4.0       # never above cap


def test_baseline_learned_from_first_100_and_persisted_value_used():
    t = Tuner(cap=4, max_conns=2, clock=Clock())
    assert t.baseline_p95 is None
    feed(t, 100, latency=0.3)
    assert abs(t.baseline_p95 - 0.3) < 1e-9
    t2 = Tuner(cap=4, max_conns=2, baseline_p95=0.5, clock=Clock())
    feed(t2, 100, latency=9.0)   # already known, must not be replaced
    assert t2.baseline_p95 == 0.5


def test_rate_up_half_step_after_500_clean_requests():
    t = Tuner(cap=4, max_conns=2, rate=1.0, baseline_p95=0.3, clock=Clock())
    ev = feed(t, 499, latency=0.3)
    assert t.rate == 1.0 and not ev
    ev = feed(t, 1, latency=0.3)
    assert t.rate == 1.5 and ev[0]['event'] == 'rate_up'
    feed(t, 500, latency=0.3)
    assert t.rate == 2.0


def test_rate_capped():
    t = Tuner(cap=1.5, max_conns=2, rate=1.0, baseline_p95=0.3, clock=Clock())
    feed(t, 1500, latency=0.3)
    assert t.rate == 1.5


def test_no_increase_when_latency_p95_above_1_5x_baseline():
    t = Tuner(cap=4, max_conns=2, rate=1.0, baseline_p95=0.2, clock=Clock())
    feed(t, 500, latency=0.35)
    assert t.rate == 1.0


def test_no_increase_if_window_had_429_or_503():
    t = Tuner(cap=4, max_conns=2, rate=2.0, baseline_p95=0.2, clock=Clock())
    clock = Clock()
    t = Tuner(cap=4, max_conns=2, rate=2.0, baseline_p95=0.2, clock=clock)
    feed(t, 10, status=429)
    r = t.rate
    clock.t += 100
    feed(t, 500, latency=0.2)
    assert t.rate == r


def test_429_halves_and_pauses_60s_once_per_pause():
    clock = Clock()
    t = Tuner(cap=4, max_conns=2, rate=4.0, baseline_p95=0.2, clock=clock)
    ev = t.record(0.2, 429, False)
    assert t.rate == 2.0 and t.paused_until == clock.t + 60
    assert ev[0]['event'] == 'rate_down'
    t.record(0.2, 503, False)          # in-flight stragglers while paused: no second halving
    assert t.rate == 2.0
    clock.t += 61
    t.record(0.2, 503, False)
    assert t.rate == 1.0


def test_rate_floor():
    clock = Clock()
    t = Tuner(cap=4, max_conns=2, rate=0.3, clock=clock)
    t.record(0.2, 429, False)
    assert t.rate == 0.25


def test_error_rate_over_20_percent_of_last_200_halts():
    t = Tuner(cap=4, max_conns=2, clock=Clock())
    ev = feed(t, 150, latency=0.1)
    assert not t.halted
    ev = feed(t, 50, latency=0.1, status=None, error=True)   # 50/200 = 25%
    assert t.halted and any(e['event'] == 'halt' for e in ev)


def test_error_rate_exactly_at_threshold_does_not_halt():
    t = Tuner(cap=4, max_conns=2, clock=Clock())
    feed(t, 160, latency=0.1)
    feed(t, 40, latency=0.1, status=None, error=True)        # 20% exactly
    assert not t.halted


def test_not_found_is_not_an_error():
    t = Tuner(cap=4, max_conns=2, clock=Clock())
    feed(t, 300, status=404)
    assert not t.halted


def test_slow_domain_grows_connections_not_rate():
    t = Tuner(cap=1.0, max_conns=4, rate=1.0, baseline_p95=4.0, clock=Clock())
    feed(t, 500, latency=3.5)
    assert t.rate == 1.0 and t.conns >= 2
    feed(t, 1500, latency=3.5)
    assert t.conns == 4 and t.rate == 1.0


def test_conns_never_exceed_max_conns():
    t = Tuner(cap=8, max_conns=3, rate=8.0, baseline_p95=2.0, clock=Clock())
    feed(t, 3000, latency=3.5)
    assert t.conns == 3


def test_interval_honours_crawl_delay():
    t = Tuner(cap=4, max_conns=2, rate=4.0, crawl_delay=2.0, clock=Clock())
    assert t.interval() == 2.0
    t2 = Tuner(cap=4, max_conns=2, rate=4.0, clock=Clock())
    assert t2.interval() == 0.25


def test_twelve_consecutive_failures_halt_early():
    t = Tuner(cap=4, max_conns=2, clock=Clock())
    feed(t, 11, status=503)
    assert not t.halted
    feed(t, 1, status=503)
    assert t.halted and 'consecutive' in t.halt_reason


def test_success_resets_consecutive_failure_counter():
    t = Tuner(cap=4, max_conns=2, clock=Clock())
    for _ in range(5):
        feed(t, 11, status=500)
        feed(t, 1, status=200)
    assert not t.halted or 'consecutive' not in t.halt_reason


def test_428_too_many_requests_backs_off_like_429():
    clock = Clock()
    t = Tuner(cap=4, max_conns=2, rate=4.0, baseline_p95=0.2, clock=clock)
    ev = t.record(0.03, 428, False)
    assert t.rate == 2.0 and t.paused_until == clock.t + 60
    assert ev[0] == {'event': 'rate_down', 'old': 4.0, 'new': 2.0, 'http': 428}
    clock.t += 100
    feed(t, 500, latency=0.2)          # the window that saw the 428 never raises the rate
    assert t.rate == 2.0
