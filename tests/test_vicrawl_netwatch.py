from r2ai.paths import ROOT

import asyncio

from vicrawl.netwatch import NetWatch


class Fake:
    def __init__(self):
        self.t = 10_000.0
        self.online = True
        self.country = 'VN'
        self.resumed = 0
        self.egress_calls = 0

    def clock(self):
        return self.t

    async def sleep(self, s):
        self.t += s
        await asyncio.sleep(0)

    async def check_online(self):
        return self.online

    async def check_egress(self):
        self.egress_calls += 1
        return self.country

    def make(self, **kw):
        return NetWatch(check_online=self.check_online, check_egress=self.check_egress, clock=self.clock, sleep=self.sleep,
                        on_resume=lambda: setattr(self, 'resumed', self.resumed + 1), **kw)


def test_normal_tick_does_not_pause():
    async def main():
        f = Fake()
        nw = f.make()
        await nw.tick_once()
        f.t += 5.3
        await nw.tick_once()
        assert not nw.paused and nw.gate.is_set()
    asyncio.run(main())


def test_clock_jump_over_60s_is_wake_pauses_then_resumes_after_egress_check():
    async def main():
        f = Fake()
        nw = f.make()
        await nw.tick_once()
        slept_from = f.t
        f.t += 3600            # laptop slept an hour
        await nw.tick_once()
        assert nw.paused and not nw.gate.is_set()
        await nw.settle()
        assert not nw.paused and nw.gate.is_set()
        assert f.egress_calls >= 1 and f.resumed == 1
        assert nw.tainted(slept_from + 100, slept_from + 200)       # a request that straddled the sleep
        assert not nw.tainted(f.t + 10, f.t + 20)
    asyncio.run(main())


def test_non_vn_egress_keeps_pause_until_back_to_vn():
    async def main():
        f = Fake()
        f.country = 'US'
        nw = f.make()
        await nw.tick_once()
        f.t += 500
        await nw.tick_once()
        task = asyncio.create_task(nw.settle())
        for _ in range(50):
            await asyncio.sleep(0)
        assert nw.paused and not task.done()
        assert f.egress_calls >= 2           # rechecked every 60s
        f.country = 'VN'
        await task
        assert not nw.paused
    asyncio.run(main())


def test_network_error_while_online_is_counted():
    async def main():
        f = Fake()
        nw = f.make()
        assert await nw.report_network_error(f.t - 1) is False
        assert not nw.paused
    asyncio.run(main())


def test_network_error_while_offline_is_discounted_and_pauses_until_back():
    async def main():
        f = Fake()
        nw = f.make()
        t0 = f.t - 20
        f.online = False
        assert await nw.report_network_error(t0) is True
        assert nw.paused
        # another straggler arriving later is also discounted without a new probe
        assert await nw.report_network_error(f.t - 1) is True
        f.online = True
        await nw.settle()
        assert not nw.paused
        assert nw.tainted(t0, t0 + 5)
    asyncio.run(main())


def test_wait_running_blocks_while_paused():
    async def main():
        f = Fake()
        nw = f.make()
        f.online = False
        await nw.report_network_error(f.t)
        waiter = asyncio.create_task(nw.wait_running())
        await asyncio.sleep(0.01)
        assert not waiter.done()
        f.online = True
        await nw.settle()
        await asyncio.wait_for(waiter, 1)
    asyncio.run(main())
