"""Sleep/wake + connectivity + egress watchdog.

Ticks every 5s on the WALL clock (monotonic clocks may not advance during suspend). A jump >60s beyond the
expected tick means the laptop just woke. While paused every domain waits on `gate`; requests whose lifetime
overlaps a pause are "tainted": their failures never count as domain errors, attempts or rate cuts."""
from __future__ import annotations

from r2ai.paths import ROOT

import asyncio
import logging
import time

log = logging.getLogger('vicrawl.net')
COUNTRY = 'VN'


class NetWatch:
    def __init__(self, *, check_online, check_egress, clock=time.time, sleep=asyncio.sleep, tick=5.0, jump=60.0,
                 recheck=60.0, probe_every=3, on_resume=None, country=COUNTRY):
        self.check_online, self.check_egress = check_online, check_egress
        self.clock, self.sleep = clock, sleep
        self.tick, self.jump, self.recheck, self.probe_every = tick, jump, recheck, probe_every
        self.on_resume, self.country = on_resume, country
        self.gate = asyncio.Event()
        self.gate.set()
        self.paused = False
        self.pause_start = 0.0
        self.intervals: list[tuple[float, float]] = []
        self._last = clock()
        self._ticks = 0
        self._task: asyncio.Task | None = None
        self.reason = ''

    # -- queries -------------------------------------------------------------------------
    async def wait_running(self):
        await self.gate.wait()

    def tainted(self, t0: float, t1: float) -> bool:
        if self.paused and t1 >= self.pause_start:
            return True
        return any(t0 <= e and t1 >= s for s, e in self.intervals)

    # -- pause / recovery ---------------------------------------------------------------
    def _enter_pause(self, start: float, reason: str):
        if self.paused:
            self.pause_start = min(self.pause_start, start)
            return
        self.paused, self.pause_start, self.reason = True, start, reason
        self.gate.clear()
        log.warning('PAUSE all domains (%s)', reason)
        self._task = asyncio.ensure_future(self._recover())

    async def _recover(self):
        while True:
            if await self.check_online():
                country = await self.check_egress()
                if country == self.country:
                    break
                log.warning('network is up but egress country=%s (need %s); staying paused, recheck in %ss', country, self.country, int(self.recheck))
                await self.sleep(self.recheck)
            else:
                await self.sleep(self.tick)
        end = self.clock()
        self.intervals.append((self.pause_start, end))
        cutoff = end - 86400
        self.intervals = [i for i in self.intervals if i[1] > cutoff]
        self.paused = False
        self._last = end
        if self.on_resume:
            self.on_resume()
        self.gate.set()
        log.warning('RESUME after %.0fs offline/asleep', end - self.pause_start)

    async def settle(self):
        while self._task and not self._task.done():
            await self._task
        self._task = None

    # -- entry points -------------------------------------------------------------------
    async def tick_once(self):
        now = self.clock()
        delta, prev = now - self._last, self._last
        self._last = now
        self._ticks += 1
        if self.paused:
            return
        if delta - self.tick > self.jump:
            self._enter_pause(prev, f'clock jumped {delta:.0f}s (sleep/wake)')
        elif self._ticks % self.probe_every == 0 and not await self.check_online():
            self._enter_pause(now - self.tick * self.probe_every, 'connectivity lost')

    async def report_network_error(self, started_at: float) -> bool:
        """Call after a network-level request failure. True => caused by our own connectivity: discount it."""
        if self.paused:
            return True
        if await self.check_online():
            return False
        self._enter_pause(started_at, 'connectivity lost (request failure)')
        return True

    async def run(self):
        while True:
            await self.sleep(self.tick)
            await self.tick_once()
