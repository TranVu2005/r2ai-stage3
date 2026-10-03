"""robots.txt interpretation (policy copied from r2ai.probe.fetcher.Fetcher.allowed, minus the I/O)."""
from __future__ import annotations

from r2ai.paths import ROOT

import re
import time
from dataclasses import dataclass
from urllib.robotparser import RobotFileParser

RETRY_UNAVAILABLE_S = 1800.0


@dataclass
class Robots:
    state: str
    parser: RobotFileParser | None = None
    crawl_delay: float | None = None
    retry_at: float | None = None
    ua: str = ''

    def can_fetch(self, url: str) -> bool:
        if self.state in ('DISALLOWED', 'UNAVAILABLE'):
            return False
        return self.parser is None or self.parser.can_fetch(self.ua, url)


def _matching_rules(text: str, agent: str):
    agents, rules, entries = [], [], []
    for line in text.splitlines() + ['User-agent: __end__']:
        line = line.split('#', 1)[0].strip()
        if ':' not in line:
            continue
        name, value = (v.strip() for v in line.split(':', 1))
        if name.lower() == 'user-agent':
            if rules:
                entries.append((agents, rules))
                agents, rules = [], []
            agents.append(value.lower())
        elif agents:
            rules.append((name.lower(), value))
    matched = next((r for a, r in entries if any(x != '*' and x in agent for x in a)), None)
    if matched is None:
        matched = next((r for a, r in entries if '*' in a), [])
    return matched


def interpret_robots(status, text: str, ua: str, *, now: float | None = None, error: str = '') -> Robots:
    now = time.time() if now is None else now
    if status is None or (isinstance(status, int) and status >= 500) or (status == 429):
        return Robots('UNAVAILABLE', retry_at=now + RETRY_UNAVAILABLE_S, ua=ua)
    if status in (401, 403):
        return Robots('DISALLOWED', ua=ua)
    if status in (404, 410) or (status is not None and 400 <= status < 500):
        return Robots('no_robots', ua=ua)
    if re.search(r'<(?:html|!doctype)', text[:500], re.I):
        return Robots('no_robots_html_fallback', ua=ua)
    rp = RobotFileParser()
    rp.parse(text.splitlines())
    agent = ua.split('/')[0].lower()
    delay = rp.crawl_delay(ua) or rp.crawl_delay('*')
    decimals = [float(v) for k, v in _matching_rules(text, agent) if k == 'crawl-delay' and re.fullmatch(r'\d+(?:\.\d+)?', v)]
    if decimals:
        delay = max(decimals)
    rate = rp.request_rate(ua) or rp.request_rate('*')
    if rate:
        delay = max(delay or 0, rate.seconds / rate.requests)
    return Robots('robots_ok', rp, float(delay) if delay else None, ua=ua)
