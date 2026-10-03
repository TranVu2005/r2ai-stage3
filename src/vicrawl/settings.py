"""Paths and per-domain configuration (config/domains.yaml)."""
from __future__ import annotations

from r2ai.paths import ROOT, CONFIG_DIR, DOCS_WRITE_DIR, LOG_DIR, OUT_DIR, RAW_DATA_DIR, RAW_DIR, STATE_DIR, resolve_path

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


BUILTIN_GROUPS = {
    'cdn_large': {'rate_cap': 8.0, 'max_conns': 8},
    'default': {'rate_cap': 4.0, 'max_conns': 2},
    'slow_server': {'rate_cap': 1.0, 'max_conns': 4},
}
QA_GATE_MIN_URLS = 5000
QA_GATE_DOCS = 300


@dataclass
class Paths:
    state_dir: Path = STATE_DIR
    raw_dir: Path = RAW_DIR
    docs_dir: Path = DOCS_WRITE_DIR
    out_dir: Path = OUT_DIR
    config: Path = CONFIG_DIR / 'domains.yaml'
    corpus: Path = RAW_DATA_DIR / 'links_corpus.parquet'
    log_dir: Path = LOG_DIR

    @property
    def db(self) -> Path:
        return self.state_dir / 'crawl.db'

    @property
    def vi_domains_csv(self) -> Path:
        return self.out_dir / 'vi_domains.csv'

    @property
    def deferred_csv(self) -> Path:
        return self.out_dir / 'deferred_domains.csv'

    @property
    def qa_dir(self) -> Path:
        return self.out_dir / 'qa_extract'


@dataclass
class DomainCfg:
    domain: str
    group: str = 'default'
    rate_cap: float = 4.0
    max_conns: int = 2
    extractor: str = 'generic'
    priority: int = 1
    qa_gate: bool | None = None
    lang: str = 'vi'
    background: bool = False
    http2: bool = True
    start_rate: float = 1.0
    challenge_retry_h: float | None = None      # bot/cookie challenge: retry gap in hours (default: state.RETRY_GAP_S)
    challenge_max_retries: int | None = None    # bot/cookie challenge: retries after the first attempt (default: MAX_ATTEMPTS - 1)

    def retry_policy(self) -> dict:
        """status -> (gap_s, max_attempts) overrides for StateDB.commit_results."""
        if self.challenge_retry_h is None and self.challenge_max_retries is None:
            return {}
        from .state import MAX_ATTEMPTS, RETRY_GAP_S
        gap = RETRY_GAP_S if self.challenge_retry_h is None else self.challenge_retry_h * 3600.0
        attempts = MAX_ATTEMPTS if self.challenge_max_retries is None else 1 + self.challenge_max_retries
        return {s: (gap, attempts) for s in ('bot_challenge', 'cookie_challenge')}


@dataclass
class Config:
    groups: dict = field(default_factory=lambda: dict(BUILTIN_GROUPS))
    defaults: dict = field(default_factory=dict)
    domains: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> 'Config':
        raw = {}
        if path and Path(path).exists():
            raw = yaml.safe_load(Path(path).read_text('utf-8')) or {}
        groups = {**BUILTIN_GROUPS, **(raw.get('groups') or {})}
        return cls(groups=groups, defaults=raw.get('defaults') or {}, domains=raw.get('domains') or {})

    def for_domain(self, domain: str) -> DomainCfg:
        merged = {'group': 'default', **self.defaults, **(self.domains.get(domain) or {})}
        g = self.groups.get(merged['group'], self.groups['default'])
        merged = {**g, **merged}
        return DomainCfg(
            domain=domain, group=merged['group'], rate_cap=float(merged['rate_cap']), max_conns=int(merged['max_conns']),
            extractor=merged.get('extractor', 'generic'), priority=int(merged.get('priority', 1)), qa_gate=merged.get('qa_gate'),
            lang=merged.get('lang', 'vi'), background=bool(merged.get('background', False)), http2=bool(merged.get('http2', True)), start_rate=float(merged.get('start_rate', 1.0)),
            challenge_retry_h=_opt(merged.get('challenge_retry_h'), float), challenge_max_retries=_opt(merged.get('challenge_max_retries'), int))


def _opt(v, typ):
    return None if v is None else typ(v)


def paths_from_args(args) -> Paths:
    p = Paths()
    for attr, key in (('state_dir', 'state_dir'), ('raw_dir', 'raw_dir'), ('docs_dir', 'docs_dir'), ('out_dir', 'out_dir'), ('config', 'config'), ('corpus', 'corpus'), ('log_dir', 'log_dir')):
        v = getattr(args, key, None)
        if v:
            setattr(p, attr, resolve_path(v))
    return p


def safe_name(domain: str) -> str:
    """File-system safe domain (':' only occurs in ip:port hosts used by tests; it is an NTFS stream separator)."""
    return domain.replace(':', '_')


def env_flag(name: str, default: str = '') -> str:
    return os.environ.get(name, default)
