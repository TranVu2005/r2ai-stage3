"""Deterministic samples, durable checkpoints and common CLI helpers."""
from __future__ import annotations

from r2ai.paths import DATA_DIR, OUT_DIR, RAW_DATA_DIR, ROOT

import argparse
import csv
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import polars as pl

OUT = OUT_DIR
SEED = 42


def normalize_url(url: str) -> str:
    u = urlsplit(url)
    # Only transport/www/fragment duplicates; path and query remain exact.
    return urlunsplit(('', u.netloc.lower().removeprefix('www.'), u.path or '/', u.query, '')).removeprefix('//')


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    temp = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    for attempt in range(20):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


class Checkpoint:
    def __init__(self, path: Path):
        self.path, self.lock, self.rows = path, threading.Lock(), {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            data = path.read_bytes()
            offset = 0
            lines = data.splitlines(keepends=True)
            for i, line in enumerate(lines):
                try:
                    r = json.loads(line)
                    self.rows[r['key']] = r
                except (ValueError, KeyError):
                    if i != len(lines) - 1:
                        raise ValueError(f'Corrupt non-final checkpoint line {i + 1}: {path}')
                    with path.open('r+b') as f:
                        f.truncate(offset)
                    break
                offset += len(line)
            if data and not data.endswith(b'\n') and offset == len(data):
                with path.open('ab') as f:
                    f.write(b'\n')

    def add(self, row: dict, *, replace=False):
        with self.lock:
            if row['key'] in self.rows and not replace:
                return
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            self.rows[row['key']] = row


def cli(description: str, n: int) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--domains', nargs='+', help='Domain names (space or comma separated).')
    p.add_argument('--n', type=int, default=n, help='URLs per domain.')
    return p


def selected_domains(values: list[str] | None) -> set[str] | None:
    return {d.strip().lower().removeprefix('www.') for v in values for d in v.split(',') if d.strip()} if values else None


def corpus_path() -> Path:
    for p in (DATA_DIR / 'links_corpus.parquet', RAW_DATA_DIR / 'links_corpus.parquet'):
        if p.exists():
            return p
    raise FileNotFoundError('links_corpus.parquet not found')


def corpus() -> pl.DataFrame:
    df = pl.read_parquet(corpus_path(), columns=['id', 'url'])
    return df.with_columns(pl.col('url').str.extract(r'^https?://([^/?#]+)', 1).str.to_lowercase().str.replace(r'^www\.', '').alias('domain'), pl.col('url').str.replace(r'^https?://(?:www\.)?', '').str.replace(r'#.*$', '').alias('norm'))


def samples(df: pl.DataFrame, domains: list[str], n: int) -> list[dict]:
    if n < 1:
        raise ValueError('--n must be positive')
    result = []
    for domain in domains:
        subset = df.filter(pl.col('domain') == domain).sort('id').unique('norm', keep='first', maintain_order=True)
        if subset.height:
            result.extend(subset.sample(min(n, subset.height), seed=SEED).sort('id').select('id', 'url', 'domain').to_dicts())
    return result


def experiment_config(path: Path, config: dict):
    """Do not merge incompatible sample sizes / UA / seed into one experiment."""
    if path.exists():
        previous = json.loads(path.read_text('utf-8'))
        if previous != config:
            raise ValueError(f'Configuration differs from {path}; use another --run-dir for a new experiment')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), 'utf-8')


def key(domain: str, url: str, arm: str) -> str:
    return hashlib.sha256(f'{domain}\n{url}\n{arm}'.encode()).hexdigest()
