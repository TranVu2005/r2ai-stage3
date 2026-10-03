"""Incremental shard -> parquet extraction (docs_vi). Idempotent: reprocessing a shard rewrites the same file."""
from __future__ import annotations

from r2ai.paths import ROOT

import hashlib
import logging
import os
import random
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .extractors import extract_doc
from .settings import Config
from .shards import list_shards, read_shard

log = logging.getLogger('vicrawl.extract')

SCHEMA = pa.schema([
    ('doc_ids', pa.list_(pa.int64())), ('url', pa.string()), ('final_url', pa.string()), ('domain', pa.string()), ('title', pa.string()),
    ('description', pa.string()), ('question', pa.string()), ('answer', pa.string()), ('body', pa.string()), ('paragraphs', pa.list_(pa.string())),
    ('lang', pa.string()), ('n_tokens_bge_m3', pa.int32()), ('text_sha1', pa.string()), ('fetched_at', pa.float64()),
    ('url_norm', pa.string()), ('status', pa.string()), ('extractor', pa.string()), ('headings', pa.list_(pa.string())),
])
MIN_OK_CHARS = 200
_tok = None


def default_tokenizer():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        try:
            tok = AutoTokenizer.from_pretrained('BAAI/bge-m3', local_files_only=True)
        except Exception:  # noqa: BLE001
            tok = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        tok.model_max_length = 10 ** 9
        _tok = lambda text: len(tok.encode(text, add_special_tokens=False))  # noqa: E731
    return _tok


def detect_lang(text: str) -> str:
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 0
        return detect(text[:1500]) if len(text) > 20 else ''
    except Exception:  # noqa: BLE001
        return ''


def extract_records(records: list[dict], extractor_for: dict, tokenizer=None) -> list[dict]:
    """ok/thin crawl records -> doc rows. Pure function (also runs inside worker processes)."""
    tokenizer = tokenizer or default_tokenizer()
    rows = []
    for r in records:
        if r.get('status') not in ('ok', 'thin') or not r.get('html'):
            continue
        doc = extract_doc(r['html'], extractor_for.get(r['domain'], 'generic'), r['domain'])
        body = doc.body
        rows.append({
            'doc_ids': r['doc_ids'], 'url': r['url'], 'final_url': r.get('final_url') or r['url'], 'domain': r['domain'], 'title': doc.title,
            'description': doc.description, 'question': doc.question, 'answer': doc.answer, 'body': body, 'paragraphs': doc.paragraphs,
            'lang': detect_lang(body), 'n_tokens_bge_m3': tokenizer(body) if body else 0, 'text_sha1': hashlib.sha1(body.encode('utf-8')).hexdigest(),
            'fetched_at': r['fetched_at'], 'url_norm': r['url_norm'], 'status': 'ok' if len(body) >= MIN_OK_CHARS else 'thin',
            'extractor': doc.extractor, 'headings': doc.headings})
    return rows


def _extract_shard_job(path: str, extractor_for: dict):
    return path, extract_records(read_shard(Path(path)), extractor_for)


def _atomic_parquet(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), tmp, compression='zstd')
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


class ExtractPipeline:
    def __init__(self, raw_dir: Path, docs_dir: Path, state_dir: Path, cfg: Config, tokenizer=None, workers: int = 1):
        self.raw_dir, self.docs_dir, self.cfg, self.tokenizer, self.workers = Path(raw_dir), Path(docs_dir), cfg, tokenizer, workers
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(Path(state_dir) / 'extract.db'), timeout=60, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('CREATE TABLE IF NOT EXISTS shards(path TEXT PRIMARY KEY, out_file TEXT, n_docs INTEGER, processed_at REAL);'
                              'CREATE TABLE IF NOT EXISTS docs(url_norm TEXT PRIMARY KEY, fetched_at REAL, out_file TEXT);')
        for p in self.docs_dir.rglob('*.tmp') if self.docs_dir.exists() else []:
            p.unlink(missing_ok=True)

    def _extractor_map(self, shards: list[Path]) -> dict:
        return {p.parent.name: self.cfg.for_domain(p.parent.name).extractor for p in shards}

    def pending(self) -> list[Path]:
        done = {r[0] for r in self.db.execute('SELECT path FROM shards')}
        return [p for p in list_shards(self.raw_dir) if str(p) not in done]

    def run_once(self) -> int:
        todo = self.pending()
        if not todo:
            return 0
        # directory names replace ':' by '_' (only for ip:port test hosts); real domains map 1:1
        emap = {}
        for p in todo:
            d = p.parent.name
            emap[d] = self.cfg.for_domain(d).extractor
        if self.workers > 1 and self.tokenizer is None:
            with ProcessPoolExecutor(self.workers) as ex:
                results = ex.map(_extract_shard_job, [str(p) for p in todo], [emap] * len(todo))
                for path, rows in results:
                    self._commit_shard(Path(path), rows)
        else:
            for p in todo:
                self._commit_shard(p, extract_records(read_shard(p), emap, self.tokenizer))
        return len(todo)

    def _commit_shard(self, shard: Path, rows: list[dict]):
        stem = shard.name.removesuffix('.jsonl.zst')
        out_file = f'{shard.parent.name}__{stem}.parquet'
        out_path = self.docs_dir / out_file
        keep, drop_from = [], {}
        for r in rows:
            row = self.db.execute('SELECT fetched_at, out_file FROM docs WHERE url_norm=?', (r['url_norm'],)).fetchone()
            if row and row[1] != out_file:
                if row[0] >= r['fetched_at']:
                    continue                                # an equal/newer copy already exists elsewhere
                drop_from.setdefault(row[1], set()).add(r['url_norm'])
            keep.append(r)
        if keep:
            _atomic_parquet(keep, out_path)
        for old_file, urls in drop_from.items():
            self._remove_from_parquet(self.docs_dir / old_file, urls)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.executemany('INSERT INTO docs(url_norm,fetched_at,out_file) VALUES (?,?,?) ON CONFLICT(url_norm) DO UPDATE SET fetched_at=excluded.fetched_at, out_file=excluded.out_file',
                                [(r['url_norm'], r['fetched_at'], out_file) for r in keep])
            self.db.execute('INSERT OR REPLACE INTO shards(path,out_file,n_docs,processed_at) VALUES (?,?,?,?)', (str(shard), out_file if keep else None, len(keep), time.time()))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        log.info('shard %s: %d docs -> %s', shard.name, len(keep), out_file if keep else '-')

    def _remove_from_parquet(self, path: Path, url_norms: set):
        if not path.exists():
            return
        rows = [r for r in pq.read_table(path).to_pylist() if r['url_norm'] not in url_norms]
        if rows:
            _atomic_parquet(rows, path)
        else:
            path.unlink()


# --------------------------------------------------------------------------------- QA report
def _latest_ok(raw_dir: Path, domain: str, n_docs: int) -> list[dict]:
    latest: dict[str, dict] = {}
    for p in list_shards(raw_dir, domain):
        for r in read_shard(p):
            if r.get('status') == 'ok' and r.get('html'):
                cur = latest.get(r['url_norm'])
                if cur is None or r['fetched_at'] >= cur['fetched_at']:
                    latest[r['url_norm']] = r
    return sorted(latest.values(), key=lambda r: r['fetched_at'])[:n_docs]


def build_qa_report(domain: str, *, raw_dir: Path, out_path: Path, cfg: Config, n_docs: int = 300, n_show: int = 5) -> Path:
    recs = _latest_ok(Path(raw_dir), domain, n_docs)
    name = cfg.for_domain(domain).extractor
    docs = [(r, extract_doc(r['html'], name, domain)) for r in recs]
    rnd = random.Random(int(hashlib.md5(domain.encode()).hexdigest()[:8], 16))
    shown = rnd.sample(docs, min(n_show, len(docs)))
    para_count = Counter(p for _, d in docs for p in set(d.paragraphs))
    fallback = sum(d.extractor == 'generic' for _, d in docs)
    qa = sum(bool(d.question) for _, d in docs)
    lines = [f'# QA extract: {domain}', '', f'- extractor cấu hình: `{name}`; doc ok đã phân tích: {len(docs)} (n_docs={n_docs})',
             f'- rơi về generic (trafilatura): {fallback}/{len(docs)}; trang hỏi-đáp (có question): {qa}/{len(docs)}',
             f'- median đoạn/doc: {sorted(len(d.paragraphs) for _, d in docs)[len(docs) // 2] if docs else 0}',
             '', f'Duyệt xong: `python crawl.py approve {domain}`', '']
    for i, (r, d) in enumerate(shown, 1):
        paras = d.paragraphs
        lines += [f'\n## Doc {i}', f'URL: {r["url"]}', f'Title: {d.title}', f'Description: {d.description[:300]}', f'Extractor: {d.extractor}; {len(paras)} đoạn; {sum(map(len, paras))} ký tự', '',
                  '**Đầu (3 đoạn):**', *[f'{j}. {p[:400]}' for j, p in enumerate(paras[:3], 1)], '', '**Cuối (3 đoạn):**', *[f'{j}. {p[:400]}' for j, p in enumerate(paras[-3:], 1)]]
        if d.question or d.answer:
            lines += ['', f'**Question:** {d.question[:500]}', '', f'**Answer:** {d.answer[:500]}']
    rep = [(c, p) for p, c in para_count.most_common(12) if c >= max(3, len(docs) // 20)]
    lines += ['', '## Đoạn lặp nhiều nhất (nghi boilerplate)', '']
    lines += [f'- {c}/{len(docs)} doc: {p[:200]}' for c, p in rep] or ['- (không có đoạn nào lặp đáng kể)']
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out_path
