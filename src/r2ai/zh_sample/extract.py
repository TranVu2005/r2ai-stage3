from __future__ import annotations

from r2ai.paths import require_inputs
from .common import RUN, RAW, DOCS, EXTRACT_STATE, preflight, atomic_json, exclusive
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from vicrawl.extractors import generic_extract
from vicrawl.extractors.base import Doc, parse_html, blocks, first_text
from vicrawl.shards import list_shards, read_shard

# Selectors are opt-in and inspected against smoke HTML before the full sample.
SELECTORS = {
    '120ask.com': (['.b_askcont', '.q-desc', '.ask-detail'],
                   ['.b_anscont_cont', '.ans-cont', '.ans_cont', '.answer-content']),
    'ask.39.net': (['.ask_cont', '.ask_cont_txt', '.question-content', '.ask-detail'],
                   ['.doc_ans', '.answer_text', '.answer-content', '.answer-con']),
    'cnkang.com': (['.ask-content', '.question', '.ques-info', '.q-cont'],
                   ['.answer-content', '.answer', '.answer-info', '.a-cont']),
    'familydoctor.com.cn': (['.question', '.question-content', '.ask-content', '.q-cont'],
                   ['.answer', '.answer-content', '.answer_text', '.a-cont']),
}


def selected_blocks(tree, selectors):
    for selector in selectors:
        found = tree.cssselect(selector)
        if found:
            return [p for node in found for p in blocks(node) if p]
    return []


def extract_zh(raw: str, domain: str) -> Doc:
    generic = generic_extract(raw)
    if domain not in SELECTORS:
        return generic
    tree = parse_html(raw)
    for node in tree.xpath('//script|//style|//nav|//footer|//aside'):
        node.drop_tree()
    q, a = (selected_blocks(tree, selectors) for selectors in SELECTORS[domain])
    if not a:
        return generic
    paras = q + a
    return Doc(title=first_text(tree, ['h1']) or generic.title, description=generic.description,
               question='\n\n'.join(q), answer='\n\n'.join(a), paragraphs=paras,
               extractor='zh_' + domain)


def extract_all():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vicrawl.extract_pipeline import SCHEMA, default_tokenizer, _atomic_parquet
    preflight()
    require_inputs(RAW)
    with exclusive('extract'):
        EXTRACT_STATE.mkdir(parents=True, exist_ok=True)
        DOCS.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(EXTRACT_STATE / 'extract.db')
        db.execute('CREATE TABLE IF NOT EXISTS shards(path TEXT PRIMARY KEY, output TEXT)')
        if 'version' not in {r[1] for r in db.execute('PRAGMA table_info(shards)')}:
            db.execute('ALTER TABLE shards ADD COLUMN version TEXT')
        version = hashlib.sha256(json.dumps(SELECTORS, sort_keys=True).encode()).hexdigest()
        done = {r[0] for r in db.execute('SELECT path FROM shards WHERE version=?', (version,))}
        tok = default_tokenizer()
        try:
            for shard in list_shards(RAW):
                if str(shard) in done:
                    continue
                rows = []
                for r in read_shard(shard):
                    if r['status'] not in ('ok', 'thin') or not r.get('html'):
                        continue
                    doc = extract_zh(r['html'], r['domain'])
                    body = doc.body
                    rows.append({'doc_ids': r['doc_ids'], 'url': r['url'], 'final_url': r['final_url'],
                        'domain': r['domain'], 'title': doc.title, 'description': doc.description,
                        'question': doc.question, 'answer': doc.answer, 'body': body,
                        'paragraphs': doc.paragraphs, 'headings': doc.headings,
                        'lang': 'zh' if sum('\u4e00' <= ch <= '\u9fff' for ch in body) > .1 * len(body) else 'unknown',
                        'n_tokens_bge_m3': tok(body) if body else 0,
                        'text_sha1': hashlib.sha1(body.encode()).hexdigest(), 'fetched_at': r['fetched_at'],
                        'url_norm': r['url_norm'], 'status': 'ok' if len(body) >= 200 else 'thin', 'extractor': doc.extractor})
                out = DOCS / 'shards' / (shard.parent.name + '__' + shard.stem + '.parquet')
                if rows:
                    _atomic_parquet(rows, out)
                db.execute('INSERT OR REPLACE INTO shards(path,output,version) VALUES (?,?,?)', (str(shard), str(out), version))
                db.commit()
        finally:
            db.close()
        # Latest fetched copy per URL; QA is deterministic and contains original HTML references.
        docs = {}
        for path in sorted((DOCS / 'shards').glob('*.parquet')):
            for r in pq.read_table(path).to_pylist():
                if r['url_norm'] not in docs or r['fetched_at'] > docs[r['url_norm']]['fetched_at']:
                    docs[r['url_norm']] = r
        by_domain = defaultdict(list)
        for r in docs.values():
            by_domain[r['domain']].append(r)
        stats = {}
        for domain, rows in sorted(by_domain.items()):
            _atomic_parquet(rows, DOCS / 'bundle' / (domain + '.parquet'))
            rows.sort(key=lambda r: hashlib.sha256(('42:' + r['url_norm']).encode()).digest())
            atomic_json(RUN / 'qa' / (domain + '.json'), rows[:50])
            stats[domain] = {'n_docs': len(rows), 'ok': sum(r['status'] == 'ok' for r in rows),
                             'answer': sum(bool(r['answer']) for r in rows), 'qa_docs': min(50, len(rows)),
                             'unknown_lang': sum(r['lang'] != 'zh' for r in rows)}
        atomic_json(RUN / 'extract_stats.json', stats)
        atomic_json(RUN / 'extract_manifest.json', {'selector_sha256': version,
                    'raw_shards': {str(p): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in list_shards(RAW)},
                    'docs_dir': str(DOCS / 'bundle'), 'docs': len(docs)})
        print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    extract_all()
