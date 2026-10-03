"""Pseudo dev set v2 from the crawled corpus only (no outside data).

  python scripts/build_pseudo_dev_v2.py [--n-a 400] [--max-per-domain 60] [--seed 42]

Type A (title-as-query): question-like titles of any vi domain in data/docs_vi, gold_text = body;
sampled in proportion to domain size with at most --max-per-domain queries per domain.
Type B (Q&A): reader questions of vinmec.com / hellobacsi.com, greeting and closing thanks stripped, gold_text = answer.
Both are de-duplicated against the 1,200 test queries and internally (LCS / max(len) >= 0.8, see build_pseudo_dev.py).
"""
from __future__ import annotations

from r2ai.paths import DEV_DIR, DOCS_DIR, RAW_DATA_DIR, auxiliary_disabled

import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_pseudo_dev import near_dup_pairs  # noqa: E402
from r2ai.gold_check.gold_check_common import load_tokenizer, normalize_text  # noqa: E402

QA_DOMAINS = ('vinmec.com', 'hellobacsi.com')
QUESTION_TITLE = re.compile(r'\?\s*$|\bcó\b.+\bkhông\b|\bbao lâu\b|\blà gì\b|\bnên\b.+\bkhông\b|\btại sao\b|\bnhư thế nào\b', re.I)
GREETING = re.compile(r'^(?:(?:xin |dạ |em |con |cháu |tôi )?(?:chào|kính chào)\s+(?:bác sĩ|bác sỹ|bs|bác|các bác sĩ)'
                      r'|(?:bác sĩ|bác sỹ|bs) ơi|thưa (?:bác sĩ|bác sỹ|bs))\b(?:\s*ạ\b)?[\s,.!:]*', re.I)
THANKS = re.compile(r'^(?:\S+\s){0,2}(?:xin |chân thành |rất )?(?:cảm ơn|cám ơn)\b[^?]{0,60}$'
                    r'|^(?:rất )?mong (?:bác sĩ|bs)\b[^?]{0,80}$'          # "Mong bác sĩ tư vấn giúp ạ."
                    r'|^\(.{2,80}\)$', re.I)                              # "(Hồng Thu, Tây Ninh)" signature after the thanks
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')


def _is_signature(p: str) -> bool:
    """Reader signature paragraph: "(Hoàng Vinh – Đồng Nai)", "Lê Kim Tuyền (1996)", "Thu Trà Lê, Cần Đước, Long An"."""
    p = p.strip()
    return bool(re.fullmatch(r'\(.{2,80}\)', p)) or (len(p.split()) <= 12 and not re.search(r'[.?!:…]$', p) and not re.search(r'\?', p))


def strip_greeting_thanks(text: str) -> str:
    """Drop a leading greeting ("Chào bác sĩ," ...), a trailing reader signature and closing thank-you sentences.
    The remaining wording is unchanged (paragraphs are joined by one space)."""
    paras = [p.strip() for p in text.strip().split('\n\n') if p.strip()]
    while len(paras) > 1 and _is_signature(paras[-1]):
        paras.pop()
    t = GREETING.sub('', ' '.join(paras), count=1).lstrip()
    sents = [s for s in SENT_SPLIT.split(t) if s.strip()]
    while len(sents) > 1 and THANKS.match(sents[-1].strip()):
        sents.pop()
    out = ' '.join(s.strip() for s in sents)
    return out[:1].upper() + out[1:] if out else out


def load_docs(docs_dir: str, cols: list[str], domains=None) -> list[dict]:
    rows = []
    pat = [f'{docs_dir}/{d}__*.parquet' for d in domains] if domains else [f'{docs_dir}/*.parquet']
    for g in pat:
        for f in sorted(glob.glob(g)):
            rows += pq.read_table(f, columns=cols).to_pylist()
    return rows


def dedup(items: list[dict], tok_key: str, test_toks: list[list[int]]):
    """-> kept items, n removed as test duplicates, n removed as internal duplicates."""
    toks = [it[tok_key] for it in items]
    pairs, _ = near_dup_pairs(toks, test_toks)
    dup_test = {i for i, _ in pairs}
    pre = [i for i in range(len(items)) if i not in dup_test]
    pairs_int, _ = near_dup_pairs([toks[i] for i in pre])
    drop = set()
    for i, j in sorted(pairs_int):
        if i not in drop:
            drop.add(j)
    kept = [items[pre[n]] for n in range(len(pre)) if n not in drop]
    return kept, dup_test, [items[pre[n]] for n in drop]


def capped_quota(sizes: dict[str, int], n: int, cap: int | None) -> dict[str, int]:
    """Proportional allocation of n over domains (largest remainder), each domain <= min(cap, size).
    Domains that hit their limit are fixed and the remainder is re-spread over the others (water-filling)."""
    take: dict[str, int] = {}
    free = dict(sizes)
    left = n
    while free and left > 0:
        tot = sum(free.values())
        quota = {d: left * s / tot for d, s in free.items()}
        lim = {d: min(s, cap) if cap else s for d, s in free.items()}
        over = [d for d in free if quota[d] >= lim[d]]
        if not over:
            base = {d: int(q) for d, q in quota.items()}
            ties = sorted(quota, key=lambda d: (-(quota[d] - base[d]), d))     # largest remainder, name breaks ties
            for d in ties[:left - sum(base.values())]:
                base[d] += 1
            take.update(base)
            break
        for d in over:
            take[d] = lim[d]
            left -= lim[d]
            del free[d]
    return take


def stratified(items: list[dict], n: int, rnd: random.Random, cap: int | None = None) -> list[dict]:
    by: dict[str, list] = {}
    for it in items:
        by.setdefault(it['domain'], []).append(it)
    if len(items) <= n and not cap:
        return list(items)
    take = capped_quota({d: len(v) for d, v in by.items()}, n, cap)
    out = []
    for d in sorted(by):
        out += rnd.sample(by[d], take.get(d, 0))
    return out


def dist(lens):
    return {'n': len(lens), 'median': float(np.median(lens)) if lens else None, 'p95': round(float(np.percentile(lens, 95)), 1) if lens else None}


def main(argv=None):
    auxiliary_disabled()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--docs-dir', default=str(DOCS_DIR))
    ap.add_argument('--queries', default=str(RAW_DATA_DIR / 'query.parquet'))
    ap.add_argument('--out', default=str(DEV_DIR / 'pseudo_vi_v2.parquet'))
    ap.add_argument('--n-a', type=int, default=400)
    ap.add_argument('--max-per-domain', type=int, default=60, help='cap on type A queries per domain (0 = no cap)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)
    tok = load_tokenizer()

    def ntok(s):
        return len(tok.encode(s, add_special_tokens=False))

    def mtoks(s):
        return tok.encode(normalize_text(s), add_special_tokens=False)

    test = [q['query'] for q in pq.read_table(args.queries).to_pylist()]
    test_toks = [mtoks(q) for q in test]
    rep: dict = {}

    # ---- type A ------------------------------------------------------------------------------------
    docs = load_docs(args.docs_dir, ['doc_ids', 'domain', 'title', 'body', 'lang', 'status'])
    a_items = []
    seen_ids = set()
    for r in docs:
        t = (r['title'] or '').strip()
        if r['lang'] != 'vi' or r['status'] != 'ok' or not t or not QUESTION_TITLE.search(t) or r['doc_ids'][0] in seen_ids:
            continue
        n = ntok(t)
        if 5 <= n <= 120:
            seen_ids.add(r['doc_ids'][0])
            a_items.append({'query': t, 'src_doc_ids_group': r['doc_ids'], 'domain': r['domain'], 'gold_text': r['body']})
    a_items.sort(key=lambda x: (x['domain'], x['src_doc_ids_group'][0]))
    for it in a_items:
        it['_t'] = mtoks(it['query'])
    a_kept, a_dup_test, a_dup_int = dedup(a_items, '_t', test_toks)
    rnd = random.Random(args.seed)
    a_pick = stratified(a_kept, args.n_a, rnd, cap=args.max_per_domain or None)
    rep['A'] = {'eligible': len(a_items), 'dup_test': len(a_dup_test), 'dup_internal': len(a_dup_int), 'pool': len(a_kept), 'out': len(a_pick),
                'eligible_by_domain_top': _count(a_items, 15), 'out_by_domain': _count(a_pick)}

    # ---- type B ------------------------------------------------------------------------------------
    qa = [r for r in load_docs(args.docs_dir, ['doc_ids', 'domain', 'question', 'answer'], QA_DOMAINS)
          if (r['question'] or '').strip() and (r['answer'] or '').strip()]
    b_items = []
    for r in qa:
        q = strip_greeting_thanks(r['question'])
        if 10 <= ntok(q) <= 350 and ntok(r['answer']) >= 30:
            b_items.append({'query': q, 'src_doc_ids_group': r['doc_ids'], 'domain': r['domain'], 'gold_text': r['answer'], '_raw': r['question']})
    b_items.sort(key=lambda x: (x['domain'], x['src_doc_ids_group'][0]))
    for it in b_items:
        it['_t'] = mtoks(it['query'])
    b_kept, b_dup_test, b_dup_int = dedup(b_items, '_t', test_toks)
    rep['B'] = {'qa_pages': len(qa), 'eligible': len(b_items), 'dup_test': len(b_dup_test), 'dup_internal': len(b_dup_int), 'out': len(b_kept),
                'out_by_domain': _count(b_kept),
                'greeting_removed': sum(GREETING.match(it['_raw'].strip()) is not None for it in b_items)}

    rows = []
    for typ, items in (('A', sorted(a_pick, key=lambda x: (x['domain'], x['src_doc_ids_group'][0]))), ('B', b_kept)):
        for it in items:
            rows.append({'qid': f'pv2{typ}{len([r for r in rows if r["type"] == typ]):04d}', 'type': typ, 'query': it['query'],
                         'src_doc_id': int(it['src_doc_ids_group'][0]), 'src_doc_ids_group': [int(x) for x in it['src_doc_ids_group']],
                         'domain': it['domain'], 'gold_text': it['gold_text']})
    schema = pa.schema([('qid', pa.string()), ('type', pa.string()), ('query', pa.string()), ('src_doc_id', pa.int64()),
                        ('src_doc_ids_group', pa.list_(pa.int64())), ('domain', pa.string()), ('gold_text', pa.string())])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), args.out, compression='zstd')
    rep['len_bge_m3'] = {'A': dist([ntok(r['query']) for r in rows if r['type'] == 'A']),
                         'B': dist([ntok(r['query']) for r in rows if r['type'] == 'B']),
                         'test': dist([ntok(q) for q in test])}
    rep['pct_end_question_mark'] = {'A': _pct([r['query'] for r in rows if r['type'] == 'A']), 'B': _pct([r['query'] for r in rows if r['type'] == 'B']),
                                    'test': _pct(test)}
    sample = random.Random(args.seed).sample([r for r in rows if r['type'] == 'A'], min(10, rep['A']['out']))
    rep['sample_A'] = [(r['qid'], r['domain'], r['query']) for r in sample]
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


def _count(items, top=None):
    from collections import Counter
    return dict(Counter(it['domain'] for it in items).most_common(top))


def _pct(qs):
    return round(100 * sum(q.rstrip().endswith(('?', '？')) for q in qs) / len(qs), 1) if qs else None


if __name__ == '__main__':
    sys.exit(main())
