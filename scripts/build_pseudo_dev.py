"""Pseudo dev set from the crawled corpus (no outside data): Q&A pages of vinmec / medlatec / hellobacsi.

  python scripts/build_pseudo_dev.py [--n 500] [--seed 42]

Steps (see data/dev/README.md): filter question/answer length (BGE-M3 tokens) -> drop questions that are
(near-)duplicates of a test query -> drop internal near-duplicates -> stratified sample by domain.
Near-duplicate = LCS(tokens) / max(len) >= 0.8 on metric-normalised text (gold_check_common.normalize_text).
A shared token 2-gram + the length ratio + the token-multiset bound are used as a coarse filter before exact LCS.
"""
from __future__ import annotations

from r2ai.paths import DEV_DIR, DOCS_DIR, RAW_DATA_DIR, auxiliary_disabled

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from r2ai.gold_check.gold_check_common import lcs_length, load_tokenizer, normalize_text  # noqa: E402

DOMAINS = ('vinmec.com', 'medlatec.vn', 'hellobacsi.com')
THRESH = 0.8


def bigram_matrix(seqs, vocab):
    rows, cols = [], []
    for i, s in enumerate(seqs):
        grams = ({(a, b) for a, b in zip(s, s[1:])} or {(s[0], -1)}) if s else set()
        for g in grams:
            cols.append(vocab.setdefault(g, len(vocab)))
            rows.append(i)
    return rows, cols


def near_dup_pairs(a, b=None):
    """Pairs (i, j) with LCS/max(len) >= THRESH between token lists a and b (b=None: within a, i<j)."""
    same = b is None
    b = a if same else b
    vocab: dict = {}
    ra, ca = bigram_matrix(a, vocab)
    rb, cb = bigram_matrix(b, vocab)
    A = sparse.csr_matrix((np.ones(len(ra)), (ra, ca)), shape=(len(a), len(vocab)))
    B = sparse.csr_matrix((np.ones(len(rb)), (rb, cb)), shape=(len(b), len(vocab)))
    shared = (A @ B.T).tocoo()
    ca_, cb_ = [Counter(s) for s in a], [Counter(s) for s in b]
    out, n_checked = [], 0
    for i, j in zip(shared.row, shared.col):
        if same and i >= j:
            continue
        la, lb = len(a[i]), len(b[j])
        m = max(la, lb)
        if not m or min(la, lb) < THRESH * m:
            continue
        if sum((ca_[i] & cb_[j]).values()) < THRESH * m:
            continue
        n_checked += 1
        if lcs_length(a[i], b[j]) >= THRESH * m:
            out.append((int(i), int(j)))
    return out, n_checked


def main(argv=None):
    auxiliary_disabled()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--docs-dir', default=str(DOCS_DIR))
    ap.add_argument('--queries', default=str(RAW_DATA_DIR / 'query.parquet'))
    ap.add_argument('--out', default=str(DEV_DIR / 'pseudo_vi.parquet'))
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)
    tok = load_tokenizer()

    def ntok(s):
        return len(tok.encode(s, add_special_tokens=False))

    def ntoks(s):
        return tok.encode(normalize_text(s), add_special_tokens=False)

    stats = {}
    docs = []
    for d in DOMAINS:
        rows = []
        for f in sorted(glob.glob(f'{args.docs_dir}/{d}__*.parquet')):
            rows += pq.read_table(f, columns=['doc_ids', 'url', 'domain', 'question', 'answer']).to_pylist()
        qa = [r for r in rows if (r['question'] or '').strip() and (r['answer'] or '').strip()]
        ok = [r for r in qa if 10 <= ntok(r['question']) <= 350 and ntok(r['answer']) >= 30]
        stats[d] = {'n_doc': len(rows), 'qa_nonempty': len(qa), 'eligible': len(ok)}
        docs += ok
    docs.sort(key=lambda r: (r['domain'], r['doc_ids'][0]))
    q_toks = [ntoks(r['question']) for r in docs]

    test = pq.read_table(args.queries).to_pylist()
    t_toks = [ntoks(q['query']) for q in test]
    pairs, checked_test = near_dup_pairs(q_toks, t_toks)
    dup_test = {i for i, _ in pairs}
    keep = [i for i in range(len(docs)) if i not in dup_test]

    pairs_int, checked_int = near_dup_pairs([q_toks[i] for i in keep])
    drop_int = set()
    for i, j in sorted(pairs_int):           # keep the first of each near-duplicate group
        if i not in drop_int:
            drop_int.add(j)
    keep = [k for n, k in enumerate(keep) if n not in drop_int]
    pre_int = [i for i in range(len(docs)) if i not in dup_test]
    for d in DOMAINS:
        stats[d]['dup_test'] = sum(docs[i]['domain'] == d for i in dup_test)
        stats[d]['dup_internal'] = sum(docs[pre_int[n]]['domain'] == d for n in drop_int)
        stats[d]['pool'] = sum(docs[i]['domain'] == d for i in keep)

    rnd = random.Random(args.seed)
    pool = len(keep)
    take = min(args.n, pool)
    picked = []
    by_dom = {d: [i for i in keep if docs[i]['domain'] == d] for d in DOMAINS}
    quota = {d: round(take * len(v) / pool) if pool else 0 for d, v in by_dom.items()}
    while sum(quota.values()) > take:        # rounding fix
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < take:
        quota[max(by_dom, key=lambda d: len(by_dom[d]) - quota[d])] += 1
    for d in DOMAINS:
        picked += rnd.sample(by_dom[d], quota[d])
    picked.sort(key=lambda i: (docs[i]['domain'], docs[i]['doc_ids'][0]))
    out_rows = [{'qid': f'pv{n:04d}', 'query': docs[i]['question'], 'src_doc_id': int(docs[i]['doc_ids'][0]),
                 'src_doc_ids_group': [int(x) for x in docs[i]['doc_ids']], 'domain': docs[i]['domain'], 'answer_text': docs[i]['answer']}
                for n, i in enumerate(picked)]
    schema = pa.schema([('qid', pa.string()), ('query', pa.string()), ('src_doc_id', pa.int64()), ('src_doc_ids_group', pa.list_(pa.int64())),
                        ('domain', pa.string()), ('answer_text', pa.string())])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), args.out, compression='zstd')

    def dist(lens):
        return {'n': len(lens), 'median': float(np.median(lens)) if lens else None, 'p95': float(np.percentile(lens, 95)) if lens else None}
    report = {'per_domain': stats, 'lcs_checked_vs_test': checked_test, 'lcs_checked_internal': checked_int,
              'n_out': len(out_rows), 'requested': args.n,
              'query_len_pseudo_bge_m3': dist([ntok(r['query']) for r in out_rows]),
              'query_len_test_bge_m3': dist([ntok(q['query']) for q in test])}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
