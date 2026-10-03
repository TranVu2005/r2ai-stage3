"""Local re-implementation of the organisers' (BTC) metric, aligned with the official spec.

    python -m eval.scorer --pred out/submissions/x.json --gold gold.json [--macro {all,has_gold}] [--per-query out.csv]
    python -m eval.scorer --bench            # time 1,200 queries x 20 predicted chunks on real corpus text

Both files use the submission format (JSON list, or a ZIP holding exactly one such JSON):

    [{"id": 1, "relevant_docs": [12, 34], "relevant_chunks": [{"doc_id": 12, "chunk_text": "...", "chunk_order": 0}]}, ...]

(`chunk_order`, an int >= 0, is optional and ignored by the metric.)

Doc level (per query):  D = set(relevant_docs), G = set(gold relevant_docs)
    P = |D∩G| / |D|, R = |D∩G| / |G|, F2 = 5PR / (4P + R), F2 = 0 when P = R = 0 (or |D| = 0).
Chunk level (per query): a predicted chunk is only compared with gold chunks of the *same* doc_id.
    Text normalisation (spec): Unicode NFKC -> html.unescape -> lowercase -> whitespace runs collapsed -> strip.
    Punctuation and HTML tags are NOT removed (the spec only mentions decoding entities).
    ASSUMPTION (not stated by BTC): tokens are BAAI/bge-m3 tokenizer ids, without special tokens.
    * overlap(c, g) = LCS_tokens(c, g) / |g|;  pred c matches gold g  iff  overlap >= 0.4
    * no merging of predictions: every submitted chunk counts, duplicates included.
    * P = #preds matching >= 1 gold chunk / #preds submitted,  R = #gold chunks matched by >= 1 pred / #gold chunks.
Each level is macro-averaged over queries, `macro` selects which:
    has_gold (default): queries with >= 1 gold item at that level (the others are skipped, not scored 0)
    all:                every gold query; one without gold at that level counts 0
A gold query absent from the prediction file counts as an empty prediction (F2 = 0).
Final = (DocF2 + ChunkF2) / 2.

The 0.4 threshold is compared in integers (5*LCS >= 2*|g|) so "exactly 40 %" is a match.
LCS is exact (bit-parallel, Allison-Dix / Hyyro), O(len(text) * ceil(len(pattern) / word)) on Python big ints.
"""
from __future__ import annotations

from r2ai.paths import DOCS_DIR, ROOT, assert_writable, require_inputs, resolve_path

import argparse
import html
import json
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence


MATCH_NUM, MATCH_DEN = 2, 5      # 0.4


def normalize_metric(text: str | None) -> str:
    """Official normalisation: NFKC -> html.unescape -> lowercase -> collapse whitespace -> strip."""
    if not text:
        return ''
    return ' '.join(html.unescape(unicodedata.normalize('NFKC', text)).lower().split())


# ---------------------------------------------------------------------------------------------------------------
# LCS
def lcs_masks(pattern: Sequence[int]) -> dict:
    masks: dict = {}
    for i, t in enumerate(pattern):
        masks[t] = masks.get(t, 0) | (1 << i)
    return masks


def lcs_with_masks(masks: dict, text: Sequence[int]) -> int:
    """Exact LCS length between the pattern behind `masks` and `text` (bit-parallel)."""
    state = 0
    get = masks.get
    for t in text:
        m = get(t)
        if m is None:
            continue                     # x = state, state & ~(state - (state<<1 | 1)) == state when the token is absent
        x = state | m
        state = x & ~(x - ((state << 1) | 1))
    return state.bit_count()


def lcs(a: Sequence[int], b: Sequence[int]) -> int:
    if not a or not b:
        return 0
    if len(a) > len(b):              # shorter one as the bit pattern
        a, b = b, a
    return lcs_with_masks(lcs_masks(a), b)


def lcs_dp(a: Sequence, b: Sequence) -> int:
    """Plain O(n*m) DP, reference implementation for tests."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------------------------------------------
# per-query metrics
def f2(p: float, r: float) -> float:
    return 0.0 if p == 0 and r == 0 else 5 * p * r / (4 * p + r)


def doc_prf(pred: Iterable[int], gold: Iterable[int]) -> tuple[float, float, float]:
    d, g = set(pred), set(gold)
    if not g:
        raise ValueError('doc_prf needs >= 1 gold doc')
    hit = len(d & g)
    p = hit / len(d) if d else 0.0
    r = hit / len(g)
    return p, r, f2(p, r)


def _matches(lcs_len: int, gold_len: int) -> bool:
    return gold_len > 0 and MATCH_DEN * lcs_len >= MATCH_NUM * gold_len


def chunk_prf(pred: list[tuple[int, Sequence[int]]], gold: list[tuple[int, Sequence[int]]]) -> tuple[float, float, float]:
    """pred / gold: (doc_id, normalised token ids). Gold chunks with 0 tokens are ignored; every pred counts in P."""
    gold = [(d, t) for d, t in gold if len(t) > 0]
    if not gold:
        raise ValueError('chunk_prf needs >= 1 non-empty gold chunk')
    gold_by_doc: dict[int, list[int]] = {}
    for gi, (d, _) in enumerate(gold):
        gold_by_doc.setdefault(d, []).append(gi)
    gold_masks = [lcs_masks(t) for _, t in gold]
    gold_hit = [False] * len(gold)
    n_pred_hit = 0
    for d, pt in pred:
        hit_any = False
        for gi in gold_by_doc.get(d, []):
            if _matches(lcs_with_masks(gold_masks[gi], pt), len(gold[gi][1])):
                gold_hit[gi] = True
                hit_any = True
        n_pred_hit += hit_any
    p = n_pred_hit / len(pred) if pred else 0.0
    r = sum(gold_hit) / len(gold)
    return p, r, f2(p, r)


# ---------------------------------------------------------------------------------------------------------------
# tokenisation
class Tokenizer:
    """Normalise + BGE-M3 encode (no special tokens), memoised. `encode_fn` lets tests inject e.g. str.split."""

    def __init__(self, encode_fn: Callable[[list[str]], list[list]] | None = None):
        if encode_fn is None:
            from r2ai.gold_check.gold_check_common import load_tokenizer
            tok = load_tokenizer()

            def encode_fn(texts):
                return tok(texts, add_special_tokens=False)['input_ids'] if texts else []
        self._encode = encode_fn
        self._cache: dict[str, tuple] = {}         # normalised text -> ids
        self._raw: dict[str, tuple] = {}           # raw text -> ids

    def encode_many(self, texts: list[str]) -> list[tuple]:
        new = [t for t in dict.fromkeys(texts) if t not in self._raw]
        norm = {t: normalize_metric(t) for t in new}
        todo = sorted({n for n in norm.values() if n and n not in self._cache})
        for n, ids in zip(todo, self._encode(todo)):
            self._cache[n] = tuple(ids)
        for t, n in norm.items():
            self._raw[t] = self._cache[n] if n else ()
        return [self._raw[t] for t in texts]

    def encode(self, text: str) -> tuple:
        return self.encode_many([text])[0]


def whitespace_tokenizer() -> Tokenizer:
    return Tokenizer(lambda texts: [t.split() for t in texts])


# ---------------------------------------------------------------------------------------------------------------
# whole files
def load_submission(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith('/')]
            if len(names) != 1:
                raise ValueError(f'{path}: ZIP must hold exactly 1 file, found {len(names)}')
            return json.loads(z.read(names[0]).decode('utf-8'))
    return json.loads(path.read_text(encoding='utf-8'))


MACROS = ('all', 'has_gold')


@dataclass
class Score:
    """doc_f2 / chunk_f2 / final / n_* follow `macro`; `by_macro` holds both variants."""
    doc_f2: float
    chunk_f2: float
    final: float
    n_doc_queries: int
    n_chunk_queries: int
    macro: str = 'has_gold'
    by_macro: dict = field(default_factory=dict)
    per_query: list[dict] = field(default_factory=list)


def score(pred: list[dict], gold: list[dict], tokenizer: Tokenizer | None = None, macro: str = 'has_gold') -> Score:
    if macro not in MACROS:
        raise ValueError(f'macro must be one of {MACROS}')
    tokenizer = tokenizer or Tokenizer()
    pred_by_id = {int(r['id']): r for r in pred}
    texts = [c['chunk_text'] for rows in (pred, gold) for r in rows for c in (r.get('relevant_chunks') or [])]
    tokenizer.encode_many(texts)                       # one batched tokenizer call, then cache hits
    rows, docs, chunks = [], [], []
    for g in gold:
        qid = int(g['id'])
        p = pred_by_id.get(qid, {})
        row = {'id': qid}
        if g.get('relevant_docs'):
            row['doc_p'], row['doc_r'], row['doc_f2'] = doc_prf(p.get('relevant_docs') or [], g['relevant_docs'])
            docs.append(row['doc_f2'])
        gc = [(int(c['doc_id']), tokenizer.encode(c['chunk_text'])) for c in (g.get('relevant_chunks') or [])]
        if any(len(t) for _, t in gc):
            pc = [(int(c['doc_id']), tokenizer.encode(c['chunk_text'])) for c in (p.get('relevant_chunks') or [])]
            row['chunk_p'], row['chunk_r'], row['chunk_f2'] = chunk_prf(pc, gc)
            chunks.append(row['chunk_f2'])
        rows.append(row)
    n = len(gold)
    by = {}
    for name, nd, nc in (('has_gold', len(docs), len(chunks)), ('all', n, n)):
        d = sum(docs) / nd if nd else 0.0
        c = sum(chunks) / nc if nc else 0.0
        by[name] = {'doc_f2': d, 'chunk_f2': c, 'final': (d + c) / 2, 'n_doc_queries': nd, 'n_chunk_queries': nc}
    m = by[macro]
    return Score(m['doc_f2'], m['chunk_f2'], m['final'], m['n_doc_queries'], m['n_chunk_queries'], macro, by, rows)


# ---------------------------------------------------------------------------------------------------------------
def _bench(n_queries: int = 1200, n_pred: int = 20, seed: int = 42) -> dict:
    """Synthetic but realistic load: real paragraphs of data/docs_vi as gold and predicted chunks."""
    import glob
    import random

    import pyarrow.parquet as pq
    rnd = random.Random(seed)
    files = sorted(glob.glob(str(DOCS_DIR / '*.parquet')))
    rnd.shuffle(files)
    docs = []
    for f in files:
        for r in pq.read_table(f, columns=['doc_ids', 'paragraphs', 'status']).to_pylist():
            if r['status'] == 'ok' and r['paragraphs'] and len(r['paragraphs']) >= 4:
                docs.append((r['doc_ids'][0], r['paragraphs']))
        if len(docs) >= 6000:
            break

    def chunk(paras):                      # ~256-token-ish window of consecutive paragraphs
        i = rnd.randrange(len(paras))
        return '\n\n'.join(paras[i:i + 3])

    gold, pred = [], []
    for q in range(n_queries):
        gdocs = rnd.sample(docs, 2)
        gold.append({'id': q, 'relevant_docs': [d for d, _ in gdocs],
                     'relevant_chunks': [{'doc_id': d, 'chunk_text': chunk(p)} for d, p in gdocs for _ in range(2)]})
        pdocs = gdocs[:1] + rnd.sample(docs, 4)
        pred.append({'id': q, 'relevant_docs': [d for d, _ in pdocs],
                     'relevant_chunks': [{'doc_id': d, 'chunk_text': chunk(p)} for d, p in pdocs for _ in range(n_pred // 5)]})
    tok = Tokenizer()
    t0 = time.perf_counter()
    s = score(pred, gold, tok)
    t1 = time.perf_counter()
    s2 = score(pred, gold, tok)                       # tokens cached: metric only
    t2 = time.perf_counter()
    lens = sorted(len(v) for v in tok._cache.values())
    return {'queries': n_queries, 'pred_chunks_per_query': n_pred, 'seconds_total_incl_tokenize': round(t1 - t0, 2),
            'seconds_metric_only_cached_tokens': round(t2 - t1, 2), 'chunk_tokens_median': lens[len(lens) // 2],
            'chunk_tokens_p95': lens[int(len(lens) * 0.95)], 'final': round(s.final, 4), 'same_result': s.final == s2.final}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred')
    ap.add_argument('--gold')
    ap.add_argument('--per-query', help='write per-query CSV here')
    ap.add_argument('--macro', choices=MACROS, default='has_gold', help='macro-average over queries with gold at that level (default) or over all gold queries')
    ap.add_argument('--bench', action='store_true')
    a = ap.parse_args(argv)
    if a.per_query:
        a.per_query = str(assert_writable(a.per_query))
    if a.bench:
        print(json.dumps(_bench(), indent=1))
        return 0
    if not (a.pred and a.gold):
        ap.error('--pred and --gold are required (or --bench)')
    a.pred, a.gold = str(resolve_path(a.pred)), str(resolve_path(a.gold))
    require_inputs(a.pred, a.gold)
    s = score(load_submission(a.pred), load_submission(a.gold), macro=a.macro)
    print(json.dumps({'macro': s.macro, **s.by_macro[s.macro], 'by_macro': s.by_macro}, indent=1))
    if a.per_query:
        import csv
        keys = ['id', 'doc_p', 'doc_r', 'doc_f2', 'chunk_p', 'chunk_r', 'chunk_f2']
        with open(a.per_query, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(s.per_query)
    return 0


if __name__ == '__main__':
    sys.exit(main())
