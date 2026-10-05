"""Build one submission from the K=100 retrieval cache (scripts/run_retrieval_k100.py).

    python -m scripts.make_submission --k-doc 5 --k-chunk 5 --chunk-mode full --out out/submissions/sub03_vi_k5_full.zip

  relevant_docs   top-K cached docs, each expanded to its whole doc_ids_group (same url_norm), rank order, unique.
  relevant_chunks chunk_mode c2:   top-2 answer/body chunks (t256) of each top-K doc by reranker score (as sub02); K <= 50.
                  chunk_mode full: 1 chunk per doc = the whole `answer` field if present (and verbatim inside the doc
                                   text), else the whole `body`, else title + description. No length cut.
                  A chunk is attached to the primary doc_id only (never repeated for the other ids of the group).
  extra chunks    --extra-chunk-docs N (off by default): the docs ranked k_chunk+1..N, which get no chunk above, get 1 extra
                  chunk each = their best answer/body t256 chunk by cached reranker score (ties: lower chunk_id), verbatim
                  from the index text (memory-mapped <index>/text.arrow). Appended after the regular chunks, rank order.
                  N <= 50 (scores are cached for the top-50 docs). --extra-zip-budget-bytes B: largest N in
                  [k_chunk, --extra-chunk-docs] whose real zip is <= B bytes (binary search on written zips).
  k_chunk search  --k-chunk-zip-budget-bytes B: largest k_chunk in [--k-chunk, --k-chunk-max] whose real zip is <= B bytes
                  (binary search on written zips; same chunk rule, so the first --k-chunk chunks stay identical).
  docs > 100      --doc-ranking P --k-doc-total K (needs --k-doc 100): relevant_docs = the K100 docs as above, then the
                  next docs of P (query_id, rank, doc_id; vi_cand.docs.parquet of run_retrieval_k100 --candidates-only)
                  that are not among the top-100 cached docs, in rank order, until K primary docs; each expanded to its
                  doc_ids_group. Cached rows ranked > 100 (a deeper cache) are ignored here.
                  relevant_chunks unchanged.
  dedupe          per query, in rank order, a later chunk is dropped when its text equals an earlier one after
                  normalisation (NFKC, html.unescape, lowercase, whitespace collapsed), or when
                  LCS_tokens(BGE-M3) / len(shorter) >= 0.8 against an earlier kept chunk.
Writes <out> (zip with the single <stem>.json), the json next to it and <stem>.stats.json (measured numbers).
"""
from __future__ import annotations

from r2ai.paths import CHUNKS_DIR, DOCS_DIR, INDEX_DIR, RAW_DATA_DIR, RUNS_DIR, assert_writable, chunks_file, index_dir, require_inputs, resolve_path

import argparse
import glob
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


from r2ai.eval.scorer import Tokenizer, lcs_masks, lcs_with_masks, normalize_metric  # noqa: E402
from r2ai.submit.window_chunks import Windows, count_tokens  # noqa: E402
from r2ai.submit.writer import format_submission, write_submission  # noqa: E402

NEAR_NUM, NEAR_DEN = 4, 5      # 0.8
C2_MAX_K = 50                  # chunk scores are cached for the top-50 docs only


def load_full_texts(doc_ids: set[int], docs_dir: Path) -> tuple[dict[int, str], dict[str, int], dict[int, int]]:
    """primary doc_id -> whole-chunk text (answer / body / title+description), verbatim from data/docs_vi."""
    out: dict[int, str] = {}
    off: dict[int, int] = {}                  # char offset of the text inside index.chunk.doc_layout's doc_text
    src = {'answer': 0, 'body': 0, 'title+description': 0}
    for f in sorted(glob.glob(str(docs_dir / '*.parquet'))):
        t = pq.read_table(f, columns=['doc_ids', 'title', 'question', 'description', 'answer', 'body', 'status'])
        for r in t.to_pylist():
            d = r['doc_ids'][0]
            if d not in doc_ids or r['status'] != 'ok':
                continue
            body = r['body'] or ''
            ans = (r['answer'] or '').strip()
            title, q = (r['title'] or '').strip(), (r['question'] or '').strip()
            b0 = (len(title) + 2 if title else 0) + (len(q) + 2 if q and q not in body else 0)
            if ans and ans in body:
                out[d], k = ans, 'answer'
                off[d] = b0 + body.find(ans)
            elif body.strip():
                out[d], k = body, 'body'
                off[d] = b0
            else:
                out[d], k = '\n\n'.join(x.strip() for x in (r['title'], r['description']) if x and x.strip()), 'title+description'
            src[k] += 1
    return out, src, off


def dedupe(texts: list[str], tok: Tokenizer) -> tuple[list[bool], int, int]:
    """Keep-flags in order; (kept, n_exact_dropped, n_near_dropped). Exact key = normalize_metric; near = LCS/shorter >= .8."""
    norm = [normalize_metric(t) for t in texts]
    ids = tok.encode_many(texts)
    vocab: dict[int, int] = {}
    for s in ids:
        for t in s:
            vocab.setdefault(t, len(vocab))
    counts = np.zeros((len(texts), max(len(vocab), 1)), np.int32)
    for i, s in enumerate(ids):
        if s:
            np.add.at(counts[i], [vocab[t] for t in s], 1)
    keep, seen, kept_idx = [], set(), []
    n_exact = n_near = 0
    for i in range(len(texts)):
        if norm[i] in seen:
            keep.append(False)
            n_exact += 1
            continue
        drop = False
        if kept_idx and ids[i]:
            ub = np.minimum(counts[kept_idx], counts[i]).sum(1)          # upper bound of LCS (multiset intersection)
            for j, u in zip(kept_idx, ub):
                short = min(len(ids[i]), len(ids[j]))
                if short and NEAR_DEN * u >= NEAR_NUM * short:           # bound can reach 0.8 -> exact LCS
                    a, b = (ids[i], ids[j]) if len(ids[i]) <= len(ids[j]) else (ids[j], ids[i])
                    if NEAR_DEN * lcs_with_masks(lcs_masks(a), b) >= NEAR_NUM * short:
                        drop = True
                        break
        if drop:
            keep.append(False)
            n_near += 1
            continue
        keep.append(True)
        seen.add(norm[i])
        kept_idx.append(i)
    return keep, n_exact, n_near


def _pack(units: list[tuple[int, int]], text: str, n_tok: list[int], cap: int) -> list[tuple[int, int]]:
    spans, cur, tot = [], None, 0
    for (s, e), n in zip(units, n_tok):
        if cur is not None and tot + n > cap:
            spans.append(cur)
            cur, tot = None, 0
        cur = (s, e) if cur is None else (cur[0], e)
        tot += n
    if cur is not None:
        spans.append(cur)
    return spans


def split_cap(text: str, cap: int, tok: Tokenizer) -> list[str]:
    """Cut `text` into consecutive verbatim pieces of <= cap BGE-M3 tokens, at paragraph, then sentence, then word boundaries."""
    if len(tok.encode(text)) <= cap:
        return [text]
    budget = cap
    while True:
        units = [(m.start(), m.end()) for m in re.finditer(r'[^\n]+', text)]                     # paragraphs
        fine = []
        for s, e in units:
            if len(tok.encode(text[s:e])) <= budget:
                fine.append((s, e))
                continue
            sents, last = [], s
            for m in re.finditer(r'(?<=[.!?…;:])\s+', text[s:e]):
                sents.append((last, s + m.start()))
                last = s + m.end()
            sents.append((last, e))
            for a, b in sents:
                if len(tok.encode(text[a:b])) <= budget:
                    fine.append((a, b))
                else:                                                          # over-long sentence: words
                    fine += [(a + m.start(), a + m.end()) for m in re.finditer(r'\S+', text[a:b])]
        lens = [len(x) for x in tok.encode_many([text[s:e] for s, e in fine])]
        if max(lens) > budget:                                                 # whitespace-free blob: cut by characters
            fixed = []
            for (s, e), n in zip(fine, lens):
                if n <= budget:
                    fixed.append((s, e))
                    continue
                step = max(1, int((e - s) * budget / n * 0.9))
                fixed += [(x, min(x + step, e)) for x in range(s, e, step)]
            fine = fixed
            lens = [len(x) for x in tok.encode_many([text[s:e] for s, e in fine])]
        pieces = [text[s:e] for s, e in _pack(fine, text, lens, budget)]
        if all(len(x) <= cap for x in tok.encode_many(pieces)):
            return pieces
        budget = int(budget * 0.95)


def dist(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {'min': int(x.min()), 'p50': float(np.percentile(x, 50)), 'p95': float(np.percentile(x, 95)),
            'max': int(x.max()), 'mean': round(float(x.mean()), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--k-doc', type=int, required=True, help='docs in relevant_docs (top-k_doc of the cache, <= 100)')
    ap.add_argument('--k-chunk', type=int, default=None, help='only the top-k_chunk docs get chunks (<= k_doc); default k_doc with --zip-budget-mib')
    ap.add_argument('--chunk-mode', choices=('c2', 'full', 'window'), required=True)
    ap.add_argument('--out', required=True, help='output .zip path; "{kc}" is replaced by the k_chunk actually used; json + stats.json next to it')
    ap.add_argument('--max-zip-mib', type=float, default=0, help='if the zip is larger, lower k_chunk by 1 until it fits (0 = off)')
    ap.add_argument('--window-tokens', type=int, default=0, help='window mode: max BGE-M3 tokens of the one chunk per doc')
    ap.add_argument('--zip-budget-mib', type=float, default=0, help='largest k_chunk whose real zip is <= B MiB (binary search)')
    ap.add_argument('--runs-dir', default=str(RUNS_DIR / 'vi-k100'))
    ap.add_argument('--chunks-dir', default=str(CHUNKS_DIR))
    ap.add_argument('--docs-dir', default=str(DOCS_DIR))
    ap.add_argument('--index-dir', default=str(INDEX_DIR))
    ap.add_argument('--queries', default=str(RAW_DATA_DIR / 'query.parquet'))
    ap.add_argument('--dedupe-scope', choices=('doc', 'query'), default='doc',
                    help='doc: drop duplicates only within the same doc_id; query: across all chunks of the query')
    ap.add_argument('--max-chunk-tokens', type=int, default=0, help='cut longer chunks into verbatim pieces (0 = no limit)')
    ap.add_argument('--target', type=int, default=256)
    ap.add_argument('--extra-chunk-docs', type=int, default=0,
                    help='docs ranked k_chunk+1..N get 1 extra chunk (best reranker-scored t256 chunk); 0 = off, N <= 50')
    ap.add_argument('--extra-zip-budget-bytes', type=int, default=0,
                    help='with --extra-chunk-docs: largest N in [k_chunk, N] whose real zip is <= B bytes (0 = off)')
    ap.add_argument('--k-chunk-zip-budget-bytes', type=int, default=0,
                    help='largest k_chunk in [--k-chunk, --k-chunk-max] whose real zip is <= B bytes (0 = off)')
    ap.add_argument('--k-chunk-max', type=int, default=0, help='upper end of the --k-chunk-zip-budget-bytes search (default k_doc)')
    ap.add_argument('--doc-ranking', default=None, help='parquet (query_id, rank, doc_id): docs after the K100 cache (with --k-doc-total)')
    ap.add_argument('--k-doc-total', type=int, default=0, help='primary docs in relevant_docs incl. --doc-ranking docs (> 100)')
    a = ap.parse_args(argv)
    if a.k_chunk is None:
        if not a.zip_budget_mib:
            ap.error('--k-chunk is required without --zip-budget-mib')
        a.k_chunk = a.k_doc
    if a.k_chunk > a.k_doc or a.k_doc > 100:
        ap.error('need k_chunk <= k_doc <= 100')
    if a.chunk_mode == 'window' and a.window_tokens <= 0:
        ap.error('--chunk-mode window needs --window-tokens N')
    if a.chunk_mode == 'c2' and a.k_chunk > C2_MAX_K:
        ap.error(f'c2 needs cached chunk scores, available for the top {C2_MAX_K} docs only')
    if a.extra_chunk_docs:
        if not a.k_chunk <= a.extra_chunk_docs <= min(a.k_doc, C2_MAX_K):
            ap.error(f'need k_chunk <= --extra-chunk-docs <= min(k_doc, {C2_MAX_K}) (chunk scores cached for the top {C2_MAX_K} docs only)')
        if a.max_zip_mib or a.zip_budget_mib:
            ap.error('--extra-chunk-docs keeps k_chunk fixed: use --extra-zip-budget-bytes, not --max-zip-mib / --zip-budget-mib')
    elif a.extra_zip_budget_bytes:
        ap.error('--extra-zip-budget-bytes needs --extra-chunk-docs')
    if a.k_chunk_zip_budget_bytes:
        a.k_chunk_max = a.k_chunk_max or a.k_doc
        if not a.k_chunk <= a.k_chunk_max <= a.k_doc:
            ap.error('need k_chunk <= --k-chunk-max <= k_doc')
        if a.max_zip_mib or a.zip_budget_mib or a.extra_chunk_docs:
            ap.error('--k-chunk-zip-budget-bytes excludes --max-zip-mib, --zip-budget-mib and --extra-chunk-docs')
        if a.chunk_mode == 'c2' and a.k_chunk_max > C2_MAX_K:
            ap.error(f'c2 needs cached chunk scores, available for the top {C2_MAX_K} docs only')
    elif a.k_chunk_max:
        ap.error('--k-chunk-max needs --k-chunk-zip-budget-bytes')
    if bool(a.doc_ranking) != bool(a.k_doc_total):
        ap.error('--doc-ranking and --k-doc-total go together')
    if a.doc_ranking and (a.k_doc != 100 or a.k_doc_total <= 100):
        ap.error('--doc-ranking needs --k-doc 100 and --k-doc-total > 100')
    a.out = str(assert_writable(a.out))
    # ZIP, JSON and stats are independent destinations; an existing sidecar
    # may be redirected even when the ZIP path is safe. Check every possible
    # budget-selected basename before inputs/models or output creation.
    candidates = range(1, a.k_doc + 1) if a.zip_budget_mib else range(1, a.k_chunk + 1)
    for candidate in candidates:
        output = assert_writable(a.out.replace('{kc}', str(candidate)))
        assert_writable(output.with_suffix('.json'))
        assert_writable(output.with_suffix('.stats.json'))
    a.queries = str(resolve_path(a.queries))
    runs, chunk_root, docs_root = resolve_path(a.runs_dir), resolve_path(a.chunks_dir), resolve_path(a.docs_dir)
    required = [runs / 'vi_k100.parquet', Path(a.queries), chunk_root / 'docs.parquet']
    if a.chunk_mode == 'c2':
        required += [runs / 'vi_k100_chunk_scores.parquet', chunks_file(a.target, base=chunk_root)]
    else:
        require_inputs(docs_root)
        document_files = list(docs_root.glob('*.parquet'))
        if not document_files:
            raise ValueError(f'Missing or empty parquet input: {docs_root}')
        if not any(pq.ParquetFile(source).metadata.num_rows for source in document_files):
            raise ValueError(f'Empty document input: {docs_root}')
    if a.extra_chunk_docs:
        required += [runs / 'vi_k100_chunk_scores.parquet', chunks_file(a.target, base=chunk_root)]
    if a.doc_ranking:
        a.doc_ranking = str(resolve_path(a.doc_ranking))
        required += [Path(a.doc_ranking)]
    if a.chunk_mode == 'window':
        required += [runs / 'vi_k100_chunk_scores.parquet', chunks_file(a.target, base=chunk_root), index_dir(a.target, base=Path(a.index_dir)) / 'dense.npy']
    require_inputs(*required)
    for source in required:
        if source.suffix == '.parquet' and pq.ParquetFile(source).metadata.num_rows == 0:
            raise ValueError(f'Empty input: {source}')
    run = pq.read_table(runs / 'vi_k100.parquet').to_pylist()
    qids = [int(x) for x in pq.read_table(a.queries, columns=['id'])['id'].to_pylist()]
    top: dict[int, list[int]] = {q: [] for q in qids}
    for r in sorted(run, key=lambda r: (r['query_id'], r['rank'])):
        if r['rank'] <= a.k_doc:
            top[r['query_id']].append(r['doc_id'])
    groups = {r['doc_id']: r['doc_ids_group'] for r in pq.read_table(chunk_root / 'docs.parquet', columns=['doc_id', 'doc_ids_group']).to_pylist()}
    need = {d for v in top.values() for d in v[:max(a.k_chunk, a.k_chunk_max)]}
    more: dict[int, list[int]] = {q: [] for q in qids}                 # --doc-ranking docs after the K100 cache
    if a.doc_ranking:
        cached = {q: set(v) for q, v in top.items()}                   # docs already taken (rank <= k_doc)
        rk = pq.read_table(a.doc_ranking, columns=['query_id', 'rank', 'doc_id']).to_pandas().sort_values(['query_id', 'rank'], kind='stable')
        for q, d in zip(rk['query_id'].tolist(), rk['doc_id'].tolist()):
            if q in more and d not in cached[q] and len(top[q]) + len(more[q]) < a.k_doc_total:
                more[q].append(d)
                cached[q].add(d)

    chunk_text: dict[int, str] = {}
    best: dict[tuple[int, int], list[int]] = {}
    full_src, full = None, {}
    if a.chunk_mode == 'c2':
        cs = pq.read_table(runs / 'vi_k100_chunk_scores.parquet').to_pylist()
        per: dict[tuple[int, int], list[tuple[float, int]]] = {}
        for r in cs:
            per.setdefault((r['query_id'], r['doc_id']), []).append((r['score'], r['chunk_id']))
        for key, lst in per.items():
            lst.sort(key=lambda x: (-x[0], x[1]))                       # score desc, chunk order on ties (as sub02)
            best[key] = [c for _, c in lst[:2]]
        want = {c for v in best.values() for c in v}
        t = pq.read_table(chunks_file(a.target, base=chunk_root), columns=['chunk_id', 'text']).to_pydict()
        chunk_text = {c: x for c, x in zip(t['chunk_id'], t['text']) if c in want}
    else:
        full, full_src, off = load_full_texts(need, docs_root)
    extra_text: dict[tuple[int, int], str] = {}                        # (query, doc) -> extra chunk text
    if a.extra_chunk_docs:
        from r2ai.retrieve.exact import take_texts, text_mmap
        rank = {(q, d): k for q, v in top.items() for k, d in enumerate(v, 1)}
        bestx: dict[tuple[int, int], tuple[float, int]] = {}
        cs = pq.read_table(runs / 'vi_k100_chunk_scores.parquet', columns=['query_id', 'doc_id', 'chunk_id', 'score'])
        for q, d, c, sc in zip(*(cs[k].to_pylist() for k in ('query_id', 'doc_id', 'chunk_id', 'score'))):
            if a.k_chunk < rank.get((q, d), 0) <= a.extra_chunk_docs:
                if (q, d) not in bestx or (-sc, c) < (-bestx[(q, d)][0], bestx[(q, d)][1]):
                    bestx[(q, d)] = (sc, c)
        del cs
        col, _ = text_mmap(chunks_file(a.target, base=chunk_root), assert_writable(index_dir(a.target, base=Path(a.index_dir)) / 'text.arrow'))
        keys = sorted(bestx)
        extra_text = dict(zip(keys, take_texts(col, [bestx[k][1] for k in keys])))
        del col
    tok = Tokenizer()
    win = None
    if a.chunk_mode == 'window':
        import torch
        from r2ai.index.bge_m3 import M3Encoder
        from r2ai.retrieve.run import ws
        rows = pq.read_table(a.queries).to_pylist()
        enc = M3Encoder(max_len=512)
        qd = {}
        for b in range(0, len(rows), 32):
            d_, _ = enc.encode_batch([ws(r['query']) for r in rows[b:b + 32]])
            qd.update({int(r['id']): v for r, v in zip(rows[b:b + 32], d_)})
        del enc
        torch.cuda.empty_cache()
        rer: dict[tuple[int, int], dict[int, float]] = {}
        for r in pq.read_table(runs / 'vi_k100_chunk_scores.parquet').to_pylist():
            rer.setdefault((r['query_id'], r['doc_id']), {})[r['chunk_id']] = r['score']
        win = Windows(a.window_tokens, tok, full, off, rer, qd, a.target, chunks_dir=chunk_root, index_base=Path(a.index_dir))

    def build(kc: int, nx: int = 0):
        out, n_docs, n_chunks, lens, xlens = [], [], [], [], []
        xst = {'extra_chunk_docs': nx, 'extra_chunks': 0, 'extra_docs_without_score': 0}
        st = {'chunks_before_dedupe': 0, 'dropped_exact': 0, 'dropped_near': 0, 'chunk_docs_without_chunk': 0, 'chunk_docs': 0,
              'queries_without_any_chunk': 0, 'docs_missing_text': 0, 'chunks_cut_by_cap': 0, 'pieces_from_cut': 0,
              'window_docs_cut': 0, 'window_docs_fallback_dense': 0}
        wlen: dict[tuple[int, str], int] = {}
        for q in qids:
            docs = list(dict.fromkeys(int(x) for d in top[q] + more[q] for x in groups[d]))
            if win is not None:
                win.ensure_len(top[q][:kc])
            cand = []
            for d in top[q][:kc]:
                if a.chunk_mode == 'c2':
                    texts = [chunk_text[c] for c in best.get((q, d), [])]
                elif a.chunk_mode == 'window':
                    if d not in full or not full[d].strip():
                        texts = []
                    else:
                        wt, wn, cut, fb = win.get(q, d)
                        texts = [wt]
                        wlen[(d, wt)] = wn
                        st['window_docs_cut'] += cut
                        st['window_docs_fallback_dense'] += cut and fb
                else:
                    texts = [full[d]] if full.get(d, '').strip() else []
                if not texts:
                    st['docs_missing_text'] += 1
                cand += [(d, t) for t in texts]
            if a.max_chunk_tokens:
                new = []
                for d, t in cand:
                    ps = split_cap(t, a.max_chunk_tokens, tok)
                    if len(ps) > 1:
                        st['chunks_cut_by_cap'] += 1
                        st['pieces_from_cut'] += len(ps)
                    new += [(d, x) for x in ps]
                cand = new
            keep, ne, nn = [], 0, 0
            if cand:
                if a.dedupe_scope == 'query':
                    keep, ne, nn = dedupe([t for _, t in cand], tok)
                else:
                    keep = [True] * len(cand)
                    for d in dict.fromkeys(d for d, _ in cand):
                        idx = [i for i, (dd, _) in enumerate(cand) if dd == d]
                        if len(idx) < 2:
                            continue
                        k, e1, n1 = dedupe([cand[i][1] for i in idx], tok)
                        ne, nn = ne + e1, nn + n1
                        for i, kk in zip(idx, k):
                            keep[i] = kk
            chunks = [{'doc_id': int(d), 'chunk_text': t} for (d, t), k in zip(cand, keep) if k]
            xs = []
            for d in top[q][kc:nx]:                                     # extra: 1 chunk per doc, after the regular ones
                t = extra_text.get((q, d), '')
                if t.strip():
                    xs.append({'doc_id': int(d), 'chunk_text': t})
                else:
                    xst['extra_docs_without_score'] += 1
            xst['extra_chunks'] += len(xs)
            if xs:
                xlens += count_tokens(tok, [c['chunk_text'] for c in xs])
            chunks += xs
            st['chunks_before_dedupe'] += len(cand)
            st['dropped_exact'] += ne
            st['dropped_near'] += nn
            st['chunk_docs'] += len(top[q][:kc])
            st['chunk_docs_without_chunk'] += len(top[q][:kc]) - len({c['doc_id'] for c in chunks})
            st['queries_without_any_chunk'] += not chunks
            out.append({'id': q, 'relevant_docs': docs, 'relevant_chunks': chunks})
            n_docs.append(len(docs))
            n_chunks.append(len(chunks))
            main_chunks = chunks[:len(chunks) - len(xs)]
            lens += [wlen[(c['doc_id'], c['chunk_text'])] for c in main_chunks] if a.chunk_mode == 'window' else \
                count_tokens(tok, [c['chunk_text'] for c in main_chunks])
        lens += xlens
        if nx:
            st |= xst | {'extra_chunk_tokens_bge_m3': dist(xlens) if xlens else None}
        if a.doc_ranking:
            npr = [len(top[q]) + len(more[q]) for q in qids]
            st |= {'k_doc_total': a.k_doc_total, 'primary_docs_per_query': dist(npr) | {'p5': float(np.percentile(npr, 5))},
                   'queries_short_of_k_doc_total': sum(n < a.k_doc_total for n in npr),
                   'docs_from_ranking': sum(len(v) for v in more.values())}
        return out, n_docs, n_chunks, lens, st

    kc, tried = a.k_chunk, []
    if a.zip_budget_mib:
        import zlib
        budget = a.zip_budget_mib * 2**20

        def zsize(k: int) -> int:
            o = build(k)[0]
            c = zlib.compressobj(9, zlib.DEFLATED, -15)
            return len(c.compress(format_submission(o).encode('utf-8')) + c.flush()) + 200       # + zip headers

        lo_k, hi_k = 0, a.k_doc                                         # largest k in [0, k_doc] with zip <= budget
        while lo_k < hi_k:
            mid = (lo_k + hi_k + 1) // 2
            sz = zsize(mid)
            tried.append({'k_chunk': mid, 'zip_mib_est': round(sz / 2**20, 2)})
            print(f'budget search k_chunk={mid}: ~{sz / 2**20:.2f} MiB', flush=True)
            if sz <= budget:
                lo_k = mid
            else:
                hi_k = mid - 1
        kc = max(lo_k, 1)
        a.max_zip_mib = a.zip_budget_mib                                # the real zip is checked (and shrunk) below

    def write(k: int, nx: int):
        res = build(k, nx)
        zp = assert_writable(a.out.replace('{kc}', str(k)))
        zp.parent.mkdir(parents=True, exist_ok=True)
        js = assert_writable(zp.with_suffix('.json'))
        write_submission(res[0], js)
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(js, js.name)
        return res, zp, js

    nx = a.extra_chunk_docs
    if a.extra_chunk_docs and a.extra_zip_budget_bytes:
        lo_n, hi_n = kc, a.extra_chunk_docs                             # largest n in [k_chunk, N] with real zip <= B
        while lo_n < hi_n:
            mid = (lo_n + hi_n + 1) // 2
            _, zp, _ = write(kc, mid)
            size = zp.stat().st_size
            tried.append({'extra_chunk_docs': mid, 'zip_bytes': size})
            print(f'extra search n={mid}: zip {size} bytes', flush=True)
            if size <= a.extra_zip_budget_bytes:
                lo_n = mid
            else:
                hi_n = mid - 1
        nx = lo_n
    if a.k_chunk_zip_budget_bytes:
        lo_k, hi_k = kc, a.k_chunk_max                                 # largest k in [k_chunk, k_chunk_max] with real zip <= B
        while lo_k < hi_k:
            mid = (lo_k + hi_k + 1) // 2
            _, zp, _ = write(mid, nx)
            size = zp.stat().st_size
            tried.append({'k_chunk': mid, 'zip_bytes': size})
            print(f'k_chunk search k={mid}: zip {size} bytes', flush=True)
            if size <= a.k_chunk_zip_budget_bytes:
                lo_k = mid
            else:
                hi_k = mid - 1
        kc = lo_k
    budget_bytes = a.extra_zip_budget_bytes or a.k_chunk_zip_budget_bytes
    while True:
        (out, n_docs, n_chunks, lens, st), zp, js = write(kc, nx)
        mib = zp.stat().st_size / 2**20
        tried.append({'k_chunk': kc, 'zip_mib': round(mib, 2)} | ({'extra_chunk_docs': nx} if nx else {})
                     | ({'zip_bytes': zp.stat().st_size} if budget_bytes else {}))
        print(f'k_chunk={kc}: zip {mib:.2f} MiB', flush=True)
        if budget_bytes and zp.stat().st_size > budget_bytes:
            raise ValueError(f'zip {zp.stat().st_size} bytes > budget {budget_bytes} even at the lower end of the search')
        if not a.max_zip_mib or mib <= a.max_zip_mib or kc <= 1:
            break
        js.unlink()
        zp.unlink()
        kc -= 1
    stats = {'name': zp.stem, 'k_doc': a.k_doc, 'k_chunk_requested': a.k_chunk, 'k_chunk': kc, 'window_tokens': a.window_tokens, 'zip_budget_mib': a.zip_budget_mib, 'tried': tried,
             'max_zip_mib': a.max_zip_mib, 'chunk_mode': a.chunk_mode, 'dedupe_scope': a.dedupe_scope,
             'max_chunk_tokens': a.max_chunk_tokens, 'queries': len(out), **st,
             'docs_per_query': dist(n_docs), 'chunks_per_query': dist(n_chunks),
             'chunk_tokens_bge_m3': dist(lens) if lens else None,
             'json_bytes': js.stat().st_size, 'zip_bytes': zp.stat().st_size, 'full_text_source_counts': full_src}
    assert_writable(zp.with_suffix('.stats.json')).write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
