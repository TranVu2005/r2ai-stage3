"""S26 P-A3 / P-A3r / P-A4: choose docs by BGE-M3 doc-level cosine (canonical doc_text, 512 tokens), not by P1's order.

Per query, K_q = len(relevant_docs) of the same row of P1-vi+10.
  pa3   top-K_q canonical docs by cosine(query, doc) over every doc of the given doc-vector file
        (pa2_doc_dense.npz = the 125,605 docs of P1-vi+10; full_doc_dense.npz = all 887,151 => P-A4).
  pa3r  top-K_q by RRF(k=60) of the P1 row order (canonical docs, first occurrence) and the doc-level order.
No zh gate (tau on BGE chunk-level) and no language quota: doc-level cosine decides for vi and zh alike.
One corpus ID per canonical doc: the ID P1-vi+10 used for it (this row first, then any row), else the canonical doc_id
when it is in its own alias group, else the first alias. P1 IDs without a canonical doc (no vector) are not ranked.
Chunks: exactly 1 per chosen doc. If P1-vi+10 ships a chunk for that corpus ID (this row first, then any row), that text
byte for byte; otherwise the canonical chunks_t256 chunk of that doc with the highest dense cosine to the query
(canonical chunk vectors, BGE-M3 dense CLS, vectors.json alignment); docs with no vector chunk take their first chunk.
With a 6th argument `nc`, only the first n_c chosen docs (variant order) get a chunk, n_c = len(relevant_chunks) of the
same P1-vi+10 row, so the chunk count per query equals P1-vi+10 (same text rule, 1 chunk per doc).
Options after `nc`: `k=<x>` scales K_q (round(K_q*x)); `ncx=<x>` scales n_c (round(n_c*x), at most K_q);
`ref=<zip>` reports the chosen canonical docs that are not in the same row of that ZIP (default pa3/P-A3-nc.zip).
In `nc` mode the chunks go to the highest-ranked chosen docs that can carry one (P1 text or >=1 t256 chunk); a chosen
doc with no chunk source (no chunks_t256 row for it or for its alias-group owner: the thin docs) stays in relevant_docs and its chunk goes to the
next doc, so every row keeps its target chunk count when enough carriers exist.
`new=body`: a chosen doc without a P1 chunk gets its whole body rebuilt from canonical (body_check.py rule, 99.0% byte
match with P1's own chunks on 200 samples) instead of a t256 chunk; if the ZIP reaches the ceiling, every such body is
cut to its first N chars (verbatim prefix), N the largest found by bisection that fits (reported as body_cut_chars).
`extra=<dir>[,<dir>...]`: add S15 handoff lots (e.g. S27 pharmacity) to the doc pool. Each <dir> holds
lots/<lot>/chunks/{docs,chunks_t256}.parquet; vectors are read from <vectors_root>/<lot>/dense_doc_level.npz with
`extravec=<vectors_root>` (same encoder/settings as full_embed.py). Extra docs are vi, own corpus ID, body = their body
chunk span (same rule as canonical). The report adds extra_chosen / extra_queries / extra_selection_rate.
`dry`: stop after the doc choice and stats (no chunk texts, no ZIP).
Usage: pa3_build.py <out_dir> <scratch_dir> <doc_vectors.npz> <name> {pa3|pa3r} [nc] [k=x] [ncx=x] [ref=zip] [new=body]
                    [extra=dir extravec=dir] [dry] [sel=ku|cos|rel tmean=x] [compact]
`sel` (mode pa3 only) replaces per-query K_q by: ku = round(tmean) docs for every query; cos = docs with cosine >= theta,
one global theta; rel = docs with cosine >= cos(top-1) - delta, one global delta. theta/delta are bisected so the mean
number of docs per query is as close to tmean as the integer counts allow (reported as sel_param / sel_mean).
"""
import hashlib, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.compose.base_io import lzma_dictionary  # noqa: E402

ZP = 'out/runs/reranker-trial/release/P1-vi+10.zip'; CEIL = 102_600_000; DATE_TIME = (2026, 10, 19, 0, 0, 0); RRF_K = 60
CAN = 'data/canonical-20261018'
out, S, vec, name, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
NC, DRY = 'nc' in sys.argv[6:], 'dry' in sys.argv[6:]
SEP = (',', ':') if 'compact' in sys.argv[6:] else None   # compact: no spaces after , and : inside a row
OPT = dict(a.split('=', 1) for a in sys.argv[6:] if '=' in a)
KX, NCX = float(OPT.get('k', 1)), float(OPT.get('ncx', 1))
NEW = OPT.get('new', 't256'); assert NEW in ('t256', 'body')
SEL, TMEAN = OPT.get('sel', 'k'), float(OPT.get('tmean', 0)); assert SEL in ('k', 'ku', 'cos', 'rel')
REF = Path(OPT.get('ref', 'out/runs/s26-gold-structure/pa3/P-A3-nc.zip'))
out.mkdir(parents=True, exist_ok=True); zpath = out / f'{name}.zip'
if zpath.exists():
    raise SystemExit(f'{zpath} exists')
t0 = time.time()
rows = []
with zipfile.ZipFile(ZP).open('P1-vi+10.json') as f:
    for line in f:
        line = line.strip().rstrip(b',')
        if line.startswith(b'{'):
            rows.append(json.loads(line))
gtext = {}                                              # corpus ID -> first chunk text P1 ships for it
for r in rows:
    for c in r['relevant_chunks']:
        gtext.setdefault(c['doc_id'], c['chunk_text'])
ct = pq.read_table(f'{CAN}/chunks_t256.parquet', columns=['doc_id', 'has_vector', 'doc_ids'])
ch_doc, ch_has = ct['doc_id'].to_numpy(), ct['has_vector'].to_numpy()
_, first = np.unique(ch_doc, return_index=True)
grp = dict(zip(ch_doc[first].tolist(), ct['doc_ids'].take(first).to_pylist())); del ct
cid2doc = {int(a): int(d) for d, ids in grp.items() for a in ids}
dq = pq.read_table(f'{CAN}/docs.parquet', columns=['doc_id', 'lang', 'quality']).to_pandas().set_index('doc_id')
doc_lang, thin = dq.lang, set(dq.index[dq.quality == 'thin'].tolist())
# alias docs (rule a): no chunks of their own; the canonical doc whose chunks_t256 alias group lists them holds the same
# text and the chunks (phys_id is not used: it often points to a doc absent from docs.parquet)
phys = {int(d): cid2doc[int(d)] for d in dq.index if d not in grp and int(d) in cid2doc}; del dq


def cdoc(d):
    """canonical doc that holds the chunks of d (d itself, or the owner of its alias group)"""
    return d if d in grp else phys.get(d, d)


ref = {}                                                # query id -> canonical docs of the reference ZIP
if REF.exists():
    with zipfile.ZipFile(REF) as zr, zr.open(zr.namelist()[0]) as f:
        for line in f:
            line = line.strip().rstrip(b',')
            if line.startswith(b'{'):
                rr = json.loads(line); ref[rr['id']] = {cid2doc.get(c, -c) for c in rr['relevant_docs']}
p1_any = {}                                             # canonical doc -> corpus ID P1 used for it in any row
for r in rows:
    for cid in r['relevant_docs']:
        d = cid2doc.get(cid)
        if d is not None:
            p1_any.setdefault(d, cid)


def corpus_id(d, row_ids):
    if d in row_ids:
        return row_ids[d]
    if d in p1_any:
        return p1_any[d]
    g = list(grp.get(d, [d]))
    return d if d in g else int(g[0])


E = np.load(vec); ids = E['doc_id'].astype(np.int64); Dh = E['dense']
extra_text, extra_spans = {}, {}                        # extra doc -> doc_text / {field: (start, end)}
if OPT.get('extra'):
    xi, xd = [ids], [Dh]
    for root in OPT['extra'].split(','):
        for lot in sorted((Path(root) / 'lots').iterdir()):
            xdoc = pd.read_parquet(lot / 'chunks' / 'docs.parquet', columns=['doc_id', 'doc_text'])
            xch = pd.read_parquet(lot / 'chunks' / 'chunks_t256.parquet', columns=['doc_id', 'field', 'char_start', 'char_end'])
            XV = np.load(Path(OPT['extravec']) / lot.name / 'dense_doc_level.npz')
            assert sorted(XV['doc_id'].tolist()) == sorted(xdoc.doc_id.tolist()), lot
            extra_text.update(zip(xdoc.doc_id.astype(int), xdoc.doc_text))
            for (d, f), g in xch.groupby(['doc_id', 'field']):
                extra_spans.setdefault(int(d), {})[f] = (int(g.char_start.min()), int(g.char_end.max()))
            xi.append(XV['doc_id'].astype(np.int64)); xd.append(XV['dense'])
    clash = set(extra_text) & (set(ids.tolist()) | set(cid2doc))
    assert not clash, f'extra doc IDs already in canonical: {sorted(clash)[:5]}'
    ids, Dh = np.concatenate(xi), np.concatenate(xd)
    for d in extra_text:                                # own group, vi, chunk source = its own spans
        grp[d] = [d]; cid2doc[d] = d
    doc_lang = pd.concat([doc_lang, pd.Series('vi', index=list(extra_text))])
    print(f'extra docs {len(extra_text)}', flush=True)
pos = {int(d): i for i, d in enumerate(ids)}
Q = np.load(S / 'q_dense.npy').astype(np.float32); qpos = {int(q): i for i, q in enumerate(np.load(S / 'q_ids.npy'))}
M = 3 * max(round(len(r['relevant_docs']) * KX) for r in rows)      # per query: doc-level top-M (index, score), blockwise fp32
if SEL in ('cos', 'rel'):
    M = max(M, int(OPT.get('m', 4000)))
top_i = np.zeros((len(Q), 0), np.int64); top_s = np.zeros((len(Q), 0), np.float32)
for b in range(0, len(ids), 50_000):
    sc = Q @ Dh[b:b + 50_000].astype(np.float32).T
    m = min(M, sc.shape[1]); part = np.argpartition(-sc, m - 1, axis=1)[:, :m]
    top_i = np.c_[top_i, part + b]; top_s = np.c_[top_s, np.take_along_axis(sc, part, 1)]
    keep = np.argsort(-top_s, axis=1, kind='stable')[:, :M]
    top_i, top_s = np.take_along_axis(top_i, keep, 1), np.take_along_axis(top_s, keep, 1)
del Dh, E
print(f'loaded {len(rows)} rows, {len(ids)} doc vectors, top-{M} done {time.time() - t0:.0f}s', flush=True)
selk, sel_param = {}, None                              # query row index -> number of docs (sel modes)
if SEL == 'ku':
    selk = {qpos[r['id']]: round(TMEAN) for r in rows}
elif SEL in ('cos', 'rel'):
    def counts(x):
        thr = x if SEL == 'cos' else top_s[:, :1] - x
        return (top_s >= thr).sum(1)
    lo, hi = (float(top_s.min()), float(top_s.max())) if SEL == 'cos' else (0.0, float((top_s[:, :1] - top_s).max()))
    for _ in range(60):                                 # counts fall with theta, rise with delta
        mid = (lo + hi) / 2; mm = counts(mid).mean()
        if (mm > TMEAN) == (SEL == 'cos'):
            lo = mid
        else:
            hi = mid
    sel_param = min((abs(counts(x).mean() - TMEAN), x) for x in (lo, hi))[1]; cnt = counts(sel_param)
    selk = {qi: int(c) for qi, c in enumerate(cnt)}
    print(f'sel {SEL} param {sel_param:.6f} mean {cnt.mean():.4f} saturated(M={M}) {(cnt >= M).sum()} zero {(cnt == 0).sum()}', flush=True)

picked = []                                             # per row: list of (canonical doc, corpus ID)
stats = []
for r in rows:
    qi, K = qpos[r['id']], round(len(r['relevant_docs']) * KX)
    row_ids, order_p1 = {}, []
    for cid in r['relevant_docs']:
        d = cid2doc.get(cid)
        if d is not None and d not in row_ids:
            row_ids[d] = cid; order_p1.append(d)
    if SEL != 'k':
        K = selk[qi]
    if mode == 'pa3':
        chosen = [int(ids[i]) for i in top_i[qi, :K]]
    else:                                               # RRF over P1 order and doc-level order (doc-level top 3*max K)
        sc = {}
        for rk, i in enumerate(top_i[qi], 1):
            sc[int(ids[i])] = sc.get(int(ids[i]), 0) + 1 / (RRF_K + rk)
        for rk, d in enumerate(order_p1, 1):
            if d in pos:                                # rankable docs only (same rule as pa3)
                sc[d] = sc.get(d, 0) + 1 / (RRF_K + rk)
        chosen = sorted(sc, key=lambda d: -sc[d])[:K]
    picked.append([(d, corpus_id(d, row_ids)) for d in chosen])
    p1set = set(order_p1)
    stats.append({'query_id': r['id'], 'K': K, 'p1_canon': len(order_p1), 'new_vs_p1': sum(d not in p1set for d in chosen),
                  'zh_new': sum(doc_lang.get(d) == 'zh' for d in chosen), 'vi_new': sum(doc_lang.get(d) == 'vi' for d in chosen),
                  'zh_p1': sum(doc_lang.get(d) == 'zh' for d in order_p1), 'vi_p1': sum(doc_lang.get(d) == 'vi' for d in order_p1),
                  'new_vs_ref': sum(d not in ref[r['id']] for d in chosen) if ref else None,
                  'no_chunk_docs': sum(cdoc(d) not in grp for d in chosen), 'thin_docs': sum(d in thin for d in chosen)})
print(f'docs chosen {time.time() - t0:.0f}s', flush=True)

# chunk texts: P1 text (this row, then any row) else best t256 chunk of the canonical doc by dense cosine


def carriers(r, pk):
    """chosen docs that get a chunk: all of them, or (nc) the first n_c*NCX able to carry one, in variant order"""
    if not NC:
        return pk
    own, n = {c['doc_id'] for c in r['relevant_chunks']}, min(round(len(r['relevant_chunks']) * NCX), len(pk))
    return [x for x in pk if x[1] in own or x[1] in gtext or cdoc(x[0]) in grp][:n]


xstat = {}
if extra_text:
    xc = [sum(d in extra_text for d, _ in pk) for pk in picked]
    xs = {d for pk in picked for d, _ in pk if d in extra_text}
    xstat = {'extra_docs': len(extra_text), 'extra_chosen': int(sum(xc)), 'extra_queries': int(sum(c > 0 for c in xc)),
             'extra_distinct_chosen': len(xs), 'extra_selection_rate': len(xs) / len(extra_text),
             'extra_chunked': int(sum(sum(d in extra_text for d, _ in carriers(r, pk)) for r, pk in zip(rows, picked)))}
    print(json.dumps(xstat), flush=True)
if DRY:
    pd.DataFrame(stats).to_parquet(out / f'{name}.query_stats.parquet', index=False)
    (out / f'{name}.dry.json').write_text(json.dumps(xstat, indent=1), encoding='utf-8')
    raise SystemExit(0)


need = {}                                               # canonical doc -> list of row indexes needing a t256 chunk
for k, (r, pk) in enumerate(zip(rows, picked)):
    own = {c['doc_id']: c['chunk_text'] for c in r['relevant_chunks']}
    for d, cid in carriers(r, pk):
        if cid not in own and cid not in gtext:
            need.setdefault(cdoc(d), []).append(k)
if NEW == 't256':
    vrow_all = np.where(ch_has, np.cumsum(ch_has) - 1, -1)
    r_need = np.flatnonzero(np.isin(ch_doc, np.fromiter(need, np.int64)))
    sub = pd.DataFrame({'index': r_need, 'doc_id': ch_doc[r_need], 'vrow': vrow_all[r_need]})   # index = chunk row
    V = np.load(f'{CAN}/vectors/dense.npy', mmap_mode='r')
    vr = sub.vrow.values; ok = vr >= 0
    CV = np.zeros((len(sub), V.shape[1]), np.float16)
    o = np.argsort(vr[ok]); rows_ok = np.flatnonzero(ok)[o]
    for b in range(0, len(rows_ok), 200_000):
        sel = rows_ok[b:b + 200_000]; CV[sel] = V[vr[sel]]
    print(f'chunk vectors {len(sub)} ({ok.sum()} with vector) {time.time() - t0:.0f}s', flush=True)
    best_row = {}                                           # (row k, canonical doc) -> chunk row
    for d, g in sub.groupby('doc_id'):
        gi = g.index.values; has = ok[gi]          # gi = positions in sub (RangeIndex)
        for k in need[d]:
            if has.any():
                sc = CV[gi[has]].astype(np.float32) @ Q[qpos[rows[k]['id']]]; best_row[(k, d)] = int(g['index'].values[has][np.argmax(sc)])
            else:
                best_row[(k, d)] = int(g['index'].values[0])
    want = np.array(sorted(set(best_row.values())), np.int64); ctext = {}
    pf = pq.ParquetFile(f'{CAN}/chunks_t256.parquet'); lo = 0
    for g in range(pf.num_row_groups):                      # read only row groups holding wanted chunk rows
        n = pf.metadata.row_group(g).num_rows; w = want[(want >= lo) & (want < lo + n)]
        if len(w):
            ctext.update(zip(w.tolist(), pf.read_row_group(g, columns=['text'])['text'].take(w - lo).to_pylist()))
        lo += n
    print(f't256 chunks {len(want)} {time.time() - t0:.0f}s', flush=True)

else:                                                   # whole body rebuilt from canonical (body_check.py rule)
    from body_check import body_span, load_spans
    # text from the chosen doc itself (always in docs.parquet), field spans from the doc holding its chunks: for an
    # alias doc whose group owner is absent from docs.parquet the owner's offsets index the alias text (verified)
    tdocs = {d for r, pk in zip(rows, picked) for d, cid in carriers(r, pk)
             if cid not in gtext and cid not in {c['doc_id'] for c in r['relevant_chunks']}}
    bt = pq.read_table(f'{CAN}/docs.parquet', columns=['doc_id', 'lang', 'doc_text'], filters=[('doc_id', 'in', sorted(tdocs - set(extra_text)))])
    spans = load_spans({cdoc(d) for d in tdocs - set(extra_text)})
    btext = {cdoc(int(d)): body_span(t, lg, spans.get(cdoc(int(d)), {})) for d, lg, t in
             zip(bt['doc_id'].to_pylist(), bt['lang'].to_pylist(), bt['doc_text'].to_pylist())}; del bt
    btext.update({d: body_span(extra_text[d], 'vi', extra_spans.get(d, {})) for d in tdocs & set(extra_text)})
    print(f'canonical bodies {len(btext)} {time.time() - t0:.0f}s', flush=True)

new_rows, src = [], {'p1_row': 0, 'p1_any': 0, 't256': 0, 'body': 0, 'none': 0}
for k, (r, pk) in enumerate(zip(rows, picked)):
    own = {c['doc_id']: c['chunk_text'] for c in r['relevant_chunks']}
    chunks = []
    for d, cid in carriers(r, pk):
        if cid in own:
            t = own[cid]; src['p1_row'] += 1
        elif cid in gtext:
            t = gtext[cid]; src['p1_any'] += 1
        elif NEW == 't256' and (k, cdoc(d)) in best_row:
            t = ctext[best_row[(k, cdoc(d))]]; src['t256'] += 1
        elif NEW == 'body' and btext.get(cdoc(d), '').strip():
            t = btext[cdoc(d)]; src['body'] += 1
            chunks.append({'doc_id': int(cid), 'chunk_text': t, '_new': True}); continue
        else:
            src['none'] += 1; continue
        chunks.append({'doc_id': int(cid), 'chunk_text': t})
    new_rows.append({'id': r['id'], 'relevant_docs': [int(c) for _, c in pk], 'relevant_chunks': chunks})
st = pd.DataFrame(stats); st.to_parquet(out / f'{name}.query_stats.parquet', index=False)
tmp = zpath.with_suffix(f'.{os.getpid()}.zip.tmp')       # per-process: concurrent builds never share a tmp


def build(N):
    """JSON payload with every new-doc body cut to its first N chars (None = whole), and its ZIP size"""
    def row(r):
        return {**r, 'relevant_chunks': [{'doc_id': c['doc_id'], 'chunk_text': c['chunk_text'][:N] if c.get('_new') and N else c['chunk_text']}
                                         for c in r['relevant_chunks']]}
    payload = ('[\n' + ',\n'.join(json.dumps(row(r), ensure_ascii=False, separators=SEP) for r in new_rows) + '\n]\n').encode('utf-8')
    with lzma_dictionary(256), zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_LZMA) as z:
        info = zipfile.ZipInfo(f'{name}.json', date_time=DATE_TIME); info.compress_type = zipfile.ZIP_LZMA
        with z.open(info, 'w', force_zip64=True) as f:
            f.write(payload)
    print(f'  N={N} json {len(payload)} zip {tmp.stat().st_size} {time.time() - t0:.0f}s', flush=True)
    return payload, tmp.stat().st_size


N, tries = None, []
payload, size = build(None); tries.append((None, size))
if size >= CEIL and NEW == 'body' and src['body']:
    # largest N with ZIP < CEIL. ZIP size is taken as linear in the JSON bytes of the new-doc bodies: pick N whose JSON
    # size the line through the last two measured points puts just under the ceiling (secant), measure, repeat; keep
    # the largest N that fits, stop when it is within 0.5% of the ceiling or after 8 measurements.
    newb = [c['chunk_text'] for r in new_rows for c in r['relevant_chunks'] if c.get('_new')]
    nb_full = sum(len(t.encode('utf-8')) for t in newb)

    def nbytes(n):                                      # UTF-8 bytes of all new bodies cut to n chars
        return sum(len(t[:n].encode('utf-8')) for t in newb)

    def n_for(target):                                  # largest n with nbytes(n) <= target (bisection, no compression)
        lo, hi = 1, max(len(t) for t in newb)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if nbytes(mid) <= target else (lo, mid - 1)
        return lo
    pts = [(nb_full, size)]; slope = size / len(payload); best = None
    while len(tries) < 9:
        x1, y1 = pts[-1]
        target_nb = x1 - (y1 - CEIL * 0.997) / slope
        n = n_for(max(target_nb, 0))
        if any(t == n for t, _ in tries):
            break
        pl, sz = build(n); tries.append((n, sz)); pts.append((nbytes(n), sz))
        if sz < CEIL and (best is None or n > best[0]):
            best = (n, pl, sz)
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        slope = (yb - ya) / (xb - xa) if xb != xa and yb != ya else slope
        if best and CEIL - best[2] < CEIL * 0.005:
            break
    if best:
        N, payload, size = best
        if tries[-1][0] != N:                               # leave the chosen N in tmp
            payload, size = build(N)
cut_n = [len(c['chunk_text']) for r in new_rows for c in r['relevant_chunks'] if c.get('_new') and N and len(c['chunk_text']) > N]
rep = {**xstat, 'body_cut_chunks': len(cut_n), 'body_cut_chars_removed': int(sum(x - N for x in cut_n)) if cut_n else 0, 'sel': SEL, 'sel_tmean': TMEAN or None, 'sel_param': sel_param, 'docs_per_query_quantiles': st.K.quantile([0, .1, .5, .9, 1]).to_dict(), 'name': name, 'mode': mode, 'new_doc_chunk': NEW, 'body_cut_chars': N, 'size_tries': tries, 'chunks_rule': 'n_c of P1 row' if NC else '1 per chosen doc', 'doc_vectors': str(vec), 'n_doc_vectors': len(ids), 'base': ZP, 'queries': len(new_rows),
       'zip_bytes': size, 'json_bytes': len(payload), 'json_sha256': hashlib.sha256(payload).hexdigest(), 'ceiling': CEIL,
       'docs_total': int(st.K.sum()), 'chunks_total': sum(len(r['relevant_chunks']) for r in new_rows),
       'new_body_chars_mean': float(np.mean([len(c['chunk_text']) for r in new_rows for c in r['relevant_chunks'] if c.get('_new')] or [0])), 'chunk_source': src,
       'new_vs_p1_mean': float(st.new_vs_p1.mean()), 'new_vs_p1_quantiles': st.new_vs_p1.quantile([.1, .5, .9]).to_dict(),
       'zh_per_query': float(st.zh_new.mean()), 'vi_per_query': float(st.vi_new.mean()),
       'zh_per_query_p1': float(st.zh_p1.mean()), 'vi_per_query_p1': float(st.vi_p1.mean()),
       'zh_share': float(st.zh_new.sum() / st.K.sum()), 'zh_share_p1': float(st.zh_p1.sum() / (st.zh_p1 + st.vi_p1).sum()),
       'k_mult': KX, 'nc_mult': NCX, 'ref': str(REF) if ref else None,
       'new_vs_ref_mean': float(st.new_vs_ref.mean()) if ref else None,
       'new_vs_ref_quantiles': st.new_vs_ref.quantile([.1, .5, .9]).to_dict() if ref else None,
       'no_chunk_docs_total': int(st.no_chunk_docs.sum()), 'thin_docs_total': int(st.thin_docs.sum()),
       'chunk_target': int(sum(min(round(len(r['relevant_chunks']) * NCX), K) for r, K in zip(rows, st.K))) if NC else None}
if size >= CEIL:
    (out / f'{name}.build.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    raise SystemExit(f'ZIP {size} >= ceiling {CEIL}; not published ({tmp})\n' + json.dumps(rep, indent=1))
tmp.rename(zpath)
rep['zip_sha256'] = hashlib.sha256(zpath.read_bytes()).hexdigest()
(out / f'{name}.build.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
print(json.dumps(rep, indent=1), flush=True)
