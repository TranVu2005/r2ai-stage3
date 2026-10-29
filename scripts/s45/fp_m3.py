"""S45 retriever fingerprint: BGE-M3 variants on the S26 zh-sample yardstick (kept in corpus v2 vs dropped).

Same data and metric as reports/s26-retriever-fingerprint.md section 3 (pool_fingerprint_doclen.py): 29,385 docs of the
pre-v2 2% zh sample, kept = any id of the doc group in links_corpus_vi_en_zh; 1,200 Vietnamese queries; per variant a
[1200, docs] score matrix -> AUC best, AUC -min-rank, kept share at min-rank 1, R-prec, AUC -min-rank per domain.

One forward per text variant yields four score modes: dense CLS (the S26 baseline), dense mean-pool, sparse lexical,
ColBERT (FlagEmbedding BGEM3 formulas, re-implemented with plain transformers like r2ai.index.bge_m3).
Scores are cached as <out>/<variant>.<mode>.npy (float16 [1200, docs]); fusions are computed from the cache.

  fp_m3.py <out> prep                        labels, query reps (all heads), token counts
  fp_m3.py <out> enc <variant>[,<variant>]   encode doc variant(s) -> score cache (resumes per 2,048-doc block)
  fp_m3.py <out> chunk                       chunk-level max / mean / top-2 mean from the t256 index vectors
  fp_m3.py <out> report                      AUC table of every cached matrix + fusion sweeps -> report.json
"""
import json, os, sys, time, unicodedata, re
from pathlib import Path
import numpy as np, pandas as pd, torch
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from sklearn.metrics import roc_auc_score  # noqa: E402

ARCH = Path(os.environ.get('S45_ARCHIVE', 'D:/_r2ai_archive/data'))
DOCS = ARCH / 'chunks_zh_sample/docs.parquet'
SEG = ARCH / 'index/zh_full_segments/seg00001-sample-old-zh_sample_t256'
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BLOCK = 2048
TOK_BUDGET = 8192          # padded tokens per forward batch (fp16, RTX 3050 6 GB)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def load_docs():
    d = pd.read_parquet(DOCS, columns=['doc_id', 'doc_ids_group', 'domain', 'title', 'doc_text'])
    d['title'] = d.title.fillna(''); d['doc_text'] = d.doc_text.fillna('')
    return d


class M3Heads:
    """BGE-M3 with dense (CLS / mean), sparse and ColBERT heads; mirrors FlagEmbedding BGEM3Model."""

    def __init__(self, max_len=512, fp16=True):
        from r2ai.index.bge_m3 import M3Encoder, _snapshot, M3
        base = M3Encoder(max_len=max_len, fp16=fp16)
        self.tok, self.model, self.sparse, self.unused, self.max_len = base.tok, base.model, base.sparse, base.unused, max_len
        self.colbert = torch.nn.Linear(1024, 1024)
        self.colbert.load_state_dict(torch.load(_snapshot(M3) / 'colbert_linear.pt', map_location='cpu'))
        self.colbert = self.colbert.to('cuda', dtype=self.model.dtype).eval()

    @torch.inference_mode()
    def forward(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors='pt').to('cuda')
        h = self.model(**enc).last_hidden_state
        m = enc['attention_mask']
        cls = torch.nn.functional.normalize(h[:, 0].float(), dim=-1)
        mean = torch.nn.functional.normalize((h.float() * m[..., None]).sum(1) / m.sum(1, keepdim=True), dim=-1)
        w = torch.relu(self.sparse(h)).squeeze(-1).float()
        w = w.masked_fill(torch.isin(enc['input_ids'], self.unused) | (m == 0), 0)
        col = torch.nn.functional.normalize((self.colbert(h[:, 1:]) * m[:, 1:, None]).float(), dim=-1)  # [B, L-1, 1024]
        return enc['input_ids'], m, cls, mean, w, col


def sparse_csr(ids, w, V):
    """per text: max weight per token id (FlagEmbedding _process_token_weights) -> csr [B, V]"""
    import scipy.sparse as sp
    ids, w = ids.cpu().numpy(), w.cpu().numpy(); rows, cols, vals = [], [], []
    for i in range(len(ids)):
        k = w[i] > 0
        s = pd.Series(w[i][k]).groupby(ids[i][k]).max()
        rows.append(np.full(len(s), i)); cols.append(s.index.values); vals.append(s.values)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(ids), V))


def prep():
    docs = load_docs()
    v2 = set(pd.read_parquet('data/raw/links_corpus_vi_en_zh.parquet', columns=['id']).id)
    kept = docs.doc_ids_group.map(lambda g: any(int(x) in v2 for x in (json.loads(g) if isinstance(g, str) else g)))
    np.save(OUT / 'y.npy', kept.values.astype(np.int8))
    log('docs', len(docs), 'kept', int(kept.sum()))
    q = pd.read_parquet('data/raw/query.parquet').sort_values('id')
    np.save(OUT / 'q_ids.npy', q.id.values)
    m3 = M3Heads()
    cls, mean, cols, qtok, csr = [], [], [], [], []
    for i in range(0, len(q), 32):
        ids, m, c, mn, w, col = m3.forward(q['query'].tolist()[i:i + 32])
        cls.append(c.cpu().numpy()); mean.append(mn.cpu().numpy()); csr.append(sparse_csr(ids, w, len(m3.tok)))
        for j in range(len(ids)):
            n = int(m[j].sum()) - 1                     # tokens after CLS, EOS kept (FlagEmbedding _process_colbert_vecs)
            cols.append(col[j, :n].half().cpu().numpy()); qtok.append(n)
    import scipy.sparse as sp
    np.save(OUT / 'q_cls.npy', np.concatenate(cls)); np.save(OUT / 'q_mean.npy', np.concatenate(mean))
    sp.save_npz(OUT / 'q_sparse.npz', sp.vstack(csr).tocsr())
    np.save(OUT / 'q_col.npy', np.concatenate(cols)); np.save(OUT / 'q_col_len.npy', np.array(qtok))
    # token counts of the full doc_text / title / body (no truncation, specials excluded)
    tok = m3.tok
    for name, texts in [('doc_text', docs.doc_text), ('title', docs.title), ('body', [body_of(t, x) for t, x in zip(docs.title, docs.doc_text)])]:
        n = np.array([len(tok(x, add_special_tokens=False)['input_ids']) for x in texts])
        np.save(OUT / f'ntok_{name}.npy', n)
        log(name, 'tokens > 510:', int((n > 510).sum()), f'({(n > 510).mean():.4f})', 'median', int(np.median(n)))


def body_of(t, x):
    return x[len(t) + 2:] if t and x.startswith(t + '\n\n') else x


def nl2sp(x):
    return re.sub(r'\s+', ' ', x).strip()


# variant -> (text function over (title, doc_text), max_len, restrict-to-long-docs)
VARIANTS = {
    'dt512': (lambda t, x: x[:4096], 512, False),                       # S26 baseline (full_embed.py)
    'title': (lambda t, x: t, 512, False),
    'body512': (lambda t, x: body_of(t, x)[:4096], 512, False),
    'dt1024': (lambda t, x: x[:8192], 1024, True),
    'dt2048': (lambda t, x: x[:16384], 2048, True),
    'dt8192': (lambda t, x: x, 8192, True),
    'nfkc': (lambda t, x: unicodedata.normalize('NFKC', x)[:4096], 512, False),
    'nl2sp': (lambda t, x: nl2sp(x)[:4096], 512, False),
    'join_sp': (lambda t, x: (t + ' ' + body_of(t, x))[:4096] if x.startswith(t + '\n\n') else x[:4096], 512, False),
    'join_dot': (lambda t, x: (t + '. ' + body_of(t, x))[:4096] if x.startswith(t + '\n\n') else x[:4096], 512, False),
    'raw_full512': (lambda t, x: x, 512, False),                         # no 4,096-char pre-cut
}


def encode(variant, modes=('cls', 'mean', 'sparse', 'colbert')):
    fn, L, only_long = VARIANTS[variant]
    docs = load_docs(); N = len(docs)
    texts = [fn(t, x) for t, x in zip(docs.title, docs.doc_text)]
    sel = np.arange(N)
    if only_long:   # docs with <= 510 content tokens get exactly the dt512 input -> reuse its scores
        sel = np.flatnonzero(np.load(OUT / 'ntok_doc_text.npy') > 510)
        log(variant, 'docs re-encoded', len(sel), f'of {N} ({len(sel) / N:.4f})')
    import scipy.sparse as sp
    qc = torch.from_numpy(np.load(OUT / 'q_cls.npy')).cuda().half()
    qm = torch.from_numpy(np.load(OUT / 'q_mean.npy')).cuda().half()
    qs = sp.load_npz(OUT / 'q_sparse.npz').astype(np.float32)
    qcol = torch.from_numpy(np.load(OUT / 'q_col.npy')).cuda()
    qlen = np.load(OUT / 'q_col_len.npy'); qseg = torch.from_numpy(np.repeat(np.arange(len(qlen)), qlen)).cuda()
    qlen_t = torch.from_numpy(qlen).cuda().float()
    m3 = M3Heads(max_len=L); V = len(m3.tok)
    ntok = np.load(OUT / 'ntok_doc_text.npy')
    order = sel[np.argsort(ntok[sel], kind='stable')]                    # length-sorted -> little padding
    bdir = OUT / f'blocks_{variant}'; bdir.mkdir(exist_ok=True)
    for b0 in range(0, len(order), BLOCK):
        f = bdir / f'{b0:06d}.npz'
        if f.exists():
            continue
        idx = order[b0:b0 + BLOCK]; res = {k: np.zeros((len(qlen), len(idx)), np.float16) for k in modes}
        i, t0 = 0, time.time()
        while i < len(idx):
            est = min(L, max(8, int(ntok[idx[min(i + 7, len(idx) - 1)]] * 1.1) + 2))
            bs = max(1, min(64, TOK_BUDGET // est))
            bi = idx[i:i + bs]
            ids, m, c, mn, w, col = m3.forward([texts[k] for k in bi])
            sl = slice(i, i + len(bi))
            if 'cls' in modes:
                res['cls'][:, sl] = (qc @ c.half().T).float().cpu().numpy()
            if 'mean' in modes:
                res['mean'][:, sl] = (qm @ mn.half().T).float().cpu().numpy()
            if 'sparse' in modes:
                res['sparse'][:, sl] = (qs @ sparse_csr(ids, w, V).T).toarray()
            if 'colbert' in modes:
                dm = m[:, 1:].bool()                                       # [B, L-1]
                for j in range(len(bi)):
                    d = col[j][dm[j]].half()                                # [Ld, 1024]
                    s = (qcol @ d.T).float().max(1).values                  # [Tq]
                    res['colbert'][:, i + j] = (torch.zeros(len(qlen), device='cuda').index_add_(0, qseg, s) / qlen_t).cpu().numpy()
            i += len(bi)
        np.savez(f, idx=idx, **res)
        log(variant, f'block {b0}/{len(order)}', f'{len(idx) / (time.time() - t0):.1f} doc/s')
    del m3; torch.cuda.empty_cache()
    # assemble; long-only variants start from the dt512 matrices
    for k in modes:
        M = np.load(OUT / f'dt512.{k}.npy') if only_long else np.zeros((len(qlen), N), np.float16)
        for f in sorted(bdir.glob('*.npz')):
            z = np.load(f); M[:, z['idx']] = z[k]
        np.save(OUT / f'{variant}.{k}.npy', M)
    log(variant, 'done')


def chunk():
    """variant a: chunk-level (t256 index vectors) max / mean / top-2 mean of CLS cosine, per doc."""
    dense = np.load(SEG / 'dense.npy', mmap_mode='r'); doc = np.fromfile(SEG / 'doc_id.bin', dtype=np.int64)
    docs = load_docs(); pos = {int(d): i for i, d in enumerate(docs.doc_id.astype(np.int64))}
    di = torch.from_numpy(np.array([pos[int(d)] for d in doc])).cuda()
    q = torch.from_numpy(np.load(OUT / 'q_cls.npy')).cuda().half(); N = len(docs); Qn = len(q)
    mx = torch.full((Qn, N), -9.0, device='cuda'); sm = torch.zeros((Qn, N), device='cuda'); cnt = torch.zeros(N, device='cuda')
    for s in range(0, len(dense), 20_000):
        sc = (q @ torch.from_numpy(np.ascontiguousarray(dense[s:s + 20_000])).cuda().half().T).float()  # [Q, c]
        g = di[s:s + 20_000]; ge = g.expand(Qn, -1)
        mx.scatter_reduce_(1, ge, sc, 'amax'); sm.index_add_(1, g, sc); cnt.index_add_(0, g, torch.ones_like(g, dtype=torch.float))
    # top-2 mean: second max = max over chunks excluding the argmax chunk
    first = mx.clone(); second = torch.full((Qn, N), -9.0, device='cuda')
    for s in range(0, len(dense), 20_000):
        sc = (q @ torch.from_numpy(np.ascontiguousarray(dense[s:s + 20_000])).cuda().half().T).float()
        g = di[s:s + 20_000]; ge = g.expand(Qn, -1)
        is_max = sc >= first.gather(1, ge)
        second.scatter_reduce_(1, ge, torch.where(is_max, torch.full_like(sc, -9.0), sc), 'amax')
    one = cnt == 1
    t2 = torch.where(one[None, :], first, (first + second.clamp(min=-1)) / 2)
    np.save(OUT / 'chunk.max.npy', mx.cpu().numpy().astype(np.float16))
    np.save(OUT / 'chunk.mean.npy', (sm / cnt.clamp(min=1)).cpu().numpy().astype(np.float16))
    np.save(OUT / 'chunk.top2.npy', t2.cpu().numpy().astype(np.float16))
    log('chunk done; docs without chunks', int((cnt == 0).sum()))


def metrics(M, y, dom=None):
    S = torch.from_numpy(M).cuda().float()
    best = S.max(0).values.cpu().numpy()
    mr = (S.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy()
    r = {'auc_best': roc_auc_score(y, best), 'auc_minrank': roc_auc_score(y, -mr),
         'kept_minrank1': float(y[mr == 1].mean()), 'rprec': float(y[np.argsort(mr, kind='stable')[:y.sum()]].mean())}
    if dom is not None:
        r['domain'] = {d: round(roc_auc_score(y[dom == d], -mr[dom == d]), 4) for d in pd.Series(dom).value_counts().index
                       if (dom == d).sum() >= 200 and 0 < y[dom == d].sum() < (dom == d).sum()}
    return r


def zrow(M):
    M = M.astype(np.float32); return (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-6)


def report():
    y = np.load(OUT / 'y.npy').astype(int); dom = load_docs().domain.values
    out = {}
    for f in sorted(OUT.glob('*.npy')):
        if f.stem.count('.') != 1:
            continue
        out[f.stem] = metrics(np.load(f), y, dom); log(f.stem, {k: round(v, 4) for k, v in out[f.stem].items() if k != 'domain'})
    # fusions on dt512 (raw-score sums as in BGEM3 compute_score, and per-query z-score sums)
    if all((OUT / f'dt512.{k}.npy').exists() for k in ('cls', 'sparse', 'colbert')):
        D, Sp, C = (np.load(OUT / f'dt512.{k}.npy').astype(np.float32) for k in ('cls', 'sparse', 'colbert'))
        fus = {'baai_0.4d_0.2s_0.4c': 0.4 * D + 0.2 * Sp + 0.4 * C, 'baai_dense+sparse_1:0.3': D + 0.3 * Sp}
        for w in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            fus[f'd+{w}s'] = D + w * Sp
        for w in (0.25, 0.5, 1.0, 2.0):
            fus[f'd+{w}c'] = D + w * C
        Dz, Sz, Cz = zrow(D), zrow(Sp), zrow(C)
        for w in (0.1, 0.2, 0.3):
            fus[f'z: d+{w}s'] = Dz + w * Sz
        fus['z: d+c'] = Dz + Cz
        for k, M in fus.items():
            out['fuse ' + k] = metrics(M, y); log('fuse', k, {a: round(b, 4) for a, b in out['fuse ' + k].items()})
    json.dump(out, open(OUT / 'report.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[2]
    if cmd == 'prep':
        prep()
    elif cmd == 'enc':
        for v in sys.argv[3].split(','):
            encode(v, tuple(sys.argv[4].split(',')) if len(sys.argv) > 4 else ('cls', 'mean', 'sparse', 'colbert'))
    elif cmd == 'chunk':
        chunk()
    elif cmd == 'report':
        report()
