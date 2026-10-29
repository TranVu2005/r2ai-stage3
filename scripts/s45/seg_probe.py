"""S45 segment-removal probes (Vu 29/10): drop one segment S of the R0 docs, and the chunks of those docs (chunk subset of relevant_docs
stays true), keep everything else byte for byte. Doc R depends on relevant_docs only, so the LB Doc R drop gives the true precision of S:
prec(S) ~ -dR * G / n_S, with G = gold per query = P * N / R of R0 and n_S = S docs dropped per query (micro ~ macro approximation).

Segments (``SEGMENTS``): ``novi`` = every vi doc (canonical lang; 120ask segment docs are zh, Pharmacity docs vi).

  seg_probe.py stats <segment>                          per-segment numbers before delivery (docs / query, carriers, p_cal, P4B, cosine)
  seg_probe.py zip <segment> <base.zip> <out.zip> [extra_dir ...]   the probe ZIP + S26 validator (``--max-chunks-per-doc 2 --no-p1``)
"""
import hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd  # noqa: E402
from r2ai.s38.build import write_bundle  # noqa: E402

DOCS = MAIN / 'out/runs/s45/pub-exact/r0-zip-docs.parquet'   # R0 ZIP docs + docsel columns + carrier flag (docs/s45.md 5g.1)
R0_LB = {'P': 0.2332, 'R': 0.6575, 'F': 0.4413}


def frame():
    m = pd.read_parquet(DOCS)
    c = pd.read_parquet(MAIN / 'data/canonical-20261018/docs.parquet', columns=['doc_id', 'lang'])
    links = pd.read_parquet(MAIN / 'data/raw/links_corpus_vi_en_zh.parquet')
    links = links[links.id.isin(set(m.doc_id))]
    host = dict(zip(links.id.astype(int), links.url.str.extract(r'^https?://([^/]+)')[0].str.lower().str.removeprefix('www.')))
    m['host'] = m.doc_id.map(host)
    m['lang'] = m.doc_id.map(dict(zip(c.doc_id.astype(int), c.lang)))
    m.loc[m.lang.isna() & m.host.str.endswith('120ask.com'), 'lang'] = 'zh'
    m.loc[m.lang.isna() & m.host.str.contains('pharmacity'), 'lang'] = 'vi'
    if m.lang.isna().any():
        raise SystemExit(f'{int(m.lang.isna().sum())} docs without a language')
    return m


SEGMENTS = {'novi': lambda m: m.lang == 'vi'}


def stats(seg):
    m = frame(); s = SEGMENTS[seg](m); nq = m.query_id.nunique()
    per_q = m[s].groupby('query_id').size().reindex(sorted(m.query_id.unique()), fill_value=0)
    def part(x):
        return {'docs_per_query': len(x) / nq, 'p_cal_mean': float(x.p_cal.mean()), 'p4b_mean': float(x.p4b.mean()),
                'cosine_rank_mean': float(x.cosine_rank.mean()), 'rrf_rank_mean': float(x['rank'].mean()),
                'carriers_per_query': float(x.carrier.sum() / nq), 'sum_p_cal_per_query': float(x.p_cal.sum() / nq)}
    G = R0_LB['P'] * len(m) / nq / R0_LB['R']
    res = {'segment': seg, 'queries': nq, 'docs': int(len(m)), 'segment_docs': int(s.sum()),
           'segment_per_query_quantiles': {k: float(v) for k, v in zip(['p10', 'p25', 'p50', 'p75', 'p90', 'max'],
                                                                      np.quantile(per_q, [.1, .25, .5, .75, .9, 1]))},
           'queries_with_segment': int((per_q > 0).sum()), 'queries_all_segment': int((per_q == m.groupby('query_id').size()).sum()),
           'segment_share_of_carriers': float(m[m.carrier & s].shape[0] / m.carrier.sum()),
           'segment_share_of_non_carriers': float(m[~m.carrier & s].shape[0] / (~m.carrier).sum()),
           'segment': part(m[s]), 'rest': part(m[~s]), 'by_lang': {l: part(g) for l, g in m.groupby('lang')},
           'gold_per_query_G': G,
           'predicted_dR_from_p_cal': -float(m[s].p_cal.sum() / nq) / G}
    print(json.dumps(res, indent=1)); return res


def rows_of(zp):
    with zipfile.ZipFile(zp) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            line = line.strip().rstrip(b',')
            if line.startswith(b'{'):
                yield json.loads(line)


def build(seg, base, out, extra):
    m = frame(); drop = set(zip(m.query_id[SEGMENTS[seg](m)].astype(int), m.doc_id[SEGMENTS[seg](m)].astype(int)))
    rows, nd, nc, empty, outside = [], 0, 0, 0, 0
    for r in rows_of(base):
        q = int(r['id'])
        docs = [d for d in r['relevant_docs'] if (q, d) not in drop]
        chunks = [c for c in r['relevant_chunks'] if (q, int(c['doc_id'])) not in drop]
        nd += len(r['relevant_docs']) - len(docs); nc += len(r['relevant_chunks']) - len(chunks); empty += not docs
        outside += sum(int(c['doc_id']) not in set(docs) for c in chunks)
        rows.append({**r, 'relevant_docs': docs, 'relevant_chunks': chunks})
    if outside:
        raise SystemExit(f'{outside} chunks outside relevant_docs: not written')
    res = write_bundle(rows, out, out.with_suffix('.json').name)
    n = len(rows)
    rep = {'segment': seg, 'base': str(base), 'base_sha256': hashlib.sha256(base.read_bytes()).hexdigest(), 'zip': str(out),
           'zip_bytes': out.stat().st_size, **res, 'docs': sum(len(r['relevant_docs']) for r in rows),
           'chunks': sum(len(r['relevant_chunks']) for r in rows), 'docs_dropped': nd, 'docs_dropped_per_query': nd / n,
           'chunks_dropped': nc, 'queries_empty_docs': empty, 'chunk_doc_not_in_docs': outside}
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/s26/validate_pa3.py'), '--max-chunks-per-doc', '2', '--no-p1', str(out), *extra]
    v = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    out.with_suffix('.validate.json').write_text(v.stdout, encoding='utf-8')
    vj = json.loads(v.stdout)
    rep['validator'] = {'ok': vj['ok'], 'zip_sha256': vj['zip_sha256'], 'errors': {k: x for k, x in vj['errors'].items() if x},
                        'chunk_check': vj['chunk_check']}
    out.with_suffix('.seg.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'stats':
        stats(sys.argv[2])
    else:
        build(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5:])
