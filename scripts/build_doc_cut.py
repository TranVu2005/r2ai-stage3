"""Build doc-cut variants of VS-append (S5): rrf_doc_set with an adaptive cut -> slug_append -> zip + validator.

    python scripts/build_doc_cut.py --variant A1-60 [--variant ...] --out-dir out/runs/doc-cut
Variants: GATE (tau -99 = no cut, must reproduce VS-append JSON 9c91c952...), A1-60 / A1-100 / A1-130 (tau calibrated by
scripts/doc_cut_calibrate.py), A2. Inputs = configs/submission-best.yaml values; base = the RRF-chunk JSON of the best replay.
"""
import argparse, json, sys, time
from pathlib import Path
from r2ai.submit.best import load_config, sha256
from r2ai.submit.rrf_docs import rrf_doc_submission
from r2ai.submit.slug_append import slug_append_submission
from r2ai.submit.doc_cut import make_cut

M = 'D:/GitHub/r2ai-stage3/'
BASE = M + 'out/runs/best-vsappend/gate-best/sub_rrf.json'
VARIANTS = {'GATE': ('a1', -99.0), 'A1-60': ('a1', 2.44), 'A1-100': ('a1', -0.62), 'A1-130': ('a1', -3.02), 'A2': ('a2', None)}

ap = argparse.ArgumentParser()
ap.add_argument('--variant', action='append', required=True, choices=sorted(VARIANTS))
ap.add_argument('--out-dir', default='out/runs/doc-cut')
a = ap.parse_args()
cfg = load_config('configs/submission-best.yaml')
s, rds, sla, exp = cfg['submission'], cfg['postprocess']['rrf_doc_set'], cfg['postprocess']['slug_append'], cfg['postprocess']['expand_clusters']
ab = lambda p: p if Path(p).is_absolute() else M + p
assert sha256(Path(BASE)) == s['rrf_json_sha256'], 'base RRF json differs from the config hash'
args = s['args']; arg = lambda k: args[args.index(k) + 1]
for v in a.variant:
    mode, tau = VARIANTS[v]
    d = Path(a.out_dir) / v; d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    st1 = rrf_doc_submission(BASE, d / 'mid.zip', chunks='base', runs_dir=ab(s['runs_dir']), doc_ranking=ab(s['doc_ranking']),
        hybrid_ranks=ab(rds['hybrid_ranks']), rerank_run=ab(rds['reranker_cache']), queries=cfg['queries'], chunks_dir=ab(s['chunks_dir']),
        docs_dir=ab(s['docs_dir']), clusters=ab(exp['clusters']), scope=exp.get('scope', 'content'), k_cache=int(arg('--k-doc')),
        k_total=int(arg('--k-doc-total')), k_docs=int(rds['k_docs']), k_chunk=int(arg('--k-chunk')), rrf_k=int(rds['k']),
        cut=make_cut(mode, tau=tau, floor=20, ceil=150))
    st2 = slug_append_submission(d / 'mid.json', d / f'sub_{v}.zip', pin=ab(sla['pin']), pin_sha256=sla['pin_sha256'], queries=cfg['queries'],
        k=int(sla.get('k', 20)), k1=float(sla['k1']), b=float(sla['b']), scan_depth=int(sla['scan_depth']))
    (d / 'mid.json').unlink(); (d / 'mid.zip').unlink()
    res = {'variant': v, 'mode': mode, 'tau': tau, 'doc_cut': st1.get('doc_cut'), 'expand_ids_added': st1['expand_ids_added'],
           'slug': {k: st2[k] for k in ('appended_total', 'appended_mean', 'queries_below_k')},
           'json_sha256': st2['json_sha256'], 'json_bytes': st2['json_bytes'], 'zip_sha256': st2['zip_sha256'], 'zip_bytes': st2['zip_bytes'],
           'zip_mib': round(st2['zip_bytes'] / 2**20, 3), 'elapsed_s': round(time.time() - t0, 1)}
    if v == 'GATE':
        res['matches_vs_append'] = res['json_sha256'] == s['expected_json_sha256']
    (d / 'build.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)
