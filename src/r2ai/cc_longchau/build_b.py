"""VS-cc-B / VS-cc-B-cut115: run the unchanged submission chain on the merged (vi + CC) retrieval outputs of merge_b.

builder (make_submission) -> H1 expand -> RRF chunk docs -> RRF doc set [+ S5 doc cut] -> slug ids (URLs without CC text only).
Nothing here changes a default: the chain functions are the ones best.py uses, only the input paths point to
out/runs/cc-longchau/B/{deep,r200,cand,chunks,docs}. The cluster file is the vi one (CC docs are singletons).
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq

from r2ai.dupes.postprocess import expand_submission
from r2ai.paths import CORPUS_FILE, QUERY_FILE
from r2ai.submit import doc_cut as C
from r2ai.submit import make_submission
from r2ai.submit.best import build_argv, load_config, sha256
from r2ai.submit.rrf_chunks import rrf_chunk_submission
from r2ai.submit.rrf_docs import rrf_doc_submission
from r2ai.submit.slug_append import slug_append_submission
from r2ai.zh_sample.append import _rows

from .common import Roots, atomic_json
from .cut_cc import calibrate
from .merge_b import CONFIG, _p

ZIP_CEILING = 104_857_600
KNOWN_ACCEPTED_ZIP = 101_870_962


def b_config(roots: Roots):
    cfg = copy.deepcopy(load_config(_p(CONFIG)))
    B = roots.run / 'B'
    s, pp = cfg['submission'], cfg['postprocess']
    s['runs_dir'] = str(B / 'deep')
    s['doc_ranking'] = str(B / 'cand' / 'vi_cand.docs.parquet')
    s['chunks_dir'] = str(B / 'chunks')
    s['docs_dir'] = str(B / 'docs')
    for k in ('rrf_chunk_docs', 'rrf_doc_set'):
        pp[k]['reranker_cache'] = str(B / 'r200' / 'vi_k100.parquet')
        pp[k]['hybrid_ranks'] = str(B / 'cand' / 'vi_cand.docs.parquet')
    return cfg


def cc_doc_ids(roots: Roots) -> set:
    from .build import active_segments, zroots
    out = set()
    for sg in active_segments(zroots(roots)):
        t = pq.read_table(roots.chunks / sg['name'] / 'docs.parquet', columns=['doc_ids_group'])
        for g in t.column('doc_ids_group').to_pylist():
            out.update(g)
    return out


def stage(roots: Roots, cfg, k_chunk=None):
    """builder -> expand -> rrf chunks; returns the RRF-chunk JSON path (shared by B and B-cut115).

    k_chunk (default: the config's 50) = docs that get a chunk; a smaller value only re-runs the RRF chunk step."""
    d = roots.guard(roots.run / 'B' / 'stage')
    d.mkdir(parents=True, exist_ok=True)
    s, pp = cfg['submission'], cfg['postprocess']
    argv0 = build_argv(cfg, False, str(d / 'sub_base.zip'))
    kc = k_chunk or int(argv0[argv0.index('--k-chunk') + 1])
    rrf_json = d / ('sub_rrf.json' if k_chunk is None else f'sub_rrf_k{kc}.json')
    if rrf_json.exists():
        return rrf_json
    exp, rrf = pp['expand_clusters'], pp['rrf_chunk_docs']
    argv = argv0
    arg = lambda k: argv[argv.index(k) + 1]
    if not (d / 'sub_expand.json').exists():
        if make_submission.main(argv):
            raise RuntimeError('make_submission failed')
        expand_submission(d / 'sub_base.json', d / 'sub_expand.zip', runs_dir=arg('--runs-dir'), doc_ranking=arg('--doc-ranking'),
                          queries=arg('--queries'), chunks_dir=arg('--chunks-dir'), clusters=_p(exp['clusters']),
                          scope=exp.get('scope', 'content'), k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')))
    zip_out = rrf_json.with_suffix('.zip')
    rrf_chunk_submission(d / 'sub_expand.json', zip_out, runs_dir=arg('--runs-dir'), doc_ranking=arg('--doc-ranking'),
                         hybrid_ranks=rrf['hybrid_ranks'], rerank_run=rrf['reranker_cache'], queries=arg('--queries'),
                         docs_dir=arg('--docs-dir'), k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')),
                         k_chunk=kc, rrf_k=int(rrf['k']), missing_rank=int(rrf['missing_rank']))
    return rrf_json


def validate(zip_path, docs_dir, log):
    from r2ai.submit.validate_submission import main as run
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap):
        code = run([str(zip_path), '--queries', str(QUERY_FILE), '--corpus', str(CORPUS_FILE), '--docs-dir', str(docs_dir), '--max-zip-mib', '100'])
    Path(log).write_text(cap.getvalue(), encoding='utf-8')
    res = json.loads(cap.getvalue().strip())
    if code != 0 or res.get('errors') != 0 or res.get('ok') is not True or res.get('rows') != 1200:
        raise ValueError(f'validator failed ({code}): {cap.getvalue()[:300]}')
    return res


def build(roots: Roots, target_mean=None, tag='', k_chunk=None):
    t0 = time.time()
    cfg = b_config(roots)
    s, pp = cfg['submission'], cfg['postprocess']
    rds, sla, exp = pp['rrf_doc_set'], pp['slug_append'], pp['expand_clusters']
    args = s['args']
    arg = lambda k: args[args.index(k) + 1]
    base = stage(roots, cfg, k_chunk)
    kc = k_chunk or int(arg('--k-chunk'))
    name = ('VS-cc-B' if target_mean is None else f'VS-cc-B-cut{int(target_mean)}') + (f'-k{kc}' if k_chunk else '') + tag
    out = roots.guard(roots.run / name)
    out.mkdir(parents=True, exist_ok=True)
    mid = out / '_mid'
    mid.mkdir(exist_ok=True)
    cut, tau, mean, cal = None, None, None, None
    if target_mean is not None:
        tau, mean, cal = calibrate(cfg, target_mean, base=base)
        cut = C.make_cut('a1', tau=tau, floor=20, ceil=150, protect_chunks=True)
    st1 = rrf_doc_submission(base, mid / 'mid.zip', chunks='base', runs_dir=_p(s['runs_dir']), doc_ranking=_p(s['doc_ranking']),
                             hybrid_ranks=_p(rds['hybrid_ranks']), rerank_run=_p(rds['reranker_cache']), queries=_p(cfg['queries']),
                             chunks_dir=_p(s['chunks_dir']), docs_dir=_p(s['docs_dir']), clusters=_p(exp['clusters']),
                             scope=exp.get('scope', 'content'), k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')),
                             k_docs=int(rds['k_docs']), k_chunk=kc, rrf_k=int(rds['k']), cut=cut)
    cc = cc_doc_ids(roots)
    stem = name.replace('-', '_')
    final_zip = out / f'{stem}.zip'
    st2 = slug_append_submission(mid / 'mid.json', final_zip, pin=_p(sla['pin']), pin_sha256=sla['pin_sha256'], queries=_p(cfg['queries']),
                                 k=int(sla.get('k', 20)), k1=float(sla['k1']), b=float(sla['b']), scan_depth=int(sla['scan_depth']),
                                 exclude_ids=cc)
    final_json = final_zip.with_suffix('.json')
    val = validate(final_zip, s['docs_dir'], out / 'validator.log')
    with open(final_json, encoding='utf-8', newline='') as f:
        rows = [r for r, _ in _rows(f.read())]
    n_docs = sorted(len(r['relevant_docs']) for r in rows)
    pct = lambda p: n_docs[min(len(n_docs) - 1, int(p * len(n_docs)))]
    cc_docs = [sum(d in cc for d in r['relevant_docs']) for r in rows]
    cc_chunks = [sum(c['doc_id'] in cc for c in r['relevant_chunks']) for r in rows]
    stats = {'variant': name, 'diagnostic_only': True, 'LB': 'chua do', 'k_chunk': kc, 'tau': tau, 'target_mean': target_mean, 'mean_primary_after_cut': mean,
             'calibration': cal, 'doc_cut': st1.get('doc_cut'), 'validator': val, 'json_bytes': final_json.stat().st_size,
             'json_sha256': sha256(final_json), 'zip_bytes': final_zip.stat().st_size, 'zip_mib': round(final_zip.stat().st_size / 2 ** 20, 3),
             'zip_sha256': sha256(final_zip), 'zip_vs_largest_accepted_VS40_bytes': final_zip.stat().st_size - KNOWN_ACCEPTED_ZIP,
             'relevant_docs_per_query': {'mean': statistics.fmean(n_docs), 'p10': pct(.1), 'p50': pct(.5), 'p90': pct(.9), 'min': n_docs[0], 'max': n_docs[-1]},
             'cc_docs_in_relevant_docs': {'total': sum(cc_docs), 'mean': statistics.fmean(cc_docs), 'queries_with_any': sum(x > 0 for x in cc_docs)},
             'cc_chunks_in_relevant_chunks': {'total': sum(cc_chunks), 'mean': statistics.fmean(cc_chunks), 'queries_with_any': sum(x > 0 for x in cc_chunks)},
             'chunks_per_query': statistics.fmean(len(r['relevant_chunks']) for r in rows),
             'slug': {k: st2[k] for k in ('appended_total', 'appended_mean', 'queries_below_k')},
             'seconds': round(time.time() - t0, 1)}
    atomic_json(roots, out / 'stats.json', stats)
    for f in mid.iterdir():
        f.unlink()
    mid.rmdir()
    return stats
