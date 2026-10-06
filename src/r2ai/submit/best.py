"""Reproduce the best leaderboard submission from configs/submission-best.yaml (no new defaults in code).

    python scripts/run_best_retrieval.py   [--config ...] [--step candidates|rerank|all]   # GPU; rerun to resume
    python scripts/build_best_submission.py [--config ...] [--from-retrieval] [--out X.zip]  # CPU; checks the JSON SHA256

retrieve: step 1 run_retrieval_k100 --candidates-only into retrieval.candidates.out_dir (skipped when its outputs exist),
          step 2 the deep rerank into retrieval.rerank.out_dir (finished queries are restored from its checkpoint).
build:    make_submission with submission.args, runs_dir / doc_ranking from the config (the caches the uploaded file
          was built from) or, with --from-retrieval, from the two retrieval out dirs; then compares the JSON SHA256 with
          submission.expected_json_sha256 (exit 1 on mismatch, unless --no-check) and the ZIP size with max_zip_bytes.
          With postprocess.expand_clusters.enabled (off when the key is absent), make_submission writes <out stem>_base.zip
          and r2ai.dupes.postprocess.expand_submission appends the duplicate-cluster mates (H1-expand) into <out>;
          submission.base_json_sha256 (optional) is then checked on the builder JSON, expected_json_sha256 on the final one.
          With postprocess.rrf_chunk_docs.enabled (off when the key is absent), the previous output becomes <out stem>_expand.zip
          (or <out stem>_base.zip without expand) and r2ai.submit.rrf_chunks.rrf_chunk_submission rewrites only relevant_chunks
          (the k_chunk D50 primary docs with the highest RRF of reranker_cache / hybrid_ranks) into <out>;
          submission.expand_json_sha256 (optional) is then checked on the expand JSON.
"""
from __future__ import annotations

from r2ai.paths import ROOT, resolve_path

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

DEFAULT_CONFIG = ROOT / 'configs' / 'submission-best.yaml'
CAND_DOCS = 'vi_cand.docs.parquet'


def load_config(path: str | Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def _p(x) -> str:
    return str(resolve_path(x))


def retrieval_argv(cfg: dict, step: str) -> list[str]:
    r = cfg['retrieval']
    s = r[step]
    return ['--target', str(r['target']), '--queries', _p(cfg['queries']), '--out-dir', _p(s['out_dir'])] + [str(x) for x in s['args']]


def build_argv(cfg: dict, from_retrieval: bool = False, out: str | None = None) -> list[str]:
    s = cfg['submission']
    runs = cfg['retrieval']['rerank']['out_dir'] if from_retrieval else s['runs_dir']
    ranking = Path(cfg['retrieval']['candidates']['out_dir']) / CAND_DOCS if from_retrieval else s['doc_ranking']
    return [str(x) for x in s['args']] + ['--runs-dir', _p(runs), '--doc-ranking', _p(ranking), '--chunks-dir', _p(s['chunks_dir']),
                                          '--docs-dir', _p(s['docs_dir']), '--queries', _p(cfg['queries']),
                                          '--out', _p(out or s['out'])]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cmd_retrieve(a, cfg) -> int:
    from r2ai.retrieve import run_retrieval_k100
    steps = ['candidates', 'rerank'] if a.step == 'all' else [a.step]
    for step in steps:
        if step == 'candidates' and (resolve_path(cfg['retrieval']['candidates']['out_dir']) / CAND_DOCS).exists():
            print(f'candidates: {CAND_DOCS} exists, skipped', flush=True)
            continue
        argv = retrieval_argv(cfg, step)
        print(f'{step}: run_retrieval_k100 {" ".join(argv)}', flush=True)
        rc = run_retrieval_k100.main(argv)
        if rc:
            return rc
    return 0


def expand_config(cfg: dict) -> dict | None:
    """postprocess.expand_clusters of the config when enabled, else None (key absent = off)."""
    e = (cfg.get('postprocess') or {}).get('expand_clusters') or {}
    return e if e.get('enabled') else None


def rrf_config(cfg: dict) -> dict | None:
    """postprocess.rrf_chunk_docs of the config when enabled, else None (key absent = off)."""
    r = (cfg.get('postprocess') or {}).get('rrf_chunk_docs') or {}
    return r if r.get('enabled') else None


def cmd_build(a, cfg) -> int:
    from r2ai.submit import make_submission
    exp, rrf = expand_config(cfg), rrf_config(cfg)
    s = cfg['submission']
    final = Path(_p(a.out or s['out']))
    mid = lambda tag: final.with_name(f'{final.stem}_{tag}.zip')
    argv = build_argv(cfg, a.from_retrieval, str(mid('base')) if exp or rrf else a.out)
    print(f'make_submission {" ".join(argv)}', flush=True)
    t0 = time.time()
    rc = make_submission.main(argv)
    if rc:
        return rc
    zp = Path(argv[argv.index('--out') + 1])
    js = zp.with_suffix('.json')
    base = {}
    arg = lambda k: argv[argv.index(k) + 1]
    if exp or rrf:
        base = {'base_json': str(js), 'base_json_bytes': js.stat().st_size, 'base_json_sha256': sha256(js),
                'builder_elapsed_s': round(time.time() - t0, 1)}
    if exp:
        from r2ai.dupes.postprocess import expand_submission
        t1, out = time.time(), mid('expand') if rrf else final
        print(f'expand_clusters: {json.dumps(exp)}', flush=True)
        st = expand_submission(js, out, runs_dir=arg('--runs-dir'), doc_ranking=arg('--doc-ranking'), queries=arg('--queries'),
                               chunks_dir=arg('--chunks-dir'), clusters=_p(exp['clusters']), scope=exp.get('scope', 'content'),
                               k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')))
        base |= {'expand': {k: st[k] for k in ('queries_changed', 'ids_added_total', 'scope', 'clusters')},
                 'expand_elapsed_s': round(time.time() - t1, 1)}
        zp, js = out, out.with_suffix('.json')
        if rrf:
            base |= {'expand_json': str(js), 'expand_json_bytes': js.stat().st_size, 'expand_json_sha256': sha256(js)}
    if rrf:
        from r2ai.submit.rrf_chunks import rrf_chunk_submission
        t1 = time.time()
        print(f'rrf_chunk_docs: {json.dumps(rrf)}', flush=True)
        st = rrf_chunk_submission(js, final, runs_dir=arg('--runs-dir'), doc_ranking=arg('--doc-ranking'),
                                  hybrid_ranks=_p(rrf['hybrid_ranks']), rerank_run=_p(rrf['reranker_cache']), queries=arg('--queries'),
                                  docs_dir=arg('--docs-dir'), k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')),
                                  k_chunk=int(arg('--k-chunk')), rrf_k=int(rrf['k']), missing_rank=int(rrf['missing_rank']))
        base |= {'rrf': {k: st[k] for k in ('chunk_docs_changed_vs_base', 'new_chunk_docs_total', 'primary_docs_without_reranker_rank',
                                            'docs_missing_text', 'kept_docs_same_chunk_text')},
                 'rrf_elapsed_s': round(time.time() - t1, 1)}
        zp, js = final, final.with_suffix('.json')
    res = {'elapsed_s': round(time.time() - t0, 1), 'json': str(js), 'json_bytes': js.stat().st_size, 'json_sha256': sha256(js),
           'zip_bytes': zp.stat().st_size, 'zip_sha256': sha256(zp), 'expected_json_sha256': s['expected_json_sha256'],
           'max_zip_bytes': s['max_zip_bytes']} | base
    res['json_matches'] = res['json_sha256'] == s['expected_json_sha256']
    for key, on in (('base_json', exp or rrf), ('expand_json', exp and rrf)):
        if on and s.get(f'{key}_sha256'):
            res[f'{key}_matches'] = res[f'{key}_sha256'] == s[f'{key}_sha256']
            res['json_matches'] = res['json_matches'] and res[f'{key}_matches']
    res['zip_within_ceiling'] = res['zip_bytes'] <= s['max_zip_bytes']
    print(json.dumps(res, indent=1), flush=True)
    if not res['zip_within_ceiling']:
        return 1
    return 0 if res['json_matches'] or a.no_check else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('retrieve')
    r.add_argument('--config', default=str(DEFAULT_CONFIG))
    r.add_argument('--step', choices=('candidates', 'rerank', 'all'), default='all')
    b = sub.add_parser('build')
    b.add_argument('--config', default=str(DEFAULT_CONFIG))
    b.add_argument('--from-retrieval', action='store_true', help='use the retrieval out dirs of the config, not the D50 caches')
    b.add_argument('--out', default=None, help='output zip (default submission.out of the config)')
    b.add_argument('--no-check', action='store_true', help='do not fail when the JSON SHA256 differs from the expected one')
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    return cmd_retrieve(a, cfg) if a.cmd == 'retrieve' else cmd_build(a, cfg)


if __name__ == '__main__':
    sys.exit(main())
