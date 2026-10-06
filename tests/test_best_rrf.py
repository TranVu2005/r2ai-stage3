"""configs: rrf = h1expand + postprocess.rrf_chunk_docs (RRF choice of the full-chunk docs); best.build replays the
r2ai.submit.rrf_chunks CLI byte for byte on a fixture, and with the key off/absent gives the H1-expand bytes unchanged.
The current best (rrf + RRF doc set, RD150) is covered by test_best_rd150.py."""
import copy

import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.paths import ROOT
from r2ai.submit import best
from r2ai.submit import rrf_chunks as R
from tests.test_best_expand import H1EXPAND, IDS, K_CHUNK, K_TOTAL, _cfg, _h1, env  # noqa: F401  (fixture)

RRF = ROOT / 'configs' / 'submission-rrf.yaml'


def _rr(env):
    order = [60, 61, 62] + [d for d in IDS[::-1] if d not in (60, 61, 62)]
    rr = [(q, k, d) for q in (1, 2) for k, d in enumerate(order, 1)]
    p = env / 'rr.parquet'
    pq.write_table(pa.table({'query_id': [r[0] for r in rr], 'rank': [r[1] for r in rr], 'doc_id': [r[2] for r in rr]}), p)
    return p


def _rrf_cli(env, base):
    out = env / 'rrf_cli' / 'sub.zip'
    assert R.main(['build', '--runs-dir', str(env / 'runs'), '--doc-ranking', str(env / 'cand.parquet'), '--rerank-run', str(_rr(env)),
                   '--queries', str(env / 'query.parquet'), '--k-total', str(K_TOTAL), '--k-chunk', str(K_CHUNK), '--base', str(base),
                   '--out', str(out), '--docs-dir', str(env / 'docs_vi')]) == 0
    return out.with_suffix('.json')


def _rrf_cfg(env, enabled, sha_id, sha_ex, sha):
    cfg = _cfg(env, True, base_sha=sha_id, sha=sha)
    cfg['submission']['expand_json_sha256'] = sha_ex
    cfg['postprocess']['rrf_chunk_docs'] = {'enabled': enabled, 'k': 60, 'reranker_cache': str(_rr(env)),
                                            'hybrid_ranks': str(env / 'cand.parquet'), 'missing_rank': 201}
    return cfg


def test_config_rrf_is_h1expand_plus_rrf_chunk_docs():
    b, h = best.load_config(RRF), best.load_config(H1EXPAND)
    assert best.rrf_config(b) == {'enabled': True, 'k': 60, 'reranker_cache': 'out/runs/rerank200/full/vi_k100.parquet',
                                  'hybrid_ranks': 'out/runs/ab-2026-10-05b/cand/vi_cand.docs.parquet', 'missing_rank': 201}
    assert best.rrf_config(h) is None
    assert b['retrieval'] == h['retrieval'] and b['queries'] == h['queries']
    assert b['postprocess']['expand_clusters'] == h['postprocess']['expand_clusters']
    keys = ('runs_dir', 'doc_ranking', 'chunks_dir', 'docs_dir', 'args', 'max_zip_bytes', 'base_json_sha256')
    assert {k: b['submission'][k] for k in keys} == {k: h['submission'][k] for k in keys}
    assert b['submission']['expand_json_sha256'] == h['submission']['expected_json_sha256'] == \
        '6c27d3f327fdcc5dac248cdaf80cf1be941fbff22666b25dbd07f32903f01106'
    assert b['submission']['expected_json_sha256'] == 'ff035b8ba7eea0a177932850d3a9cc9f1ffbcbe8297b8a8740bae23bd949d1c3'
    off = copy.deepcopy(b)
    off['postprocess']['rrf_chunk_docs']['enabled'] = False
    assert best.rrf_config(off) is None


def test_best_build_rrf_replays_cli_and_off_keeps_expand(env, monkeypatch):
    sha_id, _ = _h1(env, 'identity')
    sha_ex, js_ex = _h1(env, 'expand')
    want = _rrf_cli(env, js_ex)
    sha_rrf = best.sha256(want)
    assert want.read_bytes() != js_ex.read_bytes()
    cfg = _rrf_cfg(env, True, sha_id, sha_ex, sha_rrf)
    monkeypatch.setattr(best, 'load_config', lambda p, c=cfg: c)
    assert best.main(['build', '--out', str(env / 'on/sub.zip')]) == 0
    assert (env / 'on/sub.json').read_bytes() == want.read_bytes()
    assert (env / 'on/sub_base.json').exists() and (env / 'on/sub_expand.json').read_bytes() == js_ex.read_bytes()
    for off in (False, None):                                               # enabled false / key absent -> H1-expand bytes
        c = _rrf_cfg(env, bool(off), sha_id, sha_ex, sha_ex)
        if off is None:
            del c['postprocess']['rrf_chunk_docs']
        monkeypatch.setattr(best, 'load_config', lambda p, c=c: c)
        assert best.main(['build', '--out', str(env / f'off_{off}/sub.zip')]) == 0
        assert (env / f'off_{off}/sub.json').read_bytes() == js_ex.read_bytes()
        assert not (env / f'off_{off}/sub_expand.json').exists()


def test_best_build_rrf_fails_on_wrong_intermediate_hash(env, monkeypatch):
    sha_id, _ = _h1(env, 'identity')
    sha_ex, js_ex = _h1(env, 'expand')
    sha_rrf = best.sha256(_rrf_cli(env, js_ex))
    for e, f in (('x', sha_rrf), (sha_ex, 'x')):
        cfg = _rrf_cfg(env, True, sha_id, e, f)
        monkeypatch.setattr(best, 'load_config', lambda p, c=cfg: c)
        assert best.main(['build', '--out', str(env / 'bad/sub.zip')]) == 1
        assert best.main(['build', '--out', str(env / 'bad/sub.zip'), '--no-check']) == 0
