"""configs: best = rrf + postprocess.rrf_doc_set (RD150: RRF choice of the doc set over the 200-doc pool); best.build replays
the r2ai.submit.rrf_docs CLI (--chunks base) byte for byte on a fixture, and with the key off/absent gives the RRF bytes."""
import copy

from r2ai.paths import ROOT
from r2ai.submit import best
from r2ai.submit import rrf_docs as D
from tests.test_best_expand import K_CHUNK, K_TOTAL, _h1, env  # noqa: F401  (fixture)
from tests.test_best_rrf import RRF, _rr, _rrf_cfg, _rrf_cli

K_DOCS = 5


def _rd_cli(env, base):
    out = env / 'rd_cli' / 'sub.zip'
    assert D.main(['build', '--base', str(base), '--out', str(out), '--chunks', 'base', '--runs-dir', str(env / 'runs'),
                   '--doc-ranking', str(env / 'cand.parquet'), '--rerank-run', str(_rr(env)), '--queries', str(env / 'query.parquet'),
                   '--chunks-dir', str(env / 'chunks'), '--docs-dir', str(env / 'docs_vi'), '--clusters', str(env / 'H1/clusters.parquet'),
                   '--k-cache', '100', '--k-total', str(K_TOTAL), '--k-docs', str(K_DOCS), '--k-chunk', str(K_CHUNK)]) == 0
    return out.with_suffix('.json')


def _cfg(env, enabled, sha_rrf, sha):
    sha_id, _ = _h1(env, 'identity')
    sha_ex, _ = _h1(env, 'expand')
    cfg = _rrf_cfg(env, True, sha_id, sha_ex, sha)
    cfg['submission']['rrf_json_sha256'] = sha_rrf
    cfg['postprocess']['rrf_doc_set'] = {'enabled': enabled, 'k': 60, 'k_docs': K_DOCS, 'reranker_cache': str(_rr(env)),
                                         'hybrid_ranks': str(env / 'cand.parquet')}
    return cfg


def test_config_best_is_rrf_plus_rrf_doc_set():
    b, r = best.load_config(best.DEFAULT_CONFIG), best.load_config(RRF)
    assert best.rrf_doc_set_config(b) == {'enabled': True, 'k': 60, 'k_docs': 150,
                                          'reranker_cache': 'out/runs/rerank200/full/vi_k100.parquet',
                                          'hybrid_ranks': 'out/runs/ab-2026-10-05b/cand/vi_cand.docs.parquet'}
    assert best.rrf_doc_set_config(r) is None
    assert b['retrieval'] == r['retrieval'] and b['queries'] == r['queries']
    assert {k: b['postprocess'][k] for k in ('expand_clusters', 'rrf_chunk_docs')} == r['postprocess']
    keys = ('runs_dir', 'doc_ranking', 'chunks_dir', 'docs_dir', 'args', 'max_zip_bytes', 'base_json_sha256', 'expand_json_sha256')
    assert {k: b['submission'][k] for k in keys} == {k: r['submission'][k] for k in keys}
    assert b['submission']['rrf_json_sha256'] == r['submission']['expected_json_sha256'] == \
        'ff035b8ba7eea0a177932850d3a9cc9f1ffbcbe8297b8a8740bae23bd949d1c3'
    assert b['submission']['expected_json_sha256'] == '136c259145c6fa134bcd5d4a877575088e28346ef43636afc87d2e283508e950'
    off = copy.deepcopy(b)
    off['postprocess']['rrf_doc_set']['enabled'] = False
    assert best.rrf_doc_set_config(off) is None


def test_best_build_rd150_replays_cli_and_off_keeps_rrf(env, monkeypatch):
    _, js_ex = _h1(env, 'expand')
    rrf = _rrf_cli(env, js_ex)
    want = _rd_cli(env, rrf)
    sha_rrf, sha = best.sha256(rrf), best.sha256(want)
    assert want.read_bytes() != rrf.read_bytes()
    cfg = _cfg(env, True, sha_rrf, sha)
    monkeypatch.setattr(best, 'load_config', lambda p, c=cfg: c)
    assert best.main(['build', '--out', str(env / 'on/sub.zip')]) == 0
    assert (env / 'on/sub.json').read_bytes() == want.read_bytes()
    assert (env / 'on/sub_rrf.json').read_bytes() == rrf.read_bytes() and (env / 'on/sub_expand.json').exists()
    for off in (False, None):                                               # enabled false / key absent -> RRF bytes
        c = _cfg(env, bool(off), sha_rrf, sha_rrf)
        if off is None:
            del c['postprocess']['rrf_doc_set']
        monkeypatch.setattr(best, 'load_config', lambda p, c=c: c)
        assert best.main(['build', '--out', str(env / f'off_{off}/sub.zip')]) == 0
        assert (env / f'off_{off}/sub.json').read_bytes() == rrf.read_bytes()
        assert not (env / f'off_{off}/sub_rrf.json').exists()


def test_best_build_rd150_fails_on_wrong_rrf_hash(env, monkeypatch):
    _, js_ex = _h1(env, 'expand')
    rrf = _rrf_cli(env, js_ex)
    sha = best.sha256(_rd_cli(env, rrf))
    for r, f in (('x', sha), (best.sha256(rrf), 'x')):
        cfg = _cfg(env, True, r, f)
        monkeypatch.setattr(best, 'load_config', lambda p, c=cfg: c)
        assert best.main(['build', '--out', str(env / 'bad/sub.zip')]) == 1
        assert best.main(['build', '--out', str(env / 'bad/sub.zip'), '--no-check']) == 0
