"""configs: h1expand = D50 + H1-expand post-processing (the best before RRF), d50 = plain D50; best.build replays both byte
for byte on a fixture. The current best (h1expand + RRF chunk docs) is covered by test_best_rrf.py."""
import copy
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.eval.scorer import whitespace_tokenizer
from r2ai.paths import ROOT
import r2ai.submit.make_submission as ms
from r2ai.dupes import variants as V
from r2ai.submit import best

IDS = list(range(10, 120))
K_TOTAL, K_CHUNK = 104, 3
H1EXPAND = ROOT / 'configs' / 'submission-h1expand.yaml'
GROUPS = {d: [d] + ([d + 1000] if d % 3 == 0 else []) for d in IDS}
CLUSTERS = {0: [10, 12], 1: [11, 50], 2: [101, 110], 3: [20, 119], 4: [14, 60]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, 'Tokenizer', whitespace_tokenizer)
    for p in ('runs', 'chunks', 'docs_vi', 'H1'):
        (tmp_path / p).mkdir(parents=True)
    pq.write_table(pa.table({'id': [1, 2], 'query': ['a', 'b']}), tmp_path / 'query.parquet')
    pq.write_table(pa.table({'doc_ids': [GROUPS[d] for d in IDS], 'title': ['t'] * len(IDS), 'question': [None] * len(IDS),
                             'description': [None] * len(IDS), 'answer': [None] * len(IDS),
                             'body': [' '.join(f'd{d}w{w}' for w in range(60)) for d in IDS], 'status': ['ok'] * len(IDS)}),
                   tmp_path / 'docs_vi/p0.parquet')
    pq.write_table(pa.table({'doc_id': IDS, 'doc_ids_group': [GROUPS[d] for d in IDS]}), tmp_path / 'chunks/docs.parquet')
    order = {1: IDS, 2: IDS[::-1]}
    run = [(q, k, d) for q in (1, 2) for k, d in enumerate(order[q], 1)]
    pq.write_table(pa.table({'query_id': [r[0] for r in run], 'rank': pa.array([r[1] for r in run], pa.int32()),
                             'doc_id': [r[2] for r in run], 'score': pa.array([0.0] * len(run), pa.float32()),
                             'tier': pa.array([1] * len(run), pa.int8())}), tmp_path / 'runs/vi_k100.parquet')
    pq.write_table(pa.table({'query_id': [r[0] for r in run], 'rank': [r[1] for r in run], 'doc_id': [r[2] for r in run],
                             'score': pa.array([0.0] * len(run), pa.float32())}), tmp_path / 'cand.parquet')
    rows = [(d, c) for c, m in CLUSTERS.items() for d in m] + [(70, 9), (71, 9)]
    pq.write_table(pa.table({'doc_id': [r[0] for r in rows], 'cluster_id': [r[1] for r in rows],
                             'boilerplate': [False] * (len(rows) - 2) + [True, True]}), tmp_path / 'H1/clusters.parquet')
    return tmp_path


def _cfg(env, expand: bool, base_sha='', sha=''):
    cfg = copy.deepcopy(best.load_config(H1EXPAND))
    cfg['queries'] = str(env / 'query.parquet')
    s = cfg['submission']
    s |= {'runs_dir': str(env / 'runs'), 'doc_ranking': str(env / 'cand.parquet'), 'chunks_dir': str(env / 'chunks'),
          'docs_dir': str(env / 'docs_vi'), 'out': str(env / 'best/sub.zip'),
          'args': ['--k-doc', '100', '--k-doc-total', str(K_TOTAL), '--k-chunk', str(K_CHUNK), '--chunk-mode', 'full', '--dedupe-scope', 'doc'],
          'base_json_sha256': base_sha, 'expected_json_sha256': sha}
    cfg['postprocess']['expand_clusters'] |= {'enabled': expand, 'clusters': str(env / 'H1/clusters.parquet')}
    return cfg


def _h1(env, mode):
    out = env / f'h1_{mode}' / 'sub.zip'
    assert V.main(['build', '--mode', mode, '--out', str(out), '--out-dir', str(env / 'H1'), '--runs-dir', str(env / 'runs'),
                   '--doc-ranking', str(env / 'cand.parquet'), '--queries', str(env / 'query.parquet'), '--chunks-dir', str(env / 'chunks'),
                   '--docs-dir', str(env / 'docs_vi'), '--k-total', str(K_TOTAL), '--k-chunk', str(K_CHUNK)]) == 0
    return json.loads(out.with_suffix('.stats.json').read_text(encoding='utf-8'))['json_sha256'], out.with_suffix('.json')


def test_configs_h1expand_enables_expand_d50_does_not():
    b = best.load_config(H1EXPAND)
    d = best.load_config(ROOT / 'configs' / 'submission-d50.yaml')
    e = best.expand_config(b)
    assert e and e['scope'] == 'content' and e['clusters'] == 'out/runs/H1/clusters.parquet'
    assert best.expand_config(d) is None
    assert b['retrieval'] == d['retrieval']
    assert {k: b['submission'][k] for k in ('runs_dir', 'doc_ranking', 'chunks_dir', 'docs_dir', 'args', 'max_zip_bytes')} == \
        {k: d['submission'][k] for k in ('runs_dir', 'doc_ranking', 'chunks_dir', 'docs_dir', 'args', 'max_zip_bytes')}
    assert d['submission']['expected_json_sha256'] == b['submission']['base_json_sha256'] == \
        '2f10b78ac17f4eafdd5fe1b0b207260b062f17b29f4fcc3ee1a4b364b3c36544'
    assert b['submission']['expected_json_sha256'] == '6c27d3f327fdcc5dac248cdaf80cf1be941fbff22666b25dbd07f32903f01106'
    off = copy.deepcopy(b)
    off['postprocess']['expand_clusters']['enabled'] = False
    assert best.expand_config(off) is None


def test_best_build_replays_d50_and_expand_byte_identically(env, monkeypatch):
    sha_id, js_id = _h1(env, 'identity')
    sha_ex, js_ex = _h1(env, 'expand')
    assert sha_id != sha_ex
    for expand, want, want_js in ((False, sha_id, js_id), (True, sha_ex, js_ex)):
        cfg = _cfg(env, expand, base_sha=sha_id, sha=want)
        monkeypatch.setattr(best, 'load_config', lambda p, c=cfg: c)
        assert best.main(['build', '--out', str(env / f'best_{expand}/sub.zip')]) == 0
        js = env / f'best_{expand}/sub.json'
        assert js.read_bytes() == want_js.read_bytes()
        assert (env / f'best_{expand}/sub_base.json').exists() == expand
    d50 = json.loads(js_id.read_text(encoding='utf-8'))
    ex = json.loads((env / 'best_True/sub.json').read_text(encoding='utf-8'))
    assert [r['relevant_chunks'] for r in ex] == [r['relevant_chunks'] for r in d50]
    added = [e['relevant_docs'][len(b['relevant_docs']):] for b, e in zip(d50, ex)]
    assert added == [[119], [14, 11]]                                        # boilerplate cluster 9 (docs 70, 71) ignored


def test_best_build_fails_on_wrong_hashes_and_doc_list_mismatch(env, monkeypatch):
    sha_id, _ = _h1(env, 'identity')
    sha_ex, _ = _h1(env, 'expand')
    for base_sha, sha in ((sha_id, 'x'), ('x', sha_ex)):
        cfg = _cfg(env, True, base_sha=base_sha, sha=sha)
        monkeypatch.setattr(best, 'load_config', lambda p, c=cfg: c)
        assert best.main(['build']) == 1
        assert best.main(['build', '--no-check']) == 0
    from r2ai.dupes.postprocess import expand_submission
    _, js_id = _h1(env, 'identity')
    with pytest.raises(ValueError, match='differ from the doc list'):                 # base built with another K
        expand_submission(js_id, env / 'bad/sub.zip', runs_dir=str(env / 'runs'), doc_ranking=str(env / 'cand.parquet'),
                          queries=str(env / 'query.parquet'), chunks_dir=str(env / 'chunks'),
                          clusters=str(env / 'H1/clusters.parquet'), k_total=K_TOTAL - 1)
