"""configs/submission-d50.yaml (D50, formerly submission-best.yaml) -> the exact CLI args of the uploaded D50 run; build fails
on a JSON SHA256 mismatch. The current best config (D50 + expand) is covered by test_best_expand.py."""
import hashlib
import zipfile

import pytest

from r2ai.paths import ROOT, resolve_path
from r2ai.submit import best


@pytest.fixture
def cfg():
    return best.load_config(ROOT / 'configs' / 'submission-d50.yaml')


def _flags(argv):
    return {argv[i]: argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
            for i in range(len(argv)) if argv[i].startswith('--')}


def test_config_matches_d50_commands(cfg):
    r = _flags(best.retrieval_argv(cfg, 'rerank'))
    assert r == {'--target': '256', '--queries': str(resolve_path(cfg['queries'])), '--out-dir': str(resolve_path('out/runs/best/rerank')),
                 '--candidates': 'exact', '--tier1-docs': '100', '--tier2-docs': '50', '--chunk-score-docs': '0', '--pair-scores': True}
    assert _flags(best.retrieval_argv(cfg, 'candidates'))['--candidates-only'] is True
    b = _flags(best.build_argv(cfg))
    assert {k: b[k] for k in ('--k-doc', '--k-doc-total', '--k-chunk', '--chunk-mode', '--dedupe-scope')} == \
        {'--k-doc': '100', '--k-doc-total': '150', '--k-chunk': '50', '--chunk-mode': 'full', '--dedupe-scope': 'doc'}
    assert b['--runs-dir'] == str(resolve_path('out/runs/deep-rerank-2026-10-05/full'))
    fr = _flags(best.build_argv(cfg, from_retrieval=True))
    assert fr['--runs-dir'] == str(resolve_path('out/runs/best/rerank'))
    assert fr['--doc-ranking'] == str(resolve_path('out/runs/best/cand/vi_cand.docs.parquet'))
    assert cfg['submission']['max_zip_bytes'] == 104857600


@pytest.mark.parametrize('payload,rc', [(b'[\n]\n', 0), (b'other', 1)])
def test_build_checks_json_sha(cfg, tmp_path, monkeypatch, payload, rc):
    from r2ai.submit import make_submission
    cfg['submission']['expected_json_sha256'] = hashlib.sha256(b'[\n]\n').hexdigest()
    monkeypatch.setattr(best, 'load_config', lambda p: cfg)

    def fake_main(argv):
        zp = argv[argv.index('--out') + 1]
        js = zp[:-4] + '.json'
        open(js, 'wb').write(payload)
        with zipfile.ZipFile(zp, 'w') as z:
            z.write(js, 'x.json')
        return 0
    monkeypatch.setattr(make_submission, 'main', fake_main)
    assert best.main(['build', '--out', str(tmp_path / 's.zip')]) == rc
    assert best.main(['build', '--out', str(tmp_path / 's.zip'), '--no-check']) == 0
