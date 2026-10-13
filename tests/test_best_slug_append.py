"""configs: best = rd150 + postprocess.slug_append (VS-append: top-k vi doc ids by BM25 over URL slugs of non-crawlable vi
URLs, appended to relevant_docs, chunks untouched); rd150 = the former best. The slug candidate set and the BM25 parameters
are frozen in a pin file (SHA256 in the config), so replay does not change when state/crawl.db does."""
import copy
import json
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from r2ai.paths import ROOT
from r2ai.selective.bm25 import Slug
from r2ai.submit import best
from r2ai.submit import slug_append as S
from tests.test_best_expand import K_CHUNK, K_TOTAL, _h1, env  # noqa: F401  (fixture)
from tests.test_best_rd150 import _cfg as _rd_cfg, _rd_cli
from tests.test_best_rrf import _rrf_cli

RD150_SHA = '136c259145c6fa134bcd5d4a877575088e28346ef43636afc87d2e283508e950'
VS_SHA = '9c91c952863c4e33b992053d665d547ba1f483cd0be487047386ba09af43e2e0'

CRAWL = [  # url, doc_ids, domain, status   (crawl.db `urls` rows, natural order)
    ('https://a.vn/thuoc-ho-tre-em-1', [901], 'a.vn', 'ok'),
    ('https://a.vn/benh-tieu-duong-2', [902], 'a.vn', 'blocked_4xx'),
    ('https://b.vn/thuoc-ho-cho-tre-em-3', [903, 9031], 'b.vn', 'pending'),
    ('https://b.vn/dieu-tri-tieu-duong-tuyp-2-4', [904], 'b.vn', 'bot_challenge'),
    ('https://b.vn/khong-lien-quan-5', [905], 'b.vn', 'soft404_or_home'),
    ('https://c.vn/thuoc-ho-6', [906], 'c.vn', 'thin'),
    ('https://c.vn/tieu-duong-thuoc-7', [907], 'c.vn', 'network_error'),
]
CORPUS = [(910, 'https://www.nhathuoclongchau.com.vn/thuoc-ho-tre-em-p1'), (911, 'https://other.vn/x'),
          (912, 'https://nhathuoclongchau.com.vn/tieu-duong-tuyp-2-p2')]
QUERIES = ['thuốc ho cho trẻ em', 'điều trị tiểu đường tuýp 2']


@pytest.fixture
def crawl(tmp_path):
    db = tmp_path / 'crawl.db'
    c = sqlite3.connect(db)
    c.execute('create table urls (url_norm text, url text, doc_ids text, domain text, rank int, status text)')
    c.executemany('insert into urls values (?,?,?,?,?,?)',
                  [(u, u, json.dumps(d), dom, i, st) for i, (u, d, dom, st) in enumerate(CRAWL)])
    c.commit()
    c.close()
    corpus = tmp_path / 'corpus.parquet'
    pq.write_table(pa.table({'id': [r[0] for r in CORPUS], 'url': [r[1] for r in CORPUS]}), corpus)
    queries = tmp_path / 'q.parquet'
    pq.write_table(pa.table({'id': [1, 2], 'query': QUERIES}), queries)
    return db, corpus, queries


def _pin(tmp_path, crawl):
    db, corpus, _ = crawl
    return S.build_pin(db, corpus, tmp_path / 'pin.parquet')


def _reference(crawl, have, k=20):
    """Step 3 of r2ai.selective.step3_vs, spelled out on the fixture (the code that produced VS-append)."""
    db, corpus, queries = crawl
    rows = sqlite3.connect(db).execute('select url,doc_ids,domain,status from urls').fetchall()
    corp = [r for r in CORPUS if 'nhathuoclongchau.com.vn' in r[1]]
    urls = [r[0] for r in rows] + [r[1] for r in corp]
    ids = [json.loads(r[1]) for r in rows] + [[r[0]] for r in corp]
    st = [r[3] for r in rows] + ['not_in_crawl'] * len(corp)
    pool = np.array([i for i, s in enumerate(st) if s in S.BAD_STATUS or s == 'not_in_crawl'])
    bm = Slug(urls)
    Q = bm.queries(QUERIES)
    out = []
    Sx = (Q @ bm.WT[:, pool]).toarray()
    for q in range(len(QUERIES)):
        order = np.lexsort((np.arange(len(pool)), -Sx[q]))
        pick = []
        for j in order[:400]:
            if Sx[q][j] <= 0 or len(pick) >= k:
                break
            for d in ids[pool[j]][:1]:
                if d not in have[q] and d not in pick:
                    pick.append(d)
        out.append(pick)
    return out


BASE = ('[\n{"id": 1, "relevant_docs": [5, 6], "relevant_chunks": [{"doc_id": 5, "chunk_text": "th\\u1ecbt \\"x\\""}]},\n'
        '{"id": 2, "relevant_docs": [], "relevant_chunks": [{"doc_id": 7, "chunk_text": "y"}]}\n]\n')


def test_pin_holds_every_url_in_crawl_order_and_flags_the_pool(tmp_path, crawl):
    info = _pin(tmp_path, crawl)
    t = pq.read_table(tmp_path / 'pin.parquet').to_pydict()
    assert t['url'] == [r[0] for r in CRAWL] + [r[1] for r in CORPUS if 'nhathuoclongchau.com.vn' in r[1]]
    assert t['in_pool'] == [False, True, True, True, True, False, True, True, True]
    assert [d for d, p in zip(t['doc_id'], t['in_pool']) if p] == [902, 903, 904, 905, 907, 910, 912]
    assert info['rows'] == 9 and info['pool'] == 7 and len(info['sha256']) == 64
    assert info['sha256'] == S.sha256(tmp_path / 'pin.parquet')


def test_append_matches_the_step3_reference_and_keeps_every_other_byte(tmp_path, crawl):
    info = _pin(tmp_path, crawl)
    base, out = tmp_path / 'base.json', tmp_path / 'out.json'
    base.write_text(BASE, encoding='utf-8', newline='')
    st = S.append_slug_ids(base, out, pin=tmp_path / 'pin.parquet', pin_sha256=info['sha256'], queries=crawl[2],
                           k=20, k1=1.2, b=0.75, scan_depth=400)
    want = _reference(crawl, [{5, 6}, set()])
    assert want[0] and want[1]
    text = out.read_text(encoding='utf-8')
    assert text == BASE.replace('[5, 6]', '[5, 6, ' + ', '.join(map(str, want[0])) + ']') \
        .replace('"relevant_docs": []', '"relevant_docs": [' + ', '.join(map(str, want[1])) + ']')
    assert st['appended_total'] == sum(map(len, want)) and st['queries'] == 2
    assert st['queries_below_k'] == 2                                    # fixture pool is far below k=20


def test_append_k_limits_and_skips_docs_already_present(tmp_path, crawl):
    info = _pin(tmp_path, crawl)
    base, out = tmp_path / 'base.json', tmp_path / 'out.json'
    base.write_text(BASE.replace('[5, 6]', '[5, 6, 903, 910]'), encoding='utf-8', newline='')
    S.append_slug_ids(base, out, pin=tmp_path / 'pin.parquet', pin_sha256=info['sha256'], queries=crawl[2],
                      k=1, k1=1.2, b=0.75, scan_depth=400)
    rows = json.loads(out.read_text(encoding='utf-8'))
    first = rows[0]['relevant_docs']
    assert first[:4] == [5, 6, 903, 910] and len(first) == 5 and first[4] not in (903, 910)
    assert len(rows[1]['relevant_docs']) == 1


def test_pin_sha_mismatch_is_refused(tmp_path, crawl):
    _pin(tmp_path, crawl)
    base = tmp_path / 'base.json'
    base.write_text(BASE, encoding='utf-8', newline='')
    with pytest.raises(ValueError, match='pin'):
        S.append_slug_ids(base, tmp_path / 'o.json', pin=tmp_path / 'pin.parquet', pin_sha256='0' * 64, queries=crawl[2],
                          k=20, k1=1.2, b=0.75, scan_depth=400)


def test_replay_does_not_change_when_the_crawl_db_changes(tmp_path, crawl):
    info = _pin(tmp_path, crawl)
    base = tmp_path / 'base.json'
    base.write_text(BASE, encoding='utf-8', newline='')
    kw = dict(pin=tmp_path / 'pin.parquet', pin_sha256=info['sha256'], queries=crawl[2], k=20, k1=1.2, b=0.75, scan_depth=400)
    S.append_slug_ids(base, tmp_path / 'a.json', **kw)
    c = sqlite3.connect(crawl[0])
    c.execute("update urls set status='ok' where url like '%tieu-duong%'")
    c.execute("insert into urls values ('n','https://z.vn/thuoc-ho-tre-em-999','[999]','z.vn',0,'pending')")
    c.commit()
    c.close()
    S.append_slug_ids(base, tmp_path / 'b.json', **kw)
    assert (tmp_path / 'a.json').read_bytes() == (tmp_path / 'b.json').read_bytes()


def test_configs_best_is_rd150_plus_slug_append():
    b, r = best.load_config(ROOT / 'configs' / 'submission-best.yaml', 'VS-append'), best.load_config(ROOT / 'configs' / 'submission-rd150.yaml')
    assert r['submission']['expected_json_sha256'] == RD150_SHA and best.slug_append_config(r) is None
    sl = best.slug_append_config(b)
    assert sl['enabled'] is True and len(sl['pin_sha256']) == 64
    assert (sl['k'], sl['k1'], sl['b'], sl['scan_depth']) == (20, 1.2, 0.75, 400)
    assert b['submission']['expected_json_sha256'] == VS_SHA
    assert b['submission']['rrf_doc_set_json_sha256'] == RD150_SHA
    assert b['retrieval'] == r['retrieval'] and b['queries'] == r['queries']
    keys = ('runs_dir', 'doc_ranking', 'chunks_dir', 'docs_dir', 'args', 'max_zip_bytes', 'base_json_sha256', 'expand_json_sha256',
            'rrf_json_sha256')
    assert {k: b['submission'][k] for k in keys} == {k: r['submission'][k] for k in keys}
    assert {k: b['postprocess'][k] for k in ('expand_clusters', 'rrf_chunk_docs', 'rrf_doc_set')} == r['postprocess']
    off = copy.deepcopy(b)
    off['postprocess']['slug_append']['enabled'] = False
    assert best.slug_append_config(off) is None


def _chain(env, crawl, tmp_path, monkeypatch, enabled, pin_sha, final_sha, rd_sha, slug_k=20):
    q = env / 'query.parquet'
    pq.write_table(pa.table({'id': [1, 2], 'query': QUERIES}), q)
    _, js_ex = _h1(env, 'expand')
    rrf = _rrf_cli(env, js_ex)
    rd = _rd_cli(env, rrf)
    cfg = _rd_cfg(env, True, best.sha256(rrf), rd_sha or best.sha256(rd))
    cfg['submission'] |= {'expected_json_sha256': final_sha, 'rrf_doc_set_json_sha256': rd_sha or best.sha256(rd)}
    if enabled is not None:
        cfg['postprocess']['slug_append'] = {'enabled': enabled, 'pin': str(tmp_path / 'pin.parquet'), 'pin_sha256': pin_sha,
                                             'k1': 1.2, 'b': 0.75, 'scan_depth': 400}
        if slug_k is not None:                                                      # None = key absent (default k = 20)
            cfg['postprocess']['slug_append']['k'] = slug_k
    monkeypatch.setattr(best, 'load_config', lambda p, c=cfg: c)
    return rd


def test_best_build_runs_slug_append_last_and_off_keeps_rd150(env, crawl, tmp_path, monkeypatch):
    info = _pin(tmp_path, crawl)
    rd = _chain(env, crawl, tmp_path, monkeypatch, True, info['sha256'], 'x', None)
    S.append_slug_ids(rd, tmp_path / 'want.json', pin=tmp_path / 'pin.parquet', pin_sha256=info['sha256'],
                      queries=env / 'query.parquet', k=20, k1=1.2, b=0.75, scan_depth=400)
    want = (tmp_path / 'want.json').read_bytes()
    assert want != rd.read_bytes()
    final = best.sha256(tmp_path / 'want.json')
    _chain(env, crawl, tmp_path, monkeypatch, True, info['sha256'], final, None)
    assert best.main(['build', '--out', str(env / 'on/sub.zip')]) == 0
    assert (env / 'on/sub.json').read_bytes() == want
    assert (env / 'on/sub_rrf_doc_set.json').read_bytes() == rd.read_bytes()      # intermediate = RD150, hash-checked
    for enabled in (False, None):                                                   # off / key absent = RD150 bytes
        _chain(env, crawl, tmp_path, monkeypatch, enabled, info['sha256'], best.sha256(rd), None)
        assert best.main(['build', '--out', str(env / f'off_{enabled}/sub.zip')]) == 0
        assert (env / f'off_{enabled}/sub.json').read_bytes() == rd.read_bytes()
        assert not (env / f'off_{enabled}/sub_rrf_doc_set.json').exists()


def test_best_build_fails_on_wrong_pin_or_intermediate_hash(env, crawl, tmp_path, monkeypatch):
    info = _pin(tmp_path, crawl)
    _chain(env, crawl, tmp_path, monkeypatch, True, '0' * 64, 'x', None)
    with pytest.raises(ValueError, match='pin'):
        best.main(['build', '--out', str(env / 'bad1/sub.zip')])
    _chain(env, crawl, tmp_path, monkeypatch, True, info['sha256'], 'x', 'f' * 64)
    assert best.main(['build', '--out', str(env / 'bad2/sub.zip')]) == 1          # RD150 hash (intermediate) differs


# ---- regression guards: frozen expectations, independent of r2ai.selective.bm25 / r2ai.zh_sample.append -------------------
GOLDEN_OUT = ('[\n{"id": 1, "relevant_docs": [5, 6, 903, 910, 907], "relevant_chunks": [{"doc_id": 5, "chunk_text": "th\\u1ecbt \\"x\\""}]},\n'
              '{"id": 2, "relevant_docs": [904, 912, 902, 907], "relevant_chunks": [{"doc_id": 7, "chunk_text": "y"}]}\n]\n')


def test_golden_tokenization_is_frozen():
    """A change of r2ai.selective.bm25 tokenization (strip_marks / toks / slug_toks) changes the VS-append ids: fail loudly."""
    from r2ai.selective.bm25 import slug_toks, toks
    assert slug_toks('https://a.vn/Thuốc-ho-trẻ-em_đau-1?x=Tiểu%20đường') == ['thuoc', 'ho', 'tre', 'em', 'dau', 'tieu', 'duong']
    assert toks('Điều trị tiểu đường, tuýp 2 (type-2) ĐẠI') == ['dieu', 'tri', 'tieu', 'duong', 'tuyp', 'type', 'dai']


def test_golden_output_bytes_on_fixture(tmp_path, crawl):
    """Frozen expected bytes (ids, order and every other byte, incl. the \\u escapes) of slug_append on the fixture: a change in
    r2ai.selective.bm25 or r2ai.zh_sample.append (_rows / _addition) that moves a byte or an id makes this fail."""
    info = _pin(tmp_path, crawl)
    base, out = tmp_path / 'base.json', tmp_path / 'out.json'
    base.write_text(BASE, encoding='utf-8', newline='')
    st = S.append_slug_ids(base, out, pin=tmp_path / 'pin.parquet', pin_sha256=info['sha256'], queries=crawl[2],
                           k=20, k1=1.2, b=0.75, scan_depth=400)
    assert out.read_text(encoding='utf-8') == GOLDEN_OUT
    assert st['query_picks'] == [[903, 910, 907], [904, 912, 902, 907]]


def test_apply_cli_writes_json_and_zip_like_the_library_call(tmp_path, crawl, capsys):
    import zipfile
    info = _pin(tmp_path, crawl)
    base = tmp_path / 'base.json'
    base.write_text(BASE, encoding='utf-8', newline='')
    assert S.main(['apply', '--base', str(base), '--pin', str(tmp_path / 'pin.parquet'), '--pin-sha256', info['sha256'],
                   '--queries', str(crawl[2]), '--k', '2', '--out', str(tmp_path / 'o' / 'x.zip')]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st['k'] == 2 and st['appended_total'] == 4 and 'query_picks' not in st
    js = (tmp_path / 'o' / 'x.json').read_text(encoding='utf-8')
    assert json.loads(js)[0]['relevant_docs'] == [5, 6, 903, 910] and json.loads(js)[1]['relevant_docs'] == [904, 912]
    with zipfile.ZipFile(tmp_path / 'o' / 'x.zip') as z:
        assert z.namelist() == ['x.json'] and z.read('x.json').decode('utf-8') == js



def test_best_build_slug_k_is_a_config_parameter_with_default_20(env, crawl, tmp_path, monkeypatch):
    info = _pin(tmp_path, crawl)
    rd = _chain(env, crawl, tmp_path, monkeypatch, True, info['sha256'], 'x', None)
    shas = {}
    for k in (20, 2):
        S.append_slug_ids(rd, tmp_path / f'want{k}.json', pin=tmp_path / 'pin.parquet', pin_sha256=info['sha256'],
                          queries=env / 'query.parquet', k=k, k1=1.2, b=0.75, scan_depth=400)
        shas[k] = best.sha256(tmp_path / f'want{k}.json')
    assert shas[20] != shas[2]
    for slug_k, want in ((None, 20), (20, 20), (2, 2)):                              # absent key == k 20 == the pinned best
        _chain(env, crawl, tmp_path, monkeypatch, True, info['sha256'], shas[want], None, slug_k=slug_k)
        assert best.main(['build', '--out', str(env / f'k_{slug_k}/sub.zip')]) == 0
        assert (env / f'k_{slug_k}/sub.json').read_bytes() == (tmp_path / f'want{want}.json').read_bytes()
