"""Wrong output roots must fail before model loading or filesystem writes."""
import hashlib
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

def run_cli(tmp_path, module, argv, extra_env=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               R2AI_OLD_ROOT=str(tmp_path / 'old'), R2AI_DATA_DIR=str(tmp_path / 'old/data'),
               R2AI_OUT_DIR=str(tmp_path / 'new/out'), R2AI_LOG_DIR=str(tmp_path / 'new/logs'),
               R2AI_STATE_DIR=str(tmp_path / 'new/state'), R2AI_RAW_DIR=str(tmp_path / 'missing-raw'))
    env.update(extra_env or {})
    return subprocess.run([sys.executable, '-B', '-m', module, *map(str, argv)], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45)

@pytest.mark.parametrize('module,args', [
    ('r2ai.submit.make_submission', ['--k-doc','100','--k-chunk','19','--chunk-mode','full','--out','@OLD/sub{kc}.zip']),
    ('r2ai.index.chunk', ['--out-dir','@OLD/chunks']),
    ('r2ai.retrieve.run_retrieval_k100', ['--out-dir','@OLD/runs']),
    ('r2ai.crawl.cli', ['run','--state-dir','@OLD/state','--no-probe']),
    ('r2ai.extract.cli', ['--docs-dir','@OLD/docs']),
])
def test_cli_old_override_rejected_before_any_write(tmp_path, module, args):
    old = tmp_path / 'old'
    args = [x.replace('@OLD', str(old)) for x in args]
    proc = run_cli(tmp_path, module, args)
    assert proc.returncode != 0
    assert 'Write to OLD is forbidden' in proc.stderr + proc.stdout
    assert not old.exists()

@pytest.mark.parametrize('module,args', [
    ('r2ai.index.build', ['build']),
    ('r2ai.retrieve.run', ['dev']),
])
def test_fixed_output_root_is_guarded_without_gpu(tmp_path, module, args):
    old = tmp_path / 'old'
    key = 'R2AI_WORK_DATA_DIR' if module.endswith('build') else 'R2AI_OUT_DIR'
    proc = run_cli(tmp_path, module, args, {key: str(old)})
    assert proc.returncode != 0
    assert 'Write to OLD is forbidden' in proc.stderr + proc.stdout
    assert not old.exists()

def test_missing_submission_input_fails_before_output(tmp_path):
    out = tmp_path / 'new/result.zip'
    proc = run_cli(tmp_path, 'r2ai.submit.make_submission', ['--k-doc','100','--k-chunk','19','--chunk-mode','full','--out',out])
    assert proc.returncode != 0
    assert 'Missing or empty input' in proc.stderr + proc.stdout
    assert not out.parent.exists()

def create_status_db(path, domain):
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE urls(domain TEXT,status TEXT,next_try_at REAL); CREATE TABLE domains(domain TEXT,state TEXT,halt_reason TEXT,rate REAL,conns INT);")
        conn.execute("INSERT INTO urls VALUES (?, 'ok', NULL)", (domain,))
        conn.execute("INSERT INTO domains VALUES (?, 'done', NULL, 1, 1)", (domain,))

def test_status_defaults_to_new_and_explicit_legacy_is_read_only(tmp_path):
    new, old = tmp_path/'new/state/crawl.db', tmp_path/'old/state/crawl.db'
    create_status_db(new, 'new.example')
    create_status_db(old, 'old.example')
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (new,old)}
    proc = run_cli(tmp_path, 'r2ai.crawl.cli', ['status'], {'R2AI_LEGACY_STATE_DIR': str(old.parent)})
    assert proc.returncode == 0, proc.stderr
    assert 'new.example' in proc.stdout and 'old.example' not in proc.stdout
    proc = run_cli(tmp_path, 'r2ai.crawl.cli', ['status','--state-dir',old.parent])
    assert proc.returncode == 0, proc.stderr
    assert 'old.example' in proc.stdout and 'new.example' not in proc.stdout
    assert not (tmp_path/'new/logs').exists()
    for p in (new,old):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == before[p]

def test_report_status_connection_cannot_write(tmp_path):
    from vicrawl.report import _open
    p = tmp_path/'state/crawl.db'
    create_status_db(p, 'example')
    db = _open(p)
    try:
        with pytest.raises(sqlite3.OperationalError):
            db.conn.execute('CREATE TABLE should_not_write(x)')
    finally:
        db.close()

def test_auxiliary_cli_lock_preserves_library_import(tmp_path):
    from r2ai.gold_check.gold_check_report import generate_report
    from r2ai.gold_check.gold_check_common import ResultsStore
    from r2ai.probe.probe_common import Checkpoint
    from r2ai.probe.step1_archive_coverage import Service
    assert all(callable(x) for x in (generate_report, ResultsStore, Checkpoint, Service))
    proc = run_cli(tmp_path, 'r2ai.gold_check.gold_check_report', [])
    assert proc.returncode != 0
    assert 'auxiliary writer is disabled' in proc.stderr + proc.stdout
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('module', ['r2ai.probe.step0_dup_check', 'r2ai.probe.profile_report_v2'])
def test_auxiliary_async_or_direct_main_is_locked(tmp_path, module):
    proc = run_cli(tmp_path, module, [])
    assert proc.returncode != 0
    assert 'auxiliary writer is disabled' in proc.stderr + proc.stdout
    assert not (tmp_path/'new').exists()


def test_score_csv_override_is_guarded(tmp_path):
    proc = run_cli(tmp_path, 'r2ai.eval.scorer', ['--per-query', tmp_path/'old/result.csv'])
    assert proc.returncode != 0
    assert 'Write to OLD is forbidden' in proc.stderr + proc.stdout
    assert not (tmp_path/'old').exists()


def test_gold_report_keeps_legacy_output_basename(tmp_path):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from r2ai.gold_check.gold_check_app import create_app
    task=dict(id='q1',query='Q',word_count=1,stratum='short',searches=[])
    tasks, results=tmp_path/'tasks.json',tmp_path/'results.csv'
    tasks.write_text(json.dumps({'tasks':[task]}),encoding='utf-8')
    pq.write_table(pa.Table.from_pylist([{'id':'d1','url':'https://example.com/a'}]),tmp_path/'corpus.parquet')
    app=create_app(tasks,results,tmp_path/'corpus.parquet',tmp_path/'index.sqlite',tmp_path/'pages')
    response=app.test_client().post('/api/report',json={},headers={'X-Gold-CSRF':app.extensions['gold_csrf']})
    assert response.status_code==200
    assert Path(response.json['path']).name=='gold_check_report.md'
    assert (tmp_path/'gold_check_report.md').is_file()


@pytest.mark.parametrize('mode', ['import', 'cli'])
def test_profile_import_and_locked_cli_do_not_create_old_output(tmp_path, mode):
    old=tmp_path/'old'
    old.mkdir()
    output=old/'profile-output'
    if mode=='cli':
        proc=run_cli(tmp_path,'r2ai.probe.step0_profile',[],{'R2AI_OUT_DIR':str(output)})
        assert 'auxiliary writer is disabled' in proc.stdout+proc.stderr
    else:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',R2AI_OLD_ROOT=str(old),R2AI_DATA_DIR=str(old/'data'),R2AI_OUT_DIR=str(output))
        proc=subprocess.run([sys.executable,'-B','-c','import r2ai.probe.step0_profile'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
        assert proc.returncode==0,proc.stderr
    assert not output.exists()


def test_dup_check_import_does_not_execute_main(tmp_path):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',R2AI_OLD_ROOT=str(tmp_path/'old'),R2AI_DATA_DIR=str(tmp_path/'old/data'),R2AI_OUT_DIR=str(tmp_path/'new/out'))
    proc=subprocess.run([sys.executable,'-B','-c','from r2ai.probe.step0_dup_check import norm; assert callable(norm)'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert proc.returncode==0,proc.stderr
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('suffix',['.json','.stats.json'])
def test_submission_sidecar_redirect_is_guarded_before_input_or_output(tmp_path,monkeypatch,suffix):
    from r2ai import paths
    from r2ai.submit import make_submission as module
    old=tmp_path/'old'
    monkeypatch.setattr(paths,'OLD_ROOT',old)
    output=tmp_path/'new/result.zip'
    sidecar=output.with_suffix(suffix)
    original=Path.resolve
    def redirected(self,*args,**kwargs):
        return old/'redirected' if self==sidecar else original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',redirected)
    with pytest.raises(ValueError,match='Write to OLD is forbidden'):
        module.main(['--k-doc','1','--k-chunk','1','--chunk-mode','full','--out',str(output),'--runs-dir',str(tmp_path/'missing'),'--docs-dir',str(tmp_path/'missing-docs')])
    assert not output.parent.exists()


def test_empty_full_document_parquet_fails_before_tokenizer_and_output(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from r2ai.submit import make_submission as module
    runs,chunks,docs=(tmp_path/name for name in ('runs','chunks','docs'))
    for p in (runs,chunks,docs):p.mkdir()
    pq.write_table(pa.Table.from_pylist([{'query_id':1,'rank':1,'doc_id':10}]),runs/'vi_k100.parquet')
    pq.write_table(pa.Table.from_pylist([{'doc_id':10,'doc_ids_group':[10]}]),chunks/'docs.parquet')
    pq.write_table(pa.Table.from_pylist([{'id':1}]),tmp_path/'queries.parquet')
    schema=pa.schema([('question',pa.string()),('status',pa.string()),('doc_ids',pa.list_(pa.int64())),('answer',pa.string()),('body',pa.string()),('title',pa.string()),('description',pa.string())])
    pq.write_table(pa.Table.from_pylist([],schema=schema),docs/'empty.parquet')
    def tokenizer_should_not_run():pytest.fail('Tokenizer reached with empty docs')
    monkeypatch.setattr(module,'Tokenizer',tokenizer_should_not_run)
    output=tmp_path/'new/result.zip'
    with pytest.raises(ValueError,match='Empty document input'):
        module.main(['--k-doc','1','--k-chunk','1','--chunk-mode','full','--out',str(output),'--runs-dir',str(runs),'--chunks-dir',str(chunks),'--docs-dir',str(docs),'--queries',str(tmp_path/'queries.parquet')])
    assert not output.parent.exists()
