"""Migration failures: dotenv precedence, OLD write protection and empty inputs."""
import importlib.util
from pathlib import Path
import shutil
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/r2ai/paths.py'

def load(tmp_path, monkeypatch, dotenv='', env=None):
    for key in list(__import__('os').environ):
        if key.startswith('R2AI_') or key in ('HF_HOME', 'HF_HUB_OFFLINE'):
            monkeypatch.delenv(key, raising=False)
    root = tmp_path / 'new'
    p = root / 'src/r2ai/paths.py'
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, p)
    (root / '.env').write_text(dotenv, encoding='utf-8-sig')
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.chdir(elsewhere)
    spec = importlib.util.spec_from_file_location('isolated_paths', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_dotenv_root_not_cwd_and_process_wins(tmp_path, monkeypatch):
    mod = load(tmp_path, monkeypatch, 'R2AI_OUT_DIR="outputs with # and ="\nHF_HUB_OFFLINE=1\n', {'HF_HUB_OFFLINE': '0'})
    assert mod.OUT_DIR == tmp_path / 'new/outputs with # and ='
    assert __import__('os').environ['HF_HUB_OFFLINE'] == '0'
    assert not mod.OUT_DIR.exists()

def test_external_data_requires_explicit_old_root(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='R2AI_OLD_ROOT'):
        load(tmp_path, monkeypatch, f'R2AI_DATA_DIR={tmp_path / "old/data"}\n')

def test_guard_rejects_old_but_not_similar_sibling(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    mod = load(tmp_path, monkeypatch, f'R2AI_DATA_DIR={old / "data"}\nR2AI_OLD_ROOT={old}\n')
    for path in (old, old / 'state/crawl.db', old / 'data/../out/x'):
        with pytest.raises(ValueError):
            mod.assert_writable(path)
    assert mod.assert_writable(tmp_path / 'old-other/out') == tmp_path / 'old-other/out'
    assert mod.assert_writable('output') == tmp_path / 'new/output'
    assert not old.exists()

def test_empty_process_root_fails_instead_of_using_cwd(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='R2AI_OUT_DIR'):
        load(tmp_path, monkeypatch, 'R2AI_OUT_DIR=valid\n', {'R2AI_OUT_DIR': ''})

def test_raw_override_and_new_state_are_independent_of_legacy(tmp_path, monkeypatch):
    old = tmp_path / 'old'
    new = tmp_path / 'new'
    mod = load(tmp_path, monkeypatch, f'R2AI_DATA_DIR={old / "data"}\nR2AI_OLD_ROOT={old}\nR2AI_RAW_DIR={new / "data/raw_vi"}\nR2AI_LEGACY_STATE_DIR={old / "state"}\n')
    assert mod.RAW_DIR == mod.RAW_WRITE_DIR == new / 'data/raw_vi'
    assert mod.STATE_DIR == new / 'state'
    assert mod.LEGACY_STATE_DIR == old / 'state'
    assert mod.CHUNKS_DIR == new / 'data/chunks'
    assert mod.RUNS_DIR == new / 'out/runs'
    assert mod.LEGACY_RUNS_DIR == old / 'data/runs'

def test_windows_quoted_path_and_literal_characters(tmp_path, monkeypatch):
    load(tmp_path, monkeypatch, "HF_HOME='C:\\cache with # = value'\n")
    assert __import__('os').environ['HF_HOME'] == 'C:\\cache with # = value'

def test_malformed_dotenv_hides_value(tmp_path, monkeypatch):
    with pytest.raises(ValueError) as exc:
        load(tmp_path, monkeypatch, 'BAD SECRET VALUE\n')
    assert '.env:1' in str(exc.value)
    assert 'SECRET' not in str(exc.value)

def test_missing_and_empty_input_fails_without_output(tmp_path, monkeypatch):
    mod = load(tmp_path, monkeypatch)
    missing = tmp_path / 'missing'
    empty = tmp_path / 'empty'
    empty.mkdir()
    for path in (missing, empty):
        with pytest.raises(ValueError):
            mod.require_inputs(path)
    assert not (tmp_path / 'new/out').exists()


@pytest.mark.parametrize('nested',['index/t256/shards','state/crawl.db','chunks/docs.parquet','runs/partial.parquet','logs/crawl.log'])
def test_guard_rejects_redirected_descendant_before_directory_write(tmp_path,monkeypatch,nested):
    old=tmp_path/'old'
    mod=load(tmp_path,monkeypatch,f'R2AI_DATA_DIR={old / "data"}\nR2AI_OLD_ROOT={old}\n')
    root=tmp_path/'new/writer'
    child=root/nested
    child.parent.mkdir(parents=True,exist_ok=True)
    child.write_text('original',encoding='utf-8')
    original=Path.resolve
    def redirected(self,*args,**kwargs):
        return old/'protected' if self==child else original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',redirected)
    with pytest.raises(ValueError,match='Write to OLD is forbidden'):
        mod.assert_writable(root)
    assert child.read_text(encoding='utf-8')=='original'
    assert not old.exists()
