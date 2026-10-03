"""Direct JSON validation must work on pinned Python and retain CR checks."""
import importlib
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from r2ai.eval.scorer import whitespace_tokenizer

@pytest.mark.parametrize('module_name',['r2ai.eval.validate','r2ai.submit.validate_submission'])
@pytest.mark.parametrize('newline,expected_exit',[(b'\n',0),(b'\r\n',1)])
def test_direct_json_validation_preserves_newline_contract(tmp_path,monkeypatch,module_name,newline,expected_exit):
    module=importlib.import_module(module_name)
    if hasattr(module,'Tokenizer'):
        monkeypatch.setattr(module,'Tokenizer',whitespace_tokenizer)
    docs=tmp_path/'docs'
    docs.mkdir()
    pq.write_table(pa.Table.from_pylist([{'id':1}]),tmp_path/'queries.parquet')
    pq.write_table(pa.Table.from_pylist([{'id':10}]),tmp_path/'corpus.parquet')
    row={'doc_ids':[10],'title':'','question':'','answer':'','body':'valid document body','paragraphs':['valid document body']}
    pq.write_table(pa.Table.from_pylist([row]),docs/'doc.parquet')
    raw=b'[\n{"id": 1, "relevant_docs": [10], "relevant_chunks": [{"doc_id": 10, "chunk_text": "valid document body"}]}\n]\n'
    source=tmp_path/'submission.json'
    source.write_bytes(raw.replace(b'\n',newline))
    assert module.main([str(source),'--queries',str(tmp_path/'queries.parquet'),'--corpus',str(tmp_path/'corpus.parquet'),'--docs-dir',str(docs)])==expected_exit
