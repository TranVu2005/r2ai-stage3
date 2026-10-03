"""One real sampled query + one synthetic localhost page, isolated output."""
from __future__ import annotations

from r2ai.paths import OUT_DIR, auxiliary_disabled

import json
import sys
import threading
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


from r2ai.gold_check.gold_check_app import create_app
from r2ai.gold_check.gold_check_common import load_tokenizer, make_fetcher, ResultsStore
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    auxiliary_disabled()
    out = OUT_DIR / 'gold_check_smoke'
    out.mkdir(parents=True, exist_ok=True)
    payload = json.loads((OUT_DIR / 'gold_check_tasks.json').read_text(encoding='utf-8'))
    task = payload['tasks'][0]
    answer = ' Bác sĩ trả lời: Đây là nội dung giả để kiểm thử luồng xác minh, không phải tư vấn y khoa. '
    body = ('<!doctype html><html><head><meta charset="utf-8"><title>Hỏi đáp kiểm thử</title></head><body><main><article><h1>Câu hỏi</h1><p>' + escape(task['query']) + '</p><h2>Trả lời</h2><p>' + escape(answer * 5) + '</p></article></main></body></html>').encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        requests_seen = []

        def do_GET(self):
            self.requests_seen.append(self.path)
            content = b'User-agent: *\nAllow: /\n' if self.path == '/robots.txt' else body
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain' if self.path == '/robots.txt' else 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/fake-gold?case=1'
    try:
        (out/'tasks.json').write_text(json.dumps({'tasks':[task]}, ensure_ascii=False), encoding='utf-8')
        pq.write_table(pa.Table.from_pylist([{'id':'fake-doc-1', 'url':url}, {'id':'fake-doc-2', 'url':url+'#fragment'}]), out/'corpus.parquet')
        tokenizer = load_tokenizer()
        fetcher = make_fetcher()
        app = create_app(out/'tasks.json', out/'results.csv', out/'corpus.parquet', out/'index.sqlite', out/'pages', tokenizer=tokenizer, fetcher=fetcher)
        client = app.test_client()
        headers = {'X-Gold-CSRF':app.extensions['gold_csrf']}
        assert client.get('/').status_code == 200
        annotation = dict(query_id=task['id'], urls=url, found='verbatim', page_type='hỏi đáp', notes='SYNTHETIC smoke; excluded from production results')
        saved = client.post('/api/save', json=annotation, headers=headers)
        assert saved.status_code == 200, saved.json
        assert saved.json['rows'][0]['doc_ids'] == ['fake-doc-1', 'fake-doc-2']
        verified = client.post('/api/verify', json={'query_id':task['id']}, headers=headers)
        row = verified.json['rows'][0]
        assert row['verification_status'] == 'ok', row
        assert row['verbatim_hit'] is True, row
        assert row['lcs_ratio'] == 1.0, row
        assert row['after_match'] and row['match_start'] is not None, row
        assert Path(row['html_path']).exists() and Path(row['text_path']).exists()
        assert ResultsStore(out/'results.csv').for_query(task['id'])[0]['found'] == 'verbatim'
        report = client.post('/api/report', json={}, headers=headers)
        assert report.status_code == 200, report.json
        assert '1/1: 100.0%' in Path(report.json['path']).read_text(encoding='utf-8')
        assert Handler.requests_seen == ['/robots.txt', '/fake-gold?case=1'], Handler.requests_seen
        times = [e['started_monotonic'] for e in fetcher.request_events]
        assert len(times) == 2 and times[1]-times[0] >= .99, times
        summary = dict(query_id=task['id'], url=url, doc_ids=row['doc_ids'], verbatim_hit=row['verbatim_hit'], lcs_ratio=row['lcs_ratio'], match_start=row['match_start'], match_end=row['match_end'], request_paths=Handler.requests_seen, request_gap_s=times[1]-times[0], report=report.json['path'])
        (out/'smoke_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
