"""Measured experiment report, with unmeasured fields explicit."""
from __future__ import annotations

from r2ai.paths import ROOT
from .common import RUN, RAW, DB, DOCS, CHUNKS, INDEX, preflight, guard_owned, atomic_json
import json
import sqlite3
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta


def read_json(path, default):
    return json.loads(path.read_text('utf-8')) if path.exists() else default


def report():
    preflight()
    inventory = read_json(RUN / 'inventory.json', [])
    conn = sqlite3.connect(DB.as_uri() + '?mode=ro', uri=True)
    try:
        counts = defaultdict(Counter)
        for d, s, n in conn.execute('SELECT domain,status,count(*) FROM urls GROUP BY domain,status'):
            counts[d][s] = n
        reasons = defaultdict(Counter)
        for d, reason, n in conn.execute('SELECT domain,reason,count(*) FROM urls WHERE reason IS NOT NULL GROUP BY domain,reason'):
            reasons[d][reason] = n
        domain_states = {r[0]: r[1] for r in conn.execute('SELECT domain,state FROM domains')}
    finally:
        conn.close()
    req, seconds, page_attempts = defaultdict(Counter), defaultdict(float), defaultdict(int)
    intervals = defaultdict(list)
    if (RUN / 'crawl_sessions.jsonl').exists():
        for line in (RUN / 'crawl_sessions.jsonl').open(encoding='utf-8'):
            r = json.loads(line)
            intervals[r['domain']].append((r['t0'], r['t1']))
            seconds[r['domain']] += r['t1'] - r['t0']
    if (RUN / 'requests.jsonl').exists():
        for line in (RUN / 'requests.jsonl').open(encoding='utf-8'):
            try:
                r = json.loads(line)
            except ValueError:  # live writer may be between writes of the final line
                continue
            if any(lo <= r['t0'] <= r['t1'] <= hi for lo,hi in intervals[r['domain']]):
                req[r['domain']][r['kind']] += 1
    from vicrawl.shards import list_shards, read_shard
    encoding, replacements, raw_bytes, seen = defaultdict(Counter), defaultdict(int), defaultdict(int), defaultdict(set)
    for shard in list_shards(RAW):
        d = shard.parent.name
        raw_bytes[d] += shard.stat().st_size
        for r in read_shard(shard):
            if 'latency' in r:
                seen[d].add(r['url_norm'])
                if any(lo <= r['fetched_at'] <= hi for lo,hi in intervals[d]):
                    page_attempts[d] += 1
            if r.get('html'):
                encoding[d][r.get('encoding', '')] += 1
                replacements[d] += r['html'].count('\ufffd')
    extract = read_json(RUN / 'extract_stats.json', {})
    measured = read_json(RUN / 'yield.json', {})
    chunk_counts, indexed_docs = Counter(), Counter()
    if (CHUNKS / 'docs.parquet').exists():
        import pyarrow.parquet as pq
        domain_for = {r['doc_id']: r['domain'] for r in pq.read_table(CHUNKS / 'docs.parquet', columns=['doc_id', 'domain']).to_pylist()}
        indexed_docs.update(domain_for.values())
        if (CHUNKS / 'chunks_t256.parquet').exists():
            for batch in pq.ParquetFile(CHUNKS / 'chunks_t256.parquet').iter_batches(columns=['doc_id']):
                chunk_counts.update(domain_for[d] for d in batch.column(0).to_pylist())
    index_bytes = sum(p.stat().st_size for p in (INDEX / 'dense.npy', INDEX / 'sparse.npz') if p.exists())
    table, records = [], []
    chosen = max(('hybrid', 'dense'), key=lambda b: measured.get(b, {}).get('zh_hits', 0))
    for inv in inventory:
        d = inv['domain']
        c, e = counts[d], extract.get(d, {})
        finished = sum(n for s, n in c.items() if s not in ('pending', 'in_progress'))
        rate = req[d]['page'] / seconds[d] if seconds[d] and page_attempts[d] else None
        url_rate = page_attempts[d] / seconds[d] if seconds[d] and page_attempts[d] else None
        hours = inv['n_unique'] / url_rate / 3600 if url_rate else None
        raw_est = raw_bytes[d] * inv['n_unique'] / len(seen[d]) if seen[d] else None
        idx_est = index_bytes * chunk_counts[d] / max(1, sum(chunk_counts.values())) * inv['n_unique'] / indexed_docs[d] if index_bytes and indexed_docs[d] else None
        y = measured.get(chosen, {}).get('yield', {}).get(d, {})
        record = {**inv, 'crawl_status': dict(c), 'finished': finished,
            'crawl_ok_pct_of_finished_sample': 100*c['ok']/finished if finished else None,
            'http_page_requests': req[d]['page'], 'http_robots_requests': req[d]['robots'],
            'page_requests_per_second': rate, 'url_attempts_per_second': url_rate,
            'crawl_wall_seconds_completed_sessions': seconds[d], 'extract': e,
            'encoding': dict(encoding[d]), 'unicode_replacement_chars': replacements[d],
            'chunks': chunk_counts[d] if chunk_counts else None, 'indexed_docs': indexed_docs[d] if indexed_docs else None,
            'yield_proxy': y, 'crawl_hours_extrapolated': hours,
            'raw_bytes_measured': raw_bytes[d], 'raw_bytes_extrapolated': raw_est,
            'index_bytes_extrapolated_proportional': idx_est, 'reasons': dict(reasons[d])}
        records.append(record)
        def number(value, digits=2):
            return f'{value:.{digits}f}' if value is not None else 'chưa đo'
        extract_pct = 100*e['ok']/e['n_docs'] if e.get('n_docs') else None
        answer_pct = 100*e['answer']/e['n_docs'] if e.get('n_docs') else None
        proxy_pair = ' / '.join(number(measured.get(b, {}).get('yield', {}).get(d, {}).get('hits_per_1k_extracted_docs')) for b in ('hybrid', 'dense'))
        status = 'loại theo yêu cầu' if inv['excluded'] else ('HALTED: mẫu bị cắt bởi policy' if domain_states.get(d) == 'halted' else ('robots/chưa truy cập được' if not page_attempts[d] else 'đã fetch'))
        table.append(f'| {d} | {inv["n_sample"]:,}/{inv["n_unique"]:,} | {number(record["crawl_ok_pct_of_finished_sample"])} | {number(rate)} | {number(extract_pct)} / {number(answer_pct)} | {record["chunks"] if record["chunks"] is not None else "chưa đo"} | {proxy_pair} | {number(hours)} | {number(raw_est/2**30 if raw_est is not None else None)} / {number(idx_est/2**30 if idx_est is not None else None)} | {status} |')
    atomic_json(RUN / 'domain_metrics.json', records)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    now = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
    chunk_report = read_json(CHUNKS / 'chunk_report.json', {})
    qa_chunk_report = read_json(RUN / 'qa_chunks/chunk_report.json', {})
    qa_summary = read_json(RUN / 'qa_summary.json', {})
    baseline_gate = read_json(RUN / 'd50_replay/stats.json', {})
    model_snapshots = read_json(RUN / 'model_snapshots.json', {})
    noop_parity = read_json(RUN / 'noop_vi_parity.json', {})
    embed_rows = []
    if (RUN / 'embed_sessions.jsonl').exists():
        for line in (RUN / 'embed_sessions.jsonl').open(encoding='utf-8'):
            try:
                embed_rows.append(json.loads(line))
            except ValueError:
                continue
    embed_stats = {'completed_shard_runs': len(embed_rows), 'encoded_chunks': sum(r['chunks'] for r in embed_rows),
                   'encode_and_checkpoint_seconds': sum(r['seconds'] for r in embed_rows),
                   'last_successful_run': read_json(INDEX / 'meta.json', {}),
                   'note': 'shard timing includes encode and checkpoint writes; unfinished shards excluded'} if embed_rows else {}
    variant = read_json(RUN / 'Z/stats.json', {})
    pytest_log = (RUN / 'pytest_full.log').read_text('utf-8', errors='replace') if (RUN / 'pytest_full.log').exists() else 'chưa đo'
    test_summary = next((line for line in reversed(pytest_log.splitlines()) if 'passed' in line or 'failed' in line), 'đang chạy / chưa đo')
    eligible = [r for r in records if r['yield_proxy'].get('hits_per_1k_extracted_docs') is not None and not r['excluded']]
    eligible.sort(key=lambda r: -r['yield_proxy']['hits_per_1k_extracted_docs'])
    order = ', '.join(r['domain'] for r in eligible) if eligible else 'chưa đo yield; chưa chốt thứ tự crawl toàn bộ'
    qa_table = ['| Domain QA snapshot | Doc | Body chars p50 | Answer | Thin | Body rỗng / answer ngoài body / U+FFFD |',
                '|---|---:|---:|---:|---:|---:|']
    for d, q in sorted(qa_summary.items()):
        qa_table.append(f'| {d} | {q["n"]} | {q["body_chars_p50"]} | {q["answer_present"]} | {q["thin"]} | {q["body_empty"]} / {q["answer_outside_body"]} / {q["replacement_chars"]} |')
    branch_table = ['| Nhánh | Query có zh trong top150 trộn | Query-doc zh hit |', '|---|---:|---:|']
    for b in ('hybrid', 'dense'):
        if b in measured:
            branch_table.append(f'| {b} | {measured[b]["queries_with_zh"]} | {measured[b]["zh_hits"]} |')
    text = '\n'.join([
        '# zh-sample REPORT', f'UTC+7: {now}; HEAD đã đọc: `{head}`; seed 42.',
        'Mô hình chạy inference: BAAI/bge-m3 + BAAI/bge-reranker-v2-m3 (cache local, fp16, max_len 512).',
        f'Snapshot cache đã xác minh: `{json.dumps(model_snapshots, ensure_ascii=False)}`' if model_snapshots else 'Snapshot model: chưa xác minh.',
        f'Mẫu: {sum(r["n_sample"] for r in inventory):,} nhóm URL trên {sum(not r["excluded"] for r in inventory)} domain zh; gộp http/https/www, giữ doc_ids cả nhóm. ceil(2% URL unique), tối thiểu 200 nếu đủ.',
        'Nguồn phân loại zh là snapshot legacy có lang=zh; domain chưa xác minh ngôn ngữ được ghi trong sample_manifest.json, chưa tự coi là zh.',
        'Egress VN kiểm bằng HTTP trực tiếp, trust_env=False; robots kiểm mỗi origin và đích redirect, crawl-delay/request-rate được áp dụng; mặc định nhóm zh cap 4 req/s, 2 connection, khởi đầu 1 req/s, tự giảm khi lỗi/429/503/latency. Không fetch nếu robots lỗi/chặn.',
        'Chỉ ghi bundle zh và out/runs/zh-sample. Không sửa PROGRESS/LOG (agent rerank200 phụ trách). Không nộp LB.',
        'Retry trang theo StateDB hiện có: tối đa 3 lần, cách 1 giờ; không chốt bundle khi còn retry đến hạn trong tương lai. Halt an toàn được giữ khi resume và ghi rõ mẫu bị cắt; không tự vượt halt để hoàn tất số URL.',
        '', '| Domain | Mẫu/URL unique | Crawl ok % trên URL đã xử lý | req/s trang | Extract ok % / answer % | Chunk t256 | Proxy H / D hit/1k extract doc | Giờ toàn domain* | Raw / index GiB* | Trạng thái |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|', *table,
        '', '* Ngoại suy từ đo thực mẫu; giờ dùng URL attempt/s (đã tính redirect trong thời gian), không giả định 1 request = 1 URL. Không dự báo được thay đổi rate/lỗi/robots về sau. Index phân bổ dung lượng dense+sparse thực theo tỷ trọng chunk; không gồm ANN, query cache, shard checkpoint. Rate chỉ dùng request và thời gian của các phiên đã kết thúc; không lấy tử số của phiên đang chạy/hard-kill.',
        'QA: qa/<domain>.json lưu tối đa 50 doc đã extract, số thực có trong extract_stats.json. Domain bị robots chặn hoặc mẫu nhỏ có ít hơn 50; chưa đo QA nội dung tại các domain đó. Charset/replacement và soft-404 lưu trong domain_metrics.json; các tỷ lệ answer của trang bài viết 0% không đồng nghĩa extractor lỗi.',
        'Extract ok dùng ngưỡng body ≥200 ký tự; thin gồm cả câu trả lời ngắn có nội dung thật. Tỷ lệ extract/answer chia trên doc đã extract, không chia trên URL bị robots chặn hoặc fetch lỗi.',
        'QA snapshot kiểm cấu trúc (input cố định tại qa_chunk_input; khác QA cập nhật sau full extract):' if qa_summary else 'QA kiểm cấu trúc: chưa đo.',
        '', *(qa_table if qa_summary else []), '',
        f'Chunk smoke trên snapshot QA (khác bundle index đầy đủ): `{json.dumps(qa_chunk_report, ensure_ascii=False)}`' if qa_chunk_report else 'Chunk smoke QA: chưa đo.',
        f'D50 replay gate: `{json.dumps(baseline_gate, ensure_ascii=False)}`' if baseline_gate else 'D50 replay gate: xem d50_replay.log; stats.json chưa ghi.',
        f'Đối chiếu đường dựng Z khi không chọn zh (không phải kết quả Z): `{json.dumps(noop_parity, ensure_ascii=False)}`' if noop_parity else 'Z no-op đối chiếu D50: chưa đo.',
        f'Chunk report: `{json.dumps(chunk_report, ensure_ascii=False)}`',
        'Chunker hiện có giữ nguyên thuật toán; mọi chunk là substring doc_text. Theo dõi dropped_whitespace_free_blobs cho đoạn CJK dài, không tự đổi chunker.',
        f'Embed đo thực: `{json.dumps(embed_stats, ensure_ascii=False)}`' if embed_stats else 'Embed đo thực: chưa đo.',
        f'Hybrid/dense-only: `{json.dumps(measured.get("measurement", {}), ensure_ascii=False)}`' if measured else 'Hybrid/dense-only: chưa đo.',
        '', *(branch_table if measured else []), '',
        'Yield là PROXY: số cặp query-doc zh trong top 150 khi trộn logits reranker với đúng tập doc D50, trên 1.000 doc extract của domain (gồm thin). Không có nhãn. Membership so logits trực tiếp; giữ thứ tự tương đối vi của D50; fp16 batch padding giữa cache vi và lượt zh có thể lệch ULP.',
        f'Z: `{json.dumps(variant, ensure_ascii=False)}`' if variant else 'Z: chưa đo; chưa có artifact hoàn tất. LB chưa đo, chưa kết luận zh có lợi.',
        f'Full suite: {test_summary}. Log: pytest_full.log. Slow suite: xem pytest_slow.log nếu có.',
        f'Thứ tự crawl toàn bộ đề xuất theo proxy giảm dần: {order}.',
        'Mốc cắt tỉa đề xuất: 27/10 chốt các domain/nhánh URL có proxy đo được; 28/10 23:59 ngừng nhận crawl mới; 29–30/10 dành cho embed/rerank/build/validator, ZIP thực ≤104.857.600 byte; 31/10 public do người dùng quyết định nộp. Đây là lịch đề xuất, không phải thời gian pipeline đã đo.',
        '', 'Resume: `.venv/Scripts/python.exe -X utf8 -B -m r2ai.zh_sample.pipeline`; trạng thái riêng pipeline_state.json. Ctrl+C một lần để flush; chạy lại cùng lệnh. Không đổi raw root hoặc input hash khi đã có index checkpoint.',
        'Lỗi/giới hạn còn lại phải xem pipeline_state.json và log từng bước; các trường chưa đo không được xem là 0 hoặc pass.', ''
    ])
    guard_owned(RUN / 'REPORT.md').write_text(text, encoding='utf-8')
    print(str(RUN / 'REPORT.md'), flush=True)


if __name__ == '__main__':
    report()
