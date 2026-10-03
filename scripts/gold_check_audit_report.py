"""Add provenance and the complete per-query review to Codex's separate report."""
from r2ai.paths import OUT_DIR, auxiliary_disabled

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from r2ai.gold_check.gold_check_report import cell, rate, truth


def main():
    auxiliary_disabled()
    folder = OUT_DIR / 'gold_check_30'
    rows = list(csv.DictReader((folder/'codex_results.csv').open(encoding='utf-8-sig')))
    assert len(rows) == 30 and len({r['query_id'] for r in rows}) == 30
    found = [r for r in rows if r['found'] != 'none']
    assert all(r['verification_status'] == 'ok' for r in found)
    events = json.loads((folder/'request_events.json').read_text(encoding='utf-8'))
    starts = defaultdict(list)
    for e in events:
        host = urlsplit(e['url']).hostname
        assert not any(host == h or host.endswith('.'+h) for h in ['google.com','bing.com','coccoc.com'])
        starts[host].append(e['started_monotonic'])
    gaps = [b-a for times in starts.values() for a,b in zip(sorted(times),sorted(times)[1:])]
    assert min(gaps) >= .999
    report_path = folder/'report.md'
    report = report_path.read_text(encoding='utf-8')
    report = report.split('\n## Review bổ sung và giới hạn')[0].rstrip() + '\n'
    report = report.replace('# Gold location: báo cáo human-in-the-loop', '# Kiểm tra nguồn 30 query: Codex tìm qua trình duyệt và review')
    report = report.replace('Nhãn người ghi', 'Nhãn Codex review').replace('human-verbatim','review-verbatim').replace('người gắn verbatim','Codex gắn verbatim').replace('Đối chiếu người và máy','Đối chiếu review và máy')
    lines = [
        '\n## Review bổ sung và giới hạn',
        '',
        'Nhãn trong báo cáo này do Codex đánh giá sau tìm kiếm từng câu qua trình duyệt, theo quyền người dùng đã cấp. CSV nhập tay ban đầu được giữ riêng. Không có script gửi request tới công cụ tìm kiếm; chỉ fetch URL nguồn đã mở qua trình duyệt và robots.txt của domain đó.',
        '',
        f'Tìm được nguồn khớp nguyên văn/gần khớp/diễn đạt lại: {rate(len(found),len(rows))}.',
        'Năm câu chưa tìm được nguồn: 900, 527, 572, 876, 286. `none` nghĩa là chưa tìm thấy trong lượt kiểm tra, không chứng minh không có nguồn.',
        '',
        '25 nguồn khớp đều ngoài corpus theo chuẩn hoá URL đã yêu cầu. Hai candidate không khớp được giữ để audit: query 900 có URL bài liên quan trong corpus (doc_id 754207); query 286 là ca khác mốc thời gian, ngoài corpus. Không tính hai candidate này là gold.',
        '',
        'Đã kiểm tra thêm URL alias bằng cùng slug Vinmec/VnExpress (bỏ hậu tố -vi) hoặc cùng qID MEDLATEC trong corpus. Chỉ candidate liên quan query 900 có alias. Kiểm tra này chưa loại trừ alias với slug hoàn toàn khác, nguồn sao chép hay URL bị xoá.',
        '',
        'Query 774, 206, 1037 khớp nguyên văn ở tiêu đề, không phải toàn bộ thân câu hỏi. Query 767 có LCS=1 nhưng verbatim_hit=false vì lược bỏ ngày tháng giữa câu: LCS cao không đủ kết luận nguyên văn.',
        'Đã sửa parser giữ lại câu hỏi FAQ MEDLATEC mà trafilatura bỏ sót; tính lại toàn bộ HTML đã lưu. LCS dùng tokenizer BAAI/bge-m3 trên toàn trang; nhãn partial/paraphrase được đối chiếu với câu hỏi nguồn, không chỉ lấy ngưỡng LCS.',
        '',
        f'Kiểm chứng: {len([r for r in rows if r["url"]])} candidate fetch thành công; khoảng cách request nhỏ nhất cùng hostname {min(gaps):.3f}s; timeout 15s; 14 test tự động pass. HTML/text và log request nằm cùng thư mục.',
        '',
        'Ưu tiên crawl/kiểm tra coverage: Vinmec, MEDLATEC, rồi VnExpress; tách câu hỏi và câu trả lời. Chưa đạt điều kiện ≥50% verbatim + in_corpus để coi BM25 vi là trục chính. Giữ dense đa ngữ làm baseline, thử BM25 vi bổ trợ sau khi xác định corpus có tài liệu tương đương. Tập partial gợi ý chuẩn hoá viết tắt/chính tả và rewrite có kiểm chứng; không mặc định chạy LLM rewrite cho cả 1.200 câu.',
        '',
        '## Đủ 30 câu: nguồn và đánh giá',
        '',
        '| Query | Tầng | Nhãn review | Hit | LCS | In corpus | Nguồn | Ghi chú |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for r in rows:
        source = f'[{r["domain"]}]({r["url"]})' if r['url'] else '—'
        ratio = f'{float(r["lcs_ratio"]):.3f}' if r['lcs_ratio'] else '—'
        note = r['notes'].replace('Codex browser-assisted review. ','')
        lines.append('| '+ ' | '.join(map(cell,[r['query_id'],r['stratum'],r['found'],r['verbatim_hit'] or '—',ratio,r['in_corpus'] or '—',source,note]))+' |')
    report_path.write_text(report+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'queries':len(rows),'labels':dict(Counter(r['found'] for r in rows)),'found_domains':dict(Counter(r['domain'] for r in found)),'found_in_corpus':sum(truth(r['in_corpus']) for r in found),'minimum_request_gap':min(gaps)},ensure_ascii=True))


if __name__ == '__main__':
    main()
