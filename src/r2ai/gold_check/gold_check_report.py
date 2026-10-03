"""Report human annotations with explicit denominators and Wilson 95% CI."""
from __future__ import annotations

from r2ai.paths import auxiliary_disabled

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from r2ai.gold_check.gold_check_common import OUT, FOUND, STRATA, CorpusIndex, data_path, normalize_url


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (0.0 if k == 0 else max(0, center - half)), (1.0 if k == n else min(1, center + half))


def rate(k, n):
    ci = wilson(k, n)
    return f'{k}/{n}: {100*k/n:.1f}% [CI95% {100*ci[0]:.1f}–{100*ci[1]:.1f}%]' if ci else f'{k}/{n}: N/A'


def truth(x):
    return str(x).lower() in ('true', '1')


def cell(x):
    return str(x).replace('|', '\\|').replace('\n', ' ').replace('\r', '')


def url_found(row):
    return row.get('url_found') or row['found']


def url_page_type(row):
    return row.get('url_page_type') or row['page_type']


def generate_report(tasks_file, results_file, corpus_counts, dest, lcs_threshold=.8):
    payload = json.loads(Path(tasks_file).read_text(encoding='utf-8'))
    tasks = payload['tasks'] if isinstance(payload, dict) else payload
    tasks_by_id = {str(t['id']): t for t in tasks}
    rows = []
    if Path(results_file).exists():
        with Path(results_file).open(encoding='utf-8-sig', newline='') as f:
            rows = [r for r in csv.DictReader(f) if str(r['query_id']) in tasks_by_id]
    by_query = {}
    for row in rows:
        by_query.setdefault(str(row['query_id']), []).append(row)
    completed = [group[0] for group in by_query.values()]
    urls = [r for r in rows if r.get('url') and url_found(r) != 'none']
    # Each (query, normalized URL) is one assessment; global duplicates shown separately.
    urls = list({(str(r['query_id']), normalize_url(r['url'])): r for r in urls}.values())
    lines = ['# Gold location: báo cáo human-in-the-loop', '',
             f'Đã ghi {rate(len(completed), len(tasks))}. Query chưa ghi không được tính là `none`.',
             'Mọi tỷ lệ có CI95% Wilson; đơn vị query và URL được tách riêng. CI mô tả mẫu, không chứng minh nguồn gốc query.',
             'Mẫu cân bằng 10/10/10, không theo tỷ trọng 1.200 query: tỷ lệ tổng thể là tỷ lệ trong mẫu, không phải ước lượng không chệch cho toàn bộ tập.', '',
             '## Nhãn người ghi (mỗi query một nhãn)', '',
             '| Tầng | Đã ghi | verbatim | partial | paraphrase | none |',
             '|---|---:|---|---|---|---|']
    for s in ('all',) + STRATA:
        subset = completed if s == 'all' else [r for r in completed if tasks_by_id[str(r['query_id'])]['stratum'] == s]
        counts = Counter(r['found'] for r in subset)
        lines.append('| ' + ' | '.join([s, str(len(subset))] + [rate(counts[f], len(subset)) for f in FOUND]) + ' |')
    n = len(completed)
    found_queries = [g for g in by_query.values() if g[0]['found'] != 'none']
    external = sum(bool([r for r in g if r.get('url') and url_found(r) != 'none']) and not any(truth(r.get('in_corpus')) for r in g if r.get('url') and url_found(r) != 'none') for g in found_queries)
    joint = sum(g[0]['found'] == 'verbatim' and any(truth(r.get('in_corpus')) for r in g if r.get('url') and url_found(r) == 'verbatim') for g in by_query.values())
    confirmed = sum(g[0]['found'] == 'verbatim' and any(truth(r.get('in_corpus')) and truth(r.get('verbatim_hit')) and r.get('verification_status') == 'ok' for r in g if r.get('url') and url_found(r) == 'verbatim') for g in by_query.values())
    lines += ['', '## URL và corpus', '',
              f'URL có trong corpus (exact, giữ query string): {rate(sum(truth(r.get("in_corpus")) for r in urls), len(urls))}.',
              f'Query tìm thấy nhưng tất cả URL ngoài corpus: {rate(external, len(found_queries))}.',
              f'Query human-verbatim và có URL trong corpus: {rate(joint, n)}.',
              f'Query đồng thời human-verbatim, máy xác nhận và URL trong corpus: {rate(confirmed, n)}.',
              f'{len(urls)} cặp query–URL; {len({normalize_url(r["url"]) for r in urls})} URL chuẩn hoá duy nhất. ID trùng http/https/www không nhân số URL.', '',
              '| Domain | URL có corpus / URL domain | URL ngoài corpus / URL domain | Tỷ trọng URL tìm được | Tỷ trọng dòng corpus |',
              '|---|---|---|---|---|']
    domains = Counter(r['domain'] for r in urls)
    corpus_n = sum(corpus_counts.values())
    for d, count in domains.most_common():
        inside = sum(r['domain'] == d and truth(r.get('in_corpus')) for r in urls)
        lines.append(f'| {cell(d)} | {rate(inside, count)} | {rate(count-inside, count)} | {rate(count, len(urls))} | {rate(corpus_counts.get(d, 0), corpus_n)} |')
    lines += ['', '## Page type (đơn vị URL đã tìm thấy)', '', '| Loại | Tỷ lệ |', '|---|---|']
    types = Counter(url_page_type(r) for r in urls)
    for t in ('hỏi đáp', 'bài viết', 'khác'):
        lines.append(f'| {t} | {rate(types[t], len(urls))} |')
    ok = [r for r in urls if r.get('verification_status') == 'ok']
    bad = [r for r in urls if r.get('verification_status') != 'ok']
    lines += ['', '## Đối chiếu người và máy', '',
              f'URL đã đo thành công: {rate(len(ok), len(urls))}; chưa đo/lỗi: {rate(len(bad), len(urls))}.',
              f'Verbatim máy trên URL đo thành công: {rate(sum(truth(r.get("verbatim_hit")) for r in ok), len(ok))}.',
              f'LCS ≥ {lcs_threshold}: {rate(sum(float(r.get("lcs_ratio") or 0) >= lcs_threshold for r in ok), len(ok))}.',
              'Text đo gồm tiêu đề HTML và thân bài trafilatura; match_source phân biệt title/body. LCS = longest common subsequence / số token query BAAI/bge-m3, không cắt text. LCS cao có thể ghép token rải rác trên trang; không đồng nghĩa chép nguyên văn.', '',
              '| Nhãn | URL đo được | Máy verbatim | LCS trung bình |', '|---|---:|---|---:|']
    for f in FOUND:
        group = [r for r in rows if r.get('url') and url_found(r) == f and r.get('verification_status') == 'ok']
        mean = f'{sum(float(r.get("lcs_ratio") or 0) for r in group)/len(group):.3f}' if group else 'N/A'
        lines.append(f'| {f} | {len(group)} | {rate(sum(truth(r.get("verbatim_hit")) for r in group), len(group))} | {mean} |')
    mismatches = []
    for r in rows:
        if not r.get('url') or r.get('verification_status') != 'ok':
            continue
        machine = truth(r.get('verbatim_hit'))
        ratio = float(r.get('lcs_ratio') or 0)
        reason = ''
        if url_found(r) == 'verbatim' and not machine:
            reason = 'người: verbatim; máy: không hit'
        elif url_found(r) != 'verbatim' and machine:
            reason = 'người: không verbatim; máy: hit'
        elif url_found(r) == 'partial' and ratio < lcs_threshold:
            reason = f'partial nhưng LCS < {lcs_threshold}; cần rà (heuristic)'
        if reason:
            mismatches.append((r, reason))
    lines += ['', f'Bất đồng/cần rà trên mọi URL đã đo: {rate(len(mismatches), sum(r.get("verification_status") == "ok" and bool(r.get("url")) for r in rows))}.', '',
              '| Query | Nhãn | Hit | LCS | URL | Lý do |', '|---|---|---|---|---|---|']
    for r, reason in mismatches:
        lines.append('| ' + ' | '.join(cell(v) for v in (r['query_id'], url_found(r), r['verbatim_hit'], r['lcs_ratio'], r['url'], reason)) + ' |')
    if bad:
        lines += ['', 'URL lỗi/chưa xác minh (không quy thành negative):']
        lines += [f'- Query {cell(r["query_id"])}: {cell(r["url"])} — {cell(r.get("error") or r.get("verification_status"))}' for r in bad]
    # Gold density counts a query once per domain, independent of duplicate URLs.
    gold = Counter()
    for g in by_query.values():
        gold.update({r['domain'] for r in g if r.get('url') and url_found(r) == 'verbatim'})
    assignments = sum(gold.values())
    lines += ['', '## Mật độ gold theo domain', '',
              'Gold proxy = URL được người gắn verbatim. Một query tính tối đa một lần/domain; có thể thuộc nhiều domain. Corpus denominator là dòng tài liệu gốc, gồm ID trùng. Nhãn riêng URL tránh gán gold cho các candidate partial/paraphrase.',
              'Enrichment = tỷ trọng query–domain gold / tỷ trọng dòng corpus. Đây là tín hiệu ưu tiên crawl, chịu lệch từ tìm kiếm thủ công và các link site: được gợi ý.', '',
              '| Domain | Gold / tổng query đã ghi | Tỷ trọng gold assignment | Tỷ trọng corpus | Enrichment |', '|---|---|---|---|---|']
    for d, k in gold.most_common():
        corpus_share = corpus_counts.get(d, 0) / corpus_n if corpus_n else 0
        enrichment = f'{(k/assignments)/corpus_share:.2f}×' if corpus_share else 'N/A (ngoài corpus)'
        lines.append(f'| {cell(d)} | {rate(k, n)} | {rate(k, assignments)} | {rate(corpus_counts.get(d, 0), corpus_n)} | {enrichment} |')
    none = sum(r['found'] == 'none' for r in completed)
    top = [d for d, _ in gold.most_common(3)]
    concentration = sum(any(r.get('domain') in top and url_found(r) == 'verbatim' for r in g if r.get('url')) for g in by_query.values())
    gold_queries = sum(any(r.get('url') and url_found(r) == 'verbatim' for r in g) for g in by_query.values())
    ready = len(completed) == len(tasks) and bool(n)
    lines += ['', '## Bảng quyết định', '',
              'Các ngưỡng ngoài ≥50% verbatim+in_corpus là heuristic công khai: phần lớn = >50%; tập trung = ≥50% gold query ở top 1–3 domain.',
              'Nếu mẫu chưa hoàn tất, chỉ báo tín hiệu tạm thời. `none` sau tìm kiếm thủ công không đủ chứng minh query đã rewrite; ngoài corpus cũng không đủ chứng minh nguồn duy nhất.', '',
              '| Quy tắc | Số liệu | Hành động / trạng thái |', '|---|---|---|']
    def verdict(condition, action):
        return ('Đề xuất: ' if ready else 'Tạm thời: ') + action if condition else 'Chưa đạt điều kiện'
    lines.append(f'| ≥50% human-verbatim + in_corpus | {rate(joint, n)}; máy xác nhận {rate(confirmed, n)} | {verdict(n and joint/n >= .5, "crawl domain vi trước; BM25 vi là thành phần chính; parser tách hỏi/đáp")} |')
    lines.append(f'| Phần lớn query tìm thấy ngoài corpus | {rate(external, len(found_queries))} | {verdict(found_queries and external/len(found_queries) > .5, "khảo sát giả thuyết nguồn ngoài; dense đa ngữ làm trục chính")} |')
    lines.append(f'| Phần lớn không tìm thấy | {rate(none, n)} | {verdict(n and none/n > .5, "kiểm tra giả thuyết query viết lại; thử LLM rewrite query")} |')
    lines.append(f'| ≥50% gold ở 1–3 domain | {rate(concentration, gold_queries)}; {cell(", ".join(top) or "N/A")} | {verdict(gold_queries and concentration/gold_queries >= .5, "crawl các domain trên đầu tiên")} |')
    if not ready:
        lines += ['', '**Chưa đủ mẫu: chưa chốt thứ tự crawl hoặc vai trò BM25.**']
    lines += ['', 'Với n=30, các CI thường rộng. Đánh giá lại trên tập kiểm chứng trước khi áp dụng chiến lược cho toàn bộ 1.200 query.']
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return dest


def main():
    auxiliary_disabled()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, default=OUT / 'gold_check_tasks.json')
    p.add_argument('--results', type=Path, default=OUT / 'gold_check_results.csv')
    p.add_argument('--out', type=Path, default=OUT / 'gold_check_report.md')
    p.add_argument('--corpus', type=Path, default=data_path('links_corpus.parquet'))
    p.add_argument('--index', type=Path, default=OUT / 'gold_check_corpus.sqlite')
    p.add_argument('--lcs-threshold', type=float, default=.8)
    args = p.parse_args()
    index = CorpusIndex(args.corpus, args.index)
    print(generate_report(args.tasks, args.results, index.domain_counts(), args.out, args.lcs_threshold))


if __name__ == '__main__':
    main()
