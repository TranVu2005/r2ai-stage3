"""Reverify only user-pasted URLs; preserve human labels and production CSV."""
from r2ai.paths import auxiliary_disabled

import json
import argparse
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

from r2ai.gold_check.gold_check_common import OUT, CorpusIndex, ResultsStore, data_path, load_tokenizer, make_fetcher, verify_url
from r2ai.gold_check.gold_check_report import generate_report, rate


def main():
    auxiliary_disabled()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cached', action='store_true', help='Recompute from previously fetched review HTML; no requests.')
    args = parser.parse_args()
    target = OUT / 'gold_check_review'
    target.mkdir(parents=True, exist_ok=True)
    store = ResultsStore(OUT / 'gold_check_results.csv')
    index = CorpusIndex(data_path('links_corpus.parquet'), OUT / 'gold_check_corpus.sqlite')
    tokenizer = load_tokenizer()
    fetcher = make_fetcher()
    if args.cached:
        cached = {(r['query_id'], r.get('url')): r for r in json.loads((target/'review.json').read_text(encoding='utf-8'))}

        class CachedFetcher:
            request_events = []

            def fetch(self, url):
                r = cached[(row['query_id'], url)]
                return dict(url=url, final_url=r.get('final_url') or url, state=r.get('fetch_state'), http_status=r.get('http_status'), error=r.get('error', '')), Path(r['html_path']).read_bytes()

        fetcher = CachedFetcher()
    rows, reviewed = [], []
    for original in store.rows:
        row = dict(original)
        if row.get('url'):
            row.update(index.lookup(row['url']))
            row.update(verify_url({'id':row['query_id'], 'query':row['query']}, row['url'], index, fetcher, tokenizer, target / 'pages'))
            # Offline alias hints only. Never fetch corpus URLs that were not pasted.
            slug = row['url_norm'].split('/')[-1].split('?')[0].removesuffix('-vi')
            with closing(sqlite3.connect(index.cache)) as db:
                aliases = [dict(url_norm=r[0], doc_id=r[1]) for r in db.execute('SELECT norm,doc_id FROM urls WHERE norm LIKE ? LIMIT 30', ('%/' + slug + '%',))]
            row['corpus_slug_candidates'] = aliases
            print(json.dumps(dict(query_id=row['query_id'], human_found=row.get('url_found') or row['found'], in_corpus=row['in_corpus'], doc_ids=row['doc_ids'], verification_status=row['verification_status'], error=row['error'], verbatim_hit=row.get('verbatim_hit'), match_source=row.get('match_source'), body_verbatim_hit=row.get('body_verbatim_hit'), lcs_ratio=row.get('lcs_ratio'), aliases=aliases), ensure_ascii=True), flush=True)
            reviewed.append(row)
        rows.append(row)
    (target / 'review.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    audited = ResultsStore(target / 'reviewed_results.csv')
    audited._write(rows)
    generate_report(OUT / 'gold_check_tasks.json', audited.path, index.domain_counts(), target / 'report.md')
    lines = ['# Review các URL người dùng đã dán', '', 'Giữ nguyên nhãn người và CSV production. Chỉ fetch URL đã dán; các URL corpus cùng slug chỉ là gợi ý offline.', '', '| ID | Nhãn người | Exact corpus | Doc ID | Xác minh | Verbatim | Nguồn khớp | LCS |', '|---|---|---|---|---|---|---|---|']
    for r in reviewed:
        lines.append(f'| {r["query_id"]} | {r.get("url_found") or r["found"]} | {r["in_corpus"]} | {r["doc_ids"]} | {r["verification_status"]} | {r.get("verbatim_hit")} | {r.get("match_source")} | {r.get("lcs_ratio")} |')
    ok = [r for r in reviewed if r['verification_status'] == 'ok']
    hits = [r for r in ok if r.get('verbatim_hit')]
    lines += ['', f'Verbatim trên URL đã kiểm tra: {rate(len(hits), len(ok))}.', f'Gold verbatim có exact URL trong corpus: {rate(sum(bool(r["in_corpus"]) for r in hits), len(hits))}.', '', 'Đề nghị rà nhãn:']
    for r in ok:
        suggestion = 'verbatim / hỏi đáp' if r.get('verbatim_hit') else 'chưa xác nhận gold; LCS cao chỉ là dấu hiệu liên quan'
        lines.append(f'- Query {r["query_id"]}: {suggestion}.')
    lines += ['', 'Query không có URL không được tự tìm thay người. Mẫu này chỉ có 3 URL, đều ở Vinmec; chưa đủ để kết luận chiến lược crawl/BM25 cho 1.200 query.', 'Lỗi robots_unavailable trước đó biến mất khi chạy fetch có quyền mạng: đây là lỗi môi trường sandbox, không phải bằng chứng trang/corpus không có gold.']
    for r in reviewed:
        text = Path(r['text_path']).read_text(encoding='utf-8') if r.get('text_path') else ''
        excerpt = text[r['match_start']:r['match_end']] if r.get('match_start') is not None else ''
        lines += ['', f'## Query {r["query_id"]}', '', r['query'], '', f'URL: {r["url"]}', f'Fetch: {r.get("fetch_state")}; HTTP: {r.get("http_status")}; error: {r.get("error")}', f'Match: {r.get("match_kind")}, source={r.get("match_source")}, [{r.get("match_start")}, {r.get("match_end")})', '', 'Đoạn khớp:', '', excerpt or '(chưa có)', '', '300 ký tự sau đoạn khớp:', '', r.get('after_match') or '(chưa có)', '', 'Corpus cùng slug (chưa xác nhận tương đương, không fetch):', '', '```json', json.dumps(r['corpus_slug_candidates'], ensure_ascii=False, indent=2), '```']
    (target / 'review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if not args.cached:
        (target / 'request_events.json').write_text(json.dumps(fetcher.request_events, indent=2), encoding='utf-8')
    print(f'Review saved: {target}', flush=True)


if __name__ == '__main__':
    main()
