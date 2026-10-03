"""Fetch browser-selected sources only; Codex assessments kept separate from user CSV."""
from r2ai.paths import auxiliary_disabled

import json
import argparse
import sys
from pathlib import Path

from r2ai.gold_check.gold_check_common import OUT, CorpusIndex, ResultsStore, data_path, domain, normalize_url, load_tokenizer, make_fetcher, timestamp, verify_url
from r2ai.gold_check.gold_check_report import generate_report


def main():
    auxiliary_disabled()
    parser = argparse.ArgumentParser()
    parser.add_argument('--recompute-cached', action='store_true', help='Re-extract saved successful HTML; fetch only new sources')
    args = parser.parse_args()
    target = OUT/'gold_check_30'
    findings = json.loads((target/'browser_findings.json').read_text(encoding='utf-8'))
    tasks = json.loads((OUT/'gold_check_tasks.json').read_text(encoding='utf-8'))['tasks']
    by_id = {t['id']:t for t in tasks}
    index = CorpusIndex(data_path('links_corpus.parquet'), OUT/'gold_check_corpus.sqlite')
    tokenizer, fetcher = load_tokenizer(), make_fetcher()
    store = ResultsStore(target/'codex_results.csv')
    for finding in findings:
        task = by_id[finding['id']]
        previous = {r['url']:r for r in store.for_query(task['id'])}
        rows = []
        for url in finding.get('urls') or ['']:
            row = dict(query_id=task['id'], query=task['query'], word_count=task['word_count'], stratum=task['stratum'], url=url, url_norm=normalize_url(url) if url else '', domain=domain(url) if url else '', page_type=finding.get('page_type','hỏi đáp'), saved_at=timestamp(), notes='Codex browser-assisted review. '+finding.get('search_note',''))
            if url:
                row.update(index.lookup(url))
                old = previous.get(url)
                if old and old.get('verification_status') == 'ok' and args.recompute_cached:
                    class CachedFetcher:
                        def fetch(self, requested):
                            return {'final_url':old.get('final_url') or requested, 'state':old.get('fetch_state') or 'ok', 'http_status':int(old.get('http_status') or 200)}, Path(old['html_path']).read_bytes()
                    row.update(verify_url(task, url, index, CachedFetcher(), tokenizer, target/'pages'))
                elif old and old.get('verification_status') == 'ok':
                    row.update({k:v for k,v in old.items() if k not in ('found','url_found','page_type','url_page_type','notes','saved_at')})
                else:
                    row.update(verify_url(task, url, index, fetcher, tokenizer, target/'pages'))
                ratio = float(row.get('lcs_ratio') or 0)
                hit = str(row.get('verbatim_hit')).lower() == 'true'
                suggested = 'verbatim' if hit else 'partial' if ratio >= .85 else 'none'
                row['found'] = finding.get('found', suggested)
                if row['verification_status'] != 'ok':
                    row['notes'] += ' Unverified source: fetch failed; classification needs browser evidence.'
            else:
                row.update(found=finding.get('found','none'), verification_status='not_applicable')
            row.update(url_found=row['found'], url_page_type=row['page_type'])
            rows.append(row)
            store.replace_query(task['id'], rows)
            print(json.dumps({k:row.get(k) for k in ('query_id','found','in_corpus','doc_ids','verbatim_hit','body_verbatim_hit','match_source','lcs_ratio','verification_status','error')},ensure_ascii=True), flush=True)
    generate_report(OUT/'gold_check_tasks.json', store.path, index.domain_counts(), target/'report.md')
    events_path = target/'request_events.json'
    prior = json.loads(events_path.read_text(encoding='utf-8')) if events_path.exists() else []
    events_path.write_text(json.dumps(prior+fetcher.request_events,indent=2),encoding='utf-8')
    print(f'Completed {len(findings)}/{len(tasks)} browser investigations; output {target}', flush=True)


if __name__ == '__main__':
    main()
