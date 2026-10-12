"""Full Common Crawl index measurement for nhathuoclongchau.com.vn: every crawl of the window, newest first (index only, no WARC).

    python -m r2ai.cc_coverage.run_all --ua "<agent string>" [--rps 5 --workers 4]

Per crawl: footer map (cached), the one row group holding the domain (cached rows file), then cumulative coverage over the
79,598 corpus URLs. Resumable: a crawl with both cache files is not requested again. Stops (and says why) on 403/429, a cap of
the cumulative ledger (2 GB / 5,000 requests), a projected cap breach before the next crawl, or any anomaly.
Does not extend the window backwards: it reports the marginal gain of the oldest crawls so the user can decide.
"""
from __future__ import annotations

from r2ai.paths import assert_writable

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import choose as CH
from . import crawls as CR
from . import index_scan as IX
from .net import Blocked, BudgetExceeded, Client, InvalidResponse, Ledger, Unavailable
from .step0 import OUT, COLLINFO, corpus_urls, old_sample, records_of, scan_crawl
from .surt import surt_key

REQUESTS_PER_CRAWL = 330           # projected cost used to stop before a crawl that would not fit under the cap (measured: ~306)
OLDEST_GAIN_POINTS = 2.0


def usable_keys(table: pa.Table):
    cols = table.to_pydict()
    return {k for k, st, m in zip(cols['url_surtkey'], cols['fetch_status'], cols['content_mime_type'])
            if str(st) == '200' and m == CH.HTML}


def coverage_curve(out: Path, order: list[str], key_of: dict, total: int) -> list[dict]:
    covered, rows = set(), []
    for crawl in order:
        f = out / f'rows_{crawl}.parquet'
        if not f.exists():
            continue
        t = pq.read_table(f)
        keys = usable_keys(t)
        mine = {i for i, k in key_of.items() if k in keys}
        new = mine - covered
        covered |= mine
        rows.append({'crawl': crawl, 'domain_rows': t.num_rows, 'covered_by_crawl': len(mine), 'covered_by_crawl_pct': round(100 * len(mine) / total, 2),
                     'new_vs_previous': len(new), 'cumulative': len(covered), 'cumulative_pct': round(100 * len(covered) / total, 2),
                     'marginal_points': round(100 * len(new) / total, 2)})
    return rows


def final_summary(out: Path, order: list[str], corpus: list, report: dict) -> dict:
    prefix_of = {i: ('/benh' if '/benh/' in u else '/bai-viet') for i, u in corpus}
    key_of = {i: surt_key(u) for i, u in corpus}
    if len(set(key_of.values())) != len(key_of):
        report['warning_duplicate_surt_keys'] = len(key_of) - len(set(key_of.values()))
    by_key, snaps = {}, []
    wanted = set(key_of.values())
    for crawl in order:
        f = out / f'rows_{crawl}.parquet'
        if not f.exists():
            continue
        for r in records_of(pq.read_table(f), crawl):
            if r['url_surtkey'] in wanted:
                by_key.setdefault(r['url_surtkey'], []).append(r)
    id_of = {k: i for i, k in key_of.items()}
    for k, recs in by_key.items():
        for r in recs:
            snaps.append({'url_id': id_of[k], **{c: r[c] for c in ('crawl', 'fetch_time', 'fetch_status', 'content_mime_type', 'warc_filename',
                                                                  'warc_record_offset', 'warc_record_length')}})
    chosen = {i: CH.choose(by_key.get(k, []), CR.CORPUS_DATE) for i, k in key_of.items()}
    snap_cols = ('url_id', 'crawl', 'fetch_time', 'fetch_status', 'content_mime_type', 'warc_filename', 'warc_record_offset', 'warc_record_length')
    pq.write_table(pa.table({c: [r[c] for r in snaps] for c in snap_cols}) if snaps else pa.table({c: pa.array([], pa.int64()) for c in snap_cols}),
                   assert_writable(out / 'snapshots.parquet'))
    pick_cols = ('crawl', 'fetch_time', 'warc_filename', 'warc_record_offset', 'warc_record_length', 'after_corpus')
    cov = [{'url_id': i, 'prefix': prefix_of[i], 'n_snapshots': sum(1 for r in by_key.get(key_of[i], []) if CH.usable(r)),
            **{c: (chosen[i][c] if chosen[i] else None) for c in pick_cols}} for i in key_of]
    pq.write_table(pa.table({c: [r[c] for r in cov] for c in ('url_id', 'prefix', 'n_snapshots') + pick_cols}), assert_writable(out / 'coverage.parquet'))
    summary = CH.summarize(chosen, prefix_of)
    six = [c for c in CR.OLD_SIX if c in order]
    sample = old_sample()
    six_keys = set()
    for c in six:
        f = out / f'rows_{c}.parquet'
        if f.exists():
            six_keys |= usable_keys(pq.read_table(f))
    summary['sample_vs_old_six_crawls'] = {
        'n': len(sample), 'old_found': sum(x[2] == 'found' for x in sample),
        'new_found_in_the_same_six_crawls': sum(surt_key(x[1]) in six_keys for x in sample),
        'agree': sum((x[2] == 'found') == (surt_key(x[1]) in six_keys) for x in sample),
        'old_found_new_missing': [x[1] for x in sample if x[2] == 'found' and surt_key(x[1]) not in six_keys],
        'old_missing_new_found': [x[1] for x in sample if x[2] != 'found' and surt_key(x[1]) in six_keys]}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ua', required=True)
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--rps', type=float, default=5.0)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args(argv)
    out = assert_writable(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / 'ledger.json')
    client = Client(ua=a.ua, ledger=ledger, log_path=out / 'requests.jsonl', rps=a.rps)
    t_all, start_b, start_r = time.time(), ledger.bytes, ledger.requests
    report = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'ua': a.ua, 'rps': a.rps, 'workers': a.workers, 'crawls': {}}
    corpus = corpus_urls()
    key_of = {i: surt_key(u) for i, u in corpus}

    def finish(status, **kw):
        report.update({'status': status, **kw, 'ledger': {'bytes': ledger.bytes, 'requests': ledger.requests},
                       'run_bytes': ledger.bytes - start_b, 'run_requests': ledger.requests - start_r, 'seconds': round(time.time() - t_all, 1)})
        (out / 'run_all_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k not in ('crawls', 'window')}, ensure_ascii=False, default=str)[:1800], flush=True)
        return 0 if status == 'ok' else 2

    try:
        info = json.loads(client.get(COLLINFO).content)
        window = CR.select_crawls(info)
        report['window'] = window
        order = [c['id'] for c in sorted(window, key=lambda c: CR.crawl_key(c['id']), reverse=True)]
        for crawl in order:
            have = (out / f'rows_{crawl}.parquet').exists() and (out / f'footer_map_{crawl}.json').exists()
            if not have and ledger.requests + REQUESTS_PER_CRAWL > ledger.max_requests:
                return finish('stopped_projected_budget', reason=f'{crawl}: {ledger.requests} + {REQUESTS_PER_CRAWL} would pass the cap {ledger.max_requests}')
            res = scan_crawl(client, crawl, out, workers=a.workers)
            report['crawls'][crawl] = {k: v for k, v in res.items() if k not in ('row_group_hits',)}
            curve = coverage_curve(out, order, key_of, len(corpus))
            (out / 'coverage_curve.json').write_text(json.dumps(curve, indent=1), encoding='utf-8')
            last = next((c for c in curve if c['crawl'] == crawl), None)
            print(crawl, res['status'], res.get('domain_rows'), 'cumulative_pct', last and last['cumulative_pct'], 'ledger', ledger.requests, ledger.bytes, flush=True)
            if res['status'] != 'ok':
                return finish('stopped_anomaly', reason=f"{crawl}: {res['status']}")
        curve = coverage_curve(out, order, key_of, len(corpus))
        summary = final_summary(out, order, corpus, report)
        report.update({'summary': summary, 'coverage_curve': curve,
                       'oldest_crawl_marginal_points': curve[-1]['marginal_points'] if curve else None,
                       'ask_before_extending_back': bool(curve) and curve[-1]['marginal_points'] >= OLDEST_GAIN_POINTS,
                       'rule': f'ask before going further back if the oldest crawl still adds >= {OLDEST_GAIN_POINTS} points'})
        (out / 'summary.json').write_text(json.dumps({'summary': summary, 'coverage_curve': curve}, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        return finish('ok')
    except Blocked as e:
        return finish('stopped_blocked', reason=str(e))
    except BudgetExceeded as e:
        return finish('stopped_budget', reason=str(e))
    except (Unavailable, InvalidResponse, IX.TooBig, FileNotFoundError) as e:
        return finish('stopped_error', reason=f'{type(e).__name__}: {e}')
    finally:
        client.close()


if __name__ == '__main__':
    sys.exit(main())
