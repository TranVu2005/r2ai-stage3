"""Reclassify saved step-0 bodies OFFLINE. Never makes an HTTP request."""
from r2ai.paths import auxiliary_disabled, resolve_legacy_artifact

from collections import Counter
import json

from r2ai.probe.fetcher import STATES, classify
from r2ai.probe.probe_common import OUT, ROOT, Checkpoint, cli, read_csv, selected_domains, write_csv


def old_state(row):
    flags = [k for k in ('ok', 'thin', 'needs_js', 'blocked', 'paywall', 'is_pdf') if row.get(k) == 'true']
    return '+'.join(flags) if flags else ('network_or_robots' if row.get('error') else 'unclassified')


def main():
    auxiliary_disabled()
    p = cli(__doc__, 0)
    p.add_argument('--run-dir', default=str(OUT))
    args = p.parse_args()
    from pathlib import Path
    out = Path(args.run_dir)
    domains = selected_domains(args.domains)
    source = [r for r in read_csv(OUT / 'crawl_sample.csv') if domains is None or r['domain'] in domains]
    if args.n > 0:
        counts, filtered = Counter(), []
        for row in source:
            if counts[row['domain']] < args.n:
                filtered.append(row)
                counts[row['domain']] += 1
        source = filtered
    cp = Checkpoint(out / 'checkpoints/reclassify.jsonl')
    results = []
    for row in source:
        # Re-run offline classification cheaply so code fixes do not leave stale labels.
        raw_path = resolve_legacy_artifact(row['raw_path']) if row.get('raw_path') else None
        body = raw_path.read_bytes() if raw_path and raw_path.exists() else b''
        rec = {**row, 'key': row['id'], 'http_status': row.get('status'), 'old_state': old_state(row), 'redirect_chain': None, 'server': None, 'set_cookie': None, 'body_completeness': 'no_closing_html' if body and b'</html>' not in body.lower() else 'closing_html_present' if body else 'no_body', 'metadata_note': 'Historical probe did not save redirect hops/server/set-cookie; unavailable offline.'}
        if raw_path and not raw_path.exists():
            rec['error'] = 'saved_body_missing'
        rec = classify(rec, body)
        cp.add(rec)
        results.append(rec)
    assert len(results) == len(source)
    assert all(r['state'] in STATES for r in results)
    write_csv(out / 'reclassified.csv', results)
    transitions = Counter((r['old_state'], r['state']) for r in results)
    write_csv(out / 'reclassification_transitions.csv', [{'old_state': old, 'new_state': new, 'n': n} for (old, new), n in sorted(transitions.items())])
    print(json.dumps({'n': len(results), 'states': dict(Counter(r['state'] for r in results)), 'other': sum(r['state'] == 'other' for r in results)}, ensure_ascii=False))
    from r2ai.probe.profile_report_v2 import build_report
    build_report(out)


if __name__ == '__main__':
    main()
