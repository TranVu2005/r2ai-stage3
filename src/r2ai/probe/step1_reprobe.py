"""Local top-domain probe, paired UA diagnosis, robots and durable resume."""
from __future__ import annotations

from r2ai.paths import auxiliary_disabled

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
from pathlib import Path
import time

import requests

from r2ai.probe.fetcher import CHROME_UA, OLD_UA, Fetcher, Pacer
from r2ai.probe.probe_common import OUT, ROOT, SEED, Checkpoint, cli, corpus, corpus_path, experiment_config, key, read_csv, samples, selected_domains, write_csv


def load_tokenizer(online=False):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('BAAI/bge-m3', local_files_only=not online)
        tok.model_max_length = 10**9
        return tok, ''
    except Exception as e:
        return None, f'{type(e).__name__}: {str(e)[:180]}'


def main():
    auxiliary_disabled()
    p = cli(__doc__, 30)
    p.add_argument('--top', type=int, default=35)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--run-dir', default=str(OUT))
    p.add_argument('--download-tokenizer', action='store_true')
    p.add_argument('--retry-robots', action='store_true', help='Remeasure previous unavailable robots outcomes; preserve prior checkpoint evidence.')
    p.add_argument('--redo-ab', action='store_true', help='Repeat paired UA measurements, preserving checkpoint history.')
    p.add_argument('--cookie-diagnostic', action='store_true', help='Observe two cookie challenges on the same URL/session without solving them; never bypass robots.')
    args = p.parse_args()
    if args.n < 1 or args.workers < 1:
        p.error('--n and --workers must be positive')
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    cp = Checkpoint(out / 'checkpoints/reprobe.jsonl')
    cookie_cp = Checkpoint(out / 'checkpoints/cookie_diagnostic.jsonl')
    source = corpus_path()
    config = {'seed': SEED, 'n': args.n, 'top': args.top, 'chrome_ua': CHROME_UA, 'old_ua': OLD_UA, 'corpus_bytes': source.stat().st_size, 'corpus_mtime_ns': source.stat().st_mtime_ns}
    experiment_config(out / 'reprobe_config.json', config)
    df = corpus()
    stats = df.group_by('domain').agg(__import__('polars').len().alias('n_urls'), __import__('polars').col('norm').n_unique().alias('n_unique')).sort(['n_urls', 'domain'], descending=[True, False])
    total = df.height
    write_csv(out / 'corpus_domains_v2.csv', [{**r, 'weight_pct': 100 * r['n_urls'] / total} for r in stats.to_dicts()])
    domain_set = selected_domains(args.domains)
    domains = [r['domain'] for r in stats.head(args.top).to_dicts()] if domain_set is None else [r['domain'] for r in stats.to_dicts() if r['domain'] in domain_set]
    if domain_set and set(domains) != domain_set:
        p.error('Unknown domain(s): ' + ', '.join(domain_set - set(domains)))
    sample = samples(df, domains, args.n)
    # Store manifests per selected domain so a subset rerun cannot hide the full run.
    for domain in domains:
        experiment_config(out / f'manifests/direct-{domain}.json', [r for r in sample if r['domain'] == domain])
    del df
    tok, token_error = load_tokenizer(args.download_tokenizer)
    env_path = out / 'egress.json'
    try:
        t0 = time.monotonic()
        r = requests.get('https://ipinfo.io/json', timeout=15)
        r.raise_for_status()
        environment = r.json()
        environment.update(measured_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), elapsed_s=round(time.monotonic() - t0, 4), tokenizer='BAAI/bge-m3', tokenizer_error=token_error)
    except (requests.RequestException, ValueError) as e:
        environment = {'ip': None, 'country': None, 'error': f'{type(e).__name__}: {str(e)[:250]}', 'tokenizer_error': token_error}
    env_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2), 'utf-8')
    pacer = Pacer()
    groups = defaultdict(list)
    for row in sample:
        groups[row['domain']].append(row)
    historical = defaultdict(list)
    for r in read_csv(OUT / 'crawl_sample.csv'):
        historical[r['domain']].append(r)

    def worker(domain, rows):
        chrome, old, ab_chrome = Fetcher(pacer=pacer), Fetcher(OLD_UA, pacer=pacer), Fetcher(pacer=pacer)

        def measure(row, arm, fetcher, solve):
            k = key(domain, row['url'], arm)
            if k in cp.rows and not (args.retry_robots and cp.rows[k].get('robots') == 'UNAVAILABLE') and not (args.redo_ab and arm.startswith('ab_')):
                return cp.rows[k]
            start = time.monotonic()
            rec, body = fetcher.fetch(row['url'], solve_cookie=solve)
            rec.update(row, key=k, arm=arm, expected_n=len(rows), measured_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), egress_ip=environment.get('ip'), egress_country=environment.get('country'))
            if body:
                path = out / 'reprobe_raw' / domain / f'{row["id"]}-{arm}.html'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                rec['raw_path'] = str(path.resolve())
                if rec['state'] == 'ok' and tok is not None:
                    from r2ai.probe.fetcher import extract
                    rec['n_tok'] = len(tok(extract(body)[0], add_special_tokens=False)['input_ids'])
            rec.setdefault('n_tok', None)
            rec['worker_elapsed_s'] = round(time.monotonic() - start, 4)
            cp.add(rec, replace=args.retry_robots or args.redo_ab)
            return rec

        main_rows = [measure(row, 'chrome_cookie', chrome, True) for row in rows]
        blocked = sum(r['state'] in ('blocked_4xx', 'bot_challenge') for r in main_rows) / len(main_rows)
        prior = historical.get(domain, [])
        old_blocked = sum(r.get('blocked') == 'true' for r in prior) / len(prior) if prior else 0
        if blocked > .5 or old_blocked > .5:
            # Identical URLs and interleaved order control URL and temporal effects.
            for i, row in enumerate(rows[:10]):
                arms = [('ab_old', old), ('ab_chrome', ab_chrome)] if i % 2 == 0 else [('ab_chrome', ab_chrome), ('ab_old', old)]
                for arm, f in arms:
                    measure(row, arm, f, False)
        if args.cookie_diagnostic and domain not in cookie_cp.rows and any(r.get('cookie_solved') in (True, 'True', 'true') for r in main_rows):
            # Warm a normal session, then observe the same challenge twice on
            # the same allowed URL. Only clear anonymous anti-bot cookies.
            initial, _ = chrome.fetch(rows[0]['url'])
            name = initial.get('robots_cookie_name') or initial.get('cookie_name')
            observations = []
            for _ in range(2):
                if name:
                    for session in chrome.sessions.values():
                        for cookie in list(session.cookies):
                            if cookie.name == name:
                                session.cookies.clear(cookie.domain, cookie.path, cookie.name)
                observed, _ = chrome.fetch(rows[0]['url'], solve_cookie=False)
                observations.append({'state': observed['state'], 'cookie_hashes': observed['cookie_hashes'], 'url': observed['final_url'], 'robots': observed['robots']})
            values = [o['cookie_hashes'][-1] for o in observations if o['cookie_hashes']]
            verdict = 'changes_same_session_same_url' if len(set(values)) > 1 else 'same_value_same_session_same_url' if len(values) >= 2 else 'insufficient_observations'
            cookie_cp.add({'key': domain, 'domain': domain, 'observations': observations, 'verdict': verdict, 'successful_urls_reusing_cookie': sum(r.get('cookie_solved') in (True, 'True', 'true') for r in main_rows)})
            write_csv(out / 'cookie_diagnostic.csv', list(cookie_cp.rows.values()))
        for label, f in [('chrome', chrome), ('old', old), ('ab_chrome', ab_chrome)]:
            event_path = out / 'request_logs' / f'{domain}-{label}-{time.time_ns()}.json'
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event_path.write_text(json.dumps(f.request_events, ensure_ascii=False), 'utf-8')
        return domain, len(main_rows), sum(r['state'] == 'ok' for r in main_rows)

    write_csv(out / 'reprobe.csv', list(cp.rows.values()), ['key', 'id', 'domain', 'url', 'arm', 'state'] if not cp.rows else None)
    print(f'probe: {len(sample)} URLs / {len(domains)} domains; cached={len(cp.rows)}', flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, domain, rows): domain for domain, rows in groups.items()}
        for future in as_completed(futures):
            try:
                domain, n, ok = future.result()
                print(f'{domain}: ok={ok}/{n}', flush=True)
            except Exception as e:
                # Preserve measurements and make programmer/dependency failures visible.
                print(f'ERROR {futures[future]}: {type(e).__name__}: {e}', flush=True)
                (out / 'reprobe_errors.jsonl').open('a', encoding='utf-8').write(json.dumps({'domain': futures[future], 'error': str(e)}) + '\n')
            write_csv(out / 'reprobe.csv', list(cp.rows.values()))
    from r2ai.probe.profile_report_v2 import build_report
    build_report(out)


if __name__ == '__main__':
    main()
