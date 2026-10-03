"""Common Crawl/Wayback coverage with exact URL variants and resumable evidence."""
from __future__ import annotations

from r2ai.paths import auxiliary_disabled

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import re
import queue
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from r2ai.probe.fetcher import Pacer, classify, extract
from r2ai.probe.probe_common import OUT, ROOT, SEED, Checkpoint, cli, corpus, experiment_config, key, normalize_url, read_csv, samples, selected_domains, write_csv
from r2ai.probe.step1_reprobe import load_tokenizer



def variants(url):
    u = urlsplit(url)
    host = (u.hostname or '').removeprefix('www.')
    candidates = [urlunsplit((scheme, prefix + host, u.path or '/', u.query, '')) for scheme in (u.scheme, 'http' if u.scheme == 'https' else 'https') for prefix in (('www.', '') if (u.hostname or '').startswith('www.') else ('', 'www.'))]
    return list(dict.fromkeys(candidates))


class Service:
    """Serial requests per archive host, cached queries, independent circuit breaker."""
    def __init__(self, name, checkpoint, *, timeout=15, failures=3, pacer=None, deadline=None):
        self.name, self.cp, self.timeout, self.failure_limit = name, checkpoint, timeout, failures
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'R2AI-Stage3-ArchiveCoverage/1.0 (research; <=1 request/s)'
        self.pacer, self.lock = pacer or Pacer(), threading.Lock()
        self.consecutive_errors, self.error, self.open = 0, '', False
        self.requests, self.elapsed = 0, 0.
        self.deadline = deadline

    def status(self):
        return {'requests': self.requests, 'seconds': round(self.elapsed, 3), 'circuit_open': self.open, 'consecutive_errors': self.consecutive_errors, 'reason': self.error or '—'}

    def get(self, url, *, params=None, headers=None, binary=False):
        if self.deadline is None:
            return self._get_impl(url, params=params, headers=headers, binary=binary)
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            return {'error':'time_budget_exhausted', 'http_status':None}
        result_queue = queue.Queue(maxsize=1)
        def run():
            try:
                result_queue.put(self._get_impl(url, params=params, headers=headers, binary=binary))
            except Exception as e:
                result_queue.put({'error':f'{type(e).__name__}: {e}', 'http_status':None})
        # The coordinator returns at the absolute deadline even if a server
        # trickles forever. Daemon transports never keep the CLI process alive.
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            return result_queue.get(timeout=remaining)
        except queue.Empty:
            return {'error':'time_budget_exhausted', 'http_status':None}

    def _get_impl(self, url, *, params=None, headers=None, binary=False):
        query_key = hashlib.sha256(json.dumps([self.name, url, params, headers], sort_keys=True).encode()).hexdigest()
        if not binary and query_key in self.cp.rows:
            cached = self.cp.rows[query_key]
            if cached.get('error'):
                # Failed calls are retried on a later invocation, not frozen forever.
                pass
            else:
                return cached
        with self.lock:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                return {'key': query_key, 'error': 'time_budget_exhausted', 'http_status': None}
            if self.open:
                return {'key': query_key, 'error': 'circuit_open: ' + self.error, 'http_status': None}
            start = time.monotonic()
            result = {'key': query_key, 'url': url, 'params': params, 'http_status': None}
            for attempt in range(3):
                try:
                    current, current_params = url, params
                    redirects = []
                    for hop in range(11):
                        if self.deadline is not None and time.monotonic() >= self.deadline:
                            raise requests.Timeout('time_budget_exhausted')
                        self.pacer.wait(current)
                        if self.deadline is not None and time.monotonic() >= self.deadline:
                            raise requests.Timeout('time_budget_exhausted_after_pacing')
                        self.requests += 1
                        remaining = max(.1, self.deadline - time.monotonic()) if self.deadline else self.timeout
                        r = self.session.get(current, params=current_params, headers=headers, timeout=min(self.timeout, remaining), allow_redirects=False, stream=True)
                        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get('Location'):
                            target = urljoin(r.url or current, r.headers['Location'])
                            r.close()
                            if binary and ('wb' in self.name or 'wayback' in self.name) and urlsplit(target).hostname != 'web.archive.org':
                                raise requests.RequestException('Wayback replay redirect escaped archive host')
                            if hop == 10 or urlsplit(target).scheme not in ('http', 'https'):
                                raise requests.TooManyRedirects('archive redirect limit or invalid scheme')
                            redirects.append({'url': current, 'status': r.status_code, 'location': target})
                            current, current_params = target, None
                            continue
                        break
                    result['redirect_chain'] = redirects
                    result['http_status'] = r.status_code
                    result['final_url'] = r.url
                    result['response_headers'] = dict(r.headers)
                    if r.status_code == 429 or r.status_code >= 500:
                        r.close()
                        if attempt < 2:
                            try:
                                backoff = max(2 ** attempt, float(r.headers.get('Retry-After', 0)))
                            except ValueError:
                                backoff = 2 ** attempt
                            if self.deadline and time.monotonic() + backoff >= self.deadline:
                                result['error'] = 'time_budget_exhausted_before_retry_after'
                                break
                            time.sleep(backoff)
                            continue
                    chunks, size = [], 0
                    try:
                        for chunk in r.iter_content(65536):
                            chunks.append(chunk)
                            size += len(chunk)
                            if size > (10_000_000 if binary else 25_000_000) or time.monotonic() - start > 45 or (self.deadline and time.monotonic() > self.deadline):
                                raise requests.Timeout('archive payload size/time/budget limit')
                        downloaded = b''.join(chunks)
                        if binary:
                            result['body'] = downloaded
                        else:
                            result['text'] = downloaded.decode('utf-8-sig', errors='replace')
                    finally:
                        r.close()
                    if r.status_code not in (200, 206, 404):
                        result['error'] = f'HTTP {r.status_code}: {result.get("text", "")[:160]}'
                    else:
                        result.pop('error', None)
                    break
                except requests.RequestException as e:
                    result['error'] = f'{type(e).__name__}: {str(e)[:250]}'
                    if self.deadline and time.monotonic() >= self.deadline:
                        break
                    if attempt < 2 and (not self.deadline or time.monotonic()+2**attempt < self.deadline):
                        time.sleep(2 ** attempt)
                        continue
            result['elapsed_s'] = round(time.monotonic() - start, 4)
            self.elapsed += result['elapsed_s']
            if result.get('error'):
                self.consecutive_errors += 1
                self.error = result['error']
                self.open = self.consecutive_errors >= self.failure_limit
            else:
                self.consecutive_errors = 0
            if not binary:
                self.cp.add(result, replace=True)
            return result


def cc_lookup(url, indexes, service):
    errors = []
    for index in indexes:
        for variant in variants(url):
            r = service.get(index['cdx-api'], params={'url': variant, 'output': 'json', 'filter': 'status:200'})
            if r.get('error'):
                errors.append(r['error'])
                if service.open:
                    return {'cc_status': 'unknown', 'cc_error': '; '.join(errors)[-500:]}
                continue
            if r['http_status'] == 404:
                continue
            try:
                records = [json.loads(line) for line in r.get('text', '').splitlines() if line.strip()]
                hits = [x for x in records if str(x.get('status')) == '200' and all(k in x for k in ('url', 'filename', 'offset', 'length', 'timestamp'))]
                if hits:
                    hit = max(hits, key=lambda x: x['timestamp'])
                    return {'cc_status': 'found', 'cc_timestamp': hit['timestamp'], 'cc_record': hit, 'cc_crawl': index['id'], 'cc_variant': variant}
            except (ValueError, TypeError, KeyError) as e:
                errors.append(f'malformed_cdx: {e}')
    return {'cc_status': 'unknown' if errors else 'missing', 'cc_error': '; '.join(errors)[-500:]}


def wb_lookup(url, service):
    errors, hits = [], []
    # Wayback CDX canonicalizes URL keys; the requested exact lookup is one call.
    # Explicit transport/www variants are required for CC, not for Wayback.
    for variant in [url]:
        r = service.get('https://web.archive.org/cdx/search/cdx', params={'url': variant, 'output': 'json', 'filter': 'statuscode:200', 'limit': '-1'})
        if r.get('error'):
            errors.append(r['error'])
            if service.open:
                break
            continue
        try:
            data = json.loads(r.get('text', '[]'))
            if data:
                headers = data[0]
                for row in data[1:]:
                    record = dict(zip(headers, row))
                    if record.get('statuscode') == '200' and record.get('timestamp') and record.get('original'):
                        hits.append(record)
        except (ValueError, TypeError, KeyError) as e:
            errors.append(f'malformed_cdx: {e}')
    if hits:
        hit = max(hits, key=lambda x: x['timestamp'])
        return {'wb_status': 'found', 'wb_timestamp': hit['timestamp'], 'wb_record': hit, 'wb_latest_complete': not bool(errors)}
    return {'wb_status': 'unknown' if errors else 'missing', 'wb_error': '; '.join(errors)[-500:]}


def cc_payload(record, service):
    offset, length = int(record['offset']), int(record['length'])
    url = 'https://data.commoncrawl.org/' + record['filename']
    r = service.get(url, headers={'Range': f'bytes={offset}-{offset+length-1}', 'Accept-Encoding': 'identity'}, binary=True)
    if r.get('error'):
        raise ValueError(r['error'])
    if r['http_status'] != 206 or len(r.get('body', b'')) != length:
        raise ValueError(f'Range mismatch: HTTP {r["http_status"]}, bytes={len(r.get("body", b""))}, expected={length}')
    content_range = next((v for k, v in r.get('response_headers', {}).items() if k.lower() == 'content-range'), '')
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(?:\d+|\*)', content_range)
    if not match or (int(match[1]), int(match[2])) != (offset, offset + length - 1):
        raise ValueError(f'Content-Range mismatch: {content_range!r}')
    from warcio.archiveiterator import ArchiveIterator
    for warc in ArchiveIterator(io.BytesIO(r['body'])):
        if warc.rec_type == 'response':
            target = warc.rec_headers.get_header('WARC-Target-URI') or ''
            if not record.get('url') or normalize_url(target) != normalize_url(record['url']):
                raise ValueError(f'WARC target mismatch: {target!r}')
            if warc.http_headers and warc.http_headers.get_statuscode() != '200':
                raise ValueError('WARC embedded HTTP status is not 200')
            return warc.content_stream().read(5_000_000)
    raise ValueError('No response record in WARC Range')


def measure_content(row, source, service, tok, out):
    if row.get(source + '_status') != 'found':
        return {source + '_content_status': 'not_available', source + '_text_ok': None}
    start = time.monotonic()
    try:
        record = row[source + '_record']
        if source == 'cc':
            body = cc_payload(record, service)
        else:
            url = f'https://web.archive.org/web/{record["timestamp"]}id_/{record["original"]}'
            response = service.get(url, binary=True)
            if response.get('error') or response['http_status'] != 200:
                raise ValueError(response.get('error') or f'HTTP {response["http_status"]}')
            body = response.get('body', b'')
        text, _, _ = extract(body)
        rec = classify({'url': row['url'], 'final_url': row['url'], 'http_status': 200}, body)
        ok = len(text) >= 200 and rec['state'] == 'ok'
        path = out / 'archive_raw' / row['domain'] / f'{row["id"]}-{source}.html'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return {source + '_content_status': 'extracted', source + '_capture_timestamp': row.get(source + '_timestamp'), source + '_text_ok': ok, source + '_text_len': len(text), source + '_n_tok': len(tok(text, add_special_tokens=False)['input_ids']) if tok and text else None, source + '_payload_state': rec['state'], source + '_raw_path': str(path.resolve()), source + '_content_elapsed_s': round(time.monotonic()-start, 4)}
    except Exception as e:
        return {source + '_content_status': 'error', source + '_text_ok': None, source + '_content_error': f'{type(e).__name__}: {str(e)[:300]}', source + '_content_elapsed_s': round(time.monotonic()-start, 4)}


def content_cache_valid(row, source, saved, retry_unknown):
    if not saved:
        return False
    status = saved.get(source + '_content_status')
    if status == 'not_available' and row.get(source + '_status') == 'found':
        return False
    if saved.get(source + '_capture_timestamp') and saved[source + '_capture_timestamp'] != row.get(source + '_timestamp'):
        return False
    return not (retry_unknown and status == 'error')


def url_main():
    p = cli(__doc__, 100)
    p.add_argument('--content-n', type=int, default=20)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--run-dir', default=str(OUT))
    p.add_argument('--download-tokenizer', action='store_true')
    p.add_argument('--retry-unknown', action='store_true', help='Retry unknown index/content outcomes in existing checkpoint.')
    args = p.parse_args()
    if min(args.n, args.content_n, args.workers) < 1:
        p.error('sample sizes/workers must be positive')
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    experiment_config(out / 'archive_config.json', {'seed': SEED, 'n': args.n, 'content_n': args.content_n, 'indexes': 6, 'variants': 'http_https_www', 'timeout': 15})
    direct = defaultdict(list)
    for r in read_csv(out / 'reprobe.csv'):
        if r.get('arm') == 'chrome_cookie':
            direct[r['domain']].append(r)
    domain_filter = selected_domains(args.domains)
    eligible = [d for d, rows in direct.items() if sum(r['state'] == 'ok' for r in rows) / len(rows) < .5]
    domains = [d for d in eligible if domain_filter is None or d in domain_filter]
    if domain_filter and not domain_filter.issubset(set(eligible)):
        p.error('--domains must have measured direct success <50%: ' + ', '.join(domain_filter-set(eligible)))
    df = corpus()
    sample = samples(df, sorted(domains), args.n)
    del df
    cp = Checkpoint(out / 'checkpoints/archive.jsonl')
    api_cp = Checkpoint(out / 'checkpoints/archive_api.jsonl')
    content_cp = Checkpoint(out / 'checkpoints/archive_content.jsonl')
    pacer = Pacer()
    cc, wb, cc_data = Service('cc-index', api_cp, pacer=pacer), Service('wayback', api_cp, pacer=pacer), Service('cc-data', api_cp, pacer=pacer)
    status_path = out / 'archive_services.json'
    if status_path.exists() and not args.retry_unknown:
        prior = json.loads(status_path.read_text('utf-8'))
        for name, service in [('Common Crawl CDX', cc), ('Wayback', wb), ('Common Crawl WARC', cc_data)]:
            if prior.get(name, {}).get('circuit_open'):
                service.open = True
                service.error = prior[name].get('reason', 'previous prolonged failure; use --retry-unknown to retry')
    indexes_path = out / 'archive_indexes.json'
    collinfo = cc.get('https://index.commoncrawl.org/collinfo.json')
    indexes, index_error = [], ''
    if collinfo.get('error'):
        index_error = collinfo['error']
    else:
        try:
            indexes = sorted(json.loads(collinfo['text']), key=lambda r: r['id'], reverse=True)[:6]
            if len(indexes) < 6 or not all('cdx-api' in r for r in indexes):
                raise ValueError('Expected six valid indexes')
            indexes_path.write_text(json.dumps(indexes, indent=2), 'utf-8')
        except (ValueError, TypeError, KeyError) as e:
            index_error = f'invalid_collinfo: {e}'
    tok, token_error = load_tokenizer(args.download_tokenizer)
    groups = defaultdict(list)
    for row in sample:
        groups[row['domain']].append(row)
    updates = dict(cp.rows)
    update_lock = threading.Lock()

    def worker(domain, rows):
        experiment_config(out / f'manifests/archive-{domain}.json', rows)
        # Preselect 20 from the 100 independently of availability; avoids hit-only bias.
        content_ids = {r['id'] for r in sorted(rows, key=lambda r: hashlib.sha256(f'{SEED}-{r["id"]}'.encode()).hexdigest())[:args.content_n]}
        completed = []
        for original in rows:
            k = key(domain, original['url'], 'archive')
            row = updates.get(k)
            if row is None or (args.retry_unknown and (row.get('cc_status') == 'unknown' or row.get('wb_status') == 'unknown')):
                start = time.monotonic()
                row = {**original, 'key': k, 'content_sample': original['id'] in content_ids, 'expected_index_n': len(rows), 'expected_content_n': len(content_ids), 'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                row.update(cc_lookup(original['url'], indexes, cc) if indexes else {'cc_status': 'unknown', 'cc_error': index_error or 'no_indexes'})
                row.update(wb_lookup(original['url'], wb))
                row['index_elapsed_s'] = round(time.monotonic()-start, 4)
                cp.add(row, replace=args.retry_unknown)
            if row['content_sample']:
                for source, service in [('cc', cc_data), ('wb', wb)]:
                    content_key = key(domain, row['url'], source + '-content')
                    saved = content_cp.rows.get(content_key)
                    if content_cache_valid(row, source, saved, args.retry_unknown):
                        row.update({k: v for k, v in saved.items() if k != 'key'})
                    else:
                        measured = measure_content(row, source, service, tok, out)
                        content_cp.add({'key': content_key, **measured}, replace=args.retry_unknown)
                        row.update(measured)
            with update_lock:
                updates[k] = row
                if len(completed) % 10 == 0:
                    write_csv(out / 'archive_coverage.csv', list(updates.values()))
                    (out / 'archive_services.json').write_text(json.dumps({'Common Crawl CDX': cc.status(), 'Wayback': wb.status(), 'Common Crawl WARC': cc_data.status(), 'tokenizer': {'error': token_error}}, ensure_ascii=False, indent=2), 'utf-8')
            completed.append(row)
        return domain, len(completed), sum(r['cc_status'] == 'found' for r in completed), sum(r['wb_status'] == 'found' for r in completed)

    print(f'archive: {len(sample)} URLs / {len(domains)} domains; indexes={[r["id"] for r in indexes]}', flush=True)
    write_csv(out / 'archive_coverage.csv', list(updates.values()), ['key', 'id', 'domain', 'url', 'cc_status', 'wb_status'] if not updates else None)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, d, rows): d for d, rows in groups.items()}
        for f in as_completed(futures):
            try:
                domain, n, n_cc, n_wb = f.result()
                print(f'{domain}: CC={n_cc}/{n}; WB={n_wb}/{n}', flush=True)
            except Exception as e:
                print(f'ERROR {futures[f]}: {type(e).__name__}: {e}', flush=True)
            with update_lock:
                write_csv(out / 'archive_coverage.csv', list(updates.values()))
    (out / 'archive_services.json').write_text(json.dumps({'Common Crawl CDX': cc.status(), 'Wayback': wb.status(), 'Common Crawl WARC': cc_data.status(), 'tokenizer': {'error': token_error}}, ensure_ascii=False, indent=2), 'utf-8')
    from r2ai.probe.profile_report_v2 import build_report
    build_report(out)


PRIORITY = ['familydoctor.com.cn', 'ask.39.net', 'zysjonline.com', 'suckhoedoisong.vn', 'nhathuoclongchau.com.vn', 'thanhnien.vn', 'vinmec.com']


def archive_key(url):
    u = urlsplit(url)
    return urlunsplit(('', u.netloc.lower().removeprefix('www.'), u.path.rstrip('/'), u.query, '')).removeprefix('//')


def join_capture(url, source, lookup, complete, reason=''):
    record = lookup.get(archive_key(url))
    if record:
        return {source + '_status': 'found', source + '_timestamp': record['timestamp'], source + '_record': record, source + '_search_complete': complete}
    return {source + '_status': 'missing' if complete else 'not_observed', source + '_search_complete': complete, source + '_error': reason}


def cc_domain_index(domain, index, service, max_pages):
    params = {'url': domain + '/*', 'output': 'json', 'showNumPages': 'true', 'filter': 'status:200'}
    r = service.get(index['cdx-api'], params=params)
    if r.get('error'):
        return {}, False, {'crawl': index['id'], 'error': r['error'], 'pages_read': 0}
    if r['http_status'] == 404:
        return {}, True, {'crawl': index['id'], 'pages_total': 0, 'pages_read': 0, 'records': 0}
    try:
        pagination = json.loads(r.get('text', '{}'))
        page_count = int(pagination['pages'])
    except (ValueError, KeyError, TypeError) as e:
        return {}, False, {'crawl': index['id'], 'error': f'invalid_pagination: {e}', 'pages_read': 0}
    lookup, errors, read, records = {}, [], 0, 0
    for page in range(min(page_count, max_pages)):
        result = service.get(index['cdx-api'], params={'url': domain + '/*', 'output': 'json', 'filter': 'status:200', 'page': str(page)})
        if result.get('error'):
            errors.append(result['error'])
            if service.open or 'budget' in result['error']:
                break
            continue
        try:
            for line in result.get('text', '').splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record.get('status')) != '200' or not all(k in record for k in ('url', 'timestamp', 'filename', 'offset', 'length')):
                    continue
                records += 1
                k = archive_key(record['url'])
                if k not in lookup or record['timestamp'] > lookup[k]['timestamp']:
                    lookup[k] = {**record, 'crawl': index['id']}
            read += 1
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f'invalid_cdx: {e}')
    complete = read == page_count and not errors
    return lookup, complete, {'crawl': index['id'], 'pages_total': page_count, 'pages_read': read, 'records': records, 'search_complete': complete, 'error': '; '.join(errors)[-500:], 'limited': page_count > max_pages}


def wb_domain_index(domain, service):
    params = {'url': domain, 'matchType': 'domain', 'output': 'json', 'filter': 'statuscode:200', 'collapse': 'urlkey', 'fl': 'original,timestamp', 'limit': '50000'}
    r = service.get('https://web.archive.org/cdx/search/cdx', params=params)
    if r.get('error'):
        return {}, False, {'error': r['error'], 'records': 0, 'limit': 50000}
    try:
        rows = json.loads(r.get('text', '[]'))
        if not rows:
            return {}, True, {'records': 0, 'limit': 50000, 'search_complete': True}
        lookup = {}
        for values in rows[1:]:
            record = dict(zip(rows[0], values))
            if 'original' in record and 'timestamp' in record:
                lookup[archive_key(record['original'])] = record
        complete = len(rows) - 1 < 50000
        return lookup, complete, {'records': len(rows)-1, 'limit': 50000, 'search_complete': complete, 'limited': not complete}
    except (ValueError, KeyError, TypeError) as e:
        return {}, False, {'error': f'invalid_cdx: {e}', 'records': 0, 'limit': 50000}


def domain_main():
    p = cli('Domain-level archive indexes + local join (updated user requirements).', 50)
    p.add_argument('--content-n', type=int, default=10)
    p.add_argument('--pages', type=int, default=3)
    p.add_argument('--max-minutes', type=float, default=60)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--run-dir', default=str(OUT))
    p.add_argument('--retry-unknown', action='store_true')
    p.add_argument('--renew-budget', action='store_true', help='Explicitly start a new bounded measurement budget on resume.')
    args = p.parse_args()
    if min(args.n, args.content_n, args.pages, args.workers, args.max_minutes) <= 0:
        p.error('sizes, workers and time budget must be positive')
    start = time.monotonic()
    deadline = start + args.max_minutes * 60
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {'method': 'domain_index_local_join', 'n': args.n, 'content_n': args.content_n, 'pages_per_crawl': args.pages, 'initial_crawls': 2, 'older_if_observed_below': .3, 'max_crawls': 6, 'wayback_limit': 50000, 'seed': SEED, 'max_minutes': args.max_minutes, 'priority': PRIORITY}
    experiment_config(out / 'archive_domain_config.json', config)
    budget_path = out / 'archive_domain_budget.json'
    if budget_path.exists() and not args.renew_budget:
        budget = json.loads(budget_path.read_text('utf-8'))
    else:
        budget = {'started_epoch': time.time() if args.renew_budget else (out/'archive_domain_config.json').stat().st_mtime, 'minutes': args.max_minutes}
        budget_path.write_text(json.dumps(budget, indent=2), 'utf-8')
    deadline = time.monotonic() + max(0, budget['started_epoch'] + args.max_minutes*60 - time.time())
    # Preserve the previous exact-URL partial experiment separately.
    previous = out / 'archive_coverage_url_lookup_partial.csv'
    if (out / 'archive_config.json').exists() and (out / 'archive_coverage.csv').exists() and not previous.exists():
        previous.write_bytes((out / 'archive_coverage.csv').read_bytes())
    cp = Checkpoint(out / 'checkpoints/archive_domain.jsonl')
    api_cp = Checkpoint(out / 'checkpoints/archive_api.jsonl')
    content_cp = Checkpoint(out / 'checkpoints/archive_domain_content.jsonl')
    pacer = Pacer()
    cc = Service('cc-domain', api_cp, pacer=pacer, deadline=deadline)
    wb = Service('wb-domain', api_cp, pacer=pacer, deadline=deadline)
    data = Service('cc-data-domain', api_cp, pacer=pacer, deadline=deadline)
    replay = Service('wb-replay-domain', api_cp, pacer=pacer, deadline=deadline)
    direct = defaultdict(list)
    for r in read_csv(out / 'reprobe.csv'):
        if r.get('arm') == 'chrome_cookie':
            direct[r['domain']].append(r)
    weights = {r['domain']: int(r['n_urls']) for r in read_csv(out / 'corpus_domains_v2.csv')}
    selected = selected_domains(args.domains)
    remaining = sorted([d for d, rows in direct.items() if d not in PRIORITY and sum(r['state']=='ok' for r in rows)/len(rows)<.5], key=lambda d: -weights.get(d, 0))
    domains = [d for d in PRIORITY + remaining if d in direct and (selected is None or d in selected)]
    if selected and set(domains) != selected:
        p.error('Unknown/unselected domains: ' + ', '.join(selected-set(domains)))
    df = corpus()
    sample = samples(df, domains, args.n)
    del df
    grouped = defaultdict(list)
    for row in sample:
        grouped[row['domain']].append(row)
    info = cc.get('https://index.commoncrawl.org/collinfo.json')
    indexes, index_error = [], ''
    try:
        if info.get('error'):
            raise ValueError(info['error'])
        indexes = sorted(json.loads(info['text']), key=lambda x:x['id'], reverse=True)[:6]
        (out / 'archive_indexes.json').write_text(json.dumps(indexes, indent=2), 'utf-8')
    except (ValueError, TypeError, KeyError) as e:
        index_error = str(e)
    tok, token_error = load_tokenizer()
    updates, lock = dict(cp.rows), threading.Lock()

    def flush():
        write_csv(out / 'archive_coverage.csv', list(updates.values()), ['key', 'id', 'domain', 'url', 'cc_status', 'wb_status'] if not updates else None)
        state = {'Common Crawl CDX': cc.status(), 'Wayback': wb.status(), 'Common Crawl WARC': data.status(), 'Wayback replay': replay.status(), 'runtime': {'elapsed_s': round(time.monotonic()-start, 2), 'budget_s': args.max_minutes*60, 'budget_exhausted': time.monotonic() >= deadline}, 'tokenizer': {'error': token_error}}
        (out / 'archive_services.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), 'utf-8')

    def worker(domain):
        rows = grouped[domain]
        experiment_config(out / f'manifests/archive-domain-{domain}.json', rows)
        domain_path = out / 'archive_domain_indexes' / f'{domain}.json'
        if domain_path.exists() and not args.retry_unknown:
            summary = json.loads(domain_path.read_text('utf-8'))
            cc_lookup_map, wb_lookup_map = summary['cc_lookup'], summary['wb_lookup']
        else:
            cc_lookup_map, cc_reports = {}, []
            for index in indexes[:2]:
                lookup, complete, report = cc_domain_index(domain, index, cc, args.pages)
                cc_reports.append(report)
                # Newest crawl wins; within each crawl newest timestamp wins.
                for k, record in lookup.items():
                    cc_lookup_map.setdefault(k, record)
            observed = sum(archive_key(r['url']) in cc_lookup_map for r in rows) / len(rows)
            valid_newest = any(r.get('pages_read',0)>0 or (r.get('pages_total')==0 and not r.get('error')) for r in cc_reports)
            if observed < .3 and valid_newest:
                for index in indexes[2:]:
                    if cc.open or time.monotonic() >= deadline:
                        break
                    lookup, complete, report = cc_domain_index(domain, index, cc, args.pages)
                    cc_reports.append(report)
                    for k, record in lookup.items():
                        cc_lookup_map.setdefault(k, record)
            wb_lookup_map, wb_complete, wb_report = wb_domain_index(domain, wb)
            wanted = {archive_key(r['url']) for r in rows}
            cc_lookup_map = {k:v for k,v in cc_lookup_map.items() if k in wanted}
            wb_lookup_map = {k:v for k,v in wb_lookup_map.items() if k in wanted}
            summary = {'domain': domain, 'cc_lookup': cc_lookup_map, 'wb_lookup': wb_lookup_map, 'cc_reports': cc_reports, 'wb_report': wb_report, 'cc_complete': bool(indexes) and len(cc_reports)==6 and all(r.get('search_complete', r.get('pages_total')==0 and not r.get('error')) for r in cc_reports), 'wb_complete': wb_complete, 'cc_reason': index_error or 'partial crawl window/pages or API errors', 'elapsed_s':round(time.monotonic()-start,4), 'measured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            domain_path.parent.mkdir(parents=True, exist_ok=True)
            domain_path.write_text(json.dumps(summary, ensure_ascii=False), 'utf-8')
        joined = []
        for original in rows:
            row = {**original, 'key': key(domain, original['url'], 'archive-domain'), 'method': 'domain_index_local_join', 'expected_index_n': len(rows), 'measured_at': summary['measured_at'], 'index_metadata_path': str(domain_path.resolve())}
            row.update(join_capture(row['url'], 'cc', cc_lookup_map, summary['cc_complete'], summary.get('cc_reason', '')))
            row.update(join_capture(row['url'], 'wb', wb_lookup_map, summary['wb_complete'], summary['wb_report'].get('error', 'domain query limit')))
            row['cc_index_observed'] = any(r.get('pages_read', 0)>0 or (r.get('pages_total')==0 and not r.get('error')) for r in summary['cc_reports'])
            row['wb_index_observed'] = not bool(summary['wb_report'].get('error'))
            joined.append(row)
        # Payload trials use a reproducible sample of observed hits, ≤10/domain.
        hit_rows = [r for r in joined if r['cc_status']=='found' or r['wb_status']=='found']
        chosen = sorted(hit_rows, key=lambda r: hashlib.sha256(f'{SEED}-{r["id"]}'.encode()).hexdigest())[:args.content_n]
        selected_ids = {r['id'] for r in chosen}
        for row in joined:
            row.update(content_sample=row['id'] in selected_ids, expected_content_n=len(chosen), observed_hit_n=len(hit_rows))
            if row['content_sample']:
                for source, service in [('cc', data), ('wb', replay)]:
                    content_key = key(domain, row['url'], source+'-domain-content')
                    saved = content_cp.rows.get(content_key)
                    if content_cache_valid(row, source, saved, args.retry_unknown):
                        row.update({k:v for k,v in saved.items() if k!='key'})
                    else:
                        measured = measure_content(row, source, service, tok, out)
                        content_cp.add({'key':content_key, **measured}, replace=True)
                        row.update(measured)
            cp.add(row, replace=True)
            with lock:
                updates[row['key']] = row
        with lock:
            flush()
        return domain, sum(r['cc_status']=='found' for r in joined), sum(r['wb_status']=='found' for r in joined), len(chosen)

    print(f'archive DOMAIN: {len(sample)} URLs/{len(domains)} domains; n={args.n}, payload<={args.content_n}, budget={args.max_minutes} min; order={domains}', flush=True)
    flush()
    # Bounded dispatch preserves priority order while parallelizing two domains.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending, iterator = {}, iter(domains)
        for _ in range(args.workers):
            d = next(iterator, None)
            if d:
                pending[pool.submit(worker, d)] = d
        while pending:
            from concurrent.futures import wait, FIRST_COMPLETED
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                d = pending.pop(future)
                try:
                    result = future.result()
                    print(f'{result[0]}: CC={result[1]}/{args.n}; WB={result[2]}/{args.n}; payload URLs={result[3]}', flush=True)
                except Exception as e:
                    print(f'ERROR {d}: {type(e).__name__}: {e}', flush=True)
                next_domain = next(iterator, None)
                if next_domain:
                    pending[pool.submit(worker, next_domain)] = next_domain
    flush()
    from r2ai.probe.profile_report_v2 import build_report
    build_report(out)


def main():
    auxiliary_disabled()
    if '--mode' in sys.argv:
        pos = sys.argv.index('--mode')
        mode = sys.argv[pos+1]
        del sys.argv[pos:pos+2]
    else:
        mode = 'domain'
    if mode == 'url':
        url_main()
    elif mode == 'domain':
        domain_main()
    else:
        raise SystemExit('--mode must be domain or url')


if __name__ == '__main__':
    main()
