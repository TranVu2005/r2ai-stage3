"""Render measured data only; preserve missingness in weighted extrapolation."""
from __future__ import annotations

from r2ai.paths import LEGACY_PROFILE_DIR, auxiliary_disabled

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import math
from datetime import datetime, timezone, timedelta

from r2ai.probe.probe_common import OUT, read_csv


def table(rows, columns=None):
    if not rows:
        return '_Chưa đo._\n'
    columns = columns or list(rows[0])
    def fmt(v):
        if v is None or v == '':
            return 'chưa đo'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)] + ['| ' + ' | '.join(fmt(r.get(c)) for c in columns) + ' |' for r in rows]) + '\n'


def groups(rows, arm=None):
    g = defaultdict(list)
    for r in rows:
        if arm is None or r.get('arm') == arm:
            g[r['domain']].append(r)
    return g


def ratio(rows, field='state', value='ok'):
    return sum(r.get(field) == value for r in rows) / len(rows) if rows else None


def percentage(value):
    return f'{100 * value:.1f}%' if value is not None else 'chưa đo'


def wilson95(hits, n):
    if n == 0:
        return None, None
    z, p = 1.959963984540054, hits / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return max(0., center-half), min(1., center+half)


def local_time(value):
    try:
        return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone(timedelta(hours=7))).strftime('%Y-%m-%d %H:%M:%S +07')
    except (ValueError, AttributeError):
        return 'chưa đo'


def diagnosis(rows, old, chrome):
    if not rows:
        return 'chưa đo'
    n = len(rows)
    if sum(r['state'] == 'dead_origin' for r in rows) / n > .5:
        return 'dead (HTTP 5xx; chưa chứng minh origin chết)'
    if any(r.get('robots') in ('DISALLOWED', 'UNAVAILABLE') for r in rows):
        return 'chưa xác định: robots từ chối/không đọc được'
    if old and chrome:
        paired_old = {r['url']: r for r in old}
        paired_chrome = {r['url']: r for r in chrome}
        common = paired_old.keys() & paired_chrome.keys()
        if len(common) >= 5:
            old_block = sum(paired_old[u]['state'] in ('blocked_4xx', 'bot_challenge') for u in common)
            chrome_ok = sum(paired_chrome[u]['state'] == 'ok' for u in common)
            if old_block / len(common) > .5 and chrome_ok / len(common) >= .5:
                return 'UA-based (bằng chứng A/B)'
            if old_block / len(common) > .5 and sum(paired_chrome[u]['state'] in ('blocked_4xx', 'bot_challenge') for u in common) / len(common) > .5:
                return 'IP-or-geo-based (nghi vấn; một IP chưa tách được IP/geo/WAF)'
    if sum(r.get('http_status') == '429' for r in rows) / n > .5:
        return 'rate-based (HTTP 429; quan sát tại ≤1 req/s)'
    return 'chưa xác định' if ratio(rows) < .5 else 'không chặn đa số mẫu'


def archive_measure(rows, source):
    if rows and rows[0].get('method') == 'domain_index_local_join':
        hits = sum(r.get(source+'_status')=='found' for r in rows)
        # If the API produced no valid result, do not label the error as 0%.
        has_observation = any(r.get(source+'_index_observed') in ('True','true',True) or r.get(source+'_status') in ('found','missing') or r.get(source+'_search_complete') in ('True','true',True) for r in rows)
        if not has_observation and all(r.get(source+'_error') for r in rows):
            return 'chưa đo (API/trang giới hạn)'
        lo, hi = wilson95(hits, len(rows))
        return f'quan sát {hits}/{len(rows)}={100*hits/len(rows):.1f}% [CI95% {100*lo:.1f}–{100*hi:.1f}]; chưa xác định={sum(r.get(source+"_status")=="not_observed" for r in rows)}'
    known = [r for r in rows if r.get(source + '_status') in ('found', 'missing')]
    hits = sum(r.get(source + '_status') == 'found' for r in known)
    return f'{hits}/{len(known)} ({100 * hits / len(known):.1f}%; ?={len(rows)-len(known)})' if known else 'chưa đo' + (f' (?={len(rows)})' if rows else '')


def usable_archive(rows):
    if rows and rows[0].get('method') == 'domain_index_local_join':
        good, lost, unknown = domain_archive_outcomes(rows)
        return good if unknown < 1e-12 else None
    expected_index = max((int(r.get('expected_index_n') or 100) for r in rows), default=100)
    expected_content = max((int(r.get('expected_content_n') or min(20, expected_index)) for r in rows), default=20)
    sampled = [r for r in rows if r.get('content_sample') in ('True', 'true', True)]
    if len(rows) < expected_index or len(sampled) < expected_content:
        return None
    known = []
    for row in sampled:
        cc = row.get('cc_text_ok') in ('True', 'true', True)
        wb = row.get('wb_text_ok') in ('True', 'true', True)
        cc_negative = row.get('cc_status') == 'missing' or row.get('cc_content_status') == 'extracted'
        wb_negative = row.get('wb_status') == 'missing' or row.get('wb_content_status') == 'extracted'
        if cc or wb or (cc_negative and wb_negative):
            known.append(bool(cc or wb))
    # Only extrapolate a completed content subsample; API errors leave unknowns.
    return sum(known) / len(known) if sampled and len(known) == len(sampled) else None


def domain_archive_outcomes(rows):
    n = len(rows)
    expected = max((int(r.get('expected_index_n') or 50) for r in rows), default=50)
    if n < expected:
        return 0., 0., 1.
    missing = sum(r.get('cc_status')=='missing' and r.get('wb_status')=='missing' for r in rows) / n
    hits = [r for r in rows if r.get('cc_status')=='found' or r.get('wb_status')=='found']
    sampled = [r for r in rows if r.get('content_sample') in ('True','true',True)]
    content_expected = max((int(r.get('expected_content_n') or 0) for r in rows), default=0)
    if len(sampled) < content_expected or not sampled:
        return 0., missing, 1.-missing
    good, bad = 0, 0
    for row in sampled:
        if any(row.get(s+'_text_ok') in ('True','true',True) for s in ('cc','wb')):
            good += 1
        elif all(row.get(s+'_status')=='missing' or (row.get(s+'_status')=='found' and row.get(s+'_content_status')=='extracted') for s in ('cc','wb')):
            bad += 1
    observed_hit_fraction = len(hits) / n
    success = observed_hit_fraction * good / len(sampled)
    lost = missing + observed_hit_fraction * bad / len(sampled)
    return success, lost, max(0., 1.-success-lost)


def archive_outcomes(rows):
    if rows and rows[0].get('method')=='domain_index_local_join':
        return domain_archive_outcomes(rows)
    q = usable_archive(rows)
    return (q, 1.-q, 0.) if q is not None else (0., 0., 1.)


def payload_throughput(rows, source):
    attempted = [r for r in rows if r.get(source+'_content_status') in ('extracted', 'error')]
    seconds = sum(float(r.get(source+'_content_elapsed_s') or 0) for r in attempted)
    return (len(attempted) / seconds if seconds else None), len(attempted)


def weighted_coverage(stats, direct, archive, domain_languages, language=None):
    selected = [s for s in stats if language is None or domain_languages.get(s['domain'], 'unknown') == language]
    total = sum(int(s['n_urls']) for s in selected)
    counts = {'direct': 0., 'archive': 0., 'lost': 0., 'unknown': 0.}
    for s in selected:
        weight = int(s['n_urls'])
        rows = direct.get(s['domain'], [])
        expected = max((int(r.get('expected_n') or 30) for r in rows), default=30)
        if len(rows) < expected:
            counts['unknown'] += weight
            continue
        p = ratio(rows)
        if p is None:
            counts['unknown'] += weight
            continue
        counts['direct'] += weight * p
        remaining = weight * (1 - p)
        if remaining == 0:
            continue
        recovered, lost, unknown = archive_outcomes(archive.get(s['domain'], []))
        counts['archive'] += remaining * recovered
        counts['lost'] += remaining * lost
        counts['unknown'] += remaining * unknown
    return {'nhóm': language or 'toàn corpus', 'n URL': f'{total:,}', **{k: percentage(v / total) if total else 'chưa đo' for k, v in counts.items()}}


def build_report(out: Path = OUT):
    offline = read_csv(out / 'reclassified.csv')
    probe = read_csv(out / 'reprobe.csv')
    archived = read_csv(out / 'archive_coverage.csv')
    stats = read_csv(out / 'corpus_domains_v2.csv') or [{'domain': r['domain'], 'n_urls': r['n_urls'], 'weight_pct': r.get('pct'), 'n_unique': None} for r in read_csv(OUT / 'domains.csv')]
    stats.sort(key=lambda s: -int(s['n_urls']))
    total = sum(int(s['n_urls']) for s in stats)
    lang_source = read_csv(LEGACY_PROFILE_DIR / 'domains.csv')
    languages = {r['domain']: r['lang_guess'].rstrip('?') for r in lang_source}
    direct, old, chrome, archive = groups(probe, 'chrome_cookie'), groups(probe, 'ab_old'), groups(probe, 'ab_chrome'), groups(archived)
    environment = json.loads((out / 'egress.json').read_text('utf-8')) if (out / 'egress.json').exists() else {}
    L = ['# Profile v2 – bước 2–4', '', 'Seed=42. Nguồn: CSV/checkpoint của từng bước; ngưỡng thành công = trạng thái `ok`, text ≥200 ký tự. Tỷ trọng dùng số dòng corpus gốc; thời gian dùng số URL duy nhất sau bỏ http/https/www, giữ nguyên path/query.', '', f'Egress: **{environment.get("ip") or "chưa đo"} / {environment.get("country") or "chưa đo"}**, thời điểm {local_time(environment.get("measured_at"))}. Tokenizer BAAI/bge-m3: {"đã nạp" if environment and not environment.get("tokenizer_error") else "chưa đo: " + (environment.get("tokenizer_error") or "chưa chạy")}.', '', '## 1. Phân loại offline', '', f'{len(offline)} URL; **other={sum(r["state"] == "other" for r in offline)}**; tất cả URL có đúng một state. Dữ liệu cũ không lưu `server`, `set-cookie` và từng redirect hop: **không có dữ liệu**, không request bổ sung để tái dựng.', '']
    transitions = Counter((r.get('old_state'), r['state']) for r in offline)
    L.append(table([{'cũ': a, 'mới': b, 'n': n} for (a, b), n in sorted(transitions.items())]))
    domain_rows, time_rows, strategy_times = [], [], defaultdict(list)
    for s in stats:
        d, rs = s['domain'], direct.get(s['domain'], [])
        if d not in {x['domain'] for x in stats[:35]} and not rs and not archive.get(d):
            continue
        success, q = ratio(rs), usable_archive(archive.get(d, []))
        solved = sum(r.get('cookie_solved') in ('True', 'true', True) for r in rs)
        if success is None:
            strategy = 'chưa chọn: chưa đo'
        elif success >= .5:
            strategy = 'requests+cookie' if solved else 'requests'
        elif archive.get(d) and any(r.get('cc_text_ok') in ('True', 'true', True) for r in archive[d]):
            strategy = 'commoncrawl (mẫu trích được)'
        elif archive.get(d) and any(r.get('wb_text_ok') in ('True', 'true', True) for r in archive[d]):
            strategy = 'wayback (mẫu trích được)'
        elif ratio(rs, 'state', 'needs_js_real') >= .5:
            strategy = 'playwright (đề xuất; chưa đo render)'
        elif q == 0:
            strategy = 'bỏ (không lấy được trong mẫu)'
        else:
            strategy = 'chưa chọn: lưu trữ chưa đo đủ'
        attempted = [r for r in rs if int(r.get('request_count') or 0) > 0]
        seconds = sum(float(r.get('worker_elapsed_s') or r.get('elapsed_s') or 0) for r in attempted)
        throughput = len(attempted) / seconds if seconds else None
        baseline = ratio(rs, 'initial_state', 'ok')
        domain_rows.append({'domain': d, 'corpus %': f'{100 * int(s["n_urls"]) / total:.3f}', 'n': len(rs), 'UA cũ A/B': percentage(ratio(old.get(d, []))), 'Chrome A/B': percentage(ratio(chrome.get(d, []))), 'Chrome trước cookie': percentage(baseline), 'sau cookie': percentage(success), 'nguyên nhân': diagnosis(rs, old.get(d, []), chrome.get(d, [])), '% CC quan sát (CI95%)': archive_measure(archive.get(d, []), 'cc'), '% Wayback quan sát (CI95%)': archive_measure(archive.get(d, []), 'wb'), 'chiến lược': strategy, 'URL/s': f'{throughput:.3f}' if throughput else None})
        unique = int(s['n_unique']) if s.get('n_unique') else None
        source = 'cc' if strategy.startswith('commoncrawl') else 'wb' if strategy.startswith('wayback') else None
        payload_rate, payload_n = payload_throughput(archive.get(d, []), source) if source else (None, 0)
        archive_hours = unique / payload_rate / 3600 if unique and payload_rate else None
        direct_hours = unique / throughput / 3600 if unique and throughput else None
        time_rows.append({'domain': d, 'chiến lược': strategy, 'doc duy nhất': f'{unique:,}' if unique else None, 'giờ HTTP toàn domain': f'{direct_hours:.2f}' if direct_hours else None, 'payload URL/s (n)': f'{payload_rate:.3f} ({payload_n})' if payload_rate else '—', 'giờ payload toàn domain': f'{archive_hours:.2f}' if archive_hours else 'chưa đo' if source else '—'})
        selected_hours = archive_hours if source else direct_hours if strategy.startswith('requests') else None
        strategy_times[strategy.split(' (')[0]].append((unique, selected_hours))
    truncated = sum(r.get('body_completeness') == 'no_closing_html' for r in offline)
    L += [f'HTML lịch sử không có thẻ đóng `</html>`: {truncated} URL. Đây là dấu hiệu nội dung có thể chưa đầy đủ, không phải kết luận chắc chắn; kiểm tra file thanhnien.vn thấy body dừng giữa CSS trước `<body>`. Probe cũ dùng một lần `aiohttp.content.read(MAX_BYTES)` có thể trả phần đang sẵn có; fetcher mới đọc stream đến EOF/giới hạn.', '', '## 2. Theo domain', '', f'Probe Chrome: {sum(len(r) for r in direct.values())} URL / {len(direct)} domain. A/B dùng cùng URL, cùng vòng đời session, tối đa 10 URL/biến thể ở domain blocked >50% trong mẫu cũ hoặc mới. “Chrome trước cookie” tính cả challenge trên robots; A/B không giải challenge body URL (robots được đọc với cùng chính sách cookie). Success UA cũ chỉ có ở domain đã chạy A/B.', '', table(domain_rows), 'Chẩn đoán IP/geo là **nghi vấn** từ một egress; A/B không thể loại trừ WAF theo fingerprint/TLS, quy tắc site hoặc phân biệt IP với địa lý. 5xx không đủ chứng minh origin chết vĩnh viễn.', '']
    robots = Counter((r.get('robots', ''), r.get('crawl_delay', '')) for r in probe)
    L += ['Robots/crawl-delay (số quan sát, gồm A/B):', '', table([{'robots': r, 'crawl-delay s': delay or 'không khai báo', 'n': n} for (r, delay), n in sorted(robots.items())]), 'Cookie theo phiên: ' + '; '.join(f'{d}: {dict(Counter(r.get("cookie_stability", "") for r in rows if r.get("cookie_attempted") in ("True", "true", True)))}' for d, rows in direct.items() if any(r.get('cookie_attempted') in ('True', 'true', True) for r in rows)) + '.', '', '## 3. Độ phủ theo tỷ trọng corpus', '', table([weighted_coverage(stats, direct, archive, languages, lg) for lg in (None, 'vi', 'zh')]), 'Các phần `direct/archive/lost/unknown` cộng thành 100% trước làm tròn. Archive bổ sung = w × (1−p_direct) × tỷ lệ union capture quan sát × tỷ lệ trích được trong subsample capture hoàn tất. Giả định archive không phụ thuộc success trực tiếp trong cùng domain; đây là ngoại suy mẫu, chưa phải tỷ lệ thực tế đã crawl toàn corpus. API/page/crawl bị giới hạn và payload chưa xác định giữ unknown. `lost` chỉ là không thu được trong cửa sổ archive đã truy vấn đầy đủ, không chứng minh URL không tồn tại ở mọi kho/mọi năm. Archive đo 7 domain ưu tiên và các domain còn success<50%; các domain khác chưa đo archive.', '', '## 4. Lưu trữ và token', '']
    cookie_diagnostics = read_csv(out / 'cookie_diagnostic.csv')
    saved_cookies = [r for r in offline if r.get('cookie_value_hash')]
    if saved_cookies:
        L.insert(L.index('## 3. Độ phủ theo tỷ trọng corpus'), f'Cookie offline: {len(saved_cookies)} challenge, {len(set(r["cookie_value_hash"] for r in saved_cookies))} giá trị hash khác nhau. Các challenge cùng phiên probe cũ có giá trị giống nhau; chưa đủ phân biệt cố định theo IP hay phiên. Online có {sum(r.get("cookie_solved") in ("True","true",True) for r in direct.get("laodong.vn",[]))} URL laodong.vn dùng lại cookie thành công.\n')
    if cookie_diagnostics:
        cookie_rows = []
        for r in cookie_diagnostics:
            observations = json.loads(r['observations'])
            cookie_rows.append({'domain': r['domain'], 'kết luận': 'chưa đủ quan sát' if r['verdict']=='insufficient_observations' else r['verdict'], 'URL thành công với cookie tái dùng': r['successful_urls_reusing_cookie'], 'sau xóa cookie, cùng URL/phiên': f'{sum(o["state"]=="ok" for o in observations)}/{len(observations)} ok; literal challenge={sum(bool(o.get("cookie_hashes")) for o in observations)}'})
        L.insert(L.index('## 3. Độ phủ theo tỷ trọng corpus'), table(cookie_rows))
    archive_rows = []
    for d, rows in archive.items():
        for service in ('cc', 'wb'):
            content = [r for r in rows if r.get('content_sample') in ('True', 'true', True)]
            extracted = [r for r in content if r.get(service + '_content_status') == 'extracted']
            attempted = [r for r in content if r.get(service+'_content_status') in ('extracted','error')]
            errors = sum(r.get(service+'_content_status')=='error' for r in content)
            tokens = [int(r[service + '_n_tok']) for r in extracted if r.get(service + '_n_tok') not in (None, '')]
            direct_tokens = [int(r['n_tok']) for r in direct.get(d, []) if r.get('n_tok') not in (None, '') and r['state'] == 'ok']
            years = Counter(r.get(service + '_timestamp', '')[:4] for r in rows if r.get(service + '_status') == 'found')
            successes = sum(r.get(service+'_text_ok') in ('True','true',True) for r in extracted)
            payload_result = f'{successes}/{len(attempted)}={100*successes/len(attempted):.1f}%; errors={errors}' if attempted else 'chưa đo: không có capture trong subsample'
            archive_rows.append({'domain': d, 'nguồn': service, 'capture / n index': f'{sum(r.get(service+"_status")=="found" for r in rows)}/{len(rows)}', 'text≥200 / thử payload nguồn': payload_result, 'năm (n)': json.dumps(dict(years), ensure_ascii=False), 'token archive median (n)': f'{statistics.median(tokens):.0f} ({len(tokens)})' if tokens else None, 'token ok cùng domain median (n)': f'{statistics.median(direct_tokens):.0f} ({len(direct_tokens)})' if direct_tokens else None})
    L += [table(archive_rows), 'Phương án mới: CDX theo domain, local join 50 URL; 2 crawl CC mới nhất, thêm crawl cũ nếu tỷ lệ quan sát <30%, tối đa 3 page/crawl. Wayback tối đa 50.000 URL collapse; thử nội dung tối đa 10 URL có capture/domain. CI95% Wilson mô tả **khả năng tìm thấy dưới chính giao thức truy vấn giới hạn này**, không phải CI của toàn bộ kho lưu trữ. Tỷ lệ quan sát là cận dưới; `not_observed` khi chưa đọc đủ page/crawl/limit vẫn unknown, các URL có thể tồn tại ở phần chưa đọc. Bản Wayback collapse là bản API trả về, không bảo đảm mới nhất. Index hit không đồng nghĩa trích được text.', '']
    index_rows = []
    for d in archive:
        path = out / 'archive_domain_indexes' / (d+'.json')
        if not path.exists():
            continue
        meta = json.loads(path.read_text('utf-8'))
        cc_reports, wb_report = meta.get('cc_reports', []), meta.get('wb_report', {})
        errors = [r.get('error') for r in cc_reports if r.get('error')]
        limitations = list(dict.fromkeys(errors))
        if wb_report.get('limited'):
            limitations.append('Wayback chạm limit 50000')
        if wb_report.get('error'):
            limitations.append('Wayback: '+wb_report['error'])
        index_rows.append({'domain': d, 'CC crawl': len(cc_reports), 'CC page đọc đủ / khai báo': f'{sum(r.get("pages_read",0) for r in cc_reports)}/{sum(r.get("pages_total") or 0 for r in cc_reports)}', 'crawl CC lỗi': len(errors), 'Wayback URL đọc': wb_report.get('records'), 'Wayback đủ': wb_report.get('search_complete', False), 'phần chưa đo': '; '.join(limitations)[:400] or '—'})
    L += ['Phạm vi index đã đọc (page lỗi có thể vẫn có một phần record hợp lệ):', '', table(index_rows), '']
    budget_path, services_path = out / 'archive_domain_budget.json', out / 'archive_services.json'
    if budget_path.exists() and services_path.exists():
        budget = json.loads(budget_path.read_text('utf-8'))
        elapsed_minutes = (services_path.stat().st_mtime-budget['started_epoch']) / 60
        L += [f'Ngân sách archive: {budget["minutes"]:.0f} phút; khoảng đo theo domain gồm resume: **{elapsed_minutes:.1f} phút**. Đã join {len(archived)} URL/{len(archive)} domain; thử payload {sum(r.get("content_sample") in ("True","true",True) for r in archived)} URL duy nhất. Số thử thấp hơn 10 khi không có đủ capture; phần index lỗi/circuit giữ checkpoint để đo tiếp.', '']
    if (out / 'archive_services.json').exists():
        services = json.loads((out / 'archive_services.json').read_text('utf-8'))
        L += ['Trạng thái API ở lần chạy cuối (không cộng request từ các lần trước/cache):', '', table([{'dịch vụ': k, **v} for k, v in services.items() if 'requests' in v]), '']
    L += ['## 5. Thời gian crawl', '', table(time_rows), 'Giờ HTTP = doc duy nhất / throughput worker / 3600, gồm pacing, robots, retry, cookie và extraction. Giờ payload dùng throughput thử nội dung từng nguồn, gồm pacing/retry/extraction nhưng **chưa gồm tìm index**; ngoại suy giả định mọi URL có capture, nên không phải thời gian thu hồi thực tế toàn domain. Chưa đo chi phí index toàn corpus hoặc Playwright. Các giờ là tổng thời gian worker, không phải wall-clock khi các domain chạy song song.', '', table([{'chiến lược chọn': strategy, 'domain': len(values), 'domain có giờ đo để ngoại suy': sum(hours is not None for _,hours in values), 'tổng giờ trên phần có số đo': f'{sum(hours for _,hours in values if hours is not None):.2f}' if any(hours is not None for _,hours in values) else None} for strategy, values in strategy_times.items()]), '', '## Khuyến nghị', '']
    recs = [f'Giữ phân loại độc quyền: {len(offline)} URL, other={sum(r["state"] == "other" for r in offline)}; chuyển {sum(r["state"] == "dead_origin" for r in offline)} phản hồi sang dead_origin.', f'Giải cookie literal trước SPA: offline có {sum(r["state"] == "cookie_challenge" for r in offline)} cookie_challenge; online giải thành công {sum(r.get("cookie_solved") in ("True", "true", True) for r in probe)} URL.', f'Ưu tiên mẫu top-35: phủ {100 * sum(int(s["n_urls"]) for s in stats[:35]) / total:.3f}% số dòng corpus; đã probe {len(direct)}/35 domain.', f'Giữ seed=42 và checkpoint: {len(probe)} quan sát trực tiếp/A-B; {len(archived)} URL archive hiện có.', f'Không kết luận UA/IP/geo nếu thiếu A/B: hiện có {sum(len(r) for r in old.values())} URL UA cũ và {sum(len(r) for r in chrome.values())} URL Chrome đối chứng.', f'Không thay tỷ lệ archive lỗi bằng 0: {sum(r.get("cc_status") in ("unknown","not_observed") or r.get("wb_status") in ("unknown","not_observed") for r in archived)} URL có ít nhất một nguồn chưa xác định.']
    L += ['- ' + r for r in recs[:7]]
    path = out / 'profile_report_v2.md'
    path.write_text('\n'.join(L) + '\n', 'utf-8')
    return path


if __name__ == '__main__':
    auxiliary_disabled()
    print(build_report())
