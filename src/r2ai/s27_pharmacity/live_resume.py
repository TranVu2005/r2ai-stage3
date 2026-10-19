"""Cooldown-gated, one-probe-at-most live continuation for S27."""
from __future__ import annotations

import json
import random
import sqlite3
from datetime import datetime, timedelta, timezone

from . import pipeline as P

NEVER_RETRY_DOC_IDS = frozenset({317850})
LIVE_MIN_INTERVAL_S = 5.0
LIVE_JITTER_S = 2.0


def _utc(value) -> datetime:
    if isinstance(value, datetime):
        stamp = value
    else:
        stamp = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)


def cooldown_deadline(stop: dict) -> datetime:
    """Three hours plus one second after the recorded halt timestamp."""
    record = stop.get('halt') if isinstance(stop.get('halt'), dict) else stop
    stopped_at = record.get('fetched_at_utc') or record.get('halted_at_utc') or record.get('stopped_at_utc')
    if not stopped_at:
        raise ValueError('halt checkpoint has no UTC timestamp')
    return _utc(stopped_at) + timedelta(hours=3, seconds=1)


def eligible_live_ids(targets, pages) -> list[int]:
    """Only CC no-capture/unusable IDs without a prior page result are live eligible."""
    page_ids = set(pages)
    result = []
    for doc_id, outcome in targets:
        doc_id = int(doc_id)
        if doc_id in NEVER_RETRY_DOC_IDS or doc_id in page_ids:
            continue
        if outcome in ('no_capture', 'unusable'):
            result.append(doc_id)
    return sorted(set(result))


def is_normal_probe_result(result: dict) -> bool:
    return (int(result.get('http_status') or 0) == 200
            and result.get('status') in ('ok', 'low_title_match')
            and bool(result.get('selector_found'))
            and int(result.get('body_chars') or 0) >= P.MIN_BODY_CP)


def _read_inputs(roots: P.Roots):
    cc_path = roots.guard(roots.root / 'cc-backfill' / 'state.sqlite')
    if not cc_path.is_file() or not roots.mapping.is_file() or not roots.state.is_file():
        raise FileNotFoundError('S27 CC scan, mapping, and crawl state are required before live resume')
    cc = sqlite3.connect(cc_path.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        targets = cc.execute('SELECT doc_id,outcome FROM targets ORDER BY doc_id').fetchall()
    finally:
        cc.close()
    state = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        pages = {int(doc_id): {'status': status, 'http_status': http_status, 'body_chars': body_chars}
                 for doc_id, status, http_status, body_chars in state.execute(
                     'SELECT doc_id,status,http_status,body_chars FROM pages')}
    finally:
        state.close()
    mapping = {int(row['doc_id']): row for row in P._load_mapping(roots)}
    counts = {name: 0 for name in ('no_capture', 'unusable', 'dead', 'usable', 'pending')}
    for _doc_id, outcome in targets:
        if outcome in counts:
            counts[outcome] += 1
    if len(targets) != 5182:
        raise ValueError(f'expected 5,182 classified Common Crawl targets, got {len(targets)}')
    eligible = eligible_live_ids(targets, pages)
    return targets, pages, mapping, counts, eligible


def live_resume_status(roots: P.Roots, *, now: datetime | None = None) -> dict:
    """Read-only preflight/status, including the required CC groups and cooldown time."""
    halt_path = roots.guard(roots.reports / 'crawl_halted.json')
    if not halt_path.is_file():
        raise FileNotFoundError('recorded S27 halt checkpoint is required')
    halt = json.loads(halt_path.read_text(encoding='utf-8'))
    deadline = cooldown_deadline(halt)
    halt_record = halt.get('halt') if isinstance(halt.get('halt'), dict) else halt
    _targets, pages, _mapping, counts, eligible = _read_inputs(roots)
    current = _utc(now or datetime.now(timezone.utc))
    return {
        'batch': P.BATCH, 'status': 'cooldown_elapsed' if current >= deadline else 'waiting_for_cooldown',
        'halted_at_utc': _utc(halt_record['fetched_at_utc']).isoformat().replace('+00:00', 'Z'),
        'probe_earliest_utc': deadline.isoformat().replace('+00:00', 'Z'), 'now_utc': current.isoformat().replace('+00:00', 'Z'),
        'cc_groups': {'a_no_capture': counts['no_capture'], 'b_captured_no_usable_text': counts['unusable'],
                      'dead_all_captures_404_or_410': counts['dead'], 'usable_text': counts['usable'],
                      'pending_fetch_error': counts['pending']},
        'eligible_live_urls': len(eligible), 'prior_page_rows_excluded': len(pages),
        'doc_id_317850_never_retry': True, 'site_requests_made': 0,
    }


def _load_report(roots: P.Roots) -> tuple[dict, object]:
    path = roots.guard(roots.reports / 'live_resume.json')
    value = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    return value, path


def _save(roots: P.Roots, path, report: dict) -> None:
    P.atomic_json(roots, path, report)


def _page_for(roots: P.Roots, doc_id: int):
    c = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        return c.execute('SELECT status,http_status,body_chars FROM pages WHERE doc_id=?', (doc_id,)).fetchone()
    finally:
        c.close()


def _handoff_pending(roots: P.Roots, *, final: bool = False) -> list[dict]:
    """Publish complete 1,000-doc increments, then the final remainder at crawl end."""
    import pyarrow.parquet as pq
    from .handoff import handoff_lot

    handoff_root = roots.guard(roots.root / 'handoff' / P.BATCH)
    manifest_path = roots.guard(handoff_root / 'manifest.json')
    handed = set()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for lot in manifest.get('lots', []):
            handed.update(int(doc_id) for doc_id in lot.get('doc_ids', []))
    c = sqlite3.connect(roots.state.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        ready = {int(row[0]) for row in c.execute(
            "SELECT doc_id FROM pages WHERE status='ok' AND http_status=200 AND body_chars>=?", (P.MIN_BODY_CP,))}
    finally:
        c.close()
    pending = ready - handed
    published = []
    while len(pending) >= 1000 or (final and pending):
        limit = min(1000, len(pending))
        result = handoff_lot(roots, limit=limit)
        if result.get('status') != 'ok' or int(result.get('docs', 0)) == 0:
            raise RuntimeError(f'failed to publish S15 handoff lot: {result}')
        result['vector_handoff'] = P.embed_handoff_lot(roots, result['lot_id'])
        published.append(result)
        for doc_id in result.get('doc_ids', []):
            handed.add(int(doc_id))
        # Older handoff return values do not echo IDs; reload the cumulative manifest.
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        handed = {int(doc_id) for lot in manifest.get('lots', []) for doc_id in lot.get('doc_ids', [])}
        pending = ready - handed
    return published


def run_live_resume(roots: P.Roots, *, now: datetime | None = None) -> dict:
    """After cooldown, make one durable probe, then crawl eligible CC-missing URLs serially."""
    roots.preflight()
    status = live_resume_status(roots, now=now)
    report, report_path = _load_report(roots)
    if status['status'] != 'cooldown_elapsed':
        return status | {'status': 'waiting_for_cooldown', 'resume_report': str(report_path)}
    if report.get('status') in ('halted_challenge', 'halted_block', 'probe_stopped', 'stopped_unknown_request', 'complete'):
        return report | {'no_new_requests': True}

    new_run = not report
    targets, pages, mapping, counts, eligible = _read_inputs(roots)
    report.setdefault('batch', P.BATCH)
    report.setdefault('started_at_utc', datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'))
    report['cc_groups'] = status['cc_groups']
    report['eligible_live_urls_at_start'] = len(eligible) + sum(
        1 for doc_id, outcome in targets if outcome in ('no_capture', 'unusable') and int(doc_id) in pages
    )
    report['probe_earliest_utc'] = status['probe_earliest_utc']
    report['shared_index_written'] = False
    report['vectors_written'] = False
    report.setdefault('request_log', str(roots.guard(roots.reports / 'live_requests.jsonl')))

    # A durable in-flight marker prevents retrying a possibly completed challenge after a crash.
    current_doc = report.get('current_doc_id')
    if report.get('request_in_flight'):
        report.update(status='stopped_unknown_request', stop_reason='request_outcome_unknown_after_interruption',
                      stopped_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'))
        _save(roots, report_path, report)
        return report | {'no_new_requests': True}
    if current_doc is not None:
        saved = _page_for(roots, int(current_doc))
        if saved is None:
            if report.get('request_started'):
                report.update(status='stopped_unknown_request', stop_reason='request_result_not_persisted_after_interruption',
                              stopped_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'))
                _save(roots, report_path, report)
                return report | {'no_new_requests': True}
            # The saved ready marker proves no HTTP request started, so this URL may be resumed safely.
            report['current_doc_id'] = None
        if saved is not None and saved[0] in ('halted_captcha', 'halted_block'):
            report.update(status='halted_challenge' if saved[0] == 'halted_captcha' else 'halted_block',
                          stop_reason=saved[0], stopped_at_utc=datetime.now(timezone.utc).isoformat(
                              timespec='seconds').replace('+00:00', 'Z'))
            _save(roots, report_path, report)
            return report | {'no_new_requests': True}
        if saved is not None:
            report['current_doc_id'] = None
            report['request_in_flight'] = False

    if report.get('status') == 'probe_started':
        probe_id = int(report['probe_doc_id'])
        saved = _page_for(roots, probe_id)
        inferred = {'http_status': saved[1], 'status': saved[0], 'body_chars': saved[2], 'selector_found': saved[2] is not None and saved[2] >= P.MIN_BODY_CP} if saved else None
        if inferred and is_normal_probe_result(inferred):
            report.update(status='crawling', probe_result=inferred, current_doc_id=None, request_in_flight=False,
                          request_started=False, probe_recovered_after_interruption=True)
            _save(roots, report_path, report)
        elif report.get('request_started'):
            report.update(status='probe_stopped', stop_reason='probe_did_not_finish_as_ordinary_article',
                          stopped_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'))
            _save(roots, report_path, report)
            return report | {'no_new_requests': True}

    if new_run:
        if not eligible:
            report.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(
                timespec='seconds').replace('+00:00', 'Z'), processed=0, reason='no_live_eligible_urls')
            _save(roots, report_path, report)
            return report
        probe_id = eligible[0]
        report.update(status='probe_started', probe_doc_id=probe_id, current_doc_id=probe_id,
                      probe_started_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                      request_in_flight=False, request_started=False, processed=0)
        _save(roots, report_path, report)
    elif report.get('status') not in ('probe_started', 'crawling', 'probe_ok'):
        report.update(status='stopped_unknown_request', stop_reason='unrecognized_saved_resume_state')
        _save(roots, report_path, report)
        return report | {'no_new_requests': True}

    import requests

    session = requests.Session()
    session.trust_env = False
    session.proxies.clear()
    session.headers.update({'User-Agent': P.USER_AGENT})
    request_log = roots.guard(roots.reports / 'live_requests.jsonl')
    request_log.parent.mkdir(parents=True, exist_ok=True)

    def observe(event):
        stamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        if event['phase'] == 'start':
            report.update(request_in_flight=True, request_count=int(report.get('request_count', 0)) + 1,
                          request_started=True,
                          last_request_start=event | {'observed_at_utc': stamp})
            _save(roots, report_path, report)
        else:
            previous = report.get('last_request_start', {})
            record = previous | {'phase': 'complete', 'observed_at_utc': stamp,
                                 'http_status': event.get('http_status'), 'error': event.get('error')}
            with open(roots.guard(request_log), 'a', encoding='utf-8', newline='\n') as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
                f.flush()
            report.update(request_in_flight=False, last_request=record)
            _save(roots, report_path, report)

    connection = P._connect(roots)
    try:
        if report.get('status') == 'probe_started':
            probe_id = int(report['probe_doc_id'])
            if _page_for(roots, probe_id) is None:
                row = mapping[probe_id]
                try:
                    result = P._fetch_target(roots, connection, row, session, max_hops=1,
                                             request_observer=observe)
                except Exception as exc:  # challenge/block outcomes are persisted by _fetch_target
                    saved = _page_for(roots, probe_id)
                    halt_status = saved[0] if saved else None
                    report.update(status=('halted_challenge' if halt_status == 'halted_captcha' else
                                          'halted_block' if halt_status == 'halted_block' else 'probe_stopped'),
                                  probe_error=f'{type(exc).__name__}: {exc}', probe_page_status=halt_status,
                                  stopped_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'))
                    _save(roots, report_path, report)
                    return report | {'no_new_requests': True}
                report['probe_result'] = result
                if not is_normal_probe_result(result):
                    report.update(status='probe_stopped', stop_reason='probe_not_http_200_normal_article',
                                  stopped_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'))
                    _save(roots, report_path, report)
                    return report | {'no_new_requests': True}
                report.update(status='crawling', probe_ok_at_utc=datetime.now(timezone.utc).isoformat(
                    timespec='seconds').replace('+00:00', 'Z'), current_doc_id=None, request_in_flight=False,
                              request_started=False)
                _save(roots, report_path, report)

        # Recompute eligible IDs so completed work from a prior invocation is skipped.
        _targets, current_pages, _mapping, _counts, remaining = _read_inputs(roots)
        probe_id = int(report['probe_doc_id'])
        remaining = [doc_id for doc_id in remaining if doc_id != probe_id]
        processed = int(report.get('processed', 0))
        for doc_id in remaining:
            if doc_id in current_pages:
                continue
            report.update(status='crawling', current_doc_id=doc_id, request_in_flight=False,
                          next_doc_id=doc_id, processed=processed)
            _save(roots, report_path, report)
            try:
                result = P._fetch_target(roots, connection, mapping[doc_id], session,
                                         min_request_interval=LIVE_MIN_INTERVAL_S, jitter_s=LIVE_JITTER_S,
                                         rng=random.SystemRandom(), request_observer=observe)
            except Exception as exc:  # stop at the first challenge/block, never request another URL
                saved = _page_for(roots, doc_id)
                page_status = saved[0] if saved else None
                state = 'halted_challenge' if page_status == 'halted_captcha' else (
                    'halted_block' if page_status == 'halted_block' else 'stopped_error')
                report.update(status=state, stop_reason=f'{type(exc).__name__}: {exc}', page_status=page_status,
                              stopped_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'))
                _save(roots, report_path, report)
                return report | {'no_new_requests': state in ('halted_challenge', 'halted_block')}
            processed += 1
            report.update(current_doc_id=None, request_in_flight=False, request_started=False, last_doc_id=doc_id,
                          last_doc_result=result, processed=processed)
            _save(roots, report_path, report)
            if result.get('status') in ('halted_captcha', 'halted_block'):
                state = 'halted_challenge' if result['status'] == 'halted_captcha' else 'halted_block'
                report.update(status=state, stopped_at_utc=datetime.now(timezone.utc).isoformat(
                    timespec='seconds').replace('+00:00', 'Z'))
                _save(roots, report_path, report)
                return report | {'no_new_requests': True}
            lots = _handoff_pending(roots)
            if lots:
                report.setdefault('handoff_lots', []).extend(lots)
                _save(roots, report_path, report)
        lots = _handoff_pending(roots, final=True)
        if lots:
            report.setdefault('handoff_lots', []).extend(lots)
        report.update(status='complete', current_doc_id=None, request_in_flight=False,
                      completed_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                      processed=processed, pending_after_crawl=len(_read_inputs(roots)[-1]),
                      shared_index_written=False, vectors_written=False)
        _save(roots, report_path, report)
        return report
    finally:
        connection.close()
        session.close()
