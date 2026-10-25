from datetime import datetime, timezone
import json
from pathlib import Path

from r2ai.s27_pharmacity.live_resume import cooldown_deadline, is_normal_probe_result, eligible_live_ids
from r2ai.s27_pharmacity.pipeline import _request_delay


def test_cooldown_deadline_is_three_hours_plus_one_second_after_halt():
    stop = {'fetched_at_utc': '2026-10-19T12:55:04Z'}
    assert cooldown_deadline(stop) == datetime(2026, 10, 19, 15, 55, 5, tzinfo=timezone.utc)


def test_live_pool_includes_only_cc_no_capture_or_unusable_and_never_317850():
    targets = [(1, 'no_capture'), (2, 'unusable'), (3, 'dead'), (4, 'usable'), (317850, 'no_capture')]
    pages = {4: 'ok'}
    assert eligible_live_ids(targets, pages) == [1, 2]


def test_probe_requires_ordinary_http_200_article_result():
    article = {'http_status': 200, 'status': 'ok', 'selector_found': True, 'body_chars': 200}
    assert is_normal_probe_result(article)
    assert not is_normal_probe_result(article | {'status': 'low_title_match'})
    assert not is_normal_probe_result({'http_status': 403, 'status': 'halted_block'})
    assert not is_normal_probe_result(article | {'status': 'soft404_or_nonarticle'})
    assert not is_normal_probe_result(article | {'http_status': 302, 'status': 'low_title_match'})
    assert not is_normal_probe_result(article | {'body_chars': 199})
    assert not is_normal_probe_result({'http_status': 200, 'status': 'ok', 'body_chars': 500})


def test_live_request_delay_enforces_floor_and_adds_jitter():
    assert _request_delay(elapsed=3.0, minimum=5.0, jitter_s=1.0, jitter_value=0.75) == 2.75
    assert _request_delay(elapsed=8.0, minimum=5.0, jitter_s=1.0, jitter_value=0.75) == 0.0


class FakeRoots:
    def __init__(self, root):
        self.root = Path(root)
        self.reports = self.root / 'reports'

    def guard(self, path):
        return Path(path)

    def preflight(self):
        return None


class DummyConnection:
    def close(self):
        pass


def _stub_live_inputs(monkeypatch, tmp_path, *, fetch_result=None, fetch_error=None):
    import r2ai.s27_pharmacity.live_resume as live
    import r2ai.s27_pharmacity.pipeline as pipeline

    roots = FakeRoots(tmp_path)
    mapping = {1: {'doc_id': 1, 'original_url': 'https://pmc-ecm-healthblog.beta.pharmacity.io/x',
                   'mapped_url': 'https://www.pharmacity.vn/x.htm', 'slug': 'x'}}
    reads = 0

    def inputs(_roots):
        nonlocal reads
        reads += 1
        pages = {} if reads == 1 else {1: {'status': 'ok', 'http_status': 200, 'body_chars': 200}}
        return [(1, 'no_capture')], pages, mapping, {'no_capture': 1, 'unusable': 0, 'dead': 0, 'usable': 0, 'pending': 0}, ([] if pages else [1])

    monkeypatch.setattr(live, '_read_inputs', inputs)
    monkeypatch.setattr(live, '_page_for', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live, 'live_resume_status', lambda *_args, **_kwargs: {
        'status': 'cooldown_elapsed', 'cc_groups': {'a_no_capture': 1}, 'probe_earliest_utc': '2026-10-19T15:55:05Z'})
    monkeypatch.setattr(pipeline, '_connect', lambda _roots: DummyConnection())
    monkeypatch.setattr(live, '_handoff_pending', lambda *_args, **_kwargs: [])
    calls = []

    def fetch(*args, **kwargs):
        calls.append(kwargs)
        if fetch_error:
            raise fetch_error
        return fetch_result

    monkeypatch.setattr(pipeline, '_fetch_target', fetch)
    return roots, live, calls


def test_resume_does_not_probe_before_cooldown(monkeypatch, tmp_path):
    roots, live, calls = _stub_live_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(live, 'live_resume_status', lambda *_args, **_kwargs: {
        'status': 'waiting_for_cooldown', 'probe_earliest_utc': '2026-10-19T15:55:05Z'})
    result = live.run_live_resume(roots)
    assert result['status'] == 'waiting_for_cooldown'
    assert calls == []


def test_resume_probe_is_single_request_and_one_hop(monkeypatch, tmp_path):
    result = {'http_status': 200, 'status': 'ok', 'selector_found': True, 'body_chars': 200}
    roots, live, calls = _stub_live_inputs(monkeypatch, tmp_path, fetch_result=result)
    summary = live.run_live_resume(roots)
    assert summary['status'] == 'complete'
    assert len(calls) == 1
    assert calls[0]['max_hops'] == 1


def test_resume_retries_saved_probe_only_if_request_never_started(monkeypatch, tmp_path):
    result = {'http_status': 200, 'status': 'ok', 'selector_found': True, 'body_chars': 200}
    roots, live, calls = _stub_live_inputs(monkeypatch, tmp_path, fetch_result=result)
    roots.reports.mkdir(parents=True)
    (roots.reports / 'live_resume.json').write_text(json.dumps({
        'status': 'probe_started', 'probe_doc_id': 1, 'current_doc_id': 1,
        'request_started': False, 'request_in_flight': False,
    }), encoding='utf-8')
    summary = live.run_live_resume(roots)
    assert summary['status'] == 'complete'
    assert len(calls) == 1
    assert calls[0]['max_hops'] == 1


def test_challenge_probe_is_persisted_and_never_retried(monkeypatch, tmp_path):
    roots, live, calls = _stub_live_inputs(monkeypatch, tmp_path, fetch_error=RuntimeError('challenge'))
    reads = 0

    def page(_roots, doc_id):
        nonlocal reads
        reads += 1
        return None if reads == 1 else ('halted_captcha', 200, 0)

    monkeypatch.setattr(live, '_page_for', page)
    first = live.run_live_resume(roots)
    assert first['status'] == 'halted_challenge'
    second = live.run_live_resume(roots)
    assert second['status'] == 'halted_challenge'
    assert len(calls) == 1


def test_authorized_retry_once_obeys_24h_and_persists_single_use_claim(monkeypatch, tmp_path):
    result = {'http_status': 200, 'status': 'ok', 'selector_found': True, 'body_chars': 200}
    roots, live, calls = _stub_live_inputs(monkeypatch, tmp_path, fetch_result=result)
    roots.reports.mkdir(parents=True)
    report = {
        'status': 'halted_challenge', 'current_doc_id': 318325,
        'page_status': 'halted_captcha', 'stopped_at_utc': '2026-10-19T16:43:35Z',
        'request_in_flight': False, 'request_started': True,
    }
    (roots.reports / 'live_resume.json').write_text(json.dumps(report), encoding='utf-8')

    early = live.run_live_resume(roots, now=datetime(2026, 10, 20, 16, 43, 34, tzinfo=timezone.utc),
                                 authorized_retry_once=True)
    assert early['status'] == 'waiting_for_retry_cooldown'
    assert calls == []
    assert not (roots.reports / 'authorized_retry_once.json').exists()

    retried = live.run_live_resume(roots, now=datetime(2026, 10, 20, 16, 43, 36, tzinfo=timezone.utc),
                                   authorized_retry_once=True)
    assert retried['status'] == 'complete'
    assert retried['authorized_retry_once']['probe_doc_id'] == 1
    assert retried['processed'] == 1
    assert len(calls) == 1
    assert calls[0]['min_request_interval'] == 15.0
    assert calls[0]['jitter_s'] == 2.0
    assert (roots.reports / 'authorized_retry_once.json').exists()

    again = live.run_live_resume(roots, now=datetime(2026, 10, 20, 16, 44, tzinfo=timezone.utc),
                                 authorized_retry_once=True)
    assert again['no_new_requests'] is True
    assert len(calls) == 1


def test_saved_challenge_stays_permanently_halted_without_new_authorization(monkeypatch, tmp_path):
    result = {'http_status': 200, 'status': 'ok', 'selector_found': True, 'body_chars': 200}
    roots, live, calls = _stub_live_inputs(monkeypatch, tmp_path, fetch_result=result)
    roots.reports.mkdir(parents=True)
    saved = {
        'status': 'halted_challenge', 'current_doc_id': 318325,
        'page_status': 'halted_captcha', 'stopped_at_utc': '2026-10-19T16:43:35Z',
        'request_in_flight': False, 'request_started': True,
    }
    (roots.reports / 'live_resume.json').write_text(json.dumps(saved), encoding='utf-8')

    stopped = live.run_live_resume(roots, now=datetime(2026, 10, 20, 17, tzinfo=timezone.utc))
    assert stopped['status'] == 'halted_challenge'
    assert stopped['no_new_requests'] is True
    assert calls == []
    assert not (roots.reports / 'authorized_retry_once.json').exists()


def test_live_challenge_flushes_sub_1000_handoff_remainder(monkeypatch, tmp_path):
    roots, live, calls = _stub_live_inputs(monkeypatch, tmp_path)
    mapping = {1: {'doc_id': 1, 'original_url': 'https://pmc-ecm-healthblog.beta.pharmacity.io/a',
                   'mapped_url': 'https://www.pharmacity.vn/a.htm', 'slug': 'a'},
               2: {'doc_id': 2, 'original_url': 'https://pmc-ecm-healthblog.beta.pharmacity.io/b',
                   'mapped_url': 'https://www.pharmacity.vn/b.htm', 'slug': 'b'}}
    reads = 0

    def inputs(_roots):
        nonlocal reads
        reads += 1
        pages = {} if reads == 1 else {1: {'status': 'ok', 'http_status': 200, 'body_chars': 200}}
        eligible = [1, 2] if not pages else [2]
        return [(1, 'no_capture'), (2, 'no_capture')], pages, mapping, {'no_capture': 2}, eligible

    import r2ai.s27_pharmacity.pipeline as pipeline
    monkeypatch.setattr(live, '_read_inputs', inputs)
    fetch_results = [
        {'http_status': 200, 'status': 'ok', 'selector_found': True, 'body_chars': 200},
        RuntimeError('challenge'),
    ]

    def fetch(*_args, **_kwargs):
        result = fetch_results[len(calls)]
        if isinstance(result, Exception):
            raise result
        calls.append(True)
        return result

    monkeypatch.setattr(pipeline, '_fetch_target', fetch)
    page_reads = 0

    def page(_roots, doc_id):
        nonlocal page_reads
        page_reads += 1
        return None if page_reads == 1 else ('halted_captcha', 200, 0)

    final_flags = []
    monkeypatch.setattr(live, '_page_for', page)
    monkeypatch.setattr(live, '_handoff_pending', lambda *_args, final=False: final_flags.append(final) or [])
    result = live.run_live_resume(roots)
    assert result['status'] == 'halted_challenge'
    assert len(calls) == 1
    assert final_flags == [True]
