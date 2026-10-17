import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from r2ai.zh_v2 import healthcheck


TZ = ZoneInfo('Asia/Bangkok')


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}


class _Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def head(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def health_paths(tmp_path, monkeypatch):
    state = tmp_path / 'healthcheck.json'
    events = tmp_path / 'healthcheck.jsonl'
    monkeypatch.setattr(healthcheck.direct, 'guard', lambda path: Path(path))
    return state, events


def test_200_resumes_only_120ask_and_marks_monitor_complete(health_paths):
    state_path, events_path = health_paths
    client = _Client(_Response(200))
    launched = []

    result = healthcheck.run_once(
        now=datetime(2026, 10, 17, 12, tzinfo=TZ).timestamp(),
        client=client,
        launcher=lambda **kwargs: launched.append(kwargs) or {'pid': 42},
        state_path=state_path,
        events_path=events_path,
    )

    assert result['action'] == 'resumed'
    assert result['notify_vu'] is True
    assert launched == [{'hosts': ('120ask.com',)}]
    saved = json.loads(state_path.read_text('utf-8'))
    assert saved['completed'] is True
    assert saved['launch_info'] == {'pid': 42}


def test_521_records_origin_down_and_keeps_queue_unlaunched(health_paths):
    state_path, events_path = health_paths
    client = _Client(_Response(521))
    launched = []

    result = healthcheck.run_once(
        now=datetime(2026, 10, 17, 12, tzinfo=TZ).timestamp(),
        client=client,
        launcher=lambda **kwargs: launched.append(kwargs),
        state_path=state_path,
        events_path=events_path,
    )

    assert result['action'] == 'checked'
    assert result['classification'] == 'unreachable (origin down)'
    assert launched == []
    assert json.loads(state_path.read_text('utf-8'))['completed'] is False


def test_timeout_is_recorded_without_launch_or_immediate_retry(health_paths):
    state_path, events_path = health_paths
    client = _Client(error=TimeoutError('connect timed out'))
    launched = []
    now = datetime(2026, 10, 17, 12, tzinfo=TZ).timestamp()

    result = healthcheck.run_once(
        now=now,
        client=client,
        launcher=lambda **kwargs: launched.append(kwargs),
        state_path=state_path,
        events_path=events_path,
    )

    assert result['action'] == 'checked'
    assert result['classification'] == 'request_error'
    assert result['notify_vu'] is False
    assert launched == []
    assert len(client.calls) == 1
    saved = json.loads(state_path.read_text('utf-8'))
    assert saved['last_result'] == 'request_error'
    assert saved['last_attempt_at'] == now


def test_one_head_request_per_due_run_and_same_user_agent_no_proxy_or_redirects(health_paths):
    state_path, events_path = health_paths
    now = datetime(2026, 10, 17, 12, tzinfo=TZ).timestamp()
    client = _Client(_Response(503))

    first = healthcheck.run_once(now=now, client=client, state_path=state_path, events_path=events_path)
    second = healthcheck.run_once(now=now + 5 * 3600, client=client, state_path=state_path, events_path=events_path)

    assert first['action'] == 'checked'
    assert second['action'] == 'not_due'
    assert len(client.calls) == 1
    assert client.calls[0][0] == 'https://www.120ask.com/robots.txt'
    assert client.calls[0][1] == {'follow_redirects': False}
    options = healthcheck.client_options()
    assert options['headers']['User-Agent'] == healthcheck.direct.USER_AGENT
    assert options['trust_env'] is False
    assert options['follow_redirects'] is False


def test_deadline_ends_monitor_without_network_request(health_paths):
    state_path, events_path = health_paths
    client = _Client(_Response(200))
    now = datetime(2026, 10, 27, 0, tzinfo=TZ).timestamp()

    result = healthcheck.run_once(now=now, client=client, state_path=state_path, events_path=events_path)

    assert result['action'] == 'expired'
    assert client.calls == []
    assert json.loads(state_path.read_text('utf-8'))['expired'] is True


def test_launch_failure_after_200_is_logged_but_not_marked_complete(health_paths):
    state_path, events_path = health_paths
    client = _Client(_Response(200))

    def fail_launch(**kwargs):
        raise RuntimeError('already running')

    result = healthcheck.run_once(
        now=datetime(2026, 10, 17, 12, tzinfo=TZ).timestamp(),
        client=client,
        launcher=fail_launch,
        state_path=state_path,
        events_path=events_path,
    )

    assert result['action'] == 'launch_failed'
    assert result['notify_vu'] is True
    assert json.loads(state_path.read_text('utf-8'))['completed'] is False
    events = [json.loads(line) for line in events_path.read_text('utf-8').splitlines()]
    assert events[-1]['event'] == 'launch_failed'


def test_completed_monitor_never_probes_or_launches_again(health_paths):
    state_path, events_path = health_paths
    state_path.write_text(json.dumps({'completed': True, 'attempts': 1}), encoding='utf-8')
    client = _Client(_Response(200))
    launched = []

    result = healthcheck.run_once(
        now=datetime(2026, 10, 18, 12, tzinfo=TZ).timestamp(),
        client=client,
        launcher=lambda **kwargs: launched.append(kwargs),
        state_path=state_path,
        events_path=events_path,
    )

    assert result['action'] == 'complete'
    assert client.calls == []
    assert launched == []
