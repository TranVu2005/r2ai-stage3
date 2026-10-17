"""Check whether 120ask's origin has recovered and resume its v2 queue once."""
from __future__ import annotations

# Resolve the configured workspace roots before importing crawler or ML code.
from r2ai.zh_v2 import direct

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


URL = 'https://www.120ask.com/robots.txt'
INTERVAL_S = 6 * 60 * 60
TIMEZONE = ZoneInfo('Asia/Bangkok')
DEADLINE = datetime(2026, 10, 27, 0, 0, tzinfo=TIMEZONE)
STATE_FILE = direct.RUN / 'healthcheck.json'
EVENTS_FILE = direct.RUN / 'healthcheck.jsonl'


def client_options() -> dict:
    """Use the approved truthful UA, one local egress path, and no redirects."""
    return direct.client_options()


def _append_event(path: Path, event: dict) -> None:
    path = direct.guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text('utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'health-check state is not a JSON object: {path}')
    return value


def _classification(status_code: int | None) -> str:
    if status_code == 521:
        return 'unreachable (origin down)'
    if status_code == 200:
        return 'origin reachable'
    return f'http_{status_code}' if status_code is not None else 'request_error'


def run_once(*, now: float | None = None, client=None, launcher=None,
             state_path: Path = STATE_FILE, events_path: Path = EVENTS_FILE) -> dict:
    """Make at most one HEAD request if due; resume 120ask only after HTTP 200.

    The scheduler calls this once every six hours. Persisting ``last_attempt_at``
    before opening the socket prevents duplicate probes if a run is interrupted.
    """
    state_path, events_path = direct.guard(state_path), direct.guard(events_path)
    current = time.time() if now is None else float(now)
    state = _read_state(state_path)
    if state.get('completed'):
        return {'action': 'complete', 'attempts': state.get('attempts', 0)}
    if state.get('expired'):
        return {'action': 'expired', 'attempts': state.get('attempts', 0)}
    if datetime.fromtimestamp(current, TIMEZONE) >= DEADLINE:
        state.update({'expired': True, 'expired_at': current})
        direct._atomic_json(state_path, state)
        event = {'event': 'deadline_expired', 'at': current, 'deadline': DEADLINE.isoformat()}
        _append_event(events_path, event)
        return {'action': 'expired', 'deadline': DEADLINE.isoformat()}

    previous = state.get('last_attempt_at')
    if previous is not None and current - float(previous) < INTERVAL_S:
        return {'action': 'not_due', 'next_check_at': float(previous) + INTERVAL_S,
                'attempts': state.get('attempts', 0)}

    state.update({'completed': False, 'expired': False, 'last_attempt_at': current,
                  'attempts': int(state.get('attempts', 0)) + 1,
                  'last_result': 'request_in_progress'})
    direct._atomic_json(state_path, state)
    _append_event(events_path, {'event': 'attempt_started', 'at': current, 'url': URL,
                                'method': 'HEAD', 'user_agent': direct.USER_AGENT})

    try:
        if client is None:
            import httpx
            with httpx.Client(**client_options()) as active_client:
                response = active_client.head(URL, follow_redirects=False)
        else:
            response = client.head(URL, follow_redirects=False)
        status_code = int(response.status_code)
        classification = _classification(status_code)
        event = {'event': 'response', 'at': current, 'url': URL, 'method': 'HEAD',
                 'http_status': status_code, 'classification': classification}
        if status_code != 200:
            state.update({'last_result': classification, 'last_http_status': status_code})
            direct._atomic_json(state_path, state)
            _append_event(events_path, event)
            return {'action': 'checked', 'http_status': status_code,
                    'classification': classification, 'notify_vu': False}
    except Exception as exc:
        reason = f'{type(exc).__name__}: {str(exc)[:300]}'
        state.update({'last_result': 'request_error', 'last_error': reason, 'last_http_status': None})
        direct._atomic_json(state_path, state)
        _append_event(events_path, {'event': 'request_error', 'at': current, 'url': URL,
                                    'method': 'HEAD', 'reason': reason})
        return {'action': 'checked', 'http_status': None, 'classification': 'request_error',
                'error': reason, 'notify_vu': False}

    # A 200 is the sole resume gate. Launch only the existing 120ask queue.
    resume = direct.launch if launcher is None else launcher
    try:
        launch_info = resume(hosts=('120ask.com',))
    except Exception as exc:
        reason = f'{type(exc).__name__}: {str(exc)[:300]}'
        state.update({'last_result': 'launch_failed', 'last_http_status': 200,
                      'launch_error': reason})
        direct._atomic_json(state_path, state)
        _append_event(events_path, {'event': 'launch_failed', 'at': current, 'http_status': 200,
                                    'classification': 'origin reachable', 'reason': reason})
        return {'action': 'launch_failed', 'http_status': 200, 'notify_vu': True, 'error': reason}

    state.update({'last_result': 'resumed', 'last_http_status': 200, 'completed': True,
                  'completed_at': current, 'launch_info': launch_info})
    direct._atomic_json(state_path, state)
    _append_event(events_path, {'event': 'resumed', 'at': current, 'http_status': 200,
                                'hosts': ['120ask.com'], 'launch_info': launch_info})
    return {'action': 'resumed', 'http_status': 200, 'notify_vu': True,
            'launch_info': launch_info}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    direct.preflight()
    print(json.dumps(run_once(), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
