"""Gate to run before every upload of a compose ZIP: corpus v2 HF revision + the ZIP's own validation record.

    python -m r2ai.compose.release_check --manifest configs/best-c1.yaml --zip out/runs/c1/replay/C1.zip

Fails closed (exit != 0) when any of these does not hold:
  1. HF dataset revision, corpus/query SHA256 and the local ids parquet equal the pinned values (`python -m r2ai.corpus_v2 check`;
     exit 2 when HF cannot be reached: an unverifiable corpus is not submitted);
  2. the ZIP, its `.approval.json` and `.validator.json` agree (native validator ok, 0 errors, 1,200 rows, v2, ZIP/JSON SHA256);
  3. the ZIP is not above the ceiling, and its JSON SHA256 equals the manifest `expected.json_sha256` (derived ZIPs, e.g. a host
     filter, pass `--derived`: only 1 and 2 are checked, with the validator record of that ZIP).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from r2ai import corpus_v2

from . import manifest as M
from .archive import verify_ready_file
from .base_io import base_sha


def check_zip(zip_path, expected_json_sha256=None) -> list[str]:
    """Problems of the ZIP's validation record; empty list = ready."""
    zp = Path(zip_path)
    problems = []
    proof, done = zp.with_suffix('.approval.json'), zp.with_suffix('.validator.json')
    if not zp.is_file() or not proof.is_file() or not done.is_file():
        return [f'missing ZIP/approval/validator record next to {zp}']
    try:
        verify_ready_file(zp, json.loads(proof.read_text(encoding='utf-8')), json.loads(done.read_text(encoding='utf-8')))
    except ValueError as e:
        problems.append(str(e))
    if expected_json_sha256 is not None and base_sha(zp) != expected_json_sha256:
        problems.append(f'JSON SHA256 {base_sha(zp)} != expected {expected_json_sha256}')
    return problems


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--zip', required=True)
    ap.add_argument('--derived', action='store_true', help='ZIP derived from the compose output: do not compare with expected.json_sha256')
    ap.add_argument('--skip-hf', action='store_true', help='tests only: never for a real upload')
    a = ap.parse_args(argv)
    m = M.load(a.manifest)
    code = 0
    if not a.skip_hf:
        code = corpus_v2.main(['check'])
    problems = check_zip(a.zip, None if a.derived else m['expected']['json_sha256'])
    for p in problems:
        print('FAIL ' + p)
    if not problems:
        print('PASS ZIP validation record and JSON SHA256')
    if code or problems:
        print('NOT READY: do not submit')
        return code or 1
    print('READY: corpus v2 revision and ZIP record verified')
    return 0


if __name__ == '__main__':
    sys.exit(main())
