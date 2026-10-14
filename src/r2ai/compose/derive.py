"""Derive a host-filtered ZIP from a built compose ZIP (e.g. C1-no-vinmec) with the same validation record as the source.

    python -m r2ai.compose.derive --manifest configs/best-c1.yaml --src out/runs/c1/replay/C1.zip \
        --dst out/runs/c1/no-vinmec/C1-no-vinmec.zip --drop-host vinmec.com

Uses `r2ai.submit.corpus_filter` (--drop-host, dict 256 MiB) and the native validator in `--derived` mode (every check active); writes
`<dst>.validator.json` and `<dst>.approval.json` so that `r2ai.compose.release_check --derived` can gate the upload. Never uploaded here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from r2ai.paths import ROOT, assert_writable, resolve_path
from r2ai.submit import corpus_filter as F

from . import manifest as M
from .base_io import base_sha, sha


def derive(manifest_path, src, dst, hosts, *, max_zip_bytes=M.POLICY['ceiling_bytes']) -> dict:
    src, dst = resolve_path(src), assert_writable(dst)
    stats = F.filter_submission(src, dst, max_zip_bytes=max_zip_bytes, lzma_dict_mib=M.POLICY['dict_mib'], drop_hosts=hosts)
    log = assert_writable(dst.with_suffix('.validator.log'))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8',
           'PYTHONPATH': str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
    with log.open('w', encoding='utf-8') as f:
        r = subprocess.run([sys.executable, '-m', 'r2ai.compose.native', '--manifest', str(manifest_path), '--worker', str(dst), '--derived'],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('{')]
    if r.returncode or not records or records[-1].get('ok') is not True or records[-1].get('errors') != 0 or records[-1].get('rows') != 1200:
        raise ValueError('native validator failed on the derived ZIP: ' + str(log))
    zip_sha, json_sha = sha(dst), base_sha(dst)
    if json_sha != stats['json_sha256']:
        raise ValueError('derived JSON SHA256 differs from the filter stream')
    result = {**records[-1], 'zip_sha256': zip_sha, 'json_sha256': json_sha, 'v2': True, 'cap_bytes': M.POLICY['ceiling_bytes'],
              'native_checks': 'all schema/v2/protection/dedupe/near/verbatim active (derived mode)'}
    assert_writable(dst.with_suffix('.validator.json')).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    meta = {'zip_bytes': stats['zip_bytes'], 'zip_sha256': zip_sha, 'json_sha256': json_sha, 'rows': 1200, 'ceiling_bytes': M.POLICY['ceiling_bytes'],
            'native_validator_pending': False, 'native_validator': result, 'submission_approved': False,
            'derived_from': {'zip': str(src), 'zip_sha256': sha(src), 'json_sha256': base_sha(src), 'drop_hosts': list(hosts)},
            'filter_stats': stats}
    assert_writable(dst.with_suffix('.approval.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return meta


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--drop-host', action='append', required=True)
    a = ap.parse_args(argv)
    meta = derive(resolve_path(a.manifest), a.src, a.dst, a.drop_host)
    print(json.dumps({k: v for k, v in meta.items() if k != 'native_validator'}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
