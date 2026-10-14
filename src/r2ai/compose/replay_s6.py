"""Replay of the C1 base S6-f-K150 (JSON only) from its pinned inputs: cc-A-cut115 docs + pool-B chunk cache, mode inset, K150.

    python -m r2ai.compose.replay_s6 --manifest configs/best-c1.yaml

Inputs come from the manifest pins (cc_a_cut115_json, pool_b_cache); the render is `r2ai.cc_longchau.s6.render` unchanged. The JSON
SHA256 must equal the pinned S6-f base JSON; on a difference the first differing rows are reported per query/field and nothing
is edited. The ZIP is not rebuilt here (the compose step reads the pinned S6-f ZIP/JSON); S6-f itself stays replayable with
`python -m r2ai.cc_longchau.s6 build --name S6-f --pool B --k 150 --k-cap 200 --mode inset --zip-method lzma --lzma-dict-mib 256`.
"""
from __future__ import annotations

import json
import sys

from . import manifest as M
from .base_io import baseline, sha
from .build import row_diff

K = 150
K_CAP = 200


def replay(manifest_path, *, name='S6-f-K150-replay', limit=20) -> dict:
    from r2ai.cc_longchau import s6 as S
    from r2ai.cc_longchau.common import Roots
    m = M.load(manifest_path)
    got = M.verify(m, (*M.S6F_PINS, 'base_json'))
    if got['cc_a_cut115_json'][0] != S.resolve_path(S.CC_A_CUT115):
        raise M.PinError('manifest cc_a_cut115_json is not the file the s6 builder reads')
    expected = M._entries(m, 'base_json')[0]['sha256']
    out = S.out_dir(name)
    jp = S.guard(out / (name.replace('-', '_') + '.json'))
    if jp.exists():
        raise ValueError(f'{jp} exists; choose another --name')
    docs = S.pinned_docs()
    expect_cache = S.guard(S.S6_RUN / 'cache' / f'B_k{K_CAP}.pkl')
    if expect_cache != got['pool_b_cache'][0]:
        raise M.PinError('manifest pool_b_cache is not the cache the s6 builder reads')
    qids, new, texts = S.chunk_pool('B', Roots(), k_cap=K_CAP, use_cache=True)
    chunks = S.chunk_lists_inset(qids, new, texts, K, docs)
    del texts, new
    counts = S.render(jp, qids, docs, chunks, protect=True)
    del chunks
    digest = sha(jp)
    report = {'json': str(jp), 'json_sha256': digest, 'expected_json_sha256': expected, 'match': digest == expected, 'counts': counts}
    if digest != expected:
        diffs = []
        for (ref, _), (new_row, _) in zip(baseline(got['base_json'][0]), baseline(jp)):
            d = row_diff(ref, new_row)
            if d:
                diffs.append({'query_id': ref['id'], **d})
                if len(diffs) >= limit:
                    break
        report['first_differing_rows'] = diffs
    (out / 'replay.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    return report


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--name', default='S6-f-K150-replay')
    a = ap.parse_args(argv)
    report = replay(a.manifest, name=a.name)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report['match'] else 1


if __name__ == '__main__':
    sys.exit(main())
