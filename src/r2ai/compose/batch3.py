"""C3 opt-in (zh candidate batch 3 with superseded docs). C1 policy, selection and serializers are unchanged.

A manifest enables it with `variant: C3-batch3` and `name: C3`; its policy must equal the C1 policy except `candidate_batch: 3`. The zh docs
snapshot lists the physical docs of every segment plus `superseded` edges (doc replaced by a redo segment); live docs = physical - superseded.
A candidate row may only point at a live source. The mean number of selected physical docs per query must stay <= 20 before any ZIP is written.
"""
from __future__ import annotations

import json

CANDIDATE_SHA = 'ff20ae1a40e4251425e5fe02c2827704a414e34a3d65132518db8d24baf7225b'
VARIANT = 'C3-batch3'
COVERAGE_LIVE = 152036
PHYSICAL = 153604
SUPERSEDED = 1568
MAX_MEAN_DOCS = 20


def enabled(m) -> bool:
    return m.get('variant') == VARIANT


def check_config(m) -> None:
    from . import manifest as M
    if m.get('name') != 'C3' or m.get('expected', {}).get('coverage_docs') != COVERAGE_LIVE:
        raise M.PinError('C3 name/live coverage differs')
    M.check_declared(m, ('candidate', 'source_snapshot_report'))
    if M._entries(m, 'candidate')[0]['sha256'] != CANDIDATE_SHA:
        raise M.PinError('C3 candidate pin differs')


def live_count(m) -> int:
    """Live docs of a batch-3 snapshot manifest after validating the superseded edges (no duplicates, no cycles, known segments)."""
    if m.get('s8_variant') != VARIANT:
        raise ValueError('batch3 snapshot opt-in missing')
    segs = {s['name']: s for s in m['segments']}
    if len(segs) != len(m['segments']):
        raise ValueError('duplicate segments')
    seen = set()
    edges = {}
    sup = m.get('superseded', [])
    for r in sup:
        d, a, b = r.get('doc_id'), r.get('segment'), r.get('by')
        key = (d, a)
        if type(d) != int or a not in segs or b not in segs or a == b or key in seen:
            raise ValueError('bad superseded edge')
        seen.add(key)
        edges[key] = b
    for d, a in edges:
        trail = set()
        while (d, a) in edges:
            if a in trail:
                raise ValueError('superseded cycle')
            trail.add(a)
            a = edges[(d, a)]
    physical = sum(r['docs_v2'] for r in m['segments'])
    live = physical - len(sup)
    if m.get('s8_physical_docs') != physical or m.get('s8_superseded_docs') != len(sup) or m.get('s8_live_docs') != live:
        raise ValueError('snapshot physical/live/superseded counts differ')
    return live


def check_candidate_live(rows, m) -> None:
    gone = {(e['segment'], e['doc_id']) for e in m.get('superseded', [])}
    segments = {s['name'] for s in m['segments']}
    for r in rows:
        if r['segment'] not in segments or (r['segment'], r['doc_id']) in gone:
            raise ValueError('candidate references superseded/unknown source')


def check_mean(sels) -> None:
    if len(sels) != 1200 or sum(len(r['docs']) for r in sels.values()) / 1200 > MAX_MEAN_DOCS:
        raise ValueError('C3 mean doc/query exceeds 20 or query count differs; stop')


def check_source_report(m, paths, source_pins) -> None:
    """The CPU snapshot QA report must describe exactly this candidate, snapshot manifest and source pins."""
    from . import manifest as M
    q = json.loads(paths['source_snapshot_report'].read_text(encoding='utf-8'))
    if (q.get('ok') is not True or q.get('candidate_sha256') != CANDIDATE_SHA
            or q.get('snapshot_manifest_sha256') != M._entries(m, 'zh_manifest')[0]['sha256']
            or q.get('physical_docs') != PHYSICAL or q.get('superseded_docs') != SUPERSEDED or q.get('live_docs') != COVERAGE_LIVE
            or q.get('source_pins') != source_pins):
        raise ValueError('C3 CPU snapshot report/pins failed')
