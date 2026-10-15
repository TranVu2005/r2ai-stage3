"""Pinned compose build (S8 compose_builder + locked batch-2 policy): base S6 JSON + zh v2 docs/chunks -> ZIP_LZMA dict256.

    python -m r2ai.compose build   --manifest configs/best-c1.yaml [--prepare-only | --json-only]
    python -m r2ai.compose replay  --manifest configs/best-c1.yaml          # JSON SHA gate against `expected`, then ZIP + native validator

Selection (unchanged from S8): doc rank order, absolute tau on the raw reranker logit, cap on physical docs/query, the
primary-present skip, v2 aliases only; chunks = ZL rule (logit >= .96, top 3 docs, whole doc_text, one chunk per v2 id,
only docs already selected). Every base doc and chunk is kept (no K cut, no protection ids). Any missing pin stops the build.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from r2ai.paths import ROOT, assert_writable, resolve_path

from . import batch3 as C3
from . import manifest as M
from .archive import (OverBudget, check_pins, compression_profile, json_sha256, render_rows, shape, verify_ready_file,
                      write_archive)
from .base_io import base_sha, baseline, check_schema, sha
from .zh_select import load_source_texts, project_candidate_v2, selections_for


class ReplayMismatch(ValueError):
    """The rebuilt JSON differs from the pinned expected SHA; the diff report says where."""


def valid_sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def check_base_stats(s):
    if not isinstance(s, dict) or not re.match(r'^S6-(?:g(?:-|$)|f(?:-|$)|h)', str(s.get('variant', ''))):
        raise ValueError('only explicit S6-g/f/h bases approved')
    v = s.get('validator', {})
    if s.get('queries') != 1200 or v.get('ok') is not True or v.get('errors') != 0 or v.get('rows') != 1200 or type(s.get('k')) != int or s['k'] < 1:
        raise ValueError('base requires validated 1200 queries/K')
    return s


def source_manifest_pins(manifest, root, coverage):
    """{docs.parquet path: SHA256} for every non-empty segment of the zh docs snapshot; total docs must equal `coverage`."""
    root = resolve_path(root)
    pins = {}
    total = 0
    names = set()
    for s in manifest.get('segments', []):
        name = s.get('name')
        count = s.get('docs_v2')
        if not isinstance(name, str) or not name or name in names or type(count) != int or count < 0:
            raise ValueError('invalid manifest segment/count')
        names.add(name)
        total += count
        if not count:
            continue
        p = resolve_path(root / name / 'docs.parquet')
        if not p.is_relative_to(root) or not (s.get('verified') is True or s.get('s8_doc_source_verified') is True) or not valid_sha(s.get('s8_docs_file_sha256')):
            raise ValueError('source requires verified snapshot SHA and safe path')
        pins[str(p)] = s['s8_docs_file_sha256']
    if manifest.get('s8_variant') == C3.VARIANT:
        total = C3.live_count(manifest)
    if total != coverage or not pins:
        raise ValueError('manifest coverage mismatch')
    return pins


def load_job(manifest_path, *, scope=M.COMPOSE_PINS):
    """Manifest -> verified job. Every pin in `scope` plus the zh source files is hashed on disk before anything is read or written."""
    manifest_path = resolve_path(manifest_path)
    m = M.load(manifest_path)
    M.check_policy(m)
    if C3.enabled(m):
        scope = (*scope, 'source_snapshot_report')
    M.check_declared(m, (*scope, 'zh_sources'))
    for key in ('name', 'output_dir', 'zh_chunk_root', 'expected', 'base_measured_scores'):
        if key not in m:
            raise M.PinError(f'manifest missing {key}')
    if not isinstance(m['name'], str) or not re.fullmatch('[A-Za-z0-9_-]{1,80}', m['name']):
        raise ValueError('unsafe output name')
    exp = m['expected']
    for key in ('json_sha256', 'zip_sha256'):
        if not valid_sha(exp.get(key)):
            raise M.PinError(f'expected.{key} missing')
    resolved = M.verify(m, (*scope, 'zh_sources'))
    one = {k: v[0] for k, v in resolved.items() if k != 'zh_sources'}
    sources = {str(p): e['sha256'] for p, e in zip(resolved['zh_sources'], M._entries(m, 'zh_sources'))}
    if C3.enabled(m):
        C3.check_source_report(m, one, sources)
    return {'manifest': m, 'manifest_path': manifest_path, 'manifest_sha256': sha(manifest_path), 'paths': one, 'zh_sources': sources}


def check_base_scores(scores):
    fields = ('final', 'doc_f2', 'doc_p', 'doc_r', 'chunk_f2', 'chunk_p', 'chunk_r')
    if not isinstance(scores, dict) or any(k not in scores or type(scores[k]) not in (float, int) or not math.isfinite(scores[k]) or not 0 <= scores[k] <= 1 for k in fields):
        raise ValueError('full measured baseline F2/P/R required; no proxy')
    if abs(scores['final'] - (scores['doc_f2'] + scores['chunk_f2']) / 2) > .00011:
        raise ValueError('baseline components inconsistent')
    return dict(scores)


def prepare(job):
    """Pins, base and candidate checks, selection and plan. Returns (out, stats, plan, membership, sels, body, texts)."""
    m, p = job['manifest'], job['paths']
    check_base_scores(m['base_measured_scores'])
    out = assert_writable(M.expand(m['output_dir']))
    out.mkdir(parents=True, exist_ok=True)
    chunk_root = M.expand(m['zh_chunk_root'])
    pins = {str(job['manifest_path']): job['manifest_sha256']}
    for key in M.COMPOSE_PINS:
        pins[str(p[key])] = M._entries(m, key)[0].get('sha256')
    s = check_base_stats(json.loads(p['base_stats'].read_text(encoding='utf-8')))
    if resolve_path(s['json']) != p['base_json'] or resolve_path(s['zip']) != p['base_zip']:
        raise ValueError('stats paths differ from the pinned base files')
    for field, key in (('zip', 'base_zip'), ('json', 'base_json')):
        if s.get(field + '_sha256') != M._entries(m, key)[0]['sha256']:
            raise ValueError('stats SHA differs ' + field)
    if base_sha(p['base_zip']) != M._entries(m, 'base_json')[0]['sha256']:
        raise ValueError('base ZIP JSON SHA differs')
    header = compression_profile(p['base_zip'])
    if header['dictionary_mib'] != 256 or (header['lc'], header['lp'], header['pb']) != (3, 0, 2):
        raise ValueError('base requires approved dict256/lc-lp-pb')
    zh_manifest = json.loads(p['zh_manifest'].read_text(encoding='utf-8'))
    corpus_sha = M._entries(m, 'corpus_v2_ids')[0]['sha256']
    if zh_manifest.get('v2_ids_sha256') != corpus_sha:
        raise ValueError('manifest corpus SHA differs')
    coverage = m['expected']['coverage_docs']
    declared = source_manifest_pins(zh_manifest, chunk_root, coverage)
    if declared != job['zh_sources']:
        raise ValueError('zh_sources pins differ from the docs snapshot manifest')
    pins.update(declared)
    if C3.enabled(m):
        pins[str(p['source_snapshot_report'])] = M._entries(m, 'source_snapshot_report')[0]['sha256']
    check_pins(pins)
    member = {r['id']: r['lang'] for r in pq.read_table(p['corpus_v2_ids'], columns=['id', 'lang']).to_pylist()}
    qids = pq.read_table(p['query'], columns=['id'])['id'].to_pylist()
    if len(qids) != 1200 or len(set(qids)) != 1200:
        raise ValueError('query 1200 required')
    table = pq.read_table(p['candidate'])
    check_schema(table)
    rows, _ = project_candidate_v2(table.to_pylist(), member)
    if C3.enabled(m):
        C3.check_candidate_live(rows, zh_manifest)
    by = collections.defaultdict(list)
    for r in rows:
        if type(r['query_id']) != int or type(r['rank']) != int or r['rank'] < 1 or not math.isfinite(r['logit']) or r['doc_id'] not in r['doc_ids_group']:
            raise ValueError('candidate schema/rank/logit invalid')
        by[r['query_id']].append(r)
    if set(by) != set(qids):
        raise ValueError('candidate query coverage differs')
    for rs in by.values():
        rs.sort(key=lambda r: r['rank'])
        if [r['rank'] for r in rs] != list(range(1, len(rs) + 1)) or any(a['logit'] < b['logit'] for a, b in zip(rs, rs[1:])):
            raise ValueError('candidate rank/order differs')
    present = {}
    for r, _ in baseline(p['base_json']):
        shape(r, member)
        if r['id'] in present:
            raise ValueError('duplicate base query')
        if any(member[ch['doc_id']] == 'zh' for ch in r['relevant_chunks']):
            raise ValueError('base zh chunks require explicit source handoff; no fallback')
        present[r['id']] = r['relevant_docs']
    if set(present) != set(qids):
        raise ValueError('base query coverage differs')
    tau, cap = M.POLICY['tau_doc'], M.POLICY['doc_cap']
    sels, body = selections_for(by, present, tau=tau, cap=cap)
    if C3.enabled(m):
        C3.check_mean(sels)
    source, hashes = load_source_texts([r for rs in body.values() for r in rs], zh_manifest, chunk_root, p['corpus_v2_ids'])
    plan = {'name': m['name'], 'base_stats': s, 'candidate_batch': m['policy']['candidate_batch'], 'tau_doc': tau, 'doc_cap': cap,
            'chunk_tau': M.POLICY['chunk_tau'], 'chunk_cap': M.POLICY['chunk_doc_cap'], 'header': header, 'coverage_docs': coverage,
            'pins': pins, 'source_doc_files': hashes, 'physical_docs': sum(len(r['docs']) for r in sels.values()),
            'new_doc_ids': sum(len(r['ids']) for r in sels.values()),
            'new_chunk_ids': sum(len(r['doc_ids_v2']) for rs in body.values() for r in rs)}
    target = assert_writable(out / (m['name'] + '.plan.json'))
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != plan:
            raise ValueError('plan changed; choose new output name')
    else:
        target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return out, s, plan, member, sels, body, {k: r['doc_text'] for k, r in source.items()}


# ---------------------------------------------------------------- replay gate and diff

def row_diff(ref: dict, new: dict) -> dict:
    """Field-level difference of two submission rows (reference vs rebuilt)."""
    import hashlib
    out = {}
    if ref['id'] != new['id']:
        out['id'] = [ref['id'], new['id']]
    a, b = ref['relevant_docs'], new['relevant_docs']
    if a != b:
        out['relevant_docs'] = {'len': [len(a), len(b)], 'only_reference': [d for d in a if d not in set(b)][:20],
                                'only_new': [d for d in b if d not in set(a)][:20], 'same_set_other_order': set(a) == set(b)}
    ca, cb = ref['relevant_chunks'], new['relevant_chunks']
    if ca != cb:
        ha = {c['doc_id']: hashlib.sha256(c['chunk_text'].encode('utf-8')).hexdigest() for c in ca}
        hb = {c['doc_id']: hashlib.sha256(c['chunk_text'].encode('utf-8')).hexdigest() for c in cb}
        out['relevant_chunks'] = {'len': [len(ca), len(cb)], 'doc_ids_only_reference': [d for d in ha if d not in hb][:20],
                                  'doc_ids_only_new': [d for d in hb if d not in ha][:20],
                                  'text_differs': [d for d in ha if d in hb and ha[d] != hb[d]][:20],
                                  'same_ids_other_order': [c['doc_id'] for c in ca] != [c['doc_id'] for c in cb] and ha.keys() == hb.keys()}
    return out


def replay_json(job, prepared, *, reference_zip=None, limit=50):
    """Re-render the JSON without a ZIP, hash it and (when a reference ZIP is pinned) diff row by row. Never edits anything to match."""
    out, s, plan, member, sels, body, texts = prepared
    expected = job['manifest']['expected']['json_sha256']
    kw = dict(base=job['paths']['base_json'], selections=sels, chunk_rows=body, texts=texts, membership=member, vi_k=sys.maxsize)
    import hashlib
    h = hashlib.sha256()
    diffs = []
    ref_iter = baseline(reference_zip) if reference_zip else None
    ref_bytes = None
    n_rows = 0
    for payload, audit in render_rows(**kw):
        h.update(payload)
        if audit is None:
            continue
        n_rows += 1
        if ref_iter is not None:
            ref_row, ref_raw = next(ref_iter)
            if ref_raw.strip().removesuffix(b',') != payload.strip().removesuffix(b','):
                d = row_diff(ref_row, json.loads(payload.strip().removesuffix(b',')))
                if len(diffs) < limit:
                    diffs.append({'query_id': ref_row['id'], **d})
                else:
                    diffs.append(None)
    got = h.hexdigest()
    report = {'rows': n_rows, 'expected_json_sha256': expected, 'json_sha256': got, 'match': got == expected,
              'rows_differing': len(diffs), 'diffs': [d for d in diffs if d]}
    return report


def check_resume(meta, expected):
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError('resume identity differs ' + key)


def native_validate(job, dst, meta, plan):
    log = assert_writable(dst.with_suffix('.validator.log'))
    done = assert_writable(dst.with_suffix('.validator.json'))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONPATH': str(ROOT / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''), 'PYTHONIOENCODING': 'utf-8'}
    with log.open('w', encoding='utf-8') as f:
        r = subprocess.run([sys.executable, '-m', 'r2ai.compose.native', '--manifest', str(job['manifest_path']), '--worker', str(dst)],
                           stdout=f, stderr=subprocess.STDOUT, env=env)
    records = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith('{')]
    if r.returncode or not records or records[-1].get('ok') is not True or records[-1].get('errors') != 0 or records[-1].get('rows') != 1200:
        raise ValueError('native failed: ' + str(log))
    check_pins(plan['pins'])
    if sha(dst) != meta['zip_sha256'] or base_sha(dst) != meta['json_sha256']:
        raise ValueError('native output hash differs')
    result = {**records[-1], 'zip_sha256': meta['zip_sha256'], 'json_sha256': meta['json_sha256'], 'v2': True,
              'cap_bytes': meta['ceiling_bytes'], 'native_checks': 'all schema/v2/protection/dedupe/near/verbatim active'}
    done.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    meta.update(native_validator_pending=False, native_validator=result)
    assert_writable(dst.with_suffix('.approval.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def build(manifest_path, *, prepare_only=False, json_only=False):
    job = load_job(manifest_path)
    prepared = prepare(job)
    out, s, plan, member, sels, body, texts = prepared
    m = job['manifest']
    if prepare_only:
        print(json.dumps({k: v for k, v in plan.items() if k not in ('pins', 'source_doc_files', 'base_stats')}, indent=2))
        return plan
    reference = None
    for ref_key in ('reference_zip', 'reference_c1_zip'):
        if ref_key in m['inputs']:
            reference = M.verify(m, (ref_key,))[ref_key][0]
    report = replay_json(job, prepared, reference_zip=reference)
    report_path = assert_writable(out / (m['name'] + '.replay-json.json'))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'diffs'}, indent=2), flush=True)
    if not report['match']:
        raise ReplayMismatch(f'JSON SHA {report["json_sha256"]} != expected {report["expected_json_sha256"]}; '
                             f'{report["rows_differing"]} rows differ, report {report_path}')
    if json_only:
        return report
    dst = assert_writable(out / (m['name'] + '.zip'))
    proof = assert_writable(dst.with_suffix('.approval.json'))
    c = m['expected']
    identity = {'stage': m['name'], 'base_json_path': str(job['paths']['base_json']), 'base_json_sha256': M._entries(m, 'base_json')[0]['sha256'],
                'base_zip_path': str(job['paths']['base_zip']), 'base_zip_sha256': M._entries(m, 'base_zip')[0]['sha256'],
                'candidate_path': str(job['paths']['candidate']), 'candidate_sha256': M._entries(m, 'candidate')[0]['sha256'],
                'candidate_batch': m['policy']['candidate_batch'], 'manifest_path': str(job['paths']['zh_manifest']),
                'manifest_sha256': M._entries(m, 'zh_manifest')[0]['sha256'], 'tau_raw_logit': plan['tau_doc'],
                'cap_physical_docs': M.POLICY['doc_cap'], 'chunk_tau_raw_logit': M.POLICY['chunk_tau'],
                'chunk_doc_cap': M.POLICY['chunk_doc_cap'], 'ceiling_bytes': M.POLICY['ceiling_bytes'], 'K': s['k'],
                'compose_config_path': str(job['manifest_path']), 'compose_config_sha256': job['manifest_sha256'],
                'coverage_docs': c['coverage_docs'], 'source_doc_files': plan['source_doc_files'], 'header': plan['header'],
                'keep_base_docs_and_chunks': True, 'plan_sha256': sha(out / (m['name'] + '.plan.json'))}
    if dst.exists() or proof.exists():
        if not dst.exists() or not proof.exists():
            raise ValueError('partial publish; stop')
        meta = json.loads(proof.read_text(encoding='utf-8'))
        check_resume(meta, identity)
        if sha(dst) != meta['zip_sha256'] or base_sha(dst) != meta['json_sha256']:
            raise ValueError('resume hash differs')
        done = dst.with_suffix('.validator.json')
        if done.exists():
            verify_ready_file(dst, meta, json.loads(done.read_text(encoding='utf-8')))
            print('KEEP ready ' + m['name'])
            return meta
        native_validate(job, dst, meta, plan)
        return meta

    def gate(result):
        check_pins(plan['pins'])
        if result['removed_vi_chunks'] or result['protected_ids_added'] or result['header'] != plan['header']:
            raise ValueError('base/doc-policy/header gate failed')
        if result['json_sha256'] != c['json_sha256']:
            raise ReplayMismatch('ZIP JSON SHA differs from expected after the JSON gate passed')

    try:
        meta = write_archive(base=job['paths']['base_json'], dst=dst, selections=sels, chunk_rows=body, texts=texts, membership=member,
                             vi_k=sys.maxsize, ceiling=M.POLICY['ceiling_bytes'], proof_path=proof,
                             metadata={**identity, 'native_validator_pending': True, 'submission_approved': False}, postflight=gate)
    except OverBudget as e:
        raise ValueError(f'ZIP {e.size} > cap {e.cap}; stop, no cuts authorised') from e
    native_validate(job, dst, meta, plan)
    print('READY ' + m['name'] + ' ' + str(meta['zip_bytes']) + ' ' + meta['zip_sha256'], flush=True)
    return meta


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('cmd', choices=('build', 'replay'))
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--json-only', action='store_true', help='re-render and hash the JSON, diff against the reference; no ZIP')
    a = ap.parse_args(argv)
    result = build(a.manifest, prepare_only=a.prepare_only, json_only=a.json_only)
    if a.cmd == 'replay' and not a.prepare_only and not a.json_only:
        zip_sha = result['zip_sha256']
        want = M.load(a.manifest)['expected']['zip_sha256']
        print(json.dumps({'zip_sha256': zip_sha, 'expected_zip_sha256': want, 'zip_match': zip_sha == want}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
