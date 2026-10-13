"""S6: cc-A-cut115 relevant_docs + a K-parameterised full-chunk list from a chosen chunk pool, optional chunk-doc protection,
selectable ZIP method. Opt-in module; no default of any existing code path changes.

  relevant_docs  = those of cc-A-cut115 (JSON pinned by SHA256), byte for byte
  relevant_chunks = the first K docs of the pool's RRF chunk order, full chunk text (r2ai.submit.rrf_chunks.choose_docs +
                   make_submission.load_full_texts: same rule as RRF chunk docs); K is a parameter (K=50 reproduces the base)
  protection ON  = a chunk doc_id that is not in relevant_docs is appended to it (chunk rank order); OFF = relevant_docs untouched
                   (the validator then needs --allow-chunk-outside-docs: a risk for the official format)
  --zip-method   deflate (default, level 9) | lzma (ZIP_LZMA, Python zipfile default preset)

  python -m r2ai.cc_longchau.s6 build --name S6-a --pool B --k 75 --zip-method lzma [--no-protect]
  python -m r2ai.cc_longchau.s6 gate1            # K=50, protection ON, pool cc-A -> JSON == cc-A-cut115 (99a9e11d...)
  python -m r2ai.cc_longchau.s6 gate2 --k 80     # doc cut A1-115 recut at K (S7 semantics) + VS-append pool + LZMA -> S7 K80 JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import statistics
import subprocess
import sys
import time
from pathlib import Path

from r2ai import corpus_v2
from r2ai.paths import CORPUS_FILE, OLD_ROOT, QUERY_FILE, ROOT, RUNS_DIR, assert_writable, resolve_path
from r2ai.submit.best import build_argv, load_config, sha256
from r2ai.submit.zipio import METHODS, verify_submission_zip, write_submission_zip

from .common import Roots

S6_RUN = RUNS_DIR / 's6'
CC_A_CUT115 = RUNS_DIR / 'cc-longchau' / 'VS-cc-A-cut115' / 'VS_cc_A_cut115.json'
CC_A_CUT115_SHA256 = '99a9e11d76d2a3f0ac429d761e25eb4007a2324bab1eff6c65a5a05b954a2bfd'
CEILING = 101_870_962                  # largest ZIP the leaderboard has accepted (VS40); 413 starts somewhere below 121.6 MB
CONFIG = 'configs/submission-best.yaml'
GATE_EXPAND = 'out/runs/best-vsappend/gate-best/sub_expand.json'          # VS-append H1-expand JSON (config expand_json_sha256)
S7_K80_JSON_SHA256 = '91fd6049245313dde4a7d42c8f86bae851acb43cf1aa7f6ccc8fd8e73b373a44'
S7_K80_ZIP_SHA256 = 'cc765eead79003ec28b422cd420f5e2ad7fea31d5ea4b23bb6b11c4b03182d02'
K_CAP = 100                            # the pool cache holds the first K_CAP chunk docs per query
_SEP = (', ', ': ')


def _p(x) -> str:
    return str(resolve_path(x))


def guard(path) -> Path:
    p = resolve_path(path)
    if OLD_ROOT is not None and p.is_relative_to(OLD_ROOT):
        raise ValueError(f'Write to OLD is forbidden: {p}')
    if not p.is_relative_to(resolve_path(S6_RUN)):
        raise ValueError(f'Not an s6 output: {p}')
    return p


def out_dir(name: str) -> Path:
    d = guard(S6_RUN / name)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ inputs

def read_docs(path) -> dict[int, list[int]]:
    """{query id: relevant_docs} from a submission JSON in the one-row-per-line layout (never decodes the chunk texts)."""
    out = {}
    with open(path, encoding='utf-8', newline='') as f:
        for line in f:
            line = line.rstrip('\r\n').rstrip(',')
            if not line.startswith('{'):
                continue
            i = line.index(', "relevant_chunks": ')
            row = json.loads(line[:i] + '}')
            out[int(row['id'])] = [int(d) for d in row['relevant_docs']]
    return out


def pinned_docs() -> dict[int, list[int]]:
    got = sha256(Path(CC_A_CUT115))
    if got != CC_A_CUT115_SHA256:
        raise ValueError(f'cc-A-cut115 JSON SHA256 {got} != {CC_A_CUT115_SHA256}')
    return read_docs(CC_A_CUT115)


def pool_setup(pool: str, roots: Roots):
    """-> (cfg, docs_dir, choose_docs kwargs). cc-A / VS-append = vi-only pool (config best); B = vi + CC merged retrieval."""
    if pool == 'B':
        from .build_b import b_config
        cfg = b_config(roots)
    elif pool in ('cc-A', 'VS-append'):
        cfg = load_config(_p(CONFIG))
    else:
        raise ValueError(f'unknown pool {pool!r}')
    s, rrf = cfg['submission'], cfg['postprocess']['rrf_chunk_docs']
    argv = build_argv(cfg, False, 'x.zip')
    arg = lambda k: argv[argv.index(k) + 1]
    kw = dict(runs_dir=arg('--runs-dir'), doc_ranking=arg('--doc-ranking'), hybrid_ranks=rrf['hybrid_ranks'], rerank_run=rrf['reranker_cache'],
              queries=arg('--queries'), k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')), rrf_k=int(rrf['k']),
              missing_rank=int(rrf['missing_rank']))
    return cfg, Path(_p(s['docs_dir'])), kw


def chunk_pool(pool: str, roots: Roots, *, k_cap: int = K_CAP, use_cache: bool = True):
    """-> (qids, {q: first k_cap chunk doc ids in RRF order}, {doc id: full chunk text}). Cached pickle under out/runs/s6/cache."""
    cache = guard(S6_RUN / 'cache' / f'{pool}_k{k_cap}.pkl')
    if use_cache and cache.exists():
        with open(cache, 'rb') as f:
            return pickle.load(f)
    from r2ai.submit.make_submission import load_full_texts
    from r2ai.submit.rrf_chunks import choose_docs
    _, docs_dir, kw = pool_setup(pool, roots)
    qids, _, new, _ = choose_docs(k_chunk=k_cap, **kw)
    texts, _, _ = load_full_texts({d for q in qids for d in new[q]}, docs_dir)
    res = (qids, new, texts)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = guard(cache.with_name(cache.name + '.partial'))
    with open(tmp, 'wb') as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache)
    return res


def chunk_lists(qids, new, texts, k: int) -> dict[int, list[tuple[int, str]]]:
    """First k docs of each query's RRF order that have a full text (a doc without text gets no chunk, as in the builder)."""
    return {q: [(d, texts[d]) for d in new[q][:k] if texts.get(d, '').strip()] for q in qids}


def chunk_lists_inset(qids, new, texts, k: int, docs_by_q) -> dict[int, list[tuple[int, str]]]:
    """S6-f: first k docs of the RRF order that have a full text AND are already in the fixed relevant_docs (no doc is ever added)."""
    out = {}
    for q in qids:
        have = set(docs_by_q[q])
        out[q] = [(d, texts[d]) for d in new[q] if d in have and texts.get(d, '').strip()][:k]
    return out


def chunk_lists_hybrid(qids, new, texts, k: int, docs_by_q, split: int = 80) -> dict[int, list[tuple[int, str]]]:
    """S6-g: rank = position among docs with full text. Ranks 1..split as chunk_lists (docs missing from relevant_docs get added by
    protection); ranks split+1..k only from docs already in relevant_docs (fixed set + docs protection added at ranks <= split)."""
    out = {}
    for q in qids:
        allowed = set(docs_by_q[q])
        res, rank = [], 0
        for d in new[q]:
            if not texts.get(d, '').strip():
                continue
            rank += 1
            if rank > k:
                break
            if rank <= split:
                allowed.add(d)
                res.append((d, texts[d]))
            elif d in allowed:
                res.append((d, texts[d]))
        out[q] = res
    return out


# ------------------------------------------------------------------ render

def render(path, qids, docs_by_q, chunks_by_q, *, protect: bool, cc_ids=frozenset()) -> dict:
    """Write the submission JSON (writer.format_submission layout) and return the per-query counts."""
    n_docs, n_added, n_chunks, n_cc_chunks, n_cc_docs, n_outside = [], [], [], [], [], []
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('[\n')
        for i, q in enumerate(sorted(qids)):
            docs, chunks = list(docs_by_q[q]), chunks_by_q[q]
            have = set(docs)
            missing = []
            for d, _ in chunks:
                if d not in have:
                    have.add(d)
                    missing.append(d)
            added = missing if protect else []
            outside = 0 if protect else len(missing)
            docs += added
            row = {'id': q, 'relevant_docs': docs, 'relevant_chunks': [{'doc_id': d, 'chunk_text': t} for d, t in chunks]}
            f.write(('' if i == 0 else ',\n') + json.dumps(row, ensure_ascii=False, separators=_SEP))
            n_docs.append(len(docs))
            n_added.append(len(added))
            n_chunks.append(len(chunks))
            n_cc_chunks.append(sum(d in cc_ids for d, _ in chunks))
            n_cc_docs.append(sum(d in cc_ids for d in docs))
            n_outside.append(outside)
        f.write('\n]\n')
    mean = statistics.fmean
    return {'queries': len(n_docs), 'relevant_docs_per_query': round(mean(n_docs), 4),
            'docs_added_by_protection_per_query': round(mean(n_added), 4),
            'docs_added_by_protection_pct': round(100 * sum(n_added) / sum(n_docs), 4),
            'chunk_docs_outside_relevant_docs_per_query': round(mean(n_outside), 4),
            'chunks_per_query': round(mean(n_chunks), 4), 'chunks_total': sum(n_chunks),
            'cc_chunks_per_query': round(mean(n_cc_chunks), 4), 'cc_chunks_total': sum(n_cc_chunks),
            'cc_docs_in_relevant_docs_per_query': round(mean(n_cc_docs), 4)}


def membership(json_path) -> dict:
    """corpus v2 dry run: relevant_docs ids and chunk doc_ids that are not in corpus v2 (expected 0 for vi-only content)."""
    st = {'docs': 0, 'docs_outside': 0, 'chunks': 0, 'chunks_outside': 0, 'rows_changed': 0}
    with open(json_path, encoding='utf-8', newline='') as f:
        for line in f:
            line = line.rstrip('\r\n').rstrip(',')
            if not line.startswith('{'):
                continue
            row = json.loads(line)
            dm = corpus_v2.member_mask(row['relevant_docs'])
            cm = corpus_v2.member_mask([c['doc_id'] for c in row['relevant_chunks']])
            st['docs'] += len(dm)
            st['chunks'] += len(cm)
            st['docs_outside'] += int((~dm).sum())
            st['chunks_outside'] += int((~cm).sum())
            st['rows_changed'] += int(not (dm.all() and cm.all()))
    return st


def validate(zip_path, docs_dir, log, *, allow_outside: bool) -> dict:
    """validate_submission in a subprocess (frees this process' memory first); raises unless 0 errors and 1,200 rows."""
    cmd = [sys.executable, '-m', 'r2ai.submit.validate_submission', str(zip_path), '--queries', str(QUERY_FILE),
           '--corpus', str(CORPUS_FILE), '--docs-dir', str(docs_dir), '--max-zip-mib', '100']
    if allow_outside:
        cmd.append('--allow-chunk-outside-docs')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONIOENCODING='utf-8')
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env, cwd=str(ROOT))
    Path(guard(log)).write_text(r.stdout + r.stderr, encoding='utf-8')
    last = [x for x in r.stdout.strip().splitlines() if x.startswith('{')]
    res = json.loads(last[-1]) if last else {}
    res['allow_chunk_outside_docs'] = allow_outside
    if r.returncode != 0 or res.get('errors') != 0 or res.get('ok') is not True or res.get('rows') != 1200:
        raise ValueError(f'validator failed ({r.returncode}): {(r.stdout + r.stderr)[:400]}')
    return res


def finish(name, json_path, *, method, docs_dir, allow_outside, counts, extra=None, probe_only=False, date_time=None, reuse_zip=False,
           lzma_dict_mib=None, ceiling=None) -> dict:
    """zip -> size check -> Python zipfile read-back SHA -> validator (skipped when probe_only or over the ceiling)."""
    json_path = Path(json_path)
    stem = json_path.stem
    out = json_path.parent
    t0 = time.time()
    jsha = sha256(json_path)
    ceiling = ceiling or CEILING
    mem = membership(json_path)                          # D7: stop (and report) when anything is outside corpus v2
    if mem['docs_outside'] or mem['chunks_outside']:
        raise ValueError(f'{name}: elements outside corpus v2, stopping: {mem}')
    zp = guard(out / f'{stem}.zip')
    if not (reuse_zip and zp.exists()):                 # reuse: the read-back SHA below still proves it matches this JSON
        write_submission_zip(json_path, zp, arcname=json_path.name, method=method, date_time=date_time, lzma_dict_mib=lzma_dict_mib)
    zbytes = zp.stat().st_size
    st = {'variant': name, 'json': str(json_path), 'json_bytes': json_path.stat().st_size, 'json_sha256': jsha,
          'zip': str(zp), 'zip_method': method, 'zip_bytes': zbytes, 'zip_mib': round(zbytes / 2 ** 20, 3), 'zip_sha256': sha256(zp),
          'lzma_dict_mib': lzma_dict_mib if method == 'lzma' else None, 'corpus_v2': mem,
          'ceiling_bytes': ceiling, 'within_ceiling': zbytes <= ceiling, 'zip_seconds': round(time.time() - t0, 1)} | counts | (extra or {})
    st['zip_readback'] = verify_submission_zip(zp, jsha, expected_member=json_path.name)
    if not st['zip_readback']['json_sha256_match']:
        raise ValueError(f'{zp}: read-back SHA differs from the JSON')
    if probe_only or not st['within_ceiling']:
        st['validator'] = 'not run (size probe)' if probe_only else 'not run (over the ceiling)'
    else:
        st['validator'] = validate(zp, docs_dir, out / 'validator.log', allow_outside=allow_outside)
    (out / 'stats.json').write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
    return st


# ------------------------------------------------------------------ commands

def cmd_build(a) -> int:
    roots = Roots()
    d = out_dir(a.name)
    cfg, docs_dir, _ = pool_setup(a.pool, roots)
    docs = pinned_docs()
    qids, new, texts = chunk_pool(a.pool, roots, k_cap=a.k_cap, use_cache=not a.no_cache)
    if a.k > a.k_cap:
        raise ValueError(f'--k {a.k} > --k-cap {a.k_cap}')
    from .build_b import cc_doc_ids
    cc = cc_doc_ids(roots)
    if a.mode == 'inset':
        chunks = chunk_lists_inset(qids, new, texts, a.k, docs)
    elif a.mode == 'hybrid':
        chunks = chunk_lists_hybrid(qids, new, texts, a.k, docs, a.split)
    else:
        chunks = chunk_lists(qids, new, texts, a.k)
    del texts, new
    stem = a.name.replace('-', '_')
    jp = guard(d / f'{stem}.json')
    t0 = time.time()
    counts = render(jp, qids, docs, chunks, protect=a.protect, cc_ids=cc)
    del chunks
    counts['render_seconds'] = round(time.time() - t0, 1)
    st = finish(a.name, jp, method=a.zip_method, docs_dir=docs_dir, allow_outside=not a.protect, counts=counts, probe_only=a.probe, reuse_zip=a.reuse_zip,
                lzma_dict_mib=a.lzma_dict_mib, ceiling=a.ceiling,
                extra={'k': a.k, 'pool': a.pool, 'protect': a.protect, 'mode': a.mode, 'split': a.split if a.mode == 'hybrid' else None, 'docs_source': f'cc-A-cut115 {CC_A_CUT115_SHA256[:8]}'})
    print(json.dumps(st, ensure_ascii=False, indent=1))
    return 0


READ_SCRIPT = "\n".join([
    "import hashlib, sys, zipfile",
    "h = hashlib.sha256()",
    "with zipfile.ZipFile(sys.argv[1]) as z:",
    "    with z.open(z.infolist()[0]) as f:",
    "        for b in iter(lambda: f.read(1 << 20), b''):",
    "            h.update(b)",
    "print(h.hexdigest())",
])
BSDTAR = r'C:\Windows\System32\tar.exe'


def _peak_rss(proc, box):
    """Poll the process tree's resident set (MiB) until it exits; box['peak'] keeps the maximum."""
    import psutil
    try:
        p = psutil.Process(proc.pid)
    except psutil.Error:
        return
    while proc.poll() is None:
        try:
            box['peak'] = max(box['peak'], sum(c.memory_info().rss for c in [p] + p.children(recursive=True)) / 2 ** 20)
        except psutil.Error:
            pass
        time.sleep(0.05)


def unpack_bench(zip_path, json_sha256) -> dict:
    """Extract with Python zipfile and with bsdtar (tar.exe); SHA256 of the member + peak RSS (MiB) of each reader."""
    import threading
    out = {}
    for label, cmd in (('python_zipfile', [sys.executable, '-c', READ_SCRIPT, str(zip_path)]), ('bsdtar', [BSDTAR, '-xOf', str(zip_path)])):
        box = {'peak': 0.0}
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        th = threading.Thread(target=_peak_rss, args=(proc, box), daemon=True)
        th.start()
        if label == 'bsdtar':
            h = hashlib.sha256()
            for b in iter(lambda: proc.stdout.read(1 << 20), b''):
                h.update(b)
            sha = h.hexdigest()
            proc.wait()
        else:
            so, _ = proc.communicate()
            sha = so.decode().strip()
        th.join()
        out[label] = {'sha256': sha, 'match': sha == json_sha256, 'returncode': proc.returncode, 'peak_rss_mib': round(box['peak'], 1),
                      'seconds': round(time.time() - t0, 1)}
    return out


def cmd_dict(a) -> int:
    """ZIP size at several LZMA dictionaries + unpack check (Python zipfile and bsdtar) with the peak RAM of each reader."""
    d = out_dir(a.name)
    jp = next(d.glob('*.json'))
    jsha = sha256(jp)
    rows = []
    for mib in a.dicts:
        zp = guard(d / f'{jp.stem}_d{mib}.zip')
        t0 = time.time()
        if not zp.exists():
            write_submission_zip(jp, zp, arcname=jp.name, method='lzma', lzma_dict_mib=mib if mib != 8 else None)
        row = {'dict_mib': mib, 'zip': str(zp), 'zip_bytes': zp.stat().st_size, 'zip_sha256': sha256(zp),
               'compress_seconds': round(time.time() - t0, 1), 'unpack': unpack_bench(zp, jsha)}
        rows.append(row)
        (d / 'dict_study.json').write_text(json.dumps({'json': str(jp), 'json_sha256': jsha, 'rows': rows}, indent=1), encoding='utf-8')
        print(json.dumps(row, indent=1), flush=True)
    return 0


def cmd_gate1(a) -> int:
    """K=50, protection ON, pool cc-A (vi) -> the JSON must be byte-identical to cc-A-cut115."""
    roots = Roots()
    d = out_dir('gate1')
    _, docs_dir, _ = pool_setup('cc-A', roots)
    docs = pinned_docs()
    qids, new, texts = chunk_pool('cc-A', roots, use_cache=not a.no_cache)
    chunks = chunk_lists(qids, new, texts, 50)
    del texts, new
    jp = guard(d / 'gate1.json')
    counts = render(jp, qids, docs, chunks, protect=True)
    jsha = sha256(jp)
    ok = jsha == CC_A_CUT115_SHA256
    res = {'gate': 'K=50 + protect ON + pool cc-A == cc-A-cut115', 'json_sha256': jsha, 'expected': CC_A_CUT115_SHA256, 'match': ok,
           'docs_added_by_protection_per_query': counts['docs_added_by_protection_per_query'],
           'relevant_docs_equal': read_docs(jp) == docs}
    (d / 'gate.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    return 0 if ok else 1


def calibrate_tau_a1(docs_by_query, scores_by_query, *, target_mean, floor=20, ceil=150, protect=None):
    """Bisect an absolute A1 threshold to the target mean of kept primary docs/query (port of the S7 upload-probe routine)."""
    from r2ai.submit.doc_cut import cut_a1
    qids = list(docs_by_query)
    protect = protect or {}
    values = [float(scores_by_query[q][d]) for q in qids for d in docs_by_query[q]]
    lo, hi = min(values) - 1.0, max(values) + 1.0

    def mean_at(tau):
        return sum(len(cut_a1(docs_by_query[q], scores_by_query[q], tau, floor, ceil, protect.get(q, ()))[0]) for q in qids) / len(qids)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mean_at(mid) > target_mean else (lo, mid)
    center = (lo + hi) / 2
    cands = {round(center + off / 100, 2) for off in range(-2, 3)}
    return min(cands, key=lambda tau: (abs(mean_at(tau) - target_mean), -mean_at(tau), tau))


def cmd_gate2(a) -> int:
    """S7 semantics: RRF chunks at K -> A1-115 cut recalibrated with chunk-doc protection -> expand -> VS-append slug -> LZMA.

    The JSON must equal the one decompressed from the S7 K80 ZIP (`--k 80`)."""
    import numpy as np
    from r2ai.dupes import variants as V
    from r2ai.submit import doc_cut as C
    from r2ai.submit import rrf_chunks as R
    from r2ai.submit import rrf_docs as RD
    from r2ai.submit.rrf_docs import rrf_doc_submission
    from r2ai.submit.slug_append import slug_append_submission
    import pyarrow.parquet as pq
    roots = Roots()
    d = out_dir(f'gate2-K{a.k}')
    cfg, docs_dir, _ = pool_setup('VS-append', roots)
    s, rds, sla, exp = cfg['submission'], cfg['postprocess']['rrf_doc_set'], cfg['postprocess']['slug_append'], cfg['postprocess']['expand_clusters']
    expand = Path(_p(GATE_EXPAND))
    if sha256(expand) != s['expand_json_sha256']:
        raise ValueError('expand JSON differs from the config hash')
    qids, new, texts = chunk_pool('VS-append', roots, use_cache=not a.no_cache)
    base_docs = read_docs(expand)
    rrf_json = guard(d / f'rrf_K{a.k}.json')
    render(rrf_json, qids, base_docs, chunk_lists(qids, new, texts, a.k), protect=False)
    del texts, new
    res = {'rrf_json_sha256': sha256(rrf_json)}
    qs = sorted(qids)
    rr, hy, sc = R.ranks(_p(rds['reranker_cache'])), R.ranks(_p(rds['hybrid_ranks'])), C.pool_scores(_p(rds['reranker_cache']))
    pool = {q: sorted(rr[q], key=rr[q].get) for q in qs}
    docs = {q: RD.rrf_docs(pool[q], rr[q], hy[q], int(rds['k_docs']), int(rds['k'])) for q in qs}
    groups = V.load_groups(_p(s['chunks_dir']))
    of, members = V.load_clusters(_p(exp['clusters']), exp.get('scope', 'content'))
    cdocs = read_chunk_doc_ids(rrf_json)

    def covers(x):
        ids = {int(i) for i in groups[x]}
        c = of.get(x)
        if c is not None:
            ids |= {int(i) for m in members[c] for i in groups.get(m, [m])}
        return ids
    prot = {q: {x for x in docs[q] if cdocs[q] & covers(x)} for q in qs}
    tau = calibrate_tau_a1(docs, sc, target_mean=115, floor=20, ceil=150, protect=prot)
    cut = C.make_cut('a1', tau=tau, floor=20, ceil=150, protect_chunks=True)
    args = s['args']
    arg = lambda k: args[args.index(k) + 1]
    mid_dir = guard(d / '_stage')
    mid_dir.mkdir(exist_ok=True)
    st1 = rrf_doc_submission(str(rrf_json), mid_dir / 'mid.zip', chunks='base', runs_dir=_p(s['runs_dir']), doc_ranking=_p(s['doc_ranking']),
                             hybrid_ranks=_p(rds['hybrid_ranks']), rerank_run=_p(rds['reranker_cache']), queries=_p(cfg['queries']),
                             chunks_dir=_p(s['chunks_dir']), docs_dir=_p(s['docs_dir']), clusters=_p(exp['clusters']),
                             scope=exp.get('scope', 'content'), k_cache=int(arg('--k-doc')), k_total=int(arg('--k-doc-total')),
                             k_docs=int(rds['k_docs']), k_chunk=a.k, rrf_k=int(rds['k']), cut=cut)
    final_zip = mid_dir / 'slug.zip'
    slug_append_submission(mid_dir / 'mid.json', final_zip, pin=_p(sla['pin']), pin_sha256=sla['pin_sha256'], queries=_p(cfg['queries']),
                           k=int(sla.get('k', 20)), k1=float(sla['k1']), b=float(sla['b']), scan_depth=int(sla['scan_depth']))
    final_json = guard(d / f'sub_K{a.k}.json')
    os.replace(mid_dir / 'slug.json', final_json)
    for f in mid_dir.iterdir():
        f.unlink()
    mid_dir.rmdir()
    jsha = sha256(final_json)
    res |= {'tau': tau, 'doc_cut': st1.get('doc_cut'), 'json_sha256': jsha, 'json_bytes': final_json.stat().st_size}
    if a.k == 80:
        res |= {'expected_json_sha256': S7_K80_JSON_SHA256, 'json_matches_S7_K80': jsha == S7_K80_JSON_SHA256}
    (d / 'gate.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1), flush=True)
    if a.k == 80 and jsha != S7_K80_JSON_SHA256:
        return 1
    st = finish(f'gate2-K{a.k}', final_json, method='lzma', docs_dir=docs_dir, allow_outside=False,
                counts={'k': a.k}, probe_only=a.probe, date_time=(2026, 10, 13, 3, 48, 0) if a.k == 80 else None)
    if a.k == 80:
        st['zip_sha256_matches_S7_K80_LZMA'] = st['zip_sha256'] == S7_K80_ZIP_SHA256
        (d / 'stats.json').write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(st, ensure_ascii=False, indent=1))
    return 0


def read_chunk_doc_ids(path) -> dict[int, set[int]]:
    """{query id: {chunk doc_id}} of a one-row-per-line submission JSON."""
    out = {}
    with open(path, encoding='utf-8', newline='') as f:
        for line in f:
            line = line.rstrip('\r\n').rstrip(',')
            if not line.startswith('{'):
                continue
            row = json.loads(line)
            out[int(row['id'])] = {int(c['doc_id']) for c in row['relevant_chunks']}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--name', required=True)
    b.add_argument('--pool', choices=('B', 'cc-A', 'VS-append'), default='B')
    b.add_argument('--k', type=int, required=True, help='docs that get a full chunk (first K of the pool order)')
    b.add_argument('--no-protect', dest='protect', action='store_false', help='do not add chunk docs missing from relevant_docs (risk: official format)')
    b.add_argument('--zip-method', choices=METHODS, default='deflate')
    b.add_argument('--probe', action='store_true', help='size probe: zip but no validator')
    b.add_argument('--reuse-zip', action='store_true', help='keep an existing zip of this name (read-back SHA still checked)')
    b.add_argument('--lzma-dict-mib', type=int, default=None, help='LZMA dictionary (default: stdlib 8 MiB)')
    b.add_argument('--ceiling', type=int, default=None, help=f'ZIP byte limit (default {CEILING})')
    b.add_argument('--k-cap', type=int, default=K_CAP, help='chunk docs per query held in the pool cache')
    b.add_argument('--mode', choices=('all', 'inset', 'hybrid'), default='all',
                   help='all: first K docs (S6-a..e); inset: only docs already in relevant_docs (S6-f); hybrid: ranks <= --split as all, deeper only in-set (S6-g)')
    b.add_argument('--split', type=int, default=80)
    b.add_argument('--no-cache', action='store_true')
    dd = sub.add_parser('dict')
    dd.add_argument('--name', required=True, help='an existing s6 variant directory holding the JSON')
    dd.add_argument('--dicts', type=int, nargs='+', default=[64, 128, 256])
    g1 = sub.add_parser('gate1')
    g1.add_argument('--no-cache', action='store_true')
    g2 = sub.add_parser('gate2')
    g2.add_argument('--k', type=int, default=80)
    g2.add_argument('--probe', action='store_true')
    g2.add_argument('--no-cache', action='store_true')
    a = ap.parse_args(argv)
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    assert_writable(S6_RUN)
    return {'build': cmd_build, 'gate1': cmd_gate1, 'gate2': cmd_gate2, 'dict': cmd_dict}[a.cmd](a)


if __name__ == '__main__':
    raise SystemExit(main())
