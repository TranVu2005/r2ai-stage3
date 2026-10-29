"""S45: rebuild the public CH-rrf docsel after S44 deleted the public inputs of scripts/s41/regress_public.py.

Deleted: out/runs/s37/pool.npz and out/runs/s29-qwen-judge/query_vectors.npz (the cosine order of d300.merged_top). Still there:
the 4B / C-max / pair registries / S31-S32 frames in D:/_r2ai_archive/judge-public.tar (extracted to <restore>/out/runs), and the
CH-rrf ZIP (b12b8591...). Replacement for merged_top: ``top.parquet`` of ``python -m r2ai.newq prepare`` on the 1,200 public queries
(same canonical + Pharmacity + 120ask cosine top-300). Then, as regress_public.py: ``s37.merge_union`` + ``s37.fixed_pools(order=top)``
+ C-max + ``newq.apply_selection`` with the frozen private config.

Gate: per query, the docsel docs in docsel rank order equal ``relevant_docs`` of the CH-rrf ZIP, and the carrier set equals the doc ids
of its ``relevant_chunks``.

``--host-prior <json> --beta B`` (S45 probe P-host): before the selection, every zh pair's P4B becomes
``sigmoid(logit(P4B) + B * (logit(r_host) - logit(r_mean)) / a)``, i.e. logit(p_cal) moves by ``B * (logit r_host - logit r_mean)``
(r_host = shrunk retention of the host in the pinned table, r_mean = its prior mean; host outside the table and vi / en: no change).
``--down-only``: the shift is ``B * min(0, logit r_host - logit r_mean)``, i.e. the prior only lowers hosts kept less than the mean.
The 4B rank inside the RRF follows the moved P4B, so the doc order, G_q, the EF2 cut and the carrier flag all see the prior.

  public_rebuild.py --run out/runs/s45/pub --out out/runs/s45/pub-rebuild [--name CH-rrf] [--host-prior f.json --beta 1]
"""
import argparse, json, os, re, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = Path(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))
RESTORE = Path(os.environ.get('S45_RESTORE', MAIN / 'out/runs/s45/restore/out'))
os.environ['R2AI_OUT_DIR'] = str(RESTORE)          # RUNS_DIR of the S31-S37 frames = the extracted archive
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np, pandas as pd  # noqa: E402
from r2ai import newq, paths  # noqa: E402
from r2ai.s36 import cal, fuse, s37  # noqa: E402

ZIP = Path('D:/_r2ai_archive/zips/out/runs/s41/ch/compose/CH-rrf-full+w768-t1536.zip')
OLD_RUNS = 'D:/GitHub/r2ai-stage3/out/runs'


def remap(p):
    p = str(p).replace('\\', '/')
    return str(paths.RUNS_DIR / p[len(OLD_RUNS) + 1:]) if p.startswith(OLD_RUNS) else p


SCORE_FILES = ['s31-strong-judge/pharmacity-ext/qwen3-reranker-4b-scores-pool13.parquet', 's31-strong-judge/gpu/qwen3-reranker-4b/scores-pool13.parquet',
               's31-strong-judge/gpu/qwen3-reranker-4b/scores-extra.parquet', 's31-strong-judge/pharmacity-ext/scores/qwen3-reranker-4b/scores-all.parquet']
COSINE_FILES = ['s31-strong-judge/pool/pool_index.parquet', 's31-strong-judge/pool-extra/extra_pairs.parquet', 's37/pairs_new_final_v2.parquet',
                's36-fuse4/d300/pairs_final.parquet', 's36-fuse4/d300/pairs_d300.parquet']


def public_pool(spec, depth, newq_top=None):
    """The public pools without pool.npz: every 4B-scored (query, doc) of the S31 union files + the spec's 4B files (no score conflicts),
    ordered by the ORIGINAL cosine kept in the pair registries (S29 query vectors), top ``depth``. -> newq ``top`` frame, {pair_id: P4B}."""
    runs = paths.RUNS_DIR
    a = pd.concat([pd.read_parquet(runs / f if not Path(f).is_absolute() else f, columns=['query_id', 'doc_id', 'score'])
                   for f in SCORE_FILES + list(spec['scores_4b'])]).drop_duplicates(['query_id', 'doc_id'])
    cos = pd.concat([pd.read_parquet(runs / f, columns=['query_id', 'doc_id', 'cosine_score', 'cosine_rank']).assign(src=i)
                     for i, f in enumerate(COSINE_FILES)]).drop_duplicates(['query_id', 'doc_id']).rename(columns={'cosine_rank': 'rank0'})
    a = a.merge(cos, on=['query_id', 'doc_id'], how='left')
    if newq_top is not None:   # the 170 Pharmacity-lot pairs have no registry cosine: newq prepare's cosine (same vectors, new query encode)
        nt = newq_top[['query_id', 'doc_id', 'cosine_score']].rename(columns={'cosine_score': 'c_newq'})
        a = a.merge(nt, on=['query_id', 'doc_id'], how='left'); a['cosine_score'] = a.cosine_score.fillna(a.c_newq)
    a = a.dropna(subset=['cosine_score'])
    a['rank0'] = a.rank0.fillna(10 ** 6); a['src'] = a.src.fillna(9)
    a = a.sort_values(['query_id', 'cosine_score', 'src', 'rank0'], ascending=[True, False, True, True], kind='stable')
    a['cosine_rank'] = a.groupby('query_id').cumcount() + 1
    top = a[a.cosine_rank <= depth].reset_index(drop=True)
    top['pair_id'] = np.arange(len(top)) + 90_000_000
    return top[['query_id', 'cosine_rank', 'doc_id', 'cosine_score', 'pair_id']], dict(zip(top.pair_id.astype(int), top.score.astype(float)))


def exact_pool(spec, top_file):
    """The pools from the exact merged_top(300) of ``exact_top.py`` (S29 query vectors rebuilt byte for byte): its cosine order, the 4B
    score of each pair from the S31 union files + the spec's 4B files. -> newq ``top`` frame, {pair_id: P4B}, pairs without 4B."""
    runs = paths.RUNS_DIR
    a = pd.concat([pd.read_parquet(runs / f if not Path(f).is_absolute() else f, columns=['query_id', 'doc_id', 'score'])
                   for f in SCORE_FILES + list(spec['scores_4b'])]).drop_duplicates(['query_id', 'doc_id'])
    top = pd.read_parquet(top_file).merge(a, on=['query_id', 'doc_id'], how='left')
    top = top.sort_values(['query_id', 'cosine_rank']).reset_index(drop=True)
    top['pair_id'] = np.arange(len(top)) + 90_000_000
    have = top.dropna(subset=['score'])
    return (top[['query_id', 'cosine_rank', 'doc_id', 'cosine_score', 'pair_id']], dict(zip(have.pair_id.astype(int), have.score.astype(float))),
            int(top.score.isna().sum()))


def zip_docs():
    out, chunks = {}, {}
    with zipfile.ZipFile(ZIP) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            m = re.match(rb'\{"id":(\d+),"relevant_docs":\[([0-9,]*)\]', line)
            if m:
                q = int(m.group(1)); out[q] = [int(x) for x in m.group(2).split(b',') if x]
                chunks[q] = {int(x) for x in re.findall(rb'"doc_id":(\d+)', line)}
    return out, chunks


def host_lang(doc_ids):
    """doc_id -> (host, lang) from the corpus links (host of the URL) and the canonical language; 120ask segment docs are zh."""
    links = pd.read_parquet(MAIN / 'data/raw/links_corpus_vi_en_zh.parquet')
    links = links[links.id.isin(doc_ids)]
    host = dict(zip(links.id.astype(int), links.url.str.extract(r'^https?://([^/]+)')[0].str.lower().str.removeprefix('www.')))
    c = pd.read_parquet(MAIN / 'data/canonical-20261018/docs.parquet', columns=['doc_id', 'lang'])
    lang = dict(zip(c.doc_id.astype(int), c.lang))
    return {d: (host.get(d, '?'), lang.get(d, 'zh' if host.get(d, '').endswith('120ask.com') else 'vi')) for d in doc_ids}


def apply_prior(pools, prior, beta, a, hl, down_only=False):
    mean = prior['_default']['shrunk']; lm = cal.logit(np.array([mean]))[0]; moved = 0
    for q, rows in pools.items():
        for r in rows:
            h, lang = hl[int(r['doc_id'])]
            if lang != 'zh' or h not in prior or h.startswith('_'):
                continue
            delta = cal.logit(np.array([prior[h]['shrunk']]))[0] - lm
            if down_only:
                delta = min(0.0, delta)
            if delta == 0:
                continue
            shift = beta * delta / a
            r['judge_score'] = float(cal.sigmoid(cal.logit(np.array([r['judge_score']]))[0] + shift)); moved += 1
    from r2ai.s31 import build as s31_build
    return {q: s31_build._with_judge_rank(rows) for q, rows in pools.items()}, moved


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--name', default='CH-rrf'); ap.add_argument('--host-prior', type=Path); ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--down-only', action='store_true')
    ap.add_argument('--top', type=Path, help='exact merged_top(300) parquet of exact_top.py (instead of the registry cosines)')
    args = ap.parse_args()
    apply_dir = args.out.resolve() / 'apply'; apply_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    cfg = newq.load_config(newq.DEFAULT_CONFIG)
    spec = {**s37.default_inputs(), **json.loads((ROOT / 'configs' / 's40-abl-free-v2-noJina.json').read_text(encoding='utf-8'))}
    for k in ('pairs_new', 'scores_4b', 'cmax'):
        spec[k] = [remap(p) for p in spec[k]] if isinstance(spec[k], list) else remap(spec[k])
    if args.top:
        top, scores_4b, no_4b = exact_pool(spec, args.top)
    else:
        top, scores_4b = public_pool(spec, int(spec['depth']), pd.read_parquet(args.run / 'top.parquet') if args.run else None); no_4b = 0
    pools = newq.pools_from(top, scores_4b)
    cmax, _ = fuse.load_cmax(fuse.run_root() / 'extra' / 'cmax-scores.parquet')
    reg = pd.concat([pd.read_parquet(f, columns=['pair_id', 'query_id', 'doc_id']) for f in spec['pairs_new']])
    by_pair = dict(zip(reg.pair_id.astype(int), zip(reg.query_id.astype(int), reg.doc_id.astype(int))))
    import pyarrow.parquet as pq
    keyed = [f for f in spec['cmax'] if 'pair_id' not in pq.read_schema(f).names]
    for pid, value in s37.load_scores([f for f in spec['cmax'] if f not in keyed], 'pair_id', 'cmax').items():
        if pid in by_pair:
            cmax[by_pair[pid]] = value
    for f in keyed:
        fr = pd.read_parquet(f, columns=['query_id', 'doc_id', 'cmax'])
        cmax.update(zip(zip(fr.query_id.astype(int), fr.doc_id.astype(int)), fr.cmax.astype(float)))
    info = {'pool_pairs': int(len(top)), 'top': str(args.top) if args.top else 'registry cosine', 'pairs_without_4b': no_4b}
    res = {'name': args.name, 'load_seconds': round(time.monotonic() - t0, 1), 'pool': info}
    if args.host_prior:
        prior = json.loads(args.host_prior.read_text(encoding='utf-8'))
        hl = host_lang({int(r['doc_id']) for rows in pools.values() for r in rows})
        pools, moved = apply_prior(pools, prior, args.beta, float(cfg['a']), hl, args.down_only)
        res['host_prior'] = {'file': str(args.host_prior), 'sha256': newq.sha256(args.host_prior), 'beta': args.beta, 'down_only': args.down_only, 'pairs_moved': moved}
    got = newq.apply_selection(pools, {}, cmax, cfg, ('cosine_rank', 'judge_rank', 'cmax_rank'), True, args.name, apply_dir)
    res['carrier_info'] = got['carrier_info']; res['missing'] = got['frame'].missing
    sel = pd.read_parquet(apply_dir / f'docsel-{args.name}.parquet')
    car = pd.read_parquet(apply_dir / f'carriers-{args.name}.parquet')
    ref, ref_chunks = zip_docs()
    mine = {int(q): g.sort_values('rank').doc_id.astype(int).tolist() for q, g in sel.groupby('query_id')}
    mine_car = {int(q): set(g.doc_id.astype(int)) for q, g in car.groupby('query_id')}
    res['vs_chrrf'] = {'docs': int(len(sel)), 'ref_docs': sum(map(len, ref.values())),
                       'queries_equal_order': sum(mine.get(q) == ref[q] for q in ref),
                       'queries_equal_set': sum(set(mine.get(q, [])) == set(ref[q]) for q in ref),
                       'carrier_queries_equal': sum(mine_car.get(q, set()) == ref_chunks[q] for q in ref),
                       'only_new': sum(len(set(mine.get(q, [])) - set(ref[q])) for q in ref),
                       'only_ref': sum(len(set(ref[q]) - set(mine.get(q, []))) for q in ref)}
    res['gate_pass'] = res['vs_chrrf']['queries_equal_order'] == len(ref) and res['vs_chrrf']['carrier_queries_equal'] == len(ref)
    newq.atomic_json(args.out.resolve() / f'rebuild-{args.name}.json', res)
    print(json.dumps(res, indent=1, default=str), flush=True)


if __name__ == '__main__':
    main()
