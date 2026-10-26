"""S40 rehearsal check (reads the public free-v2 inputs ON PURPOSE; never an input of r2ai.newq): which source makes the rehearsal docsel
differ from 120A-free-v2. r2ai.newq.select is run on the 100 rehearsal queries with the sources swapped one at a time from free-v2 to new:

    0 old      free-v2 pools, P4B, C-max, Jina (must give the free-v2 docsel exactly: the selection core is equivalent)
    1 pool     new top-300 pools, every score taken from free-v2 where it exists (new score for a doc free-v2 never scored)
    2 +4b      new P4B
    3 +cmax    new C-max
    4 +jina    new Jina (= the rehearsal build)

Usage: attribute_docsel.py <run> [out.json]"""
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2ai import newq, paths  # noqa: E402
from r2ai.s31 import build as s31_build  # noqa: E402
from r2ai.s36 import d300, fuse, s37  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
run = Path(sys.argv[1])
m = pd.read_parquet(paths.RUNS_DIR / "s40" / "rehearsal-input" / "id-map-rh100.parquet")
to_old = dict(zip(m.id.astype(int), m.orig_id.astype(int)))
to_new = {v: k for k, v in to_old.items()}
spec = json.loads((ROOT / "configs" / "s36-handoff-120A-free-v2.json").read_text(encoding="utf-8"))
cfg = newq.run_config(run)
# ---- free-v2 inputs, exactly as s37.apply builds them (pool fixed 300)
_, union, _, _, _, _ = fuse.load_frames()
scores_4b = s37.load_scores(spec["scores_4b"], "pair_id", "score")
pairs = s37.read_pairs(spec["pairs_new"], scores_4b)
union = s37.merge_union(union, pairs, scores_4b, float(spec["min_4b_coverage"]))
pairs = pairs[pairs.pair_id.astype(int).isin(scores_4b)]
old_ids = list(to_new)
pools_old, _ = s37.fixed_pools({q: union[q] for q in to_new}, 300, d300.merged_top(300).query("query_id in @old_ids"))
jina_base, _ = fuse.load_jina(fuse.run_root() / "extra" / "jina-scores.parquet")
s37_jina = pd.read_parquet(spec["jina"], columns=["query_id", "doc_id", "score"])
jina_old = s37.assemble_jina(jina_base, dict(zip(zip(s37_jina.query_id.astype(int), s37_jina.doc_id.astype(int)), s37_jina.score.astype(float))))
cmax_old, _ = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")
by_pair = dict(zip(pairs.pair_id.astype(int), zip(pairs.query_id.astype(int), pairs.doc_id.astype(int))))
keyed = [f for f in spec["cmax"] if "pair_id" not in pq.read_schema(f).names]
for pid, value in s37.load_scores([f for f in spec["cmax"] if f not in keyed], "pair_id", "cmax").items():
    if pid in by_pair:
        cmax_old[by_pair[pid]] = value
for f in keyed:
    fr = pd.read_parquet(f, columns=["query_id", "doc_id", "cmax"])
    cmax_old.update(zip(zip(fr.query_id.astype(int), fr.doc_id.astype(int)), fr.cmax.astype(float)))
p4b_old = {(q, int(r["doc_id"])): float(r["judge_score"]) for q in to_new for r in union[q]}
# re-key free-v2 to the rehearsal ids
pools_old = {to_new[q]: rows for q, rows in pools_old.items()}
jina_old = {(to_new[q], d): v for (q, d), v in jina_old.items() if q in to_new}
cmax_old = {(to_new[q], d): v for (q, d), v in cmax_old.items() if q in to_new}
p4b_old = {(to_new[q], d): v for (q, d), v in p4b_old.items()}
# ---- new inputs
top = pd.read_parquet(run / "top.parquet")
p4b_new_by_pair = newq.load_4b(run)
p4b_new = {(int(q), int(d)): p4b_new_by_pair[int(p)] for q, d, p in zip(top.query_id, top.doc_id, top.pair_id)}
j = pd.read_parquet(run / "jina.parquet")
jina_new = dict(zip(zip(j.query_id.astype(int), j.doc_id.astype(int)), j.score.astype(float)))
c = pd.read_parquet(run / "cmax.parquet")
cmax_new = dict(zip(zip(c.query_id.astype(int), c.doc_id.astype(int)), c.cmax.astype(float)))
want = pd.read_parquet(paths.RUNS_DIR / "release" / "120A-free-v2" / "apply" / "docsel-120A-free-v2.parquet", columns=["query_id", "doc_id"])
want = {to_new[q]: g.doc_id.astype(int).tolist() for q, g in want[want.query_id.isin(to_new)].groupby("query_id")}


def pools_new(p4b: dict) -> dict:
    out = {}
    for q, g in top.sort_values(["query_id", "cosine_rank"]).groupby("query_id"):
        rows = [{"pair_id": int(pid), "doc_id": int(d), "cosine_score": float(cs), "cosine_rank": int(r), "judge_score": p4b[(int(q), int(d))]}
                for pid, d, cs, r in zip(g.pair_id, g.doc_id, g.cosine_score, g.cosine_rank)]
        out[int(q)] = s31_build._with_judge_rank(rows)
    return out


def prefer(old: dict, new: dict) -> dict:
    return {**new, **old}                         # free-v2 score where it exists, else the new one


steps = {
    "0 old": (copy.deepcopy(pools_old), jina_old, cmax_old),
    "1 pool": (pools_new(prefer(p4b_old, p4b_new)), prefer(jina_old, jina_new), prefer(cmax_old, cmax_new)),
    "2 +4b": (pools_new(p4b_new), prefer(jina_old, jina_new), prefer(cmax_old, cmax_new)),
    "3 +cmax": (pools_new(p4b_new), prefer(jina_old, jina_new), cmax_new),
    "4 +jina": (pools_new(p4b_new), jina_new, cmax_new),
}
res = {}
for name, (pools, jina, cmax) in steps.items():
    got = newq.select(pools, jina, cmax, cfg, fuse.FOUR_KEYS, True)
    sel = {q: [int(r["doc_id"]) for r in rows] for q, rows in got["chosen"].items()}
    same_set = sum(set(sel[q]) == set(want[q]) for q in want)
    same_order = sum(sel[q] == want[q] for q in want)
    only_new = sum(len(set(sel[q]) - set(want[q])) for q in want)
    only_old = sum(len(set(want[q]) - set(sel[q])) for q in want)
    res[name] = {"queries_same_set": same_set, "queries_same_order": same_order, "docs_only_new": only_new, "docs_only_old": only_old,
                 "docs": sum(len(v) for v in sel.values()), "predicted_f2": got["calibrated"].expected(got["keep"])["f2"],
                 "missing": got["frame"].missing}
    print(name, json.dumps(res[name]), flush=True)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
