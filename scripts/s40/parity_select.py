"""S40 regression (reads the public free-v2 inputs ON PURPOSE): r2ai.newq.select (PoolFrame, constants of configs/s40-newq-free-v2.json) on
the exact pools / Jina / C-max lookups that r2ai.s36.s37.apply builds for 120A-free-v2 must give the same keep and the same RRF order as
the hand-off (keep-120A-free-v2.json SHA256 eb8989e0...). Usage: parity_select.py [out.json]"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2ai import newq  # noqa: E402
from r2ai.s36 import cal, d300, fuse, s37  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
spec = json.loads((ROOT / "configs" / "s36-handoff-120A-free-v2.json").read_text(encoding="utf-8"))
cfg = json.loads((ROOT / "configs" / "s40-newq-free-v2.json").read_text(encoding="utf-8"))
# --- the pool / source part of s37.apply, unchanged
ranked, union, _, _, _, _ = fuse.load_frames()
scores_4b = s37.load_scores(spec["scores_4b"], "pair_id", "score")
pairs = s37.read_pairs(spec["pairs_new"], scores_4b)
union = s37.merge_union(union, pairs, scores_4b, float(spec["min_4b_coverage"]))
pairs = pairs[pairs.pair_id.astype(int).isin(scores_4b)]
pools, _ = s37.fixed_pools(union, 300, d300.merged_top(300))
jina_base, _ = fuse.load_jina(fuse.run_root() / "extra" / "jina-scores.parquet")
s37_jina = pd.read_parquet(spec["jina"], columns=["query_id", "doc_id", "score"])
jina = s37.assemble_jina(jina_base, dict(zip(zip(s37_jina.query_id.astype(int), s37_jina.doc_id.astype(int)), s37_jina.score.astype(float))))
cmax, _ = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")
by_pair = dict(zip(pairs.pair_id.astype(int), zip(pairs.query_id.astype(int), pairs.doc_id.astype(int))))
keyed = [f for f in spec["cmax"] if "pair_id" not in pq.read_schema(f).names]
for pid, value in s37.load_scores([f for f in spec["cmax"] if f not in keyed], "pair_id", "cmax").items():
    if pid in by_pair:
        cmax[by_pair[pid]] = value
for f in keyed:
    fr = pd.read_parquet(f, columns=["query_id", "doc_id", "cmax"])
    cmax.update(zip(zip(fr.query_id.astype(int), fr.doc_id.astype(int)), fr.cmax.astype(float)))
# --- reference: cal.Frame as apply calls it; candidate: newq.select
import copy  # noqa: E402
ref_frame = cal.Frame(fuse.FOUR_KEYS, "free", data={"pools": copy.deepcopy(pools), "union": union, "jina": jina, "cmax": cmax}, allow_missing=True)
ref = cal.Calibrated(ref_frame, "p4b", cfg["a"], cfg["b"], cfg["s_free"], use_tail=True, g_depth=cfg["g_depth"])
ref_keep = cal.ef2_keep(ref)
got = newq.select(copy.deepcopy(pools), jina, cmax, cfg, fuse.FOUR_KEYS, True)
want = json.loads(Path("D:/GitHub/r2ai-stage3/out/runs/release/120A-free-v2/apply/keep-120A-free-v2.json").read_text(encoding="utf-8"))
same_order = all([int(r["doc_id"]) for r in ref_frame.order[q]] == [int(r["doc_id"]) for r in got["frame"].order[q]] for q in ref_frame.qids)
keep_json = json.dumps({str(q): int(k) for q, k in got["keep"].items()}, indent=1, ensure_ascii=False, default=str) + "\n"
res = {"queries": len(got["keep"]), "keep_equal_cal_frame": got["keep"] == ref_keep, "keep_equal_handoff": {str(q): k for q, k in got["keep"].items()} == want,
       "rrf_order_equal": same_order, "predicted_f2": got["calibrated"].expected(got["keep"])["f2"], "predicted_f2_ref": ref.expected(ref_keep)["f2"]}
print(json.dumps(res, indent=1))
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
