"""S40 rehearsal metric (Vu 26/10; reads the public FREE-noJina outputs ON PURPOSE, never an input of r2ai.newq): expected-F2 gap between the
rehearsal docsel and the FREE-noJina docsel of the same queries.

(a) predicted macro Doc F2 of each docsel under its own calibration (p_cal, G_q of its own pool), over the same queries, and each docsel scored
    under the OTHER side's p_cal / G_q (a doc the other pool never scored counts p_cal 0);
(b) sum of p_cal of the docs only one side selected, and where they sit relative to the per-query cut probability F2_q / 5
    (EF2 adds a doc only while its p_cal > F2_q / 5: differences should be docs near that threshold).
Pass if |gap| <= 0.002.

Usage: expected_gap.py <run> <build name> <id-map parquet> <id-map sha256> [out.json]"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2ai import paths  # noqa: E402

run, name, id_map, id_sha = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
if not id_map.exists() or hashlib.sha256(id_map.read_bytes()).hexdigest() != id_sha:
    raise SystemExit(f"{id_map}: missing or SHA256 differs from {id_sha}")
REF = paths.RUNS_DIR / "s40" / "abl" / "apply"
REF_NAME = "ABL-noJina-free-v2"
m = pd.read_parquet(id_map)
to_new = dict(zip(m.orig_id.astype(int), m.id.astype(int)))


def side(apply_dir: Path, nm: str, remap: bool) -> dict:
    d = pd.read_parquet(apply_dir / f"docsel-{nm}.parquet", columns=["query_id", "doc_id"])
    p = pd.read_parquet(apply_dir / f"pcal-{nm}.parquet", columns=["query_id", "doc_id", "p_cal"])
    g = pd.read_parquet(apply_dir / f"perq-{nm}.parquet", columns=["query_id", "g", "f2"])
    if remap:
        d, p, g = (f[f.query_id.isin(to_new)].assign(query_id=lambda x: x.query_id.map(to_new)) for f in (d, p, g))
    return {"sel": {int(q): set(x.doc_id.astype(int)) for q, x in d.groupby("query_id")},
            "p": dict(zip(zip(p.query_id.astype(int), p.doc_id.astype(int)), p.p_cal.astype(float))),
            "g": dict(zip(g.query_id.astype(int), g.g.astype(float))), "f2": dict(zip(g.query_id.astype(int), g.f2.astype(float)))}


new = side(run / "build" / name / "apply", name, False)
old = side(REF, REF_NAME, True)
qs = sorted(new["sel"])


def f2(sel: dict, model: dict) -> float:
    vals = []
    for q in qs:
        tp = sum(model["p"].get((q, d), 0.0) for d in sel[q])
        vals.append(5 * tp / (4 * model["g"][q] + len(sel[q])))
    return float(np.mean(vals))


res = {"queries": len(qs),
       "own_model": {"new": f2(new["sel"], new), "free_noJina": f2(old["sel"], old)},
       "under_new_model": {"new": f2(new["sel"], new), "free_noJina": f2(old["sel"], new)},
       "under_free_noJina_model": {"new": f2(new["sel"], old), "free_noJina": f2(old["sel"], old)}}
res["gap_own_model"] = res["own_model"]["new"] - res["own_model"]["free_noJina"]
res["gap_under_new_model"] = res["under_new_model"]["new"] - res["under_new_model"]["free_noJina"]
res["gap_under_free_noJina_model"] = res["under_free_noJina_model"]["new"] - res["under_free_noJina_model"]["free_noJina"]
only_new = [(q, d) for q in qs for d in new["sel"][q] - old["sel"][q]]
only_old = [(q, d) for q in qs for d in old["sel"][q] - new["sel"][q]]
for label, pairs, model in (("only_new", only_new, new), ("only_free_noJina", only_old, old)):
    p = np.array([model["p"][k] for k in pairs]) if pairs else np.zeros(0)
    thr = np.array([model["f2"][q] / 5 for q, _ in pairs]) if pairs else np.zeros(0)
    rel = p / thr if len(p) else p
    res[label] = {"docs": len(pairs), "sum_p_cal": float(p.sum()), "mean_p_cal": float(p.mean()) if len(p) else 0.0,
                  "p_over_threshold_p10_p50_p90": [float(x) for x in np.quantile(rel, [.1, .5, .9])] if len(p) else [],
                  "share_within_0.5x_to_2x_threshold": float(((rel > 0.5) & (rel < 2)).mean()) if len(p) else 0.0}
res["pass_abs_gap_le_0.002"] = bool(max(abs(res["gap_own_model"]), abs(res["gap_under_new_model"]), abs(res["gap_under_free_noJina_model"])) <= 0.002)
print(json.dumps(res, indent=1))
if len(sys.argv) > 5:
    Path(sys.argv[5]).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
