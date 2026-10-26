"""S40 (26/10): the free frame at D = 400 on the public set. Exports what a cosine top-400 pool still needs scored; scores nothing.

    python scripts/s40/d400_export.py count     counts only: out/runs/s40/d400/counts.json
    python scripts/s40/d400_export.py export    Kaggle 4B dataset out/runs/s40/d400/kaggle-dataset-d400-4b/ + notebook d400-4b.ipynb,
                                                pairs_final.parquet (the new 4B pairs and their ids), cmax_pairs.parquet (top-400 pairs without a C-max)
    python scripts/s40/d400_export.py import --scores scores-extra.parquet --report collect-extra-report.json
                                                checks the Kaggle result like r2ai.s36.d300 import4b, stores it as
                                                out/runs/s40/d400/d400-4b/gpu/qwen3-reranker-4b/scores-extra.parquet (what configs/s40-d400-noJina.json pins)

Corpus: the one FREE-noJina was built on (canonical + Pharmacity cosine top-400 of out/runs/s37/pool.npz merged with every 120ask segment,
``r2ai.s36.d300.merged_top``), no en lane. "Scored" is the S36 union plus the 4B / C-max files that configs/s40-abl-free-v2-noJina.json pins,
so the pairs listed here are exactly the ones the D = 400 apply would miss. Same pinned scorer, revision and 512-token cut as d300 / final-4b.
Everything is written under out/runs/s40/d400 (nothing of out/runs/s36-fuse4/d300 is touched); pair ids start at 6,000,000, checked against
every existing registry. C-max: scripts/s37/d300_cmax.py --src out/runs/s40/d400/cmax_pairs.parquet --out out/runs/s37/cmax-d400 (local GPU).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s36 import d300, fuse, s37

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "s40-abl-free-v2-noJina.json"
OUT = paths.RUNS_DIR / "s40" / "d400"
DEPTH, OLD_DEPTH = 400, 300
TAG = "d400-4b"
SHARD = "pool-93005.parquet"
PAIR_ID_BASE = 6_000_000
KAGGLE_PAIRS_PER_S = 6.7          # S40 rehearsal, Kaggle 2x T4


def scored() -> dict[str, set[tuple[int, int]]]:
    """(query, doc) pairs with a 4B / C-max score in the FREE-noJina inputs (the config's files, not a directory glob)."""
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for f in cfg["pairs_new"] + cfg["scores_4b"] + cfg["cmax"]:
        if not Path(f).exists():
            raise SystemExit(f"pinned input missing: {f}")
    _, union, _, _, _, _ = fuse.load_frames()
    key_of = d300.pair_keys([Path(f) for f in cfg["pairs_new"]] + [p for p in d300.registry_files() if p.name == "pairs_d300.parquet"])
    four_b = {(q, int(r["doc_id"])) for q, rows in union.items() for r in rows}
    four_b |= {key_of[p] for p in s37.load_scores(cfg["scores_4b"], "pair_id", "score") if p in key_of}
    cmax, _ = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")
    cmax_keys = set(cmax)
    for f in cfg["cmax"]:
        frame = pd.read_parquet(f)
        if "pair_id" in frame.columns:
            cmax_keys |= {key_of[p] for p in frame.pair_id.astype(int) if p in key_of}
        else:
            cmax_keys |= set(zip(frame.query_id.astype(int), frame.doc_id.astype(int)))
    return {"4b": four_b, "cmax": cmax_keys}


def summary(top: pd.DataFrame) -> dict:
    new = top[~top.has_4b]
    deep = top.cosine_rank > OLD_DEPTH
    per_query = new.groupby("query_id").size().reindex(sorted(top.query_id.unique()), fill_value=0)
    return {"depth": DEPTH, "pairs": int(len(top)), "pairs_ask": int(top.is_ask.sum()),
            "pairs_4b_needed": int(len(new)), "pairs_4b_needed_rank_301_400": int((~top.has_4b & deep).sum()),
            "pairs_4b_needed_rank_le_300": int((~top.has_4b & ~deep).sum()), "pairs_4b_needed_ask": int(new.is_ask.sum()),
            "queries_needing_4b": int((per_query > 0).sum()),
            "per_query_p50_p90_max": [float(np.quantile(per_query, .5)), float(np.quantile(per_query, .9)), int(per_query.max())],
            "pairs_cmax_needed": int((~top.has_cmax).sum()), "pairs_cmax_needed_rank_301_400": int((~top.has_cmax & deep).sum()),
            "eta_hours_kaggle_2xT4": round(len(new) / KAGGLE_PAIRS_PER_S / 3600, 2)}


def main(cmd: str) -> dict:
    started = time.monotonic()
    paths.assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    have = scored()
    top = d300.need(d300.merged_top(DEPTH, d300.load_pool()), have)
    info = {"config": str(CONFIG), "config_sha256": _sha256(CONFIG), "scored": {k: len(v) for k, v in have.items()}, **summary(top),
            "pool_npz_sha256": _sha256(d300.RUN37 / "pool.npz")}
    _atomic_json(OUT / "counts.json", info)
    if cmd == "count":
        return info
    todo = top[~top.has_4b].sort_values(["query_id", "cosine_rank"]).reset_index(drop=True)
    todo["pair_id"] = PAIR_ID_BASE + np.arange(len(todo))
    taken = set(d300.pair_keys(d300.registry_files()))
    if any(PAIR_ID_BASE <= i < PAIR_ID_BASE + len(todo) for i in taken):
        raise SystemExit("pair_id range collides with an existing registry")
    ds = OUT / f"kaggle-dataset-{TAG}"
    man = d300.write_subset(todo, ds, TAG, SHARD, info, started, top)
    # write_subset puts the pair list / C-max list next to the subset: move them out of the Kaggle dataset folder
    for name in ("pairs_final.parquet", "cmax_pairs.parquet"):
        (ds / name).replace(OUT / name)
    (ds / TAG / "judge_score.py").replace(ds / "judge_score.py")
    from r2ai import newq

    nb = newq.notebook_for({TAG: int(man["pairs"])}, ds)
    (OUT / f"{TAG}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    man.update({"pairs_final_sha256": _sha256(OUT / "pairs_final.parquet"), "cmax_pairs_sha256": _sha256(OUT / "cmax_pairs.parquet"),
                "eta_hours_kaggle_2xT4": round(man["pairs"] / KAGGLE_PAIRS_PER_S / 3600, 2)})
    _atomic_json(ds / TAG / "manifest.json", man)
    return man


def import_scores(scores: Path, report: Path) -> dict:
    """Same checks as ``r2ai.s36.d300.import4b``: SHA in the report, model and revision, one score per exported pair, no NaN."""
    import shutil

    man = json.loads((OUT / f"kaggle-dataset-{TAG}" / TAG / "manifest.json").read_text(encoding="utf-8"))
    rep = json.loads(report.read_text(encoding="utf-8"))
    sc, ex = pd.read_parquet(scores), pd.read_parquet(OUT / f"kaggle-dataset-{TAG}" / TAG / "extra_pairs.parquet")
    sha, errors = _sha256(scores), []
    if sha != rep.get("scores_sha256"):
        errors.append(f"scores SHA256 {sha} != report {rep.get('scores_sha256')}")
    if rep.get("revisions") != [d300.REVISION_4B] or rep.get("model") != "qwen3-reranker-4b":
        errors.append(f"report model/revision {rep.get('model')} {rep.get('revisions')}")
    if len(sc) != man["pairs"] or sc.pair_id.nunique() != len(sc):
        errors.append(f"{len(sc)} rows / {sc.pair_id.nunique()} unique pair_id, expected {man['pairs']}")
    elif set(sc.pair_id) != set(ex.pair_id):
        errors.append("pair_id set differs from extra_pairs.parquet")
    elif sc.score.isna().any():
        errors.append("NaN scores")
    if errors:
        raise SystemExit("import refused: " + "; ".join(errors))
    dest = OUT / TAG / "gpu" / "qwen3-reranker-4b"
    paths.assert_writable(dest)
    if (dest / "scores-extra.parquet").exists():
        raise SystemExit(f"{dest} already holds scores: not overwritten")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scores, dest / "scores-extra.parquet")
    shutil.copyfile(report, dest / "collect-extra-report.json")
    return {"tag": TAG, "pairs": len(sc), "scores_sha256": sha, "dir": str(dest)}


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "import" and sys.argv[2] == "--scores" and sys.argv[4] == "--report":
        print(json.dumps(import_scores(Path(sys.argv[3]), Path(sys.argv[5])), indent=1))
    elif len(sys.argv) != 2 or sys.argv[1] not in ("count", "export"):
        raise SystemExit(__doc__)
    else:
        print(json.dumps(main(sys.argv[1]), indent=1, ensure_ascii=False, default=str))
