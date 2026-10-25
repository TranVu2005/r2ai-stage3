"""Hand-off of the two final 120A builds to S30 (S35, 26/10): fixed configs, every input pinned by SHA256, the tie rules, the ZIPs.

    PYTHONPATH=src python scripts/s36/handoff_s30.py            # writes out/runs/s36-fuse4/handoff-s30/{manifest.json, *.config.json}
    PYTHONPATH=src python scripts/s36/handoff_s30.py --verify   # also re-runs `apply` from the fixed configs and compares docsel / carriers SHA256
    PYTHONPATH=src python scripts/s36/handoff_s30.py --verify --out DIR   # release check: writes only below DIR, leaves configs/ and the pinned hand-off alone

The two builds: ``120A-p4b-v3`` (main: EF2-cut-p4b frame, pool = cosine top-n_q of the union) and ``120A-free-v2`` (general: free frame, pool =
corpus cosine top-300). The fixed configs (``configs/s36-handoff-*.json``) list every input file explicitly (no globbing).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from r2ai import paths
from r2ai.s36 import cal, fuse, s37

RUNS = paths.RUNS_DIR
S36 = RUNS / "s36-fuse4"
OUT = S36 / "handoff-s30"
COMPOSE = RUNS / "s38-chunk-metric-v2" / "compose"
WT = Path("D:/GitHub/r2ai-stage3/.superpowers/worktrees")
BUILDS = {
    "120A-p4b-v3": {"config": "configs/s36-s37-120A-p4b-v3.json", "role": "main", "frame": "EF2-cut-p4b (pool = cosine top-n_q, n_q = round(1.3 K_q))",
                    "compose": "120A-p4b-v3-full+w768-t1536", "lb": None},
    "120A-free-v2": {"config": "configs/s36-s37-120A-free-v2.json", "role": "general", "frame": "free (pool = corpus cosine top-300)",
                     "compose": "120A-free-v2-full+w768-t1536", "lb": {"final": 0.4101, "doc": 0.4402, "chunk": 0.3800}},
}
TIE_RULES = {
    "120A-p4b-v3": ("The pool of a query is the first n_q rows of its scored union sorted by (-cosine_score, order). `order` is a tuple: (0, old pool13 cosine_rank) "
                    "for the S29/S31 pool13 docs, (1, pair_id) for the pool13 pairs evicted by the Pharmacity merge, (2, pair_id) for the S31 pool-extra pairs, "
                    "(3, pair_id) for the S37 / S36 pairs (pair_id ascending: S37 1,000,000+, final-4b 5,000,000+). Docs with exactly the same cosine "
                    "(duplicate content under several doc_ids) therefore keep the pool13 doc first, then evicted, extra, new pairs by ascending pair_id. "
                    "A new pair whose doc is already in the query's union is dropped. n_q = the size of the old pool13 of the query. Rule code: "
                    "r2ai.s36.s37.merge_union + cosine13_pools."),
    "120A-free-v2": ("The pool of a query is the corpus cosine top-300 in the corpus order of r2ai.s36.d300.merged_top: the canonical + Pharmacity cosines of "
                     "out/runs/s37/pool.npz (top-400, stored cosine-descending order) followed by the 120ask segments (sorted segment name, then vector "
                     "file order), stable argsort by -cosine: on an exact cosine tie the canonical doc comes first (pool.npz order), then 120ask docs "
                     "by segment name and file order. The union rows are reordered to this order (r2ai.s36.s37.fixed_pools(union, 300, order)); every "
                     "doc of the corpus top-300 must have a 4B score (error otherwise). 25/10: 20 queries (913 docs per side) differed before this rule "
                     "(tie blocks of duplicate-content docs straddling rank 300); now 0."),
}


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def git(wt: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(wt), *args], capture_output=True, text=True, check=True).stdout.strip()


def base_inputs() -> dict[str, str]:
    """SHA256 of the S36 base data `apply` reads besides the config's files (union, Jina S35 / S36, C-max S32 / S36, calibration, pool, queries)."""
    _, _, _, _, _, files = fuse.load_frames()
    inputs = cal._inputs()
    found = {f"s31:{k}": Path(v) for k, v in files.items()}
    found.update({"s35_jina_scores": fuse.s35_jina_scores(), "s32_doc_scores": fuse.s32_doc_scores(),
                  "s36_extra_jina": S36 / "extra" / "jina-scores.parquet", "s36_extra_cmax": S36 / "extra" / "cmax-scores.parquet",
                  "cal.json": cal.cal_root() / "cal.json", "ef2-free.json": cal.cal_root() / "ef2-free.json",
                  "pool.npz": RUNS / "s37" / "pool.npz", "query_vectors.npz": RUNS / "s29-qwen-judge" / "query_vectors.npz",
                  "baseline_submission": Path(inputs["baseline"]), "p1_submission": Path(inputs["p1"]),
                  "canonical_docs.parquet": Path(inputs["canonical"]) / "docs.parquet", "s37_handoff_manifest": RUNS / "s37" / "handoff" / "manifest.json"})
    return {k: sha(v) for k, v in found.items() if Path(v).exists()}


def main() -> None:
    verify = "--verify" in sys.argv
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT      # --out: a scratch run, the repo configs are not rewritten
    write_configs = "--out" not in sys.argv
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"schema": "r2ai.s36.handoff-s30.v1", "builds": {}, "base_inputs_sha256": base_inputs(),
                      "code": {"s36": {"branch": "feat/s36-fuse4", "commit": git(WT / "s35-part-jina", "rev-parse", "HEAD")},
                               "s38": {"branch": "feat/s38-chunk-metric-v2", "commit": git(WT / "s38-chunk-metric-v2", "rev-parse", "HEAD"),
                                       "compose_commit_used": git(WT / "s38-dev", "rev-parse", "HEAD")}},
                      "s37_manifest_sha256": sha(RUNS / "s37" / "handoff" / "manifest.json"),
                      "compose": {"command": "cd <s38 worktree>; PYTHONPATH=src python -m r2ai.s38.compose run --name <name>-full+w768 --docsel <docsel> --rule full+w768 "
                                              "--carriers <carriers> --fit-carriers --rounds 10; then `compose validate --name <name>-full+w768 --out <compose dir>`",
                                  "rule": "full+w768", "ceiling_bytes": 102_600_000, "T": 1536}}
    for name, b in BUILDS.items():
        root = Path(__file__).resolve().parents[2]
        cfg = json.loads((root / b["config"]).read_text(encoding="utf-8"))
        apply = json.loads((S36 / "s37" / f"{name}.apply.json").read_text(encoding="utf-8"))
        fixed = {k: v for k, v in cfg.items() if k != "frozen"}
        fixed["cmax"] = fixed.get("cmax") or sorted(apply["inputs"]["cmax"])
        fixed["scores_4b"] = sorted(apply["inputs"]["scores_4b"])
        fixed["pairs_new"] = fixed["pairs_new"] if isinstance(fixed["pairs_new"], list) else [fixed["pairs_new"]]
        pins = {kind: {Path(f).name if kind != "scores_4b" else "/".join(Path(f).parts[-4:-3] + Path(f).parts[-1:]): h for f, h in files.items()}
                for kind, files in apply["inputs"].items()}
        compose = json.loads((COMPOSE / f"{b['compose']}.compose.json").read_text(encoding="utf-8"))
        prepare = json.loads((COMPOSE / f"{name}-full+w768.prepare.json").read_text(encoding="utf-8"))
        outputs = {k: v for k, v in apply["outputs"].items() if not k.endswith((".stdout", ".stderr"))}
        manifest["builds"][name] = {
            "role": b["role"], "frame": b["frame"], "lb": b["lb"], "fixed_config": f"configs/s36-handoff-{name}.json", "tie_rule": TIE_RULES[name],
            "inputs_sha256": pins, "apply_outputs_sha256": outputs, "apply_report": f"out/runs/s36-fuse4/s37/{name}.apply.json",
            "predicted": apply["predicted"], "docs_per_query": apply["docs_per_query"], "carriers": apply["carriers"], "pool": apply["pool"],
            "missing_scores": {k: {"pairs": v["pairs"], "queries": v["queries"]} for k, v in apply["missing_scores"].items()},
            "zip": {"path": compose["zip"], "bytes": compose["zip_bytes"], "sha256": compose["zip_sha256"], "T": compose["threshold"], "windows": compose["second_chunks"],
                    "chunks": compose["chunks"], "docs": compose["docs"], "carriers_after_thin": prepare["carriers"], "thin": prepare["thin"],
                    "validator": {k: v for k, v in compose["validator"].items() if k != "extra_sources"}, "validator_extra_segments": len(compose["validator"]["extra_sources"])}}
        (out / f"{name}.config.json").write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if write_configs:
            (root / "configs" / f"s36-handoff-{name}.json").write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if verify:
            scratch = out / f"verify-{name}"
            paths.assert_writable(scratch)
            s37.apply({**fixed, "out": str(scratch)})
            same = {f: sha(scratch / f) == outputs[f] for f in outputs if (scratch / f).exists() and not f.startswith("missing")}
            manifest["builds"][name]["verify_apply_same_sha256"] = same
            assert all(same.values()) and len(same) >= 3, (name, same)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out / "manifest.json"), "sha256": sha(out / "manifest.json"),
                      "verify": {n: all(b.get("verify_apply_same_sha256", {}).values()) for n, b in manifest["builds"].items()} if verify else None}, indent=1))


if __name__ == "__main__":
    main()
