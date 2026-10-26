"""One-command build of a final submission ZIP (S35, 26/10): ``apply`` -> ``compose`` -> validator S26 -> ZIP, checked against the hand-off manifest.

    python -m r2ai.release build --config configs/s36-handoff-120A-p4b-v3.json     # main (EF2-cut-p4b frame)
    python -m r2ai.release build --config configs/s36-handoff-120A-free-v2.json    # general (free frame, D = 300)
    python -m r2ai.release check                                                   # environment + pinned inputs of both builds, no build

``build`` steps (each timed in ``<work>/release-report.json``):

1. ``env``      Python / package versions against the ``pyproject.toml`` pins; manifest SHA256 against ``MANIFEST_SHA256``.
2. ``inputs``   SHA256 of every input file named in the config and of the S36 base data, against the manifest (a mismatch stops the build).
3. ``apply``    ``r2ai.s36.s37.apply`` (pool, 4-source RRF, EF2 cut, CR-fair carriers) into ``<work>/apply``; docsel / carriers / pcal / keep SHA256
                against the manifest.
4. ``prepare``  ``r2ai.s38.compose prepare`` (chunk texts, thin carriers).
5. ``write``    ``r2ai.s38.compose write --threshold T`` with the T of the manifest (``--search-t`` runs the full T search of
                ``compose run --fit-carriers`` instead and compares T), then the validator S26 (``--max-chunks-per-doc 2`` and the 120ask
                segments as extra verbatim sources).
6. ``compare``  ZIP SHA256, size, windows, chunks, docs against the manifest.

Nothing is submitted; the ZIP stays below ``<work>/compose``. The exit code is 0 only when every check matches the manifest byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from r2ai import paths

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs" / "s36-handoff-s30-manifest.json"
MANIFEST_SHA256 = "45ce6e493bc275aa71bd6bfda646de5c4ac5314d375866e19a58666db963e1f7"
CONFIGS = {"120A-p4b-v3": ROOT / "configs" / "s36-handoff-120A-p4b-v3.json", "120A-free-v2": ROOT / "configs" / "s36-handoff-120A-free-v2.json"}
RULE = "full+w768"
BYTE_CRITICAL = ("pyarrow", "numpy", "pandas", "polars")      # packages whose version can change a parquet / ZIP byte


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def pinned_versions() -> dict[str, str]:
    """``name -> version`` of the ``==`` pins of pyproject.toml (project dependencies and the optional groups)."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r'"([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([^"\s]+)"', text)}


def check_env() -> dict[str, Any]:
    have, wrong = {}, {}
    for name, version in pinned_versions().items():
        try:
            got = metadata.version(name)
        except metadata.PackageNotFoundError:
            got = None
        have[name] = got
        if got is None or got.split("+")[0] != version.split("+")[0]:
            wrong[name] = {"pinned": version, "installed": got}
    critical = {k: v for k, v in wrong.items() if k in BYTE_CRITICAL}
    return {"python": platform.python_version(), "platform": platform.platform(), "pin_mismatches": wrong, "byte_critical_mismatches": critical,
            "ok": not critical}


def manifest_sha256(path: Path = MANIFEST) -> str:
    """SHA256 of the manifest with Windows (CRLF) line endings: the number given to S30 (45ce6e49...). Git stores the file with LF (e5010f99...), so
    the digest is taken over the CRLF form on every checkout."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> dict[str, Any]:
    got = manifest_sha256()
    if got != MANIFEST_SHA256:
        raise SystemExit(f"manifest {MANIFEST} SHA256 (CRLF form) {got} != {MANIFEST_SHA256}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify_inputs(cfg: dict[str, Any], entry: dict[str, Any], base: dict[str, str]) -> dict[str, Any]:
    """Every file the config names against ``inputs_sha256`` of the manifest; the S36 base data against ``base_inputs_sha256``."""
    problems, checked = [], 0
    kinds = {"pairs_new": cfg["pairs_new"], "cmax": cfg["cmax"], "scores_4b": cfg["scores_4b"], "jina": [cfg["jina"]]}
    for kind, files in kinds.items():
        pins = entry["inputs_sha256"][kind]
        by_name = {(Path(f).name if kind != "scores_4b" else "/".join(Path(f).parts[-4:-3] + Path(f).parts[-1:])): f for f in files}
        if set(by_name) != set(pins):
            problems.append(f"{kind}: files {sorted(by_name)} != pinned {sorted(pins)}")
            continue
        for key, f in by_name.items():
            checked += 1
            if not Path(f).exists():
                problems.append(f"{kind}/{key}: missing {f}")
            elif sha256(f) != pins[key]:
                problems.append(f"{kind}/{key}: SHA256 differs")
    sys.path.insert(0, str(ROOT / "scripts" / "s36"))
    import handoff_s30                                      # noqa: E402  (base_inputs: the S36 base data)

    now = handoff_s30.base_inputs()
    for key, want in base.items():
        checked += 1
        if now.get(key) != want:
            problems.append(f"base {key}: {'missing' if key not in now else 'SHA256 differs'}")
    return {"checked": checked, "problems": problems, "ok": not problems}


def run(cmd: list[str], log: Path) -> int:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    with open(log, "ab") as f:
        f.write(("$ " + " ".join(cmd) + "\n").encode())
        f.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def build(config: Path, work: Path | None, search_t: bool, rounds: int, drop: int) -> dict[str, Any]:
    cfg = json.loads(config.read_text(encoding="utf-8"))
    name = cfg["name"]
    manifest = load_manifest()
    if name not in manifest["builds"]:
        raise SystemExit(f"{name}: not a hand-off build ({sorted(manifest['builds'])})")
    entry = manifest["builds"][name]
    work = paths.resolve_path(work) if work else paths.RUNS_DIR / "release" / name
    if not work.is_relative_to(paths.RUNS_DIR):
        raise SystemExit(f"work dir must stay below RUNS_DIR: {work}")
    paths.assert_writable(work)
    (work / "apply").mkdir(parents=True, exist_ok=True)
    (work / "compose").mkdir(parents=True, exist_ok=True)
    log = work / "release.log"
    report: dict[str, Any] = {"name": name, "config": str(config), "config_sha256": sha256(config), "manifest_sha256_crlf": MANIFEST_SHA256, "manifest_sha256_lf": sha256(MANIFEST), "steps": {}, "ok": False}
    clock = time.monotonic()

    def step(label: str, result: dict[str, Any]) -> None:
        nonlocal clock
        report["steps"][label] = {**result, "seconds": round(time.monotonic() - clock, 1)}
        clock = time.monotonic()
        (work / "release-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[{label}] {'ok' if result.get('ok') else 'FAILED'} {report['steps'][label]['seconds']}s", flush=True)
        if not result.get("ok"):
            raise SystemExit(f"step {label} failed; see {work / 'release-report.json'}")

    step("env", check_env())
    step("inputs", verify_inputs(cfg, entry, manifest["base_inputs_sha256"]))

    from r2ai.s36 import s37

    apply_dir = work / "apply"
    s37.apply({**cfg, "out": str(apply_dir)})
    outputs = {f: sha256(apply_dir / f) if (apply_dir / f).exists() else None for f in entry["apply_outputs_sha256"] if not f.startswith("missing")}
    differ = [f for f, h in outputs.items() if h != entry["apply_outputs_sha256"][f]]
    step("apply", {"ok": not differ and len(outputs) >= 3, "outputs": len(outputs), "differ": differ})

    base = f"{name}-{RULE}"
    compose = ["-m", "r2ai.s38.compose"]
    cdir = ["--out", str(work / "compose")]
    docsel, carriers = str(apply_dir / f"docsel-{name}.parquet"), str(apply_dir / f"carriers-{name}.parquet")
    rc = run([sys.executable, "-u", *compose, "prepare", "--name", base, "--docsel", docsel, "--rule", RULE, "--carriers", carriers, *cdir], log)
    step("prepare", {"ok": rc == 0, "returncode": rc})
    zip_info = entry["zip"]
    if search_t:
        rc = run([sys.executable, "-u", *compose, "run", "--name", base, "--docsel", docsel, "--rule", RULE, "--carriers", carriers,
                  "--fit-carriers", "--rounds", str(rounds), *cdir], log)
        step("write", {"ok": rc == 0, "returncode": rc, "search": True})
    else:
        rc = run([sys.executable, "-u", *compose, "write", "--name", base, "--threshold", str(zip_info["T"]), "--drop", str(drop), *cdir], log)
        step("write", {"ok": rc == 0, "returncode": rc, "threshold": zip_info["T"], "drop": drop})
    label = f"{base}-t{zip_info['T']}" if not drop else f"{base}-d{drop}-t{zip_info['T']}"
    result = json.loads((work / "compose" / f"{label}.compose.json").read_text(encoding="utf-8"))
    checks = {"zip_sha256": result["zip_sha256"] == zip_info["sha256"], "zip_bytes": result["zip_bytes"] == zip_info["bytes"],
              "windows": result["second_chunks"] == zip_info["windows"], "chunks": result["chunks"] == zip_info["chunks"], "docs": result["docs"] == zip_info["docs"],
              "validator_ok": bool(result.get("validator", {}).get("ok")), "under_ceiling": bool(result["under_ceiling"])}
    step("compare", {"ok": all(checks.values()), "checks": checks, "zip": result["zip"], "zip_sha256": result["zip_sha256"], "expected_zip_sha256": zip_info["sha256"]})
    report["ok"] = True
    (work / "release-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


# Measured rates behind the private-set plan. S40 rehearsal (26/10, 100 new queries, r2ai.newq): Kaggle 2x T4 4B 6.7 pairs/s, C-max 26.0 rows/s
# with 2.61 rows per pair, prepare 41 s, export about 3.5 min, compose 3.5 min (35 to 45 min for 1,200 queries, S40 ablations); RunPod not timed.
RATES = {
    "kaggle_4b_pairs_per_second": 6.7,
    "cmax_rows_per_second": 26.0,
    "cmax_rows_per_pair": 2.61,
    "compose_minutes_1200": (35, 45),
    "runpod_4b_pairs_per_second": None,
}
PRIVATE_PREREQUISITES = {
    "new-query pipeline": "src/r2ai/newq.py",
    "pinned private config (RRF cosine / 4B / C-max)": "configs/s40-newq-free-v2.json",
    "input pins": "configs/s40-newq-pins.json",
    "4B Kaggle notebook template": "notebooks/s36-kaggle-4b-final.ipynb",
    "4B RunPod script": "scripts/s40/run_runpod_4b.sh",
    "Kaggle / RunPod gate": "scripts/s40/gate_4b.py",
    "local GPU driver (C-max)": "scripts/s40/run_newq_gpu.ps1",
}


def private_plan(queries: int, new_queries: int, depth: int = 300) -> dict[str, Any]:
    """Dry run for the private set: what each step needs and how long it takes at the measured rates. Reads files, runs nothing.
    Pipeline (Vu 26/10): free frame D = 300, RRF over cosine / 4B / C-max (no Jina), full body + 768 window; docs/s40-private.md section 5."""
    present = {k: (ROOT / v).exists() for k, v in PRIVATE_PREREQUISITES.items()}
    if not new_queries:
        return {"case": "same 1,200 queries as the public set", "gpu": False,
                "steps": [{"step": "python -m r2ai.release build --config configs/s36-handoff-120A-<p4b-v3|free-v2>.json", "estimate_hours": 0.34,
                           "note": "CPU only; see README_RELEASE.md for the measured times"}]}
    pairs = new_queries * depth
    half = pairs / 2 / RATES["kaggle_4b_pairs_per_second"] / 3600
    cmax = pairs * RATES["cmax_rows_per_pair"] / RATES["cmax_rows_per_second"] / 3600
    steps = [
        {"step": "python -m r2ai.newq pins && python -m r2ai.newq prepare --queries <q.parquet> --run out/runs/s40/private", "where": "local", "estimate_hours": 0.05},
        {"step": "python -m r2ai.newq export-scoring --run out/runs/s40/private --parts-4b 2 --reg-pairs 1000", "where": "local CPU", "estimate_hours": 0.3,
         "note": "not timed at 1,200 queries (3.5 min for 100)"},
        {"step": f"4B part 1 ({pairs // 2:,} pairs + 1,000 gate pairs) on Kaggle 2x T4 (notebook <run>-4b-1.ipynb)", "where": "Kaggle", "estimate_hours": round(half, 1)},
        {"step": f"4B part 2 ({pairs // 2:,} pairs + the gate pairs) on RunPod RTX 4090 (scripts/s40/run_runpod_4b.sh), gate scripts/s40/gate_4b.py",
         "where": "RunPod", "estimate_hours": None, "note": "not timed yet; gate fail -> part 2 on Kaggle after part 1 (+%.1f h)" % half},
        {"step": f"C-max ({pairs * RATES['cmax_rows_per_pair']:,.0f} rows): scripts/s40/run_newq_gpu.ps1, then import-scores --source cmax", "where": "local GPU",
         "estimate_hours": round(cmax, 1)},
        {"step": "python -m r2ai.newq import-scores --source 4b --tag ... (each part); python -m r2ai.newq build --run out/runs/s40/private",
         "where": "local CPU", "estimate_hours": round(RATES["compose_minutes_1200"][1] / 60, 2)},
    ]
    return {"case": f"{new_queries:,} new queries", "gpu": True, "prerequisites_present": present,
            "critical_path_hours": round(max(half, cmax) + RATES["compose_minutes_1200"][1] / 60, 1),
            "kaggle_only_4b_hours": round(2 * half, 1), "steps": steps}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="apply -> compose -> validator -> ZIP, compared with the hand-off manifest")
    b.add_argument("--config", type=Path, required=True)
    b.add_argument("--work", type=Path, default=None, help="default RUNS_DIR/release/<name>")
    b.add_argument("--search-t", action="store_true", help="redo the full T / carrier-fit search instead of using the manifest T (hours)")
    b.add_argument("--rounds", type=int, default=10, help="--search-t: parallel build rounds")
    b.add_argument("--drop", type=int, default=0, help="carriers dropped by the fit (0 for both hand-off builds)")
    sub.add_parser("check", help="environment and pinned inputs of both builds, no build")
    pl = sub.add_parser("plan", help="private set, dry run: steps, prerequisites and estimated hours (no GPU, nothing is run)")
    pl.add_argument("--new-queries", type=int, default=0, help="0 = the same 1,200 queries; N = a new query set of N queries")
    pl.add_argument("--depth", type=int, default=300)
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps(private_plan(1200, args.new_queries, args.depth), indent=2, ensure_ascii=False))
        return 0
    if args.command == "check":
        manifest = load_manifest()
        out = {"env": check_env(), "builds": {n: verify_inputs(json.loads(c.read_text(encoding="utf-8")), manifest["builds"][n], manifest["base_inputs_sha256"])
                                              for n, c in CONFIGS.items()}}
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if out["env"]["ok"] and all(v["ok"] for v in out["builds"].values()) else 1
    report = build(args.config, args.work, args.search_t, args.rounds, args.drop)
    print(json.dumps({k: v for k, v in report.items() if k != "steps"}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
