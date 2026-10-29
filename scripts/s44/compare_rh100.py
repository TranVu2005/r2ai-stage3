"""S44: compare the rebuilt rh100 runs (argv[1]) with the recorded S41 runs: ledger SHA256 of every run file and the ZIP SHA256."""
import json
import os
import sys
from pathlib import Path

# ledger.json of the recorded runs; S44 used a copy (S44_REF) because out/runs/s41 was held during the check
REF = Path(os.environ.get("S44_REF", "D:/GitHub/r2ai-stage3/out/runs/s41"))
EXPECT = {
    ("rh100-final", "rh100-final-free"): "57dca1798da23ea14a7473bad3f4965ec0112b5b1aa59668d7771e004faa57b5",
    ("rh100-cosk150", "rh100-cosk150-free-cosk150"): "23d1d3ae62997627a8f29252f78da7ad2012e23c6ba512ded7874d7bdc1f115e",
}
EXPECT_EN = {("rh100-cosk150", "rh100-cosk150-free-cosk150-en3"): "f13ee54c924905d6981a208f668397095a2992f918d78096834fbc71077f8108"}


def build_json(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.JSONDecoder().raw_decode(text)[0]   # build.json is rewritten in place; read the first object


out = Path(sys.argv[1])
res = {"ledger": {}, "zip": {}, "ok": True}
for run in ("rh100-final", "rh100-cosk150"):
    new = json.loads((out / run / "ledger.json").read_text(encoding="utf-8"))
    ref = json.loads((REF / run / "ledger.json").read_text(encoding="utf-8"))
    diff = {k: [ref.get(k), new.get(k)] for k in sorted(set(ref) | set(new)) if ref.get(k) != new.get(k)}
    res["ledger"][run] = {"files": len(new), "differ": diff}
    res["ok"] &= not diff
for build_dir in sorted(out.glob("*/build/*")):
    run, name = build_dir.parent.parent.name, build_dir.name
    b = build_json(build_dir / "build.json")
    z = b.get("zip") or {}
    row = {"zip_sha256": z.get("zip_sha256"), "zip_bytes": z.get("zip_bytes"), "validator_ok": (z.get("validator") or {}).get("ok")}
    if "en_hedge" in b:
        row["en_zip_sha256"] = b["en_hedge"].get("zip_sha256")
        row["en_under_ceiling"] = b["en_hedge"].get("under_ceiling")
    if (run, name) in EXPECT:
        row["expected"] = EXPECT[(run, name)]
        row["match"] = row["zip_sha256"] == row["expected"]
        res["ok"] &= row["match"]
    if (run, name) in EXPECT_EN:
        row["expected_en"] = EXPECT_EN[(run, name)]
        row["match_en"] = row.get("en_zip_sha256") == row["expected_en"]
        res["ok"] &= row["match_en"]
    res["ok"] &= bool(row["validator_ok"])
    res["zip"][f"{run}/{name}"] = row
print(json.dumps(res, indent=1))
(out / "compare.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
