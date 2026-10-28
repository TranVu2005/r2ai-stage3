"""S44: group every inventoried path (inventory.csv, data depth 2 / out depth 3) into GIU-PRIVATE / GIU-NGUON / LUU-TRU / XOA and
write classes.csv + a per-group total. Read-only (stats the KEEP / archive files directly)."""
import csv
import os
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
REPO = "D:/GitHub/r2ai-stage3/"
sys.path.insert(0, str(S))
import hold  # noqa: E402  (KEEP list = the set the rh100 check ran with)

ARCHIVE_FILES = {   # submitted ZIPs worth keeping off the machine (path -> why)
    "out/runs/s41/ch/compose/CH-rrf-full+w768-t1536.zip": "CH-rrf, bản chốt private, LB 0,4138",
    "out/runs/s41/fallback/B0/compose/FB-B0-full+w768-t1536.zip": "B0, dự phòng 1, LB 0,4101",
    "out/runs/s41/fallback/COSK150/compose/FB-COSK150-full+w768-t1536.zip": "COS-K150, dự phòng 2, LB 0,3742",
    "out/runs/s43/probes/EN-hedge3-c5226/EN-hedge3-c5226-full+w768-t1536.zip": "EN-hedge, bản 5, LB 0,4104",
    "out/runs/release/120A-p4b-v3/compose/120A-p4b-v3-full+w768-t1536.zip": "120A-p4b-v3, bàn giao S36",
    "out/runs/s43/probes/P1-carrier-w2/compose/P1-carrier-w2-full+w768-t1536.zip": "P1-carrier-w2, LB 0,4141 (cao nhất public)",
}
ARCHIVE_DIRS = {"data/index": "vector BGE-M3 cũ theo segment (nguồn của canonical); tái tạo = embed lại, nhiều giờ GPU"}
SOURCE = {  # never candidates: AGENTS.md (raw stable to 11/11), crawl state, OLD
    "data/raw": "AGENTS: raw giữ ổn định đến sau 11/11", "data/raw_vi": "AGENTS: raw", "data/raw_zh_full": "AGENTS: raw",
    "data/raw_cc_longchau": "AGENTS: raw", "data/raw_zh_v2": "AGENTS: raw", "data/raw_zh_sample": "AGENTS: raw",
    "state": "crawl.db / extract.db (AGENTS)",
}
SOURCE_DATA = ("data/docs_", "data/chunks", "data/cc_aws", "data/pubmed_en", "data/s22_", "data/cache", "data/corpus_v2_ids")


def size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


rows = list(csv.DictReader(open(S / "inventory.csv", encoding="utf-8")))
keep = [k for k in hold.KEEP]
out = []
for r in rows:
    p = r["path"]
    if not p.startswith(REPO):
        continue
    rel = p[len(REPO):]
    if not (rel.startswith("data/") or rel.startswith("out/") or rel.startswith("state")):
        continue
    b = int(r["bytes"])
    inside = [k for k in keep if k.startswith(rel + "/")]          # partial: KEEP files below this inventory row
    if any(rel == k or rel.startswith(k + "/") for k in SOURCE):
        grp, why = "GIU-NGUON", next(v for k, v in SOURCE.items() if rel == k or rel.startswith(k + "/"))
    elif any(rel == k or rel.startswith(k + "/") for k in keep):
        grp, why = "GIU-PRIVATE", "lệnh runbook / diễn tập rh100"
    elif any(rel == k or rel.startswith(k + "/") for k in ARCHIVE_DIRS):
        grp, why = "LUU-TRU", ARCHIVE_DIRS["data/index"]
    elif rel.startswith(SOURCE_DATA):
        grp, why = "LUU-TRU", "docs/chunks trung gian (nguồn canonical-20261018); tái tạo bằng extract từ raw, không chắc trùng byte"
    else:
        grp, why = "XOA", "kết quả thí nghiệm; số liệu + SHA đã ghi trong docs/ và docs/working-notes/data"
    if inside:
        kb = sum(size(Path(REPO + k)) for k in inside if Path(REPO + k).exists())
        out.append(("GIU-PRIVATE", rel + " (phần ghim)", kb, r["newest_mtime"], ", ".join(inside)))
        b -= kb
    arch = [a for a in ARCHIVE_FILES if a.startswith(rel + "/") or a == rel]
    for a in arch:
        ab = size(Path(REPO + a)) if Path(REPO + a).exists() else 0
        out.append(("LUU-TRU", a, ab, r["newest_mtime"], ARCHIVE_FILES[a]))
        b -= ab
    out.append((grp, rel, b, r["newest_mtime"], why))
with open(S / "classes.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["group", "path", "bytes", "newest", "note"])
    w.writerows(out)
tot = {}
for g, *_rest in out:
    tot[g] = tot.get(g, 0) + _rest[1]
for g, v in sorted(tot.items()):
    print(f"{g:12s} {v / 2**30:8.2f} GiB")
