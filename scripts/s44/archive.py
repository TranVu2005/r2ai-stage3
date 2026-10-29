"""S44 (Vu duyệt 28/10): pack the LƯU-TRỮ group into D:/_r2ai_archive (copies of 6 ZIPs, judge-public.tar, out-runs-meta.tar.xz; data
dirs moved by rename), each with a SHA256 manifest. Nothing is deleted.

    python -I -X utf8 archive.py pack        # copies + tars (reads out/runs, writes only below ARCHIVE)
    python -I -X utf8 archive.py move-data   # rename the LƯU-TRỮ data dirs to ARCHIVE/data (manifest moves.jsonl)
    python -I -X utf8 archive.py hash-data   # SHA256 of every moved data file -> ARCHIVE/data-sha256.csv
"""
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

REPO = Path("D:/GitHub/r2ai-stage3")
RUNS = REPO / "out" / "runs"
ARCHIVE = Path("D:/_r2ai_archive")
NOTES = REPO / ".superpowers/worktrees/working-notes/docs/working-notes/data/archive_zips.csv"
JUDGE_NAMES = ("pool_index.parquet", "extra_pairs.parquet", "chunk_pairs.parquet", "cmax.parquet", "jina.parquet")
META_EXT = {".json", ".md", ".csv", ".tsv", ".log", ".txt", ".yaml", ".out", ".stdout"}
DATA_PREFIX = ("docs_", "chunks", "cc_aws", "pubmed_en", "s22_", "cache", "corpus_v2_ids", "index")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def run_files():
    for dp, dn, fn in os.walk(RUNS):
        rel = Path(dp).relative_to(RUNS).as_posix()
        if rel.startswith("s44-verify"):
            dn[:] = []
            continue
        for f in fn:
            yield Path(dp) / f


def is_judge(p: Path) -> bool:
    if "compose" in p.parts:
        return False
    n = p.name
    return ((n.endswith(".parquet") and ("score" in n or n.startswith("pairs") or n in JUDGE_NAMES))
            or (n.endswith(".sqlite") and "scores" in n))


def is_meta(p: Path) -> bool:
    return p.suffix in META_EXT and p.stat().st_size < 5_000_000


def tar(name: str, mode: str, files: list[Path]) -> dict:
    target = ARCHIVE / name
    members = []
    with tarfile.open(target, mode) as t:
        for p in files:
            arc = p.relative_to(REPO).as_posix()
            members.append({"path": arc, "bytes": p.stat().st_size, "sha256": sha256(p)})
            t.add(p, arcname=arc, recursive=False)
    return {"bundle": name, "bytes": target.stat().st_size, "sha256": sha256(target), "files": len(members), "members": members}


def pack():
    ARCHIVE.mkdir(exist_ok=True)
    out = {}
    zips = []
    for row in csv.DictReader(open(NOTES, encoding="utf-8")):
        src = REPO / row["path"]
        dst = ARCHIVE / "zips" / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        got = sha256(dst)
        if got != row["sha256"]:
            raise SystemExit(f"ZIP copy SHA differs: {dst}")
        zips.append({**row, "archive_path": dst.relative_to(ARCHIVE).as_posix(), "verified": True})
    out["zips"] = zips
    files = sorted(run_files())
    out["judge"] = tar("judge-public.tar", "w", [p for p in files if is_judge(p)])
    out["meta"] = tar("out-runs-meta.tar.xz", "w:xz", [p for p in files if is_meta(p)])
    (ARCHIVE / "manifest.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"zips": len(zips), "judge": {k: out["judge"][k] for k in ("bytes", "sha256", "files")},
                      "meta": {k: out["meta"][k] for k in ("bytes", "sha256", "files")}}, indent=1))


def data_dirs():
    return sorted(p for p in (REPO / "data").iterdir() if p.name.startswith(DATA_PREFIX))


def move_data():
    man = ARCHIVE / "moves.jsonl"
    with open(man, "a", encoding="utf-8") as f:
        for src in data_dirs():
            dst = ARCHIVE / src.relative_to(REPO)
            if dst.exists():
                raise SystemExit(f"exists: {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            f.write(json.dumps({"src": str(src), "dst": str(dst)}) + "\n")
            f.flush()
            os.fsync(f.fileno())
            os.rename(src, dst)
            print("moved", src.relative_to(REPO).as_posix())


def hash_data():
    with open(ARCHIVE / "data-sha256.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "sha256"])
        n = b = 0
        for p in sorted((ARCHIVE / "data").rglob("*")):
            if p.is_file():
                w.writerow([p.relative_to(ARCHIVE).as_posix(), p.stat().st_size, sha256(p)])
                n += 1
                b += p.stat().st_size
    print(n, "files", round(b / 2**30, 2), "GiB")


if __name__ == "__main__":
    {"pack": pack, "move-data": move_data, "hash-data": hash_data}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
