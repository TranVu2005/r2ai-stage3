"""S44 temporary hold: move (os.rename, same volume D:) every path below data/ and out/ that is not in KEEP to HOLD/<same relative path>,
so a run can prove KEEP is enough; ``restore`` moves everything back from the manifest. Nothing is deleted or copied.

    python -I hold.py plan     # print what would move (bytes not computed)
    python -I hold.py hold     # move; manifest HOLD/manifest.jsonl written before each move (fsync)
    python -I hold.py restore  # move back in reverse order; refuses to overwrite
"""
import json
import os
import sys
from pathlib import Path

REPO = Path("D:/GitHub/r2ai-stage3")
HOLD = Path(os.environ.get("S44_HOLD", "D:/_r2ai_s44_hold"))
ROOTS = [REPO / "data", REPO / "out"]
KEEP = [
    # GIỮ-PRIVATE: corpus, segments, models (79 pins + model snapshot)
    "data/canonical-20261018", "data/s37_120ask", "data/s37_en", "data/hf_model_cache",
    "out/runs/s26-gold-structure/full/full_doc_dense.npz",
    "out/runs/s27-pharmacity/vectors/handoff", "out/runs/s27-pharmacity/handoff/cc-s27-pharmacity",
    "out/runs/s31-strong-judge/pool-extra/kaggle/judge_score.py",
    "out/runs/s32-chunk-judge/pool/supplement_vectors.npz",
    "out/runs/s37/cmax-d300/supplement_vectors.npz",
    "out/runs/reranker-trial/model-pin.json",
    # rehearsal fixtures (rh100 only: queries + the 4B / C-max scores that stand in for Kaggle / GPU)
    "out/runs/s40/rh100", "out/runs/s40/abl/rehearsal-input",
    # AGENTS.md: raw roots stay where they are until after 11/11 (not candidates, not moved)
    "data/raw", "data/raw_vi", "data/raw_zh_full", "data/raw_cc_longchau", "data/raw_zh_v2", "data/raw_zh_sample",
]
EXTRA_KEEP = [p for p in os.environ.get("S44_EXTRA_KEEP", "").split(";") if p]
NEVER = ["out/runs/s44-verify"]          # outputs of the check itself (prefix)


def keep_set():
    return [REPO / k for k in KEEP + EXTRA_KEEP]


def plan():
    keep = keep_set()
    moves = []

    def visit(d: Path):
        for c in sorted(d.iterdir()):
            rel = c.relative_to(REPO).as_posix()
            if any(rel.startswith(n) for n in NEVER):
                continue
            if any(c == k for k in keep):
                continue
            if any(k.is_relative_to(c) for k in keep) and c.is_dir() and not c.is_junction() and not c.is_symlink():
                visit(c)
                continue
            moves.append(c)

    for r in ROOTS:
        visit(r)
    return moves


def main(cmd: str):
    man = HOLD / "manifest.jsonl"
    if cmd == "plan":
        for m in plan():
            print(m.relative_to(REPO).as_posix())
        return
    if cmd == "hold":
        if man.exists() and man.read_text(encoding="utf-8").strip():
            raise SystemExit(f"{man} not empty: restore first")
        missing = [k for k in keep_set() if not k.exists()]
        if missing:
            raise SystemExit(f"KEEP paths missing: {missing}")
        HOLD.mkdir(parents=True, exist_ok=True)
        moves = plan()
        with open(man, "a", encoding="utf-8") as f:
            for src in moves:
                dst = HOLD / src.relative_to(REPO)
                if dst.exists():
                    raise SystemExit(f"exists: {dst}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                f.write(json.dumps({"src": str(src), "dst": str(dst)}) + "\n")
                f.flush()
                os.fsync(f.fileno())
                os.rename(src, dst)
        print(f"held {len(moves)} paths")
        return
    if cmd == "restore":
        rows = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
        back = 0
        for r in reversed(rows):
            src, dst = Path(r["src"]), Path(r["dst"])
            if not dst.exists():
                if src.exists():
                    continue          # never moved (stopped before the rename) or already restored
                raise SystemExit(f"lost track: neither {dst} nor {src}")
            if src.exists():
                raise SystemExit(f"refuse to overwrite {src}")
            os.rename(dst, src)
            back += 1
        man.rename(man.with_suffix(".restored.jsonl"))
        print(f"restored {back} paths")
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
