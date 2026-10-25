"""English lane (S35, 25/10): top-20 en docs per query by doc-level cosine, scored by 4B / C-max, kept apart from the vi/zh pools.

``python -m r2ai.s36.enlane scorer``   writes the 4B scorer that also knows ``en`` (the pinned S31 scorer plus ONE INSTRUCTIONS line) and pins its SHA256
``python -m r2ai.s36.enlane gate``     CPU/GPU: re-embeds 200 random vi/zh canonical docs and compares them with the S26 vectors (min cosine >= 0.9999)
``python -m r2ai.s36.enlane pairs``    the top-20 en pairs per query (``out/runs/s36-fuse4/en/pairs_en.parquet``)

The en segment is ``data/s37_en/seg37900-en-pubmed`` (S37: 239,771 PubMed docs, ``vectors.npz`` = BGE-M3 dense CLS L2 fp16, first 4096 chars,
max_len 512, the S26 recipe). The lane never changes the vi/zh pools of the existing builds: ``export-final`` only appends the en pairs to the
``final-4b`` subset, and the en 4B / C-max scores are used by the en p_cal calibration and the joint EF2 only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s36 import fuse

SEG = paths.DATA_DIR / "s37_en" / "seg37900-en-pubmed"
QUERY_VECTORS = paths.RUNS_DIR / "s29-qwen-judge" / "query_vectors.npz"
S26_VECTORS = paths.RUNS_DIR / "s26-gold-structure" / "full" / "full_doc_dense.npz"
JUDGE_SCORE = paths.RUNS_DIR / "s31-strong-judge" / "pool-extra" / "kaggle" / "judge_score.py"
JUDGE_SHA = "688948a7c93ac01ab7c18c565755318ceec5cb041c6663ec9204f7f5695a7186"
EN_INSTRUCTION = "Given a Vietnamese medical question, judge whether the English passage answers it"
TOP = 20
PAIR_ID_BASE_EN = 6_000_000        # S31 0..433,731 / 20,000,000+; S37 1,000,000+; S36 d300 / final-4b 5,000,000+
RANK_BASE_EN = 10_000              # cosine_rank of an en pair = RANK_BASE_EN + its rank among the query's en docs
GATE_MIN_COSINE = 0.9999
ZH_LINE = '    "zh": "Given a Vietnamese medical question, judge whether the Chinese passage answers it",\n'


def out_dir() -> Path:
    return fuse.run_root() / "en"


def scorer_text(base: str) -> str:
    """The pinned scorer with the single ``en`` line added after ``zh`` (everything else byte-identical)."""
    if base.count(ZH_LINE) != 1:
        raise ValueError("INSTRUCTIONS zh line not found exactly once in the pinned scorer")
    return base.replace(ZH_LINE, ZH_LINE + f'    "en": "{EN_INSTRUCTION}",\n')


def make_scorer(dest: Path | None = None) -> dict[str, Any]:
    """Write ``judge_score.py`` with the en instruction; returns its path and SHA256 (and the base SHA256)."""
    if _sha256(JUDGE_SCORE) != JUDGE_SHA:
        raise SystemExit(f"{JUDGE_SCORE} is not the pinned scorer {JUDGE_SHA}")
    dest = dest or (out_dir() / "scorer" / "judge_score.py")
    paths.assert_writable(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(scorer_text(JUDGE_SCORE.read_bytes().decode("utf-8")).encode("utf-8"))
    return {"path": str(dest), "sha256": _sha256(dest), "base_sha256": JUDGE_SHA, "added_line": f'"en": "{EN_INSTRUCTION}"'}


def load_vectors(seg: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    seg = seg or SEG
    with np.load(seg / "vectors.npz") as z:
        return z["doc_ids"].astype(np.int64), z["vectors"]


def top_pairs(q_ids: np.ndarray, q_dense: np.ndarray, doc_ids: np.ndarray, vectors: np.ndarray, top: int = TOP, block: int = 40_000) -> pd.DataFrame:
    """Per query the ``top`` en docs by cosine: ``query_id, doc_id, en_rank, cosine_score`` (ties broken by the smaller doc_id)."""
    q32 = q_dense.astype(np.float32)
    best_s = np.full((len(q32), 0), -np.inf, np.float32)
    best_i = np.zeros((len(q32), 0), np.int64)
    for s in range(0, len(doc_ids), block):
        cos = vectors[s:s + block].astype(np.float32) @ q32.T                       # [docs, queries]
        k = min(top, cos.shape[0])
        part = np.argpartition(-cos, k - 1, axis=0)[:k]                              # [k, queries]
        sc = np.take_along_axis(cos, part, axis=0).T                                  # [queries, k]
        best_s = np.concatenate([best_s, sc], axis=1)
        best_i = np.concatenate([best_i, part.T.astype(np.int64) + s], axis=1)
        if best_s.shape[1] > top:                                                     # keep the running top
            keep = np.argpartition(-best_s, top - 1, axis=1)[:, :top]
            best_s, best_i = np.take_along_axis(best_s, keep, axis=1), np.take_along_axis(best_i, keep, axis=1)
    rows = []
    for qi, q in enumerate(q_ids.tolist()):
        d, sc = doc_ids[best_i[qi]], best_s[qi]
        order = np.lexsort((d, -sc))[:top]
        rows += [(int(q), int(d[j]), r + 1, float(sc[j])) for r, j in enumerate(order)]
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "en_rank", "cosine_score"])


def load_queries() -> tuple[np.ndarray, np.ndarray]:
    with np.load(QUERY_VECTORS) as z:
        return z["q_ids"].astype(np.int64), np.array(z["q_dense"], copy=True)


def pairs(top: int = TOP, seg: Path | None = None) -> pd.DataFrame:
    """The en lane pairs with their ids: ``pair_id, query_id, doc_id, cosine_rank, cosine_score`` (cosine_rank = RANK_BASE_EN + en_rank)."""
    doc_ids, vectors = load_vectors(seg)
    if len(doc_ids) != len(set(doc_ids.tolist())):
        raise ValueError("duplicate doc_id in the en vectors")
    q_ids, q_dense = load_queries()
    frame = top_pairs(q_ids, q_dense, doc_ids, vectors, top)
    frame = frame.sort_values(["query_id", "en_rank"]).reset_index(drop=True)
    frame["pair_id"] = PAIR_ID_BASE_EN + np.arange(len(frame))
    frame["cosine_rank"] = RANK_BASE_EN + frame.en_rank
    return frame[["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "en_rank"]]


def embed_texts(texts: list[str], device: str) -> np.ndarray:
    """The S26 ``full_embed.py`` recipe (as r2ai.s37.ingest.embed_texts): first 4096 chars, length-sorted batches (64 / 32 / 16), dense CLS L2 fp16, max_len 512."""
    from r2ai.index.bge_m3 import M3Encoder

    enc = M3Encoder(device=device, fp16=(device != "cpu"), max_len=512)
    texts = [t[:4096] for t in texts]
    order = np.argsort([len(x) for x in texts], kind="stable")
    out = np.zeros((len(texts), 1024), np.float16)
    i = 0
    while i < len(order):
        bs = 64 if len(texts[order[i]]) < 600 else 32 if len(texts[order[i]]) < 1500 else 16
        sel = order[i:i + bs]
        out[sel] = enc.encode_batch([texts[j] for j in sel])[0]
        i += bs
    return out


def gate(n: int = 200, seed: int = 20261025, device: str = "cuda") -> dict[str, Any]:
    """Regression gate: re-embed ``n`` random vi/zh canonical docs with the S26 recipe and compare with the S26 vectors."""
    import pyarrow.parquet as pq

    with np.load(S26_VECTORS) as z:
        s26_ids, s26_vec = z["doc_id"].astype(np.int64), z["dense"]
    canon = paths.DATA_DIR / "canonical-20261018" / "docs.parquet"
    docs = pq.read_table(canon, columns=["doc_id", "doc_text"]).to_pandas()
    docs = docs[docs.doc_text.fillna("").str.len() > 0]
    pick = docs.sample(n=n, random_state=seed)
    index = {int(d): i for i, d in enumerate(s26_ids)}
    keep = [i for i, d in enumerate(pick.doc_id.astype(int)) if int(d) in index]
    pick = pick.iloc[keep]
    new = embed_texts(pick.doc_text.tolist(), device).astype(np.float32)
    old = s26_vec[[index[int(d)] for d in pick.doc_id]].astype(np.float32)
    cos = (new * old).sum(1) / (np.linalg.norm(new, axis=1) * np.linalg.norm(old, axis=1))
    result = {"docs": int(len(pick)), "seed": seed, "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
              "threshold": GATE_MIN_COSINE, "ok": bool(cos.min() >= GATE_MIN_COSINE), "worst_doc_ids": pick.doc_id.iloc[np.argsort(cos)[:3]].astype(int).tolist()}
    paths.assert_writable(out_dir())
    out_dir().mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir() / "gate.json", result)
    return result


# ------------------------------------------------------------------------------------------------- calibration and joint EF2
EPS_GRID = (5e-5, 5e-4, 4e-3)      # upper bound of the mean p_cal of the en top-20: LB resolution of Doc P = 0 (EN-only-Q), 4e-3 = the E1 bound (< 0.4%)


def calibrate(p4b: np.ndarray, a: float, target: float) -> float:
    """``b_en`` such that mean sigmoid(a * logit(P4B) + b_en) over ``p4b`` equals ``target`` (bisection; the mean is increasing in b)."""
    from r2ai.s36.cal import logit, sigmoid

    x = a * logit(p4b)
    lo, hi = -60.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(sigmoid(x + mid).mean()) < target else (lo, mid)
    return 0.5 * (lo + hi)


def joint(perq: pd.DataFrame, en_scores: pd.DataFrame, a: float, eps_grid: Sequence[float] = EPS_GRID) -> dict[str, Any]:
    """Joint EF2 of the vi/zh selection and the en lane.

    ``perq``: ``query_id, k, tp, g, f2`` of the vi/zh EF2 selection (predicted TP, G_q and F2 per query); ``en_scores``: ``query_id, doc_id, score``
    (4B of the top-20 en pairs). Adding a doc with calibrated probability p to a selection with F2 = 5 TP / (4 G + k) raises F2 iff
    p > TP / (4 G + k) = F2 / 5 (the break-even); the en docs are tried by decreasing p_cal and each accepted doc raises the break-even.
    ``a_en`` = the vi/zh slope ``a`` (the LB evidence is one number per set, so only the offset ``b_en`` is fitted)."""
    from r2ai.s36.cal import logit, sigmoid

    scores = en_scores.score.to_numpy(np.float64)
    base = perq.set_index("query_id")
    out: dict[str, Any] = {"a_en": a, "queries": int(len(base)), "en_pairs": int(len(en_scores)),
                           "break_even_f2_over_5": {"min": float((base.f2 / 5).min()), "p10": float((base.f2 / 5).quantile(.1)), "mean": float((base.f2 / 5).mean())},
                           "p4b_en": {"mean": float(scores.mean()), "max": float(scores.max()), "share_ge_0.5": float((scores >= 0.5).mean())}, "grid": {}}
    for eps in eps_grid:
        b_en = calibrate(scores, a, eps)
        p = sigmoid(a * logit(scores) + b_en)
        frame = en_scores.assign(p=p)
        chosen = 0
        for q, g in frame.groupby("query_id"):
            tp, k, tg = float(base.tp[q]), int(base.k[q]), float(base.g[q])
            for pv in sorted(g.p, reverse=True):
                if pv > tp / (4 * tg + k):
                    tp, k = tp + pv, k + 1
                    chosen += 1
                else:
                    break
        out["grid"][f"{eps:g}"] = {"b_en": float(b_en), "p_cal_mean": float(p.mean()), "p_cal_max": float(p.max()), "p_cal_p99": float(np.quantile(p, .99)),
                                   "expected_relevant_en_in_top20": float(p.sum()), "docs_above_min_break_even": int((p > out["break_even_f2_over_5"]["min"]).sum()),
                                   "en_docs_chosen": int(chosen)}
    return out


def run_joint(perq_path: Path, scores_path: Path, a: float) -> dict[str, Any]:
    pairs_en = pd.read_parquet(out_dir() / "pairs_en.parquet", columns=["pair_id", "query_id", "doc_id"])
    scores = pd.read_parquet(scores_path, columns=["pair_id", "score"]).merge(pairs_en, on="pair_id")
    if len(scores) != len(pairs_en):
        raise ValueError(f"{len(pairs_en) - len(scores)} en pairs have no 4B score")
    result = joint(pd.read_parquet(perq_path), scores[["query_id", "doc_id", "score"]], a)
    result["inputs"] = {"perq": str(perq_path), "scores": str(scores_path), "scores_sha256": _sha256(scores_path)}
    paths.assert_writable(out_dir())
    _atomic_json(out_dir() / "joint.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("scorer", "gate", "pairs", "joint"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--perq", type=Path, default=None, help="joint: perq-<name>.parquet of the vi/zh EF2 selection (s37 apply)")
    parser.add_argument("--scores", type=Path, default=None, help="joint: 4B scores of the en pairs (default d300/final-4b-B/gpu/.../scores-extra.parquet)")
    parser.add_argument("--a", type=float, default=0.5298075151496865, help="joint: slope a of the vi/zh calibration (S36 cal.json, variant p4b)")
    args = parser.parse_args(argv)
    if args.command == "scorer":
        result = make_scorer()
        _atomic_json(out_dir() / "scorer" / "scorer.json", result)
    elif args.command == "gate":
        result = gate(device=args.device)
    elif args.command == "joint":
        scores = args.scores or (fuse.run_root() / "d300" / "final-4b-B" / "gpu" / "qwen3-reranker-4b" / "scores-extra.parquet")
        result = run_joint(args.perq, scores, args.a)
    else:
        frame = pairs()
        paths.assert_writable(out_dir())
        out_dir().mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out_dir() / "pairs_en.parquet", index=False)
        result = {"pairs": int(len(frame)), "queries": int(frame.query_id.nunique()), "docs": int(frame.doc_id.nunique()),
                  "cosine_p10_p50_p90": [float(x) for x in frame.cosine_score.quantile([.1, .5, .9])], "sha256": _sha256(out_dir() / "pairs_en.parquet")}
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
