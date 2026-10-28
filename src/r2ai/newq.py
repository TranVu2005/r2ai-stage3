"""A NEW query set on the free-v2 frame (S40, 26/10): the steps of ``120A-free-v2`` from the query texts alone.

    python -m r2ai.newq prepare --queries q.parquet --run <dir> [--config configs/s41-private-final.json] [--device cuda]
    python -m r2ai.newq export-scoring --run <dir> [--parts-4b K]
    python -m r2ai.newq import-scores --run <dir> --source 4b --tag <tag> --scores scores-extra.parquet --report collect-extra-report.json
    python -m r2ai.newq import-scores --run <dir> --source cmax      (after scripts/s37/run_cmax.ps1 -Run <dir>/cmax)
    python -m r2ai.newq import-scores --run <dir> --source jina      (after scripts/s37/run_jina.ps1 -Run <dir>/jina)
    python -m r2ai.newq build --run <dir> [--name N] [--jina] [--no-cmax] [--impute-4b | --cos-only | --cos-k K] [--rounds R]
    python -m r2ai.newq status --run <dir>
    python -m r2ai.newq pins [--write]        # SHA256 of every corpus / scorer input against configs/s41-newq-pins.json

Pipeline chốt cho private (Vu 27/10, frozen; LB public CH-rrf 0,4138): RRF over cosine / 4B / C-max, NO Jina, CH-rrf carriers
(``configs/s41-private-final.json``: CR-fair count, docs taken in RRF order). ``build`` and
``export-scoring`` leave Jina out unless ``--jina`` / ``--sources ...,jina`` asks for it.

Input pinning (Vu, 26/10): every command first checks the SHA256 of every corpus / scorer file it may read against
``configs/s41-newq-pins.json`` and of every run file it reads against ``<run>/ledger.json`` (written by the step that produced the file).
A missing file or a different SHA256 stops the command; nothing is searched elsewhere.

Nothing here reads a file keyed by the 1,200 public query ids (S29 query vectors, pool.npz, pool13 / K_q, S31-S37 pair registries and scores,
S32 chunk scores, the P1 / baseline ZIPs, docsel of earlier builds). What is read: the corpus (S26 doc vectors + Pharmacity lots, the
120ask segments and their chunk vectors, canonical-20261018 text / chunk vectors, the S32 / S37 supplement chunk vectors), the pinned
scorers and models, and the calibration constants of the config (a, b, s_free, t_c: fitted once on the public LB, not refitted).

Steps (each writes below ``--run``; a finished step is skipped when rerun, so a job can simply be restarted):

1. ``prepare``      queries (``id`` int, ``query`` text) -> BGE-M3 dense (CLS, L2, fp16, max_len 512, length-sorted batches of 128) ->
                    cosine top-400 over the S26 doc vectors + Pharmacity (``r2ai.s37.ingest.top_scores``) -> merged with every 120ask doc
                    (``r2ai.s36.d300.merged_top``: canonical first on a cosine tie, then segment / file order) -> ``top.parquet`` (top-``depth``).
2. ``export-scoring`` the three scoring packages: 4B Kaggle datasets (S31 pool-extra layout, pinned scorer ``688948a7``, one dataset and one
                    notebook per part), C-max pool (S32 chunk rule: top-3 t256 body/answer chunks by cosine, doc_text fallback; 0.6B scorer),
                    Jina pool (doc text cut to 512 Qwen3 tokens, rows by cosine desc / doc_id, layout b10s0 seeded by query id).
3. ``import-scores`` checks and stores each source (4B per Kaggle part; C-max = max over the pair's chunk rows; Jina scores-all).
4. ``build``        free frame: pool = top-``depth``, RRF k=60 over cosine / 4B / Jina / C-max (``--no-jina`` / ``--no-cmax`` leave the source out of
                    the RRF), p_cal = sigmoid(a logit(P4B) + b), G_q = s_free (sum p_cal over the cosine top-211 + tail), EF2 cut per query,
                    CR-fair carriers (target = ``target_per_query`` x queries; config ``carriers.order``: ``rank`` = CH-rrf, ``p4b`` = S36), docsel, then S38 compose ``full+w768`` (``--fit-carriers``,
                    T searched) with ``--query-set`` and the S26 validator on the new query ids.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai import paths

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "s41-private-final.json"   # S41 private config (CH-rrf carriers); s40-newq-free-v2.json = FREE-noJina
COS_CONFIG = ROOT / "configs" / "s41-private-fallback-cos.json"   # build --cos-only: DEFAULT_CONFIG + the pinned cosine -> p_cal map
COS_K_CONFIG = ROOT / "configs" / "s41-private-fallback-cos-k{k}.json"   # build --cos-k K: COS_CONFIG + cos_k = K
SELECTION_KEYS = ("depth", "keep", "a", "b", "s_free", "g_depth", "t_c", "rrf_k", "carriers")
PAIR_ID_BASE = 70_000_000         # pair ids of a new query set (S31 0..433,731 / 20,000,000+; S37 1,000,000+; S36 5,000,000+ / 6,000,000+; d300 C-max 60,000,000+)
CMAX_ROW_BASE = 80_000_000        # C-max chunk rows
SHARD_4B = 94_000                 # 4B shard pool-<94000 + part>.parquet
JUDGE_SHA = "688948a7c93ac01ab7c18c565755318ceec5cb041c6663ec9204f7f5695a7186"
REVISION_4B = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
REVISION_06B = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
NOTEBOOK = ROOT / "notebooks" / "s36-kaggle-4b-final.ipynb"
EMBED_BATCH = 128
SOURCES = ("4b", "cmax", "jina")


# ---------------------------------------------------------------------------------------------------------- helpers

def sha256(path: Path | str) -> str:
    from r2ai.s29.qwen_judge import _sha256
    return _sha256(Path(path))


def atomic_json(path: Path, value: Any) -> None:
    paths.assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    paths.assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def log_to(run: Path):
    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        with open(run / "newq.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return log


def load_config(path: Path | None) -> dict[str, Any]:
    """A config file, or (S41 fallback) one that ``inherits`` another, pinned by SHA256, and may only add ``impute_4b``, ``cos_k`` and a
    ``note``."""
    cfg = json.loads((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    if "inherits" in cfg:
        extra = set(cfg) - {"inherits", "inherits_sha256", "impute_4b", "cos_k", "note"}
        if extra:
            raise SystemExit(f"config {path}: an inheriting config may only add impute_4b / cos_k / note, not {sorted(extra)}")
        base = ROOT / cfg["inherits"]
        if not base.exists() or sha256(base) != cfg.get("inherits_sha256"):
            raise SystemExit(f"config {path}: {cfg['inherits']} missing or SHA256 differs from {cfg.get('inherits_sha256')}")
        cfg = {**load_config(base), **cfg}
    for key in ("depth", "keep", "a", "b", "s_free", "g_depth", "t_c", "carriers"):
        if key not in cfg:
            raise SystemExit(f"config {path or DEFAULT_CONFIG}: missing {key}")
    for rel, want in cfg.get("pins_sha256", {}).items():         # S41: the config pins the input-pin file itself
        if not (ROOT / rel).exists() or sha256(ROOT / rel) != want:
            raise SystemExit(f"config {path or DEFAULT_CONFIG}: {rel} missing or SHA256 differs from {want}")
    return cfg


def run_config(run: Path) -> dict[str, Any]:
    require(run, ["config.json"])
    return json.loads((run / "config.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------- input pinning

PINS = ROOT / "configs" / "s41-newq-pins.json"   # S41: repo files keyed "repo:<path>" (any checkout); s40-newq-pins.json keyed one worktree


def pin_key(path: Path) -> str:
    """Absolute posix path; a file of this checkout is ``repo:<path relative to ROOT>``, so the pins hold in every worktree."""
    path = path.resolve()
    try:
        return "repo:" + path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def pin_path(key: str) -> Path:
    return ROOT / key[5:] if key.startswith("repo:") else Path(key)


def pin_targets() -> list[Path]:
    """Every corpus / scorer file a step may read (models are pinned by revision)."""
    from r2ai.s37 import ingest

    canon = paths.DATA_DIR / "canonical-20261018"
    files = [canon / "docs.parquet", canon / "chunks_t256.parquet", canon / "vectors" / "dense.npy", canon / "vectors" / "vectors.json",
             canon / "manifest.json", ingest.S26_VECTORS, ROOT / "src" / "r2ai" / "s31" / "judge_score.py", NOTEBOOK,
             paths.RUNS_DIR / "s32-chunk-judge" / "pool" / "supplement_vectors.npz", paths.RUNS_DIR / "s37" / "cmax-d300" / "supplement_vectors.npz",
             paths.RUNS_DIR / "s31-strong-judge" / "pool-extra" / "kaggle" / "judge_score.py"]
    files += sorted(p for p in ingest.PHARMACITY_VECTORS.rglob("*") if p.is_file())
    files += sorted(p for p in (paths.RUNS_DIR / "s27-pharmacity" / "handoff" / "cc-s27-pharmacity").rglob("*") if p.is_file())
    for seg in sorted(p for p in ingest.STORE.iterdir() if p.is_dir()):
        files += [seg / n for n in ("docs.parquet", "vectors.npz", "chunks_t256.parquet", "chunk_vectors.npz") if (seg / n).exists()]
    return files


def write_pins() -> dict[str, Any]:
    pins = {pin_key(f): sha256(f) for f in pin_targets()}
    PINS.write_text(json.dumps({"note": "S41 r2ai.newq input pins (python -m r2ai.newq pins --write)", "files": pins}, indent=1) + "\n", encoding="utf-8")
    return {"files": len(pins), "path": str(PINS)}


def check_pins(log=print) -> dict[str, Any]:
    """Every pinned file exists with its SHA256 and no corpus file appeared that the pins do not list; otherwise SystemExit."""
    if not PINS.exists():
        raise SystemExit(f"{PINS} missing: run `python -m r2ai.newq pins --write` once, review and commit it")
    pins = json.loads(PINS.read_text(encoding="utf-8"))["files"]
    t0 = time.monotonic()
    bad = [f"missing {f}" for f in pins if not pin_path(f).exists()]
    bad += [f"SHA256 differs {f}" for f, h in pins.items() if pin_path(f).exists() and sha256(pin_path(f)) != h]
    now = {pin_key(f) for f in pin_targets()}
    bad += [f"not pinned {f}" for f in sorted(now - set(pins))]
    if bad:
        raise SystemExit("input pins: " + "; ".join(bad[:10]) + (f" (+{len(bad) - 10})" if len(bad) > 10 else ""))
    log(f"input pins ok: {len(pins)} files, {time.monotonic() - t0:.0f}s")
    return {"files": len(pins), "seconds": round(time.monotonic() - t0, 1)}


def record(run: Path, names: Sequence[str]) -> None:
    """Add the SHA256 of run files (relative names) to ``<run>/ledger.json``."""
    path = run / "ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    ledger.update({n: sha256(run / n) for n in names})
    atomic_json(path, ledger)


def require(run: Path, names: Sequence[str]) -> None:
    """Every run file exists with the SHA256 of ``<run>/ledger.json``; otherwise SystemExit."""
    path = run / "ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    bad = [n for n in names if n not in ledger or not (run / n).exists() or sha256(run / n) != ledger[n]]
    if bad:
        raise SystemExit(f"{run}: run files missing, unrecorded or changed: {bad}")


def read_queries(path: Path) -> pd.DataFrame:
    q = pd.read_parquet(path)
    if not {"id", "query"} <= set(q.columns):
        raise SystemExit(f"{path}: needs columns id, query (got {list(q.columns)})")
    q = q[["id", "query"]].copy()
    if q.id.isna().any() or q.id.duplicated().any():
        raise SystemExit(f"{path}: id must be unique and non-null")
    if (q.id.astype(np.int64) != q.id).any():
        raise SystemExit(f"{path}: id must be integers")
    q["id"] = q.id.astype(np.int64)
    q["query"] = q["query"].fillna("").astype(str)
    if (q["query"].str.strip() == "").any():
        raise SystemExit(f"{path}: {(q['query'].str.strip() == '').sum()} empty queries")
    return q.sort_values("id").reset_index(drop=True)


def query_texts(run: Path) -> dict[int, str]:
    q = pd.read_parquet(run / "queries.parquet")
    return dict(zip(q.id.astype(int), q["query"]))


def query_vectors(run: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(run / "query_vectors.npz") as z:
        return z["q_ids"].astype(np.int64), np.array(z["q_dense"], copy=True)


# ---------------------------------------------------------------------------------------------------------- 1. prepare

def embed_queries(texts: list[str], device: str) -> np.ndarray:
    """BGE-M3 dense CLS, L2-normalised, fp16 (``r2ai.index.bge_m3.M3Encoder``, max_len 512), length-sorted batches of ``EMBED_BATCH``.
    Against the S26 q_dense of the public queries this gives cosine >= 0.99992 per query (56 % of the rows bit-identical: the S26 batch
    layout is not recorded)."""
    from r2ai.index.bge_m3 import M3Encoder

    enc = M3Encoder(device=device, fp16=(device != "cpu"), max_len=512)
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), 1024), np.float16)
    for begin in range(0, len(order), EMBED_BATCH):
        sel = order[begin:begin + EMBED_BATCH]
        out[sel] = enc.encode_batch([texts[j] for j in sel])[0]
    if not np.isfinite(out.astype(np.float32)).all():
        raise ValueError("non-finite query vectors")
    return out


def prepare(queries: Path, run: Path, cfg_path: Path | None, device: str) -> dict[str, Any]:
    from r2ai.s36 import d300
    from r2ai.s37 import ingest

    paths.assert_writable(run)
    run.mkdir(parents=True, exist_ok=True)
    log = log_to(run)
    check_pins(log)
    cfg = load_config(cfg_path)
    q = read_queries(queries)
    report: dict[str, Any] = {"queries_source": str(queries), "queries_source_sha256": sha256(queries), "queries": len(q),
                              "config": str(cfg_path or DEFAULT_CONFIG), "steps": {}}
    if (run / "queries.parquet").exists():
        old = pd.read_parquet(run / "queries.parquet")
        if not old.equals(q):
            raise SystemExit(f"{run}: holds another query set; use a new --run")
    else:
        atomic_parquet(q, run / "queries.parquet")
        atomic_json(run / "config.json", cfg)
        record(run, ["queries.parquet", "config.json"])
    require(run, ["queries.parquet", "config.json"])
    t0 = time.monotonic()
    if not (run / "query_vectors.npz").exists():
        log(f"embed {len(q)} queries on {device}")
        dense = embed_queries(q["query"].tolist(), device)
        paths.assert_writable(run / "query_vectors.npz")
        np.savez(run / "query_vectors.npz", q_ids=q.id.to_numpy(np.int64), q_dense=dense)
        record(run, ["query_vectors.npz"])
    report["steps"]["embed_seconds"] = round(time.monotonic() - t0, 1)
    t0 = time.monotonic()
    q_ids, q_dense = query_vectors(run)
    if not (run / "pool.npz").exists():
        log("cosine top-%d over the S26 doc vectors + Pharmacity" % cfg["keep"])
        ids, dense = ingest.load_doc_vectors()
        scores, rows = ingest.top_scores(q_dense, dense, int(cfg["keep"]))
        del dense
        paths.assert_writable(run / "pool.npz")
        np.savez(run / "pool.npz", q_ids=q_ids, scores=scores, doc_ids=ids[rows])
        record(run, ["pool.npz"])
    report["steps"]["pool_seconds"] = round(time.monotonic() - t0, 1)
    t0 = time.monotonic()
    if not (run / "top.parquet").exists():
        with np.load(run / "pool.npz") as z:
            pool = {k: z[k] for k in z.files}
        top = d300.merged_top(int(cfg["depth"]), pool, q_dense)
        top["pair_id"] = PAIR_ID_BASE + np.arange(len(top), dtype=np.int64)
        atomic_parquet(top, run / "top.parquet")
        record(run, ["top.parquet"])
    require(run, ["query_vectors.npz", "pool.npz", "top.parquet"])
    report["steps"]["merge_seconds"] = round(time.monotonic() - t0, 1)
    top = pd.read_parquet(run / "top.parquet")
    per = top.groupby("query_id").size()
    if set(per.index) != set(q.id) or (per != int(cfg["depth"])).any():
        raise ValueError(f"top.parquet: not exactly {cfg['depth']} docs for every query")
    report.update({"pairs": int(len(top)), "pairs_ask": int(top.is_ask.sum()), "docs": int(top.doc_id.nunique()),
                   "query_vectors_sha256": sha256(run / "query_vectors.npz"), "top_sha256": sha256(run / "top.parquet")})
    atomic_json(run / "prepare.json", report)
    log(f"prepare done: {report['pairs']} pairs")
    return report


# ---------------------------------------------------------------------------------------------------------- 2. export-scoring

def texts_512(run: Path, top: pd.DataFrame, log) -> pd.DataFrame:
    """``doc_id, lang, segment, doc_text, doc_tokens``: every pool doc cut to 512 Qwen3 tokens (the 4B / Jina text rule of S31 / S36)."""
    target = run / "texts.parquet"
    if target.exists():
        require(run, ["texts.parquet"])
        return pd.read_parquet(target)
    from r2ai.s36 import d300

    docs = top.drop_duplicates("doc_id")[["doc_id", "is_ask"]].reset_index(drop=True)
    passages, langs, seg_of = d300.passages_for(docs, time.monotonic())
    frame = pd.DataFrame({"doc_id": docs.doc_id.astype(np.int64), "lang": [langs[int(d)] for d in docs.doc_id],
                          "segment": [seg_of.get(int(d), "") for d in docs.doc_id],
                          "doc_text": [passages[int(d)][0] for d in docs.doc_id], "doc_tokens": [int(passages[int(d)][1]) for d in docs.doc_id]})
    atomic_parquet(frame, target)
    record(run, ["texts.parquet"])
    log(f"texts: {len(frame)} docs")
    return frame


def split_queries(per_query: pd.Series, parts: int) -> list[list[int]]:
    """Contiguous blocks of query ids with about the same number of pairs."""
    ids, sizes = per_query.index.to_numpy(), per_query.to_numpy()
    edges = np.searchsorted(np.cumsum(sizes), np.linspace(0, sizes.sum(), parts + 1)[1:-1], side="right")
    return [blk.tolist() for blk in np.split(ids, edges) if len(blk)]


def notebook_for(tags: dict[str, int], dataset: Path) -> dict[str, Any]:
    """The pinned final-4b notebook with only TAGS / EXPECT / the dataset name changed; ``tags`` (tag -> pairs) run in this order."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    dataset_dir, hits, main = dataset.name, 0, list(tags)[-1]
    tag_list = ", ".join(f'"{t}"' for t in tags)
    expect = ", ".join(f'"{t}": {n}' for t, n in tags.items())
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        new = text.replace("`out/runs/s36-fuse4/d300/kaggle-dataset-final4b/` (`judge_score.py` + `final-4b/`)",
                           f"`{dataset.as_posix()}/` (`judge_score.py` + " + " + ".join(f"`{t}/`" for t in tags) + ")")
        new = new.replace("(`final-4b/qwen3-reranker-4b/parts`)", f"(`{main}/qwen3-reranker-4b/parts`)")
        new = new.replace('TAGS = ["final-4b"]', f"TAGS = [{tag_list}]").replace('EXPECT = {"final-4b": 47833}', f"EXPECT = {{{expect}}}")
        new = new.replace("/kaggle/working/final-4b/", f"/kaggle/working/{main}/").replace("kaggle-dataset-final4b", dataset_dir)
        if cell["cell_type"] == "markdown" and text.startswith("# S36"):
            new = (f"# S40 Qwen3-Reranker-4B on Kaggle (2x T4): {', '.join(tags)}\n\n"
                   + "; ".join(f"`{t}` {n:,} pairs" for t, n in tags.items())
                   + f" of a new query set (r2ai.newq), dataset `{dataset_dir}`, subsets scored in this order (a `-reg` subset is the "
                   "1,000-pair Kaggle / RunPod regression gate: download its scores-extra.parquet as soon as it is collected). "
                   "The cells are the pinned S36 final-4b notebook; only the subset names, the pair counts and the dataset name differ. "
                   "The notes below are those of S36.\n\n" + new)
        hits += new != text
        cell["source"] = new.splitlines(keepends=True)
    if hits < 2:
        raise ValueError("notebook template changed: TAGS / EXPECT not found")
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    return nb


def write_4b_subset(sub: Path, tag: str, todo: pd.DataFrame, texts: pd.DataFrame, queries: dict[int, str], shard_no: int,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """One 4B subset in the S31 pool-extra layout: ``sub/{shards/pool-<n>.parquet, extra_pairs.parquet, manifest.json}``."""
    from r2ai.s36 import d300

    if (sub / "manifest.json").exists():
        return json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
    todo = todo.sort_values(["query_id", "cosine_rank"]).reset_index(drop=True)
    t = texts.set_index("doc_id").loc[todo.doc_id.to_numpy()]
    shard_name = f"pool-{shard_no}.parquet"
    (sub / "shards").mkdir(parents=True, exist_ok=True)
    shard = pa.table({
        "pair_id": pa.array(todo.pair_id, pa.int64()), "query_id": pa.array(todo.query_id, pa.int64()), "doc_id": pa.array(todo.doc_id, pa.int64()),
        "cosine_rank": pa.array(todo.cosine_rank, pa.int32()), "cosine_score": pa.array(todo.cosine_score, pa.float64()),
        "in_pool13": pa.array([False] * len(todo), pa.bool_()), "lang": pa.array(t.lang.tolist(), pa.string()),
        "query_text": pa.array([queries[int(q)] for q in todo.query_id], pa.string()),
        "doc_text": pa.array(t.doc_text.tolist(), pa.string()), "doc_tokens": pa.array(t.doc_tokens.to_numpy(), pa.int32())})
    paths.assert_writable(sub / "shards" / shard_name)
    pq.write_table(shard, sub / "shards" / shard_name, compression="zstd", row_group_size=5_000)
    pd.DataFrame({"pair_id": todo.pair_id, "query_id": todo.query_id, "doc_id": todo.doc_id, "cosine_rank": todo.cosine_rank,
                  "cosine_score": todo.cosine_score, "source": f"newq-{tag}", "free": todo.cosine_rank <= 211, "capped": False}
                 ).to_parquet(sub / "extra_pairs.parquet", index=False)
    q = sorted(todo.query_id.unique())
    man = {"status": "prepared", "tag": tag, "model": "qwen3-reranker-4b", "revision": REVISION_4B, "pairs": int(len(todo)),
           "queries": [int(q[0]), int(q[-1]), len(q)], "pair_id_range": [int(todo.pair_id.min()), int(todo.pair_id.max())],
           "lang_counts": t.lang.value_counts().to_dict(), "doc_tokens_mean": float(t.doc_tokens.mean()),
           "eta_hours_kaggle_2xT4": round(len(todo) / 6.7 / 3600, 2),
           "extra_pairs_sha256": sha256(sub / "extra_pairs.parquet"), "shard_sha256": sha256(sub / "shards" / shard_name),
           "judge_score_py_sha256": JUDGE_SHA, **(extra or {})}
    atomic_json(sub / "manifest.json", man)
    return man


def export_4b(run: Path, top: pd.DataFrame, texts: pd.DataFrame, queries: dict[int, str], parts: int, log, reg_pairs: int = 0) -> dict[str, Any]:
    """``parts`` subsets ``<run>-4b-<k>`` (contiguous query blocks, each its own dataset ``kaggle-dataset-<tag>`` + notebook; the same
    folder is what ``scripts/s40/run_runpod_4b.sh`` takes). ``reg_pairs`` > 0: a regression subset ``<run>-4b-reg`` of that many random
    pairs (seed 40) of part 1, put in the part-1 dataset and scored there FIRST, and also scored by RunPod: the Kaggle / RunPod gate."""
    from r2ai.s36 import d300

    if sha256(d300.JUDGE_SCORE) != JUDGE_SHA:
        raise SystemExit("judge_score.py SHA differs from the pinned S31 scorer")
    out = run / "4b"
    name = run.name
    result = {}
    for k, block in enumerate(split_queries(top.groupby("query_id").size(), parts), 1):
        tag = f"{name}-4b-{k}"
        ds = out / f"kaggle-dataset-{tag}"
        todo = top[top.query_id.isin(block)]
        result[tag] = write_4b_subset(ds / tag, tag, todo, texts, queries, SHARD_4B + k)
        tags = {tag: int(result[tag]["pairs"])}
        if reg_pairs and k == 1:
            reg = f"{name}-4b-reg"
            pick = todo.sample(n=min(reg_pairs, len(todo)), random_state=40)
            result[reg] = write_4b_subset(ds / reg, reg, pick, texts, queries, SHARD_4B + 90, {"regression_gate": True, "subset_of": tag})
            tags = {reg: int(result[reg]["pairs"]), **tags}
        shutil.copyfile(d300.JUDGE_SCORE, ds / "judge_score.py")
        nb = notebook_for(tags, ds)
        (out / f"{tag}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"4B {', '.join(tags)}: {sum(tags.values())} pairs, dataset {ds}")
    return result


def export_4b_sample(run: Path, suffix: str, pairs: int, seed: int) -> dict[str, Any]:
    """A 4B subset ``<run>-4b-<suffix>`` of ``pairs`` random pool pairs in ``<run>/4b/sample-<suffix>/`` (RunPod rehearsal, speed and gate
    against Kaggle). It is never imported into a build (``load_4b`` only reads ``kaggle-dataset-<tag>`` parts)."""
    from r2ai.s36 import d300

    log = log_to(run)
    check_pins(log)
    require(run, ["queries.parquet", "top.parquet", "texts.parquet"])
    top, texts = pd.read_parquet(run / "top.parquet"), pd.read_parquet(run / "texts.parquet")
    tag = f"{run.name}-4b-{suffix}"
    ds = run / "4b" / f"sample-{suffix}"
    man = write_4b_subset(ds / tag, tag, top.sample(n=min(pairs, len(top)), random_state=seed), texts, query_texts(run), SHARD_4B + 80,
                          {"sample": True, "seed": seed})
    shutil.copyfile(d300.JUDGE_SCORE, ds / "judge_score.py")
    log(f"4B sample {tag}: {man['pairs']} pairs in {ds}")
    return man


def ask_chunk_rows(pairs: pd.DataFrame, queries: dict[int, str], qvec: dict[int, np.ndarray], tok, log) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """120ask pairs: top-3 t256 body/answer chunks by cosine of the query vector and the segment's chunk vector (the rule of
    ``r2ai.s37.ingest.cmax_top3`` / ``scripts/s37/d300_cmax.py``), texts cut to 512 Qwen3 tokens; docs without such a chunk on doc_text."""
    from r2ai.s29.qwen_judge import truncate_document
    from r2ai.s37.ingest import STORE

    chunk_rows, fb_rows, handled = [], [], set()
    for seg in sorted(STORE.glob("seg*")):
        if not (seg / "docs.parquet").exists():
            continue
        docs = pd.read_parquet(seg / "docs.parquet", columns=["doc_id", "doc_text"])
        dtext = dict(zip(docs.doc_id.astype(int), docs.doc_text))
        mine = pairs[pairs.doc_id.isin(dtext)]
        if mine.empty:
            continue
        ch = pd.read_parquet(seg / "chunks_t256.parquet", columns=["chunk_id", "doc_ids", "field", "text"])
        dense = np.load(seg / "chunk_vectors.npz")["t256_dense"].astype(np.float32)
        by_doc: dict[int, list[int]] = {}
        for i, (ids, f) in enumerate(zip(ch.doc_ids, ch.field)):
            if f in ("body", "answer"):
                for x in ids:
                    by_doc.setdefault(int(x), []).append(i)
        for r in mine.itertuples():
            idx = by_doc.get(int(r.doc_id), [])
            base = dict(src_pair_id=r.src_pair_id, query_id=r.query_id, doc_id=r.doc_id, cosine_rank=r.cosine_rank, cosine_score=r.cosine_score,
                        in_pool13=True, lang="zh", query_text=queries[int(r.query_id)])
            if idx:
                cos = dense[idx] @ qvec[int(r.query_id)].astype(np.float32)
                for rank, j in enumerate(np.argsort(-cos, kind="stable")[:3], 1):
                    t, n = truncate_document(tok, ch.text.iat[idx[j]])
                    chunk_rows.append({**base, "doc_text": t, "doc_tokens": n, "chunk_id": int(ch.chunk_id.iat[idx[j]]), "chunk_rank": rank,
                                       "chunk_cosine": float(cos[j]), "field": ch.field.iat[idx[j]], "origin": "s", "chunk_source": "s37-chunk"})
            else:
                t, n = truncate_document(tok, dtext[int(r.doc_id)])
                fb_rows.append({**base, "doc_text": t, "doc_tokens": n, "chunk_id": -1, "chunk_rank": 0, "chunk_cosine": np.nan, "field": "doc_text",
                                "origin": "s", "chunk_source": "fallback"})
            handled.add((int(r.query_id), int(r.doc_id)))
    log(f"C-max 120ask: {len(handled)} pairs, {len(chunk_rows)} chunk rows, {len(fb_rows)} doc_text rows")
    return pd.DataFrame(chunk_rows).assign(pair_id=-1), pd.DataFrame(fb_rows).assign(pair_id=-1), handled


def supplement_vectors(out: Path) -> Path:
    """Chunk vectors that the canonical bundle lacks: the S32 supplement plus the chunks S37 embedded for the d300 C-max (corpus data)."""
    from r2ai.s33 import pool as s33_pool

    files = [s33_pool.s32_pool() / "supplement_vectors.npz", paths.RUNS_DIR / "s37" / "cmax-d300" / "supplement_vectors.npz"]
    files = [f for f in files if f.exists()]
    if (out / "supplement_vectors.npz").exists():
        files.append(out / "supplement_vectors.npz")        # embedded here by an earlier attempt
    parts = [np.load(f) for f in files]
    comb = out / "supplement_combined.npz"
    paths.assert_writable(comb)
    np.savez(comb, origin=np.concatenate([p["origin"] for p in parts]), chunk_id=np.concatenate([p["chunk_id"] for p in parts]),
             dense=np.concatenate([p["dense"] for p in parts]))
    return comb


def export_cmax(run: Path, top: pd.DataFrame, texts: pd.DataFrame, queries: dict[int, str], log) -> dict[str, Any]:
    """C-max pool in the S32 / S37 layout (``cmax/pool/shards``, ``judge_score.py``), scored by ``scripts/s37/run_cmax.ps1 -Run <run>/cmax``."""
    from transformers import AutoTokenizer

    from r2ai.s29.qwen_judge import _model_path, truncate_document
    from r2ai.s33 import pool as s33_pool
    from r2ai.s36 import d300

    out = run / "cmax"
    pool_dir = out / "pool"
    if (pool_dir / "manifest.json").exists():
        return json.loads((pool_dir / "manifest.json").read_text(encoding="utf-8"))
    import r2ai.index.bge_m3 as m3                      # the few chunks without a stored vector are embedded on the CPU (never takes the GPU)
    orig = m3.M3Encoder.__init__
    m3.M3Encoder.__init__ = lambda self, device="cpu", fp16=False, max_len=512: orig(self, "cpu", False, max_len)
    s33_pool._prepare_out(out)
    lang = dict(zip(texts.doc_id.astype(int), texts.lang))
    ids, qd = query_vectors(run)
    qvec = {int(q): v for q, v in zip(ids, qd)}
    pairs = pd.DataFrame({"src_pair_id": top.pair_id.astype(np.int64), "query_id": top.query_id.astype(np.int64), "doc_id": top.doc_id.astype(np.int64),
                          "cosine_rank": top.cosine_rank.astype(np.int64), "cosine_score": top.cosine_score.astype(float),
                          "lang": [lang[int(d)] for d in top.doc_id], "source": "newq"})
    tok = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    s_chunks, s_fb, s_keys = ask_chunk_rows(pairs[top.is_ask.to_numpy()], queries, qvec, tok, log)
    rest = pairs[[(int(q), int(d)) not in s_keys for q, d in zip(pairs.query_id, pairs.doc_id)]].reset_index(drop=True)
    sel, fallback, info = s33_pool.select_like_s32(rest, log, supplement_vectors(out), out, qvec=qvec)
    sel = s33_pool.attach_texts(sel, log)
    sel["query_text"] = sel.query_id.map(queries)
    sel["chunk_source"], sel["in_pool13"], sel["pair_id"] = "newq", True, -1
    fb = rest[rest.src_pair_id.isin(fallback.src_pair_id)].copy()
    if len(fb):
        dt = pq.read_table(paths.DATA_DIR / "canonical-20261018" / "docs.parquet", columns=["doc_id", "doc_text"],
                           filters=[("doc_id", "in", sorted(fb.doc_id.unique().tolist()))]).to_pandas()
        text = dict(zip(dt.doc_id.astype(int), dt.doc_text))
        lost = sorted(set(fb.doc_id.astype(int)) - set(text))
        if lost:
            raise SystemExit(f"{len(lost)} doc_text fallback docs are not canonical (e.g. {lost[:5]})")
        cut = {int(d): truncate_document(tok, text[int(d)]) for d in fb.doc_id.unique()}
        fb["doc_text"] = [cut[int(d)][0] for d in fb.doc_id]
        fb["doc_tokens"] = np.array([cut[int(d)][1] for d in fb.doc_id], dtype=np.int32)
    fb["query_text"] = fb.query_id.map(queries)
    fb["chunk_id"], fb["chunk_rank"], fb["chunk_cosine"] = -1, 0, np.nan
    fb["field"], fb["origin"], fb["chunk_source"], fb["in_pool13"], fb["pair_id"] = "doc_text", "d", "fallback", True, -1
    cols = list(s33_pool.RCHK_COLUMNS)
    frames = [sel.assign(src_pair_id=sel.src_pair_id.astype(np.int64))[cols], fb[cols]] + [f[cols] for f in (s_chunks, s_fb) if len(f)]
    rows = pd.concat(frames, ignore_index=True).sort_values(["query_id", "cosine_rank", "doc_id", "chunk_rank"], kind="stable").reset_index(drop=True)
    rows["pair_id"] = CMAX_ROW_BASE + np.arange(len(rows), dtype=np.int64)
    if rows.duplicated(["query_id", "doc_id", "chunk_rank", "chunk_id"]).any():
        raise SystemExit("duplicate C-max rows")
    if set(zip(rows.query_id.astype(int), rows.doc_id.astype(int))) != set(zip(top.query_id.astype(int), top.doc_id.astype(int))):
        raise SystemExit("C-max rows do not cover exactly the pool pairs")
    (pool_dir / "shards").mkdir(parents=True, exist_ok=True)
    shards = s33_pool.write_shards(pool_dir, rows, s33_pool.RCHK_COLUMNS, log)
    s33_pool._atomic_frame(out / "chunk_pairs.parquet", rows[[c for c in cols if c not in ("query_text", "doc_text")]])
    shutil.copyfile(d300.JUDGE_SCORE, pool_dir / "judge_score.py")
    man = {"status": "prepared", "model": "qwen3-reranker-0.6b", "revision": REVISION_06B, "pairs": int(len(top)), "rows": int(len(rows)),
           "rows_per_pair": float(len(rows) / len(top)), "pairs_doc_text": int(len(fb) + len(s_fb)), "pairs_ask": len(s_keys), **info,
           "shards": shards, "judge_score_py_sha256": JUDGE_SHA, "eta_hours_local": round(len(rows) / 28.6 / 3600, 2)}
    atomic_json(pool_dir / "manifest.json", man)
    log(f"C-max pool: {len(rows)} rows for {len(top)} pairs")
    return man


def export_jina(run: Path, top: pd.DataFrame, texts: pd.DataFrame, queries: dict[int, str], log) -> dict[str, Any]:
    """``jina/pool.parquet`` for ``r2ai.s35.jina score`` (b10s0): the S37 rescoring row order (cosine desc, doc_id asc) per query."""
    out = run / "jina"
    if (out / "pool.parquet").exists():
        return json.loads((out / "pool_manifest.json").read_text(encoding="utf-8"))
    t = texts.set_index("doc_id")
    pool = top.sort_values(["query_id", "cosine_score", "doc_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    sub = t.loc[pool.doc_id.to_numpy()]
    frame = pd.DataFrame({"query_id": pool.query_id.astype(np.int64), "doc_id": pool.doc_id.astype(np.int64), "cosine_rank": pool.cosine_rank.astype(np.int64),
                          "cosine_score": pool.cosine_score.astype(float), "p4b": np.nan, "in_pool211": pool.cosine_rank <= 211, "in_pool13": False,
                          "lang": sub.lang.to_numpy(), "query_text": [queries[int(q)] for q in pool.query_id],
                          "doc_text": sub.doc_text.to_numpy(), "doc_tokens": sub.doc_tokens.to_numpy(np.int32)})
    atomic_parquet(frame, out / "pool.parquet")
    man = {"pairs": int(len(frame)), "queries": int(frame.query_id.nunique()), "layout": "b10s0", "row_order": "cosine desc, doc_id asc",
           "pool_sha256": sha256(out / "pool.parquet")}
    atomic_json(out / "pool_manifest.json", man)
    log(f"Jina pool: {len(frame)} pairs")
    return man


def export_scoring(run: Path, parts_4b: int, sources: Sequence[str], reg_pairs: int = 0) -> dict[str, Any]:
    log = log_to(run)
    check_pins(log)
    require(run, ["queries.parquet", "query_vectors.npz", "top.parquet"])
    top = pd.read_parquet(run / "top.parquet")
    queries = query_texts(run)
    t0 = time.monotonic()
    texts = texts_512(run, top, log)
    report: dict[str, Any] = {"texts_seconds": round(time.monotonic() - t0, 1)}
    for source in sources:
        t0 = time.monotonic()
        if source == "4b":
            report["4b"] = export_4b(run, top, texts, queries, parts_4b, log, reg_pairs)
        elif source == "cmax":
            report["cmax"] = export_cmax(run, top, texts, queries, log)
        else:
            report["jina"] = export_jina(run, top, texts, queries, log)
        report[f"{source}_seconds"] = round(time.monotonic() - t0, 1)
    report["next"] = {
        "4b": f"upload each {run / '4b'}/kaggle-dataset-<tag> as a Kaggle dataset, run {run / '4b'}/<tag>.ipynb (2x T4), then import-scores --source 4b --tag <tag>",
        "cmax": f"powershell -File scripts/s37/run_cmax.ps1 -Run {run / 'cmax'}, then import-scores --source cmax",
        "jina": f"powershell -File scripts/s37/run_jina.ps1 -Run {run / 'jina'} -Repo {ROOT}, then import-scores --source jina"}
    previous = json.loads((run / "export.json").read_text(encoding="utf-8")) if (run / "export.json").exists() else {}
    atomic_json(run / "export.json", {**previous, **report})
    return report


# ---------------------------------------------------------------------------------------------------------- 3. import-scores

def import_4b(run: Path, tag: str, scores: Path, report: Path) -> dict[str, Any]:
    found = [d / tag for d in sorted((run / "4b").glob("kaggle-dataset-*")) if (d / tag / "manifest.json").exists()]
    if len(found) != 1:
        raise SystemExit(f"tag {tag}: {len(found)} exported subsets")
    sub = found[0]
    man = json.loads((sub / "manifest.json").read_text(encoding="utf-8"))
    if man.get("regression_gate") or man.get("sample"):
        raise SystemExit(f"{tag} is a regression / sample subset: compare it with scripts/s40/gate_4b.py, do not import it")
    rep = json.loads(report.read_text(encoding="utf-8"))
    sc, ex = pd.read_parquet(scores), pd.read_parquet(sub / "extra_pairs.parquet")
    digest, errors = sha256(scores), []
    if digest != rep.get("scores_sha256"):
        errors.append(f"scores SHA256 {digest} != report {rep.get('scores_sha256')}")
    if rep.get("revisions") != [REVISION_4B] or rep.get("model") != "qwen3-reranker-4b":
        errors.append(f"report model/revision {rep.get('model')} {rep.get('revisions')} != qwen3-reranker-4b {REVISION_4B}")
    if len(sc) != man["pairs"] or sc.pair_id.nunique() != len(sc):
        errors.append(f"{len(sc)} rows / {sc.pair_id.nunique()} unique pair_id, expected {man['pairs']}")
    elif set(sc.pair_id) != set(ex.pair_id):
        errors.append("pair_id set differs from extra_pairs.parquet")
    elif not np.isfinite(sc.score.to_numpy(float)).all():
        errors.append("non-finite scores")
    if errors:
        raise SystemExit("import 4b refused: " + "; ".join(errors))
    dest = run / "4b" / tag
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(scores, dest / "scores-extra.parquet")
    shutil.copyfile(report, dest / "collect-extra-report.json")
    record(run, [f"4b/{tag}/scores-extra.parquet"])
    return {"tag": tag, "pairs": int(len(sc)), "scores_sha256": digest, "gpu_seconds": rep.get("scoring_gpu_seconds_total_all_parts")}


def import_cmax(run: Path, scores: Path | None) -> dict[str, Any]:
    index = pd.read_parquet(run / "cmax" / "chunk_pairs.parquet", columns=["pair_id", "query_id", "doc_id", "chunk_id", "chunk_rank", "chunk_source"])
    scores = scores or run / "cmax" / "gpu" / "qwen3-reranker-0.6b" / "scores-all.parquet"
    sc = pd.read_parquet(scores, columns=["pair_id", "score"])
    if sc.pair_id.duplicated().any() or not np.isfinite(sc.score.to_numpy(float)).all():
        raise SystemExit("duplicate or non-finite C-max scores")
    m = index.merge(sc, on="pair_id", how="left")
    if m.score.isna().any():
        raise SystemExit(f"{int(m.score.isna().sum())} C-max rows have no score")
    atomic_parquet(m[["query_id", "doc_id", "chunk_id", "chunk_rank", "chunk_source", "score"]], run / "cmax_chunks.parquet")
    g = m.groupby(["query_id", "doc_id"])
    res = g.score.max().rename("cmax").reset_index()
    fb = set(zip(m.query_id[m.chunk_source == "fallback"], m.doc_id[m.chunk_source == "fallback"]))
    res["fallback"] = [(q, d) in fb for q, d in zip(res.query_id, res.doc_id)]
    atomic_parquet(res, run / "cmax.parquet")
    record(run, ["cmax.parquet", "cmax_chunks.parquet"])
    return {"pairs": int(len(res)), "rows": int(len(m)), "fallback": int(res.fallback.sum()), "sha256": sha256(run / "cmax.parquet"),
            "scores_sha256": sha256(scores)}


def import_jina(run: Path, scores: Path | None) -> dict[str, Any]:
    scores = scores or run / "jina" / "scores-all.parquet"
    sc = pd.read_parquet(scores, columns=["query_id", "doc_id", "score"])
    top = pd.read_parquet(run / "top.parquet", columns=["query_id", "doc_id"])
    if sc.duplicated(["query_id", "doc_id"]).any() or not np.isfinite(sc.score.to_numpy(float)).all():
        raise SystemExit("duplicate or non-finite Jina scores")
    if set(zip(sc.query_id, sc.doc_id)) != set(zip(top.query_id, top.doc_id)):
        raise SystemExit("Jina scores do not cover exactly the pool pairs")
    atomic_parquet(sc, run / "jina.parquet")
    record(run, ["jina.parquet"])
    return {"pairs": int(len(sc)), "sha256": sha256(run / "jina.parquet"), "scores_sha256": sha256(scores)}


# ---------------------------------------------------------------------------------------------------------- 4. build

class PoolFrame:
    """The part of ``r2ai.s36.cal.Frame`` that ``Calibrated`` / ``ef2_keep`` / ``chosen_rows`` use, built from the new pools only
    (``cal.Frame`` also loads the baseline submission and the S36 union of the public queries). Same ranking: ``fuse.rank_missing_last``
    for Jina / C-max, ``select_rrf`` over ``keys`` with k = ``RRF_K``."""

    def __init__(self, keys: Sequence[str], pools: dict[int, list[dict[str, Any]]], jina: dict, cmax: dict, allow_missing: bool) -> None:
        from r2ai.s31.select import RRF_K, select_rrf
        from r2ai.s36 import fuse

        self.keys, self.jina, self.cmax = tuple(keys), jina, cmax
        self.missing = {"jina": 0, "cmax": 0, "queries_jina": 0, "queries_cmax": 0}
        for q, rows in pools.items():
            for source, lookup, score_key, rank_key in (("jina", jina, "jina_score", "jina_rank"), ("cmax", cmax, "cmax_score", "cmax_rank")):
                n = fuse.rank_missing_last(rows, q, lookup, score_key, rank_key)
                if n and not allow_missing and rank_key in self.keys:
                    raise ValueError(f"query {q}: {n} pairs without {source} score")
                self.missing[source] += n
                self.missing[f"queries_{source}"] += bool(n)
        self.pools = pools
        self.order = {q: select_rrf(rows, len(rows), self.keys, RRF_K) for q, rows in pools.items()}
        self.qids = sorted(pools)

    def fused(self, variant: str, p4b: np.ndarray, jina: np.ndarray) -> np.ndarray:
        if variant != "p4b":
            raise ValueError("the free-v2 frame calibrates P4B only")
        return p4b


def load_4b(run: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for f in sorted((run / "4b").glob("*/scores-extra.parquet")):
        if not (run / "4b" / f"kaggle-dataset-{f.parent.name}").exists():      # only imported parts (import_4b), not a download folder
            continue
        require(run, [f"4b/{f.parent.name}/scores-extra.parquet"])
        frame = pd.read_parquet(f, columns=["pair_id", "score"])
        if set(frame.pair_id.astype(int)) & set(out):
            raise ValueError(f"{f}: pair ids scored twice")
        out.update(zip(frame.pair_id.astype(int), frame.score.astype(float)))
    return out


def cosine_pcal_map(top: pd.DataFrame, scores_4b: dict[int, float], a: float, b: float) -> dict[str, Any]:
    """Missing-4B path (Vu 27/10): ONE global map cosine -> p_cal, ``p_cal ~ sigmoid(c cosine + d)``, least squares on the p_cal of the pool
    pairs of this run that have both a cosine and a 4B score (p_cal = sigmoid(a logit(P4B) + b), the config's a / b)."""
    from scipy.optimize import least_squares

    from r2ai.s36 import cal

    have = top[top.pair_id.astype(int).isin(scores_4b)]
    if len(have) < 1000:
        raise SystemExit(f"only {len(have)} pool pairs have a 4B score: too few to fit the cosine -> p_cal map")
    x = have.cosine_score.to_numpy(float)
    p = cal.sigmoid(a * cal.logit(np.array([scores_4b[int(i)] for i in have.pair_id])) + b)
    fit = least_squares(lambda t: cal.sigmoid(t[0] * x + t[1]) - p, np.array([10.0, -8.0]))
    c, d = map(float, fit.x)
    return {"c": c, "d": d, "pairs": int(len(have)), "rmse": float(np.sqrt(np.mean(fit.fun ** 2))),
            "pcal_mean": float(p.mean()), "pred_mean": float(cal.sigmoid(c * x + d).mean())}


def cos_only_map(cfg: dict[str, Any]) -> dict[str, Any]:
    """``build --cos-only`` (Vu 27/10): the cosine -> p_cal map pinned in the config's ``impute_4b`` (mode ``cos_only``; fitted once on the
    full public 4B), with the config's ``a`` / ``b``. Only this path reads a pinned c / d: ``--impute-4b`` fits its own map (>= 1,000 pairs)."""
    imp = cfg.get("impute_4b") or {}
    if imp.get("mode") != "cos_only" or not all(isinstance(imp.get(k), (int, float)) for k in ("c", "d")):
        raise SystemExit("config: --cos-only needs impute_4b = {mode: cos_only, c, d}")
    return {"c": float(imp["c"]), "d": float(imp["d"]), "a": float(cfg["a"]), "b": float(cfg["b"]), "mode": "cos_only", "source": imp.get("source")}


def cos_k_keep(cfg: dict[str, Any], k: int):
    """``build --cos-k K`` (Vu 27/10, COS-K): the EF2 cut of the COS branch is replaced by the top ``K`` docs of every query in the RRF order
    (cosine + C-max; the 4B source is a constant), never using p_cal to choose ``K``. The config must say the same ``cos_k``."""
    if cfg.get("cos_k") != k:
        raise SystemExit(f"--cos-k {k}: the config says cos_k = {cfg.get('cos_k')}")
    return lambda res: {q: min(k, len(res["frame"].order[q])) for q in res["frame"].qids}


def pools_from(top: pd.DataFrame, scores_4b: dict[int, float], impute: dict[str, float] | None = None) -> dict[int, list[dict[str, Any]]]:
    """Pools in cosine order with the 4B score and rank. ``impute`` (``cosine_pcal_map`` plus the config's ``a`` / ``b``): a pair without
    4B gets the P4B whose p_cal is the mapped one (``sigmoid((logit(p_hat) - b) / a)``, so p_cal, G_q, the EF2 cut and the carrier order use
    p_hat), is flagged ``p4b_imputed`` and takes the last rank of the 4B source in the RRF (``fuse.rank_missing_last`` rule)."""
    from r2ai.s31.select import add_judge_rank
    from r2ai.s36 import cal

    missing = top[~top.pair_id.astype(int).isin(scores_4b)]
    if len(missing) and impute is None:
        raise SystemExit(f"{len(missing)} pool pairs of {missing.query_id.nunique()} queries have no 4B score (import every 4B part first, "
                         "or build --impute-4b)")
    pools = {}
    for q, g in top.sort_values(["query_id", "cosine_rank"]).groupby("query_id", sort=True):
        rows, scored = [], []
        for p, d, c, r in zip(g.pair_id, g.doc_id, g.cosine_score, g.cosine_rank):
            row = {"pair_id": int(p), "doc_id": int(d), "cosine_score": float(c), "cosine_rank": int(r), "origin": "newq", "p4b_imputed": False}
            if int(p) in scores_4b:
                row["judge_score"] = float(scores_4b[int(p)])
                scored.append(row)
            else:
                p_hat = cal.sigmoid(impute["c"] * float(c) + impute["d"])
                row["judge_score"] = float(cal.sigmoid((cal.logit(np.array([p_hat]))[0] - impute["b"]) / impute["a"]))
                row["p4b_imputed"] = True
            rows.append(row)
        add_judge_rank(scored, "judge_score", "judge_rank")
        for row in rows:
            if row["p4b_imputed"]:
                row["judge_rank"] = len(rows)
        pools[int(q)] = rows
    return pools


def select(pools: dict[int, list[dict[str, Any]]], jina: dict, cmax: dict, cfg: dict[str, Any], rrf_keys: Sequence[str],
           allow_missing: bool) -> dict[str, Any]:
    """Free-v2 selection on given pools (``r2ai.s36.s37.apply``, pool ``fixed``): RRF, p_cal, G_q, EF2 cut. Returns the frame, the
    calibration, keep and the chosen rows."""
    from r2ai.s36 import cal

    frame = PoolFrame(rrf_keys, pools, jina, cmax, allow_missing)
    calibrated = cal.Calibrated(frame, "p4b", float(cfg["a"]), float(cfg["b"]), float(cfg["s_free"]), use_tail=True, g_depth=int(cfg["g_depth"]))
    keep = cal.ef2_keep(calibrated)
    return {"frame": frame, "calibrated": calibrated, "keep": keep, "chosen": cal.chosen_rows(frame, keep)}


def choose_carriers(sel: pd.DataFrame, rule: dict[str, Any], queries: int) -> tuple[pd.DataFrame, dict[str, Any], int | None]:
    """Carriers of a docsel (``query_id, doc_id, rank, p4b``). ``rule["order"]``: ``rank`` (S41 CH-rrf, the private default: CR-fair in the
    docsel RRF order, ``r2ai.s41.carriers``) or ``p4b`` (S36 CR-fair, what FREE-noJina used; also when ``order`` is absent)."""
    from r2ai.s36 import carriers as carrier_rules
    from r2ai.s41 import carriers as s41_carriers

    target = int(round(float(rule["target_per_query"]) * queries)) if "target_per_query" in rule else rule.get("target")
    order = rule.get("order", "p4b")
    if order == "p4b":
        carriers, info = carrier_rules.select_carriers(sel, rule.get("rule", "fair"), target, rule.get("drop"), rule.get("rho"))
    else:
        carriers, info = s41_carriers.select(sel, rule.get("rule", "fair"), order, target)
    return carriers, {**info, "order": order}, target


def apply_selection(pools: dict[int, list[dict[str, Any]]], jina: dict, cmax: dict, cfg: dict[str, Any], rrf_keys: Sequence[str],
                    missing_last: bool, name: str, apply_dir: Path, keep_fn=None) -> dict[str, Any]:
    """``select`` + carriers, then ``perq-``, ``carriers-``, ``pcal-``, ``docsel-``, ``keep-<name>`` in ``apply_dir`` (the part of ``build``
    before compose; ``scripts/s41/regress_public.py`` runs it on the 1,200 public queries). ``keep_fn(select result) -> {query: k}``
    replaces the EF2 cut (S41 probe ``r2ai.s41.kfit``; the docs are still the top ``k`` of the RRF order)."""
    from r2ai.s36 import cal
    from r2ai.s36.build import docsel_writer

    res = select(pools, jina, cmax, cfg, rrf_keys, missing_last)
    if keep_fn is not None:
        res["keep"] = keep_fn(res)
        res["chosen"] = cal.chosen_rows(res["frame"], res["keep"])
    frame, calibrated, keep, chosen = res["frame"], res["calibrated"], res["keep"], res["chosen"]
    perq = pd.DataFrame([{"query_id": q, "k": int(keep[q]), "tp": float(calibrated.p[q][: keep[q]].sum()), "g": float(calibrated.g[q])} for q in frame.qids])
    perq["f2"] = 5 * perq.tp / (4 * perq.g + perq.k)
    atomic_parquet(perq, apply_dir / f"perq-{name}.parquet")
    t_c = float(cfg["t_c"])
    sel = pd.DataFrame([(q, int(r["doc_id"]), k, float(r["judge_score"])) for q, rows in chosen.items() for k, r in enumerate(rows, 1)],
                       columns=["query_id", "doc_id", "rank", "p4b"])
    sel["chunk"] = sel.p4b >= t_c
    carriers, carrier_info, target = choose_carriers(sel, cfg["carriers"], len(frame.qids))
    atomic_parquet(carriers, apply_dir / f"carriers-{name}.parquet")
    chunk_ids = {int(q): g.doc_id.astype(int).tolist() for q, g in carriers.groupby("query_id")}
    pcal = pd.DataFrame([{"query_id": q, "doc_id": int(r["doc_id"]), "p_fused": float(f), "p_cal": float(p)}
                         for q in frame.qids for r, f, p in zip(frame.order[q], calibrated.fused[q], calibrated.p[q])])
    atomic_parquet(pcal, apply_dir / f"pcal-{name}.parquet")
    docsel_writer(apply_dir, "", t_c, {name: pcal})(name, chosen, chunk_ids, {"t_c": t_c})
    atomic_json(apply_dir / f"keep-{name}.json", {str(q): int(k) for q, k in keep.items()})
    return {**res, "carriers": carriers, "carrier_info": carrier_info, "target": target}


def build(run: Path, name: str | None, no_jina: bool, no_cmax: bool, missing_last: bool, rounds: int, compose: bool = True,
          impute_4b: bool = False, drop_4b: float = 0.0, drop_seed: int = 40, cos_only: bool = False, cos_config: Path | None = None,
          cos_k: int | None = None, en_hedge: int = 0, en_min_cos: float | None = None) -> dict[str, Any]:
    import subprocess
    import sys

    from r2ai.s36 import cal, fuse
    from r2ai.s36 import carriers as carrier_rules

    if cos_k is not None:                          # COS-K: the COS branch with a fixed K, from its own config
        cos_only, cos_config = True, cos_config or Path(str(COS_K_CONFIG).format(k=cos_k))
    if cos_only and (impute_4b or drop_4b):
        raise SystemExit("--cos-only uses no 4B score at all: not with --impute-4b / --test-drop-4b")
    log = log_to(run)
    check_pins(log)
    cfg = run_config(run)
    fallback = None
    if cos_only:                                   # the run's own config must be the one the fallback config inherits
        fallback = load_config(cos_config or COS_CONFIG)
        differ = [k for k in SELECTION_KEYS if fallback.get(k) != cfg.get(k)]
        if differ:
            raise SystemExit(f"--cos-only: {cos_config or COS_CONFIG} differs from the run config on {differ}")
    require(run, ["queries.parquet", "query_vectors.npz", "top.parquet"] + ([] if no_jina else ["jina.parquet"]) + ([] if no_cmax else ["cmax.parquet", "cmax_chunks.parquet"]))
    name = name or (f"{run.name}-free" + ("" if no_jina else "-jina") + ("-noCmax" if no_cmax else "")
                    + (f"-drop4b{round(drop_4b * 100)}" if drop_4b else "") + ("-impute4b" if impute_4b else "") + (("-cosonly" if cos_k is None else f"-cosk{cos_k}") if cos_only else ""))
    work = run / "build" / name
    apply_dir, compose_dir = work / "apply", work / "compose"
    paths.assert_writable(work)
    apply_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    top = pd.read_parquet(run / "top.parquet")
    scores_4b = {} if cos_only else load_4b(run)
    dropped = 0
    if drop_4b:                                    # test only: forget the 4B score of a random share of the pool pairs
        ids = top.pair_id.astype(int).to_numpy()
        gone = set(np.random.default_rng(drop_seed).choice(ids, size=int(round(drop_4b * len(ids))), replace=False).tolist())
        scores_4b = {k: v for k, v in scores_4b.items() if k not in gone}
        dropped = len(gone)
    impute = None
    if impute_4b:
        impute = {**cosine_pcal_map(top, scores_4b, float(cfg["a"]), float(cfg["b"])), "a": float(cfg["a"]), "b": float(cfg["b"])}
        log(f"impute 4B: cosine -> p_cal map {json.dumps(impute)}")
    if cos_only:
        impute = cos_only_map(fallback)
        log(f"cos-only: no 4B score read, every pair from the pinned map {json.dumps(impute)}")
    pools = pools_from(top, scores_4b, impute)
    jina, cmax = {}, {}
    if not no_jina:
        j = pd.read_parquet(run / "jina.parquet")
        jina = dict(zip(zip(j.query_id.astype(int), j.doc_id.astype(int)), j.score.astype(float)))
    if not no_cmax:
        c = pd.read_parquet(run / "cmax.parquet")
        cmax = dict(zip(zip(c.query_id.astype(int), c.doc_id.astype(int)), c.cmax.astype(float)))
    rrf_keys = [k for k in fuse.FOUR_KEYS if not (no_jina and k == "jina_rank") and not (no_cmax and k == "cmax_rank")]
    keep_fn = cos_k_keep(fallback, cos_k) if cos_k is not None else None
    applied = apply_selection(pools, jina, cmax, cfg, rrf_keys, missing_last, name, apply_dir, keep_fn)
    frame, calibrated, keep, chosen = applied["frame"], applied["calibrated"], applied["keep"], applied["chosen"]
    carriers, carrier_info, target = applied["carriers"], applied["carrier_info"], applied["target"]
    imputed = pd.DataFrame([{"query_id": q, "doc_id": int(r["doc_id"]), "pair_id": int(r["pair_id"]), "cosine_score": float(r["cosine_score"]),
                             "p4b_imputed": float(r["judge_score"])} for q, rows in pools.items() for r in rows if r["p4b_imputed"]],
                           columns=["query_id", "doc_id", "pair_id", "cosine_score", "p4b_imputed"])
    if impute is not None:
        atomic_parquet(imputed, apply_dir / f"imputed-4b-{name}.parquet")
    chosen_keys = {(q, int(r["doc_id"])) for q, rows in chosen.items() for r in rows}
    report: dict[str, Any] = {
        "name": name, "run": str(run), "queries": len(frame.qids), "rrf_keys": rrf_keys, "missing": frame.missing,
        "missing_4b": {"test_dropped_pairs": dropped, "drop_seed": drop_seed if drop_4b else None, "imputed_pairs": int(len(imputed)),
                       "imputed_chosen": int(sum((q, d) in chosen_keys for q, d in zip(imputed.query_id, imputed.doc_id))),
                       "imputed_carriers": int(len(carriers.merge(imputed[["query_id", "doc_id"]], on=["query_id", "doc_id"]))),
                       "map": impute, "cos_only": cos_only, "cos_k": cos_k,
                       "cos_config": {"path": str(cos_config or COS_CONFIG), "sha256": sha256(cos_config or COS_CONFIG)} if cos_only else None},
        "global": {k: cfg[k] for k in ("a", "b", "s_free", "g_depth", "t_c", "depth")}, "docs_per_query": cal.keep_stats(keep),
        "predicted": calibrated.expected(keep), "carriers": {**carrier_info, "target": target, **carrier_rules.describe(carriers, len(frame.qids))},
        "apply_seconds": round(time.monotonic() - t0, 1),
        "outputs": {p.name: sha256(p) for p in sorted(apply_dir.glob(f"*{name}*")) if p.is_file()}}
    report["carriers"].pop("per_query_p10_p50_p90", None)
    atomic_json(work / "build.json", report)
    log(f"build {name}: {report['docs_per_query']['total']} docs, predicted Doc F2 {report['predicted']['f2']:.4f}, {len(carriers)} carriers")
    if not compose:
        return report
    compose_dir.mkdir(parents=True, exist_ok=True)
    base = f"{name}-full+w768"
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-u", "-m", "r2ai.s38.compose", "run", "--name", base, "--docsel", str(apply_dir / f"docsel-{name}.parquet"),
           "--rule", "full+w768", "--carriers", str(apply_dir / f"carriers-{name}.parquet"), "--fit-carriers", "--rounds", str(rounds),
           "--out", str(compose_dir), "--query-set", str(run)]
    t0 = time.monotonic()
    with open(work / "compose.log", "ab") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    report["compose_seconds"] = round(time.monotonic() - t0, 1)
    report["compose_returncode"] = rc
    found = sorted(compose_dir.glob(f"{base}*-t*.compose.json"))
    results = [json.loads(p.read_text(encoding="utf-8")) for p in found]
    ok = [r for r in results if r.get("under_ceiling") and r.get("validator", {}).get("ok")]
    if ok:
        best = min(ok, key=lambda r: r["threshold"])
        report["zip"] = {k: best.get(k) for k in ("zip", "zip_bytes", "zip_sha256", "threshold", "dropped_carriers", "docs", "chunks", "second_chunks")}
        report["zip"]["validator"] = best["validator"]
        if en_hedge:                                # S43 probe: en docs after the vi/zh docs (r2ai.s43.enhedge), off by default
            from r2ai.s38 import registry as reg
            from r2ai.s43 import enhedge

            reg.use_query_set(run)
            q_ids, q_dense = query_vectors(run)
            thr = en_min_cos if en_min_cos is not None else enhedge.all_pairs_percentiles(q_dense, (99,))["p99"]
            hedge = enhedge.choose(enhedge.lane_pairs(q_ids, q_dense, en_hedge), en_hedge, thr)
            report["en_hedge"] = {"n": en_hedge, "min_cos": thr, **enhedge.run(Path(best["zip"]), hedge, compose_dir,
                                                                                f"{best['name']}-en{en_hedge}", best["zip_sha256"])}
    atomic_json(work / "build.json", report)
    log(f"compose {name}: rc={rc}, zip={report.get('zip', {}).get('zip')}")
    return report


def status(run: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run": str(run)}
    for f in ("queries.parquet", "query_vectors.npz", "pool.npz", "top.parquet", "texts.parquet", "cmax.parquet", "jina.parquet"):
        out[f] = (run / f).exists()
    out["4b_parts_exported"] = sorted(p.name for p in (run / "4b").glob("kaggle-dataset-*")) if (run / "4b").exists() else []
    out["4b_parts_imported"] = sorted(p.parent.name for p in (run / "4b").glob("*/scores-extra.parquet") if (run / "4b" / f"kaggle-dataset-{p.parent.name}").exists()) if (run / "4b").exists() else []
    out["builds"] = sorted(p.name for p in (run / "build").iterdir()) if (run / "build").exists() else []
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    e = sub.add_parser("export-scoring")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--parts-4b", type=int, default=1, help="4B datasets (one Kaggle session / machine each); private: 2 (Kaggle + RunPod)")
    e.add_argument("--reg-pairs", type=int, default=0, help="Kaggle / RunPod regression subset of part 1 (private: 1000)")
    e.add_argument("--sources", default="4b,cmax", help="add ,jina only if Jina is wanted (not in the private pipeline)")
    i = sub.add_parser("import-scores")
    i.add_argument("--run", type=Path, required=True)
    i.add_argument("--source", choices=SOURCES, required=True)
    i.add_argument("--tag", default=None)
    i.add_argument("--scores", type=Path, default=None)
    i.add_argument("--report", type=Path, default=None)
    b = sub.add_parser("build")
    b.add_argument("--run", type=Path, required=True)
    b.add_argument("--name", default=None)
    b.add_argument("--jina", action="store_true", help="add Jina to the RRF (not in the private pipeline)")
    b.add_argument("--no-jina", action="store_true", help="kept for old commands: Jina is off by default")
    b.add_argument("--no-cmax", action="store_true")
    b.add_argument("--missing-last", action="store_true", help="a pair without a Jina / C-max score takes the last rank (default: refuse)")
    b.add_argument("--impute-4b", action="store_true", help="a pair without 4B: p_cal from a global cosine map, last in the 4B source (default: refuse)")
    b.add_argument("--cos-only", action="store_true", help="no 4B at all: every pair from the cosine map pinned in --cos-config (fallback COS)")
    b.add_argument("--cos-k", type=int, default=None, help="COS with the top K docs per query in RRF order instead of EF2 (config cos-k<K>)")
    b.add_argument("--cos-config", type=Path, default=None, help=f"default {COS_CONFIG.relative_to(ROOT).as_posix()} ({COS_K_CONFIG.name} with --cos-k)")
    b.add_argument("--test-drop-4b", type=float, default=0.0, help="TEST ONLY: forget the 4B score of this share of the pool pairs")
    b.add_argument("--test-drop-seed", type=int, default=40)
    b.add_argument("--rounds", type=int, default=10)
    b.add_argument("--no-compose", action="store_true", help="docsel / carriers only")
    b.add_argument("--en-hedge", type=int, default=0, help="S43 probe: up to N en docs per query appended after the ZIP docs (0 = frozen private config)")
    b.add_argument("--en-min-cos", type=float, default=None, help="S43 --en-hedge: min en cosine (default: P99 over all (query, en doc) pairs)")
    s = sub.add_parser("status")
    s.add_argument("--run", type=Path, required=True)
    sp = sub.add_parser("export-4b-sample", help="RunPod rehearsal / gate subset (never imported)")
    sp.add_argument("--run", type=Path, required=True)
    sp.add_argument("--suffix", required=True)
    sp.add_argument("--pairs", type=int, required=True)
    sp.add_argument("--seed", type=int, default=40)
    pp = sub.add_parser("pins")
    pp.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "pins":
        print(json.dumps(write_pins() if args.write else check_pins(), indent=1))
        return 0
    run = paths.resolve_path(args.run)
    if not run.is_relative_to(paths.RUNS_DIR):
        raise SystemExit(f"--run must stay below RUNS_DIR ({paths.RUNS_DIR}): {run}")
    if args.cmd == "prepare":
        result = prepare(paths.resolve_path(args.queries), run, args.config, args.device)
    elif args.cmd == "export-scoring":
        result = export_scoring(run, args.parts_4b, [x for x in args.sources.split(",") if x], args.reg_pairs)
    elif args.cmd == "import-scores":
        if args.source == "4b":
            if not (args.tag and args.scores and args.report):
                parser.error("--source 4b needs --tag, --scores and --report")
            result = import_4b(run, args.tag, args.scores, args.report)
        elif args.source == "cmax":
            result = import_cmax(run, args.scores)
        else:
            result = import_jina(run, args.scores)
        log_to(run)(f"import {args.source}: {json.dumps(result, default=str)}")
    elif args.cmd == "export-4b-sample":
        result = export_4b_sample(run, args.suffix, args.pairs, args.seed)
    elif args.cmd == "build":
        result = build(run, args.name, not args.jina, args.no_cmax, args.missing_last, args.rounds, not args.no_compose,
                       args.impute_4b, args.test_drop_4b, args.test_drop_seed, args.cos_only,
                       paths.resolve_path(args.cos_config) if args.cos_config else None, args.cos_k, args.en_hedge, args.en_min_cos)
    else:
        result = status(run)
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
