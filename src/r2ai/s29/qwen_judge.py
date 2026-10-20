"""Build cosine candidate pools and score them with the pinned Qwen3 reranker.

Run with ``python -m r2ai.s29.qwen_judge prepare`` to make the cosine pool,
``python -m r2ai.s29.qwen_judge score --pilot`` for the 200-pair pilot, and
``python -m r2ai.s29.qwen_judge score --full`` to resume/full-score it.
All outputs stay below ``r2ai.paths.RUNS_DIR/s29-qwen-judge``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
)
PREFIX = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INSTRUCTIONS = {
    "vi": "Given a Vietnamese medical question, judge whether the Vietnamese passage answers it",
    "zh": "Given a Vietnamese medical question, judge whether the Chinese passage answers it",
}
BASELINE_NAME = "P-A4-K90-ph"
P1_NAME = "P1-vi+10"
RRF_K = 60
DOC_TOKEN_LIMIT = 512
GPU_FREE_MIN_MIB = 2560
DEFAULT_BATCH_SIZE = 8
CHECKPOINT_PAIRS = 200
DATE_TIME = (2026, 10, 19, 0, 0, 0)


def candidate_pool_size(raw_k: int, multiplier: float = 1.3) -> int:
    """S29 uses Python's round, matching the S26 builders' K/n_c rounding."""
    if raw_k < 0:
        raise ValueError("raw_k must be nonnegative")
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be finite and positive")
    return round(raw_k * multiplier)


def _validate_pilot_report(
    report: dict[str, Any], expected_pairs: int, expected_candidate_sha256: str
) -> None:
    required = min(CHECKPOINT_PAIRS, expected_pairs)
    if report.get("status") != "pilot_complete":
        raise ValueError("A completed 200-pair pilot is required before full scoring")
    if report.get("model_revision") != MODEL_REVISION:
        raise ValueError("Pilot report uses a different Qwen model revision")
    if int(report.get("pairs_total", -1)) != expected_pairs:
        raise ValueError("Pilot report belongs to a different candidate pool size")
    if report.get("candidate_pool_sha256") != expected_candidate_sha256:
        raise ValueError("Pilot report belongs to a different candidate pool SHA256")
    if int(report.get("pairs_scored_this_run", 0)) < required or int(
        report.get("pairs_checkpointed_total", 0)
    ) < required:
        raise ValueError(f"Pilot report must include at least {required} scored/checkpointed pairs")


def select_qfilter(candidates: Sequence[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Keep the highest Qwen scores; stable ties prefer the cosine order."""
    if not 0 <= keep <= len(candidates):
        raise ValueError(f"keep={keep} is outside candidate count {len(candidates)}")
    return sorted(
        candidates,
        key=lambda row: (-float(row["qwen_score"]), int(row["cosine_rank"]), int(row["doc_id"])),
    )[:keep]


def select_qrrf(
    candidates: Sequence[dict[str, Any]], keep: int, rrf_k: int = RRF_K
) -> list[dict[str, Any]]:
    """Select by reciprocal-rank fusion of cosine and Qwen ranks."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if not 0 <= keep <= len(candidates):
        raise ValueError(f"keep={keep} is outside candidate count {len(candidates)}")
    ranked = []
    for row in candidates:
        score = 1.0 / (rrf_k + int(row["cosine_rank"])) + 1.0 / (
            rrf_k + int(row["qwen_rank"])
        )
        ranked.append((score, row))
    ranked.sort(
        key=lambda item: (
            -item[0],
            int(item[1]["cosine_rank"]),
            int(item[1]["qwen_rank"]),
            int(item[1]["doc_id"]),
        )
    )
    return [{**row, "rrf_score": score} for score, row in ranked[:keep]]


def select_weighted_qrrf(
    candidates: Sequence[dict[str, Any]],
    keep: int,
    rrf_k: int = RRF_K,
    qwen_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Select with (1-w)/(k+cosine-rank) + w/(k+Qwen-rank)."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if not math.isfinite(qwen_weight) or not 0.0 <= qwen_weight <= 1.0:
        raise ValueError("qwen_weight must be finite and between 0 and 1")
    if not 0 <= keep <= len(candidates):
        raise ValueError(f"keep={keep} is outside candidate count {len(candidates)}")
    cosine_weight = 1.0 - qwen_weight
    ranked = []
    for row in candidates:
        score = cosine_weight / (rrf_k + int(row["cosine_rank"])) + qwen_weight / (
            rrf_k + int(row["qwen_rank"])
        )
        ranked.append((score, row))
    ranked.sort(
        key=lambda item: (
            -item[0],
            int(item[1]["cosine_rank"]),
            int(item[1]["qwen_rank"]),
            int(item[1]["doc_id"]),
        )
    )
    return [{**row, "rrf_score": score} for score, row in ranked[:keep]]


def _validate_pool_extension_alignment(
    source: pd.DataFrame, target: pd.DataFrame
) -> pd.DataFrame:
    """Require shared scores to match and dropped source rows to tie at the new cutoff."""
    keys = ["query_id", "doc_id"]
    required = set(keys + ["cosine_rank", "cosine_score"])
    for label, frame in (("source", source), ("target", target)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{label} candidate pool lacks required columns {sorted(required)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} candidate pool contains duplicate query/document pairs")
    aligned = source.merge(
        target,
        on=keys,
        how="left",
        suffixes=("_source", "_target"),
        validate="one_to_one",
        indicator=True,
    )
    target_boundary = target.groupby("query_id")["cosine_score"].min()
    aligned["target_boundary_score"] = aligned["query_id"].map(target_boundary)
    missing = aligned["_merge"].eq("left_only")
    if missing.any() and not np.allclose(
        aligned.loc[missing, "cosine_score_source"].to_numpy(dtype=np.float64),
        aligned.loc[missing, "target_boundary_score"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("missing source pairs are not tied at the expanded cutoff")
    shared = aligned["_merge"].eq("both")
    if not np.allclose(
        aligned.loc[shared, "cosine_score_source"].to_numpy(dtype=np.float64),
        aligned.loc[shared, "cosine_score_target"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("Expanded candidate pool changed a source cosine score")
    return aligned


def _select_pilot_pairs(
    pairs: list[dict[str, Any]], done: set[tuple[int, int]], limit: int = CHECKPOINT_PAIRS
) -> list[dict[str, Any]]:
    pending = [
        row for row in pairs if (int(row["query_id"]), int(row["doc_id"])) not in done
    ]
    random.Random(42).shuffle(pending)
    return pending[: min(limit, len(pending))]


def truncate_document(tokenizer: Any, text: str, limit: int = DOC_TOKEN_LIMIT) -> tuple[str, int]:
    """Return a decoded leading passage of at most ``limit`` tokenizer tokens."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    token_ids = tokenizer.encode(text or "", add_special_tokens=False)[:limit]
    passage = tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    # Some tokenizers normalize on decode/re-encode. Keep the passage within the
    # requested budget after reconstruction as well.
    while len(tokenizer.encode(passage, add_special_tokens=False)) > limit:
        token_ids = token_ids[:-1]
        passage = tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return passage, len(tokenizer.encode(passage, add_special_tokens=False))


def run_dir() -> Path:
    return paths.run_dir("s29-qwen-judge")


def _inputs() -> dict[str, Path]:
    out = paths.OUT_DIR
    return {
        "p1": out / "runs" / "reranker-trial" / "release" / f"{P1_NAME}.zip",
        "baseline": out / "runs" / "s26-gold-structure" / "pa4" / f"{BASELINE_NAME}.zip",
        "baseline_build": out / "runs" / "s26-gold-structure" / "pa4" / f"{BASELINE_NAME}.build.json",
        "vectors": out / "runs" / "s26-gold-structure" / "full" / "full_doc_dense.npz",
        "query": paths.DATA_DIR / "raw" / "query.parquet",
        "canonical": paths.DATA_DIR / "canonical-20261018",
        "extra_docs": out / "runs" / "s27-pharmacity" / "handoff" / "cc-s27-pharmacity",
        "extra_vectors": out / "runs" / "s27-pharmacity" / "vectors" / "handoff",
    }


def _guard_outputs(out: Path) -> None:
    """Preflight every output root and temporary/lock/checkpoint target before I/O."""
    targets = [
        out,
        out / "candidate_pool.parquet",
        out / "candidate_pool.parquet.tmp",
        out / "query_vectors.npz",
        out / "query_vectors.npz.tmp",
        out / "query_vectors.meta.json",
        out / "query_vectors.meta.json.tmp",
        out / "prepare.json",
        out / "prepare.json.tmp",
        out / "prepare.log",
        out / "qwen_scores.sqlite",
        out / "qwen_scores.sqlite-wal",
        out / "qwen_scores.sqlite-shm",
        out / "score.lock",
        out / "score.log",
        out / "pilot-report.json",
        out / "pilot-report.json.tmp",
        out / "full-score-report.json",
        out / "full-score-report.json.tmp",
        out / "scores.parquet",
        out / "scores.parquet.tmp",
        out / "Q-filter.zip",
        out / f"Q-filter.{os.getpid()}.zip.tmp",
        out / "Q-filter.query_stats.parquet",
        out / "Q-filter.query_stats.parquet.tmp",
        out / "Q-filter.build.json",
        out / "Q-filter.build.json.tmp",
        out / "Q-rrf.zip",
        out / f"Q-rrf.{os.getpid()}.zip.tmp",
        out / "Q-rrf.query_stats.parquet",
        out / "Q-rrf.query_stats.parquet.tmp",
        out / "Q-rrf.build.json",
        out / "Q-rrf.build.json.tmp",
    ]
    for name in (
        "QRRF-nc140-ph",
        "QRRF-k20",
        "QRRF-k200",
        "QRRF-wq0.65",
        "QRRF-wq0.35",
        "QRRF-pool16",
    ):
        targets.extend(
            [
                out / f"{name}.zip",
                out / f"{name}.{os.getpid()}.zip.tmp",
                out / f"{name}.query_stats.parquet",
                out / f"{name}.query_stats.parquet.{os.getpid()}.tmp",
                out / f"{name}.build.json",
                out / f"{name}.build.json.tmp",
            ]
        )
    targets.extend(
        [
            out / "seed-report.json",
            out / "seed-report.json.tmp",
        ]
    )
    for target in targets:
        paths.assert_writable(target)


def _require_under_runs(path: Path) -> Path:
    resolved = paths.resolve_path(path)
    if not resolved.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S29 output must stay below RUNS_DIR: {resolved}")
    paths.assert_writable(resolved)
    return resolved


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    paths.assert_writable(temp)
    with temp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    temp.replace(path)


def _iter_zip_rows(path: Path) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if len(names) != 1:
            raise ValueError(f"Expected one JSON member in {path}, got {names}")
        with archive.open(names[0]) as fh:
            for line in fh:
                raw = line.strip().rstrip(b",")
                if raw.startswith(b"{"):
                    yield json.loads(raw)


def _load_submission(path: Path, with_chunks: bool = False) -> list[dict[str, Any]]:
    rows = []
    for raw in _iter_zip_rows(path):
        row = {
            "id": int(raw["id"]),
            "relevant_docs": [int(doc) for doc in raw["relevant_docs"]],
            "nc": len(raw.get("relevant_chunks", [])),
        }
        if with_chunks:
            row["relevant_chunks"] = raw.get("relevant_chunks", [])
        rows.append(row)
    if not rows:
        raise ValueError(f"No submission rows found in {path}")
    return rows


def _load_query_vectors(
    query_path: Path,
    query_ids: list[int],
    out: Path,
    report_progress: Callable[[str], None],
    source_dir: Path,
) -> tuple[np.ndarray, str]:
    query_hash = _sha256(query_path)
    source_path = paths.resolve_path(source_dir)
    source_dense = source_path / "q_dense.npy"
    source_ids = source_path / "q_ids.npy"
    paths.require_inputs(source_dense)
    paths.require_inputs(source_ids)
    source_hashes = {
        "q_dense_sha256": _sha256(source_dense),
        "q_ids_sha256": _sha256(source_ids),
    }
    cache = out / "query_vectors.npz"
    meta_path = out / "query_vectors.meta.json"
    if cache.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("query_sha256") == query_hash
            and meta.get("query_ids") == query_ids
            and meta.get("source_dir") == (str(source_path) if source_path is not None else None)
            and meta.get("source_hashes") == source_hashes
        ):
            vectors = np.load(cache)
            cached_ids = vectors["q_ids"].astype(int).tolist()
            cached_dense = vectors["q_dense"]
            source_dense_values = np.load(source_dense, allow_pickle=False)
            cache_valid = (
                cached_ids == query_ids
                and cached_dense.shape == (len(query_ids), 1024)
                and cached_dense.dtype == np.float16
                and np.isfinite(cached_dense).all()
                and np.array_equal(cached_dense, source_dense_values)
            )
            if cache_valid:
                q_dense = np.array(cached_dense, copy=True)
                vectors.close()
                report_progress(f"query_vectors cache hit rows={len(query_ids)} sha256={query_hash}")
                return q_dense, str(meta.get("encoder", "BAAI/bge-m3"))
            vectors.close()

    source_query_ids = np.load(source_ids, allow_pickle=False)
    q_dense = np.load(source_dense, allow_pickle=False)
    if source_query_ids.ndim != 1 or source_query_ids.astype(np.int64).tolist() != query_ids:
        raise ValueError("S26 q_ids.npy does not exactly match sorted P1/K90-ph query IDs")
    if q_dense.shape != (len(query_ids), 1024) or q_dense.dtype != np.float16:
        raise ValueError(
            f"Expected S26 float16 query vectors with shape {(len(query_ids), 1024)}, "
            f"got shape={q_dense.shape} dtype={q_dense.dtype}"
        )
    if not np.isfinite(q_dense).all():
        raise ValueError("S26 query vectors contain non-finite values")
    q_dense = np.array(q_dense, copy=True)
    encoder = "BAAI/bge-m3; exact S26 q_dense.npy fp16"
    device = "S26 source vectors"
    dtype_description = "float16 (preserved exactly from S26 q_dense.npy)"
    report_progress(
        f"loaded exact S26 query vectors rows={len(query_ids)} "
        f"q_dense_sha256={source_hashes['q_dense_sha256']} "
        f"q_ids_sha256={source_hashes['q_ids_sha256']}"
    )

    if q_dense.shape != (len(query_ids), 1024):
        raise ValueError(f"Unexpected BGE query vector shape: {q_dense.shape}")

    temp = cache.with_name(cache.name + ".tmp")
    paths.assert_writable(temp)
    with temp.open("wb") as fh:
        np.savez(fh, q_ids=np.asarray(query_ids, dtype=np.int64), q_dense=q_dense)
        fh.flush()
        os.fsync(fh.fileno())
    temp.replace(cache)
    meta = {
        "query_sha256": query_hash,
        "query_ids": query_ids,
        "encoder": encoder,
        "max_length": 512,
        "device": device,
        "dtype": dtype_description,
        "source_dir": str(source_path) if source_path is not None else None,
        "source_hashes": source_hashes,
    }
    _atomic_json(meta_path, meta)
    return q_dense, encoder


def _load_doc_vectors(inputs: dict[str, Path]) -> tuple[np.ndarray, np.ndarray, int, list[Path]]:
    base = np.load(inputs["vectors"])
    ids = [base["doc_id"].astype(np.int64)]
    dense = [base["dense"]]
    base_count = len(ids[0])
    base.close()
    extra_files = []
    lot_root = inputs["extra_docs"] / "lots"
    vector_root = inputs["extra_vectors"]
    extra_ids = []
    for lot in sorted(path for path in lot_root.iterdir() if path.is_dir()):
        doc_file = lot / "chunks" / "docs.parquet"
        vector_file = vector_root / lot.name / "dense_doc_level.npz"
        docs = pd.read_parquet(doc_file, columns=["doc_id"])
        vectors = np.load(vector_file)
        actual_ids = vectors["doc_id"].astype(np.int64)
        if sorted(actual_ids.tolist()) != sorted(docs.doc_id.astype(np.int64).tolist()):
            raise ValueError(f"Extra document/vector IDs differ for {lot.name}")
        if len(set(actual_ids.tolist())) != len(actual_ids):
            raise ValueError(f"Duplicate extra document IDs in {lot.name}")
        ids.append(actual_ids)
        dense.append(vectors["dense"])
        vectors.close()
        extra_ids.extend(actual_ids.tolist())
        extra_files.append(vector_file)
    all_ids = np.concatenate(ids)
    if len(set(all_ids.tolist())) != len(all_ids):
        raise ValueError("Duplicate doc_id across canonical and Pharmacity vectors")
    return all_ids, np.concatenate(dense), base_count, extra_files


def prepare(
    out: Path, query_vectors_dir: Path, pool_multiplier: float = 1.3
) -> dict[str, Any]:
    if not math.isfinite(pool_multiplier) or pool_multiplier <= 0:
        raise ValueError("pool_multiplier must be finite and positive")
    prepare_started = time.monotonic()
    inputs = _inputs()
    for path in inputs.values():
        paths.require_inputs(path)
    _guard_outputs(out)
    out.mkdir(parents=True, exist_ok=True)
    def report_progress(message: str) -> None:
        event = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "message": message,
        }
        line = json.dumps(event, ensure_ascii=False)
        print(line, flush=True)
        with (out / "prepare.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(line + "\n")

    report_progress("start prepare")
    p1_rows = _load_submission(inputs["p1"])
    baseline_rows = _load_submission(inputs["baseline"])
    p1_by_id = {row["id"]: row for row in p1_rows}
    baseline_by_id = {row["id"]: row for row in baseline_rows}
    if set(p1_by_id) != set(baseline_by_id):
        raise ValueError("P1 and K90-ph query IDs differ")
    query_ids = sorted(p1_by_id)
    q_dense, encoder = _load_query_vectors(
        inputs["query"], query_ids, out, report_progress, source_dir=query_vectors_dir
    )
    raw_k = {qid: len(p1_by_id[qid]["relevant_docs"]) for qid in query_ids}
    keep_k = {qid: len(baseline_by_id[qid]["relevant_docs"]) for qid in query_ids}
    pool_k = {
        qid: candidate_pool_size(raw_k[qid], multiplier=pool_multiplier) for qid in query_ids
    }
    max_pool = max(pool_k.values())

    doc_ids, doc_dense, base_doc_count, extra_files = _load_doc_vectors(inputs)
    qpos = {qid: index for index, qid in enumerate(query_ids)}
    top_i = np.zeros((len(query_ids), 0), dtype=np.int64)
    top_s = np.zeros((len(query_ids), 0), dtype=np.float32)
    start = time.monotonic()
    for begin in range(0, len(doc_ids), 50_000):
        scores = q_dense.astype(np.float32) @ doc_dense[begin : begin + 50_000].astype(np.float32).T
        take = min(max_pool, scores.shape[1])
        part = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        top_i = np.c_[top_i, part + begin]
        top_s = np.c_[top_s, np.take_along_axis(scores, part, axis=1)]
        keep = np.argsort(-top_s, axis=1, kind="stable")[:, :max_pool]
        top_i = np.take_along_axis(top_i, keep, axis=1)
        top_s = np.take_along_axis(top_s, keep, axis=1)
        if (begin // 50_000 + 1) % 4 == 0 or begin + 50_000 >= len(doc_ids):
            report_progress(
                f"cosine {min(begin + 50_000, len(doc_ids))}/{len(doc_ids)} docs "
                f"elapsed_s={time.monotonic() - start:.0f}"
            )
    del doc_dense

    candidate_rows = []
    for qid in query_ids:
        qi = qpos[qid]
        n = pool_k[qid]
        for rank, vector_row in enumerate(top_i[qi, :n], 1):
            candidate_rows.append(
                {
                    "query_id": qid,
                    "doc_id": int(doc_ids[vector_row]),
                    "cosine_rank": rank,
                    "cosine_score": float(top_s[qi, rank - 1]),
                }
            )
    candidate = pd.DataFrame(candidate_rows)
    if candidate.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Duplicate doc in a query's cosine pool")
    candidate_path = out / "candidate_pool.parquet"
    temp = candidate_path.with_name(candidate_path.name + ".tmp")
    paths.assert_writable(temp)
    candidate.to_parquet(temp, index=False)
    temp.replace(candidate_path)

    baseline_build = json.loads(inputs["baseline_build"].read_text(encoding="utf-8"))
    query_vector_meta = json.loads((out / "query_vectors.meta.json").read_text(encoding="utf-8"))
    report = {
        "status": "prepared",
        "queries": len(query_ids),
        "raw_p1_docs_mean": float(np.mean(list(raw_k.values()))),
        "pool_docs_mean": float(np.mean(list(pool_k.values()))),
        "pool_docs_min": min(pool_k.values()),
        "pool_docs_max": max(pool_k.values()),
        "pool_pairs": len(candidate),
        "keep_docs_mean_from_k90_zip": float(np.mean(list(keep_k.values()))),
        "keep_docs_total_from_k90_zip": int(sum(keep_k.values())),
        "baseline_k90_zip_sha256": baseline_build.get("zip_sha256"),
        "p1_zip_sha256": _sha256(inputs["p1"]),
        "doc_vectors_sha256": _sha256(inputs["vectors"]),
        "base_doc_vectors": base_doc_count,
        "extra_doc_vectors": len(doc_ids) - base_doc_count,
        "extra_vector_files": [str(path) for path in extra_files],
        "qwen_model": MODEL_ID,
        "qwen_revision": MODEL_REVISION,
        "query_encoder": encoder,
        "query_vectors": str(out / "query_vectors.npz"),
        "query_vectors_dtype": str(q_dense.dtype),
        "query_vectors_sha256": _sha256(out / "query_vectors.npz"),
        "query_vectors_source": str(paths.resolve_path(query_vectors_dir)),
        "query_vectors_source_hashes": query_vector_meta.get("source_hashes"),
        "candidate_pool": str(candidate_path),
        "candidate_pool_sha256": _sha256(candidate_path),
        "pool_multiplier": pool_multiplier,
        "pool_rule": f"round({pool_multiplier:g} * len(P1 relevant_docs)); Python round",
        "keep_rule": "len(relevant_docs) read from each K90-ph ZIP row",
        "seconds": round(time.monotonic() - prepare_started, 2),
    }
    _atomic_json(out / "prepare.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def seed_scores(out: Path, source_dir: Path) -> dict[str, Any]:
    """Seed the expanded-pool checkpoint from a completed smaller-pool run."""
    target_candidate_path = out / "candidate_pool.parquet"
    source_candidate_path = source_dir / "candidate_pool.parquet"
    source_report_path = source_dir / "full-score-report.json"
    source_db_path = source_dir / "qwen_scores.sqlite"
    target_db_path = out / "qwen_scores.sqlite"
    for path in (
        target_candidate_path,
        source_candidate_path,
        source_report_path,
        source_db_path,
        out / "prepare.json",
    ):
        paths.require_inputs(path)
    _guard_outputs(out)

    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("status") != "full_complete":
        raise ValueError("Source score run is not complete")
    if source_report.get("model_revision") != MODEL_REVISION:
        raise ValueError("Source score run uses a different Qwen model revision")
    source_candidate_sha = _sha256(source_candidate_path)
    if source_report.get("candidate_pool_sha256") != source_candidate_sha:
        raise ValueError("Source score report belongs to a different candidate pool SHA256")
    source_candidate = pd.read_parquet(source_candidate_path)
    target_candidate = pd.read_parquet(target_candidate_path)
    aligned = _validate_pool_extension_alignment(source_candidate, target_candidate)

    target_prepare = json.loads((out / "prepare.json").read_text(encoding="utf-8"))
    target_candidate_sha = _sha256(target_candidate_path)
    if target_prepare.get("candidate_pool_sha256") != target_candidate_sha:
        raise ValueError("Expanded-pool prepare report belongs to a different candidate pool")
    if float(target_prepare.get("pool_multiplier", 0)) <= 1.3:
        raise ValueError("Target candidate pool must be expanded beyond 1.3x")

    expected_source_meta = {
        "candidate_sha256": source_candidate_sha,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "instruction_vi": INSTRUCTIONS["vi"],
        "instruction_zh": INSTRUCTIONS["zh"],
        "doc_token_limit": str(DOC_TOKEN_LIMIT),
    }
    expected_target_meta = {
        **expected_source_meta,
        "candidate_sha256": target_candidate_sha,
    }
    with sqlite3.connect(f"file:{source_db_path.as_posix()}?mode=ro", uri=True) as source_conn:
        source_meta = dict(source_conn.execute("SELECT key, value FROM metadata").fetchall())
        if source_meta != expected_source_meta:
            raise ValueError("Source Qwen checkpoint metadata does not match the pinned inputs")
        source_rows = source_conn.execute(
            "SELECT query_id, doc_id, cosine_rank, cosine_score, qwen_score FROM scores "
            "ORDER BY query_id, cosine_rank"
        ).fetchall()
    if len(source_rows) != len(source_candidate) or len(source_rows) != int(
        source_report.get("pairs_checkpointed_total", -1)
    ):
        raise ValueError("Source score checkpoint is incomplete for its candidate pool")
    score_candidates = pd.DataFrame(
        source_rows,
        columns=["query_id", "doc_id", "cosine_rank", "cosine_score", "qwen_score"],
    )
    score_alignment = source_candidate.merge(
        score_candidates,
        on=["query_id", "doc_id"],
        how="left",
        suffixes=("_candidate", "_score"),
        validate="one_to_one",
        indicator=True,
    )
    if (
        not score_alignment["_merge"].eq("both").all()
        or not np.array_equal(
            score_alignment["cosine_rank_candidate"].to_numpy(),
            score_alignment["cosine_rank_score"].to_numpy(),
        )
        or not np.allclose(
            score_alignment["cosine_score_candidate"].to_numpy(dtype=np.float64),
            score_alignment["cosine_score_score"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-7,
        )
        or not np.isfinite(score_alignment["qwen_score"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("Source Qwen scores do not align exactly with its candidate pool")

    target_scores = target_candidate.merge(
        score_candidates[["query_id", "doc_id", "qwen_score"]],
        on=["query_id", "doc_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    target_scores = target_scores[target_scores["_merge"].eq("both")].drop(columns="_merge")
    if not np.isfinite(target_scores["qwen_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Source Qwen checkpoint contains non-finite scores for shared candidates")

    paths.assert_writable(target_db_path)
    paths.assert_writable(out / "seed-report.json.tmp")
    if not target_db_path.exists():
        with sqlite3.connect(target_db_path) as target_conn:
            target_conn.execute("PRAGMA journal_mode=WAL")
            target_conn.execute(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            target_conn.execute(
                "CREATE TABLE scores(query_id INTEGER, doc_id INTEGER, cosine_rank INTEGER, "
                "cosine_score REAL, qwen_score REAL, PRIMARY KEY(query_id, doc_id))"
            )
            target_conn.executemany(
                "INSERT INTO metadata(key, value) VALUES(?,?)", expected_target_meta.items()
            )
            target_conn.commit()
    with sqlite3.connect(target_db_path) as target_conn:
        target_meta = dict(target_conn.execute("SELECT key, value FROM metadata").fetchall())
        if target_meta != expected_target_meta:
            raise ValueError("Expanded-pool checkpoint metadata does not match the target pool")
        existing = {
            (int(qid), int(doc_id)): (int(rank), float(cosine), float(qwen))
            for qid, doc_id, rank, cosine, qwen in target_conn.execute(
                "SELECT query_id, doc_id, cosine_rank, cosine_score, qwen_score FROM scores"
            )
        }
        source_pairs = []
        for row in target_scores.itertuples(index=False):
            qid = int(row.query_id)
            doc_id = int(row.doc_id)
            rank = int(row.cosine_rank)
            cosine = float(row.cosine_score)
            qwen = float(row.qwen_score)
            key = (qid, doc_id)
            actual = (rank, cosine, qwen)
            if key in existing:
                prior = existing[key]
                if prior[0] != actual[0] or not np.isclose(
                    prior[1], actual[1], rtol=0.0, atol=1e-7
                ) or not np.isclose(prior[2], actual[2], rtol=0.0, atol=1e-7):
                    raise ValueError(f"Existing pool16 score differs for query={qid}, doc={doc_id}")
                continue
            source_pairs.append((int(qid), int(doc_id), int(rank), float(cosine), float(qwen)))
        target_conn.executemany(
            "INSERT INTO scores(query_id, doc_id, cosine_rank, cosine_score, qwen_score) "
            "VALUES(?,?,?,?,?)",
            source_pairs,
        )
        target_conn.commit()
        checkpointed = int(target_conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
    report = {
        "status": "seeded",
        "source_candidate_pool_sha256": source_candidate_sha,
        "target_candidate_pool_sha256": target_candidate_sha,
        "source_pairs": len(source_rows),
        "source_pairs_aligned_to_target": int(aligned["_merge"].eq("both").sum()),
        "source_pairs_missing_at_tied_cutoff": int(aligned["_merge"].eq("left_only").sum()),
        "shared_pairs_with_cosine_rank_change": int(
            (
                aligned.loc[aligned["_merge"].eq("both"), "cosine_rank_source"]
                != aligned.loc[aligned["_merge"].eq("both"), "cosine_rank_target"]
            ).sum()
        ),
        "source_pairs_inserted": len(source_pairs),
        "pilot_or_prior_pairs_already_present": int(
            aligned["_merge"].eq("both").sum()
        ) - len(source_pairs),
        "target_pool_pairs": len(target_candidate),
        "target_checkpoint_pairs_after_seed": checkpointed,
        "target_pairs_remaining": len(target_candidate) - checkpointed,
        "source_score_report": str(source_report_path),
        "target_checkpoint": str(target_db_path),
    }
    _atomic_json(out / "seed-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _gpu_snapshot() -> dict[str, int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first = proc.stdout.strip().splitlines()[0]
    used, total, utilization = [int(part.strip()) for part in first.split(",")]
    return {"used_mib": used, "total_mib": total, "free_mib": total - used, "utilization_pct": utilization}


def _acquire_lock(out: Path) -> Path:
    lock = out / "score.lock"
    payload = {"pid": os.getpid(), "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    paths.assert_writable(lock)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = {}
        try:
            existing = json.loads(lock.read_text(encoding="utf-8"))
        except Exception:
            pass
        pid = int(existing.get("pid", -1))
        alive = False
        if pid > 0:
            try:
                import psutil

                alive = psutil.pid_exists(pid)
            except Exception:
                try:
                    os.kill(pid, 0)
                    alive = True
                except OSError:
                    alive = False
        if alive:
            raise RuntimeError(f"Scoring lock belongs to live PID {pid}: {lock}")
        lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    return lock


def _pid_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _model_path() -> Path:
    pin = paths.OUT_DIR / "runs" / "reranker-trial" / "model-pin.json"
    if pin.is_file():
        info = json.loads(pin.read_text(encoding="utf-8"))
        if info.get("revision") != MODEL_REVISION:
            raise ValueError("Existing Qwen model pin has a different revision")
        model_path = Path(info["snapshot_path"])
    else:
        model_path = (
            paths.DATA_DIR
            / "hf_model_cache"
            / "models--Qwen--Qwen3-Reranker-0.6B"
            / "snapshots"
            / MODEL_REVISION
        )
    paths.require_inputs(model_path)
    return model_path


def _score_batch(tokenizer: Any, model: Any, pair_batch: list[dict[str, Any]], doc_texts: dict[int, str], queries: dict[int, str], batch_size: int) -> list[float]:
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    sequences = []
    for row in pair_batch:
        lang = row["lang"]
        prompt = (
            f"<Instruct>: {INSTRUCTIONS[lang]}\n<Query>: {queries[int(row['query_id'])]}\n"
            f"<Document>: {doc_texts[int(row['doc_id'])]}"
        )
        sequences.append(prefix + tokenizer.encode(prompt, add_special_tokens=False) + suffix)
    enc = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
    enc = {key: value.to(model.device) for key, value in enc.items()}
    import torch

    with torch.inference_mode():
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
        no_id = tokenizer.convert_tokens_to_ids("no")
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        values = torch.log_softmax(torch.stack((logits[:, no_id].float(), logits[:, yes_id].float()), dim=1), dim=1)[:, 1].exp()
    return values.detach().cpu().numpy().astype(np.float64).tolist()


def _write_score_checkpoint(conn: sqlite3.Connection, pair_batch: list[dict[str, Any]], scores: list[float]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO scores(query_id, doc_id, cosine_rank, cosine_score, qwen_score) VALUES(?,?,?,?,?)",
        [
            (
                int(row["query_id"]),
                int(row["doc_id"]),
                int(row["cosine_rank"]),
                float(row["cosine_score"]),
                float(score),
            )
            for row, score in zip(pair_batch, scores)
        ],
    )
    conn.commit()


def score(out: Path, pilot: bool, batch_size: int = DEFAULT_BATCH_SIZE) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    inputs = _inputs()
    candidate_path = out / "candidate_pool.parquet"
    for path in [candidate_path, inputs["query"], inputs["canonical"] / "docs.parquet"]:
        paths.require_inputs(path)
    _guard_outputs(out)
    candidates = pd.read_parquet(candidate_path)
    candidate_hash = _sha256(candidate_path)
    total_pairs = len(candidates)
    if not pilot:
        pilot_path = out / "pilot-report.json"
        paths.require_inputs(pilot_path)
        _validate_pilot_report(
            json.loads(pilot_path.read_text(encoding="utf-8")), total_pairs, candidate_hash
        )
    gpu_initial = _gpu_snapshot()
    lock = _acquire_lock(out)
    log_path = out / "score.log"
    conn = None
    log = log_path.open("a", encoding="utf-8", newline="\n")
    def emit(message: str) -> None:
        line = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "message": message,
        }
        serialized = json.dumps(line, ensure_ascii=False)
        log.write(serialized + "\n")
        log.flush()
        print(serialized, flush=True)

    try:
        emit(
            f"start mode={'pilot' if pilot else 'full'} stage=load_inputs "
            f"batch_size={batch_size} gpu_free_mib={gpu_initial['free_mib']} pid={os.getpid()}"
        )
        emit(f"loaded candidate pool pairs={total_pairs} sha256={candidate_hash}")
        qtable = pq.read_table(inputs["query"], columns=["id", "query"]).to_pandas()
        queries = dict(zip(qtable.id.astype(int).tolist(), qtable["query"].fillna("").tolist()))
        doc_ids = set(candidates.doc_id.astype(int).tolist())
        texts: dict[int, str] = {}
        langs: dict[int, str] = {}
        extra_docs = inputs["extra_docs"] / "lots"
        extra_id_set: set[int] = set()
        for lot in sorted(path for path in extra_docs.iterdir() if path.is_dir()):
            table = pd.read_parquet(lot / "chunks" / "docs.parquet", columns=["doc_id", "doc_text"])
            table = table[table.doc_id.astype(int).isin(doc_ids)]
            texts.update({int(row.doc_id): row.doc_text or "" for row in table.itertuples(index=False)})
            extra_ids = table.doc_id.astype(int).tolist()
            extra_id_set.update(extra_ids)
            langs.update({doc_id: "vi" for doc_id in extra_ids})
        canonical_ids = doc_ids - extra_id_set
        if canonical_ids:
            docs_path = inputs["canonical"] / "docs.parquet"
            table = pq.read_table(
                docs_path,
                columns=["doc_id", "lang", "doc_text"],
                filters=[("doc_id", "in", sorted(canonical_ids))],
            ).to_pandas()
            for row in table.itertuples(index=False):
                if row.lang not in INSTRUCTIONS:
                    raise ValueError(f"Unsupported passage language {row.lang!r} for doc {row.doc_id}")
                texts[int(row.doc_id)] = row.doc_text or ""
                langs[int(row.doc_id)] = row.lang
        if set(texts) != doc_ids:
            raise ValueError(f"Missing candidate document text for {len(doc_ids - set(texts))} IDs")
        candidates["lang"] = candidates.doc_id.astype(int).map(langs)
        if candidates.lang.isna().any() or not candidates.lang.isin(INSTRUCTIONS).all():
            raise ValueError("Candidate pool includes a missing or unsupported language")
        emit(f"loaded candidate doc_text docs={len(texts)} query_rows={len(queries)}")

        db_path = out / "qwen_scores.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scores(query_id INTEGER, doc_id INTEGER, cosine_rank INTEGER, "
            "cosine_score REAL, qwen_score REAL, PRIMARY KEY(query_id, doc_id))"
        )
        expected_meta = {
            "candidate_sha256": candidate_hash,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "instruction_vi": INSTRUCTIONS["vi"],
            "instruction_zh": INSTRUCTIONS["zh"],
            "doc_token_limit": str(DOC_TOKEN_LIMIT),
        }
        existing_meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        if existing_meta and existing_meta != expected_meta:
            raise ValueError("Qwen checkpoint belongs to different inputs or model settings")
        if not existing_meta:
            conn.executemany("INSERT INTO metadata(key, value) VALUES(?,?)", expected_meta.items())
            conn.commit()
        done = {
            (int(qid), int(did))
            for qid, did in conn.execute("SELECT query_id, doc_id FROM scores").fetchall()
        }
        pairs = candidates.to_dict("records")
        if pilot:
            pairs = _select_pilot_pairs(pairs, done)
        else:
            pairs = [row for row in pairs if (int(row["query_id"]), int(row["doc_id"])) not in done]
        if not pairs:
            if not pilot and len(done) == total_pairs:
                report_path = out / "full-score-report.json"
                if report_path.is_file():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if (
                        report.get("status") == "full_complete"
                        and int(report.get("pairs_checkpointed_total", -1)) == total_pairs
                        and report.get("model_revision") == MODEL_REVISION
                        and report.get("candidate_pool_sha256") == candidate_hash
                    ):
                        emit("full score already complete; returning existing report")
                        return report
                report = {
                    "status": "full_complete",
                    "mode": "full",
                    "pid": os.getpid(),
                    "pairs_scored_this_run": 0,
                    "pairs_total": total_pairs,
                    "pairs_checkpointed_total": len(done),
                    "seconds_inference": 0.0,
                    "pairs_per_second": None,
                    "eta_total_seconds_at_measured_rate": None,
                    "eta_remaining_seconds_at_measured_rate": 0.0,
                    "initial_batch_size": batch_size,
                    "final_batch_size": batch_size,
                    "oom_retried": False,
                    "recovered_report_from_complete_checkpoint": True,
                    "model": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "candidate_pool_sha256": candidate_hash,
                    "candidate_pool": str(candidate_path),
                    "instruction_vi": INSTRUCTIONS["vi"],
                    "instruction_zh": INSTRUCTIONS["zh"],
                    "document_token_limit": DOC_TOKEN_LIMIT,
                    "checkpoint": str(db_path),
                    "log": str(log_path),
                    "lock": str(lock),
                }
                _atomic_json(report_path, report)
                emit("recovered full-complete report from the exact score checkpoint")
                return report
            raise RuntimeError("No pairs to score")
        emit(
            f"checkpoint loaded={len(done)} pairs_to_score={len(pairs)} "
            f"mode={'pilot' if pilot else 'full'}"
        )
        gpu_gate = _gpu_snapshot()
        if gpu_gate["free_mib"] < GPU_FREE_MIN_MIB:
            raise RuntimeError(
                f"GPU gate closed before model load: free {gpu_gate['free_mib']} MiB < "
                f"{GPU_FREE_MIN_MIB} MiB; used={gpu_gate['used_mib']} MiB / "
                f"{gpu_gate['total_mib']} MiB, util={gpu_gate['utilization_pct']}%"
            )
        model_path = _model_path()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required for S29 scoring")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        requested_batch_size = batch_size
        model_load_oom_retried = False
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                dtype=torch.bfloat16,
            ).cuda().eval()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            load_error = str(exc)
            model_load_oom_retried = True
            batch_size = max(1, batch_size // 2)
            del exc
            gc.collect()
            torch.cuda.empty_cache()
            emit(
                f"OOM during model load: cleared failed-load allocations and retry once; "
                f"scoring batch_size={batch_size}; first error={load_error}"
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    local_files_only=True,
                    dtype=torch.bfloat16,
                ).cuda().eval()
            except RuntimeError as retry_exc:
                if "out of memory" in str(retry_exc).lower():
                    emit(f"OOM after one reduced-batch retry during model load: {retry_exc}")
                raise
        if tokenizer.convert_tokens_to_ids("no") is None or tokenizer.convert_tokens_to_ids("yes") is None:
            raise ValueError("Qwen tokenizer lacks the expected yes/no token IDs")
        passage_cache: dict[int, tuple[str, int]] = {}
        gpu_loaded = _gpu_snapshot()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        completed = 0
        cursor = 0
        batch_oom_retried = False

        emit(
            f"start mode={'pilot' if pilot else 'full'} pairs={len(pairs)} total={total_pairs} "
            f"free_vram_preload_mib={gpu_gate['free_mib']} "
            f"used_vram_after_load_mib={gpu_loaded['used_mib']} "
            f"stage=score_with_lazy_doc_truncation pid={os.getpid()}"
        )
        while cursor < len(pairs):
            size = min(batch_size, len(pairs) - cursor)
            batch = pairs[cursor : cursor + size]
            texts_512: dict[int, str] = {}
            for row in batch:
                doc_id = int(row["doc_id"])
                if doc_id not in passage_cache:
                    if doc_id not in texts:
                        raise ValueError(f"Missing untruncated text for candidate doc {doc_id}")
                    passage_cache[doc_id] = truncate_document(tokenizer, texts.pop(doc_id))
                texts_512[doc_id] = passage_cache[doc_id][0]
                row["lang"] = str(row["lang"])
            try:
                scores = _score_batch(
                    tokenizer,
                    model,
                    batch,
                    texts_512,
                    queries,
                    batch_size,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                    raise
                if batch_oom_retried or batch_size <= 1:
                    emit(f"oom after one reduced-batch retry: {exc}")
                    raise
                batch_oom_retried = True
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                emit(f"OOM: retry once with batch_size={batch_size}")
                continue
            _write_score_checkpoint(conn, batch, scores)
            cursor += len(batch)
            completed += len(batch)
            if completed % CHECKPOINT_PAIRS < len(batch) or cursor >= len(pairs):
                elapsed = time.monotonic() - start
                rate = completed / elapsed if elapsed else 0.0
                checkpointed_total = int(conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
                remaining = max(total_pairs - checkpointed_total, 0)
                eta_s = remaining / rate if rate else math.inf
                emit(
                    f"progress={checkpointed_total}/{total_pairs} run_pairs={completed} "
                    f"rate={rate:.3f}/s eta_remaining_s={eta_s:.0f} batch_size={batch_size}"
                )
        elapsed = time.monotonic() - start
        rate = completed / elapsed if elapsed else 0.0
        gpu_after = _gpu_snapshot()
        report = {
            "status": "pilot_complete" if pilot else "full_complete",
            "mode": "pilot" if pilot else "full",
            "pid": os.getpid(),
            "pairs_scored_this_run": completed,
            "pairs_total": total_pairs,
            "pairs_checkpointed_total": conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0],
            "seconds_inference": round(elapsed, 3),
            "seconds_scoring_with_doc_truncation": round(elapsed, 3),
            "pairs_per_second": rate,
            "eta_total_seconds_at_measured_rate": total_pairs / rate if rate else None,
            "eta_remaining_seconds_at_measured_rate": max(total_pairs - conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0], 0) / rate if rate else None,
            "initial_batch_size": requested_batch_size,
            "final_batch_size": batch_size,
            "oom_retried": model_load_oom_retried or batch_oom_retried,
            "model_load_oom_retried": model_load_oom_retried,
            "batch_oom_retried": batch_oom_retried,
            "gpu_preload_gate": gpu_gate,
            "gpu_after_model_load": gpu_loaded,
            "gpu_after": gpu_after,
            "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024**2), 1),
            "peak_cuda_reserved_mib": round(torch.cuda.max_memory_reserved() / (1024**2), 1),
            "peak_rss_bytes": _pid_rss_bytes(),
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "candidate_pool_sha256": candidate_hash,
            "candidate_pool": str(candidate_path),
            "instruction_vi": INSTRUCTIONS["vi"],
            "instruction_zh": INSTRUCTIONS["zh"],
            "document_token_limit": DOC_TOKEN_LIMIT,
            "checkpoint": str(db_path),
            "log": str(log_path),
            "lock": str(lock),
        }
        target = out / ("pilot-report.json" if pilot else "full-score-report.json")
        _atomic_json(target, report)
        emit(json.dumps(report, ensure_ascii=False))
        return report
    finally:
        if conn is not None:
            conn.close()
        log.close()
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def export_scores(out: Path) -> Path:
    db = out / "qwen_scores.sqlite"
    paths.require_inputs(db)
    target = out / "scores.parquet"
    temp = target.with_name(target.name + ".tmp")
    paths.assert_writable(temp)
    with sqlite3.connect(db) as conn:
        frame = pd.read_sql_query(
            "SELECT query_id, doc_id, cosine_rank, cosine_score, qwen_score FROM scores ORDER BY query_id, cosine_rank",
            conn,
        )
    frame.to_parquet(temp, index=False)
    temp.replace(target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="S29 run directory below R2AI_OUT_DIR/runs")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="make the cosine top-(1.3*Kq) candidate pool")
    prepare_parser.add_argument(
        "--query-vectors",
        type=Path,
        required=True,
        help="directory containing exact q_dense.npy/q_ids.npy (for S26 parity)",
    )
    prepare_parser.add_argument("--pool-multiplier", type=float, default=1.3)
    score_parser = sub.add_parser("score", help="score candidate query/document pairs with Qwen")
    mode = score_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="score the deterministic 200-pair pilot")
    mode.add_argument("--full", action="store_true", help="score all candidates, resuming the checkpoint")
    score_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    seed_parser = sub.add_parser(
        "seed-scores", help="seed an expanded pool with verified scores from a smaller pool"
    )
    seed_parser.add_argument("--source-dir", type=Path, required=True)
    sub.add_parser("export-scores", help="export the SQLite checkpoint to Parquet")
    build_parser = sub.add_parser("build", help="build configured S29 bundles on CPU")
    build_parser.add_argument("--variant", action="append", default=None)
    build_parser.add_argument("--chunk-multiplier", type=float, default=None)
    build_parser.add_argument("--whole-body", action="store_true", default=None)
    args = parser.parse_args(argv)
    out = _require_under_runs(args.out or run_dir())
    _guard_outputs(out)
    if args.command == "prepare":
        prepare(
            out,
            query_vectors_dir=args.query_vectors,
            pool_multiplier=args.pool_multiplier,
        )
    elif args.command == "score":
        if args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
        score(out, pilot=args.pilot, batch_size=args.batch_size)
    elif args.command == "export-scores":
        print(export_scores(out))
    elif args.command == "seed-scores":
        seed_scores(out, source_dir=_require_under_runs(args.source_dir))
    elif args.command == "build":
        from r2ai.s29.build import build

        variants = tuple(args.variant) if args.variant else ("Q-filter", "Q-rrf")
        print(
            json.dumps(
                build(
                    out,
                    variants=variants,
                    chunk_multiplier=args.chunk_multiplier,
                    whole_body=args.whole_body,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
