"""Score pool shards with BAAI/bge-reranker-v2-m3 (cross-encoder); resumable per part, portable to Kaggle.

Reads the same ``pool-*.parquet`` shards as ``r2ai.s31.judge_score`` (``doc_text`` = document text
for the S31/S33 doc pools, chunk text for the S32/S33 chunk pools) and writes one parquet per part
(atomic rename), so a killed session resumes where it stopped. Part planning, locks, seeding from an
earlier session, ``status`` and ``collect`` are the S31 functions, so part names and checks match.

Scoring follows the model card / ``r2ai.index.bge_m3.Reranker`` defaults: fp16,
``tokenizer(query, passage, truncation='only_second', max_length=512)``, raw logit of the single
output (``score``) plus ``sigmoid(logit)`` (``prob``, FlagReranker ``normalize=True``).
No instruction or prompt template: bge-reranker-v2-m3 takes the bare (query, passage) pair.

Copy this file next to ``judge_score.py`` to run it outside the repo::

    python bge_score.py pilot   --pool POOL --out OUT --pairs 1000 --any-pairs
    python bge_score.py score   --pool POOL --out OUT --worker 0 --workers 2
    python bge_score.py collect --pool POOL --out OUT
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from r2ai.s31 import judge_score as js
except Exception:  # pragma: no cover - outside the repo judge_score.py sits next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import judge_score as js  # type: ignore[no-redef]

MODEL = "bge-reranker-v2-m3"
HF_ID = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # docs/COMPLIANCE.md, local HF cache refs/main
MAX_LENGTH = 512
SCHEMA_VERSION = 1


class BGEScorer:
    def __init__(self, model_path: str, revision: str | None, dtype: str, log: js.Logger, attn: str = "sdpa"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if dtype == "auto":
            dtype = "float16"  # the model card and the earlier repo pipeline run fp16 on every GPU
        self.dtype_name = dtype
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, dtype=torch_dtype, attn_implementation=attn, **kwargs)
        except TypeError:  # transformers < 4.56 uses torch_dtype
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, torch_dtype=torch_dtype, attn_implementation=attn, **kwargs)
        self.model = model.cuda().eval()
        self.resolved_revision = getattr(model.config, "_commit_hash", None)
        snapshot = Path(model_path).name
        if self.resolved_revision is None and len(snapshot) == 40 and all(c in "0123456789abcdef" for c in snapshot):
            self.resolved_revision = snapshot  # local HF snapshot dir: the dir name is the commit
        log("bge model loaded", model=model_path, dtype=dtype, attn=attn, revision=self.resolved_revision, gpu=js.gpu_info())

    def encode(self, queries: Sequence[str], passages: Sequence[str]) -> list[list[int]]:
        enc = self.tokenizer(list(queries), list(passages), truncation="only_second", max_length=MAX_LENGTH)
        return enc["input_ids"]

    def logits(self, sequences: list[list[int]]) -> np.ndarray:
        torch = self.torch
        enc = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        enc = {key: value.to(self.model.device) for key, value in enc.items()}
        with torch.inference_mode():
            out = self.model(**enc, return_dict=True).logits.view(-1).float().cpu().numpy()
        return out

    def is_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, self.torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def empty_cache(self) -> None:
        import gc

        gc.collect()
        self.torch.cuda.empty_cache()


def score_frame(scorer: BGEScorer, frame: pd.DataFrame, max_batch_tokens: int, max_batch_size: int,
                log: js.Logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one part; returns per-pair logits/probabilities and timing/token stats."""
    queries = [q or "" for q in frame.query_text]
    passages = [d or "" for d in frame.doc_text]
    sequences: list[list[int]] = []
    for begin in range(0, len(frame), 4_096):
        sequences.extend(scorer.encode(queries[begin : begin + 4_096], passages[begin : begin + 4_096]))
    lengths = [len(s) for s in sequences]
    logit = np.full(len(frame), np.nan, dtype=np.float32)
    budget = max_batch_tokens
    started = time.monotonic()
    pending = js.token_budget_batches(lengths, budget, max_batch_size)
    oom_retries = 0
    while pending:
        batch = pending.pop(0)
        try:
            values = scorer.logits([sequences[i] for i in batch])
        except Exception as exc:  # noqa: BLE001
            if not scorer.is_oom(exc) or len(batch) == 1:
                raise
            scorer.empty_cache()
            oom_retries += 1
            budget = max(budget // 2, max(lengths[i] for i in batch))
            log("oom; halving token budget and re-planning the rest of the part", budget=budget, batch=len(batch))
            rest = batch + [i for b in pending for i in b]
            pending = [[rest[j] for j in sub] for sub in js.token_budget_batches([lengths[i] for i in rest], budget, max_batch_size)]
            continue
        logit[batch] = values
    seconds = time.monotonic() - started
    if not np.isfinite(logit).all():
        raise FloatingPointError(f"{int((~np.isfinite(logit)).sum())} non-finite logits (fp16 overflow?); rerun with --dtype float32")
    result = pd.DataFrame({
        "pair_id": frame.pair_id.astype(np.int64).to_numpy(),
        "query_id": frame.query_id.astype(np.int64).to_numpy(),
        "doc_id": frame.doc_id.astype(np.int64).to_numpy(),
        "score": logit,
        "prob": (1.0 / (1.0 + np.exp(-logit.astype(np.float64)))).astype(np.float64),
        "input_tokens": np.asarray(lengths, dtype=np.int32),
    })
    stats = {
        "pairs": len(frame),
        "seconds": round(seconds, 3),
        "input_tokens": int(sum(lengths)),
        "max_input_tokens": int(max(lengths)) if lengths else 0,
        "inputs_at_max_length": int(sum(1 for n in lengths if n >= MAX_LENGTH)),
        "oom_retries": oom_retries,
        "final_batch_tokens": budget,
    }
    return result, stats


def make_scorer(args: argparse.Namespace, log: js.Logger) -> BGEScorer:
    model_path = args.model_path or HF_ID
    revision = None if args.model_path else (args.revision or REVISION)
    return BGEScorer(model_path, revision, args.dtype, log, attn=args.attn)


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    out = js._model_dir(Path(args.out), args.model)
    parts_dir = out / "parts"
    js._guard(out)
    parts_dir.mkdir(parents=True, exist_ok=True)
    log = js.Logger(out / f"score-w{args.worker}.log")
    js._seed_from([Path(p) for p in args.resume_from], parts_dir, args.model, log)
    parts = js.worker_parts(js.plan_parts(js.all_shards(args), args.part_pairs, args.subset, js.subset_pair_ids(args)),
                            args.worker, args.workers)
    todo = [part for part in parts if not (parts_dir / js.part_name(part)).exists()]
    log("plan", subset=args.subset, worker=args.worker, workers=args.workers, parts=len(parts), todo=len(todo),
        pairs_worker=js.part_pairs_count(parts), pairs_todo=js.part_pairs_count(todo))
    if not todo:
        return {"status": "nothing_to_do", "worker": args.worker}
    lock = out / f"score-w{args.worker}.lock"
    js._acquire_lock(lock)
    deadline = time.monotonic() + args.max_seconds if args.max_seconds else math.inf
    try:
        load_started = time.monotonic()
        scorer = make_scorer(args, log)
        load_seconds = time.monotonic() - load_started
        done_pairs, done_seconds = 0, 0.0
        cache: dict[str, pd.DataFrame] = {}
        for part in todo:
            if time.monotonic() > deadline:
                log("max-seconds reached; stopping before next part", remaining_parts=len(todo) - todo.index(part))
                break
            shard = part["shard"]
            if shard not in cache:
                cache.clear()
                cache[shard] = pd.read_parquet(part["path"])
            frame = cache[shard].iloc[part["positions"]]
            result, stats = score_frame(scorer, frame, args.max_batch_tokens, args.max_batch_size, log)
            target = parts_dir / js.part_name(part)
            meta = {**stats, "model": args.model, "backend": "hf", "dtype": scorer.dtype_name,
                    "revision": scorer.resolved_revision, "gpu": js.gpu_info().get("name"), "shard": shard,
                    "subset": args.subset, "part": part["part"], "rows": len(part["positions"]), "utc": js._now(),
                    "schema": SCHEMA_VERSION}
            js._atomic_parquet(target, result)
            js._atomic_json(target.with_suffix(".json"), meta)
            done_pairs += stats["pairs"]
            done_seconds += stats["seconds"]
            rate = done_pairs / done_seconds if done_seconds else 0.0
            left = js.part_pairs_count(todo) - done_pairs
            log("part done", part=target.name, pairs=stats["pairs"], seconds=stats["seconds"], rate=round(rate, 3),
                eta_worker_s=round(left / rate) if rate else None, oom_retries=stats["oom_retries"],
                at_max_length=stats["inputs_at_max_length"])
        report = {"status": "stopped" if done_pairs < js.part_pairs_count(todo) else "worker_complete",
                  "subset": args.subset, "worker": args.worker, "pairs_scored": done_pairs,
                  "seconds_scoring": round(done_seconds, 1), "seconds_model_load": round(load_seconds, 1),
                  "pairs_per_second": done_pairs / done_seconds if done_seconds else None, "gpu": js.gpu_info(),
                  "model": args.model, "dtype": scorer.dtype_name}
        js._atomic_json(out / f"worker-{args.worker}-{args.subset}-report.json", report)
        return report
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    out = js._model_dir(Path(args.out), args.model)
    js._guard(out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"pilot-{args.pairs}-{args.dtype}-{args.tag}" if args.tag else f"pilot-{args.pairs}-{args.dtype}"
    log = js.Logger(out / f"{tag}.log")
    shards = js.pool_shards(Path(args.pool))
    total_pairs = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    frame = js.pilot_rows(shards, args.pairs, only_pool13=not args.any_pairs)
    load_started = time.monotonic()
    scorer = make_scorer(args, log)
    load_seconds = time.monotonic() - load_started
    score_frame(scorer, frame.iloc[: min(16, len(frame))], args.max_batch_tokens, args.max_batch_size, log)  # warm-up
    result, stats = score_frame(scorer, frame, args.max_batch_tokens, args.max_batch_size, log)
    rate = stats["pairs"] / stats["seconds"] if stats["seconds"] else None
    js._atomic_parquet(out / f"{tag}.parquet", result)
    report = {
        "status": "pilot_complete", "model": args.model, "dtype": scorer.dtype_name, "revision": scorer.resolved_revision,
        "pool": str(args.pool), "pairs": stats["pairs"], "seconds": stats["seconds"], "pairs_per_second": rate,
        "pool_pairs_total": total_pairs, "eta_full_pool_seconds_one_gpu": total_pairs / rate if rate else None,
        "model_load_seconds": round(load_seconds, 1), "input_tokens_mean": stats["input_tokens"] / max(stats["pairs"], 1),
        "max_input_tokens": stats["max_input_tokens"], "inputs_at_max_length": stats["inputs_at_max_length"],
        "oom_retries": stats["oom_retries"], "max_batch_tokens": args.max_batch_tokens,
        "max_batch_size": args.max_batch_size, "gpu": js.gpu_info(), "pilot_seed": js.PILOT_SEED,
        "pilot_only_pool13": not args.any_pairs, "scores": str(out / f"{tag}.parquet"), "utc": js._now(),
    }
    try:
        import torch

        report["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    js._atomic_json(out / f"{tag}.json", report)
    log("pilot done", **{k: report[k] for k in ("pairs", "seconds", "pairs_per_second", "eta_full_pool_seconds_one_gpu")})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, model_args: bool = True) -> None:
        p.add_argument("--pool", required=True, help="dir with shards/pool-*.parquet")
        p.add_argument("--out", required=True, help="output root; results go to OUT/<model>/")
        p.add_argument("--model", default=MODEL, choices=[MODEL])
        p.add_argument("--part-pairs", type=int, default=js.DEFAULT_PART_PAIRS)
        p.add_argument("--resume-from", action="append", default=[], help="read-only earlier output root(s)")
        p.add_argument("--subset", choices=js.SUBSETS, default="all")
        p.add_argument("--pairs-file", default=None)
        p.add_argument("--extra-pool", action="append", default=[])
        if model_args:
            p.add_argument("--model-path", default=None, help="local snapshot dir instead of the HF id")
            p.add_argument("--revision", default=None)
            p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
            p.add_argument("--attn", default="sdpa")
            p.add_argument("--max-batch-tokens", type=int, default=16_384, help="padded tokens per batch")
            p.add_argument("--max-batch-size", type=int, default=64)

    pilot = sub.add_parser("pilot", help="score a deterministic sample and report speed/ETA")
    common(pilot)
    pilot.add_argument("--pairs", type=int, default=1000)
    pilot.add_argument("--any-pairs", action="store_true", help="sample every shard row, not only in_pool13 rows")
    pilot.add_argument("--tag", default="", help="suffix of the pilot file names")
    score = sub.add_parser("score", help="score this worker's parts, resuming finished ones")
    common(score)
    score.add_argument("--worker", type=int, default=0)
    score.add_argument("--workers", type=int, default=1)
    score.add_argument("--max-seconds", type=float, default=0.0)
    collect = sub.add_parser("collect", help="verify parts and write OUT/<model>/scores-<subset>.parquet")
    common(collect, model_args=False)
    status = sub.add_parser("status", help="print finished/total parts")
    common(status, model_args=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pilot":
        result = run_pilot(args)
    elif args.command == "score":
        result = run_score(args)
    elif args.command == "collect":
        result = js.run_collect(args)
    else:
        result = js.run_status(args)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
