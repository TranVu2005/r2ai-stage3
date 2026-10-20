"""Score S31 pool shards with a Qwen3 yes/no judge; resumable per part, portable to Kaggle/RunPod.

The script depends only on numpy/pandas/pyarrow/torch/transformers (vLLM optional),
so the single file can be copied to a GPU host. It reads ``pool-*.parquet`` shards
made by ``r2ai.s31.pool`` and writes one parquet per part of ``--part-pairs`` pairs
(atomic rename), so a killed session resumes where it stopped. Prompt, instruction
strings and the yes/no read-out are identical to S29 (Qwen3-Reranker-0.6B), so the
0.6B model reproduces S29 scores and 4B/8B differ only by weights.

Examples::

    python judge_score.py pilot  --pool POOL --out OUT --model qwen3-reranker-4b --pairs 1000
    python judge_score.py score  --pool POOL --out OUT --model qwen3-reranker-4b --subset pool13 --worker 0 --workers 2
    python judge_score.py collect --pool POOL --out OUT --model qwen3-reranker-4b --subset pool13
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:  # Local repo: route every write through the OLD guard. Absent on Kaggle.
    from r2ai import paths as _r2ai_paths
except Exception:  # pragma: no cover - exercised only outside the repo
    _r2ai_paths = None

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
DOC_TOKEN_LIMIT = 512
# kind="reranker": Qwen3-Reranker checkpoints. kind="generative": a chat model asked
# the same yes/no question with thinking disabled (same template the rerankers use).
MODELS: dict[str, dict[str, Any]] = {
    "qwen3-reranker-0.6b": {"hf_id": "Qwen/Qwen3-Reranker-0.6B", "kind": "reranker", "params_b": 0.6,
                            "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473"},
    "qwen3-reranker-4b": {"hf_id": "Qwen/Qwen3-Reranker-4B", "kind": "reranker", "params_b": 4.0},
    "qwen3-reranker-8b": {"hf_id": "Qwen/Qwen3-Reranker-8B", "kind": "reranker", "params_b": 8.0},
    "qwen3-14b": {"hf_id": "Qwen/Qwen3-14B", "kind": "generative", "params_b": 14.8},
    "qwen3-14b-awq": {"hf_id": "Qwen/Qwen3-14B-AWQ", "kind": "generative", "params_b": 14.8,
                      "quantization": "awq"},
}
DEFAULT_PART_PAIRS = 2_000
PILOT_SEED = 42
SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------- utils

def _guard(path: Path) -> Path:
    if _r2ai_paths is not None:
        _r2ai_paths.assert_writable(path)
    return path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: Any) -> None:
    temp = _guard(path.with_name(path.name + f".{os.getpid()}.tmp"))
    _guard(path)
    with temp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    temp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    temp = _guard(path.with_name(path.name + f".{os.getpid()}.tmp"))
    _guard(path)
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Logger:
    def __init__(self, path: Path):
        self.path = _guard(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str, **fields: Any) -> None:
        line = json.dumps({"utc": _now(), "pid": os.getpid(), "message": message, **fields}, ensure_ascii=False)
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")


def gpu_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        return {
            "cuda": True,
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_mib": round(total / 2**20),
            "free_mib": round(free / 2**20),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "bf16_native": props.major >= 8,
        }
    except Exception as exc:  # pragma: no cover
        return {"cuda": False, "error": str(exc)}


# ------------------------------------------------------------------------ pool parts

def pool_shards(pool: Path) -> list[Path]:
    shards = sorted((pool / "shards").glob("pool-*.parquet")) or sorted(pool.glob("pool-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No pool-*.parquet shards under {pool}")
    return shards


SUBSETS = ("all", "pool13", "tail")


def subset_positions(shard: Path, subset: str) -> np.ndarray:
    """Row positions of a shard in ``subset``: all rows, in_pool13 rows (1.3x), or the 1.3x-2.0x tail."""
    if subset == "all":
        return np.arange(pq.ParquetFile(shard).metadata.num_rows)
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}")
    mask = pq.read_table(shard, columns=["in_pool13"]).column("in_pool13").to_numpy(zero_copy_only=False)
    return np.flatnonzero(mask if subset == "pool13" else ~mask)


def plan_parts(shards: Sequence[Path], part_pairs: int, subset: str = "all") -> list[dict[str, Any]]:
    """Deterministic parts: consecutive ``part_pairs`` subset rows of each shard, in file order."""
    if part_pairs <= 0:
        raise ValueError("part_pairs must be positive")
    parts = []
    for shard in shards:
        positions = subset_positions(shard, subset)
        for index, begin in enumerate(range(0, len(positions), part_pairs)):
            parts.append({"shard": shard.name, "path": shard, "subset": subset, "part": index,
                          "positions": positions[begin : begin + part_pairs]})
    return parts


def part_name(part: dict[str, Any]) -> str:
    """``all`` keeps the original name; subsets get their own part files."""
    infix = "" if part.get("subset", "all") == "all" else f".{part['subset']}"
    return f"{Path(part['shard']).stem}{infix}.part-{int(part['part']):04d}.parquet"


def part_pairs_count(parts: Iterable[dict[str, Any]]) -> int:
    return int(sum(len(part["positions"]) for part in parts))


def worker_parts(parts: Sequence[dict[str, Any]], worker: int, workers: int) -> list[dict[str, Any]]:
    if not 0 <= worker < workers:
        raise ValueError(f"worker {worker} outside 0..{workers - 1}")
    return [part for index, part in enumerate(parts) if index % workers == worker]


def pilot_rows(shards: Sequence[Path], pairs: int, only_pool13: bool, seed: int = PILOT_SEED) -> pd.DataFrame:
    """Deterministic sample: ``random.Random(seed)`` over sorted pair_ids (1.3x part by default)."""
    keys = pd.concat([pd.read_parquet(s, columns=["pair_id", "in_pool13"]) for s in shards], ignore_index=True)
    if only_pool13:
        keys = keys[keys.in_pool13]
    ids = sorted(keys.pair_id.astype(int).tolist())
    picks = set(random.Random(seed).sample(ids, min(pairs, len(ids))))
    frames = [pq.read_table(s, filters=[("pair_id", "in", sorted(picks))]).to_pandas() for s in shards]
    return pd.concat(frames, ignore_index=True).sort_values("pair_id").reset_index(drop=True)


# --------------------------------------------------------------------------- scoring

def build_prompt_ids(tokenizer: Any, query: str, doc: str, lang: str, prefix_ids: list[int], suffix_ids: list[int]) -> list[int]:
    """Exactly S29 ``_score_batch``: prefix + encode(instruct/query/document) + suffix."""
    body = f"<Instruct>: {INSTRUCTIONS[lang]}\n<Query>: {query}\n<Document>: {doc}"
    return prefix_ids + tokenizer.encode(body, add_special_tokens=False) + suffix_ids


def token_budget_batches(lengths: Sequence[int], max_batch_tokens: int, max_batch_size: int) -> list[list[int]]:
    """Group indices (longest first) so that batch_size * longest_len <= max_batch_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for i in order:
        new_longest = max(longest, lengths[i])
        if current and (len(current) + 1 > max_batch_size or new_longest * (len(current) + 1) > max_batch_tokens):
            batches.append(current)
            current, new_longest = [], lengths[i]
        current.append(i)
        longest = new_longest
    if current:
        batches.append(current)
    return batches


def yes_probability(no_logit: np.ndarray, yes_logit: np.ndarray) -> np.ndarray:
    """softmax([no, yes])[1] computed stably; same value S29 stores as qwen_score."""
    diff = np.asarray(yes_logit, dtype=np.float64) - np.asarray(no_logit, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-diff))


class HFScorer:
    def __init__(self, model_path: str, revision: str | None, dtype: str, log: Logger, attn: str = "sdpa",
                 device_map: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if dtype == "auto":  # native bf16 needs sm80+; is_bf16_supported() also reports emulation (T4)
            dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
        self.dtype_name = dtype
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", **kwargs)
        load = dict(attn_implementation=attn, **kwargs)
        if device_map:
            load["device_map"] = device_map  # e.g. "auto": split an 8B model over 2x T4
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch_dtype, **load)
        except TypeError:  # transformers < 4.56 uses torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype, **load)
        self.model = (model if device_map else model.cuda()).eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        if self.yes_id is None or self.no_id is None:
            raise ValueError("Tokenizer lacks yes/no tokens")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.resolved_revision = getattr(model.config, "_commit_hash", None)
        log("hf model loaded", model=model_path, dtype=dtype, attn=attn, revision=self.resolved_revision, gpu=gpu_info())

    def logits(self, sequences: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        enc = self.tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        enc = {key: value.to(self.model.device) for key, value in enc.items()}
        with torch.inference_mode():
            try:
                out = self.model(**enc, use_cache=False, logits_to_keep=1)
            except TypeError:  # older transformers
                out = self.model(**enc, use_cache=False, num_logits_to_keep=1)
            last = out.logits[:, -1, :]
            no = last[:, self.no_id].float().cpu().numpy()
            yes = last[:, self.yes_id].float().cpu().numpy()
        return no, yes

    def is_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, self.torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def empty_cache(self) -> None:
        gc.collect()
        self.torch.cuda.empty_cache()


class VLLMScorer:
    """vLLM generate with max_tokens=1 and top-20 logprobs; reads yes/no logprobs."""

    def __init__(self, model_path: str, revision: str | None, dtype: str, log: Logger, max_model_len: int = 2048,
                 gpu_memory_utilization: float = 0.90, quantization: str | None = None):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
        self.dtype_name = dtype if dtype != "auto" else "auto"
        self.llm = LLM(model=model_path, revision=revision, dtype=dtype if dtype != "auto" else "auto",
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       enable_prefix_caching=True, quantization=quantization)
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20,
                                     allowed_token_ids=[self.yes_id, self.no_id])
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.resolved_revision = revision
        self.missing = 0
        log("vllm model loaded", model=model_path, dtype=dtype, gpu=gpu_info())

    def logits(self, sequences: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        from vllm.inputs import TokensPrompt

        outputs = self.llm.generate([TokensPrompt(prompt_token_ids=s) for s in sequences], self.params, use_tqdm=False)
        no, yes = [], []
        for output in outputs:
            top = output.outputs[0].logprobs[-1]
            floor = min(lp.logprob for lp in top.values()) - 10.0
            if self.yes_id not in top or self.no_id not in top:
                self.missing += 1
            yes.append(top[self.yes_id].logprob if self.yes_id in top else floor)
            no.append(top[self.no_id].logprob if self.no_id in top else floor)
        return np.asarray(no, dtype=np.float32), np.asarray(yes, dtype=np.float32)

    def is_oom(self, exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower()

    def empty_cache(self) -> None:
        gc.collect()


def score_frame(scorer: Any, frame: pd.DataFrame, max_batch_tokens: int, max_batch_size: int, log: Logger) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score one part; returns per-pair scores and timing/token stats."""
    tok = scorer.tokenizer
    sequences = []
    over_limit = 0
    for query, doc, lang in zip(frame.query_text, frame.doc_text, frame.lang):
        if lang not in INSTRUCTIONS:
            raise ValueError(f"Unsupported lang {lang!r}")
        sequences.append(build_prompt_ids(tok, query or "", doc or "", lang, scorer.prefix_ids, scorer.suffix_ids))
    if "doc_tokens" in frame:
        recount = [len(tok.encode(doc or "", add_special_tokens=False)) for doc in frame.doc_text]
        over_limit = int(sum(1 for n in recount if n > DOC_TOKEN_LIMIT))
    lengths = [len(s) for s in sequences]
    no = np.full(len(frame), np.nan, dtype=np.float32)
    yes = np.full(len(frame), np.nan, dtype=np.float32)
    budget = max_batch_tokens
    started = time.monotonic()
    pending = token_budget_batches(lengths, budget, max_batch_size)
    oom_retries = 0
    while pending:
        batch = pending.pop(0)
        try:
            b_no, b_yes = scorer.logits([sequences[i] for i in batch])
        except Exception as exc:  # noqa: BLE001 - OOM types differ per backend
            if not scorer.is_oom(exc) or len(batch) == 1:
                raise
            scorer.empty_cache()
            oom_retries += 1
            budget = max(budget // 2, max(lengths[i] for i in batch))
            log("oom; halving token budget and re-planning the rest of the part", budget=budget, batch=len(batch))
            rest = batch + [i for b in pending for i in b]
            pending = [[rest[j] for j in sub] for sub in token_budget_batches([lengths[i] for i in rest], budget, max_batch_size)]
            continue
        no[batch] = b_no
        yes[batch] = b_yes
    seconds = time.monotonic() - started
    if not (np.isfinite(no).all() and np.isfinite(yes).all()):
        bad = int((~np.isfinite(no) | ~np.isfinite(yes)).sum())
        raise FloatingPointError(f"{bad} non-finite yes/no logits (fp16 overflow?); rerun with --dtype bfloat16/float32")
    result = pd.DataFrame(
        {
            "pair_id": frame.pair_id.astype(np.int64).to_numpy(),
            "query_id": frame.query_id.astype(np.int64).to_numpy(),
            "doc_id": frame.doc_id.astype(np.int64).to_numpy(),
            "score": yes_probability(no, yes),
            "logit_diff": (yes.astype(np.float64) - no.astype(np.float64)).astype(np.float32),
        }
    )
    stats = {
        "pairs": len(frame),
        "seconds": round(seconds, 3),
        "prompt_tokens": int(sum(lengths)),
        "max_prompt_tokens": int(max(lengths)) if lengths else 0,
        "doc_over_token_limit": over_limit,
        "oom_retries": oom_retries,
        "final_batch_tokens": budget,
    }
    return result, stats


def make_scorer(args: argparse.Namespace, log: Logger) -> Any:
    spec = MODELS[args.model]
    model_path = args.model_path or spec["hf_id"]
    revision = None if args.model_path else (args.revision or spec.get("revision"))
    if args.backend == "vllm":
        return VLLMScorer(model_path, revision, args.dtype, log, max_model_len=args.max_model_len,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          quantization=args.quantization or spec.get("quantization"))
    if spec.get("quantization") and args.backend == "hf":
        raise ValueError(f"{args.model} needs --backend vllm (quantized checkpoint)")
    return HFScorer(model_path, revision, args.dtype, log, attn=args.attn, device_map=args.device_map)


def _acquire_lock(path: Path) -> None:
    _guard(path)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        info = {}
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        same_host = info.get("host") == socket.gethostname()
        alive = False
        if same_host and int(info.get("pid", -1)) > 0:
            try:
                os.kill(int(info["pid"]), 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            raise RuntimeError(f"Lock {path} held by live PID {info.get('pid')}")
        path.unlink()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "utc": _now()}, fh)


def _model_dir(out: Path, model: str) -> Path:
    return out / model


def _seed_from(resume_from: Sequence[Path], parts_dir: Path, model: str, log: Logger) -> int:
    """Copy finished parts from read-only earlier outputs (e.g. a previous Kaggle session)."""
    copied = 0
    for root in resume_from:
        for candidate in (root / model / "parts", root / "parts", root):
            if not candidate.is_dir():
                continue
            for src in sorted(candidate.glob("pool-*.part-*.parquet")):
                dst = parts_dir / src.name
                if not dst.exists():
                    _atomic_parquet(dst, pd.read_parquet(src))
                    meta_src = src.with_suffix(".json")
                    if meta_src.exists():
                        _atomic_json(dst.with_suffix(".json"), json.loads(meta_src.read_text(encoding="utf-8")))
                    copied += 1
    if copied:
        log("seeded finished parts from earlier output", parts=copied)
    return copied


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    pool = Path(args.pool)
    out = _model_dir(Path(args.out), args.model)
    parts_dir = out / "parts"
    _guard(out)
    parts_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out / f"score-w{args.worker}.log")
    _seed_from([Path(p) for p in args.resume_from], parts_dir, args.model, log)
    parts = worker_parts(plan_parts(pool_shards(pool), args.part_pairs, args.subset), args.worker, args.workers)
    todo = [part for part in parts if not (parts_dir / part_name(part)).exists()]
    log("plan", subset=args.subset, worker=args.worker, workers=args.workers, parts=len(parts), todo=len(todo),
        pairs_worker=part_pairs_count(parts), pairs_todo=part_pairs_count(todo))
    if not todo:
        return {"status": "nothing_to_do", "worker": args.worker}
    lock = out / f"score-w{args.worker}.lock"
    _acquire_lock(lock)
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
            target = parts_dir / part_name(part)
            meta = {**stats, "model": args.model, "backend": args.backend, "dtype": scorer.dtype_name,
                    "revision": scorer.resolved_revision, "gpu": gpu_info().get("name"), "shard": shard,
                    "subset": args.subset, "part": part["part"], "rows": len(part["positions"]), "utc": _now(),
                    "schema": SCHEMA_VERSION}
            _atomic_parquet(target, result)
            _atomic_json(target.with_suffix(".json"), meta)
            done_pairs += stats["pairs"]
            done_seconds += stats["seconds"]
            rate = done_pairs / done_seconds if done_seconds else 0.0
            left = part_pairs_count(todo) - done_pairs
            log("part done", part=target.name, pairs=stats["pairs"], seconds=stats["seconds"],
                rate=round(rate, 3), eta_worker_s=round(left / rate) if rate else None,
                oom_retries=stats["oom_retries"], doc_over_limit=stats["doc_over_token_limit"])
        report = {"status": "stopped" if done_pairs < part_pairs_count(todo) else "worker_complete",
                  "subset": args.subset, "worker": args.worker, "pairs_scored": done_pairs, "seconds_scoring": round(done_seconds, 1),
                  "seconds_model_load": round(load_seconds, 1), "pairs_per_second": done_pairs / done_seconds if done_seconds else None,
                  "gpu": gpu_info(), "model": args.model, "backend": args.backend, "dtype": scorer.dtype_name}
        _atomic_json(out / f"worker-{args.worker}-{args.subset}-report.json", report)
        return report
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    pool = Path(args.pool)
    out = _model_dir(Path(args.out), args.model)
    _guard(out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"pilot-{args.pairs}-{args.backend}-{args.dtype}"
    log = Logger(out / f"{tag}.log")
    shards = pool_shards(pool)
    total_pairs = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    frame = pilot_rows(shards, args.pairs, only_pool13=not args.any_pairs)
    load_started = time.monotonic()
    scorer = make_scorer(args, log)
    load_seconds = time.monotonic() - load_started
    # Warm-up on a few pairs so CUDA kernels/compilation do not count against the rate.
    score_frame(scorer, frame.iloc[: min(16, len(frame))], args.max_batch_tokens, args.max_batch_size, log)
    result, stats = score_frame(scorer, frame, args.max_batch_tokens, args.max_batch_size, log)
    rate = stats["pairs"] / stats["seconds"] if stats["seconds"] else None
    _atomic_parquet(out / f"{tag}.parquet", result)
    report = {
        "status": "pilot_complete",
        "model": args.model,
        "backend": args.backend,
        "dtype": scorer.dtype_name,
        "revision": scorer.resolved_revision,
        "pairs": stats["pairs"],
        "seconds": stats["seconds"],
        "pairs_per_second": rate,
        "pool_pairs_total": total_pairs,
        "eta_full_pool_seconds_one_gpu": total_pairs / rate if rate else None,
        "model_load_seconds": round(load_seconds, 1),
        "prompt_tokens_mean": stats["prompt_tokens"] / max(stats["pairs"], 1),
        "max_prompt_tokens": stats["max_prompt_tokens"],
        "doc_over_token_limit": stats["doc_over_token_limit"],
        "oom_retries": stats["oom_retries"],
        "max_batch_tokens": args.max_batch_tokens,
        "max_batch_size": args.max_batch_size,
        "gpu": gpu_info(),
        "pilot_seed": PILOT_SEED,
        "pilot_only_pool13": not args.any_pairs,
        "scores": str(out / f"{tag}.parquet"),
        "utc": _now(),
    }
    try:
        import torch

        report["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:
        pass
    _atomic_json(out / f"{tag}.json", report)
    log("pilot done", **{k: report[k] for k in ("pairs", "seconds", "pairs_per_second", "eta_full_pool_seconds_one_gpu")})
    return report


def run_status(args: argparse.Namespace) -> dict[str, Any]:
    parts_dir = _model_dir(Path(args.out), args.model) / "parts"
    parts = plan_parts(pool_shards(Path(args.pool)), args.part_pairs, args.subset)
    done = [p for p in parts if (parts_dir / part_name(p)).exists()]
    report = {"model": args.model, "subset": args.subset, "parts_done": len(done), "parts_total": len(parts),
              "pairs_done": part_pairs_count(done), "pairs_total": part_pairs_count(parts),
              "complete": len(done) == len(parts)}
    print(json.dumps(report), flush=True)
    return report


def run_collect(args: argparse.Namespace) -> dict[str, Any]:
    """Merge every finished part (any subset naming) and keep the pairs of ``--subset``.

    Each part is checked against the pool shards (pair_id -> query_id/doc_id); a pair
    scored in two part files keeps the first file in name order and is counted.
    """
    pool = Path(args.pool)
    out = _model_dir(Path(args.out), args.model)
    parts_dir = out / "parts"
    _seed_from([Path(p) for p in args.resume_from], parts_dir, args.model, Logger(out / "collect.log"))
    shards = pool_shards(pool)
    keys = pd.concat([pd.read_parquet(s, columns=["pair_id", "query_id", "doc_id", "in_pool13"]) for s in shards],
                     ignore_index=True).set_index("pair_id")
    wanted = keys.index if args.subset == "all" else keys.index[keys.in_pool13 == (args.subset == "pool13")]
    frames, metas = [], []
    for target in sorted(parts_dir.glob("pool-*.part-*.parquet")):
        frame = pd.read_parquet(target)
        ref = keys.reindex(frame.pair_id.to_numpy())
        if ref.query_id.isna().any() or not (
            (ref.query_id.to_numpy() == frame.query_id.to_numpy()).all()
            and (ref.doc_id.to_numpy() == frame.doc_id.to_numpy()).all()
        ):
            raise ValueError(f"{target.name}: pair_id/query_id/doc_id differ from the pool shards")
        frames.append(frame)
        meta_path = target.with_suffix(".json")
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    if not frames:
        raise FileNotFoundError(f"No parts under {parts_dir}")
    merged = pd.concat(frames, ignore_index=True)
    duplicates = int(merged.pair_id.duplicated().sum())
    merged = merged.drop_duplicates("pair_id", keep="first").set_index("pair_id")
    missing = wanted.difference(merged.index)
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} pairs of subset {args.subset} have no score yet")
    scores = merged.loc[wanted].reset_index().sort_values("pair_id", kind="stable")
    if not np.isfinite(scores.score).all():
        raise ValueError("Non-finite scores")
    target = out / f"scores-{args.subset}.parquet"
    _atomic_parquet(target, scores)
    gpus: dict[str, float] = {}
    for meta in metas:
        gpus[str(meta.get("gpu"))] = gpus.get(str(meta.get("gpu")), 0.0) + float(meta.get("seconds", 0.0))
    report = {
        "status": "collected",
        "model": args.model,
        "subset": args.subset,
        "pairs": len(scores),
        "part_files_read": len(frames),
        "duplicate_pairs_dropped": duplicates,
        "scoring_gpu_seconds_by_gpu_all_parts": {k: round(v, 1) for k, v in gpus.items()},
        "scoring_gpu_seconds_total_all_parts": round(sum(gpus.values()), 1),
        "dtypes": sorted({str(m.get("dtype")) for m in metas}),
        "backends": sorted({str(m.get("backend")) for m in metas}),
        "revisions": sorted({str(m.get("revision")) for m in metas}),
        "doc_over_token_limit": int(sum(int(m.get("doc_over_token_limit", 0)) for m in metas)),
        "scores": str(target),
        "scores_sha256": _sha256(target),
        "utc": _now(),
    }
    _atomic_json(out / f"collect-{args.subset}-report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, model_args: bool = True) -> None:
        p.add_argument("--pool", required=True, help="dir with shards/pool-*.parquet (or the shards themselves)")
        p.add_argument("--out", required=True, help="output root; results go to OUT/<model>/")
        p.add_argument("--model", required=True, choices=sorted(MODELS))
        p.add_argument("--part-pairs", type=int, default=DEFAULT_PART_PAIRS)
        p.add_argument("--resume-from", action="append", default=[], help="read-only earlier output root(s)")
        p.add_argument("--subset", choices=SUBSETS, default="all",
                       help="all pairs, pool13 (in_pool13: the 1.3x pool) or tail (1.3x-2.0x)")
        if model_args:
            p.add_argument("--model-path", default=None, help="local snapshot dir instead of the HF id")
            p.add_argument("--revision", default=None, help="HF revision (commit) to pin")
            p.add_argument("--backend", choices=("hf", "vllm"), default="hf")
            p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
            p.add_argument("--attn", default="sdpa", help="transformers attn_implementation")
            p.add_argument("--device-map", default=None, help='hf device_map, e.g. "auto" to split over GPUs')
            p.add_argument("--max-batch-tokens", type=int, default=16_384, help="padded tokens per batch (hf)")
            p.add_argument("--max-batch-size", type=int, default=32)
            p.add_argument("--max-model-len", type=int, default=2048, help="vllm")
            p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vllm")
            p.add_argument("--quantization", default=None, help="vllm quantization, e.g. awq")

    pilot = sub.add_parser("pilot", help="score a deterministic sample and report speed/ETA")
    common(pilot)
    pilot.add_argument("--pairs", type=int, default=1000)
    pilot.add_argument("--any-pairs", action="store_true", help="sample the whole pool, not only the 1.3x part")
    score = sub.add_parser("score", help="score this worker's parts, resuming finished ones")
    common(score)
    score.add_argument("--worker", type=int, default=0)
    score.add_argument("--workers", type=int, default=1)
    score.add_argument("--max-seconds", type=float, default=0.0, help="stop starting new parts after this many seconds")
    collect = sub.add_parser("collect", help="verify parts and write OUT/<model>/scores-<subset>.parquet")
    common(collect, model_args=False)
    status = sub.add_parser("status", help="print finished/total parts for --subset")
    common(status, model_args=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pilot":
        run_pilot(args)
    elif args.command == "score":
        print(json.dumps(run_score(args), ensure_ascii=False, indent=2), flush=True)
    elif args.command == "collect":
        run_collect(args)
    elif args.command == "status":
        run_status(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
