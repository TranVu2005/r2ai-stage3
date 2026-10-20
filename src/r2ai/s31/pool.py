"""Export the S31 judge pool: cosine top-(2.0 x K_q) pairs with Qwen3-truncated doc_text.

Run ``python -m r2ai.s31.pool export`` on CPU. The 2.0x pool reuses the S29 1.3x
pool verbatim for its leading ranks (same query vectors, doc vectors and
Pharmacity lots), so every S29 Qwen3-Reranker-0.6B score stays aligned. Output
lives in ``RUNS_DIR/s31-strong-judge/pool``: parquet shards for upload to a GPU
host, a text-free ``pool_index.parquet`` for the local builders, and
``manifest.json`` with hashes and statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29.qwen_judge import (
    DOC_TOKEN_LIMIT,
    INSTRUCTIONS,
    MODEL_ID as TOKENIZER_MODEL_ID,
    MODEL_REVISION as TOKENIZER_REVISION,
    _atomic_json,
    _inputs,
    _load_doc_vectors,
    _load_submission,
    _model_path,
    _sha256,
    truncate_document,
)

S31_NAME = "s31-strong-judge"
POOL_MULTIPLIER = 2.0
S29_POOL_MULTIPLIER = 1.3
DEFAULT_SHARD_PAIRS = 20_000
SHARD_MAX_BYTES = 500 * 1024 * 1024
# Text kept in memory per document while truncating; longer docs are re-read in full.
HEAD_CHARS = 20_000
# (window chars, margin chars): the 512th token must end at least `margin`
# characters before the cut so BPE boundary effects cannot reach it.
TRUNCATION_WINDOWS = ((4_096, 512), (16_384, 2_048))
SELF_CHECK_DOCS = 3_000
DOC_BATCH = 1_000


def pool_size(raw_k: int, multiplier: float = POOL_MULTIPLIER) -> int:
    """Python round, matching S29's ``round(1.3 * K_q)`` rule."""
    if raw_k < 0:
        raise ValueError("raw_k must be nonnegative")
    return round(raw_k * multiplier)


def run_root() -> Path:
    return paths.run_dir(S31_NAME)


def _require_under_runs(path: Path) -> Path:
    resolved = paths.resolve_path(path)
    if not resolved.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S31 output must stay below RUNS_DIR: {resolved}")
    paths.assert_writable(resolved)
    return resolved


def _finish_truncation(tokenizer: Any, token_ids: list[int], limit: int) -> tuple[str, int]:
    """Same decode/re-encode loop as ``r2ai.s29.qwen_judge.truncate_document``."""
    token_ids = token_ids[:limit]
    passage = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    while len(tokenizer.encode(passage, add_special_tokens=False)) > limit:
        token_ids = token_ids[:-1]
        passage = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return passage, len(tokenizer.encode(passage, add_special_tokens=False))


def truncate_heads(
    tokenizer: Any,
    texts: Sequence[str],
    limit: int = DOC_TOKEN_LIMIT,
    windows: Sequence[tuple[int, int]] = TRUNCATION_WINDOWS,
) -> list[tuple[str, int] | None]:
    """Truncate each text from a leading window; ``None`` means the full text is needed.

    A window is accepted only when the text is shorter than the window (then it
    is the whole text) or the ``limit``-th token ends ``margin`` characters
    before the cut, so its first ``limit`` token IDs equal the full text's.
    """
    result: list[tuple[str, int] | None] = [None] * len(texts)
    pending = list(range(len(texts)))
    for window, margin in windows:
        if not pending:
            break
        heads = [texts[i][:window] for i in pending]
        enc = tokenizer(heads, add_special_tokens=False, return_offsets_mapping=True)
        still = []
        for slot, index in enumerate(pending):
            ids = enc["input_ids"][slot]
            offsets = enc["offset_mapping"][slot]
            whole = len(texts[index]) <= window
            if whole or (len(ids) > limit and offsets[limit - 1][1] <= window - margin):
                result[index] = _finish_truncation(tokenizer, list(ids), limit)
            else:
                still.append(index)
        pending = still
    return result


def _top_cosine(
    q_dense: np.ndarray,
    doc_dense: np.ndarray,
    max_pool: int,
    report: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray]:
    """Same blockwise top-k as S29 ``prepare`` with a larger ``max_pool``."""
    top_i = np.zeros((len(q_dense), 0), dtype=np.int64)
    top_s = np.zeros((len(q_dense), 0), dtype=np.float32)
    q32 = q_dense.astype(np.float32)
    start = time.monotonic()
    for begin in range(0, len(doc_dense), 50_000):
        scores = q32 @ doc_dense[begin : begin + 50_000].astype(np.float32).T
        take = min(max_pool, scores.shape[1])
        part = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        top_i = np.c_[top_i, part + begin]
        top_s = np.c_[top_s, np.take_along_axis(scores, part, axis=1)]
        keep = np.argsort(-top_s, axis=1, kind="stable")[:, :max_pool]
        top_i = np.take_along_axis(top_i, keep, axis=1)
        top_s = np.take_along_axis(top_s, keep, axis=1)
        if (begin // 50_000 + 1) % 4 == 0 or begin + 50_000 >= len(doc_dense):
            report(f"cosine {min(begin + 50_000, len(doc_dense))}/{len(doc_dense)} docs elapsed_s={time.monotonic() - start:.0f}")
    return top_i, top_s


def merge_with_s29(
    query_id: int,
    s29_rows: pd.DataFrame,
    new_doc_ids: Sequence[int],
    new_scores: Sequence[float],
    size: int,
) -> tuple[list[tuple[int, float]], bool]:
    """Keep S29's leading ranks verbatim, then append the remaining new cosine ranks.

    Returns ``([(doc_id, cosine_score), ...], swapped)``. Docs with identical
    vectors tie exactly, and argpartition picks an arbitrary subset of a tie at
    the S29 cut; ``swapped`` counts docs of the new top-n13 that S29 did not
    keep. A swap is accepted only when both sides score exactly the boundary
    score; any other difference raises.
    """
    s29_rows = s29_rows.sort_values("cosine_rank")
    n13 = len(s29_rows)
    if list(s29_rows.cosine_rank) != list(range(1, n13 + 1)):
        raise ValueError(f"S29 pool ranks are not 1..n for query {query_id}")
    if size < n13 or len(new_doc_ids) < size:
        raise ValueError(f"Query {query_id}: pool size {size} smaller than S29 {n13} or cosine top {len(new_doc_ids)}")
    s29_pairs = list(zip(s29_rows.doc_id.astype(int), s29_rows.cosine_score.astype(float)))
    new_pairs = [(int(d), float(s)) for d, s in zip(new_doc_ids[:size], new_scores[:size])]
    new_score = dict(new_pairs)
    old_score = dict(s29_pairs)
    head = dict(new_pairs[:n13])
    boundary = new_pairs[n13 - 1][1] if n13 else None
    only_new = set(head) - set(old_score)
    only_old = set(old_score) - set(head)
    if any(head[d] != boundary for d in only_new) or any(old_score[d] != boundary for d in only_old):
        raise ValueError(f"Query {query_id}: new cosine top-{n13} differs from the S29 pool beyond a boundary tie")
    if any(new_score[d] != old_score[d] for d in set(old_score) & set(new_score)):
        raise ValueError(f"Query {query_id}: cosine score differs from S29")
    tail = [pair for pair in new_pairs if pair[0] not in old_score][: size - n13]
    if len(tail) != size - n13 or len({d for d, _ in tail}) != len(tail):
        raise ValueError(f"Query {query_id}: cannot fill the cosine tail")
    if tail and tail[0][1] > s29_pairs[-1][1]:
        raise ValueError(f"Query {query_id}: tail doc scores above the S29 cut")
    return s29_pairs + tail, len(only_new)


def _shard_groups(pairs_per_query: dict[int, int], target: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    count = 0
    for qid in sorted(pairs_per_query):
        current.append(qid)
        count += pairs_per_query[qid]
        if count >= target:
            groups.append(current)
            current, count = [], 0
    if current:
        groups.append(current)
    return groups


def _guard(out: Path, shard_count_hint: int = 200) -> None:
    targets = [out, out / "shards", out / "manifest.json", out / "manifest.json.tmp", out / "export.log",
               out / "pool_index.parquet", out / "pool_index.parquet.tmp"]
    for index in range(shard_count_hint):
        shard = out / "shards" / f"pool-{index:05d}.parquet"
        targets += [shard, shard.with_name(shard.name + ".tmp")]
    for target in targets:
        paths.assert_writable(target)


def _truncate_all(
    inputs: dict[str, Path], doc_ids: set[int], tokenizer: Any, report: Callable[[str], None]
) -> tuple[dict[int, tuple[str, int]], dict[int, str], set[int]]:
    """Stream candidate doc_text, truncating each batch as it is read.

    Returns (passage and token count, lang, docs that need the full text).
    Only a ``HEAD_CHARS`` head of each text is held at a time.
    """
    passages: dict[int, tuple[str, int]] = {}
    langs: dict[int, str] = {}
    need_full: set[int] = set()
    t0 = time.monotonic()

    def consume(batch_ids: list[int], texts: list[str]) -> None:
        heads = [text[:HEAD_CHARS] for text in texts]
        for begin in range(0, len(heads), DOC_BATCH):
            results = truncate_heads(tokenizer, heads[begin : begin + DOC_BATCH])
            for offset, result in enumerate(results):
                doc_id = batch_ids[begin + offset]
                full_len = len(texts[begin + offset])
                if result is None or (full_len > HEAD_CHARS and result[1] < DOC_TOKEN_LIMIT):
                    need_full.add(doc_id)
                else:
                    passages[doc_id] = result

    for lot in sorted(path for path in (inputs["extra_docs"] / "lots").iterdir() if path.is_dir()):
        table = pd.read_parquet(lot / "chunks" / "docs.parquet", columns=["doc_id", "doc_text"])
        ids, texts = [], []
        for doc_id, text in zip(table.doc_id.astype(int), table.doc_text):
            if doc_id in doc_ids:
                ids.append(int(doc_id))
                texts.append(text or "")
                langs[int(doc_id)] = "vi"
        consume(ids, texts)
    extra = set(langs)
    want = doc_ids - extra
    parquet = pq.ParquetFile(inputs["canonical"] / "docs.parquet")
    seen_batches = 0
    for batch in parquet.iter_batches(batch_size=20_000, columns=["doc_id", "lang", "doc_text"]):
        seen_batches += 1
        ids = batch.column("doc_id").to_pylist()
        hit = [i for i, doc_id in enumerate(ids) if doc_id in want]
        if hit:
            lang_col = batch.column("lang").to_pylist()
            text_col = batch.column("doc_text")
            hit_ids, texts = [], []
            for i in hit:
                doc_id = int(ids[i])
                if doc_id in langs:
                    raise ValueError(f"Duplicate canonical doc_id {doc_id}")
                if lang_col[i] not in INSTRUCTIONS:
                    raise ValueError(f"Unsupported passage language {lang_col[i]!r} for doc {doc_id}")
                langs[doc_id] = lang_col[i]
                hit_ids.append(doc_id)
                texts.append(text_col[i].as_py() or "")
            consume(hit_ids, texts)
        if seen_batches % 5 == 0:
            report(f"truncate canonical rows={seen_batches * 20_000} docs_done={len(passages)} full_needed={len(need_full)} elapsed_s={time.monotonic() - t0:.0f}")
    missing = doc_ids - set(langs)
    if missing:
        raise ValueError(f"Missing candidate document text for {len(missing)} IDs")
    if need_full:
        full = _full_texts(inputs, need_full)
        for doc_id in sorted(need_full):
            passages[doc_id] = truncate_document(tokenizer, full[doc_id])
    report(f"truncated docs={len(passages)} pharmacity={len(extra)} full_text_fallback={len(need_full)} seconds={time.monotonic() - t0:.0f}")
    return passages, langs, need_full


def _full_texts(inputs: dict[str, Path], doc_ids: set[int]) -> dict[int, str]:
    if not doc_ids:
        return {}
    table = pq.read_table(
        inputs["canonical"] / "docs.parquet",
        columns=["doc_id", "doc_text"],
        filters=[("doc_id", "in", sorted(doc_ids))],
    ).to_pandas()
    texts = {int(d): t or "" for d, t in zip(table.doc_id, table.doc_text)}
    for lot in sorted(path for path in (inputs["extra_docs"] / "lots").iterdir() if path.is_dir()):
        extra = pd.read_parquet(lot / "chunks" / "docs.parquet", columns=["doc_id", "doc_text"])
        texts.update({int(d): t or "" for d, t in zip(extra.doc_id, extra.doc_text) if int(d) in doc_ids})
    if set(texts) != doc_ids:
        raise ValueError("Could not reload full text for every long document")
    return texts


def export(out: Path, multiplier: float = POOL_MULTIPLIER, shard_pairs: int = DEFAULT_SHARD_PAIRS) -> dict[str, Any]:
    started = time.monotonic()
    if multiplier < S29_POOL_MULTIPLIER:
        raise ValueError("The S31 pool must contain the S29 1.3x pool")
    inputs = _inputs()
    s29_dir = paths.RUNS_DIR / "s29-qwen-judge"
    s29_pool_path = s29_dir / "candidate_pool.parquet"
    s29_prepare_path = s29_dir / "prepare.json"
    s29_qvec_path = s29_dir / "query_vectors.npz"
    for key in ("p1", "baseline", "vectors", "query", "canonical", "extra_docs", "extra_vectors"):
        paths.require_inputs(inputs[key])
    for path in (s29_pool_path, s29_prepare_path, s29_qvec_path, s29_dir / "Q-rrf.build.json"):
        paths.require_inputs(path)
    _guard(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    existing = sorted((out / "shards").glob("pool-*.parquet"))
    if existing:
        raise FileExistsError(f"{out / 'shards'} already holds {len(existing)} shards; use a new --out")

    def report(message: str) -> None:
        line = json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": os.getpid(), "message": message}, ensure_ascii=False)
        print(line, flush=True)
        with (out / "export.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(line + "\n")

    report(f"start export multiplier={multiplier} shard_pairs={shard_pairs}")
    s29_prepare = json.loads(s29_prepare_path.read_text(encoding="utf-8"))
    s29_pool_sha = _sha256(s29_pool_path)
    # S29's prepare.json predates its candidate_pool_sha256 field; Q-rrf.build.json records it.
    s29_qrrf = json.loads((s29_dir / "Q-rrf.build.json").read_text(encoding="utf-8"))
    if s29_pool_sha != s29_prepare.get("candidate_pool_sha256", s29_qrrf["candidate_pool_sha256"]):
        raise ValueError("S29 candidate pool SHA256 differs from the pool Q-rrf was built on")
    qvec_sha = _sha256(s29_qvec_path)
    if qvec_sha != s29_prepare["query_vectors_sha256"]:
        raise ValueError("S29 query_vectors.npz SHA256 differs from its prepare.json")
    doc_vec_sha = _sha256(inputs["vectors"])
    if doc_vec_sha != s29_prepare["doc_vectors_sha256"]:
        raise ValueError("Doc vectors differ from the ones S29 used")

    p1_rows = _load_submission(inputs["p1"])
    baseline_rows = _load_submission(inputs["baseline"])
    raw_k = {row["id"]: len(row["relevant_docs"]) for row in p1_rows}
    keep_k = {row["id"]: len(row["relevant_docs"]) for row in baseline_rows}
    if set(raw_k) != set(keep_k):
        raise ValueError("P1 and K90-ph query IDs differ")
    query_ids = sorted(raw_k)
    with np.load(s29_qvec_path) as qv:
        q_ids = qv["q_ids"].astype(np.int64).tolist()
        q_dense = np.array(qv["q_dense"], copy=True)
    if q_ids != query_ids:
        raise ValueError("S29 query vector IDs differ from the P1/K90-ph query IDs")
    size = {qid: pool_size(raw_k[qid], multiplier) for qid in query_ids}
    s29_size = {qid: pool_size(raw_k[qid], S29_POOL_MULTIPLIER) for qid in query_ids}

    doc_ids, doc_dense, base_count, extra_files = _load_doc_vectors(inputs)
    report(f"loaded doc vectors rows={len(doc_ids)} base={base_count} extra={len(doc_ids) - base_count}")
    top_i, top_s = _top_cosine(q_dense, doc_dense, max(size.values()), report)
    del doc_dense

    s29_pool = pd.read_parquet(s29_pool_path)
    s29_groups = dict(tuple(s29_pool.groupby("query_id")))
    rows = []
    tie_swaps: dict[int, int] = {}
    for qpos, qid in enumerate(query_ids):
        s29_rows = s29_groups[qid]
        if len(s29_rows) != s29_size[qid]:
            raise ValueError(f"S29 pool size for query {qid} is {len(s29_rows)}, expected {s29_size[qid]}")
        merged, swapped = merge_with_s29(
            qid, s29_rows, doc_ids[top_i[qpos]].tolist(), top_s[qpos].tolist(), size[qid]
        )
        if swapped:
            tie_swaps[qid] = swapped
        for rank, (doc_id, cos) in enumerate(merged, 1):
            rows.append((qid, doc_id, rank, cos, rank <= s29_size[qid]))
    index = pd.DataFrame(rows, columns=["query_id", "doc_id", "cosine_rank", "cosine_score", "in_pool13"])
    if index.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Duplicate doc in a query's pool")
    report(f"pool pairs={len(index)} pool13_pairs={int(index.in_pool13.sum())} boundary_tie_queries={len(tie_swaps)} swapped_docs={sum(tie_swaps.values())}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    unique_docs = set(index.doc_id.astype(int))
    order = sorted(unique_docs)
    passages, langs, need_full = _truncate_all(inputs, unique_docs, tokenizer, report)

    # Exactness check against S29's truncate_document on the full text.
    rng = random.Random(31)
    sample = sorted(rng.sample(order, min(SELF_CHECK_DOCS, len(order))))
    sample_full = _full_texts(inputs, set(sample))
    mismatches = [d for d in sample if truncate_document(tokenizer, sample_full[d]) != passages[d]]
    if mismatches:
        raise ValueError(f"Fast truncation differs from S29 truncate_document on {len(mismatches)} docs, e.g. {mismatches[:5]}")
    report(f"truncation self-check docs={len(sample)} mismatches=0")
    del sample_full

    qtable = pq.read_table(inputs["query"], columns=["id", "query"]).to_pandas()
    queries = dict(zip(qtable.id.astype(int), qtable["query"].fillna("")))
    index["lang"] = index.doc_id.map(langs)
    index["pair_id"] = np.arange(len(index), dtype=np.int64)
    groups = _shard_groups(index.groupby("query_id").size().to_dict(), shard_pairs)
    _guard(out, len(groups))
    shard_of_query = {qid: shard for shard, group in enumerate(groups) for qid in group}
    index["shard"] = index.query_id.map(shard_of_query).astype(np.int32)

    shard_reports = []
    for shard, group in enumerate(groups):
        part = index[index.query_id.isin(group)]
        table = pa.table(
            {
                "pair_id": pa.array(part.pair_id, pa.int64()),
                "query_id": pa.array(part.query_id, pa.int64()),
                "doc_id": pa.array(part.doc_id, pa.int64()),
                "cosine_rank": pa.array(part.cosine_rank, pa.int32()),
                "cosine_score": pa.array(part.cosine_score, pa.float64()),
                "in_pool13": pa.array(part.in_pool13, pa.bool_()),
                "lang": pa.array(part.lang, pa.string()),
                "query_text": pa.array([queries[q] for q in part.query_id], pa.string()),
                "doc_text": pa.array([passages[d][0] for d in part.doc_id], pa.string()),
                "doc_tokens": pa.array([passages[d][1] for d in part.doc_id], pa.int32()),
            }
        )
        target = out / "shards" / f"pool-{shard:05d}.parquet"
        temp = target.with_name(target.name + ".tmp")
        pq.write_table(table, temp, compression="zstd", row_group_size=5_000)
        temp.replace(target)
        nbytes = target.stat().st_size
        if nbytes > SHARD_MAX_BYTES:
            raise ValueError(f"{target.name} is {nbytes} bytes, over {SHARD_MAX_BYTES}")
        shard_reports.append({"file": target.name, "queries": len(group), "pairs": len(part), "bytes": nbytes, "sha256": _sha256(target)})
    index_path = out / "pool_index.parquet"
    temp = index_path.with_name(index_path.name + ".tmp")
    index.to_parquet(temp, index=False)
    temp.replace(index_path)

    tokens = np.array([passages[d][1] for d in order])
    manifest = {
        "status": "exported",
        "pool_rule": f"round({multiplier:g} * len(P1 relevant_docs)); leading round(1.3 * K_q) ranks copied from S29",
        "keep_rule": "len(relevant_docs) of each K90-ph row (unchanged from S29)",
        "queries": len(query_ids),
        "pairs": len(index),
        "pool13_pairs": int(index.in_pool13.sum()),
        "pool_docs_mean": float(np.mean(list(size.values()))),
        "pool_docs_min": min(size.values()),
        "pool_docs_max": max(size.values()),
        "keep_docs_total": int(sum(keep_k.values())),
        "unique_docs": len(order),
        "lang_pairs": index.lang.value_counts().to_dict(),
        "boundary_tie_swaps": {str(k): v for k, v in sorted(tie_swaps.items())},
        "boundary_tie_note": "exactly tied cosine docs at the S29 cut: S29's pick kept, other tied docs start the tail",
        "doc_token_limit": DOC_TOKEN_LIMIT,
        "truncation_tokenizer": f"{TOKENIZER_MODEL_ID}@{TOKENIZER_REVISION}",
        "truncation_rule": "r2ai.s29.qwen_judge.truncate_document on doc_text (leading window shortcut, self-checked)",
        "truncation_self_check_docs": len(sample),
        "full_text_fallback_docs": len(need_full),
        "doc_tokens_mean": float(tokens.mean()),
        "doc_tokens_max": int(tokens.max()),
        "docs_at_token_limit": int((tokens == DOC_TOKEN_LIMIT).sum()),
        "shards": shard_reports,
        "shard_bytes_total": int(sum(s["bytes"] for s in shard_reports)),
        "pool_index": str(index_path),
        "pool_index_sha256": _sha256(index_path),
        "s29_candidate_pool_sha256": s29_pool_sha,
        "s29_query_vectors_sha256": qvec_sha,
        "doc_vectors_sha256": doc_vec_sha,
        "extra_vector_files": [str(path) for path in extra_files],
        "p1_zip_sha256": _sha256(inputs["p1"]),
        "baseline_k90_ph_zip_sha256": _sha256(inputs["baseline"]),
        "instructions": INSTRUCTIONS,
        "seconds": round(time.monotonic() - started, 1),
    }
    _atomic_json(out / "manifest.json", manifest)
    report(f"done pairs={manifest['pairs']} shards={len(shard_reports)} bytes={manifest['shard_bytes_total']}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="build the 2.0x cosine pool shards on CPU")
    ex.add_argument("--out", type=Path, default=None, help="output dir below RUNS_DIR (default s31-strong-judge/pool)")
    ex.add_argument("--multiplier", type=float, default=POOL_MULTIPLIER)
    ex.add_argument("--shard-pairs", type=int, default=DEFAULT_SHARD_PAIRS)
    args = parser.parse_args(argv)
    out = _require_under_runs(args.out or run_root() / "pool")
    _guard(out)
    if args.command == "export":
        print(json.dumps(export(out, args.multiplier, args.shard_pairs), ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
