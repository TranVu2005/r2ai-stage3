"""S38 step 5 (prepare only, nothing is scored): chunk-level 0.6B judge input for the t256 chunks S32 did not score.

``python -m r2ai.s38.judge_prep`` -> ``out/runs/s38-chunk-metric-v2/judge-pool/`` (S32 shard layout, so
``r2ai.s31.judge_score score --pool <dir> --model qwen3-reranker-0.6b --subset all`` scores it as is).

* pairs = (J3-thrC carrier, canonical t256 answer/body chunk of its doc) without an S32 row for that
  (query, doc, chunk); the title/question chunks are counted in the report but not exported (never submitted).
* ``doc_text`` = chunk text cut to 512 Qwen3 tokens (``truncate_document``), prompt/instruction as S29/S32,
  ``chunk_rank`` = rank of the chunk by cosine(S29 query vector, chunk vector) among the doc's answer/body chunks,
  ``in_pool13`` = True for every row, ``src_pair_id`` = the S32 pair of (query, doc).
* ETA = pairs / the S32 local rate (RTX 3050, HF bf16, ``worker-0-all-report.json``), also scaled by the mean
  prompt length relative to S32.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s38.score import S32_ROOT, load_query_vectors, piece_vectors
from r2ai.s38.sources import _atomic_json, _atomic_parquet, _sha256, describe, run_root

SHARD_PAIRS = 20_000
S32_PROMPT_MEAN = 284.5833091864459   # S32 manifest prompt_tokens_mean (pool13 main rows)


def prepare(out: Path) -> dict[str, Any]:
    started = time.monotonic()
    from r2ai.s29.qwen_judge import DOC_TOKEN_LIMIT, INSTRUCTIONS, PREFIX, SUFFIX, _inputs, _model_path, truncate_document

    src = run_root() / "sources"
    canonical = paths.DATA_DIR / "canonical-20261018"
    pieces = pd.read_parquet(src / "pieces.parquet")
    pieces = pieces[pieces.kind == "t256"].reset_index(drop=True)
    carriers = pd.read_parquet(src / "carriers.parquet")
    cp = pd.read_parquet(paths.OUT_DIR / S32_ROOT / "pool" / "chunk_pairs.parquet",
                         columns=["src_pair_id", "query_id", "doc_id", "chunk_id", "field", "lang", "cosine_rank",
                                  "cosine_score", "source"])
    cp = cp.merge(carriers[["query_id", "doc_id"]], on=["query_id", "doc_id"])
    scored = set(zip(cp.query_id, cp.doc_id, cp.chunk_id))
    pair = cp.drop_duplicates(["query_id", "doc_id"]).set_index(["query_id", "doc_id"])

    # all t256 fields of the carrier docs, for the count with title/question
    owners = pieces.drop_duplicates("doc_id").set_index("doc_id").owner
    meta = pq.read_table(canonical / "chunks_t256.parquet", columns=["chunk_id", "doc_id", "field"],
                         filters=[("doc_id", "in", sorted(set(owners.astype(int))))]).to_pandas()
    per_owner = meta.groupby("doc_id").size()
    n_all = carriers.doc_id.map(owners).map(per_owner).fillna(0).astype(int)
    scored_per_carrier = cp.groupby(["query_id", "doc_id"]).size().reindex(
        pd.MultiIndex.from_frame(carriers[["query_id", "doc_id"]]), fill_value=0).to_numpy()

    vec = piece_vectors(pieces, canonical, None)
    qv = load_query_vectors()
    by_doc = {d: g.index.to_numpy() for d, g in pieces.groupby("doc_id", sort=False)}
    rows = []
    for qid, d in zip(carriers.query_id.astype(int), carriers.doc_id.astype(int)):
        idx = by_doc.get(d)
        if idx is None:
            continue
        cos = vec[idx].astype(np.float32) @ qv[qid].astype(np.float32)
        order = np.lexsort((pieces.idx.to_numpy()[idx], -cos))
        rank = np.empty(len(idx), dtype=np.int32)
        rank[order] = np.arange(1, len(idx) + 1)
        for r, k, c in zip(rank, idx, cos):
            cid = int(pieces.chunk_id.iat[k])
            if (qid, d, cid) not in scored:
                rows.append((qid, d, cid, int(r), float(c), k))
    todo = pd.DataFrame(rows, columns=["query_id", "doc_id", "chunk_id", "chunk_rank", "chunk_cosine", "piece_row"])
    todo = todo.join(pair[["src_pair_id", "lang", "cosine_rank", "cosine_score"]], on=["query_id", "doc_id"])
    if todo.src_pair_id.isna().any():
        raise ValueError("carrier without S32 pair")
    todo["field"] = pieces.field.to_numpy()[todo.piece_row]
    text = pieces.text.to_numpy()[todo.piece_row]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_model_path(), local_files_only=True)
    uniq = {}
    for t in dict.fromkeys(text):
        uniq[t] = None
    keys = list(uniq)
    lens = []
    for begin in range(0, len(keys), 20_000):
        lens += [len(x) for x in tokenizer(keys[begin:begin + 20_000], add_special_tokens=False)["input_ids"]]
    truncated = 0
    for t, n in zip(keys, lens):
        if n > DOC_TOKEN_LIMIT:
            uniq[t] = truncate_document(tokenizer, t)
            truncated += 1
        else:
            uniq[t] = (t, n)
    todo["doc_text"] = [uniq[t][0] for t in text]
    todo["doc_tokens"] = np.array([uniq[t][1] for t in text], dtype=np.int32)
    q = pq.read_table(_inputs()["query"], columns=["id", "query"]).to_pandas()
    todo["query_text"] = todo.query_id.map(dict(zip(q.id.astype(int), q["query"].fillna(""))))
    prefix_len = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    suffix_len = len(tokenizer.encode(SUFFIX, add_special_tokens=False))
    prompt = []
    bodies = [f"<Instruct>: {INSTRUCTIONS[lang]}\n<Query>: {qt}\n<Document>: {dt}"
              for lang, qt, dt in zip(todo.lang, todo.query_text, todo.doc_text)]
    for begin in range(0, len(bodies), 20_000):
        prompt += [len(x) for x in tokenizer(bodies[begin:begin + 20_000], add_special_tokens=False)["input_ids"]]
    del bodies
    todo["prompt_tokens"] = np.array(prompt, dtype=np.int32) + prefix_len + suffix_len
    todo = todo.sort_values(["query_id", "cosine_rank", "chunk_rank"], kind="stable").reset_index(drop=True)
    todo["pair_id"] = np.arange(len(todo), dtype=np.int64)
    todo["in_pool13"] = True
    todo["source"] = "s38_t256_unscored"
    todo["src_pair_id"] = todo.src_pair_id.astype(np.int64)

    columns = ["pair_id", "query_id", "doc_id", "cosine_rank", "cosine_score", "in_pool13", "lang", "query_text",
               "doc_text", "doc_tokens", "chunk_id", "chunk_rank", "chunk_cosine", "field", "src_pair_id",
               "prompt_tokens", "source"]
    (out / "shards").mkdir(parents=True, exist_ok=True)
    shards = []
    for k, begin in enumerate(range(0, len(todo), SHARD_PAIRS)):
        target = out / "shards" / f"pool-{k:05d}.parquet"
        paths.assert_writable(target)
        temp = target.with_name(target.name + ".tmp")
        pq.write_table(pa.Table.from_pandas(todo.iloc[begin:begin + SHARD_PAIRS][columns], preserve_index=False),
                       temp, compression="zstd", row_group_size=5_000)
        temp.replace(target)
        shards.append({"file": target.name, "pairs": min(SHARD_PAIRS, len(todo) - begin), "sha256": _sha256(target)})
    _atomic_parquet(todo[[c for c in columns if c not in ("query_text", "doc_text")]], out / "chunk_pairs.parquet")

    rep32 = json.loads((paths.OUT_DIR / S32_ROOT / "gpu" / "qwen3-reranker-0.6b" / "worker-0-all-report.json").read_text(encoding="utf-8"))
    rate = float(rep32["pairs_per_second"])
    mean_prompt = float(todo.prompt_tokens.mean())
    report = {
        "status": "prepared_not_scored",
        "carriers": len(carriers),
        "t256_chunks_of_carriers_all_fields": int(n_all.sum()),
        "s32_scored_rows_of_carriers": int(scored_per_carrier.sum()),
        "unscored_all_fields": int(n_all.sum() - scored_per_carrier.sum()),
        "pairs_exported_answer_body": len(todo),
        "pairs_by_field": todo.field.value_counts().to_dict(),
        "pairs_by_lang": todo.lang.value_counts().to_dict(),
        "carriers_with_unscored": int(todo.groupby(["query_id", "doc_id"]).ngroups),
        "unscored_per_carrier": describe(todo.groupby(["query_id", "doc_id"]).size()),
        "chunk_rank_of_unscored": describe(todo.chunk_rank),
        "prompt_tokens": describe(todo.prompt_tokens), "chunks_truncated_to_512": truncated,
        "s32_rate_pairs_per_s": rate, "s32_prompt_tokens_mean": S32_PROMPT_MEAN,
        "eta_hours_s32_rate": len(todo) / rate / 3600,
        "eta_hours_prompt_scaled": len(todo) * (mean_prompt / S32_PROMPT_MEAN) / rate / 3600,
        "command": f"python -m r2ai.s31.judge_score score --pool {out} --out {out.parent / 'judge-gpu'} "
                   f"--model qwen3-reranker-0.6b --subset all",
        "shards": shards,
        "seconds": round(time.monotonic() - started, 1),
    }
    _atomic_json(out / "manifest.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out or run_root() / "judge-pool")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S38 output must stay below RUNS_DIR: {out}")
    for name in ("", "shards", "manifest.json", "manifest.json.tmp", "chunk_pairs.parquet", "chunk_pairs.parquet.tmp"):
        paths.assert_writable(out / name if name else out)
    if (out / "shards").exists() and any((out / "shards").glob("pool-*.parquet")):
        raise FileExistsError(f"{out / 'shards'} already holds shards; use a new --out")
    out.mkdir(parents=True, exist_ok=True)
    report = prepare(out)
    print(json.dumps({k: v for k, v in report.items() if k != "shards"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
