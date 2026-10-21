"""Build the S29 Q-filter and Q-RRF bundles on CPU using the S26 body/chunk rules."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.compose.base_io import lzma_dictionary
from r2ai.s29.qwen_judge import (
    BASELINE_NAME,
    DOC_TOKEN_LIMIT,
    INSTRUCTIONS,
    P1_NAME,
    RRF_K,
    MODEL_ID,
    MODEL_REVISION,
    _inputs,
    _iter_zip_rows,
    _load_submission,
    _sha256,
    select_qfilter,
    select_qrrf,
    select_weighted_qrrf,
)


def _body_span(doc_text: str | None, lang: str, spans: dict[str, tuple[int, int]]) -> str:
    text = doc_text or ""
    if lang == "zh" or not spans:
        return text
    start, end = spans.get("answer") or spans.get("body") or (0, len(text))
    return text[start:end]


def _load_spans(chunks_path: Path, doc_ids: set[int]) -> dict[int, dict[str, tuple[int, int]]]:
    if not doc_ids:
        return {}
    table = pq.read_table(
        chunks_path,
        columns=["doc_id", "field", "char_start", "char_end"],
        filters=[("doc_id", "in", sorted(doc_ids))],
    ).to_pandas()
    grouped = table.groupby(["doc_id", "field"]).agg(
        start=("char_start", "min"), end=("char_end", "max")
    )
    result: dict[int, dict[str, tuple[int, int]]] = {}
    for (doc_id, field), row in grouped.iterrows():
        result.setdefault(int(doc_id), {})[str(field)] = (int(row.start), int(row.end))
    return result


def _chunk_target(n_c: int, selected_docs: int, multiplier: float = 1.0) -> int:
    if n_c < 0 or selected_docs < 0:
        raise ValueError("n_c and selected_docs must be nonnegative")
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("chunk multiplier must be finite and nonnegative")
    return min(round(n_c * multiplier), selected_docs)


def _resolve_variant_options(
    variant: str,
    chunk_multiplier: float | None,
    whole_body: bool | None,
) -> tuple[float, bool]:
    is_qrrf = variant.startswith("QRRF-")
    resolved_multiplier = chunk_multiplier
    if resolved_multiplier is None:
        resolved_multiplier = 1.4 if is_qrrf else 1.0
    resolved_whole_body = whole_body
    if resolved_whole_body is None:
        resolved_whole_body = is_qrrf
    return float(resolved_multiplier), bool(resolved_whole_body)


def _validate_candidate_pool_sha(prepare_report: dict[str, Any], candidate_digest: str) -> None:
    prepared_candidate_digest = prepare_report.get("candidate_pool_sha256")
    if not prepared_candidate_digest:
        raise ValueError("Prepare report is missing candidate_pool_sha256")
    if prepared_candidate_digest != candidate_digest:
        raise ValueError("Prepare report belongs to a different candidate pool SHA256")


def _load_extra_docs(inputs: dict[str, Path]) -> tuple[dict[int, str], dict[int, dict[str, tuple[int, int]]]]:
    texts: dict[int, str] = {}
    spans: dict[int, dict[str, tuple[int, int]]] = {}
    for lot in sorted(path for path in (inputs["extra_docs"] / "lots").iterdir() if path.is_dir()):
        docs = pd.read_parquet(lot / "chunks" / "docs.parquet", columns=["doc_id", "doc_text"])
        texts.update({int(doc_id): text or "" for doc_id, text in zip(docs.doc_id, docs.doc_text)})
        chunk_path = lot / "chunks" / "chunks_t256.parquet"
        table = pq.read_table(
            chunk_path, columns=["doc_id", "field", "char_start", "char_end"]
        ).to_pandas()
        grouped = table.groupby(["doc_id", "field"]).agg(
            start=("char_start", "min"), end=("char_end", "max")
        )
        for (doc_id, field), row in grouped.iterrows():
            spans.setdefault(int(doc_id), {})[str(field)] = (int(row.start), int(row.end))
    return texts, spans


def _preflight_outputs(out: Path, variants: tuple[str, ...]) -> None:
    targets = [out]
    for name in variants:
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
    for target in targets:
        paths.assert_writable(target)


def _validate_score_alignment(
    candidates: list[dict[str, Any]], score_rows: list[tuple[int, int, int, float]]
) -> None:
    expected = {
        (int(row["query_id"]), int(row["doc_id"])): (
            int(row["cosine_rank"]),
            float(row["cosine_score"]),
        )
        for row in candidates
    }
    seen: set[tuple[int, int]] = set()
    for qid, doc_id, cosine_rank, cosine_score in score_rows:
        key = (int(qid), int(doc_id))
        if key not in expected:
            raise ValueError("Qwen score checkpoint contains query/document pairs outside the current candidate pool")
        expected_rank, expected_score = expected[key]
        if int(cosine_rank) != expected_rank or not np.isclose(
            float(cosine_score), expected_score, rtol=0.0, atol=1e-7
        ):
            raise ValueError(f"Cosine rank/score differs from current pool for query={qid}, doc={doc_id}")
        seen.add(key)
    if seen != set(expected):
        raise ValueError("Qwen score checkpoint does not cover the exact current candidate query/document pairs")


def _score_rows_sha256(scores: dict[int, list[dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for query_id in sorted(scores):
        for row in sorted(scores[query_id], key=lambda item: int(item["cosine_rank"])):
            payload = json.dumps(
                [
                    int(query_id),
                    int(row["doc_id"]),
                    int(row["cosine_rank"]),
                    float(row["cosine_score"]),
                    float(row["qwen_score"]),
                ],
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(payload)
            digest.update(b"\n")
    return digest.hexdigest()


def _read_full_scores(
    out: Path,
    candidates: list[dict[str, Any]],
    candidate_path: Path,
) -> dict[int, list[dict[str, Any]]]:
    expected_pairs = len(candidates)
    report_path = out / "full-score-report.json"
    paths.require_inputs(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "full_complete" or int(report.get("pairs_checkpointed_total", -1)) != expected_pairs:
        raise ValueError("Qwen full-score report is incomplete or has the wrong pair count")
    if report.get("candidate_pool_sha256") != _sha256(candidate_path):
        raise ValueError("Qwen full-score report belongs to a different candidate pool SHA256")
    score_path = out / "qwen_scores.sqlite"
    paths.require_inputs(score_path)
    grouped: dict[int, list[dict[str, Any]]] = {}
    alignment_rows = []
    with sqlite3.connect(score_path) as conn:
        metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        expected_metadata = {
            "candidate_sha256": _sha256(candidate_path),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "instruction_vi": INSTRUCTIONS["vi"],
            "instruction_zh": INSTRUCTIONS["zh"],
            "doc_token_limit": str(DOC_TOKEN_LIMIT),
        }
        if metadata != expected_metadata:
            raise ValueError("Qwen score checkpoint belongs to different candidate or scoring settings")
        rows = conn.execute(
            "SELECT query_id, doc_id, cosine_rank, cosine_score, qwen_score FROM scores "
            "ORDER BY query_id, cosine_rank"
        )
        count = 0
        for qid, doc_id, cosine_rank, cosine_score, qwen_score in rows:
            alignment_rows.append((int(qid), int(doc_id), int(cosine_rank), float(cosine_score)))
            if not np.isfinite(qwen_score):
                raise ValueError(f"Non-finite Qwen score for query={qid}, doc={doc_id}")
            grouped.setdefault(int(qid), []).append(
                {
                    "query_id": int(qid),
                    "doc_id": int(doc_id),
                    "cosine_rank": int(cosine_rank),
                    "cosine_score": float(cosine_score),
                    "qwen_score": float(qwen_score),
                }
            )
            count += 1
    if count != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} scores, got {count}")
    _validate_score_alignment(candidates, alignment_rows)
    for rows in grouped.values():
        reranked = sorted(
            rows,
            key=lambda row: (
                -float(row["qwen_score"]),
                int(row["cosine_rank"]),
                int(row["doc_id"]),
            ),
        )
        for rank, row in enumerate(reranked, 1):
            row["qwen_rank"] = rank
    return grouped


def _row_id_maps(rows: list[dict[str, Any]], cid_to_doc: dict[int, int]) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
    p1_any: dict[int, int] = {}
    row_ids: dict[int, dict[int, int]] = {}
    for row in rows:
        mapping: dict[int, int] = {}
        for corpus_id in row["relevant_docs"]:
            doc_id = cid_to_doc.get(int(corpus_id))
            if doc_id is not None and doc_id not in mapping:
                mapping[doc_id] = int(corpus_id)
                p1_any.setdefault(doc_id, int(corpus_id))
        row_ids[int(row["id"])] = mapping
    return p1_any, row_ids


def _load_aliases(chunks_path: Path, docs_path: Path) -> tuple[dict[int, list[int]], dict[int, int]]:
    groups: dict[int, list[int]] = {}
    cid_to_doc: dict[int, int] = {}
    parquet = pq.ParquetFile(chunks_path)
    for index in range(parquet.num_row_groups):
        table = parquet.read_row_group(index, columns=["doc_id", "doc_ids"])
        doc_ids = table["doc_id"].to_pylist()
        aliases = table["doc_ids"].to_pylist()
        for doc_id, ids in zip(doc_ids, aliases):
            owner = int(doc_id)
            ids = [int(value) for value in (ids or [owner])]
            groups[owner] = ids
            for corpus_id in ids:
                cid_to_doc[corpus_id] = owner
    doc_table = pq.read_table(docs_path, columns=["doc_id"]).to_pandas()
    group_owners = set(groups)
    physical = {
        int(doc_id): cid_to_doc[int(doc_id)]
        for doc_id in doc_table.doc_id
        if int(doc_id) not in group_owners and int(doc_id) in cid_to_doc
    }
    return groups, cid_to_doc | {alias: owner for alias, owner in physical.items()}


def _canonical_owner(doc_id: int, groups: dict[int, list[int]], physical: dict[int, int]) -> int:
    return doc_id if doc_id in groups else physical.get(doc_id, doc_id)


def _corpus_id(
    doc_id: int,
    row_ids: dict[int, int],
    p1_any: dict[int, int],
    groups: dict[int, list[int]],
) -> int:
    if doc_id in row_ids:
        return row_ids[doc_id]
    if doc_id in p1_any:
        return p1_any[doc_id]
    aliases = groups.get(doc_id, [doc_id])
    return doc_id if doc_id in aliases else int(aliases[0])


def _carriers(
    row: dict[str, Any],
    picked: list[tuple[int, int]],
    own_texts: dict[int, str],
    all_p1_texts: dict[int, str],
    groups: dict[int, list[int]],
    physical: dict[int, int],
    chunk_multiplier: float = 1.0,
    thin_texts: dict[int, str] | None = None,
) -> list[tuple[int, int]]:
    """Selected docs that get a chunk, in selection order, capped at the n_c target.

    ``thin_texts`` (``--thin-chunks``): thin docs without chunks are also carriers, with
    their whole text as the chunk; they count toward the cap like any other doc.
    """
    count = _chunk_target(len(row["relevant_chunks"]), len(picked), chunk_multiplier)
    return _eligible_carriers(picked, own_texts, all_p1_texts, groups, physical, thin_texts)[:count]


def _eligible_carriers(
    picked: list[tuple[int, int]],
    own_texts: dict[int, str],
    all_p1_texts: dict[int, str],
    groups: dict[int, list[int]],
    physical: dict[int, int],
    thin_texts: dict[int, str] | None = None,
) -> list[tuple[int, int]]:
    """Selected docs that have a chunk source (P1 text, chunk group, or thin text), in order."""
    return [
        selected
        for selected in picked
        if selected[1] in own_texts
        or selected[1] in all_p1_texts
        or _canonical_owner(selected[0], groups, physical) in groups
        or (thin_texts is not None and selected[0] in thin_texts)
    ]


def _thin_carrier(
    doc_id: int,
    corpus_id: int,
    own_texts: dict[int, str],
    all_p1_texts: dict[int, str],
    groups: dict[int, list[int]],
    physical: dict[int, int],
    thin_texts: dict[int, str] | None,
) -> bool:
    """A carrier only because of ``--thin-chunks`` (no P1 text and no chunk group)."""
    return (
        thin_texts is not None
        and doc_id in thin_texts
        and corpus_id not in own_texts
        and corpus_id not in all_p1_texts
        and _canonical_owner(doc_id, groups, physical) not in groups
    )


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "relevant_docs": [int(doc_id) for doc_id in row["relevant_docs"]],
        "relevant_chunks": [
            {"doc_id": int(chunk["doc_id"]), "chunk_text": chunk["chunk_text"]}
            for chunk in row["relevant_chunks"]
        ],
    }


def _build_variant(
    out: Path,
    name: str,
    rows: list[dict[str, Any]],
    baseline: dict[int, dict[str, Any]],
    p1_any: dict[int, int],
    all_p1_texts: dict[int, str],
    row_ids: dict[int, dict[int, int]],
    groups: dict[int, list[int]],
    cid_to_doc: dict[int, int],
    doc_lang: dict[int, str],
    thin: set[int],
    canonical_path: Path,
    extra_texts: dict[int, str],
    extra_spans: dict[int, dict[str, tuple[int, int]]],
    chosen: dict[int, list[dict[str, Any]]],
    candidate_pool_sha256: str,
    candidate_pool_multiplier: float,
    score_rows_sha256: str,
    rrf_k: int | None,
    qwen_weight: float | None,
    chunk_multiplier: float,
    whole_body: bool,
    mean_doc_id_symdiff_vs_primary: float | None,
    fixed_k: bool = True,
    thin_texts: dict[int, str] | None = None,
    chunk_selector: Callable[[dict[int, list[tuple[int, int, bool]]]], dict[int, list[tuple[int, int]]]] | None = None,
) -> dict[str, Any]:
    """``chunk_selector`` replaces the n_c x multiplier carrier rule: it receives, per query,
    every selected doc with a chunk source as ``(doc_id, corpus_id, has_text)`` in selection
    order and returns the carriers to write (a subsequence of those with text)."""
    started = time.monotonic()
    inputs = _inputs()
    chunks_path = canonical_path / "chunks_t256.parquet"
    docs_path = canonical_path / "docs.parquet"
    physical = {
        doc_id: cid_to_doc[doc_id]
        for doc_id in doc_lang
        if doc_id not in groups and doc_id in cid_to_doc
    }
    picked_by_q: dict[int, list[tuple[int, int]]] = {}
    need_body: set[int] = set()
    for row in rows:
        qid = int(row["id"])
        row_own = {int(chunk["doc_id"]): chunk["chunk_text"] for chunk in row["relevant_chunks"]}
        picked = [
            (int(result["doc_id"]), _corpus_id(int(result["doc_id"]), row_ids[qid], p1_any, groups))
            for result in chosen[qid]
        ]
        picked_by_q[qid] = picked
        candidates = (
            _carriers(row, picked, row_own, all_p1_texts, groups, physical, chunk_multiplier, thin_texts)
            if chunk_selector is None
            else _eligible_carriers(picked, row_own, all_p1_texts, groups, physical, thin_texts)
        )
        for doc_id, corpus_id in candidates:
            if _thin_carrier(doc_id, corpus_id, row_own, all_p1_texts, groups, physical, thin_texts):
                continue
            if corpus_id not in row_own and corpus_id not in all_p1_texts:
                if doc_id not in extra_texts:
                    need_body.add(doc_id)
    owner_ids = {_canonical_owner(doc_id, groups, physical) for doc_id in need_body}
    spans = _load_spans(chunks_path, owner_ids)
    body_texts: dict[int, str] = {}
    if need_body - set(extra_texts):
        table = pq.read_table(
            docs_path,
            columns=["doc_id", "lang", "doc_text"],
            filters=[("doc_id", "in", sorted(need_body - set(extra_texts)))],
        ).to_pandas()
        for doc_id, lang, doc_text in zip(table.doc_id, table.lang, table.doc_text):
            doc_id = int(doc_id)
            owner = _canonical_owner(doc_id, groups, physical)
            body_texts[owner] = _body_span(doc_text, str(lang), spans.get(owner, {}))
    for doc_id in need_body & set(extra_texts):
        body_texts[doc_id] = _body_span(extra_texts[doc_id], "vi", extra_spans.get(doc_id, {}))

    carriers_by_q: dict[int, list[tuple[int, int]]] | None = None
    if chunk_selector is not None:
        offered: dict[int, list[tuple[int, int, bool]]] = {}
        for row in rows:
            qid = int(row["id"])
            row_own = {int(chunk["doc_id"]): chunk["chunk_text"] for chunk in row["relevant_chunks"]}
            items = []
            for doc_id, corpus_id in _eligible_carriers(
                picked_by_q[qid], row_own, all_p1_texts, groups, physical, thin_texts
            ):
                if corpus_id in row_own or corpus_id in all_p1_texts:
                    has_text = True
                elif _thin_carrier(doc_id, corpus_id, row_own, all_p1_texts, groups, physical, thin_texts):
                    has_text = True
                elif doc_id in extra_texts:
                    has_text = bool(body_texts.get(doc_id, "").strip())
                else:
                    has_text = bool(body_texts.get(_canonical_owner(doc_id, groups, physical), "").strip())
                items.append((doc_id, corpus_id, has_text))
            offered[qid] = items
        carriers_by_q = chunk_selector(offered)
        for qid, items in offered.items():
            allowed = [(d, c) for d, c, has_text in items if has_text]
            chosen_carriers = carriers_by_q[qid]
            position = {pair: index for index, pair in enumerate(allowed)}
            if any(pair not in position for pair in chosen_carriers) or [
                position[pair] for pair in chosen_carriers
            ] != sorted(position[pair] for pair in chosen_carriers):
                raise ValueError(f"chunk_selector returned carriers outside the offered order for query {qid}")

    target = out / f"{name}.zip"
    temp = out / f"{name}.{os.getpid()}.zip.tmp"
    stat_path = out / f"{name}.query_stats.parquet"
    stat_temp = stat_path.with_name(f"{stat_path.name}.{os.getpid()}.tmp")
    report_path = out / f"{name}.build.json"
    report_temp = report_path.with_name(f"{report_path.name}.{os.getpid()}.tmp")
    if target.is_file() and stat_path.is_file() and report_path.is_file():
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                existing_report.get("status") == "built_cpu_not_submitted"
                and existing_report.get("name") == name
                and existing_report.get("candidate_pool_sha256") == candidate_pool_sha256
                and existing_report.get("score_rows_sha256") == score_rows_sha256
                and existing_report.get("baseline_k90_ph_zip_sha256") == _sha256(inputs["baseline"])
                and existing_report.get("p1_zip_sha256") == _sha256(inputs["p1"])
                and existing_report.get("model_revision") == MODEL_REVISION
                and existing_report.get("rrf_k") == rrf_k
                and existing_report.get("qwen_weight") == qwen_weight
                and existing_report.get("chunk_multiplier") == chunk_multiplier
                and existing_report.get("whole_body") == whole_body
                and existing_report.get("zip_sha256") == _sha256(target)
                and existing_report.get("query_stats_sha256") == _sha256(stat_path)
            ):
                stats_check = pd.read_parquet(stat_path, columns=["query_id"])
                if len(stats_check) == len(rows):
                    return existing_report
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    paths.assert_writable(temp)
    paths.assert_writable(stat_temp)
    paths.assert_writable(report_temp)
    def write_bundle(body_char_limit: int | None) -> dict[str, Any]:
        stats = []
        source_counts = {"p1_row": 0, "p1_any": 0, "body": 0, "none": 0}
        if thin_texts is not None:
            source_counts["thin"] = 0
        json_hash = hashlib.sha256()
        json_bytes = 0
        chunks_total = 0
        body_segments: list[str] = []
        with lzma_dictionary(256), zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_LZMA) as archive:
            info = zipfile.ZipInfo(f"{name}.json", date_time=(2026, 10, 19, 0, 0, 0))
            info.compress_type = zipfile.ZIP_LZMA
            with archive.open(info, "w", force_zip64=True) as stream:
                opening = b"[\n"
                stream.write(opening)
                json_hash.update(opening)
                json_bytes += len(opening)
                for index, row in enumerate(rows):
                    qid = int(row["id"])
                    picked = picked_by_q[qid]
                    own = {int(chunk["doc_id"]): chunk["chunk_text"] for chunk in row["relevant_chunks"]}
                    carriers = (
                        carriers_by_q[qid]
                        if carriers_by_q is not None
                        else _carriers(row, picked, own, all_p1_texts, groups, physical, chunk_multiplier, thin_texts)
                    )
                    output_chunks = []
                    thin_chunks = 0
                    for doc_id, corpus_id in carriers:
                        is_thin = _thin_carrier(doc_id, corpus_id, own, all_p1_texts, groups, physical, thin_texts)
                        if is_thin:
                            text = thin_texts[doc_id]
                            source_counts["thin"] += 1
                            thin_chunks += 1
                        elif corpus_id in own:
                            text = own[corpus_id]
                            source_counts["p1_row"] += 1
                        elif corpus_id in all_p1_texts:
                            text = all_p1_texts[corpus_id]
                            source_counts["p1_any"] += 1
                        elif doc_id in extra_texts:
                            text = body_texts.get(doc_id, "")
                            if text.strip():
                                source_counts["body"] += 1
                                body_segments.append(text)
                            else:
                                source_counts["none"] += 1
                                continue
                        else:
                            text = body_texts.get(_canonical_owner(doc_id, groups, physical), "")
                            if text.strip():
                                source_counts["body"] += 1
                                body_segments.append(text)
                            else:
                                source_counts["none"] += 1
                                continue
                        if body_char_limit and not is_thin and (corpus_id not in own and corpus_id not in all_p1_texts):
                            text = text[:body_char_limit]
                        output_chunks.append({"doc_id": int(corpus_id), "chunk_text": text})
                    result = {
                        "id": qid,
                        "relevant_docs": [int(corpus_id) for _, corpus_id in picked],
                        "relevant_chunks": output_chunks,
                    }
                    serialized = json.dumps(
                        result, ensure_ascii=False, separators=(",", ":")
                    ).encode("utf-8")
                    prefix = b"" if index == 0 else b",\n"
                    stream.write(prefix + serialized)
                    json_hash.update(prefix)
                    json_hash.update(serialized)
                    json_bytes += len(prefix) + len(serialized)
                    chunks_total += len(output_chunks)

                    baseline_ids = set(baseline[qid]["relevant_docs"])
                    selected_ids = set(result["relevant_docs"])
                    if len(selected_ids) != len(result["relevant_docs"]):
                        raise ValueError(f"Duplicate output doc IDs for query {qid}")
                    if fixed_k and len(selected_ids) != len(baseline_ids):
                        raise ValueError(f"Output count differs from K90-ph for query {qid}")
                    counts = {"zh": 0, "vi": 0}
                    for doc_id, _ in picked:
                        lang = "vi" if doc_id in extra_texts else doc_lang.get(doc_id)
                        if lang not in counts:
                            raise ValueError(f"Missing/unsupported language for selected doc {doc_id}: {lang}")
                        counts[lang] += 1
                    changed_in = len(selected_ids - baseline_ids)
                    changed_out = len(baseline_ids - selected_ids)
                    stats.append(
                        {
                            "query_id": qid,
                            "K": len(selected_ids),
                            "n_c_p1": int(row["nc"]),
                            "chunks_written": len(output_chunks),
                            "changed_in_vs_k90_ph": changed_in,
                            "changed_out_vs_k90_ph": changed_out,
                            "changed_symmetric_vs_k90_ph": changed_in + changed_out,
                            "zh_docs": counts["zh"],
                            "vi_docs": counts["vi"],
                            "zh_share": counts["zh"] / len(selected_ids) if selected_ids else 0.0,
                            "no_chunk_docs": sum(
                                _canonical_owner(doc_id, groups, physical) not in groups for doc_id, _ in picked
                            ),
                            "thin_docs": sum(doc_id in thin for doc_id, _ in picked),
                        }
                    )
                    if thin_texts is not None:
                        stats[-1]["thin_chunks"] = thin_chunks
                closing = b"\n]\n"
                stream.write(closing)
                json_hash.update(closing)
                json_bytes += len(closing)
        return {
            "stats": stats,
            "source_counts": source_counts,
            "json_sha256": json_hash.hexdigest(),
            "json_bytes": json_bytes,
            "chunks_total": chunks_total,
            "zip_bytes": temp.stat().st_size,
            "body_segments": body_segments,
            "body_char_limit": body_char_limit,
        }

    built = write_bundle(None)
    size_tries = [{"body_cut_chars": None, "zip_bytes": built["zip_bytes"]}]
    if built["zip_bytes"] >= 102_600_000 and whole_body:
        raise ValueError(
            f"{name} ZIP exceeds 102,600,000-byte ceiling with whole-body chunks: "
            f"{built['zip_bytes']}; left at {temp}"
        )
    if built["zip_bytes"] >= 102_600_000 and built["body_segments"]:
        body_segments = built["body_segments"]
        nb_full = sum(len(text.encode("utf-8")) for text in body_segments)

        def nbytes(n: int) -> int:
            return sum(len(text[:n].encode("utf-8")) for text in body_segments)

        def n_for(target: float) -> int:
            lo, hi = 1, max(len(text) for text in body_segments)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if nbytes(mid) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        points = [(nb_full, built["zip_bytes"])]
        slope = built["zip_bytes"] / built["json_bytes"]
        best: tuple[int, dict[str, Any]] | None = None
        while len(size_tries) < 9:
            x1, y1 = points[-1]
            target_bytes = x1 - (y1 - 102_600_000 * 0.997) / slope
            body_cut = n_for(max(target_bytes, 0))
            if any(item["body_cut_chars"] == body_cut for item in size_tries):
                break
            attempt = write_bundle(body_cut)
            size_tries.append({"body_cut_chars": body_cut, "zip_bytes": attempt["zip_bytes"]})
            points.append((nbytes(body_cut), attempt["zip_bytes"]))
            if attempt["zip_bytes"] < 102_600_000 and (
                best is None or body_cut > best[0]
            ):
                best = (body_cut, attempt)
            (xa, ya), (xb, yb) = points[-2], points[-1]
            if xb != xa and yb != ya:
                slope = (yb - ya) / (xb - xa)
            if best and 102_600_000 - best[1]["zip_bytes"] < 102_600_000 * 0.005:
                break
        if best is None:
            raise ValueError(
                f"{name} ZIP exceeds 102,600,000-byte ceiling after body fallback attempts: "
                f"{size_tries}; left at {temp}"
            )
        built = write_bundle(best[0])
    elif built["zip_bytes"] >= 102_600_000:
        raise ValueError(f"{name} ZIP exceeds 102,600,000-byte ceiling without body chunks: {built['zip_bytes']}; left at {temp}")
    if built["zip_bytes"] >= 102_600_000:
        raise ValueError(f"{name} ZIP exceeds 102,600,000-byte ceiling: {built['zip_bytes']}; left at {temp}")
    stats = built["stats"]
    source_counts = built["source_counts"]
    json_bytes = built["json_bytes"]
    chunks_total = built["chunks_total"]
    zip_size = built["zip_bytes"]
    body_segments = built["body_segments"]
    body_cut_chars = built["body_char_limit"]
    body_cut_chunks = sum(len(text) > body_cut_chars for text in body_segments) if body_cut_chars else 0
    body_cut_chars_removed = (
        sum(len(text) - body_cut_chars for text in body_segments if len(text) > body_cut_chars)
        if body_cut_chars
        else 0
    )
    zip_hash = _sha256(temp)
    stat_frame = pd.DataFrame(stats)
    stat_frame.to_parquet(stat_temp, index=False)
    stat_hash = _sha256(stat_temp)

    total_docs = int(stat_frame.K.sum())
    summary = {
        "name": name,
        "status": "built_cpu_not_submitted",
        "mode": "qfilter" if name == "Q-filter" else "qrrf",
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "instructions": INSTRUCTIONS,
        "document_token_limit": DOC_TOKEN_LIMIT,
        "pool_multiplier": candidate_pool_multiplier,
        "pool_rule": f"round({candidate_pool_multiplier:g} * len(P1 relevant_docs))",
        "keep_rule": "exact len(relevant_docs) from K90-ph ZIP per query" if fixed_k
        else "per-query count chosen by the caller (total and rule in its own sidecar)",
        "rrf_k": rrf_k,
        "qwen_weight": qwen_weight,
        "chunks_rule": "round(n_c of P1 row * chunk_multiplier), S26 selected-doc carrier order"
        if chunk_selector is None else "caller chunk selector (rule in its own sidecar)",
        "chunk_multiplier": chunk_multiplier,
        "whole_body": whole_body,
        "queries": len(rows),
        "docs_total": total_docs,
        "chunks_total": chunks_total,
        "n_c_p1_mean": float(stat_frame.n_c_p1.mean()),
        "chunks_written_mean": float(stat_frame.chunks_written.mean()),
        "chunk_source": source_counts,
        "mean_changed_docs_symmetric_difference_vs_k90_ph": float(
            stat_frame.changed_symmetric_vs_k90_ph.mean()
        ),
        "mean_doc_id_symmetric_difference_vs_QRRF_nc140_ph": mean_doc_id_symdiff_vs_primary,
        "mean_replacements_vs_k90_ph": float(stat_frame.changed_in_vs_k90_ph.mean()),
        "mean_added_vs_k90_ph": float(stat_frame.changed_in_vs_k90_ph.mean()),
        "mean_removed_vs_k90_ph": float(stat_frame.changed_out_vs_k90_ph.mean()),
        "vi_docs": int(stat_frame.vi_docs.sum()),
        "zh_docs": int(stat_frame.zh_docs.sum()),
        "vi_share": float(stat_frame.vi_docs.sum() / total_docs) if total_docs else 0.0,
        "zh_share": float(stat_frame.zh_docs.sum() / total_docs) if total_docs else 0.0,
        "zip_bytes": zip_size,
        "body_cut_chars": body_cut_chars,
        "body_cut_chunks": body_cut_chunks,
        "body_cut_chars_removed": body_cut_chars_removed,
        "size_tries": size_tries,
        "json_bytes": json_bytes,
        "json_sha256": built["json_sha256"],
        "zip_sha256": zip_hash,
        "candidate_pool_sha256": candidate_pool_sha256,
        "score_rows_sha256": score_rows_sha256,
        "query_stats_sha256": stat_hash,
        "zip": str(target),
        "query_stats": str(stat_path),
        "baseline_k90_ph_zip_sha256": _sha256(inputs["baseline"]),
        "p1_zip_sha256": _sha256(inputs["p1"]),
        "built_seconds": round(time.monotonic() - started, 1),
    }
    with report_temp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    temp.replace(target)
    stat_temp.replace(stat_path)
    report_temp.replace(report_path)
    return summary


_first_chunk_by_id: dict[int, str] = {}


_VARIANTS: dict[str, tuple[str, int | None, float | None]] = {
    "Q-filter": ("filter", None, None),
    "Q-rrf": ("rrf_legacy", RRF_K, None),
    "QRRF-nc140-ph": ("rrf", 60, 0.5),
    "QRRF-k20": ("rrf", 20, 0.5),
    "QRRF-k200": ("rrf", 200, 0.5),
    "QRRF-wq0.65": ("rrf", 60, 0.65),
    "QRRF-wq0.35": ("rrf", 60, 0.35),
    "QRRF-pool16": ("rrf", 60, 0.5),
}


def build(
    out: Path,
    variants: tuple[str, ...] = ("Q-filter", "Q-rrf"),
    chunk_multiplier: float | None = None,
    whole_body: bool | None = None,
) -> list[dict[str, Any]]:
    if not variants or set(variants) - set(_VARIANTS):
        raise ValueError(f"Unsupported S29 build variants: {variants}")
    if chunk_multiplier is not None and (not np.isfinite(chunk_multiplier) or chunk_multiplier < 0):
        raise ValueError("chunk_multiplier must be finite and nonnegative")
    inputs = _inputs()
    _preflight_outputs(out, variants)
    for key in ("p1", "baseline", "baseline_build", "canonical", "extra_docs"):
        paths.require_inputs(inputs[key])
    p1_rows = _load_submission(inputs["p1"], with_chunks=True)
    baseline_rows = _load_submission(inputs["baseline"])
    p1_by_id = {int(row["id"]): row for row in p1_rows}
    baseline = {int(row["id"]): row for row in baseline_rows}
    if set(p1_by_id) != set(baseline):
        raise ValueError("P1 and K90-ph query IDs differ")

    _first_chunk_by_id.clear()
    for row in p1_rows:
        for chunk in row["relevant_chunks"]:
            _first_chunk_by_id.setdefault(int(chunk["doc_id"]), str(chunk["chunk_text"]))

    chunks_path = inputs["canonical"] / "chunks_t256.parquet"
    docs_path = inputs["canonical"] / "docs.parquet"
    groups, cid_to_doc = _load_aliases(chunks_path, docs_path)
    docs = pq.read_table(docs_path, columns=["doc_id", "lang", "quality"]).to_pandas()
    doc_lang = dict(zip(docs.doc_id.astype(int).tolist(), docs.lang.tolist()))
    thin = set(docs.loc[docs.quality == "thin", "doc_id"].astype(int).tolist())
    extra_texts, extra_spans = _load_extra_docs(inputs)
    # S26 P-A4 treats each Pharmacity handoff document as its own chunk owner.
    for doc_id in extra_texts:
        groups[doc_id] = [doc_id]
        cid_to_doc[doc_id] = doc_id
        doc_lang[doc_id] = "vi"
    p1_any, row_ids = _row_id_maps(p1_rows, cid_to_doc)

    candidate_path = out / "candidate_pool.parquet"
    score_db = out / "qwen_scores.sqlite"
    prepare_path = out / "prepare.json"
    for path in (candidate_path, score_db, prepare_path):
        paths.require_inputs(path)
    prepare_report = json.loads(prepare_path.read_text(encoding="utf-8"))
    candidate_digest = _sha256(candidate_path)
    _validate_candidate_pool_sha(prepare_report, candidate_digest)
    pool_multiplier = float(prepare_report.get("pool_multiplier", 1.3))
    if "QRRF-pool16" in variants and not np.isclose(pool_multiplier, 1.6):
        raise ValueError(f"QRRF-pool16 requires a 1.6x pool, got {pool_multiplier}")
    if any(name.startswith("QRRF-") and name != "QRRF-pool16" for name in variants) and not np.isclose(
        pool_multiplier, 1.3
    ):
        raise ValueError(f"Requested CPU sweeps require a 1.3x score pool, got {pool_multiplier}")
    candidate = pd.read_parquet(candidate_path)
    if int(prepare_report.get("pool_pairs", -1)) != len(candidate):
        raise ValueError("Prepare report pair count differs from the candidate pool")
    if candidate.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Candidate pool has duplicate query/document pairs")
    scores = _read_full_scores(
        out,
        candidate.to_dict("records"),
        candidate_path,
    )
    score_digest = _score_rows_sha256(scores)
    if set(scores) != set(p1_by_id):
        raise ValueError("Qwen score query IDs do not match the P1/K90-ph inputs")
    chosen_by_variant: dict[str, dict[int, list[dict[str, Any]]]] = {name: {} for name in variants}
    candidate_counts = candidate.groupby("query_id").size().to_dict()
    variant_settings = {name: _VARIANTS[name] for name in variants}
    for row in p1_rows:
        qid = int(row["id"])
        options = scores[qid]
        keep = len(baseline[qid]["relevant_docs"])
        if len(options) != int(candidate_counts.get(qid, 0)):
            raise ValueError(f"Score count differs from candidate pool for query {qid}")
        for name, (mode, rrf_k, qwen_weight) in variant_settings.items():
            if mode == "filter":
                chosen_by_variant[name][qid] = select_qfilter(options, keep)
            elif mode == "rrf_legacy":
                chosen_by_variant[name][qid] = select_qrrf(options, keep, int(rrf_k))
            else:
                chosen_by_variant[name][qid] = select_weighted_qrrf(
                    options, keep, int(rrf_k), float(qwen_weight)
                )

    primary = chosen_by_variant.get("QRRF-nc140-ph")
    difference_means: dict[str, float | None] = {}
    for name, selected_by_q in chosen_by_variant.items():
        if primary is None:
            difference_means[name] = None
            continue
        differences = []
        for qid, chosen in selected_by_q.items():
            selected_ids = {int(row["doc_id"]) for row in chosen}
            primary_ids = {int(row["doc_id"]) for row in primary[qid]}
            differences.append(len(selected_ids.symmetric_difference(primary_ids)))
        difference_means[name] = float(np.mean(differences))

    summaries = []
    for name in variants:
        variant_chunk_multiplier, variant_whole_body = _resolve_variant_options(
            name, chunk_multiplier, whole_body
        )
        summary = _build_variant(
                out,
                name,
                p1_rows,
                baseline,
                p1_any,
                _first_chunk_by_id,
                row_ids,
                groups,
                cid_to_doc,
                doc_lang,
                thin,
                inputs["canonical"],
                extra_texts,
                extra_spans,
                chosen_by_variant[name],
                candidate_digest,
                pool_multiplier,
                score_digest,
                variant_settings[name][1],
                variant_settings[name][2],
                variant_chunk_multiplier,
                variant_whole_body,
                difference_means[name],
            )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summaries
