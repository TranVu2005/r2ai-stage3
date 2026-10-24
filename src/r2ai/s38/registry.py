"""S38 body sources outside the canonical bundle: the S37 120ask segments (S35, 24/10).

The registry is ``out/runs/s37/pairs_new.parquet`` (``doc_id`` -> ``segment``); each segment is a canonical-schema directory under
``data/s37_120ask/<segment>/`` (``docs.parquet`` with ``doc_text``, ``chunks_t256.parquet`` with the t256 spans, ``cmax_top3.parquet``).
``r2ai.s38.compose`` asks this module for the body of a registry doc exactly where it asks the canonical bundle:

* whole-body chunk (S29 rule): the answer span, else the body span of the doc_text (``window._body_span``);
* best t256 chunk for the 768 window (CH-t256-1 rule): the S37 Qwen3-Reranker-0.6B chunk score of the pair
  (``out/runs/s37/cmax*/``, the same rows the C-max uses), ties by file order; without a chunk score the top-1 chunk by chunk cosine of
  ``cmax_top3.parquet``, else the first answer/body chunk.

A registry doc never reads the canonical docs (ids are disjoint); nothing here writes.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.index.chunk import SUBMITTABLE

RUN = paths.RUNS_DIR / "s37"
REGISTRY = RUN / "pairs_new.parquet"
REGISTRY_V2 = RUN / "pairs_new_final_v2.parquet"                                   # the crawl of 25/10 (superset of pairs_new)
S36_PAIRS = paths.RUNS_DIR / "s36-fuse4" / "d300"                                  # pairs_d300 / pairs_final: 120ask rows have a segment
STORE = paths.DATA_DIR / "s37_120ask"
SHARD = "pool-93001.parquet"
SCORES = Path("gpu") / "qwen3-reranker-0.6b" / "scores-all.parquet"


def registry_files() -> list[Path]:
    """The registries that exist, later files overriding earlier ones: ``pairs_new``, ``pairs_new_final_v2`` and the pairs the S36 D=300 export
    wrote (``pairs_d300``, ``pairs_final``); only their rows with a segment (120ask docs) count."""
    return [f for f in (REGISTRY, REGISTRY_V2, S36_PAIRS / "pairs_d300.parquet", S36_PAIRS / "pairs_final.parquet") if f.exists()]


def registry_pairs(registry: Path | None = None, columns: tuple[str, ...] = ("pair_id", "query_id", "doc_id", "segment")) -> pd.DataFrame:
    """Rows of the registry (``registry`` alone, else every file of ``registry_files``) that belong to a 120ask segment."""
    files = [registry] if registry else registry_files()
    frames = [pq.read_table(f, columns=list(columns)).to_pandas() for f in files if f.exists()]
    if not frames:
        return pd.DataFrame({c: [] for c in columns})
    table = pd.concat(frames, ignore_index=True)
    return table[table.segment.notna() & (table.segment != "")]


def registry_index(registry: Path | None = None) -> dict[int, str]:
    """doc_id -> segment name of every doc of the S37 registries (empty when none exists)."""
    table = registry_pairs(registry, ("doc_id", "segment")).drop_duplicates("doc_id", keep="last")
    return dict(zip(table.doc_id.astype(int), table.segment))


def _by_segment(index: dict[int, str], ids) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for d in sorted(set(int(i) for i in ids)):
        if d in index:
            out.setdefault(index[d], []).append(d)
    return out


def load_docs(index: dict[int, str], ids, store: Path | None = None) -> pd.DataFrame:
    """``doc_id, lang, doc_text`` of the registry docs among ``ids``."""
    store = store or STORE
    frames = [pq.read_table(store / seg / "docs.parquet", columns=["doc_id", "lang", "doc_text"],
                            filters=[("doc_id", "in", docs)]).to_pandas() for seg, docs in _by_segment(index, ids).items()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"doc_id": [], "lang": [], "doc_text": []})


def load_t256(index: dict[int, str], ids, store: Path | None = None) -> pd.DataFrame:
    """Answer/body t256 rows of the registry docs among ``ids`` in file order (``file_row`` = running index in the segment file)."""
    store = store or STORE
    frames = []
    for seg, docs in _by_segment(index, ids).items():
        table = pq.read_table(store / seg / "chunks_t256.parquet", columns=["chunk_id", "doc_id", "field", "char_start", "char_end", "text"]).to_pandas()
        table["file_row"] = np.arange(len(table))
        table = table[table.doc_id.isin(docs) & table.field.isin(SUBMITTABLE)]
        table["segment"] = seg
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        {"chunk_id": [], "doc_id": [], "field": [], "char_start": [], "char_end": [], "text": [], "file_row": [], "segment": []})


def chunk_scores(run: Path | None = None) -> dict[tuple[int, int, int], float]:
    """(query, doc, chunk_id) -> Qwen3-Reranker-0.6B P(yes) of the S37 C-max chunk rows (every ``cmax*`` run that has scores)."""
    run = run or RUN
    out: dict[tuple[int, int, int], float] = {}
    for directory in sorted(p for p in run.glob("cmax*") if p.is_dir()):
        shard, scores = directory / "pool" / "shards" / SHARD, directory / SCORES
        if not (shard.exists() and scores.exists()):
            continue
        rows = pq.read_table(shard, columns=["pair_id", "query_id", "doc_id", "chunk_id"]).to_pandas().dropna(subset=["chunk_id"])
        joined = rows.merge(pq.read_table(scores, columns=["pair_id", "score"]).to_pandas(), on="pair_id")
        out.update(zip(zip(joined.query_id.astype(int), joined.doc_id.astype(int), joined.chunk_id.astype(int)), joined.score.astype(float)))
    return out


def top_chunk_by_cosine(index: dict[int, str], pairs: pd.DataFrame, registry: Path | None = None, store: Path | None = None
                        ) -> dict[tuple[int, int], int]:
    """(query, doc) -> chunk_id of the rank-1 chunk of ``cmax_top3.parquet`` of the pair's segment."""
    store = store or STORE
    reg = registry_pairs(registry).drop_duplicates(["query_id", "doc_id"], keep="last")
    reg = reg.merge(pairs[["query_id", "doc_id"]], on=["query_id", "doc_id"])
    out: dict[tuple[int, int], int] = {}
    for seg, g in reg.groupby("segment"):
        top = pq.read_table(store / seg / "cmax_top3.parquet", columns=["pair_id", "rank", "chunk_id"]).to_pandas()
        top = top[top["rank"] == top.groupby("pair_id")["rank"].transform("min")].merge(g, on="pair_id")
        out.update(zip(zip(top.query_id.astype(int), top.doc_id.astype(int)), top.chunk_id.astype(int)))
    return out


def best_spans(pairs: pd.DataFrame, text: dict[int, str], index: dict[int, str], choose, scores: dict | None = None,
               registry: Path | None = None, store: Path | None = None
               ) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[int, pd.DataFrame]]:
    """((query, doc) -> char span of the best answer/body t256 chunk in the doc's text, doc -> its answer/body chunks) for the registry
    docs among ``pairs``. A doc whose chunk spans do not index its text verbatim is left out (as for the canonical docs)."""
    docs = set(pairs.doc_id.astype(int)) & set(index)
    chunks = load_t256(index, docs, store)
    pieces: dict[int, pd.DataFrame] = {}
    for d, g in chunks.groupby("doc_id", sort=False):
        g = g.sort_values("file_row")
        t = text[int(d)]
        if all(t[s:e] == x for s, e, x in zip(g.char_start.astype(int), g.char_end.astype(int), g.text)):
            pieces[int(d)] = g.reset_index(drop=True).assign(idx=np.arange(len(g)))
    scores = chunk_scores() if scores is None else scores
    sub = pairs[pairs.doc_id.astype(int).isin(pieces)]
    cosine_top = top_chunk_by_cosine(index, sub, registry, store) if len(sub) and (registry.exists() if registry else registry_files()) else {}
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for q, d in zip(sub.query_id.astype(int), sub.doc_id.astype(int)):
        g = pieces[d]
        cids = g.chunk_id.astype(int).tolist()
        judged = [(k, scores[(q, d, c)]) for k, c in enumerate(cids) if (q, d, c) in scores]
        if judged:
            keep, vals = [k for k, _ in judged], [v for _, v in judged]
            pick = keep[choose(vals, [int(g.idx.iat[k]) for k in keep], 1)[0]]
        elif cosine_top.get((q, d)) in cids:
            pick = cids.index(cosine_top[(q, d)])
        else:
            pick = 0
        out[(q, d)] = (int(g.char_start.iat[pick]), int(g.char_end.iat[pick]))
    return out, pieces


def body_chunk_spans(index: dict[int, str], ids, store: Path | None = None) -> dict[int, pd.DataFrame]:
    """doc -> answer/body t256 rows (``field, char_start, char_end``) for ``window._body_span``."""
    chunks = load_t256(index, ids, store)
    return {int(d): g for d, g in chunks.groupby("doc_id")}


def validator_extra(target: Path, index: dict[int, str], ids, store: Path | None = None) -> Path | None:
    """Extra verbatim source of ``scripts/s26/validate_pa3.py`` for the registry docs among ``ids``: ``target/lots/<segment>/chunks/docs.parquet``
    (``doc_id, doc_text``, the layout of the S15 lots). None when no id is a registry doc. Rewritten on every call."""
    store = store or STORE
    groups = _by_segment(index, ids)
    if not groups:
        return None
    paths.assert_writable(target)
    shutil.rmtree(target / "lots", ignore_errors=True)
    for seg, docs in groups.items():
        dest = target / "lots" / seg / "chunks" / "docs.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pq.read_table(store / seg / "docs.parquet", columns=["doc_id", "doc_text"], filters=[("doc_id", "in", docs)]), dest)
    return target


def describe(index: dict[int, str]) -> dict[str, Any]:
    segments = sorted(set(index.values()))
    return {"registries": [str(f) for f in registry_files()], "docs": len(index), "segments": len(segments)}
