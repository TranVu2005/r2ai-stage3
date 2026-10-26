"""EF2-free on the S37 data (120ask docs, Jina rescored): docsel + keep in one command (S35, 24/10). Nothing is run by importing this.

``python -m r2ai.s36.s37 apply [--config cfg.json] [--jina scores-all.parquet] [--scores-4b p ...] [--cmax p ...] [--name EF2-free-s37]``
    pool = cosine top-211 (+ adaptive deepening) of the S36 scored union plus the new S37 pairs; RRF of the four sources; EF2 cut with
    the global parameters of ``cal/cal.json`` / ``cal/ef2-free.json`` ((a, b) and s_free, nothing refitted); carriers by the chunk rule
    of the config; writes ``docsel-<name>.parquet`` (the S36 docsel schema), ``keep-<name>.json``, ``pcal-<name>.parquet``,
    ``carriers-<name>.parquet`` and ``<name>.apply.json`` (SHA256 of every input and output) under ``--out``.
    The S38 compose then builds the ZIP from the docsel alone: ``compose run --docsel docsel-<name>.parquet --rule full+w768``
    (``--carriers carriers-<name>.parquet`` for a rule other than the threshold one).
``python -m r2ai.s36.s37 jina-missing``
    ``cal/jina-missing-pairs.parquet``: the S36 supplement Jina pairs of the queries that S37 rescores, which the S37 Jina pool does not
    hold (Vu 24/10: neither ranked last nor kept; S37 adds them to its pool). ``apply`` stops when any pair of a pool has no score in a source
    and writes the missing pairs to ``missing-<name>-<source>.parquet``.

Config (JSON, every key optional): ``{"name": "EF2-free-s37", "pairs_new": ".../pairs_new.parquet" | [".../pairs_new_final_v2.parquet",
".../d300/pairs_final.parquet"], "scores_4b": [...], "cmax": [...], "jina": ".../scores-all.parquet", "t_c": 0.2030746281894702,
"carriers": {"rule": "threshold" | "fair" | "swap", "target": N, "drop": N}, "pool": "free" | "d211" | "fixed" | "pool13" | "cosine13", "depth": 300, "out": "...", "a": .., "b": .., "s_free": ..}`` (a / b / s_free: S40 recal override of cal.json / ef2-free.json).
Defaults are the S37 run layout (``out/runs/s37``) plus the 4B scores of the subsets exported by ``r2ai.s36.d300`` (``out/runs/s36-fuse4/d300``).
``pairs_new`` may list several registries (``pairs_new_final_v2`` and the D=300 pairs): a pair in two of them counts once, the copy with a 4B score
first. ``pool`` ``fixed``: every query's pool is its cosine top-``depth`` of the merged union (all pairs of it need a 4B score; checked against the
cosine ranking of ``out/runs/s37/pool.npz`` + the 120ask vectors and reported); G_q keeps the top-211 definition.
``pool`` ``pool13``: the EF2-cut-p4b frame, each pool = the pool13 docs plus the new 120ask docs (J3 sources cosine / 4B / Jina, G_q = s_thrC x
(sum p_cal over the pool + tail), no C-max in the fusion).
``pool`` ``cosine13``: the EF2-cut-p4b frame on the corpus with the new docs: pool = cosine top-n_q of the merged union (n_q = the old pool13 size, round(1.3 K_q)), four sources
(cosine / 4B / C-max / Jina), same a, b and s_thrC, EF2 cut on the calibrated P4B.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.s29.qwen_judge import _atomic_json, _sha256
from r2ai.s31 import build as s31_build
from r2ai.s31.select import doc_set_diff
from r2ai.s36 import carriers as carrier_rules
from r2ai.s36 import cal, fuse

RUN37 = paths.RUNS_DIR / "s37"
DEFAULT_NAME = "EF2-free-s37"
FUSE4_FREE_T_C = 0.2030746281894702   # r2ai.s36.build.FUSE4_FREE_T_C (global, calibrated once)


class MissingScores(SystemExit):
    """A source has no score for some pool pairs; the pairs were written to ``path``."""


def default_inputs() -> dict[str, Any]:
    return {"pairs_new": str(RUN37 / "pairs_new.parquet"),
            "scores_4b": sorted(glob.glob(str(RUN37 / "kaggle-4b" / "*" / "gpu" / "qwen3-reranker-4b" / "scores-extra.parquet")))
            + sorted(glob.glob(str(fuse.run_root() / "d300" / "*" / "gpu" / "qwen3-reranker-4b" / "scores-extra.parquet"))),
            "cmax": sorted(glob.glob(str(RUN37 / "cmax_scores*.parquet"))),
            "jina": str(RUN37 / "jina" / "scores-all.parquet")}


# ---------------------------------------------------------------------------------------------------------- union

def merge_union(union: dict[int, list[dict[str, Any]]], pairs: pd.DataFrame, scores_4b: dict[int, float],
                min_coverage: float = 1.0) -> dict[int, list[dict[str, Any]]]:
    """The scored union of S36 plus the S37 pairs (4B score from ``scores_4b`` by ``pair_id``), cosine order, ``cosine_rank`` renumbered.
    A new pair whose doc is already in the query's union is dropped. Pairs without a 4B score are dropped (they cannot be judged) when
    at least ``min_coverage`` of the pairs are scored, else the call refuses."""
    scored = pairs.pair_id.astype(int).isin(scores_4b)
    unscored = pairs[~scored]
    if len(unscored):
        if scored.mean() < min_coverage:
            raise ValueError(f"{len(unscored)} S37 pairs have no 4B score ({scored.mean():.4f} covered < {min_coverage}), "
                             f"e.g. {unscored.pair_id.head(3).tolist()}")
        pairs = pairs[scored]
    new: dict[int, list[dict[str, Any]]] = {}
    for q, d, pid, cos in zip(pairs.query_id.astype(int), pairs.doc_id.astype(int), pairs.pair_id.astype(int), pairs.cosine_score.astype(float)):
        new.setdefault(q, []).append({"pair_id": pid, "doc_id": d, "cosine_score": cos, "judge_score": float(scores_4b[pid]),
                                      "origin": "s37", "order": (3, pid)})
    out: dict[int, list[dict[str, Any]]] = {}
    for q, rows in union.items():
        rows = [dict(r) for r in rows]
        have = {int(r["doc_id"]) for r in rows}
        rows += [r for r in new.get(q, []) if r["doc_id"] not in have]
        rows.sort(key=lambda r: (-float(r["cosine_score"]), tuple(r["order"])))
        for rank, r in enumerate(rows, 1):
            r["cosine_rank"] = rank
        out[q] = rows
    extra = set(new) - set(union)
    if extra:
        raise ValueError(f"S37 pairs for queries outside the union: {sorted(extra)[:5]}")
    return out


def read_pairs(files: Sequence[str | Path], scores_4b: dict[int, float] | None = None) -> pd.DataFrame:
    """``pair_id, query_id, doc_id, cosine_score`` over several registries. One ``pair_id`` must mean one pair; a (query, doc) in two registries
    counts once, the copy whose ``pair_id`` has a 4B score first (else the first registry)."""
    frame = pd.concat([pd.read_parquet(f, columns=["pair_id", "query_id", "doc_id", "cosine_score"]) for f in files], ignore_index=True)
    frame[["pair_id", "query_id", "doc_id"]] = frame[["pair_id", "query_id", "doc_id"]].astype(int)
    same_id = frame.drop_duplicates(["pair_id", "query_id", "doc_id"]).pair_id
    if same_id.duplicated().any():
        raise ValueError(f"pair_id {int(same_id[same_id.duplicated()].iloc[0])} is two different pairs across {len(files)} registries")
    if scores_4b is not None:
        frame = frame.assign(_unscored=~frame.pair_id.isin(scores_4b)).sort_values("_unscored", kind="stable").drop(columns="_unscored")
    return frame.drop_duplicates(["query_id", "doc_id"], keep="first").sort_values("pair_id").reset_index(drop=True)


def fixed_pools(union: dict[int, list[dict[str, Any]]], depth: int, order: pd.DataFrame | None = None
                ) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Every pool = the cosine top-``depth`` rows of the merged union (no adaptive deepening, no K_q).

    Many docs share exactly the same cosine (duplicate content under several doc_ids), and a tie block can straddle the ``depth`` cut: the union
    order (cosine, origin) and the corpus order (``r2ai.s36.d300.merged_top``: canonical first, then the position in pool.npz) then keep different
    members. With ``order`` (``merged_top(depth)``: ``query_id, doc_id, cosine_rank``) the pool is exactly the corpus top-``depth`` in the corpus
    order; every doc of it must be in the union (scored by 4B)."""
    pools, ties = {}, 0
    for q, rows in union.items():
        if len(rows) < depth:
            raise ValueError(f"Query {q}: only {len(rows)} scored pairs < depth {depth}")
        if order is None:
            pools[q] = s31_build._with_judge_rank([dict(r) for r in rows[:depth]])
            continue
        by_doc = {int(r["doc_id"]): r for r in rows}
        want = order.doc_id[order.query_id == q].astype(int).tolist() if isinstance(order, pd.DataFrame) else order[q]
        missing = [d for d in want if d not in by_doc]
        if missing:
            raise ValueError(f"Query {q}: {len(missing)} docs of the corpus top-{depth} have no 4B score, e.g. {missing[:3]}")
        chosen = [dict(by_doc[d]) for d in want]
        ties += sum(1 for d, r in zip(want, rows[:depth]) if d != int(r["doc_id"]))
        for rank, r in enumerate(chosen, 1):
            r["cosine_rank"] = rank
        pools[q] = s31_build._with_judge_rank(chosen)
    stuck = sum(1 for rows in pools.values() if rows[-1]["judge_score"] >= s31_build.T_FREE)
    return pools, {"mode": "fixed", "depth": depth, "order": "corpus" if order is not None else "union",
                   "queries_bottom_doc_p4b_at_or_above_t_free": int(stuck)}


def pool13_plus(ranked: dict[int, list[dict[str, Any]]], union: dict[int, list[dict[str, Any]]]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """EF2-cut-p4b frame with the new 120ask docs: every query's pool = its pool13 docs plus the docs the S37 registries added (``origin`` s37, 4B
    scored), rows taken from the merged ``union`` so ``cosine_rank`` counts the new docs too."""
    pools, added, lost = {}, 0, 0
    for q, rows in union.items():
        base = {int(r["doc_id"]) for r in ranked.get(q, [])}
        keep = [dict(r) for r in rows if int(r["doc_id"]) in base or r.get("origin") == "s37"]
        lost += len(base - {int(r["doc_id"]) for r in keep})
        added += sum(1 for r in keep if int(r["doc_id"]) not in base)
        pools[q] = s31_build._with_judge_rank(keep)
    if lost:
        raise ValueError(f"{lost} pool13 docs are not in the scored union")
    return pools, {"mode": "pool13+s37", "pool13_docs": int(sum(len(v) for v in ranked.values())), "docs_added": int(added)}


def cosine13_pools(ranked: dict[int, list[dict[str, Any]]], union: dict[int, list[dict[str, Any]]]
                   ) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """EF2-cut-p4b frame on the corpus with the new docs: every pool = the cosine top-``n_q`` of the merged ``union``, ``n_q`` = the size of the
    query's old pool13 (``round(1.3 K_q)`` of the S29 / S31 pool rule; the old pool13 is exactly the cosine top-``n_q`` of the S36 union, so the
    new docs push out the lowest-cosine old docs). ``ranked`` is also used to report how many docs the new docs replaced."""
    pools, replaced, queries = {}, 0, 0
    for q, rows in union.items():
        pools[q] = s31_build._with_judge_rank([dict(r) for r in rows[: len(ranked[q])]])
        old = {int(r["doc_id"]) for r in ranked.get(q, [])}
        gone = len(old - {int(r["doc_id"]) for r in pools[q]})
        replaced, queries = replaced + gone, queries + bool(gone)
    return pools, {"mode": "cosine13", "pool13_docs_new": int(sum(len(v) for v in pools.values())), "pool13_docs_old": int(sum(len(v) for v in ranked.values())),
                   "old_pool13_docs_pushed_out": int(replaced), "queries_with_docs_pushed_out": int(queries)}


def check_fixed_pools(pools: dict[int, list[dict[str, Any]]], depth: int, top: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compare the pools with the cosine top-``depth`` of the corpus (canonical + Pharmacity + 120ask vectors): queries and docs that differ."""
    from r2ai.s36 import d300

    top = d300.merged_top(depth) if top is None else top
    want = {int(q): set(g.doc_id.astype(int)) for q, g in top.groupby("query_id")}
    diff = {q: len(want[q] ^ {int(r["doc_id"]) for r in rows}) for q, rows in pools.items()}
    return {"queries_differing": int(sum(1 for v in diff.values() if v)), "docs_differing": int(sum(diff.values()))}


def assemble_jina(base: dict[tuple[int, int], float], s37: dict[tuple[int, int], float]) -> dict[tuple[int, int], float]:
    """Jina is listwise: for a query S37 rescored, only its S37 scores count (the S35 and S36 scores were given in another block context).
    Queries S37 did not touch keep S35 + S36 scores."""
    rescored = {q for q, _ in s37}
    out = {k: v for k, v in base.items() if k[0] not in rescored}
    out.update(s37)
    return out


def missing_pairs(pools: dict[int, list[dict[str, Any]]], lookup: dict[tuple[int, int], float]) -> pd.DataFrame:
    rows = [(q, int(r["doc_id"]), int(r["cosine_rank"]), float(r["judge_score"])) for q, rs in pools.items() for r in rs
            if (q, int(r["doc_id"])) not in lookup]
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "cosine_rank", "p4b"])


# ------------------------------------------------------------------------------------------------- jina-missing

def jina_missing(jina_pool: Path | None = None, supplement: Path | None = None) -> dict[str, Any]:
    """S36 supplement Jina pairs in the queries S37 rescores that the S37 Jina pool does not contain."""
    jina_pool = jina_pool or RUN37 / "jina" / "pool.parquet"
    supplement = supplement or fuse.run_root() / "extra" / "jina-scores.parquet"
    pool = pd.read_parquet(jina_pool, columns=["query_id", "doc_id"])
    supp = pd.read_parquet(supplement, columns=["query_id", "doc_id", "score"]).rename(columns={"score": "s36_score"})
    rescored = set(pool.query_id.astype(int))
    in_pool = set(zip(pool.query_id.astype(int), pool.doc_id.astype(int)))
    mask = supp.query_id.isin(rescored) & ~pd.Series([(int(q), int(d)) in in_pool for q, d in zip(supp.query_id, supp.doc_id)], index=supp.index)
    missing = supp[mask].sort_values(["query_id", "doc_id"]).reset_index(drop=True)
    target = fuse.run_root() / "cal" / "jina-missing-pairs.parquet"
    paths.assert_writable(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    missing.to_parquet(target, index=False)
    per_query = missing.groupby("query_id").size()
    report = {"path": str(target), "sha256": _sha256(target), "pairs": int(len(missing)), "queries": int(missing.query_id.nunique()),
              "per_query_p10_p50_p90": [float(np.quantile(per_query, q)) for q in (.1, .5, .9)] if len(per_query) else None,
              "s36_supplement_pairs": int(len(supp)), "s36_supplement_pairs_in_unrescored_queries": int((~supp.query_id.isin(rescored)).sum()),
              "rescored_queries": len(rescored), "jina_pool": str(jina_pool), "jina_pool_sha256": _sha256(jina_pool),
              "supplement_sha256": _sha256(supplement)}
    _atomic_json(target.with_suffix(".json"), report)
    return report


# ------------------------------------------------------------------------------------------------------- apply

def load_scores(files: Sequence[str | Path], key: str, value: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for f in files:
        frame = pd.read_parquet(f, columns=[key, value])
        dup = set(frame[key].astype(int)) & set(out)
        if dup:
            raise ValueError(f"{f}: {len(dup)} {key} already scored by another file")
        out.update(zip(frame[key].astype(int), frame[value].astype(float)))
    return out


def apply(cfg: dict[str, Any]) -> dict[str, Any]:
    spec = {**default_inputs(), **{k: v for k, v in cfg.items() if v is not None}}
    last = spec.get("missing", "strict") == "last"          # "last": a pair without a Jina / C-max score takes the last rank of its pool
    min_4b = float(spec.get("min_4b_coverage", 1.0))
    no_jina = last and (spec["jina"] is False or not Path(spec["jina"]).exists())   # no S37 rescoring (``"jina": false`` pins that): the S35 + S36 scores, new docs rank last
    name = spec.get("name", DEFAULT_NAME)
    out = paths.resolve_path(spec["out"]) if spec.get("out") else fuse.run_root() / "s37"
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    t_c = float(spec.get("t_c", FUSE4_FREE_T_C))
    rule = {"rule": "fair", **spec.get("carriers", {})}        # default chunk rule from 24/10: CR-fair with the global rho
    pair_files = [spec["pairs_new"]] if isinstance(spec["pairs_new"], (str, Path)) else list(spec["pairs_new"])
    inputs = {"pairs_new": pair_files, "scores_4b": list(spec["scores_4b"]), "cmax": list(spec["cmax"]),
              "jina": [] if no_jina else [spec["jina"]]}
    for key, files in inputs.items():
        if (not files and not (key == "jina" and no_jina)) or not all(Path(f).exists() for f in files):
            raise FileNotFoundError(f"{key}: {files} (artifact not there yet)")
    ranked, union, _, _, _, _ = fuse.load_frames()
    scores_4b = load_scores(spec["scores_4b"], "pair_id", "score")
    pairs = read_pairs(pair_files, scores_4b)
    coverage_4b = float(pairs.pair_id.astype(int).isin(scores_4b).mean())
    union = merge_union(union, pairs, scores_4b, min_4b)
    pairs = pairs[pairs.pair_id.astype(int).isin(scores_4b)]
    pool_kind = spec.get("pool", "free")                       # "d211": every pool cut at the cosine top-211 (no adaptive deepening)
    if pool_kind == "fixed":                                   # every pool = the cosine top-depth of the merged union
        from r2ai.s36 import d300

        corpus_top = d300.merged_top(int(spec["depth"]))                  # the corpus order also settles the cosine ties at the cut
        pools, pool_report = fixed_pools(union, int(spec["depth"]), corpus_top)
        pool_report["check_vs_corpus"] = check_fixed_pools(pools, int(spec["depth"]), corpus_top)
    elif pool_kind == "cosine13":                              # EF2-cut-p4b on the new corpus: pool = cosine top-round(1.3 K_q), J4 sources
        pools, pool_report = cosine13_pools(ranked, union)
    elif pool_kind == "pool13":                                # EF2-cut-p4b frame: pool13 + the new docs, J3 keys, G_q of the pool13 frame
        pools, pool_report = pool13_plus(ranked, union)
    else:
        pools, _, pool_report = s31_build.free_adaptive(union, deepen=pool_kind != "d211")
    pool_report["pool"] = pool_kind
    # sources: the S36 lookups; Jina rescored per query by S37, C-max pointwise so the S37 pairs are simply added
    jina_base, jina_files = fuse.load_jina(fuse.run_root() / "extra" / "jina-scores.parquet")
    if no_jina:
        s37_jina, jina = pd.DataFrame({"query_id": []}), dict(jina_base)
    else:
        s37_jina = pd.read_parquet(spec["jina"], columns=["query_id", "doc_id", "score"])
        jina = assemble_jina(jina_base, dict(zip(zip(s37_jina.query_id.astype(int), s37_jina.doc_id.astype(int)), s37_jina.score.astype(float))))
    cmax, cmax_files = fuse.load_cmax(fuse.run_root() / "extra" / "cmax-scores.parquet")
    by_pair = dict(zip(pairs.pair_id.astype(int), zip(pairs.query_id.astype(int), pairs.doc_id.astype(int))))
    keyed_files = [f for f in spec["cmax"] if "pair_id" not in pq.read_schema(f).names]       # e.g. cmax_scores-d300.parquet: (query_id, doc_id, cmax)
    for pid, value in load_scores([f for f in spec["cmax"] if f not in keyed_files], "pair_id", "cmax").items():
        if pid in by_pair:
            cmax[by_pair[pid]] = value
    for f in keyed_files:
        frame = pd.read_parquet(f, columns=["query_id", "doc_id", "cmax"])
        for q, d, v in zip(frame.query_id.astype(int), frame.doc_id.astype(int), frame.cmax.astype(float)):
            if (q, d) in cmax:
                raise ValueError(f"{f}: ({q}, {d}) already has a C-max score")
            cmax[(q, d)] = v
    missing = {}
    for source, lookup in (("jina", jina), ("cmax", cmax)):
        gap = missing_pairs(pools, lookup)
        if len(gap):
            target = out / f"missing-{name}-{source}.parquet"
            gap.to_parquet(target, index=False)
            new_docs = set(pairs.doc_id.astype(int))
            missing[source] = {"pairs": int(len(gap)), "queries": int(gap.query_id.nunique()), "path": str(target),
                               "pairs_of_new_docs": int(gap.doc_id.isin(new_docs).sum())}
    if missing and not last:
        raise MissingScores(f"pool pairs without a score: {json.dumps(missing)}")
    fitted = json.loads((cal.cal_root() / "cal.json").read_text(encoding="utf-8"))["variants"]["p4b"]
    a, b = fitted["fit"]["a"], fitted["fit"]["b"]
    if "a" in spec or "b" in spec:                                       # S40 recal: (a, b) refitted on the free-frame LB numbers
        a, b = float(spec["a"]), float(spec["b"])
    rrf_keys = tuple(spec.get("rrf_keys") or fuse.FOUR_KEYS)             # ablations: drop one RRF source (cosine and 4B stay)
    if not set(rrf_keys) <= set(fuse.FOUR_KEYS) or not {"cosine_rank", "judge_rank"} <= set(rrf_keys):
        raise ValueError(f"rrf_keys {rrf_keys}: must be a subset of {fuse.FOUR_KEYS} that keeps cosine_rank and judge_rank")
    if pool_kind in ("pool13", "cosine13"):
        s_free = float(fitted["g_scale"])
        frame = cal.Frame(cal.J3_KEYS if pool_kind == "pool13" else rrf_keys, "thrC", data={"ranked": ranked, "pools": pools, "union": union, "jina": jina, "cmax": cmax}, allow_missing=last)
        calibrated = cal.Calibrated(frame, "p4b", a, b, s_free, use_tail=True)
    else:
        s_free = float(spec["s_free"]) if "s_free" in spec else float(json.loads((cal.cal_root() / "ef2-free.json").read_text(encoding="utf-8"))["s_free"])
        frame = cal.Frame(rrf_keys, "free", data={"pools": pools, "union": union, "jina": jina, "cmax": cmax}, allow_missing=last)
        calibrated = cal.Calibrated(frame, "p4b", a, b, s_free, use_tail=True, g_depth=cal.FREE_G_DEPTH)
    keep = cal.ef2_keep(calibrated)
    chosen = cal.chosen_rows(frame, keep)
    perq = pd.DataFrame([{"query_id": q, "k": int(keep[q]), "tp": float(calibrated.p[q][: keep[q]].sum()), "g": float(calibrated.g[q])} for q in frame.qids])
    perq["f2"] = 5 * perq.tp / (4 * perq.g + perq.k)              # predicted F2 of the query; a doc adds to it only if its p_cal > f2 / 5
    perq.to_parquet(out / f"perq-{name}.parquet", index=False)
    sel = pd.DataFrame([(q, int(r["doc_id"]), k, float(r["judge_score"])) for q, rows in chosen.items() for k, r in enumerate(rows, 1)],
                       columns=["query_id", "doc_id", "rank", "p4b"])
    sel["chunk"] = sel.p4b >= t_c                                               # fixed-whole: the global threshold carriers
    carriers, carrier_info = carrier_rules.select_carriers(sel, rule["rule"], rule.get("target"), rule.get("drop"), rule.get("rho"))
    carriers.to_parquet(out / f"carriers-{name}.parquet", index=False)
    chunk_ids = {int(q): g.doc_id.astype(int).tolist() for q, g in carriers.groupby("query_id")}
    pcal_rows = [{"query_id": q, "doc_id": int(r["doc_id"]), "p_fused": float(f), "p_cal": float(p)}
                 for q in frame.qids for r, f, p in zip(frame.order[q], calibrated.fused[q], calibrated.p[q])]
    pcal = pd.DataFrame(pcal_rows)
    pcal.to_parquet(out / f"pcal-{name}.parquet", index=False)
    from r2ai.s36.build import docsel_writer

    docsel_writer(out, "", t_c, {name: pcal})(name, chosen, chunk_ids, {"t_c": t_c})
    _atomic_json(out / f"keep-{name}.json", {str(q): int(k) for q, k in keep.items()})
    previous = fuse.run_root() / "docsel" / "docsel-EF2-free.parquet"
    diff = None
    if previous.exists():
        old = pd.read_parquet(previous, columns=["query_id", "doc_id"]).groupby("query_id").doc_id.apply(list).to_dict()
        diff = doc_set_diff({q: [int(r["doc_id"]) for r in rows] for q, rows in chosen.items()}, old)
    pins = {key: {str(f): _sha256(Path(f)) for f in files} for key, files in inputs.items()}
    report = {"name": name, "inputs": pins, "s36_sources": {"jina": jina_files, "cmax": cmax_files}, "global": {"a": a, "b": b, "s_free": s_free, "t_c": t_c, "p_free": s31_build.P_FREE, "t_free": s31_build.T_FREE},
              "pool": pool_report, "new_pairs": int(len(pairs)), "queries_with_new_pairs": int(pairs.query_id.nunique()),
              "queries_jina_rescored": int(s37_jina.query_id.nunique()), "coverage_4b": coverage_4b,
              "missing_policy": "last" if last else "strict", "missing_scores": missing, "pool_pairs": int(sum(len(v) for v in pools.values())),
              "docs_per_query": cal.keep_stats(keep), "predicted": calibrated.expected(keep), "carriers": {**carrier_info, **carrier_rules.describe(carriers)},
              "vs_EF2-free_docsel": diff,
              "outputs": {p.name: _sha256(p) for p in sorted(out.glob(f"*{name}*")) if p.is_file() and not p.name.endswith(".apply.json")}}
    _atomic_json(out / f"{name}.apply.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("apply", "jina-missing"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--jina", default=None)
    parser.add_argument("--pairs-new", nargs="*", default=None, help="one or several registries (pairs_new_final_v2, d300/pairs_final)")
    parser.add_argument("--scores-4b", nargs="*", default=None)
    parser.add_argument("--cmax", nargs="*", default=None)
    parser.add_argument("--missing", choices=("strict", "last"), default=None, help="pair without a Jina / C-max score: refuse (default) or rank last")
    parser.add_argument("--min-4b-coverage", type=float, default=None, help="share of S37 pairs that must have a 4B score (default 1.0)")
    args = parser.parse_args(argv)
    if args.command == "jina-missing":
        print(json.dumps(jina_missing(), indent=2, ensure_ascii=False))
        return 0
    cfg = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    for key, value in (("name", args.name), ("out", str(args.out) if args.out else None), ("jina", args.jina),
                       ("pairs_new", args.pairs_new), ("scores_4b", args.scores_4b), ("cmax", args.cmax),
                       ("missing", args.missing), ("min_4b_coverage", args.min_4b_coverage)):
        if value is not None and value != "":
            cfg[key] = value
    print(json.dumps(apply(cfg), indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
