"""S38 compose: any doc selection + one S38 chunk rule -> ZIP + validator S26.

``python -m r2ai.s38.compose run --name NAME --base-zip DOCREAD.zip --docsel docsel-X.parquet --rule RULE``

* base = the whole-body bundle of the doc selection (S36 ``-docread`` ZIP, or J3-thrC): its ``relevant_docs`` and
  chunk carriers must equal the docsel (``rank`` order, ``chunk`` == True) unless ``--skip-docsel-check``.
* rules (``r2ai.s38.window`` / ``r2ai.s38.clean``), applied to the base chunk of every carrier:
  ``clean``      cut the tail before the first host marker (``scores/clean_markers.json``, counted on J3-thrC) and,
                 for baophutho.vn, at the first base64 line (``--no-base64`` turns that off; CH-clean had it off);
  ``cap2048``    body > 2,048 tokens -> 2,048-token window around the best t256 chunk;
  ``full+w768``  body > T tokens -> base chunk + 768-token window around the best t256 chunk (2 chunks).
  Carriers the rule leaves alone keep the base chunk byte for byte; docs never change.
* best t256 chunk (CH-t256-1 rule): the S32 chunk judge on the scored answer/body chunks of the canonical owner
  (Pharmacity lots: their own chunks), else cosine(S29 query vector, chunk vector); ties: file order.
* ``full+w768`` picks T itself: the ZIP grows by about ``RATE_FULL`` bytes per second chunk (S38, J3-thrC), so the
  first candidates sit around the estimate; every round builds its candidates in parallel processes and the next
  round bisects between the highest T over the ceiling and the lowest under it, until they are 1 token apart or
  ``--rounds`` is used up. Result: the lowest measured T with ZIP < 102,600,000 B.

Steps (``run`` = ``prepare`` + ``write``): ``prepare`` writes ``<out>/<name>.cache.parquet`` (chunks per carrier,
body tokens) and ``<name>.prepare.json``; ``write --threshold T`` writes ``<name>[-t<T>].zip``, runs the validator
and ``<name>[-t<T>].compose.json``. Validator: ``scripts/s26/validate_pa3.py`` with the S27 Pharmacity handoff,
``--max-chunks-per-doc 2`` for full+w768.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from r2ai import paths
from r2ai.index.chunk import SUBMITTABLE
from r2ai.s38 import clean as cl
from r2ai.s38.build import replace_chunks, write_bundle
from r2ai.s38.score import S32_ROOT, S32_SCORES, choose, load_query_vectors, scores_dir
from r2ai.s38.sources import (CANONICAL, _atomic_json, _atomic_parquet, _sha256, describe, load_doc_rows,
                              load_owner_map, owner_of, read_bundle, run_root)
from r2ai.s38.window import _body_span, pieces_for

RULES = {"clean": None, "cap2048": ("window", 2048, 2048), "full+w768": ("full+window", 768, 1536)}
CEILING = 102_600_000
RATE_FULL = 94.3        # B per 768 window: (102,599,152 - 101,181,680) / 15,026 (CH-full+w768-t1849 on J3-thrC)
FULL_MIN_T = 1536
PHARM_HANDOFF = Path("runs/s27-pharmacity/handoff/cc-s27-pharmacity")
VALIDATOR = Path("scripts/s26/validate_pa3.py")


def _log(msg: str) -> None:
    print(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "message": msg}, ensure_ascii=False),
          flush=True)


# ------------------------------------------------------------------------------------------- inputs

def check_docsel(rows: list[dict[str, Any]], docsel: pd.DataFrame) -> dict[str, Any]:
    """Docs (in rank order) and chunk carriers of the base bundle against the docsel."""
    sel = docsel.sort_values(["query_id", "rank"])
    docs = sel.groupby("query_id").doc_id.apply(list).to_dict()
    carriers = set(zip(sel.query_id[sel.chunk], sel.doc_id[sel.chunk]))
    bad_docs = sum(int(r["id"]) not in docs or [int(d) for d in r["relevant_docs"]] != docs[int(r["id"])] for r in rows)
    base_car = {(int(r["id"]), int(c["doc_id"])) for r in rows for c in r["relevant_chunks"]}
    return {"queries": len(rows), "queries_docs_differ": bad_docs, "docsel_queries": len(docs),
            "carriers_base": len(base_car), "carriers_docsel": len(carriers),
            "carriers_only_base": len(base_car - carriers), "carriers_only_docsel": len(carriers - base_car)}


def doc_texts(doc_ids: set[int]) -> tuple[dict[int, str], dict[int, str], dict[int, int], set[int], dict[int, int]]:
    """(text by carrier doc, lang, owner, Pharmacity ids, text key): own canonical row, else the alias owner's
    (validator S26 rule); Pharmacity lot docs take their lot doc_text."""
    from r2ai.s29.build import _load_extra_docs
    from r2ai.s29.qwen_judge import _inputs

    canonical = paths.DATA_DIR / CANONICAL
    owners, cid_to_doc = load_owner_map(canonical)
    extra, _ = _load_extra_docs(_inputs())
    owner = {d: owner_of(d, owners, cid_to_doc) for d in doc_ids}
    rows = load_doc_rows(canonical, (doc_ids | set(owner.values())) - set(extra))
    text = dict(zip(rows.doc_id.astype(int), rows.doc_text))
    lang = dict(zip(rows.doc_id.astype(int), rows.lang))
    out_text, out_lang, key = {}, {}, {}
    for d in doc_ids:
        if d in extra:
            out_text[d], out_lang[d], key[d] = extra[d] or "", "vi", d
            continue
        k = d if d in text else owner[d]
        out_text[d], out_lang[d], key[d] = text.get(k) or "", str(lang.get(k, "vi")), k
    return out_text, out_lang, owner, set(extra) & doc_ids, key


def t256_chunks(owners: set[int], pharm: set[int]) -> pd.DataFrame:
    """Answer/body t256 chunks: canonical owners and Pharmacity lot docs; ``idx`` = file order among them."""
    from r2ai.s29.qwen_judge import _inputs
    from r2ai.s32.chunks import load_pharmacity_chunks
    from r2ai.s38.sources import load_chunks

    frames = []
    if owners:
        c = load_chunks(paths.DATA_DIR / CANONICAL, owners)
        c["origin"] = "c"
        frames.append(c)
    if pharm:
        p = load_pharmacity_chunks(_inputs())
        p = p[p.doc_id.isin(pharm)].copy()
        lots = {}
        for lot in sorted(x for x in (_inputs()["extra_docs"] / "lots").iterdir() if x.is_dir()):
            t = pq.read_table(lot / "chunks" / "chunks_t256.parquet", columns=["chunk_id", "char_start", "char_end"]).to_pandas()
            t["chunk_id"] = (1_000_000_000_000_000 + int(lot.name.rsplit("-", 1)[1]) * 1_000_000 + t.chunk_id.astype(np.int64))
            lots.update(dict(zip(t.chunk_id, zip(t.char_start, t.char_end))))
        p["char_start"] = [lots[c][0] for c in p.chunk_id]
        p["char_end"] = [lots[c][1] for c in p.chunk_id]
        p["vec_row"], p["file_row"], p["origin"] = -1, np.arange(len(p)), "p"
        frames.append(p)
    return pd.concat(frames, ignore_index=True)


def best_t256(carriers: pd.DataFrame, text: dict[int, str], owner: dict[int, int], pharm: set[int]
              ) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[int, pd.DataFrame]]:
    """((query, doc) -> char span of the best answer/body t256 chunk in the carrier's text (CH-t256-1 rule),
    doc -> its answer/body t256 chunks when their spans index the carrier's text)."""
    docs = set(carriers.doc_id.astype(int))
    own = {d: (d if d in pharm else owner[d]) for d in docs}
    chunks = t256_chunks({o for d, o in own.items() if d not in pharm}, pharm & docs)
    by_owner = {int(o): g.sort_values("file_row") for o, g in chunks.groupby("doc_id", sort=False)}
    pieces: dict[int, pd.DataFrame] = {}
    for d in docs:
        g = by_owner.get(own[d])
        if g is None:
            continue
        t = text[d]
        if not all(t[s:e] == x for s, e, x in zip(g.char_start.astype(int), g.char_end.astype(int), g.text)):
            continue                                            # spans do not index this text (sources rule)
        sub = g[g.field.isin(SUBMITTABLE)].reset_index(drop=True)
        if len(sub):
            pieces[d] = sub.assign(idx=np.arange(len(sub)))
    cp = pd.read_parquet(paths.OUT_DIR / S32_ROOT / "pool" / "chunk_pairs.parquet",
                         columns=["pair_id", "query_id", "doc_id", "chunk_id", "field"])
    cp = cp.merge(carriers[["query_id", "doc_id"]], on=["query_id", "doc_id"])
    sc = pd.read_parquet(paths.OUT_DIR / S32_SCORES, columns=["pair_id", "score"])
    judge = cp[cp.field.isin(SUBMITTABLE)].merge(sc, on="pair_id")
    judge = {(int(q), int(d), int(c)): float(s) for q, d, c, s in zip(judge.query_id, judge.doc_id, judge.chunk_id, judge.score)}
    qv = load_query_vectors()
    dense = np.load(paths.DATA_DIR / CANONICAL / "vectors" / "dense.npy", mmap_mode="r")
    with np.load(paths.RUNS_DIR / "s32-chunk-judge" / "pool" / "supplement_vectors.npz") as sup:
        sup_row = {(str(o), int(c)): k for k, (o, c) in enumerate(zip(sup["origin"], sup["chunk_id"]))}
        sup_dense = np.array(sup["dense"])
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for q, d in zip(carriers.query_id.astype(int), carriers.doc_id.astype(int)):
        g = pieces.get(d)
        if g is None:
            continue
        cids = g.chunk_id.astype(int).tolist()
        scores = [judge.get((q, d, c)) for c in cids]
        keep = [k for k, s in enumerate(scores) if s is not None]
        if keep:
            vals = [scores[k] for k in keep]
        else:
            vecs, keep = [], []
            for k, (c, vr, o) in enumerate(zip(cids, g.vec_row.astype(int), g.origin)):
                if vr >= 0:
                    vecs.append(dense[vr])
                elif (o, c) in sup_row:
                    vecs.append(sup_dense[sup_row[(o, c)]])
                else:
                    continue
                keep.append(k)
            if not keep:
                continue
            vals = (np.stack(vecs).astype(np.float32) @ qv[q].astype(np.float32)).tolist()
        pick = keep[choose(vals, [int(g.idx.iat[k]) for k in keep], 1)[0]]
        out[(q, d)] = (int(g.char_start.iat[pick]), int(g.char_end.iat[pick]))
    return out, pieces


# ------------------------------------------------------------------------------------------ prepare

def prepare(out: Path, name: str, base_zip: Path, docsel_path: Path | None, rule: str, base64: bool,
            skip_check: bool) -> dict[str, Any]:
    started = time.monotonic()
    rows = read_bundle(base_zip)
    report: dict[str, Any] = {"name": name, "rule": rule, "base_zip": str(base_zip), "base_zip_bytes": base_zip.stat().st_size,
                              "base_zip_sha256": _sha256(base_zip), "docsel": str(docsel_path) if docsel_path else None,
                              "base64_baophutho": base64 if rule == "clean" else None}
    if docsel_path is not None:
        report["docsel_sha256"] = _sha256(docsel_path)
        check = check_docsel(rows, pd.read_parquet(docsel_path, columns=["query_id", "doc_id", "rank", "chunk"]))
        report["docsel_check"] = check
        if not skip_check and (check["queries_docs_differ"] or check["carriers_only_base"] or check["carriers_only_docsel"]):
            raise ValueError(f"base bundle differs from the docsel: {check}")
    car = pd.DataFrame([(int(r["id"]), int(c["doc_id"]), c["chunk_text"]) for r in rows for c in r["relevant_chunks"]],
                       columns=["query_id", "doc_id", "body"])
    del rows
    text, lang, owner, pharm, key = doc_texts(set(car.doc_id.astype(int)))
    _log(f"carriers={len(car)} docs={car.doc_id.nunique()} pharmacity={len(pharm)}")
    from r2ai.gold_check.gold_check_common import load_tokenizer

    tok = load_tokenizer()
    out_rows: list[dict[str, Any]] = []
    stats = {"located": 0, "fallback_body": 0, "no_t256_choice": 0, "chunk_outside_body": 0, "not_verbatim_skipped": 0}
    if rule == "clean":
        markers = {h: {m["key"] for m in v["markers"]} for h, v in json.loads(
            (scores_dir() / "clean_markers.json").read_text(encoding="utf-8"))["hosts"].items()}
        canonical = paths.DATA_DIR / CANONICAL
        hosts = pq.read_table(canonical / "docs.parquet", columns=["doc_id", "host"],
                              filters=[("doc_id", "in", sorted(set(key.values())))]).to_pandas()
        host_of = dict(zip(hosts.doc_id.astype(int), hosts.host))
        cut_by_host: dict[str, int] = {}
        for q, d, body in zip(car.query_id.astype(int), car.doc_id.astype(int), car.body):
            host = host_of.get(key[d])
            t = None
            if host in markers or (base64 and host in cl.BASE64_HOSTS):
                t = cl.clean_cut(body, markers.get(host, set()), base64 and host in cl.BASE64_HOSTS)
                if t is not None and t not in text[d]:
                    stats["not_verbatim_skipped"] += 1
                    t = None
            if t is not None:
                cut_by_host[host] = cut_by_host.get(host, 0) + 1
            out_rows.append({"query_id": q, "doc_id": d, "rank": 1, "text": t, "mode": "clean" if t is not None else "base",
                             "body_tokens": -1})
        report["carriers_cut_by_host"] = cut_by_host
    else:
        spec = RULES[rule]
        best, pieces = best_t256(car[["query_id", "doc_id"]], text, owner, pharm)
        empty = pd.DataFrame({"field": [], "char_start": [], "char_end": []})
        _log(f"best t256 chunks={len(best)}")
        for begin in range(0, len(car), 5_000):
            part = car.iloc[begin:begin + 5_000]
            bodies, base = [], []
            for d, body in zip(part.doc_id.astype(int), part.body):
                t = text[d]
                at = t.find(body)
                if at >= 0:
                    stats["located"] += 1
                    bodies.append(body)
                    base.append((at, True))
                else:
                    stats["fallback_body"] += 1
                    s, e = _body_span(t, lang[d], pieces.get(d, empty))       # same rule as r2ai.s38.window
                    bodies.append(t[s:e])
                    base.append((s, False))
            enc = tok(bodies, add_special_tokens=False, return_offsets_mapping=True)
            for k, (q, d, body, (b0, located)) in enumerate(zip(part.query_id.astype(int), part.doc_id.astype(int),
                                                                 bodies, base)):
                offsets = enc["offset_mapping"][k]
                chunk = None
                span = best.get((q, d))
                if span is None:
                    stats["no_t256_choice"] += 1
                else:
                    cs, ce = span[0] - b0, span[1] - b0
                    if ce <= 0 or cs >= len(body):
                        stats["chunk_outside_body"] += 1
                    else:
                        chunk = (max(cs, 0), min(ce, len(body)))
                for rank, piece in enumerate(pieces_for(*spec, body, offsets, chunk), 1):
                    if piece is None or (piece[0] == "whole" and located):
                        out_rows.append({"query_id": q, "doc_id": d, "rank": rank, "text": None, "mode": "base",
                                         "body_tokens": len(offsets)})
                    else:
                        out_rows.append({"query_id": q, "doc_id": d, "rank": rank, "text": body[piece[1]:piece[2]],
                                         "mode": piece[0], "body_tokens": len(offsets)})
            _log(f"carriers {begin + len(part)}/{len(car)}")
    frame = pd.DataFrame(out_rows)
    target = out / f"{name}.cache.parquet"
    _atomic_parquet(frame, target)
    changed = frame[frame["mode"] != "base"]
    report.update({"carriers": len(car), "chunks": len(frame), "carriers_changed": int(changed[["query_id", "doc_id"]].drop_duplicates().shape[0]),
                   **stats, "cache": str(target), "seconds": round(time.monotonic() - started, 1)})
    if rule != "clean":
        bt = frame.drop_duplicates(["query_id", "doc_id"]).body_tokens.to_numpy()
        report["body_tokens"] = describe(bt)
    _atomic_json(out / f"{name}.prepare.json", report)
    return report


# -------------------------------------------------------------------------------------------- write

def write(out: Path, name: str, threshold: int | None, validate: bool = True) -> dict[str, Any]:
    started = time.monotonic()
    prep = json.loads((out / f"{name}.prepare.json").read_text(encoding="utf-8"))
    rule = prep["rule"]
    frame = pd.read_parquet(out / f"{name}.cache.parquet")
    label = name
    if rule == "full+w768":
        threshold = threshold if threshold is not None else FULL_MIN_T
        if threshold < FULL_MIN_T:
            raise ValueError(f"threshold below {FULL_MIN_T}: windows not prepared")
        frame = frame[(frame["rank"] == 1) | (frame.body_tokens > threshold)]
        label = f"{name}-t{threshold}"
    chosen: dict[tuple[int, int], list] = {}
    for q, d, t in zip(frame.query_id.astype(int), frame.doc_id.astype(int), frame.text):
        chosen.setdefault((q, d), []).append(None if t is None or t != t else t)
    rows = read_bundle(Path(prep["base_zip"]))
    if _sha256(Path(prep["base_zip"])) != prep["base_zip_sha256"]:
        raise ValueError("base ZIP changed since prepare")
    new_rows = [replace_chunks(r, chosen) for r in rows]
    target = out / f"{label}.zip"
    paths.assert_writable(target)
    paths.assert_writable(target.with_name(target.name + ".tmp"))
    res = write_bundle(new_rows, target, f"{label}.json")
    entry = {"name": label, "rule": rule, "threshold": threshold, "zip": str(target), "zip_bytes": target.stat().st_size,
             "zip_sha256": _sha256(target), **res, "under_ceiling": target.stat().st_size < CEILING,
             "docs": sum(len(r["relevant_docs"]) for r in new_rows),
             "chunks": sum(len(r["relevant_chunks"]) for r in new_rows),
             "second_chunks": int((frame["rank"] == 2).sum()),
             "carriers_changed": sum(any(t is not None for t in v) for v in chosen.values()),
             "docs_equal_base": all(a["relevant_docs"] == b["relevant_docs"] for a, b in zip(rows, new_rows))}
    if validate:
        cmd = [sys.executable, "-u", str(paths.ROOT / VALIDATOR)]
        if rule == "full+w768":
            cmd += ["--max-chunks-per-doc", "2"]
        cmd += [str(target), str(paths.OUT_DIR / PHARM_HANDOFF)]
        res_v = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        (out / f"{label}.validate.json").write_text(res_v.stdout, encoding="utf-8")
        try:
            v = json.loads(res_v.stdout)
            entry["validator"] = {"ok": v["ok"], "errors": {k: n for k, n in v["errors"].items() if n},
                                  "chunk_check": v["chunk_check"]}
        except (ValueError, KeyError):
            entry["validator"] = {"ok": False, "stderr_tail": res_v.stderr[-2000:]}
    entry["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / f"{label}.compose.json", entry)
    return entry


def estimate_thresholds(frame: pd.DataFrame, base_zip_bytes: int, margin: int = 0) -> dict[str, Any]:
    """T so that (number of second chunks) x RATE_FULL fits under the ceiling, and two neighbours."""
    bt = np.sort(frame[frame["rank"] == 2].body_tokens.to_numpy())    # ascending; second chunks at T = #(bt > T)
    budget = CEILING - margin - base_zip_bytes
    k = int(max(budget, 0) / RATE_FULL)

    def t_for(count: int) -> int:
        if count >= len(bt):
            return FULL_MIN_T
        return max(int(bt[len(bt) - count - 1]), FULL_MIN_T)

    return {"second_chunks_at_min_t": int(len(bt)), "budget_bytes": int(budget), "k_estimate": k,
            "t_estimate": t_for(k), "candidates": sorted({t_for(int(k * 1.03)), t_for(k), t_for(int(k * 0.97))})}


def run_full(out: Path, name: str, candidates: list[int], rounds: int) -> dict[str, Any]:
    """Build candidate thresholds in parallel processes, bisect between over/under, keep the lowest T under."""
    tried: dict[int, dict[str, Any]] = {}
    for r in range(rounds):
        todo = [t for t in candidates if t not in tried]
        _log(f"round {r + 1}: thresholds {todo}")
        procs = [(t, subprocess.Popen([sys.executable, "-m", "r2ai.s38.compose", "write", "--name", name, "--out", str(out),
                                       "--threshold", str(t)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
                 for t in todo]
        for t, p in procs:
            _, err = p.communicate()
            if p.returncode:
                raise RuntimeError(f"write t={t} failed: {err.decode('utf-8', 'replace')[-2000:]}")
            tried[t] = json.loads((out / f"{name}-t{t}.compose.json").read_text(encoding="utf-8"))
            _log(f"t={t} zip={tried[t]['zip_bytes']} under={tried[t]['under_ceiling']}")
        under = sorted(t for t, e in tried.items() if e["under_ceiling"])
        over = sorted(t for t, e in tried.items() if not e["under_ceiling"])
        if not under:
            hi = max(over)
            candidates = [hi + 25, hi + 50, hi + 100]
            continue
        lo_ok = under[0]
        below = [t for t in over if t < lo_ok]
        if not below:
            candidates = [max(FULL_MIN_T, lo_ok - 25), max(FULL_MIN_T, lo_ok - 10)] if lo_ok > FULL_MIN_T else []
            if not candidates:
                break
            continue
        hi_bad = max(below)
        if lo_ok - hi_bad <= 1:
            break
        step = max(1, (lo_ok - hi_bad) // 3)
        candidates = sorted({hi_bad + step, hi_bad + 2 * step, lo_ok - 1} - {hi_bad, lo_ok})
    under = sorted(t for t, e in tried.items() if e["under_ceiling"])
    over = sorted(t for t, e in tried.items() if not e["under_ceiling"])
    best = under[0] if under else None
    return {"tried": {str(t): {k: e[k] for k in ("zip_bytes", "under_ceiling", "second_chunks", "chunks")} | {
                "validator_ok": e.get("validator", {}).get("ok")} for t, e in sorted(tried.items())},
            "lowest_t_under_ceiling": best, "highest_t_over_ceiling_below_it": max([t for t in over if best and t < best], default=None),
            "exact_to_one_token": bool(best and any(t == best - 1 for t in over))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "write", "run", "estimate"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, default=None, help="default RUNS_DIR/s38-chunk-metric-v2/compose")
    parser.add_argument("--base-zip", type=Path)
    parser.add_argument("--docsel", type=Path)
    parser.add_argument("--rule", choices=list(RULES))
    parser.add_argument("--no-base64", action="store_true", help="clean: no base64 cut on baophutho.vn")
    parser.add_argument("--skip-docsel-check", action="store_true")
    parser.add_argument("--threshold", type=int, action="append", default=None, help="full+w768: body tokens above which the window is added")
    parser.add_argument("--rounds", type=int, default=3, help="full+w768 run: parallel build rounds")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)
    out = paths.resolve_path(args.out or run_root() / "compose")
    if not out.is_relative_to(paths.RUNS_DIR):
        raise ValueError(f"S38 output must stay below RUNS_DIR: {out}")
    paths.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    if args.command in ("prepare", "run") and not (out / f"{args.name}.prepare.json").exists():
        if not (args.base_zip and args.rule):
            parser.error("prepare needs --base-zip and --rule (and normally --docsel)")
        rep = prepare(out, args.name, paths.resolve_path(args.base_zip),
                      paths.resolve_path(args.docsel) if args.docsel else None, args.rule, not args.no_base64,
                      args.skip_docsel_check)
        print(json.dumps(rep, ensure_ascii=False, indent=1))
    if args.command == "prepare":
        return 0
    prep = json.loads((out / f"{args.name}.prepare.json").read_text(encoding="utf-8"))
    if args.command == "estimate" or (args.command == "run" and prep["rule"] == "full+w768" and not args.threshold):
        est = estimate_thresholds(pd.read_parquet(out / f"{args.name}.cache.parquet"), prep["base_zip_bytes"])
        print(json.dumps(est, indent=1))
        if args.command == "estimate":
            return 0
        res = run_full(out, args.name, est["candidates"], args.rounds)
        res["estimate"] = est
        _atomic_json(out / f"{args.name}.threshold.json", res)
        print(json.dumps(res, indent=1))
        return 0
    for t in (args.threshold or [None]):
        print(json.dumps(write(out, args.name, t, not args.no_validate), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
