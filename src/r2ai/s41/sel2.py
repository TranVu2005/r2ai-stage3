"""S41 CH-sel2 (Vu 27/10): on a validated full+w768 bundle (FREE-noJina, or the winning carrier probe), the carriers with P4B >= theta
whose body is longer than T = 1536 tokens (exactly the carriers with a 768 window) ship 2 contiguous halves of the body instead of
(full body + 768 window), FREE-half rule (``scripts/s40/split_compose.py --mode half``: cut at the sentence end nearest the middle of the
region within +-T/4 tokens, else a word start; region = the whole body when tokens <= 4T, else the 4T block centred on the best t256
chunk, shifted inside the body). Every other carrier keeps its chunks byte for byte. theta = the P4B of the ``--n-split``-th eligible
carrier (default 11,400, about 1/3 of the 34,186 docs FREE-split split).

    python -m r2ai.s41.sel2 --name CH-sel2 --base-zip <ZIP> --base-sha <SHA> --docsel <docsel.parquet> --out out/runs/s41/sel2

Eligible: (query, doc) with two chunks in the base (full body, then the window) whose body is a substring of the doc text (the halves must
be verbatim in the doc text; the others keep their chunks, counted). The best t256 chunk centre is the centre of the 768 window (compose
centres the window on it, clamped inside the body). Halves that are empty or equal after the metric normalisation keep the base chunks
(counted). Validator S26 ``--max-chunks-per-doc 2`` as compose.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s38 import compose as cp
from r2ai.s38.build import write_bundle
from r2ai.s38.sources import _atomic_json, _sha256, read_bundle

T = 1536
ROOT = Path(__file__).resolve().parents[3]


def _split_module():
    spec = importlib.util.spec_from_file_location("s40_split_compose", ROOT / "scripts" / "s40" / "split_compose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.MODE = "half"
    return mod


def two_chunk_pairs(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, int], tuple[str, str]]:
    """(query, doc) -> (body, window) for every doc with exactly two chunks in the base."""
    out: dict[tuple[int, int], list[str]] = {}
    for r in rows:
        for c in r["relevant_chunks"]:
            out.setdefault((int(r["id"]), int(c["doc_id"])), []).append(c["chunk_text"])
    return {k: (v[0], v[1]) for k, v in out.items() if len(v) == 2}


def choose(eligible: pd.DataFrame, n_split: int) -> tuple[float, pd.DataFrame]:
    """theta = P4B of the ``n_split``-th eligible carrier by P4B (descending); every eligible carrier with P4B >= theta."""
    if eligible.empty:
        raise ValueError("no eligible carrier")
    ranked = eligible.sort_values("p4b", ascending=False, kind="stable")
    theta = float(ranked.p4b.iloc[min(n_split, len(ranked)) - 1])
    return theta, ranked[ranked.p4b >= theta]


def replace_doc_chunks(rows: Sequence[dict[str, Any]], pieces: dict[tuple[int, int], list[str]]) -> list[dict[str, Any]]:
    """Rows where every chunk of a (query, doc) in ``pieces`` is replaced by the pieces, at the position of its first chunk."""
    out = []
    for r in rows:
        q, chunks, done = int(r["id"]), [], set()
        for c in r["relevant_chunks"]:
            key = (q, int(c["doc_id"]))
            if key not in pieces:
                chunks.append(c)
            elif key not in done:
                chunks += [{"doc_id": int(c["doc_id"]), "chunk_text": p} for p in pieces[key]]
                done.add(key)
        out.append({**r, "relevant_chunks": chunks})
    return out


def halves(sc, tok, body: str, window: str) -> tuple[list[str] | None, dict[str, Any]]:
    from r2ai.eval.scorer import normalize_metric

    enc = tok([body], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"][0]
    starts = np.array([s for s, _ in enc], dtype=np.int64)
    sent = {m.end() for m in sc.SENT.finditer(body) if 0 < m.end() < len(body)}
    word = {int(s) for k, s in enumerate(starts) if k and k % 8 == 0 and (body[s - 1].isspace() or not body[s].isascii())}
    chars = sorted(sent | word)
    cc = np.array(chars, np.int64)
    ct = np.searchsorted(starts, chars, side="left").astype(np.int32)
    cs = np.array([c in sent for c in chars], bool)
    at = body.find(window)
    mid = int(np.searchsorted(starts, at + len(window) / 2, side="right")) - 1 if at >= 0 else -1
    edges, fb, case = sc.edges_for(len(enc), len(body), cc, ct, cs, T, mid)
    pieces = [body[a:b].strip() for a, b in zip(edges, edges[1:])]
    norm = [normalize_metric(p) for p in pieces]
    if len(pieces) != 2 or any(not p for p in pieces) or len(set(norm)) < 2:
        return None, {"case": case}
    return pieces, {"case": case, "tokens": len(enc), "word_start_edges": fb, "window_found": at >= 0}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--docsel", type=Path, required=True, help="docsel of the base (query_id, doc_id, p4b)")
    parser.add_argument("--n-split", type=int, default=11_400)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from r2ai.gold_check.gold_check_common import load_tokenizer

    started = time.monotonic()
    out = args.out.resolve()
    target = out / f"{args.name}.zip"
    paths.assert_writable(target)
    paths.assert_writable(target.with_name(target.name + ".tmp"))
    out.mkdir(parents=True, exist_ok=True)
    if _sha256(args.base_zip) != args.base_sha:
        raise ValueError(f"{args.base_zip} SHA256 differs from the pinned {args.base_sha}")
    rows = read_bundle(args.base_zip)
    pairs = two_chunk_pairs(rows)
    docsel = pd.read_parquet(args.docsel, columns=["query_id", "doc_id", "p4b"])
    p4b = dict(zip(zip(docsel.query_id.astype(int), docsel.doc_id.astype(int)), docsel.p4b.astype(float)))
    text = cp.doc_texts({d for _, d in pairs})[0]
    verbatim = {k for k, (body, _) in pairs.items() if body in (text.get(k[1]) or "")}
    eligible = pd.DataFrame([(q, d, p4b[(q, d)]) for q, d in sorted(verbatim)], columns=["query_id", "doc_id", "p4b"])
    theta, chosen = choose(eligible, args.n_split)
    sc, tok = _split_module(), load_tokenizer()
    pieces, cases, kept, fb, tokens = {}, {}, 0, 0, []
    for q, d in zip(chosen.query_id.astype(int), chosen.doc_id.astype(int)):
        body, window = pairs[(q, d)]
        got, info = halves(sc, tok, body, window)
        cases[info["case"]] = cases.get(info["case"], 0) + 1
        if got is None:
            kept += 1
            continue
        pieces[(q, d)] = got
        fb += info["word_start_edges"]
        tokens.append(info["tokens"])
    new_rows = replace_doc_chunks(rows, pieces)
    res = write_bundle(new_rows, target, f"{args.name}.json")
    size = target.stat().st_size
    report: dict[str, Any] = {
        "name": args.name, "base_zip": str(args.base_zip), "base_zip_sha256": args.base_sha, "docsel": str(args.docsel),
        "two_chunk_carriers": len(pairs), "body_not_in_doc_text": len(pairs) - len(verbatim), "eligible": len(eligible),
        "n_split_target": args.n_split, "theta_p4b": theta, "chosen": int(len(chosen)), "split_docs": len(pieces),
        "kept_dup_or_empty": kept, "cases": cases, "word_start_edges": fb,
        "split_body_tokens_p10_p50_p90": [float(x) for x in np.quantile(tokens or [0], [.1, .5, .9])],
        "zip": str(target), "zip_bytes": size, "zip_sha256": _sha256(target), **res,
        "limit_bytes": cp.BTC_CEILING - cp.RESERVE, "under_ceiling": size < cp.BTC_CEILING - cp.RESERVE,
        "docs": sum(len(r["relevant_docs"]) for r in new_rows), "chunks": sum(len(r["relevant_chunks"]) for r in new_rows),
        "docs_equal_base": all(a["relevant_docs"] == b["relevant_docs"] for a, b in zip(rows, new_rows))}
    report["validator"] = cp.run_validator(out, args.name, "full+w768", target, {int(d) for r in new_rows for d in r["relevant_docs"]})
    report["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / f"{args.name}.sel2.json", report)
    print(json.dumps({k: report[k] for k in ("name", "theta_p4b", "split_docs", "zip_bytes", "zip_sha256", "under_ceiling", "validator")},
                     ensure_ascii=False))
    return 0 if report["validator"].get("ok") and report["under_ceiling"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
