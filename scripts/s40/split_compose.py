"""S40 chunk probe FREE-split (26/10): on the docsel / CR-fair carriers / thin of 120A-free-v2, every carrier whose body is longer than T
BGE-M3 tokens ships CONTIGUOUS, non-overlapping verbatim pieces of about T tokens instead of (full body + 768 window); bodies <= T keep the
full body as one chunk. Same docs, same carriers as free-v2; only the chunk texts change. T is the lowest value whose ZIP stays under
102,600,000 B; validator S26 with ``--max-chunks-per-doc 4``.

* T < tokens <= 4T: n = ceil(tokens / T) pieces covering the whole body, cut targets k * tokens / n.
* FREE-half (``--mode half``, Vu 26/10): the same region (the whole body if tokens <= 4T, else the 4T block below) as exactly 2 contiguous
  pieces, cut at a sentence end near the middle of the region.
* tokens > 4T (Vu 26/10): 4 pieces of about T tokens, a block of 4T tokens centred on the best t256 chunk (the chunk the 768 window of
  compose ``full+w768`` is centred on: ``r2ai.s38.compose.best_t256``), shifted inside the body when it touches the start or the end; the body
  outside the block is not shipped. Without a best chunk the block starts at the body start (counted).

    python scripts/s40/split_compose.py prepare            # candidates of every carrier body -> <out>/split.cands.parquet (one tokenisation)
    python scripts/s40/split_compose.py write --t 512      # <out>/FREE-split-t512.zip + .compose.json (+ validator unless --no-validate)
    python scripts/s40/split_compose.py search             # lowest T under the ceiling (size model from measured ZIPs, then bisection)
    python scripts/s40/split_compose.py search --mode half # FREE-half (same candidates file)

Cut rule: each edge (a cut target, and the two block ends when they are inside the body) is the sentence end (``. ! ? ; … 。 ！ ？ ；`` followed
by space or end, or a line break) whose token index is nearest to its target within +-T/4 tokens, else the nearest word start (every 8th token
if preceded by whitespace; for CJK any 8th token); edges strictly increase. Pieces are body[e_k:e_k+1] with surrounding whitespace stripped
(substrings of the body, so of the doc text). A body whose pieces would repeat a text (after the metric normalisation) or be empty keeps the
full body (counted).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r2ai import paths  # noqa: E402
from r2ai.s38 import compose as cp  # noqa: E402
from r2ai.s38 import registry as reg  # noqa: E402
from r2ai.s38.build import replace_chunks, write_bundle  # noqa: E402

SRC = paths.RUNS_DIR / "release" / "120A-free-v2" / "compose"
BASE_NAME = "120A-free-v2-full+w768"
OUT = paths.RUNS_DIR / "s40" / "split"
NAME = "FREE-split"
NAMES = {"split": "FREE-split", "half": "FREE-half"}
MODE = "split"
MAX_PIECES = 4
CEILING = cp.BTC_CEILING - cp.RESERVE
SENT = re.compile(r"(?:[.!?;…](?=\s|$)|[。！？；]|\n)\s*")


def log(m: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {m}"
    print(line, flush=True)
    with open(OUT / "split.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def base_rows() -> list[dict]:
    prep = json.loads((SRC / f"{BASE_NAME}.prepare.json").read_text(encoding="utf-8"))
    rows = cp.load_base(prep)
    swap = SRC / f"{BASE_NAME}.swap.json"
    return cp.apply_swap(rows, json.loads(swap.read_text(encoding="utf-8"))) if swap.exists() else rows


def prepare() -> dict:
    """Per carrier: body tokens and the candidate cuts (char offset, token index, sentence end or word start)."""
    from r2ai.gold_check.gold_check_common import load_tokenizer

    t0 = time.monotonic()
    rows = base_rows()
    car = [(int(r["id"]), int(c["doc_id"]), c["chunk_text"]) for r in rows for c in r["relevant_chunks"]]
    log(f"carriers {len(car)}")
    frame0 = pd.DataFrame(car, columns=["query_id", "doc_id", "body"])
    text, _, owner, pharm, key = cp.doc_texts(set(frame0.doc_id.astype(int)))
    best, _ = cp.best_t256(frame0[["query_id", "doc_id"]], text, owner, pharm)
    hosts = cp.host_map(set(key.values()))
    index = reg.registry_index()
    log(f"best t256 chunks {len(best)} of {len(car)} carriers")
    tok = load_tokenizer()
    out = {"query_id": [], "doc_id": [], "tokens": [], "cut_char": [], "cut_tok": [], "cut_sent": [], "best_mid_tok": [], "host": []}
    for begin in range(0, len(car), 5_000):
        part = car[begin:begin + 5_000]
        enc = tok([b for _, _, b in part], add_special_tokens=False, return_offsets_mapping=True)
        for (q, d, body), offs in zip(part, enc["offset_mapping"]):
            starts = np.array([s for s, _ in offs], dtype=np.int64)
            sent = {m.end() for m in SENT.finditer(body) if 0 < m.end() < len(body)}
            word = {int(s) for k, s in enumerate(starts) if k and k % 8 == 0 and (body[s - 1].isspace() or not body[s].isascii())}   # every 8th token
            chars = sorted(sent | word)
            idx = np.searchsorted(starts, chars, side="left")                     # tokens before the cut
            out["query_id"].append(q), out["doc_id"].append(d), out["tokens"].append(len(offs))
            out["cut_char"].append(np.array(chars, np.int64)), out["cut_tok"].append(idx.astype(np.int32))
            out["cut_sent"].append(np.array([c in sent for c in chars], bool))
            span, at = best.get((q, d)), text[d].find(body)
            mid = -1
            if span is not None and at >= 0 and 0 <= (span[0] + span[1]) / 2 - at < len(body):
                mid = int(np.searchsorted(starts, (span[0] + span[1]) / 2 - at, side="right")) - 1
            out["best_mid_tok"].append(max(mid, -1))
            out["host"].append("120ask.com" if d in index else "pharmacity.vn" if d in pharm else str(hosts.get(key.get(d), "")))
        log(f"tokenised {begin + len(part)}/{len(car)}")
    frame = pd.DataFrame(out)
    paths.assert_writable(OUT / "split.cands.parquet")
    frame.to_parquet(OUT / "split.cands.parquet", index=False)
    rep = {"carriers": len(frame), "tokens_p50_p90_p99": [float(x) for x in np.quantile(frame.tokens, [.5, .9, .99])], "seconds": round(time.monotonic() - t0, 1)}
    (OUT / "split.prepare.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    return rep


def pick(target: float, last: int, cut_char: np.ndarray, cut_tok: np.ndarray, cut_sent: np.ndarray, t: int) -> tuple[int | None, bool]:
    """Char offset of the edge nearest to the token ``target`` after char ``last`` (sentence end within +-t/4, else any candidate)."""
    ok = cut_char > last
    sent = ok & cut_sent & (np.abs(cut_tok - target) <= t / 4)
    pool = sent if sent.any() else ok
    if not pool.any():
        return None, False
    j = np.flatnonzero(pool)[np.argmin(np.abs(cut_tok[pool] - target))]
    return int(cut_char[j]), not sent.any()


def edges_for(tokens: int, length: int, cut_char: np.ndarray, cut_tok: np.ndarray, cut_sent: np.ndarray, t: int, mid: int
              ) -> tuple[list[int], int, str]:
    """Char edges of the pieces (first and last included), word-start fallbacks, and the case (``cover`` / ``block`` / ``block-nobest``)."""
    if tokens <= MAX_PIECES * t:
        n, start, case = math.ceil(tokens / t), 0, "cover"
        targets = [k * tokens / n for k in range(1, n)] if MODE == "split" else [tokens / 2]
        edges, end = [0], length
    else:
        case = "block" if mid >= 0 else "block-nobest"
        start = min(max((mid if mid >= 0 else 0) - MAX_PIECES * t / 2, 0), tokens - MAX_PIECES * t)
        targets = [start + k * t for k in range(1, MAX_PIECES)] if MODE == "split" else [start + MAX_PIECES * t / 2]
        edges, end = ([0] if start <= 0 else []), (length if start + MAX_PIECES * t >= tokens else None)
    fallback = 0
    if not edges:
        e, fb = pick(start, -1, cut_char, cut_tok, cut_sent, t)
        if e is None:
            return [], 0, case
        edges, fallback = [e], fallback + fb
    for target in targets:
        e, fb = pick(target, edges[-1], cut_char, cut_tok, cut_sent, t)
        if e is None:
            break
        edges.append(e)
        fallback += fb
    if end is None:
        e, fb = pick(start + MAX_PIECES * t, edges[-1], cut_char, cut_tok, cut_sent, t)
        end, fallback = (e, fallback + fb) if e is not None else (length, fallback)
    return edges + [end], fallback, case


def chosen_for(t: int, rows: list[dict]) -> tuple[dict, dict]:
    from r2ai.eval.scorer import normalize_metric

    cands = pd.read_parquet(OUT / "split.cands.parquet")
    body = {(int(r["id"]), int(c["doc_id"])): c["chunk_text"] for r in rows for c in r["relevant_chunks"]}
    chosen, stats = {}, {"split_docs": 0, "pieces": 0, "kept_dup_or_empty": 0, "word_start_edges": 0, "piece_tokens": [],
                         "cases": {"cover": 0, "block": 0, "block-nobest": 0}, "block_hosts": {}, "block_tokens_dropped": 0}
    for q, d, n_tok, cc, ct, cs, mid, host in zip(cands.query_id, cands.doc_id, cands.tokens, cands.cut_char, cands.cut_tok, cands.cut_sent,
                                                   cands.best_mid_tok, cands.host):
        if n_tok <= t:
            continue
        text = body[(int(q), int(d))]
        cc, ct, cs = np.asarray(cc), np.asarray(ct), np.asarray(cs)
        edges, fb, case = edges_for(int(n_tok), len(text), cc, ct, cs, t, int(mid))
        pieces = [text[a:b].strip() for a, b in zip(edges, edges[1:])]
        norm = [normalize_metric(p) for p in pieces]
        if len(pieces) < 2 or any(not p for p in pieces) or len(set(norm)) < len(norm):
            stats["kept_dup_or_empty"] += 1
            continue
        chosen[(int(q), int(d))] = pieces

        def tok_at(c: int) -> int:
            return int(n_tok) if c >= len(text) else 0 if c <= 0 else int(ct[np.searchsorted(cc, c)])

        te = [tok_at(c) for c in edges]
        stats["piece_tokens"] += [b - a for a, b in zip(te, te[1:])]
        stats["split_docs"] += 1
        stats["pieces"] += len(pieces)
        stats["word_start_edges"] += fb
        stats["cases"][case] += 1
        if case != "cover":
            stats["block_hosts"][host] = stats["block_hosts"].get(host, 0) + 1
            stats["block_tokens_dropped"] += int(n_tok) - (te[-1] - te[0])
    pt = np.array(stats.pop("piece_tokens") or [0])
    stats["block_hosts"] = dict(sorted(stats["block_hosts"].items(), key=lambda kv: -kv[1]))
    stats.update({"pieces_per_split_doc": round(stats["pieces"] / max(stats["split_docs"], 1), 3),
                  "piece_tokens_p10_p50_p90_max": [int(x) for x in np.quantile(pt, [.1, .5, .9])] + [int(pt.max())]})
    return chosen, stats


def run_validator(label: str, target: Path, ids: set[int]) -> dict:
    cmd = [sys.executable, "-u", str(ROOT / cp.VALIDATOR), "--max-chunks-per-doc", str(MAX_PIECES), str(target), str(paths.OUT_DIR / cp.PHARM_HANDOFF)]
    extra = reg.validator_extra(OUT / f"{label}.validator-extra", reg.registry_index(), ids)
    if extra is not None:
        cmd.append(str(extra))
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    (OUT / f"{label}.validate.json").write_text(res.stdout, encoding="utf-8")
    try:
        v = json.loads(res.stdout)
        return {"ok": v["ok"], "errors": {k: n for k, n in v["errors"].items() if n}, "chunk_check": v["chunk_check"]}
    except (ValueError, KeyError):
        return {"ok": False, "stderr_tail": res.stderr[-2000:]}


def write(t: int, validate: bool) -> dict:
    t0 = time.monotonic()
    label = f"{NAMES[MODE]}-t{t}"
    rows = base_rows()
    chosen, stats = chosen_for(t, rows)
    new_rows = [replace_chunks(r, chosen) for r in rows]
    target = OUT / f"{label}.zip"
    paths.assert_writable(target)
    res = write_bundle(new_rows, target, f"{label}.json")
    size = target.stat().st_size
    per_doc = pd.Series([len([c for c in r["relevant_chunks"] if c["doc_id"] == d]) for r in new_rows for d in {c["doc_id"] for c in r["relevant_chunks"]}])
    entry = {"name": label, "threshold": t, "zip": str(target), "zip_bytes": size, "zip_sha256": cp._sha256(target), **res,
             "limit_bytes": CEILING, "under_ceiling": size < CEILING, "docs": sum(len(r["relevant_docs"]) for r in new_rows),
             "chunks": sum(len(r["relevant_chunks"]) for r in new_rows), "carriers": int(len(per_doc)),
             "max_chunks_per_doc": int(per_doc.max()), **stats, "source": str(SRC / BASE_NAME)}
    if validate:
        entry["validator"] = run_validator(label, target, {int(d) for r in new_rows for d in r["relevant_docs"]})
    entry["seconds"] = round(time.monotonic() - t0, 1)
    (OUT / f"{label}.compose.json").write_text(json.dumps(entry, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"t={t} zip={size} under={entry['under_ceiling']} split_docs={stats['split_docs']} pieces={stats['pieces']} "
        f"validator={entry.get('validator', {}).get('ok')} {entry['seconds']}s")
    return entry


def measured() -> dict[int, dict]:
    return {int(p.stem.split("-t")[-1].split(".")[0]): json.loads(p.read_text(encoding="utf-8"))
            for p in OUT.glob(f"{NAMES[MODE]}-t*.compose.json")}


def search(start: list[int], max_builds: int) -> dict:
    """Builds without the validator until the lowest T under the ceiling is bracketed to 4 tokens (or ``max_builds``), then validates it."""
    for t in start:
        if t not in measured():
            write(t, False)
    for _ in range(max_builds):
        m = measured()
        under = sorted(t for t, e in m.items() if e["under_ceiling"])
        over = sorted(t for t, e in m.items() if not e["under_ceiling"])
        if not under:
            nxt = max(over) * 2
        elif not [t for t in over if t < under[0]]:
            nxt = max(under[0] // 2, 32) if under[0] > 32 else None
        else:
            lo, hi = max(t for t in over if t < under[0]), under[0]
            if hi - lo <= 4:
                break
            a, b = m[lo], m[hi]                                   # linear in T between the bracket
            frac = (a["zip_bytes"] - CEILING) / max(a["zip_bytes"] - b["zip_bytes"], 1)
            nxt = int(round(lo + frac * (hi - lo)))
            nxt = min(max(nxt, lo + 1), hi - 1)
        if nxt is None or nxt in m:
            break
        write(nxt, False)
    m = measured()
    best = min(t for t, e in m.items() if e["under_ceiling"])
    entry = m[best] if m[best].get("validator") else write(best, True)
    over = [t for t, e in m.items() if not e["under_ceiling"] and t < best]
    rep = {"lowest_t_under_ceiling": best, "highest_t_over_below_it": max(over) if over else None,
           "tried": {str(t): {k: e[k] for k in ("zip_bytes", "under_ceiling", "split_docs", "pieces")} for t, e in sorted(m.items())}, "best": entry}
    (OUT / f"{NAMES[MODE]}.search.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("prepare", "write", "search"))
    ap.add_argument("--t", type=int, default=None)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--start", default="512,1024")
    ap.add_argument("--max-builds", type=int, default=8)
    ap.add_argument("--mode", choices=sorted(NAMES), default="split")
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    paths.assert_writable(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.command == "prepare":
        res = prepare()
    elif a.command == "write":
        res = write(a.t, not a.no_validate)
    else:
        if not (OUT / "split.cands.parquet").exists():
            prepare()
        res = search([int(x) for x in a.start.split(",")], a.max_builds)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
