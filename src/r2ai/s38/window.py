"""S38 body-cut variants: windows / prefixes of the J3-thrC body (BGE-M3 tokens), see ``SPECS``.

``python -m r2ai.s38.window --variant CH-pre1024 --variant CH-cap2048 --variant CH-full+w768`` (CPU, after
``score select --variant CH-t256-1``) -> ``out/runs/s38-chunk-metric-v2/scores/windows.parquet`` (rows of other
variants kept; ``text`` null = the J3 chunk unchanged), ``windows.json``, ``bodies.parquet``, ``host_bodies.json``.

Per J3-thrC carrier:
* body = the J3-thrC chunk text of the carrier (the whole body it shipped); located verbatim in the carrier's
  canonical doc_text. Not found (old P1 text): the canonical body span (answer span, else body span; zh: doc_text).
  Body length = raw BGE-M3 tokens, no special tokens.
* CH-win<W>: body <= W tokens keeps its J3 chunk, else a W-token window around the best t256 chunk.
* CH-pre1024: body <= 1,024 tokens keeps its J3 chunk, else its first 1,024 tokens, the end moved back to the
  last sentence end (at most 256 tokens).
* CH-cap2048: body <= 2,048 tokens keeps its J3 chunk, else a 2,048-token window around the best t256 chunk.
* CH-full+w768: body <= 1,536 tokens keeps its J3 chunk; else two chunks: the J3 chunk and a 768-token window
  around the best t256 chunk.
Window: W consecutive tokens centred on the CH-t256-1 chunk (clamped inside the body), then cut on sentence
boundaries when possible: the start moves forward to the next sentence start and the end back to the last
sentence end, each by at most W/4 tokens and never into the chunk. Boundaries: after . ! ? … ; : followed by
whitespace (the chunker's rule), after 。！？；, and at line breaks. Leading/trailing whitespace is stripped.
``--hosts`` (offline report): body length per host and, for the 10 hosts with the longest median body, the
last 300 characters of one median-length body.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s38.score import scores_dir
from r2ai.s38.sources import J3_JSON_SHA256, J3_ZIP, _atomic_json, _atomic_parquet, describe, read_bundle, run_root

SENT_END = re.compile(r"[.!?…;:](?=\s)|[。！？；]")
BREAK = re.compile(r"\n+")
# variant -> (rule, width W, threshold T): a body of <= T tokens keeps its J3 chunk
SPECS = {
    "CH-win384": ("window", 384, 384),
    "CH-win768": ("window", 768, 768),
    "CH-pre1024": ("prefix", 1024, 1024),
    "CH-cap2048": ("window", 2048, 2048),
    "CH-full+w768": ("full+window", 768, 1536),
}
TOP_HOSTS = 10
TAIL_CHARS = 300
HOST_MIN_CARRIERS = 50


def boundaries(text: str) -> tuple[np.ndarray, np.ndarray]:
    """(sentence starts, sentence ends) as char offsets of ``text``."""
    ends, starts = {0}, {len(text)}
    for m in SENT_END.finditer(text):
        ends.add(m.end())
        k = m.end()
        while k < len(text) and text[k].isspace():
            k += 1
        starts.add(k)
    for m in BREAK.finditer(text):
        ends.add(m.start())
        starts.add(m.end())
    starts |= {0}
    ends |= {len(text)}
    return np.array(sorted(starts)), np.array(sorted(ends))


def window(text: str, offsets: Sequence[tuple[int, int]], chunk: tuple[int, int] | None, width: int) -> dict[str, Any]:
    """Window of ``text`` (char span) for ``width`` tokens; ``offsets`` = token char spans, ``chunk`` = char span or None."""
    n = len(offsets)
    if n <= width:
        return {"start": 0, "end": len(text), "mode": "whole", "tokens": n, "snap_start": False, "snap_end": False}
    starts_tok = np.array([s for s, _ in offsets])
    ends_tok = np.array([e for _, e in offsets])
    if chunk is not None and chunk[0] < chunk[1]:
        t0 = int(np.searchsorted(ends_tok, chunk[0], side="right"))          # first token ending after the chunk start
        t1 = max(t0, int(np.searchsorted(starts_tok, chunk[1], side="left")) - 1)
        centre = (t0 + t1 + 1) / 2
    else:
        t0 = t1 = None
        centre = n / 2
    ws = int(np.clip(round(centre - width / 2), 0, n - width))
    we = ws + width
    a, b = int(starts_tok[ws]), int(ends_tok[we - 1])
    slack = width // 4
    a_max = int(starts_tok[min(ws + slack, n - 1)])
    b_min = int(ends_tok[max(we - 1 - slack, 0)])
    if t0 is not None and t1 - t0 + 1 <= width:                                  # never cut into the chunk
        a_max = min(a_max, max(a, chunk[0]))
        b_min = max(b_min, min(b, chunk[1]))
    sent_starts, sent_ends = boundaries(text)
    s_ok = sent_starts[(sent_starts >= a) & (sent_starts <= a_max)]
    e_ok = sent_ends[(sent_ends <= b) & (sent_ends >= b_min)]
    snap_a, snap_b = len(s_ok) > 0 and ws > 0, len(e_ok) > 0 and we < n
    a2 = int(s_ok[0]) if snap_a else a
    b2 = int(e_ok[-1]) if snap_b else b
    if b2 <= a2:
        a2, b2, snap_a, snap_b = a, b, False, False
    piece = text[a2:b2]
    lead = len(piece) - len(piece.lstrip())
    trail = len(piece) - len(piece.rstrip())
    a2, b2 = a2 + lead, b2 - trail
    return {"start": a2, "end": b2, "mode": "window", "tokens": None, "snap_start": bool(snap_a), "snap_end": bool(snap_b)}


def prefix(text: str, offsets: Sequence[tuple[int, int]], width: int) -> dict[str, Any]:
    """The first ``width`` tokens of ``text``; the end moves back to the last sentence end (at most width/4 tokens)."""
    n = len(offsets)
    if n <= width:
        return {"start": 0, "end": len(text), "mode": "whole", "snap_start": False, "snap_end": False}
    ends_tok = np.array([e for _, e in offsets])
    b = int(ends_tok[width - 1])
    b_min = int(ends_tok[max(width - 1 - width // 4, 0)])
    _, sent_ends = boundaries(text)
    e_ok = sent_ends[(sent_ends <= b) & (sent_ends >= b_min)]
    b2 = int(e_ok[-1]) if len(e_ok) else b
    piece = text[:b2]
    return {"start": len(piece) - len(piece.lstrip()), "end": len(piece.rstrip()), "mode": "prefix",
            "snap_start": False, "snap_end": bool(len(e_ok))}


def pieces_for(rule: str, width: int, threshold: int, body: str, offsets, chunk) -> list[tuple | None]:
    """Chunks of one carrier: ``None`` = the J3 chunk unchanged, else (mode, start, end, snap_start, snap_end)."""
    if len(offsets) <= threshold:
        return [None]
    res = prefix(body, offsets, width) if rule == "prefix" else window(body, offsets, chunk, width)
    span = (res["mode"], res["start"], res["end"], res["snap_start"], res["snap_end"])
    return [None, span] if rule == "full+window" else [span]


def _body_span(doc_text: str, lang: str, spans: pd.DataFrame) -> tuple[int, int]:
    """Canonical body span as ``r2ai.s29.build._body_span`` (answer span, else body span; zh: the doc_text)."""
    if lang == "zh" or spans.empty:
        return 0, len(doc_text)
    for field in ("answer", "body"):
        sub = spans[spans.field == field]
        if len(sub):
            return int(sub.char_start.min()), int(sub.char_end.max())
    return 0, len(doc_text)


def _changed_share(frame: pd.DataFrame, key: str) -> dict[str, dict[str, float]]:
    g = frame.groupby(key).changed.agg(["mean", "size"])
    return {str(k): {"changed": float(r["mean"]), "carriers": int(r["size"])} for k, r in g.iterrows()}


def build_windows(out: Path, variants: Sequence[str]) -> dict[str, Any]:
    started = time.monotonic()
    rows = read_bundle(paths.OUT_DIR / J3_ZIP, J3_JSON_SHA256)
    src = run_root() / "sources"
    docs = pd.read_parquet(src / "docs.parquet", columns=["doc_id", "owner", "text_doc", "lang", "host"])
    pieces = pd.read_parquet(src / "pieces.parquet", columns=["doc_id", "kind", "field", "char_start", "char_end"])
    sel = pd.read_parquet(scores_dir() / "selection.parquet")
    best = sel[sel.variant == "CH-t256-1"].set_index(["query_id", "doc_id"]).piece_row.to_dict()
    t256 = pieces[pieces.kind == "t256"]
    canonical = paths.DATA_DIR / "canonical-20261018"
    import pyarrow.parquet as pq

    need = sorted(set(docs.text_doc.astype(int)))
    dt = pq.read_table(canonical / "docs.parquet", columns=["doc_id", "doc_text"], filters=[("doc_id", "in", need)])
    doc_text = dict(zip(dt["doc_id"].to_pylist(), dt["doc_text"].to_pylist()))
    del dt
    info = docs.set_index("doc_id")
    text_doc, lang_of, host_of = info.text_doc.to_dict(), info.lang.to_dict(), info.host.to_dict()
    spans_by_doc = {d: g for d, g in t256.groupby("doc_id", sort=False)}
    p_start, p_end = pieces.char_start.to_numpy(), pieces.char_end.to_numpy()

    from r2ai.gold_check.gold_check_common import load_tokenizer

    tok = load_tokenizer()
    carriers = [(int(r["id"]), int(c["doc_id"]), c["chunk_text"]) for r in rows for c in r["relevant_chunks"]]
    del rows
    out_rows: list[dict[str, Any]] = []
    body_rows: list[dict[str, Any]] = []
    stats = {"located": 0, "fallback_body": 0, "chunk_outside_body": 0, "no_t256_choice": 0}
    for begin in range(0, len(carriers), 5_000):
        part = carriers[begin:begin + 5_000]
        bodies, base = [], []
        for qid, d, j3 in part:
            text = doc_text[int(text_doc[d])] or ""
            at = text.find(j3)
            if at >= 0:
                stats["located"] += 1
                bodies.append(j3)
                base.append((at, True))
            else:
                stats["fallback_body"] += 1
                s, e = _body_span(text, str(lang_of[d]), spans_by_doc.get(d, t256.iloc[:0]))
                bodies.append(text[s:e])
                base.append((s, False))
        enc = tok(bodies, add_special_tokens=False, return_offsets_mapping=True)
        for k, ((qid, d, j3), body, (b0, located)) in enumerate(zip(part, bodies, base)):
            offsets = enc["offset_mapping"][k]
            chunk = None
            pr = best.get((qid, d))
            if pr is not None:
                cs, ce = int(p_start[pr]) - b0, int(p_end[pr]) - b0
                if ce <= 0 or cs >= len(body):
                    stats["chunk_outside_body"] += 1
                else:
                    chunk = (max(cs, 0), min(ce, len(body)))
            else:
                stats["no_t256_choice"] += 1
            body_rows.append({"query_id": qid, "doc_id": d, "lang": lang_of[d], "host": host_of[d],
                              "body_tokens": len(offsets), "body_chars": len(j3), "located": located,
                              "tail": j3[-TAIL_CHARS:]})
            for name in variants:
                rule, width, threshold = SPECS[name]
                for rank, span in enumerate(pieces_for(rule, width, threshold, body, offsets, chunk), 1):
                    if span is None or (span[0] == "whole" and located):
                        out_rows.append({"variant": name, "query_id": qid, "doc_id": d, "rank": rank, "text": None,
                                         "mode": "j3", "snap_start": False, "snap_end": False})
                    else:
                        mode, a, b, sa, sb = span
                        out_rows.append({"variant": name, "query_id": qid, "doc_id": d, "rank": rank,
                                         "text": body[a:b], "mode": mode, "snap_start": sa, "snap_end": sb})
    frame = pd.DataFrame(out_rows)
    bodies_f = pd.DataFrame(body_rows)
    has = frame.text.notna().to_numpy()
    texts = frame.text[has].tolist()
    got: list[int] = []
    for b in range(0, len(texts), 20_000):
        got += [len(x) for x in tok(texts[b:b + 20_000], add_special_tokens=False)["input_ids"]]
    frame = frame.merge(bodies_f[["query_id", "doc_id", "body_tokens", "located", "lang", "host"]], on=["query_id", "doc_id"], how="left")
    tokens = frame.body_tokens.to_numpy().copy()
    has = frame.text.notna().to_numpy()                      # same rows, merge keeps the left order
    tokens[has] = got
    frame["tokens"] = tokens.astype(np.int32)

    report = json.loads((out / "windows.json").read_text(encoding="utf-8")) if (out / "windows.json").exists() else {}
    bt = bodies_f.body_tokens.to_numpy()
    report.update({"carriers": len(carriers), **stats, "body_tokens": describe(bt),
                   "body_share_le": {str(w): float((bt <= w).mean()) for w in (384, 768, 1024, 1536, 2048)}})
    report.setdefault("variants", {})
    top_hosts = bodies_f.host.value_counts().head(TOP_HOSTS).index.tolist()
    for name in variants:
        sub = frame[frame.variant == name]
        rule, width, threshold = SPECS[name]
        sub = sub.assign(new=sub["mode"] != "j3")
        per = sub.groupby(["query_id", "doc_id"], sort=False).agg(changed=("new", "any"), lang=("lang", "first"),
                                                                  host=("host", "first")).reset_index()
        new = sub[sub.new]
        report["variants"][name] = {
            "rule": rule, "width": width, "threshold_tokens": threshold, "chunks": len(sub),
            "carriers_changed": int(per.changed.sum()), "carriers_changed_share": float(per.changed.mean()),
            "docs_changed_share": float(per.groupby("doc_id").changed.max().mean()),
            "changed_by_lang": _changed_share(per, "lang"),
            "changed_top_hosts": _changed_share(per[per.host.isin(top_hosts)], "host"),
            "tokens_all": describe(sub.tokens), "tokens_new_chunks": describe(new.tokens),
            "over_width": int((new.tokens > width).sum()),
            "snap_end_share": float(new.snap_end.mean()) if len(new) else None,
            "new_chunks_from_fallback_body": int((~new.located).sum()),
        }
    target = out / "windows.parquet"
    keep = ["variant", "query_id", "doc_id", "rank", "text", "mode", "snap_start", "snap_end", "tokens", "body_tokens"]
    frame = frame[keep]
    if target.exists():
        old = pd.read_parquet(target)
        frame = pd.concat([old[~old.variant.isin(variants)], frame], ignore_index=True)
    _atomic_parquet(frame, target)
    _atomic_parquet(bodies_f, out / "bodies.parquet")
    report["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / "windows.json", report)
    return report


def host_report(out: Path) -> dict[str, Any]:
    """Body length per host for the carriers; the TOP_HOSTS hosts (>= HOST_MIN_CARRIERS carriers) with the longest
    median body, each with the last TAIL_CHARS characters of its median-length body."""
    bodies = pd.read_parquet(out / "bodies.parquet")
    g = bodies.groupby("host").body_tokens
    table = pd.DataFrame({"carriers": g.size(), "docs": bodies.groupby("host").doc_id.nunique(),
                          "p10": g.quantile(0.1), "median": g.median(), "p90": g.quantile(0.9), "max": g.max(),
                          "share_gt_1024": g.apply(lambda x: float((x > 1024).mean())),
                          "lang": bodies.groupby("host").lang.agg(lambda x: x.mode().iat[0])})
    table = table.sort_values("carriers", ascending=False)
    longest = table[table.carriers >= HOST_MIN_CARRIERS].sort_values("median", ascending=False).head(TOP_HOSTS)
    examples = {}
    for host, row in longest.iterrows():
        sub = bodies[bodies.host == host].drop_duplicates("doc_id")
        pick = sub.iloc[int(np.argmin(np.abs(sub.body_tokens.to_numpy() - row["median"])))]
        examples[host] = {"doc_id": int(pick.doc_id), "body_tokens": int(pick.body_tokens), "tail": str(pick["tail"])}
    report = {"hosts": int(len(table)), "min_carriers_for_longest": HOST_MIN_CARRIERS,
              "top_hosts_by_carriers": json.loads(table.head(20).reset_index().to_json(orient="records")),
              "longest_median": json.loads(longest.reset_index().to_json(orient="records")), "tail_examples": examples}
    _atomic_parquet(table.reset_index(), out / "host_bodies.parquet")
    _atomic_json(out / "host_bodies.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", action="append", choices=list(SPECS), default=None)
    parser.add_argument("--hosts", action="store_true", help="only the per-host body report (needs bodies.parquet)")
    args = parser.parse_args(argv)
    out = scores_dir()
    for name in ("windows.parquet", "windows.json", "bodies.parquet", "host_bodies.parquet", "host_bodies.json"):
        paths.assert_writable(out / name)
        paths.assert_writable(out / (name + ".tmp"))
    if not args.hosts:
        variants = args.variant or ["CH-win768", "CH-win384"]
        report = build_windows(out, variants)
        print(json.dumps({k: v for k, v in report.items() if k != "variants"}, ensure_ascii=False, indent=1))
        print(json.dumps({k: report["variants"][k] for k in variants}, ensure_ascii=False, indent=1))
    print(json.dumps(host_report(out), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
