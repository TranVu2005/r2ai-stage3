"""S38 CH-win<W>: one contiguous window of at most W BGE-M3 tokens of the body around the best t256 chunk.

``python -m r2ai.s38.window [--width 384 --width 768]`` (CPU, after ``score select --variant CH-t256-1``)
-> ``out/runs/s38-chunk-metric-v2/scores/windows.parquet`` + ``windows.json``

Per J3-thrC carrier (Vu, 23/10):
* body = the J3-thrC chunk text of the carrier (the whole body it shipped); located verbatim in the carrier's
  canonical doc_text. Not found (old P1 text): the canonical body span (answer span, else body span; zh: doc_text).
* body <= W tokens (raw BGE-M3, no special tokens): the J3 chunk text unchanged.
* longer: W consecutive tokens centred on the CH-t256-1 chunk (clamped inside the body), then cut on sentence
  boundaries when possible: the start moves forward to the next sentence start and the end back to the last
  sentence end, each by at most W/4 tokens and never into the chunk. Boundaries: after . ! ? … ; : followed by
  whitespace (the chunker's rule), after 。！？；, and at line breaks. Leading/trailing whitespace is stripped,
  so a window never exceeds W tokens.
Also reports the body length distribution of the carriers (share <= 384, <= 768, > 768 tokens).
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

WIDTHS = (384, 768)
SENT_END = re.compile(r"[.!?…;:](?=\s)|[。！？；]")
BREAK = re.compile(r"\n+")


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


def _body_span(doc_text: str, lang: str, spans: pd.DataFrame) -> tuple[int, int]:
    """Canonical body span as ``r2ai.s29.build._body_span`` (answer span, else body span; zh: the doc_text)."""
    if lang == "zh" or spans.empty:
        return 0, len(doc_text)
    for field in ("answer", "body"):
        sub = spans[spans.field == field]
        if len(sub):
            return int(sub.char_start.min()), int(sub.char_end.max())
    return 0, len(doc_text)


def build_windows(out: Path, widths: Sequence[int]) -> dict[str, Any]:
    started = time.monotonic()
    rows = read_bundle(paths.OUT_DIR / J3_ZIP, J3_JSON_SHA256)
    src = run_root() / "sources"
    docs = pd.read_parquet(src / "docs.parquet", columns=["doc_id", "owner", "text_doc", "lang"])
    pieces = pd.read_parquet(src / "pieces.parquet", columns=["doc_id", "kind", "field", "char_start", "char_end"])
    sel = pd.read_parquet(scores_dir() / "selection.parquet")
    best = sel[sel.variant == "CH-t256-1"].set_index(["query_id", "doc_id"]).piece_row
    t256 = pieces[pieces.kind == "t256"]
    canonical = paths.DATA_DIR / "canonical-20261018"
    import pyarrow.parquet as pq

    need = sorted(set(docs.text_doc.astype(int)))
    dt = pq.read_table(canonical / "docs.parquet", columns=["doc_id", "doc_text"], filters=[("doc_id", "in", need)])
    doc_text = dict(zip(dt["doc_id"].to_pylist(), dt["doc_text"].to_pylist()))
    del dt
    info = docs.set_index("doc_id")
    spans_by_doc = {d: g for d, g in t256.groupby("doc_id", sort=False)}

    from r2ai.gold_check.gold_check_common import load_tokenizer

    tok = load_tokenizer()
    carriers = [(int(r["id"]), int(c["doc_id"]), c["chunk_text"]) for r in rows for c in r["relevant_chunks"]]
    del rows
    out_rows: list[dict[str, Any]] = []
    stats = {"located": 0, "fallback_body": 0, "chunk_outside_body": 0, "no_t256_choice": 0}
    body_tokens = np.zeros(len(carriers), dtype=np.int64)
    for begin in range(0, len(carriers), 5_000):
        part = carriers[begin:begin + 5_000]
        bodies, base = [], []
        for qid, d, j3 in part:
            text = doc_text[int(info.at[d, "text_doc"])] or ""
            at = text.find(j3)
            if at >= 0:
                stats["located"] += 1
                bodies.append(j3)
                base.append((at, True))
            else:
                stats["fallback_body"] += 1
                s, e = _body_span(text, str(info.at[d, "lang"]), spans_by_doc.get(d, t256.iloc[:0]))
                bodies.append(text[s:e])
                base.append((s, False))
        enc = tok(bodies, add_special_tokens=False, return_offsets_mapping=True)
        for k, ((qid, d, j3), body, (b0, located)) in enumerate(zip(part, bodies, base)):
            offsets = enc["offset_mapping"][k]
            body_tokens[begin + k] = len(offsets)
            key = (qid, d)
            chunk = None
            if key in best.index:
                pr = int(best.at[key])
                cs, ce = int(pieces.char_start.iat[pr]) - b0, int(pieces.char_end.iat[pr]) - b0
                if ce <= 0 or cs >= len(body):
                    stats["chunk_outside_body"] += 1
                else:
                    chunk = (max(cs, 0), min(ce, len(body)))
            else:
                stats["no_t256_choice"] += 1
            for w in widths:
                res = window(body, offsets, chunk, w)
                text_w = j3 if res["mode"] == "whole" and located else body[res["start"]:res["end"]]
                out_rows.append({"variant": f"CH-win{w}", "query_id": qid, "doc_id": d, "text": text_w,
                                 "mode": res["mode"], "located": located, "snap_start": res["snap_start"],
                                 "snap_end": res["snap_end"], "body_tokens": len(offsets)})
    frame = pd.DataFrame(out_rows)
    lens = []
    texts = frame.text.tolist()
    for begin in range(0, len(texts), 20_000):
        lens += [len(x) for x in tok(texts[begin:begin + 20_000], add_special_tokens=False)["input_ids"]]
    frame["tokens"] = np.array(lens, dtype=np.int32)
    _atomic_parquet(frame, out / "windows.parquet")
    bt = body_tokens
    report = {"carriers": len(carriers), **stats,
              "body_tokens": describe(bt),
              "body_share_le_384": float((bt <= 384).mean()), "body_share_le_768": float((bt <= 768).mean()),
              "body_share_gt_768": float((bt > 768).mean()),
              "body_share_by_lang": {lang: {"le_384": float((bt[m] <= 384).mean()), "le_768": float((bt[m] <= 768).mean()),
                                            "n": int(m.sum())}
                                     for lang in ("vi", "zh")
                                     for m in [np.array([info.at[d, "lang"] == lang for _, d, _ in carriers])]},
              "variants": {}}
    for w in widths:
        sub = frame[frame.variant == f"CH-win{w}"]
        win = sub[sub["mode"] == "window"]
        report["variants"][f"CH-win{w}"] = {
            "chunks": len(sub), "whole_body": int((sub["mode"] == "whole").sum()), "windowed": len(win),
            "tokens": describe(sub.tokens), "tokens_windowed": describe(win.tokens),
            "over_width": int((sub.tokens > w).sum()),
            "snap_start_share": float(win.snap_start.mean()) if len(win) else None,
            "snap_end_share": float(win.snap_end.mean()) if len(win) else None}
    report["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / "windows.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, action="append", default=None)
    args = parser.parse_args(argv)
    out = scores_dir()
    for name in ("windows.parquet", "windows.parquet.tmp", "windows.json", "windows.json.tmp"):
        paths.assert_writable(out / name)
    report = build_windows(out, args.width or list(WIDTHS))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
