"""S38 CH-clean: the J3-thrC body without its boilerplate tail, for six hosts (Vu, 23/10).

``python -m r2ai.s38.clean discover`` -> ``scores/clean_markers.json`` (candidate markers per host)
``python -m r2ai.s38.clean apply``    -> ``scores/clean.parquet`` + ``clean.json`` (+ ``build --variant CH-clean``)

* body = the J3-thrC chunk of a carrier. A cut chunk must be a verbatim span of the canonical doc_text (validator
  S26): true when the J3 chunk itself is (its prefix is too), else checked on the cut text; failing cuts are skipped.
* lines = the body split at ``"\\n"``; line key = NFKC, lowercase, digits -> 0, whitespace collapsed, first 80 chars.
  Keys shorter than ``MIN_KEY_CHARS`` are ignored (bullets, "›", "-").
* marker of a host = a key found inside the last 30% of the body (line start >= 0.7 x body length) in >= 30% of
  the host's docs (unique doc_id), both thresholds from Vu.
* chunk = the body up to the start of the first marker line that lies in the last 30% of that body, trailing
  whitespace removed; a body without such a line keeps its J3 chunk byte for byte. Other hosts: unchanged.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from r2ai import paths
from r2ai.s38.score import scores_dir
from r2ai.s38.sources import J3_JSON_SHA256, J3_ZIP, _atomic_json, _atomic_parquet, describe, read_bundle, run_root

HOSTS = ("tamanhhospital.vn", "baophutho.vn", "a-hospital.com", "nhathuoclongchau.com.vn", "youmed.vn", "medlatec.vn")
TAIL_SHARE = 0.30
DOC_SHARE = 0.30
MIN_KEY_CHARS = 10
KEY_CHARS = 80
DIGITS = re.compile(r"\d")


def line_key(line: str) -> str:
    t = unicodedata.normalize("NFKC", line).lower()
    t = " ".join(DIGITS.sub("0", t).split())
    return t[:KEY_CHARS]


def lines_of(body: str) -> list[tuple[int, str]]:
    """(start offset, key) of every non-empty line of ``body``."""
    out, pos = [], 0
    for raw in body.split("\n"):
        key = line_key(raw)
        if len(key) >= MIN_KEY_CHARS:
            out.append((pos, key))
        pos += len(raw) + 1
    return out


def tail_keys(body: str) -> dict[str, int]:
    """Key -> first start offset among the lines in the last TAIL_SHARE of the body."""
    limit = (1 - TAIL_SHARE) * len(body)
    out: dict[str, int] = {}
    for start, key in lines_of(body):
        if start >= limit and key not in out:
            out[key] = start
    return out


def discover(bodies: dict[int, str]) -> list[dict[str, Any]]:
    """Markers of one host: ``bodies`` = doc_id -> body (one per doc)."""
    n = len(bodies)
    tail, anywhere = Counter(), Counter()
    raw_example: dict[str, str] = {}
    rel = defaultdict(list)
    for body in bodies.values():
        keys = tail_keys(body)
        tail.update(keys.keys())
        for k, s in keys.items():
            rel[k].append(s / max(len(body), 1))
        seen = set()
        for start, key in lines_of(body):
            if key not in seen:
                seen.add(key)
                anywhere[key] += 1
                if key not in raw_example:
                    raw_example[key] = body[start:body.find("\n", start) if "\n" in body[start:] else len(body)][:160]
    out = [{"key": k, "docs_tail": c, "share_tail": c / n, "share_any": anywhere[k] / n,
            "rel_pos_median": float(np.median(rel[k])), "example": raw_example[k]}
           for k, c in tail.items() if c / n >= DOC_SHARE]
    return sorted(out, key=lambda x: -x["share_tail"])


def cut_point(body: str, markers: set[str]) -> int | None:
    """Start of the first marker line in the last TAIL_SHARE of ``body``, or None."""
    limit = (1 - TAIL_SHARE) * len(body)
    for start, key in lines_of(body):
        if start >= limit and key in markers:
            return start
    return None


def load_carriers() -> pd.DataFrame:
    rows = read_bundle(paths.OUT_DIR / J3_ZIP, J3_JSON_SHA256)
    frame = pd.DataFrame([(int(r["id"]), int(c["doc_id"]), c["chunk_text"]) for r in rows for c in r["relevant_chunks"]],
                         columns=["query_id", "doc_id", "body"])
    bodies = pd.read_parquet(scores_dir() / "bodies.parquet", columns=["query_id", "doc_id", "host", "lang", "located",
                                                                       "body_tokens"])
    return frame.merge(bodies, on=["query_id", "doc_id"], how="left")


def run_discover(out: Path) -> dict[str, Any]:
    car = load_carriers()
    report = {"tail_share": TAIL_SHARE, "doc_share": DOC_SHARE, "min_key_chars": MIN_KEY_CHARS, "hosts": {}}
    for host in HOSTS:
        sub = car[car.host == host].drop_duplicates("doc_id")
        report["hosts"][host] = {"docs": len(sub), "markers": discover(dict(zip(sub.doc_id, sub.body)))}
    _atomic_json(out / "clean_markers.json", report)
    return report


def run_apply(out: Path) -> dict[str, Any]:
    started = time.monotonic()
    markers = {h: {m["key"] for m in v["markers"]} for h, v in json.loads(
        (out / "clean_markers.json").read_text(encoding="utf-8"))["hosts"].items()}
    car = load_carriers()
    from r2ai.gold_check.gold_check_common import load_tokenizer
    import pyarrow.parquet as pq

    tok = load_tokenizer()
    docs = pd.read_parquet(run_root() / "sources" / "docs.parquet", columns=["doc_id", "text_doc"])
    text_doc = dict(zip(docs.doc_id.astype(int), docs.text_doc.astype(int)))
    need = sorted({text_doc[int(d)] for d, h, ok in zip(car.doc_id, car.host, car.located) if h in markers and not ok})
    canon = dict(zip(*[pq.read_table(paths.DATA_DIR / "canonical-20261018" / "docs.parquet", columns=["doc_id", "doc_text"],
                                     filters=[("doc_id", "in", need)])[k].to_pylist() for k in ("doc_id", "doc_text")])) if need else {}
    rows, not_verbatim = [], Counter()
    for qid, did, host, located, body in zip(car.query_id, car.doc_id, car.host, car.located, car.body):
        if host in markers and markers[host]:
            at = cut_point(body, markers[host])
            if at is not None:
                text = body[:at].rstrip()
                if not text.strip():
                    continue
                if not located and text not in (canon.get(text_doc[int(did)]) or ""):
                    not_verbatim[host] += 1
                    continue
                rows.append({"query_id": int(qid), "doc_id": int(did), "host": host, "cut_at": at, "text": text,
                             "removed": body[len(text):], "located": bool(located)})
    frame = pd.DataFrame(rows)
    kept = [len(x) for x in tok(frame.text.tolist(), add_special_tokens=False)["input_ids"]] if len(frame) else []
    frame["tokens"] = np.array(kept, dtype=np.int32)
    frame = frame.merge(car[["query_id", "doc_id", "body_tokens", "body"]], on=["query_id", "doc_id"])
    frame["removed_tokens"] = frame.body_tokens - frame.tokens
    report: dict[str, Any] = {"carriers": len(car), "carriers_cut": len(frame), "hosts": {}}
    for host in HOSTS:
        sub = frame[frame.host == host]
        alln = car[car.host == host]
        ex = sub.drop_duplicates("doc_id").sort_values("doc_id").head(2)
        report["hosts"][host] = {
            "markers": len(markers.get(host, ())),
            "carriers": int(len(alln)), "docs": int(alln.doc_id.nunique()),
            "carriers_cut": int(len(sub)), "docs_cut": int(sub.doc_id.nunique()),
            "carriers_cut_from_unlocated_body": int((~sub.located.astype(bool)).sum()),
            "carriers_skipped_not_verbatim": int(not_verbatim[host]),
            "removed_tokens": describe(sub.drop_duplicates("doc_id").removed_tokens),
            "removed_share": describe((sub.removed_tokens / sub.body_tokens).to_numpy()) if len(sub) else {"n": 0},
            "examples": [{"doc_id": int(r.doc_id), "body_tokens": int(r.body_tokens), "kept_tokens": int(r.tokens),
                          "before_tail": r.body[max(0, r.cut_at - 200):r.cut_at + 400],
                          "after_tail": r.text[-200:]} for r in ex.itertuples()],
        }
    _atomic_parquet(frame[["query_id", "doc_id", "host", "text", "tokens", "removed_tokens"]], out / "clean.parquet")
    report["seconds"] = round(time.monotonic() - started, 1)
    _atomic_json(out / "clean.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["discover", "apply"])
    args = parser.parse_args(argv)
    out = scores_dir()
    for name in ("clean_markers.json", "clean.parquet", "clean.json"):
        paths.assert_writable(out / name)
        paths.assert_writable(out / (name + ".tmp"))
    report = run_discover(out) if args.command == "discover" else run_apply(out)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
