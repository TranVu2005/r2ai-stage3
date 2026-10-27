"""Coverage of corpus v2 by (lang, host) for the working notes.

Read-only: reads corpus ids, the canonical bundle, the 120ask segments (S37),
the PubMed en segment and the Pharmacity lots; writes one CSV.

    python docs/working-notes/scripts/build_coverage.py --root D:/GitHub/r2ai-stage3

A corpus id counts as covered when a document with text exists for it in the
final corpus used by the submissions (canonical-20261018 + data/s37_120ask +
Pharmacity lots + data/s37_en). "good" follows the quality flag of each bundle
(>= 200 code points, or >= 50 Han characters for zh; otherwise "thin").
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "coverage_by_host.csv")


def source_of(segment: str) -> str:
    s = (segment or "").lower()
    if "webmedqa" in s:
        return "webmedqa"
    if "pharmacity" in s:
        return "pharmacity"
    if "-cc-" in s or s.startswith("cc-"):
        return "common_crawl"
    if "s37" in s:
        return "vps_120ask"
    if "en-pubmed" in s:
        return "pubmed"
    return "direct_crawl"


def read(path: str, cols: list[str]) -> pd.DataFrame:
    return pq.read_table(path, columns=cols).to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/GitHub/r2ai-stage3")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    data = os.path.join(a.root, "data")

    ids = read(os.path.join(data, "corpus_v2_ids.parquet"), ["id", "lang", "host"])
    ids["id"] = ids["id"].astype(str)

    cols = ["doc_id", "quality", "segment"]
    parts = [read(os.path.join(data, "canonical-20261018", "docs.parquet"), cols)]
    parts += [read(p, cols) for p in sorted(glob.glob(os.path.join(data, "s37_120ask", "*", "docs.parquet")))]
    for p in sorted(glob.glob(os.path.join(data, "s37_en", "*", "docs.parquet"))):
        d = read(p, ["doc_id", "quality"])
        d["segment"] = "seg37900-en-pubmed"
        parts.append(d)
    lots = os.path.join(a.root, "out", "runs", "s27-pharmacity", "handoff", "cc-s27-pharmacity", "lots")
    for p in sorted(glob.glob(os.path.join(lots, "*", "docs", "docs.parquet"))):
        # S15 lot layout: one row per page, doc_ids lists the corpus ids of the page.
        d = read(p, ["doc_ids", "body"]).explode("doc_ids").rename(columns={"doc_ids": "doc_id"})
        d["quality"] = d["body"].fillna("").str.split().str.join(" ").str.len().ge(200).map({True: "good", False: "thin"})
        d["segment"] = "cc-s27-pharmacity"
        parts.append(d[cols])
    docs = pd.concat(parts, ignore_index=True)
    docs["doc_id"] = docs["doc_id"].astype(str)
    docs["source"] = docs["segment"].map(source_of)
    # One row per corpus id: a later bundle (120ask, Pharmacity) wins over canonical,
    # good wins over thin.
    docs["good"] = (docs["quality"] == "good").astype(int)
    docs = docs.sort_values(["doc_id", "good"]).drop_duplicates("doc_id", keep="last")

    m = ids.merge(docs[["doc_id", "good", "source"]], left_on="id", right_on="doc_id", how="left")
    m["has_doc"] = m["doc_id"].notna().astype(int)
    m["good"] = m["good"].fillna(0).astype(int)
    rows = []
    for (lang, host), g in m.groupby(["lang", "host"]):
        r = {"lang": lang, "host": host, "urls": len(g), "docs": int(g.has_doc.sum()), "good": int(g.good.sum())}
        for src in ["direct_crawl", "common_crawl", "vps_120ask", "webmedqa", "pharmacity", "pubmed"]:
            r[f"good_{src}"] = int(((g.source == src) & (g.good == 1)).sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    out["good_pct"] = (100 * out.good / out.urls).round(2)
    out = out.sort_values(["lang", "urls"], ascending=[True, False])
    tot = out.groupby("lang")[[c for c in out.columns if c not in ("lang", "host", "good_pct")]].sum().reset_index()
    tot["host"] = "(total)"
    tot["good_pct"] = (100 * tot.good / tot.urls).round(2)
    out = pd.concat([out, tot[out.columns]], ignore_index=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out, index=False)
    print(tot.to_string(index=False))
    print(f"wrote {a.out} ({len(out)} rows)")


if __name__ == "__main__":
    main()
