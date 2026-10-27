"""S43 EN-hedge check, streamed from the ZIPs: per query ``relevant_docs`` of the hedge ZIP = those of the base ZIP (same ids, same order)
followed by exactly the hedge docs of that query (``hedge.parquet``, en_rank order); also the chunk doc ids of the appended docs.

    python scripts/s43/zip_docs_prefix.py BASE.zip HEDGE.zip hedge.parquet
"""
import json
import re
import sys
import zipfile

import pandas as pd

DOCS = re.compile(rb'"id":(\d+),"relevant_docs":\[([0-9,]*)\]')


def rows(path: str):
    with zipfile.ZipFile(path) as z, z.open(z.infolist()[0]) as f:
        for line in f:
            m = DOCS.search(line)
            if m:
                yield int(m.group(1)), [int(x) for x in m.group(2).split(b",") if x], line


def main() -> int:
    base = {q: d for q, d, _ in rows(sys.argv[1])}
    h = pd.read_parquet(sys.argv[3]).sort_values(["query_id", "en_rank"])
    want = {int(q): g.doc_id.astype(int).tolist() for q, g in h.groupby("query_id")}
    bad_prefix, bad_tail, bad_chunks, n, added = [], [], [], 0, {}
    for q, docs, line in rows(sys.argv[2]):
        n += 1
        b = base.get(q)
        if b is None or docs[:len(b)] != b:
            bad_prefix.append(q)
            continue
        tail = docs[len(b):]
        added[q] = len(tail)
        if tail != want.get(q, []):
            bad_tail.append(q)
        row = json.loads(line.strip().rstrip(b","))
        chunk_docs = [c["doc_id"] for c in row["relevant_chunks"]][-len(tail):] if tail else []
        if chunk_docs != tail:
            bad_chunks.append(q)
    s = pd.Series(added)
    res = {"queries_base": len(base), "queries_hedge": n, "vi_zh_prefix_identical": not bad_prefix and n == len(base),
           "tail_equals_hedge": not bad_tail, "one_chunk_per_en_doc_at_end": not bad_chunks,
           "first_bad": (bad_prefix + bad_tail + bad_chunks)[:10], "en_added": int(s.sum()), "en_per_query_mean": round(float(s.mean()), 3),
           "queries_ge1_en": int((s > 0).sum()), "en_per_query_dist": {int(k): int(v) for k, v in s.value_counts().sort_index().items()}}
    print(json.dumps(res))
    return 0 if res["vi_zh_prefix_identical"] and res["tail_equals_hedge"] and res["one_chunk_per_en_doc_at_end"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
