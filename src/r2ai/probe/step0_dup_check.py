"""Kiem tra cap URL 'trung gan' (chi khac http/https, www...) co that su cung noi dung khong.
Mau nho (~30 cap, phan tang theo domain), UA ro rang, tuan thu robots.txt, <=2 request dong thoi/domain.
    python step0_dup_check.py  ->  out/dup_pairs_check.csv
"""
from r2ai import paths  # load env before libraries

import asyncio, re, hashlib
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit
import aiohttp, polars as pl, trafilatura

SEED, PER_DOM, MAX_DOM = 42, 3, 14
UA = "R2AI-Stage3-SampleProbe/0.1 (academic competition research; small sample; honors robots.txt)"
OUT = paths.OUT_DIR

def norm(u): return re.sub(r"/+$", "", re.sub(r"^https://www\.", "https://", re.sub(r"^http://", "https://", re.sub(r"#.*$", "", u)))).lower()

def pick():
    d = pl.read_parquet(paths.CORPUS_FILE)
    d = d.with_columns(
        pl.col("url").str.replace(r"#.*$", "").str.replace(r"^http://", "https://").str.replace(r"^https://www\.", "https://")
        .str.replace(r"/+$", "").str.to_lowercase().alias("k"),
        pl.col("url").str.extract(r"^https?://(?:www\.)?([^/?#]+)", 1).alias("domain"))
    d = d.filter(pl.len().over("k") == 2)
    g = d.sort("id").group_by("k", maintain_order=True).agg(pl.col("id"), pl.col("url"), pl.col("domain").first())
    doms = g.group_by("domain").len().sort("len", descending=True).head(MAX_DOM)["domain"].to_list()
    g = g.filter(pl.col("domain").is_in(doms)).with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).over("domain").alias("r")).filter(pl.col("r") < PER_DOM)
    return [{"domain": r["domain"], "ids": r["id"], "urls": r["url"]} for r in g.to_dicts()]

async def main():
    paths.auxiliary_disabled()
    rows = pick()
    sem = asyncio.Semaphore(16); dsem = {}; robots = {}
    async with aiohttp.ClientSession(headers={"User-Agent": UA}, timeout=aiohttp.ClientTimeout(total=15),
                                     connector=aiohttp.TCPConnector(ssl=False)) as s:
        async def allowed(u):
            o = "{0.scheme}://{0.netloc}".format(urlsplit(u))
            if o not in robots:
                rp = robotparser.RobotFileParser()
                try:
                    async with s.get(o + "/robots.txt") as r:
                        if r.status == 200: rp.parse((await r.text(errors="replace")).splitlines()); robots[o] = rp
                        else: robots[o] = None
                except Exception: robots[o] = None
            return robots[o] is None or robots[o].can_fetch(UA, u)
        async def get(u):
            if not await allowed(u): return {"status": "robots", "text": "", "final": None}
            ds = dsem.setdefault(urlsplit(u).netloc, asyncio.Semaphore(2))
            async with sem, ds:
                try:
                    async with s.get(u) as r:
                        b = await r.content.read(5_000_000)
                        t = trafilatura.extract(b, favor_recall=True) or "" if r.status == 200 else ""
                        return {"status": r.status, "text": t.strip(), "final": norm(str(r.url)), "raw": hashlib.md5(b).hexdigest()}
                except Exception as e: return {"status": type(e).__name__, "text": "", "final": None}
        res = await asyncio.gather(*(asyncio.gather(get(r["urls"][0]), get(r["urls"][1])) for r in rows))
    out = []
    for r, (a, b) in zip(rows, res):
        both = a["status"] == 200 and b["status"] == 200 and a["text"] and b["text"]
        out.append({"domain": r["domain"], "id_a": r["ids"][0], "id_b": r["ids"][1], "url_a": r["urls"][0], "url_b": r["urls"][1],
                    "status_a": a["status"], "status_b": b["status"], "same_final_url": a["final"] == b["final"] and a["final"] is not None,
                    "len_a": len(a["text"]), "len_b": len(b["text"]),
                    "text_identical": bool(both and a["text"] == b["text"]),
                    "raw_identical": bool(both and a.get("raw") == b.get("raw")), "comparable": bool(both)})
    df = pl.DataFrame(out); df.write_csv(OUT / "dup_pairs_check.csv")
    
    c = df.filter(pl.col("comparable"))
    print(f"pairs={df.height} comparable={c.height} text_identical={int(c['text_identical'].sum())} same_final_url={int(df['same_final_url'].sum())}")

asyncio.run(main())
