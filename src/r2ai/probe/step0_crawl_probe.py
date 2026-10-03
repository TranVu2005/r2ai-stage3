"""Step 0b - crawl thu co kiem soat (~300 URL) + tong hop + bao cao out/profile_report.md.

    python step0_crawl_probe.py            # crawl (neu chua co out/crawl_sample.csv) + report
    python step0_crawl_probe.py --recrawl  # crawl lai
    python step0_crawl_probe.py --bench    # (tuy chon) do toc do embed BGE-M3 fp16 tren GPU local (tai model ~2GB)

Rang buoc: async, timeout 15s, <=2 request dong thoi/domain, UA ro rang, tuan thu robots.txt.
Phai chay `step0_profile.py` truoc.
"""
from __future__ import annotations

from r2ai.paths import OUT_DIR, RAW_DIR, auxiliary_disabled, data_label, resolve_legacy_artifact

import argparse
import asyncio
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlsplit

import aiohttp
import polars as pl

SEED = 42
OUT = OUT_DIR
RAW_DIR = OUT / "sample_raw"
UA = "R2AI-Stage3-SampleProbe/0.1 (academic competition research; ~300 URL sample; honors robots.txt)"
TIMEOUT = 15
PER_DOMAIN = 2
GLOBAL_CONC = 24
MAX_BYTES = 5_000_000
MIN_TEXT = 200
TOP_N, PER_TOP, TAIL_N = 30, 8, 60

BLOCK_PAT = re.compile(
    r"captcha|验证码|cf-chl|cf_chl|just a moment|attention required|access denied|滑动验证|安全验证|"
    r"verify you are human|are you a robot|请完成验证|访问频繁|403 forbidden",
    re.I)
PAYWALL_PAT = re.compile(r"paywall|subscribe to (?:read|continue)|đăng nhập để (?:xem|đọc)|vui lòng đăng nhập|请登录|登录后查看|会员专享", re.I)


# ---------------------------------------------------------------- sample
def build_sample() -> pl.DataFrame:
    df = pl.read_parquet(OUT / "corpus_profiled.parquet", columns=["id", "url", "domain", "lang_guess", "slug_type"])
    top = (df.group_by("domain").len().sort("len", descending=True).head(TOP_N))["domain"].to_list()
    a = (df.filter(pl.col("domain").is_in(top))
         .with_columns(pl.int_range(pl.len()).shuffle(seed=SEED).over("domain").alias("_r"))
         .filter(pl.col("_r") < PER_TOP).drop("_r").with_columns(pl.lit("top30").alias("stratum")))
    b = (df.filter(~pl.col("domain").is_in(top)).sample(TAIL_N, seed=SEED)
         .with_columns(pl.lit("tail").alias("stratum")))
    return pl.concat([a, b]).sort("domain", "id")


# ---------------------------------------------------------------- fetch
class Prober:
    def __init__(self) -> None:
        self.dom_sem: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(PER_DOMAIN))
        self.glob = asyncio.Semaphore(GLOBAL_CONC)
        self.robots: dict[str, robotparser.RobotFileParser | None] = {}
        self.robots_lock: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def allowed(self, sess: aiohttp.ClientSession, url: str) -> tuple[bool, str]:
        u = urlsplit(url)
        origin = f"{u.scheme}://{u.netloc}"
        async with self.robots_lock[origin]:
            if origin not in self.robots:
                rp = robotparser.RobotFileParser()
                try:
                    async with self.dom_sem[u.netloc]:
                        async with sess.get(origin + "/robots.txt") as r:
                            if r.status == 200:
                                rp.parse((await r.text(errors="replace")).splitlines())
                                self.robots[origin] = rp
                            else:  # 4xx/5xx: coi nhu khong co robots (cho phep)
                                self.robots[origin] = None
                except Exception:
                    self.robots[origin] = None
        rp = self.robots[origin]
        if rp is None:
            return True, "no_robots"
        return rp.can_fetch(UA, url), "robots_ok"

    async def fetch(self, sess: aiohttp.ClientSession, row: dict) -> dict:
        rec = {"id": row["id"], "domain": row["domain"], "url": row["url"], "stratum": row["stratum"],
               "status": None, "content_type": None, "final_url": None, "n_redirects": 0,
               "bytes": 0, "elapsed_s": None, "error": None, "robots": None, "raw_path": None}
        async with self.glob:
            ok, why = await self.allowed(sess, row["url"])
            rec["robots"] = why if ok else "DISALLOWED"
            if not ok:
                rec["error"] = "robots_disallowed"
                return rec
            async with self.dom_sem[urlsplit(row["url"]).netloc]:
                t0 = time.perf_counter()
                try:
                    async with sess.get(row["url"], allow_redirects=True) as r:
                        rec["status"] = r.status
                        rec["content_type"] = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                        rec["final_url"] = str(r.url)
                        rec["n_redirects"] = len(r.history)
                        body = await r.content.read(MAX_BYTES)
                        rec["bytes"] = len(body)
                        rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
                        if body:
                            ext = "pdf" if "pdf" in rec["content_type"] or body[:4] == b"%PDF" else "html"
                            p = RAW_DIR / f"{row['id']}.{ext}"
                            p.write_bytes(body)
                            rec["raw_path"] = data_label(p).replace("\\", "/")
                except asyncio.TimeoutError:
                    rec["error"] = "timeout"
                    rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:100]}"
                    rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
        return rec


async def crawl(sample: pl.DataFrame) -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    conn = aiohttp.TCPConnector(limit=GLOBAL_CONC, limit_per_host=PER_DOMAIN, ssl=False)
    tmo = aiohttp.ClientTimeout(total=TIMEOUT)
    hdr = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5",
           "Accept-Language": "vi,en;q=0.8,zh;q=0.6"}
    pr = Prober()
    async with aiohttp.ClientSession(connector=conn, timeout=tmo, headers=hdr) as sess:
        return await asyncio.gather(*(pr.fetch(sess, r) for r in sample.to_dicts()))


# ---------------------------------------------------------------- extract
def analyse(rec: dict, tok) -> dict:
    import trafilatura
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = SEED
    out = {"text_len": 0, "n_tok": 0, "lang": None, "is_pdf": False, "needs_js": False, "blocked": False,
           "paywall": False, "ok": False, "title": None, "visible_chars": 0, "thin": False, "has_text": False}
    if not rec["raw_path"]:
        return out
    body = (resolve_legacy_artifact(rec["raw_path"])).read_bytes()
    out["is_pdf"] = body[:4] == b"%PDF" or "pdf" in (rec["content_type"] or "")
    text = ""
    if out["is_pdf"]:
        try:
            import fitz
            with fitz.open(stream=body, filetype="pdf") as d:
                text = "\n".join(p.get_text() for p in d)
        except Exception:  # noqa: BLE001
            text = ""
    else:
        html = body.decode("utf-8", errors="ignore") if body[:2000].lower().count(b"utf-8") else None
        text = trafilatura.extract(body, favor_recall=True, include_comments=False, include_tables=True) or ""
        md = trafilatura.extract_metadata(body)
        out["title"] = getattr(md, "title", None)
        head = (html or body[:200000].decode("utf-8", errors="ignore"))
        out["blocked"] = bool(rec["status"] in (403, 429, 503) or (len(text) < 500 and BLOCK_PAT.search(head)))
        out["paywall"] = bool(len(text) < 500 and PAYWALL_PAT.search(head)) or rec["status"] in (401, 402)
        # chu hien thi trong body (bo script/style): ~0 => render phia client => can JS that su
        try:
            import lxml.html
            tree = lxml.html.fromstring(body)
            for x in tree.xpath("//script|//style|//noscript"):
                x.drop_tree()
            out["visible_chars"] = len(re.sub(r"\s+", " ", tree.text_content()).strip())
        except Exception:  # noqa: BLE001
            out["visible_chars"] = 0
        short = rec["status"] == 200 and len(text) < MIN_TEXT and not out["blocked"]
        out["needs_js"] = bool(short and out["visible_chars"] < 300)  # HTML co nhung khong co text hien thi
        out["thin"] = bool(short and not out["needs_js"])  # text co that nhung ngan (<200 ky tu)
    text = text.strip()
    out["text_len"] = len(text)
    if text:
        out["n_tok"] = len(tok(text, add_special_tokens=False)["input_ids"])
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text[:3000])) / max(1, len(text[:3000]))
        try:
            out["lang"] = "zh" if cjk > 0.2 else detect(text[:3000])
        except Exception:  # noqa: BLE001
            out["lang"] = "unk"
    out["ok"] = bool(rec["status"] == 200 and out["text_len"] >= MIN_TEXT)
    out["has_text"] = bool(rec["status"] == 200 and out["text_len"] > 0 and not out["blocked"])
    if not out["has_text"]:
        out["lang"] = None
        out["n_tok"] = 0
    return out


def run_crawl(force: bool) -> pl.DataFrame:
    path = OUT / "crawl_sample.csv"
    if path.exists() and not force:
        return pl.read_csv(path, infer_schema_length=10000)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    tok.model_max_length = 10**9
    sample = build_sample()
    print(f"sample={sample.height} domains={sample['domain'].n_unique()}; crawling...")
    t0 = time.time()
    recs = asyncio.run(crawl(sample))
    print(f"crawl done {time.time() - t0:.0f}s; extracting...")
    rows = []
    for r in recs:
        rows.append({**r, **analyse(r, tok)})
    df = pl.DataFrame(rows, infer_schema_length=None).join(
        sample.select("id", "lang_guess", "slug_type"), on="id", how="left")
    df.write_csv(path)
    return df


# ---------------------------------------------------------------- summary + resources
def domain_summary(cr: pl.DataFrame) -> pl.DataFrame:
    return (cr.group_by("domain").agg(
        pl.len().alias("n"),
        pl.col("ok").mean().round(3).alias("success_rate"),
        (pl.col("status") == 200).mean().round(3).alias("http200"),
        pl.col("needs_js").mean().round(3).alias("pct_js"),
        pl.col("is_pdf").mean().round(3).alias("pct_pdf"),
        (pl.col("blocked") | pl.col("paywall")).mean().round(3).alias("pct_block_paywall"),
        pl.col("error").is_not_null().mean().round(3).alias("pct_err"),
        pl.col("lang").drop_nulls().mode().first().alias("main_lang"),
        pl.col("n_tok").filter(pl.col("has_text")).median().alias("median_tok"),
        pl.col("thin").mean().round(3).alias("pct_thin"),
        pl.col("elapsed_s").median().alias("median_rt_s"),
        pl.col("stratum").first().alias("stratum"),
    ).sort("n", "success_rate", descending=[True, False]))


def resources(cr: pl.DataFrame, bench: dict | None) -> dict:
    prof = pl.read_parquet(OUT / "corpus_profiled.parquet", columns=["domain", "lang_guess", "dup_norm_size"])
    dom = prof.group_by("domain").agg(pl.len().alias("n"), (pl.col("dup_norm_size") > 1).sum().alias("n_dup"),
                                      pl.col("lang_guess").mode().first().alias("lg"))
    ok = cr.filter(pl.col("has_text"))
    med_dom = {r["domain"]: r["median_tok"] for r in
               ok.group_by("domain").agg(pl.col("n_tok").median().alias("median_tok")).to_dicts()}
    med_all = float(ok["n_tok"].median()) if ok.height else float("nan")
    med_lang = {r["lang_guess"]: r["m"] for r in ok.group_by("lang_guess").agg(pl.col("n_tok").median().alias("m")).to_dicts()}
    tot_tok = 0.0
    n_docs = 0
    for r in dom.to_dicts():
        uniq = r["n"] - r["n_dup"] // 2  # nhom trung chu yeu la cap http/https -> bo 1/2 dong trong nhom
        m = med_dom.get(r["domain"]) or med_lang.get(r["lg"]) or med_all
        tot_tok += uniq * m
        n_docs += uniq
    res = {"median_tok_all": med_all, "n_unique_docs_est": n_docs, "total_tokens_est": tot_tok}
    for c in (256, 512):
        # median-based; doc dai bi cat thanh nhieu chunk, doc ngan van >=1 chunk => can duoi
        res[f"chunks_{c}"] = max(tot_tok / c, n_docs)
        res[f"index_gb_fp16_{c}"] = res[f"chunks_{c}"] * 1024 * 2 / 1e9
    res["bench"] = bench
    return res


def bench_gpu(cr: pl.DataFrame) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    texts = []
    for r in cr.filter(pl.col("ok") & pl.col("raw_path").is_not_null()).to_dicts()[:60]:
        import trafilatura
        t = trafilatura.extract((resolve_legacy_artifact(r["raw_path"])).read_bytes()) if not r["is_pdf"] else None
        if t:
            texts.append(t)
    m = SentenceTransformer("BAAI/bge-m3", device="cuda", model_kwargs={"torch_dtype": torch.float16})
    m.max_seq_length = 512
    texts = (texts * 20)[:256]
    m.encode(texts[:16], batch_size=16)
    torch.cuda.synchronize(); t0 = time.time()
    m.encode(texts, batch_size=16)
    torch.cuda.synchronize(); dt = time.time() - t0
    return {"gpu": torch.cuda.get_device_name(0), "docs_per_s_512": len(texts) / dt, "n": len(texts),
            "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


# ---------------------------------------------------------------- report
def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(trống)_\n"
    cols = cols or list(rows[0].keys())
    f = lambda v: f"{v:,.2f}" if isinstance(v, float) else (f"{v:,}" if isinstance(v, int) else str(v))  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(f(r[c]) for c in cols) + " |" for r in rows]) + "\n"


def build_report(st: dict, cr: pl.DataFrame, ds: pl.DataFrame, rs: dict) -> str:
    c, q = st["corpus"], st["query"]
    n = c["rows"]
    L: list[str] = ["# Profile ViBioMIR – Road to AI Stage 3 (bước 0)\n",
                    f"Seed={SEED}. Số liệu sinh bởi `step0_profile.py` + `step0_crawl_probe.py`; ngôn ngữ/token ở mục 4 là **đo trên mẫu**, không ngoại suy ngoài mục 6.\n"]
    L += ["## 1. Integrity\n", md_table([
        {"metric": "rows", "value": n}, {"metric": "id unique", "value": c["id_unique"]},
        {"metric": "url unique", "value": c["url_unique"]}, {"metric": "id min / max", "value": f"{c['id_min']:,} / {c['id_max']:,}"},
        {"metric": "id thiếu trong [min,max]", "value": c["id_missing_in_range"]},
        {"metric": "url null / rỗng / sai định dạng", "value": f"{c['url_null']} / {c['url_empty']} / {c['url_bad_format']}"},
        {"metric": "url chứa khoảng trắng", "value": c["url_has_space"]},
    ])]
    L += ["## 2. Domain\n", f"{c['n_domains']} domain. **{c['domains_cover_80']} domain phủ 80%**, **{c['domains_cover_95']} domain phủ 95%**, top-30 = {c['top30_domain_pct']}%.\n",
          md_table(c["domain_top"])]
    L += ["## 3. URL features\n", "Đuôi file:\n", md_table(c["ext_dist"]), "TLD:\n", md_table(c["tld_dist"]),
          "Slug (tiêu đề giả từ URL):\n", md_table([{**d, "pct": pct(d["count"], n)} for d in c["slug_dist"]]),
          md_table([
              {"metric": "URL chứa CJK (ký tự thô)", "value": c["url_has_cjk"]},
              {"metric": "nhóm trùng gần (bỏ http/https, www, #frag, '/', host hoa/thường; path giữ nguyên)", "value": c["url_dup_norm_groups"]},
              {"metric": "dòng thừa theo nhóm đó", "value": c["url_dup_norm_extra_rows"]},
              {"metric": "nhóm trùng khi bỏ cả query string", "value": c["url_dup_noquery_groups"]},
              {"metric": "dòng thừa khi bỏ query (CẢNH BÁO: có thể là trang khác nhau)", "value": c["url_dup_noquery_extra_rows"]},
          ])]
    # crawl
    http = cr["status"].drop_nulls().value_counts().sort("count", descending=True).to_dicts()
    nerr = cr.filter(pl.col("error").is_not_null())["error"].str.replace(r":.*", "").value_counts().sort("count", descending=True).to_dicts()
    L += ["## 4. Crawl thử\n",
          f"Mẫu {cr.height} URL / {cr['domain'].n_unique()} domain (top30×{PER_TOP} + {TAIL_N} long-tail). UA `{UA}`, timeout {TIMEOUT}s, ≤{PER_DOMAIN} req/domain, robots.txt tôn trọng.\n",
          md_table([
              {"metric": "OK (200 & text≥200 ký tự)", "n": int(cr["ok"].sum()), "pct": pct(int(cr["ok"].sum()), cr.height)},
              {"metric": "thin (200, text thật nhưng <200 ký tự)", "n": int(cr["thin"].sum()), "pct": pct(int(cr["thin"].sum()), cr.height)},
              {"metric": "needs_js (200, text trích <200 và body hiển thị <300 ký tự)", "n": int(cr["needs_js"].sum()), "pct": pct(int(cr["needs_js"].sum()), cr.height)},
              {"metric": "PDF", "n": int(cr["is_pdf"].sum()), "pct": pct(int(cr["is_pdf"].sum()), cr.height)},
              {"metric": "blocked/captcha/403/429/503", "n": int(cr["blocked"].sum()), "pct": pct(int(cr["blocked"].sum()), cr.height)},
              {"metric": "paywall/login", "n": int(cr["paywall"].sum()), "pct": pct(int(cr["paywall"].sum()), cr.height)},
              {"metric": "robots disallowed", "n": int((cr["robots"] == "DISALLOWED").sum()), "pct": pct(int((cr["robots"] == "DISALLOWED").sum()), cr.height)},
              {"metric": "lỗi mạng/timeout", "n": int(cr["error"].is_not_null().sum()) - int((cr["robots"] == "DISALLOWED").sum()), "pct": pct(int(cr["error"].is_not_null().sum()) - int((cr["robots"] == "DISALLOWED").sum()), cr.height)},
              {"metric": "có redirect", "n": int((cr["n_redirects"] > 0).sum()), "pct": pct(int((cr["n_redirects"] > 0).sum()), cr.height)},
          ]),
          "HTTP status:\n", md_table(http), "Lỗi:\n", md_table(nerr) if nerr else "_không có_\n",
          f"Thời gian phản hồi (s): median {cr['elapsed_s'].median():.2f}, p95 {cr['elapsed_s'].quantile(.95):.2f}. Kích thước (KB): median {cr['bytes'].median()/1024:.0f}, p95 {cr['bytes'].quantile(.95)/1024:.0f}.\n",
          "Ngôn ngữ text (doc status 200 có text):\n",
          md_table(cr.filter(pl.col("has_text"))["lang"].value_counts().sort("count", descending=True).head(6).to_dicts()),
          "Tổng hợp theo domain:\n", md_table(ds.select("domain", "n", "success_rate", "pct_js", "pct_thin", "pct_pdf", "pct_block_paywall", "pct_err", "main_lang", "median_tok", "median_rt_s").to_dicts())]
    # queries
    L += ["## 5. Queries\n", md_table([
        {"metric": "n / id unique", "value": f"{q['n']} / {q['id_unique']}"},
        {"metric": "từ: median / p95 / max", "value": f"{q['words']['median']:.0f} / {q['words']['p95']:.0f} / {q['words']['max']:.0f}"},
        {"metric": "token BGE-M3: median / p90 / p95 / max", "value": f"{q['tokens']['median']:.0f} / {q['tokens']['p90']:.0f} / {q['tokens']['p95']:.0f} / {q['tokens']['max']:.0f}"},
        {"metric": "query >128 / >256 token", "value": f"{q['gt_128_tok']} / {q['gt_256_tok']}"},
        {"metric": "≥2 dấu '?'", "value": f"{q['multi_qmark']} ({pct(q['multi_qmark'], q['n'])}%)"},
        {"metric": "chứa 'và'/'đồng thời'/';'", "value": f"{q['multi_conj_va_dongthoi_semicolon']} ({pct(q['multi_conj_va_dongthoi_semicolon'], q['n'])}%)"},
        {"metric": "nhiều ý chặt (≥2 '?' hoặc ';' hoặc 'đồng thời')", "value": f"{q['multi_strict']} ({pct(q['multi_strict'], q['n'])}%)"},
        {"metric": "nhiều ý lỏng (≥2 '?' hoặc 'và'...)", "value": f"{q['multi_intent_any']} ({pct(q['multi_intent_any'], q['n'])}%)"},
        {"metric": "có viết tắt IN HOA (CT, MRI, HbA1c...)", "value": f"{q['has_abbr']} ({pct(q['has_abbr'], q['n'])}%)"},
        {"metric": "có thuật ngữ Latin/EN (heuristic)", "value": f"{q['has_en']} ({pct(q['has_en'], q['n'])}%)"},
    ]), "Viết tắt hay gặp:\n", md_table(q["top_abbr"]),
        "### 30 mẫu ngẫu nhiên\n", "\n".join(f"- Q{d['id']} ({d['n_tok_bgem3']} tok): {d['query']}" for d in q["sample30"]) + "\n",
        "### 20 câu dài nhất (cắt 160 ký tự)\n", "\n".join(f"- Q{d['id']} ({d['n_tok_bgem3']} tok): {d['query']}…" for d in q["longest20"]) + "\n"]
    # resources
    L += ["## 6. Ước lượng tài nguyên\n",
          f"Giả định: median token/doc theo domain trong mẫu có text (domain không có mẫu có text dùng median theo nhóm ngôn ngữ đoán, rồi median toàn mẫu = {rs['median_tok_all']:.0f}); loại trùng http/https; chunk ≈ tổng_token/size (sàn 1 chunk/doc); 1024-dim fp16 = 2 KB/chunk.\n",
          md_table([
              {"metric": "doc duy nhất (ước)", "value": int(rs["n_unique_docs_est"])},
              {"metric": "tổng token (ước)", "value": f"{rs['total_tokens_est']/1e9:.2f} tỷ"},
              {"metric": "chunk @256", "value": f"{rs['chunks_256']/1e6:.1f} triệu"},
              {"metric": "chunk @512", "value": f"{rs['chunks_512']/1e6:.1f} triệu"},
              {"metric": "index dense fp16 @256", "value": f"{rs['index_gb_fp16_256']:.1f} GB"},
              {"metric": "index dense fp16 @512", "value": f"{rs['index_gb_fp16_512']:.1f} GB"},
          ])]
    b = rs.get("bench")
    if b:
        rate = b["docs_per_s_512"]  # doc 512 token/giay do tren GPU local
        L += [f"Đo thực tế BGE-M3 fp16 trên **{b['gpu']}**: {rate:.1f} chunk-512/s ({b['n']} mẫu, peak VRAM {b['max_mem_gb']:.1f} GB).\n",
              md_table([{"cấu hình": "RTX 3050 6GB (đo)", "giờ embed @512": rs["chunks_512"] / rate / 3600,
                         "giờ embed @256 (giả định 2× nhanh hơn)": rs["chunks_256"] / (rate * 2) / 3600}]),
              "T4/P100 Kaggle: **chưa đo** (không có quyền truy cập từ đây); cần chạy `--bench` trên Kaggle để có số thật.\n"]
    else:
        L += ["Thời gian embed: **chưa đo** (chạy `python step0_crawl_probe.py --bench`, tải BGE-M3 ~2.2 GB). "
              "Công thức: giờ = chunk / (chunk-512/s đo được) / 3600; T4/P100 phải đo riêng trên Kaggle, không ước lượng khống ở đây.\n"]
    L += ["## Khuyến nghị\n"] + [f"- {x}" for x in recommend(st, cr, ds, rs)] + [""]
    return "\n".join(L)


def pct(x: int, n: int) -> float:
    return round(100 * x / n, 2)


def recommend(st: dict, cr: pl.DataFrame, ds: pl.DataFrame, rs: dict) -> list[str]:
    c, q = st["corpus"], st["query"]
    n = c["rows"]
    rec = []
    ok = cr["ok"].mean()
    zh_share = sum(d["count"] for d in c["lang_guess_dist"] if d["lang_guess"] == "zh?") / n
    langs = cr.filter(pl.col("has_text"))["lang"].value_counts().sort("count", descending=True).head(3).to_dicts()
    vi_share = sum(d["count"] for d in c["lang_guess_dist"] if d["lang_guess"] == "vi") / n
    rec.append(f"Ngôn ngữ: ~{zh_share*100:.0f}% URL là zh (id số/CJK slug), {vi_share*100:.0f}% vi; mẫu có text: "
               + ", ".join(f"{r['lang']} {r['count']}" for r in langs)
               + ". Query 100% tiếng Việt -> truy hồi chéo ngữ: dense BGE-M3 là trục chính; BM25 tiếng Trung (jieba/bigram) chỉ hữu ích nếu dịch query sang zh; tách index BM25 vi và zh.")
    js = cr["needs_js"].mean()
    pdf = cr["is_pdf"].mean()
    blk = (cr["blocked"] | cr["paywall"]).mean()
    rec.append(f"Crawl: requests+trafilatura đủ (text≥200) cho {ok*100:.0f}% mẫu; cần JS/Playwright cho {js*100:.1f}% (needs_js; thin {cr["thin"].mean()*100:.1f}% là trang ngắn thật), PDF {pdf*100:.1f}% (PyMuPDF), bị chặn/paywall {blk*100:.1f}%.")
    bad = ds.filter((pl.col("success_rate") < 0.5) & (pl.col("stratum") == "top30")).sort("n", descending=True)
    if bad.height:
        rec.append("Domain top-30 cần parser/xử lý riêng (success<50% trong mẫu): " + ", ".join(
            f"{r['domain']} ({r['success_rate']*100:.0f}%)" for r in bad.head(8).to_dicts()) + ".")
    rec.append(f"Ưu tiên domain: {c['domains_cover_80']} domain phủ 80% corpus, {c['domains_cover_95']} phủ 95% -> crawl theo thứ tự cum_pct, chỉ nhắm các domain có success≥50%.")
    rec.append(f"Slug có nghĩa chỉ {next(d['count'] for d in c['slug_dist'] if d['slug_type']=='meaningful_slug')/n*100:.1f}% URL, numeric_id {next(d['count'] for d in c['slug_dist'] if d['slug_type']=='numeric_id')/n*100:.1f}% -> không dùng slug làm title; phải fetch nội dung để có tiêu đề.")
    rec.append(f"Trùng URL: {c['url_dup_norm_extra_rows']:,} dòng thừa (http/https) -> crawl 1 lần/nhóm url_norm; KHÔNG dedupe bỏ query ({c['url_dup_noquery_extra_rows']-c['url_dup_norm_extra_rows']:,} dòng chỉ khác query).")
    rec.append(f"Query: {q['multi_strict']} ({q['multi_strict']/q['n']*100:.1f}%) nhiều ý chặt, {q['gt_128_tok']} >128 token, p95={q['tokens']['p95']:.0f} -> "
               f"{'cần' if q['multi_strict']/q['n'] > 0.08 else 'chưa cần'} tách truy vấn nhiều ý; {q['has_abbr']/q['n']*100:.0f}% có viết tắt EN -> BM25 + dense hybrid, giữ nguyên viết tắt.")
    return rec[:7]


def main() -> None:
    auxiliary_disabled()
    ap = argparse.ArgumentParser()
    ap.add_argument("--recrawl", action="store_true")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    st = json.loads((OUT / "profile_static.json").read_text("utf-8"))
    cr = run_crawl(a.recrawl)
    ds = domain_summary(cr)
    ds.write_csv(OUT / "crawl_domain_summary.csv")
    bench = bench_gpu(cr) if a.bench else None
    rs = resources(cr, bench)
    (OUT / "profile_report.md").write_text(build_report(st, cr, ds, rs), "utf-8")
    print("report ->", OUT / "profile_report.md")


if __name__ == "__main__":
    main()
