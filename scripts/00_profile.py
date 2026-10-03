"""Buoc 0 - profile dataset AIGuruTinix/ViBioMIR. KHONG gui request toi URL trong corpus.

Chay lai duoc (idempotent): ghi de toan bo output trong data/profile/.
    python scripts/00_profile.py
"""
from __future__ import annotations

from r2ai.paths import PROFILE_DIR, RAW_DOWNLOAD_DIR, auxiliary_disabled, data_label

import json
import re
import sys
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote

import polars as pl
import psutil
from huggingface_hub import snapshot_download

RAW = RAW_DOWNLOAD_DIR
OUT = PROFILE_DIR
REPO = "AIGuruTinix/ViBioMIR"
CHUNK = 1_100_000
SEED = 42

# --- heuristic cau hinh (domain-level, suy tu danh sach host thuc te cua corpus) ---
ZH_SITES = {
    "39.net", "120ask.com", "cnkang.com", "zysjonline.com", "a-hospital.com", "zhongyibaodian.net",
    "zydcd.com", "wujue.com", "iiyi.com", "qihuangzhishu.com", "pmphai.com", "jb39.com", "baidu.com",
    "msdmanuals.cn",
}
VI_SITES = {"vinmec.com", "hellobacsi.com", "benhvienvietduc.org", "pharmacity.io"}
VI_SLUG_WORDS = (
    r"\b(?:benh|thuoc|cach|dieu tri|suc khoe|trieu chung|nguyen nhan|bac si|phong ngua|"
    r"co the|nen|khong|nhung|cua|cho|viem|ung thu|mang thai|tre em|sinh)\b"
)
RISK_NOTES = {
    "baidu.com": "baike: JS-render + captcha/chan bot manh",
    "120ask.com": "Q&A TQ: chan bot/IP, rate-limit nghiem (dung 918k URL)",
    "cnkang.com": "TQ: co the chan IP nuoc ngoai, 963k URL -> crawl lau",
    "familydoctor.com.cn": "TQ: anti-bot/403 tu IP ngoai TQ kha nang cao",
    "39.net": "TQ (nhieu subdomain): trang cu, GBK encoding, co the redirect/chet",
    "zysjonline.com": "sach TCM: co the can login/paywall, noi dung kieu 'books'",
    "a-hospital.com": "wiki TQ: on dinh hon, nhung 169k URL",
    "zhongyibaodian.net": "TQ: server nho, de bi rate-limit",
    "youlai.cn": "TQ: co the JS-render + chan bot",
    "wujue.com": "TQ: kha nang chan bot, noi dung mong",
    "zydcd.com": "TQ: trang cu, nguy co link chet",
    "iiyi.com": "TQ: co the can cookie/anti-bot",
    "qihuangzhishu.com": "TQ: co the can login",
    "pmphai.com": "host 'test.' -> moi truong test, kha nang chet/can auth",
    "pharmacity.io": "host 'beta' -> moi truong beta, kha nang chet",
    "msdmanuals.cn": "MSD manual TQ: cloudflare/anti-bot kha nang cao",
    "nhathuoclongchau.com.vn": "e-commerce VN: JS-render (Next.js), co the can headless",
    "tiemchunglongchau.com.vn": "e-commerce VN: JS-render",
    "vinmec.com": "VN: cloudflare/WAF nhe",
    "hellobacsi.com": "VN: cloudflare + JS lazy-load",
    "thanhnien.vn": "bao VN: paywall nhe, ok neu tu toc do cham",
    "suckhoedoisong.vn": "bao VN: on, ok",
    "baohaiphong.vn": "bao dia phuong VN: server yeu, link chet cao hon",
    "baonghean.vn": "bao dia phuong VN: server yeu",
    "baodanang.vn": "bao dia phuong VN: server yeu",
    "baocantho.com.vn": "bao dia phuong VN: server yeu",
    "baoangiang.com.vn": "bao dia phuong VN: server yeu",
}

T0 = time.perf_counter()


@contextmanager
def step(name: str):
    t = time.perf_counter()
    print(f"[{time.perf_counter() - T0:7.1f}s] >> {name}", flush=True)
    yield
    print(f"[{time.perf_counter() - T0:7.1f}s] << {name} ({time.perf_counter() - t:.1f}s)", flush=True)


def ram() -> str:
    m = psutil.virtual_memory()
    return f"RAM kha dung {m.available / 2**30:.2f} GiB / {m.total / 2**30:.2f} GiB"


def md_table(df: pl.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            elif isinstance(v, int) and not isinstance(v, bool):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v).replace("|", "\\|"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def deaccent(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower().replace("đ", "d"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------- enrich
def _unquote_batch(s: pl.Series) -> pl.Series:
    return pl.Series([unquote(x) if "%" in x else x for x in s.to_list()], dtype=pl.String)


def _deaccent_nonascii(s: pl.Series) -> pl.Series:
    return pl.Series([deaccent(x) if not x.isascii() else x for x in s.to_list()], dtype=pl.String)


def _clean_tokens(e: pl.Expr) -> pl.Expr:
    """Chuan hoa chuoi path -> token cach nhau dau cach, bo token so/hex."""
    return (
        e.str.to_lowercase()
        .str.replace_all(r"[-_+.\s/]+", " ")
        .str.replace_all(r"\b\d+\b", " ")
        .str.replace_all(r"\b[0-9a-f]{8,}\b", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def enrich(df: pl.DataFrame) -> pl.DataFrame:
    rest = (
        pl.col("url").str.strip_chars()
        .str.replace(r"(?i)^[a-z][a-z0-9+.\-]*://", "")
        .str.replace(r"#.*$", "")
    )
    host = (
        rest.str.extract(r"^([^/?#]*)", 1).str.to_lowercase()
        .str.replace(r"^[^@]*@", "").str.replace(r":\d+$", "").str.replace(r"^www\.", "")
    )
    tail = rest.str.extract(r"^[^/?#]*(.*)$", 1).fill_null("").str.replace(r"/+(\?|$)", "$1")
    df = df.with_columns(domain=host, _tail=tail).with_columns(
        url_norm=pl.col("domain") + pl.col("_tail"),
        url_path=pl.col("domain") + pl.col("_tail").str.replace(r"\?.*$", ""),
        tld=pl.col("domain").str.extract(r"\.([a-z0-9\-]+)$", 1),
        site=pl.coalesce(
            pl.col("domain").str.extract(r"([^.]+\.(?:com|org|gov|net|edu|co)\.[a-z]{2})$", 1),
            pl.col("domain").str.extract(r"([^.]+\.[^.]+)$", 1),
            pl.col("domain"),
        ),
    ).drop("_tail")
    df = df.with_columns(
        _dec=pl.col("url_norm").map_batches(_unquote_batch, return_dtype=pl.String),
        _path=pl.col("url_path").map_batches(_unquote_batch, return_dtype=pl.String)
        .str.extract(r"^[^/]*(/.*)$", 1).fill_null(""),
    ).with_columns(
        _last=pl.col("_path").str.extract(r"([^/]*)$", 1).fill_null("")
        .str.replace(r"(?i)\.(?:html?|shtml|php\d?|aspx?|jsp)$", ""),
    ).with_columns(
        slug_text=_clean_tokens(pl.col("_last")),
        path_text=_clean_tokens(pl.col("_path").str.replace(r"(?i)\.(?:html?|shtml|php\d?|aspx?|jsp)$", "")),
        _han=pl.col("_last").str.contains(r"\p{Han}"),
        _han_url=pl.col("_dec").str.contains(r"\p{Han}"),
    ).with_columns(_nalpha=pl.col("slug_text").str.count_matches(r"\b\p{L}{2,}\b"))
    df = df.with_columns(
        slug_type=pl.when(pl.col("_han")).then(pl.lit("cjk_slug"))
        .when(pl.col("_last") == "").then(pl.lit("other"))
        .when(pl.col("_nalpha") >= 3).then(pl.lit("meaningful_slug"))
        .when(pl.col("_nalpha") == 0).then(pl.lit("numeric_id"))
        .otherwise(pl.lit("other")),
        lang_guess=pl.when(pl.col("_han_url")).then(pl.lit("zh"))
        .when(pl.col("tld") == "vn").then(pl.lit("vi"))
        .when(pl.col("site").is_in(list(ZH_SITES)) | pl.col("tld").is_in(["cn", "tw", "hk", "mo"])).then(pl.lit("zh"))
        .when(pl.col("site").is_in(list(VI_SITES))).then(pl.lit("vi"))
        .when(pl.col("path_text").map_batches(_deaccent_nonascii, return_dtype=pl.String)
              .str.contains(VI_SLUG_WORDS)).then(pl.lit("vi"))
        .otherwise(pl.lit("en/unknown")),
        pmid=pl.col("url_norm").str.extract(
            r"(?i)(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?/(?:\?term=)?|[?&]pmid=)(\d{4,9})", 1
        ).cast(pl.Int64, strict=False),
        pmcid=pl.col("url_norm").str.extract(r"(?i)\b(PMC\d{4,9})\b", 1).str.to_uppercase(),
        source_family=pl.when(pl.col("domain").str.contains(r"ncbi\.nlm\.nih\.gov|pubmed|europepmc")).then(pl.lit("pubmed_pmc"))
        .when(pl.col("domain").str.contains(r"wikipedia\.org")).then(pl.lit("wikipedia"))
        .when(pl.col("domain").str.contains(r"medlineplus|(^|\.)who\.int|(^|\.)cdc\.gov|(^|\.)nhs\.uk|(^|\.)nih\.gov")).then(pl.lit("gov_health"))
        .when(pl.col("domain").str.contains(r"msdmanuals")).then(pl.lit("msdmanuals"))
        .when(pl.col("domain").str.contains(r"baike")).then(pl.lit("baike"))
        .otherwise(pl.lit("")),
    ).with_columns(
        dump_level=pl.when(pl.col("source_family").is_in(["pubmed_pmc", "wikipedia"])).then(pl.lit("bulk_dump"))
        .when(pl.col("source_family") == "gov_health").then(pl.lit("api_or_feed"))
        .otherwise(pl.lit("none"))
    )
    return df.select(
        "id", "url", "url_norm", "url_path", "domain", "site", "tld", "lang_guess", "slug_type",
        "slug_text", "path_text", "pmid", "pmcid", "source_family", "dump_level",
    )


# ---------------------------------------------------------------- main
def main() -> None:
    auxiliary_disabled()
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    rep: list[str] = []

    # 1. tai du lieu
    with step("1. download"):
        print(ram())
        snapshot_download(REPO, repo_type="dataset", allow_patterns=["*.parquet", "*.md"], local_dir=str(RAW))
        files = sorted(p for p in RAW.rglob("*") if p.is_file() and ".cache" not in p.parts)
        rows = []
        for p in files:
            n = pl.scan_parquet(p).select(pl.len()).collect().item() if p.suffix == ".parquet" else None
            rows.append((data_label(p), p.stat().st_size, n))
            print(f"   {rows[-1]}")
        files_df = pl.DataFrame(rows, schema=["path", "bytes", "rows"], orient="row")
        rep.append("# Profile ViBioMIR (Buoc 0)\n\n## 1. File tai ve\n\n" + md_table(files_df) +
                   "\n\nFile goc: `links_corpus.parquet` + `query.parquet` (1 shard, khong co nhanh refs/convert/parquet).\n")

    # 2. toan ven
    with step("2. integrity"):
        print(ram())
        raw = pl.read_parquet(RAW / "links_corpus.parquet")
        n, nu = raw.height, raw["id"].n_unique()
        idmin, idmax = raw["id"].min(), raw["id"].max()
        card_total = 4_420_561
        miss_in_range = (idmax - idmin + 1) - nu
        miss_below = idmin - 1
        url_null = raw["url"].null_count()
        url_empty = raw.filter(pl.col("url").str.strip_chars() == "").height
        bad_scheme = raw.filter(~pl.col("url").str.contains(r"(?i)^https?://")).height
        exact_dup = raw.group_by("url").agg(pl.col("id").alias("ids"), pl.len().alias("n")).filter(pl.col("n") > 1)
        integ = pl.DataFrame({
            "metric": ["rows", "unique id", "min id", "max id", "id thieu trong [min,max]", "id thieu duoi min (1..min-1)",
                       "tong id thieu so voi card (max - rows)", "url null", "url rong", "url khong http(s)",
                       "nhom URL trung y het (id khac nhau)", "so id nam trong nhom URL trung y het"],
            "value": [n, nu, idmin, idmax, miss_in_range, miss_below, card_total - n, url_null, url_empty, bad_scheme,
                      exact_dup.height, int(exact_dup["n"].sum()) if exact_dup.height else 0],
        })
        rep.append("## 2. Toan ven corpus\n\n" + md_table(integ) +
                   f"\n\nGiai thich chenh lech: card ghi {card_total:,} = max id; viewer/parquet thuc te {n:,} dong. "
                   f"Chenh {card_total - n:,} = {miss_below:,} id (1..{miss_below}) khong ton tai + {miss_in_range:,} id "
                   "thung trong khoang (dong bi loai khi tao dataset). Card dem theo khoang ID, khong dem dong.\n")
        print(integ)

    # 3-5. enrich theo chunk
    with step("3-5. enrich (url_norm, domain, slug, lang) theo chunk"):
        print(ram())
        parts = []
        for i in range(0, raw.height, CHUNK):
            with step(f"   chunk {i // CHUNK + 1}/{-(-raw.height // CHUNK)}"):
                parts.append(enrich(raw.slice(i, CHUNK)))
        del raw
        df = pl.concat(parts)
        del parts
        df.write_parquet(OUT / "corpus_profiled.parquet", compression="zstd")
        print(df.estimated_size("gb"), "GB;", ram())

    with step("3b. near-dup"):
        g_path = df.group_by("url_path").agg(pl.col("id").alias("ids"), pl.len().alias("count")).filter(pl.col("count") > 1)
        g_norm = df.group_by("url_norm").agg(pl.len().alias("count")).filter(pl.col("count") > 1)
        nd = g_path.rename({"url_path": "path", "ids": "ids"}).sort("count", descending=True)
        nd.write_parquet(OUT / "near_dup_groups.parquet", compression="zstd")
        n_dup_rows = int(nd["count"].sum()) if nd.height else 0
        n_extra = n_dup_rows - nd.height  # so dong thua neu giu 1 dai dien/nhom
        nd_dom = (df.filter(pl.col("url_path").is_in(nd["path"].implode()))
                  .group_by("domain").len().sort("len", descending=True).head(10)) if nd.height else None
        norm_rows = int(g_norm["count"].sum())
        norm_extra = norm_rows - g_norm.height  # trang that su trung (khac scheme/www/fragment/slash)
        qonly = nd.height - g_norm.height       # nhom cung path, khac nhau CHI o query string
        qonly_rows = n_dup_rows - norm_rows
        nd_t = pl.DataFrame({
            "metric": ["nhom trung theo url_norm (http/https, www, #fragment, '/')", "dong thuoc nhom url_norm",
                       "dong thua THAT SU (giu 1/nhom url_norm)", "nhom trung theo url_path",
                       "nhom chi khac query string (cung path, khac ?id=...)", "dong thuoc nhom chi-khac-query",
                       "nhom lon nhat (so id)"],
            "value": [g_norm.height, norm_rows, norm_extra, nd.height, qonly, qonly_rows,
                      int(nd["count"].max()) if nd.height else 0],
        })
        rep.append("## 3. Trung gan\n\n" + md_table(nd_t) +
                   "\n\n**Luu y**: nhom trung url_norm = cung trang goc, chi khac http/https (hoac `#fragment`) -> 1 trang, nhieu id, nen "
                   "crawl 1 lan roi gan noi dung cho tat ca id. Nhom 'chi khac query' (vd `test.pmphai.com/...toPcDetail?id=X`) la TRANG KHAC NHAU "
                   "-> KHONG dedupe theo url_path, phai giu query.\n\nTop domain trong nhom trung path:\n\n" +
                   (md_table(nd_dom) if nd_dom is not None else "") + "\n")
        print(nd_t)

    with step("4. domain profile"):
        total = df.height
        dom = (df.group_by("domain").agg(
            pl.len().alias("n_urls"),
            pl.col("site").first(), pl.col("tld").first(), pl.col("lang_guess").mode().first().alias("lang_guess"),
            pl.col("source_family").first(), pl.col("dump_level").first(),
            (pl.col("slug_type") == "meaningful_slug").mean().alias("r_meaningful"),
            (pl.col("slug_type") == "numeric_id").mean().alias("r_numeric_id"),
            (pl.col("slug_type") == "cjk_slug").mean().alias("r_cjk"),
            (pl.col("slug_type") == "other").mean().alias("r_other"),
        ).sort("n_urls", descending=True).with_columns(
            pct=pl.col("n_urls") / total * 100,
        ).with_columns(cum_pct=pl.col("pct").cum_sum()))
        dom.write_csv(OUT / "domains.csv")
        top100 = dom.head(100)
        site = (df.group_by("site").agg(pl.len().alias("n_urls")).sort("n_urls", descending=True)
                .with_columns(pct=pl.col("n_urls") / total * 100).with_columns(cum_pct=pl.col("pct").cum_sum()))
        site.write_csv(OUT / "sites.csv")
        # mau 5 URL / domain
        sub = df.filter(pl.col("domain").is_in(top100["domain"].implode())).select("domain", "id", "url")
        sub = sub.with_columns(_r=pl.int_range(pl.len()).shuffle(seed=SEED)).sort(["domain", "_r"])
        samples = sub.group_by("domain", maintain_order=True).head(5).drop("_r")
        samples.write_csv(OUT / "domain_samples.csv")
        lang = (df.group_by("lang_guess").len().sort("len", descending=True)
                .with_columns(pct=pl.col("len") / total * 100).rename({"len": "n_urls"}))
        tld = df.group_by("tld").len().sort("len", descending=True).with_columns(pct=pl.col("len") / total * 100).rename({"len": "n_urls"})
        fam = df.filter(pl.col("source_family") != "").group_by(["source_family", "dump_level"]).len().rename({"len": "n_urls"})
        n_pmid = df["pmid"].drop_nulls().len()
        n_pmc = df["pmcid"].drop_nulls().len()
        print(top100.select("domain", "n_urls", "pct", "cum_pct", "lang_guess").head(100))
        rep.append(
            f"## 4. Domain\n\nSo host (domain bo www.): {dom.height}; so site (registrable gan dung): {site.height}. "
            "Sample 5 URL/domain (seed=42) o `data/profile/domain_samples.csv`.\n\n"
            "### Top 30 domain\n\n" + md_table(top100.head(30).select("domain", "n_urls", "pct", "cum_pct", "lang_guess")) +
            "\n\nToan bo 97 domain: `domains.csv`. Gop theo site (vd moi `*.39.net`): `sites.csv`.\n\n"
            "### Top 15 site\n\n" + md_table(site.head(15)) +
            "\n\n### Ngon ngu doan (heuristic, khong goi mang)\n\n" + md_table(lang) +
            "\n\n### TLD\n\n" + md_table(tld.head(10)) +
            f"\n\n### Nguon co bulk dump/API kha di\n\n" +
            (md_table(fam) if fam.height else "Khong co URL nao thuoc pubmed/ncbi/pmc/wikipedia/medlineplus/who/cdc/nhs/msd(.com)/baike.") +
            f"\n\nPMID trich duoc: {n_pmid:,}; PMCID: {n_pmc:,}. msdmanuals.cn va baidu.com (baike) xuat hien nhung "
            "khong co dump cong khai.\n")

    with step("5. slug"):
        st = (df.group_by("slug_type").len().rename({"len": "n_urls"}).sort("n_urls", descending=True)
              .with_columns(pct=pl.col("n_urls") / total * 100))
        st_lang = df.group_by(["lang_guess", "slug_type"]).len().pivot(on="slug_type", index="lang_guess", values="len").fill_null(0)
        rep.append("## 5. Slug (title mien phi tu URL)\n\n" + md_table(st) + "\n\nTheo ngon ngu doan:\n\n" + md_table(st_lang) +
                   "\n\nTheo domain (top 40; day du trong domains.csv, cot r_*):\n\n" +
                   md_table(top100.head(40).select("domain", "n_urls", "r_meaningful", "r_numeric_id", "r_cjk", "r_other"),
                            floatfmt="{:.3f}") + "\n")
        ps = (df.select((pl.col("path_text").str.count_matches(r"\b\p{L}{2,}\b") >= 3).mean()).item())
        path_note = f"Ty le URL co `path_text` (toan bo path) >=3 token chu: {ps * 100:.1f}% (cao hon slug cuoi vi gom ca thu muc).\n"
        rep.append(path_note)

    # 6. query
    with step("6. query profile"):
        q = pl.read_parquet(RAW / "query.parquet")
        q = q.with_columns(
            nw=pl.col("query").str.split(" ").list.len(),
            nq=pl.col("query").str.count_matches(r"\?"),
            nva=pl.col("query").str.count_matches(r" và "),
            nsemi=pl.col("query").str.count_matches(";"),
            nnl=pl.col("query").str.count_matches("\n"),
        )
        nw = q["nw"]
        multi = q.filter((pl.col("nq") >= 2) | (pl.col("nva") >= 2) | (pl.col("nsemi") >= 1) | (pl.col("nnl") >= 1))
        ids = q["id"]
        stats = {
            "n": q.height, "words_min": nw.min(), "words_median": nw.median(), "words_p95": nw.quantile(0.95),
            "words_max": nw.max(), "id_unique": ids.n_unique(), "id_duplicates": q.height - ids.n_unique(),
            "id_contiguous_1_1200": bool(sorted(ids.to_list()) == list(range(1, 1201))),
            "q_with_2plus_question_marks": q.filter(pl.col("nq") >= 2).height,
            "q_with_2plus_va": q.filter(pl.col("nva") >= 2).height,
            "q_with_semicolon": q.filter(pl.col("nsemi") >= 1).height,
            "q_with_newline": q.filter(pl.col("nnl") >= 1).height,
            "q_multi_intent_any": multi.height,
            "shortest10": q.sort("nw").head(10)["query"].to_list(),
            "longest10": q.sort("nw", descending=True).head(10)["query"].to_list(),
        }
        (OUT / "query_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
        s_tab = pl.DataFrame({"metric": [k for k, v in stats.items() if not isinstance(v, list)],
                              "value": [str(v) for k, v in stats.items() if not isinstance(v, list)]})
        rep.append("## 6. Query\n\n" + md_table(s_tab) + "\n\n**10 query ngan nhat**\n\n" +
                   "\n".join(f"- {x[:160]}" for x in stats["shortest10"]) + "\n\n**10 query dai nhat** (cat 200 ky tu)\n\n" +
                   "\n".join(f"- {x[:200].replace(chr(10), ' ')}..." for x in stats["longest10"]) + "\n")

    # 7. sanity
    with step("7. sanity match slug"):
        base = df.filter(pl.col("lang_guess") != "zh").select("id", "url", "slug_text", "path_text")
        base = base.with_columns(
            slug_text=pl.col("slug_text").map_batches(_deaccent_nonascii, return_dtype=pl.String),
            path_text=pl.col("path_text").map_batches(_deaccent_nonascii, return_dtype=pl.String),
        ).with_columns(_s=pl.lit(" ") + pl.col("slug_text") + pl.lit(" "), _p=pl.lit(" ") + pl.col("path_text") + pl.lit(" "))
        stop = set("va la cua cho co khong bi em toi minh may nay nhu the nao gi khi da duoc nhung mot cac voi de trong "
                   "den ra roi thi nen hay hoac bac si xin hoi con anh chi ban".split())
        sample_q = q.sample(5, seed=SEED)
        sec = ["Chi xet URL khong-zh (slug TQ khong khop duoc query tieng Viet khong dich).\n"]
        sanity = {}
        for qid, qt in sample_q.select("id", "query").iter_rows():
            toks = sorted({t for t in re.findall(r"[a-z0-9]+", deaccent(qt)) if len(t) >= 3 and t not in stop})
            m = base.with_columns(
                m_slug=sum(pl.col("_s").str.contains(f" {t} ", literal=True).cast(pl.Int8) for t in toks),
                m_path=sum(pl.col("_p").str.contains(f" {t} ", literal=True).cast(pl.Int8) for t in toks),
            )
            n2s, n2p = m.filter(pl.col("m_slug") >= 2).height, m.filter(pl.col("m_path") >= 2).height
            top = m.sort(["m_path", "m_slug"], descending=True).head(10).select("m_slug", "m_path", "url")
            sanity[int(qid)] = {"tokens": toks, "n_ge2_slug": n2s, "n_ge2_path": n2p}
            print(f"Q{qid}: {qt[:100]!r} tokens={len(toks)} >=2 slug:{n2s} path:{n2p}")
            sec.append(f"### Q{qid}: {qt[:200]}\n\n- tokens ({len(toks)}): {' '.join(toks)}\n- URL co >=2 token khop: slug cuoi={n2s:,}, "
                       f"toan path={n2p:,}\n\n" + md_table(top.with_columns(pl.col("url").str.slice(0, 110))) + "\n")
        rep.append("## 7. Sanity check: khop slug (bo dau) voi 5 query\n\n" + "\n".join(sec))
        (OUT / "sanity_match.json").write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")
        del base

    # Ham y chien luoc
    with step("8. implications"):
        bulk = df.filter(pl.col("dump_level") == "bulk_dump").height
        api = df.filter(pl.col("dump_level") == "api_or_feed").height
        cnt = {r[0]: r[1] for r in df.group_by("slug_type").len().iter_rows()}
        free = cnt.get("meaningful_slug", 0)
        cjk = cnt.get("cjk_slug", 0)
        need = cnt.get("numeric_id", 0) + cnt.get("other", 0)
        risk_rows = []
        for r in site.iter_rows(named=True):
            note = RISK_NOTES.get(r["site"])
            if note and r["n_urls"] >= 1000:
                risk_rows.append((r["site"], r["n_urls"], round(r["pct"], 2), note))
        for d in dom.iter_rows(named=True):
            if d["domain"] in RISK_NOTES and d["domain"] not in {x[0] for x in risk_rows} and d["n_urls"] >= 1000:
                risk_rows.append((d["domain"], d["n_urls"], round(d["pct"], 2), RISK_NOTES[d["domain"]]))
        risk = pl.DataFrame(risk_rows, schema=["domain/site", "n_urls", "pct", "ghi chu (doan, CHUA kiem chung bang mang)"], orient="row")
        zh_pct = lang.filter(pl.col("lang_guess") == "zh")["pct"].sum()
        rep.append(
            "## Ham y cho chien luoc crawl\n\n"
            f"- **Qua dump/API: {(bulk + api) / total * 100:.2f}%** ({bulk + api:,} URL). Corpus chi co {dom.height} host, "
            "khong co pubmed/pmc/wikipedia. Khong co duong tat bulk -> phai GET.\n"
            f"- **Slug co nghia (title mien phi, chu Latin): {free / total * 100:.1f}%** ({free:,}); slug CJK: {cjk / total * 100:.1f}% "
            f"({cjk:,}, nhung la tieng Trung nen can encoder xuyen ngon ngu/dich).\n"
            f"- **Can GET de lay title: {need / total * 100:.1f}%** ({need:,} = numeric_id + other). "
            f"Voi {zh_pct:.1f}% corpus la zh, phan nay hau het la 120ask/cnkang/familydoctor dang `/id.html`.\n"
            f"- **Nhom trung gan: {g_norm.height:,} nhom url_norm** (http/https twin) = {norm_extra:,} dong thua ({norm_extra / total * 100:.1f}%) -> crawl 1 lan, "
            f"gan cho moi id. Them {qonly:,} nhom cung path khac query (pmphai ...?id=) la trang khac, KHONG dedupe.\n"
            f"- Trang duy nhat sau dedupe url_norm: {total - norm_extra:,}; rieng nhom vi (title mien phi tu slug): "
            f"{lang.filter(pl.col('lang_guess') == 'vi')['n_urls'].sum():,} URL, la tap crawl re nhat va co the loc truoc bang slug.\n"
            "- Moi dong nay la **URL-level doc**: crawl 4.39M trang day du la kho thuc te -> uu tien crawl chon loc theo domain "
            "ung vien (vi truoc, zh sau) + dung slug de loc truoc.\n\n"
            "### Domain rui ro cao (suy tu kinh nghiem, chua kiem chung)\n\n" + md_table(risk) +
            "\n\nGoi y: thu 50-100 URL/domain tren mang SAU de do ty le 200/403/404/JS truoc khi quyet dinh ngan sach crawl.\n")

    (OUT / "REPORT.md").write_text("\n".join(rep), encoding="utf-8")
    print(f"[{time.perf_counter() - T0:7.1f}s] DONE; {ram()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
