"""Step 0a - thong ke tinh corpus + queries. KHONG gui request toi URL corpus.

    python step0_profile.py
Doc data/raw/*.parquet (chi doc), ghi out/: domains.csv, query_stats.csv,
corpus_profiled.parquet, profile_static.json.
"""
from __future__ import annotations

from r2ai.paths import OUT_DIR, RAW_DATA_DIR, auxiliary_disabled

import json
import re
from pathlib import Path

import polars as pl

SEED = 42
RAW = RAW_DATA_DIR
OUT = OUT_DIR

CJK = r"[㐀-䶿一-鿿豈-﫿]"
ABBR = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9]*|[A-Z][a-z]?[0-9][A-Za-z0-9]*|HbA1c|CT|MRI|ECG|COVID(?:-19)?)\b")
EN_WORD = re.compile(r"(?<![^\W\d_])[A-Za-z]{3,}(?![^\W\d_])")
# am tiet tieng Viet khong dau; tu ASCII khong khop cau truc nay => kha nang thuat ngu EN (heuristic)
VN_SYLLABLE = re.compile(
    r"^(?:ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnpqrstvx])?[aeiouy]{1,3}(?:ng|nh|ch|[cmnpt])?$", re.I)
VI_DIACRITIC = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)
MULTI = re.compile(r"(?:\bvà\b|\bđồng thời\b|;)", re.I)


def pct(x: int, n: int) -> float:
    return round(100 * x / n, 2)


def corpus_profile() -> dict:
    df = pl.read_parquet(RAW / "links_corpus.parquet")
    n = df.height
    res: dict = {"rows": n}

    # --- 1. integrity
    ids = df["id"]
    res["id_unique"] = ids.n_unique()
    res["id_min"], res["id_max"] = int(ids.min()), int(ids.max())
    res["id_missing_in_range"] = res["id_max"] - res["id_min"] + 1 - res["id_unique"]
    res["url_unique"] = df["url"].n_unique()
    res["url_null"] = df["url"].null_count()
    res["url_empty"] = int((df["url"].str.strip_chars() == "").sum())
    res["url_bad_format"] = int((~df["url"].str.contains(r"^https?://[^\s/]+\.[^\s/]+")).sum())
    res["url_has_space"] = int(df["url"].str.contains(r"\s").sum())

    # --- 3. URL features (tinh truoc de dung cho domain)
    host = df["url"].str.extract(r"^https?://([^/?#]+)", 1).str.to_lowercase().str.replace(r":\d+$", "")
    df = df.with_columns(
        host.str.replace(r"^www\.", "").alias("domain"),
        df["url"].str.replace(r"[?#].*$", "").alias("_nq"),
    )
    path = df["_nq"].str.replace(r"^https?://[^/]+", "")
    last = path.str.replace(r"/+$", "").str.extract(r"([^/]*)$", 1).fill_null("")
    ext = last.str.extract(r"\.([A-Za-z0-9]{1,5})$", 1).str.to_lowercase().fill_null("(none)")
    slug_base = last.str.replace(r"\.[A-Za-z0-9]{1,5}$", "")
    # slug co nghia: >=3 token chu (Latin/CJK) ngan cach bang '-', '_', '+'
    n_tok = slug_base.str.count_matches(r"[^\-_+\s]+")
    has_cjk_path = path.str.contains(CJK) | path.str.contains(r"%[Ee][0-9A-Fa-f]%")
    only_digits = slug_base.str.contains(r"^[0-9\-_]*$") | (slug_base == "")
    slug_type = (
        pl.when(has_cjk_path).then(pl.lit("cjk_slug"))
        .when(only_digits).then(pl.lit("numeric_id"))
        .when(n_tok >= 3).then(pl.lit("meaningful_slug"))
        .otherwise(pl.lit("other"))
    )
    # chuan hoa: bo http/https, www, #fragment, '/' cuoi; chi ha chu thuong HOST (path phan biet hoa/thuong)
    def _norm(u: pl.Series) -> pl.Series:
        u = u.str.replace(r"^http://", "https://").str.replace(r"^https://www\.", "https://").str.replace(r"/+$", "")
        host_lc = u.str.extract(r"^https://([^/?#]*)", 1).str.to_lowercase()
        return pl.concat_str([pl.lit("https://"), host_lc, u.str.replace(r"^https://[^/?#]*", "")])

    url_norm = _norm(df["url"].str.replace(r"#.*$", ""))
    url_noq = _norm(df["_nq"])
    df = df.with_columns(
        ext.alias("ext"),
        df["domain"].str.extract(r"\.([a-z]+)$", 1).alias("tld"),
        df["url"].str.contains(CJK).alias("url_has_cjk"),
        slug_type.alias("slug_type"),
        n_tok.alias("slug_ntok"),
        url_norm.alias("url_norm"),
        url_noq.alias("url_noquery"),
    ).drop("_nq")
    df = df.with_columns(
        pl.len().over("url_norm").alias("dup_norm_size"),
        pl.len().over("url_noquery").alias("dup_noquery_size"),
    )
    # ngon ngu doan theo domain: ket qua that se do bang crawl; o day chi dung TLD/ten host
    df = df.with_columns(
        pl.when(pl.col("tld") == "vn").then(pl.lit("vi"))
        .when(pl.col("domain").str.contains(r"vinmec|hellobacsi|benhvienvietduc|pharmacity|longchau|medlatec"))
        .then(pl.lit("vi"))
        .when(pl.col("slug_type").is_in(["numeric_id", "cjk_slug"])).then(pl.lit("zh?"))
        .otherwise(pl.lit("unknown")).alias("lang_guess")
    )

    res["url_dup_exact"] = n - res["url_unique"]
    res["url_dup_norm_groups"] = int(df.filter(pl.col("dup_norm_size") > 1)["url_norm"].n_unique())
    res["url_dup_norm_extra_rows"] = int((df["dup_norm_size"] > 1).sum()) - res["url_dup_norm_groups"]
    res["url_dup_noquery_groups"] = int(df.filter(pl.col("dup_noquery_size") > 1)["url_noquery"].n_unique())
    res["url_dup_noquery_extra_rows"] = int((df["dup_noquery_size"] > 1).sum()) - res["url_dup_noquery_groups"]
    res["url_has_cjk"] = int(df["url_has_cjk"].sum())

    # --- 2. domain
    dom = (
        df.group_by("domain").agg(
            pl.len().alias("n_urls"),
            (pl.col("slug_type") == "meaningful_slug").mean().round(3).alias("r_meaningful_slug"),
            (pl.col("slug_type") == "numeric_id").mean().round(3).alias("r_numeric_id"),
            (pl.col("slug_type") == "cjk_slug").mean().round(3).alias("r_cjk_slug"),
            (pl.col("ext") == "pdf").sum().alias("n_pdf_ext"),
            pl.col("lang_guess").mode().first().alias("lang_guess"),
            pl.col("url").sample(1, seed=SEED).first().alias("example_url"),
        ).sort("n_urls", descending=True)
        .with_columns((pl.col("n_urls") / n * 100).round(3).alias("pct"))
        .with_columns(pl.col("pct").cum_sum().round(3).alias("cum_pct"))
    )
    dom.write_csv(OUT / "domains.csv")
    res["n_domains"] = dom.height
    res["domains_cover_80"] = int((dom["cum_pct"] < 80).sum()) + 1
    res["domains_cover_95"] = int((dom["cum_pct"] < 95).sum()) + 1
    res["top30_domain_pct"] = round(float(dom["pct"].head(30).sum()), 2)
    res["domain_top"] = dom.head(30).select("domain", "n_urls", "pct", "cum_pct", "lang_guess").to_dicts()

    res["ext_dist"] = df["ext"].value_counts().sort("count", descending=True).head(12).to_dicts()
    res["tld_dist"] = df["tld"].value_counts().sort("count", descending=True).head(12).to_dicts()
    res["slug_dist"] = df["slug_type"].value_counts().sort("count", descending=True).to_dicts()
    res["lang_guess_dist"] = df["lang_guess"].value_counts().sort("count", descending=True).to_dicts()
    res["slug_by_lang"] = df.group_by("lang_guess", "slug_type").len().sort("lang_guess", "slug_type").to_dicts()

    df.select(
        "id", "url", "domain", "tld", "ext", "url_has_cjk", "slug_type", "slug_ntok",
        "lang_guess", "dup_norm_size", "dup_noquery_size",
    ).write_parquet(OUT / "corpus_profiled.parquet", compression="zstd")
    return res


def query_profile() -> dict:
    from transformers import AutoTokenizer

    q = pl.read_parquet(RAW / "query.parquet")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    texts = q["query"].to_list()
    n_tok = [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]]
    n_words = [len(t.split()) for t in texts]

    def abbr_terms(t: str) -> list[str]:
        return sorted(set(ABBR.findall(t)))

    def en_terms(t: str) -> list[str]:
        # tu Latin >=4 chu cai ma khong chua dau tieng Viet va khong phai am tiet VN pho bien -> chi la heuristic
        return sorted({w for w in EN_WORD.findall(t) if not w.isupper() and not VN_SYLLABLE.match(w)})

    stats = pl.DataFrame({
        "id": q["id"], "query": q["query"], "n_words": n_words, "n_tok_bgem3": n_tok,
        "n_qmark": [t.count("?") for t in texts],
        "multi_conj": [bool(MULTI.search(t)) for t in texts],
        "abbr_terms": [",".join(abbr_terms(t)) for t in texts],
        "en_terms": [",".join(en_terms(t)) for t in texts],
    }).with_columns(
        (pl.col("n_qmark") >= 2).alias("multi_qmark"),
        (pl.col("abbr_terms") != "").alias("has_abbr"),
        (pl.col("en_terms") != "").alias("has_en"),
    ).with_columns(
        ((pl.col("n_qmark") >= 2) | pl.col("multi_conj")).alias("multi_intent"),
        ((pl.col("n_qmark") >= 2) | pl.col("query").str.contains(r"(?i);|đồng thời")).alias("multi_strict"),
    )
    stats.write_csv(OUT / "query_stats.csv")

    def dist(c: str) -> dict:
        s = stats[c]
        return {k: float(v) for k, v in
                zip(["min", "p25", "median", "p75", "p90", "p95", "max", "mean"],
                    [s.min(), s.quantile(.25), s.median(), s.quantile(.75), s.quantile(.9), s.quantile(.95), s.max(), s.mean()])}

    nq = stats.height
    out = {
        "n": nq, "id_unique": q["id"].n_unique(),
        "words": dist("n_words"), "tokens": dist("n_tok_bgem3"),
        "gt_128_tok": int((stats["n_tok_bgem3"] > 128).sum()),
        "gt_256_tok": int((stats["n_tok_bgem3"] > 256).sum()),
        "multi_qmark": int(stats["multi_qmark"].sum()),
        "multi_conj_va_dongthoi_semicolon": int(stats["multi_conj"].sum()),
        "multi_intent_any": int(stats["multi_intent"].sum()),
        "multi_strict": int(stats["multi_strict"].sum()),
        "has_abbr": int(stats["has_abbr"].sum()),
        "has_en": int(stats["has_en"].sum()),
        "top_abbr": pl.Series([a for s in stats["abbr_terms"] for a in s.split(",") if a]).rename("abbr").value_counts()
        .sort("count", descending=True).head(15).to_dicts(),
        "sample30": stats.sample(30, seed=SEED).select("id", "n_tok_bgem3", "query").to_dicts(),
        "longest20": stats.sort("n_tok_bgem3", descending=True).head(20)
        .select("id", "n_tok_bgem3", pl.col("query").str.slice(0, 160)).to_dicts(),
    }
    return out


def main() -> None:
    auxiliary_disabled()
    OUT.mkdir(exist_ok=True)
    res = {"corpus": corpus_profile(), "query": query_profile()}
    (OUT / "profile_static.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    c, q = res["corpus"], res["query"]
    print(f"rows={c['rows']:,} domains={c['n_domains']} cover80={c['domains_cover_80']} cover95={c['domains_cover_95']}")
    print(f"queries={q['n']} tok median={q['tokens']['median']} p95={q['tokens']['p95']} multi_intent={q['multi_intent_any']}")


if __name__ == "__main__":
    main()
