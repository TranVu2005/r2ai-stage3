> Tài liệu legacy được giữ để tham khảo; root/module/CLI hiện hành theo README.md. Writer phụ đang khóa; không chạy lệnh legacy vào OLD.

# data/dev — pseudo dev sets

Both files are built **only from the crawled corpus** (`data/docs_vi`), no outside data, seed 42.

| file | generator | content |
|---|---|---|
| `pseudo_vi_v2.parquet` (current) | `python scripts/build_pseudo_dev_v2.py [--n-a 400] [--max-per-domain 60] [--seed 42]` | type A (title-as-query) + type B (Q&A) |
| `pseudo_vi.parquet` (v1, kept for reference) | `python scripts/build_pseudo_dev.py [--n 500]` | Q&A questions as written (greetings kept) |

Re-run the generators after more pages are extracted (they read whatever is in `data/docs_vi`).

## v2 columns

| column | meaning |
|---|---|
| `qid` | `pv2A0000`… / `pv2B0000`… |
| `type` | `A` = title-as-query, `B` = reader question of a Q&A page |
| `query` | A: the page title; B: the reader's question with greeting / thanks / signature stripped |
| `src_doc_id` | first `doc_id` of the source page |
| `src_doc_ids_group` | every `doc_id` of the source URL (http/https / duplicate variants merged by `url_norm`) |
| `domain` | source domain |
| `gold_text` | A: the page body; B: the doctor's answer |

## How v2 is built

Measured on the 1,200 test queries first (2026-10-03): 0.8 % start with a greeting ("Thưa bác sĩ", "Bác sĩ ơi" …),
76.3 % end with "?". Because greetings are rare (< 10 %), type B strips them.

**Type A — title-as-query.** Every `lang=vi`, `status=ok` doc in `data/docs_vi` whose title looks like a question:
ends with "?" or contains "có … không", "bao lâu", "là gì", "nên … không", "tại sao", "như thế nào".
Title length 5–120 BGE-M3 tokens. De-duplicated against test and internally (below), then 400 sampled, stratified by
domain in natural proportion (largest remainder) with **at most 60 queries per domain**: a domain whose share exceeds 60
is fixed at 60 and the rest is re-spread proportionally over the other domains (water-filling), seed 42.

**Type B — Q&A.** `question` of vinmec.com and hellobacsi.com pages with non-empty `answer`. Stripped:
a leading greeting ("Chào bác sĩ,", "Thưa bác sĩ", "Bác sĩ ơi" …), trailing reader signatures
("(Hoàng Vinh – Đồng Nai)", "Thu Trà Lê, Cần Đước, Long An") and closing sentences ("Em cảm ơn bác sĩ.",
"Mong bác sĩ tư vấn giúp ạ."). The rest of the wording is unchanged. Kept if 10–350 tokens and answer ≥ 30 tokens.
All eligible questions are kept (no sampling).

**De-duplication (both types).** Text normalised like the metric (`gold_check_common.normalize_text`: Unicode NFC,
lowercase, punctuation → space, whitespace collapsed), BGE-M3 tokens, near-duplicate if `LCS / max(len) >= 0.8`.
Coarse filter before exact LCS: one shared token 2-gram, length ratio ≥ 0.8, token-multiset overlap ≥ 0.8·max(len)
(the last two are exact upper bounds; the 2-gram filter is heuristic). Internally the first item by (domain, doc_id) is kept.

## Evaluation protocol: leave-one-out

The test set was built with the source page removed from the corpus. To mimic that, **remove every id in
`src_doc_ids_group` from the index when scoring the matching query**. Other pages (including near-identical copies
on other URLs) stay in the index.

## State of v2 (2026-10-03)

| | A | B | test |
|---|---|---|---|
| queries | 400 (pool 15,488 of 16,058 eligible; 1 test dup, 569 internal dups) | 62 (vinmec 57, hellobacsi 5) | 1,200 |
| length median / p95 (BGE-M3) | 15 / 25 | 72 / 154 | 23 / 101 |
| ends with "?" | 83.2 % | 93.5 % | 76.3 % |

Type A by domain (35 domains, re-sampled 2026-10-03 with the 60 cap; before the cap tiemchunglongchau.com.vn had 152/400
and giadinhonline.vn 67/400): tiemchunglongchau.com.vn 60, giadinhonline.vn 60, phunusuckhoe.giadinhonline.vn 52,
vietnamnet.vn 32, suckhoecongdongonline.vn 30, tuoitre.vn 28, laodong.vn 26, vov2.vov.vn 25, baogialai.com.vn 11,
vietnamplus.vn 10, vinmec.com 6, baophutho.vn 6, vnexpress.net 6, baochinhphu.vn 5, 4 each: medlatec.vn, nhandan.vn,
hellobacsi.com, baotayninh.vn; 3 each: thaythuocvietnam.vn, tienphong.vn; 2 each: suckhoedoisong.vn, baoangiang.com.vn,
benhviennhitrunguong.gov.vn, tamanhhospital.vn, dantri.com.vn, thanhnien.vn; 1 each: hanoimoi.vn, pharmacity.vn,
baoquangninh.vn, baonghean.vn, sggp.org.vn, suckhoeviet.org.vn, qdnd.vn, baocantho.com.vn, khoahocphothong.vn.
Type B is unchanged by the re-sample (no sampling there).

Caveats: some type A titles are news headlines phrased as questions rather than patient questions. Type B is small
because only 301 docs per large domain are extracted so far (medlatec.vn has no Q&A pages in the corpus: all its URLs
are `/tin-tuc/`).
