# Submission log

The pipeline never uploads; the user uploads by hand and reports the scores. Leaderboard scores are in the
"Leaderboard" section below (the LB columns of the first table are not used).

## Leaderboard (scores provided by the user)

| upload date | file | ZIP SHA256 | JSON SHA256 | config | Final | Doc F2 | Doc P | Doc R | Chunk F2 | Chunk P | Chunk R | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | sub02 / sub03 / sub04 / sub05 / **sub06** / sub07 / sub08 (OLD index, 123,874 docs) | see sections below | | see sections below | 0.0158 / 0.0273 / 0.0551 / 0.0623 / **0.0626** / 0.0545 / 0.0507 | | | sub06: 0.088 | | | | `Downloads/SCOREBOARD.md` (user, 2026-10-03); per-metric columns other than sub06 Doc R not copied here |
| 2026-10-05 | `out/runs/new-full/sub_new_vi_kd100_kc19_full.zip` (39,237,992 B) | `1d171722cd27636a901684778fa0120bf28baab126284d00fc9449820deb567a` | `50ac7367c671104c6ae3f0d686fe7cc40cf69478e2b0fc7d06674ace25f937ab` (156,211,418 B) | sub06 config on the NEW index (3,909,586 t256 chunks / 623,805 docs, K100 `--candidates exact`): k_doc 100 (whole doc_ids_group), k_chunk 19, chunk mode full, dedupe scope doc | **0.1748** | 0.2214 | 0.174 | 0.2897 | 0.1283 | 0.3025 | 0.1249 | provided by the user 2026-10-05 |

vs sub06: Final x2.79, Doc R x3.29 (0.088 -> 0.2897). Coverage of the vi corpus was the bottleneck.

| file | created | config | index | docs in index | build time | Doc F2 (LB) | Chunk F2 (LB) | Final (LB) |
|---|---|---|---|---|---|---|---|---|
| sub01_vi_k5_c2.zip | 2026-10-03 08:10 | chunks t256; hybrid w=0.7 dense + 0.3 sparse (min-max per query); top-200 chunks -> doc max-score -> top-50 docs; bge-reranker-v2-m3 fp16 on the retrieved chunks of those docs; K=5 docs (expanded to doc_ids_group), C=2 answer/body chunks per doc ranked by reranker | data/index/t256, built 2026-10-03 06:11:40, BGE-M3 dense+sparse fp16 max_len 512, FAISS IndexFlatIP, 818,080 chunks | 123,874 (data/docs_vi status ok, body >= 50 tokens) | 3,481 s total, 2.87 s/query mean (measured) | | | |

## sub01_vi_k5_c2

* Command: `python -m submission.build --name sub01_vi_k5_c2 --target 256 --w 0.7 --agg max --rerank --k 5 --c 2`
* Validator (`python -m eval.validate out/submissions/sub01_vi_k5_c2.zip`): 1,200 rows, 0 errors, 0 warnings;
  6,035 relevant_docs (4,965 unique), 11,990 chunks, all verbatim substrings of their doc.
* relevant_docs per query: 5 docs for 1,169 queries, 6 for 27, 7 for 4 (http/https group expansion).
* relevant_chunks per query: 10 for 1,190 queries, 9 for 10 (a top-5 doc with a single answer/body chunk).
  Chunk length (BGE-M3 tokens): median 226, p95 254.
* Config choice: the known-item pseudo-dev ablation is saturated (A R@1 ~ 1, B R@1 = 1: the title/question chunk of the
  source page repeats the query), so the choice used the diagnostic run without title/question chunks
  (out/retrieval/ablation_t256_no-title-question.md): w=0.7, max, rerank had the best or tied A R@5 / R@10 / R@50.
* Chunk doc_id = primary id of the group (doc_ids_group[0]); the scorer only matches chunks on the exact doc_id.
* Corpus: only the pages crawled so far (123,874 of ~654k vi URLs); no outside data.

## sub03–sub05 (K / chunk-mode variants, 2026-10-03)

Built from one cached retrieval run (`python -m scripts.run_retrieval_k100`, 13,083 s, 10.9 s/query): same index, model,
config as sub02 (t256, hybrid w=0.7, max, bge-reranker-v2-m3, seed 42), K_max=100.
Cache: `data/runs/vi_k100.parquet` (query_id, rank, doc_id, score, tier), `data/runs/vi_k100_chunk_scores.parquet`
(reranker score of every answer/body chunk of the top-50 docs, for c2). Docs per query in the cache: min 74, p50 100, 5 queries < 100.
Ranks 1–50 = sub02 procedure (tier 1; top-5 docs identical to sub02 for 1,200/1,200 queries). Ranks 51–100 = tier 2 (new,
needed for K>50): docs not in tier 1 by max hybrid score over all first-stage candidates, reranked among themselves.
Commands: `python -m scripts.make_submission --k K --chunk-mode {c2,full} --out out/submissions/<name>.zip`;
`python -m scripts.validate_submission out/submissions/<name>.zip` (0 errors for all 5 files below).
Check: c2 K=5 with dedupe reproduces sub02 exactly except for dropped chunks (1,067/1,200 queries identical, the other 133 are an ordered subsequence of sub02's).

| file | docs/query min/p50/p95/max | chunks/query min/p50/p95/max | chunk tokens BGE-M3 p50/p95/max | chunks before dedupe | dropped exact+near | top-K docs w/o chunk | json MiB | zip MiB |
|---|---|---|---|---|---|---|---|---|
| sub03_vi_k5_full | 5/5/5/7 | 3/5/5/5 | 1440/2759/12493 | 6000 | 14+71 | 85/6000 | 46.2 | 12.0 |
| sub04_vi_k20_c2 | 20/20/21/25 | 20/40/40/40 | 224/252/376 | 47951 | 347+749 | 317/24000 | 50.5 | 12.5 |
| sub04_vi_k20_full | 20/20/21/25 | 10/20/20/20 | 1423/2695/45518 | 24000 | 66+480 | 546/24000 | 178.5 | 45.7 |
| sub05_vi_k50_c2 | 50/50/51/59 | 70/98/100/100 | 223/252/382 | 119786 | 849+1891 | 796/60000 | 125.7 | 31.1 |
| sub05_vi_k50_full | 50/50/51/59 | 35/49/50/50 | 1381/2587/45518 | 60000 | 174+1233 | 1407/60000 | 425.8 | 110.0 |

Notes
* Token lengths are normalised text (NFKC, lowercase, ...) under the BGE-M3 tokenizer, no special tokens.
* "full" source over the top-K docs of sub03: answer 17, body 4,916, title+description 0 (every indexed doc has a body; few have a separate answer field).
* "full" chunks are very long (some > 8,192 tokens, max 45,518): unknown how the organisers' scorer handles that; sub05_full is 426 MiB json / 110 MiB zip, upload limit not checked.
* Dedupe also removes near-duplicates across different docs (same text on mirrored pages); a doc whose only chunk is dropped stays in relevant_docs.
* Uploaded by hand later by the user (sub03-sub05 have scores, see Leaderboard); the pipeline itself uploads nothing.

## sub04 v2 (dedupe per doc_id, optional length cap, 2026-10-03)

`make_submission.py` gained `--dedupe-scope {doc,query}` (default `doc`: duplicates only within the same doc_id) and
`--max-chunk-tokens N` (cut into consecutive verbatim pieces <= N BGE-M3 tokens at paragraph, then sentence, then word
boundary; a whitespace-free blob larger than N is cut by characters). `validate_submission.py` got the same `--dedupe-scope`.
Old files are untouched. Both v2 files: validator 0 errors. Not uploaded.
Commands: `--k 20 --chunk-mode full --dedupe-scope doc [--max-chunk-tokens 8192]`.

| file | chunks/query min/p50/max | chunk tokens p50/p95/max | dedupe dropped | top-20 docs w/o chunk | chunks cut by cap (-> pieces) | json MiB | zip MiB |
|---|---|---|---|---|---|---|---|
| sub04_vi_k20_full (old, scope=query) | 10/20/20 | 1423/2695/45518 | 546 (66 exact + 480 near) | 546/24000 | - | 178.5 | 45.7 |
| sub04_vi_k20_full_v2 (scope=doc) | 20/20/20 | 1414/2670/45518 | 0 | 0/24000 | 0 | 181.4 | 46.0 |
| sub04_vi_k20_full_cap8k_v2 (scope=doc, cap 8192) | 20/20/26 | 1422/2844/8190 | 0 | 0/24000 | 405 (-> 814) | 181.4 | 46.0 |

* Full mode has one chunk per doc, so per-doc dedupe finds nothing: every one of the 546 old drops was cross-doc.
* With the cap, 405 of 24,000 chunks (1.7 %) exceed 8,192 tokens; they became 814 pieces, so chunks/query is up to 26 (p95 22).
* Pieces keep doc_id of their doc and stay in rank order; text is verbatim (validator substring check passes).
* Only the fallback char-cut can split inside a word-less blob; it was needed to stop an endless loop in the first cap8k attempt (killed and rerun).

## sub05_vi_kd50_kc19_full, sub06_vi_kd100_kc19_full (k_doc / k_chunk split, 2026-10-03)

`make_submission.py`: `--k` replaced by `--k-doc` (relevant_docs = top-k_doc of data/runs/vi_k100.parquet + url_norm group expansion;
ranks 51–100 are tier 2) and `--k-chunk` (chunks only for the top-k_chunk docs); `--max-zip-mib X` lowers k_chunk by 1 until the
zip fits; `{kc}` in `--out` is replaced by the k_chunk used. Zip deflate level 9. `validate_submission.py` got `--max-zip-mib`.
Both: `--chunk-mode full --dedupe-scope doc`, no `--max-chunk-tokens`, limit 45.7 MiB. Validator (`--max-zip-mib 45.7`): 0 errors, 22,800 chunks each.
Not uploaded.

| file | docs/query min/p50/max | k_chunk (requested 20) | chunks/query p50 | chunk tokens p50/p95/max | json MiB | zip MiB (k_chunk tried) |
|---|---|---|---|---|---|---|
| sub05_vi_kd50_kc19_full | 50/50/59 | 19 | 19 | 1415/2694/45518 | 173.1 | 43.64 (kc20: 45.80 > 45.7) |
| sub06_vi_kd100_kc19_full | 74/100/114 | 19 | 19 | 1415/2694/45518 | 173.5 | 43.91 (kc20: 46.08 > 45.7) |

* k_chunk=20 missed the limit by 0.1–0.4 MiB (sub04_vi_k20_full_v2 itself is 46.0 MiB, above the 45.7 of the accepted old file), so k_chunk=19.
* Diff of relevant_chunks vs sub04_vi_k20_full_v2: 1,200/1,200 queries differ, and in every query the new list is exactly the first 19 chunks of the base list (the rank-20 chunk is the only one missing; 1,200 chunks fewer in total). Chunk texts of ranks 1–19 are identical.
* sub06 min 74 docs: the 5 queries with < 100 cached docs (see below).

### First-stage candidate set (config unchanged: t256, hybrid w=0.7, max)
All 1,200 queries from the cache; first-stage numbers from a **sample of 100 queries** (random.seed(42) over query rows, `scripts/measure_candidates.py`, no rerank; raw numbers in data/runs/candidate_stats.json).

| quantity | n | min | p50 | max | mean |
|---|---|---|---|---|---|
| tier-1 docs in cache | 1200 | 50 | 50 | 50 | 50.0 |
| tier-2 docs in cache (capped at 50) | 1200 | 24 | 50 | 50 | 49.9 |
| docs in cache (tier 1 + 2) | 1200 | 74 | 100 | 100 | 99.9 |
| first-stage candidate chunks (dense top-200 U sparse top-200) | 100 (sample) | 285 | 352.5 | 394 | 349.4 |
| distinct candidate docs before rerank | 100 (sample) | 99 | 242 | 325 | 233.9 |
| tier-2 pool = distinct candidate docs not in tier 1 (uncapped) | 100 (sample) | 49 | 192 | 275 | 183.9 |

* Queries with < 100 docs in the cache: 5/1,200 (all in tier 2, min total 74). In the sample: 1/100 has < 100 distinct candidate docs.
* The tier-2 cap of 50 in the cache, not the candidate pool, limits K to 100 for most queries (sample p50 pool is 192 docs); a deeper tier 2 would need a cache re-run with a larger cap.

## sub07–sub10: window chunks + zip budget (2026-10-03)

`make_submission.py`: `--chunk-mode window --window-tokens N` (new module `scripts/window_chunks.py`) and `--zip-budget-mib B`.
Window: one contiguous verbatim passage of <= N BGE-M3 tokens per doc (whole text if the doc is <= N). Centre = best t256 chunk
(bge-reranker score from vi_k100_chunk_scores for the top-50 docs, dense score otherwise); grown unit by unit (paragraph, then
sentence, then word group) toward the side with the higher-scored neighbouring chunk, until nothing fits; the finished window is
re-counted and trimmed if needed. Budget: binary search over k_chunk in [0, k_doc] on the deflate-9 size, then the real zip is written
(and checked <= B). All: `--dedupe-scope doc`, no `--max-chunk-tokens`, validator `--max-zip-mib 45.7` -> 0 errors. Not uploaded.

| file | k_doc | k_chunk | docs/query min/p50/max | chunk tokens p50/p95/max | docs cut by window | docs with dense fallback | chunk tokens vs sub04_vi_k20_full_v2 | json MiB | zip MiB | relevant_docs diff vs base |
|---|---|---|---|---|---|---|---|---|---|---|
| sub07_vi_kd20_kc20_win2048 | 20 | 20 | 20/20/25 | 1414/2032/2048 | 2,790/24,000 = 11.6% | 0 | 85.7% | 156.0 | 39.15 | 0/1200 queries differ (base sub04_vi_k20_full_v2) |
| sub08_vi_kd20_kc20_win1024 | 20 | 20 | 20/20/25 | 989/1022/1024 | 17,102/24,000 = 71.3% | 0 | 58.3% | 107.0 | 27.29 | 0/1200 (sub04_vi_k20_full_v2) |
| sub09_vi_kd100_win1024_budget45 | 100 | 32 | 74/100/114 | 988/1022/1024 | 27,067/38,400 = 70.5% | 0 | 93.0% | 171.3 | 44.00 | 0/1200 (sub06_vi_kd100_kc19_full) |
| sub10_vi_kd100_win2048_budget45 | 100 | 22 | 74/100/114 | 1414/2032/2048 | 3,087/26,400 = 11.7% | 0 | 94.3% | 172.2 | 43.52 | 0/1200 (sub06_vi_kd100_kc19_full) |

* Budget search (estimated deflate size): sub09 k_chunk 33 -> ~45.33 MiB (> 45), 32 -> 44.00 (chosen); sub10 23 -> ~45.44, 22 -> 43.52 (chosen). Zip is monotone in k_chunk in the probes.
* "Dense fallback" is 0 everywhere because k_chunk <= 50, so every chunk doc has reranker scores; the dense path is implemented but unexercised.
* "Chunk tokens vs base" = (mean tokens x chunks) / same for sub04_vi_k20_full_v2 (stats.json means, 24,000 chunks).
* 2048-window changes only 11.6% of docs (the rest are <= 2048 tokens), so its p50 equals the full-text p50 (1414).
* Chunk tokens counted on scorer-normalised text; the cap holds exactly (max = N or below after re-count).

## A/B variants of sub_new_vi_kd100_kc19_full (2026-10-05, not uploaded)

Built without re-running embed/retrieval/rerank, from `out/runs/vi-k100/` (K100 + chunk scores) and the NEW index text
(memory-mapped `data/index/t256/text.arrow`). Baseline = the uploaded file above. Files in `out/runs/ab-2026-10-05/`.

| variant | change | built | why |
|---|---|---|---|
| V1 whole doc_ids_group | relevant_docs side | **no** | the baseline already returns every id of the group (1,200/1,200 queries: relevant_docs == expansion of the K100 primaries); groups are url_norm (http/https/www) groups, max 2 ids, 166 of 119,969 K100 docs have 2 |
| V2 extra chunk for docs ranked 20..N | relevant_chunks side | **yes**, N = 45 | see below |
| V3 / V3b K doc 200 / 150 | relevant_docs side | **no** | `vi_k100.candidates.parquet` holds only counts (`n_candidates`, `n_docs`), no doc order or hybrid score beyond rank 100; ranks 101+ need the query embeddings again (re-run) |
| V4 = V1 + V2 + V3 | | **no** | V1 and V3 not built |

V2 `sub_ab_v2_kd100_kc19_full_xchunk.zip`: `make_submission.py ... --k-doc 100 --k-chunk 19 --chunk-mode full --dedupe-scope doc
--extra-chunk-docs 50 --extra-zip-budget-bytes 47919923`. Docs ranked 20..45 get 1 chunk each = best answer/body t256 chunk by
the cached bge-reranker score (scores exist for ranks 1-50 only, so N <= 50). Binary search on written zips: N 35 -> 44,505,816 B,
43 -> 47,137,420, 47 -> 48,442,791, 45 -> 47,792,584 (chosen), 46 -> 48,115,197.
45 chunks/query (19 + 26), 31,200 extra chunks, extra chunk tokens (BGE-M3) min 2 / p50 222 / p95 253 / max 385 / mean 211.93.
JSON 190,797,582 B SHA256 `a0de9f7a4edc83c5ce8ee30d416377429bf1c2bac3c48b10074d4aeecf7ce614`; ZIP 47,792,584 B (45.58 MiB)
SHA256 `af1864143dff827048c6dfbc16afd664a6758341ec8ab809cde947277bf6290c`. Validator (`--max-zip-mib 45.7`): 1,200 rows,
54,000 chunks, 0 errors. Diff vs baseline: id and relevant_docs identical for 1,200/1,200 queries (id+relevant_docs prefix
byte-identical on every line); relevant_chunks: first 19 identical for 1,200/1,200, extras in rank order 20..45 for 1,200/1,200.
N = 100 (size only, not built: ranks 51-100 have no chunk score, measured with their first answer/body chunk): ZIP 65,855,436 B
(62.81 MiB), 17,935,513 B over the 45.7 MiB budget.
Regression: the default builder after this change rebuilds the uploaded JSON byte-identically (SHA256 `50ac7367...37ab`).
