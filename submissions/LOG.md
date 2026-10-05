# Submission log

The pipeline never uploads; the user uploads by hand and reports the scores. Leaderboard scores are in the
"Leaderboard" section below (the LB columns of the first table are not used).

## Leaderboard (scores provided by the user)

| upload date | file | ZIP SHA256 | JSON SHA256 | config | Final | Doc F2 | Doc P | Doc R | Chunk F2 | Chunk P | Chunk R | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-03 | sub02 / sub03 / sub04 / sub05 / **sub06** / sub07 / sub08 (OLD index, 123,874 docs) | see sections below | | see sections below | 0.0158 / 0.0273 / 0.0551 / 0.0623 / **0.0626** / 0.0545 / 0.0507 | | | sub06: 0.088 | | | | `Downloads/SCOREBOARD.md` (user, 2026-10-03); per-metric columns other than sub06 Doc R not copied here |
| 2026-10-05 | `out/runs/new-full/sub_new_vi_kd100_kc19_full.zip` (39,237,992 B) | `1d171722cd27636a901684778fa0120bf28baab126284d00fc9449820deb567a` | `50ac7367c671104c6ae3f0d686fe7cc40cf69478e2b0fc7d06674ace25f937ab` (156,211,418 B) | sub06 config on the NEW index (3,909,586 t256 chunks / 623,805 docs, K100 `--candidates exact`): k_doc 100 (whole doc_ids_group), k_chunk 19, chunk mode full, dedupe scope doc | 0.1748 | 0.2214 | 0.174 | 0.2897 | 0.1283 | 0.3025 | 0.1249 | provided by the user 2026-10-05 |
| 2026-10-05 | V2 `out/runs/ab-2026-10-05/v2/sub_ab_v2_kd100_kc19_full_xchunk.zip` (47,792,584 B) | `af1864143dff827048c6dfbc16afd664a6758341ec8ab809cde947277bf6290c` | `a0de9f7a4edc83c5ce8ee30d416377429bf1c2bac3c48b10074d4aeecf7ce614` (190,797,582 B) | baseline + 1 extra t256 chunk (best reranker score) for docs ranked 20..45; relevant_docs unchanged | 0.1728 | 0.2214 | 0.174 | 0.2897 | 0.1243 | 0.1507 | 0.1411 | provided by the user 2026-10-05 |

| 2026-10-05 | C `out/runs/ab-2026-10-05b/C/sub_abC_kd100_fullk_probe63mib.zip` (65,062,696 B = 62.05 MiB) | `3d1b045f00245fe5421a1fbb363ae0656d6ea3f5066e8f341d2b23dcbf676bdf` | `e588e7bf49c7f64147d8c65505949dd685fd1537373915c405f44ea638498d17` (260,260,224 B) | baseline + full chunks for docs ranked 20..32 (k_chunk 32); relevant_docs unchanged | 0.1929 | 0.2214 | 0.174 | 0.2897 | 0.1644 | 0.2669 | 0.1713 | provided by the user 2026-10-05 |
| 2026-10-05 | V3 `out/runs/ab-2026-10-05b/V3/sub_abV3_kd200_kc19_full.zip` (39,775,610 B) | `fa73bf9fb387e056fa2220b935cca94182f28c82fb617c97e21530cc2d96ead8` | `984e1cce0f666ff2c9785e4937d069645988aff8374fb10bbb12b3e91afcd844` (157,121,008 B) | baseline + relevant_docs to 200 primary docs (ranks 101+ in hybrid order); chunks unchanged | 0.1758 | 0.2233 | 0.1209 | 0.36 | 0.1283 | 0.3025 | 0.1249 | provided by the user 2026-10-05 |
| 2026-10-05 | V3b `out/runs/ab-2026-10-05b/V3b/sub_abV3b_kd150_kc19_full.zip` (39,519,425 B) | `fe0f217a2343b3b7985e93cabc030307cb00a7290810daaf4eebd1a00116ec1c` | `42a641204d1ed986e89e2a376a400963ab7c440baec57b5d92aadd28ebb6ef97` (156,679,502 B) | baseline + relevant_docs to 150 primary docs (ranks 101+ in hybrid order); chunks unchanged | 0.1783 | 0.2283 | 0.1422 | 0.3359 | 0.1283 | 0.3025 | 0.1249 | provided by the user 2026-10-05 |
| 2026-10-05 | K40 `out/runs/ab-2026-10-05c/K40/sub_abK40_kd150_full40.zip` (81,164,810 B) | `7e64874b158d89bc93a1e29119920b546105dad1088c8ae6e6c18c572da3f65a` | `79bdcf65b29716dda743555019f6564863f3905dddd28a6eab35e99626b0a900` (323,820,612 B) | G (V3b docs K150 + C chunks) + full chunks for ranks 33..40 (k_chunk 40) | 0.2035 | 0.2283 | 0.1422 | 0.3359 | 0.1786 | 0.2485 | 0.1938 | provided by the user 2026-10-05 |
| 2026-10-05 | K50 `out/runs/ab-2026-10-05c/K50/sub_abK50_kd150_full50.zip` (100,135,221 B = 95.50 MiB) | `843e96d9934672e01d568c4a2e8d7396e972aadacf2d91c1d5649d987be3ee32` | `f9844fa2325dbc52883aba08bfca3c35cd3e65e4b894d9c461307d900a222383` (398,880,639 B) | G + full chunks for ranks 33..50 (k_chunk 50) | 0.2083 | 0.2283 | 0.1422 | 0.3359 | 0.1883 | 0.2263 | 0.2122 | provided by the user 2026-10-05 |
| 2026-10-05 | **D50** `out/runs/deep-rerank-2026-10-05/D50/sub_deepD50_kd150_full50.zip` (101,751,549 B = 97.04 MiB) | `0ab1983d41100a6d429bd3b1938034d7534020ff6ea9822955f874e4e2fde127` | `2f10b78ac17f4eafdd5fe1b0b207260b062f17b29f4fcc3ee1a4b364b3c36544` (405,340,395 B) | K50 with doc ranks 1..100 from the deep rerank (tier 1 = 100 docs); config `configs/submission-d50.yaml` | 0.2107 | 0.2283 | 0.1422 | 0.3359 | 0.1932 | 0.2294 | 0.2167 | provided by the user 2026-10-05 |
| 2026-10-05 | R150 `out/runs/rerank200/R150/sub_r200_kd150_full50.zip` (101,642,878 B = 96.93 MiB) | `ba7eea1a1fce0fb6c88c239be59b3548517cb23f5a24128294fd5c93d6606060` | `f8f304fb1ff9acf57339e5660e0b427bc4613600e8fccb897d89d954ebe46731` (403,369,945 B) | rerank tier 1 = 200 (nested on tier 1 = 100), top 150 by reranker; full chunks for the top 50 | 0.2076 | 0.2287 | 0.1423 | 0.3365 | 0.1865 | 0.221 | 0.2093 | provided by the user 2026-10-05 |
| 2026-10-05 | **H1-expand** `out/runs/H1/expand/sub_h1_expand.zip` (101,761,288 B = 97.05 MiB) | `565b96add077b6458ae74d06c282db056a5a7179435bef99ecda5852e9ded9e9` | `6c27d3f327fdcc5dac248cdaf80cf1be941fbff22666b25dbd07f32903f01106` (405,355,611 B) | D50 + ids of duplicate-cluster mates (content scope) appended to relevant_docs; chunks = D50; config `configs/submission-best.yaml` | **0.2109 (BEST)** | 0.2285 | 0.1414 | 0.3372 | 0.1932 | 0.2294 | 0.2167 | provided by the user 2026-10-05 |
| — | G `out/runs/ab-2026-10-05c/G/sub_abG_kd150_full32.zip` | `3cf7112f…78dc` | `eeff96dd…9287` | V3b + C | not uploaded, not needed (K50 > C) | | | | | | | |

C / V3 / V3b vs baseline (each changes one side only, and the other side's metrics are identical to the baseline, as designed):
* **Upload limit**: the LB accepted the 62.05 MiB ZIP (C). The limit is between 62.05 MiB (accepted) and 110 MiB (sub05_full never finished uploading); exact value unknown. New ceiling used for files: 100 MiB = 104,857,600 B.
* **Full chunks for ranks 20..32 still pay**: Chunk F2 0.1283 -> 0.1644 (+0.036), Chunk R 0.1249 -> 0.1713, Chunk P 0.3025 -> 0.2669. Not saturated at rank 32.
* **Doc K**: Doc F2 K150 0.2283 > K200 0.2233 > K100 0.2214; K150 kept (K200 loses more precision, 0.1209, than it gains recall).
* **C + V3b (predicted, not measured)**: Final = (Doc F2 + Chunk F2) / 2 and each variant moves only one F2, so (0.2283 + 0.1644) / 2 = 0.19635. Built as G (see "Pha 1" below).
* Best scored file at that point: C, Final 0.1929 (superseded, see below).

K40 / K50 / D50 (Final = (Doc F2 + Chunk F2) / 2: K40 0.20345, K50 0.2083, D50 0.21075; all match):
* **Best file: D50, Final 0.2107** (sub06 0.0626 -> x3.37 on 2026-10-05). Contributions (Final differences between
  uploaded files, each changing one factor): vi coverage (sub06 -> NEW index baseline) +0.1122; full chunks 19 -> 50
  (V3b -> K50) +0.0300; K doc 100 -> 150 (baseline -> V3b) +0.0035; deep rerank choosing the full-chunk docs (K50 -> D50) +0.0024.
* Full chunks: diminishing returns, Chunk F2 per added rank 19 -> 32 +0.0028, 32 -> 40 +0.0018, 40 -> 50 +0.0010. Short t256
  chunks for ranks 20..45 (V2) lost: no short chunks.
* Doc: K150 > K200 > K100; K150 kept.
* Deep rerank tier 1 = 100 only reorders the top 100 (the 150-doc set is unchanged, Doc F2 stays 0.2283) but picks better
  docs for the 50 full chunks: Chunk F2 0.1883 -> 0.1932 (+0.0049). Deep rerank is part of the best configuration.
* Upload limit: a 97.04 MiB ZIP was accepted; 110 MiB never finished. The limit is between 97 and 110 MiB.
* Uploads on 2026-10-05: 8 (baseline, V2, C, V3, V3b, K40, K50, D50).

R150 / H1-expand (Final = (Doc F2 + Chunk F2) / 2: R150 (0.2287 + 0.1865) / 2 = 0.2076, H1-expand (0.2285 + 0.1932) / 2 =
0.21085 -> 0.2109; both match):
* **Best file: H1-expand, Final 0.2109** (D50 + 0.0002). Config `configs/submission-best.yaml` = D50 + expand post-processing.
* R150 loses to D50 by 0.0031. Doc side almost unchanged (Doc F2 +0.0004) despite 24.67 new primary docs per query; Chunk F2
  -0.0067 because 9.13 full-chunk docs per query changed: the docs the reranker lifts from depth (hybrid ranks 101..200) are worse
  than the ones they replace. R120 / R180 not uploaded (same chunks as R150, so the same loss). No more investment on the doc side.
* H1-expand: Doc R +0.0013, Doc P -0.0008, chunks identical to D50 -> gold contains part of the copies (G1 in part). H1-dedup dropped.
* Rule (derived from the formula, not measured): per query F2 = 5 TP / (4 |G| + |D|), so adding one item that is right with
  probability h raises F2 iff h > F2 / 5. With the current scores the thresholds are about 4.6 % for docs (0.2283 / 5) and
  3.9 % for chunks (0.1932 / 5). The LB averages per query, so this holds per query, not exactly for the mean.

vs sub06: Final x2.79, Doc R x3.29 (0.088 -> 0.2897). Coverage of the vi corpus was the bottleneck.

V2 vs baseline: doc side identical (by design). Chunk P 0.3025 -> 0.1507, Chunk R 0.1249 -> 0.1411, Chunk F2 0.1283 -> 0.1243, Final -0.0020. The 26 short chunks are right only ~4 % of the time (estimate from the macro means, vs ~30 % for the 19 full chunks): precision halves and cancels the recall gain. **The baseline stays the best file.** Lesson: the value is in full chunks (one full chunk covers several reference chunks); no more short extra chunks.

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

## A/B variants of sub_new_vi_kd100_kc19_full (2026-10-05; V2 uploaded, score in Leaderboard)

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

## A / B / C variants of sub_new_vi_kd100_kc19_full (2026-10-05, not uploaded)

Baseline (best so far, Final 0.1748) = `out/runs/new-full/sub_new_vi_kd100_kc19_full.zip`. Each file changes exactly one
factor; files in `out/runs/ab-2026-10-05b/`. No rerank, no corpus embedding. Common builder args: `--k-doc 100 --k-chunk 19
--chunk-mode full --dedupe-scope doc --runs-dir out/runs/vi-k100`.

| variant | change | extra args | relevant_docs ids/query min/p5/p50/max | chunks/query | JSON B | ZIP B | ZIP SHA256 | validator | out-of-scope diff |
|---|---|---|---|---|---|---|---|---|---|
| A `A/sub_abA_kd100_fullk_45mib.zip` | full chunks for docs ranked 20..23 | `--k-chunk-zip-budget-bytes 47919923 --k-chunk-max 50` | = baseline | 23 | 188,731,563 | 47,277,964 | `bf2b94a36cdf159b0dcd95a3d6a02d2fdd152e07941775b2087eda7bea4d3c23` | 0 errors (`--max-zip-mib 45.7`) | 0 |
| V3 `V3/sub_abV3_kd200_kc19_full.zip` | relevant_docs to 200 primary docs | `--doc-ranking .../cand/vi_cand.docs.parquet --k-doc-total 200` | 79 / 169 / 200 / 212 | 19 | 157,121,008 | 39,775,610 | `fa73bf9fb387e056fa2220b935cca94182f28c82fb617c97e21530cc2d96ead8` | 0 errors (`--max-zip-mib 45.7`) | 0 |
| C `C/sub_abC_kd100_fullk_probe63mib.zip` | full chunks for docs ranked 20..32; **upload-limit probe, upload only after A has a score** | `--k-chunk-zip-budget-bytes 65855436 --k-chunk-max 60` | = baseline | 32 | 260,260,224 | 65,062,696 | `3d1b045f00245fe5421a1fbb363ae0656d6ea3f5066e8f341d2b23dcbf676bdf` | 0 errors (no size limit) | 0 |
| V3b `V3b/sub_abV3b_kd150_kc19_full.zip` | relevant_docs to 150 primary docs | `--doc-ranking ... --k-doc-total 150` | 79 / 150 / 150 / 159 | 19 | 156,679,502 | 39,519,425 | `fe0f217a2343b3b7985e93cabc030307cb00a7290810daaf4eebd1a00116ec1c` | 0 errors (`--max-zip-mib 45.7`) | 0 |

JSON SHA256: A `d63537dd9cf7a56f8aecb9880f9d8ce6db01a99ffda9d9c8edf258f90d0d975f`, V3 `984e1cce0f666ff2c9785e4937d069645988aff8374fb10bbb12b3e91afcd844`,
C `e588e7bf49c7f64147d8c65505949dd685fd1537373915c405f44ea638498d17`, V3b `42a641204d1ed986e89e2a376a400963ab7c440baec57b5d92aadd28ebb6ef97`.
Suggested upload order: **A -> V3 -> C -> V3b** (depending on the scores).

* A / C: k_chunk search on written zips. A: k 35 -> 70,988,656 B, 27 -> 55,335,458, 23 -> 47,277,964 (chosen), 25 -> 51,320,743,
  24 -> 49,304,898. C: 40 -> 80,883,339, 29 -> 59,335,467, 34 -> 69,006,693, 31 -> 63,212,600, 32 -> 65,062,696 (chosen),
  33 -> 67,034,795. Each rank adds ~2 MB of zip. Added full chunks: A 4,800 (chars p50 4,859 / p95 8,982 / max 47,450;
  BGE-M3 tokens p50 1,278.5 / p95 2,375 / max 12,367); C 15,600 (chars p50 4,825 / max 251,693; tokens p50 1,274 / max 69,680),
  same rule as the baseline (whole answer if inside the body, else whole body, no cap). id and relevant_docs byte-identical,
  first 19 chunks byte-identical and later chunks in K100 rank order for 1,200/1,200 queries.
* V3 / V3b: docs after the K100 cache = candidate docs not in K100, ordered by max hybrid chunk score (w 0.7, min-max over the
  query's candidates), each expanded to its doc_ids_group. relevant_chunks byte-identical and the baseline ids are a byte-identical
  prefix (same order) for 1,200/1,200 queries; no duplicate ids. Queries short of the target (not enough candidate docs): 155
  for K200, 40 for K150. Primary docs/query K200 min 79 / p5 169 / p50 200; K150 min 79 / p5 150 / p50 150.
* Candidate ranking: `run_retrieval_k100 --candidates exact --candidates-only` (queries encoded on cuda, exact scan on cpu, no
  rerank) -> `out/runs/ab-2026-10-05b/cand/` (`vi_cand.docs.parquet` 317,765 rows, `vi_cand.chunks.parquet` 431,653 rows).
  429 s (dense 95.2, sparse 253.5, union+rescore 41.3), peak RSS 5.552 GiB, private 6.411 GiB, VRAM 1,338 MiB.
  Gate: n_candidates and n_docs equal `vi_k100.candidates.parquet` for 1,200/1,200 queries; K100 docs within the candidate docs
  1,200/1,200; tier-1 set (top-200 chunks -> top-50 docs by hybrid) and tier-2 set (next docs by hybrid order) reproduced 1,200/1,200.
* Default builder after this change still rebuilds the uploaded JSON byte-identically (SHA256 `50ac7367...37ab`).

## Pha 1: G / K40 / K50 (C + V3b combined, deeper full chunks; 2026-10-05, not uploaded)

Files in `out/runs/ab-2026-10-05c/` (ignored). No GPU, no rerank, no corpus embedding; builder code unchanged (the flags
already existed). Common args: `--k-doc 100 --chunk-mode full --dedupe-scope doc --runs-dir out/runs/vi-k100
--chunks-dir data/chunks --docs-dir data/docs_vi --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet
--doc-ranking out/runs/ab-2026-10-05b/cand/vi_cand.docs.parquet --k-doc-total 150`, plus `--k-chunk 32 / 40 / 50`.
`--docs-dir` / `--chunks-dir` must be given: a first attempt with the default docs dir read the legacy docs (29,333 of
38,400 chunk docs without text); its outputs are kept apart in `_invalid_default_docs_dir/` and not used.

| file | control | change | primary docs/query min/p5/p50/max | chunks/query | JSON B | ZIP B | ZIP SHA256 | validator | out-of-scope diff |
|---|---|---|---|---|---|---|---|---|---|
| G `G/sub_abG_kd150_full32.zip` | V3b (docs), C (chunks) | relevant_docs of V3b + relevant_chunks of C | 79 / 150 / 150 / 150 | 32 | 260,728,308 | 65,344,576 | `3cf7112f011567cc9da4654d3c45400224fde4c03628d1ccd6236c07b04178dc` | 1,200 rows / 38,400 chunks / 0 errors (`--max-zip-mib 100`) | relevant_docs byte-identical to V3b 1,200/1,200; relevant_chunks byte-identical to C 1,200/1,200 |
| K40 `K40/sub_abK40_kd150_full40.zip` | G | full chunks for ranks 33..40 | = G | 40 | 323,820,612 | 81,164,810 | `7e64874b158d89bc93a1e29119920b546105dad1088c8ae6e6c18c572da3f65a` | 1,200 / 48,000 / 0 errors (`--max-zip-mib 100`) | relevant_docs byte-identical 1,200/1,200; first 32 chunks identical 1,200/1,200; added chunks in K100 rank order 33..40 1,200/1,200 |
| K50 `K50/sub_abK50_kd150_full50.zip` | G | full chunks for ranks 33..50 | = G | 50 | 398,880,639 | 100,135,221 | `843e96d9934672e01d568c4a2e8d7396e972aadacf2d91c1d5649d987be3ee32` | 1,200 / 60,000 / 0 errors (`--max-zip-mib 100`) | as K40, added ranks 33..50 |

JSON SHA256: G `eeff96dddfb75ffb517ff83870dceec1c114ad7eadaaf67d8c1d19bef8129287`, K40
`79bdcf65b29716dda743555019f6564863f3905dddd28a6eab35e99626b0a900`, K50 `f9844fa2325dbc52883aba08bfca3c35cd3e65e4b894d9c461307d900a222383`.
Each ZIP has exactly one entry (`<stem>.json`), no directory.

* K50 at k=50 is 100,135,221 B (95.50 MiB) <= the 104,857,600 B ceiling, so no k search was needed (k=50 reached).
  ZIP grows ~2.0 MB per rank (K40 -> K50: +18,970,411 B for 10 ranks).
* Added full chunks (same rule as the baseline: whole answer if inside the body, else whole body; no cut):
  K40 vs G 9,600 chunks, chars min 196 / p50 4,752.5 / p95 8,982 / max 162,183; BGE-M3 tokens min 50 / p50 1,256 / p95 2,358 / max 45,514.
  K50 vs G 21,600 chunks, chars min 196 / p50 4,631 / p95 8,789 / max 162,183; tokens min 50 / p50 1,221 / p95 2,306 / max 45,514.
* Checks: `out/runs/ab-2026-10-05c/verify.py` -> `verify.json` (line-level byte diff of the JSON files); `check_all.sh` runs it, the validators (`--corpus D:/GitHub/r2ai-stage3-old/data/raw/links_corpus.parquet --docs-dir data/docs_vi --max-zip-mib 100`) and the regression.
* Regression: the default builder still rebuilds the uploaded baseline JSON byte-identically (SHA256 `50ac7367...37ab`, `regress/`).
* Predicted (extrapolation, not measured): G Final ~0.1964 (C chunk side + V3b doc side).
* **Suggested upload order: K40 -> K50 -> G** (G only if K40 / K50 do not beat C).

## Pha 2: deep rerank, tier 1 = 100 docs -> D32 / D40 / D50 (2026-10-05, not uploaded)

Files in `out/runs/deep-rerank-2026-10-05/` (ignored). Same index, models, seed 42 and first stage as `vi-k100`.

Retrieval: `run_retrieval_k100 --tier1-docs 100 --tier2-docs 50 --chunk-score-docs 0 --pair-scores` -> `full/`
(`vi_k100.parquet` 150 docs/query except 40 short queries, min 79; `vi_k100_pairs.parquet` 258,117 reranked pairs: tier 1
188,704, tier 2 69,413). Tier 1 = the top-100 docs by max hybrid over the top-200 chunks, all reranked together on their
top-200 chunks; for the 86 queries whose top-200 chunks cover < 100 docs the hybrid order is followed past rank 200 and a doc
first met there is scored on its chunks up to that point (docs already in the top 200 keep exactly their top-200 chunks).
The top-50 docs' pairs are scored in a reranker call of their own (see gate). Tier 2 = next 50 docs by hybrid, reranked among
themselves. `--chunk-score-docs 0`: the c2 per-chunk scores of the top-50 docs are skipped (full mode does not use them).

* Measured: 3,743 s for 1,200 queries (candidates: dense 64.3 s, sparse 143.0 s, union 10.4 s), 2.915 s/query mean, p95 4.09 s;
  rerank pairs/query tier 1 157.25, tier 2 57.84; peak RSS 7.466 GiB, private 7.120 GiB; VRAM torch max allocated 1,452 MiB,
  nvidia-smi peak 2,836 MiB (14 samples at 3,938 MiB are excluded: a duplicate run started by mistake on the same out dir
  for ~30 s, killed; it wrote no checkpoint). Trial (50 queries, seed 42): 2.80 s/query; with `--chunk-score-docs 50` 7.79 s/query.
* Gate deep (`python -m r2ai.retrieve.deep_gate deep --old out/runs/vi-k100 --new .../full`): old tier-1 (query, doc) pairs
  60,000/60,000 with |delta| = 0 (max 0.0); relative order of the old 50 docs identical in 1,200/1,200 queries; chunk level
  93,775 common rows, |delta| 0. A first trial scoring all 100 docs in one call failed (50 queries: 93.8 % pairs within 1e-3,
  max 0.0117, order changed in 3/50): fp16 reranker logits move by 1-3 ULP with the batch/padding composition, and one ULP
  is >= 0.00195 for |score| >= 2. Fix: the top-50 docs' pair list is scored in its own call (same batches as the default run).
* Gate same (`--tier1-docs 50 --sample 20`): rank / doc_id / tier identical 20/20 queries, doc and chunk scores |delta| = 0
  (2,000 doc rows, 7,927 chunk rows).
* Deep top-100 doc set == old K100 set for 1,200/1,200 queries (only the order changes). Old tier-2 docs inside the new top-k:
  mean 8.4 (k 32) / 12.0 (k 40) / 17.0 (k 50).

Build: Pha 1 args with `--runs-dir out/runs/deep-rerank-2026-10-05/full` (ranks 1..100 = deep order; ranks 101..150 from
`vi_cand.docs.parquet`, hybrid order, as in G). Builder fix in this change: `--doc-ranking` skipped every doc present in the
cache file, so a 150-doc cache would have hidden ranks 101..150 and pulled hybrid ranks 151..200; it now skips only the top
k_doc cached docs (identical for the 100-row `vi-k100` cache: baseline and G rebuild byte-identically, see below).

| file | control | change | primary docs/query | chunk/query | JSON B | ZIP B | ZIP SHA256 | validator | out-of-scope diff |
|---|---|---|---|---|---|---|---|---|---|
| D32 `D32/sub_deepD32_kd150_full32.zip` | G | doc order 1..100 from deep rerank | = G | 32 | 261,441,816 | 65,717,360 | `4ee026a5068ce378f481963dbc41e85bbd1c3c3e386836fa8b1675292e0a93bc` | 1,200 / 38,400 / 0 errors | 150-doc set differs in 0/1,200 queries; chunk docs changed mean 8.40 / p50 8 / max 21 per query (1,196 queries); same doc -> same chunk text |
| D40 `D40/sub_deepD40_kd150_full40.zip` | K40 | same | = K40 | 40 | 325,781,533 | 81,757,628 | `b4604c9e5fd4dda623370641cb2428cbab73e6fab31931b2ee4ed6b048be8d88` | 1,200 / 48,000 / 0 errors | set differs 0/1,200; chunk docs changed mean 11.99 / p50 12 / max 27 (1,200 queries) |
| D50 `D50/sub_deepD50_kd150_full50.zip` | K50 | same | = K50 | 50 | 405,340,395 | 101,751,549 | `0ab1983d41100a6d429bd3b1938034d7534020ff6ea9822955f874e4e2fde127` | 1,200 / 60,000 / 0 errors | set differs 0/1,200; chunk docs changed mean 17.04 / p50 17 / max 32 (1,200 queries) |

JSON SHA256: D32 `f99476fc6b5e43a3a5ccf249969e738453070fda46049df6a9924509b8171fc8`, D40
`8fff4b51d11f07f463ef736ea0224efb33bbba1be479e17fab72f6181317e313`, D50 `2f10b78ac17f4eafdd5fe1b0b207260b062f17b29f4fcc3ee1a4b364b3c36544`.
All ZIPs: one entry, no directory, <= 104,857,600 B (D50 97.04 MiB). Validator `--max-zip-mib 100`. Checks: `compare_d.py` -> `compare_d.json`.
Regression after the builder fix: baseline JSON SHA256 `50ac7367...37ab` and G `eeff96dd...9287` rebuild byte-identically (`regress/`, `regress_G/`).
relevant_docs order differs from the control in every query (by design: ranks 1..100 reordered); the doc set is identical.

## Pha 3: rerank tier 1 = 200 docs, top K by reranker -> R150 / R120 / R180 (2026-10-05; R150 uploaded 0.2076, R120 / R180 dropped)

Agent rerank200; report `out/runs/rerank200/REPORT.md`. Retrieval `run_retrieval_k100 --candidates exact --tier1-docs 200
--tier1-base 100 --tier2-docs 0 --chunk-score-docs 0 --pair-scores` -> `out/runs/rerank200/full/` (200 docs/query, all ordered
by the reranker). `--tier1-base 100` (new, off by default) keeps the docs, chunks and the two reranker calls of the tier-1 = 100
run and scores the pairs of docs 101..200 in a third call. A plain `--tier1-docs 200` trial (20 queries, seed 42) failed the
gate: 82/2,000 old doc scores moved (max 0.0156), order changed in 2/20 queries (fp16 batch effect, see Pha 2). Nesting leaves
145 chunks (53/1,200 queries) unscored that a plain run would add to docs met past the top-200 chunks; the 200-doc set is the same.

* Trial (nested, `--sample 20`): 3.23 s/query, peak RSS 5.166 GiB, VRAM torch 1,402 MiB / nvidia-smi 2,687 MiB; gate pass.
* Full 1,200 queries: wall 4,192 s (in-process 4,187 s), **3.212 s/query** (p95 4.62), candidates dense 84.2 s / sparse 169.5 s /
  union 41.0 s, peak RSS 6.283 GiB, peak private 7.312 GiB, VRAM torch 1,452 MiB / nvidia-smi peak 2,933 MiB (whole GPU).
* Gate (`deep_gate deep --old out/runs/deep-rerank-2026-10-05/full --tol 0 --pairs`): 119,969 old tier-1 (query, doc) scores and
  188,704 (query, doc, chunk) scores |delta| = 0, relative order 1,200/1,200, 0 missing.

Build: D50 chunk config with the new order, `--k-doc 100 --k-doc-total K --doc-ranking out/runs/rerank200/full/vi_k100.parquet
--runs-dir out/runs/rerank200/full --k-chunk 50 --chunk-mode full --dedupe-scope doc --docs-dir data/docs_vi --chunks-dir data/chunks`
(`out/runs/rerank200/build_r.sh`; checks `compare_r.py` -> `compare_r.json`, control = D50 rebuilt from its caches).

| file | primary docs/query | k_chunk | JSON B | JSON SHA256 | ZIP B | ZIP SHA256 | validator | vs D50 (per query) |
|---|---|---|---|---|---|---|---|---|
| **R150** `R150/sub_r200_kd150_full50.zip` | 150 | 50 | 403,369,945 | `f8f304fb1ff9acf57339e5660e0b427bc4613600e8fccb897d89d954ebe46731` | 101,642,878 (96.93 MiB) | `ba7eea1a1fce0fb6c88c239be59b3548517cb23f5a24128294fd5c93d6606060` | 1,200 / 60,000 / 0 errors | new docs in top 150 mean 24.67 / p50 26 / max 42; full-chunk docs changed mean 9.13 / p50 9 / max 26 |
| R120 `R120/sub_r200_kd120_full50.zip` | 120 | 50 | 403,090,890 | `2e607d049996ef31152034d73339a4cdd3e8322a987096a78a63708e507a3d1a` | 101,475,556 (96.77 MiB) | `6518cc12f3eefd363df3c513ee4f2e80a788f72d1496ac934fd6411c28985426` | 0 errors | new docs 15.66 / p50 17; D50 docs dropped 45.03; chunk docs as R150 |
| R180 `R180/sub_r200_kd180_full50.zip` | 180 | 50 | 403,641,105 | `9745b242ed70ab8e8dc75a96078c2805bffdbbf0041b80764eddd1a8faaf005e` | 101,799,807 (97.08 MiB) | `db0e8aec98254fd055ee7db0e92624c20dc5184b7662c1ba08dd9eb15f7c8db0` | 0 errors | new docs 36.52 / p50 38; D50 docs dropped 8.02; chunk docs as R150 |

All ZIPs <= 104,857,600 B with k_chunk 50 (no reduction); R180 is 0.04 MiB above the accepted 97.04 MiB mark. Of the 9.13
changed full-chunk docs per query, 3.17 are docs outside D50's 150, the rest are D50 ranks 101..150 (hybrid order there)
lifted by the reranker. LB: R150 0.2076 < D50 0.2107 (Leaderboard section); R120 / R180 dropped.

## H1: duplicate document clusters -> H1-expand / H1-dedup (2026-10-05; H1-expand uploaded 0.2109 = best, H1-dedup dropped)

Agent H1; report `out/runs/H1/REPORT.md`, code `src/r2ai/dupes/` (commit `8fdc426`). Summary of its measured numbers:
623,852 ok docs -> 21,074 clusters (exact on metric-normalised body + MinHash LSH 128 perm, 5-word shingles, verified
Jaccard >= 0.8, seed 42), 47,086 docs = 7.55 % of ok docs (26,012 redundant, 4.17 %); 83 % of clusters span several domains;
401 boilerplate clusters (2,429 docs) excluded from the variants. On D50 (content scope, per query): slots in the 150 docs taken
by a same-cluster doc mean 4.06 / p50 4 / p95 9 (1,144/1,200 queries); same-cluster full chunks in the top 50 mean 1.34, of
which identical text 0.45; same-cluster docs outside the top 150 mean 1.66. Identity mode rebuilds D50 byte-identically.
* H1-expand `out/runs/H1/expand/sub_h1_expand.zip`: D50 + 2,000 ids of same-cluster docs appended (809 queries), relevant_chunks
  byte-identical to D50; ZIP 101,761,288 B (97.05 MiB), SHA256 `565b96add077b6458ae74d06c282db056a5a7179435bef99ecda5852e9ded9e9`; validator 0 errors.
* H1-dedup `out/runs/H1/dedup/sub_h1_dedup.zip`: one doc per cluster, refilled to 150; 1,144 queries changed; ZIP 103,560,765 B
  (98.76 MiB, above the accepted 97.04 MiB mark), SHA256 `08d111ce1a30f273eb28c47d56a89ecf3ce37efec83f0f84ac2100af84f46c57`; validator 0 errors.
* H1-expand uploaded: 0.2109 (best; Doc R +0.0013, Doc P -0.0008 vs D50). H1-dedup dropped (gold holds part of the copies).

## Best configuration and replay (2026-10-05)

`configs/submission-best.yaml` = D50 + H1-expand (Final 0.2109); `configs/submission-d50.yaml` = D50 (the former best config,
unchanged). Retrieval `python scripts/run_best_retrieval.py` (candidates, then deep rerank; resumable), build
`python scripts/build_best_submission.py [--config ...] [--from-retrieval]`: make_submission, then (when
`postprocess.expand_clusters.enabled`) `r2ai.dupes.postprocess.expand_submission` with `out/runs/H1/clusters.parquet`, scope
content; checks the builder JSON SHA256 (`base_json_sha256`) and the final one (`expected_json_sha256`). The clusters must be
rebuilt when the corpus changes (README).
Replays from the existing caches (no GPU), `out/runs/rerank200/partB/`:
* best: JSON 405,355,611 B, SHA256 `6c27d3f327fdcc5dac248cdaf80cf1be941fbff22666b25dbd07f32903f01106` = uploaded H1-expand (match);
  builder JSON `2f10b78a...6c36544` (match); 809 queries / 2,000 ids added; ZIP 101,761,300 B (<= 104,857,600; 12 B more than the
  uploaded ZIP: longer entry name); validator 1,200 / 60,000 / 0 errors; wall 281 s (builder 236.8 s, expand ~42 s); validator 117 s.
* d50 (`--config configs/submission-d50.yaml`): JSON 405,340,395 B, SHA256 `2f10b78a...6c36544` (match), ZIP 101,751,543 B; wall 208 s.
Earlier D50 replay: `out/runs/replay-best-2026-10-05/`, builder 222.2 s.

## RRF choice of the full-chunk docs (2026-10-05, not uploaded)

Hypothesis: the reranker errs at depth while the hybrid score still carries information. `python -m r2ai.submit.rrf_chunks
build --base out/runs/rerank200/partB/best/sub_best_d50_expand.json --out out/runs/rerank200/partC/sub_rrf_d50x_full50.zip`:
relevant_docs of the best file kept; the 50 full-chunk docs are the D50 primary docs (150) with the best
1/(60 + rank_reranker) + 1/(60 + rank_hybrid) (reranker rank from `out/runs/rerank200/full/`, 200-doc cache, all D50 docs present;
hybrid rank from `vi_cand.docs.parquet`, the order the D50 builder used); chunk text = builder full-chunk rule, RRF order.
* Measured before the build: chunk docs changed vs best mean 8.56 / p50 8 / p95 13 / max 22, all 1,200 queries; 10,272 new
  chunk docs, of which 8,282 (80.6 %) have reranker rank > 50 and 1,990 (19.4 %) hybrid rank > 100 (0 both).
* File: ZIP 101,641,704 B (96.93 MiB), SHA256 `34eb7b2a7337a5f32ea3cb87c7bd65d7d51397abf3be34a1feed83e3e930bd8f`; JSON 402,439,775 B,
  SHA256 `ff035b8ba7eea0a177932850d3a9cc9f1ffbcbe8297b8a8740bae23bd949d1c3`; validator 1,200 / 60,000 / 0 errors; diff vs best:
  id / keys / relevant_docs identical 1,200/1,200, relevant_chunks differ 1,200/1,200, every chunk doc in relevant_docs; the 49,728
  kept chunk docs have byte-identical chunk text. Build 111 s, validator 129 s.
* Score: not measured. Only the chunk side moves, so Final changes by half the Chunk F2 change; it pays only if the swapped-in
  docs are right more often than the ones they replace.

## Pending uploads

RRF (`out/runs/rerank200/partC/sub_rrf_d50x_full50.zip`) is the only built candidate; one slot, compared with the best file
(0.2109). D32 / D40, G, R120 / R180 and H1-dedup are not to be uploaded.
