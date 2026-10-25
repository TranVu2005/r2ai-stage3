# Artifacts the final builds need (retention list)

Generated from `configs/s36-handoff-s30-manifest.json` and the two fixed configs. The manifest digest is `45ce6e493bc275aa71bd6bfda646de5c4ac5314d375866e19a58666db963e1f7` over its CRLF form; git stores the file with LF, whose digest is `e5010f99ea036064b89f67516a25944213f581ab9216f96baf4f46e66ff4af79`. `python -m r2ai.release check` compares every file of sections 1 and 2 with its SHA256.

Nothing here may be deleted or regenerated before the private-set decision. Nothing was deleted by the release work.

## 1. S36 base data (both builds)

| key | path | SHA256 |
|---|---|---|
| `s31:pool13_index` | `D:/GitHub/r2ai-stage3/out/runs/s31-strong-judge/pharmacity-ext/pool13_index.parquet` | `ecf0ea1b3e179a8553f7b549a13dc0c0ff7daa6b0cb3b38260bba9afd4ac3e3f` |
| `s31:q06` | `D:/GitHub/r2ai-stage3/out/runs/s31-strong-judge/pharmacity-ext/q06_scores.parquet` | `3b04a00fdea5646286189ebce4a9a9cda453053bd4c0dc2f6cf16b43f7f0118d` |
| `s31:judge` | `D:/GitHub/r2ai-stage3/out/runs/s31-strong-judge/pharmacity-ext/qwen3-reranker-4b-scores-pool13.parquet` | `ddf24b91718d6fe7b24ef36ce8f2b3a16dbb1d1f765134d9f6e97cdfb100db0f` |
| `s31:pool_index` | `D:/GitHub/r2ai-stage3/out/runs/s31-strong-judge/pool/pool_index.parquet` | `71ae4ad0468790bd407de32d256871b06b780540b48ef0dc64a1828b22df515d` |
| `s31:orig` | `D:/GitHub/r2ai-stage3/out/runs/s31-strong-judge/gpu/qwen3-reranker-4b/scores-pool13.parquet` | `39d554f2996e44db6a915ff39eaf10f9720d4614215022f5a58999a068ea84fb` |
| `s31:extra_pairs` | `D:/GitHub/r2ai-stage3/out/runs/s31-strong-judge/pool-extra/extra_pairs.parquet` | `389f6d39568916104546d9d52968671536959c563ff30aa46f58d4e79b674c04` |
| `s31:extra_scores` | `D:/GitHub/r2ai-stage3/out/runs/s31-strong-judge/gpu/qwen3-reranker-4b/scores-extra.parquet` | `84cf53fa8b90f7db2d34bf23f25999554a899dd7a66df0529fa32b598286f0e3` |
| `s35_jina_scores` | `D:/GitHub/r2ai-stage3/out/runs/s35-part-jina/jina/scores-all.parquet` | `e9a4818eeb720b83711616b497c94f87e3c5d1cceffa6fd26b655f6c6489cf39` |
| `s32_doc_scores` | `D:/GitHub/r2ai-stage3/out/runs/s32-chunk-judge/scores/doc_scores.parquet` | `bd64145ec258ce6b5b93d1072563d307ce3d5a3940fd895a532af704589912b0` |
| `s36_extra_jina` | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/extra/jina-scores.parquet` | `a4fac30a7d29ed1601e07e442edb1a6f8d56649a58711aab792ab4350aac93b6` |
| `s36_extra_cmax` | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/extra/cmax-scores.parquet` | `af3f2e84e9fdf78f7bf5ec4c7f00613241b8b5bafc996b0d9013df69cb7d013f` |
| `cal.json` | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/cal/cal.json` | `a1517b501b12873b14bef3b3c0d60daf905f21f4e0e1d088bd012096bbaab006` |
| `ef2-free.json` | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/cal/ef2-free.json` | `88856110a7f8ce070ddb28632c45baffed9f96131c0c4e74f70199051ac98490` |
| `pool.npz` | `D:/GitHub/r2ai-stage3/out/runs/s37/pool.npz` | `45f69e563d2df74b983bc5441aeccad18251fe9606b5694bea335b2c87c31cdd` |
| `query_vectors.npz` | `D:/GitHub/r2ai-stage3/out/runs/s29-qwen-judge/query_vectors.npz` | `eed0e1046ffce6925987d21fdf832b99d548a7837c592ef21cf75e48cb02edca` |
| `baseline_submission` | `D:/GitHub/r2ai-stage3/out/runs/s26-gold-structure/pa4/P-A4-K90-ph.zip` | `b291124fa22e9f07fbddaeb3891455ff2c71757f3de1a1c0404fb82382389c6e` |
| `p1_submission` | `D:/GitHub/r2ai-stage3/out/runs/reranker-trial/release/P1-vi+10.zip` | `9e83ac7b634aeecfb2cfe91ef522249753b7ac3a845c330cdfbdc1d366221009` |
| `canonical_docs.parquet` | `D:/GitHub/r2ai-stage3/data/canonical-20261018/docs.parquet` | `80c52371d31093de717123004a44ec510019e74b272d1d64a57e458e00eaed7b` |
| `s37_handoff_manifest` | `D:/GitHub/r2ai-stage3/out/runs/s37/handoff/manifest.json` | `995a77ed655281c1e4367a228218048dae340a697dd7d883c71f700c17f88dff` |

## 2. 120A-p4b-v3 (main): inputs named by `configs/s36-handoff-120A-p4b-v3.json`

| kind | path | SHA256 |
|---|---|---|
| pairs_new | `D:/GitHub/r2ai-stage3/out/runs/s37/pairs_new_final_v2.parquet` | `6d2c08e56b76a1b424e00512bab04edfe337959ce550f150916d63d03975d4ae` |
| pairs_new | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/d300/pairs_final.parquet` | `43c1a2d399cecf02e9f3e87a8f8404613ea34385d32de9ec844eaa086fe1cd38` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores.parquet` | `7966fca4099179f83759e140b15e21e8f9dc0c87393101cc614587bafe4e5f85` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0004.parquet` | `24e8cca77930d64bfe512f69c2820a65f929a6a57dad17441c531ecec2edd358` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0005.parquet` | `5b5a7b1b6648ce0a336658a2c299011962eca59247bf8e215ccf5d74a78649fe` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0006.parquet` | `401a3d19160c99ec18b06bdde3a3099e03ce4d5955388d71c6bbb23efbbc7d10` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0007.parquet` | `c2af890c158a77bb59f67c74eb6c4cbbf62d00f5477013d07ef6f34b7c376567` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0008.parquet` | `eb0c99bd782ed36b157679508e8a7612371a9bf97fbd3f3f0a7ca856d66af6b1` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0009.parquet` | `68e2f687afb6ad69d38879e8d49392c744c949d5213ae1c46771ff568169c1be` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/d300/final-4b/gpu/qwen3-reranker-4b/scores-extra.parquet` | `3b1a60671be5b55e8faaed0cb442d687736742a88c1eb0be94d48e807e7c19d9` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s37/kaggle-4b/s37-b001-003/gpu/qwen3-reranker-4b/scores-extra.parquet` | `9ba2ca4fb997e568c32f8b07ae40ade7d374814e29791687eddf4a8ce4fc08d7` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s37/kaggle-4b/s37-b004-006/gpu/qwen3-reranker-4b/scores-extra.parquet` | `0b54d6c4ead67e1addb6cefd90c732166f985f918a55c3c266bb1984dcbbd8fc` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s37/kaggle-4b/s37-b007/gpu/qwen3-reranker-4b/scores-extra.parquet` | `c6804adf4292ee94b5e1728110d3445835c23dc208ac68f04fca4508aeecf732` |
| jina | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/s37/jina-final2-scores.parquet` | `45bfb54d3c6fce7a72dde0a690d56594396cf6103af4b745c8080c90bc886ba0` |

Outputs of `apply` (`out/runs/s36-fuse4/s37/120A-p4b-v3.apply.json`):

| file | SHA256 |
|---|---|
| `carriers-120A-p4b-v3.parquet` | `e9f543c952a8248e469ca812b536a2625cbde1e2e9ed2265743a5e796ae7360f` |
| `docsel-120A-p4b-v3.parquet` | `69540d841b9b4a6ed04a3ea867c690e42bc68daaf3a50326c8a06a4c3fb2d463` |
| `keep-120A-p4b-v3.json` | `53b0e7a6ae308dda8b8a4a4ba064e70f321a58ddc57a68f9bf7f2b20502913d7` |
| `pcal-120A-p4b-v3.parquet` | `61bbf0366b7a1e35aee4f3d0eb744f4f59408f14f42ea333f5964407129eac18` |

Result: `120A-p4b-v3-full+w768-t1536.zip` in `D:/GitHub/r2ai-stage3/out/runs/s38-chunk-metric-v2/compose`, 99,417,831 B, SHA256 `34d235f6f37704d11b803e83e9eddeabbf4c6100a38497c2728b56cb614937e2`, T 1536, windows 34,709, chunks 209,665, docs 192,742, validator ok True.

## 3. 120A-free-v2 (general): inputs named by `configs/s36-handoff-120A-free-v2.json`

| kind | path | SHA256 |
|---|---|---|
| pairs_new | `D:/GitHub/r2ai-stage3/out/runs/s37/pairs_new_final_v2.parquet` | `6d2c08e56b76a1b424e00512bab04edfe337959ce550f150916d63d03975d4ae` |
| pairs_new | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/d300/pairs_final.parquet` | `43c1a2d399cecf02e9f3e87a8f8404613ea34385d32de9ec844eaa086fe1cd38` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores.parquet` | `7966fca4099179f83759e140b15e21e8f9dc0c87393101cc614587bafe4e5f85` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0004.parquet` | `24e8cca77930d64bfe512f69c2820a65f929a6a57dad17441c531ecec2edd358` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0005.parquet` | `5b5a7b1b6648ce0a336658a2c299011962eca59247bf8e215ccf5d74a78649fe` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0006.parquet` | `401a3d19160c99ec18b06bdde3a3099e03ce4d5955388d71c6bbb23efbbc7d10` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0007.parquet` | `c2af890c158a77bb59f67c74eb6c4cbbf62d00f5477013d07ef6f34b7c376567` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0008.parquet` | `eb0c99bd782ed36b157679508e8a7612371a9bf97fbd3f3f0a7ca856d66af6b1` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-b0009.parquet` | `68e2f687afb6ad69d38879e8d49392c744c949d5213ae1c46771ff568169c1be` |
| cmax | `D:/GitHub/r2ai-stage3/out/runs/s37/cmax_scores-d300.parquet` | `832aec532893b24d570972d5c776eae23e03ea19dff90e348744b90cb80ad8f9` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/d300/final-4b/gpu/qwen3-reranker-4b/scores-extra.parquet` | `3b1a60671be5b55e8faaed0cb442d687736742a88c1eb0be94d48e807e7c19d9` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s37/kaggle-4b/s37-b001-003/gpu/qwen3-reranker-4b/scores-extra.parquet` | `9ba2ca4fb997e568c32f8b07ae40ade7d374814e29791687eddf4a8ce4fc08d7` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s37/kaggle-4b/s37-b004-006/gpu/qwen3-reranker-4b/scores-extra.parquet` | `0b54d6c4ead67e1addb6cefd90c732166f985f918a55c3c266bb1984dcbbd8fc` |
| scores_4b | `D:/GitHub/r2ai-stage3/out/runs/s37/kaggle-4b/s37-b007/gpu/qwen3-reranker-4b/scores-extra.parquet` | `c6804adf4292ee94b5e1728110d3445835c23dc208ac68f04fca4508aeecf732` |
| jina | `D:/GitHub/r2ai-stage3/out/runs/s36-fuse4/s37/jina-final2-scores.parquet` | `45bfb54d3c6fce7a72dde0a690d56594396cf6103af4b745c8080c90bc886ba0` |

Outputs of `apply` (`out/runs/s36-fuse4/s37/120A-free-v2.apply.json`):

| file | SHA256 |
|---|---|
| `carriers-120A-free-v2.parquet` | `47b646de6d92b6ba82db4171c64ee7136707c68134510ed7de79dfff0cd38d86` |
| `docsel-120A-free-v2.parquet` | `c547b9d9b9e22911ff40db17219f4a575c1132bb787f6c827b76b97236c58707` |
| `keep-120A-free-v2.json` | `eb8989e07e3702a8f6933117f2321779c094a71d03b432829670a16ceab90cce` |
| `missing-120A-free-v2-cmax.parquet` | `b47796f245c84d3de6c7c9782cfd7440359ef85f1cf7c13c8c61b96dfd4975a8` |
| `missing-120A-free-v2-jina.parquet` | `4e6f269f548994f34f745e98411909f85e992e78dbc0ff800905cff72d931612` |
| `pcal-120A-free-v2.parquet` | `2806cfaac6a5e4724fa622d26219c67db3af1196c7503b7e9394e2b1d23c35a1` |

Result: `120A-free-v2-full+w768-t1536.zip` in `D:/GitHub/r2ai-stage3/out/runs/s38-chunk-metric-v2/compose`, 100,285,047 B, SHA256 `fba176c65f98b733ce879e932ab245905d9899e96eea867583af5ae85caada3f`, T 1536, windows 34,186, chunks 209,176, docs 220,599, validator ok True.

## 4. Data `compose` reads that the manifest does not pin file by file

The release build rebuilds both ZIPs byte for byte from these (that is their check); keep them as they are.

| path (under the repository root unless absolute) | size | read by |
|---|---:|---|
| `data/canonical-20261018/` (`docs.parquet`, chunks, `vectors/dense.npy`) | 17 GB | compose (chunk text, hosts, best-chunk choice), `apply` (docs.parquet is pinned) |
| `out/runs/s32-chunk-judge/` (`pool/chunk_pairs.parquet`, `pool/supplement_vectors.npz`, `gpu/qwen3-reranker-0.6b/scores-all.parquet`) | 771 MB | compose (best t256 chunk) |
| `out/runs/s31-strong-judge/pool/` | 296 MB | S38 score code, `apply` base data (pinned entries) |
| `data/s37_120ask/` | 1.0 GB | registry: 120ask segments (bodies of the new docs) |
| `out/runs/s37/` | 2.9 GB | registries, C-max, Jina, `pool.npz`, Kaggle 4B scores (pinned entries) |
| `out/runs/s27-pharmacity/handoff/cc-s27-pharmacity` | 12 MB | validator (Pharmacity handoff) |
| `out/runs/s35-part-jina/build-j3/J3-thrC.zip` | 97 MB | S38 sources (J3 bundle) |
| `out/runs/s29-qwen-judge/` | 623 MB | query vectors, candidate pool |
| `out/runs/s36-fuse4/` | 1.5 GB | S36 extra scores, cal, d300 packages, 4B score files |
| `out/runs/s38-chunk-metric-v2/compose/` | 16 GB | the submitted ZIPs, their `compose.json`, the three ablation ZIPs |
| `out/runs/release/` | small + ZIPs | release rebuilds (`120A-p4b-v3`, `120A-free-v2`), reports |

## 5. Models (needed only to re-score, not to build)

| model | revision | used for |
|---|---|---|
| BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` | corpus / query / en vectors (dense CLS, L2) |
| Qwen/Qwen3-Reranker-4B | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | P4B (Kaggle, scorer SHA `688948a7…a7186`, en `ffc54f92…21e1`) |
| Qwen/Qwen3-Reranker-0.6B | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | C-max (S32 recipe) |
| jinaai Jina Reranker v3.5 | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | Jina listwise (layout b10s0) |
