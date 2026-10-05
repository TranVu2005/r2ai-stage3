# PROGRESS – R2AI Stage 3

- Cập nhật: 2026-10-05 10:40 (UTC+7)
- Model thực hiện: Claude Opus 5.5
- Commit HEAD đã kiểm trước cập nhật này: `6c666a6`, nhánh `main` (hash commit chứa tài liệu này được báo ở kết quả)
- Public: 31/10/2026 · private: 04/11/2026 (tối đa 5 lượt) · kết quả: 11/11/2026; mốc do người dùng cung cấp
- Quy ước: ✅ có bằng chứng · 🟡 làm một phần · ❌ chưa làm (đã tìm, không thấy) · ❓ không chứng minh được

> Ghi chú: lúc kiểm tra **không có file `PROGRESS.md` template** trong repo (`ls`, `find`) nên cấu trúc mục 1–9 và danh sách M0–M12 được dựng lại từ `R2AI_STAGE3_CONTEXT.md` §4, §6 và từ nhiệm vụ. Hãy đối chiếu nếu bạn có template gốc.
> Snapshot nguồn OLD ngày 03/10: repo git nằm ở `D:\GitHub` (gốc chung nhiều dự án). `git ls-files` trong `r2ai-stage3` chỉ có **4 file đã commit** (`eval/README.md`, `eval/scorer.py`, `eval/validate.py`, `tests/test_scorer.py`). Mọi thứ khác (crawler, extract, index, submission…) là **untracked**, nên không có commit hash để dẫn; bằng chứng là path và mtime.


> **PHA B / repo hiện tại:** `D:/GitHub/r2ai-stage3` (chuyển từ `D:/R2AI/r2ai-stage3` ngày 04/10 14:00; OLD đổi tên thành `D:/GitHub/r2ai-stage3-old`), history mới trên main. Trong các mục bên dưới, "OLD" là `D:/GitHub/r2ai-stage3-old`. Các số crawl/extract/index/LB phía dưới là snapshot legacy OLD ngày 03/10, không phải kết quả chạy lớn ở NEW. Mapping hiện tại: `eval`→`src/r2ai/eval`, `index`→`src/r2ai/index`, `retrieve`→`src/r2ai/retrieve`, `submission`→`src/r2ai/submit`, `vicrawl`→`src/vicrawl`, `config`→`configs`; CLI xem README. Đọc query/docs/chunks/index/K100 cũ ở OLD; ghi pipeline mới vào NEW. Status mặc định đọc NEW/state, đối chiếu OLD phải dùng `--state-dir` rõ ràng.
> Gate đã đo: core **36 pass**, full **272 pass / 2 deselected**, slow **2 pass / 10 deselected**; OLD full 234 pass và slow 2 pass. Replay JSON sub06 **181.960.105 byte**, SHA256 `07e0fd59f8cc4b8abbf129dc1536305bdbef17ce510b6fe39fe50fe62133acc3`, byte/query/field diff=0; validator **1.200 query / 22.800 chunk / 0 lỗi**. Không rerun model/GPU hay nộp LB. Python 3.12.13, torch 2.11.0+cu128, 102 package freeze khớp OLD. Chi tiết run ở `out/runs/restructure/` và `out/runs/reproduce-sub06/` (ignored).
> Copy raw/crawl.db: **Đã copy** 18,031 shard / 3,679,529,131 byte (3.427 GiB) và crawl.db 376,438,784 byte sang NEW; hash/inventory nguồn–đích khớp, SQL 653.970 URL / 624.596 ok / 48 domain khớp. Writer dừng, WAL0; không overwrite hay move/xóa OLD. RAW_DIR=RAW_WRITE_DIR=NEW/data/raw_vi; status mặc định NEW đã kiểm. Extract NEW đã hoàn tất (xem bên dưới); không chạy crawl thật. OLD raw hash sau copy, source161 và state6 vẫn không đổi. Manifest `out/runs/restructure/data-copy.json`.

> Pipeline sau PHA B (trạng thái 05/10): Extract NEW **18.031/18.031 shard**, **632.208 doc** (ok 623.852, thin 8.356). Chunk NEW chạy lại theo lô sau lần 1 hết RAM: **623.805 doc → 3.909.586 chunk t256** (answer/body 3.281.895, verbatim 3.909.586/3.909.586, `data/chunks/chunk_report.json`). Index NEW dense+sparse 3.909.586 chunk (embed 37.953 s), K100 exact 1.200 query 8.481 s. Submission cấu hình sub06 trên index NEW đã **nộp 05/10: Final 0,1748** (sub06 0,0626; ×2,79), Doc R 0,2897 (sub06 0,088; ×3,29). Chi tiết `submissions/LOG.md`, mục 5. Lượt chunk lần 1 hết RAM (`ArrowMemoryError`, pagefile 49 GB) giữ ở Changelog 13:51.

## 1. Crawl (vi)

Nguồn: `state/crawl.db` (SQLite, mở `mode=ro`), schema: `urls(url_norm PK, url, doc_ids, domain, rank, status, http_status, reason, attempts, fetched_at, next_try_at, shard, final_url)`, `domains(...)`, `meta`.

| Chỉ số | Giá trị |
|---|---|
| Tổng URL trong DB (đã gộp http/https) | 653.970 (48 domain) |
| `ok` | 624.596 (95,5%) |
| `pending` | 7.176 (toàn bộ `youmed.vn`) |
| `blocked_4xx` | 11.767 (`vov.vn` 9.784 + `baolangson.vn` 1.981 + 2 lẻ; `reason=robots_disallowed`) |
| `thin` | 7.713 · `soft404_or_home` 2.305 · `network_error` 395 · `bot_challenge` 17 · `dead_origin` 1 |
| Trạng thái domain | 47 `done`, 1 `halted` (`youmed.vn`: "error_rate: 12 consecutive failures") |
| Tiến trình crawler | **Không chạy** lúc kiểm tra: `logs/crawl.log` kết thúc `22:31:05 runner finished: done`; không có process `crawl.py` (Get-CimInstance) |
| Lần fetch cuối | 2026-10-03 22:31:05 (`max(fetched_at)`) |
| Raw | 18.031 shard `data/raw_vi/<domain>/*.jsonl.zst`, tổng 3,5 GB (`du -sh data/raw_vi`) |
| IP egress | `out/egress.json`: 118.71.35.77, FPT Telecom, Hà Nội, VN (đo 2026-10-02) |

Top domain theo `ok`: suckhoecongdongonline.vn 154.501 · suckhoedoisong.vn 79.284 · thanhnien.vn 71.029 · laodong.vn 31.282 · phunusuckhoe.giadinhonline.vn 27.577 · vinmec.com 26.127 · medlatec.vn 24.709 · giadinhonline.vn 16.727.
Đáng chú ý: `thaythuocvietnam.vn` chỉ 507/2.927 `ok` (2.230 `soft404_or_home`, 190 `thin`); `suckhoedoisong.vn` 6.480 `thin`.
Chưa có: zh (không có `data/raw_zh*`), `nhathuoclongchau.com.vn`, `zysjonline.com`, `bingli.iiyi.com`.

## 2. Milestones

| M | Nội dung | Trạng thái | Bằng chứng |
|---|---|---|---|
| M0 | Profile corpus/query, probe v2 từ IP VN, gold check 30 query | ✅ | `out/profile_report_v2.md`, `out/gold_check_report.md`, `out/gold_check_results.csv`, `out/verification_v2.json` |
| M1 | Crawler production vi (code + test) | 🟡 | `crawl.py`, `fetcher.py`, `vicrawl/` (17 module), `tests/test_vicrawl_*.py` (13 file). Tôi **chỉ chạy** `test_scorer`, `test_make_submission`, `test_chunk`: 36 passed (37s). Chưa chạy bộ test `vicrawl` |
| M2 | Crawl vi hoàn tất | 🟡 | 624.596/653.970 `ok`; còn `youmed.vn` (halted) và 2 domain robots chặn. Xem mục 1 |
| M3 | ≥100k doc `ok` để dựng baseline | ✅ | `out/MILESTONE_1.md`: 101.554 `ok` lúc 2026-10-02 20:20:44 |
| M4 | QA trích xuất domain vi lớn | 🟡 | 23 báo cáo `out/qa_extract/*.md` (300 doc/domain). Đây là mẫu QA, không phải duyệt toàn bộ. Không thấy báo cáo cho `bachmai`, `tamanhhospital`, `tuoitre` v.v. (domain dưới ngưỡng QA gate) |
| M5 | Extract toàn bộ doc vi | ✅ | NEW: 18.031/18.031 shard, 632.208 doc; checkpoint/raw/parquet khớp. Snapshot OLD xem mục 4; kết quả NEW ở phần đầu báo cáo. |
| M6 | Scorer local theo metric | ✅ | `eval/scorer.py`, `tests/test_scorer.py`, commit `ae6b90d`. Chưa đối chiếu với điểm LB nào (không có điểm LB) |
| M7 | Chunk + index + truy hồi baseline (BGE-M3 hybrid + reranker) | ✅ vi | NEW: `data/index/t256/` dense.npy + sparse (không FAISS), **3.909.586 chunk / 623.805 doc**; K100 exact `out/runs/vi-k100/vi_k100.parquet` (1.200 query, 119.969 dòng). Legacy OLD 818.080 chunk / 123.874 doc. Chưa có zh. Xem mục 5 |
| M8 | Nộp thử lên leaderboard | ✅ | 03/10: 7 lượt (`Downloads/SCOREBOARD.md`), tốt nhất sub06 Final 0,0626. 05/10: `sub_new_vi_kd100_kc19_full` (index NEW) **Final 0,1748** (người dùng cung cấp). Ghi ở `submissions/LOG.md` mục Leaderboard |
| M9 | Nộp thử A/B (chunk dài/ngắn; 1 id/cả nhóm; vi/zh) | 🟡 | Xong: chunk dài vs ngắn (c2 0,0158 → full 0,0273 ở K=5), quét K 5/20/50/100, cửa sổ 1024/2048, index cũ vs NEW (0,0626 → 0,1748). Đã sinh chưa nộp: V2 thêm 1 chunk cho doc hạng 20..45 (`out/runs/ab-2026-10-05/`). Cả nhóm id: mọi file đều đã trả cả nhóm (mặc định); "1 id" chưa thử. K doc 150/200: cần chạy lại truy hồi. vi vs có zh: chưa |
| M10 | Mẫu 2% zh, tính yield, thứ tự crawl zh | ❌ | Không có `data/raw_zh*`, `docs_zh*`, `config/domains.yaml` chỉ có domain vi |
| M11 | longchau qua CC columnar index; kiểm robots zysjonline | ❌ | `out/archive_*` có số đo độ phủ (bước 1), nhưng không có dữ liệu longchau đã tải. Không thấy kết luận robots zysjonline trong `out/` |
| M12 | Crawl dữ liệu train Vinmec/MEDLATEC ngoài corpus (lọc trang trùng query test) | ❌ | Không có thư mục/dữ liệu tương ứng. `data/dev/pseudo_vi_v2.parquet` (462 dòng) sinh từ chính corpus đã crawl, không phải dữ liệu ngoài |

## 3. Crawl theo domain × status

Đếm bằng `select domain,status,count(*) from urls group by 1,2` (read-only). Chỉ liệt kê các domain có status khác `ok` đáng kể; toàn bộ 48 domain còn lại gần 100% `ok`.

| domain | tổng | ok | khác |
|---|---|---|---|
| suckhoecongdongonline.vn | 154.503 | 154.501 | network_error 1, soft404 1 |
| suckhoedoisong.vn | 85.823 | 79.284 | thin 6.480, network_error 58 |
| thanhnien.vn | 72.111 | 71.029 | thin 745, network_error 336, blocked_4xx 1 |
| vov.vn | 9.784 | 0 | blocked_4xx 9.784 (robots_disallowed) |
| baolangson.vn | 1.981 | 0 | blocked_4xx 1.981 (robots_disallowed) |
| youmed.vn | 7.189 | 0 | pending 7.176, bot_challenge 13 (halted) |
| thaythuocvietnam.vn | 2.927 | 507 | soft404_or_home 2.230, thin 190 |
| bachmai.gov.vn | 2.783 | 2.514 | thin 269 |
| pharmacity.vn | 69 | 65 | bot_challenge 4 |

Kích thước raw lớn nhất: suckhoedoisong.vn 916 MB (5.884 shard) · medlatec.vn 447 MB · thanhnien.vn 407 MB · suckhoecongdongonline.vn 341 MB · hellobacsi.com 206 MB · vinmec.com 155 MB.

## 4. Extract

**Kết quả NEW tại 2026-10-04 03:22 UTC+7:** Extract NEW hoàn tất: **18.031/18.031 shard**, **632.208 doc**, status {"ok": 623852, "thin": 8356}; elapsed CLI **5438.0 s**, workers2. Checkpoint khớp tập raw NEW, số hàng parquet khớp SQLite; không còn shard pending. Doc ok: Vinmec **26.124**, MEDLATEC **24.709**. 6 hash/size state OLD giữ nguyên. Chunk → index → K100 → submission đang chạy theo chuỗi; chưa có điểm LB mới. Bằng chứng `out/runs/coverage-rebuild/extract.json`, `old-state-after-extract.json`.

Các số bên dưới lưu snapshot OLD ngày 03/10.

- Output: `data/docs_vi/<domain>__<timestamp>-<pid>-<seq>-<hash>.parquet` (3.083 file, phẳng, không có thư mục con theo domain; 474 MB). Cột: `doc_ids, url, final_url, domain, title, description, question, answer, body, paragraphs, lang, n_tokens_bge_m3, text_sha1, fetched_at, url_norm, status, extractor, headings`.
- Tổng 125.499 doc (metadata parquet), `status`: ok 123.881, thin 1.618. `body` không rỗng 99,8%.
- **Phủ rất lệch**: các domain có >5k URL (có QA gate) chỉ được extract ~300 doc. Ví dụ: suckhoedoisong.vn 324/79.284 `ok`, thanhnien.vn 312/71.029, vinmec.com 301/26.127, medlatec.vn 301/24.709, hellobacsi.com 301/11.743, khoahocphothong.vn 301/8.143. Trong khi vietnamnet 13.834 (≈100%), giadinhonline 12.209, tiemchunglongchau 11.353, laodong 10.419/31.282, suckhoecongdongonline 10.588/154.501.
- `answer`/`question` không rỗng: vinmec.com 18,9% (301 doc), hellobacsi.com 1,7%, thanhnien.vn 1,3%, còn lại 0%. Trong chunk index chỉ có 238 chunk `answer` và 79 `question` (`data/chunks/chunk_report.json`).
- `title` rỗng nhiều: vnexpress.net (title có 42,3%), baothanhhoa.vn (0,4%), benhvienvietduc.org (0%), bachmai.gov.vn (body 90,2%).
- Extractor dùng: `generic` 51.480 (41%), còn lại là extractor riêng (vietnamnet, giadinhonline*, tiemchunglongchau, suckhoecongdongonline, laodong…).
- Báo cáo QA: có, `out/qa_extract/*.md` (23 file, mỗi file 300 doc mẫu, gồm tỷ lệ rơi về trafilatura và đoạn đầu/cuối).
- Process `extract.py` không chạy lúc kiểm tra; `state/extract.db` cập nhật cuối 2026-10-03 02:43, trong khi crawl chạy tới 22:31.

## 5. Eval / retrieval / submission

| Hạng mục | Hiện trạng | Bằng chứng |
|---|---|---|
| Scorer | ✅ `python -m eval.scorer`, test pass | `eval/scorer.py`, `tests/test_scorer.py` (nằm trong 36 passed) |
| Validator | ✅ | `eval/validate.py`, `scripts/validate_submission.py`; `logs/sub01.log`: validator exit code 0 |
| Chunking | ✅ legacy t128/t256/t400 · ✅ NEW t256 | Legacy: `data/chunks/chunks_t256.parquet` 818.080 chunk. NEW (chạy lại theo lô `--batch-docs`): 623.805 doc → 3.909.586 chunk (answer/body 3.281.895; token answer/body p50 219, p95 254, max 384; verbatim 3.909.586/3.909.586), `data/chunks/chunk_report.json`. `tests/test_chunk.py` |
| Index | 🟡 legacy: FAISS `IndexFlatIP` dense + sparse, t256, `built_at` 2026-10-03 06:11:40 · ✅ NEW: 3.909.586 chunk / 623.805 doc, dense+sparse, không FAISS (`--no-ann`), `built_at` 2026-10-05 02:15:23 | Legacy `OLD/data/index/t256/meta.json` (123.874 doc, 3,12 GB float32, nnz 73.316.966 = 89,6/chunk). NEW `data/index/t256/meta.json`: embed 37.953 s (10,54 h, batch 64, 391 shard), VRAM đỉnh 1.634 MiB, sparse nnz 339.331.682 (86,8/chunk), dense.npy 8.006.832.256 byte. `index.build` có `--no-ann` và `assemble` (chỉ ghép shard, không nạp model/text) |
| Truy hồi | 🟡 hybrid w=0.7, `max`, rerank `bge-reranker-v2-m3` fp16, seed 42. Thêm `--candidates exact` (block-scan dense.npy + CSR mmap, không FAISS/CSC); gate legacy 1.199/1.200 (99,917%) K100 trước rerank giống hệt | `src/r2ai/retrieve/exact.py`, `exact_gate.py`, `out/runs/exact-gate/*.json`; `retrieve/run.py`, `data/runs/vi_k100.meta.json` (1.200 query, 13.083 s = 10,9 s/query), `vi_k100.parquet` 119.934 dòng (≥74, p50 100 doc/query) |
| Dev set | ⚠ pseudo-dev, **bão hòa và lệch xa LB** (dev chunk F2 0,358 so với LB Final 0,0158 cùng cấu hình c2) | `data/dev/pseudo_vi_v2.parquet` 462 dòng (400 A tiêu đề-làm-query + B hỏi đáp). `out/retrieval/ablation_t256.md`: A R@5 0,9975, B R@1 1,0; chunk-level B F2 0,358. Trang nguồn nằm trong index nên số này **không dự báo điểm LB** (chính `out/submissions/LOG.md` nêu điều này) |
| File nộp | Legacy sub01–sub10 (`OLD/out/submissions/`) · ✅ NEW `sub_new_vi_kd100_kc19_full` **đã nộp 05/10** · 🟡 A/B V2 đã sinh, chưa nộp | NEW: `out/runs/new-full/sub_new_vi_kd100_kc19_full.zip` 39.237.992 byte, SHA256 `1d171722…567a`; JSON 156.211.418 byte, SHA256 `50ac7367…37ab`; validator 1.200 / 22.800 / 0 lỗi. V2: `out/runs/ab-2026-10-05/v2/sub_ab_v2_kd100_kc19_full_xchunk.zip` 47.792.584 byte, SHA256 `af186414…290c`, 45 chunk/query, validator 0 lỗi. `scripts/make_submission.py`, `tests/test_make_submission*.py` |
| Điểm leaderboard | ✅ 8 lượt (người dùng cung cấp) | 05/10 `sub_new_vi_kd100_kc19_full`: **Final 0,1748** · Doc F2 0,2214 · Doc P 0,174 · Doc R 0,2897 · Chunk F2 0,1283 · Chunk P 0,3025 · Chunk R 0,1249 (Final = trung bình hai F2: (0,2214+0,1283)/2 = 0,17485, khớp). 03/10 (`Downloads/SCOREBOARD.md`) Final: sub02 0,0158 · sub03 0,0273 · sub04 0,0551 · sub05 0,0623 · sub06 0,0626 · sub07 0,0545 · sub08 0,0507 |

JSON NEW 156.211.418 byte < sub06 181.960.105 byte dù corpus lớn hơn (đo trên hai file JSON): cùng 22.800 chunk và gần cùng số id (120.135 vs 120.662); khác ở độ dài chunk_text = độ dài văn bản của doc top 19 (chế độ full). Tổng ký tự chunk_text 116.158.508 vs 135.525.792 (−14,3%), UTF-8 153.019.613 vs 178.513.823 byte (−25.494.210, ≈ toàn bộ chênh JSON 25.748.687). Ký tự/chunk p5 1.030 vs 1.955, p50 4.917 vs 5.369, p95 9.119 vs 10.149, max 57.173 vs 162.200; token BGE-M3 mean 1.340 vs 1.570. Nguồn: `answer` 993 chunk vs 29 ở sub06 (trường answer ngắn hơn body). Tỉ lệ nén ZIP NEW 3,98 (sub06 3,95).

Lưu ý: sub01/sub02 dùng K=5 doc, C=2 chunk; sub03–sub10 gồm các biến thể K/chunk dài ("full" tới 45.518 token), cửa sổ 1024/2048 token và ngân sách zip 45,7 MiB. Giới hạn dung lượng upload của leaderboard là giả định trong `LOG.md`, chưa xác minh.

## 6. Lệch so với `R2AI_STAGE3_CONTEXT.md`

Cập nhật 2026-10-04 03:27: chunk NEW đã nạp 623.805 doc đủ điều kiện (118 s), target256, min-body-tokens50; đang tokenise, chưa có kết quả chunk/index/retrieval.

Cập nhật 2026-10-04 13:51: lần chunk này thất bại do hết RAM; `index.chunk` đã chuyển sang xử lý theo lô (đã chạy lại xong trước 05/10, xem đầu báo cáo).

Cập nhật 2026-10-05 10:40: người dùng nộp `sub_new_vi_kd100_kc19_full` → Final 0,1748. Khảo sát A/B không chạy lại model: builder đã trả cả nhóm id (V1 không cần); `vi_k100.candidates.parquet` chỉ lưu số đếm nên doc hạng 101–200 cần chạy lại truy hồi (V3 không sinh); điểm reranker cấp chunk chỉ có cho top 50 doc. Thêm `--extra-chunk-docs` / `--extra-zip-budget-bytes` (mặc định tắt; builder mặc định tái tạo JSON đã nộp giống byte), sinh V2 N=45.

Cập nhật 2026-10-05 09:05: chạy thật NEW xong, chưa nộp LB. Embed 391 shard 37.953 s. Lần K100 đầu dừng ở chạy thử 50 query: `ArrowInvalid: offset overflow` (`ChunkedArray.take` trên >2 GiB text) và private 12,44 GiB (cột text giữ trong Arrow pool ~7,2 GiB); sửa `a76052b` (take theo chunk, text mmap `data/index/t256/text.arrow`). K100 exact 1.200 query: **8.481 s (6,84 s/query, đo thực)**, peak RSS 7,391 GiB, private 7,046 GiB, VRAM 1.452 MiB; sinh ứng viên dense 68,5 s / sparse 156,8 s / union+rescore 18,8 s, |V_q| 2.641. Doc phân biệt trước rerank: min 79, p5 169, p50 269; 2 query < 100 (qid 99: 79, qid 804: 90) → K100 có 1.198 query đủ 100 doc. Chạy thử 50 query đầu (không phải mẫu seed 42): 6,72 s/query → ngoại suy 2,24 h, thực tế 2,36 h. Submission sub06 (kd100, kc19, full, dedupe doc): ZIP 37,42 MiB, k_chunk không bị `--max-zip-mib` hạ; validator 0 lỗi. Log `out/runs/new-full/` (`status.json`, `pipeline.log`).

Cập nhật 2026-10-04 15:40: retrieval có `--candidates exact` (mặc định vẫn `faiss`). Gate trên index legacy 818.080 chunk, 1.200 query, không rerank: K100 doc trước rerank giống hệt **1.199/1.200 (99,917%)**, cùng thứ tự 1.199; tập ứng viên giống hệt 1.191; điểm dense/sparse trên tập giống hệt lệch 0,0. 9 ca lệch tập ứng viên đều là tie trên chunk trùng vector ở biên top-200 sparse (`argpartition` cũ chọn tuỳ ý); 1 ca (qid 694) đổi 3 doc trùng nội dung trong K100. Exact: dense 12,9 s, sparse 33,2 s, union+rescore 2,7 s, peak RSS 1,256 GiB, private 1,544 GiB. Cũ (FAISS mmap + CSC): 198,0 s, peak RSS 3,743 GiB. Số doc phân biệt trước rerank: min 74, p5 136, p50 227, 5 query < 100. Mix là min-max theo tập ứng viên nên không có phương án "hybrid một lượt quét". Embed NEW chưa chạy xong (process không còn); Bước chạy thật 3,9M chưa làm.

Cập nhật 2026-10-04 14:05: repo chuyển về `D:/GitHub/r2ai-stage3`; OLD thành `D:/GitHub/r2ai-stage3-old` (chuyển từng mục con vì thư mục gốc bị process khác giữ handle; không xóa gì). `.env` local trỏ legacy sang `-old`, data/state/out NEW nằm trong repo. `extract.db` NEW: 18.031 `shards.path` đổi prefix `D:\R2AI\r2ai-stage3\data\raw_vi\` → `D:\GitHub\r2ai-stage3\data\raw_vi\` (backup `out/runs/extract.db.before-move`), 0 thiếu file, `ExtractPipeline.pending()` = 0. `.venv` cài lại editable `--no-deps`, freeze 102 package không đổi. Hash crawl.db/extract.db OLD không đổi.

Extract NEW hoàn tất: **18.031/18.031 shard**, **632.208 doc**, status {"ok": 623852, "thin": 8356}; elapsed CLI **5438.0 s**, workers2. Checkpoint khớp tập raw NEW, số hàng parquet khớp SQLite; không còn shard pending. Doc ok: Vinmec **26.124**, MEDLATEC **24.709**. 6 hash/size state OLD giữ nguyên. Chunk lần 1 (target 256) **thất bại** 12:29 sau 32.858 s: `ArrowMemoryError` (realloc 2.818.572.288 byte) khi dựng bảng chunk; RAM 15,7 GiB, pagefile C: tự tăng lên 49 GB (đỉnh 31,6 GB), tokenize 11.455.367 paragraph mất 22.024 s do swap. `data/chunks/docs.parquet` của lần này là output dở, không dùng. `index.chunk` đã sửa sang xử lý theo lô (Changelog 13:51); chưa chạy lại trên corpus đầy đủ, RAM/thời gian mới chưa đo. Index → K100 → submission chờ chunk; chưa có điểm LB mới. Bằng chứng `out/runs/coverage-rebuild/extract.json`, `old-state-after-extract.json`.

Extract đang chạy: **1,039/18.031 shard**, **43,674 doc** trong checkpoint NEW tại 2026-10-04 01:57 UTC+7; raw/data/state đều NEW, workers2. PID launcher 22592; log `out/runs/coverage-rebuild/extract.log`. Chưa có kết quả chunk/index/K100/submission mới. Validator JSON sub06 thực đã pass: 1.200 query / 22.800 chunk / 0 lỗi (không áp budget ZIP vào file JSON181MB).

Validator JSON trực tiếp đã sửa cho Python3.12 bằng Path.open(newline=""). Một test parametrized cho cả hai CLI: LF hợp lệ/CRLF bị từ chối, 4 ca RED→GREEN; full suite **272 passed / 2 deselected / 152,22s**. Guard được kiểm chỉ gọi ở preflight CLI, không trong vòng lặp shard; giữ quét recursive để bảo vệ junction/symlink con. Không thêm gọi assert_writable mỗi shard.

Replay JSON giữ nguyên SHA256 nhưng ZIP NEW **46.083.494 byte**, OLD **46.047.946 byte**, chênh **35.548 byte** do bước nén/metadata (nguyên nhân build zlib chưa xác minh). Cả hai dưới 45,7 MiB = 47.919.923 byte. Khi ZIP sát ngưỡng, --max-zip-mib có thể chọn k_chunk khác OLD; luôn đo ZIP thực trên môi trường nộp hiện tại.

Chuyển raw/state sau gate repo: **Đã copy** 18,031 shard / 3,679,529,131 byte (3.427 GiB) và crawl.db 376,438,784 byte sang NEW; hash/inventory nguồn–đích khớp, SQL 653.970 URL / 624.596 ok / 48 domain khớp. Writer dừng, WAL0; không overwrite hay move/xóa OLD. RAW_DIR=RAW_WRITE_DIR=NEW/data/raw_vi; status mặc định NEW đã kiểm. Extract NEW đã hoàn tất (xem bên dưới); không chạy crawl thật. OLD raw hash sau copy, source161 và state6 vẫn không đổi. Manifest `out/runs/restructure/data-copy.json`.

Review độc lập toàn bộ repo: sửa 3 nhóm Important (profile/dup import, guard nested redirects/sidecars, docs parquet rỗng); 11 test RED→GREEN. Suite sau sửa: **268 pass**, slow **2 pass**, replay sub06 SHA/diff=0 trong **78.8s**. Minor kế thừa OLD về validator JSON trên Python3.12 đã sửa trong lượt tiếp theo (xem cập nhật trên).

PHA B đã sửa CONTEXT/metric theo scorer giữ dấu câu, ngưỡng 0,4 và không dedupe ở metric; builder dedupe nguyên để replay sub06. Bảng dưới lưu các lệch của snapshot OLD, không còn là danh sách lỗi chưa sửa. NEW mở pipeline extract → chunk → index → retrieve/K100 → submission; công cụ ghi phụ khóa trong main tới khi có guard. Cài torch dùng `--torch-backend cu128` để giữ wheel CUDA và lấy đúng dependency đã pin từ PyPI. Status read-only, mặc định STATE_DIR; không log vào OLD.

| # | CONTEXT nói | Thực tế | Bằng chứng |
|---|---|---|---|
| 1 | §5: crawler gồm `fetcher.py` + `crawl.py` | Logic chính nằm trong package **`vicrawl/`** (engine, tuner, robots, shards, state…). `crawl.py` 175 dòng, `fetcher.py` 324 dòng, `extract.py` 54 dòng chỉ là vỏ/CLI | `wc -l`, `ls vicrawl` |
| 2 | §5: crawl `httpx HTTP/2` | Log có `RemoteProtocolError: <StreamReset ... error_code:8>` ở `thanhnien.vn` (HTTP/2 xác nhận gián tiếp) | `state/crawl.db` `reason` |
| 3 | §4.4: `slow_server` ≤1 req/s | Đúng định nghĩa, nhưng `suckhoecongdongonline.vn` **không** dùng nhóm này: đo p50 0,74 s nên dùng nhóm `default`, tối đa 4 kết nối (cả domain xong, 154.501 `ok`). Giả định "369 giờ" của CONTEXT đã lỗi thời | `config/domains.yaml` dòng 33–37; `out/amp_check_suckhoecongdongonline.vn.md` |
| 4 | §4.4: `default` 4 req/s, `cdn_large` 8 | Khớp (`rate_cap` 4/8; thêm `max_conns` 2/8). Có thêm override: `dantri.com.vn` và `qdnd.vn` 0,3 req/s, 1 kết nối. `domains` ghi rate đã học, ví dụ thanhnien 8,0, dantri 0,25 | `config/domains.yaml`, bảng `domains` |
| 5 | §4.1: "không gồm longchau" | Có `tiemchunglongchau.com.vn` (13.277 `ok`); chỉ `nhathuoclongchau.com.vn` bị hoãn | `state/crawl.db` |
| 6 | §5: `data/docs_<group>/*.parquet` | Khớp tên, nhưng phẳng (không thư mục theo domain). Thêm `data/docs_vi_rev1/` (3,8 MB, 1 phiên bản extractor cũ?) | `ls data` — mục đích `docs_vi_rev1`: ❓ chưa đọc code |
| 7 | §5: "Index FAISS IVF-PQ hoặc int8" | Dùng `IndexFlatIP` float32 (3,12 GB) trên 818k chunk, đủ cho vi. Chưa cần IVF-PQ ở quy mô hiện tại | `data/index/t256/meta.json` |
| 8 | §5: reranker `bge-reranker-v2-m3` hoặc Qwen3-Reranker | Đã chọn `bge-reranker-v2-m3` fp16. Chưa có LLM rút gọn/dịch query | `out/submissions/LOG.md` |
| 9 | §6: scorer "tái tạo đúng metric … gộp trùng 0,8" | Commit `ae6b90d` ghi lại "align with official spec": **không gộp** chunk dự đoán trùng, giữ dấu câu, chuẩn hoá NFKC. Không khớp mô tả CONTEXT §1 (gộp LCS/hợp ≥0,8 trước khi tính P). Cần đọc spec chính thức để chốt | `eval/README.md`, `git show ae6b90d` |
| 10 | §6: dòng "Review `out/qa_extract/`" chưa tích | Đã có 23 báo cáo, nhưng chưa duyệt toàn bộ và extract chưa chạy tiếp (mục 4) | `out/qa_extract/` |
| 11 | §4.1 "Mốc 1 ≥100k doc ok → dựng baseline" | Đã đạt và baseline đã chạy, nhưng index dừng ở 123.874 doc (xem mục 7) | `out/MILESTONE_1.md`, `data/chunks/chunk_report.json` |
| 12 | Git | `git ls-files` chỉ 4 file; 3 commit gần nhất (`fa8759e`, `87307b1`) thuộc dự án khác (helicopter). `main` không tồn tại cục bộ; chỉ có `master` và `codex/helicopter-training-setup` | `git branch -a`, `git log` |

Hai dòng ❓ từ CONTEXT:
- ❓ **zysjonline.com robots.txt** (CONTEXT §6): không có kết luận trong `out/` (grep `deferred_domains.csv`, `vi_domains.csv`, `source_integrity_v2.json` không ra); xác minh cần truy cập mạng, bị cấm ở lượt kiểm tra này.
- ❓ **Hạn nộp cuối / phân bố gold theo ngôn ngữ / giới hạn dung lượng upload** (CONTEXT "Câu hỏi mở"): không có bằng chứng. Phần "độ dài chunk_text" đã có đáp: scorer nhận chunk tới 45.518 token (theo SCOREBOARD, sub04 full) và chunk dài tăng điểm rõ. Upload: 45,7 MiB lên được, 110 MiB không hoàn tất, giới hạn thật chưa rõ.

## 7. Blockers

1. **Độ phủ vẫn là nút thắt chính nhưng đã giảm**: index NEW 623.805 doc / 4.394.718 URL corpus ≈ **14,2%** (cũ 123.874, 2,8%). Doc R 0,088 → 0,2897 khi tăng độ phủ vi. Ước tính thô từ trung bình macro (không phải số đo): gold khoảng 60–70 doc/query, nên phần còn thiếu phần lớn nằm ngoài index (zh, domain chưa crawl).
2. **Chưa có dữ liệu zh**: CONTEXT ước ~77% corpus là zh; hiện 0 doc zh. Nếu gold có nhiều zh, recall bị chặn trên.
3. **K doc > 100 cần chạy lại truy hồi**: `vi_k100.candidates.parquet` chỉ lưu số đếm, không lưu thứ tự/điểm hybrid ngoài top 100 (doc phân biệt ngoài top 100: min 0, p5 69, p50 169). Điểm reranker chunk chỉ có cho top 50 doc.
4. **Ngân sách upload**: 45,7 MiB lên được, 110 MiB không; giới hạn thật chưa rõ. Chunk cho đủ 100 doc cần ZIP khoảng 62,81 MiB (đo, cách chọn chunk cho hạng 51–100 chỉ để đo size).
5. `youmed.vn` halted (bot_challenge); `vov.vn` và `baolangson.vn` chặn bởi robots (11.765 URL không lấy được, đúng theo quyết định tôn trọng robots).
6. Pseudo-dev không dự báo được LB; chỉ LB đo thật (tối đa 5 lượt private).

## 8. 3 việc tiếp theo (đường găng tới 31/10)

1. **Người dùng nộp V2** `out/runs/ab-2026-10-05/v2/sub_ab_v2_kd100_kc19_full_xchunk.zip` (45,58 MiB, chỉ thêm 1 chunk cho doc hạng 20..45) và ghi điểm vào `submissions/LOG.md`. V1 (cả nhóm id) không cần vì đã là mặc định; V3/V3b (K doc 200/150) và V4 chưa sinh.
2. **Nếu muốn K doc > 100 hoặc chunk cho hạng 51–100**: chạy lại K100 với lưu thứ tự hybrid đầy đủ của ứng viên và điểm chunk cho top 100 (cần GPU, khoảng 2,4 h theo lượt 05/10); cần người dùng duyệt.
3. **Quyết định zh**: lấy mẫu 2% mỗi domain zh, tính yield trên 1.200 query, rồi xếp thứ tự crawl.

## 9. Changelog

| Thời gian (UTC+7) | Ai | Thay đổi |
|---|---|---|
| 2026-10-03 22:50 | Claude Sonnet 5.5 | Tạo `PROGRESS.md` (template không có trong repo) từ khảo sát chỉ-đọc: `state/*.db` (ro), `data/`, `out/`, `logs/`, `git` ở HEAD `ae6b90d`; chạy 36 test (scorer, make_submission, chunk) pass |
| 2026-10-03 23:10 | Claude Sonnet 5.5 | Thêm điểm LB từ `SCOREBOARD.md` (M8, M9, mục 5, 7, 8); kiểm số học bảng điểm |
| 2026-10-04 01:02 | Codex (GPT-6) | PHA B: src layout, paths/env/guards/status, pins và docs; core36/full257/slow2 pass, replay sub06 JSON/hash/diff=0; OLD/state/fixtures bất biến. Copy raw và extract lớn chưa chạy. |
| 2026-10-04 01:22 | Codex (GPT-6) | Review độc lập + 11 regression RED→GREEN; suite268/slow2 pass, replay JSON/hash giữ nguyên, các input hash và elapsed đã lưu. Không có Critical/Important còn mở. |
| 2026-10-04 01:26 | Codex (GPT-6) | Sau gate repo/push: copy raw18.031 shard+crawl.db+domain CSV, hash/inventory/SQL khớp, OLD không đổi, raw root NEW ổn định; chưa extract lớn. |
| 2026-10-04 01:50 | Codex (GPT-6) | Sửa hai validator JSON, regression4 RED→GREEN/full272 pass; ghi chênh ZIP và guard chỉ preflight. Chuẩn bị extract đủ18.031 shard, workers2 trên RAM16GiB; chưa có kết quả extract lớn. |
| 2026-10-04 01:57 | Codex (GPT-6) | Khởi chạy extract raw NEW/state NEW workers2; snapshot 1039shards/43674docs. JSON validator thực1200/22800/0lỗi. Các bước sau chờ extract hoàn tất. |
| 2026-10-04 03:22 | Codex (GPT-6) | Extract NEW đủ18.031 shard/632208doc; checkpoint/raw/parquet khớp, state OLD bất biến. Chunk/index/K100/submission chờ kết quả; không ghi điểm LB. |
| 2026-10-04 03:27 | Codex (GPT-6) | Chunk NEW nạp623.805 doc đủ điều kiện trong118s, target256; cập nhật đường găng vì extract đã hoàn tất. Chưa có index/K100/submission mới. |
| 2026-10-04 13:51 | Claude Opus 5.5 | Chunk NEW lần 1 hết RAM (ArrowMemoryError sau 32.858 s, pagefile C: 49 GB). Sửa `index.chunk`: bucket theo khoảng `doc_id` vào `.chunk_tmp` trong out-dir, xử lý từng lô `--batch-docs` (mặc định 20.000), ghi nối `chunk_id`, output `.partial` → rename, dọn tmp/partial khi lỗi. Đối chiếu code cũ trên 301 file / 10.402 doc, target 128+256, lô 1.000 và 20.000: docs/chunks `Table.equals` + schema + `chunk_report.json` giống hệt. Thêm 2 test (lô 1/2/100 cho cùng output; lỗi không để lại partial); full suite 274 pass / 2 deselected. Chưa chạy lại toàn corpus. |
| 2026-10-04 14:05 | Claude Opus 5.5 | Chuyển repo `D:/R2AI/r2ai-stage3` → `D:/GitHub/r2ai-stage3`, OLD → `D:/GitHub/r2ai-stage3-old` (không xóa). Sửa prefix 18.031 path trong `extract.db` (pending 0), cài lại editable (freeze 102 không đổi), cập nhật path trong `.env.example`, README, AGENTS (cảnh báo `git clean -fdx` cả trong repo vì data/state là file ignored), `configs/submission-sub06.yaml`. Kiểm chứng: full suite 274 pass / 2 deselected; replay sub06 từ input `-old` JSON SHA256 `07e0fd59…acc3`, validator 1.200 query / 22.800 chunk / 0 lỗi; `crawl.py status` đọc state NEW. |
| 2026-10-04 15:40 | Claude Opus 5.5 | Exact block-scan candidates: `retrieve/exact.py` (dense.npy đọc theo khối căn ranh giới doc, CSR mmap cache tách từ `sparse.npz` mỗi lần 1 mảng, running top-k tie theo id giảm như IndexFlatIP), `--candidates exact` trong `run_retrieval_k100` (mặc định `faiss`), log `vi_k100.candidates.parquet` + cảnh báo < 100 doc, peak RSS/private trong meta. `index.build`: sửa thiếu import `save_npz` (crash shard 0) và `flat_gb` không định nghĩa (crash khi ghi meta), thêm `--no-ann` và subcommand `assemble`. Gate legacy 99,917% K100 trước rerank giống hệt; full suite **296 pass / 2 deselected**. Embed NEW không chạy; chưa có K100/submission 3,9M. |
| 2026-10-05 09:05 | Claude Opus 5.5 | Chạy thật NEW: embed 37.953 s; K100 exact 1.200 query 8.481 s, peak RSS 7,391 / private 7,046 GiB, VRAM 1.452 MiB; sửa `Index.texts` overflow >2 GiB và text mmap (`a76052b`, private 12,44 → 2,07 GiB ở bước ứng viên); submission sub06-config ZIP 39.237.992 byte SHA256 `1d171722…567a`, validator 1.200/22.800/0 lỗi; chưa nộp LB. Suite 299 pass / 2 deselected. |
| 2026-10-05 10:40 | Claude Opus 5.5 | Ghi điểm LB `sub_new_vi_kd100_kc19_full` (Final 0,1748, người dùng cung cấp) vào `submissions/LOG.md` + PROGRESS (M7–M9, mục 5/7/8, đầu file, Chunking), sắp Changelog theo thời gian, đo nguyên nhân JSON nhỏ hơn sub06. `make_submission`: thêm `--extra-chunk-docs` / `--extra-zip-budget-bytes` (mặc định tắt; JSON mặc định tái tạo đúng SHA `50ac7367…`). Sinh V2 N=45 (ZIP 47.792.584 byte, validator 0 lỗi, diff ngoài phạm vi 0); V1 không cần (đã trả cả nhóm), V3/V3b/V4 không sinh (thiếu thứ tự doc ngoài top 100). 6 test mới; full suite **305 pass / 2 deselected** (232 s). Thứ tự nộp đề xuất: V2 → (V3/V3b nếu chạy lại truy hồi) → V4 sau khi có điểm. |
