# PROGRESS – R2AI Stage 3

- Cập nhật: 2026-10-04 01:26 (UTC+7)
- Model thực hiện: Codex (GPT-6)
- Commit HEAD đã kiểm trước cập nhật này: `52c7c61`, nhánh `main` (hash commit chứa tài liệu này được báo ở kết quả PHA B)
- Public: 31/10/2026 · private: 04/11/2026 (tối đa 5 lượt) · kết quả: 11/11/2026; mốc do người dùng cung cấp
- Quy ước: ✅ có bằng chứng · 🟡 làm một phần · ❌ chưa làm (đã tìm, không thấy) · ❓ không chứng minh được

> Ghi chú: lúc kiểm tra **không có file `PROGRESS.md` template** trong repo (`ls`, `find`) nên cấu trúc mục 1–9 và danh sách M0–M12 được dựng lại từ `R2AI_STAGE3_CONTEXT.md` §4, §6 và từ nhiệm vụ. Hãy đối chiếu nếu bạn có template gốc.
> Snapshot nguồn OLD ngày 03/10: repo git nằm ở `D:\GitHub` (gốc chung nhiều dự án). `git ls-files` trong `r2ai-stage3` chỉ có **4 file đã commit** (`eval/README.md`, `eval/scorer.py`, `eval/validate.py`, `tests/test_scorer.py`). Mọi thứ khác (crawler, extract, index, submission…) là **untracked**, nên không có commit hash để dẫn; bằng chứng là path và mtime.


> **PHA B / repo hiện tại:** `D:/R2AI/r2ai-stage3`, history mới trên main. Các số crawl/extract/index/LB phía dưới là snapshot legacy OLD ngày 03/10, không phải kết quả chạy lớn ở NEW. Mapping hiện tại: `eval`→`src/r2ai/eval`, `index`→`src/r2ai/index`, `retrieve`→`src/r2ai/retrieve`, `submission`→`src/r2ai/submit`, `vicrawl`→`src/vicrawl`, `config`→`configs`; CLI xem README. Đọc query/docs/chunks/index/K100 cũ ở OLD; ghi pipeline mới vào NEW. Status mặc định đọc NEW/state, đối chiếu OLD phải dùng `--state-dir` rõ ràng.
> Gate đã đo: core **36 pass**, full **268 pass / 2 deselected**, slow **2 pass / 10 deselected**; OLD full 234 pass và slow 2 pass. Replay JSON sub06 **181.960.105 byte**, SHA256 `07e0fd59f8cc4b8abbf129dc1536305bdbef17ce510b6fe39fe50fe62133acc3`, byte/query/field diff=0; validator **1.200 query / 22.800 chunk / 0 lỗi**. Không rerun model/GPU hay nộp LB. Python 3.12.13, torch 2.11.0+cu128, 102 package freeze khớp OLD. Chi tiết run ở `out/runs/restructure/` và `out/runs/reproduce-sub06/` (ignored).
> Copy raw/crawl.db: **Đã copy** 18,031 shard / 3,679,529,131 byte (3.427 GiB) và crawl.db 376,438,784 byte sang NEW; hash/inventory nguồn–đích khớp, SQL 653.970 URL / 624.596 ok / 48 domain khớp. Writer dừng, WAL0; không overwrite hay move/xóa OLD. RAW_DIR=RAW_WRITE_DIR=NEW/data/raw_vi; status mặc định NEW đã kiểm. Extract.db mới chưa có; chưa chạy extract/crawl/GPU lớn. OLD raw hash sau copy, source161 và state6 vẫn không đổi. Manifest `out/runs/restructure/data-copy.json`.

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
| M5 | Extract toàn bộ doc vi | 🟡 | `state/extract.db`: 3.103 shard đã xử lý/18.031; `data/docs_vi`: 125.499 doc = 20,1% số `ok`. Lần ghi cuối 2026-10-03 02:43. Xem mục 4 |
| M6 | Scorer local theo metric | ✅ | `eval/scorer.py`, `tests/test_scorer.py`, commit `ae6b90d`. Chưa đối chiếu với điểm LB nào (không có điểm LB) |
| M7 | Chunk + index + truy hồi baseline (BGE-M3 hybrid + reranker) | 🟡 | `data/index/t256/` (faiss.index, dense.npy, sparse.npz, 6,8 GB), 818.080 chunk, **chỉ trên 123.874 doc**; `data/runs/vi_k100.parquet` (1.200 query). Xem mục 5 |
| M8 | Nộp thử lên leaderboard | ✅ | `Downloads/SCOREBOARD.md` (người dùng cung cấp, 2026-10-03): 7 lượt có điểm, tốt nhất sub06 Final 0,0626. Đối chiếu tên/dung lượng zip với `out/submissions/` khớp. `out/submissions/LOG.md` chưa cập nhật (vẫn ghi "nothing uploaded") |
| M9 | Nộp thử A/B (chunk dài/ngắn; 1 id/cả nhóm; vi/zh) | 🟡 | Xong: chunk dài vs ngắn (c2 0,0158 → full 0,0273 ở K=5), quét K 5/20/50/100, cửa sổ 1024/2048. Chưa làm: 1 id vs cả nhóm id trùng, vi vs có zh |
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
| Chunking | ✅ t128/t256/t400 | `data/chunks/chunks_t256.parquet` 818.080 chunk; `chunk_report.json`; `tests/test_chunk.py` |
| Index | 🟡 FAISS `IndexFlatIP` dense + sparse, t256, `built_at` 2026-10-03 06:11:40 | `data/index/t256/meta.json` (123.874 doc, 3,12 GB float32). Index **cũ**: lập trước khi crawl xong |
| Truy hồi | 🟡 hybrid w=0.7, `max`, rerank `bge-reranker-v2-m3` fp16, seed 42 | `retrieve/run.py`, `data/runs/vi_k100.meta.json` (1.200 query, 13.083 s = 10,9 s/query), `vi_k100.parquet` 119.934 dòng (≥74, p50 100 doc/query) |
| Dev set | ⚠ pseudo-dev, **bão hòa và lệch xa LB** (dev chunk F2 0,358 so với LB Final 0,0158 cùng cấu hình c2) | `data/dev/pseudo_vi_v2.parquet` 462 dòng (400 A tiêu đề-làm-query + B hỏi đáp). `out/retrieval/ablation_t256.md`: A R@5 0,9975, B R@1 1,0; chunk-level B F2 0,358. Trang nguồn nằm trong index nên số này **không dự báo điểm LB** (chính `out/submissions/LOG.md` nêu điều này) |
| File nộp | sub01–sub10 đã sinh, chưa nộp | `out/submissions/*.zip` (sub01, sub02 zip 3,5 MB; sub03–sub10 json/zip tới 110 MB). `scripts/make_submission.py`, `submission/build.py`, `tests/test_make_submission.py` |
| Điểm leaderboard | ✅ 7 lượt, nguồn `Downloads/SCOREBOARD.md` | Final: sub02 0,0158 · sub03 0,0273 · sub04 0,0551 · sub05 0,0623 · **sub06 0,0626** · sub07 0,0545 · sub08 0,0507. Final = (Doc F2 + Chunk F2)/2 kiểm lại đúng cho cả 7 dòng. sub01 không có điểm riêng; sub09, sub10 và sub05_k50_full không nộp / không upload được |

Lưu ý: sub01/sub02 dùng K=5 doc, C=2 chunk; sub03–sub10 gồm các biến thể K/chunk dài ("full" tới 45.518 token), cửa sổ 1024/2048 token và ngân sách zip 45,7 MiB. Giới hạn dung lượng upload của leaderboard là giả định trong `LOG.md`, chưa xác minh.

## 6. Lệch so với `R2AI_STAGE3_CONTEXT.md`

Chuyển raw/state sau gate repo: **Đã copy** 18,031 shard / 3,679,529,131 byte (3.427 GiB) và crawl.db 376,438,784 byte sang NEW; hash/inventory nguồn–đích khớp, SQL 653.970 URL / 624.596 ok / 48 domain khớp. Writer dừng, WAL0; không overwrite hay move/xóa OLD. RAW_DIR=RAW_WRITE_DIR=NEW/data/raw_vi; status mặc định NEW đã kiểm. Extract.db mới chưa có; chưa chạy extract/crawl/GPU lớn. OLD raw hash sau copy, source161 và state6 vẫn không đổi. Manifest `out/runs/restructure/data-copy.json`.

Review độc lập toàn bộ repo: sửa 3 nhóm Important (profile/dup import, guard nested redirects/sidecars, docs parquet rỗng); 11 test RED→GREEN. Suite sau sửa: **268 pass**, slow **2 pass**, replay sub06 SHA/diff=0 trong **78.8s**. Minor kế thừa OLD: validator JSON trực tiếp dùng Path.read_text(newline=...) không tương thích Python3.12; validator ZIP đã pass, sửa compatibility ở đợt riêng.

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

1. **Extract dừng ở 20%**: 125.499/624.596 doc `ok`; các domain lớn nhất (suckhoedoisong, thanhnien, vinmec, medlatec, suckhoecongdongonline) gần như chưa được extract. Index và mọi file nộp hiện chỉ bao phủ 123.874 doc. Vinmec và MEDLATEC (nguồn của 83% query theo gold check) có ~300 doc/domain trong index → recall sẽ thấp.
2. **Recall là nút thắt**: LB tốt nhất 0,0626 (Doc R 0,088 ở K=100). Index chỉ 2,8% corpus (123.874/4.394.718 URL) và chưa có zh. Pseudo-dev không dùng được để chọn cấu hình; chỉ LB mới đo thật. Còn chưa biết scorer local có khớp LB không (chưa có bản chấm local trên cùng file nộp, vì không có gold).
3. **Chưa có dữ liệu zh**: CONTEXT ước ~77% corpus là zh; hiện 0 doc zh. Nếu gold có nhiều zh, recall bị chặn trên.
4. `youmed.vn` halted (bot_challenge); `vov.vn` và `baolangson.vn` chặn bởi robots (11.765 URL không lấy được, đúng theo quyết định tôn trọng robots).
5. Chunk "full" của sub03–sub10 có chunk dài tới 45.518 token; không rõ scorer chính thức xử lý ra sao (đã ghi trong `LOG.md`).
6. Snapshot OLD phần lớn code chưa commit; PHA B đang lưu toàn bộ allowlist vào history mới tại NEW, không mang history helicopter.

## 8. 3 việc tiếp theo (đường găng tới 31/10)

1. **Raw/crawl state đã copy và kiểm hash; bước tiếp theo: extract mới đủ 18.031 shard (14.928 chưa có checkpoint), rebuild chunk/embed/index/K100/submission** (thời gian extract + embed toàn bộ: chưa đo; `data/index/t256/meta.json` chỉ ghi 327 s cho phần embed của lần chạy cuối). Đây là điều kiện để mọi bước sau có ý nghĩa.
2. **Sau rebuild, nộp lại cấu hình sub06 (kd=100, chunk full)** để đo gain từ độ phủ; chốt giới hạn upload (nằm giữa 45,7 MiB đã lên được và 110 MiB không lên được) bằng JSON compact; ghi điểm vào `out/submissions/LOG.md`. Còn lại A/B: 1 id vs cả nhóm id trùng.
3. **Quyết định zh**: lấy mẫu 2% mỗi domain zh, tính yield trên 1.200 query, rồi xếp thứ tự crawl; song song commit code đang untracked.

## 9. Changelog

| Thời gian (UTC+7) | Ai | Thay đổi |
|---|---|---|
| 2026-10-04 01:02 | Codex (GPT-6) | PHA B: src layout, paths/env/guards/status, pins và docs; core36/full257/slow2 pass, replay sub06 JSON/hash/diff=0; OLD/state/fixtures bất biến. Copy raw và extract lớn chưa chạy. |
| 2026-10-04 01:22 | Codex (GPT-6) | Review độc lập + 11 regression RED→GREEN; suite268/slow2 pass, replay JSON/hash giữ nguyên, các input hash và elapsed đã lưu. Không có Critical/Important còn mở. |
| 2026-10-04 01:26 | Codex (GPT-6) | Sau gate repo/push: copy raw18.031 shard+crawl.db+domain CSV, hash/inventory/SQL khớp, OLD không đổi, raw root NEW ổn định; chưa extract lớn. |
| 2026-10-03 23:10 | Claude Sonnet 5.5 | Thêm điểm LB từ `SCOREBOARD.md` (M8, M9, mục 5, 7, 8); kiểm số học bảng điểm |
| 2026-10-03 22:50 | Claude Sonnet 5.5 | Tạo `PROGRESS.md` (template không có trong repo) từ khảo sát chỉ-đọc: `state/*.db` (ro), `data/`, `out/`, `logs/`, `git` ở HEAD `ae6b90d`; chạy 36 test (scorer, make_submission, chunk) pass |
