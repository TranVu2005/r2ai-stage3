# S44 housekeeping: chốt bản, kiểm kê và đề xuất dọn ổ (28/10/2026)

Chưa xoá hay di chuyển dữ liệu nào ngoài phép thử tạm ở mục 3, và phép thử đó đã hoàn nguyên. Mọi việc ở mục 5 chờ Vu duyệt.

## 1. Chốt bản

* `origin`: `main` `7e8894d`, tag `v-private-2026-10-28` / `-28b` / `-28c` (object `d6d0bd07` / `260fad63` / `114854f6`), các nhánh `feat/s40-private`, `feat/s41`, `exp/s43`, `exp/s43-enhedge`, `feat/s22-reextract` trùng local, nên không push gì. `main` local được fast-forward `e179ffd` → `7e8894d` (worktree `D:/GitHub/r2ai-stage3-corpusv2`).
* Bảng nhánh: `docs/PROGRESS.md` mục "S44 bảng nhánh". Runbook private: `docs/private-runbook.md` (IMP30 không dùng).
* Ghi chú (nhánh `docs/working-notes`, commit `02dc26a`, chưa push): `lb_scores.csv` thêm 6 dòng LB S41–S43, `zips_sha256.csv` (SHA256 của 2.067 ZIP dưới `out/`), `archive_zips.csv`.

## 2. Kiểm kê

Đo 28/10 17:00–17:30, chỉ đọc (`scripts/s44/inventory.py`, `classify.py`, `hash_zips.py`). Chi tiết từng thư mục (nhóm, đường dẫn, byte, ngày mới nhất, ghi chú): `docs/s44-disk-inventory.csv` (336 dòng). Ổ D: 505 GB, đã dùng 466 GB, còn 39 GB; ổ C: còn 24 GB.

Trong repo (`data/`, `out/`, `state/`) có 270,4 GiB. Sáu worktree anh em có junction trỏ vào chính các thư mục này (mục 6), nên công cụ đo nào đi theo junction sẽ đếm trùng 6 lần.

## 3. Bằng chứng tập GIỮ-PRIVATE đủ

`scripts/s44/run_check.sh`, 28/10 17:30–17:59, log `out/runs/s44-verify-1/verify.log`, kết quả `out/runs/s44-verify-1/compare.json`.

1. `hold.py hold` chuyển (rename, cùng ổ D:, có manifest) **346 đường dẫn** không thuộc GIỮ-PRIVATE từ `data/` và `out/` sang `D:/_r2ai_s44_hold/`. Raw (`data/raw*`) và `state/` để nguyên chỗ theo AGENTS.md.
2. Các biến môi trường được đổi thêm như sau:
   * `HF_HOME` trỏ một thư mục chỉ có junction `models--BAAI--bge-m3`;
   * `R2AI_LEGACY_OUT_DIR` / `R2AI_LEGACY_STATE_DIR` trỏ đường dẫn không tồn tại;
   * mã lấy từ worktree tách ở tag `v-private-2026-10-28c` (`7e8894d`).
3. Dựng lại hai diễn tập đã ghi trong S41:

| run / build | kết quả | so với bản ghi |
|---|---|---|
| `rh100-final`: prepare, export, nhập 4B + C-max của `s40/rh100` | ledger 9 tệp (query vector, pool, top, texts, 4B, C-max) trùng SHA | trùng |
| `rh100-final-free` (CH-rrf) | ZIP 10.912.265 B `57dca1798da23ea14a7473bad3f4965ec0112b5b1aa59668d7771e004faa57b5`, validator ok | **trùng từng byte** |
| `rh100-final-free-noCmax` (B0) | ZIP 10.740.481 B `5afd66da…5f72`, validator ok | chưa có bản ghi |
| `rh100-final-free-en3` (CH-rrf + EN-hedge 3) | ZIP en3 `724054c6…80c9`, dưới trần, validator ok | chưa có bản ghi |
| `rh100-cosk150`: ledger 8 tệp | trùng SHA | trùng |
| `rh100-cosk150-free-cosk150` (COS-K150) | ZIP 10.491.562 B `23d1d3ae62997627a8f29252f78da7ad2012e23c6ba512ded7874d7bdc1f115e` | **trùng từng byte** |
| `…-cosk150-en3` (COS-K150 + EN-hedge 3) | ZIP en3 `f13ee54c924905d6981a208f668397095a2992f918d78096834fbc71077f8108` | **trùng từng byte** |

4. `hold.py restore` trả lại đủ 346 đường dẫn, thư mục hold không còn tệp nào, `git status` của repo chung như trước khi thử.

Phép thử này không chạy: Kaggle 4B (notebook tải 4B từ HF theo revision), chấm C-max trên GPU máy (`run_newq_gpu.ps1`; hai run dùng điểm C-max đã có), và đường `--impute-4b` (không dùng). Model 0.6B / 4B trong `data/hf_model_cache` vẫn giữ.

## 4. Bốn nhóm

| nhóm | GiB | gồm |
|---|---:|---|
| **GIỮ-PRIVATE** | 29,6 (+8,5 ở C:) | `data/canonical-20261018` (16,8), `data/hf_model_cache` (0.6B + 4B, 8,6), `data/s37_120ask` (1,0), `data/s37_en` (1,1, gồm `seg37900-en-pubmed`), `s26-gold-structure/full/full_doc_dense.npz` (1,7), Pharmacity vectors + handoff, `s31…/pool-extra/kaggle/judge_score.py`, `s32…/pool/supplement_vectors.npz`, `s37/cmax-d300/supplement_vectors.npz`, `reranker-trial/model-pin.json`, diễn tập `s40/rh100` + `s40/abl/rehearsal-input` (0,33); C: `models--BAAI--bge-m3` 8,5; ngoài đo: `.venv`, mã ở tag 28c |
| GIỮ-NGUỒN (không phải ứng viên) | 18,6 | `data/raw*` (AGENTS: raw ổn định đến sau 11/11), `state/` (crawl.db, extract.db; 9,4) |
| **GIỮ-NOTES** | < 0,01 trong git | bảng LB, SHA mọi ZIP, danh sách lưu trữ (nhánh `docs/working-notes`); `docs/s44-disk-inventory.csv`; 26.136 tệp json/md/csv/log nhỏ trong `out/runs` (552 MiB chưa nén) đóng gói vào nhóm LƯU-TRỮ |
| **LƯU-TRỮ** (ngoài máy, Vu chọn nơi) | 65,2 | xem 4.1 |
| **XOÁ ĐƯỢC** | 156,9 (+ 3,3 worktree doccut, + 1,1 ở C:) | xem 4.2 |

### 4.1 LƯU-TRỮ

| mục | GiB | ghi chú |
|---|---:|---|
| 6 ZIP (`docs/working-notes/data/archive_zips.csv`) | 0,56 | CH-rrf `b12b8591…08c8`, B0 `d2f234e9…0d29`, COS-K150 `fd331252…c2fb`, EN-hedge `ce0fabc0…8742`, 120A-p4b-v3 `34d235f6…37e2`, P1-carrier-w2 `8dc735e7…9c2b` (LB public cao nhất 0,4141) |
| gói điểm judge public | 0,57 | 912 tệp `*score*.parquet`, `*scores*.sqlite`, `pool_index` / `extra_pairs` / `chunk_pairs` / `pairs*` / `cmax` / `jina` parquet trong `out/runs` (trừ `compose/`); parquet đã nén, đóng `tar` không nén thêm |
| gói meta `out/runs` | khoảng 0,1 sau nén (552 MiB thô) | mọi `*.json`, `*.md`, `*.csv`, `*.tsv`, `*.log`, `*.txt`, `*.yaml`, `*.out`, `*.stdout` dưới 5 MB (`build.json`, `compose.json`, log), `tar.xz` |
| `data/index` | 50,8 | vector BGE-M3 cũ theo segment (t256 22,9; zh_full 21,5; en_pubmed 2,8; zh_v2 2,5; …), nguồn của canonical; tái tạo được bằng embed lại, mất nhiều giờ GPU; không cần cho private |
| `data/docs_*`, `data/chunks*`, `data/cc_aws`, `data/pubmed_en`, `data/s22_*` | 13,9 | docs/chunks trung gian (nguồn của canonical); tái tạo bằng extract từ raw, không chắc trùng từng byte; không cần cho private |

Ghi SHA256 từng tệp trong gói lúc đóng gói (manifest kèm gói). Nếu không có chỗ, có thể xoá `data/index` và docs/chunks (64,7 GiB) mà không ảnh hưởng private; đó là quyết định của Vu.

### 4.2 XOÁ ĐƯỢC

Gồm toàn bộ `out/runs` trừ GIỮ-PRIVATE, 6 ZIP lưu trữ và 3 tệp đang được git theo dõi (`out/runs/reranker-trial/quota-variants-report.json` đang sửa dở, `out/runs/s24-recovery/report.md`, `out/runs/s24-recovery/direct/vov.vn/crawl_extract.py`: không chuyển). Lớn nhất:

| thư mục | GiB | là gì | tái tạo |
|---|---:|---|---|
| `s6` | 24,1 | biến thể S6 (13–14/10, LB ≤ 0,2655) | không cần |
| `s38-chunk-metric-v2` | 15,2 | compose / cache S38 (`*.base.parquet` khoảng 390 MB mỗi bản) | cache: `r2ai.s38.compose` tự dựng lại |
| `s41` | 12,2 | hồi quy / fallback / probe S41; ZIP chính đã lưu trữ | `scripts/s41/run_fallback.sh`, `regress_public.py --compose` |
| `cc-longchau` | 9,6 | S3 Common Crawl longchau | không cần |
| `best-vsappend` | 8,9 | gate VS-append (07/10) | không cần; **làm hỏng `compose replay` C1 / VS-append** (pin `slug_pool_pin.parquet`, bản sao ở `D:/r2ai-pins/`) |
| `s26-gold-structure` (trừ tệp ghim) | 6,7 | S26 `full_emb`, extractor, pa2–pa6 | không cần (`full_doc_dense.npz` giữ) |
| `rrf-sweep-2026-10-06`, `rrf-docs-2026-10-06` | 10,4 | quét RRF tháng 10 | không cần |
| `s40` (trừ rh100 / rehearsal-input) | 5,9 | ablation, cut, release-regression | không cần |
| `c1`, `replay`, `s1-replay`, `replay-best-*` | 12,5 | replay C1 / sub06 | không cần cho private; `compose replay` cũ không chạy được nữa |
| `s25-canonical` | 4,5 | dựng canonical (18/10) | không cần (canonical giữ) |
| còn lại (khoảng 260 thư mục) | khoảng 47 | thí nghiệm 04–28/10, `out/logs`, `out/pytest-*` | không cần; số liệu và SHA ở `docs/`, `submissions/LOG.md`, bảng ghi chú |

Ngoài repo:

* `D:/GitHub/r2ai-stage3-doccut/out` (3,3 GiB): worktree `doc-cut`, đã merge, sạch.
* C: `models--jinaai--jina-reranker-v3.5` (1,1 GiB): chỉ Jina dùng, Jina không có trong private; tải lại từ HF được.

Không đề xuất, vì có thể thuộc dự án khác (có từ trước 10/2026):

* các model / dataset còn lại trong cache HF ở C: (Qwen3-Embedding-4B, e5, vietlegal, legal datasets, roberta, Paddle, bge-reranker-v2-m3);
* `D:/GitHub/r2ai-stage3-old` (OLD, AGENTS cấm đụng);
* `D:/GitHub/.git`.

**Giải phóng dự kiến ở D:** khoảng 160 GiB ngay sau khi xoá nhóm XOÁ (156,9 + 3,3); thêm 65 GiB nếu nhóm LƯU-TRỮ chuyển ra ngoài. Ổ còn trống từ 39 GB lên khoảng 210 GB, hoặc khoảng 280 GB.

## 5. Thủ tục sau khi Vu duyệt (chưa làm)

1. Trước khi chuyển, xác nhận mọi tác vụ lịch `r2ai-*` / `R2AI-*` đang Disabled (28/10 18:00: cả 4 Disabled) và không có tiến trình python nào của repo.
2. Đóng gói LƯU-TRỮ vào `D:/_r2ai_archive/` kèm manifest SHA256. Vu chép gói ra nơi lưu trữ và kiểm SHA.
3. Chuyển nhóm XOÁ vào `D:/_r2ai_to_delete/` (rename cùng ổ, giữ cấu trúc, manifest; dùng `hold.py` với `S44_HOLD`). Không dùng `rm` trực tiếp.
4. Chạy lại `run_check.sh` (rh100 từ tag 28c) cùng lúc chuyển tạm nhóm LƯU-TRỮ. Đạt thì báo Vu tự xoá `D:/_r2ai_to_delete/`.

## 6. Cảnh báo junction

Sáu worktree `D:/GitHub/r2ai-stage3-{vi,s6,chunk,cc,corpusv2,integrate}` có junction `data/`, `out/` (và `state/` ở bốn worktree đầu) trỏ vào `D:/GitHub/r2ai-stage3/…`. Chúng chỉ chứa mã (16–28 MiB mỗi worktree). Nếu xoá các thư mục này, trước hết gỡ junction bằng `rmdir <tên>` (không `/s`) hoặc dùng `git worktree remove`. Không bao giờ chạy `rm -rf` hay `Remove-Item -Recurse` lên chúng.
