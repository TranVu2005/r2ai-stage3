# S40: chuẩn bị bộ private (26/10/2026)

Nhánh `feat/s40-private` (tách từ `origin/main` `e179ffd`, worktree `.superpowers/worktrees/s40-private`). Không nộp LB (Vu nộp). Số LB trong tài liệu này chỉ là số Vu báo.

## 1. Ablation khung free (120A-free-v2, mốc LB 0,4101)

Mỗi bản bỏ đúng một thành phần của `configs/s36-handoff-120A-free-v2.json`; mọi thứ khác giữ nguyên (pool top-300 theo thứ tự corpus, a, b, s_free, cắt EF2 theo P4B, carrier CR-fair tổng 175.548, thin, compose `full+w768` với `--fit-carriers`, T tự tìm). Mã: `apply` nhận `rrf_keys` cả ở khung free (vắng khoá thì đủ bốn nguồn, free-v2 không đổi); config `configs/s40-abl-free-v2-{noJina,noCmax}.json`; driver `scripts/s40/run_abl_free.sh` (watchdog `scripts/s40/watchdog.sh`, log `out/runs/s40/abl/abl.log`). Chạy 26/10 10:03 đến 11:41 (apply 20 giây mỗi bản, compose noJina 38,5 phút, noCmax 44,3 phút, nowin 14,2 phút).

| bản | thay đổi | ZIP (B) | SHA256 | T | cửa sổ | chunk | doc | carrier bị bỏ | Doc F2 dự đoán | validator |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 120A-free-v2 (đối chứng) | không | 100.285.047 | `fba176c6…da3f` | 1536 | 34.186 | 209.176 | 220.599 | 0 | 0,4250 | ok |
| FREE-noJina | RRF 3 nguồn: cosine + 4B + C-max | 100.014.559 | `153aba8b…c918d` | 1536 | 33.898 | 208.876 | 220.436 | 0 | 0,4253 | ok, `p1_pair` 88.086, `canonical_substring` 120.790 |
| FREE-noCmax | RRF 3 nguồn: cosine + 4B + Jina | 100.327.272 | `1c19caf3…e2f4` | 1536 | 34.002 | 208.976 | 220.133 | 0 | 0,4240 | ok, `p1_pair` 88.202, `canonical_substring` 120.774 |
| FREE-nowin | chunk chỉ là full body, bỏ cửa sổ 768 | 96.951.936 | `2ffa3b2a…ce9a` | không có | 0 | 174.990 | 220.599 | 0 | 0,4250 (docsel không đổi) | ok, `p1_pair` 88.506, `canonical_substring` 86.484 |

SHA đầy đủ: noJina `153aba8b61c14e74b0ab2d26bc344111a8bdf1ef56d0e6d8850b03c97e3c918d`, noCmax `1c19caf3e2bd8c4569eced83c4a259f822d7116f1ac2dc9d633a9de658d8e2f4`, nowin `2ffa3b2a71928a514e56013f0a16b8bb217a0f334aed928ef6316b7add56ce9a`.

ZIP: `out/runs/s40/abl/compose/ABL-noJina-free-v2-full+w768-t1536.zip`, `out/runs/s40/abl/compose/ABL-noCmax-free-v2-full+w768-t1536.zip`, `out/runs/s40/abl/nowin/120A-free-v2-full+w768-t1000000000.zip` (nowin dựng bằng `compose write --threshold 1000000000` trên chính cache prepare của bản dựng lại free-v2 `out/runs/release/120A-free-v2/compose`; trùng SHA từng byte với ZIP dò T của S38 `out/runs/s38-chunk-metric-v2/compose/120A-free-v2-full+w768-t1000000000.zip`, nên dựng lại được). Cả noJina và noCmax đều ra T = 1536 (sàn của luật), 0 carrier bị bỏ.

Doc khác free-v2 (so docsel): noJina bỏ 8,71 / thêm 8,58 doc mỗi query, khác ở 1.186/1.200 query; noCmax bỏ 10,21 / thêm 9,82, khác ở 1.184/1.200 query. Doc F2 dự đoán chỉ phản ánh thứ tự RRF tới việc doc nào lọt top-k (p_cal vẫn từ P4B), không đo giá trị thật của Jina và C-max; cần điểm LB.

## 2. Lệnh cho bộ query mới (`python -m r2ai.newq`)

Xem docstring `src/r2ai/newq.py`. Chuỗi lệnh: `prepare --queries <parquet> --run <dir>` → `export-scoring --run <dir> [--parts-4b K]` → (Kaggle 4B, `scripts/s40/run_newq_gpu.ps1` cho C-max rồi Jina) → `import-scores --source {4b,cmax,jina}` → `build --run <dir> [--no-jina] [--no-cmax]`. Hằng số trong `configs/s40-newq-free-v2.json` (a 0,5298075151496865, b −1,40969757276664, s_free 1,7405181516323944, g_depth 211, t_c 0,2030746281894702, carrier `target_per_query` 146,29 = 175.548/1.200), không fit lại.

Không đọc tệp nào gắn với id của 1.200 query public. Những chỗ mã cũ đọc dữ liệu public đã được thay khi có `--query-set`: vector query S29 (C-max chọn chunk, compose chọn chunk tốt nhất), registry cặp S37 (compose: chỉ mục doc → segment lấy từ mọi segment 120ask), điểm chunk S32 / S37 (compose: điểm 0.6B của chính các dòng C-max mới), P1 (compose không lấy text chunk P1; validator `--no-p1`, `--queries <parquet>`). Cổng chọn chunk của `d300_cmax.py` (so 1.000 cặp S32 public) không chạy.

Khác với free-v2 có chủ ý (chờ Vu quyết, 26/10): (1) text body của carrier lấy từ canonical / 120ask / Pharmacity thay vì chunk P1; (2) làn en chưa có trong `newq` (cơ chế hiện chọn 0 doc en).

## 3. Diễn tập 100 query (đang chạy)

Đầu vào `scripts/s40/rehearsal_queries.py`: 100 query public ngẫu nhiên (`numpy.random.default_rng(2026)`), đánh số lại id 5001 đến 5100 (bảng nối `out/runs/s40/rehearsal-input/id-map-rh100.parquet`), để mọi lần đọc nhầm dữ liệu theo id public hiện ra thành lệch. Run `out/runs/s40/rh100`.

| bước | nơi | thời gian đo (100 query) |
|---|---|---:|
| embed query (BGE-M3, GPU) | máy | 21,3 s (gồm nạp mô hình) |
| cosine top-400 (S26 + Pharmacity) | máy, CPU | 18,0 s (gồm nạp 1,8 GB vector) |
| ghép 120ask, top-300 | máy | 1,4 s |
| cắt text 512 token (27.764 doc) | máy | khoảng 2,3 phút |
| gói 4B + pool Jina + pool C-max | máy | khoảng 1,2 phút (C-max: chọn chunk S32, 23 chunk nhúng CPU) |
| C-max 0.6B, 78.205 dòng (2,61 dòng/cặp) | GPU máy | 3.011 s (26,0 dòng/s) |
| Jina v3.5 b10s0 | GPU máy | đang chạy |
| 4B, 30.000 cặp | Kaggle 2×T4 (Vu) | đang chạy |

Pool so với free-v2 (`scripts/s40/compare_pool.py`): 82/100 query trùng tập top-300; 18 query khác tổng 543 doc. 5 query khác 77 đến 135 doc đều do một khối hoà cosine (215 đến 378 doc trùng nội dung có đúng một cosine) vắt qua mốc 300: bản sao nào lọt top-400 / top-300 là tuỳ ý ở cả hai bên; 13 query khác 1 đến 8 doc do vector query không trùng bit với q_dense S26 (cosine ≥ 0,99992 mỗi query, 56% số query trùng bit; cách chia batch của S26 không được ghi lại, thử lô 1 / 8 / 16 / 32 / 64 / 128 / 256 có và không xếp theo độ dài đều không ra 100%).

## 4. Probe chunk FREE-split (Vu, 26/10)

Trên docsel / carrier / thin của free-v2, T = 1536 cố định (Vu: trần ZIP không ràng buộc T ở luật chia vì không nhân đôi text; T nhỏ còn làm ZIP nhỏ đi do khối 4T cắt body). Body > T: các đoạn liền kề, không chồng lấn, nguyên văn, cắt ở cuối câu; n = ceil(token / T) đoạn phủ hết body khi body ≤ 4T; body > 4T: 4 đoạn ≈ T trong khối 4T căn quanh chunk t256 tốt nhất (dịch vào trong khi chạm đầu hoặc cuối), phần ngoài khối bỏ. Validator `--max-chunks-per-doc 4`. Mã `scripts/s40/split_compose.py`, driver `scripts/s40/run_split_half.sh`. FREE-half T1536 bỏ (gần như trùng split ở T này). Nếu LB của FREE-split T1536 thắng Chunk F2 0,3800 thì dựng FREE-split T1024 và FREE-half T1024; thua thì dừng hướng chia.
