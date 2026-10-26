# S40: chuẩn bị bộ private (26/10/2026)

**Pipeline chốt cho private (Vu, 26/10):** khung free D = 300, RRF(cosine, 4B, C-max), **không Jina**, chunk full body + cửa sổ 768 (T tự tìm), carrier CR-fair, thin. Mốc public mới: **FREE-noJina 0,4111** (ZIP `153aba8b…c918d`).

| bản (LB public, Vu báo 26/10) | Final | so free-v2 0,4101 |
|---|---:|---:|
| FREE-noJina | **0,4111** | +0,0010 |
| FREE-split T1536 | 0,4103 (Chunk 0,3804: R +0,018, P −0,012) | +0,0002, trung tính |
| FREE-noCmax | 0,4077 (Doc −0,0032) | −0,0024 |
| FREE-nowin | 0,4070 | −0,0031 |

Quyết định: bỏ Jina; giữ C-max; giữ full body + cửa sổ 768; dừng thử nghiệm chunk (không dựng FREE-split / FREE-half T1024).

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

Từ 26/10 chiều: `build` và `export-scoring` mặc định không Jina (`--jina` / `--sources 4b,cmax,jina` mới bật lại). Ghim đầu vào (Vu): `python -m r2ai.newq pins --write` ghi SHA256 của 79 tệp corpus / scorer (`configs/s40-newq-pins.json`: canonical docs / chunks / dense.npy, vector S26, Pharmacity, 9 segment 120ask, vector chunk bổ sung S32 / S37, scorer `688948a7`, notebook); mỗi lệnh kiểm lại (20 s) và mỗi tệp của run (`<run>/ledger.json`, ghi bởi bước sinh ra nó); thiếu tệp, lệch SHA, hay có tệp corpus chưa ghim thì dừng, không tìm nơi khác.

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
| Jina v3.5 b10s0, 100 query | GPU máy | 2.283 s (2,63 query/phút, không OOM) |
| 4B, 30.000 cặp | Kaggle 2×T4 (Vu) | 8.961 s T4 cộng hai worker, khoảng 1,24 giờ thực (6,7 cặp/s); SHA điểm `241ea4a7…d303`, revision `22e68366…`, fp16 |
| build (chọn doc), mỗi bản | máy | khoảng 2 s |

Pool so với free-v2 (`scripts/s40/compare_pool.py`): 82/100 query trùng tập top-300; 18 query khác tổng 543 doc. 5 query khác 77 đến 135 doc đều do một khối hoà cosine (215 đến 378 doc trùng nội dung có đúng một cosine) vắt qua mốc 300: bản sao nào lọt top-400 / top-300 là tuỳ ý ở cả hai bên; 13 query khác 1 đến 8 doc do vector query không trùng bit với q_dense S26 (cosine ≥ 0,99992 mỗi query, 56% số query trùng bit; cách chia batch của S26 không được ghi lại, thử lô 1 / 8 / 16 / 32 / 64 / 128 / 256 có và không xếp theo độ dài đều không ra 100%).

**Docsel diễn tập** (`newq build --no-compose`):

| bản | doc | carrier | Doc F2 dự đoán |
|---|---:|---:|---:|
| rh100-free (4 nguồn) | 17.912 | 14.629 | 0,4194 |
| rh100-free-noJina | 17.921 | 14.629 | 0,4188 |
| rh100-free-noCmax | 17.718 | 14.629 | 0,4199 |
| rh100-free-noJina-noCmax (B0) | 17.768 | 14.628 | 0,4183 |

Free-v2 trên cùng 100 query: 18.070 doc, Doc F2 dự đoán 0,4183.

**So docsel với free-v2 và nguyên nhân từng lệch** (`scripts/s40/attribute_docsel.py`: thay từng nguồn từ free-v2 sang bản mới, chạy `newq.select` mỗi bước; đếm query có tập doc trùng free-v2, và doc chỉ có ở một phía):

| bước | nguồn thay | query trùng tập doc | doc chỉ ở mới / chỉ ở free-v2 |
|---|---|---:|---:|
| 0 | không (pool, P4B, C-max, Jina của free-v2) | **100 (cả thứ tự)** | 0 / 0 |
| 0b | thêm điểm Jina mới cho 3.183 cặp (52 query) mà free-v2 không có Jina (xếp cuối) | 59 | 279 / 448 |
| 1 | pool top-300 mới (điểm free-v2 nếu có) | 50 | 326 / 458 |
| 2 | P4B mới | 44 | 331 / 462 |
| 3 | C-max mới | 25 | 380 / 561 |
| 4 | Jina mới (= bản diễn tập) | 2 | 729 / 887 |

Bước 0 trùng tuyệt đối: lõi chọn doc của `newq` (PoolFrame, hằng số trong config) cho đúng docsel free-v2, nên mọi lệch ở bước sau đến từ đầu vào. Giải thích từng nguồn:

* **Jina** (bước 0b và 4, phần lớn lệch): Jina là listwise, điểm của một doc phụ thuộc 9 doc cùng khối và cách xáo khối (hạt giống theo query id, ở đây id đã đổi). Free-v2 không chấm Jina trên top-300: điểm đến từ pool S35 (J4-K40 ∪ pool13), phần bù S36 và lần chấm lại S37, còn 31.645 cặp D = 300 không có điểm (xếp cuối). Chấm lại trên đúng pool top-300 cho ra thứ tự Jina khác; lệch điểm trên cặp chung TB 0,047, tối đa 0,31. Không có cách khớp 1:1 nếu không tái tạo đúng các pool cũ.
* **Pool** (bước 1): 18 query có tập top-300 khác (mục trên: khối hoà cosine, vector query không trùng bit).
* **P4B** (bước 2): cùng pair, cùng text, cùng scorer; lệch TB 0,0005, tối đa 0,012 (số học fp16 trên T4 giữa hai lần chạy); 6 query đổi tập doc ở vùng biên.
* **C-max** (bước 3): chọn chunk trùng ở 20.445/20.455 cặp có trong S32; trên cùng chunk, điểm 0.6B lệch p99 0,046, tối đa 0,139 (số học GPU khi chấm lại theo lô khác); 19 query đổi tập doc.

**C-max tất định (đo 26/10, `scripts/s40/cmax_determinism.py`, một part S32 2.000 dòng, GPU máy):** cùng text (60.402 dòng chung giữa S32 và diễn tập trùng 100% query_text / doc_text), cùng scorer, cùng bf16, cùng 8.192 token / 16 dòng mỗi batch, cùng part 2.000 dòng.

| cách chấm lại | lệch tối đa so với S32 | p99 | dòng trùng tuyệt đối | giây |
|---|---:|---:|---:|---:|
| bf16, cùng dòng, cùng thứ tự (= cách S32) | 0 | 0 | 100% | 79 |
| bf16, cùng dòng, thứ tự xáo (hàng xóm batch khác) | 0,072 | 0,031 | 82,5% | 77 |
| bf16, batch 1 | 0,168 | 0,065 | 25,5% | 149 |
| fp32, 8.192 / 16 | 0,093 | 0,048 | 0% | 174 |
| fp32, batch 1 | 0,093 | 0,048 | 0% | 291 |

Scorer tất định: cùng dòng và cùng cách ghép batch cho điểm trùng từng bit; `newq` ghép part theo thứ tự cố định (query, cosine_rank, doc, chunk), nên chạy lại cho cùng điểm. Lệch so với free-v2 chỉ do hàng xóm trong batch bf16 (đệm trái, độ dài batch khác): điểm C-max của free-v2 sinh trong các part S32 / S36 / S37 trộn cặp của 1.200 query, nên một bộ query khác không thể ghép lại đúng các batch đó. fp32 không phụ thuộc batch (lệch 1,8e-5) nhưng chậm 2,2 lần và lệch S32 cũng cỡ đó (vì điểm S32 chính là bf16): không khớp hơn. Giữ bf16 như free-v2 (cách dựng giống hệt).

**Tái lập theo F2 kỳ vọng, pipeline chốt (không Jina; Vu: đạt nếu |chênh| ≤ 0,002)** (`scripts/s40/expected_gap.py`, docsel diễn tập `rh100-free` so docsel FREE-noJina của cùng 100 query):

| | diễn tập | FREE-noJina | chênh |
|---|---:|---:|---:|
| Doc F2 dự đoán, mỗi bên với mô hình p_cal / G_q của nó | 0,41883 | 0,41924 | −0,00042 |
| cả hai chấm bằng mô hình của diễn tập | 0,41883 | 0,41862 | +0,00021 |
| cả hai chấm bằng mô hình của FREE-noJina | 0,41923 | 0,41924 | −0,00001 |

**Đạt.** Doc chỉ có ở diễn tập 207 (tổng p_cal 17,9), chỉ có ở FREE-noJina 203 (tổng 17,4); p_cal / ngưỡng F2_q/5 trung vị 0,91 và 0,90, 66% và 70% nằm trong 0,5 đến 2 lần ngưỡng: phần lệch là doc sát ngưỡng cắt EF2 (đổi chỗ do nhiễu số học C-max / P4B và hoà cosine ở pool), tổng giá trị kỳ vọng hai phía gần bằng nhau.

Kết luận: chuỗi lệnh tương đương free-v2 về thuật toán (bước 0); lệch của bản diễn tập là do chấm lại (chủ yếu Jina listwise), không phải lỗi mã. Với bộ private, mọi nguồn được chấm một lần trên đúng pool top-300, nên không còn cặp thiếu Jina như free-v2.

**Sự cố 26/10 12:14:54:** thư mục `out/runs/s40/rehearsal-input/` bị chuyển nguyên vẹn sang `out/runs/s40/abl/rehearsal-input/` (mtime thư mục `abl` = 12:14:54.668; SHA hai tệp không đổi). Không phải phiên này (transcript: không có lệnh nào từ 12:12:24 đến 12:23; không lệnh `mv` nào chạm thư mục đó), không phải Vu; không phiên Claude / Codex nào khác có log nhắc tới thư mục hay ghi transcript trong 12:00 đến 12:30. Windows không bật kiểm tra truy cập tệp nên không xác định được tiến trình. Hệ quả: script kiểm của diễn tập đọc đường dẫn mới cố định kèm SHA (`scripts/s40/rh_inputs.py`); mọi đầu vào của private được ghim SHA (mục 2).

## 4. Probe chunk FREE-split (Vu, 26/10)

Trên docsel / carrier / thin của free-v2, T = 1536 cố định (Vu: trần ZIP không ràng buộc T ở luật chia vì không nhân đôi text; T nhỏ còn làm ZIP nhỏ đi do khối 4T cắt body). Body > T: các đoạn liền kề, không chồng lấn, nguyên văn, cắt ở cuối câu; n = ceil(token / T) đoạn phủ hết body khi body ≤ 4T; body > 4T: 4 đoạn ≈ T trong khối 4T căn quanh chunk t256 tốt nhất (dịch vào trong khi chạm đầu hoặc cuối), phần ngoài khối bỏ. Validator `--max-chunks-per-doc 4`. Mã `scripts/s40/split_compose.py`, driver `scripts/s40/run_split_half.sh`. FREE-half T1536 bỏ (gần như trùng split ở T này). Nếu LB của FREE-split T1536 thắng Chunk F2 0,3800 thì dựng FREE-split T1024 và FREE-half T1024; thua thì dừng hướng chia.

**FREE-split T1536 (26/10 12:35):** `out/runs/s40/split/FREE-split-t1536.zip`, 96.773.825 B, SHA256 `ef709999f48524643f467f4e04c97aa5a3376b829f6f85678aaf7747c411a2b2`, validator ok (`--max-chunks-per-doc 4`, 0 lỗi, `p1_pair` 63.489, `canonical_substring` 147.482). 220.599 doc, 174.990 carrier, 210.971 chunk (tối đa 4 chunk một doc). Doc bị chia 34.186 (19,5% carrier), 70.167 đoạn, **2,053 đoạn TB/doc bị chia**; token mỗi đoạn P10/P50/P90/max = 788 / 893 / 1.210 / 1.990. Cắt ở cuối câu trừ 2 mốc (đầu từ), 0 doc phải giữ nguyên vì đoạn rỗng / trùng. 877 doc bị chia trên body span canonical vì body P1 không còn là chuỗi con của doc_text hiện tại (P1 cắt từ text cũ).

Doc dài hơn 4T = 6.144 token: **208 doc** (0,12% carrier), lấy khối 4 đoạn quanh chunk tốt nhất (cả 208 đều có chunk tốt nhất), bỏ 645.359 token ngoài khối. Host: tamanhhospital.vn 47, hellobacsi.com 33, a-hospital.com 28, thanhnien.vn 25, zhongyibaodian.net 20, msdmanuals.cn 14, medlatec.vn 7, youmed.vn 5, vietnamnet.vn 5, test.pmphai.com 5, 11 host còn lại 19.

**LB (Vu, 26/10): FREE-split T1536 = 0,4103, Chunk F2 0,3804 (R +0,018, P −0,012 so với free-v2) → trung tính.** Giữ full body + cửa sổ 768; không dựng T1024; dừng hướng chia.

Lần dựng đầu (12:07) bị validator loại (866 chunk không nguyên văn, do chia trực tiếp body P1 cũ), giữ ở `out/runs/s40/split/v1-p1body-not-verbatim/`, không nộp.

## 5. Kế hoạch private 01–04/11 (pipeline chốt: cosine + 4B + C-max, không Jina)

Giả định N = 1.200 query; giờ tỷ lệ thuận với N trừ compose. **T0 = lúc Vu có tệp query** (giờ mở đề 01/11 chưa biết). Hạn nộp 04/11 23:59 (UTC+7), tối đa 5 lượt. Tệp query: parquet có cột `id` (int, duy nhất) và `query` (text); Vu gửi kèm SHA256, `prepare` ghi SHA vào `prepare.json`, các bước sau kiểm qua `ledger.json`.

**Tốc độ đo trong diễn tập (26/10, 100 query) và ngoại suy:**

| bước | nơi | đo (100 query) | 1.200 query |
|---|---|---|---:|
| kiểm ghim 79 tệp | máy | 20 s mỗi lệnh | 20 s mỗi lệnh |
| prepare (embed, cosine top-400, ghép 120ask) | máy (GPU cho embed) | 41 s | khoảng 2 phút (chưa đo ở 1.200) |
| export 4B + C-max (text 512, chọn chunk S32) | máy CPU | khoảng 3,5 phút | 10 đến 20 phút (chưa đo ở 1.200) |
| 4B, 360.000 cặp | Kaggle 2×T4 | 6,7 cặp/s | **14,9 giờ** một luồng; 3 phần × 5,0 giờ |
| C-max, 939.600 dòng | GPU máy, bf16 | 26,0 dòng/s | **10,0 giờ** |
| build (chọn doc) | máy | 2 s | dưới 1 phút |
| compose full+w768 + dò T + validator | máy CPU | 3,5 phút | 35 đến 45 phút (ablation free 1.200 query: 38,5 và 44,3 phút) |

**Đường găng:** 4B. GPU máy chỉ còn C-max (10 giờ), chạy song song với 4B trên Kaggle. Tách 4B thành 3 phần (`export-scoring --parts-4b 3`, mỗi phần một dataset + notebook, khoảng 5 giờ). Nếu chạy được 2 phiên GPU cùng lúc (2 tài khoản Kaggle, hoặc 1 phần trên RunPod; **chưa kiểm, Vu xác nhận**), 4B xong sau khoảng 10 giờ; một phiên nối tiếp thì khoảng 15 giờ (3 phiên × 5 giờ, mỗi phiên dưới mức dừng 10,5 giờ của notebook). RunPod 4090: chưa đo; scorer ghim chạy trên mọi GPU CUDA cùng lệnh `score` / `status` / `collect`, nhưng số học khác T4 (S31: lệch tới 0,0047), nên nếu dùng thì thử 200 cặp trước 01/11. Quota Kaggle: private cần khoảng 15 giờ phiên.

**Lịch theo giờ:**

| mốc | việc | ai | đầu ra / kiểm |
|---|---|---|---|
| T0 + 0:00 | `newq pins` (20 s) rồi `newq prepare --queries <tệp> --run out/runs/s40/private` | tôi | 1.200 × 300 cặp trong `top.parquet` |
| T0 + 0:05 | `newq export-scoring --run ... --parts-4b 3` | tôi | 3 dataset 4B (120.000 cặp mỗi phần), pool C-max |
| T0 + 0:25 | Vu upload 3 dataset, chạy phần 1 (và phần 2 nếu được song song); máy chạy C-max (`scripts/s40/run_newq_gpu.ps1 -Stages cmax`, supervisor tự khởi động lại) | Vu, tôi | log `gpu.log`, ETA theo phần |
| T0 + 5:30 | phần 4B đầu về → `import-scores --source 4b --tag ...-4b-1`; Vu chạy phần tiếp | Vu, tôi | SHA, revision, số cặp |
| T0 + 10:30 | C-max xong → `import-scores --source cmax` | tôi | 939.600 dòng, 0 thiếu |
| T0 + 10:30 đến 15:30 | phần 4B còn lại về, nhập | Vu, tôi | |
| khi đủ 4B (T0 + 10 đến 16 giờ) | **bản chính** `newq build --run ...` (cosine + 4B + C-max), compose, validator (khoảng 45 phút) | tôi | ZIP < 102.600.000 B, validator 0 lỗi |
| ngay sau đó | **bản dự phòng B0** `newq build --no-cmax` (cosine + 4B) chỉ khi C-max hỏng; nếu C-max đã xong thì B0 không cần | tôi | |
| trước 03/11 | Vu nộp bản chính (lượt 1) | Vu | |
| 03/11 đến 04/11 | dự phòng: chạy lại phần lỗi, sửa định dạng | | |

**Điểm dừng an toàn:**

* C-max hỏng hoặc chậm (GPU máy lỗi): dựng **cosine + 4B** (`build --no-cmax`); LB public của cấu hình này chưa đo trực tiếp, gần nhất là FREE-noCmax 0,4077 (vẫn có Jina). Nên nộp B0 trước 03/11 nếu đến T0 + 24 giờ C-max chưa xong.
* 4B thiếu một phần: bắt buộc có 4B cho mọi cặp (p_cal và cắt EF2 dựa trên P4B); `build` từ chối khi thiếu. Chạy lại phần thiếu (notebook resume theo phần). Chưa có luật dựng khi một số query thiếu 4B: Vu quyết nếu xảy ra sau 04/11 12:00.
* Máy hỏng: mọi bước resume được từ checkpoint (phần 4B, phần C-max, `ledger.json`); compose chỉ cần CPU.

**5 lượt private (đề xuất, Vu quyết):** (1) bản chính cosine + 4B + C-max, nộp ngay khi có (trước 03/11); (2) giữ cho bản sửa nếu lượt 1 có lỗi định dạng hay validator BTC; (3) đến (5) để dành. Chưa rõ private có hiện điểm sau mỗi lượt không (Vu xác nhận); nếu có, có thể thử biến thể ở lượt 2 hoặc 3.
