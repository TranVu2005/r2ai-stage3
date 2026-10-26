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

Kết luận: chuỗi lệnh tương đương free-v2 về thuật toán (bước 0); lệch của bản diễn tập là do chấm lại (chủ yếu Jina listwise), không phải lỗi mã. Với bộ private, mọi nguồn được chấm một lần trên đúng pool top-300, nên không còn cặp thiếu Jina như free-v2.

## 4. Probe chunk FREE-split (Vu, 26/10)

Trên docsel / carrier / thin của free-v2, T = 1536 cố định (Vu: trần ZIP không ràng buộc T ở luật chia vì không nhân đôi text; T nhỏ còn làm ZIP nhỏ đi do khối 4T cắt body). Body > T: các đoạn liền kề, không chồng lấn, nguyên văn, cắt ở cuối câu; n = ceil(token / T) đoạn phủ hết body khi body ≤ 4T; body > 4T: 4 đoạn ≈ T trong khối 4T căn quanh chunk t256 tốt nhất (dịch vào trong khi chạm đầu hoặc cuối), phần ngoài khối bỏ. Validator `--max-chunks-per-doc 4`. Mã `scripts/s40/split_compose.py`, driver `scripts/s40/run_split_half.sh`. FREE-half T1536 bỏ (gần như trùng split ở T này). Nếu LB của FREE-split T1536 thắng Chunk F2 0,3800 thì dựng FREE-split T1024 và FREE-half T1024; thua thì dừng hướng chia.

**FREE-split T1536 (26/10 12:35):** `out/runs/s40/split/FREE-split-t1536.zip`, 96.773.825 B, SHA256 `ef709999f48524643f467f4e04c97aa5a3376b829f6f85678aaf7747c411a2b2`, validator ok (`--max-chunks-per-doc 4`, 0 lỗi, `p1_pair` 63.489, `canonical_substring` 147.482). 220.599 doc, 174.990 carrier, 210.971 chunk (tối đa 4 chunk một doc). Doc bị chia 34.186 (19,5% carrier), 70.167 đoạn, **2,053 đoạn TB/doc bị chia**; token mỗi đoạn P10/P50/P90/max = 788 / 893 / 1.210 / 1.990. Cắt ở cuối câu trừ 2 mốc (đầu từ), 0 doc phải giữ nguyên vì đoạn rỗng / trùng. 877 doc bị chia trên body span canonical vì body P1 không còn là chuỗi con của doc_text hiện tại (P1 cắt từ text cũ).

Doc dài hơn 4T = 6.144 token: **208 doc** (0,12% carrier), lấy khối 4 đoạn quanh chunk tốt nhất (cả 208 đều có chunk tốt nhất), bỏ 645.359 token ngoài khối. Host: tamanhhospital.vn 47, hellobacsi.com 33, a-hospital.com 28, thanhnien.vn 25, zhongyibaodian.net 20, msdmanuals.cn 14, medlatec.vn 7, youmed.vn 5, vietnamnet.vn 5, test.pmphai.com 5, 11 host còn lại 19.

Lần dựng đầu (12:07) bị validator loại (866 chunk không nguyên văn, do chia trực tiếp body P1 cũ), giữ ở `out/runs/s40/split/v1-p1body-not-verbatim/`, không nộp.

## 5. Kế hoạch private 01–04/11 (nháp 26/10, theo tốc độ đo ở diễn tập)

Giả định: N = 1.200 query (đổi N thì mọi giờ tỷ lệ thuận, trừ compose). Giờ mở đề (01/11) chưa biết: T0 = lúc Vu có tệp query. Hạn nộp 04/11 23:59. Bộ query nhận dạng `id` (int) + `query` (text), tệp parquet.

**Tốc độ đo (26/10, 100 query) và ngoại suy cho 1.200:**

| bước | nơi | đo | 1.200 query |
|---|---|---|---:|
| prepare (embed + cosine + ghép 120ask) | máy | 41 s | khoảng 2 phút (nạp mô hình / vector là phần lớn; nhân ma trận tăng tuyến tính, chưa đo ở 1.200) |
| export (text 512 + gói 4B / C-max / Jina) | máy CPU | khoảng 3,5 phút | khoảng 10 đến 20 phút (phần đọc canonical cố định khoảng 80 s; phần tokenize tăng theo số doc; chưa đo) |
| 4B, 360.000 cặp | Kaggle 2×T4 | 6,7 cặp/s (8.961 s T4 cho 30.000 cặp) | **14,9 giờ** liền, hoặc 3 phần × 5,0 giờ |
| C-max, 939.600 dòng (2,61 dòng/cặp) | GPU máy | 26,0 dòng/s | **10,0 giờ** |
| Jina b10s0 | GPU máy | 2,63 query/phút, không OOM | **7,6 giờ** (S37 đo 1,49 đến 1,96 khi GPU dùng chung: tới 13,4 giờ) |
| build (chọn doc) | máy | 2 s | dưới 1 phút |
| compose full+w768 + validator | máy CPU | (diễn tập: chờ đo) | khoảng 20 phút với T biết trước, 35 đến 45 phút khi dò T (ablation: 38,5 và 44,3 phút) |

**Đường găng:** GPU máy chỉ chạy một job: C-max rồi Jina = 17,6 giờ (tới 23 giờ nếu Jina chậm như S37). 4B trên Kaggle 14,9 giờ nếu một phiên nối tiếp (phải tách: phiên Kaggle tự dừng ở 10,5 giờ theo notebook). Hai nhánh chạy song song, nên bản đủ 4 nguồn có sau khoảng T0 + 18 đến 24 giờ.

**4B song song (giảm nửa thời gian):** `export-scoring --parts-4b 3` tạo 3 dataset + 3 notebook (5 giờ mỗi phần). Nếu Kaggle cho chạy 2 phiên GPU cùng lúc (chưa kiểm, Vu xác nhận) hoặc 1 phần chạy trên RunPod, 4B xong sau khoảng 5 đến 10 giờ. RunPod 4090: **chưa đo**; `judge_score.py` (scorer ghim) chạy được trên mọi GPU CUDA với cùng lệnh `score` / `status` / `collect`, nhưng fp16 trên 4090 khác số học T4 (S31: lệch tới 0,0047 giữa máy), nên cần một lần thử 200 cặp trước 01/11 nếu dùng. Quota Kaggle: diễn tập dùng khoảng 1,3 giờ; private cần khoảng 15 giờ phiên trong tuần 01–04/11.

**Lịch theo giờ (T0 = lúc có tệp query):**

| mốc | việc | ai |
|---|---|---|
| T0 + 0:00 | `newq prepare --queries <tệp> --run out/runs/s40/private` | tôi |
| T0 + 0:20 | `export-scoring --parts-4b 3`; Vu upload 3 dataset, chạy notebook (song song nếu được); GPU máy bắt đầu C-max (`scripts/s40/run_newq_gpu.ps1`) | tôi, Vu |
| T0 + 5 đến 15 giờ | các phần 4B về; `import-scores --source 4b --tag ...` sau mỗi phần | Vu, tôi |
| khi đủ 4B | **bản dự phòng B0 = cosine + 4B** (`build --no-jina --no-cmax`), validator; Vu nộp sớm, trước 03/11 | tôi, Vu |
| T0 + 10 giờ | C-max xong, nhập; Jina bắt đầu tự động; dựng B1 = cosine + 4B + C-max (`build --no-jina`) nếu cần | tôi |
| T0 + 18 giờ | Jina xong, nhập; **bản chính = đủ 4 nguồn** (`build`) | tôi |
| còn lại | dự phòng thời gian cho sự cố (OOM, phiên Kaggle đứt: notebook resume theo phần) | |

**Điểm dừng an toàn:** mỗi nguồn nhập độc lập; thiếu nguồn nào thì dựng bản không có nguồn đó (`--no-jina`, `--no-cmax`). 4B là bắt buộc (p_cal, cắt EF2 dựa trên P4B): nếu đến 04/11 12:00 vẫn thiếu 4B của một số query thì cần quyết định của Vu (chưa có luật cho query thiếu 4B; hiện `build` từ chối). Mọi job dài: supervisor tự khởi động lại, log và checkpoint theo phần.

**Phân bổ 5 lượt private (đề xuất, Vu quyết):** (1) B0 cosine + 4B, nộp ngay khi có, để luôn có một bản hợp lệ; (2) bản chính đủ 4 nguồn; (3) biến thể tốt nhất theo LB public của ablation / probe chunk (noJina / noCmax / FREE-split nếu thắng free-v2); (4) và (5) để dành cho sửa lỗi định dạng hay bản cuối. Chưa rõ private có hiện điểm sau mỗi lượt hay không (Vu xác nhận); nếu không hiện thì nên nộp bản chính cuối cùng.
