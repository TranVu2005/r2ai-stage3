# Private: cấu hình chốt, kế hoạch và checklist giờ G (S41, 27/10/2026)

Thay mục 5 của `docs/s40-private.md` (các phần còn lại của tài liệu đó vẫn đúng). Đóng băng theo Vu 27/10: không thử nghiệm thêm.

## 1. Cấu hình chốt

| mục | giá trị |
|---|---|
| doc | khung free, pool cosine top-300 (D = 300), RRF k = 60 trên cosine / 4B / C-max, **không Jina**, cắt EF2 theo p_cal |
| hiệu chỉnh | cũ: a 0,5298075151496865, b −1,40969757276664, s_free 1,7405181516323944, g_depth 211 |
| carrier | **CH-rrf**: CR-fair, tổng 146,29 × N (175.548 ở N = 1.200), mỗi query lấy `ceil(ρ · |D_q|)` doc đầu theo **thứ tự RRF** của docsel |
| chunk | full body + cửa sổ 768 khi body > T, T tự dò (sàn 1536) dưới trần ZIP 102.600.000 B, text canonical |
| mốc public | LB 0,4138 (Doc 0,4413, Chunk 0,3863), ZIP `b12b85912ba03ec45062fb817b806583c0182633a89feaa2ae4f1372767908c8` |
| config | `configs/s41-private-final.json` (mặc định của `python -m r2ai.newq`) |
| ghim đầu vào | `configs/s41-newq-pins.json` (79 tệp, cùng SHA256 với `s40-newq-pins.json`; tệp trong repo ghi dạng `repo:<đường dẫn>` nên đúng ở mọi checkout); config ghim SHA256 của chính tệp ghim |
| mã | **tag `v-private-2026-10-28c`** trên `main` (Vu 28/10: CH-rrf chốt; chuỗi dự phòng CH-rrf → B0 → COS-K150; bản 5 phòng hộ EN-hedge N = 3; chạy từ một worktree tách ở tag đó, không checkout trên repo chung). Tag cũ `v-private-2026-10-28` (chưa có `--cos-only`) và `v-private-2026-10-28b` (chưa có `--cos-k`, `--en-hedge`) giữ nguyên, cùng CH-rrf. Tag đã tạo không sửa |

## 2. Kiểm tra trước khi đóng băng

* **Hồi quy public** (`scripts/s41/regress_public.py`): `newq.apply_selection` với config chốt trên đúng pool / 4B / C-max của FREE-noJina (1.200 query public) cho docsel trùng FREE-noJina ở mọi cột trừ Jina và cờ carrier, và tập carrier trùng CH-rrf (175.548, ρ 0,7937, 0 lệch). Compose: xem `docs/s41.md` mục hồi quy.
* **Hồi quy public:** compose ra JSON trùng từng byte JSON trong ZIP CH-rrf (`eeb51b06…3da2`); ZIP lệch 8 B chỉ vì tên tệp JSON bên trong.
* **Diễn tập 100 query** (`scripts/s41/run_rehearsal.sh`, run `out/runs/s41/rh100-final`): prepare 50 s, export 205 s, import 2 s, build (compose gồm) 295 s; `--impute-4b` 253 s; thiếu 30% 4B + impute 257 s; ba ZIP validator ok. Chi tiết `docs/s41.md`.
* **FREE-Kfit** (số doc/query theo K_q học từ P1): LB 0,3994, thua CH-rrf 0,0144, bác (`docs/s41.md`). CH-rrf và tag `v-private-2026-10-28` là bản cuối; đóng băng. Sau đó thêm nhánh dự phòng COS (mục 6.1), tag `v-private-2026-10-28b`, cấu hình chính không đổi.

## 3. Thời gian (N = 1.200, D = 300)

| bước | nơi | thời gian |
|---|---|---:|
| kiểm ghim 79 tệp | máy | 20 s mỗi lệnh |
| prepare (embed, cosine, ghép 120ask) | máy, GPU cho embed | khoảng 2 phút |
| export 4B (2 phần) + C-max | máy CPU | 10 đến 20 phút |
| 4B phần 1, 180.000 cặp | Kaggle 2×T4, phiên 1 | **7,5 giờ** |
| 4B phần 2, 180.000 cặp | Kaggle 2×T4, phiên 2 | **7,5 giờ** |
| C-max, khoảng 940.000 dòng | GPU máy, bf16 | khoảng 10 giờ, song song với 4B |
| build + compose + dò T + validator | máy CPU | 35 đến 45 phút |

Tổng Kaggle: **khoảng 15 giờ GPU** (360.000 cặp, 6,7 cặp/s). Đường găng: hai phiên 4B nối tiếp, bản chính có sau khoảng **T0 + 16:30**.

## 4. Chỗ Vu cần điền trước 01/11

(A) giờ mở đề 01/11: `____`; (B) private có hiện điểm sau mỗi lượt không: `____`; (C) lượt lỗi định dạng có bị tính không: `____`; (D) quota GPU Kaggle còn lại trong tuần chứa 01–02/11 (cần khoảng 15 giờ): `____` giờ.

## 5. Checklist giờ G

T0 = lúc Vu có tệp query. Run `out/runs/private` (tên cố định; chưa tồn tại trước T0).

**Trước T0 (31/10)**

- [ ] Worktree tách ở tag `v-private-2026-10-28c`, `.env` chép từ repo chung (RUNS_DIR đúng).
- [ ] `python -m r2ai.newq pins` báo `input pins ok: 79 files`.
- [ ] Làn en cho EN-hedge (ngoài 79 tệp ghim, kiểm tay bằng `sha256sum`): `data/s37_en/seg37900-en-pubmed/vectors.npz` `9edfe699a76f0030443b7d9662fae54769a37008243af4ed57bd764511a17bb0`, `docs.parquet` `c0e93e14e0893595911ed32f41ac9f5869705fea7d086a49f17a630053ce41c2`. Lệch thì không dựng EN-hedge, báo Vu.
- [ ] GPU máy rảnh (không job nào khác), ổ đĩa còn ít nhất 30 GB.
- [ ] Vu đã điền (A) đến (D); quota Kaggle ≥ 15 giờ.

**T0 + 0:00 đến 0:30 (tôi)**

- [ ] Tệp query: SHA256 trùng SHA Vu gửi. `prepare` chỉ đọc parquet có cột `id` (int, duy nhất) và `query` (text không rỗng); định dạng khác thì đổi sang parquet trước, ghi lại SHA cả hai tệp.
- [ ] `python -m r2ai.newq prepare --queries <tệp> --run out/runs/private` (log `out/runs/private/newq.log`).
- [ ] `python -m r2ai.newq export-scoring --run out/runs/private --parts-4b 2`: hai dataset `kaggle-dataset-private-4b-{1,2}`, mỗi phần một notebook.
- [ ] Bắt đầu C-max trên GPU máy: `powershell -File scripts\s40\run_newq_gpu.ps1 -Run <run>` (`gpu.log`, `gpu.pid`; supervisor tự khởi động lại khi CUDA lỗi).

**Vu**

- [ ] Tải `kaggle-dataset-private-4b-1` lên Kaggle, chạy `private-4b-1.ipynb`; upload sẵn phần 2.
- [ ] Phiên 1 xong (khoảng T0 + 8:00): tải `scores-extra.parquet` và `collect-extra-report.json`, chạy ngay `private-4b-2.ipynb`.
- [ ] Phiên 2 xong (khoảng T0 + 15:45): tải kết quả.

**Nhập điểm (tôi)**

- [ ] `import-scores --source 4b --tag private-4b-1 --scores ... --report ...` (kiểm SHA, revision `22e68366…`, đủ cặp).
- [ ] C-max xong (khoảng T0 + 10:30): `gpu.log` báo import xong, `cmax.parquet` có trong `ledger.json`.
- [ ] `import-scores --source 4b --tag private-4b-2 ...`.

**Dựng (khoảng T0 + 16:30)**

- [ ] `python -m r2ai.newq build --run out/runs/private` (không cờ: không Jina, carrier CH-rrf từ config).
- [ ] `build.json`: `carriers.order` = `rank`, tổng carrier = round(146,29 × N), doc/query trung bình gần 184 (public: 183,7).
- [ ] ZIP < 102.600.000 B, validator S26 0 lỗi; báo Vu đường dẫn ZIP + SHA256.
- [ ] Vu nộp lượt 1.

**Đường dự phòng**

* Phiên Kaggle chết: chạy lại notebook của phần đó (bỏ qua part đã xong).
* Hết quota hoặc đến 04/11 vẫn thiếu 4B: `build --impute-4b` (cặp thiếu 4B lấy p_cal từ ánh xạ cosine, xếp cuối nguồn 4B). Diễn tập thiếu 30%: Doc F2 mô hình khoảng 0,389 so với 0,419 khi đủ.
* C-max hỏng hoặc quá chậm: `build --no-cmax`, nộp nếu đến T0 + 24 giờ C-max chưa xong.
* 4B thiếu mà không dùng `--impute-4b`: `build` từ chối. Không có phần 4B nào: `build --cos-k 150` (COS-K150, mục 6.2; thay `--cos-only`).

**5 lượt private:** (1) bản chính ngay khi có; (2) giữ cho bản sửa lỗi; (3), (4) để dành (chờ (B), (C)); (5) phòng hộ EN-hedge N = 3 (mục 6.4), dùng theo quy tắc chấm của BTC.

## 6. Nếu một nguồn hỏng: mất bao nhiêu, làm gì (đo trên 1.200 query public, Vu 27/10)

Mã `scripts/s41/fallback_public.py` (đi qua đúng các hàm của `newq build`: `pools_from`, `cosine_pcal_map`, `apply_selection`; config chốt), driver `scripts/s41/run_fallback.sh`, run `out/runs/s41/fallback/<bản>`. Đối chứng FULL (đủ 4B, cùng đường mã) cho docsel và carrier trùng CH-rrf (220.436 doc, 175.548 carrier, 0 lệch). Doc F2 dự đoán chấm docsel của từng bản bằng mô hình p_cal của bản **đủ 4B** (cùng G_q), nên so được với nhau; mô hình này coi EF2 đủ 4B là tối ưu và chỉ để xếp hạng, LB mới là số thật.

| nguồn hỏng | bản đo | doc (TB/query) | Doc F2 dự đoán | Δ so CH-rrf | LB (Vu) | hành động |
|---|---|---:|---:|---:|---|---|
| không (đối chứng) | CH-rrf | 220.436 (183,7) | 0,4253 | 0 | 0,4138 (Doc 0,4413 P 0,2332 R 0,6575; Chunk 0,3863) | bản chính |
| C-max (hỏng / quá chậm) | **B0**: `build --no-cmax`, RRF cosine + 4B | 220.472 (183,7) | 0,4245 | −0,0008 | **0,4101** (Doc 0,4354 P 0,2299 R 0,6513; Chunk 0,3847): −0,0037 | nộp B0 nếu C-max chưa xong ở T0 + 24 giờ |
| 30% cặp 4B (phiên Kaggle dở / hết quota) | **IMP30**: xoá 30% (seed 2026), `build --impute-4b` | 289.469 (241,2) | 0,3984 | −0,027 | chờ | `build --impute-4b` |
| toàn bộ 4B (cũ, đã thay) | **COS**: xoá 100%, `--impute-4b`, C-max giữ | 358.759 (299,0) | 0,3786 | −0,047 | **0,3607** (Doc 0,3708 P 0,1621 R 0,7073; Chunk 0,3507): −0,0531 | thay bằng COS-K150 |
| toàn bộ 4B | **COS-K150**: như COS nhưng thay EF2 bằng 150 doc đầu theo thứ tự RRF (cosine + C-max) | 180.000 (150,0) | 0,3489 (mô hình p_cal, không dùng để chọn K) | −0,076 | **0,3742** (+0,0135 so COS): −0,0396 | `build --cos-k 150` (mục 6.2, 6.3) |

Ghi chú:
* **Chuỗi dự phòng (Vu 28/10): CH-rrf → B0 → COS-K150.** COS-K150 thay COS (LB 0,3742 so 0,3607). IMP30 chưa có LB, nằm ngoài chuỗi.
* **LB (Vu 27/10):** mô hình đánh giá thấp mức mất: B0 Doc −0,0059 (dự đoán −0,0008), COS Doc −0,0705 (dự đoán −0,047). COS gần như không cắt: 299,0 doc/query (P10 = P50 = 300, chỉ 19 query dưới 300, min 101), P 0,16 / R 0,71; R của COS gần bằng recall trần của pool cosine top-300.
* **B0:** khác CH-rrf 15.784 doc (7%), Doc F2 dự đoán gần như không đổi. LB public cũ khi bỏ C-max ở cấu hình còn Jina (free-v2 0,4101 → FREE-noCmax 0,4077): mất 0,0024 Final, Doc −0,0032.
* **IMP30:** ánh xạ cosine → p_cal fit trên 70% cặp còn điểm (c 12,861, d −9,754). Cặp được suy điểm xếp cuối nguồn 4B, nhưng p_cal suy ra vẫn đủ cao để EF2 giữ nhiều doc hơn (241 doc/query): recall dự đoán tăng (0,589), precision giảm (0,199).
* **COS:** `newq build --impute-4b` vẫn **từ chối** khi còn dưới 1.000 cặp có 4B (không fit được ánh xạ). Nhánh này đi bằng `newq build --cos-only` (Vu duyệt 27/10): không đọc điểm 4B nào, mọi cặp lấy p_cal từ ánh xạ ghim trong config riêng `configs/s41-private-fallback-cos.json` (xem bên dưới). Nguồn 4B trong RRF thành hằng, nên RRF chỉ còn cosine + C-max; EF2 giữ gần cả pool (299 doc/query, chạm trần 300).
* ZIP (chưa nộp):

  | bản | ZIP | B | SHA256 | T | chunk (cửa sổ) | validator |
  |---|---|---:|---|---:|---|---|
  | B0 | `out/runs/s41/fallback/B0/compose/FB-B0-full+w768-t1536.zip` | 98.383.174 | `d2f234e992e18a202f7323d78142f96283f17bae167a3460488598a0fedd0d29` | 1536 | 207.881 (32.992) | ok, 0 carrier bị bỏ |
  | IMP30 | `out/runs/s41/fallback/IMP30/compose/FB-IMP30-full+w768-t1536.zip` | 97.989.849 | `ba25768d8d4b588dbf21490b022cf8bb676bdbc2b0d8d7b7a869886498259ea0` | 1536 | 207.825 (33.222) | ok, 0 carrier bị bỏ |
  | COS | `out/runs/s41/fallback/COS/compose/FB-COS-full+w768-t1536.zip` | 99.424.114 | `1c5600bbd54b80fc2095927d583cc37635b0920940503b172feda0b270790fc1` | 1536 | 207.939 (33.457) | ok, 0 carrier bị bỏ |
  | COS-K150 | `out/runs/s41/fallback/COSK150/compose/FB-COSK150-full+w768-t1536.zip` | 98.548.799 | `fd33125209bc508eadc459fa569a23780b3d58e8f5dc833f9b04758a3d93c2fb` | 1536 | 207.290 (33.403) | ok, 0 carrier bị bỏ |

  Driver `scripts/s41/run_fallback.sh` (tuần tự, log `out/runs/s41/fallback/fallback.log`): B0 44,8 phút, IMP30 43,5 phút, COS 44,0 phút, COS-K150 28,1 phút (28/10 03:40 đến 04:09).

### 6.1 Config COS và `--cos-only` (Vu 27/10)

Mã `--cos-only` và config COS có từ tag `v-private-2026-10-28b` (Vu duyệt 27/10; tag cũ `v-private-2026-10-28` không có). Từ 28/10 nhánh không 4B đi bằng COS-K150 (mục 6.3), cùng ánh xạ ghim này.

* `configs/s41-private-fallback-cos.json` (SHA256 `81ba043f299f674002da532c9b6c2ab3fef4000cc4e935883c57d087beca22e9`) kế thừa `configs/s41-private-final.json` (ghim SHA256 `61922113…4ca4759`) và chỉ thêm `impute_4b = {mode: cos_only, c 12,890141042157486, d −9,772626922053805, source}`. `load_config` từ chối config kế thừa có khoá nào khác ngoài `impute_4b` / `note`, hoặc khi SHA256 tệp gốc lệch. `configs/s41-private-final.json` không đổi.
* `build --cos-only` chỉ đọc c / d từ config này (mặc định, hoặc `--cos-config`). Nó dừng nếu các khoá chọn doc (depth, keep, a, b, s_free, g_depth, t_c, rrf_k, carriers) khác config của run, và không đi cùng `--impute-4b` / `--test-drop-4b`. `--impute-4b` thường vẫn fit ánh xạ riêng và vẫn cần ≥ 1.000 cặp có 4B.
* Hằng số ghim **đủ chữ số** (giá trị fit trong `out/runs/s41/fallback/COS/fallback.json`, SHA256 `5fba1514…84bb`). Bản làm tròn 3 chữ số (12,890 / −9,773) cho docsel thiếu 2 doc và lệch 1 carrier so với FB-COS, nên không ra đúng ZIP `1c5600bb…`.

**Hồi quy** (driver `scripts/s41/run_cos_regress.sh`, log `out/runs/s41/cos-regress/cos-regress.log`, 27/10 19:45 đến 21:11):

| kiểm | kết quả |
|---|---|
| a. CH-rrf mặc định (`regress_public.py --variant CH-rrf --compose`, config chốt) | docsel 220.436 dòng trùng, carrier 175.548 trùng (0 lệch); ZIP 100.651.136 B, SHA256 `b12b85912ba03ec45062fb817b806583c0182633a89feaa2ae4f1372767908c8` **trùng từng byte**; compose 43,0 phút |
| b. COS qua đường `--cos-only` (`regress_public.py --variant COS --compose`: `load_config(COS_CONFIG)`, `cos_only_map`, `pools_from` không 4B, trên pool / C-max public) | docsel 358.759 dòng trùng (SHA256 tệp cũng trùng), carrier 175.787 trùng; ZIP 99.424.114 B, SHA256 `1c5600bbd54b80fc2095927d583cc37635b0920940503b172feda0b270790fc1` **trùng từng byte**; compose 35,5 phút |
| c. rh100 `--cos-only`, không nhập phần 4B nào (run `out/runs/s41/rh100-cos`) | prepare 52 s, export 167 s, import C-max 1 s, `build --cos-only` (compose gồm) 200 s; 29.920 doc (299,2/query), 30.000 cặp suy điểm, ZIP 10.571.362 B `029afc848914901754b35988e1030262e5d37a5185b1f9e6a760a57e328c6d96`, validator ok, 0 carrier bị bỏ; Doc F2 mô hình 0,387 |

Hồi quy a và b đi qua `newq.apply_selection` trên pool public dựng lại như `r2ai.s36.s37.apply` (như hồi quy mục 2), không qua `newq prepare`, vì public không có run newq. Tên build đặt trùng bản tham chiếu (`CH-rrf`, `FB-COS`) để tên JSON trong ZIP trùng.

### 6.2 Lệnh từng nhánh và điều kiện chuyển (run `out/runs/private`, mã tag `v-private-2026-10-28c`)

Mọi lệnh dưới đây chạy trong worktree tách ở tag `v-private-2026-10-28c` (tag `…-28b` không có `--cos-k`, `--en-hedge`; tag `…-28` không có `--cos-only`).

Kiểm trước khi chọn: `python -m r2ai.newq status --run out/runs/private` (phần 4B đã nhập, có `cmax.parquet` không). Mỗi nhánh ghi vào thư mục build riêng, nên dựng được nhiều nhánh rồi chọn bản nộp. ZIP + SHA256 nằm ở `out/runs/private/build/<tên>/build.json` khoá `zip`.

| nhánh | khi nào | lệnh | tên build |
|---|---|---|---|
| CH-rrf (chính) | đủ mọi phần 4B và đã nhập C-max | `python -m r2ai.newq build --run out/runs/private` | `private-free` |
| B0 | đủ 4B; C-max hỏng hoặc chưa xong ở T0 + 24 giờ | `python -m r2ai.newq build --run out/runs/private --no-cmax` | `private-free-noCmax` |
| IMP30 | thiếu một phần 4B (phiên Kaggle chết, hết quota) mà đến 04/11 vẫn không chạy bù được; còn ≥ 1.000 cặp có 4B | `python -m r2ai.newq build --run out/runs/private --impute-4b` | `private-free-impute4b` |
| COS-K150 | không phần 4B nào nhập được (dưới 1.000 cặp có 4B: `--impute-4b` từ chối) | `python -m r2ai.newq build --run out/runs/private --cos-k 150` | `private-free-cosk150` |
| (cũ) COS | đã thay bằng COS-K150; giữ để đối chiếu | `python -m r2ai.newq build --run out/runs/private --cos-only` | `private-free-cosonly` |
| EN-hedge (bản 5) | phòng hộ, không thay bản chính: bản đã dựng của nhánh đang dùng + tối đa 3 doc en/query (mục 6.4) | thêm `--en-hedge 3 --name <tên build>-en3` vào lệnh của nhánh đó, ví dụ `python -m r2ai.newq build --run out/runs/private --en-hedge 3 --name private-free-en3` (đặt `--name` riêng để không ghi đè thư mục build của bản chính) | `private-free-en3`; ZIP `…-full+w768-t<T>-en3.zip` trong `compose/`, `build.json` khoá `en_hedge` |

* C-max cũng thiếu: thêm `--no-cmax` vào lệnh IMP30 / COS-K150. Tổ hợp này chưa đo trên public.
* Chuỗi chuyển (Vu 28/10): CH-rrf → B0 → COS-K150 (LB public 0,4138 / 0,4101 / 0,3742). IMP30 (thiếu một phần 4B) chưa có LB và không nằm trong chuỗi; dùng hay không do Vu quyết khi xảy ra.
* Một phần 4B về muộn: nhập rồi dựng lại nhánh tốt hơn; bản đã nộp giữ nguyên.

### 6.3 COS-K: K cố định thay EF2 khi không có 4B (Vu 28/10, tag `v-private-2026-10-28c`)

COS gần như không cắt (299 doc/query, P 0,16 / R 0,71), vì p_cal suy từ cosine không đủ dốc để EF2 bỏ doc. COS-K giữ đúng `K` doc đầu mỗi query theo thứ tự RRF của COS (nguồn 4B hằng, nên là cosine + C-max), carrier CR-fair như cũ.

* LB (Vu 28/10): **0,3742**, hơn COS 0,0135; thay COS trong chuỗi dự phòng. Mã từ `feat/s41` (`f5a06f5`), có từ tag `v-private-2026-10-28c` (tag `…-28b` không có).
* `configs/s41-private-fallback-cos-k150.json` kế thừa `configs/s41-private-fallback-cos.json` (ghim SHA256 `81ba043f…e22e9`) và chỉ thêm `cos_k = 150`. `build --cos-k K` đọc `configs/s41-private-fallback-cos-k{K}.json`, dừng nếu `cos_k` trong config khác K, và không đi cùng `--impute-4b` / `--test-drop-4b`. Lệnh: `python -m r2ai.newq build --run out/runs/private --cos-k 150`, tên build `private-free-cosk150`.
* Chọn K = 150 (Vu 28/10, thứ tự RRF cosine + C-max) bằng neo LB, không dùng p_cal: TP(K) = TP(299) · (K/299)^β, với β ≥ 0,33 suy từ neo CH-rrf / B0 / COS. Doc F2 so với COS: β 0,33 → +8,3%, β 0,40 → +3,2%, β 0,50 → −3,7%. K = 150 bền nhất trên dải β này. LB của COS-K150 cho thêm một neo trên thứ tự cosine để ước β.
* Public (`scripts/s41/fallback_public.py`, bản `COSK150`): 180.000 doc (150 mọi query), khác CH-rrf 24.733 doc mới / 65.169 doc bỏ; 175.200 carrier (141.022 trùng CH-rrf). ZIP ở bảng mục 6. Doc F2 mô hình p_cal 0,3489 (mô hình này coi EF2 là tối ưu nên phạt mọi cách cắt khác; chỉ ghi lại, không dùng để chọn).
* Diễn tập rh100 `--cos-k 150` (run `out/runs/s41/rh100-cos`): 15.000 doc (150/query), ZIP 10.491.554 B `448cc699d00f8ea529ed1d9c0de6a56d59a7256f0b37716cc0fb3fc9472ac41f`, T 1536, 16.966 chunk, 0 carrier bị bỏ, validator ok; compose 393 s, tổng 424 s.

### 6.4 EN-hedge: bản 5 phòng hộ (Vu 28/10, tag `v-private-2026-10-28c`)

* **Là gì:** ZIP đã dựng của nhánh đang dùng, giữ nguyên phần vi/zh (cùng thứ tự), nối vào CUỐI `relevant_docs` tối đa N = 3 doc en mỗi query: doc của làn en (`data/s37_en/seg37900-en-pubmed`, 239.771 doc) có cosine BGE-M3 cao nhất, chỉ nhận doc có cosine ≥ X. Mỗi doc en có một chunk full body, nối cuối `relevant_chunks`. Mã `r2ai.s43.enhedge` (từ `exp/s43`), cờ `newq build --en-hedge N [--en-min-cos X]`, mặc định 0 (bản chính không đổi). X mặc định là P99 cosine trên mọi cặp (query, doc en) của bộ query đang chạy; public ra 0,5226.
* **Dùng thế nào:** lượt 5 (phòng hộ), theo quy tắc chấm của BTC. Public không có gold en, nên LB public chỉ đo giá phải trả (dự đoán Final khoảng −0,003). Bản này có lời ở private chỉ khi precision en vượt khoảng 0,08.
* **Public (S43, giao Vu 27/10):** `out/runs/s43/probes/EN-hedge3-c5226/EN-hedge3-c5226-full+w768-t1536.zip`, 102.124.766 B (trần 102.600.000), SHA256 `ce0fabc01cd017ff2372351355711a31affa0152940f77c4f945acebc6188742`, validator ok. 3.594 doc en (2,995/query, 1.199 query), phần vi/zh trùng CH-rrf 1.200 / 1.200 query. Lượt nộp public lỗi phía máy chấm (`/app/output/scores.txt` không tồn tại, `docs/PROGRESS.md` 28/10 01:40), nên chưa có số LB.
* **Kích thước:** EN-hedge thêm khoảng 1,5 MB vào ZIP. CH-rrf public 100,65 MB thành 102,12 MB, chỉ cách trần 0,48 MB. `enhedge.run` ghi `under_ceiling` vào `build.json`; nếu ZIP vượt trần thì không nộp, báo Vu.
* **Đầu vào ngoài ghim:** làn en không có trong 79 tệp của `configs/s41-newq-pins.json`; kiểm tay SHA256 hai tệp ở checklist mục 5 trước khi dựng.
* Diễn tập trên rh100 (28/10, run `out/runs/s41/rh100-cosk150`): `build --cos-k 150 --en-hedge 3 --name rh100-cosk150-free-cosk150-en3` chạy hết trong 223 s. X = 0,5217 (P99 tính trên 100 query), 300 doc en (3/query, 100 query), phần trước giữ nguyên. ZIP 10.617.191 B `f13ee54c924905d6981a208f668397095a2992f918d78096834fbc71077f8108`, dưới trần, validator ok (nguồn thêm `seg37900-en-pubmed`).

### 6.5 Hồi quy trước tag `v-private-2026-10-28c` (Vu 28/10)

Trên `main` sau khi merge `feat/s41` (`--cos-k`) và `exp/s43` (`--en-hedge`, kéo theo `--w-cos-doc` / `--w-cos-carrier` mặc định 1). Driver `scripts/s41/run_28c_regress.sh`, log `out/runs/s41/regress-28c/regress-28c.log`, 28/10 07:43 đến 09:11. Test toàn bộ: 1.079 passed.

| kiểm | kết quả |
|---|---|
| CH-rrf mặc định (`regress_public.py --variant CH-rrf --compose`) | docsel 220.436 dòng trùng, carrier trùng; ZIP 100.651.136 B `b12b85912ba03ec45062fb817b806583c0182633a89feaa2ae4f1372767908c8` **trùng từng byte**; compose 37,4 phút |
| EN-hedge (`regress_enhedge.py`, nền là ZIP CH-rrf vừa dựng lại, N 3, X 0,5226) | 3.594 cặp en trùng `hedge.parquet` của S43, cả khi đọc tệp làn en lẫn khi tính lại từ vector (`lane_pairs`, đường `newq --en-hedge`); ZIP 102.124.766 B `ce0fabc01cd017ff2372351355711a31affa0152940f77c4f945acebc6188742` **trùng từng byte** bản S43 đã nộp; validator ok; 16,9 phút |
| COS-K150 (`regress_public.py --variant COSK150 --compose`: `cos_only_map` + `cos_k_keep` của config K150) | docsel 180.000 dòng trùng (SHA256 tệp trùng), carrier trùng; ZIP 98.548.799 B `fd33125209bc508eadc459fa569a23780b3d58e8f5dc833f9b04758a3d93c2fb` **trùng từng byte**; compose 28,2 phút |
| rh100 `--cos-k 150`, run mới `out/runs/s41/rh100-cosk150`, không nhập phần 4B nào | prepare 33 s, export 125 s, import C-max 0 s, `build --cos-k 150` (compose gồm) 165 s; 15.000 doc (150/query), 30.000 cặp suy điểm, T 1536, 16.966 chunk, 0 carrier bị bỏ, validator ok. ZIP 10.491.562 B `23d1d3ae62997627a8f29252f78da7ad2012e23c6ba512ded7874d7bdc1f115e`: JSON bên trong trùng từng byte bản rh100 trước (`448cc699…`), chỉ khác tên tệp JSON vì tên run khác |

### 6.7 Tuỳ chọn cho bản phụ: cắt doc non-carrier sâu (Vu 29/10, S45; không thuộc tag `v-private-2026-10-28c`)

* **Là gì:** bước hậu xử lý chạy trên ZIP đã dựng, không đụng `newq` hay config. `scripts/s45/trim_noncarrier.py` bỏ các doc **không** mang chunk có hạng cosine > 250 trong pool của run. Chunk giữ y nguyên; script kiểm chunk ⊂ relevant_docs rồi chạy validator S26.
* **Lệnh** (run private; nguồn verbatim như compose của bản gốc):
  `python scripts/s45/trim_noncarrier.py <ZIP bản chính> out/runs/private/top.parquet <ZIP mới> --queries out/runs/private/queries.parquet out/runs/s27-pharmacity/handoff/cc-s27-pharmacity <thư mục .validator-extra của compose>`
* **Lợi dự kiến:** rất nhỏ. Public: P-D250-doconly +0,0005 (Doc P +0,0071, R −0,0082), DO-200 +0,0003, dưới ngưỡng +0,003 để đổi bản chính (`docs/s45.md` mục 5f–5g). Rủi ro thấp vì chunk không đổi.
* **Dùng thế nào:** chỉ cho bản phụ (lượt 3 hoặc 4), Vu quyết. Bản chính (lượt 1) không dùng.
* **Diễn tập:** R0 public: 7,64 doc bỏ/query, validator ok. rh100 CH-rrf `57dca179…57b5`: 6,78 doc bỏ/query, ZIP `067cc6a3…fffc`, validator ok. Script có trên nhánh `exp/s45`, cần lấy về worktree private nếu dùng.
