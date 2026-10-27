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
| mã | tag `v-private-2026-10-28` trên `main` (chạy từ một worktree tách ở tag đó, không checkout trên repo chung) |

## 2. Kiểm tra trước khi đóng băng

* **Hồi quy public** (`scripts/s41/regress_public.py`): `newq.apply_selection` với config chốt trên đúng pool / 4B / C-max của FREE-noJina (1.200 query public) cho docsel trùng FREE-noJina ở mọi cột trừ Jina và cờ carrier, và tập carrier trùng CH-rrf (175.548, ρ 0,7937, 0 lệch). Compose: xem `docs/s41.md` mục hồi quy.
* **Hồi quy public:** compose ra JSON trùng từng byte JSON trong ZIP CH-rrf (`eeb51b06…3da2`); ZIP lệch 8 B chỉ vì tên tệp JSON bên trong.
* **Diễn tập 100 query** (`scripts/s41/run_rehearsal.sh`, run `out/runs/s41/rh100-final`): prepare 50 s, export 205 s, import 2 s, build (compose gồm) 295 s; `--impute-4b` 253 s; thiếu 30% 4B + impute 257 s; ba ZIP validator ok. Chi tiết `docs/s41.md`.
* **FREE-Kfit** (số doc/query theo K_q học từ P1): LB 0,3994, thua CH-rrf 0,0144, bác (`docs/s41.md`). CH-rrf và tag `v-private-2026-10-28` là bản cuối; đóng băng.

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

- [ ] Worktree tách ở tag `v-private-2026-10-28`, `.env` chép từ repo chung (RUNS_DIR đúng).
- [ ] `python -m r2ai.newq pins` báo `input pins ok: 79 files`.
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
* 4B thiếu mà không dùng `--impute-4b`: `build` từ chối; Vu quyết.

**5 lượt private:** (1) bản chính ngay khi có; (2) giữ cho bản sửa lỗi; (3) đến (5) để dành (chờ (B), (C)).

## 6. Nếu một nguồn hỏng: mất bao nhiêu, làm gì (đo trên 1.200 query public, Vu 27/10)

Mã `scripts/s41/fallback_public.py` (đi qua đúng các hàm của `newq build`: `pools_from`, `cosine_pcal_map`, `apply_selection`; config chốt), driver `scripts/s41/run_fallback.sh`, run `out/runs/s41/fallback/<bản>`. Đối chứng FULL (đủ 4B, cùng đường mã) cho docsel và carrier trùng CH-rrf (220.436 doc, 175.548 carrier, 0 lệch). Doc F2 dự đoán chấm docsel của từng bản bằng mô hình p_cal của bản **đủ 4B** (cùng G_q), nên so được với nhau; mô hình này coi EF2 đủ 4B là tối ưu và chỉ để xếp hạng, LB mới là số thật.

| nguồn hỏng | bản đo | doc (TB/query) | Doc F2 dự đoán | Δ so CH-rrf | LB (Vu) | hành động |
|---|---|---:|---:|---:|---|---|
| không (đối chứng) | CH-rrf | 220.436 (183,7) | 0,4253 | 0 | 0,4138 | bản chính |
| C-max (hỏng / quá chậm) | **B0**: `build --no-cmax`, RRF cosine + 4B | 220.472 (183,7) | 0,4245 | −0,0008 | chờ | nộp B0 nếu C-max chưa xong ở T0 + 24 giờ |
| 30% cặp 4B (phiên Kaggle dở / hết quota) | **IMP30**: xoá 30% (seed 2026), `build --impute-4b` | 289.469 (241,2) | 0,3984 | −0,027 | chờ | `build --impute-4b` |
| toàn bộ 4B | **COS**: xoá 100%, `--impute-4b`, C-max giữ | 358.759 (299,0) | 0,3786 | −0,047 | chờ | xem ghi chú COS |

Ghi chú:
* **B0:** khác CH-rrf 15.784 doc (7%), Doc F2 dự đoán gần như không đổi. LB public cũ khi bỏ C-max ở cấu hình còn Jina (free-v2 0,4101 → FREE-noCmax 0,4077): mất 0,0024 Final, Doc −0,0032.
* **IMP30:** ánh xạ cosine → p_cal fit trên 70% cặp còn điểm (c 12,861, d −9,754). Cặp được suy điểm xếp cuối nguồn 4B, nhưng p_cal suy ra vẫn đủ cao để EF2 giữ nhiều doc hơn (241 doc/query): recall dự đoán tăng (0,589), precision giảm (0,199).
* **COS:** `newq build --impute-4b` hiện **từ chối** khi còn dưới 1.000 cặp có 4B (không fit được ánh xạ). Bản đo dùng ánh xạ fit trên toàn bộ 4B public (c 12,890, d −9,773). Muốn dùng được ở private cần ghim hai hằng số này vào config. Nguồn 4B trong RRF thành hằng, nên RRF chỉ còn cosine + C-max; EF2 giữ gần cả pool (299 doc/query, chạm trần 300).
* ZIP (chưa nộp):

  | bản | ZIP | B | SHA256 | T | chunk (cửa sổ) | validator |
  |---|---|---:|---|---:|---|---|
  | B0 | `out/runs/s41/fallback/B0/compose/FB-B0-full+w768-t1536.zip` | 98.383.174 | `d2f234e992e18a202f7323d78142f96283f17bae167a3460488598a0fedd0d29` | 1536 | 207.881 (32.992) | ok, 0 carrier bị bỏ |
  | IMP30 | compose 19:00 | | | | | |
  | COS | compose 19:00 | | | | | |

  IMP30 và COS: `VARIANTS="IMP30 COS" bash scripts/s41/run_fallback.sh` (mỗi bản khoảng 45 phút; B0 mất 44,8 phút).
