# S36: FUSE4 (cosine + 4B + Jina + C-max), EF2-cut, LANG-thr

Nhánh `feat/s36-fuse4` (từ `feat/s35-part-jina` + merge `feat/s31-strong-judge` để có bản sửa lỗi Pharmacity `8bf30a0`). Cập nhật 2026-10-23 (UTC+7). **Chưa nộp bản nào, chưa có điểm LB của các bản dưới đây.** Số đo thực tế; chỗ nào là mô hình dự đoán thì ghi rõ.

## Bối cảnh: metric chunk đổi (BTC, 23/10)

Chunk đúng khi LCS/|gold| ≥ 0,4 **và** LCS/|chunk nộp| ≥ 0,4. Chunk nguyên body gần như chắc chắn bị tính sai, vế doc không đổi; S38 làm lại chunk. Vì vậy:

- Sản phẩm chính của S36 là thứ hạng + tập doc: `out/runs/s36-fuse4/docsel/docsel-<tên>.parquet`.
- ZIP nộp được có hậu tố `-docread`: chunk nguyên body như cũ, **chỉ dùng đọc Doc F2** (Chunk F2 của chúng không còn ý nghĩa).
- Mốc vế doc: J3-thrC Doc F2 0,4161; K4-thr-C Doc F2 0,4139 (Doc P 0,2357 / 0,2346; J4-K40 Doc P 0,376).

### docsel

Một dòng mỗi doc được chọn, theo thứ tự RRF: `query_id, doc_id, rank, rrf_score, cosine, p4b, jina, cmax, cosine_rank, judge_rank, jina_rank, cmax_rank, chunk, p4b_ge_tc, t_c` (+ `p_fused, p_cal` cho EF2-cut / LANG-thr; `p_fused_*`, `p_cal_*` cho J3-thrC). `chunk` = doc mang chunk trong bundle (đúng bản dựng, hoà tại t_c theo quy tắc S31); `p4b_ge_tc` = P4B ≥ t_c (nhiều hơn `chunk` vì hoà và doc không có text). `docsel-J3-thrC` lấy chunk từ ZIP đã nộp (dựng trước khi sửa lỗi Pharmacity: doc Pharmacity không có chunk), không có cột C-max (NaN, rank 0).

| File | Doc | `chunk` | t_c |
|---|---:|---:|---:|
| `docsel-FUSE4-thrC.parquet` | 195.147 | 175.548 | 0,079782 |
| `docsel-FUSE4w-thrC.parquet` | 195.147 | 175.548 | 0,096025 |
| `docsel-FUSE4-free.parquet` | 229.062 | 175.624 | 0,203075 |
| `docsel-EF2-cut-p4b.parquet` | 193.879 | 175.548 | 0,087564 |
| `docsel-EF2-cut-avg.parquet` | 192.849 | 175.548 | 0,070560 |
| `docsel-LANG-thr-avg.parquet` | 190.323 | 175.548 | 0,063710 |
| `docsel-J3-thrC.parquet` | 195.147 | 175.148 | 0,079210 |

## Bản dựng (`out/runs/s36-fuse4/build-docread/`, validator S26 `ok`, 0 lỗi, không cắt body, dưới trần 102.600.000 B)

| Bản | Doc | Chunk | ZIP B | ZIP SHA256 | JSON SHA256 | Doc khác J3-thrC (đối xứng TB/query) | vi / zh |
|---|---:|---:|---:|---|---|---:|---|
| FUSE4-thrC-docread | 195.147 | 175.548 | 101.643.457 | `f2279849…55cc` | `d22d11a1…` | 14,85 | 82,0 / 18,0 |
| FUSE4w-thrC-docread (4B ×2) | 195.147 | 175.548 | 101.547.101 | `5c1c9e72…6a79` | `cbb658bc…` | 15,07 | 81,9 / 18,1 |
| FUSE4-free-docread | 229.062 | 175.624 | 102.471.046 | `6ab20eb4…129b` | `bb1b97b7…` | 49,15 | 81,5 / 18,5 |
| EF2-cut-p4b-docread | 193.879 | 175.548 | 101.444.896 | `a5e66840…d6` | `2b28f6b0…` | 11,97 | 82,4 / 17,6 |
| LANG-thr-avg-docread | 190.323 | 175.548 | 101.558.824 | `0b3144df…c5bb` | `47cbd26b…` | 4,33 | 83,6 / 16,4 |
| EF2-cut-avg-docread | 192.849 | 175.548 | 101.558.049 | `2f1af29e…9432` | `5d309f92…` | 15,72 | 83,0 / 17,0 |

So với mốc: FUSE4-thrC khác K4-thr-C-fx 25,73 doc/query; FUSE4w-thrC 24,62; FUSE4-free khác K4-free 27,22 (K4-free: vi 80,6 / zh 19,4). Tỉ lệ zh của các bản mới nằm trong 16,4–18,5%. Chạy lại bằng `scripts/s36/run_docread.sh`, `run_free.sh`.

### Quy tắc dựng

- **Nguồn**: RRF k=60 trên cosine_rank, judge_rank (4B), jina_rank, cmax_rank (thứ hạng trong pool của query). FUSE4w-thrC: trọng số 4B = 2, còn lại 1. Số doc/query chỉ phụ thuộc P4B nên không đổi theo nguồn.
- **thrC** (pool13 của K4-thr-C): K_q′ = kẹp(số doc P4B ≥ 0,0775; ⌈0,5K_q⌉; ⌊1,3K_q⌋), tổng 195.147 doc. Mỗi doc được chọn có P4B trong 175.548 doc cao nhất nhận **một chunk nguyên body** (t_c hiệu chỉnh từng bản về 175.548 = số chunk của K4-thr-C-fx).
  Lý do không giữ t_c cố định 0,07056: với thứ hạng hợp nhất, mức đó cho ZIP 102.725.779 B, vượt trần. t_c của hai bản: 0,0798 và 0,0960.
- **free** (K4-free, không K_q): pool = cosine top-211 của scored union, đào sâu tới mọi cặp đã chấm khi doc thứ 211 có P4B ≥ 0,0775 (525 query, pool tới 522); số doc = kẹp(#P4B ≥ 0,0775; 86; độ sâu pool). Chunk theo hằng số **`FUSE4_FREE_T_C` = 0,2030746281894702**, hiệu chỉnh MỘT LẦN (`Cal-FUSE4-free`, đích 175.548 chunk của FUSE4-thrC) rồi cố định; 261 cặp hoà tại t_c đều nhận chunk nên tổng là 175.624 (+76).
  t_c cao hơn thrC nhiều vì cùng ngân sách chunk phải chia cho 229.062 doc (34 nghìn doc hơn).
- Chunk của mọi bản `-docread` gồm cả doc Pharmacity (bản sửa lỗi builder S29); số chunk khác K4-thr-C: **175.548 so với 175.148** (+400: 337 chunk Pharmacity + 63 doc có P4B đúng bằng t_c).

## Độ phủ điểm và chấm bù

| Khung | Cặp | Thiếu Jina | Thiếu C-max |
|---|---:|---:|---:|
| thrC (pool13) | 281.924 | **0** | **0** |
| pool 211 (J4-K40) | 253.200 | 0 | 20.119 (398 query, pool13 < 211) |
| free (pool đã đào sâu) | 319.505 | 33.307 (268 query) | 53.347 (611 query) |
| trong đó đào sâu ngoài top-211 | 66.305 | 33.307 | 33.228 (229 query) |

- Cặp pool13 ngoài top-211 (48.886) đã được Jina chấm ở S35 nên khung thrC đủ điểm; chỉ khung free cần chấm bù.
- **Jina bù**: 33.307 cặp, 27 phần của 10 query, khối 10 xáo trộn (b10s0) như S35; query có < 30 cặp thiếu (46 query) được thêm 1.264 doc đã chấm làm neo (điểm neo bị bỏ). Ước tính chạy riêng ≈ 55 phút; thực tế chạy song song với C-max, S38 và dựng CPU nên lâu hơn.
- **C-max bù** (Qwen3-Reranker-0,6B, cùng cấu hình S32): 154.927 dòng = 53.347 cặp (18.858 cặp dùng dòng chunk của S33, 33.228 cặp chọn chunk mới đúng quy tắc S32, 1.380 cặp doc không có chunk chấm doc_text như fallback S32). Gate chọn chunk: 1.000 cặp S32 chọn lại, 0 lệch. ETA 1,8 giờ nếu chạy riêng (24 dòng/s đo ở S32; 27 dòng/s đo lúc chạy riêng một đoạn), chạy song song 10 dòng/s.
- **Kết quả: chấm đủ, 0 cặp phải xếp hạng cuối** ở cả hai nguồn (quy tắc dự phòng "xếp hạng cuối của pool" có trong code, `missing_*` trong sidecar = 0).
- **Nhất quán của Jina bù** (`extra/jina/consistency.json`, 8 query): chấm lại riêng một tập con doc đã có điểm S35, so với điểm S35 của chính các doc đó. Tập 120 doc: Spearman 0,83, dịch trung bình −0,0016, tỉ lệ độ lệch chuẩn 0,98; tập 40 doc: 0,83; tập 10 doc (một khối): 0,67. Không có lệch hệ thống nhưng Spearman thấp hơn mức 0,87 giữa hai lượt xáo trộn đủ pool, nên cặp bù mang nhiễu tương tự; không chạy thêm lượt khử nhiễu theo yêu cầu.

### Tương quan Spearman giữa bốn nguồn (toàn cục / TB theo query; đủ 4 điểm trên mọi cặp)

| Cặp | thrC pool13 | free pool |
|---|---|---|
| cosine ~ 4B | 0,442 / 0,364 | 0,458 / 0,370 |
| cosine ~ Jina | 0,411 / 0,392 | 0,425 / 0,380 |
| cosine ~ C-max | 0,404 / 0,321 | 0,424 / 0,330 |
| 4B ~ Jina | 0,698 / 0,653 | 0,698 / 0,651 |
| 4B ~ C-max | 0,678 / 0,584 | 0,679 / 0,590 |
| Jina ~ C-max | 0,662 / 0,587 | 0,664 / 0,588 |

## Hồi quy

- `Check-FUSE4-thrC-id` (RRF cosine + 4B, đường K4-thr-C-fx) và `Check-FUSE4-thrC-th` (đúng đường ngưỡng-175.548 của hai bản FUSE4): **JSON `c105b78d…499c` = K4-thr-C-fx**, ZIP 100.763.769 B (K4-thr-C-fx 100.763.753 B; 16 B là độ dài tên). Cả hai đạt.
- K4-thr-C gốc (`d771c4f9…`, builder cũ) đã được `--no-extra-chunks` của S31 tái tạo đúng ngày 22/10; S36 không chạy lại (bỏ để nhường CPU cho GPU).
- `Check-FUSE4-free-id` (khung free với RRF cosine + 4B, luật chunk cũ, kỳ vọng JSON K4-free `896ffcbb…`): lần 1 dừng sau 80 phút ở vòng cắt body do tranh CPU/RAM; lần 2 chạy sau cùng, kết quả ở mục "Trạng thái" cuối file.
- Test: 36 test S31 + S36 qua (`tests/test_s36_fuse4.py` 6 test); 54 test S31–S35 qua trước khi thêm S36.

## Vế doc: hiệu chỉnh xác suất, EF2-cut, LANG-thr (`python -m r2ai.s36.cal run`)

Khung J3-thrC (pool13, thứ tự RRF cosine + 4B + Jina). Quyết định của Vu (23/10): chuẩn hoá Jina bằng **khớp phân vị về P4B** (ánh xạ đơn điệu toàn cục trên pool13); Doc P tính **macro** (TB theo query); LANG-thr dùng P_fused = TB(P4B, Jina) và báo thêm bản P4B-only; dựng EF2-cut-p4b, EF2-cut-avg, LANG-thr-avg (LANG-thr-p4b = J3-thrC về mặt thuật toán, không dựng ZIP).

### 1. p_cal = sigmoid(a·logit(P_fused) + b), khớp Doc P của LB trên ba tập đã nộp

| P_fused | a | b | J4-K40 (0,376) | K4-thr-C (0,2346) | J3-thrC (0,2357) | sai số lớn nhất |
|---|---:|---:|---|---|---|---:|
| P4B | 0,5298 | −1,4097 | 0,3760 | 0,2347 | 0,2356 | **0,00015** |
| TB(P4B, Jina′) | 0,6110 | −1,4020 | 0,3760 | 0,2317 | 0,2384 | 0,0029 |

Ba phương trình, hai tham số (bình phương tối thiểu). Dạng micro (gộp mọi doc) cho 0,258 / 0,259 ở K4-thr-C / J3-thrC, tức không khớp: LB Doc P là macro, đúng như Vu chọn.
P4B đơn lẻ khớp tốt hơn TB(P4B, Jina): xác suất trộn kém chuẩn hơn, đúng nhận định của Vu.

### 2. G_q và hiệu chỉnh pool bị cắt

G_q = Σ p_cal trên pool13 (TB 45,2 doc) + đuôi ngoài pool (TB 11,8% tổng; tính từ độ suy giảm hình học của p_cal trên 60 doc cuối theo thứ tự cosine, đuôi không quá 50 lần trung bình 10 doc cuối). Với G này mô hình dự đoán **Doc F2 của J3-thrC 0,526 và recall 0,79, trong khi LB cho F2 0,4161 (recall ngầm ≈ 0,51)**: pool13 thiếu rất nhiều doc gold ngoài pool. Vì thế có thêm hệ số tỉ lệ toàn cục s trên G_q (`anchored`), chọn để dự đoán F2 của J3-thrC đúng 0,4161: **s = 1,490 (P4B), 1,532 (TB)**. Kiểm độc lập: dự đoán F2 của K4-thr-C là 0,4146 (P4B; LB 0,4139, lệch 0,0007) và 0,4039 (TB; lệch 0,010). Các bản dựng ZIP dùng `anchored`; `literal` (s = 1) báo bên dưới nhưng không dựng.

### 3. EF2-cut: chọn k tối đa hoá 5Σp/(4G+k), sàn 20, trần độ sâu pool13

| | P_fused | G | Tổng doc | k/query P10 / P50 / P90 | Doc khác J3-thrC | F2 dự đoán |
|---|---|---|---:|---|---:|---:|
| J3-thrC | – | – | 195.147 | 66 / 160,5 / 261 | 0 | 0,4161 (neo) |
| EF2-cut-p4b | P4B | anchored | **193.879** | **62 / 156 / 266** | 11,97 | 0,4206 |
| EF2-cut-avg | TB | anchored | **192.849** | **66 / 157 / 259** | 15,72 | 0,4208 (J3: 0,4161) |
| (literal) p4b | P4B | s = 1 | 174.497 | 51 / 138 / 253 | 20,95 | – |
| (literal) avg | TB | s = 1 | 170.729 | 55,9 / 135 / 238 | 24,31 | – |

Mức tăng F2 dự đoán chỉ khoảng +0,0045 so với J3-thrC, nhỏ so với sai số mô hình (K4-thr-C lệch 0,0007–0,010); đọc như giả thuyết cần LB kiểm chứng, không phải kỳ vọng chắc chắn.

### 4. LANG-thr: ngưỡng P4B riêng cho vi và zh

Precision biên ở ngưỡng t của một ngôn ngữ = TB p_cal của các cặp pool13 cùng ngôn ngữ có thứ hạng P4B trong ±5% (số cặp của ngôn ngữ đó) quanh t; hai ngưỡng chọn để precision biên bằng nhau và bằng c = F2/5 (nghiệm điểm bất động, F2 dự đoán của chính lựa chọn đó).

| P_fused | c | t_vi | t_zh | Tổng doc | k/query P10 / P50 / P90 | Doc khác J3-thrC | F2 dự đoán |
|---|---:|---:|---:|---:|---|---:|---:|
| P4B (suy biến) | 0,0829 | 0,1321 | 0,1303 | 180.048 | 61 / 141 / 252 | 12,58 | 0,4145 |
| **TB(P4B, Jina)** | 0,0832 | **0,0759** | **0,1859** | **190.323** | 64 / 153 / 259 | **4,33** | 0,4159 |

Với P4B-only hai ngưỡng gần như trùng (0,1321 / 0,1303, chênh do cửa sổ ±5%), đúng như Vu dự kiến. Với TB, Jina kéo p_cal của zh xuống nên ngưỡng zh cao gấp 2,4 lần ngưỡng vi. F2 dự đoán của LANG-thr-avg (0,4159) **không cao hơn J3-thrC** (0,4161): theo mô hình, bản này chỉ đổi cách chia doc giữa hai ngôn ngữ, không thêm F2. Cách cũ (đặt cả hai ngưỡng bằng 0,0775 như J3-thrC) vẫn là J3-thrC.

## Điểm cần Vu biết

1. **t_c từng bản**: ba bản thrC và ba bản EF2/LANG dùng t_c hiệu chỉnh về 175.548 chunk (không phải 0,07056 cố định); bản FUSE4-free dùng hằng số 0,2031. Chỉ ảnh hưởng chunk, vế doc không đổi.
2. **Hệ số s của G_q** là bước thêm của tôi (lấy từ Doc F2 mốc 0,4161). Nếu Vu muốn đúng nghĩa đen G = Σ p_cal + đuôi (`literal`), EF2-cut sẽ cắt mạnh hơn (174–171 nghìn doc), nhưng mô hình khi đó dự đoán sai F2 của J3-thrC (0,526).
3. **Chấm bù Jina là nhiễu cùng cỡ lượt xáo trộn**, đã nêu ở trên. Doc FUSE4-free ngoài top-211 mang điểm Jina của lượt bù.
4. ZIP FUSE4-free-docread 102.471.046 B, sát trần 102.600.000 B (còn 128.954 B).
5. Các bản S35 cũ (PART-thrC, J3-thrC, JINA-K40) dựng trước bản sửa lỗi Pharmacity, không dựng lại.

## Lệnh

```bash
python -m r2ai.s36.extra coverage|jina-prepare|jina-check|cmax-prepare|jina-reduce|cmax-reduce   # chấm bù (scripts/s36/run_jina_extra.sh, run_cmax_extra.sh)
python -m r2ai.s36.cal run && python -m r2ai.s36.cal docsel-j3                                   # hiệu chỉnh, EF2-cut, LANG-thr, docsel J3-thrC
python -m r2ai.s36.build --variant FUSE4-thrC --variant FUSE4w-thrC --out out/runs/s36-fuse4/build-docread
python -m r2ai.s36.build --variant EF2-cut-p4b --variant LANG-thr-avg --variant EF2-cut-avg --out out/runs/s36-fuse4/build-docread
python -m r2ai.s36.build --variant FUSE4-free --out out/runs/s36-fuse4/build-docread             # t_c = FUSE4_FREE_T_C
python -m r2ai.s36.report bundles|sources
```

## Trạng thái cuối

Xem bản ghi `out/runs/s36-fuse4/free.log` cho `Check-FUSE4-free-id`.
