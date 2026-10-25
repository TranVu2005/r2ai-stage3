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

### Tham số toàn cục và giới hạn rò K_q (24/10)

- **(a, b)** = (0,5298075; −1,4096976) khớp trên ba Doc P tổng hợp của LB public (J4-K40 0,376, K4-thr-C 0,2346, J3-thrC 0,2357); **s_free** = 1,7405 khớp trên một Doc F2 tổng hợp của LB public (FUSE4-free 0,4104). Cả bốn là tham số toàn cục (một giá trị cho mọi query), không dùng nhãn, điểm LB theo query hay dữ liệu theo query nào. Bộ query mới dùng lại đúng các giá trị này.
- **Rò K_q qua độ sâu pool**: pool top-211 và ngưỡng T_FREE là hằng số chung, nhưng 525 query có pool sâu hơn 211 (đào sâu khi doc thứ 211 có P4B ≥ T_FREE) và độ sâu đã chấm của chúng đến từ pool-extra, chấm theo trần 2,0·K_q của S31. 454 trong 525 query vẫn còn doc có P4B ≥ T_FREE ở đáy pool đã chấm, nên trần EF2 của các query này (tới 522 doc) vẫn mang dấu K_q cũ. Bản `EF2-free-d211` cắt pool ở cosine top-211 cho mọi query để bỏ phần rò này (giữ nguyên (a, b), s_free, t_c, 4 nguồn RRF).
- t_c: EF2-cut-p4b chọn chunk theo top-N P4B với N = 175.548 (tổng chunk K4-thr-C-fx, t_c 0,08756); EF2-free dùng hằng số 0,20307 (hiệu chỉnh một lần trên docs FUSE4-free về cùng tổng 175.548). Không bản nào dùng K_q theo từng query; cả hai kế thừa tổng chunk của pipeline K_q cũ (n_c × 1,4).

### EF2-FUSE4 khác EF2-cut-p4b ở đâu

Cùng khung pool13, cùng (a, b), s = 1,490 và cùng G_q (G chỉ phụ thuộc P4B của pool13). Khác duy nhất: thứ tự RRF thêm nguồn C-max (4 nguồn thay 3), nên top-k đổi và k đổi theo (|Δk| TB 8,55 doc/query, 75 query cùng k, tổng 195.049 so với 193.879 doc). Có dùng K_q: pool13 = round(1,3·K_q), trần EF2 và đuôi G đều lấy từ độ sâu pool13. Chưa dựng ZIP.

### EF2-free-d211 (S35, 24/10, chưa compose)

`python -m r2ai.s36.cal ef2-free-d211` rồi `scripts/s36/run_ef2_free.sh EF2-free-d211`: pool cắt ở cosine top-211 cho 1.200 query (253.200 cặp), RRF 4 nguồn hạng tính lại trong pool 211, G_q như EF2-free (tổng p_cal cosine top-211 + đuôi). 176.213 doc, k/query P10/P50/P90 = 73,9 / 155 / 206 (min 20, max 211, 42 query ở trần); mô hình dự đoán Doc F2 0,3790 (P 0,2408 / R 0,4565; thấp hơn EF2-free 0,4153 vì mô hình vẫn tính G đầy đủ trong khi pool bị cắt). Trên 525 query sâu: EF2-free 139.174 doc, d211 94.562 doc, mất 45.304 doc (TB 86,3/query, P10/P50/P90 8,4/56/199; 43.876 doc nằm ngoài cosine top-211), thêm 692 doc. docread ZIP 79.678.698 B SHA `bce908d8…33d9`, validator ok; docsel SHA `8ce87a00…febf`. Chưa chạy compose theo yêu cầu của Vu (chờ chốt quy tắc chunk).

### Phân rã chunk EF2-free-t2002 so với mốc t1907 (24/10, offline)

LB: mốc Final 0,3914 (Doc 0,4209, Chunk 0,3619: P 0,2213 / R 0,4769); EF2-free Final 0,3834 (Doc 0,4184, P 0,226 / R 0,6076; Chunk 0,3484: P 0,2236 / R 0,4554). Số liệu: `cal/chunkdiff.json`, `chunkdiff2.json`, `chunkrule-proxy.json` (`scripts/s36/chunkdiff*.py`, `chunkrule_proxy.py`).

- Carrier chunk đầu: chung 148.614; chỉ-free 25.921 (p_cal TB 0,274, P4B TB 0,610); chỉ-mốc 26.934 (p_cal TB 0,120, P4B TB 0,232). Chỉ-mốc gồm 18.745 doc có trong docs EF2-free nhưng P4B < t_c 0,2031 (p_cal TB 0,087) và 8.189 doc EF2-free không chọn (p_cal TB 0,198). Tổng p_cal của carrier: free 53.725 so với mốc 49.860 (+7,8%); chỉ số recall thay thế (macro Σp_cal/G_q) free/mốc = 0,9936 (LB chunk recall 0,955). Carrier dồn vào ít query hơn: 100 query đầu giữ 22,3% (mốc 16,3%); carrier/query P10/P50/P90 38/118/329 so với 46/137,5/258.
- Chunk thứ hai: 9.561 so với 12.637 (−3.076, −24,3%). Với T = 2002, 2.949 carrier có body trong (1.907; 2.002] mất cửa sổ (nếu T = 1.907 thì free có 12.510); trong carrier chung, 2.526 doc có cửa sổ ở mốc mà không có ở free. Chỉ số thay thế của cửa sổ free/mốc = 0,750.
- Byte: ZIP nền free 101.717.111 B so với mốc 101.444.896 B (+272.215 B); ngân sách cho cửa sổ 882.889 so với 1.155.104 B (−23,6%), mỗi cửa sổ ≈ 92–94 B, mỗi carrier chunk đầu ≈ 580 B. relevant_docs: +192.270 B thô, +78.100 B nén (29% phần tăng); chunk_text: thô −7,4 MB nhưng nén +194 kB (71%): carrier của free nén kém hơn. JSON thô free 945.572.363 B so với mốc 964.570.736 B, ZIP cuối bằng nhau (102.594.467 / 102.593.008 B).
- Giải thích: mô hình p_cal ở mức doc dự đoán carrier của free tốt hơn mốc, nên không giải thích được chunk recall −4,5%; mất cửa sổ (−24%) giải thích một phần chưa biết. Bằng chứng S38 (nguyên body 0,3594 → thêm cửa sổ ~12 nghìn doc 0,3619) cho thấy cửa sổ chỉ cộng cỡ +0,0025 chunk F2 nên khó giải thích hết −0,0135. Phần dư chưa giải thích được offline vì không có nhãn chunk.


### Quy tắc chunk kiểm bằng LB: CR-fair và CR-swap (24/10, chưa nộp)

Cả hai dùng đúng docsel EF2-free (`9e15a8bd…1185`, docs không đổi), compose S38 `fe7f82e` + `--carriers`, thin theo nội dung, validator S26 `--max-chunks-per-doc 2`. Tập carrier do `python -m r2ai.s36.carriers` (`scripts/s36/run_cr.sh fair swap` dựng ZIP).

- **CR-fair**: mỗi query mang chunk cho top ⌈ρ·|D_q|⌉ doc theo P4B giảm dần (hoà theo rank docsel); ρ = 0,7876448 là hằng số toàn cục, giải để tổng carrier = 174.535.
- **CR-swap**: giữ ngưỡng toàn cục hiện tại, bỏ 4.400 carrier P4B thấp nhất (ngưỡng hiệu dụng 0,2031 → 0,2254), còn 170.135.

| | EF2-free t2002 (đã nộp) | CR-fair | CR-swap |
|---|---:|---:|---:|
| ZIP | 102.594.467 B, `f4b6530d…063f` | `EF2-free-CRfair-t1728.zip` 102.597.138 B, SHA `4a7c4965…8aa4` | `EF2-free-CRswap-t1536.zip` 102.492.718 B, SHA `89223b24…786c` |
| T (đúng tới 1 token) | 2002 | **1728** (1727 vượt trần) | **1536** (sàn: toàn bộ cửa sổ, còn dư 107.282 B) |
| carrier (chunk đầu) | 174.535 | 173.898 (637 cặp không có nguồn chunk bị bỏ) | 170.135 |
| cửa sổ 768 | 9.561 | 20.890 | 35.590 |
| chunk tổng | 184.096 | 194.788 | 205.725 |
| carrier/query P10 / P50 / P90 | 38 / 118 / 329 | 58 / 127 / 286 | 36 / 113 / 324 |
| tỷ trọng 100 query đầu | 22,3% | 18,8% | 22,7% |
| thin | 25 carrier | 18 carrier | 22 carrier |
| validator S26 | ok | ok, 0 lỗi | ok, 0 lỗi |

Pin: carriers CR-fair `e80c8a28…a2bc`, CR-swap `3d6c6008…1df8` (`out/runs/s36-fuse4/carriers/`); JSON CR-fair `43c34f16…16fe`, CR-swap `9c773cc7…69d4`. Nguồn chunk của carrier mới đúng quy tắc builder S29. Mô hình không dự đoán được Chunk F2 của hai bản này (xem phân rã trên); kết quả chỉ có từ LB.

### Áp EF2-free lên dữ liệu S37 (code + test, chưa chạy)

`python -m r2ai.s36.s37 apply [--config configs/s36-s37-apply.json] [--jina scores-all.parquet] [--scores-4b …] [--cmax …]` (mặc định lấy bố cục `out/runs/s37`): union S36 + cặp mới S37 (điểm 4B Kaggle theo `pair_id`), pool cosine top-211 + đào sâu, RRF 4 nguồn, cắt EF2 với (a, b) và s_free giữ nguyên, carrier theo quy tắc trong cấu hình (`threshold` / `fair` / `swap`), ghi `docsel-<tên>.parquet`, `keep-<tên>.json`, `pcal-`, `carriers-` và `<tên>.apply.json` (SHA256 mọi đầu vào/ra). Dừng khi một nguồn thiếu điểm cho cặp nào của pool và ghi cặp thiếu ra `missing-<tên>-<nguồn>.parquet`; không xếp cuối. Jina chấm lại theo query: query S37 chấm lại chỉ tin điểm S37, query không chấm lại giữ điểm S35 + bù S36. ZIP: `compose run --docsel docsel-<tên>.parquet --rule full+w768 [--carriers …]` (không cần ZIP docread; xem `docs/s38-chunk-metric-v2.md`).

`python -m r2ai.s36.s37 jina-missing` → `out/runs/s36-fuse4/cal/jina-missing-pairs.parquet` (SHA `42f7fc9e…7aa6`): 21.311 cặp điểm bù S36 của 174 query mà pool Jina S37 (975 query chấm lại) không chứa (P10/P50/P90 = 1 / 150,5 / 167 cặp mỗi query). Vu chuyển cho S37 thêm vào pool chấm lại; 11.996 cặp bù còn lại nằm ở query không chấm lại. Test `tests/test_s36_s37.py`.

### Kết quả LB của quy tắc chunk và bản 120ask (24–25/10)

CR-fair (docsel EF2-free): Chunk 0,3537, hơn EF2-free t2002 0,0053, giữ. CR-swap: Chunk 0,3493, bỏ (cửa sổ thêm không đáng giá hơn carrier). CR-fair với ρ toàn cục (thứ tự P4B giảm dần, hoà theo rank docsel) là quy tắc chunk mặc định (`DEFAULT_RHO` = 0,7876448 trong `r2ai.s36.carriers`; `--target` giải ρ theo tổng carrier). Mốc CR-fair Final 0,3861.

**120A-free-v0** (trước là 120A-p4b): máy EF2-free (pool cosine top-211 thích nghi, `s_free`, 4 nguồn RRF) áp lên dữ liệu S37 bằng `python -m r2ai.s36.s37 apply --config configs/s36-s37-120A-p4b.json` (manifest S37 SHA `6b9e090e…1ce3`: 4B 100%, C-max 100%, Jina 36% đang chờ; cặp thiếu Jina hoặc C-max xếp cuối nguồn đó), CR-fair giải ρ = 0,76444 cho tổng 175.548 carrier (bằng mốc t1907), compose `full+w768`, thin theo nội dung. Không phải EF2-cut-p4b thuần: trùng 91% với docsel EF2-cut-p4b. **LB: Final 0,4033 (Doc 0,4344 / Chunk 0,3721), bản tốt nhất, mốc mới.**

| | 120A-free-v0 |
|---|---:|
| doc | 228.844 (120ask 14.255, 11.663 doc khác nhau, 6,2%; 811/1.200 query có ít nhất một, TB 11,9/query) |
| Doc F2 mô hình | 0,4225 (LB 0,4344) |
| T / cửa sổ | 1536 (sàn, dư 1,39 MB) / 34.212 |
| carrier giao | 174.962 (586 cặp không có nguồn chunk bị bỏ: 441 doc canonical, chủ yếu ask.39.net 385 và zh 470, không có doc 120ask), P10/P50/P90 = 58/127/286, top-100 19,2% |
| thin | 14 carrier (13 doc) |
| ZIP | `120A-free-v0-full+w768-t1536.zip` 101.213.371 B, SHA256 `55127059b31540b032fbed375a9252040d0de977bbe6ea4a8088d5a4ce357dc8` |
| bị đẩy ra so với EF2-cut-p4b | 17.091 doc (vi 12.150, zh 4.907; suckhoedoisong.vn 1.997, nhathuoclongchau.com.vn 1.808, 120ask.com 1.525, thanhnien.vn 1.504); vào 52.056 doc |

Tên file apply trong `s37/` vẫn là `120A-p4b` (SHA ghim trong `120A-p4b.apply.json`); `s37/120A-free-v0.names.json` ghi ánh xạ. Validator: compose cũ chỉ truyền lô Pharmacity làm nguồn verbatim phụ nên báo 10.750 `chunk_no_canonical_doc` (chunk của doc 120ask). Đã sửa (`registry.validator_extra`, `compose.run_validator`, lệnh `compose validate` trên nhánh `feat/s38-compose-registry`, commit `6f0540c`): bảy segment 120ask vào làm nguồn verbatim; `compose validate --name 120A-free-v0-full+w768 --threshold 1536` cho ok, 0 lỗi, `canonical_substring` 120.408.

**p4b-CRfair** (EF2-cut-p4b + CR-fair ρ = 0,7876): chỉ 152.687 carrier (mốc t1907 có 175.548), ZIP 91.413.363 B (T 1536, 32.113 cửa sổ, P10/P50/P90 = 49/123/210), không so được với 0,3914; thay bằng **p4b-CRfair175**: ρ = 0,9022 cho 175.549 carrier (174.771 có nguồn chunk, 778 cặp bị bỏ), compose `full+w768` trên docsel EF2-cut-p4b (SHA `db94540a…5454`). **T = 1752** (1751 vượt trần 102.600.000 B: 102.606.854 B), ZIP `out/runs/s38-chunk-metric-v2/compose/p4b-CRfair175-t1752.zip` 102.596.795 B, SHA256 `27048e4d7d796a01f45e3e8adfb475f84125eae223d33ffff9a19a9e8bf9a4ab`, JSON SHA `11aa43bd…6add`. Doc 193.879 (như mốc t1907), chunk 194.539 = 174.771 carrier + 19.768 cửa sổ 768 (mốc t1907: 175.548 + 12.637, T 1907); carrier mỗi query P10/P50/P90 = 55,9 / 141 / 240, top-100 query 15,1%; Doc F2 mô hình 0,4206 (docs không đổi, LB mốc 0,4209); validator S26 `--max-chunks-per-doc 2` ok, 0 lỗi; thin 18 doc. Dùng để biết khung p4b + CR-fair có hơn khung free + CR-fair (Vu nộp nếu còn lượt); chưa nộp.

### d211: phân tích (a)(b)(c) và quyết định dừng (25/10)

Mô hình: d211 0,3790, EF2-free 0,4153, thiệt Doc 0,0363 (Final ít nhất −0,018 chỉ riêng Doc). Dừng hẳn d211; compose d211-CRfair bị dừng, không có ZIP. Số liệu: `cal/d211-analysis.json`, mã `scripts/s36/d211_analysis.py`, chạy trên `cal.Frame`/`Calibrated` không đổi.

(a) Số query có doc hạng 211 với P4B ≥ 0,0775 / 0,1 / 0,15 / 0,203: 628 / 600 / 538 / 478. d211 mất 45.304 doc của EF2-free trên 521 query (vi 35.868 = 19,9%, zh 9.436 = 23,1%; 43.876 nằm sau cosine 211, 1.428 trong top-211 do RRF xếp lại; chỉ lấy thêm 692 doc), P4B TB 0,568, kỳ vọng ~11.800 doc vàng.

(b) Độ sâu cố định D cho mọi query, cận dưới từ điểm đã có (phần chưa chấm tính bằng 0): D=211 0,3790; D=300 0,4092; D=400 0,4201; toàn bộ độ sâu đã chấm (≤522, phụ thuộc K_q) 0,4238; EF2-free thích nghi 0,4153. Cặp cần chấm thêm (corpus không 120ask): D=300 53.233 4B / 54.196 Jina / 54.162 C-max; D=400 150.866 / 153.126 / 153.078. Tính lại trên corpus CÓ 120ask: xem mục D=300 dưới.

(c) Sai số mô hình dưới G của d211 (`s_free` × tổng p_cal cosine top-211 + đuôi): FUSE4-free 0 (neo), EF2-free −0,0031 (LB 0,4184), EF2-cut-p4b −0,0231 (LB 0,4209), J3-thrC −0,0218, K4-thr-C −0,0210. Mô hình xếp EF2-cut-p4b thấp hơn EF2-free 0,017 còn LB cao hơn 0,0025: với bộ doc từ pool ngắn kiểu pool13 mô hình thấp hơn LB khoảng 0,022. Nếu d211 chịu cùng độ lệch thì Doc d211 khoảng 0,401 (thiệt ~0,017, Final thiệt ≥ 0,009 trước khi tính chunk); chunk còn thiệt thêm vì d211 chỉ 139.405 carrier. Chưa chứng minh được thiệt ≤ 0,01 nên không nộp.

### Chuẩn bị D=300 trên corpus có 120ask (25/10, chỉ CPU, chưa chấm điểm)

`python -m r2ai.s36.d300 count --depth 211 300 400` và `export --depth 300` (mã `src/r2ai/s36/d300.py`, test `tests/test_s36_d300.py`). Xếp hạng gộp: cosine top-400 canonical + Pharmacity của `out/runs/s37/pool.npz` hợp với cosine mọi doc 120ask (vector segment), lấy top-D (canonical trước khi hoà). Cặp coi là đã chấm khi có điểm 4B (union S36 + Kaggle S37), C-max (S32 + bù S36 + S37) hoặc Jina.

| D | cặp cần 4B | 120ask / canonical | query | giờ 2×T4 | cặp cần C-max | cặp thiếu Jina |
|---|---:|---|---:|---:|---:|---:|
| 211 | 941 | 0 / 941 | 18 | 0,05 | 991 | 32.499 |
| 300 | 33.233 | 2.169 / 31.064 | 629 | 1,59 | 34.023 | 79.029 (120ask 47.156) |
| 400 | 107.975 | 10.772 / 97.203 | 995 | 5,17 | 109.952 | 164.384 |

D=211 lẽ ra bằng 0; 941 cặp canonical còn lại (18 query) do xếp hạng top-400 mới đổi chỗ một số doc Pharmacity so với union S31. Phần thêm không chấm Jina (xếp cuối nguồn Jina). Gói Kaggle `out/runs/s36-fuse4/d300/kaggle/` (bố cục S31/S37, `judge_score.py` SHA `688948a7…a7186`): 33.233 cặp, `pair_id` 5.000.000–5.033.232, vi 24.614 / zh 8.619, doc_tokens TB 412 (2.586 hàng doc_text rỗng của 588 doc canonical, như S31 pool-extra), SHA `extra_pairs` `49cbf440…5d2b`, shard `pool-93002.parquet` `a3750fb3…18b1`; `pairs_d300.parquet` `e5417d9f…4480` (để `apply` về sau), `cmax_pairs.parquet` (34.023 cặp) `f8e90f3d…1d2b`. Chưa chạy chấm điểm; Vu quyết sau khi kiểm quota Kaggle. **Gói `d300` này KHÔNG chạy** (Vu crawl thêm 120ask trong đêm 25/10): khi S37 có `pairs_new_final_v2`, dựng MỘT subset `final-4b` bằng `python -m r2ai.s36.d300 export-final --depth 300` (top-300 của corpus mới chưa có 4B ∪ cặp chưa chấm của `pairs_new_final_v2`, loại trùng theo (query, doc); `pair_id` của registry thắng, cặp mới giữ id đã xuất ở `pairs_d300`/`pairs_final` hoặc lấy từ 5.000.000 trở lên; ra `d300/final-4b/` + `pairs_final.parquet`); báo số cặp và giờ Kaggle. Điểm Kaggle nhập bằng `python -m r2ai.s36.d300 import4b --tag final-4b --scores … --report …` (lưu `gpu/qwen3-reranker-4b/scores-extra.parquet`, `apply` tự lấy).

`apply` nhận nhiều registry: `"pairs_new": [".../pairs_new_final_v2.parquet", ".../d300/pairs_final.parquet"]` (`s37.read_pairs`: một `pair_id` chỉ là một cặp, cặp (query, doc) trùng tính một lần, bản có điểm 4B trước) và `"pool": "fixed", "depth": 300` (pool mỗi query = cosine top-300 của union gộp, không đào sâu thích ứng, đối chiếu với xếp hạng cosine của corpus và ghi `pool.check_vs_corpus`); G_q vẫn là định nghĩa top-211. Cấu hình: `configs/s36-s37-120A-free-v2.json` (khung free, pool cố định D=300), `configs/s36-s37-120A-p4b-v2.json` (khung EF2-cut-p4b, xem dưới), `configs/s36-s37-120A-free-v1.json` cho 120A-free-v1 (120A-free-v0 với Jina đầy đủ cho 120ask; bản trung gian, KHÔNG nộp; bản chính là v2). `registry_index` của compose đã đọc `pairs_new_final_v2` và `pairs_final`.

## Lệnh

```bash
python -m r2ai.s36.extra coverage|jina-prepare|jina-check|cmax-prepare|jina-reduce|cmax-reduce   # chấm bù (scripts/s36/run_jina_extra.sh, run_cmax_extra.sh)
python -m r2ai.s36.cal run && python -m r2ai.s36.cal docsel-j3                                   # hiệu chỉnh, EF2-cut, LANG-thr, docsel J3-thrC
python -m r2ai.s36.build --variant FUSE4-thrC --variant FUSE4w-thrC --out out/runs/s36-fuse4/build-docread
python -m r2ai.s36.build --variant EF2-cut-p4b --variant LANG-thr-avg --variant EF2-cut-avg --out out/runs/s36-fuse4/build-docread
python -m r2ai.s36.build --variant FUSE4-free --out out/runs/s36-fuse4/build-docread             # t_c = FUSE4_FREE_T_C
python -m r2ai.s36.report bundles|sources
python -m r2ai.s36.cal ef2-free && bash scripts/s36/run_ef2_free.sh                              # EF2-free: chọn k, docread + docsel + validator
(cd ../s38-chunk-metric-v2 && bash scripts/s38/compose.sh EF2-free full+w768)               # chunk full+w768
```

## Trạng thái cuối

`Check-FUSE4-free-id` (lượt chạy thứ hai, xong 23/10 17:32) qua: JSON `896ffcbb6b6aaa3c3dcd319b7f2dd56d94186c6857df0b3bce0688df51f6bbba` bằng JSON K4-free, `regression_pass` true, ZIP 102.321.554 B. Lượt đầu bị dừng sau 80 phút ở vòng cắt body.

LB 23/10 (Vu báo): EF2-cut-p4b Doc 0,4209 (mô hình dự đoán 0,4206), FUSE4-thrC Doc 0,4178, FUSE4-free Doc 0,4104.

## EF2-free (S35, 24/10): EF2 trong khung tổng quát, không dùng K_q

Mục tiêu: so với mốc EF2-cut-p4b-full+w768-t1907 (ZIP SHA `af427ae3…1264`, LB Final 0,3914 = Doc 0,4209 / Chunk 0,3619) chỉ khác khung free. Ngưỡng Vu chọn EF2-free làm bản cuối: Final ≥ 0,3814. Dữ liệu chỉ là pool và điểm hiện tại (canonical + Pharmacity, Jina S35/S36 cũ, C-max S32/S36); không dùng 120ask, không dùng Jina chấm lại của S37.

Chỗ EF2-cut-p4b dùng K_q hoặc thông tin pipeline cũ, và thay thế (Vu duyệt 24/10):

| Chỗ | Thay thế toàn cục |
|---|---|
| pool13 = round(1,3·K_q) | cosine top-211 của union đã chấm, đào sâu thích ứng nếu doc thứ 211 có P4B ≥ T_FREE 0,0775 (hằng số chung, đúng khung FUSE4-free); trần EF2 = độ sâu pool đã chấm |
| G_q = Σ p_cal trên pool13 + đuôi sau độ sâu pool13 | G_q = s_free · (Σ p_cal trên cosine top-211 + đuôi hình học sau 211), cùng độ sâu 211 cho mọi query, không phụ thuộc phần đào sâu |
| s = 1,490 neo vào Doc F2 LB của J3-thrC (pool13) | s_free = 1,7405 neo một lần vào Doc F2 LB 0,4104 của FUSE4-free; (a, b) giữ nguyên |
| (a, b) khớp 3 Doc P tổng hợp của LB | giữ (a, b) = (0,5298075; −1,4096976); dữ liệu fit chỉ là ba Doc P tổng hợp (J4-K40 0,376, K4-thr-C 0,2346, J3-thrC 0,2357) và một Doc F2 tổng hợp cho s_free; không dùng nhãn hay điểm LB theo query |
| t_c đặt theo 175.548 chunk của pipeline K_q | hằng số `FUSE4_FREE_T_C` 0,2030746 (đã ghi một lần cho khung free), chunk nguyên body, không cắt |
| nguồn RRF | 4 nguồn (cosine, 4B, Jina, C-max) như FUSE4-free; EF2-cut-p4b dùng 3 nguồn J3 |
| sàn, F_FREE = 86 | không dùng; sàn EF2 cố định 20 |

Tồn dư K_q: 525 query có pool sâu hơn 211; độ sâu đã chấm của chúng (pool-extra) ban đầu theo trần 2,0·K_q. 454 query vẫn còn doc có P4B ≥ T_FREE ở đáy pool đã chấm.

Kết quả (`cal/ef2-free.json`, `python -m r2ai.s36.cal ef2-free`):

| | EF2-free | EF2-cut-p4b |
|---|---:|---:|
| doc tổng | 220.825 | 193.879 |
| k/query P10 / P50 / P90 | 74 / 161 / 363 (min 20, max 485) | 62 / 156 / 266 (max 332) |
| mô hình dự đoán Doc F2 | 0,4153 | 0,4206 (LB 0,4209) |
| mô hình dự đoán P / R | 0,2334 / 0,5295 | 0,2413 / 0,5299 |

Độ sâu pool P10/P50/P90 = 211/211/428 (max 522). G trung bình 79,5, đuôi 12,4% (P90 17,5%). Kiểm mô hình dưới G free: FUSE4-free 0,4104 (neo), K4-free 0,4095. Bốn query ở sàn, hai query ở trần. Khác EF2-cut-p4b: 32,43 doc chỉ có ở EF2-free, 9,98 doc chỉ có ở EF2-cut-p4b mỗi query; 1 query trùng hẳn. Khác FUSE4-free: 15,87 doc/query.

Bundle (docsel `docsel/docsel-EF2-free.parquet` SHA `9e15a8bd…1185`, ZIP docread 101.717.111 B SHA `c1c1605f…11c1`, validator ok), rồi compose S38 `full+w768` (nền `fe7f82e`, trần BTC − reserve, thin theo nội dung, THIN_JUNK_SHARE 0,5, T tự tìm): T = 2002, đúng tới một token (T = 2001 ZIP 102.602.368 B vượt trần 102.600.000 B). ZIP cuối `out/runs/s38-chunk-metric-v2/compose/EF2-free-full+w768-t2002.zip`, 102.594.467 B, SHA256 `f4b6530d7865f70d6822c4256ebd920ac733900a4f87fca200680a775c79063f`, JSON SHA `9c1bf23e…983e8d`. Validator S26 `--max-chunks-per-doc 2`: ok, 0 lỗi, 1.200 id, chunk thuộc relevant_docs, ZIP một file ở gốc.

So với mốc EF2-cut-p4b-full+w768-t1907 (`cal/ef2free-vs-ref.json`):

| | EF2-free t2002 | mốc t1907 |
|---|---:|---:|
| doc | 220.825 | 193.879 |
| doc/query TB / trung vị / P10–P90 | 184,02 / 161 / 74–363 | 161,57 / 156 / 62–266 |
| chunk | 184.096 | 188.185 |
| chunk carrier (chunk đầu) | 174.535 | 175.548 |
| chunk thứ hai (w768) | 9.561 | 12.637 |
| T | 2002 | 1907 |

Trùng doc: 82,4% doc của EF2-free nằm trong mốc, 93,8% doc của mốc nằm trong EF2-free. Carrier chung 148.614.

Thin: 25 carrier bị thin (21 doc, cả 21 là baophutho.vn), 25 chunk chuyển sang doc kế tiếp, t_c trước/sau 0,20307 / 0,20181. Dưới ngưỡng 50 carrier nên không dừng báo trước khi đóng gói.

Giới hạn: mô hình dự đoán EF2-free 0,4153 thấp hơn EF2-cut-p4b 0,4206 vì s_free neo vào FUSE4-free (LB 0,4104 so với 0,4161 của J3-thrC). Doc F2 thực của EF2-free chỉ biết sau khi nộp.

EF2-FUSE4 (pool13, 4 nguồn): chọn k xong (`cal/keep-EF2-FUSE4.json`, 195.049 doc, dự đoán 0,4206) nhưng chưa dựng ZIP; ưu tiên EF2-free.

## Hai bản v2 song song (25/10)

LB p4b-CRfair175 = 0,3918 (Doc 0,4209 / Chunk 0,3628): cùng CR-fair, khung p4b hơn khung free 0,0057 (CR-fair 0,3861); CR-fair ở khung p4b gần như không đổi so với EF2-cut-p4b-full+w768-t1907 (0,3914). Hai bản dựng khi có dữ liệu 120ask mới và Jina:

* **120A-free-v2**: khung free, pool cố định D=300 + 120ask mới, CR-fair (tổng carrier 175.548), full+w768 (T tự tìm). Config `configs/s36-s37-120A-free-v2.json`.
* **120A-p4b-v2**: khung EF2-cut-p4b: mỗi query lấy pool13 cộng các doc 120ask mới (đã có 4B), xếp hạng RRF cosine/4B/Jina (`J3_KEYS`, không C-max), G_q = s_thrC x (tổng p_cal trên pool + đuôi), cắt EF2, phân bổ carrier CR-fair với tổng 175.548, full+w768. Config `configs/s36-s37-120A-p4b-v2.json` (`"pool": "pool13"`, chỉ `pairs_new_final_v2`: các cặp độ sâu D=300 của doc canonical không thuộc khung này). Mã: `s37.pool13_plus`.

Cặp 4B cần trong gói Kaggle `final-4b`: khung free cần (cặp độ sâu D=300 chưa có điểm) + (cặp registry nằm trong top-300); khung p4b cần mọi cặp registry chưa có điểm. `export-final` ghi cả hai số (`pairs_needed_free_frame`, `pairs_needed_p4b_frame`, `eta_hours_*`) vào `manifest.json`; gói là hợp của hai (loại trùng), nên chạy gói một lần là đủ cho cả hai khung.

## 120A-p4b-v0 (25/10): khung EF2-cut-p4b thuần trên corpus có 120ask v0

Vu: LB p4b-CRfair175 0,3918 hơn khung free 0,0057, nên dựng khung p4b thuần trên dữ liệu 120ask v0, nộp trước 07:00. `apply` có `"pool": "cosine13"` (`s37.cosine13_pools`, config `configs/s36-s37-120A-p4b-v0.json`): mỗi query lấy top-n cosine của union gộp, n = kích thước pool13 cũ (round(1,3 K_q), kiểm: pool13 cũ đúng bằng top-n cosine của union S36, nên doc 120ask đẩy ra các doc cosine thấp nhất); nguồn RRF cosine / 4B / C-max / Jina (cặp thiếu Jina xếp cuối nguồn Jina), cùng a, b và s_thrC, cắt EF2 theo P4B, G_q = s_thrC x (tổng p_cal trên pool + đuôi). Không dùng máy EF2-free. Chunk: CR-fair tổng carrier 175.548 (ρ 0,9038), full+w768 (`scripts/s36/run_120a_p4b_v0.sh`).

Kết quả docsel: 281.924 doc trong pool (36.160 doc cũ bị đẩy ra ở 958 query; 36.160 cặp mới thiếu Jina), 193.547 doc được chọn (P10/P50/P90 = 60/159/267), trong đó 12.653 doc 120ask (10.471 doc khác nhau, TB 10,54 mỗi query, 784 query có ≥1), Doc F2 dự đoán 0,4191 (mô hình, G_q kiểu thrC).

Compose (25/10, 02:30 đến 03:12): `120A-p4b-v0-full+w768-t1536.zip`, 100.664.783 B (trần 102.600.000), SHA256 `dd8d4d76cfe4bf7cc9664ce9db984705633e0c537d916f9aafc790d821463a22`. T = 1536 (`FULL_MIN_T`, sàn của luật; ZIP còn dư 1,9 MB), 35.220 cửa sổ 768 token, 174.894 carrier sau thin (thin theo nội dung: 17 doc, 20 carrier đổi), 210.114 chunk, 193.547 doc (bằng docsel). Validator S26 (`--max-chunks-per-doc 2`) chạy lại bằng `compose validate` với 7 segment 120ask làm nguồn verbatim: ok, 0 lỗi, `p1_pair` 91.546, `canonical_substring` 118.568. Log: `out/runs/s36-fuse4/validate-120A-p4b-v0.log`. Chưa nộp (Vu nộp). Chưa tính phân bố ngôn ngữ/host của 36.160 doc cũ bị đẩy khỏi pool13.

## LB 120A-p4b-v0 = 0,4111 (25/10): bản tốt nhất, config cố định

LB 120A-p4b-v0 = 0,4111 (Doc 0,4394 / Chunk 0,3828): mốc mới, khung p4b giữ lợi thế +0,0078 so với khung free (120A-free-v0 0,4033). Bản này là **dự phòng cho private**. `configs/s36-s37-120A-p4b-v0.json` đã cố định: đường dẫn tường minh của pairs_new, điểm 4B và C-max (không glob, nên điểm 4B/Jina nhập sau không đổi bản này), `"jina": false` (chỉ điểm Jina của S35 + S36; 36.160 cặp pool của doc 120ask v0 xếp cuối nguồn Jina), khối `frozen` ghi SHA đầu vào, docsel, carriers và ZIP (`dd8d4d76…3a22`, 100.664.783 B, T 1536, 35.220 cửa sổ). Dựng lại bằng config này cho đúng SHA docsel `8a24b9b1…cca4` và carriers `f5830f26…1bd0` (đã kiểm). `apply` nhận `"jina": false`.

Bản chính v2 = cấu hình này + 120ask mới + Jina đầy đủ (`configs/s36-s37-120A-p4b-v2.json`, `"pool": "cosine13"`); bản tổng quát v2 = khung free D=300 cùng dữ liệu (`configs/s36-s37-120A-free-v2.json`).

Gói `final-4b` phải phủ cả hai khung: pool13 của khung p4b là top-n_q cosine (n_q = round(1,3 K_q), tối đa 339, 195 query có n_q > 300) và pool của khung free là top-300. `export-final` lấy top-max(300, n_q) mỗi query, kiểm mọi cặp chưa có điểm của cả hai khung nằm trong subset (lỗi nếu thiếu), và ghi vào `manifest.json` số cặp theo khung (`pairs_needed_free_frame`, `pairs_needed_p4b_frame`, `pairs_needed_either_frame`, `pairs_registry_below_both_cuts`, giờ Kaggle từng khung). Trên corpus 120ask v0 (chưa có registry v2): top-300 thiếu 33.233 cặp 4B, top-n_q thiếu 303 cặp (đều là doc canonical, nằm trong top-300).

## 120A-free-v1 và gói final-4b (25/10)

**120A-free-v1** (trung gian, không nộp): 120A-free-v0 với Jina đầy đủ cho 120ask. S37 xong Jina lúc 12:30 (`jina-final/scores-all.parquet`, 590 query, cộng `reused_scores.parquet`, 386 query: 325.839 cặp của 976 query, không trùng, phủ hết pool; gộp thành `out/runs/s36-fuse4/s37/jina-final-scores.parquet`, `configs/s36-s37-120A-free-v1.json` trỏ vào file này). `apply`: 1.249 cặp thiếu Jina (19 query) và 1.238 cặp thiếu C-max (10 query), không cặp nào thuộc doc 120ask; 228.059 doc (P10/P50/P90 = 73/165/373), Doc F2 dự đoán 0,4234 (mô hình), carrier CR-fair 175.552 (ρ 0,7670; P10/P50/P90 = 56/127/287, top-100 19,19%). Compose (`scripts/s36/run_120a_free_v1.sh`, 12:37 đến 14:07): `120A-free-v1-full+w768-t1536.zip`, 100.964.865 B, SHA256 `b9ed09e602c36420fa5732467ad7f554272990f1db8dc2721398cc4c11819c10`, T 1536 (sàn), 34.146 cửa sổ, 174.990 carrier sau thin (13 doc thin, 14 carrier đổi), 209.136 chunk. Validator (`compose validate`, segment 120ask làm nguồn verbatim) ok, 0 lỗi, `p1_pair` 88.649, `canonical_substring` 120.487.

**Gói `final-4b`** (`python -m r2ai.s36.d300 export-final --depth 300`, registry `pairs_new_final_v2.parquet` SHA `6d2c08e5…d4ae`, 13:56): 47.833 cặp (zh 24.718, vi 23.115; token TB 336), ids 1.059.378 đến 5.033.693; `extra_pairs` SHA `19be3713…e20b`, shard `pool-93003.parquet` SHA `71c70ab3…e30`, `pairs_final` SHA `43c1a2d3…cd38`, danh sách C-max SHA `b77dd626…4d6`. Phủ theo khung (đã kiểm: mọi cặp chưa có điểm của cả hai khung nằm trong gói): khung free top-300 cần 42.882 cặp (2,05 giờ Kaggle 2xT4); khung p4b (pool13 = top-n_q) cần 9.736 cặp (0,47 giờ); hợp của hai 42.937 cặp (2,06 giờ); 4.896 cặp registry nằm dưới cả hai ngưỡng (doc 120ask mới crawl, thêm theo yêu cầu, 0,23 giờ). Cả gói 47.833 cặp = 2,29 giờ. Thành phần: chỉ từ độ sâu 30.722, chỉ từ registry 4.896, cả hai 12.215 (registry chưa có điểm 17.111). Jina không chấm cho phần này (xếp cuối nguồn Jina; 85.230 cặp của top-300 thiếu Jina, 55.871 là doc 120ask). Chưa chạy Kaggle.

