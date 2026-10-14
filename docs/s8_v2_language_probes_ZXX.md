# S8 — inverse G/ZL/ZX, ZXX-G và builder bản ghép

2026-10-14 17:15 UTC+7 · GPT-6 (Codex) · HEAD đã kiểm trước commit `e1267ec12738390e68ea384b39c368f017f78a84` · nhánh `diag-lang`.

## Điểm LB do Vu cung cấp

| Bản | Final | Doc F2 | Doc P | Doc R | Chunk F2 | Chunk P | Chunk R |
|---|---:|---:|---:|---:|---:|---:|---:|
| G | 0.2595 | 0.2594 | 0.1563 | 0.3726 | 0.2596 | 0.1786 | 0.3461 |
| ZL-G | 0.2630 | 0.2649 | 0.1559 | 0.3849 | 0.2610 | 0.1784 | 0.3490 |
| ZX-G | 0.2637 | 0.2663 | 0.1548 | 0.3900 | 0.2611 | 0.1781 | 0.3494 |
| ZXX-G | 0.2639 | 0.2667 | 0.1523 | 0.3949 | 0.2610 | chưa cung cấp riêng | chưa cung cấp riêng |

**Best đã đo do Vu cung cấp: ZXX-G = 0,2639**, so ZX-G +0,0002 (thành phần F2 cho delta +0,00015, khác do rounding).

## ZXX-G đã kiểm

[ZXX-G.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/ZXX-G.zip) — **88,289,377 B**, dưới trần 102.600.000 B; dư 14,310,623 B.

Base S6-g/K150 giữ nguyên toàn bộ doc/chunk/thứ tự. Candidate đợt 1 SHA `180174d76dadd9b476712f5f97483192d17efbaaebd66be4f3f47eb5475f87bd`, coverage **120.028/240.000 = 50,011667%**. ZIP-LZMA dict **268.435.456 B (256 MiB)**, lc/lp/pb **3/0/2**, header `5d00000010` khớp base.

tau_doc **-0.87255859375**, cap **30**, **18,000 doc vật lý / 18,001 ID** (15 doc/query). Chunk giữ nguyên **1,706 body của ZL-G**: tau >=0,96/top3, nguyên docs.parquet.doc_text, một chunk/ID v2, chỉ cho doc đã chọn. Không thêm ID bảo vệ chunk, không cắt vi.

Phần doc thêm so ZX: 6.000 ID, gồm 1339 do cap25→30 tại tau ZX và 4661 do hạ tau. Đây là tăng ngân sách trung bình 10→15, không phải một dải hạng cố định trên mọi query.

**Validator native 0 lỗi / 1.200 query / 135,425 chunk**; schema/v2/protection/dedupe/near/verbatim đều bật. SHA input/source/ZIP/JSON readback, CRC, header, byte cap, full base prefix và wholebody parity đạt.

ZIP SHA `680a8cfd52dc7958c40fe00636b1ef1d96549f6d190722f4bb7cf6b37eb130d8`

JSON SHA `46e9333970be95efe68f0a0a196f54c56b1c7be349bd8d4b642992fffdfd7078`

## Dự báo trước nộp so với ZX-G

Chunk ZXX trùng ZL-G nên Chunk F2 lấy số đã đo **0,2610**, so ZX **0,2611**: DeltaChunk **−0,0001**, khoảng rounding [−0,0002; 0]. Bảng dưới là dự báo trước LB, từng dùng precision biên ZX **8,480113%/ID** làm tham chiếu. LB thực ZXX-G hiện 0,2639.

| Precision phần doc mới | Delta Doc F2 | Delta Final so ZX-G | Final dự báo |
|---:|---:|---:|---:|
| 3.000000% | -0.00143428 | -0.00076714 | 0.26293286 |
| 6.000000% | +0.00041561 | +0.00015780 | 0.26385780 |
| 7.000000% | +0.00103224 | +0.00046612 | 0.26416612 |
| 8.000000% | +0.00164887 | +0.00077444 | 0.26447444 |
| 8.480113% | +0.00194492 | +0.00092246 | 0.26462246 |
| 10.000000% | +0.00288213 | +0.00139107 | 0.26509107 |
| 13.000000% | +0.00473202 | +0.00231601 | 0.26601601 |

Ngưỡng hòa vốn có điều kiện **5,488172%/ID**, đã gồm phần giảm chunk −0,0001. Dự báo dùng F/G chung, N=1.200, public subset đại diện và k/m thực từng query; không phải điểm LB hay khoảng tin cậy. Residual Doc P vượt rounding; G fitted cho ZL và biên ZX khác nhau.

[Forecast JSON](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/forecast-ZXX-G.json) · [Native proof](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/ZXX-G.validator.json).

## Chốt bản ghép đợt 2 (2026-10-14 17:15)

Doc biên ZX→ZXX: **5,988226%/ID**, rounding **5,809125–6,179002%**; G có điều kiện61,104348. Doc P residual −0,00036163 vẫn vượt rounding, nên không coi là precision gold chính xác. Phần thêm gồm 6.000 ID, có cả tăng cap và hạ tau; mức ngân sách trung bình10→15 không là dải hạng cố định ở mọi query.

**Khóa τ_doc = −0,87255859375 tuyệt đối, cap30; không hiệu chỉnh theo mean.** Chunk theo ZL (tau0,96/top3, whole doc_text/everyv2ID, chỉ doc đã chọn). Base thắng giữa S6-f/S6-h1/S6-h2 chờ Vu báo; candidate đợt2 chờ S11 handoff path/SHA và snapshot nguồn. Không dựng/fallback đợt1 trong lúc chờ.

[Entry point khóa](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/locked_compose.py) · [Policy](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/locked-policy-b2.json) · [Job pending](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/pending-b2.json) · [Điểm/inverse mới](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/measured-ZXX-G.json).

Wrapper tự báo số doc vật lý/query, ID/query, chunk/query và delta so ZXX-G: **15 / 15,000833 / 1,421667**. 11 kiểm thử guard/so sánh số đếm đạt (0,38s); job pending đã xác nhận dừng vì thiếu pin trước build. Giữ generic builder cũ nguyên vẹn. Toàn bộ gate SHA/source/native/v2/readback và trần/dict giữ nguyên.

## Builder bản ghép (API đã chuẩn bị trước quyết định khóa)

[compose_builder.py](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/compose_builder.py) · [native worker](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/validate_compose.py) · [config mẫu](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/example-G-b1.json) · [hướng dẫn](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/README.md).

CLI dùng JSON tham số base S6-g/S6-f/S6-h*, candidate đợt 1/2, absolute tau_doc đã chọn theo LB, cap<=30. Base ZIP/JSON SHA, candidate SHA, snapshot manifest SHA và SHA nguồn doc bắt buộc; thiếu pin/snapshot verification thì dừng. Giữ toàn bộ base bất kể K label; chunk tau0,96/top3 chỉ cho doc đã chọn. Native/schema/v2/protection/dedupe/near/verbatim/CRC/readback luôn bật. Resume bind base/config/candidate/source/tau/header/plan SHA. Base có chunk zh thiếu source handoff thì dừng, không fallback.

Đã chạy preflight thật trên mẫu G/batch1: 12.000 doc vật lý, 12.001 ID, 1.706 body; **không sinh thêm ZIP**. Real build yêu cầu điểm base đầy đủ. Chưa dựng bản ghép F/H/đợt2: chờ Vu handoff điểm và pins.

```powershell
D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe "C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/compose_builder.py" --config "C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/compose/example-G-b1.json" --prepare-only
```

Bỏ --prepare-only sẽ dựng và validate. Cấu hình mẫu là G/batch1; đổi sang F/H/đợt2 phải truyền pin mới, không sửa một file config đã gắn vào artifact ready. Đặt output/name mới.

**82 test CPU đạt**, một fresh review không Critical/Important; Minor đơn vị ID/query đã sửa. Đã đo encoder peak khoảng2,95GiB; native worker peak4,73GiB; preflight reader peak0,51GiB. Không có file N+1 đã handoff nên chưa overlap native/encode; chỉ được overlap khi tổng RAM S8 <=12GiB và dừng overlap nếu vượt. Lượt ZXX này nén khoảng22 phút, ZIP đã nén đến native gate hoàn tất khoảng21 phút; RAM khả dụng quan sát chỉ0,48GiB ở pha cuối, đã giữ chạy tuần tự. Preflight/config chạy trong vài phút; full ZIP/native cần thời gian nén/kiểm.

Không GPU/crawl/nộp/push/pipeline/default changes. Helper/builder/config nằm trong out ignored, được giữ để dùng vòng sau; commit chỉ docs.

## Inverse chi tiết


Candidate đợt 1 SHA `180174d76dadd9b476712f5f97483192d17efbaaebd66be4f3f47eb5475f87bd`; coverage 120.028/240.000. Các số dưới là nghiệm theo mô hình macro, không phải precision gold được nhận diện chính xác.

| Phần doc | ID thêm | Precision từ F2/R | Khoảng do rounding | G chung có điều kiện |
|---|---:|---:|---:|---:|
| G → ZL-G | 7743 | 12.588829971% | 12.328391–12.860096% | 66.040183242 |
| ZL-G → ZX-G | 4258 | 8.480112869% | 8.154997–8.831215% | 59.000523850 |

| Phần chunk | Chunk thêm | Precision từ P/R | Khoảng do rounding | Delta Chunk F2 đo |
|---|---:|---:|---:|---:|
| G → ZL-G | 1706 | 16.215051472% | 15.397577–17.032526% | +0.0014000 |
| ZL-G → ZX-G | 368 | 7.040993962% | 3.446325–10.635663% | +0.0001000 |

Khoảng xét từng metric ±0,00005 độc lập; không phải khoảng tin cậy. Delta Chunk F2 của phần ZX thêm nằm trong **[0; +0,0002]** do rounding: bão hòa theo mức điểm quan sát, chưa chứng minh contribution đúng bằng 0.

Không đủ phép đo để tách precision thật hạng 1–3 / 4–6 / 7–10. ZL có 6,4525 doc vật lý/ID trên mỗi query; ZX 10 doc vật lý và 10,000833 ID/query. Cap là 20 và 25, nên mức trung bình không tương đương các dải hạng cố định trên từng query.

ZX thêm 956 ID do cap20→25 ở τ ZL, và 3302 ID do hạ τ; inverse 8,480113% chỉ áp dụng tổng 4.258 ID. ZXX thêm 1339 ID do cap25→30 ở τ ZX và 4661 ID do hạ τ; precision của 6.000 ID này chưa đo.

## Công thức và giới hạn

`F2_i = 5 T_i / (4 G_i + k_i)`; khi thêm m_i ID, h_i hit:

`DeltaF2_i = (5 h_i - m_i F2_i)/(4 G_i + k_i + m_i)`.

Giải F2/R với h_i = m_i p, G_i = G chung, thay F2_i bằng F2_macro; DeltaR = p sum(m_i)/(1200 G). Dùng k_i/m_i thực của từng query. Giả định public subset đại diện cho 1.200 query. Hai fit cho G khác nhau (66,040183 và 59,000524), cho thấy đây là mô hình theo phần query được chọn, không đo số gold thật.

| Fit doc | Delta P quan sát | Delta P dự báo (macro P thay local) | Residual |
|---|---:|---:|---:|
| ZL_all_docs | -0.00040000 | -0.00117476 | -0.00077476 |
| ZX_extra_docs | -0.00110000 | -0.00153966 | -0.00043966 |

Residual Doc P vượt rounding; số precision nhiều chữ số chỉ là độ chính xác số học của mô hình. Không lấy harmonic mean của P_macro/R_macro thay F2_macro.

Chunk không cắt vi: p_new = P0 + DeltaP / mean(m_i/(C_i+m_i)); eta = p_new sum(m_i)/(1200 DeltaR). Eta = G_chunk/beta cho phép một chunk khớp nhiều gold. Vẫn giả định macro P/R dùng được trên từng query. Chunk ZL có F2 residual +0,0001274; chunk biên ZX residual −0,0000392.

[JSON đầy đủ](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/inverse-G-ZL-ZX.json).
