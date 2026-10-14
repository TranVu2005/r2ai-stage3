# S8 — inverse A và probes D

2026-10-14 00:33 UTC+7 · GPT-6 (Codex) · HEAD kiểm trước commit `63b744bb15e02cdffec62c3462a7bc01be965154`, nhánh `diag-lang`.

## Inverse từ điểm LB A do Vu cung cấp

Z1-A Final 0,2497 / Doc F2 0,2622 P 0,1563 R 0,3790; chunk như S6-a. Z2-A Final 0,2500 / Chunk F2 0,2379 P 0,2053 R 0,2989; doc như Z1-A. Dùng **3.601 ID / 589 chunk thực**, không dùng 662 chunk trước cắt byte.

| Inverse | Precision có điều kiện | Tham số giải cùng | Biên chỉ do rounding 4 chữ số |
|---|---:|---:|---:|
| Doc ΔF2/R | 12,651311990950104% | Gdoc 59,319498020066675 | 12,153849%–13,188459% |
| Chunk ΔP | 15,61145672664172% | Gc/β 58,94325648712698 qua ΔR | 13,966942%–17,255971% |
| Chunk ΔF2/R | 17,14777735949102% | Gc/β 64,74385169705147 | trong biên rounding P |

Chunk P/R fit dự đoán ΔF2=0,000766567 vs 0,0008 hiển thị. Kiểm chéo Doc P: constant P_selected=Pmacro cho 15,63%; F2/R commonG/Fmacro dự đoán ΔP−0,000164007, ngoài hiệu rounding±0,0001. Do đó các macro substitutions không tái tạo đồng thời 3 metric; không có precision gold chính xác. Nếu cho F_selected tự do và fit đồng thời ΔF/P/R, doc precision = **11,97528910024888%**, G_selected = 56,149760 và F_selected = 0,237476; đây là một nghiệm theo giả định khác, không phải gold precision xác định duy nhất. Không có gold precision chính xác: cần common gold/constant precision/selected quality như toàn bộ và N=1200 (scorer has_gold có thể khác N). **Không là CI**, không gồm heterogeneity. Doc nominal A 5,188%; chunk one-to-one A 4,742%; conditional full-body model 5,726129%, không universal. D nominal doc 5,154%, chunk one-to-one 4,934%. F2(macroP,R) khác macroF2. Nếu LB chỉ chấm một tập con query, cần giả định tập đó có cùng phân bố số doc/chunk thêm và chất lượng với toàn bộ 1.200 query; số đếm toàn bộ không tự xác định denominator hoặc số đếm trên tập LB.

[Inverse chi tiết/công thức](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/inverse-A.md); [JSON inverse](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/inverse-A.json).

### Công thức inverse và giả định

Với N = 1.200, K_i/C_i là số doc/chunk trong base A, m_i là số ID/chunk thực thêm ở query i:

```text
w_doc(G) = (1/N) sum_i m_i / (4G + K_i + m_i)
Delta Doc F2 = (5p_doc - F0) w_doc(G)
Delta Doc R  = p_doc sum_i(m_i) / (N G)

w_chunk = (1/N) sum_i m_i / (C_i + m_i)
p_chunk = P0 + Delta Chunk P / w_chunk
eta_chunk = G_chunk/beta = p_chunk sum_i(m_i) / (N Delta Chunk R)
```

Doc: F0 = 0,2594, ΔF2 = 0,0028, ΔR = 0,0064, tổng m_i = 3.601; giải chung p/G. Chunk: P0 = 0,2056, ΔP = −0,0003, ΔR = 0,0013, tổng m_i = 589; w_chunk = 0,006062390150554401. Giả định precision và G chung, F_selected/P_selected như macro baseline; beta là số gold mới được phủ trung bình trên chunk đúng. Số gold và beta không tách được từ các macro này.

## Base D và artifact

S6-d **K100**, dict 64, ZIP **102.292.334 B**, SHA ZIP `7230ed06a9dc5d51252c1c1e33c8763093c74c45f0b5b017308a390223f9a05a`, JSON `43e2bc2f81a3d648b084861643c8285b526cb2553480a0d37d88f376c19a7af3`. **Điểm LB thật do Vu cung cấp:** Final 0,2522; Doc F2 0,2577 P 0,1525 R 0,3805; Chunk F2 0,2467 P 0,1868 R 0,3292.

Z2-D: doc ZG-v2 τ 0,96/cap 10/cả nhóm ID v2; toàn `docs.parquet.doc_text` rank 1 ≥0,96, một chunk/ID và bảo vệ doc. Đã khôi phục đủ 662 chunk, không dùng A đã bỏ 73 chunk. ZL-D: **τ_doc=0.44287109375**, cap 20; **7.200 doc / 6,0 doc/query**, 7.201 ID. Chunk mọi doc ≥0,96/cap 3: **1.611 / 1,3425 chunk/query**. Không lọc taxonomy. Snapshot index của candidate b7 **77.523/240.000=32,30125%**; không ngoại suy recall. Candidate SHA `b7a03d58e6087f7e888f57ee7ff2520258770286ddccf29d5e0b8a8b3f345ed3`, v2 SHA `dad50fb81c812a5c68386a2313f880db488b457e85707b92d7c478c29523374a`.

| File | ZIP byte | Dưới trần | K vi cuối | Chunk vi bỏ | Doc zh / ID thêm | Chunk zh thêm | τdoc |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Z2-D.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/Z2-D.zip) | 102,064,951 | 535,049 | 99 | 1195 | 3601 / 3601 | 662 | 0.96 |
| [ZL-D.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/ZL-D.zip) | 101,896,118 | 703,882 | 98 | 2390 | 7200 / 7201 | 1611 | 0.44287109375 |

Bỏ vi chỉ ở cuối danh sách theo K cuối; query ít hơn K giữ nguyên. **Mọi relevant_docs gốc giữ nguyên**, kể cả ID từng bảo vệ chunk bị bỏ. Nội dung/thứ tự chunk vi giữ lại nguyên vẹn; không cắt text zh. ZL được giảm K hơn Z2 theo Vu chốt; không dưới K98. `*.attempts.json` ghi kích thước thật từng lần.

## Snapshot và provenance

Manifest chung thêm 5 segment chưa verified trong lúc nén, làm gate SHA chặn bản K99 đầu. Vu duyệt đóng băng 29 segment verified / 77.523 doc vào `manifest-candidate-b7.json`; cả 15 file nguồn top3 giữ nguyên SHA. Candidate b7 không đổi; không dùng segment mới hoặc qihuang. Snapshot ghi SHA manifest hiện hành lúc trích, không tuyên bố byte-identical với manifest cũ. K100 đã đo vượt trần nên tiếp tục từ K99 với dữ liệu đã pin.

Provenance chi tiết: [snapshot frozen](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/manifest-candidate-b7.json), [audit/source SHA Z2-D](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/Z2-D.approval.json), [audit/source SHA ZL-D](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/ZL-D.approval.json). Snapshot SHA256 `53a6d222a2219a84a5ea59a76107cccc089bbfefadb592e55e3ec562e6d8ec54`; manifest đang chia sẻ lúc trích có SHA256 `aeda8ad82c581b8d17c53bb0f9baea4fde071f1c38c3433577a753f546ef5a9f`.

## Validator và SHA

Cả hai **native 0 lỗi / 1.200 query**, schema/v2/protection/dedupe/near/verbatim đầy đủ, không skip. Source adapter chỉ đọc vi/CC B docs và mọi doc_text zh cần kiểm. CRC/SHA giải nén, retained vi prefix, doc ownership/order kiểm lại trước báo cáo. **31 test CPU pass**. Fresh review tìm lỗi công bố ZIP trước gate SHA; đã sửa RED→GREEN trước dựng thật. Không GPU/crawl/nộp/push; không sửa pipeline/default.

Base và outputs **ZIP-LZMA dict 67.108.864 B / lc-lp-pb 3/0/2/props `5d00000004`**, encoder như writer base. Mỗi ZIP chỉ 1 JSON.

### Z2-D

ZIP SHA256 `b7c5ea88b0ea22a5c5c181fe2b9b8ad0d347e18840bc9be61c713de93851e73e`

JSON SHA256 `4ca0b554711c2fcbdaaa39f63d8748334aa855b51de0cdbff3cf8f8c42bf1c41`

### ZL-D

ZIP SHA256 `53be84725583e78ad9dab601e77f64a0718421a11bc7a4795225aaef1783b7c9`

JSON SHA256 `06a3e12038115345d7641aa801b780996e029858e235b1f831ee01541f1d9bcc`


## Dự báo ΔFinal — chưa đo LB D probes

Tất cả Δ so **S6-d K100 thật**, gồm loss giảm K; doc không đổi khi cắt vi. Mô hình nhiều-nhiều:

```text
ΔDoc=mean[(5E_hit_i−m_i Fdoc)/(4Gdoc+Kdoc_i+m_i)]
Fchunk(H,R,C)=5HR/(4H+RC)
H'=P0 C−removed_vi*p_tail+E_hit_zh
R'=R0−removed_vi*p_tail/eta_vi+E_hit_zh/eta_zh
C'=C−removed_vi+new_zh
ΔFinal=(ΔDoc+ΔChunk)/2
```

Gdoc 59,319498; eta_zh=Gc/β_zh = 58,943256 fit A. eta_vi = 70,246783 và p_tail = 11,132389% là estimate cho **23.928 chunk vi hạng 81–100**, đã so prefix A/D khớp 1.200 query. **Hạng 99/100 riêng chưa đo**; dùng chỉ như scenario, CSV có p_tail 0% và 18,68%. Fdoc/Pchunk/Rchunk macro thay local là giả định; model dự báo hiệu trên baseline pseudo-local, không thay điểm base thật. Kiểm chéo A→D: vi-tail P/R model dự đoán ΔChunk F2 +0,012488931, nhưng LB đo +0,009600; residual +0,002888931. Không fit đồng thời cả F2/P/R, nên loss cắt vi vẫn chỉ là sensitivity theo giả định, không phải số đo hạng 99/100. K/m/C lấy artifact thật; không CI.

### Mọi zh mới cùng precision biên p

| p | ΔDoc Z2 | ΔChunk Z2 | ΔFinal Z2 | ΔDoc ZL | ΔChunk ZL | ΔFinal ZL | ΔFinal ZL−Z2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3% | -0.000798 | -0.000836 | -0.000817 | -0.001567 | -0.001773 | -0.001670 | -0.000853 |
| 6% | +0.000314 | -0.000589 | -0.000138 | +0.000616 | -0.001174 | -0.000279 | -0.000142 |
| 10% | +0.001796 | -0.000260 | +0.000768 | +0.003526 | -0.000376 | +0.001575 | +0.000807 |
| 13% | +0.002908 | -0.000014 | +0.001447 | +0.005709 | +0.000223 | +0.002966 | +0.001519 |

### Dùng precision core A ước tính; phần mở rộng dùng precision biên p

Core 3.601 doc ở 12,651312% và 589 chunk ở 15,611457%. Z2 phục hồi 73 top-1 logit thấp; ZL thêm 3.600 doc ID và 949 chunk rank 2/3 so Z2. Những phần chưa đo này dùng p 3/6/10/13%. Bảng dùng cùng p cho doc/chunk; nếu hai precision khác nhau, lấy ΔDoc ở hàng p_doc và ΔChunk ở hàng p_chunk rồi chia tổng cho 2:

| p phần mới | ΔDoc Z2 | ΔChunk Z2 | ΔFinal Z2 | ΔDoc ZL | ΔChunk ZL | ΔFinal ZL | ΔFinal ZL−Z2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3% | +0.002778 | +0.000087 | +0.001433 | +0.001946 | -0.000852 | +0.000547 | -0.000886 |
| 6% | +0.002778 | +0.000114 | +0.001446 | +0.003037 | -0.000472 | +0.001282 | -0.000164 |
| 10% | +0.002778 | +0.000150 | +0.001464 | +0.004492 | +0.000034 | +0.002263 | +0.000799 |
| 13% | +0.002778 | +0.000177 | +0.001478 | +0.005582 | +0.000414 | +0.002998 | +0.001520 |

CSV bổ sung mô hình tuyến tính với tỷ lệ trọng số do Vu đo: **doc P:R = 1,9:1**, **chunk P:R = 1:4,5**. Hệ số hiệu chỉnh từ ΔA: doc γP/γR = 0,83125/0,4375; chunk = 0,144144/0,648649. Cột `delta_*_empirical_weights` báo dự báo thứ hai; giả định thay macro P/R cho từng query nên không bảo đảm chuyển đúng từ A sang D.

[Forecast Z2-D CSV](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/forecast-Z2-D.csv); [Forecast ZL-D CSV](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-D/forecast-ZL-D.csv). Có cả 2 mode, p 3/6/10/13 và 3 scenario vi-tail; cột `delta_chunk_vi_cut_only` tách riêng loss cắt vi. Không cộng ZIP từng nguồn để suy byte.

## Diễn giải lượt LB tiếp theo

Nếu ZL-D có K thấp hơn Z2-D, ΔChunk giữa hai file gồm cả thêm zh và bỏ vi; không inverse precision chunk biên từ hiệu LB đó như một phép append thuần. ΔDoc vẫn chỉ do thay phần zh vì mọi doc vi giữ nguyên. N cần xác nhận theo denominator LB trước inverse; không suy precision gold từ Final đơn độc.

## Gửi duyệt

Gửi **Z2-D trước**, ZL-D sau; Vu duyệt trước mỗi lần nộp. Agent chưa nộp/push. Probe-en chờ S10 / τ Q75 / bỏ title_only. Taxonomy/ICD giữ ở [báo cáo A](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/docs/s8_v2_language_probes.md), chỉ phân tích lỗi.
