# S8 — probes zh đợt 1 trên S6-g/K117

2026-10-14 16:15 UTC+7 · GPT-6 (Codex) · HEAD đã kiểm trước commit `27eaf789100e870e2e38ee173a7907bd34db0f23` · nhánh `diag-lang`.

## Base chính và file gửi Vu

**Base cố định S6-g:** Final 0,2595; Doc F2/P/R 0,2594 / 0,1563 / 0,3726; Chunk F2/P/R 0,2596 / 0,1786 / 0,3461. **LB mới do Vu cung cấp: ZL-G Final 0,2630; ZX-G best 0,2637.** Doc F2 ZL/ZX 0,2649/0,2663; Chunk F2 0,2610/0,2611. Inverse và ZXX xem [báo cáo mới](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/docs/s8_v2_language_probes_ZXX.md).

Candidate đợt 1 SHA `180174d76dadd9b476712f5f97483192d17efbaaebd66be4f3f47eb5475f87bd`, **60.000 dòng / 1.200 query**, coverage **120.028 / 240.000 doc zh = 50,011667%**. Inputs được đóng băng trong out; không dùng candidate live lúc dựng/validator.

Bốn ZIP dùng **ZIP-LZMA, dict 268.435.456 B (256 MiB), lc/lp/pb = 3/0/2**, header `5d00000010` khớp từng base. Trần **102.600.000 B**. Giữ nguyên relevant_docs, relevant_chunks, thứ tự và K của base; không cắt chunk vi.

| File | K | ZIP byte | Dưới trần byte | Doc zh vật lý / ID | Chunk zh thêm |
|---|---:|---:|---:|---:|---:|
| [ZL-E.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/ZL-E.zip) | 117 | 92,280,122 | 10,319,878 | 7,743 / 7,743 | 1,706 |
| [ZL-G.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/ZL-G.zip) | 150 | 88,242,781 | 14,357,219 | 7,743 / 7,743 | 1,706 |
| [ZX-E.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/ZX-E.zip) | 117 | 92,599,302 | 10,000,698 | 12,000 / 12,001 | 2,074 |
| [ZX-G.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/ZX-G.zip) | 150 | 88,546,146 | 14,053,854 | 12,000 / 12,001 | 2,074 |

## Cấu hình và nguyên tắc chunk

- **ZL:** tau_doc = **0,44287109375**, cap 20 doc/query; chunk logit >= **0,96**, tối đa 3 doc/query. Ngưỡng ZL giữ tuyệt đối, không hiệu chỉnh theo độ phủ.
- **ZX:** tau_doc = **-0,179931640625**, cap 25 doc/query, đạt đúng trung bình 10 doc vật lý/query; chunk logit >= **0,44**, tối đa 3 doc/query.
- Mỗi ID thuộc v2 nhận một chunk là nguyên `docs.parquet.doc_text`. Chỉ thêm chunk khi ID doc đã nằm trong relevant_docs sau bước chọn doc, trước bước chunk.
- **Đã kiểm trên 1.200 query của ZL-G và ZX-G: 0 ID được thêm để bảo vệ chunk, mọi body ID là tập con của doc đã chọn.** Hai ZIP hiện có đáp ứng nguyên tắc mới, không cần nén lại.

| Bản | Doc vật lý/query | ID/query | Chunk/query | Delta doc/query so ZL-D | Delta chunk/query so ZL-D |
|---|---:|---:|---:|---:|---:|
| ZL-D đợt 0 | 6,000000 | 6,000833 | 1,342500 | 0 | 0 |
| ZL đợt 1 | 6,452500 | 6,452500 | 1,421667 | +0,452500 | +0,079167 |
| ZX đợt 1 | 10,000000 | 10,000833 | 1,728333 | +4,000000 | +0,385833 |

ZL tăng 543 doc vật lý, 542 ID và 95 chunk so với ZL-D. ZX thêm **4.258 ID và 368 body** so với ZL. ZL giữ 5.650 ID cũ, thêm 2.093 và bỏ 1.551; không phải tập bao hàm ZL-D. Số query không có doc zh giảm 426 → 397. Do đó chuyển precision từ ZL-D sang đợt 1 là giả định cho tập được chọn mới. Không nhân điểm/recall trực tiếp theo tỷ lệ coverage.

## Dự báo trước nộp trên điểm base thật

Bảng dưới được lập **trước khi có LB ZL-G/ZX-G**. Final dự báo 0,263056/0,264646 so với LB thực 0,2630/0,2637. Kịch bản chuyển precision ZL-D đã dự báo quá cao phần ZX; inverse mới dùng điểm thực nằm ở báo cáo ZXX.

Kịch bản chính chuyển precision tổng ZL-D **11,822765%/doc ID**, và precision chunk zh **15,582649%** sang đợt 1. Cả hai là inverse có điều kiện; chunk còn phụ thuộc precision vi bị bỏ ở ZL-D. Với ZX, kịch bản chính giả định phần mở rộng có precision tương đương phần ZL.

| Probe | Delta Doc F2 | Delta Chunk F2 | Delta Final | Final dự báo |
|---|---:|---:|---:|---:|
| ZL-G | +0.00508379 | +0.00202893 | +0.00355636 | 0.26305636 |
| ZX-G | +0.00782753 | +0.00246438 | +0.00514596 | 0.26464596 |

Sensitivity riêng phần ZX mở rộng so với ZL (giữ giả định phần ZL; precision biên doc/chunk đặt bằng nhau để trình bày ngắn):

| Precision biên doc/chunk | Delta Final so S6-g | Final ZX-G dự báo |
|---:|---:|---:|
| 3% | +0.00299129 | 0.26249129 |
| 6% | +0.00369505 | 0.26319505 |
| 10% | +0.00463339 | 0.26413339 |
| 13% | +0.00533714 | 0.26483714 |

**Đây là dự báo, không phải điểm LB hay khoảng tin cậy.** Giả định N=1.200, public subset đại diện, số gold và F2 chung được thay vào từng query; sử dụng số doc/chunk thực tế của từng query. Residual Doc P vượt sai số làm tròn, nên precision gold không được nhận diện chính xác. Không dùng F2(P_macro,R_macro) thay điểm F2_macro. Sensitivity JSON giữ cả hai giả định vi-tail khả thi và doc/chunk biên độc lập.

[ZL-G forecast JSON](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/forecast-ZL-G.json) · [ZX-G forecast JSON](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/forecast-ZX-G.json) · [ZX-G CSV](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/forecast-ZX-G.csv).

E/K117 vẫn có hai ZIP đã kiểm; chưa dự báo DeltaFinal trên E vì thiếu điểm base đầy đủ. S6-g là base chính theo Vu.

## Nguồn, schema và kiểm tra

8 segment mới có source `zh_v2_ingest`, thiếu index verified flag và cột doc_ids_v2 trong docs.parquet. Exporter đưa full group vào candidate. Adapter S8 riêng trong out chiếu alias theo corpus v2 zh: **259 lượt alias ngoài v2** bị loại khỏi 60.000 dòng, giữ nguyên raw candidate, rank, logit, full group và text. Không có qihuang được chọn ở ngưỡng ZL/ZX.

Đã kiểm CPU **42.505 record mới**, 8 segment; tổng **37 file nguồn không rỗng / 120.028 primary ID duy nhất**. Kiểm schema/count/group/v2/nonempty doc_text/SHA; 17 nguồn cũ cần cho body giữ nguyên SHA. Snapshot xác minh **nguồn doc**, không tuyên bố xác minh dense/CSR index. Không sửa manifest live hay pipeline.

Manifest raw SHA `fc8da646613134462377d701dd6a2006f517f90889bae50a4b5dfea5d5484d26`; snapshot doc-source SHA `02955d4c0cdae6718baa3d2ce67a28c79fb9c049e5300699102d9096ffcf67fc`.

**Bốn validator native: 0 lỗi / 1.200 query**, schema/v2/protection/dedupe/near/verbatim đều bật. Đã kiểm lại input/source pins, ZIP SHA, JSON SHA sau giải nén đầy đủ, CRC, header, trần byte và giữ nguyên phần base. ZL-G tổng 135.425 chunk; ZX-G 135.793 chunk.

**65 kiểm thử CPU đạt (3,71 s)**. Review batch 1 không có Critical/Important/Minor. Không GPU/crawl/nộp/push; không đổi code/default pipeline. Taxonomy query dùng để phân tích, không lọc.

[Delivery gates](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/delivery-gates.json) · [Chunk/doc policy](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/chunk-doc-policy-G.json) · [Source pins](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/source-doc-verification.json) · [Freeze provenance](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-E-b1/batch1-freeze-provenance.json).

## SHA đầy đủ

### ZL-E

ZIP `a7475b47d33f763405ec253b81e30a37ee520eaaa7d31164cb2549ae14265294`

JSON `5a04a79b0cab76c059a0301898c21e67cddcd88dd5e3c35b8cfc33dc5f5804f7`

### ZL-G

ZIP `10fee23e44ba72832d21864a39860d8e4581802667425a14e3e4a6bd5cfbe74a`

JSON `429f0756ce7fb74e6ceee66014260a7c2fa99aeb5fccfe74de9ef82934280bce`

### ZX-E

ZIP `2d62b3f0569d9c40dd0141cae57df53c453000490118e3d6fe77cd01cee53ad5`

JSON `5abb38e6a94586db683dd95d8449b14eb605d04ff2a89e1aef97767713aefb16`

### ZX-G

ZIP `2de5f63d07c9d4ba62a8e9f71853c0a27ab4076cbac7b38ca166e3ea95369897`

JSON `e85d5d27456e410afc4c365c11ba6df5e6c6219992f817829a52d0b2d53bf8c7`

## Inverse ZL-D tham chiếu


**Tham chiếu đợt 0:** inverse từ LB ZL-D đã đo; thiết kế E/G ở cuối thuộc candidate b7 đợt0 và đã được thay bởi `batch1-counts.md`/candidate180174d đợt1. Không dùng tau ZX đợt0 cho probe đợt1.

## Điểm LB đã đo

| Bản | Final | Doc F2 | Doc P | Doc R | Chunk F2 | Chunk P | Chunk R |
|---|---:|---:|---:|---:|---:|---:|---:|
| S6-d K100 | 0.2522 | 0.2577 | 0.1525 | 0.3805 | 0.2467 | 0.1868 | 0.3292 |
| ZL-D K98 | 0.2549 | 0.2624 | 0.1518 | 0.3918 | 0.2475 | 0.1879 | 0.3297 |

Delta thực: Doc F2 +0,0047; Doc R +0,0113; Doc P -0,0007; Chunk F2 +0,0008; Chunk R +0,0005; Chunk P +0,0011.

## Tách doc theo đúng ID thêm

- Core ZG: 3.601 ID, logit >=0,96, cap10.
- Tăng cap10->20: 1.450 ID >=0,96.
- Dải nới tau: 2.150 ID trong [0,44287109375; 0,96).
- Tổng ZL-D: 7.201 ID / 7.200 doc vật lý. Không gọi toàn bộ phần thêm 3.600 ID là dải nới tau.

Vu chọn sensitivity cho precision của 1.450 ID tăng cap. Core A dùng inverse 12,651312%, chỉ là giả định chuyển precision sang base D.

| Precision tăng cap (giả định) | Precision dải nới tau | Khoảng do làm tròn D/ZL-D | Khoảng gồm làm tròn core A |
|---:|---:|---:|---:|
| 3.000000% | 16.127706% | 15.278255%-17.013753% | 14.397120%-17.829904% |
| 6.000000% | 14.215085% | 13.366007%-15.100756% | 12.484808%-15.916964% |
| 10.000000% | 11.664351% | 10.815759%-12.549530% | 9.934478%-13.365813% |
| 13.000000% | 9.750875% | 8.902641%-10.635692% | 8.021298%-11.452030% |
| 12.651312% | 9.973295% | 9.125021%-10.858155% | 8.243685%-11.674486% |

Các khoảng giữ precision tăng cap cố định; mỗi điểm LB 4 chữ số được xét +-0,00005. Đây là khoảng có điều kiện do làm tròn, không phải khoảng tin cậy thống kê.

**Kiểm tra không khớp:** mô hình dự báo Delta Doc P khoảng -0,00116 đến -0,00119, trong khi đo -0,0007. Chênh lệch 0,00046-0,00049 vượt sai số làm tròn. Vì vậy không thể gọi precision dải nới tau là số đo chính xác từ LB; cần giữ sensitivity và giả định macro.

## Chunk zh: có lẫn ảnh hưởng bỏ chunk vi

ZL-D thêm 1.611 chunk zh nhưng bỏ 2.390 chunk vi khi K100->K98. Precision chunk zh không được xác định riêng nếu chưa biết precision/recall contribution của phần vi bị bỏ.

| Precision vi bị bỏ (giả định) | Precision chunk zh | Khoảng làm tròn P/R | G_chunk/beta | Delta Chunk F2 kiểm tra |
|---:|---:|---:|---:|---:|
| 0.000000% | Không có nghiệm vật lý | - | - | - |
| 11.132389% | 15.582649% | 14.831094%-16.334204% | 57.215474 | +0.000803134 |
| 18.680000% | 26.919496% | 26.167940%-27.671051% | 62.349845 | +0.000805781 |

Kịch bản vi-tail 11,132389% lấy từ inverse ranks81-100 giữa A và D, **chưa đo riêng ranks99-100**. Nó cho chunk zh 15,582649%, khoảng làm tròn 14,831094%-16,334204%. Kịch bản vi-tail18,68% cũng khớp Delta Chunk F2 nhưng cho zh26,919496%; đây là bằng chứng phần zh chưa được nhận diện riêng.

## Công thức và giả định macro

Doc từng query: F2 = 5*T/(4*G+k). Khi thêm m doc, h hit:

    Delta F2_i = (5*h_i - m_i*F2_i)/(4*G_i + k_i + m_i)

Mô hình dùng N=1200, G_i=G chung, F2_i=F2_macro chung trên các query có thêm doc. Với core/cap/band có số ID a_i/b_i/c_i:

    h_i = a_i*p_core + b_i*p_cap + c_i*p_band
    Delta R = sum(h_i)/(N*G)
    Delta F2 = mean((5*h_i - (a_i+b_i+c_i)*F0)/(4*G+k_i+a_i+b_i+c_i))

Giải đồng thời G và p_band, giữ p_cap theo sensitivity. Việc thay F2_i/G_i thực bằng số chung chỉ là xấp xỉ; public LB phải đại diện cho 1.200 query. Không dùng F2(P_macro,R_macro) thay cho F2_macro.

Chunk: C_i là số chunk base, v_i vi bị bỏ, m_i zh thêm, P0 là macro P base. Đặt w_m=mean(m_i/(C_i-v_i+m_i)), w_v=mean(v_i/(C_i-v_i+m_i)):

    p_zh = P0 + (Delta P - w_v*(P0-p_vi))/w_m
    eta_zh = sum(m_i)*p_zh/(N*Delta R + p_vi*sum(v_i)/eta_vi)

eta=G_chunk/beta cho phép một chunk cover nhiều gold; giả định contribution hit/eta chung. Kiểm tra Delta F2 bằng mô hình P/R từng query với số C_i thực, không lấy harmonic mean của macro P/R làm điểm LB.
