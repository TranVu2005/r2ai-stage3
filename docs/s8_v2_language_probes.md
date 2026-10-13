# S8 v2 — thiết kế probe en/zh và nhãn query

Cập nhật 2026-10-13 15:31 UTC+7 · GPT-6 (Codex) · HEAD kiểm trước commit `c0d5c87dd035d317fc65994e92c01a568dfc2827`, nhánh `diag-lang`.

**Trạng thái:** hoàn thành phân loại rule-based, thiết kế và bộ sinh offline với 11 test giả pass. Chưa có probe thật, chưa chạy GPU/crawl, chưa nộp/push. S8 v1 đã bị Vu thay thế; các file `out/diag-lang/probe-no-*` là artifact cũ, không dùng. Output hiện hành: `out/diag-lang/v2/` trong worktree `C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3`.

## 1. Baseline và dữ liệu đã xác minh

- Base: `D:/GitHub/r2ai-stage3/out/runs/s9/sub_K80_LZMA_v2.zip`.
- ZIP **78,734,571 byte**, SHA256 `3e2935437eacb68cc0d7e372badae5fbccf0d82e35f605239caea12f99ec0b4e`.
- JSON SHA256 `91fd6049245313dde4a7d42c8f86bae851acb43cf1aa7f6ccc8fd8e73b373a44` khớp bản K80 đã nộp theo Vu. 1.200 query duy nhất, **163,613 doc entry**, **95,999 chunk entry**; toàn bộ ID có nhãn vi trong membership v2. Nhãn ngôn ngữ lấy từ membership, không chạy detector text.
- K/query: trung bình 136.344167, min/max 99/204; chunk/query: 79.999167, min/max 79/80.
- Điểm **do Vu cung cấp, không có lượt đo riêng trên LB v2**: Final 0,2334; Doc F2 0,2446, P 0,1528, R 0,345; Chunk F2 0,2222, P 0,1955, R 0,2744. cc-A-cut115 Doc F2 0,2593 là tham chiếu khác cấu hình, không cộng vào forecast của K80.
- Query nguồn đọc OLD, không ghi: `D:/GitHub/r2ai-stage3-old/data/raw/query.parquet`, SHA `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08`; S11 dùng cùng query hash.
- Membership hiện hành: `D:/GitHub/r2ai-stage3/data/corpus_v2_ids.parquet`, SHA `dad50fb81c812a5c68386a2313f880db488b457e85707b92d7c478c29523374a`. File corpus v2 thực tế có 1.220.598 URL: vi 740.598, en 240.000, zh 240.000; tỷ lệ 60/20/20 là mô tả làm tròn, không dùng chia byte.

S11 chưa bàn giao `cand_zh_v2.parquet`; chỉ dùng bản export đủ 1.200 query sau khi chạy xong. **Không dựng lại từ cache index cũ**, không chạy GPU riêng. Coverage công bố: **77.523/240.000 doc zh = 32,30125%**. ΔDoc R của probe phản ánh phần index hiện có, là cận dưới của khả năng phủ; không nhân tuyến tính ×3,1 để dự báo index đầy đủ và không coi ΔF2 là cận dưới.

S10 chưa bàn giao `cand_en.parquet`; cần đường dẫn/schema và mapping chunk/docs tương ứng trước khi tạo probe-en. Vu đã chốt **loại doc `title_only=true`** khỏi probe-en. Bộ sinh hỗ trợ contract đã chuẩn hóa giống S11; chưa khẳng định artifact S10 thực tế dùng cùng layout.

## 2. Phân loại 1.200 query — chỉ phân tích lỗi

`query_tags.parquet/csv` lưu ID, nguyên query, nhãn chính, mọi nhãn khớp, từ khóa chứng cứ và `needs_review`; `type_x_chapter.csv` là bảng chéo. Generator không nhận taxonomy làm input, không lọc query theo nhãn/ngôn ngữ.

| Loại chính | Query | Tỷ lệ |
|---|---:|---:|
| Chẩn đoán | 446 | 37.17% |
| Chưa xác định | 163 | 13.58% |
| Điều trị | 343 | 28.58% |
| Triệu chứng | 53 | 4.42% |
| Thuốc | 188 | 15.67% |
| Định nghĩa | 7 | 0.58% |

Rule `S8-v2-lexical-3`: chuẩn hóa chữ/dấu cho tìm từ khóa; giữ bản có dấu để phân biệt chưa/chữa, mất/mắt, rằng/răng, sỏi/sởi, phụ/phù, gân/gan/gần, thần/thận… Ưu tiên phần hỏi cuối, giữ mọi chủ đề bệnh sử làm nhãn phụ. Dinh dưỡng/phòng ngừa vào điều trị; vaccine/sử dụng thuốc vào thuốc; an toàn/tiên lượng/diễn giải xét nghiệm vào chẩn đoán. Tie-order: thuốc → điều trị → chẩn đoán → định nghĩa → triệu chứng. Đây là quy ước phân tích; ví dụ câu hỏi vaccine có thể đồng thời là chẩn đoán/điều trị.

Chương ICD-10 là **topic tag**, không phải chẩn đoán lâm sàng hay mã hóa chính thức. Từ khóa bệnh cụ thể nặng hơn từ chỉ cơ quan; lưu đa nhãn và giữ unknown. Kiểm mẫu đã sửa va chạm dấu; **chưa đo accuracy**. Có 755 query cờ review (gồm đa intent, không đồng nghĩa 755 lỗi). ICD chính: **276 unknown**, **70 multiple**; số còn lại trong bảng:

| Chương/chủ đề | Dải mã | Query chính |
|---|---|---:|
| I — Nhiễm trùng và ký sinh trùng | A00-B99 | 69 |
| II — U tân sinh | C00-D48 | 23 |
| III — Máu và miễn dịch | D50-D89 | 29 |
| IV — Nội tiết, dinh dưỡng, chuyển hóa | E00-E90 | 47 |
| V — Tâm thần và hành vi | F00-F99 | 7 |
| VI — Thần kinh | G00-G99 | 29 |
| VII — Mắt | H00-H59 | 33 |
| VIII — Tai và xương chũm | H60-H95 | 13 |
| IX — Tuần hoàn | I00-I99 | 37 |
| X — Hô hấp | J00-J99 | 44 |
| XI — Tiêu hóa | K00-K93 | 76 |
| XII — Da và mô dưới da | L00-L99 | 27 |
| XIII — Cơ xương khớp | M00-M99 | 28 |
| XIV — Sinh dục và tiết niệu | N00-N99 | 82 |
| XV — Thai kỳ, sinh đẻ và hậu sản | O00-O99 | 73 |
| XVI — Chu sinh | P00-P96 | 6 |
| XVII — Dị tật bẩm sinh | Q00-Q99 | 9 |
| XVIII — Triệu chứng và dấu hiệu chưa xác định | R00-R99 | 109 |
| XIX — Chấn thương, ngộ độc và hậu quả | S00-T98 | 13 |
| XX — Nguyên nhân bên ngoài | V01-Y98 | 1 |
| XXI — Chăm sóc và yếu tố sức khỏe | Z00-Z99 | 96 |
| XXII — Mã đặc biệt | U00-U85 | 3 |

Dải mã tham khảo [WHO ICD-10 2019](https://icd.who.int/browse10/2019/en) và [BfArM ICD-10-WHO 2019](https://klassifikationen.bfarm.de/icd-10-who/kode-suche/htmlamtl2019/index.htm). Rulebook SHA `96c25355aec06df8aee58b6fbb43df32f0411ab6305a77a57ab74f797e9c8aef`. Không dùng mẫu review để tuyên bố accuracy; muốn đánh giá cần tập nhãn tay độc lập.

## 3. Hai probe và ngưỡng đề xuất

Cố định một baseline cho cả cặp. Với **mọi 1.200 query**, chọn doc có rank cao nhất còn mới, đủ ngôn ngữ và raw logit ≥ τ. Nếu nhóm ID đã xuất hiện trong doc/chunk base thì bỏ nhóm đó, xét rank tiếp theo; tối đa **1 ID + 1 chunk/query**. Không mở rộng mọi alias của nhóm; nếu ID đại diện ngoài v2 thì chọn alias đầu tiên đã xác minh trong v2. Đây là precision theo ID mới, không phải precision của cả nhóm alias.

- **probe-zh:** τ_zh **0,96** cho lượt đầu; giữ mặc định ZG cũ. Đây là ngưỡng thăm dò, không khẳng định calibration giữa index cũ và S11. Sau bàn giao, thống kê số query/doc vượt ngưỡng trên logit S11 thực.
- **probe-en:** τ_en = **Q75 của điểm top-1 doc mới không title-only**, tính trên query có candidate còn lại. Chỉ là khởi tạo khoảng 25% query đủ candidate (≈300 nếu đủ 1.200, có thể nhiều hơn do ties). Chưa có số raw logit để đưa τ_en số cụ thể. Dùng raw logit max theo chunk, không sigmoid thành xác suất và không mượn phân bố cache zh cũ.
- Mỗi doc thêm **một chunk max-logit** theo candidate. Ưu tiên đo doc precision riêng qua Doc F2/R, đo chunk utility qua Chunk F2/R. Vòng S6 sau giữ max1; chỉ cân nhắc max2 khi phần mở rộng chứng minh lợi ích doc và chunk, tính cả bất định. Nhiều chunk/doc cần đo lợi ích recall/byte, chưa thay mặc định pipeline.

**Lịch:** tối đa 3 lượt/ngày theo tổng lượt Vu duyệt. Nếu S6-d/S6-a dùng 2 lượt hôm nay, còn tối đa 1 probe. K80 có điểm đã đo nên không cần nộp lại base để có mốc. Khi Vu gửi điểm S6, chọn bản tốt hơn và pin SHA mới; sinh lại cả cặp, tính lại số doc chọn/byte/forecast. Không trừ điểm hai baseline khác nhau để suy precision.

## 4. Giải ngược precision doc — giới hạn của macro

K_i = số ID predicted duy nhất; G_i = số gold doc; T_i = số hit; F_i = 5T_i/(4G_i+K_i). Thêm đúng một ID mới với y_i ∈ {0,1}:

```text
ΔF_i = (5y_i − F_i)/(4G_i + K_i + 1)
ΔR_i = y_i/G_i
N ΔDocF2 = Σ_selected (5y_i − F_i)/(4G_i + K_i + 1)
N ΔDocR  = Σ_selected y_i/G_i
```

N=1.200 **chỉ khi macro tính trên toàn bộ query hoặc mỗi query có gold**. Scorer local mặc định `has_gold`: nếu LB bỏ query không gold, phải dùng N_doc thực và A chỉ trên tập được chấm. N_chunk có thể khác N_doc. Hiện chưa có gold/private denominator để xác nhận.

Với A query có thêm ID, precision thường p = Σy_i/A **không xác định duy nhất** từ hai tổng khi G_i/K_i/F_i khác nhau. ΔR đo precision có trọng số 1/G_i; chính xác khi có hit:

```text
p = (N/A) × ΔDocR × harmonic_mean(G_i của các query có hit mới)
```

Để ước lượng điểm, giả định G_i=g chung, K_i=k chung (hoặc thay bằng mean K_selected là xấp xỉ), và baseline F_selected=f có thể thay bằng 0,2446 **chỉ dưới giả định tập chọn có cùng chất lượng với toàn bộ query**:

```text
g = [N ΔDocF2 (k+1) + A f] / [N (5ΔDocR − 4ΔDocF2)]
p = N g ΔDocR / A
```

Không clip: g≤0, p ngoài [0,1], mẫu số≤0 hoặc gần0 ⇒ giả định/độ phân giải không hỗ trợ ước lượng. `invert_doc_precision()` đã kiểm bằng dữ liệu đồng nhất biết trước (N=1200,A=600,G=50,K=100,f=0,3,p=0,2), thu lại đúng G/p.

Điểm 4 chữ số: mỗi hiệu giữa hai điểm làm tròn có sai số tối đa **±0,0001**. Chạy lại inverse trên biên sai số, các f_selected hợp lý và dải G; nhiều biên không hợp lệ thì báo không đủ độ phân giải. Sai số này chưa bao gồm heterogeneity và không là CI thống kê. Với A nhỏ, rủi ro lớn. Nếu chỉ có Final và Chunk F2, có thể lấy ΔDocF2=2ΔFinal−ΔChunkF2 nhưng sai số làm tròn lớn hơn; **vẫn cần ΔDoc R** để giải hệ trên. ΔDoc F2 + ΔChunk F2 riêng không đủ xác định precision doc.

Hòa vốn doc dưới giả định đồng nhất: p > f/5 ≈ **4,892%** với f=0,2446. Đây không phải ngưỡng universal cho tập query được score-select; ngưỡng đúng phụ thuộc F_i và trọng số mẫu số.

### Chunk không dùng cùng công thức doc

Scorer local `src/r2ai/eval/scorer.py` tính match nhiều-nhiều: một predicted chunk có thể cover nhiều gold; predicted match có thể không tăng gold coverage. Đặt H = số pred match, B = số gold đã cover, C = số pred, Gc = số gold chunk:

```text
F2 = 5 H B / (4 H Gc + B C)
F2' = 5(H+h)(B+b) / [4(H+h)Gc + (B+b)(C+m)]
```

h là số pred thêm match, b là số gold **mới** được cover, m là số pred thêm. ΔChunk F2 không suy ra precision doc/chunk đơn nhất; chỉ dùng đánh giá utility. Tỷ lệ Vu đo Pdoc đáng 1,9×Rdoc và Rchunk đáng 4,5×Pchunk được dùng định hướng ưu tiên, không suy ra absolute slope. F2 của macro P/R không bằng macro F2.

## 5. Forecast trước khi gửi Vu

Hiện **chưa có forecast hiệu chỉnh theo logit hay số doc chọn thật**. Bảng dưới là kịch bản minh họa A=300, Gdoc=Gc=60, K/C=mean K80, fdoc=0,2446; Pchunk/Rchunk macro chỉ làm xấp xỉ local; xác suất chunk match khi doc hit=0,5 và một gold mới/match. Không dùng các số này như kỳ vọng đo được:

| p_doc giả định | ΔDoc F2 | ΔDoc R | ΔChunk F2 | ΔFinal |
|---:|---:|---:|---:|---:|
| 0.05 | +0.000004 | +0.000208 | -0.000108 | -0.000052 |
| 0.10 | +0.000169 | +0.000417 | -0.000011 | +0.000079 |
| 0.20 | +0.000500 | +0.000833 | +0.000184 | +0.000342 |
| 0.40 | +0.001163 | +0.001667 | +0.000574 | +0.000868 |

`forecast_sensitivity_A300.csv` gồm 216 kịch bản: Gdoc/Gc ∈ {20,60,120}, p_doc ∈ {0,05;0,10;0,20;0,40}, q_chunk_given_doc ∈ {0,25;0,5;0,75}, gold mới/match ∈ {1,2}. Khoảng ΔFinal minh họa **−0,000086 đến +0,003570**, không là khoảng tin cậy. Candidate có thể tệ hơn phạm vi p đã chọn. Khi có file, `forecast_approval.py` dùng K/C từng query selected, vẫn ghi rõ giả định p/G/chunk và chưa calibrated.

Gói duyệt bắt buộc: file ZIP thật + JSON/ZIP SHA, baseline/candidate/membership SHA, τ raw, A/số chunk/tỷ lệ query không thêm, policy alias/title-only, coverage index, byte thực, kết quả validator, forecast theo kịch bản và sai số làm tròn. Vu duyệt trước mỗi lượt; generator không có lệnh upload.

## 6. Ngân sách byte cho S6

Dùng cap bảo thủ **98.999.999 byte** (<99 MB thập phân); chưa đo byte tăng của en/zh thật. Giữ baseline cho probe, dự phòng 1.000.000 byte:

| Phần | Ngân sách đề xuất (byte) |
|---|---:|
| vi — toàn bộ ZIP K80 hiện tại | 78.734.571 |
| en — headroom tăng tạm chia | 9.632.714 |
| zh — headroom tăng tạm chia | 9.632.714 |
| dự phòng | 1.000.000 |
| tổng | 98.999.999 |

Đây là ngân sách incremental mềm, **không phải cộng ba kích thước ZIP độc lập**. LZMA liên đới giữa text/ID/ngôn ngữ; đo ZIP hợp nhất thực. Một-doc/một-chunk probe có thể dùng ít hơn nhiều headroom. Sau LB chia phần tăng cho ngôn ngữ có ΔFinal/byte tốt hơn, xét uncertainty và mục tiêu recall chunk; corpus20/20 không chứng minh utility20/20. Khi chuyển S6 baseline, H=97.999.999−ZIP(base mới), chia lại H và bảo đảm cap thực. H<0 ⇒ không tạo probe với budget này, cần Vu chốt thay baseline/budget; không cắt vi tự động.

## 7. Bộ sinh, kiểm thử, bàn giao

- `generate_probe.py`: đọc đủ 1.200 qid, schema/rank/logit finite, membership/lang và hash; global chunk ID → local qua manifest `global_start`; kiểm group/alias; chunk phải là substring nguyên văn trong `docs.parquet.doc_text`; bảo toàn prefix/thứ tự mọi array base; đọc lại ZIP và SHA trước publish; cap fail xóa đúng temp đã guard. Input base/candidate/corpus phải giữ hash trong lúc build.
- ZIP một JSON, ZIP-LZMA dictionary64 MiB (khớp base). Không dùng `compresslevel` để giả định preset LZMA. Không sửa pipeline/config mặc định; mọi script này nằm output ignored.
- `test_generate_probe.py`: **11 passed**, log `pytest_result.txt`. Candidate giả đúng 9 cột S11; kiểm ngưỡng equality, next-new-doc, remap alias, thiếu query/schema/NaN/null/outside alias, source text không nguyên văn, manifest/base SHA, cap, inverse và forecast, policy en title-only.
- Synthetic duy nhất được ghim ở `out/diag-lang/v2/synthetic/SYNTHETIC_TEST_ONLY.zip`; không nộp. 1.200 query, 1.199 doc/chunk thêm; ZIP SHA `21ba3efe0668594a70a7098f40e8cc28003794c4916898e9660a859220cda7d5`, JSON SHA `132b7799697c1a3ad3f8fa5d7d5d0120671dc3633a186871dc1a6863497ee3e5`. Đây **không phải probe-zh thật**.
- Ví dụ lệnh khi S11 bàn giao (chưa chạy):

```powershell
$env:PYTHONPATH='C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/src'
$env:R2AI_OLD_ROOT='D:/GitHub/r2ai-stage3-old'
D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe out/diag-lang/v2/generate_probe.py `
  --base D:/GitHub/r2ai-stage3/out/runs/s9/sub_K80_LZMA_v2.zip `
  --expected-base-sha256 91fd6049245313dde4a7d42c8f86bae851acb43cf1aa7f6ccc8fd8e73b373a44 `
  --candidates <DUONG_DAN_S11_VU_BAN_GIAO> `
  --corpus D:/GitHub/r2ai-stage3/data/corpus_v2_ids.parquet `
  --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet `
  --chunks-root D:/GitHub/r2ai-stage3/data/chunks_zh_v2 `
  --segment-manifest D:/GitHub/r2ai-stage3/data/index/zh_v2_segments/manifest.json `
  --lang zh --tau 0.96 --coverage-docs 77523 --corpus-docs 240000 `
  --dst out/diag-lang/v2/probe-zh.zip
D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe out/diag-lang/v2/forecast_approval.py `
  --approval out/diag-lang/v2/probe-zh.approval.json --dst out/diag-lang/v2/probe-zh.forecast.csv
```

Cần Vu bàn giao tiếp: candidate S11 đầy đủ, artifact/mapping S10, điểm và file S6 nếu tốt hơn. Không tự thay đầu vào hay nộp probe.
