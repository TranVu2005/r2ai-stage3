# S8 v2 — thiết kế probe en/zh và nhãn query

Cập nhật 2026-10-13 19:24 UTC+7 · GPT-6 (Codex) · HEAD kiểm trước commit `bac0aabfe954351dbfbaf67d46f544eb023ae783`, nhánh `diag-lang`.

**Trạng thái hiện hành (Vu duyệt sửa 13/10):** base phải là bản Final cao nhất giữa S6-a/S6-d, chờ Vu báo điểm tối nay và pin đúng ZIP đã nộp. **Chưa sinh Z1/Z2 thật trước khi có điểm.** Đã kiểm candidate S11 đầy đủ, chuẩn bị bộ sinh và 18 test giả pass. Không GPU/crawl/nộp/push. Thiết kế zh một-doc/một-chunk cũ bị thay bằng Z1/Z2 dưới đây; S8 v1/probe-no-* cũng đã bị thay thế. Output: `out/diag-lang/v2/` trong worktree `C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3`.

## 1. Baseline: chờ điểm S6; K80 là tham chiếu cũ

- Tham chiếu K80 trước phê duyệt mới: `D:/GitHub/r2ai-stage3/out/runs/s9/sub_K80_LZMA_v2.zip`; **không dùng làm base cho Z1/Z2 đã duyệt**.
- ZIP **78,734,571 byte**, SHA256 `3e2935437eacb68cc0d7e372badae5fbccf0d82e35f605239caea12f99ec0b4e`.
- JSON SHA256 `91fd6049245313dde4a7d42c8f86bae851acb43cf1aa7f6ccc8fd8e73b373a44` khớp bản K80 đã nộp theo Vu. 1.200 query duy nhất, **163,613 doc entry**, **95,999 chunk entry**; toàn bộ ID có nhãn vi trong membership v2. Nhãn ngôn ngữ lấy từ membership, không chạy detector text.
- K/query: trung bình 136.344167, min/max 99/204; chunk/query: 79.999167, min/max 79/80.
- Điểm **do Vu cung cấp, không có lượt đo riêng trên LB v2**: Final 0,2334; Doc F2 0,2446, P 0,1528, R 0,345; Chunk F2 0,2222, P 0,1955, R 0,2744. cc-A-cut115 Doc F2 0,2593 là tham chiếu khác cấu hình, không cộng vào forecast của K80.
- Query nguồn đọc OLD, không ghi: `D:/GitHub/r2ai-stage3-old/data/raw/query.parquet`, SHA `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08`; S11 dùng cùng query hash.
- Membership hiện hành: `D:/GitHub/r2ai-stage3/data/corpus_v2_ids.parquet`, SHA `dad50fb81c812a5c68386a2313f880db488b457e85707b92d7c478c29523374a`. File corpus v2 thực tế có 1.220.598 URL: vi 740.598, en 240.000, zh 240.000; tỷ lệ 60/20/20 là mô tả làm tròn, không dùng chia byte.

**S11 đã bàn giao:** `D:/GitHub/r2ai-stage3-zhv2/artifacts/cand_zh_v2.parquet`, SHA256 `b7a03d58e6087f7e888f57ee7ff2520258770286ddccf29d5e0b8a8b3f345ed3`. Đã đọc/kiểm: **60.000 rows, 1.200 query duy nhất, 50 doc/query**, schema S11, logit finite/thứ tự rank, tất cả alias v2 thuộc zh. Không dùng cache index cũ.

Coverage index: **77.523/240.000 doc zh = 32,30125%**, tổng manifest đã đối chiếu. Kết quả recall mới phản ánh index hiện có, không ngoại suy ×3,1; ΔF2 không là cận dưới. Giả thuyết Vu: lọc khoảng93% doc ngoài v2 giúp precision cao hơn ZG cũ; **precision tăng chưa đo**, tỷ lệ lọc corpus không phải nhãn relevance.

Ở τ=0,96/cap10 **trước đối chiếu base**: **3.601 doc / 3.601 ID** (3,000833/query), 538 query0 doc, 212 query chạm cap10. Với tập chọn này mọi nhóm giữ đúng1 ID trong v2; vẫn triển khai mở rộng cả nhóm nếu có nhiều ID. Z2 rank1 đủ ngưỡng: **662 query / 662 ID**, 624 source doc khác nhau.

Đã xác minh source text của rank1 trong **14 segment**, hash từng `docs.parquet` ở `z2_full_text_profile.json`. Nguyên `doc_text` cho662 chunk: **1.725.179 byte UTF-8** trước escape JSON/nén (median1083, min227, max74130 byte). **Chưa đo ZIP tăng**; không sinh ZIP để thử trước điểm base.

S10 chờ fetch **14/10/2026** và bàn giao artifact/mapping text. Probe-en tạm giữ τ_en=Q75 raw logit top1 doc mới sau loại `title_only=true`; chưa có ngưỡng số cụ thể. Policy title-only do Vu chốt. Probe-en chưa sinh.

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

## 3. Thiết kế Z1/Z2 đã duyệt

| File | Doc mới | Chunk mới | Mục tiêu |
|---|---|---|---|
| Z1 | ZG-v2: logit≥0,96, ≤10 doc/query; cả nhóm ID thuộc v2 | Không thêm, base chunk giữ nguyên | Precision zh ở quy mô≈3 doc/query; so tham chiếu ZG cũ≈6%/doc |
| Z2 | Z1, thêm ID bảo vệ chunk nếu cần | Nguyên `docs.parquet.doc_text` của **rank1**, chỉ khi logit≥0,96; một chunk cho mỗi ID v2 trong nhóm | Utility/precision chunk qua Z2−Z1 |

Vu đã xác nhận **nguyên doc_text**, gồm layout đầy đủ hiện có; không cắt title/answer/body và không dùng chunk t256 max-logit để thay thế. Top1 là rank1 trong candidate S11, không phải doc mới đầu tiên sau bỏ base. Mỗi ID sở hữu chunk phải có trong relevant_docs. Nếu bảo vệ làm Z2 có thêm ID so với Z1, báo số ID và ΔDoc riêng; không giả định ΔDoc=0 trước validator. Khi doc Z2 đã đầy đủ trong Z1, Doc F2 của hai bản giống nhau chính xác.

Z1 giữ rule S11: skip khi **ID chính** đã trong base; stable rank order; cap theo doc vật lý; append các ID nhóm chưa có, lọc membership v2. Không bỏ cả nhóm chỉ vì một alias đã có; không cắt nhóm sau cap theo ID. Giữ mọi array base làm prefix và không đổi thứ tự. Z2 tránh append chunk y hệt đã có, báo số thực thêm.

Cặp dùng **cùng một base S6 đã chốt theo Final cao nhất**; pin JSON/ZIP SHA và bằng chứng validator của đúng artifact. Nếu điểm bằng nhau, hỏi Vu chọn base. Chưa có điểm thì builder từ chối trước I/O. Không dùng K80 tự động hay tự chọn từ các file size-probe S6 đang có.

Tối đa3 lượt/ngày, tính các lượt S6 đã dùng. Nếu S6-a/S6-d đã dùng2 lượt, còn tối đa1; Vu quyết định lịch Z1 rồi Z2. Không nộp tự động. Gửi từng file + hash + byte + forecast để Vu duyệt trước lượt nộp.

## 4. Inverse doc cho Z1 và inverse chunk cho Z2 — giả định macro

### Tham chiếu phép thêm một ID — không áp dụng trực tiếp cho Z1

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

### Công thức Z1 khi mỗi query thêm số ID khác nhau

Đặt m_i = số **ID mới** query i, y_i = số hit trong m_i. Identity đúng từng query:

```text
ΔDocF2_i = (5y_i − m_i F_i)/(4G_i + K_i + m_i)
ΔDocR_i  = y_i/G_i
```

Không dùng inverse one-doc ở trên cho Z1. Với G_i=g chung, expected precision/ID p chung, K_i/m_i đo từ append diff, F_i lấy theo giả định chất lượng selected (thường thay bằng Fbase, không có gold để biết F_i):

```text
M = Σ m_i
p = N g ΔDocR / M
ΔDocF2 = (1/N) Σ m_i(5p − F_i)/(4g + K_i + m_i)
```

`zg_pair.invert_multi()` giải g bằng bisection, rồi p; kiểm miền hợp lệ và sensitivity sai số làm tròn±0,0001. Đây là estimate theo common-gold/common-precision, không phải xác định chính xác từ macro. Báo precision/ID riêng với doc vật lý; không suy xác suất “ít nhất một hit/nhóm” từ aggregate. **Tập Z1 hiện đo trước overlap có3601 ID=3601 doc**, nên có thể so đơn vị với≈6%/doc cũ; vẫn khác base/index/selected query, con số6% trước đó cũng là estimate.

### Inverse chunk có điều kiện

Z2−Z1 cô lập chunk metric. Với H_i/B_i/Gc_i không có gold, ΔChunk F2 không xác định đơn nhất precision mới. Phải giả định Gc và baseline P/R trên tập được thêm, cùng β = số gold **mới** cover/matching chunk. Với m_i chunk mới:

```text
h_i = m_i p_chunk
b_i = β h_i
ΔChunkF2 = mean[F2(H_i+h_i,B_i+b_i,C_i+m_i) − F2(H_i,B_i,C_i)]
```

`zg_pair.invert_chunk()` giải p_chunk số trên[0,1] khi cung cấp Gc/P0/R0/β. Dùng macro P/R như local là xấp xỉ, cần sensitivity. Body đầy đủ có thể cover nhiều gold nhưng cũng có matching chunk không tạo coverage mới; β không biết từ LB aggregate. Không lấy ΔChunkF2 để khẳng định precision doc. Không clip kết quả khi giả định vô nghiệm.

## 5. Dự báo trước lượt nộp

**Inverse chỉ dùng sau khi có điểm probe.** Trước LB, dùng công thức thuận với precision/gold giả định; không thể có forecast xác định chỉ từ logit hoặc việc lọc93% corpus.

Khi Vu gửi base S6 đã đo:

- Z1: tính ΔDoc F2 theo K_i/m_i thực; scenario p_doc∈{0,06;0,10;0,15;0,20}, Gdoc∈{20;60;120}. Mức0,06 là tham chiếu ZG cũ, chưa chứng minh mức mới cao hơn. **ΔChunk F2=0 chính xác** vì chunks không đổi.
- Z2: cùng doc Z1 nếu không thêm protection; ΔDoc F2=0 so Z1 khi doc diff0. Forecast ΔChunk F2 qua `chunk_delta`, scenario p_chunk∈{0,02;0,05;0,10;0,20}, Gc∈{20;60;120}, β∈{1;2;4}, cùng baseline P/R nếu Vu có. Không dùng P/R của K80 thay S6. Thiếu P/R thì ghi rõ chưa đủ đầu vào cho forecast chunk, hoặc báo sensitivity theo các cặp P/R giả định tương thích F2 (vẫn chưa calibrated).
- Ghi cả Z1−base, Z2−Z1, Z2−base; Final delta=(ΔDocF2+ΔChunkF2)/2.

Gói duyệt: ZIP thật, baseline/candidate/membership/source SHA, nén/dictionary khớp base, byte thực, doc vật lý/ID/chunk/query counts, coverage77.523/240.000, validator0 lỗi, forecast và giả định/rounding. Sau LB cần **Doc R** ngoài Doc F2 để inverse; Chunk P/R giúp sensitivity nhưng không xóa bất định nhiều-nhiều.

Các CSV forecast A300/K80 cũ giữ làm ví dụ lịch sử; **không phải dự báo Z1/Z2 đang được duyệt**. Hiện chưa có điểm base S6, nên chưa công bố số ΔF2 của hai probe thật.

## 6. ZIP và byte

Dùng **đúng phương pháp nén của ZIP base được Vu chốt**, không mặc định dictionary K80. `compression_profile()` đọc ZIP-LZMA properties/dictionary trực tiếp; writer dùng cùng ZIP-LZMA/filter-properties. Nếu handoff dùng encoder/options khác, cần metadata writer để khớp trước build; ZIP header không chứng minh mọi encoder option.

Cap bảo thủ **98.999.999 byte**, không resize baseline hoặc cắt doc_text tự động. Đo ZIP từng Z1/Z2 thật sau điểm base. Base đã vượt cap hoặc Z2 vượt cap ⇒ dừng, báo Vu; không đổi phương pháp nén/cắt text để lách yêu cầu. Các stats S6 local gồm file size-probe và nhiều cách nén; chỉ dùng đúng file đã nộp, không suy từ tên.

Ngân sách vi/en/zh 78,7/9,6/9,6 MB trước đó theo K80 đã bị base S6 thay thế. Recompute H=98.999.999−ZIP(baseS6); ưu tiên chỗ cho cặp zh đã duyệt, probe-en chờ14/10. Tăng byte của Z2 **chưa đo**, 1.725.179 byte chỉ là UTF-8 text thô. Không cộng ZIP riêng để tính headroom. Vòng S6 sau chia headroom dựa ΔFinal/byte đo được, tỷ lệ utility docP≈1,9×docR/chunkR≈4,5×chunkP là định hướng, không là absolute slope.

## 7. Implementation và kiểm tra

- **Generator hiện hành cho zh:** `out/diag-lang/v2/zg_pair.py`. CLI cần base-evidence JSON chứa `scores:{label,final,doc_f2,chunk_f2}`, `json_sha256`, `validator:{ok:true,errors:0,rows:1200,json_sha256}`. Evidence phải thuộc đúng base đã đo/đã validate; `--base-evidence` không thay việc Vu báo điểm hay duyệt nộp.
- Pin candidate SHA; kiểm full qid/schema/order/membership; validate source group/aliases/full text; hash14 source docs; guard mọi output/temp/cleanup; preserve base prefix, Z1 chunks identical, Z2 full-text exact và doc protection; streaming ZIP/readback SHA/byte. Z1 hợp lệ có thể được giữ nếu Z2 vượt budget để Vu xem, không tự nộp.
- Validator chia bằng chứng rõ: toàn bộ ID/chunk ownership/append diff đọc lại; new zh text kiểm trực tiếp source; base text kế thừa bằng chứng validator0 lỗi gắn SHA và byte/content không đổi. Nếu base thiếu bằng chứng validator thì không build; chạy validator base phù hợp vi/CC trước.
- `forecast_zg.py` đọc approval của cả cặp cùng SHA, tính scenario với K/m/C từng query; yêu cầu P/R của S6 hoặc giá trị giả định ghi rõ, không fallback K80. Chưa chạy trên dữ liệu thật trước điểm base.
- **18 test giả pass**, gồm 11 test cũ và7 test Z1/Z2: group/cap/stable skip, body mỗi alias, protection, inverse doc với m_i khác nhau, inverse chunk giả định, gate chưa có điểm, matching dictionary, cả cặp1200 query. File `pytest_zg_result.txt`. Các ZIP trong thư mục `zg-final/` là synthetic; **chưa sinh Z1.zip/Z2.zip thật**.
- Candidate profile: `candidate_zh_profile.json`; full-body/source SHA: `z2_full_text_profile.json`; kế hoạch: `zg_pair_plan.md`. Old `generate_probe.py` chỉ là bộ một-doc tham chiếu/en prototype, không dùng thay Z1/Z2.

Tiếp theo: Vu báo điểm S6-a/S6-d tối13/10 và bàn giao đúng ZIP/JSON SHA, nén và validator; pin bản tốt hơn, đo overlap và byte thực, tạo hai probe cùng base, tính forecast gửi Vu duyệt. Probe-en chờ S10 fetch14/10, τ_en=Q75 đã tạm duyệt. Không tự nộp/push/GPU/crawl.
