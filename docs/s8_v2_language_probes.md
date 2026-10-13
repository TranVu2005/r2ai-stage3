# S8 — gói duyệt Z1-A / Z2-A

2026-10-13 20:50 UTC+7 · GPT-6 (Codex) · HEAD kiểm trước commit `f7f656493a033b987c3220a89d65f62657bb1f0a`, nhánh `diag-lang`.

## Base và trạng thái

Base **S6-a K80**, ZIP-LZMA dict 8 MiB, ZIP 101.131.654 B; SHA ZIP `e58a53bbf5404084608bfe96f4657c665871bce72a55900639db31b455f1b430`, JSON `799b16358508113e79b76e04abab70b903fd10fd6b2f88c588d23d6859070e69`. Điểm **LB thật do Vu cung cấp**: Final **0,2483**; Doc F2 **0,2594**, P0,1563/R0,3726; Chunk F2 **0,2371**, P0,2056/R0,2976. Không dùng proxy.

Z1 giữ toàn bộ chunk base, thêm ZG-v2 logit≥0,96/cap 10 và cả nhóm ID v2. Z2 giữ doc Z1, thêm nguyên `docs.parquet.doc_text` của rank1 đủ ngưỡng, một chunk/ID v2, bảo vệ doc. Cả hai thêm **3.601 doc/ID** (3,000833/query), 538 query không thêm doc. Index coverage **77.523/240.000 =32,30125%**; candidate 60.000 rows / 1.200 query / top 50 SHA `b7a03d58e6087f7e888f57ee7ff2520258770286ddccf29d5e0b8a8b3f345ed3`.

| File duyệt | ZIP byte | Còn dưới trần | Chunk zh thêm | Chunk zh bỏ |
|---|---:|---:|---:|---:|
| [Z1-A.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/prebuild/Z1-A.zip) | 101,149,603 | 721,359 | 0 | 0 |
| [Z2-A.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/prebuild/Z2-A.zip) | 101,862,687 | 8,275 | 589 | 73 |

Native validator **0 lỗi**, đủ 1.200 query, toàn bộ ID thuộc v2, dedupe/near/verbatim kiểm đầy đủ; source adapter chỉ đọc vi/CC B docs và zh S11. Base vi chunk được kiểm nguyên văn qua sourceB, new zh text qua source `doc_text`; CRC/JSON SHA giải nén khớp. Đã so **doc arrays Z1=Z2**, mọi chunk vi giữ nội dung/thứ tự. **21 test giả pass** trước chạy artifact thật. Không GPU/crawl/nộp/push.

## Header và nén

Base, Z1-A, Z2-A cùng **props `5d00008000`, dict_size 8.388.608, lc/lp/pb 3/0/2**, ZIP-LZMA version 9/4. Z1 dùng encoder base. Z2 dùng preset `9 | PRESET_EXTREME` đặt dict 8 MiB tường minh theo Vu; effort không ghi trong props nên header vẫn giống. JSON compact chỉ bỏ khoảng trắng ngoài chuỗi; chunk_text/rank/schema nguyên vẹn.

Z2 encoder base: **101.969.237B** (dư98.275B); compact: **101.945.561B** (dư74.599B). Preset9 extreme với đủ 662 chunk đạt **101.943.269B** (dư72.307B). Sau fallback đã duyệt, file cuối và số chunk bỏ ở bảng. Đã bỏ nguyên chunk zh theo top1logit tăng dần (tie query_id tăng dần), không cắt text và không đổi phần vi/doc Z1. Audit `Z2-A.approval.json`/`Z2-A.attempts.json` ghi danh sách/thông số; 73 query bỏ có top1 logit từ0,962890625 đến1,185546875; file temp B đã dừng, không phát hành ZIP B.

## Sensitivity ΔF2 — không phải điểm zh đã đo

N=1.200, Gdoc=Gc=60, p_doc=p_chunk theo từng dòng; baseline macro F/P/R thay cho giá trị trên tập selected; β=1 gold **mới** cover mỗi matching chunk. K/m/C và số chunk giữ lấy từ artifact thật. Không là CI hoặc xác suất đã calibrated:

| Precision giả định | ΔDoc Z1/Z2 | ΔChunk Z1 | ΔFinal Z1 | ΔChunk Z2 | ΔFinal Z2 |
|---:|---:|---:|---:|---:|---:|
| 3% | -0.000815 | 0 | -0.000408 | -0.000214 | -0.000515 |
| 6% | +0.000303 | 0 | +0.000151 | +0.000015 | +0.000159 |
| 10% | +0.001793 | 0 | +0.000897 | +0.000322 | +0.001057 |
| 15% | +0.003656 | 0 | +0.001828 | +0.000704 | +0.002180 |
| 25% | +0.007382 | 0 | +0.003691 | +0.001470 | +0.004426 |

Kèm CSV `.doc/.chunk/.sensitivity` của `forecast-A-pair`: p3/6/10/15/25%, Gdoc 20/60/120, Gc 20/60/120, β 1/2/4. Gc20 với P/R local giả định cho H>B ở baseline một chunk/ID nên đánh dấu không hợp lệ ở model nhiều-nhiều, không đưa con số đó vào range. Cột chunk one-to-one chỉ là tham chiếu theo F2/5; full-body không có ngưỡng4,742% universal.

### Công thức và inverse sau LB

```text
Doc: ΔF_i=(5 y_i−m_i F_i)/(4G_i+K_i+m_i), ΔR_i=y_i/G_i
Dự báo: ΔDocF2=(1/N)Σ m_i(5p−F_i)/(4g+K_i+m_i)
Inverse (common g,p): p=N g ΔDocR/Σm_i; giải g số trong công thức ΔDocF2
Chunk: F2=5HB/(4HGc+BC); h=m p_chunk, b=βh
ΔChunkF2=mean[F2(H+h,B+b,C+m)−F2(H,B,C)]
```

N=1.200 là giả định macro chấm toàn bộ query hoặc mọi query có gold. Nếu LB dùng has_gold và loại query không gold, cần N_doc/N_chunk thật. G/P/F selected và gold distribution không có từ LB aggregate; inverse precision doc là estimate theo common-gold/common-p, không là macro precision mới xác định duy nhất. m_i khác nhau phải dùng inverse multi-ID; hiện ID/doc=1 trong nhóm đã chọn. Chunk many-to-many cần giả định β/Gc; không suy duy nhất precision chunk từ ΔChunkF2. Doc hoà vốn nominal **5,188%** với Fbase0,2594. Ngưỡng chunk **4,742%** chỉ cho mô hình one-to-one. Rounding4decimals: mỗi hiệu score±0,0001; common-gold và macro-substitution có bất định lớn hơn. Không dùng F2(macroP,R) như macroF2. Riêng Pchunk0,2056/Rchunk0,2976 cho F2(Pmacro,Rmacro)=0,273154, khác F2macro0,2371: model chunk chỉ xấp xỉ hiệu tăng trên baseline pseudo-local, không chứng minh mức tăng LB. Độ chênh này đã được giữ làm giới hạn của forecast, không gọi là điểm base mới.

## Phân loại query — kết quả giữ nguyên

[query_tags.csv](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/query_tags.csv) / [query_tags.parquet](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/query_tags.parquet) lưu ID, nguyên query, nhãn chính, mọi nhãn khớp, từ khóa chứng cứ và `needs_review`; [type_x_chapter.csv](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/type_x_chapter.csv) là bảng chéo. Generator không nhận taxonomy làm input, không lọc query theo nhãn/ngôn ngữ.

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



## SHA

### Z1-A

ZIP SHA256 `bd7d9dc844e00201ca62ae211eb775b9527c8c5d421d4e4632911c857aee8cce`

JSON SHA256 `a004f723a5b90a30d0cf9c3ef49c73de65f7b2da82aa67931dbe5e77bd979e94`

### Z2-A

ZIP SHA256 `71ad9650b765df0573b3b8637f59eec26a488bd4bbc2f80b11f8dc8a98766d88`

JSON SHA256 `57ba1f10126d6dac0d5e4d95dd3c8d1727e72d7b1b6a2561919fc55fa58de510`


## Đề xuất vòng S6 và ngân sách byte

τ_zh=0,96/cap 10 doc vật lý mỗi query cho ZG-v2, chunk tối đa rank1; τ_en=Q75 logit top1 sau bỏ title_only, đề xuất tối đa1 doc en/query kèm chunk nguyên văn. Ngưỡng en số cụ thể và byte en chưa đo, chờ S10. Không dùng taxonomy để lọc.

Cặp A dùng trần riêng đã duyệt101.870.962B. Base S6-a đã vượt trần99.000.000B của vòng S6 ban đầu **2.131.654B**, nên không thể chia vi/en/zh dưới99MB chỉ bằng append. Sau Z2-A chỉ còn8.275B ở trần riêng; không có ngân sách thực dụng cho en.

Nếu S6-d thắng và cấu hình zh có LB dương, baseB97.226.606B tạo headroom **1.773.394B** dưới99MB. Đề xuất envelope để S6 thử: phần base97.226.606B, zh≤800.000B, en≤873.394B, dự phòng100.000B. Envelope là kế hoạch **chưa đo trên B/en**; khoản zh dựa mức tăng cặpA731.033B, không suy nénB giốngA. Phải đo ZIP chung thật và kiểm cap; không cộng ZIP từng nguồn như số chính xác. Nếu B không thắng, S6 cần cấu hình nền vi nhỏ hơn hoặc Vu chốt lại trần, không tự sửa cặpA. Ưu tiên utility/byte sau LB; doc P≈1,9 lần R và chunk R≈4,5 lần P là ràng buộc Vu đã đo, không biến thành absolute slope mới.

## Duyệt và lượt sau

Gửi Z1-A trước, Z2-A sau cho Vu duyệt; agent **không nộp**. Tổng ≤ 3 lượt/ngày tính cả S6. S6-d chỉ khi điểm cao hơn S6-a mới dựng tương ứng, và chỉ cho cấu hình zh đã có kết quả LB dương. Probe-en chờ S10 fetch 14/10/2026; τ_en=Q75 raw top1 sau bỏ title_only, đã tạm duyệt. Taxonomy 1.200 query/rulebook vẫn chỉ phân tích lỗi, chưa đo accuracy, không lọc retrieval.
