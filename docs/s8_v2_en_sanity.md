# E1 — sanity mapping en (CPU, nguồn chỉ đọc)

Cập nhật: 2026-10-15 08:08 (UTC+7) · GPT-6 (Codex) · HEAD trước commit `b8d9ca0171439c6fd82dc85c40fc8f6eded70714`.

Kết quả: **20/20** doc có chunk; **10/10** doc không có chunk. Seed42; lấy mẫu từ sorted unique doc_id, RNG reset riêng cho hai nhóm.
Population: 1768 unique EN docs; 405 có chunk; 1363 không có chunk ở bất kỳ query nào trong E1. Query-title5: random42 từ query có body en.

Tra toàn bộ 239771 record S10 bằng hash nội dung rồi so chuỗi đầy đủ; không dùng doc_id của builder để tìm record cùng text. Kiểm thêm doc_id thuộc record group, PMID source_url và cand_en của từng query xuất hiện. Pin5/5 kiểm trước/sau; E1 JSON SHA đúng bản đã nộp; không sửa input/pipeline.

## 20 doc có chunk

| doc_id | PMID URL | cand PMID | record PMID từ exact text | Khớp |
|---:|---:|---|---|---|
| 6961560 | [34786770](https://pubmed.ncbi.nlm.nih.gov/34786770/) | [34786770] | [34786770] | True |
| 5270953 | [14614196](https://pubmed.ncbi.nlm.nih.gov/14614196/) | [14614196] | [14614196] | True |
| 4918092 | [8325147](https://pubmed.ncbi.nlm.nih.gov/8325147/) | [8325147] | [8325147] | True |
| 7307814 | [38660895](https://pubmed.ncbi.nlm.nih.gov/38660895/) | [38660895] | [38660895] | True |
| 5662105 | [20433126](https://pubmed.ncbi.nlm.nih.gov/20433126/) | [20433126] | [20433126] | True |
| 5610671 | [19692116](https://pubmed.ncbi.nlm.nih.gov/19692116/) | [19692116] | [19692116] | True |
| 5538777 | [18661769](https://pubmed.ncbi.nlm.nih.gov/18661769/) | [18661769] | [18661769] | True |
| 5317450 | [15478742](https://pubmed.ncbi.nlm.nih.gov/15478742/) | [15478742] | [15478742] | True |
| 7300919 | [38547193](https://pubmed.ncbi.nlm.nih.gov/38547193/) | [38547193] | [38547193] | True |
| 5255721 | [12887118](https://pubmed.ncbi.nlm.nih.gov/12887118/) | [12887118] | [12887118] | True |
| 7066023 | [35729228](https://pubmed.ncbi.nlm.nih.gov/35729228/) | [35729228] | [35729228] | True |
| 6485148 | [30088776](https://pubmed.ncbi.nlm.nih.gov/30088776/) | [30088776] | [30088776] | True |
| 5217103 | [12387696](https://pubmed.ncbi.nlm.nih.gov/12387696/) | [12387696] | [12387696] | True |
| 6714315 | [32514397](https://pubmed.ncbi.nlm.nih.gov/32514397/) | [32514397] | [32514397] | True |
| 6110022 | [25822101](https://pubmed.ncbi.nlm.nih.gov/25822101/) | [25822101] | [25822101] | True |
| 4990901 | [9134944](https://pubmed.ncbi.nlm.nih.gov/9134944/) | [9134944] | [9134944] | True |
| 4982234 | [9032590](https://pubmed.ncbi.nlm.nih.gov/9032590/) | [9032590] | [9032590] | True |
| 5227257 | [12518131](https://pubmed.ncbi.nlm.nih.gov/12518131/) | [12518131] | [12518131] | True |
| 5531405 | [18564325](https://pubmed.ncbi.nlm.nih.gov/18564325/) | [18564325] | [18564325] | True |
| 5587097 | [19384786](https://pubmed.ncbi.nlm.nih.gov/19384786/) | [19384786] | [19384786] | True |

## 10 doc không có chunk

| doc_id | PMID URL | cand PMID | Khớp |
|---:|---:|---|---|
| 7361311 | [39547339](https://pubmed.ncbi.nlm.nih.gov/39547339/) | [39547339] | True |
| 5328160 | [15615817](https://pubmed.ncbi.nlm.nih.gov/15615817/) | [15615817] | True |
| 4892082 | [8056022](https://pubmed.ncbi.nlm.nih.gov/8056022/) | [8056022] | True |
| 5874687 | [23160077](https://pubmed.ncbi.nlm.nih.gov/23160077/) | [23160077] | True |
| 5758315 | [21765119](https://pubmed.ncbi.nlm.nih.gov/21765119/) | [21765119] | True |
| 5689361 | [20821354](https://pubmed.ncbi.nlm.nih.gov/20821354/) | [20821354] | True |
| 5424102 | [16945050](https://pubmed.ncbi.nlm.nih.gov/16945050/) | [16945050] | True |
| 5294675 | [15117584](https://pubmed.ncbi.nlm.nih.gov/15117584/) | [15117584] | True |
| 7078196 | [35837400](https://pubmed.ncbi.nlm.nih.gov/35837400/) | [35837400] | True |
| 5256494 | [12897470](https://pubmed.ncbi.nlm.nih.gov/12897470/) | [12897470] | True |

## 5 query để Vu xem bằng mắt

### Query 955

Loét bàn chân do tiểu đường có nguy hiểm không?

**Title:** Diabetic foot infection.
Doc ID 5537932 · PMID 18649613 · rank1 · logit 1.2724609375.

### Query 162

Tôi năm nay 45 tuổi, giới tính nam. Cách đây 2 năm tôi bị hội chứng thận hư và gout, hiện vẫn đang uống thuốc ngày 1 viên. Xin hỏi trường hợp của tôi có tiêm được vắc xin phòng Covid-19 không ạ?

**Title:** Risk of gout flares after COVID-19 vaccination: A case-crossover study.
Doc ID 7074059 · PMID 35797765 · rank1 · logit 2.0390625.

### Query 27

Đau khớp cổ tay, cổ chân và khớp ngón tay, ngón chân. Con khoai và được chuẩn đoán viêm túi dịch khớp gối đã được bác sĩ tại viện cho uống thuốc meloxicam nhưng đến nay vẫn đau và đi lại rất khó khăn vậy bác sĩ cho em thêm tư vấn.

**Title:** Meloxicam.
Doc ID 5217103 · PMID 12387696 · rank1 · logit 2.314453125.

### Query 1088

Vaccine COVID-19 AstraZeneca nên tiêm cho đối tượng nào? Giữa người cao tuổi và người trẻ tuổi, đối tượng nào tiêm vaccine sẽ an toàn hơn?

**Title:** The Impact of Age Difference on the Efficacy and Safety of COVID-19 Vaccines: A Systematic Review and Meta-Analysis.
Doc ID 6978319 · PMID 34938287 · rank1 · logit 2.642578125.

### Query 387

Con em 13 tháng bú mẹ hoàn toàn mà táo bón từ lúc 2 tháng. Hiện tại phân thành khuôn và rất to. Tuần ỉa 1 lần . Vẫn tự ỉa dc đầu phân như phân dê thì giờ em nên làm gì. Em có bổ sung men mà vẫn vậy . Cũng ăn nhiều rau mà vẫn không ăn thua.

**Title:** Evaluation of a 13-Month-Old Anemic Child With Gushing Gums.
Doc ID 7246417 · PMID 37644949 · rank1 · logit 0.974609375.

## LB và kết luận

C1→E1: Final0,2655→0,2648 (−0,0007 theo điểm hiển thị); DocF20,2687→0,2675 (−0,0012), P0,1567→0,1553, R0,3886→0,3886; ChunkF20,2623→0,2620 (−0,0003), P0,1857→0,1851, R0,3402→0,3402. Điểm do Vu cung cấp; R bằng nhau ở độ chính xác LB hiển thị, không chứng minh recall chính xác không đổi bên dưới làm tròn.

**en không có lời** trên probe E1 đã đo; sanity không phát hiện lệch mapping trong 20+10 mẫu. Không nộp E2; không dựng thêm probe en.

Chi tiết từng mẫu/hash/URL/query và pin nằm ở mapping20.json, mapping10-no-chunk.json, query-title5.json, stats.json cạnh report.

## Ghi chú thực thi

Parser kiểm phụ ban đầu chỉ nhận PubMed URL, nên dừng ở source_url Europe PMC. Đã đối chiếu code S10 europepmc.py:247: source_url=MED/{pmid}; thêm đúng parser cho URL nguồn này trong script phân tích, regression5/5 đạt (PubMed/MED hợp lệ, PMC/host lạ/ID chữ bị từ chối). Không có mismatch dữ liệu trong lỗi parser ban đầu; không sửa nguồn hay pipeline.

5 pin SHA nguồn, E1 ZIP/JSON, sampling population, query IDs và từng mapping được lưu trong `out/diag-lang/v2/probes-en/sanity-E1/` (ignored). Chỉ cập nhật report/decision; không thông báo S10 vì điều kiện mapping lệch không xảy ra. Không dựng probe, không nộp E2, không GPU/crawl/push.
