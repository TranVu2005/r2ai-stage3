# S24 recovery — measured status

Measured: 2026-10-18 04:16 (UTC+7). Branch `feat/s24-recovery`; pre-report HEAD `4af39ec2ffac4296c338134018798b1e6ed73348`.

## Ranh giới và kết quả theo host × nguồn

D21 bỏ qua robots theo yêu cầu. Direct dùng UA `r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)`, cùng egress IP, `trust_env=false`, không proxy, không theo redirect. Dừng ngay với 403/429/CAPTCHA. Baolangson log có khoảng cách request nhỏ nhất 1,000 giây.

**Đính chính trạng thái S23:** S23 đã final lúc 20:16 ngày 17/10; `vov.vn` có 1.392 doc từ CC. `waiting_robots` thuộc queue S21, không thuộc S23. Segment `seg20004-cc-s23-vi-news-20261017-vi-news` giữ 1.463 doc/8.327 chunk và checksum file không đổi. Chỉ 8.392 URL `vov.vn` còn thiếu được đưa vào S24; `vov2.vov.vn` loại khỏi crawl. Wayback đã dừng hẳn; `thaythuocvietnam.vn` được đánh dấu dead, không crawl.

| Host | Nguồn | URL mục tiêu / request | Hit | Text tốt | Payload / thời gian | Kết quả |
|---|---|---:|---:|---:|---|---|
| 120ask.com | Direct chẩn đoán cục bộ | 3 hostname; TCP 6; HTTP 12 (HEAD 6, GET 6) | 0 | 0 | 0 B; timeout | DNS trả 3 IP; TCP/HTTP timeout; chưa xác minh host mobile; S21 còn 75.410 pending. Không đủ bằng chứng kết luận chặn theo địa lý. |
| 120ask.com | Wayback CDX prefix | 98.160 corpus URL / 1 request | Không đo được (429) | — | 162 B; 2,234 s | `120ask.com/question/*` trả HTTP 429 lúc 01:03; dừng ngay, không thử prefix host khác, không tải snapshot. Ba exact probe trước đó cũng 429. |
| thaythuocvietnam.vn | Wayback CDX prefix | 2.927 corpus URL / 0 request | Không đo được | — | 0 B | Không truy vấn vì dừng toàn bộ CDX sau 429 của 120ask. |
| vov2.vov.vn | Wayback CDX prefix | 6.899 corpus URL / 0 request | Không đo được | — | 0 B | Không truy vấn vì dừng toàn bộ CDX sau 429 của 120ask. |
| baolangson.vn | Direct | 1.981 URL duy nhất / 1.984 attempt | 1.979 HTTP 200 (99,90%) | 1.978 ≥200 ký tự (99,85%) | 299.164.993 B; 68,97 phút | Mẫu 5/5 text tốt. 2 URL 302, không theo redirect; 1×502 và 2 disconnect được thử lại sau cooldown và phục hồi. Không gặp 403/429/CAPTCHA. |
| baolangson.vn | S15 | 1.978 URL có text tốt | 1.978 doc `ok` | 1.978 | raw shard nén 5.905.862 B; extract 78,0 s; chunk 5,09 s; encode 137,61 s | 20 shard; 14.942 chunk/vector; verbatim 14.942/14.942; segment `seg20005-s24-direct-baolangson-baolangson.vn` active. |
| zysjonline.com | Direct | 1 URL / 1 request | 0 | 0 | 5.645 B; 0,828 s | HTTP 403, Cloudflare “Just a moment”, state CAPTCHA; dừng ngay. |
| article.iiyi.com | Direct mẫu | 729 URL / 1 request | 0 | 0 | 0 B; ConnectTimeout 36,156 s | Dừng host sau lỗi kết nối; không full crawl. |
| bingli.iiyi.com | Direct mẫu | 5 URL / 1 request | 0 | 0 | 0 B; ConnectTimeout 36,125 s | Dừng host sau lỗi kết nối; không full crawl. |
| thaythuocvietnam.vn | Dead (chỉ report S24; không crawl) | 2.927 URL / 0 request | 0 | 0 | 0 B | Wayback/direct đều dừng; không có crawl mới. |
| vov.vn | Direct S24 | 8.392 URL / 8.392 attempt | 8.385 HTTP 200; 5×301 + 2×302 không follow | 8.364 ≥200 ký tự | 2.130.843.326 B; span 10.123,043 s (2h48m43s); min gap 1,200 s | Mẫu 5/5 tốt; không có 403/429/CAPTCHA; hoàn tất đủ mục tiêu. |
| vov.vn | S15 extract/provenance | 8.385 HTTP 200 / 8.385 doc | 8.385 doc + provenance row | 8.364 tốt; 21 thin | docs parquet 23.438.896 B; provenance parquet 1.170.244 B | Batch `s24-direct-vov`; `chunking=false`, `embedding=false`, `shared_index_written=false`; dữ liệu dưới `out/`. |
| vov2.vov.vn | Excluded | 6.899 corpus URL / 0 request | 0 | 0 | 0 B | Ngoài phạm vi theo yêu cầu. |

## Wayback CDX prefix recheck (2026-10-18 01:03 UTC+7)

| Host | Corpus v2 URL | CDX hit | Hit rate | Wilson 95% CI | Trạng thái |
|---|---:|---:|---:|---:|---|
| 120ask.com | 98.160 | Không đo được | Không đo được | Không tính được | Prefix đầu tiên HTTP 429; 1 request, 162 B, 2,234 s. |
| thaythuocvietnam.vn | 2.927 | Không đo được | Không đo được | Không tính được | Không truy vấn sau 429 toàn cục. |
| vov2.vov.vn | 6.899 | Không đo được | Không đo được | Không tính được | Không truy vấn sau 429 toàn cục. |

Đã gửi đúng một CDX prefix request đầu tiên, URL có `matchType=prefix`, `filter=statuscode:200`, `output=json`, `showResumeKey=true`; HTTP 429 trả HTML, không có record để ghép offline. Tuân thủ điều kiện dừng: không gửi request tiếp theo hoặc retry; không tải snapshot. Corpus URL counts lấy offline từ `data/corpus_v2_ids.parquet`, đếm hostname chuẩn hóa chữ thường; vì không có dữ liệu CDX thành công nên không báo hit rate/Wilson CI giả định. Chi tiết request/body/metadata: `out/runs/s24-recovery/wayback/prefix-recheck-20261018/`.

## webMedQA (Apache-2.0)

Tải ba ZIP: 74.896.426 B; 245.507 dòng, 245.505 hợp lệ, 2 malformed, 49.260 ID duy nhất. Đối chiếu ID chính xác với 98.151 URL số v2 cho 2.464 ID (2,51%). Trong 446 ID đã có text trong index, 433 câu hỏi đạt LCS ≥80% sau `normalize_metric`: **97,09%**, Wilson 95% CI **95,08–98,29%**; qua ngưỡng 90%.

Đã ingest 2.018 ID còn thiếu text theo điều kiện người dùng; ghép question + valid answers theo thứ tự nguồn, provenance ghi webMedQA/Apache-2.0/ID. Batch `cc-s24-webmedqa`: 2.018 doc, 10.785 chunk; private segment provisional `seg20006-cc-s24-webmedqa-120ask.com`, ID 20006, active. Payload vector được tái sử dụng, không encode lại trong lượt này; 11 embed blocks; mọi dữ liệu ở `out/runs/s24-recovery/s15` và `out/runs/zh-full/segments/s24-recovery`.

**Đính chính:** báo cáo trước ghi 3.166 ID overlap; tính lại từ các ZIP có SHA-256 khớp cho 2.464 ID. Số 3.166 không tái lập được và được thay bằng số đã kiểm chứng.

Dataset phụ tìm thấy: `zirui3/webMedQA-instructions` (metadata CC-BY-4.0, 316.110 dòng) chưa xác nhận schema ID/URL và không tải; crawler `Censyu/yp-120ask-spider` chỉ cho trang sản phẩm, không khớp corpus câu hỏi.

## Output, chi phí và việc còn chờ

Tất cả crawl, shard, provenance, docs, chunks và segment mới nằm dưới `out/`; index chung không được ghi. AWS cost: $0; chi phí băng thông local chưa đo. Raw Baolangson: 299.164.993 B nhận từ host qua các lần thử; shard S15 nén 5.905.862 B.

Wayback đã dừng hẳn sau prefix 120ask trả 429; không có hit rate/snapshot. `thaythuocvietnam.vn` dead, không crawl. iiyi chưa full crawl do ConnectTimeout. VOV direct đã xong theo mục tiêu S24; VOV2 bị loại. 120ask chưa đủ bằng chứng kết luận geo-block hoặc đề nghị VPS HK/SG.


## VOV direct S24 (2026-10-18 04:16 UTC+7)

Targets lấy offline bằng corpus v2 `vov.vn` trừ ID đã có trong S23 `seg20004`: 8.392 URL duy nhất. Attempt đủ 8.392; 8.385 HTTP 200 tạo raw shard/doc/provenance, 5 HTTP 301 và 2 HTTP 302 không follow, không có stop condition. Response body tổng 2.130.843.326 B; crawl span 10.123,043 s, khoảng cách request nhỏ nhất đo 1,200026 s. Sample 5/5 đạt ≥200 ký tự. Extraction generic tạo 8.385 doc: 8.364 `ok` ≥200 ký tự, 21 `thin`; không chunk, không encode, không ghi index chung. S23 manifest/file hashes kiểm lại: segment 20004 vẫn `seg20004-cc-s23-vi-news-20261017-vi-news`, active, 1.463 docs/8.327 chunks, hashes không đổi. Private webMedQA đã đổi sang provisional 20006, 2.018 docs/10.785 chunks, 11 embed blocks; payload vectors unchanged/reused. Request log, raw, docs, provenance, manifest nằm dưới `out/runs/s24-recovery/`. File mtime output cho biết extraction hoàn tất khoảng 4m47s sau request cuối; không có stopwatch riêng cho extraction.

## S15 handoff: s24-direct-vov

Đã bàn giao batch `s24-direct-vov` cho task S15: 8.385 doc từ `vov.vn` (8.364 `ok`, 21 `thin`), chưa chunk/embed ở S24. Input: `out/runs/s24-recovery/s15/docs/s24-direct-vov/vov.vn.parquet`; provenance: `out/runs/s24-recovery/s15/run/provenance/s24-direct-vov/vov.vn-direct.parquet`; manifest: `out/runs/s24-recovery/s15/run/batches/s24-direct-vov/manifest.json`. S15 xác nhận sẽ chunk t256 trong staging trước; embed theo thứ tự GPU đã giao: delta vi đợt 1 → delta zh (S22 + webMedQA 20006) → delta vi đợt 2 (baolangson + VOV). Không chạy encode từ S24. Tại thời điểm ghi nhận, S15 còn kiểm trạng thái delta/GPU và VOV chưa được xác nhận chunk/embed hoàn tất.
