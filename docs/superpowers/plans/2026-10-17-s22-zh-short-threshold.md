# S22 follow-up: ngưỡng zh theo chữ Hán

## Mục tiêu và giới hạn

- Đánh giá raw HTML/shard đã lưu, không truy cập mạng hoặc crawl.
- Thực hiện trong worktree `codex/zh-short-threshold`; giữ nguyên checkout dùng chung.
- Không ghi index. Stage bundle riêng để S15 nhập theo CC contract hiện có.
- Giữ quy tắc vi ≥200 code point; với zh chấp nhận ≥200 code point hoặc ≥50 chữ Hán CJK.

## Kết quả kiểm chứng

### Audit nội dung

- Đếm Han CJK unified/compatibility ideographs, gồm các extension, trên body của doc `thin` có raw shard tồn tại.
- Histogram 12.665 doc có raw: bin 0 = 0; 1–49 = 1.698; 50–99 = 4.183; 100–199 = 6.784. Có 10.967 doc trong bin 50–199.
- Lấy mẫu tất định tối đa 10 doc mỗi host; 60/60 mẫu ở sáu host có nội dung y khoa. Ask.39 có question và answer thật ở 10/10 mẫu. Bin 0 của ask.39 là 0/10.317, nên không cần mẫu raw bổ sung.
- Bản sửa ask.39 từ C3 được tìm thấy tại commit `8f899a2`; `unnest_sele_txt` giữ cấu trúc và thứ tự Q&A.
- Báo cáo đầy đủ theo host: `reports/s22_zh_short_threshold.md`; metadata mẫu: `reports/s22_zh_short_samples.csv`.

### Threshold và segmentation

- Thêm helper đếm CJK và eligibility theo ngôn ngữ. Vi giữ threshold 200 code point; zh giữ điều kiện cũ hoặc đạt 50 chữ Hán.
- Trong 10.967 doc mới đạt threshold, 825 có dưới 50 BGE-M3 token (ask.39: 760; cnkang: 65). Đặt min-token zh = 1; vi vẫn = 50.
- Re-segment 10 CC batch zh trong bundle riêng `data/s22_zh_short_threshold`, cập nhật manifest. Kết quả toàn bộ batch zh: 12.343 doc / 35.479 chunk t256; 16 doc Youlai cũng đạt threshold mới.
- 22.238 doc và các segment trong bundle có 100% ID thuộc v2; chunk text khớp nguyên văn docs. Không ghi index chung.
- CC schema giữ nguyên; `source='reextract-raw'`; `raw_shard` giữ đường dẫn shard gốc. Mỗi host là một batch S15; segment IDs provisional.

### Kiểm thử và bàn giao

- Regression tests: CJK/extension counting, threshold zh, giữ threshold vi, token floor theo ngôn ngữ, status doc và chunker filter.
- Lệnh tập trung: `PYTHONPATH=src python -m pytest tests/test_s22_zh_short.py tests/test_s22_raw_recovery.py tests/test_vicrawl_extractors.py tests/test_chunk.py -q -k 'not test_sample_raw_pages' --basetemp .scratch/zhshort-final`.
- Kết quả: 76 passed, 3 deselected (ba fixture raw không có sẵn).
- Bundle staging là artifact ignored trong worktree, không đưa vào commit; chỉ stage allowlist mã nguồn, tests, báo cáo và tiến độ.

## Checklist

- [x] Kiểm kê raw và đo histogram theo host.
- [x] Phân loại mẫu 50–199; kiểm tra Q&A ask.39.
- [x] Đặt ngưỡng zh 50 chữ Hán; giữ vi 200 code point.
- [x] Đo min-token losses và đặt floor zh riêng.
- [x] Re-segment batch zh, cập nhật manifest và xác minh v2 membership.
- [x] Chạy test tập trung và cập nhật báo cáo / `docs/PROGRESS.md`.
