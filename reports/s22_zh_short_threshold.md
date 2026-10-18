# S22 follow-up — ngưỡng zh theo số chữ Hán

- Chỉ đọc raw HTML/shard đã lưu; không có request mạng. PubMed EN không thuộc phạm vi.

## Histogram và thu hồi

- CJK được đếm là Unicode Han CJK Unified/Compatibility Ideographs (kể cả extension); body đã extract, chỉ các doc `thin` của S22 có raw shard tồn tại. `0` là đúng 0 chữ Hán.
- Raw gốc còn ở `data/raw_zh_v2` cho ask.39; các host còn lại ở `data/raw_zh_full`. Tất cả raw shard của 12.665 doc trong histogram còn tồn tại.

| Host | Raw root | Thin có raw | 0 | 1–49 | 50–99 | 100–199 | Không raw (local failures) | Thu hồi mới (≥50 CJK) | Còn thin có raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ask.39.net` | `raw_zh_v2` | 10317 | 0 | 1359 | 3485 | 5473 | 0 | 8958 | 1359 |
| `cnkang.com` | `raw_zh_full` | 1558 | 0 | 233 | 489 | 836 | 547 | 1325 | 233 |
| `a-hospital.com` | `raw_zh_full` | 383 | 0 | 0 | 163 | 220 | 7 | 383 | 0 |
| `test.pmphai.com` | `raw_zh_full` | 98 | 0 | 0 | 34 | 64 | 1 | 98 | 0 |
| `zhongyibaodian.net` | `raw_zh_full` | 180 | 0 | 6 | 68 | 106 | 17 | 174 | 6 |
| `jbk.39.net` | `raw_zh_full` | 29 | 0 | 0 | 4 | 25 | 0 | 29 | 0 |
| `120ask.com` | `raw_zh_full` | 100 | 0 | 100 | 0 | 0 | 32 | 0 | 100 |

Tổng histogram: **12,665** doc raw-backed; bin 50–199 có **10,967** doc. Trong bảy host, **604** lỗi ban đầu không có raw; sau ngưỡng mới còn **2,302** lỗi (raw-backed dưới 50 cộng no-raw).

## Mẫu 50–199

- Lấy mẫu tất định SHA-256 theo `(host, doc_id)`, tối đa 10 doc/host; xem ID và nhãn trong `reports/s22_zh_short_samples.csv`. Có **60** mẫu ở 6 host; 120ask không có doc trong bin 50–199.
- ask.39: **10/10** có cả trường question và answer với nội dung lâm sàng; giữ cấu trúc Q&A.
- Bản C3 ask.39 đã tìm thấy tại commit `8f899a2`; `unnest_sele_txt` giữ thứ tự các đoạn trả lời lồng nhau.
- cnkang, a-hospital, test.pmphai, zhongyibaodian, jbk: **50/50** có nội dung y khoa thực (bài/định nghĩa/triệu chứng/excerpt), không phải menu-only. Một số mẫu có footer bình luận, paywall hoặc link “详细>>” bên cạnh nội dung.
- ask.39 bin 0 = **0/10.317**, không chiếm đa số; không cần lấy thêm 5 raw zero-text để phân loại JS shell/xóa/selector.

## Ngưỡng, chunker và segment

- Điều kiện text tốt zh: giữ quy tắc cũ ≥200 code point **hoặc** thêm ≥50 chữ Hán CJK; vi vẫn ≥200 code point.
- Chunker lọc `status=ok` và `n_tokens_bge_m3 >= min_body_tokens`. Trong 10.967 doc của bảy host đạt ≥50 CJK, **825** có token <50 (ask.39 **760**, cnkang **65**, các host khác **0**); min zh được hạ 50 → **1**, vi giữ 50.
- Re-segment đủ 10 batch zh theo CC contract hiện có. Bundle độc lập: `data/s22_zh_short_threshold`; manifest `data/s22_zh_short_threshold/manifest.db`. Không ghi index chung.
- Thu hồi thêm trong bảy host đã audit: **10.967** doc. Áp ngưỡng thống nhất khi re-segment toàn zh còn có **16** doc Youlai được nâng status; stage có **12.343 doc / 35,479 chunk t256** trong segment zh.
- CC schema/provenance được giữ nguyên (`source=reextract-raw`, `raw_shard` giữ đường dẫn shard). Kiểm tra membership: **100% doc ID thuộc corpus v2** trên 22.238 doc bundle và các segment; chunk text verbatim đạt 100%. `data/index` và `state/cc_longchau` không được ghi.

## Contract S15

- Mỗi host là batch `cc-s22-{host}`; schema CC, manifest và segment chunk t256 giữ nguyên; segment IDs là provisional để S15 cấp ID/embed khi nhập.
- Các doc <50 CJK vẫn thin. Raw không có cho các lỗi ban đầu được đếm riêng; không tự đi crawl.

## Kiểm tra tính tất định và nguyên nhân chênh hash

- Re-extract **100/552** doc hash khác, phân tầng theo host (familydoctor 24, jbk 24, youlai 24, zhongyibaodian 24, wujue 4), dùng raw shard từ provenance. Chạy hai process độc lập trên cùng raw HTML với `PYTHONHASHSEED=1` và `777`; SHA-256 body: **100/100 trùng, 0 lệch**. Không dựng lại bundle.
- 552 hash khác giữa raw-recovery cũ và bundle short-threshold không phải nondeterminism được tái lập. `662f285` đổi predicate trong `_extract_recovery_doc` từ `good_text` (≥200 code point) sang `good_text_for_lang` (zh nhận thêm ≥50 CJK). Số đo xác nhận: toàn bộ 552 body mới nằm trong bin 50–199 CJK; provenance extractor đổi giữa `generic_recovery` và extractor riêng host. Breakdown: youlai 255, familydoctor 189, zhongyibaodian 65, jbk 39, wujue 4. Predicate mới khiến nhánh primary/fallback được chọn khác. Hai process cho hash trùng; chưa có bằng chứng phụ thuộc thứ tự dict/set hoặc chuẩn hóa whitespace. Không sửa extractor.
- Bundle cũ từng ở `C:\Users\Admin\.codex\worktrees\zh-short-threshold\r2ai-stage3\data\s22_zh_short_threshold`. Artifact registry hiện ghi worktree `zh-short-threshold` đã archived; thư mục bundle không còn ở checkout đó. `git check-ignore` xác nhận `data/` bị ignore. Archive snapshot lưu tracked và non-ignored untracked, còn ignored files phải được bảo toàn riêng; bundle ignored đã biến mất cùng checkout khi archive. Không có dấu vết cho thấy bundle bị ghi đè riêng hay bị sandbox giữ lại.
