# S11 — zh v2 (240.000 URL zh của corpus v2, D11)

Nhánh `feat/zh-v2`, worktree `D:/GitHub/r2ai-stage3-zhv2`. Cập nhật 2026-10-13 (UTC+7). Số liệu đo từ `state/zh_full/crawl.db` (mode=ro / sau backup), `data/corpus_v2_ids.parquet`, manifest ingest. Chưa push.

## 1. Bảng theo host zh của v2

Đơn vị: số **id v2** (1 id = 1 URL v2). `khác` = network_error, soft404_or_home, dead_origin, blocked_4xx. `chưa có` = id v2 không nằm trong `doc_ids` của bất kỳ dòng crawl.db nào. `trong index` = id v2 thuộc nhóm của doc đã chunk vào segment ACTIVE (gồm mẫu 2% và b001/b002).

Tổng: 240.000 id zh. ok 78.396 · thin 811 · pending 78.256 · deferred 883 · khác 217 · chưa có 81.437 · **trong index 77.526**.

| host | tổng | ok | thin | pending | deferred | khác | chưa có | trong index | ghi chú |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 120ask.com | 98.160 | 21.170 | 0 | 74.906 | 0 | 117 | 1.967 | 22.742 | robots 521 |
| cnkang.com | 40.046 | 35.859 | 22 | 3.350 | 0 | 16 | 799 | 32.934 | |
| ask.39.net | 32.772 | 0 | 0 | 0 | 0 | 0 | 32.772 | 0 | ngoài kế hoạch crawl |
| familydoctor.com.cn | 26.819 | 0 | 0 | 0 | 0 | 0 | 26.819 | 0 | ngoài kế hoạch crawl |
| zysjonline.com | 6.591 | 0 | 0 | 0 | 0 | 0 | 6.591 | 0 | ngoài kế hoạch crawl |
| zhongyibaodian.net | 6.061 | 5.671 | 261 | 0 | 0 | 0 | 129 | 5.730 | |
| a-hospital.com | 4.701 | 4.605 | 0 | 0 | 0 | 0 | 96 | 4.311 | |
| youlai.cn | 4.139 | 3.666 | 378 | 0 | 0 | 17 | 78 | 4.122 | |
| baby.39.net | 3.368 | 0 | 0 | 0 | 0 | 0 | 3.368 | 0 | ngoài kế hoạch crawl |
| test.pmphai.com | 2.989 | 2.865 | 62 | 0 | 0 | 0 | 62 | 2.907 | |
| jbk.39.net | 2.033 | 1.972 | 14 | 0 | 0 | 0 | 47 | 2.033 | |
| care.39.net | 1.282 | 1.256 | 0 | 0 | 0 | 0 | 26 | 1.282 | |
| msdmanuals.cn | 905 | 745 | 70 | 0 | 0 | 67 | 23 | 833 | |
| **wujue.com** | 708 | 36 | 0 | 0 | **661** | 0 | 11 | 46 | **hoãn** |
| jb39.com | 527 | 506 | 0 | 0 | 0 | 0 | 21 | 527 | |
| **zydcd.com** | 204 | 0 | 3 | 0 | **197** | 0 | 4 | 7 | **hoãn** |
| health.people.com.cn | 50 | 42 | 0 | 0 | 0 | 0 | 8 | 50 | |
| **qihuangzhishu.com** | 29 | 3 | 1 | 0 | **25** | 0 | 0 | 0 (D5, loại khỏi index) | **D5** |
| 25 host còn lại (39.net: heart, fk, nk, cancer, tnb, woman, food, ek, man, shen, gan, wei, sj, fitness, gk, pf, gc, byby; iiyi: article, bingli; baidu.com; familydoctor: baidianfeng, qy, ask, familydoctor.cn) | 8.616 | 0 | 0 | 0 | 0 | 0 | 8.616 | 2 | ngoài kế hoạch crawl |

Đầy đủ: `out/runs/zh-v2/host_table.csv` (`python -m r2ai.zh_v2.scope table`).

**Phát hiện cần Vu quyết:**
1. **81.437 id v2 (34%) không có trong crawl.db.** Phần lớn (78.166 id) thuộc host chưa từng nằm trong kế hoạch crawl full: `ask.39.net` 32.772, `familydoctor.com.cn` 26.819, `zysjonline.com` 6.591, `baby.39.net` 3.368 và 25 host nhỏ (39.net, iiyi, familydoctor, baidu; 8.616). Chúng cũng không có trong mẫu 2% (index 0 doc). Với crawl hiện tại, trần phủ zh của v2 ≈ 240.000 − 78.166 (host ngoài kế hoạch) − 883 (deferred) ≈ 160,9 nghìn id (67%), chưa tính 3.271 id nhỏ ở host đã crawl vốn đã nằm trong mẫu hoặc thiếu. Mở thêm các host này là crawl host mới → cần Vu duyệt (và kiểm robots) trước khi làm.
2. **Host hoãn / D5 có trong v2:** wujue 708 id (661 deferred, 36 ok), zydcd 204 (197 deferred), qihuangzhishu 29 (25 deferred; D5 loại). Tổng 883 id deferred. **Chưa mở lại host nào**; chờ Vu.
3. Phần "chưa có" ở host đã crawl (120ask 1.967, cnkang 799, ...) ≈ 2% là URL đã reuse từ mẫu 2% (không crawl lại, đã nằm trong segment `sample-v2`).

## 2. Giới hạn crawler về v2

- Crawler **đang dừng** từ 13:37 (stop_request pid 23596, exit 130), `watchdog.pause` còn nguyên; tôi **không** khởi động lại crawl.
- Backup: `state/zh_full/crawl-pre-zh-v2-1791877076.db`, SHA256 `450e572e5cad8931a15cfc5901e10eb05a0cf9d7dc4222564b17605efa01226a` (copy byte, so SHA nguồn = bản sao; WAL 0 trước khi chép).
- Công cụ: thêm `r2ai.selective.manage.defer_urls` / `defer-urls` (không có sẵn thao tác "defer danh sách"; `apply` từ chối vì `select_meta` đã có). Mỗi dòng chuyển sang `deferred_select` đồng thời ở `urls.status` **và** `select_meta.new_status` (lỗi 08/10: chỉ đổi một trong hai làm watchdog báo vi phạm). `rank`, `old_status` giữ nguyên nên `reopen`/`rollback` vẫn khôi phục được. Cổng: chỉ cột `status` của đúng 413.970 dòng đổi, mọi dòng/cột khác giống backup (so từng dòng); đối chiếu cuối: 0 dòng lệch `urls.status` vs `select_meta.new_status`; deferred 560.514 = 146.544 + 413.970.
- Dòng crawl = 1 URL chuẩn hoá chứa nhiều id; dòng có ≥ 1 id v2 (kể cả dòng lẫn id ngoài v2) **được giữ**. Chuyển sang deferred: 413.341 pending + 629 network_error chờ retry (120ask 289.541, cnkang 124.429). `ok/thin/…` cũ ngoài v2 giữ nguyên (không xoá cache).
- Còn lại: 120ask 74.898 pending + 102 network_error; cnkang 3.350 pending + 5 network_error; tổng `remaining` 78.355 dòng. Không đổi rate, không đụng robots.
- **ETA (ƯỚC TÍNH, chưa đo lại sau khi chạy):**
  - cnkang 3.355 dòng: tốc độ `ok` trước khi dừng 0,70/s (24 giờ), 0,52/s (12 giờ), 0,27/s (6 giờ) → **≈ 1,3–3,5 giờ** (tối đa ≈ 11 giờ nếu giữ 0,085/s của 3 giờ cuối).
  - 120ask 75.000 dòng: **không có ETA**. Lần `ok` cuối 12/10 06:09; robots.txt trả 521/timeout từ đó (`robots_last` = 521); crawler giữ `waiting_robots`, không vượt. Nếu robots trả 200 lại, rate cấu hình 0,5625 req/s (≈ 0,5 ok/s) → ≈ 42 giờ.
  - Deadline mặc định của crawler 20/10 23:59.

## 3. Index zh chỉ gồm doc v2

`python -m r2ai.zh_v2.index build --verify` (42 giây, CPU). Giữ chunk/embedding có sẵn: chunk thuộc doc có ≥ 1 id nhóm (`doc_ids_group`) thuộc v2 zh; dense fp16, CSR, text sao chép nguyên byte, giữ thứ tự dòng (doc liền nhau), cùng tên và `global_start` với segment nguồn, chỉ đánh số dòng lại; `row_map.parquet` cho ánh xạ mới → cũ. **Không embed lại.** qihuangzhishu loại (D5).

- 29 segment (30 ACTIVE trừ qihuangzhishu): **448.858 chunk, 77.523 doc** (từ 5.947.231 chunk nguồn của các segment đó). Cả 29 segment qua kiểm tra từng dòng: dense, doc_id, text, CSR bằng nguồn.
- Đường dẫn: `data/index/zh_v2_segments/` (manifest.json kèm SHA256 `corpus_v2_ids.parquet`), `data/chunks_zh_v2/<seg>/{chunks_t256,docs,row_map}.parquet`. Bản nguồn `zh_full_segments` không đổi.
- Doc v2 mới crawl (cnkang 3.350 + 120ask khi robots mở): chưa có. Khi có: `driver launch` cũ (extract → chunk t256 → embed), rồi chạy lại `r2ai.zh_v2.index build` (idempotent: segment đã lọc được bỏ qua, segment mới được lọc). Chunker t256 giữ nguyên, nên τ = 0,96 vẫn áp dụng.

## 4. ZG trên v2 và `artifacts/cand_zh_v2.parquet`

**Chưa đo.** Code đã viết (`r2ai.zh_v2.retrieve`: first-stage CPU → rerank GPU → export → zg), chưa chạy được vì first-stage CPU từ chối khi RAM trống < 2 GiB (đo 1,06–1,77 GiB; các phiên khác đang chiếm: PID 22180 `validate_submission` S6 4,1 GB). Không bỏ qua guard. Một vòng thử lại nền đang chạy. Rerank GPU ≈ 470.000 cặp ≈ 1,4 giờ (đo lần b001: 5.110 giây).

## Việc cần Vu quyết

1. Mở thêm crawl cho ask.39.net / familydoctor.com.cn / zysjonline.com / các host 39.net khác (81.437 id, 34% v2 zh)? Cần kiểm robots và duyệt rõ.
2. Mở lại wujue (661) / zydcd (197) / qihuangzhishu (25, D5)?
3. Cho chạy lại crawler (bỏ `watchdog.pause`) cho cnkang; 120ask đang bị robots 521.
