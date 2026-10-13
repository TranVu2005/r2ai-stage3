# S11 — zh v2 (240.000 URL zh của corpus v2, D11 + D13)

Nhánh `feat/zh-v2`, worktree `D:/GitHub/r2ai-stage3-zhv2`. Cập nhật 2026-10-13 19:35 (UTC+7). Số liệu đo từ `state/zh_full/crawl.db`, `state/zh_v2/crawl.db` (mode=ro), `data/corpus_v2_ids.parquet`, `out/runs/zh-v2/`. Chưa push.

## 1. Bảng theo host zh của v2 (lúc bàn giao D11)

Đơn vị: số **id v2**. `khác` = network_error, soft404_or_home, dead_origin, blocked_4xx. `chưa có` = id v2 không nằm trong `doc_ids` của dòng nào trong crawl.db zh_full. `trong index` = id thuộc nhóm của doc đã chunk vào segment ACTIVE.

Tổng: 240.000 id zh. ok 78.396 · thin 811 · pending 78.256 · deferred 883 · khác 217 · chưa có 81.437 · **trong index 77.526** (phủ 32%).

| host | tổng | ok | thin | pending | deferred | khác | chưa có | trong index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 120ask.com | 98.160 | 21.170 | 0 | 74.906 | 0 | 117 | 1.967 | 22.742 |
| cnkang.com | 40.046 | 35.859 | 22 | 3.350 | 0 | 16 | 799 | 32.934 |
| ask.39.net | 32.772 | 0 | 0 | 0 | 0 | 0 | 32.772 | 0 |
| familydoctor.com.cn | 26.819 | 0 | 0 | 0 | 0 | 0 | 26.819 | 0 |
| zysjonline.com | 6.591 | 0 | 0 | 0 | 0 | 0 | 6.591 | 0 |
| zhongyibaodian.net | 6.061 | 5.671 | 261 | 0 | 0 | 0 | 129 | 5.730 |
| a-hospital.com | 4.701 | 4.605 | 0 | 0 | 0 | 0 | 96 | 4.311 |
| youlai.cn | 4.139 | 3.666 | 378 | 0 | 0 | 17 | 78 | 4.122 |
| baby.39.net | 3.368 | 0 | 0 | 0 | 0 | 0 | 3.368 | 0 |
| test.pmphai.com | 2.989 | 2.865 | 62 | 0 | 0 | 0 | 62 | 2.907 |
| jbk.39.net | 2.033 | 1.972 | 14 | 0 | 0 | 0 | 47 | 2.033 |
| care.39.net | 1.282 | 1.256 | 0 | 0 | 0 | 0 | 26 | 1.282 |
| msdmanuals.cn | 905 | 745 | 70 | 0 | 0 | 67 | 23 | 833 |
| wujue.com | 708 | 36 | 0 | 0 | 661 | 0 | 11 | 46 |
| jb39.com | 527 | 506 | 0 | 0 | 0 | 0 | 21 | 527 |
| zydcd.com | 204 | 0 | 3 | 0 | 197 | 0 | 4 | 7 |
| health.people.com.cn | 50 | 42 | 0 | 0 | 0 | 0 | 8 | 50 |
| qihuangzhishu.com | 29 | 3 | 1 | 0 | 25 | 0 | 0 | 0 |
| 25 host nhỏ (39.net ×18, iiyi ×2, baidu.com, familydoctor ×4) | 8.616 | 0 | 0 | 0 | 0 | 0 | 8.616 | 2 |

Đầy đủ: `out/runs/zh-v2/host_table.csv` (`python -m r2ai.zh_v2.scope table`). 883 id deferred ở bảng này đã được mở lại theo D13 (mục 2).

## 2. Crawler zh_full giới hạn về v2 và việc mở lại (D11, D13)

- Backup `state/zh_full/crawl-pre-zh-v2-1791877076.db`, SHA256 `450e572e5cad8931a15cfc5901e10eb05a0cf9d7dc4222564b17605efa01226a`.
- `r2ai.selective.manage.defer_urls` / `defer-urls`: mỗi dòng ngoài v2 chuyển `deferred_select` đồng thời ở `urls.status` **và** `select_meta.new_status`; `rank`, `old_status` giữ nguyên nên `reopen`/`rollback` khôi phục được. Cổng: chỉ cột `status` của 413.970 dòng đổi, mọi dòng/cột khác giống backup. Không xoá cache ngoài v2.
- D13: mở lại 883 URL deferred (wujue, zydcd, qihuangzhishu) bằng `manage reopen-urls` (cả hai bảng), bỏ `watchdog.pause`, chạy lại crawler zh_full. 120ask giữ `waiting_robots` (robots 521 từ 12/10). cnkang: 0 pending, `ok` 536.434 dòng (xong phần v2).
- Không đổi rate, không đụng robots của host cũ.

## 3. Host mới (81.437 id): probe, crawler zh_v2, ngoại lệ 39.net

Probe (`python -m r2ai.zh_v2.probe`, 1 req/s, robots + 5 URL/host, VN egress): `out/runs/zh-v2/probe/table.md`, HTML mẫu ở `probe/html/`.

| host | v2 URL | robots | cách lấy | CJK trung vị | hỏi-đáp |
|---|---:|---|---|---:|---|
| ask.39.net | 32.772 | 404 → no_robots | static HTML | 1.466 | 5/5 |
| familydoctor.com.cn | 26.819 | 200 robots_ok | static HTML | 3.507 | 0/5 |
| baby.39.net | 3.368 | 200 (HTML) → no_robots_html_fallback | static HTML | 1.983 | 5/5 |
| 18 host `*.39.net` (heart, fk, nk, cancer, tnb, woman, food, ek, man, shen, gan, wei, sj, fitness, gk, pf, gc, byby) | 7.806 | HTML fallback | static HTML | 1.446–2.531 | 5/5 |
| baidianfeng/qy/familydoctor.cn | 64 | 404 / 200 | static HTML | 3.060–5.447 | 0 |
| zysjonline.com | 6.591 | 403 DISALLOWED | không crawl | | |
| baidu.com | 11 | robots_ok nhưng chặn path | không crawl | | |
| article.iiyi.com, bingli.iiyi.com | 734 | 521 UNAVAILABLE | chờ robots | | |
| ask.familydoctor.com.cn | 1 | 503 UNAVAILABLE | chờ robots | | |

**Lý do cho phép crawl 19 subdomain 39.net (quyết định của Vu):** `/robots.txt` của các subdomain này redirect sang trang 404 của `www.39.net`. RFC 9309 §2.3.1.3: robots.txt trả 4xx (kể cả sau redirect) → crawler được truy cập mọi tài nguyên. Crawler mặc định fail-closed với redirect robots khác host, nên thêm ngoại lệ chỉ cho họ `*.39.net` (`ROBOTS_REDIRECT_SUFFIX` trong `zh_v2/crawl.py`; `zh_sample/crawl.py` mặc định `()` = hành vi cũ, mọi host khác vẫn fail-closed). Tốc độ = nhóm tuner default (rate_cap 4 req/s, 2 conn, bắt đầu 1 req/s, giảm khi 429/5xx), cả họ 39.net chung 1 lane. Có test.

Crawler zh_v2 (`python -m r2ai.zh_v2.crawl prepare|run|stop|status`): engine của zh_full nạp vào namespace riêng, `state/zh_v2/crawl.db`, `data/raw_zh_v2`, `out/runs/zh-v2-crawl/`; 71.564 URL (4 lane: ask.39.net / familydoctor / baby.39.net+18 sub / iiyi+ask.familydoctor). Không đụng DB và raw của zh_full.

Tình trạng 19:27: familydoctor.com.cn xong (`ok` 26.814, blocked_4xx 5), ask.39.net `ok` 7.263 / pending 25.452, baby.39.net đang chạy (`ok` 169, 0,35 URL/s, ETA đo 2,5 giờ cho riêng baby), 18 sub 39.net pending (7.806, cùng lane với baby). **ask.39.net đang `waiting_robots`**: robots.txt `ConnectTimeout` (10,5 s) lúc 19:19; crawler giữ nguyên, thử lại sau 30 phút, không vượt. Tốc độ ask.39.net lần chạy trước 0,81 ok/s (p50 1,65 s) → nếu robots mở lại ≈ 8,7 giờ cho 25.452 URL (ước tính). iiyi (729+5) và ask.familydoctor (1) chờ robots 521/503. Deadline crawler 20/10 23:59.

## 4. Watchdog zh_v2

`python -m r2ai.zh_v2.watchdog` (giống zh_full): đọc heartbeat `runtime_state.json` (60 s), ghi `out/runs/zh-v2-crawl/status.json` (ALIVE / DEAD / DONE / PAUSED / HOLD) và `watchdog.jsonl`; chết hoặc treo (> 600 s) → (kill rồi) chạy lại; giới hạn 6 lần/giờ, không restart khi mất mạng. `--status` in status.json, exit 1 khi DEAD; `--dry-run`. 5 test. Đã chạy thật: lần spawn đầu (PID 26264) chết im lặng, log rỗng, chưa rõ nguyên nhân (chạy tay 40 s thì ổn); lần chạy lại watchdog giữ sống, heartbeat 43 s. Cần theo dõi. **Task Windows chưa đăng ký** (cần quyền admin và mật khẩu Windows của Vu): `powershell -ExecutionPolicy Bypass -File scripts\register_zh_v2_watchdog.ps1` (khởi động máy + mỗi 5 phút, WakeToRun). Tôi không nhập mật khẩu.

## 5. Common Crawl cho host bị robots chặn (không tải WARC)

`python -m r2ai.zh_v2.cc_queue scan|queue`: quét **chỉ index cột** (parquet, range request, 13 crawl cửa sổ corpus) ≤ 2 req/s qua ledger (5.000 request / 2 GB), không bao giờ gọi WARC, không chạm site đích. Đầu ra `out/runs/zh-v2/cc_queue.parquet` (URL, crawl, WARC file, offset, length) cho fetcher của S3.

Tiến độ: 4/13 crawl xong (2026-39, 34, 30, 25): zysjonline.com **0** dòng, baidu.com **0** dòng (CC tôn trọng robots nên không có là hợp lý). Dùng 1.720 request / 433 MB. Quét 9 crawl còn lại chạy nền (~25 phút/crawl); kết thúc tự dựng `cc_queue.parquet`. Nếu cuối vẫn 0 thì queue rỗng và phủ CC cho 6.602 id = 0%.

**120ask (74.898 pending + 102 network_error): chưa quét CC.** Robots vẫn 521. Theo D13 chỉ làm nếu 16/10 robots còn 521, và hỏi Vu trước.

## 6. Index zh v2 và ZG-v2

- `python -m r2ai.zh_v2.index build --verify` (42 s CPU): giữ chunk/embedding có sẵn, lọc theo doc có ≥ 1 id nhóm thuộc v2 zh; dense fp16, CSR, text sao chép byte; 29 segment, **448.858 chunk, 77.523 doc**; không embed lại. `data/index/zh_v2_segments/` (manifest kèm SHA corpus_v2_ids), `data/chunks_zh_v2/`. Đã bỏ loại trừ qihuangzhishu (D13 thay D5); lần build kế sẽ lấy thêm segment đó.
- Retrieval đúng quy tắc cũ: dense top-200 ∪ sparse top-200 mỗi query, rerank bge-reranker-v2-m3 fp16, mỗi nhánh top-150 doc theo logit chunk cao nhất. Rerank 1.200 query: 4,01 s/query (4.811 s).
- **ZG-v2** (τ = 0,96 cố định, nhánh dense, ≤ 10 doc/query, không nằm trong base VS-append `9c91c952…43e2e0`, lọc v2): 3.601 doc / 1.200 query = **3,00 doc/query** (trung vị 1, max 10; 538 query = 0 doc; 212 query chạm trần 10). Trước lọc v2: 4,80 id/query. Doc ≥ τ không thuộc base: 6,09/query. Không dựng JSON/ZIP (S8 dựng probe từ cand_zh_v2).
- **`artifacts/cand_zh_v2.parquet`** (batch 0, 77.523 doc, phủ 32,3% của 240.000): 60.000 dòng (1.200 query × top-50), cột `query_id, rank, doc_id, logit, domain, segment, chunk_id_v2, doc_ids_group, doc_ids_v2`; SHA256 `b7a03d58e6087f7e888f57ee7ff2520258770286ddccf29d5e0b8a8b3f345ed3`.

## 7. Ingest theo đợt

Code: `r2ai.zh_v2.extract` (extractor theo host; hỏi-đáp ask.39.net và baby.39.net giữ câu hỏi + trả lời trong body), `r2ai.zh_v2.ingest cut|launch|adopt` (chunker t256, segment id ≥ 100), `retrieve rerank --reuse` (chỉ rerank cặp mới) + `export` + `zg`. Mỗi đợt xuất lại `cand_zh_v2.parquet` kèm SHA và độ phủ.
**Đợt 1 chưa chạy.** Điều kiện của Vu: ask.39.net + cnkang ≥ 30.000 doc mới. Hiện: ask.39.net 7.263 `ok` + cnkang b003 ≈ 3.350 → ≈ 10,6 nghìn < 30.000, và ask.39.net đang chờ robots. familydoctor.com.cn đã có 26.814 `ok` sẵn sàng; nếu tính gộp là ≈ 37 nghìn (cần Vu cho phép đổi điều kiện).

## 8. Kiểm thử

Full suite sau khi thêm cc_queue/ingest/retrieve/watchdog: **717 passed, 2 deselected** (387,7 s), so với mốc 682 pass: +35 test; 0 lỗi. Log `out/runs/zh-v2/pytest_full.log`.

## Việc cần Vu

1. Chạy `scripts\register_zh_v2_watchdog.ps1` (admin, mật khẩu của Vu) để có task tự khởi động lại sau khi tắt máy.
2. Đợt ingest 1: giữ điều kiện ≥ 30k từ ask.39.net + cnkang, hay gộp cả familydoctor.com.cn (26.814 doc đã sẵn)?
3. 120ask: nếu 16/10 robots còn 521, cho phép quét CC (75.000 URL)?
4. ask.39.net robots đang timeout (site chậm/chặn tạm): chỉ chờ, không có bước nào vượt robots.
