# S11 — zh v2 (240.000 URL zh của corpus v2, D11 + D13)

Nhánh `feat/zh-v2`, worktree `D:/GitHub/r2ai-stage3-zhv2`. Cập nhật 2026-10-14 17:50 (UTC+7). Số liệu đo từ `state/zh_full/crawl.db`, `state/zh_v2/crawl.db` (mode=ro), `data/corpus_v2_ids.parquet`, `out/runs/zh-v2/`. Chưa push.

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

`python -m r2ai.zh_v2.watchdog` (giống zh_full): đọc heartbeat `runtime_state.json` (60 s), ghi `out/runs/zh-v2-crawl/status.json` (ALIVE / DEAD / DONE / PAUSED / HOLD) và `watchdog.jsonl`; chết hoặc treo (> 600 s) → (kill rồi) chạy lại; giới hạn 6 lần/giờ, không restart khi mất mạng. `--status` in status.json, exit 1 khi DEAD; `--dry-run`. 5 test. Đã chạy thật: lần spawn đầu (PID 26264) chết im lặng, log rỗng, chưa rõ nguyên nhân (chạy tay 40 s thì ổn); lần chạy lại watchdog giữ sống, heartbeat 43 s. Cần theo dõi. **Task Windows đã đăng ký 14/10 02:00** theo kiểu task S10 (không admin, không lưu mật khẩu): `R2AI-zh-v2-watchdog`, user Admin, LogonType Interactive, RunLevel Limited, trigger AtLogOn + lặp mỗi 5 phút, WakeToRun, StartWhenAvailable, MultipleInstances=IgnoreNew, giới hạn 10 phút/lần; chạy khi Admin đang đăng nhập (sau tắt máy cần đăng nhập lại). Script `scripts/register_zh_v2_watchdog.ps1`.

## 5. Common Crawl cho host bị robots chặn (không tải WARC)

`python -m r2ai.zh_v2.cc_queue scan|queue`: quét **chỉ index cột** (parquet, range request, 13 crawl cửa sổ corpus) ≤ 2 req/s qua ledger (5.000 request / 2 GB), không bao giờ gọi WARC, không chạm site đích. Đầu ra `out/runs/zh-v2/cc_queue.parquet` (URL, crawl, WARC file, offset, length) cho fetcher của S3.

Kết quả: **13/13 crawl xong** (2025-38 … 2026-39): zysjonline.com **0** dòng, baidu.com **0** dòng (CC không chứa URL robots chặn). 4.673 request / 1,48 GB (trần 5.000 / 2 GB). `out/runs/zh-v2/cc_queue.parquet` (bản sao `cc/cc_queue.parquet`, SHA256 `b467b0c9640b679cfe840d76e7bb8ec9fd321b3e8c823b6764f93456cb397dcf`) có **0 dòng**, đủ schema; phủ CC cho 6.602 id = 0%. Fetcher của S3 không có gì để xếp hàng cho 2 host này; **Quyết định của Vu (14/10): bỏ zysjonline.com** (robots chặn, CC rỗng); 6.591 id không vào index, trần phủ giảm tương ứng.

**120ask (74.898 pending + 102 network_error): chưa quét CC.** Robots vẫn 521. Vu đã duyệt (14/10): ngày 16/10, nếu robots còn 521 thì quét CC index (chỉ metadata, ≤ 2 req/s) cho ≈ 75.000 URL v2 chưa crawl; độ phủ ≥ 20% → `cc_queue_120ask.parquet` cho S3, dưới 20% → báo Vu và dừng. Code: `cc_queue scan --hosts 120ask.com --tag 120ask --only-missing` (ledger/budget riêng, không đụng queue của zysjonline; độ phủ ghi ở `cc_queue_report_120ask.json`). Đã đặt nhắc 16/10 08:07 trong phiên.

## 6. Index zh v2 và ZG-v2

- `python -m r2ai.zh_v2.index build --verify` (42 s CPU): giữ chunk/embedding có sẵn, lọc theo doc có ≥ 1 id nhóm thuộc v2 zh; dense fp16, CSR, text sao chép byte; 29 segment, **448.858 chunk, 77.523 doc**; không embed lại. `data/index/zh_v2_segments/` (manifest kèm SHA corpus_v2_ids), `data/chunks_zh_v2/`. Đã bỏ loại trừ qihuangzhishu (D13 thay D5); lần build kế sẽ lấy thêm segment đó.
- Retrieval đúng quy tắc cũ: dense top-200 ∪ sparse top-200 mỗi query, rerank bge-reranker-v2-m3 fp16, mỗi nhánh top-150 doc theo logit chunk cao nhất. Rerank 1.200 query: 4,01 s/query (4.811 s).
- **ZG-v2** (τ = 0,96 cố định, nhánh dense, ≤ 10 doc/query, không nằm trong base VS-append `9c91c952…43e2e0`, lọc v2): 3.601 doc / 1.200 query = **3,00 doc/query** (trung vị 1, max 10; 538 query = 0 doc; 212 query chạm trần 10). Trước lọc v2: 4,80 id/query. Doc ≥ τ không thuộc base: 6,09/query. Không dựng JSON/ZIP (S8 dựng probe từ cand_zh_v2).
- **`artifacts/cand_zh_v2.parquet`** (batch 0, 77.523 doc, phủ 32,3% của 240.000): 60.000 dòng (1.200 query × top-50), cột `query_id, rank, doc_id, logit, domain, segment, chunk_id_v2, doc_ids_group, doc_ids_v2`; SHA256 `b7a03d58e6087f7e888f57ee7ff2520258770286ddccf29d5e0b8a8b3f345ed3`.

## 7. Ingest theo đợt

Code: `r2ai.zh_v2.extract` (extractor theo host; hỏi-đáp ask.39.net và baby.39.net giữ câu hỏi + trả lời trong body), `r2ai.zh_v2.ingest cut|cut-zf|launch|adopt` (chunker t256, segment id ≥ 100), `retrieve rerank --reuse` + `export` + `zg`. `cut-zf` cắt dòng của crawl.db zh_full chưa từng ingest và có ≥ 1 id v2 (cnkang…): lọc theo v2 nên 45 nghìn dòng cnkang ngoài v2 không bị embed.

**Đợt 1 (cắt theo lệnh Vu: ≥ 15k doc mới, ưu tiên ask.39.net và cnkang; chạy 13–14/10):**
- `v2n-002` (cnkang 5.471+27, wujue 661, zydcd 197, qihuangzhishu 29, zhongyibaodian 316; 6.701 URL): 6.178 doc / 32.290 chunk, 5 segment (id 100–104), embed ≈ 100 chunk/s.
- `v2n-001` (ask.39.net 7.316, baby.39.net 3.364, familydoctor.com.cn 26.814, familydoctor.cn 2; 37.496 URL, 3.786 shard): 36.329 doc / 171.270 chunk (id 105–108), xong 00:59.
- Hai doc familydoctor.cn đã có trong segment mẫu 2% (seg00006): bỏ segment trùng seg00106, `adopt` giờ tự bỏ segment trùng và từ chối segment trùng một phần; `index build` giờ giữ segment đã adopt (trước đó ghi đè manifest và làm mất chúng, đã sửa).
- Index sau đợt 1: 37 segment có dữ liệu (qihuangzhishu sample 0 doc v2), **652.406 chunk, 120.028 doc = 50,0% của 240.000** (trước: 77.523 = 32,3%).
- Rerank `--reuse` (cache 1.200 query, chỉ chấm cặp mới): 1.888 s (1,57 s/query) khi GPU rảnh; lượt chạy lại sau khi bỏ segment trùng 113 s.
- **`artifacts/cand_zh_v2.parquet` batch 1**: 60.000 dòng (1.200 × top-50), SHA256 `180174d76dadd9b476712f5f97483192d17efbaaebd66be4f3f47eb5475f87bd`, độ phủ 120.028 / 240.000 doc. Bản batch 0 (SHA `b7a03d58…f345ed3`, 77.523 doc) giữ ở `out/runs/zh-v2/cand_zh_v2.batch0.parquet`. Top-50: 25,9% dòng từ segment mới; 71,6% cặp (query, doc) của batch 0 còn trong top-50; domain đứng đầu 120ask 26.318, cnkang 13.614, familydoctor.com.cn 8.446, ask.39.net 4.309.
- **ZG-v2 batch 1** (τ = 0,96): 3.880 doc / 1.200 query = **3,23 doc/query** (batch 0: 3,00); 508 query = 0 doc (batch 0: 538); 245 query chạm trần 10 (212). Doc ≥ τ không thuộc base 7,37/query (6,09). Không dựng JSON/ZIP.
- Chưa vào index: 18 sub 39.net (≈ 7,8 nghìn URL đã `ok`) và phần ask.39.net crawl thêm sau khi cắt (ok 9.693 / 32.772 lúc 01:50) → đợt 2.
- Điều phối GPU (Vu, 14/10): S10 được ưu tiên từ sau 08:05 (QA → embed en). Tôi dùng GPU xong lúc 01:36. **Đợt zh 2 chỉ chạy GPU khi S10 báo đã embed xong.**
- Đợt 2: cắt khi ask.39.net (phần crawl thêm) + 18 subdomain 39.net + phần crawl thêm đạt ≥ 15.000 doc mới (lúc 01:50: ≈ 2,4 nghìn ask.39.net + ≈ 7,8 nghìn sub 39.net ≈ 10 nghìn).

**Đợt 2 (Vu: chạy ngay, không chờ S10; 14/10 15:20 → 17:46):**
- Trước launch: `nvidia-smi --query-compute-apps` chỉ có tiến trình giao diện (dwm, explorer, Chrome, Claude…), không có python/compute lạ; không kill gì. GPU 704 MiB là bộ nhớ hiển thị.
- `v2n-003` cắt ngay trước launch: 25.266 URL = ask.39.net 17.456, baby.39.net 4, 18 sub 39.net 7.806 (heart 1.055, fk 928, nk 762, cancer 736, tnb 717, woman 687, food 594, ek 448, man 382, shen 325, gan 294, wei 225, sj 197, fitness 185, gk 90, pf 85, gc 59, byby 37). Không có dòng zh_full mới để ingest (`v2n-004` rỗng).
- Embed xong 17:23 (≈ 80 chunk/s); 20 segment mới (id 109–128), rerank `--reuse` 1.059 s (0,88 s/query), trả GPU 17:46.
- Index sau đợt 2: 57 segment có dữ liệu, **747.429 chunk, 144.097 doc = 60,0% của 240.000** (đợt 1: 50,0%).
- **`artifacts/cand_zh_v2.parquet` đợt 2**: 60.000 dòng, SHA256 `86d326173ce2e0ec5d6268a28416c7e6f9b2cb9f6fe11fd1371521d8d8ebd25c`; bản đợt 1 (SHA `180174d7…`) ở `out/runs/zh-v2/cand_zh_v2.batch1.parquet`. 16,6% dòng top-50 từ segment mới; 81,8% cặp (query, doc) của đợt 1 còn trong top-50; domain đứng đầu 120ask 22.189, cnkang 11.428, ask.39.net 11.305, familydoctor.com.cn 6.962.
- **ZG-v2 đợt 2** (τ = 0,96): 4.103 doc / 1.200 query = **3,42 doc/query** (đợt 1: 3,23; đợt 0: 3,00); 488 query = 0 doc (508; 538); 261 query chạm trần 10 (245; 212). Doc ≥ τ không thuộc base 8,22/query.
- Còn chờ: ask.39.net ≈ 10 nghìn URL, 120ask (robots 521, quét CC 16/10), iiyi (robots 521).

## 8. Kiểm thử

Full suite: **720 passed, 2 deselected** (375,6 s), so với mốc 717 (lần 1) là +3 test mới cho `ingest cut-zf` (chỉ lấy dòng có id v2, bỏ URL đã ingest/out-of-v2/pending) và `adopt` (trùng hoàn toàn → bỏ, trùng một phần → lỗi); so với mốc 682 ban đầu: +38. 0 lỗi. Log `out/runs/zh-v2/pytest_full2.log` (lần 1: `pytest_full.log`).

## Việc cần Vu

Không có việc chờ quyết định. Theo dõi: 16/10 (120ask), đợt 2 (≥ 15k doc mới, GPU sau S10).
