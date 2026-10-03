> Tài liệu legacy được giữ để tham khảo; root/module/CLI hiện hành theo README.md. Writer phụ đang khóa; không chạy lệnh legacy vào OLD.

# Crawler tiếng Việt – giai đoạn 1

Chạy trên 1 laptop (Windows native / WSL2 / Linux), mạng VN. Không cần tmux/nohup: dừng bằng Ctrl+C, gập máy, mất wifi, tắt máy – chạy lại là tiếp tục đúng chỗ.

```powershell
python -m pip install -r requirements-crawl.txt
python crawl.py init                  # 1 lần: phân loại domain, gộp URL trùng, dựng state/crawl.db (đã chạy xong ở repo này)
python crawl.py run --keep-awake      # chạy hết, tới khi xong / Ctrl+C
python crawl.py run --hours 3         # hoặc: --until 23:30 ; --domains a.vn,b.vn ; --group vi ; --limit 50
python crawl.py status                # bảng tiến độ đọc từ SQLite, crawler không cần đang chạy
python extract.py --watch             # cửa sổ thứ 2: trích xuất tăng dần (hoặc chạy sau, `python extract.py`)
python crawl.py approve vinmec.com    # sau khi đọc out/qa_extract/vinmec.com.md
python crawl.py reset-errors <domain> # đưa URL lỗi về hàng đợi, bỏ trạng thái halted
```

Ctrl+C lần 1: ngừng nhận URL mới, chờ request đang bay (tối đa 30s), ghi shard, commit SQLite, in tóm tắt rồi thoát. Ctrl+C lần 2: thoát ngay (an toàn, xem dưới).

## Dữ liệu và thư mục

| đường dẫn | nội dung |
|---|---|
| `state/crawl.db` | SQLite WAL: `urls` (url_norm, doc_ids, domain, status, http_status, attempts, fetched_at, shard…), `domains` (rate đã học, conns, baseline latency, state), `meta` |
| `data/raw_vi/<domain>/<shard>.jsonl.zst` | HTML thô, mỗi file là 1 frame zstd hoàn chỉnh; mỗi dòng JSON = 1 URL (cả URL lỗi, `html` = null) |
| `data/docs_vi/<domain>__<shard>.parquet` | kết quả `extract.py`: `doc_ids, url, final_url, domain, title, description, question, answer, body, paragraphs, lang, n_tokens_bge_m3, text_sha1, fetched_at` + `url_norm, status, extractor, headings` |
| `state/extract.db` | shard đã xử lý + url_norm→file (để khử trùng, giữ bản mới nhất) |
| `out/vi_domains.csv`, `out/deferred_domains.csv`, `out/domain_classification_all.csv` | phân loại domain |
| `out/qa_extract/<domain>.md` | báo cáo review cho domain lớn |
| `out/crawl_vi_report.md` | báo cáo (mỗi giờ + khi thoát) |
| `out/MILESTONE_1.md` | sinh khi tổng doc `ok` ≥ 100k |
| `logs/crawl.log` | log xoay vòng (5 MB × 5) |
| `config/domains.yaml` | `group / rate_cap / max_conns / extractor / priority` theo domain |

`body` nối các đoạn bằng `\n\n`; `paragraphs` giữ đúng ranh giới đoạn. Chỉ chuẩn hoá khoảng trắng, không dịch/viết lại. Doc cùng `text_sha1` được giữ cả, không xoá.

Dung lượng: ≈ 3–7 KB/URL sau nén zstd (đo trên 150 trang thật) → khoảng 3–5 GB cho 654k URL.

## Thiết kế an toàn khi dừng

* **Shard nguyên tử**: ghi `*.tmp` → fsync → `os.replace`. `*.tmp` dở dang bị xoá khi khởi động.
* **Thứ tự commit**: một URL chỉ được ghi trạng thái cuối trong SQLite *sau khi* shard chứa nó đã rename xong. Flush mỗi 500 doc / 8 MB / 30 s. Bị giết giữa chừng ⇒ URL còn `in_progress`/`pending` ⇒ tải lại; bản trùng trong shard được `extract.py` khử theo `url_norm` (giữ `fetched_at` mới nhất).
* SQLite WAL + `synchronous=NORMAL`, mọi ghi đi qua đúng 1 thread writer.
* **Ngủ/wake, mất mạng**: đồng hồ treo tường nhảy > 60 s so với tick 5 s, hoặc request lỗi mạng + ping thất bại ⇒ tạm dừng mọi domain, đợi mạng, gọi lại `ipinfo.io/json`; country ≠ VN thì giữ pause và kiểm tra lại mỗi 60 s. Request chồng lên khoảng offline không tính vào tỷ lệ lỗi, không tăng `attempts`, không giảm rate; pool kết nối và DNS cache được làm mới.
* Rate học được lưu theo domain; chạy lại bắt đầu từ 75 % rate đó.
* Khởi động: country ≠ VN ⇒ dừng ngay (exit code 3).

## Rate / tin cậy

Bắt đầu 1 req/s (`start_rate`). Mỗi 500 request không có 429/503 và p95 latency ≤ 1,5× baseline (p95 của 100 request đầu) ⇒ +0,5 req/s tới `rate_cap`. 429/503 ⇒ rate/2 và tạm dừng 60 s. Lỗi > 20 % trong 200 request gần nhất (hoặc 12 lỗi liên tiếp khi domain chết hẳn) ⇒ dừng domain, ghi `halt_reason`. p50 > 3 s ⇒ không tăng rate, mở thêm kết nối (≤ 4). Tôn trọng `Crawl-delay`; robots 5xx/timeout ⇒ chờ 30 phút rồi thử lại, không coi là cho phép; robots 401/403 ⇒ coi là cấm. URL lỗi tạm thời (network, 5xx, 429, challenge) thử lại tối đa 3 lần, cách ≥ 1 giờ (lần chạy sau tự nhặt).

Thứ tự (`priority`): wave 1 = domain nhỏ/vừa chạy song song; wave 2 (suckhoedoisong, thanhnien, laodong, vinmec, medlatec) bắt đầu khi wave 1 xong; domain `background: true` (suckhoecongdongonline.vn) chạy cùng từ đầu. `--domains` bỏ qua wave.

QA gate: domain ≥ 5000 URL dừng sau 300 doc `ok`, sinh `out/qa_extract/<domain>.md` (5 doc ngẫu nhiên: URL, title, 3 đoạn đầu/cuối, question/answer, + các đoạn lặp nhiều nhất để bắt boilerplate), chờ `crawl.py approve`. Crawler đang chạy tự nhặt approve trong ~10 s.

## Kiểm thử

```powershell
python -m pytest tests -q              # nhanh (~1,5 phút)
python -m pytest tests -q -m slow      # kill 5 lần ngẫu nhiên + chặn mạng 2 phút (~4 phút)
```
