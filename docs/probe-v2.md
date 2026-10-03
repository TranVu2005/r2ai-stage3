> Tài liệu legacy được giữ để tham khảo; root/module/CLI hiện hành theo README.md. Writer phụ đang khóa; không chạy lệnh legacy vào OLD.

# Probe v2 (bước 2–4)

Không ghi dữ liệu gốc. Corpus thực tế: `data/raw/links_corpus.parquet`.

```powershell
python -m pip install -r requirements-probe.txt
python -m unittest discover -s tests -v
python step1_reclassify.py
python step1_reprobe.py --workers 8
python step1_archive_coverage.py --workers 4
python profile_report_v2.py
```

Reclassify hoàn toàn offline. Hai bước online phải chạy từ máy có quyền mạng local. Tokenizer BAAI/bge-m3 lấy từ cache; dùng `--download-tokenizer` nếu cache chưa có. `warcio` đã cài cục bộ trong `.probe_deps` ở lần chạy này.

Một process cho mỗi bước; song song giữa domain do `--workers` điều phối. Mỗi domain/host tối đa một request/giây, kể cả robots, redirect và retry. Crawl-delay chỉ lấy từ nhóm User-agent áp dụng; robots 401/403 từ chối, lỗi mạng/5xx/challenge chưa đọc được chính sách sẽ dừng URL. Robots 200 trả HTML fallback không có chỉ thị được coi là không có chính sách; các chỉ thị hợp lệ vẫn được parse.

Chạy lại một phần (mẫu không đổi seed):

```powershell
python step1_reprobe.py --domains laodong.vn familydoctor.com.cn
python step1_reprobe.py --domains laodong.vn --cookie-diagnostic
python step1_reprobe.py --domains familydoctor.com.cn --redo-ab
python step1_reprobe.py --domains bingli.iiyi.com --retry-robots
python step1_archive_coverage.py --domains bingli.iiyi.com --retry-unknown
```

Thí nghiệm mới với số mẫu khác dùng run-dir riêng để tránh trộn số liệu:

```powershell
python step1_reprobe.py --domains vinmec.com --n 5 --run-dir out/partial-vinmec
python step1_archive_coverage.py --n 10 --content-n 5 --run-dir out/partial-vinmec
```

Checkpoint JSONL được flush/fsync từng kết quả; dòng cuối bị ghi dở được cắt bỏ khi resume. Giữ lịch sử khi đo lại; CSV chỉ xuất bản ghi mới nhất theo key. Main sample top-35 ×30 URL; bỏ trùng http/https/www, giữ nguyên path/query. A/B 10 URL ghép cặp, hai UA có session riêng với cùng vòng đời; domain được chọn nếu blocked >50% ở probe cũ hoặc mới. Không kết luận IP/geo từ một egress.

Archive mặc định theo yêu cầu cập nhật: **50 URL/domain**, CDX theo domain rồi join local (bỏ scheme/www/slash cuối, giữ query). CC dùng `showNumPages=true`, tối đa **3 page/crawl**, **2 crawl mới nhất** trước; thêm crawl cũ (tối đa 6) nếu tỷ lệ join quan sát <30%. Wayback dùng `matchType=domain`, `filter=statuscode:200`, `collapse=urlkey`, `fl=original,timestamp`, `limit=50000`. CI95% Wilson chỉ cho tỷ lệ **quan sát dưới giao thức truy vấn giới hạn**, không phải toàn bộ archive; URL ngoài page/limit giữ `not_observed`/unknown.

Ưu tiên familydoctor.com.cn, ask.39.net, zysjonline.com, suckhoedoisong.vn, nhathuoclongchau.com.vn, thanhnien.vn, vinmec.com; sau đó các domain còn success trực tiếp <50%. Thử nội dung **tối đa 10 URL có capture/domain**, chọn ngẫu nhiên xác định theo seed; payload thành công báo theo số thử từng nguồn. Độ phủ dùng tỷ lệ union capture × tỷ lệ trích được ở subsample capture, kèm giả định ngoại suy. WARC phải đúng HTTP 206, Content-Range, độ dài và Target-URI.

Ngân sách mặc định **60 phút**, giữ thời điểm bắt đầu khi resume. Transport chạy trong daemon có deadline tuyệt đối để response trickle không giữ CLI quá ngân sách. Khi hết thời gian, giữ checkpoint và ghi rõ chưa đo. Một ngân sách mới sau phiên này phải yêu cầu rõ bằng `--renew-budget`; đo lại lỗi dùng `--retry-unknown`. Mẫu/giới hạn khác dùng `--run-dir` riêng.

```powershell
python step1_archive_coverage.py --n 50 --content-n 10 --pages 3 --max-minutes 60
python step1_archive_coverage.py --domains bingli.iiyi.com --retry-unknown --renew-budget
```

Phương án exact-URL ban đầu vẫn có qua `--mode url` (n=100, 6 crawl, 20 payload); dữ liệu partial cũ giữ ở `archive_coverage_url_lookup_partial.csv`. Báo cáo chính dùng kết quả phương án domain mới.

Mỗi API có circuit breaker độc lập sau 3 truy vấn thất bại liên tiếp đã hết retry. API khác/domain khác tiếp tục; index và replay tách riêng. `--retry-unknown` đo lại phần chưa xác định; không coi API timeout/403/504 là không có capture.

Output chính: `out/profile_report_v2.md`, `out/reclassified.csv`, `out/reprobe.csv`, `out/archive_coverage.csv`. `manifests/`, `checkpoints/`, `request_logs/`, raw HTML, egress/index/service metadata là bằng chứng bổ sung. Độ phủ weighted chỉ ngoại suy từ mẫu domain hoàn tất; mọi phần chưa đo giữ unknown, không được đổi thành 0.

API references: [Common Crawl Index](https://index.commoncrawl.org/), [Wayback CDX API](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server).
