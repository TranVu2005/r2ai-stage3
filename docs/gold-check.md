> Tài liệu legacy được giữ để tham khảo; root/module/CLI hiện hành theo README.md. Writer phụ đang khóa; không chạy lệnh legacy vào OLD.

# Kiểm tra gold thủ công

Chạy tại thư mục repository:

```powershell
python -m pip install -r requirements-gold-check.txt
python gold_check_sample.py
python gold_check_app.py
```

Mở http://127.0.0.1:8765. Máy hiện tại đã có dependency; Flask cài trong `.probe_deps`, app tự nạp thư mục đó. Lần đầu dựng cache SQLite từ 4,39M dòng; những lần sau tái sử dụng cache. Có thể chọn `--port`, `--tasks`, `--results`, `--corpus`, `--index`, `--pages`.

Tokenizer chỉ nạp `BAAI/bge-m3` từ cache, không tải model weights. Nếu chưa có tokenizer, chạy `python gold_check_app.py --download-tokenizer` hoặc `--tokenizer PATH` trỏ tới tokenizer đã lưu. Không tự đổi tokenizer khi lỗi.

## Quy trình

1. Mở link Google/Bing/Cốc Cốc bằng nút trên trang, tìm thủ công.
2. Dán URL trang nội dung (mỗi dòng một URL), chọn nhãn và ghi chú.
3. **Lưu & xác minh**: ghi CSV và hiện lookup trước; fetch tiếp từng URL, ghi sau mỗi URL. Nếu các URL có mức khớp/loại trang khác nhau, sửa nhãn riêng URL ở vùng kết quả rồi lưu lại. Ctrl+Enter lưu và sang query tiếp sau xác minh.
4. Mở lại app để tiếp tục ở query chưa ghi đầu tiên. Mở Trước/Sau để sửa; lưu lại thay các dòng của query đó, không cộng trùng. Nếu fetch lỗi, nhãn vẫn được giữ; lưu lại để thử xác minh lại.
5. Bấm **Xuất báo cáo** hoặc chạy `python gold_check_report.py`.

Output production:

- `out/gold_check_tasks.json`: seed 42; 10 query <15 từ, 10 query 15–40 từ, 10 query >40 từ; đếm từ bằng whitespace, IDF từ toàn bộ 1.200 query.
- `out/gold_check_corpus.sqlite`: cache tra URL exact và gợi ý loose; đối chiếu path/size/mtime/version để dựng lại khi corpus thay đổi. Đọc parquet theo batch 50.000 dòng; SQLite cache 32 MB và sort trên disk. Không nạp dict 4,39M dòng vào RAM. Dành khoảng 2 GB disk cho cache và quá trình dựng.
- `out/gold_check_results.csv`: một dòng/query–URL, hoặc một dòng URL rỗng cho `none`. `found/page_type` là nhãn chung query; `url_found/url_page_type` là nhãn riêng URL (mặc định lấy nhãn chung). Thống kê domain, joint verbatim+corpus và bất đồng máy dùng nhãn riêng URL. UTF-8 BOM; không để CSV mở trong Excel khi lưu.
- `out/gold_check_pages/`: HTML response gốc và text trafilatura, tên hash từ ID query và URL.
- `out/gold_check_report.md`: denominator rõ, Wilson CI95%, đối chiếu người/máy và bảng quyết định.

## Quy ước và giới hạn

- URL exact bỏ scheme, www, fragment, slash cuối; hostname lowercase; **giữ nguyên path case, port, query string và thứ tự tham số**. Lookup bỏ query string chỉ gợi ý, không đặt `in_corpus=true`. Redirect được ghi và tra riêng; báo cáo dùng URL người dán.
- App không gửi request tìm kiếm; chặn URL Google/Bing/Cốc Cốc cả khi redirect. Fetch chỉ phát sinh sau khi lưu URL người dán, kèm robots.txt, redirect và cookie retry từ `fetcher.py`. Giữ robots policy hiện có, timeout 15s mỗi request và ≤1 req/s/host (www được gộp). Tổng một lượt có thể lâu hơn 15s do pacing/retry.
- Verbatim chuẩn hoá NFC/lowercase, dấu câu thành khoảng trắng và gộp khoảng trắng; text gồm tiêu đề HTML (nếu trafilatura bỏ tiêu đề) rồi thân bài trafilatura. `match_source` ghi title/body; `body_verbatim_hit` phân biệt khớp riêng trong thân bài. Ký tự `[start,end)` là offset **trong file text đã lưu**, không phải HTML. Hiện đúng 300 ký tự ngay sau end. Không có verbatim thì hiện đoạn token liên tiếp dài nhất với `match_kind=partial_tokens`; không xem đó là LCS span hay nguyên văn toàn query.
- LCS là subsequence chính xác trên token BAAI/bge-m3, không truncation, bỏ special tokens. Thuật toán bit-parallel giảm RAM/thời gian; LCS cao vẫn có thể do token rải rác.
- Query ≤12 từ dùng cả câu (bỏ dấu câu cuối) làm chuỗi chính và thêm 1–2 đoạn nguyên văn ngắn hơn. Query dài chọn 2–3 cửa sổ 8–12 từ bằng IDF, số, viết tắt hoa, danh sách Latin/y khoa heuristic và phạt cụm hỏi chung. Query <3 từ chỉ có chuỗi cả câu; mẫu hiện tại đều có 2–3 chuỗi.
- Nhãn người là kết quả tìm kiếm thủ công, không phải nhãn nguồn gốc đã chứng minh. Lỗi HTTP/robots/extraction/tokenizer được giữ là unavailable, không tính `verbatim_hit=false` giả. Query chưa ghi không tính `none`.
- Wilson cho tỷ lệ trong mẫu; không dùng tỷ lệ chung 10/10/10 làm ước lượng không chệch trên toàn bộ 1.200 query. Gợi ý site search gây lệch domain; enrichment là proxy ưu tiên crawl, không phải phép kiểm định nguồn.

## Kiểm thử

```powershell
python -m unittest discover -s tests -p test_gold_check.py
python scripts/gold_check_smoke.py
```

Smoke dùng 1 query thật trong mẫu và 1 URL **giả localhost**, fetch thật bằng `fetcher.py`, trafilatura và tokenizer thật. Mọi fixture/results/report nằm riêng trong `out/gold_check_smoke/`, không đi vào báo cáo production.
