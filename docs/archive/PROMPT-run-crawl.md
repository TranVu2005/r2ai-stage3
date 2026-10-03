> Tài liệu legacy được giữ để tham khảo; root/module/CLI hiện hành theo README.md. Writer phụ đang khóa; không chạy lệnh legacy vào OLD.

# Prompt cho session chạy crawl + extract (dán nguyên khối bên dưới)

````
Làm việc trong D:\GitHub\r2ai-stage3 (Windows, PowerShell/Git Bash). Đọc README.md và README-crawl.md trước.

## Bối cảnh
Crawler tiếng Việt giai đoạn 1 của cuộc thi Road to AI Stage 3 (truy hồi y khoa vi/en/zh) đã viết xong và test xong
(150 test nhanh + 2 test chậm kill/mất mạng đều pass). Nhiệm vụ của session này là CHẠY nó, giám sát, và báo cáo.
Không viết lại crawler.

- Môi trường: `.venv` (Python 3.12). Luôn dùng `.\.venv\Scripts\python.exe` hoặc kích hoạt `.\.venv\Scripts\Activate.ps1`.
- State đã khởi tạo: `state/crawl.db` có 653.970 URL của 48 domain vi (từ data/raw/links_corpus.parquet), 150 URL đã crawl thử
  (laodong.vn, vinmec.com, suckhoecongdongonline.vn – mỗi domain 50). Chỉ crawl các URL trong dataset được phát, không dò thêm link.
- Cấu hình đã chốt, KHÔNG đổi: config/domains.yaml (cdn_large 8 req/s; suckhoecongdongonline.vn group default, background;
  wave 1 = domain nhỏ/vừa, wave 2 = suckhoedoisong, thanhnien, laodong, vinmec, medlatec).
- Ngoài phạm vi: nhathuoclongchau.com.vn (out/deferred_domains.csv) và pmc-ecm-healthblog.beta.pharmacity.io (403).

## Việc cần làm
1. Kiểm tra nhanh: `python crawl.py status`; egress phải là VN (crawler tự dừng với exit code 3 nếu không – nếu vậy báo tôi, đừng tìm cách vòng).
2. Chạy crawl NỀN, không chặn session: dùng Bash/PowerShell với run_in_background:
   `python crawl.py run --keep-awake`  (tùy ý thêm `--hours N` nếu tôi yêu cầu giới hạn thời gian).
3. Chạy extract song song, cũng nền: `python extract.py --watch`.
4. Giám sát bằng `python crawl.py status`, `logs/crawl.log`, `out/crawl_vi_report.md` (cập nhật mỗi giờ). KHÔNG poll bằng vòng sleep;
   kiểm tra khi được báo hoàn tất hoặc khi tôi hỏi. Dừng sạch bằng Ctrl+C/kill nhẹ (lần 1), không `kill -9` trừ khi treo.
5. QA gate: domain ≥ 5000 URL tự dừng sau 300 doc ok, sinh `out/qa_extract/<domain>.md`, trạng thái `awaiting_qa`.
   Khi có file mới: đọc, tóm tắt cho tôi (extractor đúng chưa, boilerplate lặp, Q/A tách đúng chưa, đoạn lặp nhiều nhất) rồi DỪNG LẠI HỎI.
   Chỉ chạy `python crawl.py approve <domain>` SAU KHI tôi nói rõ đồng ý domain đó. Không tự approve.
   Nếu extractor sai, sửa selector trong vicrawl/extractors/sites.py + thêm/sửa test trong tests/test_vicrawl_extractors.py,
   rồi chạy lại `python extract.py` (HTML thô đã lưu nên không cần crawl lại: xoá file parquet của domain đó trong data/docs_vi
   và dòng tương ứng trong state/extract.db bảng shards/docs, hoặc báo tôi nếu không chắc).
6. Domain `halted` (lỗi >20%/200 request, hoặc 12 lỗi liên tiếp): đọc `halt_reason` trong status/log, tìm nguyên nhân
   (bot-block, 403/429, server chết). Báo tôi trước khi `python crawl.py reset-errors <domain>`. Không dùng proxy xoay vòng, captcha solver,
   cookie đăng nhập, hay vượt 4 req/s (8 với cdn_large).
7. Mất mạng / máy sleep: crawler tự pause và resume, không cần can thiệp. Nếu ipinfo báo country ≠ VN nó giữ pause và cảnh báo – báo tôi.
8. Khi `out/MILESTONE_1.md` xuất hiện (≥ 100k doc ok), báo tôi ngay và chạy `python extract.py` để docs_vi đủ cho baseline.

## Báo cáo định kỳ (khi tôi hỏi, và khi có sự kiện: QA gate, halted, milestone, xong)
- Tiến độ tổng và theo domain (done/total, ok%, rate hiện tại, ETA), số URL còn chờ retry.
- Domain halted/awaiting_qa kèm lý do.
- Từ docs_vi: số doc, median n_tokens_bge_m3, % thin, % trùng text_sha1, phân bố extractor (generic vs riêng).
- Bất thường: rate bị giảm do 429/503, nhiều cookie_challenge/bot_challenge, tỷ lệ soft404 cao.

## Quy tắc
- Không sửa dữ liệu gốc (data/raw/*). Không xoá data/raw_vi, state/crawl.db.
- Không commit/push nếu tôi chưa yêu cầu. Không gửi dữ liệu ra ngoài.
- Gặp lỗi code thật: viết test tái hiện trước, sửa tối thiểu, chạy `python -m pytest tests -q` trước khi chạy lại crawl.
- Dung lượng ước tính 3–5 GB cho data/raw_vi; kiểm tra ổ đĩa D: còn trống trước khi bắt đầu.
````
