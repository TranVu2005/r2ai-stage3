# Quy tắc làm việc

- Trả lời tiếng Việt, ngắn gọn, ưu tiên số liệu đo thực tế; chưa đo thì ghi rõ.
- Repo độc lập, code ở src/r2ai và src/vicrawl. Mọi root/runtime path qua r2ai.paths; .env tự nạp trước thư viện HF. CLI > process env > .env > mặc định.
- Không đổi thuật toán, version thư viện, API, schema, thứ tự rank/chunk, dedupe hoặc metric trong việc tổ chức code/path. Chạy test phù hợp và ghi kết quả.
- Không ghi, xóa, rename/move OLD `D:/GitHub/r2ai-stage3`. OLD là backup cho đến khi người dùng xác nhận.
- Cấm `git clean -fdx`, reset/remove tại OLD hoặc Git cha `D:/GitHub`: data/out khoảng 15,366 GiB không tracked có thể mất. Stage theo allowlist; không commit corpus, state, output, .env, secret hay token; tracked file <=5.000.000 byte.
- Guard đích ghi kể cả override/temp/cleanup trước I/O. Công cụ phụ bị khóa trong main/__main__; giữ import thư viện cho test. Chỉ mở khóa sau khi thêm guard, không thêm bypass OLD.
- Không crawl thật khi chưa được duyệt rõ. Youmed halted; --limit là cutoff rank, không phải số pending tiếp theo. Status mặc định STATE_DIR; đối chiếu OLD phải truyền --state-dir rõ, đọc URI ro và không tạo log.
- Sau gate repo, copy raw_vi/crawl.db sang NEW trước extract lớn, giữ nguyên OLD, kiểm writer dừng/WAL0/hash/inventory. Crawl/extract cùng raw NEW; không mang extract.db/docs cũ vào bundle mới.
- Không đổi raw root sau khi có checkpoint extract lớn nếu chưa migration mọi shards.path. Giữ raw NEW ổn định đến sau 11/11/2026.
- Replay sub06 đọc cache/chunks/docs OLD rõ ràng; pipeline mới chọn toàn bộ bundle NEW. Không tự fallback/trộn bundle.
- Sau mỗi việc cập nhật docs/PROGRESS.md: header UTC+7/model/HEAD thực đã xác minh trước commit, thêm Changelog, cập nhật §6. Không ghi commit tự tham chiếu hay điểm leaderboard chưa đo. Deadline public 31/10/2026; private 04/11/2026 tối đa 5 lượt; kết quả 11/11/2026.

- assert_writable quét recursive chỉ ở preflight CLI; không gọi trong vòng lặp shard. ZIP nén khác OLD có thể đổi kc sát budget; đo ZIP thực, giữ hash JSON làm gate parity.
