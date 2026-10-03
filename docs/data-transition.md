# Copy raw/crawl state trước extract lớn

Thực hiện sau gate repo/replay, ngoài timebox tạo repo. OLD giữ nguyên; không move/xóa. Đích NEW/data/raw_vi và NEW/state/crawl.db.

1. Kiểm không có crawl/extract writer; crawl.db-wal/extract.db-wal phải 0 byte. Nếu WAL khác 0, dừng copy trực tiếp để lập snapshot nhất quán riêng; không checkpoint/UPDATE DB gốc.
2. Kiểm dung lượng trống; không overwrite raw/state NEW đã có nội dung khác. Copy raw_vi giữ cây domain/tên file, copy crawl.db, giữ configs/domain CSV đầu vào. Dữ liệu này ignored, không stage.
3. So inventory/hash từng file raw và hash/count crawl.db nguồn–đích. urls.shard là basename tương đối, không cần UPDATE SQLite.
4. Không copy extract.db/docs cũ. Trước extract lớn kiểm extract.db NEW chưa chứa checkpoint trỏ raw OLD; nếu đã có thì dừng để migration path đầy đủ, không giả định chỉ 3.103 dòng.
5. Đổi R2AI_RAW_DIR trong .env sang NEW/data/raw_vi; DATA_DIR vẫn OLD. Xác nhận RAW_DIR=RAW_WRITE_DIR=NEW/data/raw_vi theo config thực (process env có ưu tiên hơn .env). Cả crawl/extract truyền --raw-dir NEW rõ ràng. Status không cờ đọc NEW/state; status OLD chỉ khi có --state-dir rõ.
6. Giữ raw NEW ổn định đến sau 11/11/2026. Sau đó extract mới 18.031 shard → chunk docs NEW → embed/index → K100 → make_submission/validator cùng bundle NEW. Không chạy crawl thật nếu chưa duyệt riêng; youmed vẫn halted.

Copy không có nghĩa đã extract/index đủ corpus; thời gian chạy lớn chưa đo. Quy trình giữ lại extract.db cũ là phương án khác, không dùng lần này. Kiểm chứng thực tế được ghi vào PROGRESS.
