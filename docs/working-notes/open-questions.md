# Câu hỏi cho Vu (S42, 27/10/2026)

Chỗ số liệu lệch, thiếu nguồn, hoặc cần Vu quyết cách viết. Chưa có câu trả lời thì bản nháp để trống hoặc ghi "chờ xác nhận"; không đoán.

## A. Định dạng (chặn việc viết văn)

1. Ngôn ngữ bài: tiếng Việt hay tiếng Anh?
2. Template / định dạng (LaTeX, Word, Markdown?), giới hạn số trang, file mẫu của BTC (đường dẫn).
3. Tên đội, danh sách tác giả (thứ tự, đơn vị), email liên hệ.
4. Hạn nộp working notes và kênh nộp.

## B. Provenance và tuân thủ (cần Vu quyết cách mô tả, vì bài phải trung thực)

1. **robots.txt.** Khung bài Vu gửi mô tả crawl trực tiếp "có giới hạn tốc độ và User-Agent rõ ràng". Docs ghi thêm: quyết định D21 "bỏ qua robots theo duyệt" (S24, 17/10; `docs/PROGRESS.md:576`); `vov.vn` (9.784 URL) và `baolangson.vn` (1.981 URL) ban đầu bị chặn do `robots_disallowed` (`docs/PROGRESS.md:66,105-106`), sau đó S24 crawl trực tiếp 8.364 + 1.978 doc tốt (`feat/s24-recovery:out/runs/s24-recovery/report.md`); crawler S37 cho 120ask "không đọc robots (D21)" (`deploy/s37/README.md:5`). Bài cần nêu điều này thế nào? Ai duyệt D21 (BTC cho phép, hay quyết định của đội)?
2. **120ask từ VPS thứ hai.** Docs: 24/10 mọi kết nối tới 120ask từ VPS Sydney timeout (cả 4G của Vu cũng không vào được); ghi chú lúc đó "đổi IP khi site có thể đang chặn là vượt chặn", Vu chốt không crawl thêm. 25/10 Vu crawl nốt 16.619 URL từ VPS mới (`docs/PROGRESS.md` mục S37, dòng "Lô cuối"). Bài nên mô tả lý do (site sập tạm thời, không phải chặn?) theo lời Vu; xin Vu một câu xác nhận.
3. **`docs/COMPLIANCE.md` đã cũ:** ghi "Không dùng LLM" và chỉ liệt kê BGE-M3, bge-reranker-v2-m3. Thực tế đã dùng Qwen3-Reranker-4B / 0,6B, Jina Reranker v3.5 (trong các bản nộp J3-thrC, 120A-*-v0..v3, free-v2), Qwen3-Embedding-4B (probe), `qwen2.5:7b` qua Ollama để dịch query sang zh (chỉ phân tích fingerprint, `reports/s26-retriever-fingerprint.md` mục 3). Cần bảng license đã kiểm cho từng mô hình (tôi chưa kiểm license Qwen3-Reranker, Jina v3.5, qwen2.5; không đoán). Có cần cập nhật `COMPLIANCE.md` trên nhánh khác không?
4. **Pharmacity "theo hướng dẫn BTC":** chưa thấy văn bản / tin nhắn BTC trong docs; xin nguồn (ngày, kênh) để trích.
5. **"BTC xác nhận doc không truy cập được không có trong nhãn":** chưa thấy trong docs; xin nguồn.
6. Kaggle: 4B private chạy trên một tài khoản (`docs/s40-private.md` mục 5). Public có lần nào dùng 2 tài khoản không? (để mô tả chi phí đúng).

## C. Số liệu LB lệch hoặc thiếu nguồn

1. Không tìm thấy trong docs (chỉ có ở bảng Vu): JINA-K40 0,340, BGR-K40-doc 0,343 (là Doc P? bảng ghi chung với "J4-K40 Doc P 0,376"), CH-clean 0,3580, CH-cap2048 0,3606 (Chunk F2), 120A-p4b-v3 0,4176 (docs chỉ có "free-v2 thua p4b-v3 0,0075"; thiếu Doc / Chunk). Xin xác nhận và P/R nếu có.
2. Chunk F2 của các probe chunk (win768, win384, sec-1, clean, cap2048, full+w768): xin Final / Chunk P / R nếu có. Doc của các probe này là J3-thrC (Doc F2 0,4161), riêng "full body 0,3594" là K4-thr-C-extD (doc K4-thr-C, khác J3 9,35 doc/query): so sánh trong hình chunk-size lệch một chút về doc; có chấp nhận không?
3. "Encode theo phần" (PART-thrC, `docs/PROGRESS.md:297`): có số LB không? Docs ghi dựng xong, chưa thấy điểm.
4. Ngày LB của J3-thrC (0,4190) và V1-v+10 (0,2683) không ghi trong docs; CSV đang để 23/10 và 18/10.
5. Bảng Vu bắt đầu ở K80-v2 0,2334 (13/10). Docs còn chuỗi trước đó: corpus cũ 03–13/10 (sub06 0,0626 → cc-A-cut115 0,2279), dãy S6 / C1 / R1m 13–15/10 (0,2483 → C1 0,2655 → R1m 0,2673; `submissions/LOG.md`, `docs/PROGRESS.md:15`). Có đưa vào hình tiến trình không? "K80 LZMA" của bảng Vu có đúng là K80 v2 (ZIP 78.734.571 B, 13/10, `docs/PROGRESS.md:422`)?
6. Chi phí VPS "~1–4 USD": không có trong docs; xin số thực (nhà cung cấp, giờ chạy, hoá đơn).
7. Baseline BTC: `docs/PROGRESS.md:302` ghi "khoảng 38 doc/query", `reports/s33-btc-baseline.md` ghi "khoảng 30 doc/query". Số nào đúng (và nguồn: slide Mentor Day 12/10?)
8. Doc P / R còn thiếu cho: 120A-p4b-v0, 120A-p4b-v3, EF2-cut-p4b, p4b-CRfair175, FREE-nowin, FREE-split, D400, các bản CH-*, K4-thr-C (R), J3-thrC (R), J4-rrf-ph979, QRRF-nc140-ph, P-A3-nc. Chunk P / R còn thiếu cho phần lớn bản từ 24/10. Nếu Vu còn ảnh chụp LB, tôi bổ sung vào `data/lb_scores.csv`.

## D. Nội dung cần Vu quyết

1. Bảng độ phủ: đưa đủ 94 host vào phụ lục hay chỉ top 15 + tổng?
2. Bài có trình bày khung p4b (bản public cao nhất 0,4176) như kết quả chính, hay chỉ là mốc so sánh cho khung free (cấu hình private)?
3. Mục 4(e) "thiên lệch do pool" của baseline BTC: Vu muốn lập luận nào? Docs có: baseline P 0,88 ở ~30–38 doc/query, R 0,51; bge-reranker trên pool cosine không tái tạo được (Doc P 0,343). Tôi chưa thấy phân tích "gold sinh từ pool của baseline" có số đo; không viết nếu không có nguồn.
4. Nhiễu LB: có lần nào nộp lại cùng một ZIP để đo độ dao động không? (FREE-noJina-canon trùng FREE-noJina ở mọi chữ số, nhưng ZIP khác nhau.)
5. Kết quả private (sau 04/11): cập nhật bảng và tóm tắt.
6. Mục 8.2 (phân bổ số doc theo query): chờ kết quả Kfit của S41 (LB nếu có) rồi viết.

## E. Đã chốt (Vu, 27/10)

- Ba hạn chế ở outline mục 8 chỉ ghi vào bài; không mở lại cấu hình private CH-rrf.
- 8.1: pipeline kiểu baseline (hybrid chunk + bge-reranker) đã chạy ở sub02–RD150; ghi kèm điều kiện corpus v1 / chỉ vi / độ phủ thấp hơn và fingerprint hybrid 0,701 vs 0,924. (Trả lời một phần C5.)
- 8.3: ngưỡng chấp nhận |Δ Final| ≈ 0,003; bảng `data/noise_decisions.csv`; Jina bỏ vì chi phí và tính tất định.
