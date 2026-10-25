# Kịch bản bộ private (viết và chạy khô, 26/10/2026)

Chưa chạy GPU, chưa embed, chưa chấm gì. Phần chạy được là `python -m r2ai.release plan [--new-queries N]` (đọc tệp, kiểm đủ điều kiện đầu vào, in bước và giờ ước tính). Số giờ lấy từ tốc độ đã đo ở S35 đến S37; bước nào chưa từng đo thì ghi "chưa đo", không đoán.

## A. Private dùng lại đúng 1.200 query

Không cần GPU và không cần mạng. Chạy hai lệnh ở `README_RELEASE.md` mục 3 (`r2ai.release build` cho `120A-p4b-v3` và `120A-free-v2`), mỗi bản khoảng 20 phút. Kết quả trùng ZIP đã bàn giao từng byte (đã kiểm 26/10). Chọn bản nộp là quyết định của Vu theo LB công khai.

## B. Private có bộ query mới (N query)

Chỉ dùng được bản **free-v2**. Lý do: bản p4b-v3 lấy pool là cosine top-n_q với n_q = round(1,3 K_q) của pool13 cũ, K_q là số doc của bản nền cho từng query của bộ công khai; query mới không có K_q và pool13. Khung free chỉ cần cosine top-300 của corpus nên dùng được với mọi query.

Dữ liệu đã có sẵn (corpus cố định: vector doc S26 + segment 120ask + Pharmacity, mô hình và scorer ghim): không phải embed lại doc, không phải chấm lại cặp của bộ công khai. Mọi cặp (query mới, doc) đều chưa có điểm, nên phải chấm 4B, C-max, Jina cho toàn bộ pool top-300 của từng query.

| bước | việc | nơi chạy | mã / lệnh | ước tính cho N = 1.200 |
|---|---|---|---|---|
| 1 | Embed query (BGE-M3 dense, công thức S29, revision `5617a9f6…`), ghi `query_vectors.npz` đúng bố cục S29 | GPU máy | `r2ai.s29.qwen_judge` (ghi `query_vectors.npz`) | chưa đo (1.200 query ngắn) |
| 2 | Pool: cosine top-400 của corpus + 120ask/Pharmacity cho từng query, `pool.npz` | máy | `r2ai.s37.ingest.build_pool` (đọc query qua `load_queries()`, trỏ sang tệp query mới), `r2ai.s36.d300.merged_top` | chưa đo |
| 3 | Xuất gói 4B top-300: 360.000 cặp, tách thành nhiều dataset; Vu chạy `notebooks/s36-kaggle-4b-final.ipynb` (2×T4) trên Kaggle | Kaggle | `python -m r2ai.s36.d300 export-final --depth 300`, nhập điểm bằng `import4b` | 360.000 cặp ÷ 20.900 cặp/giờ ≈ **17,2 giờ**, tức hơn một phiên Kaggle (12 giờ theo thông tin chung về Kaggle, cần kiểm lại), chia ít nhất 2 phiên |
| 4 | C-max (Qwen3-Reranker-0,6B, công thức S32): 360.000 cặp × 2,67 chunk = 961.200 dòng | GPU máy | `scripts/s37/d300_cmax.py`, `scripts/s37/run_d300_cmax.ps1` | 961.200 ÷ 28,6 dòng/s ≈ **9,3 giờ** |
| 5 | Jina Reranker v3.5 listwise, layout b10s0, 1.200 query | GPU máy | `r2ai.s37.jinapool`, `scripts/s37/run_jina.ps1` | 1,49 đến 1,96 query/phút ≈ **10,2 đến 13,4 giờ** (GPU 6 GiB dùng chung, từng bị OOM, phải chạy tuần tự với bước 4) |
| 6 | `apply` (khung free, `pool: fixed`, `depth: 300`) → compose → validator: sao chép `configs/s36-handoff-120A-free-v2.json`, trỏ `pairs_new`, `scores_4b`, `cmax`, `jina` sang tệp mới | CPU | `python -m r2ai.release build --config <config mới>` (phải thêm mục manifest cho bản mới, vì `build` so với manifest) | khoảng 20 phút |

Tổng GPU máy tuần tự cho bước 4 và 5 khoảng 20 đến 23 giờ, cộng 17 giờ Kaggle (chạy song song với máy được, vì bước 3 chạy trên Kaggle). Bước 1 và 2 chưa đo, nhưng nhỏ so với các bước còn lại.

### Điểm chưa chắc (cần làm khi thật sự có bộ query mới)

1. Mã bước 1 và 2 hiện đọc `out/runs/s29-qwen-judge/query_vectors.npz` và `candidate_pool.parquet` cố định cho 1.200 query công khai; cần trỏ sang tệp query mới. Đây là sửa mã nhỏ nhưng chưa làm và chưa chạy thử, vì private chưa có.
2. Kích thước hiệu chỉnh `a`, `b`, `s_free` (S36 `cal.json`, `ef2-free.json`) được khớp trên điểm LB của bộ công khai; áp cho bộ query mới là giả định hiệu chỉnh không đổi theo bộ query.
3. `release build` so ZIP với manifest của bản đã bàn giao; bản dựng cho query mới cần manifest riêng (không có ZIP chuẩn để so), nên chỉ kiểm được validator và tính nhất quán (dựng hai lần ra cùng SHA).
4. Không có thử nghiệm nào ở đây đo thời gian bước 1 và 2.

## Lệnh chạy khô

```bash
PYTHONPATH=src python -m r2ai.release plan                       # trường hợp A
PYTHONPATH=src python -m r2ai.release plan --new-queries 1200    # trường hợp B
```

Kết quả `plan` ngày 26/10: cả 7 điều kiện đầu vào (vector query, pool, notebook 4B, scorer ghim, script C-max, script Jina, `d300`) có mặt.
