# S35: encode theo phần (cos_doc2) và Jina Reranker v3.5

Nhánh `feat/s35-part-jina`, base `81dad2a` (S33). Hai ý từ slide Mentor Day 2 (22/10): encode theo phần (slide 7) và Jina Reranker v3.5 (slide 10). Quyết định của Vu (22/10): dựng **y hệt K4-thr-C** (pool13, K_q' kẹp [0,5; 1,3]·K_q, t = 0,0775), chỉ thay nguồn hạng trong RRF, để đo giá trị thuần của tín hiệu; bản cuối K_q-free sẽ dựng riêng, gộp mọi tín hiệu thắng (chưa làm). Mọi bản **chưa nộp**.

Ghi chú tên: yêu cầu gọi bge-reranker là "S34"; trong repo đó là S33 (`feat/s33-btc-baseline`, R-doc / R-chk). Cả hai chỉ là cùng một nguồn điểm.

## Phần 1: cos_doc2 (CPU, đo)

`python -m r2ai.s35.part` → `out/runs/s35-part-jina/part/` (`part_table.parquet`, `part_report.json`).

- `cos_doc512` = `cosine_score` của pool (BGE-M3 doc-level, doc_text 512). `cos_part` = max cosine(query S29, vector chunk t256) trên các chunk của doc, lấy từ `chunk_cosine` của `chunk_rank = 1` trong pool R-chk của S32/S33 (S32/S33 đã chấm cosine **mọi** chunk của doc owner rồi giữ top-3, nên hạng 1 là max). `cos_doc2 = max(cos_doc512, cos_part)`; doc không có chunk giữ `cos_doc512`.
- **Embed bổ sung: 0 chunk ở S35.** Vector chunk lấy từ canonical (`dense.npy`); chunk thiếu vector đã được embed trước đó cùng `M3Encoder` (S32: 7.668 chunk / 1.322 doc, cổng `vector_verify` trung bình 0,99999; S33: 451 chunk embed thêm + 488 dùng lại vector S32). Hai nguồn trùng nhau ở cặp chung: lệch tối đa 1e-9, kiểm trong hook.

Pool top-211 (253.200 cặp, 1.200 query):

| Đại lượng | Giá trị |
|---|---|
| Cặp có chunk / không chunk | 249.458 / 3.742 (doc thin, không có nguồn chunk) |
| **% cặp có cos_part > cos_doc512** | **81,56%** (203.448 cặp); 80,35% tính cả cặp không chunk. Theo query: P10 68,2%, median 82,0%, P90 91,5% |
| cos_part − cos_doc512 (cặp có chunk) | P10 −0,008, median +0,023, P90 +0,082; TB cos_doc512 0,6303, cos_part 0,6617, cos_doc2 0,6625 |
| P4B trung bình | cặp part thắng 0,410, cặp doc thắng/hoà 0,283 |

Spearman trên 253.200 cặp (toàn cục / TB theo query / P10 theo query):

| Cặp | Toàn cục | TB query | P10 query |
|---|---:|---:|---:|
| cos_doc2 ~ cos_doc512 | 0,794 | 0,528 | 0,370 |
| cos_doc2 ~ cos_part | 0,993 | 0,981 | 0,952 |
| **cos_doc2 ~ P4B** | **0,528** | 0,358 | 0,172 |
| cos_doc512 ~ P4B | 0,499 | 0,355 | 0,173 |
| cos_part ~ P4B | 0,521 | 0,355 | 0,167 |

Hạng cos_doc2 khác hạng cosine nhiều (dịch hạng tuyệt đối median 39, P90 96; top-40 theo cosine và theo cos_doc2 khác nhau 20,6 doc/query). Đọc: tín hiệu part gần cos_part (0,993), tương quan với P4B tăng nhẹ ở mức toàn cục (+0,029) và gần như không đổi theo query (+0,003).

### PART-thrC

`python -m r2ai.s35.build --variant Check-PART-id --variant PART-thrC` → `out/runs/s35-part-jina/build-part/` (qua `r2ai.s31.build.build`, thêm `rank_hook` / `rank_keys_override` / `variant_alias`, mặc định không đổi hành vi).

Quy tắc = K4-thr-C: pool13 (round(1,3·K_q), Pharmacity 979), điểm 4B, K_q' kẹp [0,5; 1,3]·K_q với t = 0,07751765647688268 (tổng 195.147 doc), chunk = một chunk nguyên body cho doc đã chọn có P4B ≥ t_c, t_c hiệu chỉnh để tổng chunk = 175.148. Khác duy nhất: RRF(**hạng cos_doc2 trong pool13**, hạng 4B) thay RRF(hạng cosine, hạng 4B). Hạng 4B (và hoà của nó) vẫn tính theo hạng cosine cũ, nên K_q' và t **giống hệt** K4-thr-C: số doc mỗi query bằng nhau, chỉ đổi doc nào được chọn. cos_part có đủ cho 281.924 cặp pool13 (279.119 có chunk, 2.805 không chunk).

| Bản | Doc | Chunk | t_c | ZIP B | JSON SHA256 | ZIP SHA256 |
|---|---:|---:|---:|---:|---|---|
| K4-thr-C (LB 0,4183) | 195.147 | 175.148 | 0,070560 | 100.471.357 | `d771c4f9…c9b0` | `31980453…5a87` |
| **Check-PART-id** (cos_doc2 := cos_doc512) | 195.147 | 175.148 | 0,070560 | 100.471.367 | `d771c4f9b0b5f7b8c68f934b28aeb207179a047a2f7a83f29c7b62628939c9b0` | `3fd9b9d1…14d5` |
| **PART-thrC** | 195.147 | 175.148 | **0,075858** | 101.507.086 | `feb2b359b3baea69…` | `f17e8c0dcd867bd8…` |

- **Hồi quy đạt:** Check-PART-id ra đúng JSON K4-thr-C (1.200/1.200 query giống hệt). ZIP dài hơn 10 B chỉ vì tên file trong header ZIP.
- PART-thrC khác K4-thr-C **17,03 doc/query** (sym-diff 34,06; chỉ 1 query giống hệt); khác J4-rrf 27,0 doc/query. ZIP dưới trần 102.600.000 B, không cắt body (N = 0), mọi query có chunk. Validator S26 báo `ok: true`, 0 lỗi cả hai ZIP.
- **t_c khác 0,0706:** giữ nguyên quy tắc K4-thr-C (tổng chunk 175.148) thì t_c hiệu chỉnh ra 0,075858, vì PART chọn doc có P4B cao hơn. Nếu cố định t_c = 0,0706 thì có khoảng 177.200 doc đủ ngưỡng (≈ +2.000 chunk, ZIP thêm ≈ 1,2 MB, vượt trần 102,6 MB và phải cắt body), nên không dùng. **Vu xác nhận (22/10): giữ t_c hiệu chỉnh về 175.148 chunk.**

## Phần 2: Jina Reranker v3.5 (GPU local)

- Model `jinaai/jina-reranker-v3.5`, revision `e8a93f33f0b22108f8c2364f8484ce3422552fbc`. **Commit 30/07/2026 11:35 UTC** (đã kiểm bằng API commits của HF; trước 01/08/2026; commit trọng số cuối 21/07, README/GGUF sau đó), **CC-BY-NC-4.0** (phi thương mại, hợp luật cuộc thi), **0,6B** tham số (596.836.352, BF16, nền Qwen3-0.6B). Vu cho phép tải và chạy `trust_remote_code` (Vu, 22/10); `modeling.py` đã đọc: không gọi mạng ngoài nạp tokenizer, không ghi file. 12 file, 1,19 GB vào cache HF.
- Cách gọi: listwise. Mỗi lần forward đặt mọi doc của một lô vào cùng một ngữ cảnh (`<passage id=i>` + token embed), query ở đầu và cuối; embedding doc chiếu về 512-d, điểm = cosine với embedding query của lô. `rerank()` gốc gộp nhiều lô bằng trung bình có trọng số `max((1+s)/2)` của embedding query rồi chấm cosine từng doc với một vector query duy nhất.
- Bản của S35 (`r2ai.s35.jina`) chạy lô nhỏ hơn và khớp `rerank()` gốc với 1 lô (lệch tối đa 6e-8). Sửa cần thiết cho 6 GB: HF dựng mask n×n và `enable_gqa` làm torch rơi về kernel math (24–42 GiB ở 25–37k token, OOM); dùng attention causal cuDNN không mask, lặp KV. Đỉnh VRAM 1,7 GiB cho lô 50 doc dài nhất (25,6k token), nên **vừa 6 GB, không cần Kaggle**.
- Pool chấm: J4-K40 pool (cosine top-211, 253.200 cặp) ∪ pool13 mở rộng = **302.086 cặp** (253.200 + 48.886 cặp chỉ có trong pool13), TB 251,7 doc/query (min 211, max 339). `doc_text` = văn bản S31 cắt 512 token Qwen3, giống 4B / bge. J3-thrC cần cả pool13 nên phải chấm phần 48.886 cặp thêm.

### Pilot 20 query (cosine top-211, RTX 3050 6 GB, bf16)

| Layout | s/query | tok/s | Spearman TB query với P4B | với cosine | Spearman với b50s0 | top-40 trùng b50s0 | ETA 1.200 query (302k cặp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| b50s0 (lô 50, xáo trộn) | 33,2 | 2.870 | 0,646 | 0,332 | – | – | 13,2 h |
| b50s1 (xáo trộn khác) | 33,2 | 2.871 | 0,638 | 0,328 | 0,892 | 0,77 | 13,2 h |
| b50cos (lô 50, thứ tự cosine) | 33,3 | 2.864 | 0,601 | **0,118** | 0,840 | 0,69 | 13,2 h |
| b25s0 | 23,7 | 4.018 | 0,644 | 0,327 | 0,908 | 0,80 | 9,4 h |
| **b10s0 (chọn)** | **17,1** | 5.568 | **0,662** | 0,316 | 0,882 | 0,78 | **6,8 h** |

Pilot 2 (cùng 20 query): b10s0 / b10s1 / b10s2 / b25s1. Hai lần xáo trộn khác nhau của lô 10 chỉ đồng ý Spearman 0,866 (P10 0,80); trung bình 3 lần xáo trộn có Spearman với P4B 0,686 so với 0,662 một lần (đồng ý với lần giữ lại 0,893 so với 0,867).

**Chọn cách chuẩn hoá và lô (báo theo yêu cầu):**
- Điểm so được giữa các lô nhờ **một vector query chung** (cách gộp của chính model), không z-score theo lô: Spearman điểm gộp với cosine-lô là 0,997, nhưng z-score theo lô kém ổn định hơn rõ (0,77 so với 0,87 giữa hai lần xáo trộn ở lô 10; 0,885 so với 0,892 ở lô 50). Không dùng lô chồng lấn.
- **Lô 10 doc, thứ tự xáo trộn cố định** (seed = 35 + hạt layout, query_id): thứ tự cosine làm doc đầu lô bị lệch (Spearman với cosine tụt 0,12), lô 10 không kém lô 50 (còn khớp P4B hơn) và nhanh gấp 2. Ngưỡng ≤ 50 doc/lần vẫn thoả.
- Một lượt xáo trộn là nhiễu đáng kể (0,87). Lượt 1 `b10s0` đang chạy; các lượt `b10s1`, `b10s2` (mỗi lượt 6,8 h) có thể thêm và lấy trung bình (`collect --layout b10s0,b10s1`).

### Chạy và dựng

`scripts/s35/run_part2.sh b10s0`: chấm 302.086 cặp (checkpoint mỗi 10 query, chạy lại tự bỏ qua part đã có) → `JINA-K40` + `Check-J4-K40` (`r2ai.s35.topk`) → `J3-thrC` (`r2ai.s35.build --variant J3-thrC`) → validator S26. Nhật ký `out/runs/s35-part-jina/part2.log`. Tiếp tục sau khi máy tắt: chạy lại đúng lệnh trên.

Kết quả đầy đủ của phần này được ghi bên dưới khi chạy xong.
