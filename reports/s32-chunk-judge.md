# S32: judge chấm từng chunk (Qwen3-Reranker-0.6B)

Cập nhật 2026-10-22 04:00 (UTC+7), nhánh `feat/s32-chunk-judge`. Mọi số dưới đây là số đo trên máy local; **chưa nộp bản nào, chưa có điểm LB**.

## Giả thuyết và phép thử

BTC có thể gán nhãn gold bằng cách cho LLM chấm từng chunk, doc là gold khi có ít nhất một chunk liên quan. Pipeline S29/S31 cho judge chấm 512 token đầu của doc nên bỏ lỡ đoạn trả lời nằm giữa hoặc cuối bài. Phép thử: thay điểm judge mức doc bằng điểm cao nhất trên các chunk của doc, giữ mọi thứ khác như bản đã có điểm (pool, số doc/query, nc140, nguyên body, Pharmacity, RRF k=60).

## Thiết kế

- **Pool:** đúng pool13 của S31, 281.924 cặp (query, doc), không đổi doc nào. Thêm 170 cặp mà Pharmacity lô 3 đưa vào pool 979 (chỉ dùng cho các bản ph979).
- **Chunk:** `chunks_t256` của `data/canonical-20261018`, mọi field (title, body, answer, question), nguyên văn. Alias doc đi qua owner map của S29. Pharmacity dùng `chunks_t256` của chính các lô (chunk_id lô bắt đầu từ 0 nên được đổi thành id duy nhất ≥ 1e15).
- **Chọn chunk:** với mỗi (query, doc), lấy top-3 chunk theo cosine(vector query S29, vector chunk BGE-M3 dense). Hoà thì giữ chunk đứng trước trong file. Doc ít hơn 3 chunk lấy hết.
- **Doc không có chunk** (doc thin): dùng điểm 0.6B doc_text của S29 (cùng đầu vào, không chấm lại), gắn cờ `fallback`.
- **Judge:** Qwen3-Reranker-0.6B, revision `e61197ed45024b0ed8a2d74b80b4d909f1255473` (snapshot `data/hf_model_cache`), instruction và template y hệt S29/S31 (`r2ai.s31.judge_score`), bf16, đầu vào là chunk thay cho doc_text. Chunk dài hơn 512 token bị cắt (3 chunk).

### Vector chunk bổ sung (đã Vu đồng ý)

Canonical có 5.645 chunk của 1.047 doc VOV (s24/s25) không có vector, và chunk Pharmacity chỉ có vector mức doc. Tôi embed bổ sung 7.668 chunk (1.322 doc) bằng đúng `r2ai.index.bge_m3.M3Encoder` (fp16, max_len 512).

**Xác minh vector** (1.000 chunk đã có vector, seed 32, cùng encoder): cosine giữa vector mã hoá lại và vector đã lưu có TB 0,999998, min 0,999874, p01 0,999912, 0 chunk dưới 0,999 (`pool/vector_verify.json`). Chọn lại độc lập top-3 cho 300 cặp bằng đường code khác (alias qua danh sách `doc_ids`, vector đọc thẳng `dense.npy`): 0 chỗ lệch; 6.000 dòng shard so với bảng gốc: 0 lệch text (`pool/audit.json`).

### Đặc điểm chunk (`pool/manifest.json`)

| | |
|---|---|
| Cặp (query, chunk), pool13 | **824.813** (top-2: 558.210; hạng 3: 266.603) |
| Cặp thêm cho ph979 | 505 |
| Tổng chấm | 825.318 |
| Cặp pool13 có chunk / fallback doc_text | 279.119 / 2.805 (1.306 doc thin, không có chunk) |
| Chunk chọn mỗi cặp | 3 chunk: 266.603 cặp; 2 chunk: 12.488; 1 chunk: 28 |
| Số chunk của doc | TB 6,79; p10 3; p50 6; p90 11; p99 18 |
| Chunk duy nhất | 552.668 |
| Prompt TB | **284,6 token Qwen3** (top-2: 281,2; tối đa 802), so với 561 của doc_text 512 |
| Field của chunk được chọn | body 67,9%; **title 20,9%**; answer 6,1%; question 5,1% |

## Chạy GPU (RTX 3050 6 GB, hf, bf16)

- **Pilot** 1.000 cặp (lấy mẫu mọi hạng): 26,1 cặp/s, ETA 31.620 s (8,8 h) cho 825.318 cặp. Dưới ngưỡng 10 h nên giữ top-3, không hạ xuống top-2. Thử batch 24.576 token/64 cặp: 25,9 cặp/s, không nhanh hơn.
- **Full:** 446 part × 2.000 cặp, checkpoint theo part, resume được. Tổng thời gian chấm cộng dồn **34.003,6 s (9,4 h)**, TB khoảng 24 cặp/s. 0 NaN, 0 chunk quá 512 token khi chấm.
- **Sự cố:** lần chạy đầu và lần thứ hai sập vì `CUDA error: out of memory` (GPU dùng chung với desktop; lỗi này làm hỏng context CUDA nên không retry trong tiến trình). Tôi hạ batch xuống 8.192 token/16 cặp và thêm `scripts/s32/run_score.ps1` tự chạy lại sau mỗi lần chết (dừng nếu có NaN hoặc 3 lần liên tiếp không ra part mới). Sau đó không sập lần nào. Có một lần tạm dừng theo yêu cầu của Vu ở part 353/446.
- Điểm gộp: `gpu/qwen3-reranker-0.6b/scores-all.parquet`, SHA256 `cc117752…1a17`.

## Điểm doc và tương quan

Ba điểm doc tính từ điểm chunk (CPU, `r2ai.s32.scores reduce`), thêm biến thể theo yêu cầu của Vu:

- **C-max** = max P(yes) trên các chunk đã chấm.
- **C-top2** = trung bình hai điểm cao nhất (doc chỉ có 1 chunk thì lấy điểm đó).
- **C-mix** = max(C-max, điểm 0.6B doc_text của S29).
- **C-max-nt** = max trên các chunk không phải title; doc chỉ có chunk title thì giữ điểm title.
- Doc fallback: cả bốn điểm bằng điểm 0.6B doc_text của S29.

Phân bố: C-max p10/p50/p90 = 0,029 / 0,827 / 0,993; điểm 0.6B doc_text (S29) = 0,006 / 0,438 / 0,972. C-max cao hơn điểm doc_text ở 86,2% cặp; 13,6% cặp có C-max ≥ 0,99 (doc_text: 3,7%), nên phần trên của bảng xếp hạng bão hoà. C-max là max của ba điểm tương quan nên lệch lên so với điểm doc; RRF chỉ dùng thứ hạng trong query nên không bị ảnh hưởng bởi thang.

**Chunk tốt nhất nằm ở đâu** (cặp pool13 có chunk): hạng cosine 1: 47,9%; hạng 2: 32,7%; hạng 3: 19,4%. Tức 52,1% doc có chunk tốt nhất không phải chunk cosine cao nhất. **Chunk title là chunk tốt nhất ở 18,8% doc** (C-max đến từ title); C-max-nt khác C-max ở 18,4% cặp. Điểm TB theo field: body 0,494; title 0,417; answer 0,378; question 0,288.

**Spearman trên 281.924 cặp pool13** (toàn cục / TB theo query / p10 theo query):

| Cặp điểm | Toàn cục | TB/query | p10/query |
|---|---|---|---|
| **C-max ~ 0.6B doc_text (S29)** | 0,860 | 0,758 | 0,618 |
| **C-max ~ 4B doc_text (S31)** | 0,678 | 0,585 | 0,388 |
| **C-max ~ cosine** | 0,404 | 0,321 | 0,151 |
| C-max ~ C-max-nt | 0,966 | 0,936 | 0,869 |
| C-max ~ C-top2 | 0,951 | 0,921 | 0,820 |
| C-top2 ~ 0.6B doc_text | 0,871 | 0,772 | 0,632 |
| C-mix ~ 0.6B doc_text | 0,880 | 0,791 | 0,658 |
| C-max-nt ~ 0.6B doc_text | 0,844 | 0,731 | 0,575 |
| (tham chiếu) 0.6B ~ 4B | 0,737 | 0,687 | 0,493 |
| (tham chiếu) 4B ~ cosine | 0,442 | 0,364 | 0,179 |

Chỉ tính cặp có chunk (bỏ fallback) cho số gần như y hệt (`scores/correlations.json`). C-max xa 4B doc_text hơn điểm 0.6B doc_text (0,678 so với 0,737) và gần 0.6B doc_text hơn nhiều so với 4B.

## Bản dựng (CPU, chưa nộp)

Builder `r2ai.s32.build` dùng `r2ai.s29.build._build_variant` và `r2ai.s31.select.select_rrf` không đổi: 1.200 query, 195.147 doc (162,6225 doc/query = `len(relevant_docs)` của K90-ph), n_c ×1,4, nguyên body, RRF k=60. Validator S26 (`scripts/s26/validate_pa3.py` + lô Pharmacity) báo `ok`, 0 lỗi cả ba ZIP.

| Bản | Pool | Nguồn RRF | Mốc | Doc thay/query so mốc (đối xứng) | Query y hệt mốc | ZIP byte | ZIP SHA256 | JSON SHA256 | Chunk |
|---|---|---|---|---|---|---|---|---|---|
| **C06-rrf** | pool13, Pharmacity 612 | cosine × C-max | QRRF-nc140-ph | **13,57** (27,15) | 4/1200 | 99.237.610 | `d59e6e8be6c27e3f60002094c92c3632606e91cf416c891316ab0321c463b0a7` | `8c482126…c2c0` | 175.379 |
| **C06-rrf4** | ph979 (Pharmacity 979) | cosine × C-max × 4B doc | J4-rrf-ph979 | **12,67** (25,35) | 1/1200 | 99.147.433 | `8e75135d7cd2a0fd8dd17f89cc65c3524b295e3f36cbf93b9b90a8e19b9b845a` | `76ce297c…bc87` | 175.192 |
| **C06-rrf-nt** | pool13, Pharmacity 612 | cosine × C-max-nt | QRRF-nc140-ph | **14,28** (28,57) | 4/1200 | 99.908.887 | `524a92397de820631e58e28e342075fe811e60e5f54e16bce7025402bea2b489` | `ae2560bc…c81a` | 175.366 |

Mọi ZIP dưới trần 102.600.000 byte. C06-rrf-nt khác C06-rrf 4,79 doc/query (9,58 đối xứng, 50 query y hệt). C06-rrf4 chọn 131 doc Pharmacity lô 3 (J4-rrf-ph979: 91). Sidecar từng bản: `build/<tên>.s32.json`; thống kê theo query: `build/<tên>.query_stats.parquet`.

**Hồi quy (cổng):**

- `Check-q06-rrf` (thay C-max bằng điểm 0.6B doc_text S29, qua đường code S32) ra JSON SHA256 `da6cf34bbb71b25a1662c904f9db4e7a0e33408301137ca5c73c335595d63fa7` = JSON của QRRF-nc140-ph; 1.200/1.200 query y hệt doc set mốc.
- `Check-j4-rrf` (RRF cosine × 4B trên pool 979 qua đường code S32) ra đúng JSON SHA256 đã ghi trong sidecar của J4-rrf-ph979 (`965331b3…3950`), 1.200/1.200 query y hệt. Hai cổng xác nhận phần pool, điểm và chọn doc; khác biệt của ba bản trên chỉ đến từ nguồn rank chunk.

## Cách đọc LB

Các bản cùng pool, cùng số doc/query, cùng chunk nc140, cùng Pharmacity và RRF k=60; chỉ khác nguồn rank.

- **C06-rrf so với QRRF-nc140-ph (0,3940):** thay duy nhất điểm 0.6B doc_text bằng điểm 0.6B max-chunk. Cao hơn nghĩa là chấm theo chunk giúp, ủng hộ giả thuyết BTC chấm theo chunk. Bằng hoặc thấp hơn nghĩa là đoạn trả lời giữa/cuối bài không phải điểm yếu, hoặc max-chunk thêm nhiễu (title, chunk ngắn).
- **C06-rrf4 so với J4-rrf-ph979 (0,3997):** C06-rrf4 thêm C-max làm nguồn thứ ba, nên chênh lệch pha trộn hai hiệu ứng (thêm nguồn thứ ba, và nguồn đó là chunk). Để tách riêng hiệu ứng chunk, so với `J4-rrf3-ph979` (cosine × 0.6B doc_text × 4B, cùng cấu trúc ba nguồn) nếu có LB.
- **C06-rrf-nt so với C06-rrf:** cùng nguồn, chỉ khác có tính chunk title hay không. Cao hơn nghĩa là chunk title làm nhiễu.
- Điểm C-max bão hoà ở đầu (13,6% cặp ≥ 0,99), nên thứ hạng giữa các doc rất cao do nhiễu và cosine quyết định phần lớn; nếu LB không đổi thì chưa kết luận được giả thuyết sai.

## File và lệnh

Code: `src/r2ai/s32/chunks.py` (prepare, audit), `scores.py` (reduce), `build.py`; `scripts/s32/run_score.ps1`; test `tests/test_s32_chunk_judge.py` (10 test). Dữ liệu: `out/runs/s32-chunk-judge/{pool,gpu,scores,build,build-check}`.

```bash
python -m r2ai.s32.chunks prepare            # chọn chunk, embed bổ sung, kiểm 1.000 vector, ghi shard
python -m r2ai.s32.chunks audit              # chọn lại độc lập 300 cặp
powershell -File scripts/s32/run_score.ps1   # chấm GPU, tự chạy lại sau CUDA OOM
python -m r2ai.s31.judge_score collect --pool .../pool --out .../gpu --model qwen3-reranker-0.6b --subset all
python -m r2ai.s32.scores reduce --scores .../gpu/qwen3-reranker-0.6b/scores-all.parquet
python -m r2ai.s32.build --variant C06-rrf --variant C06-rrf4 --variant C06-rrf-nt
```

## Hạn chế

- Điểm title: chunk title chiếm 20,9% số chunk được chọn và 18,8% số doc có chunk tốt nhất là title; đã thêm C-max-nt để đo ảnh hưởng.
- Chỉ chấm top-3 chunk theo cosine, không chấm hết chunk của doc (TB 6,8 chunk/doc), nên chunk trả lời có cosine thấp vẫn có thể bị bỏ.
- Pool chỉ có doc đã vào top cosine; chấm chunk không thêm doc mới vào pool.
- 2.805 cặp (1,0%) là fallback doc_text vì doc thin không có chunk.
- Chưa có điểm LB nên chưa kiểm được giả thuyết.
