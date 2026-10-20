# S31: judge mạnh hơn (Qwen3-Reranker-4B/8B) trên pool cosine

Nhánh `feat/s31-strong-judge`, base `d07bdd9` (S29). Luật: chỉ model open-weight ≤ 15B, phát hành trước 01/08/2026, không dùng API đóng. Qwen3-Reranker-0.6B/4B/8B (06/2025) và Qwen3-14B (04/2025) đạt luật.

## Pha 1: chuẩn bị (CPU local), đã chạy 20/10/2026

### Pool 2,0 × K_q

`python -m r2ai.s31.pool export` → `out/runs/s31-strong-judge/pool/` (ignored).

- `K_q` = số doc P1 (như S29); pool mỗi query = `round(2,0 × K_q)` theo cosine BGE-M3 doc-level, vector S26 + 612 doc Pharmacity, query vector đúng file S29 (`query_vectors.npz` SHA `eed0e104…edca`).
- **Đo:** 1.200 query, **433.732 cặp** (TB 361,44 doc/query, min 170, max 522), trong đó 281.924 cặp thuộc pool 1,3× (`in_pool13`); 220.613 doc khác nhau; vi 319.787 / zh 113.945 cặp.
- Ranks đầu `round(1,3 × K_q)` chép nguyên từ pool S29 (SHA `708482f7…5222`), nên mọi điểm 0.6B của S29 dùng lại được và J4-rrf trên pool 1,3× so trực tiếp với Q-rrf. 10 query có hoà cosine tuyệt đối ở biên cắt S29 (doc trùng vector, tổng 211 doc): giữ lựa chọn của S29, các doc hoà còn lại xếp đầu phần đuôi. Ghi trong `manifest.json` (`boundary_tie_swaps`).
- `doc_text` cắt 512 token bằng tokenizer Qwen3 (`truncate_document` của S29, Vu chọn 20/10; không dùng tokenizer BGE-M3). Cả họ Qwen3 dùng chung tokenizer này. Kiểm đối chiếu trên 3.000 doc ngẫu nhiên: 0 khác biệt so với cách cắt S29. TB 439,8 token/doc, 151.936 doc chạm trần 512.
- 22 shard parquet zstd, **tổng 299.365.751 B** (lớn nhất 14,5 MB, ≤ 500 MB). Cột: `pair_id, query_id, doc_id, cosine_rank, cosine_score, in_pool13, lang, query_text, doc_text, doc_tokens`. `pool_index.parquet` (không text) dùng cho builder local.
- `python -m r2ai.s31.compare ref` → `s29_qwen06_scores.parquet`: điểm 0.6B S29 theo `pair_id` (281.924 cặp, SHA `0cd38441…fe55`), dùng cho cổng hồi quy trên máy GPU.

### Script chấm `src/r2ai/s31/judge_score.py`

Một file, chỉ cần numpy/pandas/pyarrow/torch/transformers (vLLM tuỳ chọn), copy được sang Kaggle/RunPod. Prompt, instruction vi/zh và đọc logit yes/no giống hệt S29.

- Model: `qwen3-reranker-0.6b` (pin `e61197ed…`), `qwen3-reranker-4b`, `qwen3-reranker-8b`, `qwen3-14b` (generative judge, cùng template yes/no, tắt thinking, lấy P(yes)), `qwen3-14b-awq` (chỉ backend vLLM).
- Backend `hf` (mặc định, đã chạy local) hoặc `vllm` (**chưa chạy thử**: máy local Windows không có vLLM; chỉ dùng sau khi pilot vLLM khớp pilot hf).
- Checkpoint: mỗi part 2.000 cặp ghi 1 parquet (rename nguyên tử) + JSON thời gian/GPU. Chạy lại tự bỏ qua part đã có; `--resume-from` chép part từ output phiên trước (Kaggle). `--worker i --workers n` chia part cho nhiều GPU. `--max-seconds` dừng trước giới hạn phiên. `collect` kiểm đủ part, đúng thứ tự `pair_id`, ghi `scores.parquet` và tổng giây GPU.
- dtype `auto`: bf16 nếu GPU sm80+, ngược lại fp16 (T4). Logit không hữu hạn → dừng part, báo lỗi (không ghi điểm hỏng).
- `--device-map auto` cho phép chia 8B qua 2×T4 (chậm, xem bảng dưới).

### Kiểm hồi quy 0.6B (RTX 3050 6 GB, transformers 5.18.0, torch 2.11.0+cu128)

1.000 cặp ngẫu nhiên (seed 42) trong pool 1,3×, so điểm S29:

| dtype | Spearman | Pearson | |Δ| TB | |Δ| max | tốc độ |
|---|---|---|---|---|---|
| bf16 | **0,99960** | 0,99952 | 0,0062 | 0,072 | 11,99 cặp/s |
| fp16 | **0,99966** | 0,99959 | 0,0066 | 0,071 | 11,80 cặp/s |

Cổng ≥ 0,99 qua cả hai. Δ còn lại do padding theo lô khác S29 (S29 lô 8 không sắp xếp; S31 sắp theo độ dài). Chạy `score` thật 1 part (2.000 cặp, bf16): 10,09 cặp/s; 1.332 cặp thuộc 1,3× có Spearman 0,99959 với S29. Prompt TB 561 token, max 965.

## Pha 2: chạy GPU (Vu chọn nền tảng)

Notebook `notebooks/s31-kaggle-judge.ipynb` (mặc định Kaggle 2×T4, fp16, Internet bật, không dùng token/secret nào):

1. Upload thư mục `out/runs/s31-strong-judge/pool/` thành một Kaggle dataset riêng tư (đã có sẵn `judge_score.py`, `manifest.json`, `s29_qwen06_scores.parquet`, `shards/`).
2. Notebook: tải model công khai → cổng 0.6B fp16 so S29 (dừng nếu Spearman < 0,99) → pilot 4B 1.000 cặp (tốc độ, ETA) → 2 worker (mỗi T4 một tiến trình) dừng trước 12h → `collect` nếu đủ part.
3. Chạy dở: thêm output notebook làm input phiên sau, ghi đường dẫn vào `RESUME`, đặt `REVISION` = commit in ra ở phiên đầu để mọi phiên dùng cùng trọng số.

Có GPU 24 GB (AWS g5 / RunPod): chạy trực tiếp

```bash
python judge_score.py pilot --pool POOL --out OUT --model qwen3-reranker-8b --pairs 1000
python judge_score.py score --pool POOL --out OUT --model qwen3-reranker-8b
python judge_score.py collect --pool POOL --out OUT --model qwen3-reranker-8b
```

### Thời gian và chi phí GPU (để lập kế hoạch, **chỉ 0.6B là số đo**)

Số đo duy nhất: 0.6B trên RTX 3050 laptop 12,0 cặp/s ⇒ pool 2,0× ≈ 10h03m một GPU, pool 1,3× ≈ 6h32m. Các dòng khác là **ước tính** theo FLOP (2 × tham số không-embedding × 561 token/cặp; hiệu suất giả định ghi trong bảng), sai số có thể ±50%; pilot 1.000 cặp trên máy thật sẽ thay các số này.

| Model | Máy | Giả định thông lượng | Pool 2,0× (433.732) | Pool 1,3× (281.924) | Chi phí |
|---|---|---|---|---|---|
| 0.6B | RTX 3050 6 GB (local) | đo 12,0 cặp/s | 10,0 h | 6,5 h | 0 |
| 4B fp16 | Kaggle 2×T4 | ~20 TFLOPS/T4 → ~5 cặp/s/T4, ~10 cặp/s | ~12 h (2 phiên) | ~8 h | 0 (quota Kaggle 30 h GPU/tuần) |
| 8B fp16 | Kaggle 2×T4, `--device-map auto` | ~2,5 cặp/s (2 GPU nối tiếp) | ~48 h | ~31 h | vượt quota tuần, không khả thi trước 31/10 |
| 8B bf16 | AWS g5.xlarge (A10G 24 GB) | ~35 TFLOPS → ~4,5 cặp/s | ~27 h | ~17 h | ~27 USD / ~17 USD ở giá on-demand ~1,0 USD/h (giá tham khảo, cần kiểm lại) |
| 8B bf16 | RunPod RTX 4090 24 GB | ~70 TFLOPS → ~9 cặp/s | ~13 h | ~9 h | ~5–9 USD ở 0,4–0,7 USD/h (giá tham khảo, cần kiểm lại) |
| 14B AWQ | vLLM, T4/24 GB | chưa ước tính; backend vLLM chưa chạy thử | – | – | – |

Giai đoạn private (01–04/11): nếu private là bộ query mới cùng cỡ, phải chấm lại toàn pool mới với cùng chi phí trên; chọn model theo số pilot đo được và theo LB pha public.

## Pha 3: dựng bản (chờ builder S29)

Theo Vu (20/10): chờ S29 push builder (`chunk_multiplier`, weighted RRF), merge `feat/s29-qwen-judge` vào nhánh này trước Pha 3; kiểm hồi quy sau merge: dựng lại Q-rrf (n_c P1) phải ra đúng JSON SHA `6b21fbb5…9afc` của bản S29 đã nộp.

Đã có sẵn và có test (`src/r2ai/s31/select.py`):

- `select_rrf(rows, keep, rank_keys, k=60)`: RRF nhiều nguồn; với `(cosine_rank, qwen_rank)` cho kết quả và điểm trùng từng bit với `select_qrrf` của S29 (test trên 300 doc có điểm hoà).
- `doc_set_diff`, `judge_correlations`: số doc khác Q-rrf TB/query, Spearman toàn cục và theo query giữa các judge.

Các bản (doc = số doc K90-ph/query, chunk nc140, JSON gọn): J4-rrf (pool 1,3×), J4-rrf-pool20 (pool 2,0×), J4-rrf3 (cosine + 0.6B + 4B, pool 1,3×); J8-* nếu có điểm 8B. J4-rrf3-pool20 chỉ làm khi J4-rrf-pool20 thắng J4-rrf trên LB (cần chấm thêm 0.6B cho phần 1,3×–2,0×).
