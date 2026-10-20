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
- Backend `hf` (mặc định, đã chạy local) hoặc `vllm` (**chưa chạy thử**: máy local Windows không có vLLM). `run_runpod.sh` chỉ chuyển sang vLLM khi pilot vLLM khớp pilot hf (Spearman ≥ 0,99 trên cùng 1.000 cặp).
- Checkpoint: mỗi part 2.000 cặp ghi 1 parquet (rename nguyên tử) + JSON thời gian/GPU. Chạy lại tự bỏ qua part đã có; `--resume-from` chép part từ output phiên trước (Kaggle). `--worker i --workers n` chia part cho nhiều GPU. `--max-seconds` dừng trước giới hạn phiên. `--subset pool13|tail|all`: chỉ chấm cặp `in_pool13` (281.924 cặp, 150 part), phần đuôi 1,3×–2,0× (151.808 cặp, 86 part) hoặc tất cả; part mỗi subset có tên riêng (`pool-XXXXX.pool13.part-YYYY.parquet`). `status` in số part xong/tổng. `collect --subset S` gộp mọi part đã có (mọi cách đặt tên), kiểm `pair_id`/`query_id`/`doc_id` khớp pool, ghi `scores-S.parquet` và tổng giây GPU; `collect --subset all` chạy được sau khi xong pool13 + tail.
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

Thứ tự chung: **pool13 trước** (đủ cho J4-rrf, J4-rrf3), phần đuôi chấm ở phiên sau (chỉ cần cho J*-rrf-pool20).

### Kaggle 2×T4 (mặc định, 4B)

Notebook `notebooks/s31-kaggle-judge.ipynb` (fp16, Internet bật, không dùng token/secret nào):

1. Upload thư mục `out/runs/s31-strong-judge/pool/` thành một Kaggle dataset riêng tư (đã có sẵn `judge_score.py`, `run_runpod.sh`, `manifest.json`, `s29_qwen06_scores.parquet`, `shards/`).
2. Phiên 1 (mặc định `SUBSET = "pool13"`): tải model công khai → cổng 0.6B fp16 so S29 (dừng nếu Spearman < 0,99) → pilot 4B 1.000 cặp (tốc độ, ETA) → 2 worker (mỗi T4 một tiến trình) dừng trước 12h → `collect --subset pool13` nếu đủ part.
3. Phiên sau: thêm output phiên trước làm input, ghi vào `RESUME`; đặt `REVISION` = commit in ra ở phiên 1; còn thiếu pool13 thì giữ `SUBSET = "pool13"`, xong rồi thì `SUBSET = "tail"`, `RUN_GATE = RUN_PILOT = False`. Khi đủ cả hai, notebook gộp thêm `scores-all.parquet`.

### RunPod Linux 24 GB (8B): `scripts/s31/run_runpod.sh`

Script tự làm: venv + `vllm==0.11.0` (đổi bằng `VLLM_VERSION`) → tải 0.6B và 8B (pin commit vào `$OUT/qwen3-reranker-8b.revision`) → cổng 0.6B so S29 → pilot 8B 1.000 cặp bằng hf và vllm, Spearman(hf, vllm) ≥ 0,99 thì chấm bằng vLLM, ngược lại hf (`backend-choice.json`) → `score --subset pool13` + collect → `score --subset tail` + collect tail/all → nén `/workspace/s31-qwen3-reranker-8b-results.tar.gz`. Chạy lại sau khi mất kết nối: bỏ qua bước đã xong, part đã có giữ nguyên.

Các bước Vu gõ:

1. Trên RunPod: tạo pod 1 GPU 24 GB (RTX 4090 / A5000 / A10 / L4 24 GB), template PyTorch Ubuntu, volume `/workspace` ≥ 60 GB (8B ~16 GB, venv vLLM ~10 GB), bật SSH (khoá công khai trong Settings của RunPod).
2. Trên máy (Git Bash), gửi thư mục pool (đã chứa `run_runpod.sh`), chọn một trong hai cách:

```bash
runpodctl send out/runs/s31-strong-judge/pool
```

   rồi trên pod: `cd /workspace && runpodctl receive <mã in ra>`; hoặc dùng scp với IP/cổng SSH hiển thị trên trang pod:

```bash
scp -P <PORT> -r out/runs/s31-strong-judge/pool root@<IP>:/workspace/
```

3. Trên pod:

```bash
tmux new -s s31
```

```bash
bash /workspace/pool/run_runpod.sh 2>&1 | tee -a /workspace/s31-run.log
```

   Thoát tmux không dừng job: `Ctrl-b d`; vào lại: `tmux attach -t s31`. Chỉ chấm pool13: thêm `SUBSETS=pool13` trước `bash`.
4. Lấy kết quả về máy: trên pod `runpodctl send /workspace/s31-qwen3-reranker-8b-results.tar.gz`, trên máy `runpodctl receive <mã>`; hoặc

```bash
scp -P <PORT> root@<IP>:/workspace/s31-qwen3-reranker-8b-results.tar.gz out/runs/s31-strong-judge/
```

   Giải nén vào `out/runs/s31-strong-judge/gpu/` (gồm `qwen3-reranker-8b/scores-*.parquet`, `parts/`, report, log, `pip-freeze.txt`).
5. Dừng/xoá pod để ngừng tính tiền.

Có thể lấy `scores-pool13.parquet` về ngay khi bước pool13 xong (trong `/workspace/s31-out/qwen3-reranker-8b/`) để dựng J8-rrf trong lúc phần đuôi còn chạy.

### Thời gian và chi phí GPU (để lập kế hoạch, **chỉ 0.6B là số đo**)

Số đo duy nhất: 0.6B trên RTX 3050 laptop 12,0 cặp/s ⇒ pool 2,0× ≈ 10h03m một GPU, pool 1,3× ≈ 6h32m. Các dòng khác là **ước tính** theo FLOP (2 × tham số không-embedding × 561 token/cặp; hiệu suất giả định ghi trong bảng), sai số có thể ±50%; pilot 1.000 cặp trên máy thật sẽ thay các số này.

| Model | Máy | Giả định thông lượng | Tất cả (433.732) | pool13 (281.924) | Chi phí |
|---|---|---|---|---|---|
| 0.6B | RTX 3050 6 GB (local) | đo 12,0 cặp/s | 10,0 h | 6,5 h | 0 |
| 4B fp16 | Kaggle 2×T4 | ~20 TFLOPS/T4 → ~5 cặp/s/T4, ~10 cặp/s | ~12 h (2 phiên) | ~8 h | 0 (quota Kaggle 30 h GPU/tuần) |
| 8B fp16 | Kaggle 2×T4, `--device-map auto` | ~2,5 cặp/s (2 GPU nối tiếp) | ~48 h | ~31 h | vượt quota tuần, không khả thi trước 31/10 |
| 8B bf16 | AWS g5.xlarge (A10G 24 GB) | ~35 TFLOPS → ~4,5 cặp/s | ~27 h | ~17 h | ~27 USD / ~17 USD ở giá on-demand ~1,0 USD/h (giá tham khảo, cần kiểm lại) |
| 8B bf16 | RunPod RTX 4090 24 GB | ~70 TFLOPS → ~9 cặp/s | ~13 h | ~9 h | ~5–9 USD ở 0,4–0,7 USD/h (giá tham khảo, cần kiểm lại) |
| 14B AWQ | vLLM, T4/24 GB | chưa ước tính; backend vLLM chưa chạy thử | – | – | – |

Phần đuôi (151.808 cặp) tốn thêm ~54% thời gian của pool13. Theo luật đã ghi, mỗi lần nộp gồm đủ 1.200 query và hệ thống tự tách public/private, nên điểm judge chấm một lần dùng cho cả hai giai đoạn: **chi phí GPU dự kiến chỉ một lần — chờ BTC xác nhận**.

## Pha 3: dựng bản (chờ builder S29)

Theo Vu (20/10): chờ S29 push builder (`chunk_multiplier`, weighted RRF), merge `feat/s29-qwen-judge` vào nhánh này trước Pha 3; kiểm hồi quy sau merge: dựng lại Q-rrf (n_c P1) phải ra đúng JSON SHA `6b21fbb5…9afc` của bản S29 đã nộp.

Đã có sẵn và có test (`src/r2ai/s31/select.py`):

- `select_rrf(rows, keep, rank_keys, k=60)`: RRF nhiều nguồn; với `(cosine_rank, qwen_rank)` cho kết quả và điểm trùng từng bit với `select_qrrf` của S29 (test trên 300 doc có điểm hoà).
- `doc_set_diff`, `judge_correlations`: số doc khác Q-rrf TB/query, Spearman toàn cục và theo query giữa các judge.

Các bản (doc = số doc K90-ph/query, chunk nc140, JSON gọn): **J4-rrf (pool 1,3×) dựng ngay khi có `scores-pool13.parquet` của 4B, không chờ phần đuôi**; J4-rrf3 (cosine + 0.6B + 4B, pool 1,3×) cũng chỉ cần pool13; J4-rrf-pool20 (pool 2,0×) dựng khi có `scores-all.parquet`; J8-* tương tự nếu có điểm 8B. J4-rrf3-pool20 chỉ làm khi J4-rrf-pool20 thắng J4-rrf trên LB (cần chấm thêm 0.6B cho phần 1,3×–2,0×).
