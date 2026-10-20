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

**Quyết định 21/10 (Vu):** LB 0.6B: QRRF-nc140-ph 0,3940 (tốt nhất), QRRF-pool16 0,3931, QRRF-wq0.65 0,3933 — pool rộng và trọng số không giúp. **Chỉ chấm pool13 (1,3×, 281.924 cặp); không chấm phần đuôi 1,3×–2,0× cho 4B.** Cờ `--subset tail` vẫn còn trong script nhưng không dùng.

### Kaggle 2×T4 (mặc định, 4B)

Notebook `notebooks/s31-kaggle-judge.ipynb` (fp16, Internet bật, không dùng token/secret nào):

1. Upload thư mục `out/runs/s31-strong-judge/pool/` thành một Kaggle dataset riêng tư (đã có sẵn `judge_score.py`, `run_runpod.sh`, `manifest.json`, `s29_qwen06_scores.parquet`, `shards/`).
2. `SUBSET = "pool13"` cố định (notebook dừng nếu đổi). `SESSION = 1`: tải model công khai → cổng 0.6B fp16 so S29 (dừng nếu Spearman < 0,99) → pilot 4B 1.000 cặp (tốc độ, ETA) → 2 worker (mỗi T4 một tiến trình) dừng trước 12h.
3. **Phiên 2 (mặc định hiện tại `SESSION = 2`)**: thêm output phiên 1 làm input rồi Save & Run All. Notebook tự tìm mọi `/kaggle/input/**/qwen3-reranker-4b/parts`, chép part đã xong, lấy commit model từ metadata part (dừng nếu các part khác commit), bỏ qua cổng/pilot, chỉ chấm các part pool13 còn thiếu, rồi `collect --subset pool13` khi đủ 150/150 part. Chưa đủ thì thêm output phiên 2 làm input và chạy lại.
4. Tải về `/kaggle/working/s31/qwen3-reranker-4b/scores-pool13.parquet` + `collect-pool13-report.json`, đặt vào `out/runs/s31-strong-judge/gpu/qwen3-reranker-4b/`.

### RunPod Linux 24 GB (8B): `scripts/s31/run_runpod.sh`

Mặc định `SUBSETS=pool13` (theo quyết định 21/10; phần đuôi chỉ khi đặt `SUBSETS="pool13 tail"`). Script tự làm: venv + `vllm==0.11.0` (đổi bằng `VLLM_VERSION`) → tải 0.6B và 8B (pin commit vào `$OUT/qwen3-reranker-8b.revision`) → cổng 0.6B so S29 → pilot 8B 1.000 cặp bằng hf và vllm, Spearman(hf, vllm) ≥ 0,99 thì chấm bằng vLLM, ngược lại hf (`backend-choice.json`) → `score --subset pool13` + collect → `score --subset tail` + collect tail/all → nén `/workspace/s31-qwen3-reranker-8b-results.tar.gz`. Chạy lại sau khi mất kết nối: bỏ qua bước đã xong, part đã có giữ nguyên.

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

Theo quyết định 21/10 chỉ cần cột pool13 (phần đuôi 151.808 cặp, +54% thời gian, không chấm). Theo luật đã ghi, mỗi lần nộp gồm đủ 1.200 query và hệ thống tự tách public/private, nên điểm judge chấm một lần dùng cho cả hai giai đoạn: **chi phí GPU dự kiến chỉ một lần — chờ BTC xác nhận**.

## Pha 3: dựng bản (`src/r2ai/s31/build.py`)

Builder S29 `e451fe7` (nc140, weighted RRF, JSON gọn) đã merge vào nhánh này ngày 20/10. `python -m r2ai.s31.build --variant ...` đọc `pool_index.parquet` (chỉ `in_pool13`), điểm 0.6B S29 (`s29_qwen06_scores.parquet`) và điểm judge (`--scores .../scores-pool13.parquet`), chọn `len(relevant_docs)` của K90-ph/query bằng RRF k=60 (`select_rrf`, trùng từng bit với `select_qrrf` S29 khi 2 nguồn), rồi gọi `r2ai.s29.build._build_variant` với n_c ×1,4, nguyên body, Pharmacity, JSON gọn. Mỗi bản có thêm `<tên>.s31.json`: nguồn rank, SHA file điểm, số doc khác Q-rrf, tương quan judge (vì `build.json` của builder S29 luôn ghi tên model 0.6B).

Theo quyết định 21/10 chỉ dựng trên pool 1,3×:

| Bản | Nguồn RRF (k=60) | Cần |
|---|---|---|
| `J4-rrf` | cosine + 4B | `scores-pool13.parquet` của 4B |
| `J4-rrf3` | cosine + 0.6B + 4B | như trên |
| `Jcheck-q06-rrf` | cosine + 0.6B | không cần GPU; cổng hồi quy |

```bash
python -m r2ai.s31.build --judge qwen3-reranker-4b --scores out/runs/s31-strong-judge/gpu/qwen3-reranker-4b/scores-pool13.parquet --variant J4-rrf --variant J4-rrf3
```

Output `out/runs/s31-strong-judge/build/`. J8-* tương tự với `--judge qwen3-reranker-8b`. Không dựng bản pool20.

Chỉ dựng J4-rrf và J4-rrf3 (Vu, 21/10). Không có J4-only-rrf2 riêng: RRF(cosine, 4B) chính là J4-rrf. Không dựng bản chỉ dùng 4B mà không trộn cosine, vì Q-filter (0.6B một mình) đã thua bản trộn.

### Cách đọc kết quả LB

Ba bản cùng pool 1,3×, cùng số doc K90-ph/query, cùng chunk nc140, Pharmacity và RRF k=60; chỉ khác nguồn rank:

| So sánh | Khác nhau | Đọc ra |
|---|---|---|
| J4-rrf vs QRRF-nc140-ph (0,3940) | judge 4B thay cho 0.6B (đều trộn với cosine) | đóng góp của 4B so với 0.6B |
| J4-rrf3 vs J4-rrf | thêm 0.6B làm nguồn thứ ba | đóng góp thêm của 0.6B khi đã có 4B |

`<tên>.s31.json` của mỗi bản ghi số doc khác Q-rrf TB/query; số này đọc kèm chênh LB (chênh nhỏ khi ít doc đổi thì không kết luận được).

### Hồi quy sau merge (20/10, đo)

- **Q-rrf (n_c P1) dựng lại bằng builder S29 đã merge: JSON SHA `99be0252…153a` ≠ `6b21fbb5…9afc` của bản đã nộp.** Nguyên nhân: `e451fe7` đổi `json.dumps` sang `separators=(",", ":")` cho mọi bản, kể cả Q-rrf (JSON 629.390.612 B so với 630.093.327 B). Parse cả hai: 1.200/1.200 dòng **giống hệt** (doc, chunk, thứ tự). Cổng SHA theo đúng chữ không đạt; nội dung khớp. Không sửa code S29.
- **`Jcheck-q06-rrf` qua đường S31: JSON SHA `da6cf34b…3fa7` = đúng `QRRF-nc140-ph` (LB 0,3940)**; 1.200/1.200 query cùng tập doc với Q-rrf. ZIP 97.666.826 B (hơn 2 B do tên member JSON dài hơn 1 ký tự). Build CPU 1.196,6 s.
- Tương quan trên pool13: Spearman(cosine, 0.6B) toàn cục 0,419, TB theo query 0,367 (p10 0,176).
