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
| `K4-scale` / `K4-thr` / `K4-mix` | cosine + 4B, K_q' biến thiên, tổng như J4-rrf | điểm 4B (mục LB 21/10) |
| `K4-tfree-t<t>` / `K4-thr-C` / `K4-free-p13` | cosine + 4B; t cố định / chunk theo ngưỡng / hằng số không K_q | điểm 4B (mục LB 21/10 lượt 2) |

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

## Kết quả 21/10/2026 (đo, chưa nộp)

### Điểm 4B pool13 (Kaggle)

`gpu/qwen3-reranker-4b/scores-pool13.parquet`: SHA `39d554f2…84fb` (khớp số Vu gửi), 281.924 cặp, 150/150 part, revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, fp16, hf, 0 doc vượt 512 token. **GPU: 98.273,2 s T4 (27,3 giờ T4) ⇒ 2,87 cặp/s mỗi T4**, thấp hơn mức ước tính ~5 cặp/s. Chạy 2×T4 song song thì khoảng 13,6 giờ thật.

### Pharmacity lô 3 (612 → 979 doc)

`python -m r2ai.s31.pharmacity_ext prepare` tính cosine cho 367 doc lô 3, cùng cách tính và cùng vector query với S29, rồi trộn vào pool 1,3× của từng query. Số cặp mỗi query giữ nguyên `round(1,3 × K_q)`, doc mới lọt vào thì cặp cuối bị đẩy ra.

- **103 doc lô 3 lọt pool13: 170 cặp mới, 134 query thay đổi, 170 cặp bị đẩy ra.** Tổng vẫn 281.924 cặp. Index SHA `ecf0ea1b…e3f`.
- Chấm local trên RTX 3050: 0.6B bf16 170 cặp trong 14,2 s; 4B fp16 với `--device-map auto --max-gpu-memory 4GiB` (offload, cùng snapshot `22e68366…`) 170 cặp trong 101,6 s.
- **Kiểm 4B local so với Kaggle trên 200 cặp pool13: Spearman 0,99998**, |Δ| TB 0,00085, max 0,0054.
- Ghép điểm (`merge-scores`): 0.6B `q06_scores.parquet` SHA `3b04a00f…118d`, 4B SHA `ddf24b91…db0f`, mỗi file 281.924 cặp.

### Bản dựng

Mọi bản: 1.200 query, 195.147 doc, n_c ×1,4, nguyên body (không cắt), LZMA dict 256 MiB. Validator S26 (`scripts/s26/validate_pa3.py` + lô Pharmacity) báo `ok: true` cho cả 4 ZIP.

| Bản | Pharmacity | ZIP B | ZIP SHA256 | Chunk | vi / zh doc | Doc thay so với QRRF-nc140-ph TB/query (sym-diff) | Doc lô 3 được chọn |
|---|---|---:|---|---:|---|---:|---:|
| QRRF-nc140-ph (LB 0,3940) | 612 | 97.666.824 | `c57f3e99…59d0` | 175.313 | 151.036 / 44.111 | 0 | 0 |
| J4-rrf | 612 | 97.996.591 | `80494fb6f5a432fb3b76b7ed61edbffb2282096390df9ac243f771f072e98606` | 175.270 | 152.965 / 42.182 | 15,71 (31,41) | 0 |
| J4-rrf3 | 612 | 98.029.728 | `72f1bdc0a6304a2e6de7ed2422ec4045a4f09e01a37f991d0517f7182a78eab1` | 175.249 | 151.606 / 43.541 | 10,86 (21,72) | 0 |
| **J4-rrf-ph979** | 979 | 97.920.078 | `585685f18cde4493372a58efa4fb01846c205e73ec7f5252f86fd825af39e5d4` | 175.148 | 152.994 / 42.153 | 15,76 (31,53) | 91 doc / 136 cặp |
| **J4-rrf3-ph979** | 979 | 97.925.448 | `8f34a80322a7240729debe3e2260662a8936201bff3596d908c12e3c78623a0d` | 175.123 | 151.639 / 43.508 | 10,94 (21,88) | 89 doc / 135 cặp |

Cổng hồi quy chạy lại cùng lượt build (lô 3 đã có trong thư mục lots): `Jcheck-q06-rrf` JSON vẫn bằng `da6cf34b…3fa7`. Như vậy builder không bị ảnh hưởng khi thư mục lots có thêm lô 3.

Tương quan thứ hạng trên pool13 (bản 612; bản 979 lệch < 0,0002):

| Cặp | Spearman toàn cục | TB theo query | p10 theo query |
|---|---:|---:|---:|
| 4B ~ 0.6B | 0,7369 | 0,6869 | 0,4933 |
| 4B ~ cosine | 0,4416 | 0,3636 | 0,1791 |
| 0.6B ~ cosine | 0,4192 | 0,3669 | 0,1757 |

Đọc LB: J4-rrf so với QRRF-nc140-ph cho biết đóng góp của 4B so với 0.6B (cùng 612 doc Pharmacity); J4-rrf3 so với J4-rrf cho biết đóng góp thêm của 0.6B. Cặp `-ph979` so với cặp 612 tương ứng cho biết tác động của lô 3 (136 / 135 cặp doc lô 3 được chọn, ≈ 0,11 doc/query; chênh LB nhỏ ở mức này khó tách khỏi nhiễu).

## LB 21/10 và bản K_q' biến thiên, thin chunks (đo trên CPU, chưa nộp)

LB do Vu báo: **J4-rrf-ph979 0,3997** (Doc 0,3989 / Chunk 0,4005), là nền mới; J4-rrf3-ph979 0,3994. Vì vậy bỏ 0.6B và không chạy 8B.

### K4-scale / K4-thr / K4-mix (`--variant K4-scale --variant K4-thr --variant K4-mix`)

Các bản dùng chung: pool 1,3× đã gộp Pharmacity 979, điểm 4B, chunk nc140, nguyên body. Tổng doc bằng J4-rrf-ph979 (195.147). Mỗi query lấy top-K_q' theo RRF(cosine, 4B), với K_q = `len(relevant_docs)` của K90-ph (Vu chọn). Sàn và trần là `ceil(lo·K_q)` .. `floor(hi·K_q)`.

- **scale**: K_q × r_q / TB(r), với r_q = P4B trung bình của top-K_q theo RRF (Vu chọn); kẹp [0,6; 1,3]·K_q.
- **thr**: K_q' = kẹp(số doc trong pool có P4B ≥ t) vào [0,5; 1,3]·K_q, sau đó lấy top-K_q' theo RRF (Vu chọn). t được chỉnh để tổng ra 195.147: t = 0,07752. Có 247 cặp đúng bằng t, 130 trong số đó nằm trên vết cắt (hoà được tách theo thứ tự query, rồi rank 4B).
- **mix**: sqrt(K_q × n_q(t)), dùng cùng t với thr; kẹp [0,5; 1,3]·K_q.
- scale và mix đạt đúng tổng nhờ nhân một hệ số c rồi mới kẹp (Vu chọn): c = 1,00052 cho scale, 1,01291 cho mix. Phần dư sau làm tròn chia cho các query có phần lẻ lớn nhất.

| Bản | K_q'/K_q P10 / median / P90 | Chạm sàn | Chạm trần | Doc khác J4-rrf-ph979 TB/query: thêm / bớt (sym-diff) | Chunk | ZIP B | ZIP SHA256 |
|---|---|---:|---:|---:|---:|---:|---|
| J4-rrf-ph979 (LB 0,3997) | 1 / 1 / 1 | – | – | 0 | 175.148 | 97.920.078 | `585685f1…e5d4` |
| K4-scale | 0,603 / 0,985 / 1,298 | 21,6% | 28,1% | 20,42 / 20,42 (40,84) | 159.318 | 90.063.032 | `2cf1635c5cccc5dace81d36a6d93aa3bb9bcf881c5a5362f5fe6054f473f67e3` |
| K4-thr | 0,505 / 1,021 / 1,298 | 12,2% | 21,2% | 19,58 / 19,58 (39,15) | 159.744 | 90.228.065 | `9aa9342574418c8614a595fb6a69fb84353115862f528bf002aa36dd24cb307c` |
| K4-mix | 0,691 / 1,024 / 1,191 | 2,6% | 0% | 11,96 / 11,96 (23,91) | 167.412 | 94.140.782 | `839851e4b9a000b5af59eab6b6ea99675c9fc83cff4b3669aad60e4c4e78ed2e` |

Vì tổng doc bằng nhau, số doc thêm TB/query luôn bằng số doc bớt TB/query.

Chunk giảm vì số chunk mỗi query là `min(round(1,4·n_c), K_q')`. Query bị giảm K_q' thì mất chunk, còn query được tăng K_q' không có thêm chunk, do trần chunk tính theo n_c của P1. Như vậy chênh LB của các bản K4 gồm cả tác động lên Doc lẫn tác động lên Chunk.

Validator S26 báo `ok` cho cả 3 bản. Output nằm ở `out/runs/s31-strong-judge/build-k4/`.

### `--thin-chunks` và J4-rrf-ph979-thin

Cờ `--thin-chunks` mặc định tắt. Khi bật: doc `quality == thin` có ≥ 20 ký tự text (sau strip) và chưa có chunk sẽ được làm carrier, chunk là nguyên `doc_text` (verbatim, không cắt kể cả khi có cắt body). Các chunk này tính vào trần n_c ×1,4 như doc thường. Corpus có 7.105 doc thin, trong đó 6.490 đủ điều kiện.

| Bản | Chunk thin thêm | Query có chunk thin | Chunk tổng | ZIP B | ZIP SHA256 |
|---|---:|---:|---:|---:|---|
| J4-rrf-ph979 | 0 | 0 | 175.148 | 97.920.078 | `585685f1…e5d4` |
| J4-rrf-ph979-thin | 1.139 | 377 (TB 3,0, max 35) | 175.328 | 97.561.097 | `21797e25205aca5b3cadb1f455262a0bfafe981c1e298d1d18c146ee6cfc3ba2` |

Do có trần, 1.139 chunk thin đẩy ra 959 chunk body/P1 xếp sau. Kết quả: tổng chunk chỉ +180, số chunk thay đổi ở 53 query, ZIP nhỏ đi 358.981 B vì text thin ngắn hơn body bị đẩy ra. Doc giữ nguyên. Validator S26 báo `ok`.

**Hồi quy** (`out/runs/s31-strong-judge/build-thin-regress/`): dựng lại J4-rrf-ph979 với cờ tắt, sau khi đã có thay đổi builder (`fixed_k`, `thin_texts`). JSON `965331b3…3950` và ZIP `585685f1…e5d4` khớp đúng bản đã nộp.

## LB 21/10 (lượt 2), nền K4-thr: t tự do, chunk theo ngưỡng, K4-free, pool-extra (CPU, chưa nộp)

LB do Vu báo: **K4-thr 0,4122** (Doc 0,4139 / Chunk 0,4104) là nền mới. K4-mix 0,4087, K4-scale 0,4083. J4-rrf-ph979-thin 0,3994 nên bỏ cờ thin (code vẫn giữ, mặc định tắt).

Mọi bản dưới đây dùng pool 1,3× đã gộp Pharmacity 979, điểm 4B, thứ tự RRF(cosine, 4B). Validator S26 báo `ok` cho cả 5 bản. Output nằm ở `out/runs/s31-strong-judge/build-k4b/` và `build-free/`.

### K4-tfree-t{0,06; 0,08; 0,10}: t cố định, không ràng buộc tổng doc

K_q' = kẹp(số doc trong pool có P4B ≥ t) vào [⌈0,5·K_q⌉, ⌊1,3·K_q⌋]. Chunk vẫn theo n_c ×1,4.

| Bản | Tổng doc | Doc/query P10 / P50 / P90 | Chạm sàn | Chạm trần | So với K4-thr: thêm / bớt TB/query | Chunk (P10 / P50 / P90 mỗi query) | ZIP B | ZIP SHA256 |
|---|---:|---|---:|---:|---:|---|---:|---|
| K4-thr (t = 0,07752, LB 0,4122) | 195.147 | 66 / 160,5 / 261 | 12,2% | 21,2% | 0 / 0 | 159.744 (64 / 138 / 189) | 90.228.065 | `9aa93425…307c` |
| K4-tfree-t0.06 | 201.620 | 69 / 168 / 264 | 9,6% | 24,3% | 5,39 / 0 | 162.168 (68 / 141 / 189) | 91.466.498 | `b68c408c2964948196a533fa9d3144f93c1b2aaca991a4f9145b3cf5c5c77d89` |
| K4-tfree-t0.08 | 194.296 | 66 / 159 / 261 | 12,4% | 20,8% | 0 / 0,71 | 159.406 (64 / 137 / 189) | 90.053.028 | `aa4217d1829af943029322786d980f8c33dfb6c8787021f5bddc5e43e0fd7fab` |
| K4-tfree-t0.1 | 188.312 | 64 / 151 / 257 | 14,2% | 18,1% | 0 / 5,70 | 156.868 (61 / 134 / 189) | 88.695.686 | `46538130a9ac50b22e34d19e6d3a0351f8c2f5c209356bf95e7a3ac994fdafbd` |

### K4-thr-C: doc giữ nguyên K4-thr, chunk theo ngưỡng

Duyệt doc đã chọn theo thứ tự RRF; mỗi doc có P4B ≥ t_c nhận đúng 1 chunk (nguyên body). t_c được chỉnh để tổng chunk bằng 175.148 (số chunk của J4-rrf-ph979): **t_c = 0,070560**. Có 123 cặp đúng bằng t_c, 60 trong số đó được lấy (hoà tách theo query_id, rồi vị trí RRF). 353 doc không có text nên không nhận chunk.

- Danh sách doc (kể cả thứ tự) giống hệt K4-thr; đã kiểm bằng cách so JSON của hai ZIP.
- Chunk/query P10 / P50 / P90 = 47 / 140 / 251 (min 4, max 305). Không query nào thiếu chunk.
- **ZIP 100.471.357 B, dưới trần 102.600.000 B, không phải cắt body (N = 0).** SHA256 `31980453608212846a916b84e306482bc57e3de2f7c040304b8a38a204815a87`.

### K4-free-p13 (bản tạm, vẫn phụ thuộc K_q qua pool13)

Hằng số trong `r2ai.s31.build`:
- `P_FREE = 211` = round(TB 1,3 × len(K90-ph)).
- `F_FREE = 86` = trung vị của ⌈0,5 × len(K90-ph)⌉.
- `T_FREE = 0,07751765647688268` = t của K4-thr.

Pool là cosine top-min(211, pool13). 398 query có pool13 < 211, nên bản này vẫn phụ thuộc K_q; bản đủ 211 cần điểm pool-extra. Chọn doc có P4B ≥ T_FREE theo thứ tự RRF, sàn 86, trần bằng pool. Chunk: doc có P4B ≥ t_c, với t_c chỉnh để tổng chunk = 159.744 (tổng chunk K4-thr, Vu chọn).

Ban đầu đích là 175.148, nhưng không đạt được: chỉ có 170.548 doc có text.

- Doc: tổng 172.131. Doc/query P10 / P50 / P90 = 86 / 145 / 205. Chạm sàn 22,0% query, chạm trần (bằng pool) 2,4%.
- So với K4-thr TB/query: thêm 4,63, bớt 23,81 (sym-diff 28,44). 175 query giống hệt.
- Chunk: **t_c (lần 1, bản p13) = 0,027796**. Tổng 159.744, chunk/query P10 / P50 / P90 = 65 / 135 / 203. Không cắt body.
- ZIP 91.749.446 B, SHA256 `cfd278a34d8381454b4dd9b1cd565beb737f4c9f4c08efaecec54d5c613a4731`.

Khi có điểm pool-extra, K4-free đủ 211 sẽ hiệu chỉnh t_c lần 2, cũng về 159.744 (`--calibrate-tc`), rồi ghi thành hằng số `K4_FREE_T_C`. Sau đó không chỉnh nữa.

### pool-extra: các cặp 4B còn thiếu, gộp thành một subset Kaggle (`python -m r2ai.s31.pool_extra`)

Nguồn cặp, loại trùng theo (query_id, doc_id):

- **free**: cosine top-211 mới của mọi query, tính trên canonical + đủ 3 lô Pharmacity, trừ các cặp đã có điểm 4B. Cặp đã có điểm gồm pool13 mở rộng và 170 cặp pool13 cũ bị đẩy ra (đã chấm trên Kaggle).
- **capped**: 255 query chạm trần ở K4-thr. Lấy đuôi 1,3×–2,0× của pool S31, cộng các doc mới (lô 3, hoà cosine) lọt vào top-round(2,0 × len(P1)) mới.

| | Cặp |
|---|---:|
| **Tổng (sau loại trùng)** | **56.115** |
| free (top-211) | 20.299 |
| capped (đuôi query chạm trần) | 37.106 |
| thuộc cả hai | 1.290 |
| đã có trong shard S31 (giữ pair_id) | 53.395 |
| mới (export, shard `pool-91000.parquet`, pair_id từ 20.000.000) | 2.720 (2.540 doc, trong đó 30 cặp / 23 doc Pharmacity) |

- Query: 398 query có pool13 < 211; 255 query chạm trần; 27 query thuộc cả hai.
- Top-n13 cosine mới chỉ khác pool13 mở rộng ở 10 query / 211 doc. Đây đúng là các doc hoà cosine tại biên S29 đã ghi ở Pha 1.
- `pool211_index.parquet` có 253.200 cặp (1.200 × 211), nguồn điểm của từng cặp: pool13 232.858, evicted 43, extra 20.299.
- **ETA trên Kaggle 2×T4: 56.115 / 5,8 cặp/s ≈ 2,7 giờ**, gọn trong một phiên.
- Revision 4B giữ `22e683669bc0f0bd69640a1354a6d0aebcfeede5`.

Chạy trên Kaggle:
1. Tạo dataset mới "pool-extra" từ thư mục `out/runs/s31-strong-judge/pool-extra/kaggle/`, gồm `extra_pairs.parquet` (SHA `389f6d39…4c04`), `shards/pool-91000.parquet` (`27b7fb6c…8153`) và `judge_score.py` mới (`--subset extra`, `--extra-pool`).
2. Notebook `notebooks/s31-kaggle-judge.ipynb` (mặc định đã là `SUBSET = "extra"`, `SESSION = 2`, `REVISION = 22e68366…`): Add Input gồm dataset pool cũ và pool-extra. Không thêm output pool13 cũ. Chạy Save & Run All.
3. Tải về `/kaggle/working/s31/qwen3-reranker-4b/scores-extra.parquet` và `collect-extra-report.json`.

Sau khi có `scores-extra.parquet` sẽ dựng hai bản: **K4-free** (đủ top-211, không còn phụ thuộc K_q) và **K4-thr-ext** (K4-thr, trần mở rộng cho 255 query chạm trần).
