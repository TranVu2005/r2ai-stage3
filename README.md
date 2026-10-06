# R2AI Stage 3

Repo độc lập cho crawl/extract y khoa, BGE-M3 hybrid, reranker và scorer. Logic giữ nguyên từ OLD; src layout và đường dẫn tập trung ở `src/r2ai/paths.py`. Không commit corpus/state/output. Xem [CONTEXT](docs/CONTEXT.md), [PROGRESS](docs/PROGRESS.md), [metric](docs/metric.md), [lịch sử submission](submissions/LOG.md).

## Setup native Windows / PowerShell

Môi trường đã kiểm: Python **3.12.13**, uv **0.11.28**, torch **2.11.0+cu128**, CUDA build **12.8**, RTX 3050 6GB. Linux/WSL chưa kiểm. Chạy ở root repo mới:

```powershell
Set-Location D:/GitHub/r2ai-stage3
# Lần đầu, nếu chưa có .env; sửa root/cache cho máy của bạn.
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
./scripts/setup.ps1
if ($LASTEXITCODE -ne 0) { throw 'Setup failed' }
./.venv/Scripts/Activate.ps1
$env:PYTHONDONTWRITEBYTECODE='1'
```

`.env` được nạp tự động trước HF imports: CLI > process env > .env > defaults. Process env đã set phải được bỏ/đổi nếu muốn dùng giá trị .env mới. HF_HOME dùng cache cũ; HF_HUB_OFFLINE=1, không tải model. Torch có extra `torch-cu128`; setup cài riêng với backend cu128, sau đó editable probe/ml/dev + constraints, có dry-run và dừng khi lỗi. Freeze được ghi ở configs/environment.constraints.txt. Model snapshots trong configs/retrieval.yaml là run record; YAML không tự thay CLI/thuật toán.

## Path đọc và ghi

DATA_DIR đọc legacy OLD/data. WORK_DATA_DIR ghi NEW/data; OUT_DIR/RUNS_DIR/STATE_DIR/LOG_DIR đều NEW. R2AI_OLD_ROOT bắt buộc nếu DATA_DIR ngoài repo; mọi writer chính từ chối ghi OLD kể cả CLI override/temp/cleanup. Input thiếu/rỗng phải lỗi trước tạo output; không fallback/trộn bundle.

Đã copy/kiểm vào 2026-10-04 01:26 UTC+7: **18.031 shard (3.427 GiB)** và crawl.db sang NEW, hash/inventory/SQL khớp; .env đã chọn raw NEW. Extract chưa chạy. Trong phiên tạo repo trước copy, state mới rỗng. Trước copy, crawl ghi NEW/data/raw_vi nhưng extract đọc RAW_DIR ở OLD: shard crawl mới không được extract nếu thiếu `--raw-dir` NEW. **Copy raw_vi và crawl.db sang NEW trước extract lớn**; giữ nguyên OLD, không copy extract.db/docs cũ. Copy chỉ khi writer dừng/WAL0, kiểm hash/inventory, không overwrite đích đã có dữ liệu. Xem [quy trình copy](docs/data-transition.md).

Sau copy, .env đổi `R2AI_RAW_DIR=D:/GitHub/r2ai-stage3/data/raw_vi`, RAW_DIR=RAW_WRITE_DIR. Giữ raw NEW ổn định đến sau 11/11/2026; extract.db mới lưu path tuyệt đối này. Crawl resume DB bản sao; extract chạy mới đủ **18.031 shard**, làm lại 3.103 shard (17,2%), còn 14.928 chưa có checkpoint. Chưa đo thời gian extract/embed lớn. Không crawl thật khi chưa được duyệt riêng.

## Status (không ghi DB/log)

```powershell
python scripts/crawl.py status
# Chỉ dùng OLD khi cần đối chiếu snapshot legacy:
python scripts/crawl.py status --state-dir D:/GitHub/r2ai-stage3-old/state
```

Mặc định status đọc STATE_DIR của NEW, kể cả sau copy; LEGACY_STATE_DIR không tự đổi nguồn status. URI read-only; snapshot OLD cần writer dừng/WAL0. Baseline legacy: 653.970 URL, 624.596 ok, 7.176 pending, 47 done và youmed halted.

## Pipeline NEW sau copy raw/state

Các lệnh dưới đây chưa được chạy trên corpus lớn trong migration. Chạy tuần tự, mỗi lệnh fail thì dừng; giữ cùng bundle NEW. Extract không đọc crawl.db. Chunk mặc định docs legacy nên phải truyền docs NEW rõ ràng. Corpus/query vẫn đọc OLD.

```powershell
python scripts/extract.py --raw-dir D:/GitHub/r2ai-stage3/data/raw_vi --docs-dir D:/GitHub/r2ai-stage3/data/docs_vi --state-dir D:/GitHub/r2ai-stage3/state --workers 4
if ($LASTEXITCODE -ne 0) { throw 'Extract failed' }
python -m r2ai.index.chunk --docs-dir D:/GitHub/r2ai-stage3/data/docs_vi --out-dir D:/GitHub/r2ai-stage3/data/chunks --targets 256
if ($LASTEXITCODE -ne 0) { throw 'Chunk failed' }
python scripts/build_index.py build --target 256 --batch-size 16 --shard-size 10000 --flat-max-gb 4   # 3,9M chunk: thêm --no-ann, dùng --candidates exact
if ($LASTEXITCODE -ne 0) { throw 'Index failed' }
python scripts/run_retrieval_k100.py --target 256 --candidates exact --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet --out-dir D:/GitHub/r2ai-stage3/out/runs/vi-k100
if ($LASTEXITCODE -ne 0) { throw 'K100 failed' }
python scripts/make_submission.py --k-doc 100 --k-chunk 19 --chunk-mode full --dedupe-scope doc --runs-dir D:/GitHub/r2ai-stage3/out/runs/vi-k100 --chunks-dir D:/GitHub/r2ai-stage3/data/chunks --docs-dir D:/GitHub/r2ai-stage3/data/docs_vi --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet --out D:/GitHub/r2ai-stage3/out/runs/new-full/submission.zip
if ($LASTEXITCODE -ne 0) { throw 'Submission failed' }
python scripts/validate_submission.py D:/GitHub/r2ai-stage3/out/runs/new-full/submission.zip --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet --corpus D:/GitHub/r2ai-stage3-old/data/raw/links_corpus.parquet --docs-dir D:/GitHub/r2ai-stage3/data/docs_vi
```

`--candidates exact` quét chính xác theo khối `dense.npy` và cache CSR mmap `<index>/sparse_mmap` (tạo từ `sparse.npz`, không sửa file gốc), không cần `faiss.index`; mặc định `faiss` giữ nguyên. Nếu đã có shard mà thiếu dense/sparse/meta: `python scripts/build_index.py assemble --target 256 --no-ann`. Gate tương đương: `python -m r2ai.retrieve.exact_gate --help`.

Rerank sâu (mặc định tắt): `run_retrieval_k100 ... --tier1-docs 100 --tier2-docs 50 --chunk-score-docs 0 --pair-scores --out-dir <thư mục mới>`; chạy lại đúng lệnh để resume (checkpoint atomic, Ctrl+C an toàn, khóa chặn lần chạy thứ hai cùng `--out-dir`). Tier 1 sâu hơn mà giữ điểm lượt cũ bit-identical: thêm `--tier1-base N0` (ví dụ `--tier1-docs 200 --tier1-base 100`; gate `deep_gate deep --tol 0 --pairs`). Gate: `python -m r2ai.retrieve.deep_gate --help`. Trong PowerShell gọi `python` trực tiếp hoặc Git Bash (`& "C:\Program Files\Git\bin\bash.exe"`); `bash` của WSL không thấy `/d/...`. Builder luôn truyền `--docs-dir data/docs_vi --chunks-dir data/chunks` (mặc định docs-dir có thể trỏ legacy).

## Tái tạo bản tốt nhất (RD150 = D50 + expand + RRF chunk + RRF tập doc)

Bản tốt nhất (Final 0,2139, 06/10) = RD150: D50 + expand + chọn doc nhận chunk theo RRF + chọn tập doc theo RRF, mô tả trong `configs/submission-best.yaml`. RRF (Final 0,2125) giữ ở `configs/submission-rrf.yaml`, H1-expand (Final 0,2109) ở `configs/submission-h1expand.yaml`, D50 thuần (Final 0,2107) ở `configs/submission-d50.yaml`. Expand nối vào cuối `relevant_docs` các id (cả `doc_ids_group`) của doc cùng cụm `content` với 150 doc chính. RRF (`r2ai.submit.rrf_chunks`) giữ nguyên `relevant_docs`, chỉ đổi `relevant_chunks`: 50 doc nhận chunk full = 50 doc trong 150 doc chính có 1/(60 + hạng reranker) + 1/(60 + hạng hybrid) cao nhất (hoà: hạng reranker rồi hạng hybrid thấp hơn), theo thứ tự đó. Hạng reranker lấy từ cache rerank tier 1 = 200 `out/runs/rerank200/full/vi_k100.parquet` (lượt GPU 05/10, `run_best_retrieval.py` không tạo ra file này), hạng hybrid từ `submission.doc_ranking`. Tập doc theo RRF (`r2ai.submit.rrf_docs`, `postprocess.rrf_doc_set`) chỉ đổi `relevant_docs`: `doc_ids_group` của 150 doc có RRF (k 60) cao nhất trong toàn bộ 200 doc của cache rerank đó (hạng hybrid cũng từ `submission.doc_ranking`), theo thứ tự RRF, rồi expand cụm như trên; `relevant_chunks` giữ nguyên byte. Bật/tắt bằng `postprocess.expand_clusters.enabled`, `postprocess.rrf_chunk_docs.enabled` và `postprocess.rrf_doc_set.enabled` (cần expand); thiếu key = tắt. Mặc định trong code không đổi. Chạy trong PowerShell ở root repo, không cần bash/WSL:

```powershell
# 0. Cụm doc trùng (CPU, đo 05/10: features 329,5 s + cluster 43,6 s, RSS đỉnh 1,37 GiB) -> out/runs/H1/clusters.parquet.
#    Bắt buộc chạy lại khi corpus đổi (thêm zh, extract/chunk lại): cụm cũ không phủ doc mới và doc_id có thể đổi.
python -m r2ai.dupes.scan features --docs-dir data/docs_vi --out-dir out/runs/H1 --workers 3
python -m r2ai.dupes.scan cluster  --docs-dir data/docs_vi --out-dir out/runs/H1 --workers 3
python -m r2ai.dupes.report --out-dir out/runs/H1
# 1. Truy hồi (GPU, khoảng 1,2 h đo 05/10): candidates-only -> out/runs/best/cand, rerank sâu tier 1 = 100 doc -> out/runs/best/rerank.
#    Bị ngắt (Ctrl+C, sleep, crash) thì chạy lại đúng lệnh này để resume từ checkpoint; một --out-dir chỉ cho 1 tiến trình.
python scripts/run_best_retrieval.py
if ($LASTEXITCODE -ne 0) { throw 'Retrieval failed' }
# 2. Build (CPU): make_submission (K doc 150, chunk full top 50) -> <out>_base.zip, expand (r2ai.dupes) -> <out>_expand.zip,
#    chọn lại doc nhận chunk theo RRF (r2ai.submit.rrf_chunks) -> <out>_rrf.zip, rồi chọn tập doc theo RRF (r2ai.submit.rrf_docs)
#    -> <out>; so SHA256 JSON (builder, sau expand, sau RRF, cuối) với bản đã nộp, ZIP <= 104.857.600 byte.
python scripts/build_best_submission.py --from-retrieval --out out/runs/best/submission/sub_best_rd150.zip --no-check
python scripts/validate_submission.py out/runs/best/submission/sub_best_rd150.zip --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet --corpus D:/GitHub/r2ai-stage3-old/data/raw/links_corpus.parquet --docs-dir data/docs_vi --max-zip-mib 100
```

Replay từ cache (không GPU, JSON phải giống byte bản đã nộp, lệch thì exit 1): `python scripts/build_best_submission.py --out out/runs/<thư mục mới>/sub_best_rd150.zip` (best = RD150 đã nộp, JSON `136c2591…e950`), hoặc thêm `--config configs/submission-rrf.yaml` (RRF, `ff035b8b…d1c3`) / `--config configs/submission-h1expand.yaml` (H1-expand, `6c27d3f3…1106`) / `--config configs/submission-d50.yaml` (D50 thuần, `2f10b78a…6544`). Đo 06/10 (cả bốn khớp SHA): best wall 344 s (builder 196,5 s, expand 38,0 s, RRF chunk 72,7 s, RRF tập doc 34,3 s), validator 106 s; rrf 381 s; h1expand 198 s; d50 151 s. Khi build với `--from-retrieval` sau một lượt truy hồi mới, hoặc sau khi dựng lại cụm, `--no-check` bỏ yêu cầu SHA phải khớp; kết quả SHA vẫn được in ra.

Retrieval dev (có guard, cần index/chunks NEW và dev legacy): `python scripts/retrieve.py dev --target 256 --dev D:/GitHub/r2ai-stage3-old/data/dev/pseudo_vi_v2.parquet`. Output OUT_DIR/retrieval. Index build giữ nguyên nhánh FAISS theo flat-max-gb, không ép thuật toán corpus mở rộng.

Crawl thật sau duyệt riêng: `python scripts/crawl.py run --raw-dir D:/GitHub/r2ai-stage3/data/raw_vi --state-dir D:/GitHub/r2ai-stage3/state --no-report`. Youmed halted, pending rank13–7188; --limit 5 là rank<5, không phải 5 pending tiếp theo. Không tự reset-errors hay chạy smoke thật.

## Replay sub06 legacy (đã kiểm)

```powershell
python scripts/make_submission.py --k-doc 100 --k-chunk 19 --chunk-mode full --dedupe-scope doc --runs-dir D:/GitHub/r2ai-stage3-old/data/runs --chunks-dir D:/GitHub/r2ai-stage3-old/data/chunks --docs-dir D:/GitHub/r2ai-stage3-old/data/docs_vi --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet --out D:/GitHub/r2ai-stage3/out/runs/reproduce-sub06/sub06_vi_kd100_kc19_full.zip
if ($LASTEXITCODE -ne 0) { throw 'Replay failed' }
python scripts/validate_submission.py D:/GitHub/r2ai-stage3/out/runs/reproduce-sub06/sub06_vi_kd100_kc19_full.zip --queries D:/GitHub/r2ai-stage3-old/data/raw/query.parquet --corpus D:/GitHub/r2ai-stage3-old/data/raw/links_corpus.parquet --docs-dir D:/GitHub/r2ai-stage3-old/data/docs_vi --max-zip-mib 45.7
```

JSON byte diff = 0; replay sau review 78.8s; SHA256 `07e0fd59f8cc4b8abbf129dc1536305bdbef17ce510b6fe39fe50fe62133acc3`, 181.960.105 byte; 1.200 query / 22.800 chunk; validator 0 lỗi. ZIP bytes có thể khác do metadata/nền tảng nén. Run config/metrics nằm cạnh artifact; không rerun embedding/reranker hoặc nộp leaderboard. Final 0,0626 là điểm cũ theo PROGRESS, chưa có scoreboard gốc.

## Score và test

```powershell
python scripts/score.py --pred out/runs/new-full/submission.zip --gold C:/path/to/gold.json
python -B -m pytest tests/test_scorer.py tests/test_make_submission.py tests/test_chunk.py -q -p no:cacheprovider
python -B -m pytest -q -p no:cacheprovider
python -B -m pytest tests/test_vicrawl_integration.py -m slow -q -p no:cacheprovider
```

Core 36 test giữ nguyên assertions. Suite đã đo sau review: **272 pass**, hai test slow **2 pass** riêng. Suite mặc định loại 2 test slow nên chạy dòng cuối riêng. Validator ZIP và JSON trực tiếp đã kiểm trên Python3.12, giữ kiểm tra CRLF. Bare clone cần legacy `.env`, HF tokenizer cache và OLD/out/{crawl_sample.csv,sample_raw} để chạy parity/extractor tests; không bỏ assertion/skip hoặc commit cả out. Test mockserver chỉ dùng localhost/temp.

## Công cụ phụ và lịch sử

src/vicrawl giữ package/API crawler; src/r2ai/{crawl,extract,index,retrieve,eval,submit,probe,gold_check}. Wrapper trong scripts nạp paths trước ML/HF.

Các writer phụ measure_candidates, pseudo-dev, profile/downloader, probe/gold-check CLI, fixture refresh, submission/build tạm khóa trong main/__main__ đến khi hoàn tất guard. Thư viện generate_report/ResultsStore/Checkpoint/Service vẫn import được. Các tài liệu probe/crawl/gold-check và docs/archive mô tả quy trình legacy; lệnh hiện hành là README này. Không xóa code R2AI phụ, không mang history helicopter.

OLD `D:/GitHub/r2ai-stage3-old` là backup. **Không git clean -fdx trong repo này, ở OLD hay Git cha D:/GitHub**: data/state/out của repo là file ignored, OLD còn khoảng 15,366 GiB chưa tracked; lệnh clean sẽ xóa mất. Public 31/10/2026; private 04/11/2026 tối đa 5 lượt; kết quả 11/11/2026.
