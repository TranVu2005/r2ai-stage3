# R2AI Stage 3

Repo độc lập cho crawl/extract y khoa, BGE-M3 hybrid, reranker và scorer. Logic giữ nguyên từ OLD; src layout và đường dẫn tập trung ở `src/r2ai/paths.py`. Không commit corpus/state/output. Xem [CONTEXT](docs/CONTEXT.md), [PROGRESS](docs/PROGRESS.md), [metric](docs/metric.md), [lịch sử submission](submissions/LOG.md).

## Setup native Windows / PowerShell

Môi trường đã kiểm: Python **3.12.13**, uv **0.11.28**, torch **2.11.0+cu128**, CUDA build **12.8**, RTX 3050 6GB. Linux/WSL chưa kiểm. Chạy ở root repo mới:

```powershell
Set-Location D:/R2AI/r2ai-stage3
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

Sau copy, .env đổi `R2AI_RAW_DIR=D:/R2AI/r2ai-stage3/data/raw_vi`, RAW_DIR=RAW_WRITE_DIR. Giữ raw NEW ổn định đến sau 11/11/2026; extract.db mới lưu path tuyệt đối này. Crawl resume DB bản sao; extract chạy mới đủ **18.031 shard**, làm lại 3.103 shard (17,2%), còn 14.928 chưa có checkpoint. Chưa đo thời gian extract/embed lớn. Không crawl thật khi chưa được duyệt riêng.

## Status (không ghi DB/log)

```powershell
python scripts/crawl.py status
# Chỉ dùng OLD khi cần đối chiếu snapshot legacy:
python scripts/crawl.py status --state-dir D:/GitHub/r2ai-stage3/state
```

Mặc định status đọc STATE_DIR của NEW, kể cả sau copy; LEGACY_STATE_DIR không tự đổi nguồn status. URI read-only; snapshot OLD cần writer dừng/WAL0. Baseline legacy: 653.970 URL, 624.596 ok, 7.176 pending, 47 done và youmed halted.

## Pipeline NEW sau copy raw/state

Các lệnh dưới đây chưa được chạy trên corpus lớn trong migration. Chạy tuần tự, mỗi lệnh fail thì dừng; giữ cùng bundle NEW. Extract không đọc crawl.db. Chunk mặc định docs legacy nên phải truyền docs NEW rõ ràng. Corpus/query vẫn đọc OLD.

```powershell
python scripts/extract.py --raw-dir D:/R2AI/r2ai-stage3/data/raw_vi --docs-dir D:/R2AI/r2ai-stage3/data/docs_vi --state-dir D:/R2AI/r2ai-stage3/state --workers 4
if ($LASTEXITCODE -ne 0) { throw 'Extract failed' }
python -m r2ai.index.chunk --docs-dir D:/R2AI/r2ai-stage3/data/docs_vi --out-dir D:/R2AI/r2ai-stage3/data/chunks --targets 256
if ($LASTEXITCODE -ne 0) { throw 'Chunk failed' }
python scripts/build_index.py build --target 256 --batch-size 16 --shard-size 10000 --flat-max-gb 4
if ($LASTEXITCODE -ne 0) { throw 'Index failed' }
python scripts/run_retrieval_k100.py --target 256 --queries D:/GitHub/r2ai-stage3/data/raw/query.parquet --out-dir D:/R2AI/r2ai-stage3/out/runs/vi-k100
if ($LASTEXITCODE -ne 0) { throw 'K100 failed' }
python scripts/make_submission.py --k-doc 100 --k-chunk 19 --chunk-mode full --dedupe-scope doc --runs-dir D:/R2AI/r2ai-stage3/out/runs/vi-k100 --chunks-dir D:/R2AI/r2ai-stage3/data/chunks --docs-dir D:/R2AI/r2ai-stage3/data/docs_vi --queries D:/GitHub/r2ai-stage3/data/raw/query.parquet --out D:/R2AI/r2ai-stage3/out/runs/new-full/submission.zip
if ($LASTEXITCODE -ne 0) { throw 'Submission failed' }
python scripts/validate_submission.py D:/R2AI/r2ai-stage3/out/runs/new-full/submission.zip --queries D:/GitHub/r2ai-stage3/data/raw/query.parquet --corpus D:/GitHub/r2ai-stage3/data/raw/links_corpus.parquet --docs-dir D:/R2AI/r2ai-stage3/data/docs_vi
```

Retrieval dev (có guard, cần index/chunks NEW và dev legacy): `python scripts/retrieve.py dev --target 256 --dev D:/GitHub/r2ai-stage3/data/dev/pseudo_vi_v2.parquet`. Output OUT_DIR/retrieval. Index build giữ nguyên nhánh FAISS theo flat-max-gb, không ép thuật toán corpus mở rộng.

Crawl thật sau duyệt riêng: `python scripts/crawl.py run --raw-dir D:/R2AI/r2ai-stage3/data/raw_vi --state-dir D:/R2AI/r2ai-stage3/state --no-report`. Youmed halted, pending rank13–7188; --limit 5 là rank<5, không phải 5 pending tiếp theo. Không tự reset-errors hay chạy smoke thật.

## Replay sub06 legacy (đã kiểm)

```powershell
python scripts/make_submission.py --k-doc 100 --k-chunk 19 --chunk-mode full --dedupe-scope doc --runs-dir D:/GitHub/r2ai-stage3/data/runs --chunks-dir D:/GitHub/r2ai-stage3/data/chunks --docs-dir D:/GitHub/r2ai-stage3/data/docs_vi --queries D:/GitHub/r2ai-stage3/data/raw/query.parquet --out D:/R2AI/r2ai-stage3/out/runs/reproduce-sub06/sub06_vi_kd100_kc19_full.zip
if ($LASTEXITCODE -ne 0) { throw 'Replay failed' }
python scripts/validate_submission.py D:/R2AI/r2ai-stage3/out/runs/reproduce-sub06/sub06_vi_kd100_kc19_full.zip --queries D:/GitHub/r2ai-stage3/data/raw/query.parquet --corpus D:/GitHub/r2ai-stage3/data/raw/links_corpus.parquet --docs-dir D:/GitHub/r2ai-stage3/data/docs_vi --max-zip-mib 45.7
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

OLD `D:/GitHub/r2ai-stage3` là backup. **Không git clean -fdx ở D:/GitHub/OLD**: khoảng 15,366 GiB data/out chưa tracked có thể mất. Public 31/10/2026; private 04/11/2026 tối đa 5 lượt; kết quả 11/11/2026.
