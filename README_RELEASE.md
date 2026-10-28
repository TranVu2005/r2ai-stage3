# README_RELEASE: dựng hai bản nộp cuối

Nhánh `release/final`, S35, 26/10/2026. Hai bản nộp được dựng lại **từng byte** bằng một lệnh, không cần GPU, không cần mạng.

| bản | vai trò | config | ZIP | SHA256 | LB |
|---|---|---|---|---|---|
| 120A-p4b-v3 | bản chính (khung EF2-cut-p4b, pool cosine top-n_q) | `configs/s36-handoff-120A-p4b-v3.json` | 99.417.831 B | `34d235f6f37704d11b803e83e9eddeabbf4c6100a38497c2728b56cb614937e2` | Vu nộp, chưa có số trong ghi chú này |
| 120A-free-v2 | bản tổng quát (khung free, pool cosine top-300 của corpus) | `configs/s36-handoff-120A-free-v2.json` | 100.285.047 B | `fba176c65f98b733ce879e932ab245905d9899e96eea867583af5ae85caada3f` | Final 0,4101 (Doc 0,4402 / Chunk 0,3800) |

Quy tắc chọn bản sao khi hoà cosine của từng khung: `docs/s36-handoff-s30.md`. Nội dung từng bước và các kết quả LB: `docs/s36-fuse4.md`.

## 1. Môi trường

Đã dựng và kiểm trên: Windows 11 Home 10.0.26200, Python 3.12.13 (`requires-python >=3.11,<3.14`), RAM 15,7 GB. Tiến trình compose quan sát được dùng khoảng 5,4 GB RAM; chạy **một bản một lần** (và không chạy song song với job GPU hay pytest nặng).

| gói | phiên bản ghim (`pyproject.toml`) | ghi chú |
|---|---|---|
| pyarrow | 25.0.1 | quyết định byte của parquet/ZIP, `release` dừng nếu lệch |
| numpy | 2.5.3 | như trên |
| polars | 1.44.2 | như trên |
| pandas | 3.0.6 | đã cài trong môi trường dựng; `pyproject.toml` chưa ghim pandas, nên đây là phiên bản đã kiểm chứng |
| scipy / scikit-learn | 1.18.1 / 1.9.1 | `apply`, hiệu chỉnh |
| transformers / tokenizers | 5.18.0 / 0.23.2 | tokenizer cho đếm token của compose (cache HF, offline) |
| torch | 2.11.0+cu128 | chỉ cần khi chấm lại bằng mô hình (mục 6), không cần để dựng |

Thiết lập (`.env`, không đưa vào git; mẫu `.env.release.example`): `R2AI_DATA_DIR`, `R2AI_OLD_ROOT`, `R2AI_OUT_DIR`, `R2AI_STATE_DIR`, `R2AI_LOG_DIR`, `HF_HOME`, `HF_HUB_OFFLINE=1`. Mọi đường dẫn trong config cố định là tuyệt đối `D:/GitHub/r2ai-stage3/out/runs/...`; nếu dữ liệu nằm chỗ khác phải đổi `R2AI_OUT_DIR` **và** sao chép config với đường dẫn mới (SHA256 các tệp vẫn phải khớp manifest).

Revision mô hình (chỉ để chấm lại; bản dựng không tải mô hình nào): BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181`, Qwen3-Reranker-4B `22e683669bc0f0bd69640a1354a6d0aebcfeede5`, Qwen3-Reranker-0.6B `e61197ed45024b0ed8a2d74b80b4d909f1255473`, Jina Reranker v3.5 `e8a93f33f0b22108f8c2364f8484ce3422552fbc`.

## 2. Artifact bắt buộc

- Manifest: `configs/s36-handoff-s30-manifest.json`. SHA256 `45ce6e493bc275aa71bd6bfda646de5c4ac5314d375866e19a58666db963e1f7` tính trên dạng CRLF của tệp (số đã bàn giao cho S30); git lưu tệp dạng LF, SHA256 của dạng LF là `e5010f99ea036064b89f67516a25944213f581ab9216f96baf4f46e66ff4af79`. `r2ai.release` tính SHA trên dạng CRLF nên khớp ở mọi checkout.
- Danh sách đầy đủ (đường dẫn + SHA256 từng tệp đầu vào, từng đầu ra của `apply`, thư mục compose đọc, mô hình): `docs/release-artifacts.md`. Không xoá và không dựng lại gì trong danh sách đó.
- Gồm 19 tệp nền S36, mỗi bản thêm 2 registry cặp, 4 tệp điểm 4B (Kaggle), 7 tệp C-max, 1 tệp Jina; cộng dữ liệu compose đọc (`data/canonical-20261018/` 17 GB, `data/s37_120ask/`, `out/runs/s32-chunk-judge/`, `out/runs/s27-pharmacity/handoff/`, ...).

## 3. Một lệnh dựng

Chạy từ gốc repo (cần `PYTHONPATH=src` hoặc cài `pip install -e .`):

```bash
PYTHONPATH=src python -m r2ai.release check
PYTHONPATH=src python -m r2ai.release build --config configs/s36-handoff-120A-p4b-v3.json
PYTHONPATH=src python -m r2ai.release build --config configs/s36-handoff-120A-free-v2.json
```

`check` kiểm môi trường và SHA256 mọi đầu vào của cả hai bản, không dựng gì (khoảng 10 giây). `build` ghi vào `out/runs/release/<tên>/` (`apply/`, `compose/`, `release-report.json`, `release.log`); mã thoát 0 chỉ khi mọi so sánh với manifest khớp. Kết quả là ZIP `out/runs/release/<tên>/compose/<tên>-full+w768-t1536.zip`. Không nộp gì lên LB.

Thời gian đo ngày 26/10 trên máy dựng (CPU, một bản một lần):

| bước | việc | 120A-p4b-v3 | 120A-free-v2 |
|---|---|---:|---:|
| env | phiên bản gói, SHA manifest | 0,1 s | 0,2 s |
| inputs | SHA256 các đầu vào của config và dữ liệu nền (33/34 mục) | 4,7 s | 4,6 s |
| apply | `r2ai.s36.s37.apply`: pool, RRF 4 nguồn, cắt EF2, carrier CR-fair; so SHA docsel, carriers, pcal, keep | 14,7 s | 16,7 s |
| prepare | `compose prepare`: bundle gốc từ docsel, chunk carrier, thin | 470,6 s | 505,2 s |
| write | `compose write --threshold 1536`: ZIP, validator S26 | 683,2 s | 682,4 s |
| compare | SHA256, kích thước, cửa sổ, chunk, doc, validator so với manifest | dưới 1 s | dưới 1 s |
| tổng | | 19,6 phút | 20,2 phút |

Kết quả lần dựng này từ `release/final`: ZIP p4b-v3 `34d235f6…37e2` và ZIP free-v2 `fba176c6…da3f`, trùng SHA từng byte với bản đã bàn giao; `scripts/s36/handoff_s30.py --verify --out <thư mục>` cho cả hai config cũng trùng docsel, carriers, pcal, keep.

`T = 1536` và carrier bị bỏ `= 0` lấy từ manifest. Muốn tìm lại T và số carrier từ đầu (dò nhiều vòng build song song, vài giờ): thêm `--search-t` (và `--rounds N`); kết quả phải ra T = 1536 và 0 carrier bị bỏ như ghi trong `docs/s36-fuse4.md`.

Nếu một so sánh không khớp: không sửa config hay manifest cho khớp; xem `release-report.json` (bước nào, tệp nào), rồi kiểm phiên bản pyarrow/numpy/pandas/polars và SHA tệp đầu vào.

## 4. Kiểm bằng script hồi quy cũ

```bash
PYTHONPATH=src python scripts/s36/handoff_s30.py --verify --out out/runs/release/handoff-verify
```

Chạy lại `apply` từ cả hai config cố định và so SHA256 với manifest; `--out` giữ nguyên `configs/` và bàn giao gốc. Không có `--out` script ghi đè `configs/s36-handoff-*.json` và bàn giao gốc (hành vi cũ).

## 5. Kiểm thử

`python -m pytest` chạy `tests/` (`pytest.ini: testpaths = tests`; trước đó `pytest` không đối số nhặt nhầm `scripts/s26/*_test.py`). Với `.env` đúng (mục 1: `HF_HOME` và `HF_HUB_OFFLINE=1` trỏ tới cache HF có tokenizer BGE-M3) kết quả trên `release/final` là **1.047 test, 0 đỏ**. Không có `.env`, 16 test đỏ chỉ vì thiếu tokenizer `BAAI/bge-m3` và dữ liệu mẫu (đo trên `main` cũ khi chưa có `.env`: `test_scorer`, `test_vicrawl_*`, `test_zh_full_ingest`). `tests/test_segments.py::test_real_synthetic_benchmark_writes_and_scans_all_rows` cần ≥ 2 GiB RAM trống, nên đỏ khi compose đang chạy. Chi tiết hợp nhất và từng lần chạy pytest: `docs/release-merge-report.md`.

## 6. Bộ private

**Private 01–04/11 (Vu 28/10): tag `v-private-2026-10-28c`** trên `main` (cấu hình chốt CH-rrf `configs/s41-private-final.json`; chuỗi dự phòng CH-rrf → B0 `--no-cmax` → COS-K150 `--cos-k 150` (`configs/s41-private-fallback-cos-k150.json`); bản 5 phòng hộ EN-hedge `--en-hedge 3`). Chạy `python -m r2ai.newq` theo `docs/private-final.md` (checklist giờ G mục 5, lệnh từng nhánh mục 6.2) từ một worktree tách ở tag đó. Tag cũ `v-private-2026-10-28`, `v-private-2026-10-28b` giữ nguyên; tag đã tạo không sửa. Phần dưới đây là đường S35 cũ (`r2ai.release`).

Dựng lại bằng đúng lệnh ở mục 3 nếu private dùng lại 1.200 query; nếu là bộ query mới thì xem `docs/release-private.md` và chạy `python -m r2ai.release plan --new-queries N` (chạy khô, không GPU).
