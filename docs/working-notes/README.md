# docs/working-notes

Kho số liệu, hình và khung bài cho working notes (Road to AI Stage 3). Chỉ đọc dữ liệu; không chạy GPU, không nộp LB, không sửa mã pipeline.

| tệp | nội dung |
|---|---|
| `outline.md` | khung bài theo mục, mỗi số có nguồn (file:mục) |
| `open-questions.md` | chỗ lệch / thiếu nguồn / cần Vu quyết |
| `data/lb_scores.csv` | mọi điểm LB public trong docs và bảng Vu: ngày, metric (v1 trước 23/10, v2 từ 23/10), Final, Doc F2/P/R, Chunk F2/P/R, doc/query, cột `derived` ghi số nào suy ra, `in_vu_table`, `source` |
| `data/ablation_free.csv` | ablation khung free (một thành phần mỗi dòng) |
| `data/coverage_by_host.csv` | độ phủ text theo (ngôn ngữ, host) và theo nguồn, dựng bằng `scripts/build_coverage.py` |
| `data/fingerprint_auc.csv` | AUC nhận dạng retriever qua pool zh (biến thể, độ dài, extractor) |
| `data/chunk_size_curve.csv` | Chunk F2 (metric v2) theo kích cỡ chunk |
| `data/doc_count_curve.csv` | Doc F2 theo số doc/query |
| `data/chunk_count_curve.csv` | Chunk F2 theo số chunk/query (n_c), precision biên |
| `data/constants.csv` | hằng số pipeline chốt và nguồn hiệu chỉnh |
| `data/noise_decisions.csv` | quyết định có \|Δ Final\| dưới ngưỡng ≈ 0,003 và lý do giữ / bỏ |
| `data/zips_sha256.csv` | S44 28/10: mọi ZIP dưới `out/` lúc kiểm kê (2.067 tệp, 1.136 SHA khác nhau): đường dẫn, byte, mtime, SHA256 |
| `data/archive_zips.csv` | S44: ZIP đề xuất lưu ngoài máy (vai trò, LB, byte, SHA256) |
| `figures/` | `lb_progress`, `fingerprint_auc`, `chunk_f2_by_size`, `doc_f2_by_count`, `ablation_free`, `pipeline` (PNG 200 dpi + PDF) |

Dựng lại:

```bash
python docs/working-notes/scripts/build_coverage.py --root D:/GitHub/r2ai-stage3
python docs/working-notes/scripts/make_figures.py
```

`build_coverage.py` cần pandas + pyarrow và dữ liệu ignored của repo chính (`data/`, `out/runs/s27-pharmacity/handoff`). `make_figures.py` cần matplotlib + pandas (không có trong `.venv` của pipeline; đã chạy bằng Python 3.14, matplotlib 3.10.8, pandas 3.0.3).

Số thập phân trong CSV dùng dấu chấm; trong `.md` dùng dấu phẩy như docs của repo. Commit nguồn: mọi đường dẫn không ghi nhánh là trên `feat/s41` `ba9850c` (gốc của nhánh này).
