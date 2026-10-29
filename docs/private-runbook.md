# Private runbook (1 trang, S44 28/10/2026)

Tóm tắt thao tác của `docs/private-final.md` (mục 5, 6.2 là nguồn chi tiết). Mã: **tag `v-private-2026-10-28c`** (object `114854f6…`, commit `7e8894d3…`). Không sửa tag, không thử nghiệm thêm. IMP30 (`--impute-4b`): **không dùng**.

## 0. Trước T0 (31/10) — checklist SHA

```bash
git -C D:/GitHub/r2ai-stage3 worktree add --detach .superpowers/worktrees/private-28c v-private-2026-10-28c
cp D:/GitHub/r2ai-stage3/.env D:/GitHub/r2ai-stage3/.superpowers/worktrees/private-28c/.env
cd D:/GitHub/r2ai-stage3/.superpowers/worktrees/private-28c && export PYTHONPATH=$PWD/src PYTHONIOENCODING=utf-8
PY=D:/GitHub/r2ai-stage3/.venv/Scripts/python.exe
git rev-parse HEAD                      # 7e8894d34184716b1a668bb5c8f2c00e184439a2
$PY -m r2ai.newq pins                   # "input pins ok: 79 files"
sha256sum configs/s41-private-final.json configs/s41-newq-pins.json configs/s41-private-fallback-cos.json \
  configs/s41-private-fallback-cos-k150.json notebooks/s36-kaggle-4b-final.ipynb
sha256sum D:/GitHub/r2ai-stage3/data/s37_en/seg37900-en-pubmed/{vectors.npz,docs.parquet}
```

| tệp / mục | SHA256 / giá trị |
|---|---|
| `configs/s41-private-final.json` | `619221139c45d9310413dab2e595cb9736c47f02986e4589329fecab01ca4759` |
| `configs/s41-newq-pins.json` (79 tệp) | `c60237e457bbc3d69d7d14aaf2f26bbb16b17738d1f40f5adc9054e5e2ae27f9` |
| `configs/s41-private-fallback-cos.json` | `81ba043f299f674002da532c9b6c2ab3fef4000cc4e935883c57d087beca22e9` |
| `configs/s41-private-fallback-cos-k150.json` | `7c020320529bfef2afaa478bc7e0fa9cdaf49dd6c39964c2e3536410e220653d` |
| `notebooks/s36-kaggle-4b-final.ipynb` | `58c645a733e2cff5ae5a1c89ab5b6ec1a0b6bf95e9ec631a2a312ee227779d7d` |
| `judge_score.py` (Kaggle) | `688948a7c93ac01ab7c18c565755318ceec5cb041c6663ec9204f7f5695a7186` |
| en `vectors.npz` / `docs.parquet` | `9edfe699…17bb0` / `c0e93e14…41c2` (lệch: bỏ EN-hedge, báo Vu) |
| model | 4B `22e683669bc0f0bd69640a1354a6d0aebcfeede5` (Kaggle tải HF), 0.6B `e61197ed45024b0ed8a2d74b80b4d909f1255473` (máy), BGE-M3 (HF cache máy, offline) |
| máy | GPU rảnh, ổ D: trống ≥ 30 GB, quota Kaggle ≥ 15 giờ, Vu điền (A)–(D) mục 4 `private-final.md` |

## 1. Thứ tự lệnh theo giờ G (T0 = lúc có tệp query; run cố định `out/runs/private`)

| giờ | ai | lệnh / việc | kiểm |
|---|---|---|---|
| T0 | tôi | `sha256sum <tệp query>` | trùng SHA Vu gửi; parquet cột `id` int duy nhất, `query` khác rỗng |
| T0+0:05 | tôi | `$PY -m r2ai.newq prepare --queries <tệp> --run out/runs/private` | `prepare.json`: pairs = 300 × N |
| T0+0:10 | tôi | `$PY -m r2ai.newq export-scoring --run out/runs/private --parts-4b 2` | `4b/kaggle-dataset-private-4b-{1,2}`, `4b/private-4b-{1,2}.ipynb`, `cmax/pool/manifest.json` |
| T0+0:30 | tôi | `powershell -File scripts\s40\run_newq_gpu.ps1 -Run D:\GitHub\r2ai-stage3\out\runs\private` (C-max 0.6B, khoảng 10 giờ) | `gpu.log` có `cmax imported`, `ledger.json` có `cmax.parquet` |
| T0+0:30 | Vu | Kaggle phiên 1: upload dataset 1, chạy `private-4b-1.ipynb` (2×T4, khoảng 7,5 giờ) | notebook tự assert pairs, revision, SHA |
| T0+8:00 | Vu | tải `scores-extra.parquet` + `collect-extra-report.json`; chạy ngay phiên 2 `private-4b-2.ipynb` | |
| T0+8:10 | tôi | `$PY -m r2ai.newq import-scores --run out/runs/private --source 4b --tag private-4b-1 --scores <…>/scores-extra.parquet --report <…>/collect-extra-report.json` | SHA, revision `22e68366…`, đủ cặp |
| T0+15:45 | tôi | như trên với `--tag private-4b-2` | `newq status`: 2 phần imported |
| T0+16:30 | tôi | **CH-rrf:** `$PY -m r2ai.newq build --run out/runs/private` | `build/private-free/build.json`: `carriers.order` = rank, carrier = round(146,29 × N), khoảng 184 doc/query, ZIP < 102.600.000 B, validator ok |
| ngay sau | tôi | **EN-hedge (bản 5):** `$PY -m r2ai.newq build --run out/runs/private --en-hedge 3 --name private-free-en3` | `build.json` khoá `en_hedge.under_ceiling` = true |
| | Vu | nộp lượt 1 = CH-rrf; báo Vu ZIP + SHA256 từ `build.json` khoá `zip` | |

## 2. Điều kiện chuyển nhánh (chuỗi CH-rrf → B0 → COS-K150)

| nhánh | điều kiện | lệnh | build | LB public |
|---|---|---|---|---|
| CH-rrf | đủ 2 phần 4B + C-max | `build --run out/runs/private` | `private-free` | 0,4138 |
| B0 | đủ 4B, C-max hỏng hoặc chưa xong ở T0 + 24 giờ | `build --run out/runs/private --no-cmax` | `private-free-noCmax` | 0,4101 |
| COS-K150 | không phần 4B nào nhập được | `build --run out/runs/private --cos-k 150` (thêm `--no-cmax` nếu C-max cũng thiếu: chưa đo) | `private-free-cosk150` | 0,3742 |
| IMP30 | **không dùng** | | | |
| EN-hedge | phòng hộ lượt 5, trên nhánh đang dùng | lệnh của nhánh + `--en-hedge 3 --name <build>-en3` | `<build>-en3` | 0,4104 (public không có gold en; `exp/s43` `docs/s43.md`) |

* Thiếu đúng một phần 4B: chạy bù phần đó trên Kaggle (notebook bỏ qua part đã xong); chưa bù được thì xuống COS-K150 (IMP30 không dùng), Vu quyết.
* Phần 4B về muộn: nhập rồi dựng lại nhánh tốt hơn; bản đã nộp giữ nguyên.
* 5 lượt: (1) bản chính; (2) giữ cho bản sửa lỗi; (3), (4) để dành; (5) EN-hedge.
