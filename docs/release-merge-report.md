# Báo cáo hợp nhất vào `release/final` (S35, 26/10/2026)

Worktree riêng `D:/GitHub/r2ai-stage3/.superpowers/worktrees/release-final`, nhánh `release/final` tách từ `main` local (`8dc8c80`, cũng là `origin/main`). Không rebase, không force-push; mọi hợp nhất là merge commit thường (`--no-ff`).

**Về "main chưa từng push":** `origin/main` có trên GitHub, nhưng dừng ở `8dc8c80` (17/10/2026), trước S25 đến S38. Mọi nhánh feat tách từ `c160bbf`, chậm `main` 4 commit. Chưa có mã nào của hai bản cuối nằm trên `origin/main`; nên "rủi ro lớn nhất" đúng ở chỗ main trên GitHub là bản cũ.

## 1. Trạng thái các nhánh trước khi hợp nhất

| nhánh | HEAD | đã push (`origin/<nhánh>` = HEAD) | thay đổi chưa commit | ghi chú |
|---|---|---|---|---|
| feat/s22-reextract | `6e600bf` | có | không có worktree | **không hợp nhất** (mục 4) |
| feat/s25-canonical | `8b1142a` | có | không | |
| feat/s26-gold-structure | `08befe6` | có | không | |
| feat/s27-pharmacity | `dbaaf53` | có | không có worktree | |
| feat/s29-qwen-judge | `e451fe7` | có | không | đã nằm trong s31 |
| feat/s31-strong-judge | `bd5cc8d` | có | không | |
| feat/s32-chunk-judge | `410e54e` | có | không | đã nằm trong s38 |
| feat/s33-btc-baseline | `81dad2a` | có | không | đã nằm trong s38 |
| feat/s35-part-jina | `5c7f392` | có | không có worktree | đã nằm trong s36 và s38 |
| feat/s36-fuse4 | `6171e61` | có (đã push 26/10) | lúc kiểm có 1 tệp sửa và 2 tệp mới, là công việc ablation của chính S35; đã commit và push | |
| feat/s37-120ask-vps | `4e1eed2` | có | worktree chung `D:/GitHub/r2ai-stage3` có 7 mục chưa theo dõi (`.scratch/`, `=ro`, `artifacts/`, `docs/superpowers/`, `tmp/`, `src/r2ai/cc_coverage/stage3_index.py`, `tests/test_cc_coverage_stage3.py`), không thuộc hai bản cuối, không đụng tới | hai tệp mã chưa theo dõi thuộc việc khác (cc_coverage) |
| feat/s38-chunk-metric-v2 | `74722ad` | có | không | |
| feat/s38-compose-registry | `a4b57c1` | **không** (không có `origin/`) | không | commit nằm trong s38-chunk-metric-v2 (đã push), nên không cần push |
| feat/s33-structure-signals | `a3edad2` | có | không | không hợp nhất (hai bản cuối không dùng) |

Compose đã dùng cho cả hai ZIP là `a4b57c1` (worktree `s38-dev`); trong release đó là mã của s38-chunk-metric-v2, và hồi quy byte (mục 3) chứng minh hai mã cho cùng ZIP.

## 2. Thứ tự hợp nhất, xung đột, pytest

Thứ tự theo phụ thuộc: s25, s26, s27, s31 (kéo theo s29), s38 (kéo theo s32, s33-btc, s35, s31), s36, s37; s22 bị bỏ.

Xung đột: **mỗi lần chỉ `docs/PROGRESS.md`** (3 đến 4 khối, đầu tệp và hai khối đuôi), không có xung đột mã. Cách giải (script `out/runs/s36-fuse4/resolve_progress.py`, áp đều): giữ phía release, thêm mọi dòng mục "- **…**" của nhánh vào, bỏ ba dòng metadata đầu tệp của nhánh (Cập nhật, Model thực hiện, Commit HEAD). Tệp là nhật ký nên giữ hai phía là đúng; dòng đầu của PROGRESS được viết lại một lần ở cuối (mục 5). Một dòng bảng S9 xuất hiện hai lần ngay trong `main` cũ, không do hợp nhất.

| sau khi hợp nhất | commit | pytest (passed / failed) | đỏ mới so với baseline | xử lý |
|---|---|---|---|---|
| (baseline `main` `8dc8c80`, chưa có `.env`) | `8dc8c80` | 813 / 16 | | 16 test đỏ chỉ vì thiếu tokenizer `BAAI/bge-m3` trong cache HF (chưa có `.env`) và dữ liệu mẫu vicrawl |
| s25-canonical | `52eb4be` | 820 / 18 | 2: `test_canonical::test_guard_refuses_outside_and_existing`; `test_segments::test_real_synthetic_benchmark…` | cái đầu đỏ ngay trên nhánh s25 (chạy riêng trong worktree s25: 1 failed, 8 passed): `atomic_write_bytes` không truyền `roots` cho tệp `.partial`; đã sửa ở mục 5. Cái sau cần ≥ 2 GiB RAM trống, đo được 0,8 đến 1,7 GiB vì ablation đang chạy; không phải lỗi hợp nhất |
| s26-gold-structure | `fc3ecdd` | 820 / 18 | không | lần đầu pytest dừng vì `pytest` không đối số nhặt `scripts/s26/pool_test.py` và `vi_pool_test.py` (không phải test); chuyển sang `pytest tests` và thêm `testpaths = tests` vào `pytest.ini` |
| s27-pharmacity | `a1c3516` | 859 / 18 | không | |
| s31-strong-judge (+ s29) | `71231c2` | 904 / 18 | không | |
| s38-chunk-metric-v2 (+ s32, s33-btc, s35) | `a86403b` | 980 / 18 | không | |
| s36-fuse4 | `202e9f7` | 1.006 / 17 | không (một test xanh lại) | |
| s37-120ask-vps | `13284be` | 1.026 / 21 | 3: `test_s37_ingest` (2), `test_vicrawl_integration` (ctrl-c) | chạy riêng 3 test cùng file: 18 passed. Đỏ do tải (compose chạy nền), không do hợp nhất |
| s22-reextract | không hợp nhất | | | mục 4 |
| bản cuối, có `.env` | `50cc8dc` | 1.046 / 1 | 1: test guard s25 | mục 5 |

Mọi baseline so sánh nằm ở `out/runs/s36-fuse4/release-pytest-*.log` và `release-pytest-*.failed.txt`.

## 3. Hồi quy byte (bắt buộc)

Từ `release/final`, `python -m r2ai.release build --config <config>` (apply → compose → validator → ZIP) và so với manifest `45ce6e49…e1f7`:

| bản | ZIP SHA256 dựng lại | khớp ZIP bàn giao | thời gian |
|---|---|---|---|
| 120A-p4b-v3 | `34d235f6f37704d11b803e83e9eddeabbf4c6100a38497c2728b56cb614937e2` | có, từng byte; kích thước, cửa sổ, chunk, doc, validator cùng khớp | 19,6 phút |
| 120A-free-v2 | `fba176c65f98b733ce879e932ab245905d9899e96eea867583af5ae85caada3f` | có, từng byte | 20,2 phút |

`scripts/s36/handoff_s30.py --verify --out out/runs/release/handoff-verify` cho cả hai config: `docsel`, `carriers`, `pcal`, `keep` trùng SHA, `inputs_sha256`, `apply_outputs_sha256`, `zip` của manifest mới sinh ra trùng với manifest bàn giao; 19 tệp nền trùng. Chỉ phần `code` khác (commit hiện tại), đúng dự kiến. Không sửa config nào cho khớp.

## 4. s22-reextract: không hợp nhất (quyết định của Vu, 26/10)

- Nhánh `feat/s22-reextract`, HEAD `6e600bfbb6bf4473de4f5506c6a571e4ecf54895`, đã nằm nguyên trạng trên `origin/feat/s22-reextract` (đối chiếu `git ls-remote` 26/10, cùng hash); không push gì thêm.
- Năm commit trên nền `a65b56a`: `21b484d` (pipeline crawl zh_v2 trực tiếp, resume được), `3af405e` (stage các lô S20 recovery cho S15), `662f285` (nhận doc CJK ngắn khi recovery), `dc040cd` (bundle resegment zh theo ngưỡng token tối thiểu), `6e600bf` (ghi kiểm tra tính tất định của extract S22).
- Phạm vi: extract lại zh với ngưỡng từ 50 chữ Hán. Dữ liệu kết quả đã nằm trong corpus canonical (`data/canonical-20261018/`, ghim SHA trong `docs/release-artifacts.md`), nên hai bản cuối không cần mã của nhánh này.
- Không hợp nhất vì merge xung đột ở 31 khối mã: `src/r2ai/zh_v2/ingest.py` 13, `src/r2ai/zh_full_ingest/extract.py` 8, `src/r2ai/zh_v2/extract.py` 4, `src/r2ai/zh_v2/index.py` 3, `src/r2ai/index/segments.py` 2, `src/r2ai/zh_v2/__init__.py` 1 (cộng 3 khối PROGRESS), cùng hiện tượng `main` đã ghi ở `8dc8c80` ("S22 merge blocked on zh_v2 conflicts"). Đã `git merge --abort`; `release/final` và `main` không chứa s22. Muốn dùng lại mã s22 thì cần giải các khối đó bởi người nắm zh_v2.

## 5. Thay đổi trong release ngoài các merge

- `pytest.ini`: `testpaths = tests`.
- `src/r2ai/canonical/guard.py`: `atomic_write_bytes` và `write_parquet` truyền cùng `roots` cho tệp `.partial` như cho tệp đích (mặc định không đổi); sửa lỗi có sẵn trên nhánh s25, test `test_guard_refuses_outside_and_existing` xanh (9/9 của `test_canonical.py`).
- `src/r2ai/release.py` (mới): `build`, `check`, `plan`. `.env.release.example` (mới). `scripts/s36/handoff_s30.py`: `--out`, `base_input_paths`.
- `docs/PROGRESS.md`: dòng đầu cập nhật (26/10, Claude Sonnet 5.5) và mục S35 release.
- Tài liệu mới: `README_RELEASE.md`, `docs/release-artifacts.md`, `docs/release-private.md`, báo cáo này.
