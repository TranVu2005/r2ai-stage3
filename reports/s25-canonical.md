# S25 — Corpus canonical (mỗi doc_id một phiên bản)

- Đo ngày 2026-10-18 (UTC+7), Claude Opus 5.5, nhánh `feat/s25-canonical` (worktree `.superpowers/worktrees/s25-canonical`, tách từ `ecd8e3f`).
- Nguồn chuẩn ID: corpus v2 S16, 1.220.598 URL, `data/corpus_v2_ids.parquet` SHA256 `dad50fb8…374a` (đã kiểm). en theo D17: chỉ có số liệu, không xử lý.
- Thư mục canonical mới: `data/canonical-20261018/` (17 GB). Không file cũ nào bị sửa, xoá hay di chuyển. Pha 3 **chưa thực hiện**: mới có đề xuất, chờ Vu duyệt và chờ S15 báo hàng chờ GPU rỗng.
- Code: `src/r2ai/canonical/` (`inventory`, `versions`, `select`, `build`, `verify`, `deleteplan`); test `tests/test_canonical.py`. Mọi writer chỉ ghi dưới `data/canonical-20261018` hoặc `out/runs/s25-canonical`, từ chối ghi đè (guard ở `guard.py`).

## Tóm tắt

| Chỉ số | Giá trị đo |
|---|---:|
| Artifact đã kiểm kê (repo chính + 7 worktree Codex) | 387.763 file / 231.832.168.786 B |
| Worktree ngoài phạm vi (10 sibling + 3 `.superpowers`), chỉ liệt kê | 854 file / 3.561.217.977 B |
| File doc-level có text (catalog phiên bản) | 174.834 (156.765 file thật, 18.069 hard link) |
| Dòng phiên bản (doc_id × file) / doc_id v2 có text | 2.743.888 / 887.151 |
| doc_id canonical (vi + zh), mỗi ID đúng 1 bản | **887.151** (vi 727.681, zh 159.470) |
| Chunk t256 canonical / có vector khớp text / cần re-embed | 5.647.948 / 5.599.214 / 48.734 |
| ID có hơn 1 phiên bản text (ghi vào `decisions.parquet`) | 89.816 |
| Kiểm bất biến (ID ∈ v2, duy nhất, chunk verbatim, vector mẫu) | đạt 100% |
| Đối chiếu P1.zip/P2.zip | còn lệch, liệt kê đủ (mục 4) |
| Đề xuất di chuyển (Pha 3) | 128 mục / 24.052.056.637 B, **chưa thực hiện** |

## 1. Kiểm kê (Pha 1, chỉ đọc)

Phạm vi quét: `D:\GitHub\r2ai-stage3\{data,out,state,artifacts,tmp,.scratch,submissions}` và mọi `C:\Users\Admin\.codex\worktrees\*\r2ai-stage3` (bỏ `src/tests/docs/configs/scripts` của worktree vì đã thuộc git). Theo xác nhận của Vu, quét thêm chỉ-đọc 10 worktree `D:\GitHub\r2ai-stage3-*` (trừ OLD) và `.superpowers/worktrees/*`. Không follow junction/symlink: worktree `doc-cut-best-a1-115` có `data/` và `out/` là junction vào repo chính, nếu follow sẽ đếm trùng.

Mỗi dòng kiểm kê: đường dẫn, số file, byte, mtime min/max, số file thuộc git, nhóm, tag session (suy từ đường dẫn, heuristic), số dòng, số doc_id phân biệt và số doc_id thuộc v2 (đọc cột ID của Parquet, `doc_id.bin` của segment; SQLite mở `immutable=1`; ZIP chỉ liệt kê member, không giải nén). Thư mục có trên 20 file cùng loại được gộp một dòng (shard crawl/extract).

| Nhóm | File | GiB | Thuộc git |
|---|---:|---:|---:|
| meta (json/yaml, phần lớn JSON submission) | 21.236 | 87,41 | 6 |
| vectors (dense/CSR) | 7.240 | 44,65 | 0 |
| zip | 2.006 | 22,79 | 0 |
| parquet (docs/candidate/…) | 181.386 | 14,74 | 0 |
| raw (HTML/WARC/jsonl.zst/CC) | 165.409 | 11,01 | 0 |
| db (SQLite) | 1.299 | 10,69 | 0 |
| chunks (chunk parquet, text.arrow, doc_id.bin) | 3.096 | 10,64 | 0 |
| model-cache (`data/hf_model_cache`) | 40 | 8,63 | 0 |
| scores (sqlite/pkl) | 1.433 | 3,33 | 1 |
| log | 909 | 0,37 | 0 |
| provenance | 224 | 0,04 | 0 |

Vùng lớn nhất: `data/index` 54,3 GB; worktree `upload-probe/out` 27,8 GB (ZIP nén thử); `out/runs/s6` 25,9 GB; `out/runs/cc-longchau` 10,3 GB; `out/runs/best-vsappend` 9,5 GB; `state/zh_full` 8,6 GB. Bảng đủ: `reports/s25/inventory-by-area.csv` (474 vùng) và `reports/s25/inventory-artifacts.csv` (11.834 artifact doc/chunk/vector/zip/score/db/provenance, kèm số doc_id). Bản thô: `out/runs/s25-canonical/inventory{,-extra}/inventory.parquet`.

Lưu ý đo được:
- `out/runs/cc-longchau/B/docs` gồm 18.006 hard link tới `data/docs_vi` (cùng inode): di chuyển một bên không thu hồi byte.
- 278 file không đọc được metadata (89 BadZipFile, 85 Parquet hỏng/rỗng, 66 OSError, 28 SQLite không hợp lệ), phần lớn trong thư mục pytest; ghi ở cột `note`.
- `tmp/`, `.scratch/`, `artifacts/` của repo chính không thuộc git.

### Phiên bản text theo doc_id

Catalog đọc mọi Parquet doc-level: segment docs (`doc_text`) và output extract (`title/question/answer/body` → `doc_text` theo đúng `r2ai.index.chunk.doc_layout`; đã kiểm 1.134/1.134 doc mẫu trùng byte với `data/chunks/docs.parquet`). Một phiên bản = (doc_id, SHA-256 của doc_text). Chất lượng đo trên body đã gộp whitespace: số code point và số chữ Hán.

1.473 file nằm dưới thư mục test (pytest scratch, fixture, `synthetic-*`) bị loại: đó là dữ liệu giả, không phải phiên bản của doc thật. Lần build đầu chưa loại các file này (5 doc vi lấy text từ `out/pytest-reranker-s15/…`); đã sửa trước bản build hiện tại.

- `out/runs/s25-canonical/versions/versions.parquet` (2.743.888 dòng) + `sources.parquet` (174.834 file).
- `out/runs/s25-canonical/versions/versions-by-doc.parquet`: 978.684 phiên bản (doc_id, sha, body_cp, han, live, danh sách file chứa, cờ được chọn). 797.335 ID có 1 phiên bản, 88.099 có 2, 1.717 có 3.

## 2. Quy tắc chọn (Pha 2, bước 3)

**View P2 (quy tắc a).** P2 = P1 + delta zh S22/webMedQA. Bản P2 đang phục vụ là các dòng live của index:
- vi: `data/index/vi_s18_segments/manifest.json` (SHA `564eb351…`) = base-t256 + seg00400…00406, trừ 4.264 cạnh supersede; cộng facade cc seg20000…20004 của snapshot P1 `564eb3511ebc1c06-s23-cumulative/facade-map.json`.
- zh: `data/index/zh_v2_segments/manifest.json` (SHA `38f0e180…`) = 73 segment gồm S22 seg30001…30009 và webMedQA seg20006, trừ 11.064 cạnh supersede, trong đó đúng 9.496 cạnh S22 (`superseded-s22.json`).
- View này có 869.104 ID v2 live (vi 712.194, zh 156.910) và **0 ID có hai bản live**.

Thứ tự áp dụng:
- (a) ID live trong view P2: lấy đúng text đó.
- (b) ID P2 không dùng: lấy bản đạt ngưỡng (vi ≥200 code point; zh ≥200 code point hoặc ≥50 chữ Hán) của lần extract mới nhất. "Mới nhất" đo bằng thời điểm sinh của lần chạy extract: mtime sớm nhất của các file doc-level trong thư mục nguồn đầu tiên chứa đúng text đó. Một lần chạy = một thư mục = một thế hệ extractor. Đây là proxy: các file không lưu version extractor (cột `extractor` chỉ là tên host-specific, ví dụ `generic`, `zhv2_qa_ask.39.net`).
- (c) Hai bản đạt ngưỡng cùng lần chạy: lấy body dài hơn. Không ID nào rơi vào (c).
- Không bản nào đạt ngưỡng: giữ bản body dài nhất với `quality=thin`, không chunk, không vector (theo Vu duyệt).

| lang | (a) good | (a) thin | (b) good | thin (không bản nào đạt) | Tổng |
|---|---:|---:|---:|---:|---:|
| vi | 712.175 | 19 | 10.394 | 5.093 | 727.681 |
| zh | 156.863 | 47 | 614 | 1.946 | 159.470 |

Có 66 ID "(a) thin": P2 đang phục vụ text dưới ngưỡng; giữ nguyên để khớp P2.

(b) chủ yếu: vov.vn 8.363 (S24 direct seg20007), baolangson.vn 1.978 (S24 seg20005), cùng 666 doc chỉ có ở output extract (zh cnkang 449, 120ask 151, …). Thin chủ yếu: laodong.vn 4.022, ask.39.net 1.338, suckhoedoisong.vn 450.

`decisions.parquet` (89.816 ID có ≥2 text): vi (a) 86.276, (b) 79, thin 340; zh (a) 3.116, (b) 2, thin 3. Mỗi dòng ghi luật, lý do, SHA được chọn và danh sách bản thay thế (sha, body_cp, han, birth, live, đạt ngưỡng, số file). Bản thay thế phổ biến nhất:
- overlay `P1-source/docs.parquet` (80.435 ID vi): text "full" do builder P1 ghi, không phải doc_text.
- 4.264 ID base đã bị S18 Tâm Anh/S22 supersede.
- Bản zh cũ trước S22 (seg00108/00109/00105) và bản S22 ngưỡng cũ (`s22_reextract`).

## 3. Canonical đã ghi (Pha 2, bước 4)

`data/canonical-20261018/manifest.json` SHA256 `d90db47fd97dcd4cc52785b4d479d12d8dfa92c0c046d7dddfea4ad4b95e2f11`. Manifest ghi SHA của mọi input (corpus, hai manifest index, facade P1, `superseded-s22.json`, catalog) và mọi file output:

| File | Byte | SHA256 |
|---|---:|---|
| docs.parquet | 1.316.555.347 | `80c52371…aed7b` |
| chunks_t256.parquet | 1.363.084.469 | `104211f1…f4d92` |
| decisions.parquet | 7.344.977 | `e044297a…6bba0` |
| provenance.parquet | 34.738.656 | `2bc9ed55…ff978` |
| reembed.parquet | 204.758 | `56194f1d…cf95c` |
| vectors/dense.npy (fp16 5.599.214×1024) | 11.467.190.400 | `a46e26af…4123b` |
| vectors/data.bin · indices.bin (CSR) | 1.896.084.028 ×2 | `d6636533…` · `9642a4af…` |
| vectors/indptr.bin · source_row.bin | 44.793.720 · 44.793.712 | `f93cfbdc…` · `7b97d009…` |

- `docs.parquet`: 1 dòng mỗi doc_id: text, lang, host, url, phys_id, body_cp, han, quality, rule, SHA text, segment và file nguồn.
- `chunks_t256.parquet`: chunk của bản được chọn (luật a, hoặc quality good). Chunk ID global giữ nguyên quy ước production `(65536 + segment_id) << 32` + chunk local; base-t256 bắt đầu từ 0. Cột `doc_ids` liệt kê mọi ID v2 dùng chung text vật lý đó.
- 666 doc (b) chỉ có ở output extract được chunk bằng `r2ai.index.chunk` (t256, chunker production): 1.433 chunk trong dải riêng seg25000.
- `vectors/`: chỉ copy vector khi text dòng tương ứng trong `text.arrow` của index nguồn trùng byte với text chunk. Kết quả: 0 dòng lệch, 5.599.214 vector copy từ 86 index (base t256, segment vi/cc/zh và index cô lập S24 baolangson).
- `reembed.parquet` (48.734 chunk, chưa encode): VOV seg20007 47.298 (S24 chưa BGE-embed), seg25000 1.433 (chunk mới), `chunks_zh_sample` 3 (không có index cùng số dòng). Bản CSV: `reports/s25/reembed.csv`.
- `provenance.parquet`: file nguồn + SHA-256 file, extractor/status, con trỏ raw (WARC/raw shard) khi có `provenance*.parquet`.

## 4. Xác minh (Pha 2, bước 5)

| Kiểm | Kết quả |
|---|---|
| doc_id ∈ v2, đúng lang, không trùng, không có en | 887.151/887.151 đạt; 0 ngoài v2; 0 trùng |
| Chunk verbatim `doc_text[char_start:char_end] == text` với mọi ID trong `doc_ids` | 5.659.660/5.659.660 tham chiếu đạt; 5.647.948 chunk ID duy nhất |
| Vector | 5.599.214 dòng = số chunk `has_vector`; mẫu 2.000 dòng trùng byte với index nguồn |

**Đối chiếu ZIP** (`out/runs/s25-canonical/verify-zip4/`). P1.zip SHA `6a96fe98…b7f3`; P2.zip SHA `e202470b…120f`, JSON `47515fd7…88ff`, được S15 duyệt lúc 15:01. Mỗi `relevant_chunk` được phân loại theo cách nó suy ra từ canonical:

| Loại | P1 | P2 |
|---|---:|---:|
| relevant_docs tìm thấy | 196.729 | 198.099 |
| relevant_docs không có trong canonical | 2.111 (1.261 ID) | 2.111 (1.261 ID) |
| chunk = doc_text canonical (zh: cả doc) | 2.458 | 2.996 |
| chunk = một chunk t256 canonical | 4.043 | 4.043 |
| chunk = body span của doc_text (full text vi) | 114.604 | 114.604 |
| chunk = answer span của doc_text | 1.635 | 1.635 |
| chunk chỉ là chuỗi con của doc_text | 47 | 45 |
| chunk lệch text | 3.441 | 3.435 |

Hai loại "body span" và "answer span" là cách `make_submission.load_full_texts` lấy text: answer nếu trang có answer, ngược lại cả body. Kết quả này suy ra xác định từ canonical.

Toàn bộ trường hợp lệch nằm trong `reports/s25/zip-mismatches.csv`:
1. **1.261 ID không có text ở bất kỳ đâu** (thaythuocvietnam 499, nhathuoclongchau 432, youmed 279, thanhnien 26, medlatec 17, suckhoedoisong 3). Catalog không có dòng nào cho các ID này: đó là ID chỉ-có-ID (slug-append) trong `relevant_docs`. Canonical không thể chứa chúng; đây không phải lỗi dựng.
2. **3.435 chunk Tâm Anh (1.662 doc) trong P2 là body cũ của base-t256**, bản đã bị seg00400 S18 supersede. Đã kiểm 3.435/3.435 trùng body base cũ. Index P2 dùng text S18, nhưng ZIP P2 (thừa kế C3/S6-f) vẫn chứa text cũ. Canonical đang theo index (S18). **Cần Vu quyết** (câu hỏi Q1).
3. P1 có thêm 6 chunk zh ask.39.net lệch và 2 chuỗi con: đó là text trước S22. P2 đã thay bằng text S22, nên ở P2 không còn.
4. 45 chunk vi P2 chỉ là chuỗi con: answer bị cắt khác biên span chunk (vinmec/thanhnien/hellobacsi). Text vẫn nằm nguyên trong doc_text canonical.

## 5. Độ phủ cuối so với S20

`reports/s25/coverage-final.csv` liệt kê theo (lang, host): URL v2 / tốt / thin / thiếu, kèm cột S20 tương ứng (`s20-coverage-audit.csv` 2026-10-17).

| lang | URL v2 | Tốt | Thin | Thiếu | S20 tốt/proxy | Δ tốt |
|---|---:|---:|---:|---:|---:|---:|
| vi | 740.598 | 722.569 | 5.112 | 12.917 | 644.341 | +78.228 |
| zh | 240.000 | 157.477 | 1.993 | 80.530 | 141.368 | +16.109 |
| en | 240.000 | — | — | — | 239.353 | D17, không xử lý |

Thay đổi lớn nhất so với S20:
- nhathuoclongchau: +58.052 (S19 CC; S20 chỉ có proxy 18.816).
- vov.vn: 0 → 9.755.
- ask.39.net: +9.751 (S22).
- youmed: +5.080.
- suckhoedoisong: +2.707.
- cnkang: +2.533.
- baolangson: 0 → 1.978.
- 120ask: +1.892.

Còn thiếu nhiều nhất:
- zh 120ask.com: 73.009.
- zysjonline.com: 6.591.
- pmc-ecm-healthblog (pharmacity): 5.333.
- nhathuoclongchau: 2.730.
- thaythuocvietnam: 2.331.
- youmed: 2.109.
- article.iiyi.com: 729.

Cách đếm "tốt" của S25 khác S20: S25 đo trên body của bản được chọn. S20 dùng proxy `extract_ok` cho 4 host đang chạy S3/S11.

## 6. Delete-plan (Pha 3, bước 6, chỉ đề xuất)

**Điều kiện Pha 3 chưa đạt.** Vu chưa duyệt danh sách. Theo `out/runs/reranker-trial/delta-queue.json` (11:20), S15 vẫn còn `delta_vi_2` (embed VOV + baolangson) trong hàng chờ GPU. Vì vậy **chưa di chuyển gì**. Bước kiểm sau di chuyển (dựng lại P2 từ canonical, so SHA JSON) cũng chưa chạy.

`reports/s25/delete-plan.csv`: 128 thư mục, 24.052.056.637 B. Không có hard link nên số byte thu hồi bằng số byte liệt kê.

| Nhóm | Mục | Byte | Lý do |
|---|---:|---:|---|
| `data/index/zh_full_segments/seg000*` (không gồm `cc_longchau`) | 31 | 19.784.767.607 | Bản index zh_full chưa lọc v2; bản lọc v2 ở `data/index/zh_v2_segments` mới là live. Có chứa vector của doc ngoài v2 (v1-only); text gốc vẫn ở `data/docs_zh_full`. |
| `data/chunks_zh_full` | 1 | 2.473.740.118 | Chunk zh_full chưa lọc v2; bản live là `data/chunks_zh_v2` |
| pytest scratch (`out/**/pytest*`, `tmp/pytest*`) | 94 | 1.788.223.401 | Fixture test tổng hợp, không là input của build nào |
| `data/chunks_cc_pilot`, `data/docs_cc_pilot` | 2 | 5.325.511 | Pilot CC; thay bởi seg20000 / `docs_cc_longchau` |

Bằng chứng cho mỗi mục nằm ở cột `evidence`:
- không có tham chiếu trong 19.555 file code/config/manifest/pin (json, yaml, py, csv…) của repo, `out/runs`, `state`, `data/index` và các worktree. Nhắc tới chỉ trong `.md` được ghi riêng, coi là lịch sử;
- không phải input của canonical hay segment live: không thuộc file nguồn trong provenance, không thuộc 86 index vector, không thuộc 85 segment live;
- không chứa file bảo vệ: raw/WARC/shard, provenance, log request, queue/heartbeat S21, scores/pair_cache, ZIP, approval/validator, crawl/extract/manifest DB.

Pytest scratch được miễn bộ lọc mẫu vì `crawl.db`/`*.zip`/`*.log` bên trong là fixture giả.

Bị giữ: 22 mục (`reports/s25/delete-plan-kept.csv`):
- `data/index/zh_sample_t256`: `zh_full_ingest/ingest.py` tham chiếu.
- `data/chunks_zh_sample`: canonical đang dùng 3 doc làm nguồn.
- 20 thư mục `tmp/pytest*`: CSV ownership của S15 tham chiếu.

Không đưa vào đề xuất: các run lớn chứa ZIP đã nộp, JSON nguồn, cache điểm Qwen/BGE hoặc pin của replay (`out/runs/s6` 25,9 GB, `best-vsappend`, `c1`, `rrf-*`, `replay`, `cc-longchau`, `upload-probe`…). Muốn dọn các run này cần tách ở mức file cùng Vu.

Sau khi đóng băng, `data/canonical-20261018` (17 GB) sẽ trùng với index/chunk live (`data/index/t256`, `data/chunks`, các segment). Nhóm này để dành cho vòng dọn sau, khi pipeline chuyển hẳn sang canonical.

## 7. Câu hỏi cho Vu

- **Q1 (chặn kiểm SHA sau di chuyển):** 1.662 doc Tâm Anh. Index P2 dùng text S18 (seg00400). ZIP P2 vẫn chứa 3.435 chunk là body base cũ đã bị supersede. Canonical nên giữ S18 (như hiện tại, đúng theo index) hay giữ text cũ để "dựng lại P2 từ canonical → JSON giống hệt"? Nếu giữ S18 thì phép kiểm SHA ở bước 8 chắc chắn lệch đúng 3.435 chunk này, trừ khi lấy text đó từ bản thay thế được ghi trong `decisions.parquet`.
- **Q2:** 1.261 ID chỉ-có-ID trong `relevant_docs` không có text ở bất kỳ đâu. Xác nhận canonical không cần chứa chúng (dựng lại P2 sẽ lấy ID từ nguồn gốc của luồng slug).
- **Q3:** Duyệt `reports/s25/delete-plan.csv` (128 mục / 24,05 GB). Ngay cả khi duyệt, chỉ di chuyển sau khi S15 xác nhận hàng chờ GPU rỗng.

## 8. Tái tạo

```
set PYTHONPATH=<worktree>\src
python -m r2ai.canonical.inventory                 # -> out/runs/s25-canonical/inventory
python -m r2ai.canonical.inventory --extra --out out/runs/s25-canonical/inventory-extra
python -m r2ai.canonical.versions --workers 4      # -> out/runs/s25-canonical/versions (711 s)
python -m r2ai.canonical.build                     # -> data/canonical-20261018 (506 s)
python -m r2ai.canonical.verify --zip P1=out/runs/reranker-trial/release/P1.zip --zip P2=out/runs/reranker-trial/release/P2.zip
python -m r2ai.canonical.deleteplan                # -> out/runs/s25-canonical/delete-plan (CSV only)
```

Hai build thất bại trước đó được chuyển sang `out/runs/s25-canonical/failed-build-{1,2}`, không xoá:
- lần 1: tràn offset string 32-bit khi `take` trên `text.arrow` của base; đã chuyển sang `take_texts`;
- lần 2: dừng ở bước chunk extract vì nguồn pytest không có cột `paragraphs`; sau đó thêm bộ lọc nguồn test.

Test: `pytest -q` trong worktree: **819 passed, 2 deselected** (393 s), gồm 7 test mới ở `tests/test_canonical.py`.
