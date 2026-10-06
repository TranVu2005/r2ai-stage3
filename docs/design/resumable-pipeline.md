# Pipeline zh chạy không giám sát, resume an toàn

Trạng thái: **đã duyệt có sửa (2026-10-06)**; S1 (watchdog) đang triển khai. Ngày 2026-10-06.

Phạm vi rút gọn đã duyệt: mục tiêu ~3 ngày công, batch đầu seal trước 10–11/10. Một worker tuần tự (không lease), ~10 ca fault ở ranh giới, tất định byte cho extract/chunk/seal, embed theo nguyên tắc "shard done không bao giờ tính lại". Xoay log và `ingest_seq` để sau.

Phạm vi: crawl → extract → chunk → embed → segment → truy hồi → build submission cho corpus zh-full, chạy trên laptop (RAM 15,7 GiB, RTX 3050 6 GB) tới khoảng 01/11/2026.

Nguyên tắc: tái dùng code có sẵn, thay đổi nhỏ và tăng dần, không viết lại. Crawler đang chạy (PID 4892) **không bị dừng** để triển khai. Mọi thay đổi ở crawler chỉ áp dụng ở lần restart có kế hoạch (Ctrl+C → exit 130 → resume).

---

## 1. Khảo sát hiện trạng (Phase 1, chỉ đọc code)

### 1.1 Bảng theo bước

| Bước | Đơn vị | Trạng thái lưu ở | Ghi atomic? | Chạy lại đơn vị đã xong | Kill giữa chừng | Khoá chạy trùng | RAM/GPU |
|---|---|---|---|---|---|---|---|
| Crawl (`zh_full/crawl.py` + `vicrawl/state.py`) | URL; buffer 10 record → 1 shard raw | `state/zh_full/crawl.db` (WAL, `synchronous=NORMAL`), cột `urls.status/shard` | Raw shard: tmp → fsync → `os.replace`, sau đó mới `commit_results` | No-op: URL `ok` không được lấy lại | Giữa ghi shard: còn `.tmp` mồ côi. Sau rename, trước commit DB: shard **mồ côi** có record, URL vẫn `in_progress` → `recover()` trả về pending → fetch lại → **record trùng ở shard thứ hai, HTML có thể khác**. Buffer chưa flush: mất ≤10 record/lane, URL được requeue (đúng) | `msvcrt` lock `out/runs/zh-full/crawl.lock`; OS tự nhả khi process chết (không có khoá cũ) | ~65 MiB RSS, mạng, IP VN |
| Extract zh (`zh_sample/extract.py`, dùng `_atomic_parquet` của `vicrawl/extract_pipeline.py`) | 1 raw shard → 1 parquet | `extract.db` bảng `shards(path, output, version)`. Journal mặc định, không WAL | tmp → `os.replace`, **không fsync** | No-op theo `(path, version)` | Giữa ghi parquet: `.tmp` mồ côi (zh path **không dọn**, vi path có dọn). Sau rename, trước `commit`: chạy lại sẽ ghi đè cùng nội dung (idempotent). Mất điện sau rename: parquet có thể rỗng/hỏng mà DB đã ghi done | `exclusive('extract')` | Tokenizer + 1 shard; workers=1 |
| Gộp docs (cuối `extract_all`) | Toàn bộ corpus | Không có. Mỗi lần chạy đọc lại **mọi** parquet | `atomic_json` + `_atomic_parquet` | Không no-op: làm lại toàn bộ, O(corpus) | Kill: làm lại từ đầu, kết quả không đổi | như trên | RAM tăng theo corpus (`docs` dict giữ mọi row): **không dùng được cho 1,7 triệu doc** |
| Chunk (`index/chunk.py`) | Toàn corpus; bucket `--batch-docs` | Không có checkpoint giữa bucket | `.partial` → rename sau khi xong **mọi** target; `chunk_report.json` ghi không atomic | Không no-op: chạy lại từ đầu | Kill cứng: `.partial` còn lại (lần sau ghi đè), mất toàn bộ tiến độ. `chunk_id` toàn cục liên tục 0..N−1 → **không cuốn chiếu được** | Không | Theo bucket |
| Embed (`index/build.py`, `zh_sample/index.py`) | Shard 1.000–10.000 chunk theo `chunk_id` | Sự tồn tại `{i}.sparse.npz` (+`dense.npy`) | Dense: tmp → rename (vi có fsync, zh **không**). Sparse: `save_npz` tmp → rename, **không fsync** | No-op nếu file shard tồn tại | Kill giữa shard: mất shard đang làm (đúng). Shard đánh số theo vị trí, **không gắn hash input** → chunk file đổi (rechunk) mà shard cũ vẫn bị coi là xong (vi). zh có `input_manifest.json` chặn trộn bundle (tốt) | zh: `exclusive('gpu')` trong `out/runs/zh-sample/` + `gpu_idle()` (nvidia-smi). vi: **không khoá GPU** | GPU ~VRAM batch 16; RAM giữ toàn bộ text |
| Assemble (`assemble_index`) | Toàn index | Không | `meta.json` vi ghi không atomic | Làm lại toàn bộ | Kill: làm lại, ổn | Không | Ghép dense O(N) |
| Segment (`index/segments.py`) | 1 segment = 1 lô ingest | `segment.json` (ghi cuối, mode `'x'`) | Các file dữ liệu **không fsync**; `segment.json` ghi trực tiếp, **không tmp+rename** | Writer từ chối thư mục đã tồn tại | Kill trước manifest: thư mục dở không load được (đúng), nhưng writer **không tự dọn** → kẹt. Kill giữa ghi `segment.json`: JSON cụt → load lỗi, cũng kẹt. Manifest **không có SHA/số dòng file** | Không | Ghi 0,51 GiB private; scan 8M hàng 0,70 GiB |
| Truy hồi (`retrieve/run_retrieval_k100.py`) | Query; checkpoint mỗi N query | `_partial*.pkl` + `.lock` | tmp → fsync → replace | Bỏ qua query đã xong | Ổn | `.lock` | Tuỳ reranker/GPU |
| Build submission (`submit/*`) | Một file | Không cần | Builder tất định, gate bằng SHA JSON | Tất định | Ổn | Không | ~5 GiB private khi validate |
| Điều phối | — | **Không có** | — | — | Mỗi bước chạy tay | — | — |
| Sống sót sleep/logoff | — | — | — | — | 17:32 ngày 06/10: Event Log ghi User Logoff + `SetSuspendState` (Kernel-Power 187/42) → crawler chết. **Không có watchdog** | — | — |

### 1.2 Danh sách hổng (mức độ: Cao = có thể hỏng dữ liệu/kẹt pipeline; TB = làm lại việc hoặc lệch kết quả; Thấp = vận hành)

| # | Bước | Hổng | Mức |
|---|---|---|---|
| H1 | Điều phối | Không có hàng đợi, lease, retry, cách ly lỗi | Cao |
| H2 | Vận hành | Không có watchdog/Task Scheduler; logoff/sleep giết crawler | Cao |
| H3 | Crawl→Extract | Shard mồ côi (rename xong, DB chưa commit) gây record trùng; extract quét **thư mục** thay vì đọc DB | Cao (lệch byte) |
| H4 | Chunk | Không cuốn chiếu được; `chunk_id` phụ thuộc toàn corpus | Cao |
| H5 | Gộp docs | O(corpus) RAM và thời gian mỗi lần chạy | Cao ở quy mô full |
| H6 | Segment | Thư mục dở làm kẹt writer; `segment.json` không atomic; không SHA | Cao |
| H7 | Extract/Embed/Segment | Thiếu fsync trước rename (parquet, sparse, dense zh, file segment) → mất điện có thể để lại file rỗng đã "done" | TB |
| H8 | Embed | Shard xác định theo vị trí, không theo hash input (vi) | TB |
| H9 | GPU | Khoá GPU nằm trong thư mục riêng của zh-sample; job vi/agent khác không dùng chung | TB |
| H10 | Extract zh | Không dọn `.tmp` mồ côi; `extract.db` không WAL | Thấp |
| H11 | Theo dõi | Không có `status` tổng; log không xoay vòng (`requests.jsonl` tăng mãi) | Thấp |
| H12 | Tất định GPU | Embed lại cùng shard có thể khác bit nếu kernel không tất định | TB (xem §4) |

---

## 2. Kiến trúc tổng quát

```
             (dịch vụ chạy liên tục)                    (theo lô, một worker tuần tự)  
crawl.db ──► LEDGER (append-only, thứ tự cố định) ──► batch k ─► extract ─► chunk ─► embed(shard) ─► seal segment k
                                                                                                       │
                       (chỉ khi có lệnh)    snapshot danh sách segment đã seal ◄────────────────────────┘
                                             └─► truy hồi ─► build submission (ghi hash snapshot)
```

- **Nguồn sự thật**: `crawl.db` cho crawl; `state/zh_full/pipeline.db` cho mọi bước sau.
- **Hai loại process**:
  - Crawler: process riêng, hiện tại không đụng vào.
  - `pipeline`: gồm supervisor và worker. Worker chạy **mỗi đơn vị là một subprocess**, để kill tách biệt và RAM được trả lại.
- **Mọi output là hàm thuần** của (đầu vào đã được ghi bất biến + phiên bản code/cấu hình). Trạng thái "done" chỉ là một con trỏ tới output đã có manifest.

---

## 3. Hàng đợi công việc (câu 1)

### 3.1 Schema `state/zh_full/pipeline.db`

PRAGMA: `journal_mode=WAL`, `synchronous=FULL`. Mọi chuyển trạng thái chạy trong `BEGIN IMMEDIATE`.

```sql
ledger(seq INTEGER PRIMARY KEY, url_norm TEXT UNIQUE, domain TEXT, raw_shard TEXT, line_no INTEGER,
       fetched_at REAL, doc_ids TEXT, record_sha256 TEXT, added_at REAL);
batches(batch_id INTEGER PRIMARY KEY, seq_lo INTEGER, seq_hi INTEGER,      -- [lo, hi)
        manifest_sha256 TEXT, created_at REAL);
tasks(task_id TEXT PRIMARY KEY,               -- ví dụ 'embed:b0007:s0003'
      kind TEXT, batch_id INTEGER, part INTEGER,
      input_sha256 TEXT, code_version TEXT,
      status TEXT CHECK(status IN ('pending','running','done','failed','quarantined')),
      attempts INTEGER DEFAULT 0, not_before REAL, last_error TEXT,
      output_path TEXT, output_manifest_sha256 TEXT, updated_at REAL);
deps(task_id TEXT, needs TEXT);               -- DAG trong từng batch
events(ts REAL, task_id TEXT, event TEXT, detail TEXT);   -- nhật ký, chỉ append
segments(batch_id INTEGER PRIMARY KEY, path TEXT, global_start INTEGER, rows INTEGER,
         manifest_sha256 TEXT, sealed_at REAL);
```

### 3.2 Vòng đời (một worker tuần tự)

- Supervisor giữ khoá đơn phiên `exclusive('pipeline')` (msvcrt, OS nhả khi process chết) và chạy **một** task tại một thời điểm, trong subprocess.
- Khi supervisor khởi động: mọi task `running` → `pending` (khoá đơn phiên bảo đảm không còn ai đang chạy chúng). Không có lease/heartbeat/TTL.
- `pending → running`: `UPDATE … WHERE status='pending' AND <mọi dep đã done> AND (not_before IS NULL OR not_before<=now)`.
- `running → done`: chỉ sau khi output đã rename và manifest đã kiểm (§4). Cùng transaction ghi `output_manifest_sha256`.
- `running → failed`: subprocess thoát khác 0. `attempts += 1`, lưu `last_error` (stderr cắt 4 KB). Retry với `not_before = now + 2^attempts` phút.
- **Cách ly**: `attempts ≥ 3` → `quarantined`.
  - Chỉ các task phụ thuộc trong **cùng batch** bị chặn. Batch khác vẫn chạy, vì ID của mỗi batch nằm trong không gian riêng (§6.2).
  - `status` liệt kê các task bị cách ly. Lệnh `pipeline retry <task>` đưa task về `pending` (do người vận hành gọi).
- **Lỗi tài nguyên** (RAM, đĩa, GPU bận, mất IP VN) **không tính là attempt**: task về `pending` kèm `not_before`.

### 3.3 Ledger: tách lô tất định khỏi crawler đang chạy

Crawler không bị sửa trong lượt này, nên pipeline tự xây ledger từ `crawl.db`:

1. Đọc `crawl.db` read-only (`mode=ro` URI).
2. Lấy URL có `status IN ('ok','thin') AND next_try_at IS NULL`, chưa có trong ledger.
3. Chỉ nhận record có `fetched_at < now − 900 s`. Biên an toàn này lớn hơn thời gian buffer tối đa (10 record ở domain chậm nhất ≈ 85 s) và thời gian flush khi dừng.
4. Sắp theo `(fetched_at, url_norm)`, append vào ledger với `seq` tăng dần trong **một** transaction.
5. Mỗi record trỏ tới đúng `urls.shard` trong DB và số dòng của nó trong shard đó.

Hệ quả:
- Extract chỉ đọc đúng record mà ledger trỏ tới. Shard mồ côi (H3) bị bỏ qua tự nhiên.
- Mỗi `url_norm` vào ledger **đúng một lần**. Nếu sau đó crawl ghi đè (retry), thay đổi chỉ được ghi `events`, không sửa ledger. Segment đã seal không bao giờ phải xoá dòng.
- **Batch**:
  - `batch k = seq ∈ [k·M, (k+1)·M)`. Chỉ cắt khi ledger đủ M record.
  - Lô cuối không đủ M chỉ được cắt bằng lệnh `pipeline freeze` (dùng trước hạn khoá ingest).
  - Cách chia lô vì thế chỉ phụ thuộc ledger, không phụ thuộc thời điểm supervisor thức dậy.
- Để sau (không chặn đường găng): thêm `urls.ingest_seq` gán trong `commit_results` để bỏ biên 900 s.

---

## 4. Ghi atomic + manifest (câu 2)

Helper chung `r2ai/zh_full/atomic.py`, mở rộng từ `common.atomic_json`:

- `atomic_write(path, writer_fn)`, theo thứ tự:
  1. Ghi `path.name + '.tmp-<pid>-<uuid>'`.
  2. `flush` + `os.fsync(file)`.
  3. `os.replace`, retry `PermissionError` 20 × 0,1 s (antivirus Windows, như `vicrawl/shards._replace`).
  4. fsync thư mục: Windows không hỗ trợ, nên ghi rõ giới hạn này. Bù lại, manifest và DB commit đến **sau**.
- **Manifest đơn vị** `<output>.manifest.json`, ghi sau cùng bằng `atomic_write`. Gồm:
  - `task_id`, `input_sha256`, `code_version` (git HEAD + hash cấu hình).
  - Danh sách file: `{name, bytes, sha256, rows}`.
  - `created_at`. Trường này chỉ để thông tin, không tham gia SHA tất định.
- **Thứ tự commit**: output rename → manifest rename → `tasks.status='done'` + `output_manifest_sha256` (một transaction).
- **Kiểm khi resume** (supervisor khởi động, và trước khi dùng output của dep):
  - Task `done` mà manifest thiếu hoặc SHA lệch → trả về `pending`, ghi `events: manifest_mismatch`.
  - Output có manifest đúng nhưng task chưa `done` (kill giữa rename và commit) → **nhận lại** (`adopt`): kiểm SHA, rồi đánh dấu done, không làm lại.
- **Dọn mồ côi**: xoá `*.tmp-*` và thư mục segment chưa seal, chỉ trong các root sở hữu (`guard`) và chỉ khi supervisor đang giữ khoá đơn phiên và không có task `running`. Ghi danh sách đã xoá vào `events`.
- Sửa H7: fsync trước rename cho parquet extract, dense/sparse embed, file dữ liệu segment.

---

## 5. Tất định (câu 3)

Cam kết (đã duyệt): **extract, chunk, seal là hàm thuần của (tiền tố ledger, code_version)**: kill và resume ở bất kỳ điểm nào cho output giống byte với lượt không bị ngắt có cùng ledger. **Embed**: shard đã `done` không bao giờ bị tính lại, nên segment bất biến qua resume; không cam kết embed lại cho giống bit. Bản thân nội dung crawl (mạng) không tất định; nó được "đóng băng" khi vào ledger.

| Bước | Biện pháp |
|---|---|
| Extract | Input là danh sách `(raw_shard, line_no)` theo `seq`. Output parquet theo thứ tự `seq`. Không dùng `time.time()` trong row. Parquet ghi với `compression='zstd'` cố định, `write_statistics` cố định, không metadata thời gian (kiểm `created_by`: cố định theo version pyarrow → version là một phần `code_version`) |
| Chunk | Chạy **theo batch**: docs của batch sắp theo `(doc_id, seq)`. `chunk_id` cục bộ 0..n−1. Thuật toán chunk giữ nguyên (`chunk_paragraphs`, kiểm verbatim) |
| Embed | Shard = `chunk_id ∈ [j·S, (j+1)·S)` trong batch. Batch nội bộ shard sắp theo độ dài như hiện tại (`encode`), seed 42. Shard `done` có manifest SHA và không bao giờ bị tính lại; kill chỉ làm mất shard chưa done |
| Segment | Ghi từ shard đã done theo thứ tự `j`. `global_start` cố định theo batch (§6.2) |
| Truy hồi | Tie `(score↓, global id↓)` như `retrieve.exact`. Snapshot segment cố định (§6) |
| Song song | Không có output nào phụ thuộc thứ tự hoàn tất: mọi gộp đều sắp theo khoá (`seq`, `chunk_id`, `j`), không theo thời điểm xong |

---

## 6. Segment bất biến và snapshot (câu 4)

### 6.1 Seal

`SegmentWriter` ghi vào `…/zh_full_segments/.staging/b{k:05d}-<uuid>/`. Seal theo thứ tự:
1. fsync mọi file.
2. Ghi `segment.json` qua `atomic_write`, kèm `{rows, nnz, global_start, batch_id, files:{name:{bytes,sha256}}, input_manifest_sha256}`.
3. `os.replace(staging_dir, b{k:05d})`. Trên cùng volume NTFS, rename thư mục là atomic.
4. Ghi `segments` row trong `pipeline.db`.

Thư mục trong `.staging` chưa seal sẽ bị dọn khi resume. Sửa H6 bằng cách bọc ngoài `segments.py`: staging + rename + SHA. Thuật toán ghi/scan giữ nguyên.

### 6.2 Không gian ID

- `global_start = ZH_BASE + batch_id · 2^24`, với `ZH_BASE = 2^40`.
- Mỗi batch giới hạn < 16,7 triệu chunk: M = 20.000 doc × ~5,5 chunk/doc là đủ thừa.
- `_validate_segments` chỉ cấm chồng lấn, nên khoảng trống ID hợp lệ.
- Tách hẳn khỏi ID vi (0..3,9 triệu): truy hồi gộp vi+zh không xung đột ID, và tie giữa vi/zh có thứ tự cố định.
- Batch bị cách ly không đẩy lệch ID của batch sau.

### 6.3 Snapshot

- `pipeline snapshot` ghi `snapshots/<sha>.json`, gồm danh sách `{batch_id, path, segment.json sha256}` của segment đã seal, sắp theo `batch_id`.
- `sha` = SHA256 của chính danh sách đó.
- Truy hồi **chỉ** nhận đường dẫn snapshot, không bao giờ liệt kê thư mục.
- Build submission ghi `segments_snapshot_sha256` vào stats/manifest của submission.

### 6.4 Giao diện cho phiên xếp hạng (không sửa `retrieve/`, `submit/` lượt này)

- `r2ai.index.segments.load_snapshot(path) -> list[Segment]`, kiểm SHA từng `segment.json`.
- `scan_topk(segments, qd, qsp, k)` như hiện có.
- `take_segment_texts(segments, ids)` như hiện có.
- Ánh xạ `global id → (doc_id, chunk text)` lấy từ segment. `doc_ids_group` cần thêm một file `doc_group.arrow` trong segment (khi seal).
- Phiên xếp hạng tự quyết định cách trộn ứng viên zh vào luồng vi. Đầu vào duy nhất là đường dẫn snapshot.
- **Top-k cộng dồn** (§13): `scan_topk_incremental(snapshot, prev_state, qd, qsp, k) -> state`. `state` lưu theo từng `batch_id` top-k dense và sparse mỗi query (`ids int64`, `scores float32`). Chỉ scan segment chưa có trong `prev_state`, rồi gộp bằng `merge_topk` (tie `score↓, id↓`). Kết quả giống byte với scan toàn bộ snapshot, vì top-k toàn cục ⊆ hợp top-k từng segment.
- **Cache điểm reranker**: khoá `(query_id, global_chunk_id, reranker_version)`, giá trị float32. Chỉ chấm cặp mới. Ranh giới batch gọi reranker cố định: cặp được sắp theo `(query_id, global_chunk_id)` rồi cắt lô kích thước cố định, tránh lệch fp16 do thay đổi thành phần batch.

### 6.5 Số segment và gộp định kỳ

M = 20.000 doc → khoảng 90 segment cho corpus full (ước tính). S3 đo block-scan 1.200 query trên ~100 segment nhỏ (CPU, dữ liệu giả, cùng tổng số hàng) so với 4 segment. Nếu chi phí tăng đáng kể (> 15 % thời gian hoặc > 20 % peak RAM), thêm bước `compact`: ghép các segment liền nhau thành một segment mới (không đổi `global id` của hàng, vì segment chứa mảng ID tường minh hoặc nhiều dải), seal, rồi đổi snapshot. Snapshot cũ vẫn hợp lệ vì segment cũ không bị xoá cho tới khi không còn snapshot nào tham chiếu.

---

## 7. Điều phối (câu 5)

Lệnh: `python -m r2ai.zh_full.pipeline <cmd>`.

| Lệnh | Việc |
|---|---|
| `run` | Supervisor (giữ `exclusive('pipeline')`): mỗi 60 s ① đảm bảo crawler sống (§9), ② cập nhật ledger, ③ cắt batch khi đủ M, ④ sinh task DAG cho batch mới, ⑤ nếu không có task đang chạy và tài nguyên cho phép: chọn task kế tiếp, chạy trong subprocess (một task tại một thời điểm). Truy hồi/build **không** bao giờ tự chạy |
| `status` | §10 |
| `daily` | §13 (phiên xếp hạng triển khai) |
| `freeze` | Cắt lô cuối (không đủ M) và dừng nhận ledger mới. Dùng trước hạn khoá ingest 28/10 |
| `snapshot` | Ghi snapshot segment đã seal |
| `retry <task>` / `quarantine-list` | Vận hành |
| `stop` | Ghi `stop_request.json` gắn PID (như crawler) |

DAG mỗi batch:

```
extract:b{k}  →  chunk:b{k}  →  embed:b{k}:s{j} (j = 0..⌈n/S⌉−1)  →  seal:b{k}
```

- `chunk:b{k}` sinh số shard embed sau khi chạy xong (dynamic fan-out). Supervisor tạo task embed khi chunk done.
- Ưu tiên: task GPU của batch cũ nhất trước, để seal sớm; extract/chunk của batch mới chạy khi GPU bận hoặc không có việc GPU.
- Chu kỳ ingest: theo **số lượng** (M doc), không theo giờ, để tất định (§3.3). Ở tốc độ hiện tại ~1,4 URL/s tổng, M = 20.000 tương ứng khoảng 4 giờ/batch (ước tính).
- Tái dùng code: extract dùng `extract_zh` + selector version, chunk dùng `chunk_paragraphs`/`doc_layout`, embed dùng `M3Encoder`/`encode`/`to_csr`, segment dùng `SegmentWriter`. Phần mới chỉ là adapter đọc/ghi theo batch.

---

## 8. Tài nguyên (câu 6)

| Tài nguyên | Quy tắc | Chặn ở |
|---|---|---|
| GPU | Khoá file **toàn máy** `state/locks/gpu.lock` (msvcrt; OS nhả khi chết) **và** `gpu_idle()` (nvidia-smi, không có compute process khác) trước mỗi shard. Job GPU của agent khác chưa dùng khoá này → `gpu_idle()` là hàng rào thứ hai. Đề nghị các phiên khác dùng chung helper (việc của họ) | task embed |
| RAM trống | Trước bước nặng: extract ≥ 4 GiB (theo AGENTS.md), chunk ≥ 3 GiB, embed ≥ 3 GiB, seal ≥ 2 GiB. Theo dõi trong lúc chạy (như `run_full_tests.py`): RAM trống < 1,5 GiB thì kill subprocess, trả task về pending, không tính attempt | mọi task |
| Một việc nặng tại một thời điểm | Supervisor chạy tuần tự đúng 1 task. Truy hồi/validator chạy tay phải lấy `state/locks/heavy.lock` | supervisor |
| Đĩa | Trước mỗi task: còn trống ≥ 70 GiB (reserve trong REPORT) thì chạy bình thường. Dưới 30 GiB thì dừng task mới. Dưới 15 GiB thì `stop` crawler qua `stop_request.json` (crawl không có cách dừng khác) | supervisor |
| IP VN | Crawler đã tự kiểm egress (`Network.check`). Pipeline không dùng mạng | crawler |

---

## 9. Sống sót qua sleep/logoff/reboot (câu 7)

Bằng chứng sự cố 06/10 17:32: Winlogon *User Logoff*, Kernel-Power 187 (*user-mode process gọi SetSuspendState*), 42 (*entering sleep*). Cài đặt hiện tại: sleep khi cắm điện = Never; wake timers khi cắm điện = *Important only*.

- **Task Scheduler** (cần người dùng duyệt): hai task, cùng *Run whether user is logged on or not*, *Run with highest privileges* = không:
  - `r2ai-zh-watchdog`: trigger *At startup* + lặp 5 phút; action `python -m r2ai.zh_full.watchdog`. *Wake the computer to run this task* = bật. Cần wake timers AC = *Enable* (`powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP RTCWAKE 1`), **thay đổi hệ thống → hỏi trước**.
  - Không tạo task riêng cho pipeline: watchdog lo cả crawler lẫn supervisor.
- **Watchdog** (one-shot, idempotent, ~100 dòng):
  - Đọc `runtime_state.json`/`pipeline.db`. Crawler hoặc supervisor coi là chết khi: PID không sống, `create_time` lệch, hoặc heartbeat cũ hơn 10 phút.
  - Chết → khởi động lại bằng **đúng lệnh** trong `launcher*.json`, tách khỏi phiên (`DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`).
  - Ghi `out/runs/zh-full/watchdog.jsonl` gồm thời gian, process, exit code nếu biết, số lần restart.
  - Quá 6 lần restart trong 1 giờ thì dừng thử và ghi cảnh báo (tránh vòng crash).
- **`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`** trong crawler và supervisor: chặn **idle** sleep. Không chặn được `SetSuspendState` chủ động hay đóng nắp. Vì vậy watchdog + wake là bắt buộc.
- **Khoá đơn phiên**: dùng `msvcrt.locking` như hiện tại; OS nhả khi process chết, nên không có khoá cũ. File khoá ghi thêm `{pid, create_time, started_at}` để `status` hiển thị chủ sở hữu.
- **Reboot/mất mạng**:
  - Sau reboot, watchdog chạy *At startup*.
  - Mất mạng do crawler tự xử lý (`EGRESS_UNAVAILABLE` → chờ 60 s). Pipeline không cần mạng.

---

## 10. Dừng êm và theo dõi (câu 8, 9)

**Dừng êm**: supervisor bắt SIGINT/SIGBREAK như crawler.
- Ngừng chọn task mới.
- Gửi CTRL_BREAK cho subprocess đang chạy. Subprocess làm xong **đơn vị nhỏ hiện tại** nếu ≤ 60 s (shard embed ~1.000 chunk), nếu không thì bỏ.
- Task về `pending` (không tăng attempt). Exit 130.
- Kill cứng: task còn `running`; lần khởi động sau chuyển nó về `pending` (§3.2).

**`status`** in bảng, đồng thời ghi `out/runs/zh-full/status.json`:
- Mỗi bước: done/pending/running/failed/quarantined.
- Tốc độ 60 phút gần nhất (từ `events`), ETA (ghi "ước tính").
- Heartbeat cuối của crawler/supervisor; nhật ký watchdog (số lần restart 24 giờ).
- RAM/đĩa/GPU hiện tại, snapshot mới nhất, số batch đã seal.
- Bảng `progress.csv` (§13).

**Log**: để sau (không chặn đường găng). Khi làm: `RotatingFileHandler` 50 MB × 5 cho supervisor/watchdog; `requests.jsonl` xoay theo ngày ở lần restart crawler sau.

---

## 11. Kiểm thử chịu lỗi (câu 10)

- **Điểm tiêm lỗi**: `R2AI_FAULT=<point>` → `os._exit(137)` khi đi qua điểm đó. Chi phí khi tắt: một lần đọc biến môi trường lúc import.
- **~10 ca ở ranh giới** (pytest, dữ liệu nhỏ: 2 domain giả × 100 record, 2 batch, encoder giả tất định CPU):
  1. `tmp_before_rename` (extract parquet)
  2. `rename_before_manifest` (extract)
  3. `manifest_before_commit` (chunk)
  4. `tmp_before_rename` (chunk)
  5. `manifest_before_commit` (embed shard)
  6. `seal_before_rename` (segment staging)
  7. `seal_after_rename_before_commit`
  8. `ledger_mid_append` (giữa transaction)
  9. Supervisor chết khi task `running` → khởi động lại → task về pending
  10. Chết hai lần liên tiếp ở hai điểm khác nhau
  Mỗi ca: chạy → chết → resume → SHA mọi artifact extract/chunk/seal giống bản chạy sạch; không còn `.tmp`; shard embed đã done không bị tính lại (đếm từ `events`).
- **Kill thật** một lần cho mỗi bước (dữ liệu nhỏ; embed GPU thật chỉ khi `gpu_idle()`): `taskkill /F` giữa extract, chunk, embed, seal; kill supervisor; kill crawler để watchdog resume ≤ 5 phút.

---

## 12. Danh sách thay đổi theo thứ tự (phạm vi đã duyệt)

Khối lượng là ước tính.

| Giai đoạn | Thay đổi | File/module | Khối lượng | Rủi ro | Hạn |
|---|---|---|---|---|---|
| **S1** | Watchdog one-shot + test; `SetThreadExecutionState` trong crawler; lệnh `powercfg`/`schtasks` cho người dùng; restart crawler có kế hoạch; kill thật | `zh_full/watchdog.py`, `zh_full/crawl.py`, `tests/test_zh_watchdog.py` | ~0,5 ngày | Thấp | 06–07/10 |
| **S2** | `atomic_write` + fault points; `pipeline.db` (queue tuần tự, retry, quarantine, `status`); ledger + cắt batch M | `zh_full/atomic.py`, `zh_full/queue.py`, `zh_full/ledger.py`, `zh_full/pipeline.py` | ~1 ngày | TB (biên 900 s ledger) | 08/10 |
| **S3** | Extract/chunk/embed theo batch (adapter tái dùng code có sẵn), khoá GPU toàn máy `state/locks/gpu.lock` + `gpu_idle()`; seal staging + SHA + snapshot + `load_snapshot`; đo scan ~100 segment nhỏ vs 4 (§6.5); ~10 ca fault + kill thật | `zh_full/stages.py`, `zh_full/locks.py`, `index/segments.py` (thêm hàm), `tests/test_zh_pipeline_faults.py` | ~1,5 ngày | TB (chunk theo batch phải khớp `index.chunk` trên mẫu; VRAM 6 GB) | batch đầu seal 10–11/10 |
| **S4** | Vòng lặp nộp hằng ngày (§13) | phiên xếp hạng: `retrieve/`, `submit/`; phía zh_full chỉ cung cấp snapshot + `scan_topk_incremental` | — (phiên xếp hạng ước tính) | TB | sau batch đầu |
| Để sau | Xoay log; `ingest_seq`; `compact` segment nếu S3 đo thấy cần | | | | |

Tổng phía zh-full: ~3 ngày công (ước tính).

---

## 13. Vòng lặp nộp hằng ngày (S4, phiên xếp hạng triển khai)

Mục tiêu: người dùng thấy tiến độ trên LB mỗi ngày trong lúc crawl vẫn chạy.

- Lệnh `pipeline daily` (phía zh-full chỉ gọi giao diện; logic truy hồi/dựng file thuộc phiên xếp hạng):
  `snapshot` → truy hồi zh **cuốn chiếu** → rerank có cache → dựng file → validator → ghi `out/runs/zh-full/daily/<YYYY-MM-DD>/`. **Không tự nộp.**
- **Top-k cộng dồn**: lưu top-k dense/sparse mỗi query theo segment (§6.4); segment mới chỉ scan phần mới rồi gộp. Hybrid min-max tính lại trên tập ứng viên hợp (rẻ).
- **Cache điểm reranker** theo cặp `(query, chunk)`, chỉ chấm cặp mới; ranh giới batch gọi reranker cố định (§6.4).
- Mỗi ngày dựng 3 nhóm file (base = best hiện tại; RD150 lúc viết):
  1. **Tiến độ**: base + append top N zh (chunk giống byte base). ΔDoc R so với base = gold zh thu được tới hôm nay.
  2. **Đo h theo lát**: append zh hạng 1–10 và 11–30 (rời nhau). h ≈ ΔDoc R × |G| / n, |G| ≈ 64 (ước tính).
  3. **Ứng viên thật**: zh thay đuôi vi, |D| giữ 150, theo quota hoặc RRF hạng (không trộn bằng logit reranker).
- `out/runs/zh-full/daily/progress.csv`: ngày, % URL crawl theo domain, số segment, số chunk zh, ΔDoc R (người dùng điền sau khi nộp), Final tốt nhất. `status` in bảng này.
- Mọi job GPU (kể cả phiên khác) phải lấy khoá `state/locks/gpu.lock` + kiểm `gpu_idle()`.
- Chi phí mỗi ngày (ước tính): scan ≈ 11 phút ở 8 triệu chunk (benchmark 657,9 s, 1.200 query); với top-k cộng dồn chỉ scan segment mới. Rerank zh mới cỡ chục phút GPU.

---

## 14. Quyết định của người dùng (2026-10-06)

| # | Quyết định |
|---|---|
| 1 Task Scheduler | Duyệt. Người dùng tự nhập mật khẩu Windows khi đăng ký |
| 2 Nguồn điện | Duyệt `RTCWAKE` AC = Enable. Người dùng tự kiểm Lenovo Vantage, cắm sạc, không gập máy/không bấm Sleep |
| 3 RAM | Duyệt 4/3/3/2 GiB (extract/chunk/embed/seal); kill khi < 1,5 GiB |
| 4 Đĩa | Duyệt dừng task mới < 30 GiB, dừng crawler < 15 GiB |
| 5 Ingest | Duyệt cắt theo M = 20.000 doc |
| 6 GPU | Duyệt tự chạy khi `gpu_idle()` + khoá toàn máy |
| 7 Restart crawler | Duyệt 1 lần, ngay sau khi watchdog chạy được, gộp `SetThreadExecutionState` |

Ghi chú xác minh: bản sửa Ctrl+C → exit 130 nằm trong `src/r2ai/zh_full/crawl.py` (`fetch_record` requeue khi `CancelledError`, handler SIGINT/SIGBREAK, `run()` trả 130), đã có trong commit `e7abe94`. `out/runs/zh-full/graceful_signal.py` chỉ là công cụ gửi `CTRL_C_EVENT` để kiểm thử, crawler không import.
