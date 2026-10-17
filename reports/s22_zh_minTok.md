# S22 — S11 zh min-token re-segment

- Nguồn: docs và chunk segments của index zh S11; chỉ đọc, không fetch mạng.
- Tập chọn: `status=ok`, body không rỗng, `1 <= n_tokens_bge_m3 < 50`, primary ID và toàn bộ alias thuộc corpus v2, chưa xuất hiện trong doc groups của segment S11 hiện hành.
- Đếm chính xác: **1.722 doc**, tất cả 27–49 token. Vượt ngưỡng 500 nên đã re-segment với min zh token = 1, target 256.
- Bundle CC độc lập: `data/s22_zh_minTok/manifest.db`; worktree `C:\Users\Admin\.codex\worktrees\zh-short-threshold\r2ai-stage3`. Manifest dùng `CC_MANIFEST_SCHEMA_SQL`; provenance dùng `PROV_SCHEMA`; `source='s11-minTok'`, `raw_shard` là đường dẫn repo-relative tới shard S11 gốc.
- 5 batch theo host; segment IDs 30100–30104 là provisional. S15 cấp ID production và embed khi nhập.

| Host | Bị min-token loại | Docs bundle | Chunk t256 | Raw shard |
|---|---:|---:|---:|---:|
| `ask.39.net` | 1.706 | 1.706 | 5.002 | 981 |
| `fitness.39.net` | 1 | 1 | 2 | 1 |
| `heart.39.net` | 1 | 1 | 2 | 1 |
| `wujue.com` | 13 | 13 | 26 | 5 |
| `zydcd.com` | 1 | 1 | 2 | 1 |
| **Tổng** | **1.722** | **1.722** | **5.034** | **989 unique** |

## Kiểm chứng

- Membership: doc primary ID và toàn bộ `doc_ids_group` của docs/segments/chunks đều thuộc corpus v2 (**1.722/1.722 docs, 100%**). Không có ID trùng segment S11 hiện hành.
- Raw: 989 shard tồn tại; kích thước, `mtime_ns`, SHA-256 trong manifest khớp nguồn cho tất cả shard. Không thiếu raw.
- Schema: docs theo `vicrawl.extract_pipeline.SCHEMA`; provenance theo S15 `PROV_SCHEMA`; manifest theo CC schema hiện có.
- Chunker: 5 segment `kind=cc`, `status=chunked`, `target=256`, `min_body_tokens=1`; 5.034/5.034 chunk là substring verbatim không rỗng.
- Index chung không được ghi. Bundle nằm dưới `data/` ignored và không được commit; S15 cần nhập từ worktree tại đường dẫn ghi phía trên. Chỉ report và `docs/PROGRESS.md` được đưa vào commit/push.
