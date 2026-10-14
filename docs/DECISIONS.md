# Quyết định dự án — 12/10/2026

## D1. Common Crawl

Được dùng Common Crawl để lấy text chỉ cho URL có trong `links_corpus.parquet`. Lưu provenance nối từng text với `id`, URL corpus và snapshot Common Crawl; crawl ID đã điền ở `docs/COMPLIANCE.md` (13 crawl, `CC-MAIN-2025-38` … `CC-MAIN-2026-39`).

## D2. Hợp nhất nhánh

Merge worktree vào `main` bằng `git merge --no-ff`.

## D3. Biến thể zh

ZG là biến thể zh mặc định; ngưỡng logit `τ = 0,96` cố định.

## D4. Bỏ biến thể thử nghiệm

Xóa `out/runs/zh-full-ingest/ZF/ZF-5p` và `out/runs/zh-full-ingest/ZF/ZF-10p`. Đã xóa hai thư mục khỏi output ignored ngày 12/10/2026.

## D5. Domain

Bỏ `qihuangzhishu` khỏi kế hoạch crawl/ingest.

## D6. Batch zh

`b002` chỉ gồm `120ask` và `cnkang`.
