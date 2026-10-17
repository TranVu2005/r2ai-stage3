# S23 — Common Crawl cho 4 host tin tức

- Hoàn tất: 2026-10-17 13:16:54 UTC (20:16:54 UTC+7). Tổng thời gian từ 09:53:07 UTC: 12.227,5 giây (3 giờ 23 phút 47,5 giây).
- Nguồn: Common Crawl index và WARC qua HTTPS Range. AWS STS không hợp lệ nên không dùng S3.
- Input corpus v2: 1.220.598 URL; SHA256 `dad50fb81c812a5c68386a2313f880db488b457e85707b92d7c478c29523374a`. Base docs: 623.805 dòng; SHA256 `5b2d22b6b0382bb2b325b644357a31c7afc2cb0740ba6e54f7fcd3f3799099c8`.

## Phủ Common Crawl và text

Khớp URL chính xác hoặc chuẩn hoá scheme/host www/path slash trên đủ 13 crawl: 3.192 exact + 3 chỉ sau chuẩn hoá = 3.195 URL duy nhất; 3.195 có index status 200 text/html.

| Host | URL thiếu | CC exact | CC chỉ chuẩn hoá | CC có | WARC hợp lệ | Text ≥200 ký tự | ≥50 BGE-M3 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| suckhoedoisong.vn | 3.205 | 1.336 | 0 | 1.336 | 1.336 | 39 | 27 |
| vov.vn | 9.784 | 1.462 | 3 | 1.465 | 1.465 | 1.392 | 1.392 |
| laodong.vn | 4.098 | 348 | 0 | 348 | 348 | 0 | 0 |
| thanhnien.vn | 504 | 46 | 0 | 46 | 46 | 44 | 44 |
| **Tổng** | **17.591** | **3.192** | **3** | **3.195** | **3.195** | **1.475** | **1.463** |

Ngưỡng text tốt là body ≥200 ký tự Unicode; ngưỡng vector là status `ok` và ≥50 BGE-M3 tokens. **Lao Động: dead-in-CC.** Bằng chứng S23: 348 WARC response HTTP 200 nhưng body 177 byte là Cloudrity `document.cookie`/reload challenge (21 biến thể), 0 text tốt. Không theo đuổi 5 capture cũ chưa xác minh (4 capture chưa thử), theo chỉ thị Vu.

### Tái extract SKĐS theo selector S22 (offline)

- Đọc lại 1.336/1.336 WARC local (171.103.812 byte nén), không gửi request mạng. Cả 1.336 response là HTTP 200 `text/html`; URL đích sau chuẩn hoá, WARC record-id và WARC date khớp provenance 1.336/1.336.
- Dùng selector `vicrawl` `suckhoedoisong`; source blob `src/vicrawl/extractors/sites.py` SHA `207fee63f675e049623ad8f7c9d011c3bf1c1259` trùng hoàn toàn với `feat/s22-reextract`. Kết quả: 1.335 trang dùng selector SKĐS, 1 trang fallback generic; 39 body ≥200 ký tự và 27 doc status `ok` + ≥50 BGE-M3 tokens, bằng số trước re-extract.
- Cả 27 doc đủ điều kiện đều đã có trong segment 20004; **không có ID mới đủ chuẩn**. Vì vậy không cấp segment `cc-s23-sksds-20261017`, không embed; segment 20004 giữ nguyên. Thời gian re-extract đo được 33,87 giây; chi phí mạng $0.

## Crawl và tải WARC

| Crawl | Dòng khớp trong index |
|---|---:|
| CC-MAIN-2026-39 | 404 |
| CC-MAIN-2026-34 | 374 |
| CC-MAIN-2026-30 | 352 |
| CC-MAIN-2026-25 | 214 |
| CC-MAIN-2026-21 | 394 |
| CC-MAIN-2026-17 | 498 |
| CC-MAIN-2026-12 | 120 |
| CC-MAIN-2026-08 | 164 |
| CC-MAIN-2026-04 | 137 |
| CC-MAIN-2025-51 | 121 |
| CC-MAIN-2025-47 | 147 |
| CC-MAIN-2025-43 | 154 |
| CC-MAIN-2025-38 | 192 |

- Mỗi crawl được quét đầy đủ; có gồm CC-MAIN-2025-43 và CC-MAIN-2025-38.
- 3.195/3.195 WARC range cuối cùng xác thực hợp lệ; không còn lỗi WARC. Dung lượng record dự kiến 223.633.165 byte.
- Ledger Common Crawl: 2.826.402.308 byte / 3.738 request; tải WARC bằng range, concurrency tối đa 1. Có retry 503 và lỗi DNS trên 5 capture thay thế.
- Thời gian extraction thực đo: 56,1 giây. Embedding: 8.327 chunks trong 74,19 giây, 112,24 chunks/giây.

## Segment và provenance

- Segment active trong manifest `state/cc_longchau/manifest.db`: `seg20004-cc-s23-vi-news-20261017-vi-news`, segment_id 20004, batch `cc-s23-vi-news-20261017`, domain `vi-news`.
- 1.463 docs có chunk, 8.327 chunks t256; `verbatim_ok=8327/8327`. Model BAAI/bge-m3; dense và sparse vectors đã ghi.
- Segment artifacts: `data/chunks_cc_longchau/seg20004-cc-s23-vi-news-20261017-vi-news/`; vectors: `data/index/zh_full_segments/cc_longchau/seg20004-cc-s23-vi-news-20261017-vi-news/`.
- Provenance có 3.195 dòng/ID duy nhất, mỗi dòng có crawl-id, WARC date và capture timestamp. Kiểm tra 3.195/3.195 text SHA khớp docs; không thiếu provenance; chênh WARC date/capture timestamp lớn nhất 0 giây.
- SHA256 provenance parquet: `68dfb555fc3edda7bd2d84a59aa58408af502ea4a36f50b6c76fec9ab3fc36c2`.
- Chỉ ghi segment-specific CC manifest/index; không ghi `data/index/t256` chung. Đã gửi thông tin handoff cho S15.

## Chi phí và pha B

- AWS/S3: $0, không dùng S3. Common Crawl là nguồn HTTPS công khai; chi phí mạng/ISP cục bộ không đo được.
- Pha B không chạy trong phiên S23. Direct crawl VOV (~8.400 URL) do S24 phụ trách; phiên này không crawl trực tiếp host nào.