# S27 Pharmacity — báo cáo cuối

## Kết luận

Crawl Pharmacity đã đóng vĩnh viễn theo chỉ thị của Vu. Tổng phạm vi 5.337 URL; 979 doc đủ điều kiện đã được bàn giao thành batch cc-s27-pharmacity, gồm 7.503 chunk và vector dense đủ 3 lô. Không ghi shared index.

## Mapping và disposition URL

Mapping giữ nguyên doc_id: 5.333 URL beta có slug parse được, 0 slug lỗi; cộng 4 URL pharmacity.vn thiếu, tổng 5.337.

| Trạng thái loại trừ lẫn nhau | Số URL | Ghi chú |
|---|---:|---|
| Bàn giao | 979 | HTTP 200, body ≥200 codepoint, title match ≥50%; gồm 3 lô S15 |
| HTTP 200, low_title_match | 5 | Không đưa vào docs/chunks/vector |
| HTTP 200, body dưới 200 codepoint | 3 | Không đưa vào docs/chunks/vector |
| Redirect lệch slug | 22 | Loại khỏi batch, raw giữ lại |
| Unknown | 1 | doc_id 317850, HTTP 404 có challenge marker; không thử lại riêng |
| Dừng tại Cloudflare challenge | 2 | doc_id 318325 (HTTP 404) và 318709 (HTTP 307) |
| Chưa request | 4.325 | Pending khi crawl dừng vĩnh viễn |
| **Tổng** | **5.337** | 1.012 URL đã request + 4.325 chưa request |

Title thấp hơn 50% là 5/5.337 (0,094%), dưới ngưỡng 5% đã chốt. 22 redirect mismatch bị loại. doc_id 317850 được ghi unknown riêng theo chỉ thị; hai lần halt chính thức do challenge là 318325 và 318709. Không gửi request Pharmacity sau lần halt cuối lúc 2026-10-20 21:58:57 UTC.

## Common Crawl và lý do dừng

Đã đối chiếu 5.182 URL trên 13 Common Crawl index gần nhất; 5.182 không có capture, 0 usable text, 0 WARC tải. Crawl trực tiếp được cho phép theo từng lượt nhưng dừng ở challenge theo quy tắc; lần retry một lần đã dùng hết và kết thúc ở doc_id 318709. Thăm dò archive sau đó không lấy được nội dung mới.

## Provenance

979/979 provenance row có doc_id, original_url, mapped_url và fetched_at; doc_id duy nhất 979/979. Provenance còn lưu final_url, HTTP status, selector, body length, raw SHA-256, title-match metadata, fetch status và reason. Selector nội dung là #pmc-content-blog-detail. Doc_text được nhận khi body đạt ≥200 codepoint và title match đạt ≥50%.

## S15 chunks và vector handoff

| Lô | Docs | Chunks t256 | Verbatim | Dense vector | SHA-256 vector |
|---|---:|---:|---:|---|---|
| lot-0001 | 146 | 1.225 | 1.225/1.225 | 146×1.024 fp16 | 31ee1721a03951b1955f8039eefd846ebd963ed0012142fff601226d037d9fdc |
| lot-0002 | 466 | 3.564 | 3.564/3.564 | 466×1.024 fp16 | cc7de8be748631ab96f07bd63405ec20ad073b2e86124e244e03dcfc9522043a |
| lot-0003 | 367 | 2.714 | 2.714/2.714 | 367×1.024 fp16 | 5ef8d090fb9b6a9d3d0e8296adb6efec258b16581075a9767c63ad79d8b74abd |
| **Tổng** | **979** | **7.503** | **7.503/7.503** | **979×1.024 fp16** | — |

Chunk target 256 token, minimum 50 Vietnamese tokens. Encoder BAAI/bge-m3 qua r2ai.index.bge_m3.M3Encoder; input doc_text, max_len 512, L2-normalized CLS. Mỗi vector manifest ghi S26 gate done 887151; file SHA-256 đã kiểm khớp manifest. Không ghi shared index.

Artifact manifests và raw/provenance nằm trong ignored output tại out/runs/s27-pharmacity/. Chúng không được đưa vào Git. Report này và docs/PROGRESS.md là phần được commit.

## Trạng thái cuối

- Crawl: halted permanently; không còn retry/automation.
- Batch: cc-s27-pharmacity, 979 docs / 7.503 chunks.
- Vector handoff: đủ lot-0001, lot-0002, lot-0003.
- Shared index: không thay đổi.
