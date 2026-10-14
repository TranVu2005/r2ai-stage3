# Tuân thủ mô hình và nguồn dữ liệu

## Mô hình

| Model | License | Quy mô / nền | Tài liệu | Snapshot dùng trong repo |
|---|---|---|---|---|
| `BAAI/bge-m3` | MIT | XLM-RoBERTa-large, khoảng 0,57B tham số | [arXiv:2402.03216](https://arxiv.org/abs/2402.03216), nộp tháng 02/2024; [model card](https://huggingface.co/BAAI/bge-m3) | `5617a9f61b028005a4858fdac845db406aefb181` |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 0,6B tham số | [model card](https://huggingface.co/BAAI/bge-reranker-v2-m3); bản model được cập nhật tháng 06/2024 trong [BAAI collection](https://huggingface.co/collections/BAAI/bge) | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |

Snapshot hash được đối chiếu với cấu hình retrieval và thư mục snapshot trong Hugging Face cache tại thời điểm ghi tài liệu.

## LLM

Không dùng LLM. Pipeline truy hồi dùng `BAAI/bge-m3` và `BAAI/bge-reranker-v2-m3`; không sinh, dịch hoặc viết lại query/text bằng LLM.

## Nguồn text và provenance

- Nguồn text là live crawl và Common Crawl. `links_corpus.parquet` chỉ cung cấp `id` và `url`; chỉ lấy text từ URL thuộc corpus.
- Với Common Crawl, không dùng URL ngoài corpus. Ghi provenance cho từng bản ghi text: corpus `id`, URL corpus, loại nguồn, crawl ID (xem mục "Common Crawl: nhathuoclongchau.com.vn" bên dưới), WARC filename/offset/length, trạng thái/thời điểm lấy và SHA256 text sau extract.
- Không đưa text tổng hợp hoặc text từ nguồn ngoài các URL corpus vào submission.

## Common Crawl: nhathuoclongchau.com.vn (S3, 12–13/10/2026)

- Phạm vi: chỉ 79.598 URL longchau có trong `links_corpus.parquet`; 68.164 URL (85,64%) có bản HTTP 200 HTML trong 13 crawl. Mỗi URL lấy bản mới nhất (không sau 29/09/2026; mọi bản chụp thực tế ≤ 17/09/2026).
- Crawl ID dùng: `CC-MAIN-2026-04`, `-08`, `-12`, `-17`, `-21`, `-25`, `-30`, `-34`, `-39` và `CC-MAIN-2025-38`, `-43`, `-47`, `-51`. Số bản dùng từng crawl nằm trong `out/runs/cc-longchau/provenance.parquet` (cột `crawl`).
- Tải: HTTP Range tới `https://data.commoncrawl.org/<warc_filename>` (offset/length từ cột index CC), User-Agent `R2AI-Stage3-CCFetch/1.0` (không có contact vì chưa được cung cấp), 1 kết nối, ≤ 1,2 request/giây, backoff khi 503/429. Có thử `commoncrawl.s3.amazonaws.com` không xác thực: 403 AccessDenied ở 3 request, đã bỏ; không dùng credential AWS.
- Provenance mỗi doc: `doc_id`, URL, crawl, WARC filename/offset/length, `WARC-Date`, `WARC-Record-ID`, SHA256 bản ghi nén và body, extractor (`generic` = trafilatura 2.2.0 theo `configs/domains.yaml`), SHA256 text sau extract.
- Text CC chỉ vào pipeline qua shard riêng (`data/index/zh_full_segments/cc_longchau`, segment 20000); index vi `data/index/t256` không bị ghi.
