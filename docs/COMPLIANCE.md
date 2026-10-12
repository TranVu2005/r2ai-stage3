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
- Với Common Crawl, không dùng URL ngoài corpus. Ghi provenance cho từng bản ghi text: corpus `id`, URL corpus, loại nguồn, crawl ID (`[S3 điền]`), WARC filename/offset/length, trạng thái/thời điểm lấy và SHA256 text sau extract.
- Không đưa text tổng hợp hoặc text từ nguồn ngoài các URL corpus vào submission.
