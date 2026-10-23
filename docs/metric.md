# Metric R2AI Stage 3

Theo đặc tả người dùng cung cấp và scorer nguồn commit ae6b90d; chưa có bản scoreboard gốc trong repo. Code theo metric chunk v2 (BTC 23/10/2026), v1 giữ để so. Tokenizer BGE-M3 là giả định triển khai của scorer; snapshot local ghi trong configs/retrieval.yaml.

- Chuẩn hóa: Unicode NFKC → decode HTML entity → lowercase → gộp whitespace → strip. Giữ dấu câu.
- Doc P = |D ∩ G| / |D|; Doc R = |D ∩ G| / |G|, dùng doc_id của corpus.
- So chunk chỉ trong cùng doc_id (cùng query). **Từ 23/10/2026 (BTC đổi, v2, mặc định của scorer):** c khớp g khi LCS_tokens(c,g)/|g| >= 0,4 **và** LCS_tokens(c,g)/|c| >= 0,4. Trước đó (v1, `--chunk-metric v1`) chỉ cần LCS_tokens(c,g)/|g| >= 0,4. Hệ quả: chunk dài hơn 2,5 lần gold không bao giờ khớp, nên chunk nguyên body gần như luôn sai.
- Chunk P = số predicted chunk khớp ít nhất một reference / tổng predicted chunk. Chunk R = số reference chunk khớp ít nhất một predicted / tổng reference chunk.
- Scorer không gộp chunk trùng/gần trùng ở ngưỡng 0,8. Builder có dedupe riêng để giữ hành vi submission; không dùng dedupe builder để diễn giải metric.
- F2 = 5PR/(4P+R). Macro mặc định theo query có nhãn dương ở cấp tương ứng; Final = (macro Doc F2 + macro Chunk F2)/2. Quy ước tập rỗng theo src/r2ai/eval/scorer.py và test hiện có.
- JSON đủ 1.200 query, mỗi id một lần: relevant_docs là list int; relevant_chunks chứa doc_id, chunk_text nguyên văn; chunk_order:int>=0 tùy chọn theo schema scorer/validator chính. ZIP chỉ chứa một JSON ở root.
- scripts/validate_submission.py là validator bổ sung dùng để replay builder; nó giữ schema builder hai trường doc_id/chunk_text. Validator chính `python -m r2ai.eval.validate` hỗ trợ chunk_order.
- Public 31/10/2026; private 04/11/2026 tối đa 5 lượt; kết quả 11/11/2026, theo mốc người dùng.
