# S26 P-A — Nhận dạng retriever BTC qua pool zh, và kiểm tra pool vi

2026-10-19 (UTC+7) · Claude Opus 5.5 · nhánh `feat/s26-gold-structure`. Tiếp nối `reports/s26-gold-structure.md` mục 2. Không nộp, không ghi `data/index`. Script ở `scripts/s26/`; output trung gian ở `out/runs/s26-gold-structure/fingerprint/` (ignored).

en đã đóng: Vu báo EN-only-Q (top-20 Qwen) = 0 trên 24.000 doc. Nếu en v2 cũng được chọn theo query thì đó là hard negative, không phải pool gold. Report này không kiểm thêm en.

## 0. Tóm tắt

- **zh: retriever của BTC gần như chắc chắn là BGE-M3 dense (hoặc model rất giống) nhúng nguyên doc (doc_text: title một lần rồi body), cắt ở 512 token, với query tiếng Việt gốc.** Biến thể này đạt AUC −min-rank **0,924** (max-chunk t256 chỉ 0,833), AUC trong từng domain 0,91. AUC đạt đỉnh ở 512 rồi giảm ở 1024/2048; 512 là `max_length` mặc định của BGE-M3.
- Query dịch sang tiếng Trung, BM25 zh, sparse và e5-small đều kém hơn. Qwen3-Embedding-4B (NF4, doc_text 512, mẫu con 10k) chỉ đạt AUC −min-rank 0,757, thấp hơn hẳn BGE-M3 trên cùng mẫu (0,915) ⇒ không phải retriever của BTC.
- Trần đo được gần 1: decile min-rank thấp nhất có 0/2.939 doc được giữ ⇒ BTC không lấp ngẫu nhiên; phần AUC còn thiếu (0,924 → 1) nhiều khả năng do khác bản extract text (BTC nhúng text của họ, mình nhúng text extract của mình).
- **vi: không có dấu hiệu pool.** Trên 7 site chỉ nói về y tế, bài trong corpus không gần query hơn bài cùng site ngoài corpus (AUC 0,52). Riêng Vinmec ngược chiều (AUC 0,44): BTC đã loại bớt bài giống query (trang nguồn).
- **P1-docrank (P-A2) LB 0,2937; P-A3-nc 0,3205** (doc-level thuần, n_c chunk như P1). P-A4 (top-K_q trên 887k, body canonical cho doc mới) dựng 4 bản: P-A4, K90, K110, nc120; validate 0 lỗi, chưa nộp (mục 5d).
- Đề xuất: probe P-A2 xếp hạng lại vi+zh theo BGE-M3 dense doc-level doc_text 512 (mục 5), để kiểm giả thuyết "gold sinh từ cùng retriever".

## 1. Dữ liệu và phép đo

- Mẫu đối chứng: 2% zh của corpus v1, crawl trước khi có v2. Index `data/index/zh_full_segments/seg00001-sample-old-zh_sample_t256`, doc `data/chunks_zh_sample/docs.parquet`: 29.385 doc, trong đó 2.413 được giữ ở v2 (kept, 8,21%) và 26.972 bị loại.
- Với mỗi biến thể retriever: điểm doc trên 1.200 query. "AUC best" = AUC của điểm cao nhất qua các query. "AUC −min-rank" = AUC của hạng tốt nhất của doc qua các query (trong mẫu); đây là phép đo hợp với cách chọn top-N mỗi query. "R-prec" = tỷ lệ kept trong 2.413 doc có min-rank tốt nhất.
- Query vector: `r2ai.index.bge_m3.M3Encoder` (CLS, L2, fp16), max_len 512.

## 2. Trần AUC — BTC có lấp ngẫu nhiên không

Nếu một phần doc kept được lấp ngẫu nhiên (không theo điểm), chúng rải đều mọi bin điểm. Do đó tỷ lệ kept ở bin thấp nhất chặn trên tỷ lệ lấp.

| Điểm (BGE-M3 dense max-chunk) | Bin thấp nhất | Doc kept trong bin | Tỷ lệ lấp ngẫu nhiên tối đa |
|---|---|---:|---:|
| best, 10 bin | 2.939 doc | 6 | 2,5% |
| best, 20 bin | 1.470 doc | 2 | 1,7% |
| min-rank, 10/20/50 bin | 2.939 / 1.470 / 588 doc | 0 | 0% |

⇒ Doc kept được chọn hoàn toàn theo điểm của một retriever. Một retriever khớp tuyệt đối với bản text của BTC có thể đạt AUC gần 1.

## 3. Fingerprint zh

| Biến thể | AUC best | AUC −min-rank | Kept khi min-rank = 1 | R-prec |
|---|---:|---:|---:|---:|
| BGE-M3 sparse max-chunk, query vi | 0,559 | 0,555 | 0,102 | |
| BM25 jieba title+body, query zh | 0,617 | 0,695 | 0,203 | |
| multilingual-e5-small title lặp + doc 512 | 0,674 | 0,690 | 0,266 | |
| hybrid z 0,7 dense + 0,3 sparse, query vi | 0,618 | 0,701 | 0,178 | |
| BGE-M3 sparse max-chunk, query zh | 0,679 | 0,704 | 0,205 | |
| BGE-M3 dense max-chunk, query zh | 0,736 | 0,775 | 0,330 | |
| BGE-M3 dense title only | 0,757 | 0,782 | 0,460 | 0,351 |
| BGE-M3 dense chunk đầu | 0,757 | 0,783 | 0,461 | |
| BGE-M3 dense title lặp + doc 512, query zh | 0,769 | 0,815 | 0,371 | |
| BGE-M3 dense max-chunk t256 (cách S26 cũ) | 0,796 | 0,833 | 0,433 | |
| BGE-M3 dense mean-chunk | 0,795 | 0,835 | 0,521 | |
| BGE-M3 dense mean top-2 chunk | 0,798 | 0,853 | 0,462 | |
| BGE-M3 dense title lặp + doc 128 | 0,801 | 0,878 | 0,480 | 0,416 |
| BGE-M3 dense title lặp + doc 2048 | 0,822 | 0,904 | 0,514 | 0,450 |
| BGE-M3 dense title lặp + doc 256 | 0,818 | 0,904 | 0,531 | 0,451 |
| BGE-M3 dense title lặp + doc 1024 | 0,823 | 0,906 | 0,516 | 0,450 |
| BGE-M3 dense title lặp + doc 512 | 0,825 | 0,913 | 0,528 | 0,462 |
| **BGE-M3 dense doc_text nguyên 512** | **0,833** | **0,924** | **0,548** | **0,493** |
| Qwen3-Embedding-4B (NF4) doc_text 512, query vi, mẫu con 10k* | 0,756 | 0,757 | 0,545 | |

\* Mẫu con 10.000 doc (đủ 2.413 kept + 7.587 dropped ngẫu nhiên, seed 42) vì bản 4B 4-bit chạy chậm trên RTX 3050 (12.564 s cho 10k doc). Mốc BGE-M3 doc_text 512 trên đúng mẫu con: AUC best 0,828, AUC −min-rank 0,915, kept khi min-rank = 1 là 0,771 (mẫu nhỏ hơn nên min-rank không so trực tiếp với dòng 29k). Query Qwen3 theo prompt model-card "Instruct: Given a web search query, retrieve relevant passages that answer the query".

Ghi chú: biến thể "per-query z-score" chỉ đổi AUC best (0,814), không đổi min-rank. Bản dịch zh do `qwen2.5:7b` (Ollama, temperature 0, seed 42) tạo; 23/1.200 bản còn sót chữ Việt.

**BGE-M3 doc_text 512 theo domain (AUC −min-rank):** cnkang 0,906 (759 kept), 120ask 0,921 (1.160), a-hospital 0,929, zhongyibaodian 0,933, youlai 0,959, care.39 0,948, msdmanuals 0,951; ngoại lệ jbk.39 0,729, test.pmphai 0,794, jb39 0,522 (mẫu nhỏ). Trung bình theo số kept: 0,909 ⇒ không phải hạn ngạch theo domain.

**Profile min-rank (doc_text 512):** min-rank ≤1: 940 doc, 54,6% kept, phủ 21,3% kept; ≤3: 48,8% / 48,3%; ≤10: 34,9% / 84,7%; ≤20: 25,3% / 95,2%.

**Diễn giải.**
1. Đơn vị nhúng là cả doc, không phải chunk: doc-level tăng từ 0,833 lên 0,913–0,924.
2. Độ dài 512 token là tối ưu (128: 0,878; 256: 0,904; 512: 0,913; 1024: 0,906; 2048: 0,904), khớp max_length mặc định của BGE-M3/FlagEmbedding.
3. Lặp title (ghép title thêm một lần trước doc_text vốn đã có title) làm giảm AUC (0,913 so với 0,924): BTC nhúng doc nguyên, không nhấn title. Lưu ý đính chính: bản trước của report gọi biến thể tốt nhất là "chỉ body"; thực tế `docs.parquet.doc_text` đã chứa title ở đầu, các biến thể "title + body" là title lặp hai lần.
4. Query gốc tiếng Việt tốt hơn query dịch (0,913 so với 0,815): BTC không dịch query.
5. Phần còn thiếu tới 1 nhiều khả năng do khác text đầu vào (boilerplate, cách cắt, extractor).

## 4. Kiểm tra pool vi

vi v2 = vi v1 (740.598 URL), không có tập bị loại. Đối chứng thay thế: link nội bộ trong HTML thô của trang corpus (`data/raw_vi/<host>`, ≤1.500 trang/host) trỏ tới bài **cùng site**. Gắn nhãn trong/ngoài corpus theo URL chuẩn hoá. Cả hai nhóm được biểu diễn bằng cùng loại text (anchor text dạng tiêu đề, ≥6 từ, ≥25 ký tự), nhúng BGE-M3 dense. Giữ 24 host có ≥100 link mỗi nhóm: 99.496 link, 45.344 trong corpus.

| Nhóm host | Số host | AUC best (trọng số theo số link trong corpus) | AUC −min-rank |
|---|---:|---:|---:|
| Chỉ nói về y tế (vinmec, medlatec, hellobacsi, tiemchunglongchau, suckhoedoisong, suckhoeviet, phunusuckhoe) | 7 | 0,531 | 0,518 |
| Báo tổng hợp | 17 | 0,674 | 0,655 |

- Site chỉ nói về y tế: AUC ≈0,5 ⇒ bài trong corpus không gần query hơn bài cùng site ngoài corpus. vi **không** được chọn theo query.
- Báo tổng hợp: AUC 0,52–0,84 (baoquangtri 0,845, baohaiphong 0,790). Chênh lệch này giải thích được bằng việc BTC lấy chuyên mục sức khoẻ: link ngoài corpus phần lớn là tin không y tế.
- Vinmec ngược chiều: AUC 0,444 (7.002 link, 1.887 trong corpus). Trong các link cos ≥0,8 với query: 4 trong corpus, 47 ngoài (kỳ vọng khoảng 14 nếu ngẫu nhiên theo tỷ lệ 27%). ⇒ BTC đã loại phần lớn bài Vinmec giống query (trang nguồn), khớp gold-check cũ; 18 trang nguồn còn sót (S26 mục 4) là phần lọt. Medlatec 0,484, nhưng link cos cao ngoài corpus chủ yếu là trang tag, không phải bài viết.
- Hạn chế: anchor text chỉ gần đúng tiêu đề; trang corpus thường link tới bài liên quan nên nhóm ngoài corpus đã bị lệch về phía gần chủ đề (làm AUC thấp đi, tức kết luận "không pool" là thận trọng về phía có pool).

## 5. Đề xuất

**P-A2 (giả thuyết: gold sinh từ cùng retriever).** Nếu BTC dùng BGE-M3 doc-level để chọn pool zh, nhiều khả năng họ cũng dùng nó (kèm LLM judge hoặc không) để tạo gold cho cả vi và zh. Probe:
1. Nhúng BGE-M3 dense doc_text 512 cho doc ứng viên. Rẻ: chỉ các doc trong pool ứng viên hiện có của P1 (ví dụ top-300/query, số doc unique cần đo). Đầy đủ: toàn bộ 887.151 doc canonical, ước khoảng 12 giờ trên RTX 3050 (đo 29.385 doc ở 512 token hết khoảng 33 phút khi GPU dùng chung).
2. Xếp hạng vi+zh mỗi query theo cosine doc-level, chọn K doc (giữ K và chunk = nguyên body như P1), so LB với P1.
3. Phiên bản an toàn hơn: trộn RRF giữa thứ tự P1 và thứ tự doc-level.

Kỳ vọng: chưa ước lượng được. Nếu gold = top-k của retriever BTC thì precision có thể tăng mạnh; nếu gold do LLM judge độc lập với pool thì chỉ thay đổi nhỏ. Một lượt nộp sẽ phân xử.

**Offline trước khi nộp (0 lượt):** đo trùng lặp giữa top-50 doc-level và top-50 P1 trên vi; nếu gần như trùng hoàn toàn thì P-A2 không có thông tin mới.

## 5b. P-A2 đã chạy (19/10, chưa nộp)

**Nhúng.** 125.605 doc canonical ứng với 127.873 ID của P1-vi+10 (vi 111.081, zh 14.524; 2.113 ID không map được về canonical), BGE-M3 dense `doc_text` nguyên văn, max_len 512 (đúng biến thể tốt nhất ở mục 3). Chạy sau khi job vi2 full-score của session khác thoát (10:37), mất 3.229 s. Output `out/runs/s26-gold-structure/pa2/pa2_doc_dense.npz`.

**So thứ tự doc-level với P1 (trên pool doc của P1, 99,45% doc có embedding).**

| | Doc-level thật | Embedding ngẫu nhiên (mốc nền) |
|---|---:|---:|
| Trùng top-50 (trung bình / P10 / median / P90) | 23,4 / 15 / 23 / 32 | 15,3 / 9 / 14 / 24 |
| Tỷ lệ doc nhận chunk trùng P1 | 81,2% | 78,4% |
| Hạng doc-level của doc top-1 P1 (median) | 9 | 82 |

Thứ tự doc-level tương quan với P1 nhưng khác rõ: khoảng một nửa top-50 khác nhau. Tỷ lệ doc nhận chunk trùng cao ở cả hai cột vì chỉ doc đã có text chunk trong P1 mới nhận chunk được.

**`P1-docrank.zip` (chưa nộp).** `out/runs/s26-gold-structure/pa2/P1-docrank.zip`: 80.706.521 B, SHA256 `b4248359cb103251f98776e025c7f12b38223cae0ea0e22340d2479c2d75ce31`; JSON 848.557.795 B, SHA256 `60a9e96297d0c2d5871087a9d740e0133c486b3d66de7f72310dd8f5d825dd12`; LZMA dict 256 MiB props `5d00000010`; dưới trần 102.600.000 B.
- `relevant_docs` giữ nguyên từng query (cùng ID, cùng thứ tự) ⇒ Doc P/R phải bằng P1-vi+10 (0,1521/0,4499).
- Số chunk mỗi query giữ nguyên (126.228). Chunk dồn cho doc có cosine doc-level cao nhất thay vì theo thứ tự P1; một chunk mỗi doc_id.
- Text chunk lấy lại từng byte từ chunk P1-vi+10 đã ship cho cùng doc_id (canonical `doc_text` chỉ trùng byte 2.009/82.451 nên không dùng). Doc không có text chunk trong P1 bị bỏ qua.
- Kiểm: 0 doc đổi, 0 cặp (doc_id, text) ngoài tập P1, 0 chunk ngoài relevant_docs, 0 doc_id trùng chunk. Validator native chưa chạy: adapter của P1 (`compose_matrix.py`) gắn với pin riêng của P1; mọi chunk đều trùng nguyên văn một chunk P1 đã qua native.

**Đọc kết quả khi nộp.** Chỉ Chunk P/R thay đổi. Nếu Chunk P tăng rõ so với 0,1936 ⇒ thứ tự doc-level chính xác hơn thứ tự P1, ủng hộ giả thuyết gold gắn với retriever của BTC ⇒ bước tiếp là nhúng toàn bộ 887k doc và xếp hạng lại cả tập doc. Nếu giảm hoặc không đổi ⇒ retriever BTC chỉ dùng để chọn pool zh, gold không theo nó.

## 5c. P-A3 / P-A3r / P-A4: chọn doc theo doc-level (19/10, chưa nộp)

**LB P1-docrank (Vu báo):** 0,2937. Chunk F2 0,2773 → 0,2978 (P 0,1936 → 0,2100, R 0,3644 → 0,3883), Doc giữ nguyên. Thứ tự BGE-M3 doc-level (doc_text 512) dự đoán gold tốt hơn thứ tự P1, nên các bản sau dùng nó để chọn doc.

**Quy tắc chung (`pa3_build.py … nc`).** Hai bản chỉ khác nhau ở cách chọn doc.
- K_q = `len(relevant_docs)` của cùng query trong P1-vi+10 (trung bình 180,7; tổng 216.866).
- **Không gate τ** (zh chunk-level) và không quota ngôn ngữ. Cosine doc-level quyết định cho cả vi lẫn zh.
- Một ID corpus cho mỗi doc canonical, theo thứ tự ưu tiên: ID P1 đã dùng trong cùng query; ID P1 đã dùng ở query khác; `doc_id` canonical nếu nằm trong nhóm alias; cuối cùng là alias đầu tiên. 2.113 ID của P1 không map được về canonical (không có vector) nên không được xếp hạng.
- **Số chunk = n_c của P1-vi+10** (cùng query; tổng 126.228, trung bình 105,2). Chunk dồn cho n_c doc đứng đầu theo thứ tự của biến thể, 1 chunk/doc. Text chunk chọn theo thứ tự:
  1. Chunk P1 đã ship cho ID đó (nguyên body như P1), ưu tiên cùng query (`p1_row`), không có thì lấy ở query khác (`p1_any`), lấy nguyên byte.
  2. Doc không có chunk P1: chunk `chunks_t256` của doc đó có cosine dense với query cao nhất (`t256`).

**Validator (`validate_pa3.py`).**
- Format, trần dung lượng và dict 256 MiB.
- Đủ 1.200 id. Doc nằm trong `links_corpus_vi_en_zh` và không trùng. Chunk thuộc relevant_docs, mỗi doc_id tối đa 1 chunk.
- Verbatim: hoặc cặp (doc_id, text) đã có trong P1 (đã qua native), hoặc text là substring của `doc_text` canonical (S25 đã xác minh). Chưa chạy validator native.

| | P1-vi+10 | P-A3-nc | P-A3r-nc |
|---|---:|---:|---:|
| Chọn doc | pipeline P1 | top-K_q cosine trên 125.605 doc của P1 | RRF(k=60) thứ tự P1 × thứ tự doc-level |
| Doc khác P1 / query (TB; P10/median/P90) | 0 | 94,5 (57/93/135) | 33,3 (17/31/53) |
| zh / vi mỗi query | 14,7 / 163,1 | 19,0 / 161,7 | 14,8 / 165,9 |
| Tỷ lệ zh | 8,25% | 10,51% | 8,22% |
| Chunk / query | 105,2 | 105,2 | 105,2 |
| Nguồn chunk p1_row / p1_any / t256 | | 55.968 / 41.956 / 28.304 | 92.684 / 15.846 / 17.698 |
| ZIP (B) | 80.706.521 (P1-docrank) | **55.402.549** | **71.799.786** |
| Validator | | 0 lỗi | 0 lỗi |

Ghi chú bảng: cột P1-vi+10 đếm doc canonical của P1. zh/vi đếm theo `lang` canonical. Mốc ZIP của P1-vi+10 lấy từ P1-docrank (cùng doc, cùng chunk).

**ZIP (chưa nộp), LZMA props `5d00000010`, dưới trần 102.600.000 B.**
- `out/runs/s26-gold-structure/pa3/P-A3-nc.zip`: 55.402.549 B, SHA256 `010aedc527067d38492c42faf9dd1a89c01133bee79436013a512a50fe8ce4e8`; JSON 608.802.469 B, SHA256 `72e06b202b0052b08abbd2eca96cf03628bc29afa7425b16977f2c940d3018e6`. Validator: 97.924 cặp P1 + 28.304 substring canonical.
- `out/runs/s26-gold-structure/pa3/P-A3r-nc.zip`: 71.799.786 B, SHA256 `592f8a02291ab98f4470de5ae5846cbfaeb6fc13516e1b5a4a092e7cdd1a27ed`; JSON 735.740.305 B, SHA256 `1a5b7e070e3c5105231847883f9bcb021068caa4d4150fa39b4c74e3d4082319`. Validator: 108.530 cặp P1 + 17.698 substring canonical.

**Bản cũ (1 chunk cho mỗi doc chọn, 180,7 chunk/query), thay bằng hai bản trên theo quyết định của Vu.** P-A3 86.264.388 B (SHA `19391ad9…516c`, 0 lỗi) và P-A3r 109.611.284 B (SHA `acb2513a…a140`, vượt trần). Log và ZIP P-A3 chuyển sang `pa3/kq-chunks/`. `pa3/P-A3r.zip` vẫn ở chỗ cũ vì đang bị một process khác mở.

**LB (Vu báo):** P-A3-nc **0,3205** (Doc 0,3332 / Chunk 0,3078), tốt nhất. P-A3r-nc 0,3191. P-A3 (1 chunk cho mỗi doc chọn, 180,7 chunk/query) 0,3130. Quyết định: dùng doc-level thuần với quy tắc n_c cho P-A4.

## 5d. P-A4: top-K_q doc-level trên toàn bộ 887.151 doc (19/10, chưa nộp)

**Nhúng.** `full_embed.py` nhúng 761.546 doc trong 19.102 s (39,9 doc/s) và tái sử dụng 125.605 vector của pa2, ra `full/full_doc_dense.npz` (887.151 × 1024 fp16). Thứ tự nhúng: zh trước, sau đó vi theo host tamanhhospital, tiemchunglongchau, medlatec, vinmec, hellobacsi, rồi các host còn lại theo tỷ lệ doc nằm trong P1-vi+10 giảm dần. Checkpoint mỗi 5.000 doc.

**Chunk của doc ngoài P1 = nguyên body dựng lại từ canonical** (`body_check.py`, `pa3_build.py … new=body`). Quy tắc:
- vi có field `answer`: lấy `doc_text[min char_start : max char_end]` của các chunk answer.
- vi còn lại: lấy span các chunk `body` theo cùng cách.
- zh: lấy nguyên `doc_text`.

Kiểm chứng trên chunk P1 đã ship:
- 200 mẫu (seed 26): khớp byte **198/200 = 99,0%** (đạt ngưỡng ≥95%). 2 ca lệch đều là Tâm Anh: chunk P1 có thêm đuôi boilerplate `Chia sẻ: / Chủ đề: / BÀI VIẾT CÙNG CHỦ ĐỀ ›` mà canonical đã bỏ. Bản dựng lại vẫn là chuỗi con của chunk P1.
- 5.000 mẫu (seed 27): khớp 4.874/4.992 = 97,6%. Trong 118 ca lệch, 117 là chuỗi con của chunk P1 (Tâm Anh 115, Vinmec 2); 1 ca ask.39.net khác thật.

Doc có chunk P1 vẫn dùng đúng chunk đó, nguyên byte. Nếu ZIP vượt trần, body của doc mới sẽ được cắt còn N ký tự đầu (N lớn nhất vừa trần, tìm bằng secant trên số byte JSON). Cả 4 bản đều dưới trần nên không cắt; nhánh cắt N chưa được chạy thật.

**Doc alias và doc thin.**
- 4.327 doc canonical là alias (rule a). Chúng không có chunk riêng, nhưng doc sở hữu nhóm alias trong `chunks_t256` có cùng text. 2.626 doc sở hữu không có trong `docs.parquet`; offset chunk của chúng vẫn khớp text của doc alias (1.589/1.589 chunk). Body lấy text của chính doc được chọn và span theo chunk của doc sở hữu.
- Validator kiểm chunk theo text canonical của chính corpus ID; nếu ID không có trong `docs.parquet` thì dùng text của doc sở hữu nhóm.
- Chỉ doc `thin` (7.039 doc, text rất ngắn) không có chunk. Theo quyết định của Vu, doc thin vẫn nằm trong relevant_docs và chunk của nó chuyển cho doc kế tiếp.
- Bản P-A4 đầu tiên (71.362.706 B) coi doc alias là không có chunk. Bản đó đã thay thế và chuyển sang `pa4/_v1-alias-bug/`.

**Kiểm tra hồi quy.** Sau mọi sửa đổi, `pa3_build.py … P-A3-nc pa3 nc` (mặc định t256) vẫn dựng lại đúng JSON SHA256 `72e06b20…18e6`.

| | P-A3-nc (mốc) | P-A4 | P-A4-K90 | P-A4-K110 | P-A4-nc120 |
|---|---:|---:|---:|---:|---:|
| Doc / query | 180,7 | 180,7 | 162,6 | 198,8 | 180,7 |
| Doc mới so với P-A3-nc / query (TB; P10/median/P90) | 0 | 69,8 (33/65/115) | 60,4 (28/56/101) | 79,4 (39/74/128) | 69,8 (33/65/115) |
| vi / zh mỗi query | 161,7 / 19,0 | 140,0 / 40,7 | 127,0 / 35,6 | 152,9 / 45,9 | 140,0 / 40,7 |
| Tỷ lệ zh | 10,5% | 22,5% | 21,9% | 23,1% | 22,5% |
| Lượt doc không có chunk (= thin) | 0 | 1.932 | 1.659 | 2.207 | 1.932 |
| Chunk (mục tiêu) | 126.228 | 126.133 (126.228) | 126.133 (126.228) | 126.133 (126.228) | 151.339 (151.485) |
| Nguồn chunk P1 cùng query / P1 query khác / body | | 46.294 / 25.186 / 54.653 | như P-A4 | như P-A4 | 50.168 / 31.601 / 69.570 |
| ZIP (B) | 55.402.549 | **71.055.364** | **70.962.833** | **71.139.849** | **84.113.963** |
| Cắt N | | không | không | không | không |
| Validator | 0 lỗi | 0 lỗi | 0 lỗi | 0 lỗi | 0 lỗi |

Ghi chú bảng:
- Doc thin của P-A4 gồm 1.060 doc khác nhau (677 zh, 383 vi), phần lớn là hỏi đáp ngắn trên ask.39.net.
- Thiếu chunk chỉ xảy ra ở 4 query hỏi chỉ số xét nghiệm (401, 556, 714, 1009). Ở đó doc thin chiếm 70/116, 65/91, 86/146 và 77/99 doc của top-K_q, nên không còn đủ doc mang được chunk. Tăng K không bù được vì số doc thin cũng tăng theo.
- Đổi K_q chỉ thêm hoặc bớt doc ở đuôi danh sách, nên chunk của K90 và K110 giống hệt P-A4.

**ZIP (chưa nộp), `out/runs/s26-gold-structure/pa4/`, LZMA props `5d00000010`:**
- `P-A4.zip` SHA256 `f84e1cccb899463299656e87e27657a66ed8eb3f261e63427abbbd6d6de827fb`
- `P-A4-K90.zip` SHA256 `bc17b562ba664fd5653bb10b19d3499dbfce5e90a0392450ed1c63814f11dd17`
- `P-A4-K110.zip` SHA256 `24b6b8bf60cbe4ec22c835a6a084003f5b17d4ba506621b1170a4c0185864b2b`
- `P-A4-nc120.zip` SHA256 `d2fac345f6ef8d034ca104ab17862125e85681b2a28ba6e16085d73dfa51526f`

## 5e. Gộp S27 pharmacity vào pool doc-level (chuẩn bị, chờ điểm P-A4)

Theo Vu: bản tiếp theo dựng trên bản P-A4 tốt nhất (sau khi có điểm) và gộp 612 doc pharmacity S27 (lô 1: 146, lô 2: 466) vào pool doc-level.

**Kiểm tương thích.**
- Cả 612 ID đều có trong `links_corpus_vi_en_zh` và không trùng doc canonical nào.
- Vector doc-level (`out/runs/s27-pharmacity/vectors/handoff/<lot>/dense_doc_level.npz`) dùng cùng encoder và cùng cấu hình như `full_embed.py`: BGE-M3 CLS L2 fp16, `doc_text` 4.096 ký tự đầu, max_len 512.
- `doc_text` có cấu trúc title + `

` + body. Span của các chunk `body` là đuôi của `doc_text` ở cả 612 doc, nên chunk = nguyên body theo cùng quy tắc với canonical.
- Validator nhận thêm thư mục handoff làm nguồn verbatim.

**Chạy thử (`dry`, không ghi ZIP) trên cấu hình P-A4** (top-K_q trên 887.763 doc, n_c chunk):

| | Giá trị |
|---|---:|
| Lượt doc pharmacity lọt top-K_q (tổng) | 233 |
| Query có ≥1 doc pharmacity | 184 / 1.200 |
| Doc pharmacity khác nhau được chọn | 146 |
| Selection rate (146 / 612) | **23,9%** |
| Lượt doc pharmacity nhận chunk (nằm trong n_c đầu) | 134 |
| Doc mới so với P-A3-nc / query | 69,86 (P-A4: 69,76) |

Số liệu trên dùng K_q ×1,0. Khi có điểm, bản thật sẽ dựng với hệ số K/n_c của bản P-A4 tốt nhất nên số có thể lệch nhẹ. Lệnh: `pa3_build.py … <name> pa3 nc new=body [k=x] [ncx=x] extra=out/runs/s27-pharmacity/handoff/cc-s27-pharmacity extravec=out/runs/s27-pharmacity/vectors/handoff`; validate: `validate_pa3.py <zip> out/runs/s27-pharmacity/handoff/cc-s27-pharmacity`. Hồi quy P-A3-nc sau khi sửa: JSON `72e06b20…18e6`.

## 5f. P-A5: cách chọn số doc mỗi query, trên nền K90 (20/10, chưa nộp)

**LB (Vu báo):** P-A4 0,3747; **K90 0,3750** (Doc 0,3811 / Chunk 0,3689), tốt nhất; K110 0,3738. K90 được chọn làm nền. Bản nc120 do Vu nộp tiếp.

**Cấu hình.** Cả ba bản đều xếp theo doc-level trên 887.151 doc và dùng chunk như P-A4: n_c của P1, nguyên body canonical cho doc mới, doc thin vẫn giữ trong relevant_docs. Không gộp pharmacity. Mục tiêu trung bình doc/query là 162,6225, bằng trung bình K90 (tổng 195.147). Lệnh: `pa3_build.py … pa3 nc new=body sel={ku|cos|rel} tmean=162.6225` (`pa5_run.sh`).
- **P-A5-Ku:** 163 doc cho mọi query.
- **P-A5-cos:** chọn doc có cosine ≥ θ, một θ chung cho mọi query. θ = **0,6400**.
- **P-A5-rel:** chọn doc có cosine ≥ cos(top-1) − δ, một δ chung. δ = **0,07902**.
- θ và δ tìm bằng bisection trên top-4.000 doc mỗi query; không query nào chạm trần 4.000.

| | K90 (nền) | P-A5-Ku | P-A5-cos | P-A5-rel |
|---|---:|---:|---:|---:|
| Doc / query: min / P10 / median / P90 / max | — | 163 / 163 / 163 / 163 / 163 | 0 / 2 / 53,5 / 433 / 3.610 | 1 / 11 / 82 / 406 / 2.224 |
| Query có 0 doc | 0 | 0 | **73** | 0 |
| Doc có trong bản này nhưng không có trong K90 / query | 0 | 18,3 | 78,4 | 75,5 |
| Doc có trong K90 nhưng không có trong bản này / query | 0 | 17,9 | 78,4 | 75,5 |
| vi / zh mỗi query | 127,0 / 35,6 | 126,2 / 36,8 | 123,2 / 39,4 | 115,6 / 47,1 |
| Tỷ lệ zh | 21,9% | 22,6% | 24,2% | 28,9% |
| Chunk (mục tiêu) | 126.133 (126.228) | 126.133 (126.228) | 76.462 (76.602) | 80.757 (80.918) |
| Lượt doc thin (không chunk) | 1.659 | 2.429 | 1.019 | 9.163 |
| ZIP (B) | 70.962.833 | **70.968.154** | **44.579.326** | **46.840.671** |
| Validator | 0 lỗi | 0 lỗi | 0 lỗi | 0 lỗi |

Số doc khác nhau giữa từng cặp bản, trung bình mỗi query (doc có ở hàng nhưng không có ở cột):

| | K90 | Ku | cos | rel |
|---|---:|---:|---:|---:|
| K90 | 0 | 17,9 | 78,4 | 75,5 |
| Ku | 18,3 | 0 | 87,6 | 73,9 |
| cos | 78,4 | 87,2 | 0 | 102,3 |
| rel | 75,5 | 73,5 | 102,3 | 0 |

**Đọc kết quả.**
- Với θ hoặc δ chung, số doc mỗi query rất lệch: một nửa số query nhận dưới 54 (cos) hoặc dưới 82 (rel) doc, trong khi một số query nhận hàng nghìn doc.
- Số chunk giảm còn khoảng 61–64% của K90, vì nhiều query có ít doc hơn n_c.
- P-A5-cos có 73 query với relevant_docs rỗng. Validator chỉ cảnh báo trường hợp này, không coi là lỗi; điểm của các query đó chắc chắn bằng 0.
- P-A5-rel làm tỷ lệ zh tăng lên 28,9% và số lượt doc thin lên 9.163, vì các query có top-1 thấp kéo theo nhiều doc ngắn.
- P-A5-Ku chỉ khác K90 khoảng 18 doc mỗi query.

**Đính chính đơn vị đếm "doc mới".** Cột "doc mới so với mốc" trong `*.build.json` đếm doc alias khác với doc sở hữu nhóm, nên hơi cao. Ở mục 5d, đếm lại với cùng ánh xạ ở cả hai phía cho kết quả: P-A4 69,6 (thay vì 69,8), K90 60,3 (thay vì 60,4), K110 79,2 (thay vì 79,4), nc120 69,6. Mục này đã dùng cách đếm đúng.

**ZIP (chưa nộp), `out/runs/s26-gold-structure/pa5/`, LZMA props `5d00000010`:**
- `P-A5-Ku.zip` SHA256 `3895980fd46f57c0716e5c2008c511c59e704438d3d9497c5c480c800f8284fb`
- `P-A5-cos.zip` SHA256 `2b9ddac0ee53ea3874588c826015931401c3185cb9b4d6373b3119b024bfaf93`
- `P-A5-rel.zip` SHA256 `63fdb3d4dca3b2230c5d37cd48da48089478c0ec79a38988c046d3b02439b111`

## 6. Tái tạo

`pool_ceiling.py` (mục 2); `pool_fingerprint.py`, `pool_fingerprint_cpu.py`, `pool_fingerprint_e5.py`, `pool_fingerprint_zhq.py`, `pool_fingerprint_doclen.py`, `pool_fingerprint_qwen3.py` (mục 3, chạy bằng `.venv` khi cần bitsandbytes/jieba/rank_bm25); `pool_body512_domain.py` (theo domain; tên file giữ chữ body512 nhưng là doc_text nguyên); `translate_zh.py` (Ollama, chia shard `k/3`); `vi_links.py`, `vi_pool_test.py` (mục 4). P-A2: `pa2_texts.py` (map ID P1 → canonical, text), `pa2_embed.py` (nhúng, checkpoint 5.000 doc), `pa2_build.py` (so thứ tự + ZIP). P-A3/P-A4: `full_embed.py <out> <pa2_dir>` (nhúng toàn bộ canonical, resume theo khối), `pa3_build.py <out> <scratch> <doc_vectors.npz> <name> {pa3|pa3r} [nc]`, `validate_pa3.py <zip>`. P-A4: `body_check.py <pa2_dir> [N] [seed]` (dựng lại chunk P1 từ canonical), `pa4_run.sh <name> nc new=body [k=x] [ncx=x]` (`S26_SCRATCH` = thư mục `q_dense.npy`). P-A5: `pa5_run.sh` (`sel=ku|cos|rel tmean=…`). Data và src lấy từ checkout chính qua `R2AI_ROOT`.
