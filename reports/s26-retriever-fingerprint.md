# S26 P-A — Nhận dạng retriever BTC qua pool zh, và kiểm tra pool vi

2026-10-19 (UTC+7) · Claude Opus 5.5 · nhánh `feat/s26-gold-structure`. Tiếp nối `reports/s26-gold-structure.md` mục 2. Không nộp, không ghi `data/index`. Script ở `scripts/s26/`; output trung gian ở `out/runs/s26-gold-structure/fingerprint/` (ignored).

en đã đóng: Vu báo EN-only-Q (top-20 Qwen) = 0 trên 24.000 doc. Nếu en v2 cũng được chọn theo query thì đó là hard negative, không phải pool gold. Report này không kiểm thêm en.

## 0. Tóm tắt

- **zh: retriever của BTC gần như chắc chắn là BGE-M3 dense (hoặc model rất giống) nhúng nguyên doc, cắt ở 512 token, với query tiếng Việt gốc.** Biến thể "chỉ body, 512 token" đạt AUC −min-rank **0,924** (max-chunk t256 chỉ 0,833), AUC trong từng domain 0,91. AUC đạt đỉnh ở 512 rồi giảm ở 1024/2048; 512 là `max_length` mặc định của BGE-M3.
- Query dịch sang tiếng Trung, BM25 zh, sparse và e5-small đều kém hơn. Qwen3-Embedding-4B **chưa đo**: GPU đang chạy job `run_vi2_score_launcher.py --command full-score --snapshot-tag s24-vi2` của session khác (từ 01:16 19/10); hai lần chạy chung GPU bị tràn VRAM, không ra kết quả sau 5,5 giờ, đã dừng để không làm chậm job đó.
- Trần đo được gần 1: decile min-rank thấp nhất có 0/2.939 doc được giữ ⇒ BTC không lấp ngẫu nhiên; phần AUC còn thiếu (0,924 → 1) nhiều khả năng do khác bản extract text (BTC nhúng text của họ, mình nhúng text extract của mình).
- **vi: không có dấu hiệu pool.** Trên 7 site chỉ nói về y tế, bài trong corpus không gần query hơn bài cùng site ngoài corpus (AUC 0,52). Riêng Vinmec ngược chiều (AUC 0,44): BTC đã loại bớt bài giống query (trang nguồn).
- Đề xuất: probe P-A2 xếp hạng lại vi+zh theo BGE-M3 dense doc-level body 512 (mục 5), để kiểm giả thuyết "gold sinh từ cùng retriever".

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
| multilingual-e5-small doc 512 | 0,674 | 0,690 | 0,266 | |
| hybrid z 0,7 dense + 0,3 sparse, query vi | 0,618 | 0,701 | 0,178 | |
| BGE-M3 sparse max-chunk, query zh | 0,679 | 0,704 | 0,205 | |
| BGE-M3 dense max-chunk, query zh | 0,736 | 0,775 | 0,330 | |
| BGE-M3 dense title only | 0,757 | 0,782 | 0,460 | 0,351 |
| BGE-M3 dense chunk đầu | 0,757 | 0,783 | 0,461 | |
| BGE-M3 dense doc 512, query zh | 0,769 | 0,815 | 0,371 | |
| BGE-M3 dense max-chunk t256 (cách S26 cũ) | 0,796 | 0,833 | 0,433 | |
| BGE-M3 dense mean-chunk | 0,795 | 0,835 | 0,521 | |
| BGE-M3 dense mean top-2 chunk | 0,798 | 0,853 | 0,462 | |
| BGE-M3 dense title + body 128 | 0,801 | 0,878 | 0,480 | 0,416 |
| BGE-M3 dense title + body 2048 | 0,822 | 0,904 | 0,514 | 0,450 |
| BGE-M3 dense title + body 256 | 0,818 | 0,904 | 0,531 | 0,451 |
| BGE-M3 dense title + body 1024 | 0,823 | 0,906 | 0,516 | 0,450 |
| BGE-M3 dense title + body 512 | 0,825 | 0,913 | 0,528 | 0,462 |
| **BGE-M3 dense body 512 (không title)** | **0,833** | **0,924** | **0,548** | **0,493** |
| Qwen3-Embedding-4B (NF4) body 512, query vi | chưa đo | chưa đo | | |

Qwen3-Embedding-4B: script `pool_fingerprint_qwen3.py --body-only --subsample 10000` (đủ 2.413 kept + 7.587 dropped ngẫu nhiên seed 42, in kèm mốc BGE-M3 body 512 trên cùng tập) sẵn sàng chạy khi GPU rảnh.

Ghi chú: biến thể "per-query z-score" chỉ đổi AUC best (0,814), không đổi min-rank. Bản dịch zh do `qwen2.5:7b` (Ollama, temperature 0, seed 42) tạo; 23/1.200 bản còn sót chữ Việt.

**BGE-M3 body 512 theo domain (AUC −min-rank):** cnkang 0,906 (759 kept), 120ask 0,921 (1.160), a-hospital 0,929, zhongyibaodian 0,933, youlai 0,959, care.39 0,948, msdmanuals 0,951; ngoại lệ jbk.39 0,729, test.pmphai 0,794, jb39 0,522 (mẫu nhỏ). Trung bình theo số kept: 0,909 ⇒ không phải hạn ngạch theo domain.

**Profile min-rank (body 512):** min-rank ≤1: 940 doc, 54,6% kept, phủ 21,3% kept; ≤3: 48,8% / 48,3%; ≤10: 34,9% / 84,7%; ≤20: 25,3% / 95,2%.

**Diễn giải.**
1. Đơn vị nhúng là cả doc, không phải chunk: doc-level tăng từ 0,833 lên 0,913–0,924.
2. Độ dài 512 token là tối ưu (128: 0,878; 256: 0,904; 512: 0,913; 1024: 0,906; 2048: 0,904), khớp max_length mặc định của BGE-M3/FlagEmbedding.
3. Bỏ title tốt hơn có title (0,924 so với 0,913): BTC nhúng phần nội dung, hoặc title của họ khác title mình extract.
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
1. Nhúng BGE-M3 dense body 512 cho doc ứng viên. Rẻ: chỉ các doc trong pool ứng viên hiện có của P1 (ví dụ top-300/query, số doc unique cần đo). Đầy đủ: toàn bộ 887.151 doc canonical, ước khoảng 12 giờ trên RTX 3050 (đo 29.385 doc body 512 hết khoảng 33 phút khi GPU dùng chung).
2. Xếp hạng vi+zh mỗi query theo cosine doc-level, chọn K doc (giữ K và chunk = nguyên body như P1), so LB với P1.
3. Phiên bản an toàn hơn: trộn RRF giữa thứ tự P1 và thứ tự doc-level.

Kỳ vọng: chưa ước lượng được. Nếu gold = top-k của retriever BTC thì precision có thể tăng mạnh; nếu gold do LLM judge độc lập với pool thì chỉ thay đổi nhỏ. Một lượt nộp sẽ phân xử.

**Offline trước khi nộp (0 lượt):** đo trùng lặp giữa top-50 doc-level và top-50 P1 trên vi; nếu gần như trùng hoàn toàn thì P-A2 không có thông tin mới.

## 6. Tái tạo

`pool_ceiling.py` (mục 2); `pool_fingerprint.py`, `pool_fingerprint_cpu.py`, `pool_fingerprint_e5.py`, `pool_fingerprint_zhq.py`, `pool_fingerprint_doclen.py`, `pool_fingerprint_qwen3.py` (mục 3, chạy bằng `.venv` khi cần bitsandbytes/jieba/rank_bm25); `pool_body512_domain.py` (theo domain); `translate_zh.py` (Ollama, chia shard `k/3`); `vi_links.py`, `vi_pool_test.py` (mục 4). Data và src lấy từ checkout chính qua `R2AI_ROOT`.
