# S26 — Cấu trúc dữ liệu/gold chưa khai thác

2026-10-18 (UTC+7) · Claude Opus 5.5 · nhánh `feat/s26-gold-structure` (worktree `.superpowers/worktrees/s26-gold-structure`, base `ecd8e3f`).

Phạm vi: phân tích và đề xuất probe. Không ghi index, không nộp. Ngoại lệ do Vu yêu cầu giữa phiên (H7): dựng ZIP EN-only để Vu tự nộp. Mọi số dưới đây đo trên máy local, trừ điểm LB do Vu cung cấp. Script ở `scripts/s26/` (chỉ đọc `data/`, ghi vào scratchpad hoặc `out/runs/s26-gold-structure/`).

Mốc tham chiếu: P1-vi+10 Doc P/R 0,1521/0,4499, Chunk P/R 0,1936/0,3644 (Vu cung cấp). P1-vi+10 trung bình 180,7 ID/query (165,1 vi + 15,65 zh), 105,2 chunk/query, chunk = nguyên body doc (96,5% chunk dài ≥90% body). Suy ngược thô: khoảng 27 doc đúng/query trên khoảng 60 gold/query (giả định gold chung, không phải số đo).

## 0. Tóm tắt

| Giả thuyết | Tín hiệu | Kết luận ngắn |
|---|---|---|
| H1 dữ liệu chưa dùng | Mạnh (gián tiếp) | Không có train/dev/qrels ở bất kỳ revision nào. Nhưng lịch sử repo HF cho thấy corpus v1→v2 (12/10): zh bị cắt từ 3,65M còn đúng 240.000, en thêm đúng 240.000 (= 1.200 × 200). |
| Pool test (mới, từ H1) | **Mạnh** | zh được giữ ở v2 gần query test hơn hẳn zh bị loại: AUC 0,80 (max-cos), 0,83 (min-rank). zh v2 là pool chọn theo query. BTC không dùng đúng BGE-M3 dense của mình (AUC chưa tới 0,95). |
| H7 en | Mạnh (âm) | E1 thêm 1.837 doc en mà Doc R giữ nguyên 0,3886 ⇒ precision en dưới khoảng 0,4%. EN-only đã dựng để đo trực tiếp. |
| H2 query sinh từ doc | Yếu | 2,7% query có doc nguồn rõ (cover ≥60%), 30/32 đã có trong P1-vi+10, hạng median 1. |
| H3 gần trùng | Yếu–vừa | 20,4% doc đã nộp nằm trong cụm ≥2. Cụm nhỏ (≤3) chỉ thêm 9,7 doc/query chưa nộp. |
| H4 host/URL | Yếu | Không query nào dồn ≥90% về một host; median top-host share 0,28. Longchau chiếm 24,4% top-50 (gấp 2,3 lần tỷ trọng corpus). |
| H5 chunk | Đã khai thác | P1 đã dùng nguyên body; scorer chấm LCS/|ref| nên body chứa trọn ref là tối ưu theo doc. Đòn bẩy là byte, không phải ranh giới chunk. |
| H6 query | Không có cấu trúc khai thác | 0 cặp query gần trùng (Jaccard 3-gram ≥0,5), 0 trùng chuẩn hoá; 18 query trùng nguyên tiêu đề doc (đều Vinmec/…). |

Không tìm thấy một cấu trúc nào giải thích ngay khoảng cách 0,28→0,6. Phát hiện có giá trị nhất là **zh v2 (và nhiều khả năng en v2) được BTC chọn theo query**: BTC đã chạy một retriever trên toàn kho zh để chọn 240k doc. Nếu gold cũng sinh từ retriever đó (pool + judge), việc nhận dạng retriever của BTC là probe có trần cao nhất. Probe này chạy offline, không tốn lượt nộp.

## 1. H1 — Dữ liệu chưa dùng

**Bằng chứng.**
- `AIGuruTinix/ViBioMIR`: 1 branch `main` (`71be0e12…`), 0 tag, 0 PR, 1 commit "Duplicate from gnuhbear/ViBioMIR" (12/10 13:52 UTC). Có thêm ref convert `parquet`. File: README, 7 hình, `links_corpus_vi_en_zh.parquet`, `query.parquet`. SHA local khớp (`87e44a79…`, `ce13508b…`).
- Repo nguồn `gnuhbear/ViBioMIR` có 7 commit (12/10 03:55→12:54 UTC). Commit đầu "Duplicate from AIGuruTinix/ViBioMIR" chứa `links_corpus.parquet` v1 (35.111.121 B). Sau đó xoá v1, upload `links_corpus_vi_en_zh.parquet` hai lần (36.887.910 B rồi 37.689.434 B). Không revision nào có train/dev/qrels/sample/chunk tham chiếu. `query.parquet` không đổi qua mọi revision.
- Org khác: `AIGuruTinix` chỉ có thêm `ViFinQA`; `tinixai` có 3 dataset không liên quan; `gnuhbear` chỉ có ViBioMIR.
- README v1→v2 chỉ đổi tên file, số liên kết (1.220.598 = vi 740.598 + en 240.000 + zh 240.000) và câu "nguồn tiếng Anh gồm các tài liệu thuộc PubMed". Không có câu nào về cách tạo query/gold. Hai hình không nhúng trong README (`04_medical_topic_signals`, `05_intent_distribution_pie`) chỉ là thống kê chủ đề/intent của query (Diagnosis 383, Therapy 196, Advice 151, Caution 124, Metric Interpretation 113, Outcome 93, Other 140). `tokens_by_category` cho thấy kho gốc MedKB có 45,3 tỷ token, 41,3 tỷ là tạp chí học thuật ⇒ corpus thi là mẫu con của kho MedKB.
- ID en sắp theo PMID (corr 0,994), dải 4.421.055–7.466.161: kho MedKB có khoảng 3,05M bài PubMed, BTC chọn 240k. Khoảng cách ID tối đa 2.243, xa hơn hẳn mức của lấy mẫu đều (p≈7,9%) ⇒ lấy mẫu không đều.

**Độ mạnh.** Không có dữ liệu nhãn. Tín hiệu gián tiếp mạnh: hai con số tròn 240.000 = 1.200 × 200 cho zh và en dẫn tới pool test ở mục 2.

**Probe.** Không có probe nộp từ H1. Đề xuất hỏi BTC bằng văn bản: gold được gán trên corpus v1 hay v2, và zh/en v2 được chọn thế nào (thuộc vùng xám, cần Vu quyết).

## 2. Pool test — zh v2 được chọn theo query

**Bằng chứng.** Đối chứng là mẫu 2% zh của v1, crawl trước khi có v2 (index `seg00001-sample-old-zh_sample_t256`, 29.385 doc). 2.413 doc thuộc v2 (kept), 26.972 bị loại (dropped). Điểm doc = max cosine BGE-M3 dense trên các chunk và trên 1.200 query.

| | kept | dropped |
|---|---:|---:|
| median max-cos | 0,651 | 0,604 |
| tỷ lệ max-cos ≥0,65 | 50,3% | 13,8% |
| tỷ lệ max-cos ≥0,70 | 12,8% | 1,6% |

- AUC(max-cos→kept) = 0,796; trong từng domain: cnkang 0,744, 120ask 0,716, a-hospital 0,821, zhongyibaodian 0,819, wujue 0,922. Tỷ lệ kept theo decile max-cos tăng đều từ 0,2% lên 29,8%.
- Min-rank (hạng tốt nhất của doc trên mọi query, trong mẫu): AUC 0,833. Doc có min-rank 1: 43,4% kept (nền 8,2%). Min-rank ≤50 gom 88,3% doc kept.
- zh canonical (157.524/240.000 ID đã có) có median max-cos 0,649, khớp nhóm kept ⇒ bất biến trên toàn bộ zh đã crawl.
- Fingerprint retriever, AUC −min-rank: BGE-M3 dense max-chunk 0,833; dense chunk đầu 0,783; sparse 0,555; hybrid z 0,7/0,3 là 0,701; multilingual-e5-small (doc 512 token) 0,690. Không biến thể nào gần 0,95 ⇒ BTC dùng retriever khác (mạnh hơn, hoặc dịch query sang zh, hoặc ngưỡng/pool khác).
- vi không có dấu hiệu pool: max-cos vi phân tán (median 0,613, P10 0,483). suckhoecongdongonline có median 0,501 và 70,3% doc dưới 0,55 (xổ số, tin không y tế). vi v2 = 100% vi v1.
- en: không có đối chứng PubMed bị loại nên chưa kiểm được. Gán doc en về query gần nhất cho phân bố lệch (median 114 doc/query, 3 query có 0) ⇒ không kết luận được.

**Diễn giải (suy luận, chưa đo).** Gold nhiều khả năng được gán khi corpus còn là v1 (vi+zh, chưa có en). Khi rút zh xuống 240k, BTC phải giữ mọi doc zh gold, rồi lấp phần còn lại bằng doc gần query theo retriever của họ. Ước lượng hỗn hợp thô: khoảng 80% zh v2 trông "liên quan query", khoảng 20% giống phân bố dropped. Điều này nhất quán với E1: en được thêm sau nên không chứa gold.

**Độ mạnh.** Mạnh với zh (AUC 0,80–0,83, nhất quán trong từng domain). Chưa kiểm được với en.

**Probe đề xuất.**
- P-A (offline, 0 lượt): nhận dạng retriever BTC bằng AUC kept/dropped trên mẫu 29k doc. Ứng viên: Qwen3-Embedding-4B (có cache local, cần 4/8-bit vì GPU 6 GB), Qwen3-Embedding-0.6B, gte-multilingual, BGE-M3 với query đã dịch sang zh (Qwen ≤15B), BM25 zh trên query dịch. Tiêu chí: AUC ≥0,95. Nếu đạt, dùng đúng retriever đó xếp hạng vi/zh. Mức tăng: chưa ước lượng được; đây là giả thuyết duy nhất đủ lớn để giải thích khoảng cách 2 lần, nhưng cũng có thể không tăng gì nếu gold do LLM judge độc lập với pool. Chi phí khoảng 2–4 GPU-giờ.
- P-C: crawl nốt 73.249 ID 120ask còn thiếu (31% zh v2, độ phủ hiện 25%) cùng 6.591 zysjonline. Ngoại suy từ lịch sử (S6-f 0,2599 → C1 0,2655 khi zh phủ 144k) cho khoảng +0,003 Final; có thể cao hơn nếu 120ask (dạng hỏi–đáp) giàu gold. Bị chặn bởi timeout/geo (S24), cần Vu duyệt nguồn.

## 3. H7 — Nhánh en (mở lại D17)

**Cấu hình E1 (đã nộp, 15/10).** Base C1. Candidate `artifacts/cand_en.parquet`: hybrid BGE-M3 dense+sparse (w 0,7, agg max) lấy top-100 doc → bge-reranker-v2-m3 chấm mọi chunk → top-50 PMID/query. Gate: logit ≥0,96, tối đa 10 doc vật lý/query, bỏ `title_only`; chunk chỉ top-1 logit ≥0,96 và chỉ cho doc đã chọn. Thêm 1.837 doc (1,53/query; 414 query nhận ≥1 doc, 786 query không), 414 chunk. LB: E1 0,2648 (Doc F2 0,2675, P 0,1553, R 0,3886; Chunk F2 0,2620, P 0,1851, R 0,3402) so với C1 0,2655 (Doc 0,2687/0,1567/0,3886; Chunk 0,2623/0,1857/0,3402).

**Suy ra.** Doc R không đổi ở 4 chữ số ⇒ ΔR < 0,0001. Với G≈61/query và 1.200 query, số doc en trúng gold < khoảng 7 trên 1.837 ⇒ precision en < khoảng 0,4% ngay ở nhóm logit cao nhất (giả định gold chung, có làm tròn).

**EN-only (đã dựng, chưa nộp).**
- `out/runs/s26-gold-structure/en-only/EN-only.zip`: 9.154.860 B, SHA256 `8b72528ede03f5cae4ab1a62876ff298185b15f32f5f77bb8d874d1bdd683907`; JSON 48.721.897 B, SHA256 `be991816a4455b0de51b03ca065241eeabd5498cc886385976294dbeacae609e`.
- Retriever: BGE-M3 dense CLS fp16, inner product exact trên `data/index/en_pubmed` (581.102 chunk); doc = max chunk; query nguyên văn tiếng Việt (M3Encoder max_len 512). Mỗi query 50 doc, mỗi doc 1 chunk (chunk có điểm cao nhất). Không có vi/zh. Không lọc `title_only` (0,07% chunk).
- Kiểm: validator riêng `scripts/s26/validate_en.py` 0 lỗi (1 file trong ZIP, LZMA dict 256 MiB props `5d00000010`, 1.200 id đúng, doc ∈ corpus v2, không trùng, chunk nguyên văn trong `record_text` S10 của đúng doc). Validator native `r2ai.eval.validate`: 60.000 lỗi, toàn bộ là "doc_id not in data/docs_vi", vì validator này chỉ kiểm verbatim trên vi. Không có lỗi schema/id/corpus.
- Trùng với `cand_en` (hybrid + reranker): median 29/50 doc mỗi query.
- Kỳ vọng: nếu en không có gold, Final ≈ 0. Giá trị của lượt nộp là thông tin (khoá hẳn D17), không phải điểm.

**EN-only-Q (Qwen3-Reranker-0.6B).** Xem mục 9: trạng thái tại thời điểm commit. Lưu ý: metric chấm theo tập, nên cùng 50 doc đổi thứ tự sẽ cho **cùng điểm** với EN-only. Muốn đo được tác dụng của Qwen phải cắt top-k hoặc đổi chunk (cần Vu chọn).

## 4. H2 — Query sinh từ doc

**Phương pháp.** Chuẩn hoá theo metric (NFKC → unescape → lower → gộp whitespace → strip). Token = từ `\w+` (âm tiết tiếng Việt, không phải subword BGE-M3). n = 8 (cả query nếu ngắn hơn). Quét toàn bộ 727.681 doc vi canonical. cover = tỷ lệ 8-gram của query có trong doc (xấp xỉ LCS).

**Kết quả (toàn bộ 1.200 query; mẫu 100 seed 42 trong ngoặc).**

| Ngưỡng | Query có doc | Trong P1-vi+10 | Hạng median | Hạng ≤10 |
|---|---:|---:|---:|---:|
| ≥1 8-gram | 299 = 24,9% (32%) | 90 | 8,5 | 47 |
| cover ≥0,3 | 54 = 4,5% (7%) | 45 | 1 | 35 |
| cover ≥0,6 | 32 = 2,7% (6%) | 30 | 1 | 27 |
| cover ≥0,9 | 27 = 2,2% (6%) | 27 | 1 | 25 |

- Host của doc nguồn (cover ≥0,6): vinmec 21, medlatec 3, longchau 2, suckhoedoisong 2, còn lại 1 mỗi host. 18 query trùng nguyên tiêu đề doc. Query có nguồn ngắn hơn (median 11 từ so với 19).
- Kiểm chéo bằng embedding: 51 cặp query–doc cos ≥0,95 đều có trong P1-vi+10, hạng median 1.
- Hai doc nguồn chưa nộp: Q858 → doc 176866 (medlatec, cover 0,75); Q965 → doc 510850 (suckhoedoisong, cover 0,60).
- Phát hiện này trái với gold-check cũ ("0/25 trang nguồn có trong corpus"): một phần trang nguồn Vinmec có trong corpus, nhưng chỉ khoảng 2–3% query.

**Độ mạnh.** Yếu. Doc nguồn đã có trong submission ở hạng đầu.

**Probe.** Thêm 2 doc nguồn còn thiếu: mức tăng khoảng +0,0003, không đáng một lượt; gộp vào lượt khác nếu có.

## 5. H3 — Cụm gần trùng

**Phương pháp.** MinHash 64 hoán vị, LSH 16 band × 4 row, xác nhận bằng signature ≥0,7, union-find. Shingle: vi 5 từ; zh 5 ký tự (zh không có ranh giới từ). Không bucket nào vượt trần 2.000. Lưu ý: union-find nối chuỗi tạo cụm khổng lồ theo chủ đề (bản tin COVID, trang sản phẩm Longchau) nên dưới đây tách theo kích thước cụm.

**Kết quả.**
- 887.151 doc → 738.081 cụm; 685.341 doc đơn. Cụm theo kích thước: 2: 35.514; 3: 7.865; 4–5: 4.859; 6–10: 2.641; 11–50: 1.625; 51–1.000: 234; >1.000: 2. Doc trong cụm ≥2: vi 161.961/727.681, zh 39.849/159.470.
- Cụm lớn nhất 1.710 doc (suckhoecongdongonline/baohaiphong/thanhnien), thứ hai 1.126 (805 Longchau).
- P1-vi+10: 20,4% doc đã nộp nằm trong cụm ≥2. Số bản gần trùng chưa nộp: cụm ≤2 là 0,029/doc đã nộp (5,1/query); ≤3 là 0,055 (9,7/query); ≤5 là 0,100 (17,8/query); ≤10 là 0,195 (34,7/query); ≤50 là 0,857 (152/query). 2.113/216.866 ID nộp không map được về canonical (doc không còn trong bundle).

**Độ mạnh.** Yếu–vừa: chỉ có ích nếu gold gán theo nội dung (bản gần trùng của gold cũng là gold).

**Probe H3 (ứng viên, chưa dựng).** Với mỗi doc đã nộp thuộc cụm kích thước 2–3, thêm các thành viên chưa nộp (khoảng +9,7 doc/query). Để giữ tổng số doc, cắt cùng số doc ở đuôi rank. Kỳ vọng: nếu bản gần trùng có xác suất gold bằng precision hiện tại (khoảng 15%), ΔFinal khoảng +0,002–0,003; nếu "thừa kế" gold mạnh thì tới khoảng +0,006. Chi phí: CPU, 1 lượt.

## 6. H4 — Cấu trúc host/URL

**Kết quả trên top-50 của P1-vi+10.**
- Tỷ trọng host: longchau 24,4%, suckhoedoisong 13,2%, medlatec 11,2%, vinmec 9,9%, tiemchunglongchau 7,2%, thanhnien 6,6%. Longchau gấp 2,3 lần tỷ trọng trong corpus vi (10,7%).
- Top-host share mỗi query: median 0,28, P90 0,42, max 0,84; trung bình 13,2 host/query, entropy 3,04 bit. Có 41 query ≥0,5 (Longchau 17, Vinmec 11, Medlatec 6), 3 query ≥0,7, 0 query ≥0,9. Tỷ lệ chuyên mục (2 đoạn path đầu) gần như trùng tỷ lệ host, vì phần lớn site không có chuyên mục trong URL.
- Phân cụm theo host trội: Longchau 667 query, suckhoedoisong 194, medlatec 108, vinmec 100, thanhnien 50. Query dài nghiêng nhẹ về host tập trung hơn (top-host share 0,297 → 0,320 từ nhóm ≤15 từ sang nhóm >60 từ).

**Độ mạnh.** Yếu. Không có query nào dồn về một host/chuyên mục. Câu hỏi còn mở: Longchau có giàu gold như tỷ trọng gợi ý không.

**Probe.** P-E: cắt Longchau tối đa 20% top-50, bù bằng host kế tiếp, cùng tổng doc. Đây là probe thông tin (đo gold-rate theo host), ΔFinal có thể âm. Chi phí: 1 lượt.

## 7. H5 — Chunk

**Spec BTC.** README chỉ mô tả `query.parquet`/`links_corpus`. Ràng buộc chunk lấy từ đề bài (docs/CONTEXT.md, docs/metric.md): `chunk_text` trích nguyên văn, khớp khi LCS_tokens(c,g)/|g| ≥0,4, P/R theo số chunk. Không có thông tin nào về độ dài chunk gold.

**Đã biết.** P1 dùng nguyên body (median chunk 5.275 ký tự, 1 chunk/doc, 58,1% doc có chunk). Lịch sử M9: full > c2, thêm chunk t256 ngắn cho hạng 20..45 làm giảm 0,0020. Vì overlap chia cho |ref|, chunk chứa trọn body sẽ khớp mọi ref trong doc; precision chunk = tỷ lệ doc có ≥1 ref. Chunk ngắn không thể vượt body về P/R trên cùng tập doc; nó chỉ giúp tiết kiệm byte để phủ thêm doc.

**Probe P-F (đoạn văn).** Giữ nguyên body cho các doc đang có chunk. Với 41,9% doc chưa có chunk, thêm 1–2 đoạn văn đầu (lead paragraph, ≥40 token) trong cùng trần byte. So với cách hiện tại (không có chunk). Kỳ vọng ±0,003 (chưa đo), vì doc hạng thấp có precision thấp và đoạn ngắn chỉ khớp ref nếu phủ ≥40% ref.

## 8. H6 — Phân bố query

- Độ dài (từ `\w+`): median 18, P25 13, P75 40, P90 67, P95 84, min 5, max 283. Phân nhóm: ≤10: 128; 11–15: 338; 16–20: 186; 21–30: 141; 31–50: 191; 51–100: 193; >100: 23.
- Ngôn ngữ: 1.200/1.200 tiếng Việt có dấu, 0 query chứa chữ Hán.
- Dạng câu: 916 kết thúc bằng "?"; 126 có ≥2 dấu hỏi; 518 có ngôi kể bệnh nhân (em/tôi/cháu/con…); 214 nêu tuổi/tháng/tuần; 11 mở đầu bằng lời chào. Đuôi phổ biến: "không ạ" 84, "thế nào" 82, "được không" 63, "sao không" 52, "bệnh gì" 39.
- Tên bệnh/thuốc: theo hình chính thức của BTC, 41,9% thuốc/điều trị, 18,8% xét nghiệm/chỉ số, 13,9% thai sản. Taxonomy S8 cũ: chẩn đoán 37,2%, điều trị 28,6%, thuốc 15,7%.
- Trùng: 0 cặp trùng chuẩn hoá, 0 cặp Jaccard 3-gram ≥0,5. 18 query trùng nguyên tiêu đề doc (mục 4).

**Độ mạnh.** Không có cấu trúc khai thác được (không có cụm query để chia sẻ gold).

## 9. Xếp hạng probe theo kỳ vọng/chi phí

| # | Probe | Chi phí | Kỳ vọng ΔFinal | Ghi chú |
|---:|---|---|---|---|
| 1 | P-A nhận dạng retriever BTC qua pool zh | 0 lượt, 2–4 GPU-giờ | chưa ước lượng; trần cao | Chỉ giả thuyết này đủ lớn để giải thích khoảng cách. Dừng nếu không model nào đạt AUC ≥0,9. |
| 2 | EN-only (đã dựng) | 1 lượt | ≈0 (thông tin) | Khoá D17. Nếu R en > 0,02 thì mở lại nhánh en. |
| 3 | P-D gần trùng cụm 2–3 | 1 lượt, CPU | +0,002…+0,006 | Giữ tổng doc. |
| 4 | P-C crawl 120ask/zysjonline còn thiếu | crawl bị chặn | ≈+0,003 (ngoại suy) | Cần Vu duyệt nguồn/IP. |
| 5 | P-F lead-paragraph cho doc chưa có chunk | 1 lượt | ±0,003 | Cùng trần byte. |
| 6 | P-E cắt Longchau | 1 lượt | có thể âm | Thông tin theo host. |
| 7 | H2 thêm 2 doc nguồn | gộp lượt khác | +0,0003 | |

## 10. Cần Vu quyết

1. EN-only-Q: cùng 50 doc đổi thứ tự sẽ cho cùng điểm với EN-only (metric theo tập). Chọn: (a) giữ đúng yêu cầu, (b) chỉ giữ top-k theo Qwen (ví dụ 20), hay (c) Qwen chấm mọi chunk của 50 doc để đổi chunk tốt nhất.
2. Có hỏi BTC bằng văn bản về cách chọn zh/en v2 và corpus dùng khi gán gold không (vùng xám).
3. Có duyệt P-A (tải thêm model embedding, ví dụ Qwen3-Embedding-0.6B, gte-multilingual) không.

## 11. Tái tạo

Script (đường dẫn `data/` và `src/` trỏ về checkout chính qua `R2AI_ROOT`, mặc định `D:/GitHub/r2ai-stage3`):
`en_only.py`, `validate_en.py`, `qwen_rerank_en.py` (H7); `pool_test.py`, `pool_rank.py`, `pool_fingerprint.py`, `pool_fingerprint_e5.py`, `en_pool.py`, `scan_canonical.py` (pool); `ngram_scan.py` (H2); `minhash.py` (H3, chạy với `PYTHONHASHSEED=0`); `host_path.py` (H4); `query_stats.py` (H6); `sub_docs.py` (trích doc/chunk từ ZIP). Query vector: `r2ai.index.bge_m3.M3Encoder` max_len 512, seed không dùng (encode tất định). Output trung gian ở scratchpad phiên, không commit.
