# Working notes — khung bài và kho số liệu (S42, 27/10/2026)

Trạng thái: **khung + số liệu, chưa viết văn.** Chờ Vu trả lời ngôn ngữ, template, giới hạn trang, tên đội, tác giả, hạn nộp (`open-questions.md` mục A). Mọi số dưới đây có nguồn `file:mục` trên nhánh `feat/s41` (`ba9850c`, cũng là gốc của nhánh này) trừ khi ghi nhánh khác. "Vu" = số LB Vu báo, chép lại từ docs; "suy ra" = tính từ Final = (Doc F2 + Chunk F2) / 2.

Kho số liệu: `data/*.csv` (mô tả ở `README.md`). Hình: `figures/*.png|pdf`, dựng bằng `scripts/make_figures.py`.

---

## 1. Tóm tắt

- Bài toán: query y khoa tiếng Việt → doc vi / zh / en (ViBioMIR, 1.200 query; corpus v2 1.220.598 link = vi 740.598 + zh 240.000 + en 240.000). Nguồn: `reports/s26-gold-structure.md` mục 1.
- Kết quả public: cấu hình chốt private **CH-rrf 0,4138** (Doc F2 0,4413 / Chunk F2 0,3863). Bản public cao nhất **120A-p4b-v3 0,4176** (khung p4b, không dùng được cho query mới). Nguồn: `docs/s41.md` (LB 27/10 chiều), `docs/s36-handoff-s30.md`.
- Ba phát hiện chính (chi tiết mục 4): pool zh do BGE-M3 dense doc-level 512 token chọn (AUC 0,924); en không có trong gold; gold chunk dài (nguyên body tốt hơn mọi cửa sổ ngắn).
- Pipeline (mục 5): BGE-M3 doc-level → top-300 → Qwen3-Reranker-4B + C-max 0,6B → RRF → hiệu chỉnh xác suất → cắt EF2 → carrier CR-fair → full body + cửa sổ 768.
- Câu về private: chờ kết quả sau 04/11.

## 2. Bài toán và metric

- Doc F2 macro theo query (β = 2). Chunk F2: chunk dự đoán khớp gold khi cùng (query, doc) và
  - v1 (đến 22/10): LCS_tokens(c, g) / |g| ≥ 0,4;
  - v2 (từ 23/10): thêm LCS_tokens(c, g) / |c| ≥ 0,4. Nguồn `docs/s38-chunk-metric-v2.md` mục Metric mới; `docs/metric.md`.
- Final = (Doc F2 + Chunk F2) / 2 (kiểm số học trên mọi dòng LB: `data/lb_scores.csv`).
- Ràng buộc nộp: ZIP ≤ 100 MiB (104.857.600 B); đội dùng trần nội bộ 102.600.000 B. Nguồn `src/r2ai/s38/compose.py:82-83`.
- Private: tối đa 5 lượt, hạn 04/11, kết quả 11/11 (`AGENTS.md`).
- Quy tắc ngưỡng F2 (dùng xuyên suốt): thêm một doc có lợi khi precision biên của nó > F2 hiện tại / (1 + β²) = F2 / 5. Kiểm trên LB: chuỗi n_c (`data/chunk_count_curve.csv`; ngưỡng 7,4–7,6%, precision biên 12,9% → 9,8% → 7,7%, bão hoà ở n_c × 1,4–1,6) và chuỗi cut (`docs/s41.md`: precision biên 0,103 và 0,126 > F2/5 ≈ 0,088 nên bớt doc là lỗ).

## 3. Dữ liệu

### 3.1 Độ phủ text theo host (bảng chính) — `data/coverage_by_host.csv`

Đo 27/10 trên bộ dữ liệu cuối (canonical-20261018 + 9 segment 120ask S37 + lô Pharmacity S27 + segment en S37), script `scripts/build_coverage.py`. "good" = cờ quality của bundle (≥ 200 code point, hoặc ≥ 50 chữ Hán với zh).

| ngôn ngữ | URL v2 | có doc | good | % good |
|---|---:|---:|---:|---:|
| vi | 740.598 | 728.660 | 723.548 | 97,70 |
| zh | 240.000 | 232.470 | 227.509 | 94,80 |
| en | 240.000 | 239.771 | 239.353 | 99,73 |

Good theo nguồn: vi crawl trực tiếp 639.158, Common Crawl 83.411, Pharmacity 979; zh crawl trực tiếp 143.110, Common Crawl 12.349, VPS 120ask 70.032, webMedQA 2.018; en PubMed 239.353.

Host thiếu nhiều (< 90%): laodong.vn 87,1%, youmed.vn 70,7% (halted, bot challenge; phần có là CC), pmc-ecm-healthblog.beta.pharmacity.io 18,4%, thaythuocvietnam.vn 20,2%, zysjonline.com 0% (Cloudflare CAPTCHA), article.iiyi.com 0% (timeout). Cần đưa bảng đầy đủ vào phụ lục hay chỉ top host: chờ template.

Mốc so sánh lịch sử: S20 (17/10) vi 87,00%, zh 58,90% (`codex/s23-cc-news:reports/s20-coverage-audit.md`, `2d3f44c`).

### 3.2 Thu thập và provenance (gạch đầu dòng, mỗi ý có nguồn)

- Crawl trực tiếp: UA `r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)`, ≤ 1 req/s + jitter, 1 IP, dừng toàn bộ khi 403/429/401/451/challenge (`deploy/s37/README.md`; S24 `feat/s24-recovery:out/runs/s24-recovery/report.md`).
- Dừng khi gặp chặn chủ động: nhathuoclongchau (Cloudflare 403 5/5, dừng, chuyển sang CC; `docs/PROGRESS.md` M11), zysjonline (Cloudflare CAPTCHA, dừng; S24 report), youmed (bot challenge, halted).
- **robots: xem open-questions B1** (D21 bỏ qua robots có duyệt; vov.vn và baolangson.vn từng bị robots chặn, sau đó crawl trực tiếp ở S24).
- Common Crawl: chỉ URL thuộc corpus, provenance WARC filename/offset/length, crawl ID, SHA256 (`docs/COMPLIANCE.md` mục Common Crawl). Dùng cho longchau 76.868, vov 1.392, ask.39 9.751, cnkang 1.326, …
- webMedQA (Apache-2.0): 2.464 ID khớp chính xác URL 120ask của corpus; kiểm 433/446 câu hỏi LCS ≥ 80% (97,09%) (S24 report mục webMedQA).
- 120ask: site không vào được từ máy ở VN (timeout); crawl từ VPS AWS Sydney 23–24/10 (56.373 URL), site timeout từ VPS 24/10, phần còn lại 16.619 URL crawl 25/10 từ VPS mới (`docs/PROGRESS.md` mục S37). **Cách mô tả: open-questions B2.**
- Pharmacity: lô S27 qua CC (`reports/s27_pharmacity_final.md`). "Theo hướng dẫn BTC": cần nguồn (open-questions B4).
- Doc không truy cập được: "BTC xác nhận không có trong nhãn" — chưa tìm thấy trong docs (open-questions B5).
- Text: extractor theo domain + generic; dedupe canonical (887.151 doc vi+zh; alias theo nội dung) (`reports/s25-canonical.md`).

## 4. Phân tích benchmark

(a) **Fingerprint retriever qua pool zh** — `data/fingerprint_auc.csv`, `figures/fingerprint_auc.png`. Nguồn `reports/s26-retriever-fingerprint.md` mục 0–3, 5g.
- Đối chứng: mẫu 2% zh v1 (29.385 doc), 2.413 kept ở v2 / 26.972 dropped. Decile min-rank thấp nhất có 0 doc kept ⇒ không lấp ngẫu nhiên.
- Tốt nhất: BGE-M3 dense nhúng nguyên doc_text, 512 token, query vi gốc: AUC −min-rank 0,924 (max-chunk t256 0,833; query dịch zh 0,815; BM25 zh 0,695; sparse 0,555; Qwen3-Embedding-4B 0,757 trên mẫu 10k vs BGE-M3 0,915).
- Độ dài: 128 0,878 · 256 0,904 · 512 0,913 · 1024 0,906 · 2048 0,904 (title lặp); đỉnh ở 512 = max_length mặc định.
- Extractor: extractor của đội 0,924 > trafilatura + title 0,915 > … > jusText 0,739.
- Hệ quả đo trên LB: P1-docrank (đổi thứ tự chunk theo doc-level) 0,2822 → 0,2937; P-A3-nc (chọn doc theo doc-level) 0,3205; P-A4-K90 (toàn corpus) 0,3750.

(b) **vi không chọn theo query**: link nội bộ trong/ngoài corpus, 7 site y tế AUC 0,531 / 0,518; Vinmec 0,444 (BTC loại bài giống query). `reports/s26-retriever-fingerprint.md` mục 4.

(c) **en không có trong gold** (hai phép thử LB + một phép thử offline):
- E1 (15/10): +1.837 doc en (bge-reranker logit ≥ 0,96), Doc R giữ nguyên 0,3886 ⇒ precision en < ~0,4% (`reports/s26-gold-structure.md` mục 3).
- EN-only-Q: 20 doc en/query chọn bằng Qwen3-Reranker-0,6B → Final 0, Doc P 0 (`reports/s26-retriever-fingerprint.md` dòng 5; `docs/s36-fuse4.md` gói B).
- 4B trên 24.000 cặp en: TB 0,240, 22,6% ≥ 0,5, 4,8% ≥ 0,9 ⇒ reranker đánh giá "liên quan" nhưng không phải gold (`docs/s36-fuse4.md` mục Gói final-4b-B).
- Gợi ý giải thích: en v2 = 240.000 = 1.200 × 200 thêm vào ngày 12/10, sau khi gold đã gán (suy luận, `reports/s26-gold-structure.md` mục 2).

(d) **Gold chunk dài** — `data/chunk_size_curve.csv`, `figures/chunk_f2_by_size.png`. Doc J3-thrC giữ nguyên, chỉ đổi chunk: sec-1 (~190 token) 0,0904 · win384 0,1805 · win768 0,3433 · full body 0,3594 · cap2048 0,3606 · full+w768 0,3636 · clean 0,3580 (`docs/s38-chunk-metric-v2.md` LB lượt 1; ba số cuối: open-questions C2). Body carrier: median 999 token, > 768 token 61,3%.

(e) **Baseline BTC precision 0,88**: Doc F2 0,5587, P 0,8805, R 0,5146, Chunk F2 0,1941 (Final 0,3764). Thử tái tạo bằng bge-reranker-v2-m3 trên pool cosine: BGR-K40-doc Doc P 0,343 vs J4-K40 0,376 ⇒ không tái tạo được (`reports/s33-btc-baseline.md`; số BGR từ bảng Vu). Giải thích thiên lệch do pool: viết sau khi Vu xác nhận khung lập luận (open-questions D3).

(f) **Không có rò rỉ cấu trúc**: không train/dev/qrels ở mọi revision HF; 0 cặp query gần trùng; query sinh từ doc chỉ 2,7%; host/ID không mang tín hiệu (`reports/s26-gold-structure.md` mục 0; `feat/s33-structure-signals:reports/s33-structure-signals.md`, `a3edad2`).

## 5. Phương pháp (khung free) — `figures/pipeline.png`, `data/constants.csv`

1. Embed: BGE-M3 dense doc-level, doc_text 512 token, query vi nguyên văn (revision `5617a9f6…`).
2. Pool: cosine top-300 của corpus (canonical + Pharmacity + 120ask; top-400 rồi ghép), hoà cosine: canonical trước (`docs/s36-handoff-s30.md`).
3. Chấm: Qwen3-Reranker-4B (P4B, doc 512 token Qwen3, Kaggle 2×T4) và C-max = max điểm Qwen3-Reranker-0,6B trên top-3 chunk t256 body/answer theo cosine (GPU máy).
4. RRF k = 60 trên ba hạng cosine / 4B / C-max (bỏ Jina, ablation).
5. Hiệu chỉnh: p_cal = σ(a · logit P4B + b), a 0,5298, b −1,4097 (khớp 3 Doc P tổng hợp); G_q = s · (Σ p_cal cosine top-211 + đuôi), s 1,7405 (khớp 1 Doc F2 tổng hợp).
6. Cắt EF2: duyệt theo RRF, giữ doc khi p_cal > F2_q kỳ vọng / 5 (trần độ sâu 300).
7. Carrier CR-fair: ⌈ρ |D_q|⌉ doc mang chunk mỗi query, ρ 0,7937 giải cho tổng 175.548; chọn theo thứ tự RRF (CH-rrf).
8. Chunk: nguyên body; body > T token thêm cửa sổ 768 quanh chunk t256 tốt nhất; T nhỏ nhất dưới trần ZIP (= sàn 1536). Carrier thin (≥ 50% base64 / placeholder) bỏ chunk.
- Minh bạch: hằng số toàn cục hiệu chỉnh trên điểm tổng hợp LB public; không dùng file chấm chi tiết từng query (`docs/s36-fuse4.md:133`, `docs/s40-private.md` mục 7).

## 6. Thí nghiệm

- Hình tiến trình: `figures/lb_progress.png` (dữ liệu `data/lb_scores.csv`, chỉ các dòng có trong bảng Vu).
- Bảng ablation khung free: `data/ablation_free.csv`, `figures/ablation_free.png` (`docs/s41.md` Bảng ablation).
- Doc F2 theo số doc/query: `data/doc_count_curve.csv`, `figures/doc_f2_by_count.png`.
- Kết quả âm (mỗi ý cần số LB):
  - hybrid / BM25: AUC fingerprint (mục 4a); số LB pipeline hybrid cũ (sub02–RD150, corpus cũ): open-questions C5.
  - encode theo phần (PART-thrC): chưa thấy số LB trong docs — open-questions C3.
  - bge-reranker: BGR-K40-doc Doc P 0,343 < J4-K40 0,376.
  - Jina: bỏ Jina +0,0010 (FREE-noJina); JINA-K40 Doc P 0,340.
  - chunk ngắn: win384 0,1805, sec-1 0,0904.
  - split: FREE-split T1536 +0,0002, CH-sel2 +0,0002.
  - D400: −0,0074.
  - cắt mạnh hơn: cut150 −0,0007, cut130 −0,0071.
  - carrier theo C-max: −0,0087.
  - 0,6B thêm vào 4B: J4-rrf3-ph979 0,3994 vs J4-rrf-ph979 0,3997.

## 7. Tổng quát hoá và tái lập

- Khung free vs p4b: p4b-v3 0,4176 vs free-v2 0,4101 (−0,0075); p4b cần K_q (số doc của bản nền cho từng query public), không có với query mới (`docs/release-private.md` mục B).
- Pipeline query mới: `python -m r2ai.newq prepare → export-scoring → (Kaggle 4B, GPU C-max) → import-scores → build`; ghim SHA 79 tệp (`docs/s40-private.md` mục 2; `configs/s41-private-final.json`).
- Diễn tập 100 query đổi id: Doc F2 kỳ vọng chênh −0,0004 so với FREE-noJina (ngưỡng ±0,002) (`docs/s40-private.md` mục 3).
- Một lệnh dựng lại trùng SHA: `python -m r2ai.release build --config …` (19,6 / 20,2 phút CPU, trùng SHA từng byte) (`README_RELEASE.md` mục 3).
- Chi phí (đo): 4B 6,7 cặp/s trên Kaggle 2×T4 (360.000 cặp ≈ 15 giờ, 2 phiên); C-max 26,0 dòng/s RTX 3050 6 GB (939.600 dòng ≈ 10 giờ); build + compose 35–45 phút CPU (`docs/s40-private.md` mục 5). VPS 1–4 USD: chưa có nguồn (open-questions C6).

## 8. Hạn chế và bài học

- Hiệu chỉnh trên public dễ ngoại suy sai: fit lại (a, b, s) trên 18 số LB (RMSE 0,0042) dự đoán cut150 +0,04 Doc F2, thực −0,004; cut130 thực −0,010 (`docs/s41.md`; `docs/s40-private.md` mục 7b). Mọi bản fit có 180–190 doc/query.
- Nhiễu LB: nhiều khác biệt ±0,0002–0,001 (split, sel2, canon); không có ước lượng phương sai — open-questions D4.
- Phụ thuộc text extract: AUC theo extractor 0,739–0,924; trần AUC < 1 nhiều khả năng do khác text với BTC.
- Tái lập điểm số học: C-max bf16 phụ thuộc cách ghép batch (lệch tối đa 0,072 khi xáo thứ tự); Jina listwise không tái lập được (`docs/s40-private.md` mục 3).

## 9. Kết luận — viết sau.

## Phụ lục

- Model + revision (+ license: open-questions B3): BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181`; Qwen3-Reranker-4B `22e683669bc0f0bd69640a1354a6d0aebcfeede5`; Qwen3-Reranker-0.6B `e61197ed45024b0ed8a2d74b80b4d909f1255473`; Jina Reranker v3.5 `e8a93f33f0b22108f8c2364f8484ce3422552fbc` (thí nghiệm, không trong bản chốt); bge-reranker-v2-m3 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (thí nghiệm) (`README_RELEASE.md` mục 1; `docs/COMPLIANCE.md`).
- Hằng số: `data/constants.csv`.
- Lệnh tái lập: `README_RELEASE.md` mục 3; `docs/private-final.md`.
