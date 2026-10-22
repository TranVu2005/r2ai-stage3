# S33: kiểm giả thuyết "gold = pipeline baseline của BTC" bằng bge-reranker-v2-m3

Cập nhật 2026-10-22 18:30 (UTC+7), nhánh `feat/s33-btc-baseline` (S31 `e3f49e7` + merge S32 `410e54e`). Mọi số dưới đây đo trên máy local. **Chưa nộp, chưa có điểm LB.**

## Giả thuyết

Baseline BTC trên LB: Doc F2 0,5587, P 0,8805 / R 0,5146, khoảng 30 doc/query. J4-K40 của mình chỉ đạt Doc P 0,376 ở 40 doc/query. Slide Mentor Day (12/10) mô tả baseline gồm BM25 + BGE-M3 dense, hợp bằng RRF, rerank bằng cross-encoder BGE-reranker-v2-m3; chunk theo mục và giữ cặp hỏi-đáp. Giả thuyết: gold được tạo bằng chính pipeline này. Pha A kiểm phần rerank: giữ nguyên pool cosine top-211, chỉ thay thứ tự chọn doc bằng bge-reranker-v2-m3.

**Lịch sử repo (để đọc kết quả):**
- Repo đã từng dùng bge-reranker-v2-m3, trái với tiền đề "chưa dùng lần nào". Từ sub02 đến RD150 (03–06/10), pipeline là BGE-M3 hybrid dense+sparse trên chunk t256, rerank top-200 chunk bằng bge-reranker-v2-m3, rồi gộp về doc.
- Điểm LB khi đó (`submissions/LOG.md`): K100 Doc P 0,174 / R 0,290; RD150 Doc P 0,143 / R 0,341 ở K150. Khi ấy corpus còn là index cũ (623.805 doc, chưa có canonical), nên các số này không bác được giả thuyết.
- E1 (15/10) thêm 1.837 doc en chọn bằng bge-reranker-v2-m3 (logit ≥ 0,96), Doc R không đổi ⇒ precision en dưới khoảng 0,4% (`reports/s26-gold-structure.md`).

## Thiết kế

- **Model:** `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (cache HF local, khớp `docs/COMPLIANCE.md`).
- **Cách chấm:** fp16, `tokenizer(query, passage, truncation='only_second', max_length=512)`, giống model card và `r2ai.index.bge_m3.Reranker`. Điểm là logit thô; ghi thêm `sigmoid` (`normalize=True` của FlagReranker). Không có instruction.
- **Pool:** cosine top-211 BGE-M3 doc-level trên scored union của S31, đúng pool của COS-K40 / J4-K40.
  - Pool này khác `pool211_index.parquet` ở 186 cặp trong 9 query: cả hai cắt một khối cosine hoà tuyệt đối (doc trùng vector) ở hạng 211 theo thứ tự khác nhau.
  - 186 cặp đó được chấm riêng (`pool-rdoc-tie`). Cả 186 đều là doc thin không có chunk.
- **R-doc:** (query, `doc_text` S31). `doc_text` đã cắt 512 token Qwen3; cặp đầu vào lại cắt 512 token bge, và 34% cặp chạm trần này.
- **R-chk:** max logit trên top-3 chunk t256 của doc, chọn chunk đúng cách S32 (cosine giữa vector query S29 và vector chunk, hoà thì giữ thứ tự file).
  - Cặp đã có trong pool S32 giữ nguyên dòng S32.
  - 20.299 cặp pool-extra được chọn lại bằng helper S32; 451 chunk chưa có vector được embed bằng `M3Encoder`.
  - **Cổng:** chọn lại 2.000 cặp S32 bằng đường code mới cho 0 lệch.
  - 3.742 cặp không có chunk (doc thin) lấy R-doc làm R-chk.
- **Bản dựng:** top-40 theo R-doc hoặc R-chk, hoà thì theo hạng cosine rồi doc_id. Không trộn cosine. Mỗi doc được chọn có text nhận đúng một chunk nguyên body, không cắt, như J4-K40.
- **Cổng hồi quy:** `Check-J4-K40` = RRF(cosine, 4B) top-40 dựng qua đường S33. JSON ra `e464bbd1…cafc`, đúng JSON của J4-K40; 1.200/1.200 query giống hệt. ZIP dài hơn 12 B do tên member dài hơn. Như vậy BGR chỉ khác J4-K40 ở phần chọn doc.

## Chạy GPU (RTX 3050 6 GB, transformers 5.18.0, torch 2.11.0+cu128)

| | Cặp | Pilot 1.000 cặp | Chấm full (giây GPU) | Token TB |
|---|---:|---:|---:|---:|
| R-doc | 253.200 + 186 tie | 47,4 cặp/s | 5.615,7 + 0,5 | 441,6 |
| R-chk | 737.254 dòng chunk | 97,9 cặp/s | 7.191,9 | 209,4 |

- ETA local khoảng 3,6 giờ, dưới ngưỡng 6 giờ, nên không cần Kaggle.
- Một lần mọi tiến trình con bị kill lúc 14:52 (không có lỗi trong log, đang ở 73/139 part R-doc). Supervisor `scripts/s33/run_score.sh` chạy lại và resume theo part. Không có OOM, không có NaN.
- File điểm và SHA256:
  - `gpu-rdoc/.../scores-all.parquet`: `1408e07a…befc`
  - `gpu-rchk/.../scores-all.parquet`: `413c0826…538c`
  - `gpu-rdoc-tie/.../scores-all.parquet`: `6c47a6e7…cae`

## Tương quan (Spearman, 253.200 cặp pool211)

| Cặp | Toàn cục | TB/query | p10/query |
|---|---:|---:|---:|
| R-doc ~ R-chk | 0,879 | 0,743 | 0,613 |
| R-doc ~ Qwen 4B | 0,690 | 0,610 | 0,399 |
| R-doc ~ cosine | 0,627 | 0,430 | 0,242 |
| R-chk ~ Qwen 4B | 0,653 | 0,542 | 0,326 |
| R-chk ~ cosine | 0,619 | 0,379 | 0,203 |
| (tham chiếu) 4B ~ cosine | 0,499 | 0,355 | 0,173 |

Chunk tốt nhất của R-chk theo hạng cosine trong 3 chunk đã chọn: hạng 1 48,5%, hạng 2 32,9%, hạng 3 18,5%. Theo loại field: body 73,8%, title 15,1%, answer 7,4%, question 3,7%.

## Bản dựng (CPU, chưa nộp)

Cả hai bản: 1.200 query, 48.000 doc (40/query), không cắt body. Validator S26 (`scripts/s26/validate_pa3.py` + lô Pharmacity) báo `ok`, 0 lỗi.

| Bản | Chunk | Chunk/query min / p10 / median | Doc vi / zh | ZIP B | ZIP SHA256 | JSON SHA256 |
|---|---:|---|---|---:|---|---|
| J4-K40 (tham chiếu) | 47.512 | 9 / 39 / 40 | 82,9% / 17,1% | 30.091.475 | `3b706c55…32f4` | `e464bbd1…cafc` |
| **BGR-K40-doc** | 47.527 | 22 / 39 / 40 | 88,2% / 11,8% | 30.614.714 | `be362e5e1ae94353cdca3933f218287ba0d22fbc997c59c7527ba21aa15c50de` | `b17a3664…3099` |
| **BGR-K40-chk** | 47.792 | 22 / 39 / 40 | 88,5% / 11,5% | 32.857.489 | `de2bac96a377391c00fefe98815c195140fd3a1d14aa5c08f8acfda1928278c3` | `5b6c29f0…c7fe` |

Doc chỉ có trong bản thứ nhất, TB/query (tổng 40 doc nên hai chiều bằng nhau; sym-diff = 2×):

| | BGR-K40-chk | J4-K40 | COS-K40 |
|---|---:|---:|---:|
| BGR-K40-doc | 15,44 | **17,04** | 20,51 |
| BGR-K40-chk | – | **20,18** | 22,66 |
| J4-K40 | – | – | 11,63 |

Không query nào của BGR trùng hoàn toàn với J4-K40.

**Nhận xét (chưa có LB):**
- bge-reranker đổi khoảng 17–20 trong 40 doc so với J4-K40, nhiều hơn mức J4-K40 đổi so với cosine (11,6).
- bge chọn ít doc zh hơn hẳn (11,5–11,8% so với 17,1%), có thể do query tiếng Việt + doc tiếng Trung cho logit thấp hơn.
- Output: `out/runs/s33-btc-baseline/build/` (`<bản>.s33.json` có SHA file điểm, diff, tương quan, profile).

## Cách đọc LB

Ba bản cùng pool, cùng K = 40, cùng quy tắc chunk; chỉ khác nguồn xếp hạng (4B+cosine RRF, R-doc, R-chk).

- **Ngưỡng của Pha A:** Doc P ≥ 0,55 thì làm Pha B (BM25 + dense trên chunk → RRF → rerank bge → gộp doc, sweep K/ngưỡng theo quy tắc F/5).
- Nếu Doc P của BGR ≈ 0,376 (J4-K40) hoặc thấp hơn: rerank bằng bge trên pool cosine không tái tạo được gold. Giả thuyết yếu đi, nhưng chưa loại hẳn, vì pool BTC có thể khác (BM25 + chunk-level).
- BGR-K40-chk so với BGR-K40-doc cho biết BTC chấm theo chunk hay theo đầu doc.

## Pha C (en): chưa làm

Chờ Vu quyết. Lý do: E1 (15/10) đã nộp doc en chọn bằng bge-reranker-v2-m3 mà Doc R không đổi, và PROGRESS ghi đã chốt "dừng thêm probe en".

## File và lệnh

- Code: `src/r2ai/s33/bge_score.py` (chấm, dùng lại part/lock/collect của `r2ai.s31.judge_score`), `pool.py` (`rdoc`, `rchk`, `tie`), `build.py`.
- Supervisor: `scripts/s33/run_score.sh`. Test: `tests/test_s33_btc_baseline.py`.

```bash
python -m r2ai.s33.pool rdoc
python -m r2ai.s33.pool rchk
python -m r2ai.s33.pool tie
bash scripts/s33/run_score.sh pool-rdoc pool-rchk pool-rdoc-tie
python -m r2ai.s33.bge_score collect --pool out/runs/s33-btc-baseline/pool-rdoc --out out/runs/s33-btc-baseline/gpu-rdoc
python -m r2ai.s33.build --variant BGR-K40-doc --variant BGR-K40-chk
```
