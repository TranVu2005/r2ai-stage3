# S38: chọn chunk theo metric chunk mới (BTC 23/10/2026)

Nhánh `feat/s38-chunk-metric-v2`, base `feat/s36-fuse4` (`e5567b4`). Mọi bản **chưa nộp**.

## Metric mới

BTC đổi (23/10): chunk dự đoán c khớp gold g khi `LCS_tokens(c,g)/|g| >= 0,4` **và** `LCS_tokens(c,g)/|c| >= 0,4`, vẫn so trong cùng query + doc_id, chuẩn hoá như cũ. `r2ai.eval.scorer` dùng v2 mặc định (`--chunk-metric v1` giữ cách cũ), so ngưỡng bằng số nguyên (5·LCS ≥ 2·|g| và 5·LCS ≥ 2·|c|). 7 test mới trong `tests/test_scorer.py` (biên 40% hai phía, chunk dài trượt, chunk ngắn vẫn cần vế gold).

Số Vu cung cấp (23/10): K4-thr-C-extD (chunk JSON giống K4-thr-C) chấm theo metric mới Chunk F2 **0,3594** (cũ 0,4227; P 0,225 / R 0,469). Nguyên body vẫn đúng khoảng 85% vì gold chunk dài; chỉ doc dài trượt vế LCS/|c|. Mốc so: 0,3594. CH-full (J3-thrC nguyên trạng) không cần nộp.

## Nền và doc nhận chunk

`relevant_docs` = J3-thrC (LB 0,4190), giữ nguyên từng query (`docs_equal_j3` đúng ở mọi bản). Doc nhận chunk = đúng 175.148 cặp (query, doc) mang chunk của J3-thrC (P4B ≥ t_c = 0,079210), 114.500 doc, 114.387 owner canonical. J3-thrC đọc từ ZIP, pin SHA ZIP `546b6307…3033` và JSON `e11e51e6…e500`.

**Độ dài body của doc nhận chunk** (chunk J3-thrC, token BGE-M3 thô): TB 1.056, P10 264, median 999, P90 1.798, max 23.442. **≤ 384: 16,8% · ≤ 768: 38,7% · > 768: 61,3%.** Theo ngôn ngữ: vi (146.412) ≤ 384 10,0%, ≤ 768 30,5%; zh (28.736) ≤ 384 51,5%, ≤ 768 80,4%.

## Nguồn chunk (`python -m r2ai.s38.sources`, CPU 43 phút)

Mọi chunk là đoạn nguyên văn của `doc_text` canonical (`data/canonical-20261018`), chỉ field answer/body (title/question không nộp).

- **t256**: `chunks_t256` canonical của owner.
- **t128**: cùng chunker (`r2ai.index.chunk.chunk_paragraphs`, overlap 1 đoạn, bỏ blob) target 128, trên đoạn văn dựng lại từ doc_text (vùng body tách theo `"\n\n"`, field theo chunk canonical chứa đầu đoạn). **Cổng: dựng lại với target 256 ra đúng chunk canonical ở 114.494/114.500 doc (99,995%; vi 100%, zh 99,975%).**
- **cấu trúc**: chunker S12 (`r2ai.vi_quality.chunker.structured_chunks`, target 256, heading của extractor docs_vi cho 71.658 doc base). Doc không cấu trúc (không có đoạn heading/item/row và không có field question trong body; quyết định của Vu) lấy t256: 40.471 doc; có cấu trúc 74.029 (vi 68,7%, zh 49,7%).

| Loại | Chunk | Token thô TB | P10 | median | P90 | max | chunk/doc TB | doc không có chunk |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t256 | 680.138 | 208,9 | 148 | 222 | 252 | 384 | 5,94 | 147 |
| t128 | 1.339.006 | 101,2 | 66 | 106 | 127 | 192 | 11,69 | 261 |
| cấu trúc | 710.190 | 175,2 | 80 | 190 | 247 | 1.175 | 6,20 | 4 |

## Điểm chọn chunk (`python -m r2ai.s38.score select`)

- t256: judge S32 chunk-level (Qwen3-Reranker-0,6B, top-3 chunk/doc theo cosine, pool13) trên các chunk answer/body đã chấm (quyết định của Vu: chỉ xếp chunk body/answer đã chấm, cùng thang); carrier không có chunk body nào đã chấm thì cosine BGE-M3 trên mọi chunk answer/body. 174.926 carrier dùng judge, 36 dùng cosine, 186 carrier không có chunk t256 answer/body (giữ chunk J3-thrC).
- cấu trúc / t128: cosine(vector query S29, vector chunk), một thang mỗi doc. Chunk trùng span t256 dùng vector canonical; còn lại nhúng bằng `M3Encoder` (max_len 512, fp16): cấu trúc 404.707 text (63,3 M token), 10.061 s GPU local (chia GPU với S36 phần lớn thời gian). t128 hạ ưu tiên (gold không ngắn): chưa nhúng (1.292.602 text cần nhúng), chưa dựng CH-t128-1.
- Doc không cấu trúc dùng chunk t256 với điểm judge (58.960 carrier); có cấu trúc dùng cosine (116.184). CH-sec-1 trùng text CH-t256-1 ở 42,9% carrier.
- Chunk của một carrier trùng nhau sau chuẩn hoá metric chỉ tính một lần (giữ chunk điểm cao nhất); thiếu bước này CH-t256-2 có 59 lỗi `chunk_dup_text_doc`.

## Bản dựng (`python -m r2ai.s38.window`, `python -m r2ai.s38.build`)

CH-win: body = chunk J3-thrC của carrier (định vị nguyên văn trong doc_text: 173.306; 1.842 không định vị được thì lấy span body canonical). Body ≤ W token → giữ nguyên chunk J3. Dài hơn → W token liên tục căn giữa chunk t256 tốt nhất, kẹp trong body, rồi cắt theo ranh giới câu (đầu tiến tới đầu câu, cuối lùi về cuối câu, mỗi bên tối đa W/4 token, không cắt vào chunk). Ranh giới: sau `. ! ? … ; :` + khoảng trắng, sau `。！？；`, xuống dòng. 26 carrier có chunk tốt nhất nằm ngoài body (cửa sổ căn giữa body), 186 không có chunk t256 (căn giữa body).

| Bản | Chunk | Body nguyên | Cửa sổ | Token thô TB / median / max | Token metric TB | ZIP B | ZIP SHA256 | JSON SHA256 | Validator S26 |
|---|---:|---:|---:|---|---:|---:|---|---|---|
| CH-full (J3-thrC) | 175.148 | 175.148 | 0 | 1.056 / 999 / 23.442 | 1.045,5 | 101.181.680 | `546b6307…3033` | `e11e51e6…e500` | ok (S35) |
| **CH-win768** | 175.148 | 67.829 | 107.319 | 620,8 / 735 / 769 | 615,0 | 61.627.691 | `4b7dfa6667658a073417aa27d5f9db158f956200ea500e146bca6561c850abc2` | `05c1ded5…909e` | ok, 0 lỗi |
| **CH-win384** | 175.148 | 29.391 | 145.757 | 337,1 / 359 / 385 | 334,2 | 37.016.768 | `e1d652e5f2371c73f2defee56a398ad6e053d1772495964cfb922ec2917051c2` | `f9c77c25…1792` | ok, 0 lỗi |
| **CH-sec-1** | 175.148 | – | – | 192,0 / 207 / 1.175 | 190,5 | 23.625.667 | `8c29909adeb2b0a41a671011c5a8d8eda1e9a7ac15124264d43375e6675bd14f` | `84236a5c…30cf` | ok, 0 lỗi |
| **CH-t256-1** | 175.148 | – | – | 209,4 / 223 / 1.581 | 207,6 | 25.385.911 | `80eac412ee6f8f346950c391ca4d599f3c4f10d7f13cdad58cdc9286d4e970ae` | `89a14c67…8e5b` | ok, 0 lỗi |
| CH-t256-2 | 281.380 | – | – | 211,7 / 224 / 1.581 | 209,9 | 35.892.207 | `41f0be149a790c48593545ed1147af153a67583bed3cd424190d98d68e92075d` | `244fc390…9028` | ok, 0 lỗi (`--max-chunks-per-doc 2`) |

- Cửa sổ cắt theo câu: win384 đầu 70,0% / cuối 85,4% cửa sổ; win768 52,7% / 81,5%. Đếm lại token sau khi cắt: 228 (win384) / 62 (win768) chunk dài hơn W đúng 1 token (token hoá lại tại biên khác token hoá cả body).
- CH-t256-2: 106.232 carrier có chunk thứ 2 (điểm ≥ 0,5 × chunk 1, cùng thang).
- Carrier không có chunk nguồn giữ chunk J3-thrC (nên tập doc nhận chunk giữ đúng J3-thrC): CH-t256-1/2 186 (max 1.581 token là từ đây), CH-sec-1 4, CH-win 0.
- File ở `out/runs/s38-chunk-metric-v2/build/` (ZIP, `*.validate.json`, `build.json`); nguồn ở `sources/`, điểm và cửa sổ ở `scores/`.

Validator S26 (`scripts/s26/validate_pa3.py`): thêm `--max-chunks-per-doc N` (mặc định 1, hành vi cũ) và lỗi `chunk_dup_text_doc` (hai chunk một doc trùng sau chuẩn hoá metric).

**Thứ tự nộp (Vu):** CH-win768 → CH-win384 → CH-sec-1 → CH-t256-1. Mốc 0,3594.

## Điểm LB lượt 1 (Vu, 23/10, metric mới, doc J3-thrC)

Chunk F2: **CH-win768 0,3433 · CH-win384 0,1805 · CH-sec-1 0,0904**; nguyên body (K4-thr-C) 0,3594. Kết luận của Vu: gold chunk dài (phần lớn > ~1.000 token), nguyên body tốt nhất; bỏ chunk ngắn (CH-t256-*, CH-t128-1 không nộp), **hoãn hẳn bước 5** (job judge giữ ở dạng chuẩn bị, không chạy).

## Lượt 2: cắt body dài (CPU, chưa nộp)

`python -m r2ai.s38.window --variant CH-pre1024 --variant CH-cap2048 --variant CH-full+w768` rồi `python -m r2ai.s38.build` cùng ba tên. Carrier không đổi giữ nguyên chunk J3-thrC (byte), nên phần còn lại giống CH-full.

- **CH-pre1024**: body > 1.024 token → 1.024 token đầu body, cuối lùi về cuối câu (tối đa 256 token; cắt theo câu ở 99,9% bản bị cắt); còn lại nguyên body.
- **CH-cap2048**: body > 2.048 token → cửa sổ 2.048 token quanh chunk t256 tốt nhất (cách cắt như CH-win); còn lại nguyên body.
- **CH-full+w768**: body > 1.536 token → 2 chunk: nguyên body + cửa sổ 768 quanh chunk t256 tốt nhất (luôn khác body); còn lại 1 chunk nguyên body. Validator `--max-chunks-per-doc 2`.

Body của carrier: ≤ 1.024 51,1% · ≤ 1.536 78,6% · ≤ 2.048 95,1%.

| Bản | Chunk | Carrier đổi | % carrier đổi (vi / zh) | % doc đổi | Token TB / median / max | ZIP B | ZIP SHA256 | JSON SHA256 | Validator S26 |
|---|---:|---:|---|---:|---|---:|---|---|---|
| **CH-pre1024** | 175.148 | 85.622 | 48,9% (56,1% / 11,9%) | 47,0% | 767 / 973 / 1.025 | 71.823.922 | `5e827c71fdce15e71ff6b161b780df029daf320127f7e68d76b69536954d21d6` | `c699214b…9e3c` | ok, 0 lỗi |
| **CH-cap2048** | 175.148 | 8.500 | 4,9% (5,3% / 2,6%) | 4,8% | 1.019 / 999 / 2.049 | 97.778.445 | `113f9a28ec0a307600e88f336ea86c892d3159c3bba2c16c5c097a23a07e04e3` | `1884a6ca…a3c2` | ok, 0 lỗi |
| CH-full+w768 | 212.582 | 37.434 | 21,4% (24,6% / 4,9%) | 20,6% | 1.001 / 780 / 23.442 | **104.802.138** | `20a9a78c…fd96` | `bf82d6c2…f335` | **lỗi `zip_over_ceiling`** (trần nội bộ 102.600.000 B; dưới trần BTC 104.857.600 B 55.462 B) |

Token = BGE-M3 thô của từng chunk (chunk giữ nguyên tính bằng độ dài body). Vượt W đúng 1 token khi token hoá lại: pre1024 13, cap2048 7, full+w768 12 chunk. Carrier có body không định vị được trong doc_text (1.842) mà bị cắt thì lấy đoạn từ span body canonical: pre1024 1.814, cap2048 243, full+w768 983 chunk.

**% carrier đổi theo 10 host nhiều carrier nhất** (pre1024 / cap2048 / full+w768): suckhoedoisong.vn 47,1 / 2,4 / 11,3 · nhathuoclongchau.com.vn 90,3 / 7,4 / 53,7 · thanhnien.vn 12,4 / 0,4 / 1,5 · medlatec.vn 99,1 / 2,3 / 34,0 · vinmec.com 56,2 / 10,9 / 30,7 · ask.39.net 0,6 / 0,1 / 0,2 · suckhoecongdongonline.vn 25,6 / 2,1 / 6,9 · tiemchunglongchau.com.vn 97,5 / 17,1 / 64,2 · 120ask.com 6,2 / 1,3 / 2,7 · cnkang.com 16,6 / 2,9 / 6,8.

Độ dài body theo host (offline) và đuôi 300 ký tự của 10 host có body trung vị dài nhất: `reports/s38-host-bodies.md` (`python -m r2ai.s38.window --hosts`).

## Lượt 3 (CPU, chưa nộp; mốc Chunk F2 0,3594)

**CH-full+w768 dưới trần (Vu: nâng ngưỡng body, giữ cửa sổ 768, không cắt body).** Cửa sổ của mọi body > 1.536 token đã có từ lượt 2, nên chỉ lọc chunk thứ 2 theo ngưỡng T (`python -m r2ai.s38.build --variant CH-full+w768 --full-threshold T`, ZIP tên `CH-full+w768-t<T>.zip`). Ước lượng tuyến tính (96,7 B/chunk thứ 2) cho T ≈ 1.858; dựng thật:

| T | Chunk thứ 2 | ZIP B | So với trần 102.600.000 |
|---:|---:|---:|---|
| 1.848 | 15.083 | 102.602.464 | vượt 2.464 B |
| **1.849** | **15.026** | **102.599.152** | **dưới 848 B** |
| 1.850 | 14.979 | 102.596.108 | dưới 3.892 B |
| 1.870 | 14.132 | 102.541.183 | dưới 58.817 B |

**Ngưỡng thấp nhất T = 1.849 token**: 15.026 carrier nhận chunk 2 (cửa sổ 768), 190.174 chunk; ZIP `out/runs/s38-chunk-metric-v2/build/full-search/t1849/CH-full+w768-t1849.zip` SHA256 `393b15acf79c672fe4444ea3b77803fbbe9a3958a9ab36d39a2b6517a7961be6`, JSON `559396d2…6a55`; validator S26 (`--max-chunks-per-doc 2`) ok, 0 lỗi. Ngưỡng chính xác tới 1 token theo ZIP thực (kích thước ZIP gần như đơn điệu theo T ở vùng này, đo 4 điểm).

**CH-clean** (`python -m r2ai.s38.clean discover|apply`, `build --variant CH-clean`): nguyên body, cắt đuôi boilerplate theo marker đếm trên toàn host (≥ 30% doc, trong 30% cuối body). 25.391 carrier bị cắt (tamanhhospital 2.320, baophutho 77, a-hospital 462, nhathuoclongchau 21.128, youmed 1.404; **medlatec 0**: đoạn quảng cáo diễn đạt khác nhau theo bài, không khoá dòng nào đạt 30% doc). Token bỏ trung vị theo doc: tamanhhospital 394, baophutho 379, a-hospital 19, nhathuoclongchau 55, youmed 189. Carrier còn lại giữ nguyên byte J3-thrC (149.757). ZIP 100.650.275 B, SHA256 `9c29f885528209bd07a346c86d4a1f8f2d04baca2feaf245d0d672d513535721`, JSON `51b89803…a620`; validator S26 ok, 0 lỗi. Marker, số doc bị cắt và 2 ví dụ trước/sau mỗi host: `reports/s38-clean.md`.

## Bước 5: judge 0,6B chunk-level cho chunk t256 chưa chấm (chuẩn bị; Vu hoãn hẳn, không chạy)

`python -m r2ai.s38.judge_prep` → `out/runs/s38-chunk-metric-v2/judge-pool/` (layout shard S32, prompt/instruction S29). Doc nhận chunk có 1.260.431 chunk t256 (mọi field), S32 đã chấm 519.343 → còn 741.088; chỉ xuất answer/body: **670.779 cặp** (body 663.310, answer 7.469; vi 631.573, zh 39.206) ở 138.533 carrier (TB 4,8 cặp/carrier), prompt TB 334 token (S32: 285), 2 chunk cắt về 512 token. **ETA GPU local (RTX 3050 6 GB, HF bf16, 25,5 cặp/s đo ở S32): 7,3 h; quy theo độ dài prompt 8,6 h**, nếu GPU không chia với job khác. Lệnh: `python -m r2ai.s31.judge_score score --pool out/runs/s38-chunk-metric-v2/judge-pool --out out/runs/s38-chunk-metric-v2/judge-gpu --model qwen3-reranker-0.6b --subset all`.

## Test

`pytest tests`: 906 pass / 2 deselected (587 s); `tests/test_s38_chunks.py` 14 test (dựng lại đoạn văn, cổng t256, cấu trúc, chọn chunk, cửa sổ, ghi lại hàng).

## Chưa làm

- CH-t128-1 (hạ ưu tiên): cần nhúng 1.292.602 chunk t128.
- Ghi chú để thử sau (Vu): doc hỏi-đáp, chunk answer là một phần của gold giữ cặp hỏi-đáp; nếu CH-t256-1 thấp bất thường ở host hỏi-đáp (ask.39, 120ask, vinmec) thì thử ghép question.
