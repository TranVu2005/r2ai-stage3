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

## Ghép cuối: docsel bất kỳ + quy tắc chunk (`r2ai.s38.compose`, chuẩn bị cho sáng 24/10)

Vu (23/10): mọi bản ghép cuối dựng trên docsel EF2 (S36), không phải J3-thrC; CH-full+w768 phải đo lại ZIP và hiệu chỉnh lại T.

```bash
scripts/s38/compose.sh EF2-cut-avg clean        # hoặc EF2-cut-p4b; RULE = clean | cap2048 | full+w768
```

= `python -m r2ai.s38.compose run --name <DOCSEL>-<RULE> --base-zip out/runs/s36-fuse4/build-docread/<DOCSEL>-docread.zip --docsel out/runs/s36-fuse4/docsel/docsel-<DOCSEL>.parquet --rule <RULE>` → `out/runs/s38-chunk-metric-v2/compose/`.

- Nền = ZIP `-docread` của S36 (chunk nguyên body, dựng sau bản sửa Pharmacity). Script kiểm `relevant_docs` (thứ tự `rank`) và tập doc mang chunk khớp docsel, lệch thì dừng.
- `clean`: marker đuôi đếm trên J3-thrC (`scores/clean_markers.json`) + **marker base64 cho baophutho.vn** (dòng ≥ 200 ký tự `[A-Za-z0-9+/=]`, cắt từ đó trở đi; `--no-base64` để tắt). `cap2048` / `full+w768`: cửa sổ quanh chunk t256 tốt nhất, chọn như CH-t256-1 (judge S32 → cosine; doc Pharmacity dùng chunk lô của nó).
- `full+w768`: script tự chọn T. Ước lượng T từ ngân sách (102.600.000 − ZIP nền) / 94,3 B mỗi chunk thứ 2 (đo ở CH-full+w768-t1849), dựng song song 3 ứng viên, rồi chia đôi giữa T vượt trần cao nhất và T dưới trần thấp nhất cho tới khi cách nhau 1 token (tối đa `--rounds 3`). Kết quả trong `<name>.threshold.json`. **Tên file trong ZIP đổi độ dài thì ZIP đổi vài byte** (gate-full-t1849 102.599.146 B so với CH-full+w768-t1849 102.599.152 B), nên T được đo với đúng tên sẽ nộp.
- Thời gian: `prepare` (đọc nền, chọn chunk t256, token hoá body) ~1–2 phút cho clean, 7–15 phút cho quy tắc cửa sổ; mỗi ZIP (LZMA dict 256) ~10–15 phút; full+w768 thêm 1–3 vòng dựng song song.
- **Cổng hồi quy** (trên J3-thrC + docsel-J3-thrC, đã chạy): JSON SHA khớp đúng CH-clean (`--no-base64`) `51b89803…a620`, CH-cap2048 `1884a6ca…a3c2`, CH-full+w768-t1849 `559396d2…6a55`; ước lượng T trên J3-thrC ra đúng 1.849.

### Quyết định 24/10 (Vu) và cache sẵn

- **Nền ghép: EF2-cut-p4b** (đã có LB). Không dựng EF2-cut-avg; cache EF2 cũ (chưa có thin) dời sang `compose/superseded-nothin-20261024/`.
- **Thin theo nội dung, mọi host** (bật mặc định, `--no-thin` để tắt; Vu 24/10 đổi từ lọc theo host baophutho.vn): carrier có chunk gốc mà **≥ 50% ký tự là dòng base64 (`^[A-Za-z0-9+/=]{200,}$`) hoặc token placeholder `{\w+}`** (`compose.is_thin`, `THIN_JUNK_SHARE`) thì mất chunk, chunk chuyển cho doc kế theo **thứ tự toàn cục của bộ chọn t_c S31** (P4B giảm dần, rồi query, rồi rank) trong số doc chưa mang chunk mà có nguồn chunk như builder S29 (chunk P1 của query → chunk P1 đầu tiên của doc → body span của doc có chunk canonical / doc lô Pharmacity), không rỗng và không thin. Độ trung thực của nguồn: dựng lại nguồn S29 cho 175.548 carrier của EF2-cut-p4b ra **trùng từng byte cả 175.548**; 749/772 doc có P4B ≥ t_c mà không mang chunk đúng là không có nguồn, 23 còn lại hoà đúng t_c.
- Trên EF2-cut-p4b: 24 carrier thin (20 doc) bị bỏ chunk; 24 doc nhận thay (23 doc hoà đúng t_c 0,087564, 1 doc P4B 0,086942), nguồn: P1 bất kỳ 10, body 9, P1 cùng query 5 (chi tiết `compose/EF2-cut-p4b-<rule>.swap.json`). t_c thực tế 0,087564 → 0,086942; tổng chunk giữ 175.548.
- **Quy tắc kết hợp `clean+cap2048`** (chuẩn bị, chờ Vu): clean trước (cùng marker), rồi cap2048 trên body đã cắt (cửa sổ 2.048 quanh chunk t256 tốt nhất khi body đã cắt vẫn > 2.048 token).
- **Đã ghép EF2-cut-p4b + full+w768 (24/10; Vu chỉ yêu cầu bản EF2-cut-p4b + quy tắc w768-t1849; chưa nộp)**: T thấp nhất dưới trần đã đo là **1907** (vòng 1: 1908 / 1917 / 1929 dưới trần; vòng 2: 1883 / 1898 trên; vòng 3: 1901 / 1904 trên, 1907 dưới; 1905–1906 chưa đo). ZIP `out/runs/s38-chunk-metric-v2/compose/EF2-cut-p4b-full+w768-t1907.zip` 102.593.008 B, SHA256 `af427ae3…1264`, JSON 964.570.736 B SHA256 `b100adf3…d688`; 1.200 id, 193.879 doc, 188.185 chunk (12.637 chunk thứ 2), validator S26 (`--max-chunks-per-doc 2`) 0 lỗi (95.711 chunk trùng cặp P1, 92.474 chunk là substring canonical). Thin bật. Nền: `EF2-cut-p4b-docread.zip` SHA256 `a5e66840…e6d9`, docsel `docsel-EF2-cut-p4b.parquet` SHA256 `db94540a…5454`; probe đối chiếu `CH-full+w768-t1849.zip` SHA256 `393b15ac…1be6`. relevant_docs khớp từng phần tử với EF2-cut-p4b (0 khác biệt). Phân phối so với probe t1849 (nền J3): chunk/query TB 156,8 vs 158,5, trung vị 147,5 vs 151; độ dài chunk trung vị 3.255 vs 3.222 ký tự.
- **Khác probe t1849 ở 3 yếu tố, nên điểm không so trực tiếp với 0,3636 được**: (1) docsel EF2-cut-p4b thay vì J3-thrC; (2) T=1907 thay vì 1849: doc có body 1.849–1.907 token chỉ còn full body, không còn cửa sổ 768; (3) **thin bật**, probe t1849 **không** có thin (dựng trước khi thin tồn tại). Thin: 24 carrier (20 doc baophutho.vn) có chunk chỉ gồm placeholder `{title}`… trước khối base64 bị bỏ chunk; 24 doc kế theo thứ tự P4B toàn cục (chưa mang chunk, có nguồn chunk như builder S29; 23 doc hoà đúng t_c 0,087564, 1 doc P4B 0,086942) nhận chunk thay; tập relevant_docs không đổi, tổng chunk giữ 175.548 chunk hạng 1.
- **T=1907 do agent tự chọn, vì ràng buộc kích thước**: T=1849 trên nền EF2 ước lượng vượt trần (T cần ≈ 1.917 theo ngân sách còn lại, trần 102.600.000 B). Phiên này hỏi nhanh qua hộp thoại chọn 'Hiệu chỉnh T' (đề xuất của agent); Vu xác nhận không chỉ định việc này.
- **Trần kích thước = một tham số (Vu 24/10)**: ZIP phải < `BTC_CEILING − reserve`, với `BTC_CEILING` = **104.857.600 B = 100 MiB** (trần BTC cho file mới; nguồn: `docs/PROGRESS.md`, mục giới hạn upload, «Trần dùng cho file mới: 100 MiB = 104.857.600 byte»; LB từng nhận ZIP 62,05 MiB, giới hạn thật chưa biết chính xác) và `reserve` mặc định `RESERVE` = **2.257.600 B** (`--reserve`) nên giới hạn mặc định vẫn **102.600.000 B**, trùng `CEILING` cũ. Không còn `CEILING` riêng trong `r2ai.s38.compose` (`build.py` của lượt J3 giữ hằng nội bộ cũ). `compose run` tìm T nhỏ nhất thoả giới hạn đó, không chỉnh tay từng bản; dùng được cho S35 và docsel khác. **Bản T=1907 (102.593.008 B) hợp lệ theo mặc định này** (có test), không dựng lại. (Bản sửa trước, reserve 500.000 B ⇒ 102.100.000 B, đã bị thay bằng cách trên.)
- **Quy tắc thin theo nội dung: ngưỡng và kiểm** — đo trên 175.548 carrier của EF2-cut-p4b: thin (có base64) có tỷ lệ junk 0,79–0,96; mọi carrier còn lại ≤ 0,059; không carrier nào nằm giữa, nên ngưỡng 0,5 bắt đúng nhóm này (0,3–0,7 cho cùng 24 carrier; 0,8 bỏ sót 1; 0,9 bỏ sót 6). Không dùng tiêu chí «tỷ lệ ký tự không phải chữ»: ngưỡng 0,4 bắt 67 carrier, 0,5 bắt 5, không cái nào là base64 (bảng số, mã, công thức; base64 phần lớn là chữ cái nên tiêu chí này cũng không phân biệt được base64). **Kiểm diff trên docsel EF2-cut-p4b = 0**: quy tắc mới bắt đúng 24 carrier của 20 doc như quy tắc host cũ, 24 doc nhận thay trùng từng byte (kể cả body), t_c 0,087564 → 0,086942; cache `*.swap.json` giữ nguyên hiệu lực (`compose/thin-content-rule-check.json`).
- **Đếm thin trên toàn corpus (chỉ đếm, không dựng; quy tắc áp lên `doc_text` cả doc, không phải body span của carrier)**: canonical-20261018 887.151 doc → **979 doc thin**: baophutho.vn 627 (tỷ lệ junk 0,69–0,97, trung vị 0,92; doc không thin của host ≤ 0,02) và **baothanhhoa.vn 352** (0,57–0,96, trung vị 0,90; không thin ≤ 0,003; host mới so với quy tắc cũ, cùng dạng khối base64 ở đầu). 120ask S37 (6 segment `data/s37_120ask`, 55.991 doc) → **0 doc thin**. Không doc baothanhhoa nào là carrier của EF2-cut-p4b, nên diff vẫn 0 (`compose/thin-content-rule-corpus-count.json`).
- **Quyết định của Vu (24/10): giữ quy tắc thin theo nội dung (`THIN_JUNK_SHARE` = 0,5) cho mọi host, KHÔNG thêm bộ lọc host.** Quy tắc bắt thêm **352 doc baothanhhoa.vn** (dạng khối base64 đầu doc + câu tiếng Việt thật ở cuối). **Đã biết và chấp nhận**: carrier thin mất chunk, kể cả phần chữ thật ở cuối doc; tác động không đáng kể (trên EF2-cut-p4b không có carrier baothanhhoa nào, chỉ 24 carrier baophutho). Phần chữ thật ở cuối doc thin là việc để mục «Hướng cải tiến» bên dưới.
- **Việc cho S35 khi dựng EF2-free**: báo **số carrier bị thin và host của chúng** trong docsel mới (`compose` ghi `thin` vào `<name>.prepare.json` và danh sách ở `<name>.swap.json`). **Nếu > 50 carrier thì báo Vu trước khi đóng gói.**
- Các quy tắc khác (`clean`, `cap2048`, `clean+cap2048`) vẫn có cache sẵn, chưa ghi ZIP: `scripts/s38/compose.sh EF2-cut-p4b <rule>`.
- **Cổng hồi quy sau khi thêm thin / clean+cap2048** (`--no-thin`, J3-thrC): cache clean và cap2048 trùng hệt cache cổng trước (đã khớp JSON SHA CH-clean, CH-cap2048), nên JSON không đổi.

| Cache (EF2-cut-p4b, có thin) | Carrier | Carrier đổi | prepare | Ghi chú |
|---|---:|---:|---:|---|
| EF2-cut-p4b-clean | 175.548 | 25.232 | 93 s | nhathuoclongchau 21.028, tamanhhospital 2.302, youmed 1.389, a-hospital 460, baophutho 53 |
| EF2-cut-p4b-cap2048 | 175.548 | 8.501 | 506 s | |
| EF2-cut-p4b-full+w768 | 175.548 | 37.355 | 435 s | T ước lượng 1.917 (ứng viên 1.908, 1.917, 1.929; ngân sách 1.155.104 B theo ZIP nền) |
| EF2-cut-p4b-clean+cap2048 | 175.548 | 31.142 | 419 s | cắt như clean; mọi body tamanhhospital không định vị được đều định vị được sau khi cắt |

**baophutho.vn, marker base64 (Vu 23/10).** Khối base64 có ở 21/61 doc, bắt đầu ở 0,1–0,4% body và chiếm 79–95% body. Đó là trang lỗi trích xuất: body = `{title}\n\n{publish}\n\n{head}` (placeholder) + base64 + khung bình luận + tin khác, không có bài viết. Cắt từ base64 trở đi còn lại đúng chuỗi `{title}\n\n{publish}\n\n{head}`. Ngày 24/10 Vu chọn coi các doc này là thin (xem trên); cắt base64 vẫn còn trong clean nhưng không còn chạm doc nào của EF2-cut-p4b.

## Bước 5: judge 0,6B chunk-level cho chunk t256 chưa chấm (chuẩn bị; Vu hoãn hẳn, không chạy)

`python -m r2ai.s38.judge_prep` → `out/runs/s38-chunk-metric-v2/judge-pool/` (layout shard S32, prompt/instruction S29). Doc nhận chunk có 1.260.431 chunk t256 (mọi field), S32 đã chấm 519.343 → còn 741.088; chỉ xuất answer/body: **670.779 cặp** (body 663.310, answer 7.469; vi 631.573, zh 39.206) ở 138.533 carrier (TB 4,8 cặp/carrier), prompt TB 334 token (S32: 285), 2 chunk cắt về 512 token. **ETA GPU local (RTX 3050 6 GB, HF bf16, 25,5 cặp/s đo ở S32): 7,3 h; quy theo độ dài prompt 8,6 h**, nếu GPU không chia với job khác. Lệnh: `python -m r2ai.s31.judge_score score --pool out/runs/s38-chunk-metric-v2/judge-pool --out out/runs/s38-chunk-metric-v2/judge-gpu --model qwen3-reranker-0.6b --subset all`.

## Test

`pytest tests`: 906 pass / 2 deselected (587 s); `tests/test_s38_chunks.py` 14 test (dựng lại đoạn văn, cổng t256, cấu trúc, chọn chunk, cửa sổ, ghi lại hàng).

## Chưa làm

- CH-t128-1 (hạ ưu tiên): cần nhúng 1.292.602 chunk t128.
- Ghi chú để thử sau (Vu): doc hỏi-đáp, chunk answer là một phần của gold giữ cặp hỏi-đáp; nếu CH-t256-1 thấp bất thường ở host hỏi-đáp (ask.39, 120ask, vinmec) thì thử ghép question.

## Hướng cải tiến (working notes)

- **Cứu phần chữ thật của doc thin**: doc thin hiện mất cả chunk. Doc baothanhhoa.vn / baophutho.vn có khối base64 ở đầu rồi câu tiếng Việt thật ở cuối (baophutho còn khung bình luận). Thay vì bỏ chunk, có thể cắt bỏ dòng base64 và token `{placeholder}`, giữ phần chữ còn lại làm chunk nếu đủ dài (ngưỡng độ dài cần đo), rồi mới xét thin. Chỉ đáng làm khi số carrier thin trong docsel đủ lớn (S35 báo; trên EF2-cut-p4b chỉ 24 carrier, tác động không đáng kể).
- **Ghép question cho host hỏi–đáp** (ask.39, 120ask, vinmec): xem «Chưa làm» ở trên.
- **Tiêu chí tỷ lệ ký tự không phải chữ** không dùng cho thin (xem phần quyết định 24/10); nếu cần bắt thêm kiểu rác khác thì phải xét riêng từng dạng, có kiểm diff trên docsel.

## Compose: carriers tuỳ chọn, bundle dựng từ docsel, body 120ask theo registry (S35, 24/10)

Ba thêm vào `r2ai.s38.compose`, không đổi hành vi khi không dùng (nền `fe7f82e`; commit `37cbeae`, `dde831c`, `f355f73`):

- **`--carriers carriers.parquet`** (`query_id, doc_id`): tập carrier thay cho tập của bundle nền, nằm trong các doc của docsel. Carrier bỏ ra mất chunk; carrier mới nhận chunk nguyên body theo nguồn S29 (`s29_chunk_texts`; cặp không có nguồn chunk bị bỏ và đếm); thin tính trên tập mới; mọi thay đổi gộp vào `<name>.swap.json` so với bundle nền nên `write` không đổi. Ước lượng T tính theo `RATE_CARRIER` (580 B mỗi carrier) và `RATE_DOC` (2,9 B mỗi doc).
- **Không `--base-zip`**: bundle dựng từ docsel (`relevant_docs` theo `rank`, một chunk nguyên body mỗi carrier, lưu `<name>.base.parquet`), không cần ZIP `-docread` của S36. Cổng offline: nguồn S29 cho đúng chunk của ZIP docread EF2-free trên 4.000/4.000 carrier mẫu.
- **Registry 120ask** (`r2ai.s38.registry`): doc có trong `out/runs/s37/pairs_new.parquet` lấy `doc_text`, span t256 và chunk tốt nhất cho cửa sổ 768 từ segment `data/s37_120ask/<segment>/` (điểm chunk 0,6B của các run `cmax*`, không có thì chunk đầu theo cosine của `cmax_top3`, không có nữa thì chunk body đầu).
- Bộ chọn T (`run_full`): tái dùng phép đo cũ (`<name>-t<T>.compose.json` còn khớp SHA ZIP và mới hơn `prepare.json`) và, khi mọi ứng viên dưới trần, ước lượng lại T từ phần dư byte đo được thay vì bước mù; `--rounds` mặc định 4.
- Test: `tests/test_s38_compose_registry.py` (hoán đổi carrier, gộp swap, base từ docsel, registry, chọn T).
- **`--fit-carriers`** (run): khi bundle nền chưa có cửa sổ đã vượt trần, bỏ số carrier P4B thấp nhất ít nhất (bước `--fit-step`, mặc định 25) để ZIP vừa trần rồi mới tìm T; `write --drop N` ghi bản bỏ N carrier. 120A-free-v0 không cần bỏ carrier nào.
- **Validator và segment 120ask** (S35, 25/10): `registry.validator_extra` ghi các doc registry của bundle thành thư mục kiểu lô S15 (`lots/<segment>/chunks/docs.parquet`) để validator S26 coi là nguồn verbatim phụ; `compose.run_validator` truyền nó cùng lô Pharmacity. Trước đó chunk của doc 120ask bị báo `chunk_no_canonical_doc`. Lệnh `compose validate --name N [--threshold T] [--drop N]` chạy lại validator trên ZIP đã có (từ chối nếu ZIP khác `compose.json`) và ghi kết quả vào `<label>.compose.json`. 120A-free-v0: ok, 0 lỗi, `canonical_substring` 120.408.
