# S12 – chất lượng doc vi

Nền: `483f4df` (A1-115 là best mặc định trong repo; K80 chưa commit nên chưa đo ZIP, xem mục 4).
Nhánh `feat/vi-quality`. Không dùng LLM. Số liệu đo trên 623.805 doc đã index (`data/chunks/docs.parquet`,
`chunks_t256.parquet` 3.909.586 chunk). Mẫu duyệt: `reports/s12_vi_samples.md`.

## 1. Kiểm chéo crawl.db / extract / index (chỉ xác nhận, không sửa)

Doc xấu đã bị loại trước khi index. `crawl.db` (toàn bộ): `ok` 624.596 URL; loại `thin` 7.713, `soft404_or_home` 2.305,
`blocked_4xx` 11.767, `network_error` 395, `bot_challenge` 17, `dead_origin` 1 (tổng 22.198 URL không có text), `pending` 7.176.
Extract: 632.208 doc, `ok` 623.852, `thin` 8.356 (thin không vào chunk/index); index 623.805 doc (47 doc `ok` không vào index).
Không có status `interstitial` riêng; trang chọn ngôn ngữ nếu có đã rơi vào `soft404_or_home`/`thin`.

| host | doc index | crawl ok | crawl thin | soft404_or_home | extract thin | bị loại trước index |
|---|---:|---:|---:|---:|---:|---:|
| suckhoecongdongonline.vn | 154.230 | 154.501 | 0 | 1 | 259 | 2 |
| suckhoedoisong.vn | 82.618 | 79.284 | 6.480 | 1 | 3.129 | 6.539 |
| thanhnien.vn | 71.607 | 71.029 | 745 | 0 | 160 | 1.082 |
| phunusuckhoe.giadinhonline.vn | 27.575 | 27.577 | 0 | 0 | 2 | 0 |
| laodong.vn | 27.184 | 31.282 | 0 | 0 | 4.091 | 0 |
| vinmec.com | 26.124 | 26.127 | 0 | 2 | 3 | 2 |
| medlatec.vn | 24.709 | 24.709 | 0 | 53 | 0 | 53 |
| giadinhonline.vn | 16.726 | 16.727 | 1 | 0 | 2 | 2 |
| baohaiphong.vn | 16.246 | 16.286 | 0 | 0 | 40 | 0 |
| baonghean.vn | 15.728 | 15.941 | 0 | 0 | 211 | 0 |

("bị loại trước index" = thin + soft404 + blocked + network + bot + dead trong crawl.db; cột extract thin tính theo
doc đã crawl `ok` nhưng extract đánh `thin`.) Ví dụ vinmec `/vie/bai-viet/...` kiểu "chưa có nội dung tiếng Anh":
quét cụm "chưa có nội dung / chọn ngôn ngữ" trên doc index ngắn (<1.200 ký tự) không ra trang interstitial nào.

## 2. Rule (src/r2ai/vi_quality/rules.py, test `tests/test_vi_quality.py`)

Đã bỏ cờ `title_sus` và `nav` theo yêu cầu (dương tính giả). Doc ngắn không bị loại chỉ vì độ dài: mẫu <300 ký tự
là bài chỉ có đoạn dẫn nhưng là nội dung thật.

- Strip đuôi (doc giữ lại, cắt từ marker đến hết): `laodong_comment` (khối "Bình luận:/Bạn nghĩ gì về nội dung này?/…/Gửi bình luận"
  kèm tiếp theo là chặn quảng cáo hoặc "Cùng chuyên mục"), `giadinhonline_related` ("Nên đọc" / "Tin liên quan" đến hết, 2 host
  giadinhonline), `baonghean_onecms_footer`. Không bao giờ làm doc rỗng.
- Loại cả doc: `lottery_result` (title kiểu XSDN/XSMB/KQXS/"Kết quả xổ số" **và** có ngày dd/mm; tin về Vietlott không dính),
  `error_title` (title là `…/404`), `menu_only` (≥ 80% dòng body là mục menu, ≥ 10 dòng), `title_is_site` (title chỉ là tên host).

## 3. Kết quả quét (index-wide)

| host | doc | loại | % | strip | % | ký tự cắt |
|---|---:|---:|---:|---:|---:|---:|
| suckhoecongdongonline.vn | 154.230 | 5.793 | 3,76% | 0 | 0 | 0 |
| thanhnien.vn | 71.607 | 4 | 0,01% | 0 | 0 | 0 |
| baonghean.vn | 15.728 | 2 | 0,01% | 96 | 0,61% | 3.736 |
| laodong.vn | 27.184 | 0 | 0 | 1.539 | 5,66% | 232.956 |
| giadinhonline.vn | 16.726 | 0 | 0 | 554 | 3,31% | 537.888 |
| phunusuckhoe.giadinhonline.vn | 27.575 | 0 | 0 | 19 | 0,07% | 21.299 |
| 38 host còn lại | | 0 | 0 | 0 | 0 | 0 |
| **Tổng** | **623.805** | **5.799** | 0,93% | **2.208** | 0,35% | 795.879 |

`title_is_site` = 0 doc. Danh sách: `reports/s12_vi_drop_doc_ids.tsv` (5.799), `reports/s12_vi_strip_doc_ids.tsv` (2.208).
Tác động chunk index-wide (`chunks_t256`): chunk bị bỏ 38.481 (30,36 MB text; submittable answer/body 30.444 chunk, 29,72 MB),
chunk bị cắt cụt 2.201 (1,51 MB). Đây là cận trên cho index; byte trong ZIP phụ thuộc chunk nào được nộp (xem mục 4).

## 4. Byte giải phóng trong ZIP K80

Chưa đo: chờ đường dẫn bản K80 (Final 0,2334; B-cut115 + pool chunk B + k_chunk 80, ZIP_LZMA). Không dùng A1-115 thay thế.
Sẽ báo số doc/chunk rác đang nộp và MB giải phóng; nếu < 2 MB thì dừng ở rule strip, không đề xuất re-fetch.

## 5. Ghi chú lệch so với yêu cầu

- Bản tin COVID trùng title ở suckhoecongdongonline (ví dụ "Việt Nam ghi nhận 1 ca tử vong do COVID-19", 30 doc): 30/30 doc có
  text khác nhau (ngày, tỉnh, số ca). Cả host có 839 nhóm trùng title / 1.890 doc, chỉ 36 nhóm (73 doc) trùng nguyên văn. Gộp theo title
  sẽ mất sự kiện riêng của từng bản tin nên **chưa gộp**; chỉ dedupe khi text trùng nguyên văn (mục sau).
- Trùng nguyên văn sau strip: 2.233 doc (`reports/s12_vi_dup_map.tsv`, drop_id → keep_id). Cả 2.233 đã nằm cùng cụm với
  keeper trong `out/runs/H1/clusters.parquet` (cụm H1 dùng cho expand của best), nên ánh xạ ID đã có sẵn; không áp thêm.
- Ngoài phạm vi (c) nhưng cùng loại trang mẫu, không phải y khoa, ở suckhoecongdongonline: dự báo thời tiết 1.180, lịch thi đấu 927,
  giá vàng 780, soi kèo 248, dự đoán tỉ số 176, nhận định bóng đá 122, tỷ giá 122 (tổng ≈ 4.100 theo title). Chưa áp; cần Vu quyết.

## 6. Chunker theo cấu trúc (`src/r2ai/vi_quality/chunker.py`, chưa nối vào pipeline)

Luật: heading (từ `headings` của extractor, nhãn kết thúc bằng ":", tiêu đề đánh số ngắn) đi cùng khối bên dưới và không đứng cuối
chunk; cặp hỏi-đáp chung một chunk nếu vừa `target`; list/bảng là một khối, chỉ tách ở ranh giới mục khi bản thân vượt `target`;
section vừa `target` là một chunk, gộp section liền kề chỉ khi chunk đang mở < 64 token; không overlap; chunk luôn là substring
nguyên văn của `doc_text` (số, đơn vị, phủ định giữ nguyên, test `tests/test_vi_chunker.py`). Tái dùng cho en/zh (S10b, S11): chỉ cần
đoạn + `headings` tùy chọn, heuristic heading không phụ thuộc tiếng Việt (dấu câu cuối gồm `。！？`; chưa chạy thử trên zh).

So với chunker index hiện tại (t256, `index.chunk.chunk_paragraphs`) trên cùng 12.636 doc (300 doc đầu mỗi host theo thứ tự shard,
44 host, 48,68 triệu ký tự body), tokenizer BGE-M3 thật, `target` = 256 (`reports/s12_vi_chunker_compare.json`):

| | t256 hiện tại | theo cấu trúc |
|---|---:|---:|
| số chunk answer/body | 72.738 | 70.101 (−3,6%) |
| token/chunk mean · p50 · p95 | 246 · 219 · 254 | 213 · 188 · 253 |
| ký tự trong chunk (body gốc 48,68 M) | 60,23 M (+23,7% do overlap) | 49,04 M (+0,7%) |
| JSON các chunk | 78,98 MB | 64,09 MB (−18,9%) |
| ZIP deflate-9 các chunk | 18,59 MB | 17,77 MB (−4,4%) |
| LZMA các chunk | 12,09 MB | 11,61 MB (−4,0%) |
| chunk kết thúc bằng heading | 6.036 (8,3%) | 162 (0,2%) |
| list/bảng bị cắt ngang / tổng | 1.649 / 4.646 (35%) | 847 / 4.646 (18%; phần còn lại dài hơn `target`) |
| cặp hỏi-đáp chung chunk | 0 / 13 | 13 / 13 |
| chunk < 32 token | 0,55% | 2,25% |

Đọc kết quả: ZIP giảm ít hơn JSON vì phần overlap bị bỏ nén rất tốt. Mẫu chỉ có 13 cặp hỏi-đáp (vinmec), chưa đủ nói về tỉ lệ
hỏi-đáp toàn corpus. Chunk < 32 token tăng (mục cuối cùng của section nhỏ không gộp được); chưa chỉnh.
Lưu ý quan trọng: `chunk_mode full` của builder hiện nộp **cả body** của doc, không nộp chunk t256, nên chunker chỉ đổi byte ZIP nếu
bản nộp dùng chunk con (kiểu `window`/c2). Không có gold offline: chưa đo Chunk F2, mọi con số trên là cấu trúc/dung lượng.

## 7. Đề xuất 1 probe LB (không tự nộp)

Chỉ đề xuất sau khi có K80: giữ doc set và thứ tự doc của K80, giữ nguyên số doc nhận chunk; thay `chunk_text` body đầy đủ bằng
chunk cấu trúc ≤ 256 token chứa chunk t256 có điểm reranker cao nhất (ánh xạ theo tâm span, không cần chấm lại reranker), cộng
phần body còn lại không đổi cho doc ngoài top. Byte giải phóng dùng để tăng `k_chunk`. Chưa có số byte vì cần JSON K80; Chunk F2
chỉ đo được bằng 1 lượt LB. Rủi ro đã biết: A/B trước đây cho thấy chunk full thắng chunk ngắn ở cùng K (c2 0,0158 → full 0,0273),
nên probe chỉ đáng nộp nếu byte giải phóng cho phép tăng K đáng kể.
