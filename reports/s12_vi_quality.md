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
| **Tổng** | **623.805** | **5.799** | 0,93% | **2.208** | 0,35% | 796.000 |

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
