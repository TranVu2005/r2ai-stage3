# S33. Tìm tín hiệu cấu trúc chưa dùng (22/10/2026)

**Bối cảnh (Vu cung cấp):** baseline BTC đạt Doc P 0,88 / R 0,51 với khoảng 30 doc/query. J4-K40 của mình đạt Doc P 0,38 / R 0,29 với 40 doc/query.

**Suy ra từ P, R và số doc** (chưa đo, chỉ là phép tính): |G| ≈ P·|D|/R. Cả hai bản đều ra khoảng 52 doc gold mỗi query (0,88·30/0,51 ≈ 51,8; 0,38·40/0,29 ≈ 52,4). Số doc đúng mỗi query: baseline khoảng 26,4/30, mình khoảng 15,2/40.

**Nguồn dữ liệu dùng trong báo cáo:**
- Thứ hạng J4 lấy theo thứ tự `relevant_docs` trong `out/runs/s31-strong-judge/build-topk/J4-K40.zip` và `J4-K60.zip`. Builder ghi theo thứ tự `select_rrf`: RRF(k=60) của cosine_rank và hạng 4B, trên top-211 cosine của pool hợp.
- Kết quả trung gian nằm ở `out/runs/s33-structure-signals/`. Script ở `scripts/s33/`.

## 1. Schema

| File | Dòng | Cột | Ghi chú |
|---|---:|---|---|
| `data/raw/links_corpus_vi_en_zh.parquet` (v2) | 1.220.598 | `id` int64, `url` string | 14 row group, arrow 17.0.0, không có key-value metadata |
| `links_corpus.parquet` (v1, chỉ có ở OLD, chỉ đọc) | 4.394.718 | `id` int64, `url` string | 44 row group, DuckDB 1.3.2 |
| `data/raw/query.parquet` | 1.200 | `id` int64 not null (1–1200, tăng dần), `query` string not null | độ dài 18–1.243 ký tự, trung vị 81 |

- Không file nào có cột nào khác ngoài `id` và `url` (hoặc `id` và `query`): không có lang, title, host hay ngày. Ngôn ngữ chỉ suy được từ khoảng ID: vi < 766.442, zh 766.442–4.420.561, en (pubmed) ≥ 4.421.055.
- **Thứ tự dòng của v1 = thứ tự ID, và cũng là thứ tự chuỗi URL.** v1 gồm hai khối, khối vi (id 583–766.441) rồi khối zh (766.442–4.420.561). Trong mỗi khối, URL tăng nghiêm ngặt theo thứ tự chuỗi (cả v1 chỉ có 1 chỗ ngắt, đúng ở ranh giới vi/zh). Vì `http://` < `https://`, các URL `http` đứng trước trong khối.
- **ID của v1 có lỗ:**
  - ID 1–582 không có;
  - 25.261 ID khác bị khuyết, nằm trong 7 khối liền, toàn bộ ở khối vi;
  - mỗi khối nằm ngay sau URL cuối của một host: tamanhhospital 14.096, hellobacsi 4.098, sggp 3.389, vietnamnet 3.083, tienphong 368, dantri 118, suckhoedoisong 109.

  README ghi khoảng ID 1–4.420.561. Các URL đã bị rút khỏi bản công bố nên không biết là URL nào.
- **v2 = toàn bộ khối vi của v1 + mẫu zh + pubmed:**
  - vi: 740.598 dòng, bằng đúng số dòng vi của v1;
  - zh: 240.000 dòng chọn từ 3.654.120 dòng zh của v1;
  - pubmed: 240.000 ID mới, 4.421.055–7.466.161.

  980.598 dòng v2 có ID trùng v1, và URL trùng 100%. Thứ tự dòng v2 là vi, pubmed, rồi zh, nên ID có 1 chỗ giảm (dòng 980.598). Trong khối pubmed, ID không theo đúng thứ tự chuỗi URL (4 chỗ ngắt).

## 2. Độ phân tán doc_id của top-20 J4

Với mỗi query lấy top-N J4 và so với hai mốc. **Ngẫu nhiên cùng ngôn ngữ:** mỗi doc thay bằng một doc ngẫu nhiên cùng ngôn ngữ trong v2. **Ngẫu nhiên cùng host:** mỗi doc thay bằng một doc ngẫu nhiên cùng host. Đo theo ID và theo vị trí dòng trong parquet v2; hai cách cho kết quả gần như nhau.

Theo vị trí dòng (trung bình 1.200 query):

| N | Bộ | Khoảng cách trung vị giữa doc kề nhau | Số host | % doc có láng giềng cách ≤ 1 dòng | ≤ 5 | ≤ 20 | ≤ 100 | Cụm ≤ 1 / query |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20 | J4 | 16.178 | 9,24 | **6,1%** | 10,2% | 13,1% | 17,2% | 0,57 |
| 20 | ngẫu nhiên cùng host | 17.070 | 9,24 | 0,0% | 0,07% | 0,4% | 1,8% | 0,00 |
| 20 | ngẫu nhiên cùng ngôn ngữ | 27.987 | 11,87 | 0,02% | 0,03% | 0,1% | 0,5% | 0,002 |
| 5 | J4 | 93.254 | 3,72 | 2,8% | 4,8% | 5,9% | 7,0% | 0,07 |
| 40 | J4 | 7.213 | 13,58 | 7,6% | 12,5% | 16,4% | 22,2% | 1,37 |

- Ở quy mô lớn **không có cụm**: khoảng cách trung vị của J4 xấp xỉ mốc cùng host. Phần chênh so với mốc cùng ngôn ngữ chỉ do top-N dồn vào ít host hơn.
- Ở quy mô nhỏ có **cụm vi mô**: 6,1% doc trong top-20 có một doc liền kề (cách đúng 1 dòng) cũng nằm trong top-20, so với khoảng 0% ở hai mốc.
- **Bản chất cụm** (1.371 cặp cách ≤ 5 dòng): vì ID là thứ tự URL, các doc này cùng host và có slug bắt đầu giống nhau, tức tiêu đề mở đầu giống nhau. Ví dụ `vinmec …/polyp-tui-mat-co-phai-mo-khong` và `…/polyp-tui-mat-co-tu-het-khong-khi-nao-can-mo`. Chúng không phải bản trùng:
  - chỉ 0,8% cặp cùng `phys_id`, 3,3% cặp có text giống hệt;
  - độ giống text (difflib, 1.500 ký tự đầu) có trung vị 0,25 và chỉ 6,1% cặp ≥ 0,9.

  Host đóng góp nhiều cặp nhất: suckhoedoisong 282, medlatec 234, nhathuoclongchau 219, vinmec 180, tiemchunglongchau 118, thanhnien 110.

## 3. Mở rộng theo cụm từ top-5

Từ top-5 J4 của mỗi query, lấy doc ở vị trí dòng ±w trong cùng host, bỏ các doc đã có trong top-5. Mốc so sánh: 2 doc ngẫu nhiên cùng host cho mỗi doc nguồn. Các cột:
- *Hạng cosine toàn cục*: hạng trên 887.151 doc canonical, dùng BGE-M3 doc-level (`full_doc_dense.npz`) và vector query của S26.
- *P4B*: điểm 4B của S31, chỉ có khi cặp nằm trong pool13 hoặc pool-extra.

| w | Bộ | Số doc | Hạng cosine trung vị | Trong top-100 cosine | Trong top-1000 cosine | Trong J4 top-40 | Có điểm 4B | P4B TB (nếu có) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | láng giềng | 11.822 | 69.906 | 6,8% | 15,9% | 4,4% | 10,6% | 0,593 |
| 1 | ngẫu nhiên cùng host | 12.000 | 378.941 | 0,06% | 0,3% | 0,04% | 0,1% | 0,551 |
| 2 | láng giềng | 11.931 | 105.089 | 3,8% | 10,6% | 2,3% | 6,5% | 0,554 |
| 3 | láng giềng | 11.949 | 125.287 | 2,6% | 8,4% | 1,5% | 4,9% | 0,566 |
| 5 | láng giềng | 11.967 | 152.379 | 1,5% | 5,8% | 0,8% | 3,1% | 0,544 |
| 10 | láng giềng | 11.980 | 185.603 | 0,9% | 3,4% | 0,5% | 1,7% | 0,544 |

- Mốc: P4B trung bình của J4 hạng 6–40 là **0,697**, 76,3% cặp có P4B ≥ 0,5.
- Doc kề ±1 gần query hơn doc ngẫu nhiên cùng host rất nhiều (6,8% so với 0,06% nằm trong top-100 cosine). Nhưng tín hiệu này giảm nhanh khi w tăng và dense đã bắt được nó.
- 11.307 doc kề ±1 nằm ngoài J4 top-40. Hạng cosine toàn cục của chúng: P10 625, P25 7.762, trung vị 81.413. Có 734 doc trong số đó đã có điểm 4B, P4B trung bình **0,484**, thấp hơn hẳn 0,697 của doc đang chọn ở hạng 6–40.
- **Kết luận: mở rộng theo ID chỉ thêm doc yếu hơn doc đang chọn.** Không đề xuất dựng biến thể theo hướng này.

## 4. Query ↔ doc nguồn (title gần trùng nguyên văn câu hỏi)

**Cách làm:**
- Văn bản dò: trường `title` (718.446) và `question` (6.331) của chunk canonical thuộc doc vi. Query là tiếng Việt nên doc zh không thể trùng nguyên văn.
- Chuẩn hoá: NFC, chữ thường, bỏ dấu câu.
- Lấy ứng viên bằng bigram từ (tối đa 100 văn bản mỗi query).
- Tính LCS theo ký tự, r = LCS / max(|query|, |title|).
- Doc nguồn gồm cả các ID alias trong cùng nhóm `doc_ids`.

| Ngưỡng r | Số query có doc nguồn | Hạng J4 tốt nhất của doc nguồn |
|---|---:|---|
| ≥ 0,95 | **18 / 1.200 (1,5%)** | hạng 1: 17; hạng 5: 1; ngoài top-60: 0 |
| ≥ 0,9 | 19 | hạng 1: 17; hạng 5: 1; ngoài top-60: 1 |
| ≥ 0,8 | 50 | hạng 1: 20; ≤ 5: 24; ≤ 20: 27; ≤ 60: 32; ngoài: 18 |
| ≥ 0,7 | 177 | hạng 1: 26; ≤ 5: 41; ≤ 20: 57; ≤ 60: 75 |

- Các cặp trùng thật (r ≥ 0,95) gần như đều là trang hỏi đáp vinmec (16/18), cộng 1 nhathuoclongchau và 1 qdnd. J4 đã xếp chúng hạng 1 ở 17/18 query.
- Dải 0,7–0,9 chủ yếu là **tiêu đề cùng khuôn câu nhưng khác bệnh**. Ví dụ "Mụn thịt hình thành thế nào, điều trị ra sao?" khớp với "Sỏi thận hình thành thế nào, điều trị ra sao?" (r = 0,88). 18 doc khớp ở ngưỡng 0,8 nằm ngoài top-60 đều thuộc loại này. Chúng không phải doc nguồn.
- **Kết luận:** chỉ 1,5% query có doc nguồn trùng tiêu đề, và J4 đã xếp đúng các doc đó. Đây không phải nguồn của khoảng cách P 0,38 so với 0,88.

## 5. Tài liệu BTC công bố về baseline

| Tài liệu | Vị trí | Nội dung liên quan |
|---|---|---|
| Slide Mentor Day 1 (12/10/2026, 48 trang, ViBioMIR, Nguyễn Hoàng Trung, AI Guru/TiniX) | `C:/Users/Admin/Downloads/261012_R2_AI_S3_MentorDay1.pdf` | Nhóm phương pháp baseline: lexical (TF-IDF, BM25), dense (mBERT, BGE-M3), hybrid (RRF), reranking (cross-encoder, ví dụ BGE-reranker-v2-m3). Chunk theo mục và giữ cặp hỏi–đáp cùng đoạn. Chunk khớp khi LCS token / \|g\| ≥ 0,40 sau NFKC, HTML entity, chữ thường và khoảng trắng; F2 macro theo query. Phân nhóm ICD-10. Corpus 4.394.718 URL / 97 host. **Không có số P/R/F2 của baseline và không có cấu hình cụ thể** (K, số chunk, mô hình nào thực sự dùng). |
| README dataset (bản sao) | `docs/dataset.md` | Thống kê query (18 từ trung vị), "Tổng số liên kết 4.420.561, khoảng ID 1–4.420.561", schema 2 cột. Không nói về baseline hay cách tạo gold. |
| Slide trang 48 trích dẫn "ViBioMIR: README và báo cáo kiểm kê dữ liệu" | không tìm thấy trong repo, Downloads hay cache HF | "Báo cáo kiểm kê dữ liệu" chưa có bản local. |

Không tìm thấy notebook baseline hay số liệu P 0,88 / R 0,51 trong máy. Các số này do Vu cung cấp.

## Tóm tắt

- Thứ tự ID chỉ mã hoá thứ tự URL theo chữ cái. ID kề nhau là slug giống nhau, dense đã khai thác, và mở rộng theo ID thêm doc yếu hơn.
- Title trùng query chỉ có ở 1,5% query, và đã đứng hạng 1.
- Corpus công bố không có cột nào ngoài `id` và `url`.
- Thông tin cấu trúc duy nhất chưa dùng:
  - 25.843 ID bị rút (1–582 cùng 7 khối cuối host vi). Chúng nằm ngoài corpus nên không thể nằm trong gold đã chấm, trừ khi gold có ID ngoài corpus.
  - v2 chỉ giữ mẫu 240k zh; trước đây đã biết.

## Tái tạo

Chạy từ gốc repo main. `<S>` là thư mục ra; `S26_SCRATCH` là thư mục chứa `q_dense.npy` / `q_ids.npy` của S26.

```
python scripts/s33/schema.py data/raw/links_corpus_vi_en_zh.parquet <OLD>/data/raw/links_corpus.parquet data/raw/query.parquet
python scripts/s33/order.py; python scripts/s33/order2.py; python scripts/s33/order3.py; python scripts/s33/holes.py <S>
python scripts/s33/idspread.py <S>        # mục 2 (ghi corpus_meta.parquet, idspread.parquet)
python scripts/s33/adjpairs.py <S>        # mục 2, bản chất cặp kề
python scripts/s33/neighbors.py <S>       # mục 3 (GPU, cosine toàn cục)
python scripts/s33/srcdoc.py <S>; python scripts/s33/srcrank.py <S>; python scripts/s33/srcexact.py <S>; python scripts/s33/srcex.py <S>   # mục 4
```
