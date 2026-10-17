# S22 — Khôi phục text từ raw HTML cục bộ

- Chỉ đọc raw đã lưu trong repo; không có request mạng.
- Tập host: S20 `empty_or_extract_error >= 50`, VI/ZH; loại PubMed EN theo D17.
- S20 input: `D:\GitHub\r2ai-stage3\reports\s20-coverage-audit.csv` nếu có; checkout hiện chuyển sang `feat/s21-120ask` không có `reports/`, nên fallback đọc blob local `2d3f44c:reports/s20-coverage-audit.csv`.
- Text tốt: ≥200 Unicode code point sau gộp whitespace. 120ask dùng `status != ok` để chọn lỗi theo proxy S20, rồi đo lại theo ngưỡng 200.
- Một batch/domain cho mỗi host theo CC contract S15. Bundle tách riêng; không ghi `data/index`, `state/cc_longchau`, hay index chung.

| Lang | Host | Lỗi S20 | Lỗi local | Raw có | Raw thiếu | Thu hồi ≥200 | Còn rỗng local | Docs staged | Docs segment | V2 | Lệch snapshot | Nguyên nhân mẫu |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| vi | `bachmai.gov.vn` | 274 | 274 | 274 | 0 | 0 | 274 | 274 | 0 | 100% (274) | +0 | thật sự rỗng 20 |
| vi | `baodanang.vn` | 53 | 53 | 53 | 0 | 48 | 5 | 53 | 48 | 100% (53) | +0 | selector sai 19, thật sự rỗng 1 |
| vi | `baonghean.vn` | 224 | 224 | 224 | 0 | 88 | 136 | 224 | 88 | 100% (224) | +0 | selector sai 14, thật sự rỗng 6 |
| vi | `laodong.vn` | 4091 | 4091 | 4091 | 0 | 69 | 4022 | 4091 | 69 | 100% (4091) | +0 | thật sự rỗng 20 |
| vi | `suckhoecongdongonline.vn` | 259 | 259 | 259 | 0 | 255 | 4 | 259 | 255 | 100% (259) | +0 | selector sai 20 |
| vi | `suckhoedoisong.vn` | 3130 | 3130 | 3130 | 0 | 2680 | 450 | 3130 | 2674 | 100% (3130) | +0 | selector sai 19, thật sự rỗng 1 |
| vi | `thanhnien.vn` | 165 | 165 | 165 | 0 | 152 | 13 | 165 | 152 | 100% (165) | +0 | selector sai 18, thật sự rỗng 2 |
| zh | `120ask.com` | 132 | 132 | 100 | 32 | 0 | 132 | 100 | 0 | 100% (100) | +0 | không raw 4, thật sự rỗng 16 |
| zh | `a-hospital.com` | 384 | 390 | 383 | 7 | 0 | 390 | 383 | 0 | 100% (383) | +6 | thật sự rỗng 20 |
| zh | `ask.39.net` | 11109 | 11111 | 11111 | 0 | 794 | 10317 | 11111 | 794 | 100% (11111) | +2 | selector sai 3, thật sự rỗng 17 |
| zh | `cnkang.com` | 2088 | 2105 | 1558 | 547 | 0 | 2105 | 1558 | 0 | 100% (1558) | +17 | không raw 5, thật sự rỗng 15 |
| zh | `familydoctor.com.cn` | 188 | 190 | 190 | 0 | 190 | 0 | 190 | 190 | 100% (190) | +2 | selector sai 20 |
| zh | `jbk.39.net` | 68 | 68 | 68 | 0 | 39 | 29 | 68 | 39 | 100% (68) | +0 | selector sai 14, thật sự rỗng 6 |
| zh | `test.pmphai.com` | 98 | 99 | 98 | 1 | 0 | 99 | 98 | 0 | 100% (98) | +1 | thật sự rỗng 20 |
| zh | `wujue.com` | 90 | 90 | 6 | 84 | 5 | 85 | 6 | 5 | 100% (6) | +0 | không raw 20 |
| zh | `youlai.cn` | 271 | 279 | 271 | 8 | 255 | 24 | 271 | 255 | 100% (271) | +8 | encoding 7, selector sai 13 |
| zh | `zhongyibaodian.net` | 270 | 274 | 257 | 17 | 77 | 197 | 257 | 77 | 100% (257) | +4 | selector sai 8, thật sự rỗng 12 |

Tổng: 22.894 lỗi theo S20; 22.934 lỗi trong snapshot local (+40). Có raw khớp cho 22.238 doc, 696 ID không có raw. Trong mẫu 340 doc (20/host): 148 selector sai, 156 raw không có text đạt ngưỡng, 7 lỗi encoding, 29 thiếu raw; không gặp trang chặn/lỗi. Sau re-extract, 4.652 doc đạt ≥200 code point, 17.586 raw-backed doc vẫn dưới ngưỡng; thêm 696 doc thiếu raw nên tổng còn rỗng local là 18.282. Segment có 4.646 doc / 11.781 chunk; 6 doc đạt text threshold nhưng bị min-token filter.

## Bundle S15

- Root: `D:\GitHub\r2ai-stage3\data\s22_reextract`; manifest: `D:\GitHub\r2ai-stage3\data\s22_reextract\manifest.db` theo `r2ai.zh_full_ingest.ingest.SCHEMA_SQL`.
- Mỗi host có batch `cc-s22-{host}`, `docs/<host>.parquet` theo `vicrawl.extract_pipeline.SCHEMA`, `provenance.parquet` theo S15 `PROV_SCHEMA`, `source=reextract-raw`, `raw_shard` là đường dẫn repo-relative.
- Mỗi batch có segment chunked t256 (`docs.parquet`, `chunks_t256.parquet`, `chunk_report.json`). Segment ID trong bundle là provisional; S15 cấp ID production và embed trước khi index.
- Membership được kiểm trên doc ID chính và toàn bộ `doc_ids` group cho docs; chunk docs được kiểm độc lập trong v2.
- Segment artifact không chứa vector và không được đăng ký vào index chung.

### Đường dẫn từng batch

- `cc-s22-bachmai.gov.vn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-bachmai.gov.vn` — docs=274, segment docs=0, v2=100% (274).
- `cc-s22-baodanang.vn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-baodanang.vn` — docs=53, segment docs=48, v2=100% (53).
- `cc-s22-baonghean.vn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-baonghean.vn` — docs=224, segment docs=88, v2=100% (224).
- `cc-s22-laodong.vn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-laodong.vn` — docs=4091, segment docs=69, v2=100% (4091).
- `cc-s22-suckhoecongdongonline.vn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-suckhoecongdongonline.vn` — docs=259, segment docs=255, v2=100% (259).
- `cc-s22-suckhoedoisong.vn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-suckhoedoisong.vn` — docs=3130, segment docs=2674, v2=100% (3130).
- `cc-s22-thanhnien.vn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-thanhnien.vn` — docs=165, segment docs=152, v2=100% (165).
- `cc-s22-120ask.com`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-120ask.com` — docs=100, segment docs=0, v2=100% (100).
- `cc-s22-a-hospital.com`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-a-hospital.com` — docs=383, segment docs=0, v2=100% (383).
- `cc-s22-ask.39.net`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-ask.39.net` — docs=11111, segment docs=794, v2=100% (11111).
- `cc-s22-cnkang.com`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-cnkang.com` — docs=1558, segment docs=0, v2=100% (1558).
- `cc-s22-familydoctor.com.cn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-familydoctor.com.cn` — docs=190, segment docs=190, v2=100% (190).
- `cc-s22-jbk.39.net`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-jbk.39.net` — docs=68, segment docs=39, v2=100% (68).
- `cc-s22-test.pmphai.com`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-test.pmphai.com` — docs=98, segment docs=0, v2=100% (98).
- `cc-s22-wujue.com`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-wujue.com` — docs=6, segment docs=5, v2=100% (6).
- `cc-s22-youlai.cn`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-youlai.cn` — docs=271, segment docs=255, v2=100% (271).
- `cc-s22-zhongyibaodian.net`: `D:\GitHub\r2ai-stage3\data\s22_reextract\cc-s22-zhongyibaodian.net` — docs=257, segment docs=77, v2=100% (257).

## Chẩn đoán và raw thiếu

- Mẫu chẩn đoán: 340 dòng trong `reports/s22_sample_diagnosis.csv` (tối đa 20 lỗi/host).
- ID không đọc được raw: 696 dòng trong `reports/s22_raw_missing_ids.csv`.
- Lệch snapshot được giữ riêng ở cột “Lệch snapshot”: aggregate S20 không chứa danh sách ID; local failure set được chọn lại từ docs parquet hiện có.
