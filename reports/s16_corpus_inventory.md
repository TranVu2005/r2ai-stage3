# S16 — Kiểm kê corpus hiện tại

- Đo 2026-10-16 11:45:55 UTC+7; Python .venv, CPU; SQLite mode=ro + query_only. Không ghi state/index, không thao tác tiến trình.
- Nguồn chuẩn data/corpus_v2_ids.parquet: 1.220.598 URL ID, SHA256 dad50fb81c812a5c68386a2313f880db488b457e85707b92d7c478c29523374a [S00]. HF revision pin 71be0e1239a19dfd7a717e2bc13e3bec21bd0334 [S01].
- .env hiện trỏ R2AI_DATA_DIR sang OLD; tiến trình kiểm kê đặt biến này về repo/data trước import r2ai.paths. OLD không được đọc.
- Host ≥500 URL giữ riêng, còn lại gộp thành khác. Index URL = ID v2 phủ bởi doc_ids_group của doc sống; doc sống = primary doc_id distinct; chunk = số row. Zh supersede loại theo (segment, doc_id).

## 1. Phủ theo ngôn ngữ × host

Crawl/fetch OK: vi crawl.db; zh hợp hai crawl DB (OK thắng trạng thái chờ); Longchau CC ledger; en Europe PMC records. Extract lỗi = status ngoài ok/thin; không có record không tính thành lỗi extract.

| Ngôn ngữ | Host | V2 URL | Crawl/fetch OK | Extract OK | Thin | Lỗi | Doc sống | ID v2 trong index | Chunk | Phủ | Robots | Challenge | CC không snapshot | Chờ fetch | Soft404/thin | Chưa xử lý | Khác | Thiếu tổng |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vi | suckhoecongdongonline.vn | 154.503 | 154.501 | 154.242 | 259 | 0 | 154.230 | 154.230 | 902.871 | 99,82% | 0 | 0 | 0 | 0 | 260 | 12 | 1 | 273 |
| vi | suckhoedoisong.vn | 85.823 | 79.284 | 82.635 | 3.129 | 0 | 82.618 | 82.618 | 573.719 | 96,27% | 0 | 0 | 0 | 0 | 3.141 | 6 | 58 | 3.205 |
| vi | nhathuoclongchau.com.vn | 79.598 | 20.630 | 18.816 | 0 | 0 | 18.816 | 18.816 | 199.746 | 23,64% | 0 | 0 | 11.434 | 47.521 | 0 | 1.814 | 13 | 60.782 |
| vi | thanhnien.vn | 72.111 | 71.029 | 71.614 | 160 | 0 | 71.607 | 71.607 | 319.473 | 99,30% | 1 | 0 | 0 | 0 | 161 | 6 | 336 | 504 |
| vi | laodong.vn | 31.282 | 31.282 | 27.191 | 4.091 | 0 | 27.184 | 27.184 | 116.265 | 86,90% | 0 | 0 | 0 | 0 | 4.091 | 7 | 0 | 4.098 |
| vi | phunusuckhoe.giadinhonline.vn | 27.577 | 27.577 | 27.575 | 2 | 0 | 27.575 | 27.575 | 155.537 | 99,99% | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 |
| vi | vinmec.com | 26.129 | 26.127 | 26.124 | 3 | 0 | 26.124 | 26.124 | 223.233 | 99,98% | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 5 |
| vi | medlatec.vn | 24.762 | 24.709 | 24.709 | 0 | 0 | 24.709 | 24.709 | 223.764 | 99,79% | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 53 |
| vi | giadinhonline.vn | 16.729 | 16.727 | 16.726 | 2 | 0 | 16.726 | 16.726 | 105.914 | 99,98% | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 3 |
| vi | baohaiphong.vn | 16.286 | 16.286 | 16.246 | 40 | 0 | 16.246 | 16.246 | 81.618 | 99,75% | 0 | 0 | 0 | 0 | 40 | 0 | 0 | 40 |
| vi | baonghean.vn | 15.941 | 15.941 | 15.730 | 211 | 0 | 15.728 | 15.728 | 90.106 | 98,66% | 0 | 0 | 0 | 0 | 211 | 2 | 0 | 213 |
| vi | vietnamnet.vn | 15.543 | 15.542 | 15.535 | 7 | 0 | 13.840 | 15.535 | 81.529 | 99,95% | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 8 |
| vi | baodanang.vn | 14.500 | 14.500 | 14.447 | 53 | 0 | 14.447 | 14.447 | 81.175 | 99,63% | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 53 |
| vi | tiemchunglongchau.com.vn | 13.281 | 13.277 | 13.277 | 0 | 0 | 13.277 | 13.277 | 136.947 | 99,97% | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 4 |
| vi | baocantho.com.vn | 12.016 | 12.016 | 12.009 | 7 | 0 | 12.008 | 12.008 | 62.894 | 99,93% | 0 | 0 | 0 | 0 | 7 | 1 | 0 | 8 |
| vi | hellobacsi.com | 11.747 | 11.743 | 11.742 | 2 | 0 | 11.742 | 11.742 | 123.579 | 99,96% | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 5 |
| vi | baoangiang.com.vn | 10.738 | 10.738 | 10.695 | 43 | 0 | 10.695 | 10.695 | 62.522 | 99,60% | 0 | 0 | 0 | 0 | 43 | 0 | 0 | 43 |
| vi | vov.vn | 9.784 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0,00% | 9.784 | 0 | 0 | 0 | 0 | 0 | 0 | 9.784 |
| vi | khoahocphothong.vn | 8.143 | 8.143 | 8.097 | 46 | 0 | 8.096 | 8.096 | 55.570 | 99,42% | 0 | 0 | 0 | 0 | 46 | 1 | 0 | 47 |
| vi | youmed.vn | 7.189 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0,00% | 0 | 13 | 0 | 7.176 | 0 | 0 | 0 | 7.189 |
| vi | vov2.vov.vn | 6.899 | 6.899 | 6.898 | 1 | 0 | 6.898 | 6.898 | 38.874 | 99,99% | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| vi | phuyen.baodaklak.vn | 6.252 | 6.252 | 6.248 | 4 | 0 | 6.248 | 6.248 | 31.151 | 99,94% | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 4 |
| vi | suckhoeviet.org.vn | 5.627 | 5.626 | 5.625 | 1 | 0 | 5.625 | 5.625 | 42.997 | 99,96% | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 |
| vi | baoquangtri.vn | 5.483 | 5.481 | 5.481 | 2 | 0 | 5.481 | 5.481 | 35.604 | 99,96% | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 |
| vi | pmc-ecm-healthblog.beta.pharmacity.io | 5.333 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0,00% | 0 | 0 | 0 | 0 | 0 | 5.333 | 0 | 5.333 |
| vi | nhandan.vn | 5.225 | 5.222 | 5.224 | 0 | 0 | 5.222 | 5.224 | 37.850 | 99,98% | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| vi | baovinhlong.com.vn | 5.113 | 5.113 | 5.107 | 6 | 0 | 5.107 | 5.107 | 32.476 | 99,88% | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 6 |
| vi | tamanhhospital.vn | 4.261 | 4.259 | 4.259 | 0 | 0 | 4.259 | 4.259 | 52.366 | 99,95% | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 |
| vi | tuoitre.vn | 3.893 | 3.893 | 3.893 | 0 | 0 | 3.893 | 3.893 | 23.920 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | qdnd.vn | 3.700 | 3.699 | 3.699 | 0 | 0 | 3.699 | 3.699 | 22.308 | 99,97% | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| vi | vietnamplus.vn | 3.695 | 3.695 | 3.695 | 0 | 0 | 3.695 | 3.695 | 20.311 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | baogialai.com.vn | 3.290 | 3.287 | 3.287 | 0 | 0 | 3.287 | 3.287 | 17.612 | 99,91% | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 3 |
| vi | tienphong.vn | 3.021 | 3.021 | 3.021 | 0 | 0 | 3.021 | 3.021 | 21.332 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | thaythuocvietnam.vn | 2.927 | 507 | 592 | 4 | 0 | 592 | 592 | 4.258 | 20,23% | 0 | 0 | 0 | 0 | 2.335 | 0 | 0 | 2.335 |
| vi | baoquangninh.vn | 2.912 | 2.911 | 2.909 | 2 | 0 | 2.909 | 2.909 | 20.402 | 99,90% | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 3 |
| vi | bachmai.gov.vn | 2.783 | 2.514 | 2.509 | 274 | 0 | 2.509 | 2.509 | 17.095 | 90,15% | 0 | 0 | 0 | 0 | 274 | 0 | 0 | 274 |
| vi | baochinhphu.vn | 2.302 | 2.301 | 2.301 | 0 | 0 | 2.301 | 2.301 | 17.723 | 99,96% | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| vi | dantri.com.vn | 2.063 | 2.041 | 2.063 | 0 | 0 | 2.063 | 2.063 | 14.333 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | vnexpress.net | 2.055 | 2.054 | 2.055 | 0 | 0 | 2.055 | 2.055 | 11.813 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | baolangson.vn | 1.981 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0,00% | 1.981 | 0 | 0 | 0 | 0 | 0 | 0 | 1.981 |
| vi | sggp.org.vn | 1.853 | 1.853 | 1.853 | 0 | 0 | 1.853 | 1.853 | 9.312 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | baophutho.vn | 1.288 | 1.288 | 1.288 | 0 | 0 | 1.288 | 1.288 | 8.767 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | hanoimoi.vn | 1.283 | 1.283 | 1.283 | 0 | 0 | 1.283 | 1.283 | 10.112 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | benhviennhitrunguong.gov.vn | 1.200 | 1.200 | 1.193 | 7 | 0 | 1.193 | 1.193 | 8.671 | 99,42% | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 7 |
| vi | baotayninh.vn | 1.198 | 1.198 | 1.198 | 0 | 0 | 1.198 | 1.198 | 6.763 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| vi | baothanhhoa.vn | 782 | 782 | 782 | 0 | 0 | 782 | 782 | 3.484 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | 120ask.com | 98.160 | 21.170 | 23.005 | 132 | 0 | 22.741 | 22.742 | 82.800 | 23,17% | 4 | 0 | 0 | 74.904 | 140 | 263 | 107 | 75.418 |
| zh | cnkang.com | 40.046 | 39.185 | 37.925 | 2.101 | 0 | 37.923 | 37.925 | 189.807 | 94,70% | 0 | 0 | 0 | 0 | 2.117 | 1 | 3 | 2.121 |
| zh | ask.39.net | 32.772 | 32.761 | 32.058 | 703 | 0 | 30.352 | 30.352 | 102.676 | 92,62% | 0 | 0 | 0 | 0 | 714 | 1.706 | 0 | 2.420 |
| zh | familydoctor.com.cn | 26.819 | 26.814 | 26.814 | 0 | 0 | 26.814 | 26.814 | 132.330 | 99,98% | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| zh | zysjonline.com | 6.591 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0,00% | 0 | 0 | 0 | 0 | 0 | 6.591 | 0 | 6.591 |
| zh | zhongyibaodian.net | 6.061 | 5.671 | 6.054 | 7 | 0 | 6.046 | 6.046 | 74.558 | 99,75% | 0 | 0 | 0 | 0 | 14 | 1 | 0 | 15 |
| zh | a-hospital.com | 4.701 | 4.605 | 4.311 | 390 | 0 | 4.311 | 4.311 | 54.308 | 91,70% | 0 | 0 | 0 | 0 | 390 | 0 | 0 | 390 |
| zh | youlai.cn | 4.139 | 3.666 | 4.122 | 0 | 0 | 4.122 | 4.122 | 12.540 | 99,59% | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 17 |
| zh | baby.39.net | 3.368 | 3.368 | 3.367 | 1 | 0 | 3.367 | 3.367 | 17.943 | 99,97% | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| zh | test.pmphai.com | 2.989 | 2.865 | 2.907 | 82 | 0 | 2.907 | 2.907 | 22.085 | 97,26% | 0 | 0 | 0 | 0 | 82 | 0 | 0 | 82 |
| zh | jbk.39.net | 2.033 | 1.972 | 2.033 | 0 | 0 | 2.033 | 2.033 | 8.777 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | care.39.net | 1.282 | 1.256 | 1.282 | 0 | 0 | 1.282 | 1.282 | 8.143 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | heart.39.net | 1.055 | 1.055 | 1.055 | 0 | 0 | 1.054 | 1.054 | 5.697 | 99,91% | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| zh | fk.39.net | 928 | 928 | 928 | 0 | 0 | 928 | 928 | 4.871 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | msdmanuals.cn | 905 | 745 | 833 | 3 | 0 | 833 | 833 | 10.670 | 92,04% | 0 | 0 | 0 | 0 | 70 | 2 | 0 | 72 |
| zh | nk.39.net | 762 | 762 | 761 | 1 | 0 | 761 | 761 | 3.499 | 99,87% | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| zh | cancer.39.net | 736 | 736 | 736 | 0 | 0 | 736 | 736 | 4.190 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | article.iiyi.com | 729 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0,00% | 236 | 0 | 0 | 493 | 0 | 0 | 0 | 729 |
| zh | tnb.39.net | 717 | 717 | 717 | 0 | 0 | 717 | 717 | 3.749 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | wujue.com | 708 | 697 | 708 | 0 | 0 | 694 | 694 | 2.610 | 98,02% | 0 | 0 | 0 | 0 | 0 | 14 | 0 | 14 |
| zh | woman.39.net | 687 | 687 | 687 | 0 | 0 | 687 | 687 | 3.524 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | food.39.net | 594 | 594 | 594 | 0 | 0 | 594 | 594 | 2.262 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| zh | jb39.com | 527 | 506 | 527 | 0 | 0 | 527 | 527 | 1.702 | 100,00% | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| en | pubmed.ncbi.nlm.nih.gov | 240.000 | 239.771 | 239.353 | 418 | 0 | 239.771 | 239.771 | 581.102 | 99,90% | 0 | 0 | 0 | 0 | 0 | 0 | 229 | 229 |
| vi | khác | 497 | 492 | 492 | 0 | 0 | 492 | 492 | 3.336 | 98,99% | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 5 |
| zh | khác | 2.691 | 2.411 | 2.609 | 3 | 0 | 2.607 | 2.607 | 13.570 | 96,88% | 62 | 0 | 0 | 6 | 4 | 12 | 0 | 84 |

Tổng theo ngôn ngữ và trần nếu hoàn tất phần chờ fetch:

| Ngôn ngữ | V2 URL | ID trong index | Phủ hiện tại | Chờ fetch | Trần khi hết chờ | Robots riêng | Challenge riêng |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vi | 740.598 | 644.318 | 87,00% | 54.697 | 94,39% | 11.767 | 17 |
| zh | 240.000 | 152.039 | 63,35% | 75.403 | 94,77% | 307 | 0 |
| en | 240.000 | 239.771 | 99,90% | 0 | 99,90% | 0 | 0 |

Tổng corpus 1.220.598 URL, đã index 1.036.128 (84,89%); chờ fetch 130.100. Hash main/WAL pre-post ở S06B; nếu khác nhau thì ledger đang cập nhật trong lúc snapshot.

## 2. Index hiện có

Doc/chunk đo từ dữ liệu có sẵn; zh đã trừ các doc cũ bị supersede.

| Index/segment | Doc | Chunk | Chunker | Model embed | Build | SHA manifest / segment | Đường dẫn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vi t256 | 623.805 | 3.909.586 | r2ai.index.chunk.chunk_paragraphs; target=256; min_body_tokens=50 [S13] | BAAI/bge-m3 | 2026-10-05 02:15:23 | meta e2e2a68315b4513fa1940b7cc0daf7a09e5359db27a9bdac6eadf958173288d9 | data/index/t256; data/chunks |
| seg20000-cc-longchau-nhathuoclongchau.com.vn | 15.820 | 168.120 | {"target": 256, "min_body_tokens": 50, "batch_docs": 5000, "module": "r2ai.index.chunk"} | {"model": "BAAI/bge-m3", "max_len": 512, "fp16": true, "batch_size": 16, "block": 1000, "dense": "CLS, L2-normalised, fp16", "sparse": "relu(sparse_linear) max per token id"} | 2026-10-13 00:28:04 UTC+7 | manifest 2096134f3b8a15f9d6cf609ff6d79d2baaeb36417e52bdabcb65a612b088f7c6; segment d7a064fb716bc7d2954e4f9745c937acabb64cca512475ea1ef3de7deb215178 | data/chunks_cc_longchau/seg20000-cc-longchau-nhathuoclongchau.com.vn |
| seg20001-cc-longchau-b2-nhathuoclongchau.com.vn | 2.996 | 31.626 | {"target": 256, "min_body_tokens": 50, "batch_docs": 5000, "module": "r2ai.index.chunk"} | {"model": "BAAI/bge-m3", "max_len": 512, "fp16": true, "batch_size": 16, "block": 1000, "dense": "CLS, L2-normalised, fp16", "sparse": "relu(sparse_linear) max per token id"} | 2026-10-13 03:30:48 UTC+7 | manifest 2096134f3b8a15f9d6cf609ff6d79d2baaeb36417e52bdabcb65a612b088f7c6; segment 756e0573ff58cefbb9d444f9d96c68ec587ba022e98def3998b590fe4651d2e2 | data/chunks_cc_longchau/seg20001-cc-longchau-b2-nhathuoclongchau.com.vn |
| zh v2 (63 segment) | 152.036 | 773.549 | r2ai.index.chunk.chunk_paragraphs; target=256; min_body_tokens=50 [S13] | BAAI/bge-m3 | 2026-10-15 13:09:22 UTC+7 | manifest 51bd64942ef64935daf80ba9270a5e146d719014cc3cbdc3cdc673bccef77ef2 | data/index/zh_v2_segments; data/chunks_zh_v2 |
| en PubMed (59 shard) | 239.771 | 581.102 | r2ai.vi_quality.chunker.structured_chunks [S14,S15] | BAAI/bge-m3 | 2026-10-15 00:30:00 | meta 53d25d9f8e1f957e9fe9fa919b2deadf76ce17411c9bd9a77590c09f17e19606 | data/index/en_pubmed; data/pubmed_en |

Zh có 63 segment; 20 thuộc batch v2n-003; manifest build 2026-10-15 13:09:22 UTC+7; 1.568 cặp doc cũ bị supersede [S07A,S07B].

## 3. Thành phần best V1

JSON trong release ZIP có SHA256 32b806299733bb4f79201127731c0b19112225df0c434fdc28bcfaa3133b2c32; 1200 query, validator ghi 126.228 chunk, errors=0 [S10A,S10B]. doc_id là lần xuất hiện / distinct ID; byte là UTF-8 chunk_text; trung bình chia 1200 query.

| Ngôn ngữ | Host | doc_id (lần / distinct) | Chunk | Byte chunk_text | TB/query (doc / chunk / byte) | Loại doc_id |
| --- | --- | --- | --- | --- | --- | --- |
| vi | suckhoecongdongonline.vn | 6.287 / 4.208 | 5.035 | 24.580.705 | 5.24 / 4.20 / 20483.9 | có text: 6.287 |
| vi | suckhoedoisong.vn | 23.945 / 15.066 | 19.458 | 109.010.019 | 19.95 / 16.21 / 90841.7 | có text: 23.939, id slug (không có text): 6 |
| vi | nhathuoclongchau.com.vn | 25.765 / 11.410 | 14.799 | 124.349.917 | 21.47 / 12.33 / 103624.9 | id slug (không có text): 2.882, doc CC: 22.883 |
| vi | thanhnien.vn | 12.113 / 7.736 | 9.825 | 44.492.021 | 10.09 / 8.19 / 37076.7 | có text: 12.077, id slug (không có text): 36 |
| vi | laodong.vn | 2.434 / 1.727 | 1.978 | 7.753.718 | 2.03 / 1.65 / 6461.4 | có text: 2.434 |
| vi | phunusuckhoe.giadinhonline.vn | 4.923 / 3.275 | 3.983 | 20.300.247 | 4.10 / 3.32 / 16916.9 | có text: 4.923 |
| vi | vinmec.com | 16.751 / 9.631 | 13.913 | 82.835.901 | 13.96 / 11.59 / 69029.9 | có text: 16.750, id slug (không có text): 1 |
| vi | medlatec.vn | 19.423 / 10.114 | 16.117 | 127.048.754 | 16.19 / 13.43 / 105874.0 | có text: 19.384, id slug (không có text): 39 |
| vi | giadinhonline.vn | 4.104 / 2.698 | 3.299 | 16.014.675 | 3.42 / 2.75 / 13345.6 | có text: 4.104 |
| vi | baohaiphong.vn | 1.356 / 913 | 1.057 | 5.336.904 | 1.13 / 0.88 / 4447.4 | có text: 1.356 |
| vi | baonghean.vn | 2.054 / 1.421 | 1.623 | 7.929.002 | 1.71 / 1.35 / 6607.5 | có text: 2.054 |
| vi | vietnamnet.vn | 2.442 / 1.538 | 1.828 | 9.106.967 | 2.04 / 1.52 / 7589.1 | có text: 2.442 |
| vi | baodanang.vn | 948 / 673 | 765 | 3.845.414 | 0.79 / 0.64 / 3204.5 | có text: 948 |
| vi | tiemchunglongchau.com.vn | 12.603 / 6.197 | 10.403 | 94.360.680 | 10.50 / 8.67 / 78633.9 | có text: 12.601, id slug (không có text): 2 |
| vi | baocantho.com.vn | 1.641 / 1.136 | 1.310 | 7.339.658 | 1.37 / 1.09 / 6116.4 | có text: 1.641 |
| vi | hellobacsi.com | 5.552 / 3.071 | 4.581 | 46.345.294 | 4.63 / 3.82 / 38621.1 | có text: 5.551, id slug (không có text): 1 |
| vi | baoangiang.com.vn | 1.320 / 881 | 1.023 | 4.820.178 | 1.10 / 0.85 / 4016.8 | có text: 1.320 |
| vi | vov.vn | 941 / 752 | 0 | 0 | 0.78 / 0.00 / 0.0 | id slug (không có text): 941 |
| vi | khoahocphothong.vn | 1.883 / 1.207 | 1.508 | 8.494.621 | 1.57 / 1.26 / 7078.9 | có text: 1.883 |
| vi | youmed.vn | 1.429 / 922 | 0 | 0 | 1.19 / 0.00 / 0.0 | id slug (không có text): 1.429 |
| vi | vov2.vov.vn | 1.038 / 653 | 856 | 4.123.682 | 0.86 / 0.71 / 3436.4 | có text: 1.038 |
| vi | phuyen.baodaklak.vn | 1.016 / 647 | 842 | 3.624.074 | 0.85 / 0.70 / 3020.1 | có text: 1.016 |
| vi | suckhoeviet.org.vn | 507 / 384 | 406 | 3.117.170 | 0.42 / 0.34 / 2597.6 | có text: 506, id slug (không có text): 1 |
| vi | baoquangtri.vn | 465 / 294 | 371 | 2.212.325 | 0.39 / 0.31 / 1843.6 | có text: 465 |
| vi | pmc-ecm-healthblog.beta.pharmacity.io | 0 / 0 | 0 | 0 | 0.00 / 0.00 / 0.0 | — |
| vi | nhandan.vn | 620 / 412 | 472 | 2.622.580 | 0.52 / 0.39 / 2185.5 | có text: 620 |
| vi | baovinhlong.com.vn | 917 / 598 | 743 | 4.143.388 | 0.76 / 0.62 / 3452.8 | có text: 917 |
| vi | tamanhhospital.vn | 4.641 / 1.991 | 3.868 | 52.166.484 | 3.87 / 3.22 / 43472.1 | có text: 4.641 |
| vi | tuoitre.vn | 453 / 323 | 360 | 1.914.833 | 0.38 / 0.30 / 1595.7 | có text: 453 |
| vi | qdnd.vn | 355 / 240 | 265 | 1.230.862 | 0.30 / 0.22 / 1025.7 | có text: 355 |
| vi | vietnamplus.vn | 260 / 175 | 188 | 1.337.586 | 0.22 / 0.16 / 1114.7 | có text: 260 |
| vi | baogialai.com.vn | 265 / 196 | 214 | 945.946 | 0.22 / 0.18 / 788.3 | có text: 265 |
| vi | tienphong.vn | 586 / 380 | 469 | 2.964.447 | 0.49 / 0.39 / 2470.4 | có text: 586 |
| vi | thaythuocvietnam.vn | 1.155 / 639 | 182 | 1.626.488 | 0.96 / 0.15 / 1355.4 | có text: 230, id slug (không có text): 925 |
| vi | baoquangninh.vn | 370 / 227 | 291 | 1.622.378 | 0.31 / 0.24 / 1352.0 | có text: 370 |
| vi | bachmai.gov.vn | 740 / 464 | 564 | 3.399.539 | 0.62 / 0.47 / 2832.9 | có text: 740 |
| vi | baochinhphu.vn | 123 / 88 | 93 | 791.387 | 0.10 / 0.08 / 659.5 | có text: 123 |
| vi | dantri.com.vn | 374 / 276 | 298 | 1.658.769 | 0.31 / 0.25 / 1382.3 | có text: 374 |
| vi | vnexpress.net | 824 / 457 | 674 | 2.714.549 | 0.69 / 0.56 / 2262.1 | có text: 824 |
| vi | baolangson.vn | 171 / 129 | 0 | 0 | 0.14 / 0.00 / 0.0 | id slug (không có text): 171 |
| vi | sggp.org.vn | 94 / 67 | 76 | 300.303 | 0.08 / 0.06 / 250.3 | có text: 94 |
| vi | baophutho.vn | 75 / 56 | 65 | 579.219 | 0.06 / 0.05 / 482.7 | có text: 75 |
| vi | hanoimoi.vn | 140 / 96 | 115 | 597.562 | 0.12 / 0.10 / 498.0 | có text: 140 |
| vi | benhviennhitrunguong.gov.vn | 384 / 276 | 297 | 1.892.339 | 0.32 / 0.25 / 1576.9 | có text: 384 |
| vi | baotayninh.vn | 74 / 46 | 58 | 337.032 | 0.06 / 0.05 / 280.9 | có text: 74 |
| vi | baothanhhoa.vn | 26 / 20 | 18 | 117.261 | 0.02 / 0.01 / 97.7 | có text: 26 |
| zh | 120ask.com | 6.448 / 4.661 | 631 | 864.204 | 5.37 / 0.53 / 720.2 | có text: 6.448 |
| zh | cnkang.com | 3.166 / 2.777 | 252 | 947.933 | 2.64 / 0.21 / 789.9 | có text: 3.166 |
| zh | ask.39.net | 4.458 / 3.771 | 485 | 522.432 | 3.71 / 0.40 / 435.4 | có text: 4.458 |
| zh | familydoctor.com.cn | 2.169 / 1.978 | 194 | 535.063 | 1.81 / 0.16 / 445.9 | có text: 2.169 |
| zh | zysjonline.com | 0 / 0 | 0 | 0 | 0.00 / 0.00 / 0.0 | — |
| zh | zhongyibaodian.net | 304 / 260 | 34 | 426.219 | 0.25 / 0.03 / 355.2 | có text: 304 |
| zh | a-hospital.com | 197 / 169 | 10 | 153.959 | 0.16 / 0.01 / 128.3 | có text: 197 |
| zh | youlai.cn | 528 / 452 | 63 | 86.866 | 0.44 / 0.05 / 72.4 | có text: 528 |
| zh | baby.39.net | 390 / 351 | 31 | 101.621 | 0.33 / 0.03 / 84.7 | có text: 390 |
| zh | test.pmphai.com | 72 / 70 | 1 | 1.722 | 0.06 / 0.00 / 1.4 | có text: 72 |
| zh | jbk.39.net | 75 / 64 | 2 | 7.876 | 0.06 / 0.00 / 6.6 | có text: 75 |
| zh | care.39.net | 98 / 89 | 13 | 59.488 | 0.08 / 0.01 / 49.6 | có text: 98 |
| zh | heart.39.net | 72 / 65 | 12 | 30.197 | 0.06 / 0.01 / 25.2 | có text: 72 |
| zh | fk.39.net | 86 / 81 | 6 | 16.859 | 0.07 / 0.01 / 14.0 | có text: 86 |
| zh | msdmanuals.cn | 110 / 67 | 12 | 120.830 | 0.09 / 0.01 / 100.7 | có text: 110 |
| zh | nk.39.net | 60 / 57 | 7 | 13.087 | 0.05 / 0.01 / 10.9 | có text: 60 |
| zh | cancer.39.net | 65 / 60 | 1 | 6.760 | 0.05 / 0.00 / 5.6 | có text: 65 |
| zh | article.iiyi.com | 0 / 0 | 0 | 0 | 0.00 / 0.00 / 0.0 | — |
| zh | tnb.39.net | 95 / 75 | 3 | 12.807 | 0.08 / 0.00 / 10.7 | có text: 95 |
| zh | wujue.com | 43 / 36 | 6 | 8.643 | 0.04 / 0.01 / 7.2 | có text: 43 |
| zh | woman.39.net | 82 / 74 | 7 | 20.842 | 0.07 / 0.01 / 17.4 | có text: 82 |
| zh | food.39.net | 75 / 65 | 6 | 10.698 | 0.06 / 0.01 / 8.9 | có text: 75 |
| zh | jb39.com | 9 / 6 | 0 | 0 | 0.01 / 0.00 / 0.0 | có text: 9 |
| en | pubmed.ncbi.nlm.nih.gov | 0 / 0 | 0 | 0 | 0.00 / 0.00 / 0.0 | — |
| vi | khác | 192 / 142 | 148 | 979.580 | 0.16 / 0.12 / 816.3 | có text: 192 |
| zh | khác | 174 / 148 | 14 | 34.852 | 0.14 / 0.01 / 29.0 | có text: 174 |

CSV chi tiết theo lang × host × doc_type: out/runs/s16/inventory_best_v1.csv [S10A].

## 4. Chưa dùng

Doc sống chưa từng có primary doc_id trong V1 relevant_docs pool:

| Ngôn ngữ | Doc sống ngoài pool V1 |
| --- | --- |
| vi | 553.095 |
| zh | 136.661 |
| en | 239.771 |

Longchau: CC priority snapshot 68.164 URL v2; fetch OK 20.630, chờ ledger 47.521, unavailable 13; index có 18.816 ID v2. Fetch OK chưa ingest/index: 1.814 [S06A–S08E].
En: Europe PMC record 239.771 ID v2; abstract 239.353, title-only/thin 418; 3 PMID không có record. Vu ghi D17 đã bỏ khỏi bản chính; repo không có artifact/tag D17 để tách riêng, số D17 không đo được [S09A–S09C].

## 5. Chất lượng đã đo (không chạy lại)

- S12 vi: crawl OK 624.596; robots 11.767; challenge 17; extract 632.208 (ok 623.852, thin 8.356); index 623.805 doc / 3.909.586 chunk. Quality scan loại 9.354 doc (1,50%), strip 2.208 (0,35%); K80 tác động 86 chunk / 86.132 byte UTF-8. [S11A, SHA fd596ef352b982205ed382b77865b8ad56c4f675bbda6a330f455c5d6fc71554]
- S11 zh, host mới: sample ask.39 n=100, QA OK 99/100; full scan ask.39 318/22.413 bad (1,42%). Không chạy lại QA. [S11B].
- QA en: n=3; full-body similarity mean 0,9964, min 0,9892; section mean 0,9919, min 0,9818; structured sample 0. [S09F].

## 6. Trần độ phủ hợp lệ

Cận số học = index hiện tại + toàn bộ nhóm chờ fetch, giả sử phần chờ đều extract/index được. Robots/challenge là blockers riêng, không cộng vào trần; đây không phải dự báo ingest.

| Ngôn ngữ | V2 URL | Index hiện tại | Phủ hiện tại | Chờ fetch | Trần khi hết chờ | Robots block | Bot challenge |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vi | 740.598 | 644.318 | 87,00% | 54.697 | 94,39% | 11.767 | 17 |
| zh | 240.000 | 152.039 | 63,35% | 75.403 | 94,77% | 307 | 0 |
| en | 240.000 | 239.771 | 99,90% | 0 | 99,90% | 0 | 0 |

## Kết luận khoảng trống lớn nhất

- Vi thiếu 96.280 ID; gồm 54.697 chờ fetch và 11.767 robots block.
- Zh phủ 63,35%; chờ fetch 75.403 và robots 307.
- Longchau có 47.521 URL snapshot chờ fetch; 1.814 fetch OK chưa vào index.
- En phủ 99,90%; 3 PMID không có record, 418 record title-only/thin.
- Toàn corpus phủ 84,89%; pending-only ceiling 95,55%.
- Robots/challenge tách khỏi pending; S16 không crawl, không QA lại, không embedding hoặc ingest.

## Nguồn, đường dẫn và SHA256

SHA cây = SHA256 mỗi file rồi SHA256 các dòng canonical relative_path NUL size NUL file_sha256 LF. SQLite mode=ro; hash main/WAL pre/post báo ledger đổi trong khi fetch.

| Ref | Đường dẫn repo-relative | SHA256 / trạng thái | Vai trò |
| --- | --- | --- | --- |
| S00 | data/corpus_v2_ids.parquet | dad50fb81c812a5c68386a2313f880db488b457e85707b92d7c478c29523374a | Corpus v2 |
| S01 | reports/s9_corpus_v2.md | ec775a0c2f961b5e200e3b875419eaf642a20567de63478f97a9417a094afec2 | HF revision pin |
| S01b | data/raw/links_corpus_vi_en_zh.parquet | 87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7 | File parquet nguồn HF đã pin trong S9; 37.689.434 B |
| S02 | state/crawl.db | main 18cdfb66cce995e573f0beea64ca81ef0b0bbaf3c169070a58dc369ef7a04018; WAL e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, crawl vi |
| S03 | data/docs_vi | c8467333d3cb1ffa4451ddd0f7164b01a2c95c6e675bb942a27c1d053ba7316c (tree 18006 file; 2427965673 B) | Parquet extraction vi |
| S04A | state/zh_full/crawl.db | main 935ff2c7cd33f66476040537a828870ce5754272bb6fb86e53a5e5fee88f6283; WAL 3f4580444d919eb7b299653da84617035e32f471ddbf6896f919884acfea1690 (4181832 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, crawl zh_full |
| S04B | state/zh_v2/crawl.db | main ec13214574b8f7d02b479de6e59cd837d9a4a42bf3d1916f6f4d09b43d75a886; WAL f87f5fbfef4225c6027769c1ab600dc7856f7ef92f4821f09b383273e866d80a (4181832 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, crawl zh_v2 |
| S05A | state/zh_full_ingest/manifest.db | main cdf9e1f0d722141dc40fa7b25b1b86aab08bcfbcfb7b8c0f134304dbe5036a8b; WAL e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, extraction manifest zh |
| S05B | state/zh_v2_ingest/manifest.db | main 0cb77d5d13c1d254e0a6f2ad8dd0aebe653dd1b8ca8428bc4c720e49cddf99ee; WAL e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, extraction manifest zh |
| S05C | state/zh_v2_ingest_redo/manifest.db | main 17c30995b05d347f91095f539e7a887a86cb9049f070142d6b2be5983898e078; WAL e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, extraction manifest zh |
| S05D | state/zh_v2_ingest_redo2/manifest.db | main dc6a01904e97ef92c83287dd6bde75c9c279807c99ed64d1c8ebfd9ceca5c0df; WAL e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, extraction manifest zh |
| S06A | state/cc_longchau/priority.parquet | f5e7b22df6218251c9be59b001cc1ea94a23857e32c8306a34607e70a9fffea3 | CC priority snapshot |
| S06B | state/cc_longchau/fetch.db | main 066b32ba9d58b00e8926c36de490565cb4c7fd978e9f8207a71fe08a4b270dce; WAL e6db24ef11e4ae5627908693f1a0828cae92622e926cc2b3c1b2762f03beb6de (3559712 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro; ledger may update |
| S06C | data/docs_cc_longchau | 5030db538d4197403297df94d2a6445713fd0ce32fcac028bc80df1e1bceaef9 (tree 63 file; 91441991 B) | Parquet extraction CC Longchau |
| S07A | data/index/zh_v2_segments/manifest.json | 51bd64942ef64935daf80ba9270a5e146d719014cc3cbdc3cdc673bccef77ef2 | Zh index manifest; 63 segments and supersede map |
| S07B | data/chunks_zh_v2 | e189bf5580712869352c3c8498efa120e6371ed71abc4eb97ae002c6b734234c (tree 124 file; 360023744 B) | Parquet docs/chunks zh v2 |
| S08A | data/index/t256/meta.json | e2e2a68315b4513fa1940b7cc0daf7a09e5359db27a9bdac6eadf958173288d9 | Vi index meta |
| S08B | data/chunks/chunk_report.json | d04ccf453cb59a3502314394f2c555f7b65198b899ab58e2af07d530f9bf3e26 | Vi chunk report |
| S08C | data/chunks/docs.parquet | 5b2d22b6b0382bb2b325b644357a31c7afc2cb0740ba6e54f7fcd3f3799099c8 | Vi index docs |
| S08D | data/chunks/chunks_t256.parquet | 3ba99ad1ed192aa4f6f3ad72df7422ed73766a875dcdc511699ce06a818792ea | Vi chunks |
| S08E | data/chunks_cc_longchau | ecc3814915869edca73b5d9487817db46ef7285ed6ce30f8bca2c38243b7c2c0 (tree 4 file; 89309412 B) | Parquet docs/chunks CC Longchau |
| S09A | data/pubmed_en/europepmc-summary.json | b401ac10082520e7d58445fa096cff0d0b70cd8816cd8ba8a625d268d02eaa9e | Europe PMC summary |
| S09B | data/pubmed_en/missing-pmids.json | 51bcca008a854693803cabb2cf2d06a1c9a0e6b3a0f792da511f56addec6cf1e | Missing PMID |
| S09C | data/pubmed_en/records.parquet | 92e38b55ac23f41978a37e4d4baf69c6daef23fa15b33790b6a8762658a7da61 | Europe PMC records |
| S09D | data/pubmed_en/chunks.parquet | 1bcd9c28b2d1af3eab757f2814374cdae4ba27f39714977bce755d24b37a0d61 | Europe PMC chunks |
| S09E | data/index/en_pubmed/meta.json | 53d25d9f8e1f957e9fe9fa919b2deadf76ce17411c9bd9a77590c09f17e19606 | EN index metadata |
| S09F | out/runs/s10-en-pubmed-qa/qa_cc_sample.json | 1aff2b9cbe2d5e85a30768b3caa327e25f9ed78c0bf768afe7456076c50eec6b | QA en n=3 |
| S10A | out/runs/reranker-trial/release/V1.zip | 5371cfffc174aefccef393c80047dcadd7c410b25b1276fa2f3f3c2047007f67 | V1 release ZIP |
| S10B | out/runs/reranker-trial/release/V1.validator.json | 4ae5726dbc3eef06cc6027107dbf5a0deee02963912ac54bf0c55c7e66b982cd | V1 validator |
| S11A | reports/s12_vi_quality.md | fd596ef352b982205ed382b77865b8ad56c4f675bbda6a330f455c5d6fc71554 | S12 vi |
| S11B | out/runs/zh-v2/qa_extract/qa_extract.json | 5de95834558e7b04fb2b2dd3a287eea5cd675749b6b183928239da3e729be4ec | S11 zh QA |
| S12B | state/cc_longchau/manifest.db | main 2096134f3b8a15f9d6cf609ff6d79d2baaeb36417e52bdabcb65a612b088f7c6; WAL e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, CC index manifest |
| S12C | state/extract.db | main 62b2bf16e4ff265e31ed58509dc2e7c4cf1f595a1fb925167a6f361c20761db8; WAL e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 (0 B); main/WAL pre-post ổn định trong lượt đo | SQLite mode=ro, vi extract checkpoint |
| S13 | src/r2ai/index/chunk.py | ba28b47a3c1e2535458d635224c62d838e8e3b61aade84b530bd7409e2580332 | Chunker vi/zh t256 |
| S14 | src/r2ai/vi_quality/chunker.py | 07d6a48147b6fe63b2a33ffcfa324883dfb799de292bbb667d3bd57a0fb380a5 | Hàm structured_chunks dùng cho en |
| S15 | src/r2ai/vi_quality/README.md | 518ce96ce2237f2d0596023175802d98d3531292e7a593a323c9ac66660ffdef | Ghi nhận chunker cấu trúc là mặc định index en |

CSV host: out/runs/s16/inventory_by_host.csv.
