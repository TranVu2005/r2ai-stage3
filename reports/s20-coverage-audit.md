# S20 — Audit độ phủ corpus v2 và ưu tiên lấp khoảng trống

- Đo ngày 2026-10-17; corpus theo snapshot S16 2026-10-16. Corpus v2: **1.220.598 URL**, SHA256 `dad50fb81c812a5c68386a2313f880db488b457e85707b92d7c478c29523374a` (HF revision `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`). S16 là nguồn chuẩn theo xác nhận của Vu; bỏ số 1.194.469.
- Mục tiêu N: body sau làm sạch, gộp whitespace, dài **≥200 Unicode code point**. Bốn host đang chạy S3/S11 dùng `extract_ok` của S16 làm proxy vì S16 không lưu body-length; các host còn lại đo trực tiếp trên Parquet snapshot. Proxy không phải phép đo body length.
- Corpus có 94 cặp (lang, host): 50 vi, 43 zh, 1 en. en chỉ có số liệu (D17), không có đề xuất hành động.

## Tóm tắt

| Lang | URL v2 | Text tốt/proxy | Rỗng/lỗi extract | Thiếu hoàn toàn | Chờ S3/S11 | Index hiện tại | Chưa top-300 / rate | URL hiện thiếu index |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vi | 740,598 | 644,341 (87.00%) | 8,380 | 19,906 | 67,971 | 705,753 | 562,401 (79.69%) | 33,148 |
| zh | 240,000 | 141,368 (58.90%) | 14,826 | 8,054 | 75,752 | 152,036 | 55,713 (36.64%) | 87,961 |
| en | 240,000 | 239,353 (99.73%) | 418 | 229 | 0 | — | — | — |

Tầng 2 dùng đúng 1.200 query, lấy unique doc xuất hiện trong top-300 first-stage riêng từng index (vi và zh). Doc live trong current index nhưng vắng cache vi first-stage, gồm phần Longchau/Youmed đang chạy, được tính là chưa top-300; vì vậy đây là phép đo theo cache hiện có, không phải đánh giá lại retrieval.

## Độ phủ theo ngôn ngữ × host

| Lang | Host | URL v2 | Text tốt/proxy | Rỗng/lỗi | Thiếu hoàn toàn | Chờ S3/S11 | Index hiện tại | Chưa top-300 | Rate | Thiếu URL index | V1/index hiện tại | CC 200 HTML/200 | Proxy thu hồi* | Proxy/giờ CC* |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| en | `pubmed.ncbi.nlm.nih.gov` | 240,000 | 239,353 (99.73%) | 418 | 229 | 0 | 0 | 0 | — | — | — | — | — | — |
| vi | `bachmai.gov.vn` | 2,783 | 2,509 (90.15%) | 274 | 0 | 0 | 2,509 | 1,686 | 67.20% | 274 | 18.49% | — | — | — |
| vi | `baoangiang.com.vn` | 10,738 | 10,695 (99.60%) | 43 | 0 | 0 | 10,695 | 8,617 | 80.57% | 43 | 8.24% | — | — | — |
| vi | `baocantho.com.vn` | 12,016 | 12,009 (99.94%) | 7 | 0 | 0 | 12,008 | 9,552 | 79.55% | 8 | 9.46% | — | — | — |
| vi | `baochinhphu.vn` | 2,302 | 2,301 (99.96%) | 0 | 1 | 0 | 2,301 | 2,087 | 90.70% | 1 | 3.82% | — | — | — |
| vi | `baodanang.vn` | 14,500 | 14,447 (99.63%) | 53 | 0 | 0 | 14,447 | 12,657 | 87.61% | 53 | 4.66% | — | — | — |
| vi | `baogialai.com.vn` | 3,290 | 3,287 (99.91%) | 0 | 3 | 0 | 3,287 | 2,870 | 87.31% | 3 | 5.96% | — | — | — |
| vi | `baohaiphong.vn` | 16,286 | 16,245 (99.75%) | 41 | 0 | 0 | 16,246 | 14,279 | 87.89% | 40 | 5.62% | — | — | — |
| vi | `baolangson.vn` | 1,981 | 0 (0.00%) | 0 | 1,981 | 0 | 0 | 0 | — | 1,981 | — | 0/200 (0.0%) | — | — |
| vi | `baonghean.vn` | 15,941 | 15,717 (98.59%) | 224 | 0 | 0 | 15,728 | 12,751 | 81.07% | 213 | 9.03% | — | — | — |
| vi | `baophutho.vn` | 1,288 | 1,288 (100.00%) | 0 | 0 | 0 | 1,288 | 1,132 | 87.89% | 0 | 4.35% | — | — | — |
| vi | `baoquangninh.vn` | 2,912 | 2,909 (99.90%) | 2 | 1 | 0 | 2,909 | 2,473 | 85.01% | 3 | 7.80% | — | — | — |
| vi | `baoquangtri.vn` | 5,483 | 5,481 (99.96%) | 2 | 0 | 0 | 5,481 | 4,761 | 86.86% | 2 | 5.36% | — | — | — |
| vi | `baotayninh.vn` | 1,198 | 1,198 (100.00%) | 0 | 0 | 0 | 1,198 | 1,055 | 88.06% | 0 | 3.84% | — | — | — |
| vi | `baothanhhoa.vn` | 782 | 782 (100.00%) | 0 | 0 | 0 | 782 | 705 | 90.15% | 0 | 2.56% | — | — | — |
| vi | `baovinhlong.com.vn` | 5,113 | 5,106 (99.86%) | 7 | 0 | 0 | 5,107 | 3,992 | 78.17% | 6 | 11.71% | — | — | — |
| vi | `benhviennhitrunguong.gov.vn` | 1,200 | 1,193 (99.42%) | 7 | 0 | 0 | 1,193 | 749 | 62.78% | 7 | 23.13% | — | — | — |
| vi | `benhvienvietduc.org` | 425 | 425 (100.00%) | 0 | 0 | 0 | 425 | 247 | 58.12% | 0 | 0.00% | — | — | — |
| vi | `dantri.com.vn` | 2,063 | 2,063 (100.00%) | 0 | 0 | 0 | 2,063 | 1,501 | 72.76% | 0 | 13.38% | — | — | — |
| vi | `giadinhonline.vn` | 16,729 | 16,726 (99.98%) | 2 | 1 | 0 | 16,726 | 11,685 | 69.86% | 3 | 16.13% | — | — | — |
| vi | `hanoimoi.vn` | 1,283 | 1,283 (100.00%) | 0 | 0 | 0 | 1,283 | 1,091 | 85.04% | 0 | 7.48% | — | — | — |
| vi | `hellobacsi.com` | 11,747 | 11,742 (99.96%) | 2 | 3 | 0 | 11,742 | 6,891 | 58.69% | 5 | 26.15% | — | — | — |
| vi | `khoahocphothong.vn` | 8,143 | 8,097 (99.44%) | 46 | 0 | 0 | 8,096 | 5,892 | 72.78% | 47 | 14.91% | — | — | — |
| vi | `laodong.vn` | 31,282 | 27,191 (86.92%) | 4,091 | 0 | 0 | 27,184 | 23,427 | 86.18% | 4,098 | 6.35% | 15/200 (7.5%) | 19.53 | 33.36 |
| vi | `medlatec.vn` | 24,762 | 24,709 (99.79%) | 0 | 53 | 0 | 24,709 | 10,079 | 40.79% | 53 | 40.93% | — | — | — |
| vi | `mega.vietnamplus.vn` | 2 | 2 (100.00%) | 0 | 0 | 0 | 2 | 2 | 100.00% | 0 | 0.00% | — | — | — |
| vi | `nhandan.vn` | 5,225 | 5,224 (99.98%) | 0 | 1 | 0 | 5,222 | 4,397 | 84.20% | 1 | 7.89% | — | — | — |
| vi | `nhathuoclongchau.com.vn` | 79,598 | 18,816 (23.64%) | 0 | 0 | 60,782 | 76,868 | 76,868 | 100.00% | 2,730 | 14.84% | 1/200 (0.5%) | 2.03 | 4.02 |
| vi | `pharmacity.vn` | 69 | 65 (94.20%) | 0 | 4 | 0 | 65 | 26 | 40.00% | 4 | 0.00% | — | — | — |
| vi | `phunusuckhoe.giadinhonline.vn` | 27,577 | 27,575 (99.99%) | 2 | 0 | 0 | 27,575 | 21,424 | 77.69% | 2 | 11.88% | — | — | — |
| vi | `phuyen.baodaklak.vn` | 6,252 | 6,248 (99.94%) | 4 | 0 | 0 | 6,248 | 5,021 | 80.36% | 4 | 10.36% | — | — | — |
| vi | `pmc-ecm-healthblog.beta.pharmacity.io` | 5,333 | 0 (0.00%) | 0 | 5,333 | 0 | 0 | 0 | — | 5,333 | — | — | — | — |
| vi | `qdnd.vn` | 3,700 | 3,699 (99.97%) | 0 | 1 | 0 | 3,699 | 3,122 | 84.40% | 1 | 6.49% | — | — | — |
| vi | `sggp.org.vn` | 1,853 | 1,853 (100.00%) | 0 | 0 | 0 | 1,853 | 1,686 | 90.99% | 0 | 3.62% | — | — | — |
| vi | `special.vietnamplus.vn` | 1 | 0 (0.00%) | 0 | 1 | 0 | 0 | 0 | — | 1 | — | — | — | — |
| vi | `suckhoecongdongonline.vn` | 154,503 | 154,242 (99.83%) | 259 | 2 | 0 | 154,230 | 144,236 | 93.52% | 273 | 2.73% | — | — | — |
| vi | `suckhoedoisong.vn` | 85,823 | 82,634 (96.28%) | 3,130 | 59 | 0 | 82,618 | 56,862 | 68.83% | 3,205 | 18.24% | 77/200 (38.5%) | 225.02 | 267.00 |
| vi | `suckhoeviet.org.vn` | 5,627 | 5,625 (99.96%) | 1 | 1 | 0 | 5,625 | 4,675 | 83.11% | 2 | 6.83% | — | — | — |
| vi | `tamanhhospital.vn` | 4,261 | 4,259 (99.95%) | 0 | 2 | 0 | 4,259 | 1,470 | 34.52% | 2 | 46.75% | — | — | — |
| vi | `thanhnien.vn` | 72,111 | 71,609 (99.30%) | 165 | 337 | 0 | 71,607 | 55,175 | 77.05% | 504 | 10.80% | 20/200 (10.0%) | 5.44 | 10.59 |
| vi | `thaythuocvietnam.vn` | 2,927 | 591 (20.19%) | 5 | 2,331 | 0 | 592 | 390 | 65.88% | 2,335 | 107.94% | 0/200 (0.0%) | 0.00 | 0.00 |
| vi | `tiemchunglongchau.com.vn` | 13,281 | 13,277 (99.97%) | 0 | 4 | 0 | 13,277 | 4,631 | 34.88% | 4 | 46.67% | — | — | — |
| vi | `tienphong.vn` | 3,021 | 3,021 (100.00%) | 0 | 0 | 0 | 3,021 | 2,321 | 76.83% | 0 | 12.58% | — | — | — |
| vi | `tuoitre.vn` | 3,893 | 3,893 (100.00%) | 0 | 0 | 0 | 3,893 | 3,201 | 82.22% | 0 | 8.30% | — | — | — |
| vi | `vietnamnet.vn` | 15,543 | 15,533 (99.94%) | 9 | 1 | 0 | 13,840 | 10,900 | 78.76% | 8 | 11.11% | — | — | — |
| vi | `vietnamplus.vn` | 3,695 | 3,695 (100.00%) | 0 | 0 | 0 | 3,695 | 3,220 | 87.14% | 0 | 4.74% | — | — | — |
| vi | `vinmec.com` | 26,129 | 26,124 (99.98%) | 3 | 2 | 0 | 26,124 | 10,650 | 40.77% | 5 | 36.87% | — | — | — |
| vi | `vnexpress.net` | 2,055 | 2,055 (100.00%) | 0 | 0 | 0 | 2,055 | 1,288 | 62.68% | 0 | 22.24% | — | — | — |
| vi | `vov.vn` | 9,784 | 0 (0.00%) | 0 | 9,784 | 0 | 0 | 0 | — | 9,784 | — | 38/200 (19.0%) | — | — |
| vi | `vov2.vov.vn` | 6,899 | 6,898 (99.99%) | 1 | 0 | 0 | 6,898 | 5,577 | 80.85% | 1 | 9.47% | — | — | — |
| vi | `youmed.vn` | 7,189 | 0 (0.00%) | 0 | 0 | 7,189 | 5,080 | 5,080 | 100.00% | 2,109 | 18.15% | 0/200 (0.0%) | 0.00 | 0.00 |
| zh | `120ask.com` | 98,160 | 23,005 (23.44%) | 132 | 0 | 75,023 | 22,741 | 3,397 | 14.94% | 75,418 | 20.50% | 0/200 (0.0%) | 0.00 | 0.00 |
| zh | `a-hospital.com` | 4,701 | 4,224 (89.85%) | 384 | 93 | 0 | 4,311 | 2,103 | 48.78% | 390 | 3.92% | 0/200 (0.0%) | 0.00 | 0.00 |
| zh | `article.iiyi.com` | 729 | 0 (0.00%) | 0 | 0 | 729 | 0 | 0 | — | 729 | — | 0/200 (0.0%) | — | — |
| zh | `ask.39.net` | 32,772 | 21,652 (66.07%) | 11,109 | 11 | 0 | 30,352 | 7,369 | 24.28% | 2,420 | 12.42% | 0/200 (0.0%) | 0.00 | 0.00 |
| zh | `ask.familydoctor.com.cn` | 1 | 0 (0.00%) | 0 | 1 | 0 | 0 | 0 | — | 1 | — | — | — | — |
| zh | `baby.39.net` | 3,368 | 3,348 (99.41%) | 20 | 0 | 0 | 3,367 | 1,369 | 40.66% | 1 | 10.42% | — | — | — |
| zh | `baidianfeng.familydoctor.com.cn` | 60 | 0 (0.00%) | 0 | 60 | 0 | 0 | 0 | — | 60 | — | — | — | — |
| zh | `baidu.com` | 11 | 0 (0.00%) | 0 | 11 | 0 | 0 | 0 | — | 11 | — | — | — | — |
| zh | `bingli.iiyi.com` | 5 | 0 (0.00%) | 0 | 5 | 0 | 0 | 0 | — | 5 | — | — | — | — |
| zh | `byby.39.net` | 37 | 37 (100.00%) | 0 | 0 | 0 | 37 | 23 | 62.16% | 0 | 0.00% | — | — | — |
| zh | `cancer.39.net` | 736 | 734 (99.73%) | 2 | 0 | 0 | 736 | 283 | 38.45% | 0 | 8.15% | — | — | — |
| zh | `care.39.net` | 1,282 | 1,255 (97.89%) | 4 | 23 | 0 | 1,282 | 561 | 43.76% | 0 | 6.94% | — | — | — |
| zh | `cnkang.com` | 40,046 | 37,143 (92.75%) | 2,088 | 815 | 0 | 37,923 | 17,566 | 46.32% | 2,121 | 7.32% | 0/200 (0.0%) | 0.00 | 0.00 |
| zh | `ek.39.net` | 448 | 448 (100.00%) | 0 | 0 | 0 | 448 | 159 | 35.49% | 0 | 0.00% | — | — | — |
| zh | `familydoctor.cn` | 2 | 2 (100.00%) | 0 | 0 | 0 | 2 | 1 | 50.00% | 0 | 0.00% | — | — | — |
| zh | `familydoctor.com.cn` | 26,819 | 26,626 (99.28%) | 188 | 5 | 0 | 26,814 | 11,954 | 44.58% | 5 | 7.38% | — | — | — |
| zh | `fitness.39.net` | 185 | 183 (98.92%) | 2 | 0 | 0 | 184 | 52 | 28.26% | 1 | 0.00% | — | — | — |
| zh | `fk.39.net` | 928 | 923 (99.46%) | 5 | 0 | 0 | 928 | 324 | 34.91% | 0 | 8.73% | — | — | — |
| zh | `food.39.net` | 594 | 570 (95.96%) | 24 | 0 | 0 | 594 | 215 | 36.20% | 0 | 10.94% | — | — | — |
| zh | `gan.39.net` | 294 | 291 (98.98%) | 3 | 0 | 0 | 294 | 134 | 45.58% | 0 | 0.00% | — | — | — |
| zh | `gc.39.net` | 59 | 59 (100.00%) | 0 | 0 | 0 | 59 | 29 | 49.15% | 0 | 0.00% | — | — | — |
| zh | `gk.39.net` | 90 | 89 (98.89%) | 1 | 0 | 0 | 90 | 51 | 56.67% | 0 | 0.00% | — | — | — |
| zh | `health.people.com.cn` | 50 | 44 (88.00%) | 0 | 6 | 0 | 50 | 23 | 46.00% | 0 | 0.00% | — | — | — |
| zh | `heart.39.net` | 1,055 | 1,043 (98.86%) | 12 | 0 | 0 | 1,054 | 474 | 44.97% | 1 | 6.17% | — | — | — |
| zh | `jb39.com` | 527 | 508 (96.39%) | 0 | 19 | 0 | 527 | 353 | 66.98% | 0 | 1.14% | — | — | — |
| zh | `jbk.39.net` | 2,033 | 1,927 (94.79%) | 68 | 38 | 0 | 2,033 | 1,223 | 60.16% | 0 | 3.15% | — | — | — |
| zh | `man.39.net` | 382 | 379 (99.21%) | 3 | 0 | 0 | 382 | 191 | 50.00% | 0 | 0.00% | — | — | — |
| zh | `msdmanuals.cn` | 905 | 811 (89.61%) | 6 | 88 | 0 | 833 | 272 | 32.65% | 72 | 8.04% | — | — | — |
| zh | `nk.39.net` | 762 | 759 (99.61%) | 3 | 0 | 0 | 761 | 363 | 47.70% | 1 | 7.49% | — | — | — |
| zh | `pf.39.net` | 85 | 84 (98.82%) | 1 | 0 | 0 | 85 | 47 | 55.29% | 0 | 0.00% | — | — | — |
| zh | `qihuangzhishu.com` | 29 | 28 (96.55%) | 1 | 0 | 0 | 28 | 18 | 64.29% | 1 | 0.00% | — | — | — |
| zh | `qy.familydoctor.com.cn` | 2 | 0 (0.00%) | 0 | 2 | 0 | 0 | 0 | — | 2 | — | — | — | — |
| zh | `shen.39.net` | 325 | 324 (99.69%) | 1 | 0 | 0 | 325 | 166 | 51.08% | 0 | 0.00% | — | — | — |
| zh | `sj.39.net` | 197 | 194 (98.48%) | 3 | 0 | 0 | 197 | 102 | 51.78% | 0 | 0.00% | — | — | — |
| zh | `test.pmphai.com` | 2,989 | 2,833 (94.78%) | 98 | 58 | 0 | 2,907 | 1,070 | 36.81% | 82 | 2.41% | — | — | — |
| zh | `tnb.39.net` | 717 | 715 (99.72%) | 2 | 0 | 0 | 717 | 282 | 39.33% | 0 | 10.46% | — | — | — |
| zh | `wei.39.net` | 225 | 221 (98.22%) | 4 | 0 | 0 | 224 | 140 | 62.50% | 1 | 0.00% | — | — | — |
| zh | `woman.39.net` | 687 | 685 (99.71%) | 2 | 0 | 0 | 687 | 270 | 39.30% | 0 | 10.77% | — | — | — |
| zh | `wujue.com` | 708 | 607 (85.73%) | 90 | 11 | 0 | 694 | 449 | 64.70% | 14 | 5.19% | — | — | — |
| zh | `youlai.cn` | 4,139 | 3,781 (91.35%) | 271 | 87 | 0 | 4,122 | 1,599 | 38.79% | 17 | 10.97% | — | — | — |
| zh | `zhongyibaodian.net` | 6,061 | 5,665 (93.47%) | 270 | 126 | 0 | 6,046 | 2,932 | 48.49% | 15 | 4.30% | — | — | — |
| zh | `zydcd.com` | 204 | 171 (83.82%) | 29 | 4 | 0 | 202 | 149 | 73.76% | 2 | 0.00% | — | — | — |
| zh | `zysjonline.com` | 6,591 | 0 (0.00%) | 0 | 6,591 | 0 | 0 | 0 | — | 6,591 | — | 0/200 (0.0%) | — | — |

* `Proxy thu hồi` = URL thiếu hiện tại × tỷ lệ mẫu CC × tỷ lệ chọn V1/index hiện tại. `Proxy/giờ CC` chia cho ETA mô hình, không phải số gold đã xác minh. Tỷ lệ chọn là cross-snapshot và có host >100%; không cap vì vậy, nhưng flag là proxy không phải xác suất.

## CC: mẫu và provenance

- Lấy 200 URL/host từ URL ID vắng trong static index hiện tại tại ngày 2026-10-17; 14 host top-15 theo số thiếu (trừ Pharmacity do chờ D18), tổng **2.800 URL**. Mẫu và ID được đóng băng trong `reports/s20-cc-samples.csv`; thuật toán/seed lấy mẫu không được ghi trong S16 nên khoảng tin cậy dưới đây giả định mẫu ngẫu nhiên, chưa kiểm chứng.
- Đối chiếu exact URL với CC index Parquet metadata-only trên 11/13 crawls (partial; lower bound): `CC-MAIN-2026-39` đến `CC-MAIN-2025-47`. Tính URL có ít nhất một capture `fetch_status=200` và `content_mime_type=text/html` trong các crawl đã quét. Không tải WARC, không fetch site. UA: `r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)`.
- Scan trạng thái `BudgetExceeded`: 11/13 crawl trong 721.4s; crawl chưa quét: CC-MAIN-2025-43, CC-MAIN-2025-38. Bytes của crawl đã hoàn tất: 3,531,770,599; ledger cộng dồn kể cả lượt scan trước bị dừng: 3,998,414,083 bytes / 525 request (cap 4,000,000,000 bytes, 5,000 request). Capture metadata: `reports/s20-cc-captures.csv`.

### 15 host nhiều URL thiếu nhất trong static index ngày 2026-10-17

| # | Lang/host | URL thiếu | Mẫu CC 13 crawl |
|---:|---|---:|---:|
| 1 | zh `120ask.com` | 75,418 | 0/200 (0.0%) |
| 2 | vi `vov.vn` | 9,784 | 38/200 (19.0%) |
| 3 | zh `zysjonline.com` | 6,591 | 0/200 (0.0%) |
| 4 | vi `pmc-ecm-healthblog.beta.pharmacity.io` | 5,333 | Không lấy mẫu — chờ D18 |
| 5 | vi `laodong.vn` | 4,098 | 15/200 (7.5%) |
| 6 | vi `suckhoedoisong.vn` | 3,205 | 77/200 (38.5%) |
| 7 | vi `nhathuoclongchau.com.vn` | 2,730 | 1/200 (0.5%) |
| 8 | zh `ask.39.net` | 2,420 | 0/200 (0.0%) |
| 9 | vi `thaythuocvietnam.vn` | 2,335 | 0/200 (0.0%) |
| 10 | zh `cnkang.com` | 2,121 | 0/200 (0.0%) |
| 11 | vi `youmed.vn` | 2,109 | 0/200 (0.0%) |
| 12 | vi `baolangson.vn` | 1,981 | 0/200 (0.0%) |
| 13 | zh `article.iiyi.com` | 729 | 0/200 (0.0%) |
| 14 | vi `thanhnien.vn` | 504 | 20/200 (10.0%) |
| 15 | zh `a-hospital.com` | 390 | 0/200 (0.0%) |

| Host | CC hit | Tỷ lệ | Wilson 95%* |
|---|---:|---:|---:|
| `120ask.com` | 0/200 | 0.0% | 0.0–1.9% |
| `nhathuoclongchau.com.vn` | 1/200 | 0.5% | 0.1–2.8% |
| `vov.vn` | 38/200 | 19.0% | 14.2–25.0% |
| `youmed.vn` | 0/200 | 0.0% | 0.0–1.9% |
| `zysjonline.com` | 0/200 | 0.0% | 0.0–1.9% |
| `laodong.vn` | 15/200 | 7.5% | 4.6–12.0% |
| `suckhoedoisong.vn` | 77/200 | 38.5% | 32.0–45.4% |
| `ask.39.net` | 0/200 | 0.0% | 0.0–1.9% |
| `thaythuocvietnam.vn` | 0/200 | 0.0% | 0.0–1.9% |
| `cnkang.com` | 0/200 | 0.0% | 0.0–1.9% |
| `baolangson.vn` | 0/200 | 0.0% | 0.0–1.9% |
| `article.iiyi.com` | 0/200 | 0.0% | 0.0–1.9% |
| `thanhnien.vn` | 20/200 | 10.0% | 6.6–14.9% |
| `a-hospital.com` | 0/200 | 0.0% | 0.0–1.9% |

* Wilson 95% là khoảng tin cậy nhị thức, chỉ hợp lệ nếu 200 URL/host được chọn ngẫu nhiên; seed/phương pháp lấy mẫu chưa có bằng chứng nên coi là tham khảo.

## Xếp hạng khoảng trống — top 10 theo proxy thu hồi/giờ

Chỉ xếp các host trong top-15 thiếu có index hiện tại và V1 denominator >0. Hosts chưa index (VOV, Zysjonline, Baolangson, iiyi) không có selection-rate denominator; Pharmacity bị loại khỏi CC do D18. CC rate hiện là lower bound từ crawl đã quét; ranking bên dưới chỉ provisional và phải tính lại sau 2 crawl còn lại. Các score bằng nhau được tie-break theo URL thiếu hiện tại giảm dần.

ETA mô hình cho mỗi host: `0,5 giờ` chuẩn bị/trích xuất/QC + `số URL CC kỳ vọng / 3.600` (giả định 1 WARC range request/URL, 1 request/s). Dung lượng = số URL CC kỳ vọng × mean `warc_record_length` của capture mới nhất trong mẫu. Chưa tính giờ GPU/index vì chưa có throughput phù hợp theo từng nguồn; không gọi ETA này là end-to-end.

| # | Lang/host | Thiếu URL | CC mẫu | V1/index | URL CC kỳ vọng | Doc proxy | WARC record bytes ước tính | ETA CC (h) | Proxy doc/h | Cách lấy / ETA / rủi ro |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | vi `suckhoedoisong.vn` | 3,205 | 77/200 (38.5%) | 18.2% | 1233.9 | 225.0 | 156,618,110 | 0.84 | 267.0 | CC index → tải range WARC → extract → index; chạy theo batch, đánh giá top-300 sau index. Rủi ro: CC hit không đồng nghĩa text/gold; precision theo host chưa đo. |
| 2 | vi `laodong.vn` | 4,098 | 15/200 (7.5%) | 6.4% | 307.4 | 19.5 | 238,094 | 0.59 | 33.4 | CC index → tải range WARC → extract → index; chạy theo batch, đánh giá top-300 sau index. Rủi ro: CC hit không đồng nghĩa text/gold; precision theo host chưa đo. |
| 3 | vi `thanhnien.vn` | 504 | 20/200 (10.0%) | 10.8% | 50.4 | 5.4 | 6,109,017 | 0.51 | 10.6 | CC index → tải range WARC → extract → index; chạy theo batch, đánh giá top-300 sau index. Rủi ro: CC hit không đồng nghĩa text/gold; precision theo host chưa đo. |
| 4 | vi `nhathuoclongchau.com.vn` | 2,730 | 1/200 (0.5%) | 14.8% | 13.7 | 2.0 | 765,956 | 0.50 | 4.0 | Để S3 hiện hành hoàn tất 61k URL rồi đối chiếu delta; không chạy song song trên state. CC là dự phòng. Rủi ro: CC exact-match trong 11 crawl thấp; độ phủ archive không đồng nhất với precision gold. |
| 5 | zh `120ask.com` | 75,418 | 0/200 (0.0%) | 20.5% | 0.0 | 0.0 | — | 0.50 | 0.0 | S11 đang theo dõi robots; chỉ dùng CC index/WARC, tuyệt đối không fetch site. Chờ gate S11 trước batch. Rủi ro: CC exact-match trong 11 crawl thấp; độ phủ archive không đồng nhất với precision gold. |
| 6 | zh `ask.39.net` | 2,420 | 0/200 (0.0%) | 12.4% | 0.0 | 0.0 | — | 0.50 | 0.0 | CC index → tải range WARC → extract → index; chạy theo batch, đánh giá top-300 sau index. Rủi ro: CC exact-match trong 11 crawl thấp; độ phủ archive không đồng nhất với precision gold. |
| 7 | vi `thaythuocvietnam.vn` | 2,335 | 0/200 (0.0%) | 107.9% | 0.0 | 0.0 | — | 0.50 | 0.0 | CC index → tải range WARC → extract → index; chạy theo batch, đánh giá top-300 sau index. Rủi ro: V1/index >100% do lệch snapshot/ID; proxy bị phóng đại. CC exact-match trong 11 crawl thấp; độ phủ archive không đồng nhất với precision gold. |
| 8 | zh `cnkang.com` | 2,121 | 0/200 (0.0%) | 7.3% | 0.0 | 0.0 | — | 0.50 | 0.0 | CC index → tải range WARC → extract → index; chạy theo batch, đánh giá top-300 sau index. Rủi ro: CC exact-match trong 11 crawl thấp; độ phủ archive không đồng nhất với precision gold. |
| 9 | vi `youmed.vn` | 2,109 | 0/200 (0.0%) | 18.1% | 0.0 | 0.0 | — | 0.50 | 0.0 | Để S3 hiện hành hoàn tất 61k URL rồi đối chiếu delta; không chạy song song trên state. CC là dự phòng. Rủi ro: CC exact-match trong 11 crawl thấp; độ phủ archive không đồng nhất với precision gold. |
| 10 | zh `a-hospital.com` | 390 | 0/200 (0.0%) | 3.9% | 0.0 | 0.0 | — | 0.50 | 0.0 | CC index → tải range WARC → extract → index; chạy theo batch, đánh giá top-300 sau index. Rủi ro: CC exact-match trong 11 crawl thấp; độ phủ archive không đồng nhất với precision gold. |

## Precision biên và cách hiểu selection rate

- Điểm đo toàn cục đã biết: zh doc 13→6%, vi doc 7→4,9%, hòa vốn khoảng 5,5%. Chưa có phép tách precision theo host từ S16/V1; không gán các con số này cho từng host.
- `V1 docs / current indexed docs` là selection rate proxy theo định nghĩa của Vu. Nó không phải precision. Một số host có denominator index hiện tại khác snapshot V1; `thaythuocvietnam.vn` cho tỷ lệ >100% do numerator V1 `Tổng` có ID slug/no-text. Công thức top-10 giữ nguyên tỷ lệ thô, nên score này chỉ dùng xếp thăm dò và cần reconcile ID trước khi chi ngân sách lớn.
- `CC 200 HTML` chỉ chứng minh có bản archive theo exact URL trong các crawl đã quét; không chứng minh WARC giải nén được, extract tốt, còn đúng corpus date hoặc chứa gold. Mẫu URL cùng path nhưng khác scheme/www/trailing slash không được tính; normalized-match chưa đo. Top-10 “doc proxy” có thể over/underestimate gold.

## Chia session lấp khoảng trống

1. **Giữ S3/S11 nguyên trạng**; ghi nhận đến khi có snapshot kết thúc. Sau S3, reconcile lại Longchau/Youmed theo ID để tránh tải trùng. Với 120ask/iiyi tiếp tục chỉ CC route; không live fetch.
2. **Sau khi đóng băng ID/index snapshot**, chia các host độc lập thành batch CC riêng; metadata lookup và ước lượng WARC có thể song song nếu cùng một shared rate limiter, output path tách biệt và còn quota/disk. Không chạy song song writer trên state/index chung.
3. Mỗi batch đi tuần tự: CC index → WARC range download → validate/extract → ingest/index → truy vấn lại 1.200 query/top-300. Mỗi gate phụ thuộc hash/inventory và dung lượng trước batch kế tiếp.
4. **Pharmacity chờ D18**; hiện chỉ báo cáo 5.333 URL thiếu index, không lấy mẫu CC và không xếp hạng hành động.

## Giả định và giới hạn

- S16 chuẩn: 1.220.598 URL (hash ở đầu báo cáo); S16 inventory đo 2026-10-16 11:45:55 UTC+7, HF revision `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`.
- Tầng 1 dùng `extract_ok` proxy cho Longchau, Youmed, 120ask, iiyi theo quyết định của Vu; các host còn lại đếm trực tiếp body sau whitespace collapse với ngưỡng 200 code point.
- Tầng 2 top-300 riêng trên mỗi index vi/zh. Cache vi dùng `out/runs/pool300-2026-10-06/full/vi_k100_pairs.parquet` (S16 t256 SHA khớp); zh dùng snapshot 62 segment/782.638 row tại 2026-10-17. Trạng thái index tĩnh đã không còn bất biến do S3/S11 chạy; ghi nhận thời điểm trong tên CSV.
- Current selection denominator = live docs ở static index ngày 2026-10-17; numerator = `relevant_docs_unique_doc_ids` từ `out/runs/s16/inventory_best_v1.csv`, chỉ dòng `doc_type=Tổng` (V1 JSON stats đã ghi ở S16). Đây là tỷ số cross-snapshot.
- CC URL sample frame là các ID không có trong static current index; 200/host, Pharmacity bỏ qua. So khớp exact URL nên các biến thể URL không được đếm. CC chỉ dùng cho URL corpus v2, UA do Vu cung cấp. Không tải WARC/fetch site/ingest/crawl hàng loạt/nộp bài trong S20.
- File số liệu: `reports/s20-coverage-audit.csv` (94 host rows, raw + derived); `reports/s20-cc-samples.csv` (2.800 sample URL); `reports/s20-cc-captures.csv` (exact capture metadata).
