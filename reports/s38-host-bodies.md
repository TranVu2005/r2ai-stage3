# S38: độ dài body theo host (doc nhận chunk của J3-thrC)

Nguồn: `out/runs/s38-chunk-metric-v2/scores/bodies.parquet` (175.148 carrier = cặp query–doc mang chunk của J3-thrC; body = chunk J3, token BGE-M3 thô không special token). 81 host. Toàn bộ: median 999, P90 1.798; ≤ 384: 16,8%, ≤ 768: 38,7%, ≤ 1024: 51,1%, ≤ 1536: 78,6%, ≤ 2048: 95,1%.

## 10 host nhiều carrier nhất

| Host | Lang | Carrier | Doc | P10 | Median | P90 | % > 1.024 |
|---|---|---:|---:|---:|---:|---:|---:|
| suckhoedoisong.vn | vi | 24.914 | 14.873 | 364 | 986 | 1.576 | 47,1% |
| nhathuoclongchau.com.vn | vi | 23.067 | 15.049 | 1.040 | 1.569 | 1.981 | 90,3% |
| thanhnien.vn | vi | 15.495 | 9.258 | 217 | 528 | 1.079 | 12,4% |
| medlatec.vn | vi | 15.078 | 8.530 | 1.324 | 1.472 | 1.729 | 99,1% |
| vinmec.com | vi | 11.778 | 7.337 | 174 | 1.171 | 2.082 | 56,2% |
| ask.39.net | zh | 7.173 | 5.895 | 87 | 210 | 510 | 0,6% |
| suckhoecongdongonline.vn | vi | 6.935 | 4.412 | 363 | 725 | 1.361 | 25,6% |
| tiemchunglongchau.com.vn | vi | 6.687 | 4.058 | 1.335 | 1.638 | 2.290 | 97,5% |
| 120ask.com | zh | 6.452 | 5.211 | 99 | 200 | 764 | 6,2% |
| cnkang.com | zh | 6.022 | 5.255 | 218 | 524 | 1.289 | 16,6% |

## 10 host có body trung vị dài nhất (≥ 50 carrier)

| Host | Lang | Carrier | Doc | P10 | Median | P90 | Max | % > 1.024 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| youmed.vn | vi | 1.524 | 1.015 | 1.398 | 1.914 | 3.185 | 7.593 | 98,0% |
| baophutho.vn | vi | 77 | 61 | 1.239 | 1.749 | 14.779 | 23.442 | 96,1% |
| thaythuocvietnam.vn | vi | 166 | 106 | 1.323 | 1.688 | 2.193 | 2.981 | 96,4% |
| tiemchunglongchau.com.vn | vi | 6.687 | 4.058 | 1.335 | 1.638 | 2.290 | 6.270 | 97,5% |
| hellobacsi.com | vi | 3.351 | 2.156 | 970 | 1.605 | 2.667 | 9.391 | 88,3% |
| nhathuoclongchau.com.vn | vi | 23.067 | 15.049 | 1.040 | 1.569 | 1.981 | 7.058 | 90,3% |
| tamanhhospital.vn | vi | 2.320 | 1.308 | 1.192 | 1.546 | 2.126 | 12.478 | 98,1% |
| medlatec.vn | vi | 15.078 | 8.530 | 1.324 | 1.472 | 1.729 | 12.815 | 99,1% |
| a-hospital.com | zh | 499 | 416 | 342 | 1.184 | 5.197 | 14.790 | 56,7% |
| benhviennhitrunguong.gov.vn | vi | 358 | 244 | 484 | 1.176 | 1.902 | 3.204 | 59,5% |

## Đuôi body (300 ký tự cuối, một doc có độ dài gần trung vị của host)

**youmed.vn** (doc 765160, 1.914 token)

```text
rong bài viết của chúng tôi. Tìm hiểu về Quy trình biên tập để hiểu rõ hơn cách chúng tôi đảm bảo nội dung luôn chính xác, minh bạch và tin cậy.

-

Remedies for Low Sperm Count and Motilityhttps://www.webmd.com/infertility-and-reproduction/remedies-low-sperm-count-motilityNgày tham khảo: 17/04/2021
```

**baophutho.vn** (doc 81311, 1.749 token)

```text
 thiết bị và hạ tầng số đang được ngành Y tế Phú Thọ triển khai với tổng nguồn lực hàng nghìn tỷ...

2026-09-29 14:16:00

Từ chỗ người bệnh phải chờ 2,5-3 giờ để nhận kết quả xét nghiệm, thời gian này tại Bệnh viện Bạch Mai nay được rút xuống dưới 1 giờ; với bệnh nhân cấp cứu, chỉ khoảng 45 phút....
```

**thaythuocvietnam.vn** (doc 661686, 1.677 token)

```text
nh thai, cấy que tránh thai,… Nếu phải uống thuốc tránh thai khẩn cấp trong những tình huống bất khả kháng, chị em phụ nữ nên sử dụng thuốc đúng hướng dẫn, kết hợp với thực hiện các biện pháp khắc phục tình trạng rối loạn kinh nguyệt thường gặp do thuốc tránh thai khẩn cấp gây ra.

BS Đoàn Thị Nhung
```

**tiemchunglongchau.com.vn** (doc 672690, 1.638 token)

```text
m lấn do các nhóm A, C, W-135 và Y, có thể gây tử vong nhanh nếu không điều trị kịp thời. MenQuadfi có thể tiêm cho trẻ từ 6 tuần tuổi trở lên và người lớn, mang lại hiệu quả miễn dịch lâu dài. Việc kết hợp các loại vắc xin phù hợp theo độ tuổi sẽ giúp nâng cao hiệu quả phòng bệnh toàn diện cho trẻ.
```

**hellobacsi.com** (doc 126906, 1.605 token)

```text
ợp (bệnh nhiễm độc tuyến giáp do T3) thì T3 tăng nhưng T4 lại bình thường.

Một số trường hợp có chỉ số T3 bình thường xét nghiệm hormone TSH cho chỉ số cao, chẳng hạn như trong suy giáp.

Nếu bạn có bất kỳ câu hỏi nào, hãy tham khảo ý kiến bác sĩ để được tư vấn phương pháp hỗ trợ điều trị tốt nhất.
```

**nhathuoclongchau.com.vn** (doc 265565, 1.569 token)

```text
 cho sức khỏe.

Xem thêm: Mổ u tuyến giáp nằm viện bao lâu? Cần lưu ý những gì?

Thông tin và sản phẩm gợi ý trong bài viết chỉ mang tính chất tham khảo, vui lòng liên hệ với Bác sĩ, Dược sĩ hoặc chuyên viên y tế để được tư vấn cụ thể. Xem thêm

Dược sĩ Đại họcNguyễn Mỹ Huyền

Đã kiểm duyệt nội dung
```

**tamanhhospital.vn** (doc 570191, 1.546 token)

```text
, Phường Cầu Giấy, TP. Hà Nội

- (Đ/c cũ: 265 Cầu Giấy, P.Dịch Vọng, Q.Cầu Giấy, TP. Hà Nội)

- Hotline: 024 3872 3872 – 024 7106 6858

- Fanpage: https://www.facebook.com/tamanhhospital.vn

- Website: tamanhhospital.vn

Cập nhật lần cuối: 05:01 10/07/2026

Chia sẻ:

Chủ đề:

BÀI VIẾT CÙNG CHỦ ĐỀ

›
```

**medlatec.vn** (doc 175163, 1.472 token)

```text
ũ chuyên gia, y bác sĩ đầu ngành, dày dặn kinh nghiệm cùng hệ thống máy móc, thiết bị hiện đại, đồng bộ các chuyên khoa nhằm phục vụ tốt nhất nhu cầu kiểm tra sức khỏe của người dân.

Để được tư vấn hoặc đặt lịch khám tại MEDLATEC, người dân vui lòng liên hệ tổng đài 1900 56 56 56, hoạt động 24/24h.
```

**a-hospital.com** (doc 3103610, 1.184 token)

```text
佛卡因1-2ml封闭。急性颈肌劳损引起者可用醋酸可的松1ml（加1%奴佛加因1-2ml）封闭。因颈椎增生或损伤引起者应加颈椎牵引，并加用颈托以巩固牵引疗效。

五、神经炎性头痛：除按神经炎原则治疗外，可在眶上切迹、“风池”穴等处多次用2%奴佛加因0.5-1ml（或加入Vi.t B1 50mg或Vit.B12 100μg）封闭，或一次用无水酒精0.5ml封闭。口服苯妥英钠或卡马西平也对止痛有效。对颈椎增生引起的枕大神经痛应加用颈椎牵引。

| 头痛的诊断问题 \| 癫痫 |

| 关于“神经病学/头痛的治疗原则”的留言： | 订阅讨论RSS |

| 目前暂无留言 | |

| 添加留言 | |
```

**benhviennhitrunguong.gov.vn** (doc 97567, 1.176 token)

```text
 vai trò tích cực và chủ động trong tất cả các hoạt động can thiệp cho trẻ. Can thiệp tự kỷ là một quá trình lâu dài, không có hạn định về thời gian. Vì vậy, điều quan trọng là sự quyết tâm, kiên trì, tin tưởng, yêu thương và sự đồng hành của cha mẹ với trẻ” –Bác sĩ Thành Ngọc Minh cho biết.

Lê Mai
```
## Đọc nhanh boilerplate ở đuôi (một doc mỗi host, chưa đếm trên toàn host)

- Rõ boilerplate: **tamanhhospital.vn** (địa chỉ, hotline, fanpage, "Cập nhật lần cuối", "Chia sẻ / Chủ đề / BÀI VIẾT CÙNG CHỦ ĐỀ"); **baophutho.vn** (danh sách tin khác kèm giờ đăng; P90 14.779 token, max 23.442: body gồm cả các tin liên quan); **a-hospital.com** (khung bình luận "留言 / 订阅讨论RSS / 添加留言"); **nhathuoclongchau.com.vn** (câu miễn trừ "Thông tin và sản phẩm gợi ý…", tên dược sĩ, "Đã kiểm duyệt nội dung"); **youmed.vn** (câu về quy trình biên tập, nguồn tham khảo kèm URL); **medlatec.vn** (đoạn quảng cáo đặt lịch, tổng đài).
- Gần như sạch: thaythuocvietnam.vn, hellobacsi.com, tiemchunglongchau.com.vn, benhviennhitrunguong.gov.vn (đuôi là nội dung hoặc tên tác giả).
- Chưa đo boilerplate trên toàn host (mới xem 1 doc/host); nếu cần, bước tiếp là đếm các dòng đuôi lặp lại nguyên văn theo host.
