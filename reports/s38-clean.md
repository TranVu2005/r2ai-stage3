# S38 CH-clean: cắt đuôi boilerplate theo host

`python -m r2ai.s38.clean discover` → `apply` → `python -m r2ai.s38.build --variant CH-clean`. Chunk = tiền tố liền mạch của body J3-thrC, kết thúc trước dòng marker đầu tiên nằm trong 30% cuối body (khoảng trắng cuối bị bỏ). Marker của host = khoá dòng (NFKC, chữ thường, chữ số → 0, gộp khoảng trắng, 80 ký tự đầu, ≥ 10 ký tự) có mặt trong 30% cuối body ở ≥ 30% doc của host (đếm doc_id duy nhất, trên mọi carrier của host). Doc của host khác và doc không có marker giữ nguyên byte J3-thrC. Một chunk cắt phải là đoạn nguyên văn của doc_text canonical (validator S26).

| Host | Marker | Carrier | Doc | Carrier cắt | Doc cắt | Token bỏ (median / TB, theo doc) | % body bỏ (median) |
|---|---:|---:|---:|---:|---:|---|---:|
| tamanhhospital.vn | 22 | 2.320 | 1.308 | 2.320 | 1.308 | 394 / 376 | 25,2% |
| baophutho.vn | 10 | 77 | 61 | 77 | 61 | 379 / 370 | 21,7% |
| a-hospital.com | 2 | 499 | 416 | 462 | 383 | 19 / 19 | 1,4% |
| nhathuoclongchau.com.vn | 2 | 23.067 | 15.049 | 21.128 | 13.807 | 55 / 59 | 3,5% |
| youmed.vn | 2 | 1.524 | 1.015 | 1.404 | 935 | 189 / 230 | 9,9% |
| medlatec.vn | 0 | 15.078 | 8.530 | 0 | 0 | – | – |

Tổng: 25.391 / 175.148 carrier bị cắt. tamanhhospital: 1.842 carrier có chunk J3 không nằm nguyên văn trong doc_text canonical (chunk cũ của P1); tiền tố cắt của cả 1.842 đều là đoạn nguyên văn của doc_text nên vẫn dùng được.

**medlatec.vn: không cắt.** Đoạn quảng cáo đặt lịch diễn đạt khác nhau theo bài: khoá dòng 80 ký tự đạt tối đa 1% doc; thử khoá 40 / 30 / 20 ký tự đầu cũng chỉ 1,5% / 4,0% / 9,4% doc, đều dưới ngưỡng 30%.

**baophutho.vn:** marker là khung bình luận (`{name} - {time}`, "Ý kiến của bạn sẽ được biên tập…") và danh sách tin khác (dòng giờ đăng, tiêu đề tin). P90 body 14.779 token là do một khối base64 nằm trước khung bình luận; khối này không phải marker nên vẫn còn sau khi cắt (ví dụ 1 dưới đây).

## Marker theo host

**tamanhhospital.vn** (1.308 doc): `Cập nhật lần cuối: 11:02 09/09/2024` (100,0% doc); `- Hotline: 024 3872 3872 – 024 7106 6858` (95,9% doc); `- Hotline: 093 180 6858 – 0287 102 6789` (95,9% doc); `- Bệnh viện Đa khoa Tâm Anh – Quận 8:` (95,9% doc); `- 316C Phạm Hùng, Phường Chánh Hưng, TP.HCM` (95,9% doc); `- (Đ/c cũ: 316C Phạm Hùng, P.5, Q.8, TP.HCM)` (95,9% doc); `- Phòng khám Đa khoa Tâm Anh Quận 7:` (95,9% doc); `- 25 Nguyễn Hữu Thọ, Phường Tân Hưng, TP.HCM` (95,9% doc); `- (Đ/c cũ: 25 Nguyễn Hữu Thọ, P.Tân Hưng, Q.7, TP.HCM)` (95,9% doc); `- Phòng khám Đa khoa Tâm Anh Cầu Giấy:` (95,9% doc); `- 265 Cầu Giấy, Phường Cầu Giấy, TP. Hà Nội` (95,9% doc); `- (Đ/c cũ: 265 Cầu Giấy, P.Dịch Vọng, Q.Cầu Giấy, TP. Hà Nội)` (95,9% doc); `- Fanpage: https://www.facebook.com/tamanhhospital.vn` (95,9% doc); `- Website: tamanhhospital.vn` (95,9% doc); `- 2B Phổ Quang, Phường Tân Sơn Hòa, TP.HCM` (95,8% doc); `- (Đ/c cũ: 2B Phổ Quang, P.2, Q.Tân Bình, TP.HCM)` (95,8% doc); `- Bệnh viện Đa khoa Tâm Anh TP.HCM:` (95,7% doc); `- (Đ/c cũ: 108 Hoàng Như Tiếp, P.Bồ Đề, Q.Long Biên, TP. Hà Nội)` (94,8% doc); `- 108 Phố Hoàng Như Tiếp, Phường Bồ Đề, TP. Hà Nội` (93,7% doc); `- Bệnh viện Đa khoa Tâm Anh Hà Nội:` (92,7% doc); `HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH` (91,7% doc); `BÀI VIẾT CÙNG CHỦ ĐỀ` (72,3% doc)

**baophutho.vn** (61 doc): `2026-10-02 10:32:00` (100,0% doc); `baophutho.vn Những năm qua, cùng với nâng cao chất lượng khám, chữa bệ` (100,0% doc); `Hàng loạt vướng mắc từ danh mục kĩ thuật, quy định báo cáo cho đến áp` (100,0% doc); `baophutho.vn Xây dựng môi trường khám, chữa bệnh an toàn, trong lành,` (100,0% doc); `baophutho.vn Hàng loạt dự án đầu tư cơ sở vật chất, trang thiết bị và` (100,0% doc); `Từ chỗ người bệnh phải chờ 2,5-3 giờ để nhận kết quả xét nghiệm, thời` (100,0% doc); `baophutho.vn Thiết kế nhỏ gọn, nhiều màu sắc, kiểu dáng và hương vị đa` (96,7% doc); `Cục Phòng bệnh, Bộ Y tế cho biết, trong một số tuần gần đây, số trường` (95,1% doc); `Ý kiến của bạn sẽ được biên tập trước khi đăng. Vui lòng gõ tiếng Việt` (86,9% doc); `{name} - {time}` (85,2% doc)

**a-hospital.com** (416 doc): `| 添加留言 | |` (92,1% doc); `| 目前暂无留言 | |` (87,7% doc)

**nhathuoclongchau.com.vn** (15.049 doc): `Thông tin và sản phẩm gợi ý trong bài viết chỉ mang tính chất tham khả` (91,6% doc); `Đã kiểm duyệt nội dung` (80,0% doc)

**youmed.vn** (1.015 doc): `Nguồn tham khảo / Source` (92,1% doc); `Trang tin y tế YouMed chỉ sử dụng các nguồn tham khảo có độ uy tín cao` (92,1% doc)

**medlatec.vn** (8.530 doc): không có

## Ví dụ trước / sau (2 doc mỗi host)

Trước = 200 ký tự trước điểm cắt + 400 ký tự sau; sau = 200 ký tự cuối của chunk đã cắt.

**tamanhhospital.vn**, doc 569097: 1.845 → 1.452 token

Trước:
```text
 tình dục, rối loạn cương, rối loạn xuất tinh, bộ phận sinh dục teo nhỏ… nên chủ động khám giúp tăng tỷ lệ thành công, giảm chi phí, cải thiện chất lượng cuộc sống.

*Tên người bệnh đã được thay đổi

HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH

- Bệnh viện Đa khoa Tâm Anh Hà Nội:

- 108 Phố Hoàng Như Tiếp, Phường Bồ Đề, TP. Hà Nội

- (Đ/c cũ: 108 Hoàng Như Tiếp, P.Bồ Đề, Q.Long Biên, TP. Hà Nội)

- Hotline: 024 3872 3872 – 024 7106 6858

- Bệnh viện Đa khoa Tâm Anh TP.HCM:

- 2B Phổ Quang, Phường Tân Sơn Hòa, TP.HCM

- (Đ/c cũ: 2B Phổ Quang, P.2, Q.Tân Bình, TP.HCM)

- Hotline: 093 180 6858 – 0287 102 
```
Sau (đuôi):
```text
ốn tình dục, rối loạn cương, rối loạn xuất tinh, bộ phận sinh dục teo nhỏ… nên chủ động khám giúp tăng tỷ lệ thành công, giảm chi phí, cải thiện chất lượng cuộc sống.

*Tên người bệnh đã được thay đổi
```

**tamanhhospital.vn**, doc 569098: 1.858 → 1.465 token

Trước:
```text
 kỳ 6 tháng một lần để phát hiện và điều trị kịp thời, tránh bệnh diễn tiến nặng. Đặc biệt là nữ giới 6 tháng đến 1 năm không sử dụng biện pháp tránh thai mà không có thai, nên chủ động đi khám sớm.

HỆ THỐNG BỆNH VIỆN ĐA KHOA TÂM ANH

- Bệnh viện Đa khoa Tâm Anh Hà Nội:

- 108 Phố Hoàng Như Tiếp, Phường Bồ Đề, TP. Hà Nội

- (Đ/c cũ: 108 Hoàng Như Tiếp, P.Bồ Đề, Q.Long Biên, TP. Hà Nội)

- Hotline: 024 3872 3872 – 024 7106 6858

- Bệnh viện Đa khoa Tâm Anh TP.HCM:

- 2B Phổ Quang, Phường Tân Sơn Hòa, TP.HCM

- (Đ/c cũ: 2B Phổ Quang, P.2, Q.Tân Bình, TP.HCM)

- Hotline: 093 180 6858 – 0287 102 
```
Sau (đuôi):
```text
nh kỳ 6 tháng một lần để phát hiện và điều trị kịp thời, tránh bệnh diễn tiến nặng. Đặc biệt là nữ giới 6 tháng đến 1 năm không sử dụng biện pháp tránh thai mà không có thai, nên chủ động đi khám sớm.
```

**baophutho.vn**, doc 80459: 15.486 → 15.107 token

Trước:
```text
HhuqNd4bugJcOq4bqtN33hu6A2xq9d4bugZDkk4bugNFvDoCbhu6A24buqN1vhu6Bb4bqzN8OU4buUL+G6qeG7luG7lOG6qeG7oCQ1xahjY0zhu57huqlqxJFkWzhi4bue4buWREPhu6Dhu5JEWyY44bugY8SRJDRbOCYlOF1jODd9w5RmN+G7mOG7lC/huqnhu5Y=

{name} - {time}

{body}

Ý kiến của bạn sẽ được biên tập trước khi đăng. Vui lòng gõ tiếng Việt có dấu

2026-10-02 10:32:00

Cục Phòng bệnh, Bộ Y tế cho biết, trong một số tuần gần đây, số trường hợp mắc cúm mùa có xu hướng gia tăng nhẹ. Tuy nhiên, hệ thống giám sát chưa ghi nhận sự gia tăng bất...

2026-10-02 08:15:00

baophutho.vn Thiết kế nhỏ gọn, nhiều màu sắc, kiểu dáng và hương vị đa dạng, t
```
Sau (đuôi):
```text
xJHhuqNd4bugJcOq4bqtN33hu6A2xq9d4bugZDkk4bugNFvDoCbhu6A24buqN1vhu6Bb4bqzN8OU4buUL+G6qeG7luG7lOG6qeG7oCQ1xahjY0zhu57huqlqxJFkWzhi4bue4buWREPhu6Dhu5JEWyY44bugY8SRJDRbOCYlOF1jODd9w5RmN+G7mOG7lC/huqnhu5Y=
```

**baophutho.vn**, doc 80483: 1.786 → 1.407 token

Trước:
```text
hận biết sớm các dấu hiệu đột quỵ như yếu tay chân, méo miệng, nói khó, chóng mặt, đau đầu dữ dội... để kịp thời đưa người bệnh đến các cơ sở y tế chuyên khoa.

TS (Biên soạn theo suckhoedoisong.vn)

{name} - {time}

{body}

Ý kiến của bạn sẽ được biên tập trước khi đăng. Vui lòng gõ tiếng Việt có dấu

2026-10-02 10:32:00

Cục Phòng bệnh, Bộ Y tế cho biết, trong một số tuần gần đây, số trường hợp mắc cúm mùa có xu hướng gia tăng nhẹ. Tuy nhiên, hệ thống giám sát chưa ghi nhận sự gia tăng bất...

2026-10-02 08:15:00

baophutho.vn Thiết kế nhỏ gọn, nhiều màu sắc, kiểu dáng và hương vị đa dạng, t
```
Sau (đuôi):
```text
 nhận biết sớm các dấu hiệu đột quỵ như yếu tay chân, méo miệng, nói khó, chóng mặt, đau đầu dữ dội... để kịp thời đưa người bệnh đến các cơ sở y tế chuyên khoa.

TS (Biên soạn theo suckhoedoisong.vn)
```

**a-hospital.com**, doc 2992448: 399 → 380 token

Trước:
```text
是上皮细胞堆积而引起。一般无任何症状，对口腔颌面部的发育和健康没有任何影响，不需做任何处理，切不可用针挑或用毛巾擦，否则容易损伤粘膜，造成感染。如果长期不脱落，应请医生诊治。

出自A+医学百科 “上皮珠”条目 https://www.a-hospital.com/w/%E4%B8%8A%E7%9A%AE%E7%8F%A0 转载请保留此链接

| 关于“上皮珠”的留言： | 订阅讨论RSS |

| 目前暂无留言 | |

| 添加留言 | |
```
Sau (đuôi):
```text
。这是上皮细胞堆积而引起。一般无任何症状，对口腔颌面部的发育和健康没有任何影响，不需做任何处理，切不可用针挑或用毛巾擦，否则容易损伤粘膜，造成感染。如果长期不脱落，应请医生诊治。

出自A+医学百科 “上皮珠”条目 https://www.a-hospital.com/w/%E4%B8%8A%E7%9A%AE%E7%8F%A0 转载请保留此链接

| 关于“上皮珠”的留言： | 订阅讨论RSS |
```

**a-hospital.com**, doc 2992878: 1.696 → 1.677 token

Trước:
```text
看

| |

| 女性内生殖器 | |

| 女性外生殖器 | |

| | |

|---|---|

| | |

| 其他 | |

| 其他 | |

| 先天畸形 | |

| |

| 不孕症 | |

| 生育药物 | |

| 体外人工受精 (IVF) | |

| 其他技术 | |

| 捐献 | |

| 伦理 | |

| 关于“不孕症”的留言： | 订阅讨论RSS |

| 目前暂无留言 | |

| 添加留言 | |
```
Sau (đuôi):
```text

参看

| |

| 女性内生殖器 | |

| 女性外生殖器 | |

| | |

|---|---|

| | |

| 其他 | |

| 其他 | |

| 先天畸形 | |

| |

| 不孕症 | |

| 生育药物 | |

| 体外人工受精 (IVF) | |

| 其他技术 | |

| 捐献 | |

| 伦理 | |

| 关于“不孕症”的留言： | 订阅讨论RSS |
```

**nhathuoclongchau.com.vn**, doc 204354: 1.991 → 1.936 token

Trước:
```text
ng bạn trong việc chăm sóc và bảo vệ sức khỏe của bạn và gia đình. Hãy theo dõi trang web của Nhà Thuốc Long Châu để cập nhật thêm những tin tức y khoa bổ ích nhé!

Ánh Vũ

Nguồn tham khảo: Tổng hợp

Thông tin và sản phẩm gợi ý trong bài viết chỉ mang tính chất tham khảo, vui lòng liên hệ với Bác sĩ, Dược sĩ hoặc chuyên viên y tế để được tư vấn cụ thể. Xem thêm

Dược sĩ Đại họcNguyễn Thị Thảo Nguyên

Đã kiểm duyệt nội dung
```
Sau (đuôi):
```text
cùng bạn trong việc chăm sóc và bảo vệ sức khỏe của bạn và gia đình. Hãy theo dõi trang web của Nhà Thuốc Long Châu để cập nhật thêm những tin tức y khoa bổ ích nhé!

Ánh Vũ

Nguồn tham khảo: Tổng hợp
```

**nhathuoclongchau.com.vn**, doc 204356: 1.967 → 1.912 token

Trước:
```text
tháng uống 5 lần thuốc tránh thai khẩn cấp sẽ gây ra điều gì và các biện pháp tránh thai khác bạn có thể tham khảo. Hy vọng bài viết cung cấp đến bạn những thông tin hữu ích và cần thiết.

Xem thêm:

Thông tin và sản phẩm gợi ý trong bài viết chỉ mang tính chất tham khảo, vui lòng liên hệ với Bác sĩ, Dược sĩ hoặc chuyên viên y tế để được tư vấn cụ thể. Xem thêm

Dược sĩ Đại họcNgô Kim Thúy

Đã kiểm duyệt nội dung
```
Sau (đuôi):
```text
1 tháng uống 5 lần thuốc tránh thai khẩn cấp sẽ gây ra điều gì và các biện pháp tránh thai khác bạn có thể tham khảo. Hy vọng bài viết cung cấp đến bạn những thông tin hữu ích và cần thiết.

Xem thêm:
```

**youmed.vn**, doc 759253: 1.642 → 1.497 token

Trước:
```text
ời đồng thời bạn cũng không được quên bổ sung những chất dinh dưỡng cần và sắp xếp giấc ngủ phù hợp để tăng trưởng chiều cao một cách tối ưu nhất nhé.

Xem thêm: Cách tăng chiều cao tại nhà hiệu quả

Nguồn tham khảo / Source

Trang tin y tế YouMed chỉ sử dụng các nguồn tham khảo có độ uy tín cao, các tổ chức y dược, học thuật chính thống, tài liệu từ các cơ quan chính phủ để hỗ trợ các thông tin trong bài viết của chúng tôi. Tìm hiểu về Quy trình biên tập để hiểu rõ hơn cách chúng tôi đảm bảo nội dung luôn chính xác, minh bạch và tin cậy.

-

10 cách tăng chiều cao tự nhiên hiệu quả nhấthttps:
```
Sau (đuôi):
```text
gười đồng thời bạn cũng không được quên bổ sung những chất dinh dưỡng cần và sắp xếp giấc ngủ phù hợp để tăng trưởng chiều cao một cách tối ưu nhất nhé.

Xem thêm: Cách tăng chiều cao tại nhà hiệu quả
```

**youmed.vn**, doc 759263: 1.688 → 1.603 token

Trước:
```text
ông ngờ đến. Thường gặp nhất là tác động từ stress và những bệnh lý làm thay đổi đột ngột cơ thể của bạn. Bạn nên tìm đến sự hỗ trợ từ bác sĩ chuyên khoa để cơ thể sớm hoạt động bình thường lại nhé.

Nguồn tham khảo / Source

Trang tin y tế YouMed chỉ sử dụng các nguồn tham khảo có độ uy tín cao, các tổ chức y dược, học thuật chính thống, tài liệu từ các cơ quan chính phủ để hỗ trợ các thông tin trong bài viết của chúng tôi. Tìm hiểu về Quy trình biên tập để hiểu rõ hơn cách chúng tôi đảm bảo nội dung luôn chính xác, minh bạch và tin cậy.
```
Sau (đuôi):
```text
không ngờ đến. Thường gặp nhất là tác động từ stress và những bệnh lý làm thay đổi đột ngột cơ thể của bạn. Bạn nên tìm đến sự hỗ trợ từ bác sĩ chuyên khoa để cơ thể sớm hoạt động bình thường lại nhé.
```
