# S12 – mẫu vi, top 10 host (20 doc/host)

Phân tầng: 6 ngẫu nhiên (rand), 5 ngắn <500 ký tự (short), 4 title đáng ngờ (title: <12 ký tự, bắt đầu http, trùng ≥10 doc trong host, hoặc chứa 404/not found), 5 body giống menu (nav: >60% dòng body <30 ký tự, ≥8 dòng). Thiếu tầng thì bù bằng rand.
Nguồn: `data/chunks/docs.parquet` (623.805 doc đã index; doc_text = title + body). Seed 42. `⏎` = xuống dòng.

## Tỉ lệ cờ thô theo host (chưa phải rule cuối)

| host | doc | short<500 | title_sus | nav |
|---|---:|---:|---:|---:|
| suckhoecongdongonline.vn | 154,230 | 1.48% | 0.06% | 0.47% |
| suckhoedoisong.vn | 82,618 | 4.92% | 0.24% | 0.08% |
| thanhnien.vn | 71,607 | 5.04% | 0.72% | 0.03% |
| phunusuckhoe.giadinhonline.vn | 27,575 | 0.11% | 0.09% | 0.71% |
| laodong.vn | 27,184 | 5.64% | 0.00% | 0.47% |
| vinmec.com | 26,124 | 0.19% | 0.03% | 0.11% |
| medlatec.vn | 24,709 | 0.03% | 0.00% | 0.02% |
| giadinhonline.vn | 16,726 | 0.33% | 0.02% | 0.07% |
| baohaiphong.vn | 16,246 | 3.68% | 0.09% | 0.03% |
| baonghean.vn | 15,728 | 1.63% | 0.04% | 0.22% |

## suckhoecongdongonline.vn (154,230 doc)

- **[short]** doc_id `473865` · 459 ký tự, 5 dòng body · title: `XSDN 26/1 - Kết quả xổ số Đồng Nai hôm nay thứ 4 ngày 26/1/2022`  
  https://suckhoecongdongonline.vn/xsdn-26-1-ket-qua-xo-so-dong-nai-hom-nay-thu-4-ngay-26-1-2022  
  đầu: XSDN 26/1 - Kết quả xổ số Đồng Nai hôm nay thứ 4 ngày 26/1/2022⏎⏎XSDN 26/1 - Kết quả Xổ Số Đồng Nai hôm nay thứ 4 ngày 26/1/2022. Mời quý vị xem trực tiếp và dò kết quả của tất cả các giải thưởng xổ số kiến thiết Đồng Nai dưới đây.⏎⏎Chúc các bạn may mắn.⏎⏎Xem trực tiếp XSDN 26/1 - Kết quả Xổ Số Đồng Nai hôm nay thứ 4 ngày 26/1/2022 tại đây.⏎⏎Quý vị có thể xem lại kết quả xổ số   
  cuối: …ác bạn may mắn.⏎⏎Xem trực tiếp XSDN 26/1 - Kết quả Xổ Số Đồng Nai hôm nay thứ 4 ngày 26/1/2022 tại đây.⏎⏎Quý vị có thể xem lại kết quả xổ số Đồng Nai lần trước thứ 4 ngày 19/1/2022 - XSDN⏎⏎Một vài thống kê xổ số Đồng Nai

- **[short]** doc_id `411710` · 292 ký tự, 1 dòng body · title: `Nguy cơ hỏa hoạn do đốt vàng mã rằm tháng 7`  
  https://suckhoecongdongonline.vn/nguy-co-hoa-hoan-do-dot-vang-ma-ram-thang-7  
  đầu: Nguy cơ hỏa hoạn do đốt vàng mã rằm tháng 7⏎⏎Đi các tuyến phố Hàng Mã, Hàng Lược, Hàng Đường, Hàng Bè, Mã Mây, Hàng Cá, Hàng Ngang… đâu đâu cũng có mùi đốt vàng mã và kéo theo đó là những làn khói như sương mù, làm ô nhiễm không khí, gây khó chịu cho người đi đường và cả hàng xóm xung quanh.  
  cuối: …àng Lược, Hàng Đường, Hàng Bè, Mã Mây, Hàng Cá, Hàng Ngang… đâu đâu cũng có mùi đốt vàng mã và kéo theo đó là những làn khói như sương mù, làm ô nhiễm không khí, gây khó chịu cho người đi đường và cả hàng xóm xung quanh.

- **[short]** doc_id `473204` · 404 ký tự, 3 dòng body · title: `XSBL 1/12 - Kết quả xổ số Bạc Liêu hôm nay thứ 3 ngày 1/12/2020`  
  https://suckhoecongdongonline.vn/xsbl-1-12-ket-qua-xo-so-bac-lieu-hom-nay-thu-3-ngay-1-12-2020  
  đầu: XSBL 1/12 - Kết quả xổ số Bạc Liêu hôm nay thứ 3 ngày 1/12/2020⏎⏎XSBL 1/12 - Kết quả Xổ Số Bạc Liêu hôm nay thứ 3 ngày 1/12/2020. Mời quý vị xem trực tiếp và dò kết quả của tất cả các giải thưởng xổ số kiến thiết Bạc Liêu dưới đây.⏎⏎Xem trực tiếp XSBL 1/12 - Kết quả Xổ Số Bạc Liêu hôm nay thứ 3 ngày 1/12/2020 tại đây.⏎⏎Quý vị có thể xem lại kết quả xổ số Bạc Liêu lần trước thứ   
  cuối: …giải thưởng xổ số kiến thiết Bạc Liêu dưới đây.⏎⏎Xem trực tiếp XSBL 1/12 - Kết quả Xổ Số Bạc Liêu hôm nay thứ 3 ngày 1/12/2020 tại đây.⏎⏎Quý vị có thể xem lại kết quả xổ số Bạc Liêu lần trước thứ 3 ngày 24/11/2020 - XSBL

- **[short]** doc_id `343572` · 404 ký tự, 1 dòng body · title: `Cá ngắc ngoải giãy giụa khi bị đóng băng giữa trời -30°C`  
  https://suckhoecongdongonline.vn/ca-ngac-ngoai-giay-giua-khi-bi-dong-bang-giua-troi-30c  
  đầu: Cá ngắc ngoải giãy giụa khi bị đóng băng giữa trời -30°C⏎⏎Thường Châu, Trung Quốc tiếp tục chuyển lạnh, có thể có bão tuyết Năm nay, Trung Quốc phải hứng chịu những đợt lạnh kỉ lục. Nhiều nơi có nhiệt độ xuống tới âm hàng chục độ. Theo truyền thông Trung Quốc, tại thành phố Thường Châu, Giang Tô, Trung Quốc – nơi sẽ diễn ra trận chung kết giữa U23 Việt Nam và U23 Uzbekistan, tr  
  cuối: …iều nơi có nhiệt độ xuống tới âm hàng chục độ. Theo truyền thông Trung Quốc, tại thành phố Thường Châu, Giang Tô, Trung Quốc – nơi sẽ diễn ra trận chung kết giữa U23 Việt Nam và U23 Uzbekistan, trời tiếp tục chuyển lạnh.

- **[short]** doc_id `410742` · 306 ký tự, 1 dòng body · title: `Người dân tò mò về chiếc máy lấy nước từ không khí đầu tiên tại Hà Nội`  
  https://suckhoecongdongonline.vn/nguoi-dan-to-mo-ve-chiec-may-lay-nuoc-tu-khong-khi-dau-tien-tai-ha-noi  
  đầu: Người dân tò mò về chiếc máy lấy nước từ không khí đầu tiên tại Hà Nội⏎⏎Người dân Hà Nội thích thú với hệ thống máy lọc nước hồ chạy bằng... sức người 6 bộ thiết bị xe đạp lọc nước được lắp đặt ven hồ Hoàng Cầu, những sản phẩm này không chỉ giúp rèn luyện sức khỏe mà còn cải thiện chất lượng nước tại đây.  
  cuối: …ội thích thú với hệ thống máy lọc nước hồ chạy bằng... sức người 6 bộ thiết bị xe đạp lọc nước được lắp đặt ven hồ Hoàng Cầu, những sản phẩm này không chỉ giúp rèn luyện sức khỏe mà còn cải thiện chất lượng nước tại đây.

- **[title]** doc_id `363037` · 1051 ký tự, 8 dòng body · title: `Đề thi và đáp án môn Tiếng Anh THPT quốc gia 2019 mã đề 404`  
  https://suckhoecongdongonline.vn/de-thi-va-dap-an-mon-tieng-anh-thpt-quoc-gia-2019-ma-de-404  
  đầu: Đề thi và đáp án môn Tiếng Anh THPT quốc gia 2019 mã đề 404⏎⏎Đề thi và đáp án môn Tiếng Anh THPT quốc gia 2019 mã đề 404⏎⏎Chiều 26/6, thí sinh tham dự kỳ thi THPT quốc gia năm 2019 đã làm bài thi môn Tiếng Anh với thời gian là 60 phút. Bộ Giáo Dục và Đào Tạo ra đề thi theo nội dung chương trình học, bám sát cấu trúc, mức độ đề thi của kỳ thi THPT quốc gia 2019.⏎⏎Dưới đây là đ  
  cuối: …HPT quốc gia 2019 mã đề 406 được chúng tôi giới thiệu dưới đây.⏎⏎Đề thi và đáp án môn Tiếng Anh THPT quốc gia 2019 mã đề 402 Đề thi và đáp án môn Tiếng Anh THPT quốc gia 2019 mã đề 402 được chúng tôi giới thiệu dưới đây.

- **[title]** doc_id `461111` · 3051 ký tự, 13 dòng body · title: `Uber là gì?`  
  https://suckhoecongdongonline.vn/uber-la-gi  
  đầu: Uber là gì?⏎⏎Uber là tên gọi của một phần mềm đặt xe taxi thông qua điện thoại thông minh trên nền tảng google maps có sử dụng dịch vụ định vị toàn cầu. Uber cũng là tên gọi của một công ty kinh doanh dịch vụ vận tải với chức năng là cầu nối giữa lái xe và người đi xe.⏎⏎Những điều cần biết về Uber⏎⏎Uber là gì, Uber là một công ty kinh doanh mạng lưới giao thông vận tải và công   
  cuối: … uber khắc phục dần.⏎⏎GrabTaxi là gì? Grab là ứng dụng đặt đặt xe trên điện thoại smartphone, giúp kết nối giữa hành khách và tài xế có đăng ký sử dụng ứng dụng. Với Grabtaxi bạn sẽ được di chuyển với mức chi phí hợp lý.

- **[title]** doc_id `464245` · 1417 ký tự, 8 dòng body · title: `Việt Nam: Thêm 1 ca tử vong do COVID-19`  
  https://suckhoecongdongonline.vn/viet-nam-them-1-ca-tu-vong-do-covid-19-811755360957  
  đầu: Việt Nam: Thêm 1 ca tử vong do COVID-19⏎⏎- Từ 17h30 ngày 10/6 đến 17h30 ngày 11/6 ghi nhận 1 ca tử vong tại: Tây Ninh (1).⏎⏎- Trung bình số tử vong ghi nhận trong 07 ngày qua: 0 ca.⏎⏎- Tổng số ca tử vong do COVID-19 tại Việt Nam tính đến nay là 43.083 ca, chiếm tỷ lệ 0,4% so với tổng số ca nhiễm.⏎⏎(Ảnh minh hoạ)⏎⏎- Tổng số ca tử vong xếp thứ 24/227 vùng lãnh thổ, số ca tử vong   
  cuối: …xuyên trao đổi với Tổ chức Y tế thế giới (WHO) để cập nhật thông tin kịp thời, chính xác về các biến chủng. Xây dựng phương án bảo đảm công tác y tế ứng phó với các tình huống dịch COVID-19 giai đoạn 2022-2023.⏎⏎Bảo Ngọc

- **[title]** doc_id `463972` · 625 ký tự, 6 dòng body · title: `Việt Nam ghi nhận 1 ca tử vong do COVID-19`  
  https://suckhoecongdongonline.vn/viet-nam-ghi-nhan-1-ca-tu-vong-do-covid-19-641755329399  
  đầu: Việt Nam ghi nhận 1 ca tử vong do COVID-19⏎⏎- Từ 17h30 ngày 28/8 đến 17h30 ngày 29/8, Việt Nam ghi nhận 1 ca tử vong tại: Hải Dương.⏎⏎(Ảnh minh hoạ)⏎⏎- Trung bình số tử vong ghi nhận trong 07 ngày qua: 1 ca.⏎⏎- Tổng số ca tử vong do COVID-19 tại Việt Nam tính đến nay là 43.113 ca, chiếm tỷ lệ 0,4% so với tổng số ca nhiễm.⏎⏎- Tổng số ca tử vong xếp thứ 24/227 vùng lãnh thổ, số c  
  cuối: …n xếp thứ 134/227 quốc gia, vùng lãnh thổ trên thế giới. So với châu Á, tổng số ca tử vong xếp thứ 6/49(xếp thứ 3 ASEAN), tử vong trên 1 triệu dân xếp thứ 27/49 quốc gia, vùng lãnh thổ châu Á (xếp thứ 5 ASEAN).⏎⏎Bảo Ngọc

- **[nav]** doc_id `473406` · 1679 ký tự, 30 dòng body · title: `XSBTH 20/12 - Kết quả Xổ Số Bình Thuận hôm nay thứ 5 ngày 20/12/2018`  
  https://suckhoecongdongonline.vn/xsbth-2012-ket-qua-xo-so-binh-thuan-hom-nay-thu-5-ngay-20122018  
  đầu: XSBTH 20/12 - Kết quả Xổ Số Bình Thuận hôm nay thứ 5 ngày 20/12/2018⏎⏎XSBTH 20/12 - Kết quả Xổ Số Bình Thuận hôm nay thứ 5 ngày 20/12/2018 Mời quý vị đón xem trực tiếp và dò kết quả của tất cả các giải thưởng xổ số kiến thiết Bình Thuận dưới đây.⏎⏎Xem trực tiếp XSBTH 20/12 - Kết quả Xổ Số Bình Thuận hôm nay thứ 5 ngày 20/12/2018 tại đây⏎⏎Quý vị có thể xem lại kết quả xổ số Bình  
  cuối: … xe xuống sông Bức ảnh cuối cùng của nữ ca sĩ đồng tính này là chụp cạnh dòng sông nơi cô ấy tử vong.⏎⏎12 bộ phim LGBT hay nhất năm 2018 (Phần 2) Dưới đây là danh sách những bộ phim LGBT đã làm mưa làm gió suốt năm 2018.

- **[nav]** doc_id `473544` · 1774 ký tự, 30 dòng body · title: `XSCM 11/2 - Kết quả xổ số Cà Mau hôm nay thứ 2 ngày 11/2/2019`  
  https://suckhoecongdongonline.vn/xscm-112-ket-qua-xo-so-ca-mau-hom-nay-thu-2-ngay-1122019  
  đầu: XSCM 11/2 - Kết quả xổ số Cà Mau hôm nay thứ 2 ngày 11/2/2019⏎⏎XSCM 11/2 - Kết quả Xổ Số Cà Mau hôm nay thứ 2 ngày 11/2/2019 Mời quý vị xem trực tiếp và dò kết quả của tất cả các giải thưởng xổ số kiến thiết XSCM dưới đây.⏎⏎Xem trực tiếp XSCM 11/2 - Kết quả Xổ Số Cà Mau hôm nay thứ 2 ngày 11/2/2019 tại đây.⏎⏎Quý vị có thể xem lại kết quả xổ số XSCM lần trước thứ 2 ngày 4/2/2019  
  cuối: …g quyết rời xa mẹ đến Canada khiến cộng đồng mạng rơi nước mắt Nhiều người bày tỏ sự xúc động về suy nghĩ trưởng thành của Ngô Trác Lâm. Cô thực sự là bờ vai vững chắc cho mẹ là Ngô Ỷ Lợi sau khi bà bị Thành Long bỏ rơi.

- **[nav]** doc_id `476139` · 1862 ký tự, 30 dòng body · title: `XSST 27/2 - Kết quả xổ số Sóc Trăng hôm nay thứ 4 ngày 27/2/2019`  
  https://suckhoecongdongonline.vn/xsst-272-ket-qua-xo-so-soc-trang-hom-nay-thu-4-ngay-2722019  
  đầu: XSST 27/2 - Kết quả xổ số Sóc Trăng hôm nay thứ 4 ngày 27/2/2019⏎⏎XSST 27/2 - Kết quả xổ số Sóc Trăng hôm nay ngày 27/2/2019 được phát trực tiếp tại đây nhanh và chính xác, cùng giờ với giờ quay thưởng tại Sóc Trăng.⏎⏎Mời quý vị theo dõi trực tiếp Kết quả xổ số Sóc Trăng hôm nay được quay thưởng vào lúc 16h15 ngày 27/2/2019.⏎⏎Xem lại kết quả XSST 20/2 - Kết quả Xổ Số Sóc Trăng   
  cuối: …ẩn bị kết hôn.⏎⏎Đơn xin hiến tạng của chàng trai Hà Nội viết chỉ hai ngày trước khi mất Hai ngày sau khi chàng trai 30 tuổi qua đời do ung thư dạ dày, lá đơn hiến tạng mới đến được Trung tâm Điều phối ghép tạng quốc gia.

- **[nav]** doc_id `473565` · 1903 ký tự, 31 dòng body · title: `XSCM 19/11 - Kết quả xổ số Cà Mau hôm nay thứ 2 ngày 19/11/2018`  
  https://suckhoecongdongonline.vn/xscm-1911-ket-qua-xo-so-ca-mau-hom-nay-thu-2-ngay-19112018  
  đầu: XSCM 19/11 - Kết quả xổ số Cà Mau hôm nay thứ 2 ngày 19/11/2018⏎⏎Mời quý vị xem trực tiếp XSCM 19/11 - Kết quả Xổ Số Cà Mau hôm nay thứ 2 ngày 19/11/2018 và dò kết quả của tất cả các giải thưởng xổ số kiến thiết XSCM dưới đây.⏎⏎Kết quả Xổ Số Cà Mau hôm nay thứ 2 ngày 19/11/2018 được quay thưởng tại Cà Mau và có kết quả trực tiếp tại đây, mời quý vị độc giả đón xem.⏎⏎Xem trực ti  
  cuối: …hoàn toàn không có viền.⏎⏎Ngấm đòn ông Trump, doanh số hãng xe TQ rớt thê thảm Đang là ngôi sao sáng, hãng xe nội địa Geely nói riêng và ngành công nghiệp ôtô Trung Quốc trở nên ảm đạm bởi cuộc chiến thương mại Mỹ-Trung.

- **[nav]** doc_id `362006` · 4648 ký tự, 151 dòng body · title: `Dấu hiệu của việc thừa hay thiếu vitamin`  
  https://suckhoecongdongonline.vn/dau-hieu-cua-viec-thua-hay-thieu-vitamin  
  đầu: Dấu hiệu của việc thừa hay thiếu vitamin⏎⏎Thiếu vitamin C⏎⏎- Chảy máu chân răng, chảy máu cam⏎⏎- Xuất huyết dưới da, da hay bị bầm tím⏎⏎- Da xuất hiện nhăn, đề kháng suy giảm.⏎⏎- Hay bị dị ứng, eăng dễ rụng⏎⏎- Viêm lợi, hay bị cúm.⏎⏎Thừa vitamin C⏎⏎- Rối loan tiêu hoá, loét dạ dày tá tràng, viêm bàng quang, tiêu chảy.⏎⏎- Tạo sỏi thận⏎⏎- Gây bệnh gout⏎⏎- Cản trở hấp thu vitamin   
  cuối: …ông ngon giấc ở trẻ em.⏎⏎- Răng dễ vàng ố và gẫy răng⏎⏎Thừa vitamin D⏎⏎- Tăng calci huyết.⏎⏎- Đau đầu, chóng mặt, chán ăn, khô miệng.⏎⏎- Chuột rút, táo bón, buồn nôn, đau cơ.⏎⏎- Mạch máu bị vôi hoá.⏎⏎Theo Y học thực hành

- **[rand]** doc_id `392647` · 2878 ký tự, 10 dòng body · title: `Khuyến cáo mới giúp ngăn ngừa đột quỵ tái phát`  
  https://suckhoecongdongonline.vn/khuyen-cao-moi-giup-ngan-ngua-dot-quy-tai-phat  
  đầu: Khuyến cáo mới giúp ngăn ngừa đột quỵ tái phát⏎⏎Theo Tiến sĩ khoa tim mạch Andrew Freeman: AHA và ASA khuyến nghị bước đầu tiên để giảm nguy cơ đột quỵ lần thứ hai là nhanh chóng xác định nguyên nhân dẫn đến cơn đột quỵ đầu tiên và hoàn thành các xét nghiệm chẩn đoán trong vòng 48 giờ kể từ khi các triệu chứng đột quỵ bắt đầu, tiến hành chụp CT hoặc MRI não để đánh giá mức độ t  
  cuối: …lại di chứng nặng nề hơn cơn đột quỵ lần đầu. Ðây được xem là biện pháp vô cùng hiệu quả để giảm thiểu chi phí và tác dụng phụ của tai biến mạch máu não, đồng thời cải thiện sức khỏe tổng thể về lâu dài.⏎⏎Theo Healthline

- **[rand]** doc_id `391554` · 3650 ký tự, 32 dòng body · title: `Khí hư báo hiệu có thai liệu có đúng?`  
  https://suckhoecongdongonline.vn/khi-hu-bao-hieu-co-thai-lieu-co-dung  
  đầu: Khí hư báo hiệu có thai liệu có đúng?⏎⏎Từ xưa ông bà ta thường truyền miệng những mẹo nhận biết có thai thông qua nhiều đặc điểm trên cơ thể như lông mày dựng, đầu ngực thâm, da mặt tái nhợt… Bên cạnh đó những dấu hiệu thay đổi bên trong cơ thể cũng giúp các chị em có thể nhận biết khả năng có thai dễ dàng hơn. Có ý kiến cho rằng khí hư báo hiệu có thai, song dấu hiệu này liệu   
  cuối: …ỏe cơ thể ổn định.⏎⏎Khí hư báo hiệu có thai nhưng cũng cần lưu ý đến các dấu hiệu khác cũng như áp dụng phương pháp thử thai như là siêu âm, xét nghiệm máu, xét nghiệm nước tiểu để đảm bảo kết quả mang thai là chính xác.

- **[rand]** doc_id `407315` · 2101 ký tự, 9 dòng body · title: `Ngày 28/10, Hà Nội ghi nhận 33 ca mắc Covid-19 mới`  
  https://suckhoecongdongonline.vn/ngay-28-10-ha-noi-ghi-nhan-33-ca-mac-covid-19-moi  
  đầu: Ngày 28/10, Hà Nội ghi nhận 33 ca mắc Covid-19 mới⏎⏎Phân bố theo quận, huyện Quốc Oai (12), Mê Linh (09), Hoàng Mai (03), Đống Đa (02), Nam Từ Liêm (02), Long Biên (01), Đông Anh (01), Gia Lâm (01), Bắc Từ Liêm (01), Hà Đông (01).⏎⏎Phân bố theo các chùm ca bệnh, ổ dịch: Chùm liên quan ổ dịch tại Sài Sơn, thị trấn Quốc Oai có 13 ca bệnh ở Hà Đông (01), Quốc Oai (12).⏎⏎Chùm liên   
  cuối: …nghiệm, kết quả có 46 mẫu dương tính và 856 mẫu âm tính; 5129 người liên quan khác, người trong khu vực phong tỏa cũng đã được lấy mẫu xét nghiệm, đã có 3740 mẫu âm tính và 03 mẫu dương tính, số còn lại đang chờ kết quả.

- **[rand]** doc_id `450160` · 1926 ký tự, 24 dòng body · title: `Tối 31/10: Ghi nhận 9 ca mắc COVID-19 mới`  
  https://suckhoecongdongonline.vn/toi-31-10-ghi-nhan-9-ca-mac-covid-19-moi  
  đầu: Tối 31/10: Ghi nhận 9 ca mắc COVID-19 mới⏎⏎TÌNH HÌNH DỊCH COVID-19 TẠI VIỆT NAM:⏎⏎Kể từ đầu dịch đến nay Việt Nam có 11.624.114 ca nhiễm, đứng thứ 13/231 quốc gia và vùng lãnh thổ, trong khi với tỷ lệ số ca nhiễm/1 triệu dân, Việt Nam đứng thứ 120/231 quốc gia và vùng lãnh thổ (bình quân cứ 1 triệu người có 117.470 ca nhiễm).⏎⏎TÌNH HÌNH ĐIỀU TRỊ NGÀY 30/10⏎⏎(Số liệu do Sở Y tế   
  cuối: ….965.655 liều: Mũi 1 là 9.130.889 liều; Mũi 2 là 9.021.382 liều; Mũi nhắc lại lần 1 là 5.813.384 liều.⏎⏎+ Số liều tiêm cho trẻ từ 5-11 tuổi là 18.725.280 liều: Mũi 1 là 10.236.628 liều; Mũi 2 là 8.488.652 liều.⏎⏎Bảo Ngọc

- **[rand]** doc_id `422545` · 1300 ký tự, 6 dòng body · title: `Phillip Nguyễn đăng tải loạt ảnh Linh Rin xinh như nữ thần cũng lời nhắn nhủ ngọt đến lịm `  
  https://suckhoecongdongonline.vn/phillip-nguyen-dang-tai-loat-anh-linh-rin-xinh-nhu-nu-than-cung-loi-nhan-nhu-ngot-den-lim-tim  
  đầu: Phillip Nguyễn đăng tải loạt ảnh Linh Rin xinh như nữ thần cũng lời nhắn nhủ ngọt đến lịm tim⏎⏎Phillip Nguyễn đăng tải loạt ảnh Linh Rin xinh như nữ thần⏎⏎Phillip Nguyễn và Linh Rin ngày càng trở nên ngọt ngào, nồng nàn. Họ dành cho nhau những cử chỉ ân cần nhất, như nắm tay không rời khi tham gia một sự kiện, nhìn nhau âu yếm và luôn ở bên cạnh ủng hộ đối phương dù họ làm gì.⏎  
  cuối: … viên nhau khi khó khăn và âm thầm cổ vũ người kia.⏎⏎Dù Phillip Nguyễn và Linh Rin cũng vẫn có những sóng gió nhất chắc chắn những gì cặp đôi đã và đang làm cho nhau khiến netizen tin rằng họ sẽ có một cái kết hạnh phúc.

- **[rand]** doc_id `385217` · 2851 ký tự, 9 dòng body · title: `Phạm Hương đẹp tựa 'nữ thần' trong khoảnh khắc trao lại vương miện cho tân Hoa hậu Hoàn vũ`  
  https://suckhoecongdongonline.vn/hoa-hau-hoan-vu-viet-nam-dan-he-lo-ung-vien-tro-thanh-nguoi-ke-vi-tuong-lai  
  đầu: Phạm Hương đẹp tựa 'nữ thần' trong khoảnh khắc trao lại vương miện cho tân Hoa hậu Hoàn vũ⏎⏎Cuộc thi Hoa hậu Hoàn vũ Việt Nam 2017 chính thức khép lại và danh hiệu cao nhất đã thuộc về thí sinh H’Hen Niê, số báo danh 424. Vương miện đã tìm được chủ nhân sở hữu sau hành trình dài tìm kiếm người kế vị. Hai danh hiệu Á hậu lần lượt gọi tên thí sinh Hoàng Thị Thùy và Mâu Thị Thanh   
  cuối: …àn sao Việt hội tụ trên thảm đỏ sự kiện thời trang Elle Fashion Show với các phong cách đa dạng. Nếu như Phạm Hương, Minh Hằng xuất hiện cá tính và phong cách thì diva Hồng Nhung, ca sĩ Đoan Trang lại mắc lỗi trang phục.

## suckhoedoisong.vn (82,618 doc)

- **[short]** doc_id `490948` · 292 ký tự, 1 dòng body · title: `Bất ngờ thói quen nhiều người Việt mắc phải là nguyên nhân gây ung thư tuyến giáp`  
  https://suckhoedoisong.vn/bat-ngo-thoi-quen-nhieu-nguoi-viet-mac-phai-la-nguyen-nhan-gay-ung-thu-tuyen-giap-169240930153818614.htm  
  đầu: Bất ngờ thói quen nhiều người Việt mắc phải là nguyên nhân gây ung thư tuyến giáp⏎⏎SKĐS - Ung thư tuyến giáp là căn bệnh hay gặp ở nữ giới. Đặc biệt, những thói quen phổ biến mà nhiều người Việt đang mắc phải lại là một trong số những lý do gây bệnh và khiến tình trạng bệnh trở nên nặng hơn.  
  cuối: …uyến giáp⏎⏎SKĐS - Ung thư tuyến giáp là căn bệnh hay gặp ở nữ giới. Đặc biệt, những thói quen phổ biến mà nhiều người Việt đang mắc phải lại là một trong số những lý do gây bệnh và khiến tình trạng bệnh trở nên nặng hơn.

- **[short]** doc_id `550673` · 354 ký tự, 1 dòng body · title: `Thủ tướng Chính phủ ra công điện chỉ đạo xử lý, tập trung cứu chữa các bệnh nhân bị ngộ độ`  
  https://suckhoedoisong.vn/thu-tuong-chinh-phu-ra-cong-dien-chi-dao-xu-ly-tap-trung-cuu-chua-cac-benh-nhan-bi-ngo-doc-thuc-pham-tai-tp-long-khanh-169240504123051145.htm  
  đầu: Thủ tướng Chính phủ ra công điện chỉ đạo xử lý, tập trung cứu chữa các bệnh nhân bị ngộ độc thực phẩm tại TP Long Khánh⏎⏎SKĐS - Trước diễn biến phức tạp của vấn đề ngộ độc thực phẩm thời gian gần đây, ngày 3/5, Thủ tướng Chính phủ đã có công điện số 44/CĐ-TTg yêu cầu các bộ, ngành, địa phương triển khai các biện pháp ngăn ngừa, xử lý ngộ độc thực phẩm.  
  cuối: …diễn biến phức tạp của vấn đề ngộ độc thực phẩm thời gian gần đây, ngày 3/5, Thủ tướng Chính phủ đã có công điện số 44/CĐ-TTg yêu cầu các bộ, ngành, địa phương triển khai các biện pháp ngăn ngừa, xử lý ngộ độc thực phẩm.

- **[short]** doc_id `547699` · 322 ký tự, 1 dòng body · title: `Tảo mộ xuân Bính Ngọ 2026 vào dịp nào là tốt, những điều kiêng kỵ cần tránh khi đi tảo mộ`  
  https://suckhoedoisong.vn/tao-mo-xuan-binh-ngo-2026-vao-dip-nao-la-tot-nhung-dieu-kieng-ky-can-tranh-khi-di-tao-mo-169260305111817264.htm  
  đầu: Tảo mộ xuân Bính Ngọ 2026 vào dịp nào là tốt, những điều kiêng kỵ cần tránh khi đi tảo mộ⏎⏎SKĐS - Mỗi dịp cuối năm, đầu xuân hay lễ Thanh Minh, con cháu lại cùng nhau đi tảo mộ để dọn dẹp phần mộ tổ tiên, dâng hương tưởng niệm và bày tỏ lòng thành kính. Vậy xuân Bính Ngọ 2026 nên đi tảo mộ vào ngày nào, giờ nào đẹp nhất?  
  cuối: …dịp cuối năm, đầu xuân hay lễ Thanh Minh, con cháu lại cùng nhau đi tảo mộ để dọn dẹp phần mộ tổ tiên, dâng hương tưởng niệm và bày tỏ lòng thành kính. Vậy xuân Bính Ngọ 2026 nên đi tảo mộ vào ngày nào, giờ nào đẹp nhất?

- **[short]** doc_id `502396` · 292 ký tự, 1 dòng body · title: `Cắt bỏ khối u to gấp 3 lần quả thận`  
  https://suckhoedoisong.vn/cat-bo-than-trai-vi-khoi-u-to-gap-3-lan-qua-than-169240126131946361.htm  
  đầu: Cắt bỏ khối u to gấp 3 lần quả thận⏎⏎SKĐS - Theo người nhà bệnh nhân D.T.D (87 tuổi, trú tại xã Đồng Trạch, huyện Bố Trạch), thời gian gần đây, cụ D. sờ thấy có khối cứng ở vùng mạng mỡ phía bên trái nên được đưa vào Bệnh viện Hữu nghị Việt Nam - Cuba Đồng Hới (tỉnh Quảng Bình) để thăm khám.  
  cuối: ….D (87 tuổi, trú tại xã Đồng Trạch, huyện Bố Trạch), thời gian gần đây, cụ D. sờ thấy có khối cứng ở vùng mạng mỡ phía bên trái nên được đưa vào Bệnh viện Hữu nghị Việt Nam - Cuba Đồng Hới (tỉnh Quảng Bình) để thăm khám.

- **[short]** doc_id `532377` · 252 ký tự, 1 dòng body · title: `Ngồi nhiều có nguy cơ mắc trĩ đúng hay sai?`  
  https://suckhoedoisong.vn/ngoi-nhieu-co-nguy-co-mac-tri-dung-hay-sai-169221014144043158.htm  
  đầu: Ngồi nhiều có nguy cơ mắc trĩ đúng hay sai?⏎⏎Trĩ là một bệnh rất phổ biến, tỷ lệ mắc trong cộng đồng khá cao nhất là những người lớn tuổi. Bệnh xảy ra do nhiều nguyên nhân khác nhau trong đó chủ yếu do chế độ ăn và một số thói quen sinh hoạt không tốt.  
  cuối: …ng hay sai?⏎⏎Trĩ là một bệnh rất phổ biến, tỷ lệ mắc trong cộng đồng khá cao nhất là những người lớn tuổi. Bệnh xảy ra do nhiều nguyên nhân khác nhau trong đó chủ yếu do chế độ ăn và một số thói quen sinh hoạt không tốt.

- **[title]** doc_id `524986` · 1570 ký tự, 12 dòng body · title: `Lao xương`  
  https://suckhoedoisong.vn/lao-xuong-169106467.htm  
  đầu: Lao xương⏎⏎Là tình trạng hệ thống xương khớp nhiễm khuẩn do trực khuẩn lao có tên Mycobacterium tuberculosis gây ra. Đây là tình trạng lao ngoài phổi phổ biến, chiếm khoảng 7% tổng số các thể lao.⏎⏎Triệu chứng⏎⏎•Lao cột sống còn gọi là bệnh Pott, là tình trạng viêm đốt sống - đĩa đệm do lao hay gặp nhất.⏎⏎•Biểu hiện toàn thân: bệnh nhân sốt nhẹ về chiều, vã mồ hôi, gầy sút, da   
  cuối: …rạng.⏎⏎Lao xương khớp có thể gặp ở tất cả các lứa tuổi, tất cả các xương khớp trong cơ thể; trong đó lao cột sống chiếm tỷ lệ cao nhất, khoảng 60-70% tổng số lao xương khớp, sau đó đến lao khớp háng (10%), khớp gối (5%)…

- **[title]** doc_id `560953` · 1617 ký tự, 20 dòng body · title: `Viêm tai`  
  https://suckhoedoisong.vn/viem-tai-169105948.htm  
  đầu: Viêm tai⏎⏎Triệu chứng⏎⏎• Bé có thể sốt, khóc và một bên đầu muốn cọ vào đâu đó. Đôi khi mủ trong tai chảy ra.⏎⏎• Bé cũng thường vò tai, thậm chí là giật giật tai. Bú mớm hay nuốt nước miếng đều có cảm giác đau.⏎⏎• Ngoài ra, tai của bé cũng có thể tỏa mùi khó chịu khi ghé mũi vào.⏎⏎Nguyên nhân⏎⏎• Tình trạng nhiễm trùng này thường bắt đầu ít ngày sau khi bé bị cảm lạnh hay ngạt m  
  cuối: …i 3 tuổi có thể viêm mũi 4 - 6 lần trong một năm vì hệ miễn dịch chưa hoàn thiện.⏎⏎• Khi dùng thuốc xịt chống nghẹt mũi cho trẻ, bạn cần tìm loại dành riêng cho trẻ em và không nên nhỏ quá 3 lần/ngày để tránh sung huyết.

- **[title]** doc_id `560226` · 1356 ký tự, 16 dòng body · title: `Viêm Amidan`  
  https://suckhoedoisong.vn/viem-amidan-169105974.htm  
  đầu: Viêm Amidan⏎⏎Triệu chứng⏎⏎• Lưu ý nếu trẻ ngủ ngáy to, thở bằng mồm mãn tính, hay thức giấc trong đêm, mệt mỏi, đái dầm, học lực kém…⏎⏎• Nếu trẻ phát âm như giọng mũi hoặc khó khăn khi phát âm cần phải khám amidan ngay.⏎⏎• Amidan quá to có thể làm trẻ nuốt vướng, khó ăn, ăn uống chậm chạp hàng giờ.⏎⏎• Trẻ bị quá phát amidan thường có hơi thở hôi, ho về đêm, ho khan kéo dài.⏎⏎•   
  cuối: … đọc “A” , dùng đèn pin rọi vào sẽ thấy amidan sưng to.⏎⏎• Viêm amidan nếu không được điều trị hoặc điều trị không đúng cách có thể gây ra một số biến chứng như: viêm tai giữa, viêm xoang, áp xe họng và viêm vi cầu thận.

- **[title]** doc_id `545119` · 2049 ký tự, 13 dòng body · title: `Sơn chi tử`  
  https://suckhoedoisong.vn/son-chi-tu-16944706.htm  
  đầu: Sơn chi tử⏎⏎Sơn chi tử là vị thuốc Đông y lấy từ hạt dành dành, tên khoa học là Gardenia Jasminoides Ellis, thuộc họ cà phê. Cây thường gặp ở miền núi chỗ ẩm mát và có bóng râm như ven suối, bờ hồ lớn. Là loại cây nhỏ, thân cao khoảng từ 2-3m. Lá mọc đối, hình tròn bầu dục xanh bóng. Mùa hè hoa nở trắng có 6 cánh đều, uốn cong, mùi thơm. Lúc hoa sắp tàn biến màu vàng nhạt, vào   
  cuối: …mộc thông, hạt mã đề, cù mạch, biển súc, hoạt thạch mỗi vị 12g; đại hoàng 8g; cam thảo nướng 6g. Cho tất cả vào ấm, sắc với 700 ml nước còn 150ml, uống làm hai lần trong ngày. Mỗi liệu trình 10- 15 ngày.⏎⏎Bác sĩ Trần Huy

- **[nav]** doc_id `493366` · 1773 ký tự, 8 dòng body · title: `Bị bệnh sởi rồi, có nên chích ngừa sởi không?`  
  https://suckhoedoisong.vn/bi-benh-soi-roi-co-nen-chich-ngua-soi-khong-16978848.htm  
  đầu: Bị bệnh sởi rồi, có nên chích ngừa sởi không?⏎⏎Bé Phan Quốc Kh, bảy tháng rưỡi tuổi, nhà ở Bình Đại, Bến Tre, đang điều trị bệnh sởi có biến chứng viêm phổi. Sau một tuần điều trị, ngày 21/5/2014, bé được xuất viện. Trước khi về, bà mẹ hỏi bác sĩ mai mốt bé được 9 tháng tuổi có nên đi chích ngừa sởi hay không? Bị bệnh sởi rồi có bị mắc lại không?⏎⏎Bác sĩ giải thích cho bà mẹ là  
  cuối: …ài đến suốt đời. Còn trường hợp bé bị sởi mà gia đình không biết nếu chích ngừa sởi thêm nữa thì vẫn không có hại gì cho bé.⏎⏎BS. NGUYỄN THÀNH ÚT⏎⏎Bệnh Sởi⏎⏎Sởi và biến chứng⏎⏎(cập nhật liên tục)⏎⏎Sởi và cách phòng chống

- **[nav]** doc_id `486371` · 3134 ký tự, 73 dòng body · title: `8 động tác kéo giãn giảm đau lưng có thể thực hiện mọi lúc`  
  https://suckhoedoisong.vn/8-dong-tac-keo-gian-giam-dau-lung-co-the-thuc-hien-moi-luc-169221017160752894.htm  
  đầu: 8 động tác kéo giãn giảm đau lưng có thể thực hiện mọi lúc⏎⏎Theo BS. Jason DelCollo, chuyên gia xương khớp tại Đại học Philadelphia, Hoa Kỳ, các bài tập giãn cơ có tác dụng với tất cả các cơ lưng chính để bạn có thể hỗ trợ lưng khỏe mạnh, giảm đau lưng và ngăn ngừa các vấn đề trong tương lai.⏎⏎1. Khởi động⏎⏎Thực hiện từ 5- 10 phút hoạt động trước khi thực hiện bài tập, chẳng hạ  
  cuối: ….⏎⏎Giữ, đếm đến 10.⏎⏎Quay trở lại vị trí bắt đầu.⏎⏎Bây giờ, uốn cong thân người sang bên phải trong khi trượt tay phải xuống đùi phải và vươn cánh tay trái qua đầu, áp tai.⏎⏎Giữ, đếm đến 10.⏎⏎Lặp lại chuỗi này 9 lần nữa.

- **[nav]** doc_id `478327` · 12900 ký tự, 470 dòng body · title: `100 mâm cơm ba miền Bắc – Trung – Nam ngày Tết`  
  https://suckhoedoisong.vn/100-mam-com-ba-mien-bac-trung-nam-ngay-tet-169221231204322201.htm  
  đầu: 100 mâm cơm ba miền Bắc – Trung – Nam ngày Tết⏎⏎Ý nghĩa mâm cơm gia đình⏎⏎Mâm cơm gia đình nói chung và mâm cơm ngày Tết là một nét văn hóa truyền thống mà bao đời nay người dân Việt vẫn gìn giữ. Chẳng có gì tuyệt vời hơn trong thời khắc chuyển giao giữa năm cũ và năm mới các thành viên trong gia đình được sum họp, quây quần bên mâm cơm, cùng trò chuyện, chia sẻ mọi điều trong   
  cuối: … 99⏎⏎- Sườn heo nướng BBQ ăn kèm khoai tây chiên, bánh mỳ⏎⏎- Pizza⏎⏎- Cá hồi nướng ăn kèm sốt kem⏎⏎- Salad rau củ⏎⏎Mâm cơm 100⏎⏎- Cá hồi áp chảo⏎⏎- Bít tết⏎⏎- Pizza⏎⏎- Khoai tây chiên⏎⏎- Mỳ ý⏎⏎- Salad rau củ⏎⏎- Giăm bông

- **[nav]** doc_id `487091` · 9361 ký tự, 145 dòng body · title: `9 loại đậu và cây họ đậu tốt cho sức khỏe`  
  https://suckhoedoisong.vn/9-loai-dau-va-cay-ho-dau-tot-cho-suc-khoe-169241114233804492.htm  
  đầu: 9 loại đậu và cây họ đậu tốt cho sức khỏe⏎⏎Đậu và các loại cây họ đậu là quả hoặc hạt của một họ thực vật có tên là Fabaceae. Chúng là nguồn cung cấp chất xơ và protein chay tuyệt vời.⏎⏎Đậu và các loại đậu có một số lợi ích cho sức khỏe. Ăn đậu có thể giúp giảm cholesterol, giảm lượng đường trong máu và tăng vi khuẩn đường ruột có lợi.⏎⏎Dưới đây là 9 loại đậu và cây họ đậu lành  
  cuối: …y cơ tử vong do nhiều nguyên nhân, bao gồm bệnh tim, đột quỵ, ung thư và đái tháo đường. Tuy nhiên, những nghiên cứu này chỉ mang tính quan sát nên không thể chứng minh rằng ăn đậu phộng có thể làm giảm những rủi ro này.

- **[nav]** doc_id `483294` · 10972 ký tự, 154 dòng body · title: `5 thực phẩm hàng đầu tốt cho người tăng huyết áp`  
  https://suckhoedoisong.vn/5-thuc-pham-hang-dau-tot-cho-nguoi-tang-huyet-ap-169220128013520511.htm  
  đầu: 5 thực phẩm hàng đầu tốt cho người tăng huyết áp⏎⏎Các chuyên gia nhận định, tăng huyết áp ngày càng trẻ hóa. Cuộc sống hiện đại với những căng thẳng, áp lực tại nơi làm việc, cộng với lối sống ít vận động và chế độ ăn kiêng không phù hợp khiến cho nhiều người ở độ tuổi ngoài 30 đã mắc bệnh tăng huyết áp.⏎⏎Áp suất tâm thu (số trên cùng) cho biết áp suất trong động mạch khi tim đ  
  cuối: …n có thể giảm căng thẳng bằng cách trẻ hóa bản thân. Nghe nhạc, khiêu vũ hoặc làm vườn và vẽ tranh để giữ cho tâm trí luôn được tham gia. Điều này sẽ làm giảm mức độ hormone căng thẳng, cortisol và giảm huyết áp của bạn.

- **[rand]** doc_id `512736` · 6422 ký tự, 33 dòng body · title: `Điều trị bệnh lý tuyến giáp`  
  https://suckhoedoisong.vn/dieu-tri-benh-ly-tuyen-giap-169240322080913625.htm  
  đầu: Điều trị bệnh lý tuyến giáp⏎⏎Tuyến giáp là gì?⏎⏎Tuyến giáp là một tuyến hình bướm nằm ở cổ, tiết ra hormone tuyến giáp giúp cơ thể sử dụng năng lượng, giữ ấm và giúp cho não, tim, cơ và các cơ quan khác hoạt động bình thường.⏎⏎Nhân giáp (khối u) là những khối bất thường phát triển phổ biến trong tuyến giáp. Người ta ước tính rằng trong tỷ lệ 20 bệnh nhân thì sẽ có 1 người mọc n  
  cuối: …liên hệ: Văn phòng Đại diện Tập đoàn Y tế Parkway tại Hà Nội Tầng 5, số 110 Bà Triệu, quận Hai Bà Trưng, HN. Hotline: 0988 155 855 hoặc 084 308 3637Email: info@parkway.com.vnFB page: https://www.facebook.com/parkwayhanoi

- **[rand]** doc_id `505398` · 2567 ký tự, 21 dòng body · title: `Chữa bệnh bằng rau húng`  
  https://suckhoedoisong.vn/chua-benh-bang-rau-hung-16921915.htm  
  đầu: Chữa bệnh bằng rau húng⏎⏎Mọi người thường nghĩ rau thơm chỉ là thứ gia vị “trang điểm” thêm cho bữa ăn nhưng ít ai biết được tác dụng phòng và chữa bệnh của chúng... Xin giới thiệu một số cách dùng rau húng chanh, húng quế - loại gia vị rất quen thuộc nhưng cũng là cây thuốc thông dụng phòng trị nhiều bệnh.⏎⏎Rau húng quế⏎⏎Rau húng quế còn gọi là húng chó, húng giổi, é trắng. Tê  
  cuối: … húng chanh thêm chút muối, nhai nuốt nước dần.⏎⏎Giảm đau nhức do bị kiến độc đốt, rết, bọ cạp đốt: húng chanh 20g, muối ăn vài hạt, tất cả đem giã nhỏ hoặc nhai kỹ, nuốt nước, bã đắp vào chỗ ong đốt.⏎⏎Lương y Đình Thuấn

- **[rand]** doc_id `555280` · 3330 ký tự, 17 dòng body · title: `Trẻ bị viêm phế quản có cần dùng thuốc giảm ho và kháng sinh không?`  
  https://suckhoedoisong.vn/tre-bi-viem-phe-quan-co-can-dung-thuoc-giam-ho-va-khang-sinh-khong-169230725181114264.htm  
  đầu: Trẻ bị viêm phế quản có cần dùng thuốc giảm ho và kháng sinh không?⏎⏎‏1. Nguyên tắc chung trong điều trị viêm phế quản ở trẻ em‏⏎⏎Thông thường tình trạng viêm phế quản có thể kéo dài từ 1 đến 2 tuần, gây ra các triệu chứng như sốt, ho nhiều, khó thở, thở khò khè… Việc điều trị viêm phế quản thường tập trung vào việc giảm triệu chứng và hỗ trợ quá trình phục hồi. ‏⏎⏎‏- Đảm bảo n  
  cuối: …quản như terbutaline, salbutamol dạng ống hít.Thuốc thường được chỉ định ở trẻ co thắt phế quản với triệu chứng thở khò khè. Cần lưu ý sử dụng thuốc theo hướng dẫn của bác sĩ, không tự ý sử dụng thuốc tại nhà cho trẻ.⏎⏎‏

- **[rand]** doc_id `508900` · 3873 ký tự, 18 dòng body · title: `Cứu sống bệnh nhân 17 tuổi hôn mê do u não hiếm gặp tại Bệnh viện K`  
  https://suckhoedoisong.vn/cuu-song-benh-nhan-17-tuoi-hon-me-do-u-nao-hiem-gap-tai-benh-vien-k-169251125103534611.htm  
  đầu: Cứu sống bệnh nhân 17 tuổi hôn mê do u não hiếm gặp tại Bệnh viện K⏎⏎'Đưa con đến bệnh viện trong tình trạng còn nước còn tát'⏎⏎Trước đó, em Lý Văn D. (17 tuổi, Tuyên Quang) nhập bệnh viện K trong tình trạng hôn mê, khối u não ở vùng tuyến tùng phát triển lan tràn quanh não thất, gây ứ nước não. Gia đình đưa người bệnh đến bệnh viện K với tâm trạng còn nước còn tát.⏎⏎Các bác sĩ  
  cuối: …ập nhật chuyên sâu hơn. Bởi vậy, rất nhiều trường hợp như em Lý Văn D. đã được điều trị hiệu quả, có thể trở lại cuộc sống thường nhật và hạn chế những lo ngại về biến chứng hay nguy cơ tái phát bệnh trong thời gian dài.

- **[rand]** doc_id `541507` · 2008 ký tự, 9 dòng body · title: `Phòng ngừa viêm quanh răng để tránh mất răng`  
  https://suckhoedoisong.vn/phong-tranh-viem-quanh-rang-de-tranh-mat-rang-169210806123250855.htm  
  đầu: Phòng ngừa viêm quanh răng để tránh mất răng⏎⏎Hỏi: Tôi 37 tuổi, sức khỏe bình thường. Gần đây tôi thấy phần lợi xung quanh răng hơi sưng đỏ, đánh răng thấy có máu. Nhiều người nói tôi bị viêm quanh răng. Xin bác sĩ tư vấn nguyên nhân và cách phòng tránh căn bệnh này.⏎⏎Trần Hùng (Hải Phòng)⏎⏎Nguyên nhân gây viêm quanh răng⏎⏎Viêm quanh răng là một bệnh viêm mạn tính của lợi do nh  
  cuối: …quanh răng nếu không điều trị sớm có thể gặp các biến chứng như tụt lợi, chảy máu chân răng, mất răng…Vì vậy, nếu thực hiện những điều trên mà không đỡ bạn nên đến ngay cơ sở nha khoa uy tín để khám và điều trị kịp thời.

- **[rand]** doc_id `560553` · 5208 ký tự, 16 dòng body · title: `Viêm họng hạt hay tái phát, vì sao?`  
  https://suckhoedoisong.vn/viem-hong-hat-hay-tai-phat-vi-sao-169146668.htm  
  đầu: Viêm họng hạt hay tái phát, vì sao?⏎⏎Viêm họng hạt là tình trạng viêm họng mạn tính, viêm nhiễm kéo dài và quá lâu. Bệnh tuy không gây ảnh hưởng tới tính mạng người bệnh nhưng gây khó khăn trong giao tiếp do ngứa, vướng họng khi đang nói chuyện, vừa nói vừa phải dừng lại để khạc đờm và làm suy giảm chất lượng cuộc sống của họ. Vậy viêm họng có nguy hiểm và tại sao lại hay tái p  
  cuối: … kiêng rượu mạnh, tránh ăn kem, uống nước đá lạnh... Luôn giữ ấm vùng cổ, tập thở sâu (phương pháp thở bụng) dài hơi, nhẹ nhàng và đều đặn. Khi bị viêm họng cấp cần điều trị dứt điểm, không để kéo dài trở thành mạn tính.

## thanhnien.vn (71,607 doc)

- **[short]** doc_id `622798` · 446 ký tự, 2 dòng body · title: `Khoai tây giúp chống cảm cúm`  
  https://thanhnien.vn/khoai-tay-giup-chong-cam-cum-185291535.htm  
  đầu: Khoai tây giúp chống cảm cúm⏎⏎Khoai tây giúp chống cảm cúm⏎⏎Các chuyên gia y tế thuộc Trung tâm dinh dưỡng Yale (Mỹ) cho biết, khoai tây là một thực phẩm giàu vitamin C nên chúng có thể giúp phòng chống cảm cúm. Khoai tây còn chứa nhiều kali và chất xơ. Theo các chuyên gia, ăn khoai tây lúc còn tươi tốt hơn vì nó vẫn còn giữ nhiều vitamin C, vốn có tác dụng ngăn chặn sự lây nhi  
  cuối: …chứa nhiều kali và chất xơ. Theo các chuyên gia, ăn khoai tây lúc còn tươi tốt hơn vì nó vẫn còn giữ nhiều vitamin C, vốn có tác dụng ngăn chặn sự lây nhiễm cảm cúm và giúp duy trì hệ thống miễn dịch khỏe mạnh. (H.Y/MSN)

- **[short]** doc_id `614516` · 404 ký tự, 1 dòng body · title: `Điện thoại di động và chứng mất ngủ`  
  https://thanhnien.vn/dien-thoai-di-dong-va-chung-mat-ngu-185300308.htm  
  đầu: Điện thoại di động và chứng mất ngủ⏎⏎Theo AFP, các chuyên gia kể trên đã nghiên cứu trên 71 người từ 18-45 tuổi trong vòng 18 tháng, bằng cách cho một số người trong số họ tiếp xúc với các tín hiệu vô tuyến ở tần số 884 MHz, tương đương với lượng bức xạ họ nhận được khi nói chuyện qua điện thoại di động. Kết quả cho thấy những người này đã bị đau đầu, khó ngủ và nếu có ngủ được  
  cuối: … các tín hiệu vô tuyến ở tần số 884 MHz, tương đương với lượng bức xạ họ nhận được khi nói chuyện qua điện thoại di động. Kết quả cho thấy những người này đã bị đau đầu, khó ngủ và nếu có ngủ được thì giấc ngủ không sâu.

- **[short]** doc_id `623699` · 370 ký tự, 2 dòng body · title: `Kiểm tra tim mạch trước khi chơi thể thao`  
  https://thanhnien.vn/kiem-tra-tim-mach-truoc-khi-choi-the-thao-185316672.htm  
  đầu: Kiểm tra tim mạch trước khi chơi thể thao⏎⏎Ngoài những kiểm tra thông thường, họ được khảo sát thêm điện tâm đồ và siêu âm tim nhằm phát hiện khả năng mắc bệnh tim mạch. Kết quả, phát hiện khoảng 10% vận động viên có bất thường trên điện tâm đồ cần theo dõi, có chín người mắc bệnh tim mạch và hai người cần phải dừng chơi các môn thể thao cạnh tranh cao.⏎⏎Theo Tuổi Trẻ  
  cuối: … mắc bệnh tim mạch. Kết quả, phát hiện khoảng 10% vận động viên có bất thường trên điện tâm đồ cần theo dõi, có chín người mắc bệnh tim mạch và hai người cần phải dừng chơi các môn thể thao cạnh tranh cao.⏎⏎Theo Tuổi Trẻ

- **[short]** doc_id `641655` · 416 ký tự, 4 dòng body · title: `Phẫu thuật thành công túi thừa thực quản`  
  https://thanhnien.vn/phau-thuat-thanh-cong-tui-thua-thuc-quan-185315347.htm  
  đầu: Phẫu thuật thành công túi thừa thực quản⏎⏎Kết quả chụp X-quang cho thấy ông Đoàn có một túi thừa thực quản nằm trên cơ hoành. Đây là chứng bệnh rất hiếm gặp, người bị bệnh này ăn vào là nôn mửa.⏎⏎Ngày 9.1, Bệnh viện đa khoa T.Ư Cần Thơ kết hợp Bệnh viện Chợ Rẫy tiến hành mổ nội soi ổ bụng cắt túi thực quản thừa trên cơ hoành có kích thước 7x8cm.⏎⏎Hiện sức khỏe ông Đoàn đã ổn đị  
  cuối: …Ngày 9.1, Bệnh viện đa khoa T.Ư Cần Thơ kết hợp Bệnh viện Chợ Rẫy tiến hành mổ nội soi ổ bụng cắt túi thực quản thừa trên cơ hoành có kích thước 7x8cm.⏎⏎Hiện sức khỏe ông Đoàn đã ổn định, có thể ăn uống được.⏎⏎Thanh Dũng

- **[short]** doc_id `601579` · 293 ký tự, 1 dòng body · title: `Bú sữa mẹ giúp bé ngoan hơn`  
  https://thanhnien.vn/bu-sua-me-giup-be-ngoan-hon-185127109.htm  
  đầu: Bú sữa mẹ giúp bé ngoan hơn⏎⏎Và mức độ ngoan của trẻ phụ thuộc vào thời gian mẹ cho trẻ bú, nghĩa là trẻ được bú mẹ đến 1 tuổi ít bị những vấn đề về hành vi trong 5 năm đầu đời so với trẻ chỉ bú mẹ được 2 tháng. Các nghiên cứu trước đó cũng khẳng định cả mẹ và bé đều có lợi khi trẻ bú sữa mẹ.  
  cuối: …ian mẹ cho trẻ bú, nghĩa là trẻ được bú mẹ đến 1 tuổi ít bị những vấn đề về hành vi trong 5 năm đầu đời so với trẻ chỉ bú mẹ được 2 tháng. Các nghiên cứu trước đó cũng khẳng định cả mẹ và bé đều có lợi khi trẻ bú sữa mẹ.

- **[title]** doc_id `604935` · 1387 ký tự, 6 dòng body · title: `Cây cúc tần`  
  https://thanhnien.vn/cay-cuc-tan-18593660.htm  
  đầu: Cây cúc tần⏎⏎Rất dễ trồng, mọc hoang (thân dày, lá to, vị hơi chát đắng), mọc ở vườn (còn nhỏ thân lá chua chát, già thân lá nồng gắt). Vị nhẫn đắng, mùi the, tinh dầu thơm hương cam, tính ấm. Lá dày 3-5mm, to cỡ 3-5cm. Khi lá già có tác dụng tiêu tán phong hàn, trị dứt tiêu thũng chỉ thống (giảm đau, xoa dịu cơn đau rất nhanh). Người cao tuổi, phụ nữ vừa hộ sản lạnh, tê hai ch  
  cuối: …cúc trắng xắt sợi nhỏ, 100gr đu đủ vừa chớm chín, 100gr não lợn. Nấu trong 800ml nước (sôi khoảng 15 phút mới bỏ não lợn vào), hầm trong 20-30 phút. Ăn nóng trước bữa cơm, 2 lần/ngày, liền 7 ngày.⏎⏎Lương y Dương Tấn Hưng

- **[title]** doc_id `638906` · 1554 ký tự, 5 dòng body · title: `Nồi lá xông`  
  https://thanhnien.vn/noi-la-xong-185300354.htm  
  đầu: Nồi lá xông⏎⏎Lương y Lê Ngọc Vân, Chủ tịch Hội Đông y TP Phan Rang - Tháp Chàm, tỉnh Ninh Thuận cho biết, có rất nhiều loại lá trồng quanh vườn hoặc mua ở các hàng rau ngoài chợ như: lá é, húng quế, húng đứng, tía tô, sả, bạc hà, rau tần, ngổ, gừng, riềng, hành, chanh, cam, bưởi, bạch đàn, ngũ trảo, tre, ngãi diệp... để làm nồi lá xông. Công dụng chính là giúp cơ thể người bệnh  
  cuối: …ương y Lê Ngọc Vân, nếu dùng nhiều lần mà bệnh không thuyên giảm thì tìm phương cách khác theo đúng nguyên nhân gây bệnh để chữa trị, vì xông hơi kéo dài sẽ mất nhiều nước cho cơ thể, gây ra các tác hại khác.⏎⏎Thiện Nhân

- **[title]** doc_id `621204` · 951 ký tự, 5 dòng body · title: `Hộp thư khỏe - đẹp`  
  https://thanhnien.vn/hop-thu-khoe-dep-185291594.htm  
  đầu: Hộp thư khỏe - đẹp⏎⏎- Tinh dịch bình thường có màu trắng xám hoặc hơi vàng, không lẫn máu. Nếu tinh dịch có màu đỏ, đỏ nhạt, màu nâu hoặc có lẫn sợi máu đều gọi là huyết tinh.⏎⏎Nguyên nhân chủ yếu của bệnh huyết tinh là tinh nang bị viêm do sự lây lan vi khuẩn ở các bộ phận lân cận như trực tràng, bàng quang, niệu đạo, tiền liệt tuyến. Ở nước ta vi khuẩn gây viêm thường thấy là  
  cuối: … nên đến Bệnh viện Bình Dân để kiểm tra và chữa trị gấp. Khi viêm tinh nang hết thì huyết tinh cũng biến mất.⏎⏎Mọi thắc mắc có liên quan đến sức khỏe, xin vui lòng gửi thư về tòa soạn hoặc địa chỉ e-mail: tnct@hcm.vnn.vn

- **[title]** doc_id `653882` · 1190 ký tự, 7 dòng body · title: `Trị tóc gàu`  
  https://thanhnien.vn/tri-toc-gau-1855528.htm  
  đầu: Trị tóc gàu⏎⏎Có thể dùng dầu ô liu để dưỡng ẩm da đầu và ngừa gàu - Ảnh: Shutterstock⏎⏎- Các loại dầu tự nhiên: Dầu dừa, dầu hạnh nhân hoặc dầu ô liu có thể được sử dụng để dưỡng ẩm da đầu và ngừa gàu. Chỉ cần đổ dầu ra chén, sau đó thoa lên da đầu và mát xa nhẹ. Dùng khăn quấn đầu và để như vậy qua đêm. Gội sạch tóc vào sáng hôm sau.⏎⏎- Giấm: Sau khi gội đầu, dùng 2 chén giấm   
  cuối: … một lần nữa để loại bỏ hết bột. Aspirin có chứa salicylate, một loại hoạt chất giúp trị gàu.⏎⏎Nhất Linh⏎⏎>> Lợi ích từ dầu ô liu >> Mau no nhờ dùng dầu ô liu >> Phục hồi da với dầu ô liu >> Xương chắc khỏe nhờ dầu ô liu

- **[nav]** doc_id `642427` · 808 ký tự, 68 dòng body · title: `Phục hồi sức khỏe hậu covid-19`  
  https://thanhnien.vn/phuc-hoi-suc-khoe-hau-covid-19-bac-si-huong-dan-dieu-tri-1851454530.htm  
  đầu: Phục hồi sức khỏe hậu covid-19⏎⏎Trang chủ⏎⏎Video⏎⏎Thời sự⏎⏎Tôi viết⏎⏎Thế giới⏎⏎Chính trị⏎⏎Văn hóa⏎⏎Thể thao⏎⏎Đời sống⏎⏎Kinh doanh⏎⏎Giới trẻ⏎⏎Giáo dục⏎⏎Công nghệ⏎⏎Game⏎⏎Sức khỏe⏎⏎Xe⏎⏎Trang chủ⏎⏎Video⏎⏎Thời sự⏎⏎Tôi viết⏎⏎Thế giới⏎⏎Chính trị⏎⏎Văn hóa⏎⏎Thể thao⏎⏎Đời sống⏎⏎Kinh doanh⏎⏎Giới trẻ⏎⏎Giáo dục⏎⏎Công nghệ⏎⏎Game⏎⏎Sức khỏe⏎⏎Xe⏎⏎XEM NHANH⏎⏎ĐÓNG⏎⏎BÌNH LUẬN⏎⏎XEM THÊM⏎⏎ĐÓNG⏎⏎Covi  
  cuối: …an tâm nhất⏎⏎Mới nhất⏎⏎Xem thêm bình luận⏎⏎Top⏎⏎Bạn không thể gửi bình luận liên tục. Xin hãy đợi⏎⏎60⏎⏎giây nữa.⏎⏎Bình luận (⏎⏎0⏎⏎)⏎⏎Gửi bình luận⏎⏎Đăng nhập để bình luận⏎⏎Quan tâm nhất⏎⏎Mới nhất⏎⏎Xem thêm bình luận⏎⏎Top

- **[nav]** doc_id `620725` · 774 ký tự, 68 dòng body · title: `Hỏi đáp nhanh`  
  https://thanhnien.vn/hoi-dap-nhanh-covid-19-bac-si-chuyen-gia-tra-loi-giai-dap-1851454532.htm  
  đầu: Hỏi đáp nhanh⏎⏎Trang chủ⏎⏎Video⏎⏎Thời sự⏎⏎Tôi viết⏎⏎Thế giới⏎⏎Chính trị⏎⏎Văn hóa⏎⏎Thể thao⏎⏎Đời sống⏎⏎Kinh doanh⏎⏎Giới trẻ⏎⏎Giáo dục⏎⏎Công nghệ⏎⏎Game⏎⏎Sức khỏe⏎⏎Xe⏎⏎Trang chủ⏎⏎Video⏎⏎Thời sự⏎⏎Tôi viết⏎⏎Thế giới⏎⏎Chính trị⏎⏎Văn hóa⏎⏎Thể thao⏎⏎Đời sống⏎⏎Kinh doanh⏎⏎Giới trẻ⏎⏎Giáo dục⏎⏎Công nghệ⏎⏎Game⏎⏎Sức khỏe⏎⏎Xe⏎⏎XEM NHANH⏎⏎ĐÓNG⏎⏎BÌNH LUẬN⏎⏎XEM THÊM⏎⏎ĐÓNG⏎⏎Covid-19: Điều trị và  
  cuối: …an tâm nhất⏎⏎Mới nhất⏎⏎Xem thêm bình luận⏎⏎Top⏎⏎Bạn không thể gửi bình luận liên tục. Xin hãy đợi⏎⏎60⏎⏎giây nữa.⏎⏎Bình luận (⏎⏎0⏎⏎)⏎⏎Gửi bình luận⏎⏎Đăng nhập để bình luận⏎⏎Quan tâm nhất⏎⏎Mới nhất⏎⏎Xem thêm bình luận⏎⏎Top

- **[nav]** doc_id `630302` · 2089 ký tự, 27 dòng body · title: `Nếu nhiễm Covid-19, triệu chứng giữa người đã tiêm và chưa tiêm vắc xin như thế nào?`  
  https://thanhnien.vn/neu-nhiem-covid-19-trieu-chung-giua-nguoi-da-tiem-va-chua-tiem-vac-xin-nhu-the-nao-1851102805.htm  
  đầu: Nếu nhiễm Covid-19, triệu chứng giữa người đã tiêm và chưa tiêm vắc xin như thế nào?⏎⏎Không những vậy, người chưa tiêm vắc xin có nguy cơ bị nhiều triệu chứng hơn khi nhiễm Covid-19. Các triệu chứng cũng ở mức độ nặng hơn, Business Insider đưa tin.⏎⏎Vắc xin giúp ngăn ngừa nguy cơ bệnh tiến triển nặng khi nhiễm Covid-19. Dữ liệu của Trung tâm Kiểm soát và Phòng ngừa Dịch bệnh Mỹ  
  cuối: …xin và người không tiêm vắc xin Covid-19. Những người đã tiêm đủ 2 mũi bị các triệu chứng nhẹ hơn, thậm chí chỉ là triệu chứng thoáng qua, thời gian hồi phục cũng nhanh hơn người chưa tiêm vắc xin, theo Business Insider.

- **[nav]** doc_id `641899` · 1517 ký tự, 31 dòng body · title: `Phòng dịch COVID-19: 3 công thức nước ép giúp tăng cường hệ miễn dịch mạnh mẽ`  
  https://thanhnien.vn/phong-dich-covid-19-3-cong-thuc-nuoc-ep-giup-tang-cuong-he-mien-dich-manh-me-185936419.htm  
  đầu: Phòng dịch COVID-19: 3 công thức nước ép giúp tăng cường hệ miễn dịch mạnh mẽ⏎⏎Một cách hiệu quả để tăng cường hệ miễn dịch là tiêu thụ các nguồn thực phẩm tăng cường miễn dịch.⏎⏎Sau đây là 3 loại nước ép tăng cường hệ miễn dịch mạnh mẽ nhất, theo Vitally For Life.⏎⏎Công thức 1⏎⏎Công thức này chứa một lượng lớn vitamin và khoáng chất mà hệ miễn dịch sẽ yêu thích. Tất cả trong m  
  cuối: …ả dứa⏎⏎• 1 cục gừng⏎⏎• 2 tép tỏi⏎⏎Ngoài ra, còn có công thức nước ép bổ dưỡng từ nghệ, cam quýt và cà rốt và công thức từ giấm táo, gừng và mật ong cũng có sức khả năng cải thiện hệ miễn dịch mạnh mẽ, theo Medical Daily.

- **[nav]** doc_id `638712` · 1255 ký tự, 17 dòng body · title: `Những vị thuốc tăng sắc đẹp`  
  https://thanhnien.vn/nhung-vi-thuoc-tang-sac-dep-185137673.htm  
  đầu: Những vị thuốc tăng sắc đẹp⏎⏎Bổ cốt chỉ hoàn⏎⏎* Xuất xứ: từ đời Thanh⏎⏎* Nguyên liệu:⏎⏎+ Bổ cốt chỉ (150gr), đem sao thơm, tán thành bột mịn⏎⏎+ Đường cát trắng (120gr)⏎⏎+ Hồ đào nhục (60gr), tán nhuyễn như cháo⏎⏎* Cách chế biến: Hồ đào nhục và đường cát tán nhuyễn như hồ, cho bột Bổ cốt chỉ vào trộn đều, rồi đem hấp chín, làm thành viên vừa uống. Cho vào lọ đậy kín, mỗi lần dùn  
  cuối: …g 3 ngày. Ăn liên tục trong 6 tháng, giảm dần sau 6 tháng, sau một năm. Bài thuốc này có thể dùng thay bữa ăn sáng. Dùng đúng và liên tục có công dụng giúp vóc dáng cơ thể kiều diễm, giữ lâu nét tươi trẻ.⏎⏎Khánh Vy (ghi)

- **[rand]** doc_id `645486` · 2135 ký tự, 6 dòng body · title: `Sức khỏe tâm thần có liên quan đến tuổi thọ như thế nào?`  
  https://thanhnien.vn/suc-khoe-tam-than-co-lien-quan-den-tuoi-tho-nhu-the-nao-185978243.htm  
  đầu: Sức khỏe tâm thần có liên quan đến tuổi thọ như thế nào?⏎⏎Nghiên cứu mới được công bố trên Tạp chí Nghiên cứu Tâm lý học, dẫn đầu bởi các nhà nghiên cứu Đại học Toronto (Canada), đã theo dõi 12.424 người Canada trên 18 tuổi từ giữa những năm 1990 đến năm 2011. Các nhà nghiên cứu phát hiện ra rằng những người có sức khỏe tâm thần kém khi bắt đầu nghiên cứu chết sớm hơn trung bìn  
  cuối: … sức khỏe tâm thần dưới mức tối ưu và tử vong sớm là một mối quan hệ mạnh mẽ, không phụ thuộc vào tình trạng sức khỏe, đau đớn, hạn chế chức năng và các hành vi sức khỏe tiêu cực”, PC dẫn lời tiến sĩ Esme Fuller-Thomson.

- **[rand]** doc_id `633361` · 1254 ký tự, 7 dòng body · title: `Ngừa rụng tóc không khó`  
  https://thanhnien.vn/ngua-rung-toc-khong-kho-185369752.htm  
  đầu: Ngừa rụng tóc không khó⏎⏎Không hút thuốc. Thuốc lá có tác động tiêu cực tới hệ tuần hoàn của cơ thể. Vì vậy, nếu bạn muốn ngừa rụng tóc, đồng thời tiết kiệm một số tiền đáng kể thì hãy chịu khó, hạ quyết tâm từ bỏ hút thuốc lá.⏎⏎Tập thể dục, hạn chế căng thẳng. Quá nhiều căng thẳng có thể khiến tóc bạn rụng nhanh, chậm mọc lại tóc mới. Tập thể dục có thể giúp bạn đối phó với că  
  cuối: …ng caffeine hấp thụ vào cơ thể mỗi ngày.⏎⏎Mai Duyên⏎⏎>> Rụng tóc không liên quan với viêm da >> Rụng tóc do ăn kiêng giảm cân cấp tốc >> Rụng tóc >> Trị rụng tóc tại nhà >> Ngừa rụng tóc >> Chống rụng tóc tuổi trung niên

- **[rand]** doc_id `597515` · 2010 ký tự, 7 dòng body · title: `Bé gái 7 tuổi nhiễm Covid-19 bị hội chứng hiếm gặp`  
  https://thanhnien.vn/be-gai-7-tuoi-nhiem-covid-19-bi-hoi-chung-hiem-gap-1851429229.htm  
  đầu: Bé gái 7 tuổi nhiễm Covid-19 bị hội chứng hiếm gặp⏎⏎Ngày 13.2, theo thông tin từ Bệnh viện Nhi đồng Cần Thơ, Khoa hồi sức Covid-19 của bệnh viện này vừa cứu sống kịp thời bé gái N.N.H. (7 tuổi, quê Vĩnh Long) nhập viện trong tình trạng nhiễm Covid-19 và bị hội chứng đa hệ thống hiếm gặp.⏎⏎Bé H. đã được các bác sĩ Bệnh viện Nhi đồng Cần Thơ cứu sống⏎⏎Trước nhập viện, bé H. bị số  
  cuối: …t hiện sớm MIS-C, tránh nguy cơ tử vong ở trẻ. Cách tốt nhất để bảo vệ trẻ là thực hiện phòng ngừa cho trẻ và toàn bộ gia đình không bị nhiễm Covid-19. Tái khám hậu Covid-19 sau 2 tuần ngưng cách ly tại nhà”, BS Tạo nói.

- **[rand]** doc_id `601847` · 1055 ký tự, 6 dòng body · title: `BVĐK Hồng Đức 3 trở thành vệ tinh của BV Nhi đồng 2`  
  https://thanhnien.vn/bvdk-hong-duc-3-tro-thanh-ve-tinh-cua-bv-nhi-dong-2-185669242.htm  
  đầu: BVĐK Hồng Đức 3 trở thành vệ tinh của BV Nhi đồng 2⏎⏎Nhằm góp phần giải quyết tình trạng quá tải bệnh nhân, đáp ứng nhu cầu khám chữa bệnh của người dân, hạn chế tình trạng lây nhiễm chéo, BV Nhi đồng 2 đã đề xuất liên kết với BVĐK Hồng Đức 3 trong công tác khám chữa bệnh nhi.⏎⏎Việc thăm khám, điều trị, phẫu thuật sẽ được thực hiện bởi các bác sĩ của BV Nhi đồng 2 tại BVĐK Hồng  
  cuối: …tế TP.HCM công bố kết quả khảo sát ý kiến người dân về những điểm không hài lòng khi khám, chữa bệnh thông qua hệ thống ki ốt (điểm đặt máy) sau gần 1 tháng triển khai tại các bệnh viện (BV) công lập trên địa bàn TP.HCM.

- **[rand]** doc_id `647490` · 1065 ký tự, 3 dòng body · title: `Tập thể dục tốt cho bệnh nhân ung thư vú`  
  https://thanhnien.vn/tap-the-duc-tot-cho-benh-nhan-ung-thu-vu-185402548.htm  
  đầu: Tập thể dục tốt cho bệnh nhân ung thư vú⏎⏎Theo healthday.com, các chuyên gia thuộc Hội đồng về tập thể dục Mỹ đưa ra những lợi ích tiềm năng của tập thể dục như sau: có thể làm giảm nguy cơ tái phát ung thư vú, tử vong do ung thư vú và tử vong vì các nguyên nhân khác so với những người sống sót sau ca phẫu thuật ung thư vú mà không tập thể dục; có thể giúp ngăn ngừa một số vấn   
  cuối: …vú >> Phát hiện quan trọng về ung thư vú >> Thiếu ngủ dễ tái phát ung thư vú >> Biết mắc bệnh ung thư vú nhờ... chó >> Thuốc trị ung thư vú kém tác dụng với phụ nữ béo phì >> Ung thư vú và liệu pháp phẫu thuật bảo tồn vú

- **[rand]** doc_id `636657` · 1866 ký tự, 13 dòng body · title: `Những cách tự nhiên giúp loại bỏ mụn đầu đen hiệu quả`  
  https://thanhnien.vn/nhung-cach-tu-nhien-giup-loai-bo-mun-dau-den-hieu-qua-185923263.htm  
  đầu: Những cách tự nhiên giúp loại bỏ mụn đầu đen hiệu quả⏎⏎Mụn đầu đen được hình thành bởi melanin ô xy hóa, sắc tố đen được tạo ra bởi các tế bào trong da. Khi lỗ chân lông trên da bị tắc, dầu bã nhờn hoặc tế bào da chết trong lỗ chân lông mở sẽ phản ứng với ô xy trong không khí và chuyển sang màu đen. Đây là cách mà mụn đầu đen hình thành. Các tác nhân như tuổi tác và thay đổi nộ  
  cuối: …ũng như mụn đầu đen. Thoa nước cốt chanh tươi bằng bông gòn để làm sạch da, tập trung vào vùng mụn đầu đen. Sau vài phút, rửa sạch bằng nước ấm. Bạn cũng có thể để nó qua đêm để có tác dụng tốt hơn, theo The Health Site.

## phunusuckhoe.giadinhonline.vn (27,575 doc)

- **[short]** doc_id `299431` · 312 ký tự, 2 dòng body · title: `Làm sao để sở hữu kiểu tóc đẹp và hớp hồn phái mạnh mùa đi biển?`  
  https://phunusuckhoe.giadinhonline.vn/lam-sao-de-so-huu-kieu-toc-dep-va-hop-hon-phai-manh-mua-di-bien-c12a289286.html  
  đầu: Làm sao để sở hữu kiểu tóc đẹp và hớp hồn phái mạnh mùa đi biển?⏎⏎Một chiếc khăn lụa mỏng và dải viền màu cột kiểu đuôi ngựa thay vì xõa tóc, bạn sẽ thấy thú vị hơn rất nhiều nếu kết hợp cùng khăn choàng in sáng màu.⏎⏎Cùng chiếc khăn lụa bạn có thể sáng tạo với nhiều kiểu khác cho mùa hè này từ những gợi ý sau:  
  cuối: … dải viền màu cột kiểu đuôi ngựa thay vì xõa tóc, bạn sẽ thấy thú vị hơn rất nhiều nếu kết hợp cùng khăn choàng in sáng màu.⏎⏎Cùng chiếc khăn lụa bạn có thể sáng tạo với nhiều kiểu khác cho mùa hè này từ những gợi ý sau:

- **[short]** doc_id `292612` · 337 ký tự, 1 dòng body · title: `Cập nhật 9 kiểu tóc dự đoán sẽ lên ngôi vào năm 2018, hội con gái xem ngay kẻo lỡ`  
  https://phunusuckhoe.giadinhonline.vn/cap-nhat-9-kieu-toc-du-doan-se-len-ngoi-vao-nam-2018-hoi-con-gai-xem-ngay-keo-lo-c12a132228.html  
  đầu: Cập nhật 9 kiểu tóc dự đoán sẽ lên ngôi vào năm 2018, hội con gái xem ngay kẻo lỡ⏎⏎Chỉ còn vài chục ngày nữa là bước sang năm mới 2018, điều này đồng nghĩa với việc bạn cần phải đổi mới để đón "Mùa lễ hội" sắp đến. Để trở nên tươi mới và nổi bật giữa đám đông, hãy thay mới bản thân với 9 kiểu tóc dự đoán lên ngôi vào năm 2018 dưới đây.  
  cuối: …sang năm mới 2018, điều này đồng nghĩa với việc bạn cần phải đổi mới để đón "Mùa lễ hội" sắp đến. Để trở nên tươi mới và nổi bật giữa đám đông, hãy thay mới bản thân với 9 kiểu tóc dự đoán lên ngôi vào năm 2018 dưới đây.

- **[short]** doc_id `297623` · 397 ký tự, 2 dòng body · title: `Hàng loạt sao Hàn cũng phải "nát mặt" vì mụn bọc, mụn trứng cá`  
  https://phunusuckhoe.giadinhonline.vn/hang-loat-sao-han-cung-phai-nat-mat-vi-mun-boc-mun-trung-ca-c12a5228.html  
  đầu: Hàng loạt sao Hàn cũng phải "nát mặt" vì mụn bọc, mụn trứng cá⏎⏎Lịch làm việc dày đặc, kèm theo việc phải trang điểm thường xuyên khiến làn da các sao gặp rất nhiều vấn đề nhất là mụn và các vết thâm do mụn để lại. Dù được che giấu kỹ bằng lớp make up dày và photoshop nhưng vẫn đôi lúc làn da thật sự của các sao Hàn bị phơi bày.⏎⏎Cùng xem những sao Hàn ngày ngày phải khổ sở vị   
  cuối: …là mụn và các vết thâm do mụn để lại. Dù được che giấu kỹ bằng lớp make up dày và photoshop nhưng vẫn đôi lúc làn da thật sự của các sao Hàn bị phơi bày.⏎⏎Cùng xem những sao Hàn ngày ngày phải khổ sở vị mụn trứng cá nhé:

- **[short]** doc_id `295127` · 347 ký tự, 2 dòng body · title: `Dành cho những cô nàng vụng về: Trang điểm chưa bao giờ dễ đến thế!`  
  https://phunusuckhoe.giadinhonline.vn/danh-cho-nhung-co-nang-vung-ve-trang-diem-chua-bao-gio-de-den-the-c12a9422.html  
  đầu: Dành cho những cô nàng vụng về: Trang điểm chưa bao giờ dễ đến thế!⏎⏎Trang điểm tuy không khó nhưng có rất nhiều điều bạn cần lưu ý và đây chính là lý do khiến các bạn gái muốn học trang điểm không biết nên bắt đầu từ đâu. Chỉ với những mẹo dưới đây, bạn sẽ thấy việc tô vẽ gương mặt không còn là nỗi sợ hãi mỗi ngày nữa.⏎⏎Chúc các bạn thành công!  
  cuối: …ưu ý và đây chính là lý do khiến các bạn gái muốn học trang điểm không biết nên bắt đầu từ đâu. Chỉ với những mẹo dưới đây, bạn sẽ thấy việc tô vẽ gương mặt không còn là nỗi sợ hãi mỗi ngày nữa.⏎⏎Chúc các bạn thành công!

- **[short]** doc_id `291037` · 390 ký tự, 2 dòng body · title: `Các chỉ số sức khỏe bạn cần biết`  
  https://phunusuckhoe.giadinhonline.vn/cac-chi-so-suc-khoe-ban-can-biet-c12a305040.html  
  đầu: Các chỉ số sức khỏe bạn cần biết⏎⏎Nhịp đập tim trung bình của người bình thường dao động trong khoảng 60-100 nhịp trên một phút. Nhịp tim của người ở độ tuổi thanh niên thường chậm hơn so với người lớn tuổi. Nhịp tim của nam giới thường chậm hơn nữ giới.⏎⏎Nhiệt độ cơ thể vượt quá ngưỡng 36,2-37,2 độ C nhiều khả năng bạn bị sốt và cần điều trị bằng thuốc hạ sốt kết hợp với chế đ  
  cuối: …hường chậm hơn so với người lớn tuổi. Nhịp tim của nam giới thường chậm hơn nữ giới.⏎⏎Nhiệt độ cơ thể vượt quá ngưỡng 36,2-37,2 độ C nhiều khả năng bạn bị sốt và cần điều trị bằng thuốc hạ sốt kết hợp với chế độ ăn uống.

- **[title]** doc_id `287812` · 1360 ký tự, 16 dòng body · title: ``  
  https://phunusuckhoe.giadinhonline.vn/8-meo-lam-dep-tiet-kiem-trong-nhung-ngay-cuoi-thang-nhan-tui-c12a297902.html  
  đầu: 8 mẹo làm đẹp tiết kiệm trong những ngày cuối tháng nhẵn túi⏎⏎Tận dụng cây mascara cũ hay trộn vài món mỹ phẩm vào nhau để có được công thức mới sẽ giúp bạn tiết kiệm được một khoản không nhỏ.⏎⏎Tin liên quan⏎⏎9 bí quyết làm đẹp và trang điểm thông minh giúp phụ nữ vừa tỏa sáng vừa tiết kiệm 'cả...⏎⏎Brightside đã thu thập được 9 mẹo làm đẹp và trang điểm tốt nhất đối với mọi phụ  
  cuối: …à quyến rũ hơn. Nhưng đôi khi những nỗ lực đó lại phản...⏎⏎4 bước vẽ son môi ombre không cần trang điểm chuyên nghiệp cũng làm được⏎⏎Bạn cần chuẩn bị hai màu son cùng tone một đậm một nhạt để tạo được kiểu làn môi pha...

- **[title]** doc_id `284147` · 1416 ký tự, 17 dòng body · title: ``  
  https://phunusuckhoe.giadinhonline.vn/10-nguyen-lieu-re-tien-co-tac-dung-tri-mun-khang-viem-va-lam-sang-da-bep-nha-nao-cung-co-c12a297369.html  
  đầu: Đây là những nguyên liệu rất tốt để chăm sóc da, chị em hãy sử dụng thường xuyên để vừa đảm bảo an toàn cho da lại vừa tiết kiệm tiền nhé!⏎⏎10 nguyên liệu rẻ tiền có tác dụng trị mụn. kháng viêm và làm sáng da bếp nhà nào cũng có⏎⏎Nếu biết tác dụng tuyệt vời của các nguyên liệu này, ắt hẳn bạn sẽ phải ngạc nhiên.⏎⏎Tin liên quan⏎⏎6 công thức giúp sáng da, sạch mụn với bột nghệ r  
  cuối: …c, dầu mè cũng là "thần dược" ngăn ngừa lão hóa...⏎⏎7 cách trị mụn lưng bằng nguyên liệu thiên nhiên, ai không biết thật quá phí!⏎⏎Chỉ cần bạn lắng nghe và kiên trì thực hiện theo cách trị mụn lưng, mụn dần sẽ rút lui...

- **[title]** doc_id `286174` · 1077 ký tự, 19 dòng body · title: ``  
  https://phunusuckhoe.giadinhonline.vn/5-kieu-toc-gian-di-ma-dep-lung-linh-con-gai-cu-de-la-duoc-khen-xinh-het-loi-c12a127013.html  
  đầu: 1. Tóc bob uốn cúp đuôi⏎⏎2. Tóc lob xoăn nhẹ⏎⏎3. Tóc suôn thẳng rẽ ngôi giữa⏎⏎4. Tóc buộc thấp⏎⏎5. Tóc buộc nửa đầu mái thưa⏎⏎Phòng bệnh⏎⏎5 kiểu tóc giản dị mà đẹp lung linh, con gái cứ để là được khen xinh hết lời⏎⏎Không nhiều chi tiết và cầu kỳ nhưng 5 kiểu tóc dưới đây lại giúp các nàng trở nên xinh xắn, trẻ trung vô cùng.⏎⏎30/10/2017 10:55⏎⏎1 ngày 17 giờ trước⏎⏎Đau họng buổ  
  cuối: … tiếp và thu...⏎⏎Rửa trứng ngay sau khi mua tưởng giúp làm sạch vi khuẩn, nhưng có thể ảnh hưởng đến lớp màng...⏎⏎Thức ăn thừa không nhất thiết phải bỏ. Nếu được làm lạnh sớm, bảo quản ở nhiệt độ phù hợp...⏎⏎17 giờ trước

- **[title]** doc_id `284110` · 1272 ký tự, 16 dòng body · title: ``  
  https://phunusuckhoe.giadinhonline.vn/10-loi-trang-diem-khien-ban-trong-kem-sang-c12a303140.html  
  đầu: 10 lỗi trang điểm khiến bạn trông kém sang⏎⏎Không dưỡng ẩm da trước khi trang điểm sẽ khiến lớp makeup nhanh bị mốc, không bền màu.⏎⏎Tin liên quan⏎⏎25 học sinh bị thương trong vụ sập giàn giáo, 15 em đã chuyển vào bệnh viện Chợ Rẫy⏎⏎Trong số các học sinh gặp nạn vì sập giàn giáo tại trường tiểu học Huỳnh Văn Bánh, có 10...⏎⏎Sập giàn giáo trong tường tiểu học ngày 20/11, 25 học   
  cuối: …ó làn da khô và dễ bị tổn thương khi thời tiết lạnh thì việc chuẩn...⏎⏎7 mẹo nhỏ giúp việc trang điểm dễ dàng, tiết kiệm hơn⏎⏎Đừng vội vàng vứt bỏ cây mascara đã khô, hãy thử 'hồi sinh' bằng cách ngâm sản phẩm vào bát...

- **[nav]** doc_id `296617` · 1233 ký tự, 11 dòng body · title: `Dùng nước gạo theo cách này trong 7 ngày da đen bao nhiêu cũng trắng bóc`  
  https://phunusuckhoe.giadinhonline.vn/dung-nuoc-gao-theo-cach-nay-trong-7-ngay-da-den-bao-nhieu-cung-trang-boc-c12a54862.html  
  đầu: Dùng nước gạo theo cách này trong 7 ngày da đen bao nhiêu cũng trắng bóc⏎⏎Từ xa xưa con người đã sử dụng nó như một loại “mỹ phẩm” để cải thiện vẻ đẹp của làn da. Bên cạnh đó, theo các nghiên cứu đã chỉ ra, nước vo gạo chứa rất nhiều vitamin B, chất chống oxy hóa và khoáng chất.⏎⏎Sử dụng nước vo gạo thường xuyên làn da sẽ luôn thông thoáng, sạch sẽ, đồng thời các nếp nhăn, đốm   
  cuối: …ửa sơ mặt với nước lạnh sau đó đắp hỗn hợp này lên da mặt, massage nhẹ cho da mặt rồi nằm thư giãn khoảng 30 phút. Rửa sạch lại mặt bằng nước ấm. Mỗi tuần bạn có thể thực hiện 2 -3 lần như vậy để da nhanh trắng hồng nhé.

- **[nav]** doc_id `306285` · 839 ký tự, 12 dòng body · title: `Sắp Tết rồi, học ngay 9 công thức mặt nạ từ tự nhiên này để sở hữu làn da hoàn hảo, không `  
  https://phunusuckhoe.giadinhonline.vn/sap-tet-roi-hoc-ngay-9-cong-thuc-mat-na-tu-tu-nhien-nay-de-so-huu-lan-da-hoan-hao-khong-ti-vet-chi-sau-2-tuan-c12a307535.html  
  đầu: Sắp Tết rồi, học ngay 9 công thức mặt nạ từ tự nhiên này để sở hữu làn da hoàn hảo, không tì vết chỉ sau 2 tuần⏎⏎Có rất nhiều yếu tố ảnh hưởng đến làn da của chúng ta như tuổi tác, khí hậu, căng thẳng... Để ngăn ngừa tối ưu các vấn đề về da, bạn nên bắt đầu chăm sóc từ khi làn da còn khỏe đẹp. Dưới đây là 9 công thức mặt nạ từ tự nhiên có thể giúp bạn đạt được mong muốn duy trì  
  cuối: …ì và sữa tươi⏎⏎9. Trứng và dưa leo⏎⏎Thử nghiệm ngay một trong những công thức mặt nạ từ thành phần tự nhiên trên khoảng 2 đến 3 lần mỗi tuần, chắc chắn chỉ sau thời gian ngắn, bạn sẽ ngỡ ngàng về sự cải thiện của làn da.

- **[nav]** doc_id `296563` · 2040 ký tự, 26 dòng body · title: `Dùng mật ong theo cách này, đẩy lùi lão hóa sớm, xóa mờ nếp nhăn giúp da mặt trẻ ra cả chụ`  
  https://phunusuckhoe.giadinhonline.vn/dung-mat-ong-theo-cach-nay-day-lui-lao-hoa-som-xoa-mo-nep-nhan-giup-da-mat-tre-ra-ca-chuc-tuoi-c12a302197.html  
  đầu: Dùng mật ong theo cách này, đẩy lùi lão hóa sớm, xóa mờ nếp nhăn giúp da mặt trẻ ra cả chục tuổi⏎⏎Để trì hoãn những dấu hiệu như nếp nhăn, đốm đen... bạn có thể nhờ đến các sản phẩm chống lão hóa có sẵn trên thị trường dưới dạng các loại kem, kem.. Nhưng điều đáng lo ngại là đa số mỹ phẩm đều chứa một số loại hóa chất sẽ làm tổn hại thêm cho làn da. Vì vậy, phương pháp tự nhiên  
  cuối: …nh mạnh như ngủ sớm, không hút thuốc, hạn chế bia rượu... Đồng thời, áp dụng chế độ dinh dưỡng khoa học với nhiều rau xanh, trái cây nhằm thanh lọc cơ thể, ngăn ngừa tích tụ độc tố, từ đó trẻ hóa làn da từ sâu bên trong.

- **[nav]** doc_id `284320` · 1154 ký tự, 17 dòng body · title: `14 mẹo trị cháy nắng "một phát ăn ngay" mọi cô gái cần biết`  
  https://phunusuckhoe.giadinhonline.vn/14-meo-tri-chay-nang-mot-phat-an-ngay-moi-co-gai-can-biet-c12a5288.html  
  đầu: 14 mẹo trị cháy nắng "một phát ăn ngay" mọi cô gái cần biết⏎⏎Bất kể lúc nào khi đi ra ngoài ngày hè, bạn cũng cần phải thoa kem chống nắng có chỉ số SPF ít nhất là 30 và thoa lại trong vòng 2 giờ đồng hồ. Nếu không "tuân thủ" điều này, da bị cháy nắng là điều tất yếu. Da bị tổn thương bởi ánh nắng mặt trời thường gây đỏ ứng, đau rát và dần dần sạm màu.⏎⏎Để "cứu cánh" cho làn da  
  cuối: … lotion sữa là lựa chọn thích hợp⏎⏎7. Mặc quần áo rộng⏎⏎8. Thư giãn bằng gel lô hội⏎⏎9. Chườm đá lô hội⏎⏎10. Dùng trà túi lọc⏎⏎11.Bôi thuốc aspirin⏎⏎12. Dùng khăn ướt em bé⏎⏎13. Uống thuốc chống viêm⏎⏎14. Tắm baking soda

- **[nav]** doc_id `302103` · 1940 ký tự, 28 dòng body · title: `Nếu bạn muốn giảm 7kg trong vòng 1 tuần, hãy dùng đậu đen theo 3 cách này`  
  https://phunusuckhoe.giadinhonline.vn/neu-ban-muon-giam-7kg-trong-vong-1-tuan-hay-dung-dau-den-theo-3-cach-nay-c12a101771.html  
  đầu: Nếu bạn muốn giảm 7kg trong vòng 1 tuần, hãy dùng đậu đen theo 3 cách này⏎⏎Với hàm lượng anthocyanin dồi dào, đậu đen được xem là “thần dược” có khả năng đốt cháy mỡ thừa vô cùng nhanh chóng. Hơn nữa, các dưỡng chất, vitamin trong loại đậu này còn giúp làn da trở nên mềm mịn, không xuất hiện vết rạn sau khi giảm cân.⏎⏎Dưới đây là 3 cách chế biến đậu đen giúp cơ thể giảm cân tro  
  cuối: …ứt và nước lọc vào nồi, ninh nhỏ lửa đến khi các nguyên liệu mềm nhừ thì tắt bếp.⏎⏎Cách sử dụng:⏎⏎Nên ăn cháo đậu đen và gạo lứt này thay cơm mỗi ngày, sau 1 tuần bạn có thể giảm tối đa 7kg mà không cần nhịn ăn kham khỗ.

- **[rand]** doc_id `305791` · 1657 ký tự, 13 dòng body · title: `Quên tắm trắng đi, Trương Quỳnh Anh tiết lộ loại mặt nạ tự nhiên 'thần thánh' giúp sở hữu `  
  https://phunusuckhoe.giadinhonline.vn/quen-tam-trang-di-truong-quynh-anh-tiet-lo-loai-mat-na-tu-nhien-than-thanh-giup-so-huu-lan-da-trang-boc-c12a105849.html  
  đầu: Quên tắm trắng đi, Trương Quỳnh Anh tiết lộ loại mặt nạ tự nhiên 'thần thánh' giúp sở hữu làn da trắng bóc⏎⏎Trương Quỳnh Anh có vẻ ngoài xinh xắn, dễ thương với làn da trắng không tì vết. Dù đã có gia đình con nhỏ và khá bận bịu với công việc nhưng cô vẫn dành thời gian để chăm sóc da mỗi ngày. Tuy nhiên, không chỉ chăm chỉ dưỡng da ở các spa đắt tiền, Trương Quỳnh Anh vẫn có n  
  cuối: … Quỳnh Anh chia sẻ thêm kinh nghiệm để có làn da tươi tắn, vóc dáng săn chắc thọn gọn bằng cách tập gym và yoga đều đặn mỗi ngày. Đi kèm với đó là một chế độ ăn hợp lý sẽ giúp cơ thể luôn dẻo dai, chống lão hóa hiệu quả.

- **[rand]** doc_id `293456` · 1913 ký tự, 12 dòng body · title: `Chú ý: Hướng dẫn “từng li từng tí” 4 bước uống vitamin E chuẩn nhất để da luôn đẹp như ý m`  
  https://phunusuckhoe.giadinhonline.vn/chu-y-huong-dan-tung-li-tung-ti-4-buoc-uong-vitamin-e-chuan-nhat-de-da-luon-dep-nhu-y-ma-chang-bao-gio-bi-phan-tac-dung-c12a86383.html  
  đầu: Chú ý: Hướng dẫn “từng li từng tí” 4 bước uống vitamin E chuẩn nhất để da luôn đẹp như ý mà chẳng bao giờ bị phản tác dụng⏎⏎1. Cách uống vitamin E⏎⏎- Uống 1 viên/ngày sau khi ăn sáng 30 phút.⏎⏎- Uống liên tục và đều đặn trong 3 tháng, sau đó nghỉ 2 tháng rồi lại tiếp tục uống.⏎⏎Chỉ cần uống đúng cách, vitamin E sẽ phát huy tối đa tác dụng ngăn ngừa tình trạng lão hóa da, loại b  
  cuối: …c loại rau quả, trái cây chứa hàm lượng vitamin E dồi dào như bưởi, cam, quýt, dưa hấu,… Những loại trái cây này vừa giúp giải nhiệt, vừa tốt cho da lại bồi bổ sức khỏe cực tốt.⏎⏎Chúc các bạn thành công và luôn xinh đẹp!

- **[rand]** doc_id `300497` · 3189 ký tự, 13 dòng body · title: `Măng rất bổ dưỡng và "đưa cơm" những người này, tuyệt đối cấm ăn kẻo mang họa`  
  https://phunusuckhoe.giadinhonline.vn/mang-rat-bo-duong-va-ua-com-nhung-nguoi-nay-tuyet-oi-cam-an-keo-mang-hoa-c12a492235.html  
  đầu: Măng rất bổ dưỡng và "đưa cơm" những người này, tuyệt đối cấm ăn kẻo mang họa⏎⏎Theo quan niệm Đông y, măng là một thực phẩm không chỉ có giá trị dinh dưỡng cao mà còn là một món ăn có giá trị chữa bệnh. Dựa vào giá trị dinh dưỡng của măng sẽ giúp bạn nhận biết mình có nên ăn măng hay không.⏎⏎Các thành phần dinh dưỡng trong măng được xem là vô cùng phong phú, nhờ sự giàu protein  
  cuối: …c trong máu, làm bệnh gout trở nên trầm trọng hơn. Các loại thực phẩm có tốc độ tăng trưởng nhanh như: măng tre, măng trúc, măng tây sẽ làm gia tăng tốc độ tổng hợp acid uric trong cơ thể, vì thế bệnh nhân gút cần tránh.

- **[rand]** doc_id `299292` · 1511 ký tự, 6 dòng body · title: `Lại nổ súng ở Củ Chi, 1 người chết`  
  https://phunusuckhoe.giadinhonline.vn/lai-no-sung-o-cu-chi-1-nguoi-chet-c12a336723.html  
  đầu: Lại nổ súng ở Củ Chi, 1 người chết⏎⏎Sáng 30-1, nguồn tin của Tuổi Trẻ Online xác nhận hơn 0h sáng 30-1, một người đàn ông đi xe máy trên tỉnh lộ 15, thuộc ấp Phú Thuận, xã Phú Hòa Đông, huyện Củ Chi, TP.HCM bị một người dùng súng bắn chết. Sau khi gây án, thủ phạm đã lấy chiếc xe máy của nạn nhân tẩu thoát khỏi hiện trường.⏎⏎Danh tính nạn nhân bước đầu được xác định là anh V.C.  
  cuối: …đường tỉnh lộ 15, ấp Phú Thuận, xã Phú Hoà Đông (H.Củ Chi, TP.HCM) vẫn còn vết máu của nạn nhân bị bắn chết. Nhiều người dân tập trung bàn tán về vụ việc. Lực lượng cảnh sát cơ động được điều động bảo vệ gần hiện trường.

- **[rand]** doc_id `284724` · 2611 ký tự, 28 dòng body · title: `3 công thức mặt nạ từ Vitamin B1 giúp loại bỏ mọi khuyết điểm trên da sau 1 tuần`  
  https://phunusuckhoe.giadinhonline.vn/3-cong-thuc-mat-na-tu-vitamin-b1-giup-loai-bo-moi-khuyet-diem-tren-da-sau-1-tuan-c12a282707.html  
  đầu: 3 công thức mặt nạ từ Vitamin B1 giúp loại bỏ mọi khuyết điểm trên da sau 1 tuần⏎⏎Với 3 công thức mặt nạ dưỡng da bằng Vitamin B1 này, bạn hoàn toàn có thể áp dụng ngay tại nhà. Đặc biệt, nếu kết hợp loại dưỡng chất này với một vài nguyên liệu từ nhiên khác, hiệu quả đạt được sẽ cao hơn rất nhiều.⏎⏎Vitamin B1 có tác dụng gì đối với làn da?⏎⏎Loại dưỡng chất này có khả năng tăng   
  cuối: … 2 - 3 phút. Thư giãn thêm 15 phút nữa thì rửa sạch bằng nước mát.⏎⏎Kiên trì, chăm chỉ áp dụng các công thức mặt nạ dưỡng da từ Vitamin B1 cùng những nguyên liệu tự nhiên trên, hiệu quả sẽ đến với bạn sau thời gian ngắn.

- **[rand]** doc_id `292254` · 4008 ký tự, 23 dòng body · title: `Căn bệnh Thủ tướng Nhật Bản Shinzo Abe mắc phải trước khi từ chức nguy hiểm thế nào?`  
  https://phunusuckhoe.giadinhonline.vn/can-benh-thu-tuong-nhat-ban-shinzo-abe-mac-phai-truoc-khi-tu-chuc-nguy-hiem-the-nao-c12a356184.html  
  đầu: Căn bệnh Thủ tướng Nhật Bản Shinzo Abe mắc phải trước khi từ chức nguy hiểm thế nào?⏎⏎Vào chiều ngày 28/8, Thủ tướng Nhật Bản Shinzo Abe đã tổ chức một cuộc họp báo tại tư dinh của Thủ tướng ở Tokyo và tuyên bố từ chức vì lý do sức khỏe.⏎⏎Phát biểu trong cuộc họp báo, ông Abe cho biết: “Tôi không muốn bệnh tật dẫn tới những sai lầm trong các quyết sách quan trọng. Tôi không   
  cuối: …củ, quả).⏎⏎- Thực hiện sổ giun định kỳ 6 tháng/lần.⏎⏎- Hỗ trợ hệ tiêu hóa bằng các men tiêu hóa sống.⏎⏎- Khi có những dấu hiệu của bệnh viêm đại tràng cần đi khám sớm, để được chẩn đoán và có biện pháp điều trị kịp thời.

## laodong.vn (27,184 doc)

- **[short]** doc_id `143259` · 329 ký tự, 7 dòng body · title: `3 thực phẩm giúp hạn chế đường huyết tăng không kiểm soát`  
  https://laodong.vn/ban-tin/3-thuc-pham-giup-han-che-duong-huyet-tang-khong-kiem-soat-1403160.ldo  
  đầu: 3 thực phẩm giúp hạn chế đường huyết tăng không kiểm soát⏎⏎3 thực phẩm giúp hạn chế đường huyết tăng không kiểm soát⏎⏎Thứ sáu, 04/10/2024 09:07 (GMT+7)⏎⏎Để kiểm soát đường huyết, bạn nên bổ sung 3 loại thực phẩm sau vào chế độ ăn uống.⏎⏎Bình luận:⏎⏎Bạn nghĩ gì về nội dung này?⏎⏎Bạn cần đăng nhập để gửi bình luận.⏎⏎Gửi bình luận  
  cuối: …ểm soát⏎⏎Thứ sáu, 04/10/2024 09:07 (GMT+7)⏎⏎Để kiểm soát đường huyết, bạn nên bổ sung 3 loại thực phẩm sau vào chế độ ăn uống.⏎⏎Bình luận:⏎⏎Bạn nghĩ gì về nội dung này?⏎⏎Bạn cần đăng nhập để gửi bình luận.⏎⏎Gửi bình luận

- **[short]** doc_id `171661` · 303 ký tự, 7 dòng body · title: `Giải độc gan tự nhiên nhờ món canh ngọt lành`  
  https://laodong.vn/video/giai-doc-gan-tu-nhien-nho-mon-canh-ngot-lanh-1591780.ldo  
  đầu: Giải độc gan tự nhiên nhờ món canh ngọt lành⏎⏎Giải độc gan tự nhiên nhờ món canh ngọt lành⏎⏎Thứ tư, 15/10/2025 15:23 (GMT+7)⏎⏎Canh đậu xanh nấu rau củ có thể hỗ trợ giải độc gan hiệu quả nếu sử dụng đúng cách.⏎⏎Bình luận:⏎⏎Bạn nghĩ gì về nội dung này?⏎⏎Bạn cần đăng nhập để gửi bình luận.⏎⏎Gửi bình luận  
  cuối: …ọt lành⏎⏎Thứ tư, 15/10/2025 15:23 (GMT+7)⏎⏎Canh đậu xanh nấu rau củ có thể hỗ trợ giải độc gan hiệu quả nếu sử dụng đúng cách.⏎⏎Bình luận:⏎⏎Bạn nghĩ gì về nội dung này?⏎⏎Bạn cần đăng nhập để gửi bình luận.⏎⏎Gửi bình luận

- **[short]** doc_id `171419` · 291 ký tự, 1 dòng body · title: `Các nữ sinh cần làm gì khi bị kẻ xấu tấn công`  
  https://laodong.vn/video/cac-nu-sinh-can-lam-gi-khi-bi-ke-xau-tan-cong-849829.ldo  
  đầu: Các nữ sinh cần làm gì khi bị kẻ xấu tấn công⏎⏎Sau hơn 4 ngày mất tích, đến ngày 27.10, thi thể nữ sinh Học viện Ngân hàng bị sát hại, cướp tài sản đã được tìm thấy dưới lòng sông Nhuệ. Để những trường hợp đáng tiếc như vậy không xảy ra, các chuyên gia đã có những lời khuyên với các bạn nữ.  
  cuối: … đến ngày 27.10, thi thể nữ sinh Học viện Ngân hàng bị sát hại, cướp tài sản đã được tìm thấy dưới lòng sông Nhuệ. Để những trường hợp đáng tiếc như vậy không xảy ra, các chuyên gia đã có những lời khuyên với các bạn nữ.

- **[short]** doc_id `144463` · 320 ký tự, 1 dòng body · title: `Hàng trăm người ở Quảng Bình phải đi tiêm phòng dại sau khi bị chó mèo cắn`  
  https://laodong.vn/ban-tin/hang-tram-nguoi-o-quang-binh-phai-di-tiem-phong-dai-sau-khi-bi-cho-meo-can-1320663.ldo  
  đầu: Hàng trăm người ở Quảng Bình phải đi tiêm phòng dại sau khi bị chó mèo cắn⏎⏎Quảng Bính - Từ đầu năm đến nay, tại tỉnh Quảng Bình có 444 trường hợp bị chó, mèo cắn phải đến các cơ sở y tế để tiêm huyết thanh phòng dại. Trước thực trạng trên, ngành chức năng tỉnh này đang triển khai các biện pháp để phòng tránh bệnh dại.  
  cuối: …đến nay, tại tỉnh Quảng Bình có 444 trường hợp bị chó, mèo cắn phải đến các cơ sở y tế để tiêm huyết thanh phòng dại. Trước thực trạng trên, ngành chức năng tỉnh này đang triển khai các biện pháp để phòng tránh bệnh dại.

- **[short]** doc_id `172479` · 271 ký tự, 1 dòng body · title: `Vào cấp cứu ngày cận Tết, bệnh nhân mong sớm hồi phục về với gia đình`  
  https://laodong.vn/video/vao-cap-cuu-ngay-can-tet-benh-nhan-mong-som-hoi-phuc-ve-voi-gia-dinh-1302184.ldo  
  đầu: Vào cấp cứu ngày cận Tết, bệnh nhân mong sớm hồi phục về với gia đình⏎⏎Cần Thơ - Không may gặp vấn đề sức khỏe phải nhập viện điều trị khi Tết đã cận kề, nằm ở phòng cấp cứu bệnh nhân kì vọng vào sự chăm sóc của bác sĩ, mong sớm bình phục để về đón Tết cùng với gia đình.  
  cuối: …ục về với gia đình⏎⏎Cần Thơ - Không may gặp vấn đề sức khỏe phải nhập viện điều trị khi Tết đã cận kề, nằm ở phòng cấp cứu bệnh nhân kì vọng vào sự chăm sóc của bác sĩ, mong sớm bình phục để về đón Tết cùng với gia đình.

- **[nav]** doc_id `151789` · 2016 ký tự, 53 dòng body · title: `Chuẩn bị bữa sáng giảm cân chỉ trong ít phút`  
  https://laodong.vn/lam-dep/chuan-bi-bua-sang-giam-can-chi-trong-it-phut-1206234.ldo  
  đầu: Chuẩn bị bữa sáng giảm cân chỉ trong ít phút⏎⏎1. Bánh mì nguyên cám với phô mai và đào⏎⏎Chuẩn bị:⏎⏎1 lát bánh mì nguyên cám⏎⏎50 - 60g phô mai⏎⏎1 quả đào tươi⏎⏎Cà phê hoặc trà⏎⏎Thực hiện: Phết phô mai lên bánh mì nguyên cám, trang trí lên trên một ít đào tươi là có thể thưởng thức. Thưởng thức món ăn này cùng cà phê hoặc trà.⏎⏎2. Trứng chiên rau củ⏎⏎1 quả trứng gà⏎⏎Rau củ tùy th  
  cuối: …ỡ lớn⏎⏎1 thìa cà phê bơ⏎⏎Một ít tiêu và muối⏎⏎Thực hiện:⏎⏎Đập trứng vào bát, đánh tan, trộn đều với muối và hạt tiêu⏎⏎Đun nóng chảo, cho bơ vào chảo, ngoáy đến lúc bơ tan.⏎⏎Cho trứng vào chiên chín là có thể thưởng thức.

- **[nav]** doc_id `153553` · 1103 ký tự, 32 dòng body · title: `3 món salad giàu protein tự làm để tăng cường cơ bắp và giảm cân`  
  https://laodong.vn/suc-khoe/3-mon-salad-giau-protein-tu-lam-de-tang-cuong-co-bap-va-giam-can-1241516.ldo  
  đầu: 3 món salad giàu protein tự làm để tăng cường cơ bắp và giảm cân⏎⏎Salad xanh với bánh mì⏎⏎Nguyên liệu:⏎⏎- 100 gram rau trộn salad⏎⏎- 1/2 quả dưa chuột thái lát⏎⏎- 1/2 củ cà rốt bào sợi⏎⏎- 2 thìa cà phê dầu ô liu nguyên chất⏎⏎- 2 thìa cà phê giấm, muối và tiêu xay (tuỳ khẩu vị)⏎⏎- Bánh mì làm từ lúa mì nguyên hạt (nướng)⏎⏎Cách làm:⏎⏎Cho tất cả rau xanh, dưa chuột và cà rốt vào t  
  cuối: …am các loại củ nướng⏎⏎- 50 gram đậu phụ chiên⏎⏎- 2 thìa cà phê hạt bí⏎⏎- 4 thìa cà phê dầu giấm⏎⏎Cách làm:⏎⏎Cho tất cả các loại rau xanh, rau củ, đậu phụ và hạt bí vào tô. Thêm dầu giấm lên món salad ngay trước khi dùng.

- **[nav]** doc_id `154470` · 2166 ký tự, 53 dòng body · title: `5 cách sử dụng nước ép mận trị táo bón nhanh nhóng và hiệu quả`  
  https://laodong.vn/suc-khoe/5-cach-su-dung-nuoc-ep-man-tri-tao-bon-nhanh-nhong-va-hieu-qua-1093303.ldo  
  đầu: 5 cách sử dụng nước ép mận trị táo bón nhanh nhóng và hiệu quả⏎⏎Nước ép mận và sữa Magie⏎⏎Nguyên liệu:⏎⏎1 cốc nước ép mận⏎⏎2 thìa sữa magie⏎⏎Cách làm:⏎⏎Thêm sữa magie vào nước ép mận và cho vào lò vi sóng trong khoảng 30-45 giây.⏎⏎Uống khi còn ấm.⏎⏎Phương thuốc này sẽ giúp bạn giảm đau sau 1-3 giờ.⏎⏎Công dụng:⏎⏎Sữa magie có tên gọi hóa học là magie hydroxit, là một thành phần p  
  cuối: …o bằng lò vi sóng trong 45-60 phút.⏎⏎Trộn cà phê hòa tan với nước nóng và thêm thứ này vào nước ép mận.⏎⏎Uống khi còn nóng.⏎⏎Công dụng:⏎⏎Loại nước uống này giúp thúc đẩy chuyển động của phân và giúp bạn đại tiện dễ dàng.

- **[nav]** doc_id `148817` · 1091 ký tự, 16 dòng body · title: `Cách làm kem chuối mít đơn giản, thơm ngon khó cưỡng cho ngày hè nắng nóng`  
  https://laodong.vn/dinh-duong-am-thuc/cach-lam-kem-chuoi-mit-don-gian-thom-ngon-kho-cuong-cho-ngay-he-nang-nong-907928.ldo  
  đầu: Cách làm kem chuối mít đơn giản, thơm ngon khó cưỡng cho ngày hè nắng nóng⏎⏎Mùa hè nóng bức mà được thưởng thức que kem chuối mít thơm ngậy của sữa, bùi bùi của lạc rang và thanh mát của dừa sợi thật vô cùng hấp dẫn. Dưới đây là cách làm kem chuối mít đơn giản nhất do chị Nguyễn Hà chia sẻ trên cộng đồng "Cùng nhau học nấu ăn ngon".⏎⏎Nguyên liệu:⏎⏎3-4 trái chuối chín⏎⏎3-4 múi m  
  cuối: … lớp cuối cùng là kem. Sau đó, rắc đậu phụng và dừa non đầy mặt. Cho khay kem chuối-mít vào ngăn đá tủ lạnh ít nhất 5 tiếng là có thể ăn ngon.⏎⏎Trình bày: cắt kem mít ra từng miếng vừa ăn, khi ăn các bạn dùng dĩa để cầm.

- **[nav]** doc_id `158366` · 1427 ký tự, 27 dòng body · title: `Cách làm bánh sandwich rau nướng giảm cân cho bữa sáng`  
  https://laodong.vn/suc-khoe/cach-lam-banh-sandwich-rau-nuong-giam-can-cho-bua-sang-1406794.ldo  
  đầu: Cách làm bánh sandwich rau nướng giảm cân cho bữa sáng⏎⏎Nguyên liệu:⏎⏎Với phần sốt hành tây ăn kèm bao gồm:⏎⏎- 90ml sữa đông⏎⏎- 1 tép tỏi⏎⏎- 10gram rau mùi thái nhỏ⏎⏎- 1 củ hành tây nhỏ, thái nhỏ⏎⏎- 2 quả ớt xanh⏎⏎- 10ml nước cốt chanh⏎⏎- Muối vừa ăn⏎⏎- Bột ớt đỏ vừa ăn⏎⏎- Bột oregano vừa ăn⏎⏎Với phần nhân và bánh mì sandwich bao gồm:⏎⏎- 2 lát bánh mì ngũ cốc⏎⏎- 1/2 củ hành tây  
  cuối: …àm này, mỗi bánh sandwich sẽ không chứa quá 200 calo và giàu chất béo tốt từ sữa đông. Cùng với đó, món ăn này cũng rất giàu chất xơ, vitamin và carbohydrate và protein từ rau, bánh mì ngũ cốc và trứng luộc hoặc thịt gà.

- **[rand]** doc_id `149587` · 1175 ký tự, 7 dòng body · title: `Lợi ích của tempeh trong việc kéo dài tuổi thọ`  
  https://laodong.vn/dinh-duong-am-thuc/loi-ich-cua-tempeh-trong-viec-keo-dai-tuoi-tho-1411136.ldo  
  đầu: Lợi ích của tempeh trong việc kéo dài tuổi thọ⏎⏎Tempeh (tương nén, thực phẩm có nguồn gốc từ Indonesia, thường được sử dụng trong chế độ ăn chay) được làm từ đậu nành rửa sạch, hấp chín, để nguội và lên men (được tạo ra bằng quá trình nuôi cấy và lên men có kiểm soát).⏎⏎Tempeh chứa hàm lượng vitamin B1 và vitamin B2 dồi dào.⏎⏎Chất isoflavone đậu nành trong tempeh có vai trò qua  
  cuối: … lượng lipoprotein mật độ thấp trong cơ thể, đồng thời tăng hàm lượng lipoprotein mật độ cao.⏎⏎Do đó, tempeh là thực phẩm ngoài tác dụng làm gia vị còn có tác dụng chữa bệnh và phòng ngừa bệnh tật, giúp kéo dài tuổi thọ.

- **[rand]** doc_id `156222` · 1964 ký tự, 11 dòng body · title: `Axit uric cao nên sử dụng mật ong như thế nào để an toàn với sức khỏe?`  
  https://laodong.vn/suc-khoe/axit-uric-cao-nen-su-dung-mat-ong-nhu-the-nao-de-an-toan-voi-suc-khoe-1344903.ldo  
  đầu: Axit uric cao nên sử dụng mật ong như thế nào để an toàn với sức khỏe?⏎⏎Axit uric cao có nên dùng mật ong?⏎⏎Từ lâu, mật ong là loại thực phẩm quen thuộc. Mật ong có vị ngọt, hương thơm và được chế biến thành nhiều loại nước uống khác nhau hoặc làm gia vị trong các món ăn.⏎⏎Trong thành phần của mật ong có chứa carbohydrate (82%), đường glucose, đường fructose, đường sucrose và n  
  cuối: …c. Đem lá bạc hà và lá trà xanh đun cùng nước sôi trong khoảng 10 phút, sau khi dung dịch nguội bớt cho thêm mật ong, nước chanh vào khuấy đều và sử dụng. Người bệnh nên sử dụng mỗi ngày để cảm nhận được hiệu quả rõ rệt.

- **[rand]** doc_id `167307` · 1238 ký tự, 6 dòng body · title: `Tai nạn sinh hoạt tăng hơn 100% trong 9 ngày Tết tại Bệnh viện Chợ Rẫy`  
  https://laodong.vn/suc-khoe/tai-nan-sinh-hoat-tang-hon-100-trong-9-ngay-tet-tai-benh-vien-cho-ray-1659052.ldo  
  đầu: Tai nạn sinh hoạt tăng hơn 100% trong 9 ngày Tết tại Bệnh viện Chợ Rẫy⏎⏎Từ ngày 14 - 22.2.2026, Bệnh viện Chợ Rẫy trung bình mỗi ngày tiếp nhận 273 bệnh nhân vào cấp cứu, dao động từ 200 đến 360 lượt, tăng 3,4% so với Tết Ất Tỵ 2025.⏎⏎Đáng chú ý, số ca tai nạn giao thông lên tới 429 trường hợp. Tai nạn sinh hoạt tăng mạnh nhất, với 171 ca, trung bình 19 ca/ngày, tăng tới 108,5%  
  cuối: …khi nhiều gia đình sum vầy đón xuân, đội ngũ y bác sĩ tại Bệnh viện Chợ Rẫy vẫn làm việc liên tục để đảm bảo công tác cấp cứu, điều trị được duy trì thông suốt, kịp thời cứu chữa người bệnh trong những ngày cao điểm Tết.

- **[rand]** doc_id `153906` · 1692 ký tự, 11 dòng body · title: `4 điều gây hại cho gan ở nam giới tuổi trung niên`  
  https://laodong.vn/suc-khoe/4-dieu-gay-hai-cho-gan-o-nam-gioi-tuoi-trung-nien-1495884.ldo  
  đầu: 4 điều gây hại cho gan ở nam giới tuổi trung niên⏎⏎Uống thuốc bừa bãi⏎⏎Nhiều người tự ý uống thuốc để giải quyết những bệnh lý nhẹ. Trên thực tế, nhiều loại thuốc chỉ có thể điều trị triệu chứng chứ không thể phục hồi sức khỏe. Hơn nữa, quá trình chuyển hóa thuốc có liên quan chặt chẽ đến gan. Việc sử dụng thuốc bừa bãi hoặc mù quáng sẽ gây hại trực tiếp đến gan và làm suy giảm  
  cuối: …gủ (khoảng từ 23h đến 2h).⏎⏎Thức khuya thường xuyên sẽ làm ảnh hưởng đến quá trình phục hồi và điều chỉnh của gan. Nếu gan không được phục hồi và điều chỉnh trong thời gian dài, gan sẽ yếu đi, khả năng làm việc giảm sút.

- **[rand]** doc_id `156298` · 2109 ký tự, 9 dòng body · title: `Bác sĩ chẩn đoán viêm thực quản trào ngược, thủ phạm lại là dị vật`  
  https://laodong.vn/suc-khoe/bac-si-chan-doan-viem-thuc-quan-trao-nguoc-thu-pham-lai-la-di-vat-825597.ldo  
  đầu: Bác sĩ chẩn đoán viêm thực quản trào ngược, thủ phạm lại là dị vật⏎⏎ThS.BS Đỗ Trọng Khiếu - Phó trưởng khoa Nội soi Tiêu hóa (Bệnh viện Hữu nghị Việt Đức, Hà Nội) cho biết: Các bác sĩ Bệnh viện Hữu nghị Việt Đức đã tiếp nhận nam bệnh nhân N.X.D, 29 tuổi, trú tại Kinh Môn (Hải Dương) với chẩn đoán: Dị vật đường tiêu hóa. Kết quả chụp Xquang cho thấy hình ảnh dị vật nằm ở vị trí   
  cuối: …ĩ khuyến cáo, khi bị dị vật người bệnh cần được chẩn đoán kịp thời, đúng cách và chuyển ngay đến các cơ sở uy tín để được khám và kiểm tra, lấy bỏ dị vật, điều trị các thương tổn phối hợp như thủng thực quản, thủng ruột.

- **[rand]** doc_id `156537` · 1788 ký tự, 19 dòng body · title: `Bài tập yoga đơn giản giúp bạn đánh bay mỡ bụng hiệu quả`  
  https://laodong.vn/suc-khoe/bai-tap-yoga-don-gian-giup-ban-danh-bay-mo-bung-hieu-qua-1322509.ldo  
  đầu: Bài tập yoga đơn giản giúp bạn đánh bay mỡ bụng hiệu quả⏎⏎Dưới đây là một số bài tập yoga giúp bạn giảm mỡ bụng hiệu quả và cũng rất đơn giản.⏎⏎Tư thế con thuyền⏎⏎Động tác này tác động đến cơ bụng của bạn để duy trì sự ổn định đồng thời làm săn chắc vòng 2 của bạn.⏎⏎Ngồi thẳng lưng với hai chân duỗi thẳng trước mặt, hai tay đặt hai bên cơ thể và lòng bàn tay chạm sàn, các ngón   
  cuối: …nh tay lên trên đầu, song song với nhau.⏎⏎Khi thở ra, uốn người xuống như ngồi ghế, uốn gối và hạ hông xuống như khi ngồi ghế thật.⏎⏎Giữ tư thế này trong khoảng 30 giây đến 1 phút, sau đó thở vào và từ từ đứng thẳng lên.

- **[rand]** doc_id `164818` · 8503 ký tự, 13 dòng body · title: `Nhồi máu cơ tim đang "trẻ hóa" - chuyện không nhỏ`  
  https://laodong.vn/suc-khoe/nhoi-mau-co-tim-dang-tre-hoa-chuyen-khong-nho-601522.ldo  
  đầu: Nhồi máu cơ tim đang "trẻ hóa" - chuyện không nhỏ⏎⏎Trưa ngày 28.3, Cơ trưởng L.K.H, 35 tuổi, bị ngất ngay khi máy bay Boeing 737 - 800, chuyến bay AE1858 đang chờ lệnh ra đường băng để cất cánh. Đội y tế cảng hàng không Tân Sơn Nhất nhanh chóng sơ cứu và đưa phi công đến BV 175. BV thông báo nạn nhân đã ngừng tim, ngừng thở trước khi đến viện, nguyên nhân do nhồi máu cơ tim (NM  
  cuối: …lycerine khi có cơn đau thắt ngực. Sau 5 phút, cơn đau, khó thở không giảm hoặc mất có thể dùng thuốc lần 2. Nếu có Aspirin được kê đơn thì uống 1 viên để phòng huyết khối và nhanh chóng đến BV, không nên để quá 15 phút.

- **[rand]** doc_id `164122` · 3409 ký tự, 18 dòng body · title: `Nghiện 'sát thủ số 1', người đàn ông bất ngờ phát hiện bệnh cực nặng`  
  https://laodong.vn/suc-khoe/nghien-sat-thu-so-1-nguoi-dan-ong-bat-ngo-phat-hien-benh-cuc-nang-740638.ldo  
  đầu: Nghiện 'sát thủ số 1', người đàn ông bất ngờ phát hiện bệnh cực nặng⏎⏎GS Mai Trọng Khoa, nguyên Giám đốc TT Y học hạt nhân và ung ướu, BV Bạch Mai cho biết, bệnh nhân Đặng Văn Chính, 67 tuổi là một trong những trường hợp mắc ung thư phổi giai đoạn muộn được điều trị thành công.⏎⏎Ông Chính đến viện khám vào tháng 11/2017 vì đau ngực phải âm ỉ và ho có đờm trắng trong suốt 1 thán  
  cuối: …y nhiên, ngày nay với nhiều tiến bộ trong sàng lọc, chẩn đoán cũng như điều trị đã kéo dài cũng như nâng cao chất lượng sống cho người bệnh đặc biệt là những bệnh nhân ở giai đoạn muộn.⏎⏎* Tên bệnh nhân đã được thay đổi.

- **[rand]** doc_id `157734` · 2627 ký tự, 16 dòng body · title: `Bộ Y tế trả lời quyền bổ nhiệm Giám đốc, Phó Giám đốc trạm y tế từ 1.1.2026`  
  https://laodong.vn/suc-khoe/bo-y-te-tra-loi-quyen-bo-nhiem-giam-doc-pho-giam-doc-tram-y-te-tu-112026-1623451.ldo  
  đầu: Bộ Y tế trả lời quyền bổ nhiệm Giám đốc, Phó Giám đốc trạm y tế từ 1.1.2026⏎⏎Bộ trưởng Bộ Y tế ban hành Thông tư số 43/2025/TT-BYT hướng dẫn chức năng, nhiệm vụ, quyền hạn và cơ cấu tổ chức của Trạm Y tế xã, phường, đặc khu thuộc tỉnh, thành phố trực thuộc Trung ương.⏎⏎Trong đó, về cơ cấu tổ chức, lãnh đạo trạm y tế cấp xã gồm: Giám đốc và Phó Giám đốc. Số lượng Phó Giám đốc th  
  cuối: … tế cấp xã.⏎⏎+ Các trạm y tế cấp xã được áp dụng quy định tại khoản 8 Điều 69 Nghị định sổ 188/2025/NĐ-CP ngày 01 tháng 7 năm 2025 của Chính phủ quy định chỉ tiết và hướng dẫn thi hành một số điều của Luật Bảo hiểm y tế.

- **[rand]** doc_id `164673` · 2016 ký tự, 6 dòng body · title: `Nhận bồi thường bảo hiểm trong vòng 24h với FWD`  
  https://laodong.vn/suc-khoe/nhan-boi-thuong-bao-hiem-trong-vong-24h-voi-fwd-838876.ldo  
  đầu: Nhận bồi thường bảo hiểm trong vòng 24h với FWD⏎⏎Với thế mạnh công nghệ vượt trội, FWD đã rút ngắn thời gian xử lý hồ sơ yêu cầu và có thể chi trả quyền lợi bảo hiểm chỉ trong vòng 24 tiếng, nhanh chóng và thuận tiện cho khách hàng hơn bao giờ hết.⏎⏎Dịch vụ 24h E-Claim được áp dụng cho toàn bộ các sản phẩm có quyền lợi trợ cấp viện phí và chi phí phẫu thuật của FWD. Với dịch vụ  
  cuối: … thị trường hàng đầu thế giới là Forrester. Rõ ràng, những dịch vụ như 24h E-Claim đã đem lại những quyền lợi thiết thực cho khách hàng, đồng thời thể hiện rõ tầm nhìn thay đổi cảm nhận của mọi người về bảo hiểm của FWD.

## vinmec.com (26,124 doc)

- **[short]** doc_id `754673` · 418 ký tự, 3 dòng body · title: `Trắc nghiệm: Vitamin thì sao? Bạn hiểu nó thế nào?`  
  https://www.vinmec.com/vie/bai-viet/trac-nghiem-vitamin-thi-sao-ban-hieu-no-nao-vi  
  đầu: Trắc nghiệm: Vitamin thì sao? Bạn hiểu nó thế nào?⏎⏎Vitamin là một thành phần có vai trò quan trọng đối với sức khỏe con người nhưng không phải ai cũng có hiểu biết đầy đủ về nó. Bạn thực sự hiểu được bao nhiêu về vitamin?⏎⏎Vitamin là một thành phần có vai trò quan trọng đối với sức khỏe con người nhưng không phải ai cũng có hiểu biết đầy đủ về nó. Bạn thực sự hiểu được bao nhi  
  cuối: …ợc bao nhiêu về vitamin?⏎⏎Vitamin là một thành phần có vai trò quan trọng đối với sức khỏe con người nhưng không phải ai cũng có hiểu biết đầy đủ về nó. Bạn thực sự hiểu được bao nhiêu về vitamin?⏎⏎Bài dịch từ: webmd.com

- **[short]** doc_id `754425` · 442 ký tự, 3 dòng body · title: `Trắc nghiệm: Bạn biết gì về sức khỏe trẻ em?`  
  https://www.vinmec.com/vie/bai-viet/trac-nghiem-ban-biet-gi-ve-suc-khoe-tre-em-vi  
  đầu: Trắc nghiệm: Bạn biết gì về sức khỏe trẻ em?⏎⏎Sức khỏe trẻ em là vấn đề cần được quan tâm để trẻ phát triển đầy đủ về thể chất và tinh thần. Trả lời các câu trắc nghiệm dưới đây xem bạn hiểu như thế nào về chăm sóc sức khỏe trẻ em.⏎⏎Sức khỏe trẻ em là vấn đề cần được quan tâm để trẻ phát triển đầy đủ về thể chất và tinh thần. Trả lời các câu trắc nghiệm dưới đây xem bạn hiểu nh  
  cuối: …e trẻ em.⏎⏎Sức khỏe trẻ em là vấn đề cần được quan tâm để trẻ phát triển đầy đủ về thể chất và tinh thần. Trả lời các câu trắc nghiệm dưới đây xem bạn hiểu như thế nào về chăm sóc sức khỏe trẻ em.⏎⏎Bài dịch từ: webmd.com

- **[short]** doc_id `754636` · 430 ký tự, 3 dòng body · title: `Trắc nghiệm: Thử hiểu biết của bạn về kem chống nắng`  
  https://www.vinmec.com/vie/bai-viet/trac-nghiem-thu-hieu-biet-cua-ban-ve-kem-chong-nang-vi  
  đầu: Trắc nghiệm: Thử hiểu biết của bạn về kem chống nắng⏎⏎Sử dụng kem chống nắng sẽ giúp loại bỏ các nguy cơ da bị cháy nắng, thâm sạm... Hãy cùng trả lời một số câu hỏi dưới đây để đánh giá mức độ hiểu về kem chống nắng của bạn nhé!⏎⏎Sử dụng kem chống nắng sẽ giúp loại bỏ các nguy cơ da bị cháy nắng, thâm sạm... Hãy cùng trả lời một số câu hỏi dưới đây để đánh giá mức độ hiểu về k  
  cuối: …g nắng của bạn nhé!⏎⏎Sử dụng kem chống nắng sẽ giúp loại bỏ các nguy cơ da bị cháy nắng, thâm sạm... Hãy cùng trả lời một số câu hỏi dưới đây để đánh giá mức độ hiểu về kem chống nắng của bạn nhé!⏎⏎Bài dịch từ: webmd.com

- **[short]** doc_id `758927` · 437 ký tự, 4 dòng body · title: `Xét nghiệm men gan`  
  https://www.vinmec.com/vie/bai-viet/xet-nghiem-men-gan-vi  
  đầu: Xét nghiệm men gan⏎⏎Đo hoạt độ ALT máu của tôi là 91,8 có phải cao quá không thưa bác sĩ?⏎⏎Vũ Mạnh Hà (1982)⏎⏎Hiện tại men gan của bạn tăng gần 02 lần so với bình thường, bạn cần điều chỉnh chế độ ăn, không uống bia, rượu, sau 01 tháng nếu xét nghiệm men gan còn tăng có thể đến khám chuyên khoa gan – mật- truyền nhiễm tại Vinmec.⏎⏎Trưởng khoa Khám bệnh - Nội khoa, Khoa Khám bện  
  cuối: …rượu, sau 01 tháng nếu xét nghiệm men gan còn tăng có thể đến khám chuyên khoa gan – mật- truyền nhiễm tại Vinmec.⏎⏎Trưởng khoa Khám bệnh - Nội khoa, Khoa Khám bệnh & Nội khoa - Bệnh viện Đa khoa Quốc tế Vinmec Hải Phòng

- **[short]** doc_id `754545` · 477 ký tự, 3 dòng body · title: `Trắc nghiệm: Hydrat hóa là gì bạn biết không?`  
  https://www.vinmec.com/vie/bai-viet/trac-nghiem-hydrat-hoa-la-gi-ban-biet-khong-vi  
  đầu: Trắc nghiệm: Hydrat hóa là gì bạn biết không?⏎⏎Hydrat hóa là quá trình bổ sung các phân tử nước vào các hợp chất hữu cơ. Uống đủ nước sẽ tạo điều kiện cho các phản ứng hydrat hóa được diễn ra nhiều hơn nhằm bổ sung nước cho các cơ quan trong cơ thể.⏎⏎Hydrat hóa là quá trình bổ sung các phân tử nước vào các hợp chất hữu cơ. Uống đủ nước sẽ tạo điều kiện cho các phản ứng hydrat h  
  cuối: … hóa là quá trình bổ sung các phân tử nước vào các hợp chất hữu cơ. Uống đủ nước sẽ tạo điều kiện cho các phản ứng hydrat hóa được diễn ra nhiều hơn nhằm bổ sung nước cho các cơ quan trong cơ thể.⏎⏎Bài dịch từ: webmd.com

- **[title]** doc_id `740502` · 4498 ký tự, 36 dòng body · title: `Đái máu`  
  https://www.vinmec.com/vie/bai-viet/dai-mau-vi  
  đầu: Đái máu⏎⏎Bài viết bởi Bác sĩ Trần Hải Hà - Khoa Cấp cứu - Bệnh viện Đa khoa quốc tế Vinmec Times City⏎⏎Nhìn thấy máu trong nước tiểu của bạn có thể đáng báo động. Trong nhiều trường hợp, nguyên nhân có thể vô hại, nhưng máu trong nước tiểu (tiểu máu) có thể chỉ ra một rối loạn nghiêm trọng.⏎⏎I. Triệu chứng⏎⏎Đái máu đại thể tạo ra nước tiểu màu hồng, đỏ hoặc màu cola do sự hiện   
  cuối: …ười bệnh sự thoải mái, thân thiện, yên tâm.⏎⏎Nếu có nhu cầu tư vấn và thăm khám tại các Bệnh viện và phòng khám thuộc hệ thống Y tế Vinmec trên toàn quốc, Quý khách hàng vui lòng liên hệ Website để được phục vụ tốt nhất.

- **[title]** doc_id `752823` · 2441 ký tự, 29 dòng body · title: `TCI là gì?`  
  https://www.vinmec.com/vie/bai-viet/tci-la-gi-vi  
  đầu: TCI là gì?⏎⏎Phương pháp gây mê kiểm soát nồng độ đích (TCI) được áp dụng tại Việt Nam từ năm 2001 nhờ vào bơm tiêm tiện chuyên biệt. Thuốc mê thường sử dụng trong gây mê tĩnh mạch TCI là Propofol. Ngày nay, phương pháp này được sử dụng rộng rãi vì độ an toàn cao cũng như giúp bác sĩ dự đoán được thời gian tỉnh lại của bệnh nhân.⏎⏎1. TCI là gì?⏎⏎TCI là phương pháp gây mê kiểm so  
  cuối: … tục thấp.⏎⏎Dùng atropin nếu mạch chậm.⏎⏎3.4 Đảm bảo đủ thuốc giãn cơ⏎⏎Propofol có tính chất giãn cơ yếu hơn so với các thuốc mê thể khí. Như vậy, liều thuốc giãn cơ có thể cao hơn khi sử dụng propofol để duy trì gây mê.

- **[title]** doc_id `741640` · 5701 ký tự, 59 dòng body · title: `Đau vú`  
  https://www.vinmec.com/vie/bai-viet/dau-vu-vi  
  đầu: Đau vú⏎⏎Đau vú là tình trạng thường gặp ở phụ nữ khi đến khám tại phòng khám bệnh lý tuyến vú, dùng để mô tả các tình trạng như cảm giác căng, nóng rát, siết chặt... trong mô tuyến vú. Cơn đau vú có thể liên tục hoặc thỉnh thoảng xảy ra.⏎⏎1. Triệu chứng đau vú⏎⏎Khoảng 70% phụ nữ có cơn đau vú ở 1 vài thời điểm trong đời, nhưng chỉ có khoảng 15% trường hợp cần điều trị thuốc.⏎⏎Đ  
  cuối: …h toàn diện, chuyên nghiệp; không gian khám chữa bệnh văn minh, lịch sự, an toàn và tiệt trùng tối đa. Khách hàng khi chọn thực hiện các xét nghiệm tại đây có thể hoàn toàn yên tâm về độ chính xác của kết quả xét nghiệm.

- **[title]** doc_id `741422` · 6807 ký tự, 57 dòng body · title: `Đau ngực`  
  https://www.vinmec.com/vie/bai-viet/dau-nguc-vi  
  đầu: Đau ngực⏎⏎Đau ngực là dấu hiệu của rất nhiều bệnh lý từ thành ngực, phổi, màng phổi, dạ dày đến các vấn đề nghiêm trọng của bệnh lý tim mạch... Do đó, việc tìm hiểu về triệu chứng này có khả năng giúp bạn sớm có biện pháp đối phó ngay từ đầu, đồng thời hạn chế phát sinh biến chứng nguy hiểm⏎⏎1. Các nguyên nhân gây đau ngực⏎⏎Triệu chứng đau ngực liên quan đến nhiều vấn đề sức kh  
  cuối: …quả khám sức khỏe tổng quát, nếu phát hiện các bệnh lý cần khám và điều trị chuyên sâu, Quý khách có thể sử dụng dịch vụ từ các chuyên khoa khác ngay tại Bệnh viện với chất lượng điều trị và dịch vụ khách hàng vượt trội.

- **[nav]** doc_id `741870` · 4284 ký tự, 65 dòng body · title: `Dị ứng bạc hà: Nguyên nhân, triệu chứng và hơn thế nữa`  
  https://www.vinmec.com/vie/bai-viet/di-ung-bac-ha-nguyen-nhan-trieu-chung-va-hon-the-nua  
  đầu: Dị ứng bạc hà: Nguyên nhân, triệu chứng và hơn thế nữa⏎⏎Dị ứng bạc hà là tình trạng hiếm gặp nhưng có thể gây ra nhiều phản ứng khó chịu như ngứa, nổi mẩn, khó thở hoặc sưng tấy, đặc biệt khi tiếp xúc trực tiếp hoặc tiêu thụ các sản phẩm chứa bạc hà. Bạc hà thường được sử dụng rộng rãi trong thực phẩm, mỹ phẩm và dược phẩm, khiến người bị dị ứng dễ gặp phải các triệu chứng bất   
  cuối: …m chứa bạc hà cần tránh. Các sản phẩm khác có chứa bạc hà bao gồm:⏎⏎Thuốc lá⏎⏎Kem mát-xa giảm đau cơ bắp⏎⏎Gel làm mát da bị cháy nắng⏎⏎Son dưỡng môi⏎⏎Sữa dưỡng thể⏎⏎Thuốc trị viêm họng⏎⏎Kem chân bạc hà⏎⏎Nước hoa⏎⏎Dầu gội

- **[nav]** doc_id `733823` · 6031 ký tự, 132 dòng body · title: `Bài thuốc đông y chữa bốc hỏa`  
  https://www.vinmec.com/vie/bai-viet/bai-thuoc-dong-y-chua-boc-hoa-vi  
  đầu: Bài thuốc đông y chữa bốc hỏa⏎⏎Mãn kinh là một giai đoạn trong cuộc đời mà bất cứ người phụ nữ nào cũng sẽ phải trải qua. Trong thời kỳ mãn kinh, việc sản xuất các hormone sinh dục giảm, kéo theo những thay đổi về sức khỏe, gây ra các tình trạng khó chịu như chứng bốc hỏa, da khô, đau lưng, đau xương khớp,... Bài viết này sẽ trình bày các bài thuốc Đông y chữa bốc hỏa ở phụ nữ   
  cuối: …n trong điều trị chứng bốc hỏa vào những năm 1940. Nghiên cứu cắt ngang vào năm 1998 trên 120 phụ nữ sử dụng vitamin E với 35 liều 800IU/ngày trong vòng 4 tuần thì nhận thấy rằng vitamin E đã làm giảm 1 cơn bốc hỏa/ngày.

- **[nav]** doc_id `733833` · 4157 ký tự, 96 dòng body · title: `Bài thuốc đông y trị bệnh sỏi tiết niệu`  
  https://www.vinmec.com/vie/bai-viet/bai-thuoc-dong-y-tri-benh-soi-tiet-nieu-vi  
  đầu: Bài thuốc đông y trị bệnh sỏi tiết niệu⏎⏎Sỏi tiết niệu là tình trạng trong nước tiểu có các chất bất thường kết tụ lại trong bàng quang. Sỏi tiết niệu thường gặp ở nam giới nhiều hơn nữ giới, gây ra các triệu chứng đau đớn khi sỏi di chuyển, đi tiểu đau nhói, nóng buốt,... Sỏi tiết niệu là bệnh thường gặp, vì vậy từ xưa đã có các bài thuốc đông y trị bệnh sỏi tiết niệu. Điều tr  
  cuối: …iác giao 12g⏎⏎Quy bản 12g⏎⏎Bài thuốc trên sắc uống, ngày 01 thang.⏎⏎Trên đây là những bài thuốc Đông y trị bệnh sỏi tiết niệu được đánh giá là có hiệu quả cao trong việc điều trị bệnh. Bạn có thể tham khảo, áp dụng theo.

- **[nav]** doc_id `742782` · 8701 ký tự, 142 dòng body · title: `Đông y chữa tràn dịch khớp gối`  
  https://www.vinmec.com/vie/bai-viet/dong-y-chua-tran-dich-khop-goi-vi  
  đầu: Đông y chữa tràn dịch khớp gối⏎⏎Tràn dịch khớp gối có thể là hậu quả của chấn thương như bong gân, rách dây chằng hoặc gãy xương, do nhiễm trùng, viêm xương khớp, viêm khớp dạng thấp,... Tràn dịch khớp gối gây đau đớn và khó chịu cho bệnh nhân, ảnh hưởng đến chức năng vận động. Bài viết này sẽ trình bày cách chữa tràn dịch khớp gối bằng đông y.⏎⏎1. Tràn dịch khớp gối là gì?⏎⏎Tr  
  cuối: …ch khớp gối:⏎⏎Huyệt Độc tỵ⏎⏎Huyệt Ủy Trung.⏎⏎Như vậy, cách chữa tràn dịch khớp gối bằng đông y là sử dụng các bài thuốc, có thể phối hợp thêm xoa bóp, châm cứu... bạn có thể tham khảo và áp dụng theo từng bài thuốc trên.

- **[nav]** doc_id `754513` · 4504 ký tự, 49 dòng body · title: `Trắc nghiệm: Cuộc sống bí mật của các Hormone!`  
  https://www.vinmec.com/vie/bai-viet/trac-nghiem-cuoc-song-bi-mat-cua-cac-hormone-vi  
  đầu: Trắc nghiệm: Cuộc sống bí mật của các Hormone!⏎⏎Hormone hầu như quyết định tới toàn bộ các chức năng quan trọng của cơ thể. Nó làm việc miệt mài giống như những người phát tín hiệu đến các cơ quan, các mô và tới các tế bào nhất định. Hãy cùng bài trắc nghiệm sau củng cố thêm kiến thức về vai trò cũng như cách thức các hormone tác động lên cơ thể.⏎⏎1. Đàn ông cũng trải qua thời   
  cuối: …g cơ thể. Nó được giải phóng khi chúng ta được ôm hoặc âu yếm và nó được liên kết với sự gắn kết giữa bạn tình, mẹ và con. Nó cũng có tác dụng giúp các bà mẹ trong lúc chuyển dạ và cho con bú.⏎⏎Nguồn tham khảo: webmd.com

- **[rand]** doc_id `746055` · 4808 ký tự, 24 dòng body · title: `Lọc máu, chạy thận: Những điều cần biết`  
  https://www.vinmec.com/vie/bai-viet/loc-mau-chay-nhung-dieu-can-biet-vi  
  đầu: Lọc máu, chạy thận: Những điều cần biết⏎⏎Suy thận mạn là tình trạng bệnh lý gây ra bởi nhiều nguyên nhân khác nhau khiến thận không thể thực hiện được vai trò lọc và loại bỏ các độc tố ra khỏi cơ thể, lúc này việc lọc máu chạy thận cho bệnh nhân là phương pháp điều trị cần thiết để thay thế thận. Phương pháp lọc máu chu kỳ đang được áp dụng rộng rãi giúp bệnh nhân vẫn có thể là  
  cuối: …hó, nguy hiểm trong thời gian ngắn, hỗ trợ việc chẩn đoán, điều trị của bác sĩ hiệu quả nhất. Không gian bệnh viện được thiết kế theo tiêu chuẩn khách sạn 5 sao, mang đến cho người bệnh sự thoải mái, thân thiện, yên tâm.

- **[rand]** doc_id `753087` · 4591 ký tự, 19 dòng body · title: `Thế nào là suy dinh dưỡng thể phù (Kwashiorkor)?`  
  https://www.vinmec.com/vie/bai-viet/the-nao-la-suy-dinh-duong-the-phu-kwashiorkor-vi  
  đầu: Thế nào là suy dinh dưỡng thể phù (Kwashiorkor)?⏎⏎Bài viết được tham vấn chuyên môn cùng Thạc sĩ, Bác sĩ Ngô Thị Oanh - Bác sĩ Nhi - Khoa Nhi - Sơ sinh - Bệnh viện Đa khoa Quốc tế Vinmec Hạ Long.⏎⏎Suy dinh dưỡng thể phù (Kwashiorkor) là tình trạng trẻ bị suy dinh dưỡng mà khi cân nặng còn 60-80% mặc dù bị phù. Dạng suy dinh dưỡng này thường gặp ở trẻ trên 1 tuổi, đặc biệt ở gia  
  cuối: …hệ tiêu hóa của bé không kịp thích nghi và hoàn toàn không tốt. Vì vậy, cha mẹ phải thực sự kiên trì đồng hành cùng con và thường xuyên truy cập website vinmec.com để cập nhật những thông tin chăm sóc cho bé hữu ích nhé.

- **[rand]** doc_id `745073` · 4368 ký tự, 40 dòng body · title: `Khi nào cần uống thuốc chống đông máu?`  
  https://www.vinmec.com/vie/bai-viet/khi-nao-can-uong-thuoc-chong-dong-mau-vi  
  đầu: Khi nào cần uống thuốc chống đông máu?⏎⏎Thuốc chống đông có tác dụng ngăn ngừa sự hình thành, phát triển các cục máu đông nên thường được chỉ định dùng để phòng ngừa hoặc điều trị các bệnh lý do sự hình thành huyết khối gây ra. Vậy khi nào cần uống thuốc chống đông máu?⏎⏎1. Thuốc chống đông máu là gì?⏎⏎Thuốc chống đông máu là tên gọi chung để chỉ các loại thuốc giúp phòng ngừa   
  cuối: …i trước với bác sĩ.⏎⏎Tốt nhất khi có ý định sử dụng thuốc chống đông máu, người bệnh nên trao đổi với bác sĩ và dùng thuốc đúng theo chỉ định. Không nên tự ý dùng thuốc bởi điều này tiềm ẩn rất nhiều rủi ro tới sức khỏe.

- **[rand]** doc_id `758447` · 5755 ký tự, 33 dòng body · title: `Viêm loét đại tràng, Probiotic và Prebiotic`  
  https://www.vinmec.com/vie/bai-viet/viem-loet-dai-trang-probiotic-va-prebiotic-vi  
  đầu: Viêm loét đại tràng, Probiotic và Prebiotic⏎⏎Viêm loét đại tràng là một bệnh mãn tính kéo dài do viêm và lở loét ở đại tràng.. Hiện bệnh nhân viêm đại tràng được điều trị bằng thuốc hoặc phẫu thuật, nhưng không có cách chữa trị dứt điểm. Sự tái phát mạnh hơn có thể xảy ra ngay cả sau khi điều trị. Một số người chuyển sang sử dụng men vi sinh và prebiotic với hy vọng cải thiện t  
  cuối: …về tiêu hóa.⏎⏎>>Xem thêm: Hiệu quả của liệu pháp đơn trị liệu Thiopurine trong viêm loét đại tràng- Bài viết của Thạc sĩ, Bác sĩ Mai Viễn Phương - Khoa Khám bệnh & Nội khoa - Bệnh viện Đa khoa Quốc tế Vinmec Central Park

- **[rand]** doc_id `740067` · 4738 ký tự, 41 dòng body · title: `Công dụng của tứ quân tử thang`  
  https://www.vinmec.com/vie/bai-viet/cong-dung-cua-tu-quan-tu-thang-vi  
  đầu: Công dụng của tứ quân tử thang⏎⏎Bài thuốc tứ quân tử thang còn được gọi là tứ quân thang. Đây là phương thuốc nổi tiếng chuyên điều trị các bệnh về “khí” trong Y Học Cổ Truyền, ứng dụng trong trị các chứng bệnh suy nhược cơ thể, viêm phế quản,...⏎⏎1. Thành phần bài thuốc tứ quân tử thang⏎⏎Bài thuốc tứ quân tử thang có thành phần gồm: 12g nhân sâm (hoặc đảng sâm) + 12g phục linh  
  cuối: …ài thuốc này. Người bệnh khi dùng bài thuốc này cần chú ý tuân thủ mọi hướng dẫn của thầy thuốc đông y về liều dùng, thời gian dùng thuốc,... để đảm bảo hiệu quả điều trị bệnh và tránh những tác dụng phụ không mong muốn.

- **[rand]** doc_id `751399` · 4998 ký tự, 20 dòng body · title: `Siêu âm doppler - Siêu âm khảo sát tiên lượng khả năng có thai trong chuyển phôi`  
  https://www.vinmec.com/vie/bai-viet/sieu-am-doppler-sieu-am-khao-sat-tien-luong-kha-nang-co-thai-trong-chuyen-phoi-vi  
  đầu: Siêu âm doppler - Siêu âm khảo sát tiên lượng khả năng có thai trong chuyển phôi⏎⏎Niêm mạc tử cung có thể ví như cánh đồng phù sa, một cánh đồng phù sa màu mỡ được cung cấp nguồn dinh dưỡng, nguồn nước tốt thì cây cối sẽ tốt tươi. Siêu âm doppler niêm mạc tử cung, siêu âm khảo sát niêm mạc tử cung đã giúp bác sĩ có định hướng, tiên lượng tốt trong việc đi tìm nguyên nhân thất b  
  cuối: …a của Trung tâm Hỗ trợ sinh sản IVF Vinmec có khả năng triển khai đồng bộ và toàn diện các kỹ thuật hỗ trợ sinh sản tiên tiến nhất hiện nay, giúp hiện thực hóa ước mơ làm cha mẹ của hàng trăm gia đình trên khắp Việt Nam.

## medlatec.vn (24,709 doc)

- **[short]** doc_id `182325` · 400 ký tự, 6 dòng body · title: `Đột quỵ gia tăng vào mùa lạnh - 5 cách hữu hiệu để phòng ngừa bệnh`  
  https://medlatec.vn/tin-tuc/dot-quy-gia-tang-vao-mua-lanh-5-cach-huu-hieu-de-phong-ngua-benh  
  đầu: Đột quỵ gia tăng vào mùa lạnh - 5 cách hữu hiệu để phòng ngừa bệnh⏎⏎Tin tức⏎⏎Đột quỵ gia tăng vào mùa lạnh - 5 cách hữu hiệu để phòng ngừa bệnh⏎⏎Mùa đông là thời điểm nguy cơ đột quỵ gia tăng, do nhiệt độ lạnh làm co mạch máu, tăng huyết áp. Để giảm thiểu nguy cơ, việc phòng ngừa đột quỵ là rất quan trọng.⏎⏎Bình luận ()⏎⏎Lựa chọn dịch vụ⏎⏎Quý khách hàng vui lòng lựa chọn dịch v  
  cuối: …uỵ gia tăng, do nhiệt độ lạnh làm co mạch máu, tăng huyết áp. Để giảm thiểu nguy cơ, việc phòng ngừa đột quỵ là rất quan trọng.⏎⏎Bình luận ()⏎⏎Lựa chọn dịch vụ⏎⏎Quý khách hàng vui lòng lựa chọn dịch vụ y tế theo nhu cầu!

- **[short]** doc_id `195750` · 331 ký tự, 1 dòng body · title: `Tư vấn trực tuyến - Chuyên gia chia sẻ: Bệnh do virus corona hiểu sao cho đúng, phòng bệnh`  
  https://medlatec.vn/tin-tuc/tu-van-truc-tuyen--chuyen-gia-chia-se-benh-do-virus-corona-hieu-sao-cho-dung-phong-benh-the-nao-s194-n17022  
  đầu: Tư vấn trực tuyến - Chuyên gia chia sẻ: Bệnh do virus corona hiểu sao cho đúng, phòng bệnh thế nào?⏎⏎Hãy cùng nghe chuyên gia Truyền nhiễm Thạc sĩ - BSCKII Nguyễn Hồng Hà - Nguyên Phó Giám đốc Bệnh viện Nhiệt đới Trung Ương, Phó Chủ tịch Hội truyền nhiễm Việt Nam chia sẻ về đại dịch do loại virus nguy hiểm này tại video dưới đây:  
  cuối: …ghe chuyên gia Truyền nhiễm Thạc sĩ - BSCKII Nguyễn Hồng Hà - Nguyên Phó Giám đốc Bệnh viện Nhiệt đới Trung Ương, Phó Chủ tịch Hội truyền nhiễm Việt Nam chia sẻ về đại dịch do loại virus nguy hiểm này tại video dưới đây:

- **[short]** doc_id `175656` · 391 ký tự, 9 dòng body · title: `(Bài dự thi M30: TH 026 -15065) Muôn sắc MEDLATEC`  
  https://medlatec.vn/tin-tuc/bai-du-thi-m30-muon-sac-medlatec  
  đầu: (Bài dự thi M30: TH 026 -15065) Muôn sắc MEDLATEC⏎⏎MEDLATEC – Nơi niềm tin khởi nguồn⏎⏎Giọt máu hồng gửi gắm yêu thương⏎⏎Áo blouse trắng âm thầm bền bỉ⏎⏎Giữ bình yên qua tháng ngày vô thường.⏎⏎Muôn sắc hòa chung một màu tận tụy⏎⏎Chăm sóc bằng tâm, chữa lành bằng tình⏎⏎MEDLATEC – Vững vàng cùng đất nước⏎⏎Vì sức khỏe Việt, trọn vẹn niềm tin!⏎⏎Tác giả: Nguyễn Thị Trang - Ban Kiểm   
  cuối: …háng ngày vô thường.⏎⏎Muôn sắc hòa chung một màu tận tụy⏎⏎Chăm sóc bằng tâm, chữa lành bằng tình⏎⏎MEDLATEC – Vững vàng cùng đất nước⏎⏎Vì sức khỏe Việt, trọn vẹn niềm tin!⏎⏎Tác giả: Nguyễn Thị Trang - Ban Kiểm soát Nội bộ

- **[short]** doc_id `174401` · 479 ký tự, 4 dòng body · title: `(Bài thi M30: TH 024 -11554) Med Group trong tôi`  
  https://medlatec.vn/tin-tuc/-bai-thi-m30-th-024-11554-med-group-trong-toi  
  đầu: (Bài thi M30: TH 024 -11554) Med Group trong tôi⏎⏎1.2.3… đoàn danh Y – dược, trách nhiệm dẫn đầu Xây dựng văn hóa bền sâu Cùng nhau dịch vụ dẫn đầu tiên phong.⏎⏎Hạnh phúc quý khách hài lòng Chất lượng đảm bảo, an toàn, chóng nhanh Bao năm vẫn một màu xanh Văn minh tiêu chí “chân – thật” dẫn đầu.⏎⏎Con người – đồng nghiệp gắn nhau Chí tiến chí thăng đồng sức 250 Med – group căng   
  cuối: … minh tiêu chí “chân – thật” dẫn đầu.⏎⏎Con người – đồng nghiệp gắn nhau Chí tiến chí thăng đồng sức 250 Med – group căng buồm tiến xa Sắc màu cuộc sống – tương lai vẹn toàn.⏎⏎Tác giả: Trần Thị Mai Diểm - Medlatec Đà Nẵng

- **[short]** doc_id `192991` · 429 ký tự, 4 dòng body · title: `Thông báo lịch làm vệc Tết Dương lịch 2026`  
  https://medlatec.vn/tin-tuc/thong-bao-lich-lam-vec-tet-duong-lich-2026  
  đầu: Thông báo lịch làm vệc Tết Dương lịch 2026⏎⏎Hệ thống Y tế MEDLATEC hoạt động bình thường trong kỳ nghỉ Tết Dương lịch⏎⏎Quý khách hàng cần được tư vấn và hỗ trợ đặt lịch khám, xét nghiệm vui lòng liên hệ tổng đài 24/7: 1900 56 56 56⏎⏎MEDLATEC trân trọng cảm ơn sự tin tưởng và đồng hành của Quý Khách hàng. Kính chúc Quý Khách hàng và gia đình kỳ nghỉ Lễ vui vẻ, bình an, mạnh khỏe  
  cuối: …ài 24/7: 1900 56 56 56⏎⏎MEDLATEC trân trọng cảm ơn sự tin tưởng và đồng hành của Quý Khách hàng. Kính chúc Quý Khách hàng và gia đình kỳ nghỉ Lễ vui vẻ, bình an, mạnh khỏe và tràn đầy năng lượng cho năm mới.⏎⏎Trân trọng!

- **[title]** doc_id `190808` · 5614 ký tự, 28 dòng body · title: `PTSD là gì?`  
  https://medlatec.vn/tin-tuc/ptsd-la-gi  
  đầu: PTSD là gì?⏎⏎1. PTSD là gì?⏎⏎Nhiều người thắc mắc “PTSD là gì”. Đây là cụm từ viết tắt của căn bệnh rối loạn stress sau sang chấn. Khi chiến tranh kết thức, căn bệnh này rất thường gặp ở những cựu quân nhân, do đó, bệnh PTSD còn được gọi là “hội chứng mệt mỏi sau chiến tranh” hay “sốc vỏ đạn”.⏎⏎Bệnh PTSD ngày càng phổ biến⏎⏎Khi trải qua những sự kiện kinh hoàng chẳng hạn như ch  
  cuối: …, triệu chứng, nguyên nhân cũng như cách điều trị bệnh. Khi thấy có dấu hiệu bất ổn, cần đưa người bệnh đi khám sớm để được kiểm tra, điều trị kịp thời và sớm cải thiện bệnh, quay trở lại cuộc sống sinh hoạt bình thường.

- **[nav]** doc_id `189315` · 5109 ký tự, 120 dòng body · title: `Những điều cần biết về khám sức khỏe đi Nhật`  
  https://medlatec.vn/tin-tuc/nhung-dieu-can-biet-ve-kham-suc-khoe-di-nhat-s28-n14599  
  đầu: Những điều cần biết về khám sức khỏe đi Nhật⏎⏎1. Khám sức khỏe đi Nhật cần đảm bảo những điều kiện gì?⏎⏎Từ trước đến nay, điều kiện chung nhất khi khám sức khỏe đi Nhật đó là phải có một sức khỏe tốt, không bị các bệnh truyền nhiễm, bệnh cấp tính, mãn tính chưa chữa khỏi hay bị dị tật và phụ nữ mang thai. Đây chính là những tiêu chuẩn cơ bản để đạt yêu cầu đi Nhật.⏎⏎Khám sức kh  
  cuối: … vậy, khi đi khám sức khỏe đi Nhật, các bạn cần chọn nơi uy tín, an toàn và đảm bảo nhất, Tại bệnh viện MEDLATEC có cung cấp gói khám sức khỏe đi Nhật với chất lượng đảm bảo và chi phí hợp lý cho các bạn đang có nhu cầu.

- **[nav]** doc_id `174393` · 696 ký tự, 31 dòng body · title: `(Bài dự thi M30: TH 025 - 206) Chất MED trong ta`  
  https://medlatec.vn/tin-tuc/-bai-du-thi-m30-th-025-206-chat-med-trong-ta  
  đầu: (Bài dự thi M30: TH 025 - 206) Chất MED trong ta⏎⏎MEDLATEC vươn xa⏎⏎Cùng anh em ta bứt phá!⏎⏎Nghe này!⏎⏎Sáng thức dậy⏎⏎Khoác áo trắng chỉnh tề⏎⏎Tác phong chuyên nghiệp⏎⏎Không có chỗ để chê⏎⏎Nụ cười trên môi⏎⏎Đón khách bằng cái "Tâm"⏎⏎Thương hiệu MEDLATEC⏎⏎Vang danh suốt bao năm!⏎⏎Thống nhất hành vi⏎⏎Một ý chí đồng lòng⏎⏎Xây dựng niềm tin⏎⏎Chữ "Tín" chảy trong dòng⏎⏎Uy tín là và  
  cuối: …lạnh thấu xương⏎⏎Màu áo MEDLATEC⏎⏎Vẫn len lỏi mọi cung đường⏎⏎Lấy mẫu tận nơi⏎⏎Trao tận tay sự an tâm⏎⏎Kết quả chính xác⏎⏎Chuẩn công nghệ⏎⏎Qua bao năm⏎⏎Chuyên nghiệp!⏎⏎(Chuyên nghiệp!)⏎⏎Tác giả: Hoàng Thị Ánh Tuyết - PLS

- **[nav]** doc_id `183989` · 1242 ký tự, 50 dòng body · title: `(Bài thi M30: TH005 - 9464) Hành trình MEDLATEC`  
  https://medlatec.vn/tin-tuc/hanh-trinh-medlatec  
  đầu: (Bài thi M30: TH005 - 9464) Hành trình MEDLATEC⏎⏎Từ thuở năm chín sáu⏎⏎Có Một phòng Lab nhỏ⏎⏎Ở Giữa lòng Thủ đô⏎⏎Vỏn vẹn ba thầy cô.⏎⏎Tận tâm và có tầm⏎⏎Hy sinh cả tuổi xuân⏎⏎Mang tiện ích đến gần⏎⏎Cho cuộc sống nhân dân.⏎⏎Nhận lấy mẫu tại nhà⏎⏎Xóa bỏ cảnh hàng dài⏎⏎Giảm nỗi lo chờ đợi⏎⏎Ngày tháng cứ trôi qua.⏎⏎Từ ba người thầy cô⏎⏎Lab nhỏ đã lớn mạnh⏎⏎Với tên gọi thân quen⏎⏎Me  
  cuối: …Campuchia nối liền⏎⏎Những con tim hòa chung.⏎⏎Mỗi người mỗi tỉnh thành⏎⏎Nhưng cùng chung một nhà⏎⏎Dưới mái nhà Med lớn⏎⏎Tình thân thiết chan hòa.⏎⏎(Bài dự thi của CB Nguyễn Thị Tân - Phòng Xét Nghiệm, MEDLATEC Thanh Hóa)

- **[nav]** doc_id `185059` · 904 ký tự, 17 dòng body · title: `Infographic: Kiến thức nhất định phải biết về những bệnh lý hô hấp thường gặp mùa nồm ẩm`  
  https://medlatec.vn/tin-tuc/infographic-kien-thuc-nhat-dinh-phai-biet-ve-nhung-benh-ly-ho-hap-thuong-gap-mua-nom-am  
  đầu: Infographic: Kiến thức nhất định phải biết về những bệnh lý hô hấp thường gặp mùa nồm ẩm⏎⏎Tin tức⏎⏎Infographic: Kiến thức nhất định phải biết về những bệnh lý hô hấp thường gặp mùa nồm ẩm⏎⏎Mùa đông xuân ở miền Bắc thường được chú ý bởi kiểu thời tiết đặc trưng mưa phùn, nồm ẩm. Tình trạng này kéo dài nhiều ngày khiến độ ẩm trong không khí tăng cao, môi trường sống ẩm thấp, ngột  
  cuối: … đang bị xem nhẹ⏎⏎Từ khoá:⏎⏎nồm ẩm⏎⏎viêm đường hô hấp⏎⏎viêm xoang⏎⏎bệnh viêm phổi⏎⏎viêm mũi dị ứng⏎⏎hen phế quản⏎⏎viêm phế quản⏎⏎Bình luận ()⏎⏎Lựa chọn dịch vụ⏎⏎Quý khách hàng vui lòng lựa chọn dịch vụ y tế theo nhu cầu!

- **[nav]** doc_id `193008` · 1637 ký tự, 15 dòng body · title: `Thông báo V/v cung cấp dữ liệu xét nghiệm Covid-19 cho Cơ sở dữ liệu quốc gia về dân cư ph`  
  https://medlatec.vn/tin-tuc/thong-bao-vv-cung-cap-du-lieu-xet-nghiem-covid19-cho-co-so-du-lieu-quoc-gia-ve-dan-cu-phuc-vu-kiem-soat-cong-dan-vao-san-van-dong-my-dinh-s134-n26041  
  đầu: Thông báo V/v cung cấp dữ liệu xét nghiệm Covid-19 cho Cơ sở dữ liệu quốc gia về dân cư phục vụ kiểm soát công dân vào Sân vận động Mỹ Đình⏎⏎Kính gửi Quý khách hàng!⏎⏎Theo yêu cầu từ Bộ Công An và Cục cảnh sát QLHC về TTXH, để phục vụ kiểm soát khán giả vào sân cổ vũ trận đấu đội tuyển Việt Nam và đội tuyển Nhật Bản ngày hôm nay (11/11), Ban Tổ chức đã thống nhất thực hiện kiểm  
  cuối: …Hệ thống Y tế MEDLATEC, cần xuất trình vé xem hai trận đấu và giấy tờ tùy thân nêu trên để được hưởng ưu đãi xét nghiệm cũng như đảm bảo quy định của bệnh viện và Bộ Công an trong công tác phòng, chống dịch.⏎⏎Trân trọng!

- **[rand]** doc_id `186479` · 5435 ký tự, 32 dòng body · title: `Mách bạn cách trị rụng tóc nhiều hiệu quả và nhanh chóng`  
  https://medlatec.vn/tin-tuc/mach-ban-cach-tri-rung-toc-nhieu-hieu-qua-va-nhanh-chong-s195-n20358  
  đầu: Mách bạn cách trị rụng tóc nhiều hiệu quả và nhanh chóng⏎⏎1. Những nguyên nhân dẫn đến rụng tóc nhiều?⏎⏎Thông thường, khi già và hư tổn, tóc sẽ rụng đi để tạo cơ hội cho nang tóc phát triển sợi mới. Hiện tượng này hết sức bình thường và được gọi là rụng tóc sinh lý và bạn không cần phải lo lắng. Tuy nhiên, nếu tóc rụng quá nhiều một cách bất thường, rụng khoảng hơn 100 sợi mỗi   
  cuối: …iều bất thường có nhiều nguy cơ là do tiềm ẩn bệnh lý. Vì thế cách tốt nhất là đi khám để được điều trị kịp thời. Mọi thắc mắc và mong muốn đặt lịch khám sớm, hãy gọi đến Bệnh viện Đa khoa MEDLATEC theo số 1900 56 56 56.

- **[rand]** doc_id `185747` · 5320 ký tự, 38 dòng body · title: `Không lo ngứa vùng kín về đêm nhờ những tuyệt chiêu hữu ích`  
  https://medlatec.vn/tin-tuc/khong-lo-ngua-vung-kin-ve-dem-nho-nhung-tuyet-chieu-huu-ich-s74-n29297  
  đầu: Không lo ngứa vùng kín về đêm nhờ những tuyệt chiêu hữu ích⏎⏎1. Tác nhân khiến cho vùng kín bị ngứa về đêm⏎⏎Ngứa vùng kín về đêm là hiện tượng “cô bé” bị ngứa rát gây khó chịu khi đêm đến. Cũng có thể đây là tình trạng ngứa thông thường không có các biểu hiện lạ đi kèm, chỉ xuất hiện trong một thời gian ngắn rồi tự khỏi. Trong trường hợp bị ngứa liên tục kèm theo mùi hôi ở vùng  
  cuối: … ra hướng điều trị hiệu quả.⏎⏎Mọi thắc thắc mắc liên quan đến các vấn đề sản phụ khoa, quý khách vui lòng gọi đến hotline: 1900 56 56 56 các bác sĩ chuyên khoa tại Bệnh viện Đa khoa MEDLATEC sẽ tư vấn và hỗ trợ tận tình.

- **[rand]** doc_id `192095` · 6210 ký tự, 37 dòng body · title: `Tại sao bị mất giọng - Điểm qua 10 nguyên nhân phổ biến!`  
  https://medlatec.vn/tin-tuc/tai-sao-bi-mat-giong--diem-qua-10-nguyen-nhan-pho-bien-s98-n27867  
  đầu: Tại sao bị mất giọng - Điểm qua 10 nguyên nhân phổ biến!⏎⏎1. Mất giọng nghĩa là gì?⏎⏎Mất giọng không phải là một loại bệnh mà là một triệu chứng của bệnh lý mà chúng ta mắc phải. Giọng nói của bạn cũng sẽ trở nên khàn hơn, không còn được trong trẻo hay bạn sẽ gặp những vấn đề như khó phát ra tiếng nói.⏎⏎Nguyên nhân gây mất giọng là gì?⏎⏎Trong giao tiếp hằng ngày, con người luôn  
  cuối: …ày kéo dài gây ảnh hưởng đến cuộc sống.⏎⏎Nếu có thắc mắc hoặc muốn đặt lịch khám, các bạn có thể đến trực tiếp khoa Tai Mũi Họng Bệnh viện Đa khoa MEDLATEC hoặc gọi đến tổng đài 1900 56 56 56 để được hỗ trợ chi tiết hơn.

- **[rand]** doc_id `195490` · 5608 ký tự, 40 dòng body · title: `Trồng răng Implant có đau không và những cách giảm đau hiệu quả`  
  https://medlatec.vn/tin-tuc/trong-rang-implant-co-dau-khong-va-nhung-cach-giam-dau-hieu-qua  
  đầu: Trồng răng Implant có đau không và những cách giảm đau hiệu quả⏎⏎1. Trồng răng Implant và những ưu điểm nổi bật⏎⏎Trồng răng Implant là quy trình chỉnh nha bằng cách cấy ghép một chiếc vít titan vào xương hàm làm răng giả cố định thay thế răng đã mất mà vẫn đảm bảo vẻ đẹp tự nhiên và độ chắc khỏe của răng. Khi quá trình này được thực hiện thành công sẽ giúp:⏎⏎- Khôi phục nụ cười  
  cuối: …ục nhanh chóng.⏎⏎Mọi thắc mắc hoặc cần tư vấn chi tiết hơn về phương pháp trồng răng Implant, quý khách hàng có thể liên hệ Hotline 1900 56 56 56 của Hệ thống Y tế MEDLATEC để được giải đáp thông tin đầy đủ và chính xác.

- **[rand]** doc_id `174770` · 5577 ký tự, 38 dòng body · title: `5 cách chữa táo bón cho bà bầu nhanh nhất tại nhà không dùng thuốc`  
  https://medlatec.vn/tin-tuc/5-cach-chua-tao-bon-cho-ba-bau-nhanh-nhat-tai-nha-khong-dung-thuoc  
  đầu: 5 cách chữa táo bón cho bà bầu nhanh nhất tại nhà không dùng thuốc⏎⏎1. Vì sao bà bầu dễ bị táo bón?⏎⏎Ngoài việc tìm hiểu về cách chữa táo bón cho bà bầu nhanh nhất thì bạn cũng cần phải biết nguyên nhân gây ra tình trạng này trong quá trình mang thai để có biện pháp khắc phục hiệu quả và phù hợp với bản thân.⏎⏎Thay đổi hormone⏎⏎Trong quá trình mang thai, cơ thể sản xuất nhiều h  
  cuối: …TEC để kiểm tra, thăm khám và đưa ra hướng xử lý an toàn nhất. Mọi thông tin cần được tư vấn và hỗ trợ, quý khách hàng vui lòng liên hệ với MEDLATEC theo số 1900 56 56 56, sẽ có nhân viên tiếp nhận và hướng dẫn chi tiết.

- **[rand]** doc_id `193063` · 5933 ký tự, 45 dòng body · title: `Thông Xoang Tán - Có nên dùng hay không?`  
  https://medlatec.vn/tin-tuc/thong-xoang-tan--co-nen-dung-hay-khong--s98-n27476  
  đầu: Thông Xoang Tán - Có nên dùng hay không?⏎⏎1. Những điều cần biết về thuốc Thông Xoang Tán⏎⏎Thông Xoang Tán là một loại thuốc được sử dụng nhiều trong quá trình hỗ trợ điều trị bệnh viêm mũi, viêm xoang cấp - mạn tính. Đây là một sản phẩm thảo dược được kế thừa từ các bài thuốc cổ phương Tân Di Tán với lịch sử hàng ngàn năm.⏎⏎Với một thị trường sôi nổi như hiện nay thì có rất nh  
  cuối: …bác sĩ của Chuyên khoa Tai Mũi Họng sẽ tư vấn mọi thắc mắc cần giải đáp của bạn. Với trình độ chuyên môn sâu và sự dày dặn kinh nghiệm của các bác sĩ tại MEDLATEC sẽ đảm bảo có thể giải quyết mọi vấn về sức khỏe của bạn.

- **[rand]** doc_id `188746` · 5173 ký tự, 41 dòng body · title: `Nhận diện triệu chứng và cách chẩn đoán loãng xương ở phụ nữ mãn kinh`  
  https://medlatec.vn/tin-tuc/nhan-dien-trieu-chung-va-cach-chan-doan-loang-xuong-o-phu-nu-man-kinh  
  đầu: Nhận diện triệu chứng và cách chẩn đoán loãng xương ở phụ nữ mãn kinh⏎⏎1. Nhận diện triệu chứng loãng xương ở phụ nữ mãn kinh⏎⏎1.1 Triệu chứng loãng xương giai đoạn đầu⏎⏎Các triệu chứng loãng xương ở giai đoạn đầu thường không rõ nét nên dễ bị bỏ qua:⏎⏎- Đau lưng thường xuyên tái diễn, nhất là vùng lưng dưới. Mức độ đau không quá mạnh nhưng kéo dài và xuất hiện thường xuyên. Nh  
  cuối: …⏎Quý khách hàng có nhu cầu sàng lọc loãng xương ở phụ nữ mãn kinh có thể liên hệ đặt lịch khám cùng bác sĩ chuyên khoa Cơ xương khớp - Hệ thống Y tế MEDLATEC qua tổng đài 1900 56 56 56 để được tư vấn phương pháp phù hợp.

- **[rand]** doc_id `176630` · 6365 ký tự, 55 dòng body · title: `Bệnh trĩ ngoại và cách điều trị hiệu quả`  
  https://medlatec.vn/tin-tuc/benh-tri-ngoai-va-cach-dieu-tri-an-toan-hieu-qua  
  đầu: Bệnh trĩ ngoại và cách điều trị hiệu quả⏎⏎1. Nguyên nhân bệnh trĩ ngoại là gì?⏎⏎bệnh trĩ ngoại xuất hiện thường là do kết quả của các yếu tố sau:⏎⏎- Ngồi hoặc đứng quá lâu một chỗ⏎⏎Thói quen ngồi hoặc đứng lâu một chỗ có thể tạo ra áp lực lên các đám tĩnh mạch ở vùng hậu môn và gây ra bệnh trĩ ngoại. Điều này thường xảy ra ở những người làm công việc văn phòng, yêu cầu người la  
  cuối: …iều khách hàng tin tưởng sử dụng.⏎⏎Quý khách hàng có nhu cầu thăm khám, điều trị trĩ ngoại an toàn, hiệu quả, có thể liên hệ đặt lịch cùng bác sĩ Chuyên khoa Ngoại - Bệnh viện Đa khoa MEDLATEC qua tổng đài 1900 56 56 56.

- **[rand]** doc_id `190372` · 5593 ký tự, 62 dòng body · title: `Phân loại và ý nghĩa của kết quả xét nghiệm thuỷ đậu tại nhà`  
  https://medlatec.vn/tin-tuc/phan-loai-va-y-nghia-cua-ket-qua-xet-nghiem-thuy-dau-tai-nha -s58-n17243  
  đầu: Phân loại và ý nghĩa của kết quả xét nghiệm thuỷ đậu tại nhà⏎⏎1. Bạn biết gì về thuỷ đậu?⏎⏎1.1. Thuỷ đậu do đâu mà ra?⏎⏎Bệnh thủy đậu là bệnh nhiễm trùng ngoài da do virus Varicella-zoster gây nên. Đây là loại bệnh dễ lây và dễ bùng phát thành dịch. Virus Varicella Zoster khi gây ra bệnh có thể lây nhiễm cho người khác qua đường hô hấp và nốt mụn nước trên da. Vì thế, những trư  
  cuối: …được đánh giá là đi đầu về chất lượng kết quả và thời gian xét nghiệm⏎⏎Mọi thông tin chi tiết hơn về dịch vụ xét nghiệm thuỷ đậu tại nhà nêu trên khách hàng có thể liên hệ hotline 1900 56 56 56 để được giải đáp miễn phí.

## giadinhonline.vn (16,726 doc)

- **[short]** doc_id `111938` · 416 ký tự, 3 dòng body · title: `Những sai lầm khi ăn quả bơ hầu như ai cũng mắc phải`  
  https://giadinhonline.vn/nhung-sai-lam-khi-an-qua-bo-hau-nhu-ai-cung-mac-phai-d98281.html  
  đầu: Những sai lầm khi ăn quả bơ hầu như ai cũng mắc phải⏎⏎Bạn vẫn thường ăn bơ và bỏ lại phần vỏ xanh? Bạn muốn mau giảm cân và cố ăn thật nhiều bơ để đạt được mục tiêu?... Đó là một trong những sai lầm mà nhiều người ăn bơ vẫn thường mắc phải. Nên tránh những sai lầm này để giúp cơ thể bạn có thể hấp thụ tốt nhất những giá trị dinh dưỡng từ bơ.⏎⏎Những sai lầm khi ăn quả bơ hầu như  
  cuối: …ầm mà nhiều người ăn bơ vẫn thường mắc phải. Nên tránh những sai lầm này để giúp cơ thể bạn có thể hấp thụ tốt nhất những giá trị dinh dưỡng từ bơ.⏎⏎Những sai lầm khi ăn quả bơ hầu như ai cũng mắc phải:⏎⏎(Nguồn: Youtube)

- **[short]** doc_id `108630` · 432 ký tự, 2 dòng body · title: `Kế hoạch khắc phục chứng "nghiện ăn" của bạn`  
  https://giadinhonline.vn/ke-hoach-khac-phuc-chung-nghien-an-cua-ban-d32003.html  
  đầu: Kế hoạch khắc phục chứng "nghiện ăn" của bạn⏎⏎Kiểm tra chính xác cân nặng của bạn⏎⏎Những con số chỉ cân nặng không nói dối về thực trạng của bạn. Phải giảm 3500 calo/tuần, bạn mới cắt giảm được 1 cân. Vì thế, trừ khi bạn ăn 6 lát bánh và uống 8 ly rượu vang trong bữa tiệc tối qua, thì rõ ràng bạn đang bị nghiện ăn. Trong khi tìm cách tống những thứ đã ăn ra ngoài cơ thể, những   
  cuối: … khi bạn ăn 6 lát bánh và uống 8 ly rượu vang trong bữa tiệc tối qua, thì rõ ràng bạn đang bị nghiện ăn. Trong khi tìm cách tống những thứ đã ăn ra ngoài cơ thể, những cách dưới đây sẽ giúp bạn khắc phục chứng nghiện ăn.

- **[short]** doc_id `102822` · 302 ký tự, 2 dòng body · title: `99% dân số thế giới sử dụng ít hơn 10% bộ não! Còn bạn thì sao?`  
  https://giadinhonline.vn/99-dan-so-the-gioi-su-dung-it-hon-10-bo-nao-con-ban-thi-sao-d89783.html  
  đầu: 99% dân số thế giới sử dụng ít hơn 10% bộ não! Còn bạn thì sao?⏎⏎Theo một nghiên cứu, nếu có thể sử dụng hết 90% còn lại của bộ não thì con người sẽ có những khả năng siêu nhiên. Vậy trên thực tế chúng ta đang sử dụng bao nhiêu phần trăm của bộ não?⏎⏎Cùng thực hiện bài trắc nghiệm sau đây các bạn nhé:  
  cuối: …ứu, nếu có thể sử dụng hết 90% còn lại của bộ não thì con người sẽ có những khả năng siêu nhiên. Vậy trên thực tế chúng ta đang sử dụng bao nhiêu phần trăm của bộ não?⏎⏎Cùng thực hiện bài trắc nghiệm sau đây các bạn nhé:

- **[short]** doc_id `112552` · 300 ký tự, 3 dòng body · title: `Nước uống thảo dược giải khát, kích thích tinh thần`  
  https://giadinhonline.vn/nuoc-uong-thao-duoc-giai-khat-kich-thich-tinh-than-d90117.html  
  đầu: Nước uống thảo dược giải khát, kích thích tinh thần⏎⏎Từ các nguyên liệu như chanh, gừng, mật ong, lá kinh giới cay (oregano), hương thảo (rosemary), lá bạc hà,… bạn đã có ngay ly nước giúp tinh thần thêm sảng khoái.⏎⏎Nước uống thảo dược giải khát, kích thích tinh thần:⏎⏎Minh Trang (theo Bon Appétit)  
  cuối: …h, gừng, mật ong, lá kinh giới cay (oregano), hương thảo (rosemary), lá bạc hà,… bạn đã có ngay ly nước giúp tinh thần thêm sảng khoái.⏎⏎Nước uống thảo dược giải khát, kích thích tinh thần:⏎⏎Minh Trang (theo Bon Appétit)

- **[short]** doc_id `110986` · 309 ký tự, 4 dòng body · title: `Những ai có nguy cơ thiếu Axit Folic?`  
  https://giadinhonline.vn/nhung-ai-co-nguy-co-thieu-axit-folic-d98847.html  
  đầu: Những ai có nguy cơ thiếu Axit Folic?⏎⏎Axit Folic có vai trò quan trọng trong việc dự phòng dị tật thai nhi. Những người nào có nhiều nguy cơ thiếu axit folic và làm thế nào để bổ sung? Hãy nghe chuyên gia dinh dưỡng tư vấn.⏎⏎Video: Ai có nguy cơ thiếu Axit Folic⏎⏎(Nguồn Video: Youtube)⏎⏎Phương Vũ (Tổng hợp)  
  cuối: …ng dị tật thai nhi. Những người nào có nhiều nguy cơ thiếu axit folic và làm thế nào để bổ sung? Hãy nghe chuyên gia dinh dưỡng tư vấn.⏎⏎Video: Ai có nguy cơ thiếu Axit Folic⏎⏎(Nguồn Video: Youtube)⏎⏎Phương Vũ (Tổng hợp)

- **[title]** doc_id `109783` · 6113 ký tự, 29 dòng body · title: `Mè đen`  
  https://giadinhonline.vn/me-den-d70058.html  
  đầu: Mè đen⏎⏎Vừng tên khác là mè, chi ma, hắc chi ma, hồ ma, dầu ma, kén ma nga (Thái)… Theo Đông y, vừng vị ngọt tính bình đi vào can, phế, tỳ, thận.⏎⏎Theo đông y, vừng đen vị ngọt, tính bình, tác dụng dưỡng huyết, nhuận táo, bổ ngũ tạng, hư nhược, ích khí lực, bền gân cốt, sáng mắt, thêm thông minh, dùng ngoài đắp trị sưng tấy, vết bỏng và làm cao dán nhọt. Mè đen còn giúp bổ gan   
  cuối: …ín hiện đang có bán rộng rãi tại các hệ thống siêu thị Vinmart, Sapo Mart, Hapro Mart, Đức Thành, Intimex, … và hệ thống các cửa hàng đại lý các tỉnh Nghệ An, Hà Tĩnh, Hà Nội, Hải Dương, Hưng Yên, Hải Phòng, Ninh Bình, …

- **[title]** doc_id `107657` · 1849 ký tự, 9 dòng body · title: `Đơn thuốc`  
  https://giadinhonline.vn/don-thuoc-d207746.html  
  đầu: Đơn thuốc⏎⏎Mới đây trên trang cá nhân của mình, PGS.TS.BS Nguyễn Lân Hiếu - Giám đốc Bệnh viện Đại học Y Hà Nội chia sẻ về chủ đề “đơn thuốc” thu hút sự quan tâm của đông đảo mọi người.⏎⏎Gia đình Việt Nam xin chia sẻ lại bài viết của BS Nguyễn Lân Hiếu, cụ thể:⏎⏎“Hôm qua tôi có cô bệnh nhân phúc hậu đến từ thành phố Hoa Phượng Đỏ. Đúng như tính cách của người Hải Phòng là "khôn  
  cuối: …dò, chỉ dẫn chính là phương pháp điều trị chủ đạo. Xin hãy đọc kỹ trước khi sử dụng còn "lăn tăn" trong đầu đừng ngần ngại hỏi lại cho rõ hoặc bốc máy lên gọi những số điện thoại được ghi trong đơn thuốc của chúng tôi!".

- **[title]** doc_id `105509` · 3016 ký tự, 10 dòng body · title: `Cái ao cạn`  
  https://giadinhonline.vn/cai-ao-can-d114128.html  
  đầu: Cái ao cạn⏎⏎Buổi chiều khi cả nhà chuẩn bị ăn cơm chị Thanh mới giật mình không thấy con đâu. Bình thường con bé vẫn quanh quẩn chơi trong sân nhà sau khi được bà đón từ trường mẫu giáo về, thế mà chị gọi con mấy câu không thấy đâu.⏎⏎Chị nghĩ con bé trốn trong nhà nên đi vào nhà tìm cũng không thấy. Bà ngoại thì đang cho gà ăn, cứ ngỡ cháu vẫn ở trong sân. Chị Thanh chạy sang h  
  cuối: …hể để phòng tai nạn khi chơi hay leo chèo ở độ cao. Nếu cái gì cũng cấm thì con sẽ trở nên nhút nhát, e dè mọi thứ”. Cô Duyên nói thế và bắt đầu tranh thủ tư vấn cho chị Thanh những bài học đầu tiên cần dạy con.⏎⏎Lê Tuấn

- **[nav]** doc_id `101188` · 1648 ký tự, 22 dòng body · title: `13 triệu chứng nhiễm Omicron ở người đã tiêm đầy đủ vaccine COVID-19`  
  https://giadinhonline.vn/13-trieu-chung-nhiem-omicron-o-nguoi-da-tiem-day-du-vaccine-covid-19-d179049.html  
  đầu: 13 triệu chứng nhiễm Omicron ở người đã tiêm đầy đủ vaccine COVID-19⏎⏎Nghiên cứu được công bố trên tạp chí dịch tễ học Eurosurveillance đã xem xét các trường hợp nhiễm trùng được ghi lại tại một bữa tiệc Giáng sinh được tổ chức ở Oslo, Na Uy, vào ngày 30 tháng 11 năm ngoái.⏎⏎Người ta thấy rằng 66 trong số 117 người tham dự đã xác nhận nhiễm biến thể Omicron. Một trong những   
  cuối: …̉ Omicron gặp phải là chảy nước mũi, nhức đầu, mệt mỏi, hắt hơi và đau họng.⏎⏎Khoảng một nửa số người được cho là đã trải qua 3 triệu chứng phổ biến khi nhiễm Covid-19 như ho liên tục, sốt, mất khứu giác hoặc vị giác.

- **[nav]** doc_id `104679` · 2909 ký tự, 48 dòng body · title: `Các chất gây nghiện tồn tại trong cơ thể người trong bao lâu?`  
  https://giadinhonline.vn/cac-chat-gay-nghien-ton-tai-trong-co-the-nguoi-trong-bao-lau-d175949.html  
  đầu: Các chất gây nghiện tồn tại trong cơ thể người trong bao lâu?⏎⏎Chất gây nghiện là gì?⏎⏎Chất gây nghiện nói chung là bất kỳ chất nào khi hấp thụ vào cơ thể của một sinh vật sống có thể làm thay đổi chức năng bình thường của cơ thể theo hướng tạo ra sự phụ thuộc của cơ thể đối với chất đó hoặc cảm giác thèm thuồng, ghiền, nghiện ở các mức độ khác nhau.⏎⏎Chất gây nghiện nói chung   
  cuối: …ày⏎⏎Sử dụng chất gây nghiện đang từng ngày bào mòn cơ thể người sử dụng. Thời gian tồn tại trong cơ thể người càng lâu thì càng nguy hiểm, gây hệ luỵ xấu không chỉ cho bản thân họ mà còn hệ luỵ cho cả gia đình và xã hội.

- **[nav]** doc_id `109408` · 1918 ký tự, 55 dòng body · title: `Lượng calo trong những loại đồ uống hàng ngày`  
  https://giadinhonline.vn/luong-calo-trong-nhung-loai-do-uong-hang-ngay-d46891.html  
  đầu: Lượng calo trong những loại đồ uống hàng ngày⏎⏎Các loại thực phẩm nhiều calo, dầu mỡ và chất béo, thói quen lười vận động thường được nhắc đến như là những tác nhân chính gây nên tình trạng tích tụ mỡ cho cơ thể, làm bạn khó giảm cân. Tuy nhiên, có một “thủ phạm” khiến bạn mau sở hữu vòng eo bánh mì, hoặc không thể duy trì cân nặng hợp lý nhưng ít được quan tâm chú ý, đó chính   
  cuối: …ăng lượng: 104 kcal⏎⏎Rượu sâm banh⏎⏎Thể tích: 120 ml⏎⏎Năng lượng: 89 kcal⏎⏎Rượu⏎⏎Thể tích: 150 ml⏎⏎Năng lượng: 125 kcal⏎⏎Rượu mạnh: Vodka, rượu rum, whisky, gin (40% rượu)⏎⏎Thể tích: 45 ml⏎⏎Năng lượng: 96 kcal⏎⏎Phương Vũ

- **[nav]** doc_id `111241` · 2147 ký tự, 16 dòng body · title: `Những dấu hiệu của cơn đau báo hiệu mắc bệnh nặng không được bỏ qua`  
  https://giadinhonline.vn/nhung-dau-hieu-cua-con-dau-bao-hieu-mac-benh-nang-khong-duoc-bo-qua-d94670.html  
  đầu: Những dấu hiệu của cơn đau báo hiệu mắc bệnh nặng không được bỏ qua⏎⏎Tình trạng đau đầu dữ dội⏎⏎Đau đầu có nhiều yếu tố gây nên trong đó có thể là do ngạt mũi, viêm xoang, mất nước hoặc khó chịu cũng khiến bạn đau đầu. Tuy nhiên, sau khi bạn dùng thuốc mà cơn đau đầu không hề thuyên giảm hoặc thậm chí còn đâu dữ dội hơn thì rất có thể đây chính là dấu hiệu xuất huyết não hoặc đ  
  cuối: …ường hợp này là do cơ thể có khả năng bị trầm cảm khiến các bộ phận không được khỏe trong người phát cơn đau. Lời khuyên tốt nhất là nên đi khám bác sĩ chứ không nên chủ quan để biết được bệnh tình của mình.⏎⏎Hà An (T/h)

- **[nav]** doc_id `103260` · 2916 ký tự, 48 dòng body · title: `Bà bầu ăn hạt điều được không?`  
  https://giadinhonline.vn/ba-bau-an-hat-dieu-duoc-khong-d44947.html  
  đầu: Bà bầu ăn hạt điều được không?⏎⏎Bà bầu ăn hạt điều được không?⏎⏎Duy trì một chế độ ăn uống lành mạnh là điều quan trọng nhất cần nhớ khi đang mang thai hoặc có kế hoạch có em bé. Cách tốt nhất để làm điều đó là ăn nhiều loại thực phẩm dinh dưỡng, tránh loại có thể có khả năng gây hại.⏎⏎Bà bầu ăn hạt điều được không?⏎⏎Các chuyên gia cho rằng, bà bầu có thể ăn một số loại hạt tro  
  cuối: …, nên cho vào lọ thủy tinh sạch và khô, đậy kín, giữ nơi khô ráo, tránh ánh sáng. Hạt điều để lâu sẽ bị mềm, bạn có thể cho vào chảo, rang sơ trên lửa nhỏ, sau đó để nguội, hạt điều sẽ giòn trở lại.⏎⏎An Nguyên (Tổng hợp)

- **[rand]** doc_id `111250` · 1794 ký tự, 13 dòng body · title: `Những dấu hiệu không nên xem thường với bệnh nhân tiểu đường`  
  https://giadinhonline.vn/nhung-dau-hieu-khong-nen-xem-thuong-voi-benh-nhan-tieu-duong-d26150.html  
  đầu: Những dấu hiệu không nên xem thường với bệnh nhân tiểu đường⏎⏎Tiểu đường được xem là căn bệnh ở các nước đang phát triển, gắn liền với nhiều biến chứng ở các cơ quan như não, thần kinh, thận mắt, mạch máu và đặc biệt là tim mạch. Bạn nên để ý đến những biểu hiện sau của cơ thể để thăm khám, phát hiện kịp thời bệnh tiểu đường. Việc phát hiện và điều trị sớm, sẽ giúp bạn tránh đư  
  cuối: …à sorbitol gây cản trở tầm nhìn của bạn.⏎⏎Hay bị cảm lạnh hoặc cảm cúm⏎⏎Các hệ thống miễn dịch suy yếu làm cho vết cắt và vết bầm tím lâu lành cũng có thể làm cho bạn dễ bị tổn thương và dễ bị cảm lạnh và cúm.⏎⏎Phương Vũ

- **[rand]** doc_id `101386` · 2384 ký tự, 12 dòng body · title: `3 thực phẩm gây xơ gan cao nhưng 99% người Việt sử dụng mỗi ngày`  
  https://giadinhonline.vn/3-thuc-pham-gay-xo-gan-cao-nhung-99-nguoi-viet-su-dung-moi-ngay-d168383.html  
  đầu: 3 thực phẩm gây xơ gan cao nhưng 99% người Việt sử dụng mỗi ngày⏎⏎Nội tạng của động vật là thực phẩm gây xơ gan⏎⏎Thói quen của rất nhiều người là thường thích ăn nội tạng của động vật, óc, tim cật, lòng lợn, tràng… những không hề biết đây là thực phẩm gây xơ gan.⏎⏎Chúng ta chỉ nên ăn một lượng vừa phải, đúng cách để không gây ra quá nhiều vấn đề ảnh hưởng tới sức khỏe của mình.  
  cuối: …u “tấn công” nhiều lần sẽ có khả năng cao bị xơ gan. Chính vì vậy, nên tránh xa rượu bia càng nhiều càng tốt.⏎⏎* Tiêu đề bài viết đã được thay đổi⏎⏎->5 "thủ phạm" chính dẫn đến bệnh gan nhiễm mỡ, tuyệt đối chớ coi thường

- **[rand]** doc_id `110888` · 2290 ký tự, 13 dòng body · title: `Nhiễm trùng nguy kịch sau 1 tuần tự đắp lá cây trị vết thương tại nhà`  
  https://giadinhonline.vn/nhiem-trung-nguy-kich-sau-1-tuan-tu-dap-la-cay-tri-vet-thuong-tai-nha-d204949.html  
  đầu: Nhiễm trùng nguy kịch sau 1 tuần tự đắp lá cây trị vết thương tại nhà⏎⏎Bệnh nhân N.X.Q. (sinh năm 1940, trú tại Động Linh, Thị xã Quảng Yên, tỉnh Quảng Ninh) không may bị tổn thương bàn chân trái, mất da.⏎⏎Ban đầu, vì chủ quan và nghe theo lời khuyên dân gian, bệnh nhân đã tự điều trị tại nhà bằng cách đắp lá cây với hy vọng vết thương sẽ nhanh chóng lành lại. Tuy nhiên, sau mộ  
  cuối: …p của bệnh nhân N.X.Q là một bài học quan trọng, nhắc nhở mọi người cần thận trọng trong việc xử lý vết thương. Việc điều trị sớm và đúng phương pháp sẽ giúp vết thương phục hồi nhanh hơn, tránh các biến chứng nguy hiểm.

- **[rand]** doc_id `106805` · 3184 ký tự, 26 dòng body · title: `Đàn ông đau 6 bộ phận này nên đi khám sớm trước khi phát bệnh`  
  https://giadinhonline.vn/dan-ong-dau-6-bo-phan-nay-nen-di-kham-som-truoc-khi-phat-benh-d185955.html  
  đầu: Đàn ông đau 6 bộ phận này nên đi khám sớm trước khi phát bệnh⏎⏎Theo "Báo cáo phát triển y học giảm đau Trung Quốc", đau đã trở thành vấn đề sức khỏe lớn thứ ba sau các bệnh tim mạch, mạch máu não và khối u.⏎⏎Đối với nam giới, tuy cơ thể cường tráng hơn nữ giới nhưng đôi khi cũng gặp một số vấn đề do bất cẩn. Khi mắc bệnh, một số nam giới thấy cơ thể có cảm giác đau nhức.⏎  
  cuối: …hám kịp thời nếu có vấn đề để tránh những thiệt hại không thể khắc phục được.⏎⏎Suy cho cùng, một số bệnh nam khoa khi xảy ra sẽ gây nguy hại đến sức khỏe thể chất lớn hơn, chỉ cần cảnh giác và chú ý là có thể tránh được.

- **[rand]** doc_id `105935` · 4237 ký tự, 28 dòng body · title: `Chạy bộ mỗi ngày 20km, ăn kiêng 10 năm vẫn suýt nhồi máu cơ tim`  
  https://giadinhonline.vn/chay-bo-moi-ngay-20km-an-kieng-10-nam-van-suyt-nhoi-mau-co-tim-d207020.html  
  đầu: Chạy bộ mỗi ngày 20km, ăn kiêng 10 năm vẫn suýt nhồi máu cơ tim⏎⏎Theo thông tin từ trang aboluowang, ông Lưu (45 tuổi) từng là một người ăn uống vô độ, ít vận động. Lối sống thiếu kiểm soát khiến ông mắc phải huyết áp cao, tiểu đường và mỡ máu cao – hay còn gọi là “tam cao”.⏎⏎Năm 35 tuổi, ông quyết tâm thay đổi. Từ những ngày đầu cố gắng lết 3km một cách khó nhọc, ông dần nâng   
  cuối: …g mạch, giảm lưu thông máu dễ gây đau thắt ngực, lạnh chân, thậm chí nhồi máu cơ tim, đột quỵ.⏎⏎Cục máu đông: Tĩnh mạch dễ bị giãn nếu làm việc đứng lâu hoặc khuân vác nặng gây suy giãn tĩnh mạch, đau nhức, khó vận động.

- **[rand]** doc_id `108480` · 2301 ký tự, 10 dòng body · title: `Hồi hộp đánh trống ngực... hóa ra bị rối loạn nhịp tim hiếm gặp ở người trẻ`  
  https://giadinhonline.vn/hoi-hop-danh-trong-nguc-hoa-ra-bi-roi-loan-nhip-tim-hiem-gap-o-nguoi-tre-d137256.html  
  đầu: Hồi hộp đánh trống ngực... hóa ra bị rối loạn nhịp tim hiếm gặp ở người trẻ⏎⏎Nam bệnh nhân 32 tuổi, quê ở Thanh Hóa này xuất hiện các triệu chứng trên trong một thời gian dài nhưng chỉ thoáng qua nên ít để ý. Chỉ đến khi vào Trung tâm Tim mạch (Bệnh viện E) điều trị, bệnh nhân mới biết mình bị mắc bệnh rối loạn nhịp tim hiếm gặp ở người trẻ.⏎⏎Ths.BS. Vũ Văn Bạ, Khoa Nội tim mạc  
  cuối: …nh nhân mắc rung nhĩ và điều trị ở tuổi 32 có thể coi là trẻ nhất cho tới thời điểm hiện nay và chưa xác định được nguyên nhân cũng như yếu tố nguy cơ.⏎⏎Hàng triệu người phải thán phục với clip cô gái vừa mổ vừa hát vang

- **[rand]** doc_id `114101` · 3499 ký tự, 11 dòng body · title: `Thanh niên còn ngần ngại khi mua bao cao su`  
  https://giadinhonline.vn/thanh-nien-con-ngan-ngai-khi-mua-bao-cao-su-d195935.html  
  đầu: Thanh niên còn ngần ngại khi mua bao cao su⏎⏎Thời gian qua, Việt Nam đạt được nhiều thành công nhất định trong chăm sóc sức khỏe sinh sản, sức khỏe tình dục cho vị thành niên, thanh niên. Tuy nhiên, tình trạng quan hệ tình dục sớm, quan hệ tình dục không an toàn, mang thai ngoài ý muốn và phá thai không an toàn của vị thành niên, thanh niên vẫn chưa được cải thiện đáng kể, đặc   
  cuối: …ền hôn nhân, các cặp đôi sẽ được tư vấn để bắt đầu một cuộc sống tình dục lành mạnh, an toàn. Bên cạnh đó, có thể phát hiện và điều trị sớm một số bệnh tật có thể ảnh hưởng đến vấn đề tình dục, mang thai, sinh đẻ về sau.

## baohaiphong.vn (16,246 doc)

- **[short]** doc_id `53674` · 345 ký tự, 5 dòng body · title: `[Infographics] Chín nhóm đối tượng ưu tiên và miễn phí tiêm vaccine phòng Covid-19`  
  https://baohaiphong.vn/infographics-chin-nhom-doi-tuong-uu-tien-va-mien-phi-tiem-vaccine-phong-covid-19-197568.html  
  đầu: [Infographics] Chín nhóm đối tượng ưu tiên và miễn phí tiêm vaccine phòng Covid-19⏎⏎Chính phủ vừa ban hành Nghị quyết số 21/NQ-CP ngày 26.2 về mua và sử dụng vaccine phòng COVID-19, trong đó nêu rõ 9 nhóm đối tượng ưu tiên tiêm và miễn phí.⏎⏎[Infographics] Chín nhóm đối tượng ưu tiên và miễn phí tiêm vaccine phòng Covid-19⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định  
  cuối: …Q-CP ngày 26.2 về mua và sử dụng vaccine phòng COVID-19, trong đó nêu rõ 9 nhóm đối tượng ưu tiên tiêm và miễn phí.⏎⏎[Infographics] Chín nhóm đối tượng ưu tiên và miễn phí tiêm vaccine phòng Covid-19⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định

- **[short]** doc_id `46935` · 268 ký tự, 2 dòng body · title: `5 lưu ý an toàn thực phẩm trong dịp Tết Nguyên đán`  
  https://baohaiphong.vn/5-luu-y-an-toan-thuc-pham-trong-dip-tet-nguyen-dan-372775.html  
  đầu: 5 lưu ý an toàn thực phẩm trong dịp Tết Nguyên đán⏎⏎Trong dịp Tết, việc bảo đảm vệ sinh an toàn thực phẩm là một trong những điều hết sức quan trọng. Chỉ cần thực hiện một vài lưu ý giúp chúng ta đón Tết vui vẻ và khỏe mạnh bên người thân và gia đình.⏎⏎TB (theo TTXVN)  
  cuối: …án⏎⏎Trong dịp Tết, việc bảo đảm vệ sinh an toàn thực phẩm là một trong những điều hết sức quan trọng. Chỉ cần thực hiện một vài lưu ý giúp chúng ta đón Tết vui vẻ và khỏe mạnh bên người thân và gia đình.⏎⏎TB (theo TTXVN)

- **[short]** doc_id `55213` · 388 ký tự, 2 dòng body · title: `Nam Sách: Tiêm phòng vụ xuân cho đàn vật nuôi đã đạt trên 73% kế hoạch`  
  https://baohaiphong.vn/nam-sach-tiem-phong-vu-xuan-cho-dan-vat-nuoi-da-dat-tren-73-ke-hoach-340599.html  
  đầu: Nam Sách: Tiêm phòng vụ xuân cho đàn vật nuôi đã đạt trên 73% kế hoạch⏎⏎Trạm thú y huyện đang tăng cường hướng dẫn, chỉ đạo đội ngũ thú y viên rà soát, theo dõi chặt chẽ số vật nuôi đã tiêm phòng, đồng thời tiến hành tiêm bổ sung cho vật nuôi mới nhập đàn. Huyện đang thống kê số lượng đàn trâu, bò chuẩn bị các điều kiện cần thiết phục vụ công tác tiêm phòng đạt hiệu quả cao.⏎⏎D  
  cuối: …ẽ số vật nuôi đã tiêm phòng, đồng thời tiến hành tiêm bổ sung cho vật nuôi mới nhập đàn. Huyện đang thống kê số lượng đàn trâu, bò chuẩn bị các điều kiện cần thiết phục vụ công tác tiêm phòng đạt hiệu quả cao.⏎⏎Dương Hòa

- **[short]** doc_id `47200` · 351 ký tự, 1 dòng body · title: `80% số phụ nữ mang thai được xét nghiệm HIV`  
  https://baohaiphong.vn/80-so-phu-nu-mang-thai-duoc-xet-nghiem-hiv-304374.html  
  đầu: 80% số phụ nữ mang thai được xét nghiệm HIV⏎⏎Trong đó có 80% số phụ nữ mang thai được xét nghiệm HIV, phát hiện 2 người dương tính với HIV và 4 trẻ sinh ra từ bà mẹ nhiễm HIV/AIDS được chuyển tiếp đến các dịch vụ chăm sóc toàn diện. Năm 2014, toàn tỉnh dự kiến có khoảng 65% số phụ nữ mang thai được tư vấn và xét nghiệm HIV trong quá trình mang thai.  
  cuối: …với HIV và 4 trẻ sinh ra từ bà mẹ nhiễm HIV/AIDS được chuyển tiếp đến các dịch vụ chăm sóc toàn diện. Năm 2014, toàn tỉnh dự kiến có khoảng 65% số phụ nữ mang thai được tư vấn và xét nghiệm HIV trong quá trình mang thai.

- **[short]** doc_id `51704` · 486 ký tự, 5 dòng body · title: `[Đồ họa] Hướng dẫn tiêm vaccine COVID-19 liều cơ bản và nhắc lại`  
  https://baohaiphong.vn/do-hoa-huong-dan-tiem-vaccine-covid-19-lieu-co-ban-va-nhac-lai-164796.html  
  đầu: [Đồ họa] Hướng dẫn tiêm vaccine COVID-19 liều cơ bản và nhắc lại⏎⏎[Đồ họa] Hướng dẫn tiêm vaccine COVID-19 liều cơ bản và nhắc lại⏎⏎31/01/2022 05:54⏎⏎Bộ Y tế cho biết người đã tiêm đủ liều cơ bản bằng vaccine do Sinopharm sản xuất thì có thể tiêm liều bổ sung và nhắc lại bằng vaccine do AstraZeneca sản xuất.⏎⏎Bình luận của bạn đã được gửi và sẽ hiển thị sau khi được duyệt bởi b  
  cuối: …c lại bằng vaccine do AstraZeneca sản xuất.⏎⏎Bình luận của bạn đã được gửi và sẽ hiển thị sau khi được duyệt bởi ban biên tập.⏎⏎Ban biên tập giữ quyền biên tập nội dung bình luận để phù hợp với qui định nội dung của Báo.

- **[title]** doc_id `61538` · 1940 ký tự, 10 dòng body · title: `Vẫn còn may`  
  https://baohaiphong.vn/van-con-may-288680.html  
  đầu: Vẫn còn may⏎⏎Chị không biết những loại thuốc giảm đau đó ngày một vắt kiệt sức lực chị. Chị thấy dạ dày bất thường, cả người phù thũng. Nhưng vốn là người tiết kiệm quá mức nên chị tự an ủi mình đó là chuyện thường của những người bị căn bệnh viêm đa khớp. Một năm gần đây, thấy các khớp sưng tấy và đau đớn hơn trước, ngoài uống thuốc giảm đau, chị còn bốc thuốc nam của một ông   
  cuối: … bảo mọi người:⏎⏎- Mình nghiệm ra rồi, cứ ốm đau là phải đi bệnh viện khám, có bác sĩ kê đơn, chữa trị là an toàn nhất. Dùng thuốc phải khoa học chứ không thể à uôm được. Kể cả thuốc nam, dùng sai cũng trở thành độc hại.

- **[title]** doc_id `54019` · 1457 ký tự, 10 dòng body · title: `Khổ vì ăn`  
  https://baohaiphong.vn/kho-vi-an-308860.html  
  đầu: Khổ vì ăn⏎⏎- Cuối giờ làm việc chiều nay, cơ quan ta tổ chức đi thăm anh T. đang nằm điều trị tại Bệnh viện Đa khoa tỉnh, đề nghị các anh sắp xếp thời gian, công việc nhé.⏎⏎Nghe xong, mọi người cùng nhìn nhau vì bất ngờ. Anh A. tỏ vẻ sốt ruột:⏎⏎- Sao mới đầu năm mà anh T. đã nằm viện, có phải anh ấy bị tai nạn giao thông không?⏎⏎- Không phải bị tai nạn giao thông. Anh T. nhập v  
  cuối: …a đình, bạn bè từ nay đừng ăn tiết canh và các món gỏi nữa, vì nguy cơ nhiễm bệnh rất cao. Khi nhiễm bệnh vừa thiệt thân, vừa khổ gia đình lại còn mang tiếng là khổ vì ăn!⏎⏎Nghe tới đây, mọi người cùng gật đầu tán thành.

- **[title]** doc_id `61158` · 1950 ký tự, 4 dòng body · title: `Tự cấp cứu`  
  https://baohaiphong.vn/tu-cap-cuu-482669.html  
  đầu: Tự cấp cứu⏎⏎(HPĐT)- Bác sĩ chuyên khoa 1 Nguyễn Minh Hải, Phó trưởng Khoa Ngoại tổng hợp (Bệnh viện Trẻ em Hải Phòng) cho biết, cuối năm 2022, anh và các cộng sự điều trị thành công trường hợp bệnh nhi N.M.H. 8 tuổi, vào viện với chẩn đoán viêm ruột thừa cấp nghi đã vỡ.⏎⏎Trong quá trình phẫu thuật nội soi cấp cứu cắt ruột thừa, bác sĩ phát hiện ruột thừa của cháu bé viêm rõ, bị  
  cuối: … giúp người bị nạn vượt qua khoảnh khắc nguy hiểm. Như trường hợp cháu bé N.M.H., nếu gia đình đưa bé đến cơ sở y tế kịp thời để lấy dị vật, cháu có thể không phải chịu đau đớn trong thời gian dài và nguy hiểm như vậy./.

- **[title]** doc_id `60063` · 2115 ký tự, 16 dòng body · title: `Thuốc rởm`  
  https://baohaiphong.vn/thuoc-rom-291195.html  
  đầu: Thuốc rởm⏎⏎Chốc chốc lại thấy muỗi đốt chân, ông Phúc vừa gãi vừa nói:⏎⏎- Quê mình mùa này lắm muỗi nhỉ! Từ nãy tới giờ ngồi đây mà không biết bao nhiêu con muỗi “hỏi thăm” rồi.⏎⏎Thấy anh hỏi, ông Phú đặt chén nước xuống nói giọng bức xúc:⏎⏎- Mùa này lắm muỗi bác ạ. Cách đây chục hôm em ra quán ngoài đường cái kia mua lọ thuốc muỗi về phun nhưng chẳng ăn thua gì cả. Bảo tầm hai  
  cuối: … trong vòng bốn đến sáu tháng sẽ không lo bị muỗi đốt.⏎⏎- Em biết rồi. Chờ đám cưới xong xuôi em cũng phải lên trung tâm y tế hỏi mua để phun lại cho yên tâm. Thấy đứa cháu suốt ngày bị muỗi đốt sưng hết tay chân mà xót.

- **[nav]** doc_id `61678` · 327 ký tự, 8 dòng body · title: `[Video] Chính sách bảo hiểm y tế mới tạo thuận lợi cho bệnh nhân nặng`  
  https://baohaiphong.vn/video-chinh-sach-bao-hiem-y-te-moi-tao-thuan-loi-cho-benh-nhan-nang-201572.html  
  đầu: [Video] Chính sách bảo hiểm y tế mới tạo thuận lợi cho bệnh nhân nặng⏎⏎Người dân cần hiểu đúng quy định về việc thông tuyến tỉnh trong khám, chữa bệnh bảo hiểm y tế để tránh thiệt thòi.⏎⏎Xem clip⏎⏎THANH NGA - THÀNH CHUNG⏎⏎Bài liên quan⏎⏎[Video] Chính sách bảo hiểm y tế mới tạo thuận lợi cho bệnh nhân nặng⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định  
  cuối: …việc thông tuyến tỉnh trong khám, chữa bệnh bảo hiểm y tế để tránh thiệt thòi.⏎⏎Xem clip⏎⏎THANH NGA - THÀNH CHUNG⏎⏎Bài liên quan⏎⏎[Video] Chính sách bảo hiểm y tế mới tạo thuận lợi cho bệnh nhân nặng⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định

- **[nav]** doc_id `58039` · 1820 ký tự, 23 dòng body · title: `Sáng 3.5, Việt Nam không ghi nhận ca mắc mới COVID-19`  
  https://baohaiphong.vn/sang-3-5-viet-nam-khong-ghi-nhan-ca-mac-moi-covid-19-353672.html  
  đầu: Sáng 3.5, Việt Nam không ghi nhận ca mắc mới COVID-19⏎⏎Nhân viên y tế phun khử khuẩn tại khu vực gần ca nghi nhiễm. (Ảnh: Phạm Hùng/TTXVN phát)⏎⏎Theo thông tin từ Bộ Y tế, tính từ 18 giờ ngày 2.5 đến 6 giờ ngày 3.5, Việt Nam không ghi nhận ca mắc mới COVID-19.⏎⏎Như vậy, tính đến 6 giờ ngày 35, Việt Nam có tổng cộng 1.595 ca mắc COVID-19 do lây nhiễm trong nước, trong đó số lượn  
  cuối: …ời⏎⏎6- Sơn La: 454 người⏎⏎7- Quảng Trị: 158 người⏎⏎8- Gia Lai: 492 người⏎⏎9- TP Hồ Chí Minh: 155 người⏎⏎10- Trà Vinh: 129 người⏎⏎11- Bình Phước: 67 người⏎⏎12- Kiên Giang: 1.811 người⏎⏎13- Cà Mau: 83 người.⏎⏎Theo Vietnam+

- **[nav]** doc_id `61702` · 322 ký tự, 8 dòng body · title: `[Video] Nữ nhân viên y tế đầu tiên ở Hải Dương được tiêm vaccine Covid-19`  
  https://baohaiphong.vn/video-nu-nhan-vien-y-te-dau-tien-o-hai-duong-duoc-tiem-vaccine-covid-19-196870.html  
  đầu: [Video] Nữ nhân viên y tế đầu tiên ở Hải Dương được tiêm vaccine Covid-19⏎⏎Sáng 8.3, mũi tiêm vaccine Covid-19 đầu tiên dành cho chị Phạm Thị Tuyết Nhung, cán bộ Trung tâm Y tế TP Hải Dương.⏎⏎Xem clip⏎⏎Theo Zing⏎⏎Bài liên quan⏎⏎[Video] Nữ nhân viên y tế đầu tiên ở Hải Dương được tiêm vaccine Covid-19⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định  
  cuối: …Covid-19 đầu tiên dành cho chị Phạm Thị Tuyết Nhung, cán bộ Trung tâm Y tế TP Hải Dương.⏎⏎Xem clip⏎⏎Theo Zing⏎⏎Bài liên quan⏎⏎[Video] Nữ nhân viên y tế đầu tiên ở Hải Dương được tiêm vaccine Covid-19⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định

- **[nav]** doc_id `53690` · 414 ký tự, 8 dòng body · title: `[Infographics] Đợt dịch COVID-19 thứ 4 có nhiều ca mắc trong nước nhất`  
  https://baohaiphong.vn/infographics-dot-dich-covid-19-thu-4-co-nhieu-ca-mac-trong-nuoc-nhat-190735.html  
  đầu: [Infographics] Đợt dịch COVID-19 thứ 4 có nhiều ca mắc trong nước nhất⏎⏎Theo Bộ Y tế, đợt dịch COVID-19 thứ 4 (từ 27.4 đến nay) là đợt có nhiều ca mắc trong nước nhất với 1.502 ca.⏎⏎Tính đến 6 giờ ngày 19.5, Việt Nam có tổng cộng 3.072 ca mắc COVID-19 ghi nhận trong nước và 1.471 ca nhập cảnh.⏎⏎Theo TTXVN⏎⏎Bài liên quan⏎⏎[Infographics] Đợt dịch COVID-19 thứ 4 có nhiều ca mắc tr  
  cuối: …iờ ngày 19.5, Việt Nam có tổng cộng 3.072 ca mắc COVID-19 ghi nhận trong nước và 1.471 ca nhập cảnh.⏎⏎Theo TTXVN⏎⏎Bài liên quan⏎⏎[Infographics] Đợt dịch COVID-19 thứ 4 có nhiều ca mắc trong nước nhất⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định

- **[nav]** doc_id `53702` · 488 ký tự, 8 dòng body · title: `[Infographics] Không chủ quan với diễn biến nặng bệnh sốt xuất huyết`  
  https://baohaiphong.vn/infographics-khong-chu-quan-voi-dien-bien-nang-benh-sot-xuat-huyet-176515.html  
  đầu: [Infographics] Không chủ quan với diễn biến nặng bệnh sốt xuất huyết⏎⏎Sốt xuất huyết khác với các bệnh truyền nhiễm khác là diễn biến nặng thường vào ngày thứ 4 đến ngày thứ 7 của bệnh, thời điểm bệnh nhân thường giảm hoặc hết sốt.⏎⏎Do đó, việc chăm sóc người bệnh là không thể chủ quan. Bệnh nhân cần được chẩn đoán sớm ngay từ đầu để được theo dõi và điều trị phù hợp.⏎⏎Theo TTX  
  cuối: …không thể chủ quan. Bệnh nhân cần được chẩn đoán sớm ngay từ đầu để được theo dõi và điều trị phù hợp.⏎⏎Theo TTXVN⏎⏎Bài liên quan⏎⏎[Infographics] Không chủ quan với diễn biến nặng bệnh sốt xuất huyết⏎⏎-⏎⏎Cỡ chữ⏎⏎Mặc định

- **[rand]** doc_id `55240` · 2892 ký tự, 11 dòng body · title: `Nạn nhân nặng nhất vụ cháy chung cư mini chưa hồi phục trí nhớ`  
  https://baohaiphong.vn/nan-nhan-nang-nhat-vu-chay-chung-cu-mini-chua-hoi-phuc-tri-nho-400838.html  
  đầu: Nạn nhân nặng nhất vụ cháy chung cư mini chưa hồi phục trí nhớ⏎⏎Thiếu tá Nguyễn Văn Chương, 38 tuổi, nạn nhân nặng nhất vụ cháy chung cư mini ở Thanh Xuân hiện hồi phục về vận động, thể lực, song trí nhớ vẫn rối loạn.⏎⏎Ngày 18/12, anh Chương cùng bố đến Bệnh viện Bạch Mai, nhân dịp nơi này tổ chức tái khám cho tất cả nạn nhân trong vụ cháy chung cư mini ở Khương Hạ xảy ra tháng  
  cuối: …cháy chung cư mini ở Khương Hạ xảy ra nửa đêm 12/9/2023. Chung cư này gồm 9 tầng, một tum, với khoảng 45 căn hộ, hơn 150 nhân khẩu. Công an Hà Nội xác định 56 người tử vong, 37 trường hợp bị thương.⏎⏎T.H (theo VnExpress)

- **[rand]** doc_id `51128` · 1377 ký tự, 5 dòng body · title: `Đẩy mạnh tuyên truyền vận động người dân tiêm mũi 3 vaccine phòng Covid-19`  
  https://baohaiphong.vn/day-manh-tuyen-truyen-van-dong-nguoi-dan-tiem-mui-3-vaccine-phong-covid-19-156751.html  
  đầu: Đẩy mạnh tuyên truyền vận động người dân tiêm mũi 3 vaccine phòng Covid-19⏎⏎Gần 12% số trẻ từ 5 đến dưới 12 tuổi ở Hải Dương đã được tiêm vaccine phòng Covid-19, chưa xuất hiện trường hợp nào gặp biến chứng⏎⏎Tính đến ngày 4.5, toàn tỉnh có 901.741 người từ 18 tuổi trở lên đã tiêm 3 mũi vacicne, đạt 67,09%. Các huyện Cẩm Giàng, Kim Thành dẫn đầu toàn tỉnh về tỷ lệ người đã tiêm   
  cuối: …hưa có hướng dẫn cụ thể.⏎⏎Đến nay, toàn tỉnh đã có 28.524 trẻ từ 5 đến dưới 12 tuổi tiêm vaccine phòng Covid-19, đạt gần 12%. Các địa phương đang chờ đợt phân bổ vaccine tiếp theo để tiếp tục tiêm cho nhóm đối tượng này.

- **[rand]** doc_id `48496` · 1419 ký tự, 8 dòng body · title: `Bệnh viện Quốc tế Sản nhi Hải Phòng - Bệnh viện thẩm mỹ Kangnam: Hợp tác chuyên môn trong `  
  https://baohaiphong.vn/benh-vien-quoc-te-san-nhi-hai-phong-benh-vien-tham-my-kangnam-hop-tac-chuyen-mon-trong-linh-vuc-phau-thuat-tham-my-456521.html  
  đầu: Bệnh viện Quốc tế Sản nhi Hải Phòng - Bệnh viện thẩm mỹ Kangnam: Hợp tác chuyên môn trong lĩnh vực phẫu thuật thẩm mỹ⏎⏎(HPĐT)- Sáng 3-4, Bệnh viện Quốc tế Sản nhi Hải Phòng phối hợp Bệnh viện thẩm mỹ Kangnam tổ chức hội thảo phẫu thuật thẩm mỹ, chia sẻ, tư vấn những kinh nghiệm trong lĩnh vực làm đẹp .⏎⏎Đông đảo chị em phụ nữ tới dự hội thảo và đăng ký các gói làm đẹp.⏎⏎Tại buổ  
  cuối: …trong hội thảo, Bệnh viện Quốc tế Sản nhi Hải Phòng và Bệnh viện thẩm mỹ Kangnam ký kết hợp tác về cơ sở vật chất, chuyên môn, đặc biệt trong chuyên khoa gây mê và Khoa Phẫu thuật tạo hình thẩm mỹ./.⏎⏎Tin và ảnh: Đỗ Hiền

- **[rand]** doc_id `55055` · 4357 ký tự, 19 dòng body · title: `Một người mắc bệnh sởi có thể lây cho 12 - 18 người`  
  https://baohaiphong.vn/mot-nguoi-mac-benh-soi-co-the-lay-cho-12-18-nguoi-407341.html  
  đầu: Một người mắc bệnh sởi có thể lây cho 12 - 18 người⏎⏎Theo Bộ Y tế, từ đầu năm đến nay cả nước ghi nhận 40.000 trường hợp nghi mắc bệnh sởi, có 5 trường hợp tử vong liên quan đến sởi. 1 người mắc bệnh có thể lây cho 12 - 18 người khác.⏎⏎Chiều 15/3, Bộ Y tế tổ chức hội nghị trực tuyến toàn quốc về công tác phòng chống bệnh sởi.⏎⏎Tại hội nghị, Bộ trưởng Y tế Đào Hồng Lan cho biết   
  cuối: … huy động sự vào cuộc của các cấp, các ngành; không chỉ riêng ngành y tế. Các địa phương chuẩn bị sẵn cơ sở vật chất, trang thiết bị, vật tư y tế, nhất là các vùng có nguy cơ cao để xử lý ngay dịch bệnh.⏎⏎VN (theo TTXVN)

- **[rand]** doc_id `46374` · 899 ký tự, 3 dòng body · title: `13 F1 ở xã Đại Đức hoàn thành cách ly tập trung`  
  https://baohaiphong.vn/13-f1-o-xa-dai-duc-hoan-thanh-cach-ly-tap-trung-173626.html  
  đầu: 13 F1 ở xã Đại Đức hoàn thành cách ly tập trung⏎⏎Từ ngày 1.11, điểm chợ cóc thôn Nguyễn Bạo hoạt động trở lại⏎⏎Trước đó, từ ngày 1.11, việc mua bán tại khu vực chợ cóc thôn Nguyễn Bạo đã hoạt động trở lại; toàn bộ trẻ em tại điểm Trường Mầm non xã ở thôn Đại Đức được trở lại trường; học sinh 2 lớp 1A, 3A3, Trường Tiểu học Đại Xuyên trở lại học trực tiếp. Gần 200 trường hợp F2 c  
  cuối: … 19. Sau đó, anh C. được xác định là bệnh nhân Covid-19. Qua truy vết, xã Đại Đức đã phát hiện 13 trường hợp F1, gần 200 F2 liên quan đến anh Q.V.C. Xã đã quyết định tạm dừng một số hoạt động để bảo đảm phòng chống dịch.

- **[rand]** doc_id `60799` · 4652 ký tự, 10 dòng body · title: `Trong ngày 5-11-2021, Hải Phòng ghi nhận 7 ca mắc COVID-19 mới`  
  https://baohaiphong.vn/trong-ngay-5-11-2021-hai-phong-ghi-nhan-7-ca-mac-covid-19-moi-466470.html  
  đầu: Trong ngày 5-11-2021, Hải Phòng ghi nhận 7 ca mắc COVID-19 mới⏎⏎(HPĐT)- Tối 5-11, Sở Y tế thông tin về 7 ca bệnh ghi nhận trong ngày và biện pháp phòng, chống dịch COVID-19. Cụ thể, gồm:⏎⏎Lãnh đạo Sở Y tế làm việc với chính quyền huyện An Dương, xã An Đồng khi ghi nhận ca mắc COVID-19 trên địa bàn.⏎⏎* Ca bệnh tại quận Lê Chân là V.T. D., nữ;sinh năm 1982, nghề nghiệp: nhân viên  
  cuối: … triển khai các biện pháp phòng, chống dịch, điều tra truy vết, cách ly y tế, lấy mẫu xét nghiệm đối với người tiếp xúc gần và phun thuốc khử khuẩn, phong tỏa tạm thời khu vực nhà của ca bệnh..../.⏎⏎Tin và ảnh: Minh Minh

## baonghean.vn (15,728 doc)

- **[short]** doc_id `66219` · 377 ký tự, 1 dòng body · title: `7 bệnh nguy hiểm trẻ nhỏ dễ mắc phải vào mùa đông`  
  https://baonghean.vn/7-benh-nguy-hiem-tre-nho-de-mac-phai-vao-mua-dong-10187087.html  
  đầu: 7 bệnh nguy hiểm trẻ nhỏ dễ mắc phải vào mùa đông⏎⏎(Baonghean.vn) - Mùa đông là thời điểm thuận lợi để nhiều loại siêu vi trùng phát triển và hoành hành. Vào thời điểm này, trẻ nhỏ rất dễ mắc phải những căn bệnh như viêm phế quản, tiêu chảy, quai bị hay hạ thân nhiệt… Dưới đây là những nguyên nhân, cách phòng tránh bệnh cha mẹ cần biết để bảo vệ sức khỏe cho con mùa giá rét.  
  cuối: … thời điểm này, trẻ nhỏ rất dễ mắc phải những căn bệnh như viêm phế quản, tiêu chảy, quai bị hay hạ thân nhiệt… Dưới đây là những nguyên nhân, cách phòng tránh bệnh cha mẹ cần biết để bảo vệ sức khỏe cho con mùa giá rét.

- **[short]** doc_id `72610` · 313 ký tự, 5 dòng body · title: `Hướng dẫn làm mặt nạ sữa chua để tái tạo da sau 1 tuần làm việc căng thẳng`  
  https://baonghean.vn/huong-dan-lam-mat-na-sua-chua-de-tai-tao-da-sau-1-tuan-lam-viec-cang-thang-10194222.html  
  đầu: Hướng dẫn làm mặt nạ sữa chua để tái tạo da sau 1 tuần làm việc căng thẳng⏎⏎Hướng dẫn làm mặt nạ sữa chua để tái tạo da sau 1 tuần làm việc căng thẳng⏎⏎Sữa chua và bột nghệ đều chứa nhiều chất chống oxy hóa cùng các loại vitamin, có khả năng cải thiện làn da xỉn màu hiệu quả.⏎⏎ONECMS - A PRODUCT OF⏎⏎×⏎⏎Đăng nhập  
  cuối: … nạ sữa chua để tái tạo da sau 1 tuần làm việc căng thẳng⏎⏎Sữa chua và bột nghệ đều chứa nhiều chất chống oxy hóa cùng các loại vitamin, có khả năng cải thiện làn da xỉn màu hiệu quả.⏎⏎ONECMS - A PRODUCT OF⏎⏎×⏎⏎Đăng nhập

- **[short]** doc_id `66415` · 499 ký tự, 8 dòng body · title: `7 món cháo giải cảm cực nhanh và hiệu quả`  
  https://baonghean.vn/7-mon-chao-giai-cam-cuc-nhanh-va-hieu-qua-10181978.html  
  đầu: 7 món cháo giải cảm cực nhanh và hiệu quả⏎⏎(Baonghean.vn) - Cảm cúm, cảm lạnh là bệnh thường gặp lúc giao mùa - khi tiết trời nắng mưa thất thường làm suy yếu hệ miễn dịch. Ngoài uống thuốc theo chỉ dẫn của bác sỹ, bạn cũng cần bổ sung nguồn dinh dưỡng để bệnh chóng khỏi. Dưới đây là 5 món cháo chế biến rất đơn giản nhưng công dụng giải cảm lại rất hiệu quả.⏎⏎#1. Cháo thịt bằm   
  cuối: …ây là 5 món cháo chế biến rất đơn giản nhưng công dụng giải cảm lại rất hiệu quả.⏎⏎#1. Cháo thịt bằm gừng tươi⏎⏎#2. Cháo hành⏎⏎#3. Cháo cải cúc⏎⏎#4. Cháo lá tía tô⏎⏎#5. Cháo táo đỏ, bí ngô⏎⏎#6. Cháo đậu xanh⏎⏎#7. Cháo gà

- **[short]** doc_id `72701` · 485 ký tự, 3 dòng body · title: `[Infographic] Làm sao để lựa chọn hải sản an toàn?`  
  https://baonghean.vn/infographic-lam-sao-de-lua-chon-hai-san-an-toan-10026744.html  
  đầu: [Infographic] Làm sao để lựa chọn hải sản an toàn?⏎⏎(Baonghean.vn) - Bộ Y tế vừa đưa ra khuyến cáo các loại hải sản sống ở tầng nổi, hải sản tại đầm nuôi của 4 tỉnh miền Trung (Hà Tĩnh, Quảng Bình, Quảng Trị, Thừa Thiên Huế) đều đảm bảo an toàn để sử dụng làm thực phẩm. Ngược lại, các loại hải sản sống ở tầng đáy trong vòng 20 hải lý chưa nên sử dụng.⏎⏎Cùng tìm hiểu các loại hả  
  cuối: …phẩm. Ngược lại, các loại hải sản sống ở tầng đáy trong vòng 20 hải lý chưa nên sử dụng.⏎⏎Cùng tìm hiểu các loại hải sản đã được kiểm nghiệm là an toàn và tránh những loại được Bộ Y tế khuyến cáo không nên ăn:⏎⏎Phan Minh

- **[short]** doc_id `80241` · 439 ký tự, 2 dòng body · title: `Xã Quỳnh Nghĩa xây trạm y tế cao tầng gần 1 tỷ đồng`  
  https://baonghean.vn/xa-quynh-nghia-xay-tram-y-te-cao-tang-gan-1-ty-dong-10029330.html  
  đầu: Xã Quỳnh Nghĩa xây trạm y tế cao tầng gần 1 tỷ đồng⏎⏎Trạm y tế xã Quỳnh Nghĩa có khuôn viên thoáng mát, một ngôi nhà cấp 4, giếng nước, nhà vệ sinh và vườn trồng thuốc nam. Các bác sĩ, y sĩ, y tá, hộ lý ở trạm thời gian qua đã làm tốt công tác chăm sóc sức khỏe cho dân nên không có dịch bệnh xảy ra. Trạm coi trọng việc tiêm chủng phòng bệnh cho các cháu mầm non, và phối hợp với  
  cuối: … qua đã làm tốt công tác chăm sóc sức khỏe cho dân nên không có dịch bệnh xảy ra. Trạm coi trọng việc tiêm chủng phòng bệnh cho các cháu mầm non, và phối hợp với Hội Người cao tuổi khám bệnh cho các cụ ông, cụ bà.⏎⏎Tú An

- **[title]** doc_id `74006` · 6205 ký tự, 18 dòng body · title: `Mẹo trị ho`  
  https://baonghean.vn/meo-tri-ho-10014190.html  
  đầu: Mẹo trị ho⏎⏎Cùng nghe những bà mẹ "người thật việc thật" kể về kinh nghiệm "đánh bay" cơn ho của con. Vì vẫn đang trong giai đoạn hoàn thiện và phát triển, hệ thống miễn dịch còn kém nên gần như cứ mỗi lần chuyển mùa, trẻ lại dễ bị lạnh, cảm sốt, ho hắng. Nếu để kéo dài dễ dẫn đến các bệnh về đường hô hấp. Thông thường mỗi khi trẻ bị ho, mẹ Việt thường rất “lười” đưa con đến bệ  
  cuối: … mẹ nên lưu ý chỉ dùng khi con bị ho, có đờm nhẹ và vẫn cần theo dõi phản ứng của trẻ. Ăn không ngon, tâm trạng chơi đùa không tốt, thì cần thiết, nên đưa con đến bệnh viện, không được sử dụng thuốc tuỳ tiện. Theo eva.vn

- **[title]** doc_id `77459` · 967 ký tự, 72 dòng body · title: `https://baonghean.vn/404`  
  https://baonghean.vn/sang-choi-nha-hang-xom-be-trai-9-tuoi-bi-cho-de-can-vao-duong-vat-10223805.html  
  đầu: https://baonghean.vn/404⏎⏎Tiếng Việt⏎⏎Tiếng Việt⏎⏎English⏎⏎Français⏎⏎Русский⏎⏎Nghe An News⏎⏎Nghệ An Nhật báo⏎⏎Nghệ AN cuối tuần⏎⏎Đăng nhập⏎⏎Báo Nghệ An:⏎⏎Truyền hình Nghệ An:⏎⏎Mới nhất⏎⏎Thời sự⏎⏎Diễn đàn⏎⏎Nhân sự⏎⏎Chính sách mới⏎⏎Xây dựng Đảng⏎⏎Bảo vệ nền tảng tư tưởng⏎⏎Đưa nghị quyết vào cuộc sống⏎⏎Kinh tế⏎⏎Công nghiệp⏎⏎Nông nghiệp⏎⏎Thị trường⏎⏎Bất động sản⏎⏎Xe⏎⏎Quốc tế⏎⏎Bình   
  cuối: …ideo⏎⏎Suy ngẫm⏎⏎Chuyển đổi số⏎⏎Công nghệ⏎⏎Không tìm thấy đường dẫn hoặc thông tin này⏎⏎Bạn có thể quay về⏎⏎trang chủ⏎⏎để tìm kiếm thông tin⏎⏎POWERED BY⏎⏎ONE⏎⏎CMS⏎⏎- A PRODUCT OF⏎⏎NEKO⏎⏎×⏎⏎Đăng nhập⏎⏎Đăng nhập bằng Google

- **[title]** doc_id `79012` · 717 ký tự, 7 dòng body · title: `Tin buồn`  
  https://baonghean.vn/tin-buon-10029357.html  
  đầu: Tin buồn⏎⏎Ban lễ tang và gia đình vô cùng thương tiếc báo tin:⏎⏎Cụ Vũ Thị Chắt, thường gọi bà Giáo Dương⏎⏎Sinh năm 1916 tại làng Hậu Luật - xã Diễn Bình - huyện Diễn Châu - tỉnh Nghệ An. Thường trú: Xóm 5 – thị trấn Yên Thành – huyện Yên Thành - tỉnh Nghệ An.⏎⏎Cán bộ lão thành Cách mạng (từ 1934).⏎⏎Sau nhiều năm lâm bệnh nặng, mặc đù đã được các y bác sỹ, bà con và con cháu tận  
  cuối: …ng 04 năm Tân Mão.⏎⏎Lễ viếng từ 09 giờ, ngày 08 tháng 05 năm 2011. Lễ truy điệu và đưa tang được cử hành vào 13 giờ 30nphút, ngày 09 tháng 05 năm 2011. An táng tại Nghĩa trang thị trấn Yên Thành.⏎⏎Ban lễ tang và gia đình

- **[title]** doc_id `72916` · 909 ký tự, 5 dòng body · title: `Khó...!`  
  https://baonghean.vn/kho-10030248.html  
  đầu: Khó...!⏎⏎"Để bảo đảm sức khỏe cho bản thân và gia đình, mỗi bà nội trợ hãy là một người tiêu dùng thông thái!". Thế nhưng, để trở thành "người tiêu dùng thông thái" trong thời buổi này thật quá khó!⏎⏎Hiện nay, tình trạng vi phạm ATVSTP ngày càng tinh vi hơn, khiến người tiêu dùng không thể phòng tránh nổi. Nào là ruốc bông làm từ bã sắn dây, mực khô làm từ cao su, thịt lợn ngâm  
  cuối: …n ngoài chứ lấy căn cứ đâu để mua thực phẩm "sạch"?⏎⏎Khi người tiêu dùng không thể "thông thái", xem ra phải nhờ đến cơ quan chức năng vào cuộc để bảo vệ người tiêu dùng. Đừng để Tháng ATVSTP chỉ là khẩu hiệu...⏎⏎Duy Nam

- **[nav]** doc_id `78561` · 1723 ký tự, 23 dòng body · title: `Thu hồi 17 loại mỹ phẩm không đạt chuẩn`  
  https://baonghean.vn/thu-hoi-17-loai-my-pham-khong-dat-chuan-10069390.html  
  đầu: Thu hồi 17 loại mỹ phẩm không đạt chuẩn⏎⏎Cục quản lý Dược (Bộ Y tế) vừa ra quyết định xử phạt, thu hồi 17 loại mỹ phẩm không đạt chuẩn.⏎⏎Một dòng sản phẩm của EBC bị thu hồi⏎⏎Ông Nguyễn Tất Đạt, Phó Cục trưởng Cục Quản lý Dược cho biết, 17 sản phẩm này đều do Công ty TNHH EBC Việt Nam (số 262/5 Cách Mạng Tháng Tám, phường 10, quận 3, Tp. Hồ Chí Minh) sản xuất.⏎⏎Công ty này đã c  
  cuối: …U⏎⏎4. SKIN LIGHTENING⏎⏎5. MELASMA PRO⏎⏎6. MELASMA CLEARR⏎⏎7. DAILY UV CARE⏎⏎8. ACNEPRO⏎⏎9. SPOTLESS ACNE⏎⏎10. BODY WHITE SHOWER⏎⏎11. FRESH SKIN⏎⏎12. EYES CARE⏎⏎13. HEEL CARE⏎⏎14. STRETCH MARKS⏎⏎15. LIP CARE⏎⏎Theo Dân trí

- **[nav]** doc_id `71855` · 967 ký tự, 72 dòng body · title: `https://baonghean.vn/404`  
  https://baonghean.vn/gap-con-dia-chui-vao-duong-vat-cu-ong-71-tuoi-10254722.html  
  đầu: https://baonghean.vn/404⏎⏎Tiếng Việt⏎⏎Tiếng Việt⏎⏎English⏎⏎Français⏎⏎Русский⏎⏎Nghe An News⏎⏎Nghệ An Nhật báo⏎⏎Nghệ AN cuối tuần⏎⏎Đăng nhập⏎⏎Báo Nghệ An:⏎⏎Truyền hình Nghệ An:⏎⏎Mới nhất⏎⏎Thời sự⏎⏎Diễn đàn⏎⏎Nhân sự⏎⏎Chính sách mới⏎⏎Xây dựng Đảng⏎⏎Bảo vệ nền tảng tư tưởng⏎⏎Đưa nghị quyết vào cuộc sống⏎⏎Kinh tế⏎⏎Công nghiệp⏎⏎Nông nghiệp⏎⏎Thị trường⏎⏎Bất động sản⏎⏎Xe⏎⏎Quốc tế⏎⏎Bình   
  cuối: …ideo⏎⏎Suy ngẫm⏎⏎Chuyển đổi số⏎⏎Công nghệ⏎⏎Không tìm thấy đường dẫn hoặc thông tin này⏎⏎Bạn có thể quay về⏎⏎trang chủ⏎⏎để tìm kiếm thông tin⏎⏎POWERED BY⏎⏎ONE⏎⏎CMS⏎⏎- A PRODUCT OF⏎⏎NEKO⏎⏎×⏎⏎Đăng nhập⏎⏎Đăng nhập bằng Google

- **[nav]** doc_id `77708` · 4614 ký tự, 57 dòng body · title: `Sự thật về phương pháp 'lột da' làm trắng cấp tốc`  
  https://baonghean.vn/su-that-ve-phuong-phap-lot-da-lam-trang-cap-toc-10165677.html  
  đầu: Sự thật về phương pháp 'lột da' làm trắng cấp tốc⏎⏎Dược sĩ Phạm Minh Hữu Tiến - 28/02/2018 07:23⏎⏎Các tai biến của thay da hóa học có thể chỉ gây bỏng nhưng cũng có trường hợp bị nhiễm trùng, tạo sẹo.⏎⏎Mong muốn sở hữu làn da trắng sau thời gian ngắn, nhiều chị em không ngần ngại tìm đến những giải pháp làm đẹp tức thời bằng cách “lột da”.⏎⏎Lột da hóa học là gì?⏎⏎Có tên thuật n  
  cuối: …:39⏎⏎-⏎⏎23/08/2025 16:56⏎⏎-⏎⏎23/08/2025 15:13⏎⏎-⏎⏎20/08/2025 15:00⏎⏎-⏎⏎20/08/2025 12:00⏎⏎-⏎⏎17/08/2025 22:03⏎⏎-⏎⏎17/08/2025 17:09⏎⏎-⏎⏎17/08/2025 15:51⏎⏎-⏎⏎13/08/2025 18:30⏎⏎POWERED BY⏎⏎ONECMS - A PRODUCT OF⏎⏎×⏎⏎Đăng nhập

- **[nav]** doc_id `68466` · 1440 ký tự, 23 dòng body · title: `Bí quyết giúp da trắng nõn nà với bí đao`  
  https://baonghean.vn/bi-quyet-giup-da-trang-non-na-voi-bi-dao-10022845.html  
  đầu: Bí quyết giúp da trắng nõn nà với bí đao⏎⏎Bí đao có rất nhiều công dụng trong việc làm đẹp lẫn chăm sóc sức khỏe. Có thể nói đến công dụng nổi trội chính là làm trắng da.⏎⏎Cách 1⏎⏎Chuẩn bị:⏎⏎500g bí đao⏎⏎500 dưa hấu⏎⏎Đường trắng.⏎⏎Nước ép bí đao bổ sung vitamin giúp trắng da từ bên trong⏎⏎Cách làm: bí đao gọt bỏ vỏ và ruột sau đó rửa sạch, củ cải gọt bỏ vỏ, dưa hấu gọt lấy ruột  
  cuối: … một ít nước này và thoa đều lên da mặt. Đây chính là cách dưỡng làm trắng da bên ngoài với bí đao. Sáng hôm sau rửa lại thật sạch. Nó sẽ giúp bạn sán da, chống và trị nám, làm mờ vết thâm cực hiệu quả.⏎⏎Theo Alobacsi.vn

- **[nav]** doc_id `69020` · 1660 ký tự, 41 dòng body · title: `Các loại quả dễ nhiễm thuốc trừ sâu nhất`  
  https://baonghean.vn/cac-loai-qua-de-nhiem-thuoc-tru-sau-nhat-10015140.html  
  đầu: Các loại quả dễ nhiễm thuốc trừ sâu nhất⏎⏎Nhóm Công tác Môi trường của Mỹ vừa công bố danh sách cập nhật các loại trái cây dễ bị phơi nhiễm nhiều thuốc trừ sâu nhất và kèm theo một danh sách các loại trái cây sạch nhất.⏎⏎Để hoàn thành danh sách này, Nhóm Công tác Môi trường đã tập trung vào 48 loại trái cây, rau quả phổ biến và dựa trên một phân tích gồm 32.000 mẫu được kiểm tr  
  cuối: …u⏎⏎- 89% dứa, 82% kiwi, 80% đu dủ, 88% xoài, 61% dưa đỏ không có dư lượng thuốc trừ sâu.⏎⏎Hai danh sách trên giúp cho người tiêu dùng, đặc biệt là các bà nội trợ có lựa chọn tốt nhất cho sức khoẻ gia đình mình!⏎⏎Theo NLĐ

- **[rand]** doc_id `74435` · 6633 ký tự, 23 dòng body · title: `Nâng cao hiệu quả phát hiện, điều trị dự phòng bệnh lao trẻ em tại Nghệ An`  
  https://baonghean.vn/nang-cao-hieu-qua-phat-hien-dieu-tri-du-phong-benh-lao-tre-em-tai-nghe-an-10155698.html  
  đầu: Nâng cao hiệu quả phát hiện, điều trị dự phòng bệnh lao trẻ em tại Nghệ An⏎⏎(Baonghean) - Với mục tiêu giảm tỷ lệ mắc và tử vong do lao ở trẻ em tại Nghệ An, từ tháng 3/2016 Bệnh viện Lao và Bệnh phổi Nghệ An phối hợp với tổ chức PATH, Johnson & Johnson Việt Nam triển khai dự án “Hơi thở cuộc sống”. Đến nay, số trẻ được phát hiện lao, dự phòng INH tại 4 huyện của dự án tăng từ   
  cuối: …hêm: “Nghệ An tiếp tục phát huy kết quả dự án, đồng thời tiếp tục mở rộng triển khai rộng khắp trên địa bàn tỉnh; để công tác phòng chống lao trẻ em và phòng chống lao nói chung đi vào chiều sâu và lan tỏa”.⏎⏎Đinh Nguyệt

- **[rand]** doc_id `67778` · 3739 ký tự, 12 dòng body · title: `Bé gái tuổi mẫu giáo đã bị viêm phần phụ, bác sĩ chỉ ra nguyên nhân gây bệnh bất ngờ`  
  https://baonghean.vn/be-gai-tuoi-mau-giao-da-bi-viem-phan-phu-bac-si-chi-ra-nguyen-nhan-gay-benh-bat-ngo-10223404.html  
  đầu: Bé gái tuổi mẫu giáo đã bị viêm phần phụ, bác sĩ chỉ ra nguyên nhân gây bệnh bất ngờ⏎⏎Người mẹ cho rằng em bé bị xâm hại tình dục, tuy nhiên, khi bác sĩ khám lâm sàng thì không thấy dấu hiệu xâm hại. Vậy đâu là nguyên nhân?⏎⏎BSCKII. Diêm Thị Thanh Thủy, Trưởng khoa Khám Sản tự nguyện, BV Phụ Sản Hà Nội cho biết, chị từng gặp nhiều trường hợp đến khám bệnh là các cháu bé 5 tuổi   
  cuối: … việc viêm nhiễm tái đi tái lại nhiều lần cần làm những việc đơn giản như vệ sinh phần phụ đúng cách, sử dụng thuốc theo chỉ định của bác sĩ, ngoài ra cần ăn uống đầy đủ, da dạng thực phẩm và tập luyện thể dục, thể thao.

- **[rand]** doc_id `74637` · 886 ký tự, 3 dòng body · title: `Nghệ An: 683 người vào viện khám nghi do tai nạn giao thông trong dịp nghỉ lễ`  
  https://baonghean.vn/nghe-an-683-nguoi-vao-vien-kham-nghi-do-tai-nan-giao-thong-trong-dip-nghi-le-10296552.html  
  đầu: Nghệ An: 683 người vào viện khám nghi do tai nạn giao thông trong dịp nghỉ lễ⏎⏎Tổng số bệnh nhân vào viện khám nghi do tai nạn giao thông là 683 người. Trong đó, có 204 người phải nhập viện điều trị nội trú; có 23 trường hợp nặng, nguy kịch.⏎⏎Thông tin từ Sở Y tế Nghệ An: Trong dịp nghỉ Lễ 30/4 - 01/5, từ ngày 29/4/2025 đến ngày 04/5/2025, toàn tỉnh không có ổ dịch bệnh mới nào  
  cuối: …ao thông là 683 người. Trong đó có 204 người phải nhập viện điều trị nội trú (giảm 30,6% so với năm 2024); không ghi nhận tử vong nội viện; có 03 trường hợp tử vong ngoài bệnh viện; hiện có 23 trường hợp nặng, nguy kịch.

- **[rand]** doc_id `69175` · 4455 ký tự, 36 dòng body · title: `Cách đảo ngược bệnh gan nhiễm mỡ bạn cần biết`  
  https://baonghean.vn/cach-dao-nguoc-benh-gan-nhiem-mo-ban-can-biet-10289676.html  
  đầu: Cách đảo ngược bệnh gan nhiễm mỡ bạn cần biết⏎⏎Bệnh gan nhiễm mỡ liên quan đến rối loạn chuyển hóa còn được gọi là bệnh gan nhiễm mỡ không do rượu. Điều này có nghĩa là có sự tích tụ chất béo trong gan.⏎⏎Các chuyên gia không hiểu đầy đủ lý do tại sao một số người mắc bệnh gan nhiễm mỡ và những người khác thì không.⏎⏎Nhưng bạn có nhiều khả năng mắc bệnh này nếu bạn thừa cân hoặc  
  cuối: …ười, nhưng khoa học vẫn chưa chứng minh được điều đó.⏎⏎Nếu bạn muốn thử loại thực phẩm bổ sung này, hãy trao đổi với bác sĩ trước. Nó có thể không an toàn cho tất cả mọi người và có liên quan đến ung thư tuyến tiền liệt.

- **[rand]** doc_id `79666` · 2540 ký tự, 10 dòng body · title: `Uống cà phê trộn bột pin nguy hiểm như thế nào?`  
  https://baonghean.vn/uong-ca-phe-tron-bot-pin-nguy-hiem-nhu-the-nao-10170521.html  
  đầu: Uống cà phê trộn bột pin nguy hiểm như thế nào?⏎⏎Liên quan đến 3 tấn cà phê bị nhuộm đen bằng bột pin mới được phát hiện, các chuyên gia cho biết, loại cà phê này ảnh hưởng rất lớn đến sức khỏe người uống.⏎⏎ThS. Hoàng Trọng Phú, giảng viên khoa Hóa ĐH Khoa học Tự nhiên TP.HCM cho rằng, trong quá trình tạo ra một viên pin khô như loại pin Con Ó, nhà sản xuất phải sử dụng hợp chấ  
  cuối: …oan khai nhận, cà phê sau khi được nhuộm, cơ sở đem rang, xay rồi đóng gói bán ra thị trường. Tính từ đầu năm 2018 đến nay, cơ sở của bà đã xuất bán ra thị trường hơn 3 tấn cà phê bẩn đã được nhuộm đen bằng pin như trên.

- **[rand]** doc_id `72746` · 1713 ký tự, 8 dòng body · title: `Kết quả điều tra vụ học sinh ngộ độc tại Mường Nhé`  
  https://baonghean.vn/ket-qua-dieu-tra-vu-hoc-sinh-ngo-doc-tai-muong-nhe-10050483.html  
  đầu: Kết quả điều tra vụ học sinh ngộ độc tại Mường Nhé⏎⏎Theo thông tin từ Cục An toàn Thực phẩm (Bộ Y tế), ngày 19/9, Chi cục an toàn vệ sinh thực phẩm tỉnh Điện Biên đã có kết quả điều tra ban đầu nguyên nhân vụ ngộ độc thực phẩm làm 10 học sinh tại Trường phổ thông dân tộc nội trú huyện Mường Nhé, tỉnh Điện Biên bị ngộ độc thực phẩm.⏎⏎Kết quả cho thấy, nguyên nhân ngộ độc nghi ng  
  cuối: … như truyền dịch, uống than hoạt, trợ tim mạch, hô hấp tại Trung tâm y tế huyện, điều tra, xử lý vụ ngộ độc thực phẩm.⏎⏎Đến sáng ngày 18/9/2013, tất cả 10 bệnh nhân đã hồi phục sức khỏe và ra viện./.⏎⏎Theo Vietnam+ - M.Đ
