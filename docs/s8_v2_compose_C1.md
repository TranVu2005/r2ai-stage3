# S8 — Bản ghép C1/C2, candidate zh đợt 2

C1 đã gửi Vu trước C2. Cả hai đã qua đủ gate; chưa tự nộp. C2 chờ Vu quyết định sau điểm C1. Không GPU/crawl/push; chỉ commit docs.

## Artifact và cấu hình

- C1: `out/diag-lang/v2/compose/C1/C1.zip`, **82,047,434 B**; native **0 lỗi / 1.200 query / 126.208 chunk**.
- C2: `out/diag-lang/v2/compose/C2/C2.zip`, **95,046,678 B**; native **0 lỗi / 1.200 query / 146.814 chunk**.
- Cả hai giữ toàn bộ base K150, ZIP-LZMA dict 256 MiB, lc/lp/pb 3/0/2, props `5d00000010`; trần 102.600.000 B. Không cắt chunk vi hoặc thêm ID bảo vệ.
- C1 base S6-f-K150: LB do Vu cung cấp Final 0,2599; Doc F2/P/R 0,2593/0,1603/0,3630; Chunk F2/P/R 0,2605/0,1858/0,3366.
- C2 base S6-h2: Final 0,2601, doc như F; Chunk F2/P/R 0,2608/0,1704/0,3547. `relevant_docs` của hai base được đối chiếu giống hệt đủ 1.200 query.
- Candidate đợt 2 đủ 60.000 dòng/1.200 query; coverage **144.097/240.000 (60,040417%)**. τ_doc tuyệt đối **−0,87255859375**, cap **30 doc physical/query**; không hiệu chỉnh theo mean. Chunk ZL: τ 0,96/top-3 doc physical, nguyên `doc_text`, mỗi ID v2 một chunk, chỉ doc đã chọn.

## Số zh thực tế

- C1/C2 cùng thêm **18.526 doc physical / 18.527 ID / 1.770 chunk**: **15,438333 doc / 15,439167 ID / 1,475000 chunk/query**.
- So với ZXX-G: **+526 doc/ID, +64 chunk**, tương ứng **+0,438333 doc và +0,053333 chunk/query**.
- Actual archive C1/C2 có doc arrays và phần chunk zh giống hệt cả thứ tự lẫn doc_id/SHA text nguyên trên 1.200 query. Tổng doc entries mỗi file: 182.306.
- So đợt 1: doc thêm 3.256/thay ra 2.730/giữ 15.271 ID; body thêm 342/thay ra 278/giữ 1.428. Base F có 0 doc zh, nên counterfactual không đếm trùng.

## Dự báo C1 — chưa đo LB

Mốc Vu duyệt: 0,2599 + (0,2639 − 0,2595) = **0,2643**. Precision ID doc thay ra 5,988226%; chunk vào/ra 16,215051%; P/R base F là số đo thật. Chuyển delta zh đo trên G sang giả định F+đợt 1; tính macro từng query với Gdoc chung 61,104348 và ηchunk 79,491028. Đây là sensitivity có điều kiện, không phải precision gold xác định hay khoảng tin cậy.

| Precision doc thêm | ΔDoc F2 đợt 2 | ΔChunk F2 đợt 2 | Final dự báo | Final nếu chunk đợt 2 không tăng |
|---|---|---|---|---|
| 3% | -0.00094071 | +0.00005744 | 0.26385837 | 0.26382965 |
| 6% | +0.00003994 | +0.00005744 | 0.26434869 | 0.26431997 |
| 10% | +0.00134746 | +0.00005744 | 0.26500245 | 0.26497373 |

## SHA256 và bằng chứng

- C1 ZIP `e7c84e2c37c996fe3d046bf22a4d25b2fd9e8e1c2568d928fa45797108720c13`; JSON `83375428b79957b1580299f97b05299bb292957826bd4cb8bdc602a83b8dd5f3`.
- C2 ZIP `ec978a92e198dee79013580a54cfeced30aa626f7fbb37819ba01e19fb633e67`; JSON `917804b2d28a73338cd0bc846b5106160db92d397bf74223d4c98e34835f4fdd`.
- Base F ZIP `0f0bf619a326b812a2de4f5a971490c7147ca800ec2e7f691d12704cbc7e8b67`; JSON `d0da38b1ee8943b6cb3a5dbc6a7606b4ef7abd79b6cab39fe29567e3e763dca1` (pin đầy đủ từ stats và file thực).
- Base h2 ZIP `67678ae80026b17b1fb44e96ef4b2c9c38a201b73cd96bf8989b03c2b15b134d`; JSON `f1564c6b734ff16b5919ca204c8d046df8f8504f5f0c015e8e67f2c980cc3361`.
- Candidate `86d326173ce2e0ec5d6268a28416c7e6f9b2cb9f6fe11fd1371521d8d8ebd25c`.
- Raw manifest `8430a6303e57dd07135806341a1346eeef9434072a8276a3fbafd13dcee57517`; snapshot doc-source `0310c2cb16a845da86822ad1dbabecabe266dd6f9ab5ca8060558448b736582f`.
- Snapshot 57 nguồn nonempty: 37 giữ nguyên SHA từ đợt 1, 20 nguồn mới/24.069 doc kiểm CPU schema/v2/group/body/hash/count. Đây là gate nguồn doc, không đánh dấu dense/CSR index đã verified. Qihuang có 3 candidate row nhưng 0 vượt ngưỡng, không được thêm.
- Job/plan/approval/validator/vs-ZXX, source verification, parity và forecast ở hai thư mục output riêng. Corpus/query và SHA từng source trong plan được recheck sau native. `C2/final-gates-C1-C2.json` xác nhận lại cả hai file sau khi hoàn tất.
- Native giữ toàn bộ kiểm schema/v2/protection/dedupe/near/verbatim; CRC và giải nén SHA khớp. Suite mới **90/90 CPU test, 9,85 s**. Review cuối đã sửa gate SHA forecast qua RED→GREEN 5/5; không còn Critical/Important.
- Probe en trên C1 chờ S10 bàn giao candidate/pins/chunk nguyên văn; bỏ `title_only`, τ_en = Q75 top-1 như đã duyệt.
