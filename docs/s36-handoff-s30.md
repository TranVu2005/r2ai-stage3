# Bàn giao cho S30: hai bản 120A cố định (26/10)

LB (do Vu báo): **120A-p4b-v3 là bản chính** (khung EF2-cut-p4b, Jina đủ, ứng viên bản cuối), **120A-free-v2 là bản tổng quát** (khung free, pool D=300): Final 0,4101 (Doc 0,4402 / Chunk 0,3800), thua p4b-v3 0,0075 (≤ 0,01), nên giữ cả hai. Mốc tham chiếu cũ: 120A-p4b-v0 0,4111, 120A-free-v0 0,4033.

## Tệp bàn giao

* `configs/s36-handoff-120A-p4b-v3.json`, `configs/s36-handoff-120A-free-v2.json`: cấu hình cố định cho `python -m r2ai.s36.s37 apply --config <file>` (đường dẫn tường minh cho pairs_new, điểm 4B, C-max, Jina; không glob nên điểm nhập thêm về sau không đổi kết quả).
* `configs/s36-handoff-s30-manifest.json` (bản sao của `out/runs/s36-fuse4/handoff-s30/manifest.json`, SHA256 `45ce6e49…e1f7`): với mỗi bản, SHA256 của mọi tệp đầu vào do config liệt kê (`inputs_sha256`), của 19 tệp dữ liệu nền (`base_inputs_sha256`: union S31, Jina S35 và bản bổ sung S36, C-max S32 và bản bổ sung S36, `cal.json`, `ef2-free.json`, `pool.npz`, vector truy vấn, hai ZIP nền, `canonical-20261018/docs.parquet`, manifest S37 `995a77ed…8dff`), của đầu ra `apply` (docsel, carriers, pcal, keep), ZIP (kích thước, SHA256, T, cửa sổ, carrier sau thin, validator), commit mã.
* `scripts/s36/handoff_s30.py [--verify]`: sinh lại manifest; `--verify` chạy lại `apply` từ config cố định và so SHA256 của docsel, carriers, pcal, keep. Đã chạy 26/10: **cả hai bản trùng SHA từng byte**.

## Hai bản

| | 120A-p4b-v3 (chính) | 120A-free-v2 (tổng quát) |
|---|---|---|
| Khung | EF2-cut-p4b, pool = top-n_q cosine (n_q = round(1,3 K_q) của pool13 cũ), bốn nguồn RRF | free, pool = top-300 cosine của corpus, bốn nguồn RRF |
| Config | `pool: cosine13` | `pool: fixed`, `depth: 300` |
| a, b, G | a 0,5298, b −1,4097, G = s_thrC × (Σp_cal + đuôi) | cùng a, b, G = s_free × (Σp_cal top-211 + đuôi) |
| Doc | 192.742 doc, Doc F2 dự đoán 0,4206 | 220.599 doc, Doc F2 dự đoán 0,4250 (mô hình lệch ~−0,022 ở pool ngắn, không so trực tiếp) |
| Carrier | CR-fair 175.546, ρ 0,9076 | CR-fair 175.547, ρ 0,7931 |
| Jina / C-max | Jina bù đủ 1.002 query; không cặp nào thiếu | Jina bù 1.002 query, 31.645 cặp D=300 thêm xếp cuối; C-max đủ (có `cmax_scores-d300.parquet`) |
| ZIP | `120A-p4b-v3-full+w768-t1536.zip`, 99.417.831 B, SHA256 `34d235f6…37e2`, T 1536, 34.709 cửa sổ, 174.956 carrier sau thin | `120A-free-v2-full+w768-t1536.zip`, 100.285.047 B, SHA256 `fba176c6…da3f`, T 1536, 34.186 cửa sổ, 174.990 carrier sau thin |
| Validator | S26 ok, 0 lỗi, 9 segment 120ask làm nguồn verbatim | ok, 0 lỗi, 9 segment |

Dựng ZIP: `scripts/s36/run_120a_p4b_v3.sh`, `scripts/s36/run_120a_free_v2.sh` (S38 `compose run ... --rule full+w768 --carriers ... --fit-carriers`, rồi `compose validate`; mã S38 ở `feat/s38-chunk-metric-v2` commit `74722ad`, compose dùng commit `a4b57c1`, đã nằm trong nhánh này). Luật chunk: full+w768 (chunk nguyên thân + cửa sổ 768 token cho thân dài hơn T), T tự tìm = 1536 (sàn `FULL_MIN_T`), trần 102.600.000 B, thin theo nội dung, validator S26 `--max-chunks-per-doc 2`.

## Quy tắc chọn bản sao khi hoà cosine (từng khung)

Nhiều doc trùng nội dung (khác doc_id) có cosine bằng nhau tuyệt đối; khi khối hoà vắt qua mốc cắt pool thì bản sao nào vào pool phụ thuộc quy tắc xếp:

* **p4b-v3**: pool = n_q hàng đầu của union đã chấm điểm, xếp theo `(−cosine_score, order)`; `order` là cặp `(ưu tiên, id)`: `(0, hạng cosine pool13 cũ)` cho doc pool13 S29/S31, `(1, pair_id)` cho cặp pool13 bị đẩy ra khi gộp Pharmacity, `(2, pair_id)` cho cặp pool-extra S31, `(3, pair_id)` cho cặp S37 và S36 (pair_id tăng dần: S37 từ 1.000.000, final-4b từ 5.000.000). Một cặp mới mà doc đã có trong union của query thì bị bỏ. Mã: `s37.merge_union`, `s37.cosine13_pools`. Hoà: doc pool13 trước, rồi evicted, extra, cặp mới theo pair_id tăng.
* **free-v2**: pool = top-300 theo thứ tự corpus của `d300.merged_top`: cosine canonical + Pharmacity của `out/runs/s37/pool.npz` (top-400, thứ tự lưu, giảm dần) nối với các segment 120ask (tên segment tăng dần, thứ tự trong tệp vector), `argsort` ổn định theo −cosine; hoà: doc canonical trước (đúng thứ tự `pool.npz`), rồi doc 120ask theo tên segment và thứ tự tệp. Hàng union được xếp lại theo thứ tự này (`s37.fixed_pools(union, 300, order)`); doc nào của top-300 corpus thiếu điểm 4B thì báo lỗi. Trước sửa (25/10) có 20 query lệch 913 doc mỗi phía vì hai thứ tự khác nhau; sau sửa 0 lệch (`check_vs_corpus`).

Hai quy tắc khác nhau có chủ ý: khung p4b giữ thứ tự union đã được LB kiểm (p4b-v0 0,4111) và cho phép thay đổi nhỏ khi hoà; khung free theo thứ tự corpus để pool khớp đúng top-300. Không đổi quy tắc nào của bản đã nộp.

## Điều không nằm trong bàn giao

Làn en: không có doc en nào được chọn (xem docs/s36-fuse4.md, mục Làn en), nên cả hai bản không chứa doc en. Ba ablation trên p4b-v3 (nowin, noJina, noCmax) đã dựng 26/10, xem docs/s36-fuse4.md, mục Ba ablation; chúng không thuộc hai bản bàn giao.
