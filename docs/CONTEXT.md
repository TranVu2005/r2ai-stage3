# R2AI Stage 3 – Truy hồi tài liệu y khoa đa ngôn ngữ: Ngữ cảnh dự án

> Cập nhật cấu trúc/metric: 2026-10-04 (UTC+7). Khảo sát corpus bên dưới là snapshot legacy; kết quả migration ở docs/PROGRESS.md. Đây là ngữ cảnh cho AI assistant và agent (Claude, Codex) làm việc trong dự án. Mục 1 là đề bài (cố định); mục 2 trở đi là phát hiện, quyết định và trạng thái hiện tại.

---

## 0. Mốc thời gian

Public **31/10/2026**; private **04/11/2026**, tối đa **5 lượt**; kết quả **11/11/2026**. Theo mốc người dùng cung cấp.

## 1. Đề bài (tóm tắt)

**Nhiệm vụ:** cho 1.200 truy vấn y khoa tiếng Việt (`query.parquet`: id, query), truy hồi và xếp hạng tài liệu và đoạn nội dung liên quan từ corpus đa ngôn ngữ vi/en/zh (`links_corpus.parquet`: id, url). Đội thi **tự crawl** nội dung từ URL.
- Nguồn dữ liệu: HF dataset `AIGuruTinix/ViBioMIR`.

**Đánh giá:** macro theo query (chỉ tính các query có ≥1 nhãn dương ở cấp tương ứng). **Điểm cuối = (Doc F2 + Chunk F2) / 2**, với F2 = 5PR / (4P + R), tức Recall nặng gấp 4 lần Precision.
- **Doc:** P = |D∩G|/|D|, R = |D∩G|/|G|. doc_id là `id` (int) trong `links_corpus.parquet`; tài liệu ngoài danh sách không được tính.
- **Chunk:** chỉ so reference cùng doc_id. Chuẩn hóa NFKC → decode entity → lowercase → gộp whitespace → strip, giữ dấu câu. Scorer dùng tokenizer BGE-M3 (giả định triển khai). overlap=LCS_tokens(c,g)/|g|, khớp khi >=0,4. P = số predicted chunk khớp ít nhất một reference / tổng predicted chunk; R = số reference chunk được tìm / tổng reference chunk. **Không có bước gộp trùng ở ngưỡng 0,8 trong scorer**; dedupe builder là bước riêng giữ nguyên để tái tạo submission.

**Định dạng nộp:** JSON `[{id:int, relevant_docs:[doc_id], relevant_chunks:[{doc_id, chunk_text, chunk_order?:int>=0}]}]`, đủ 1.200 query (trường không có kết quả để list rỗng), nén ZIP chỉ chứa 1 file, nộp tại leaderboard.aiguru.com.vn. Số lần nộp mỗi ngày có giới hạn.
- **`chunk_text` phải trích nguyên văn từ tài liệu gốc**, không dịch, không viết lại.

**Ràng buộc mô hình:** chỉ dùng mô hình open-weight (HF…), **≤15B**, phát hành **trước 01/08/2026**. Không dùng LLM đóng (GPT, Gemini…). Dữ liệu ngoài được phép nhưng phải trích dẫn nguồn.

---

## 2. Dữ liệu – các con số chính

| Mục | Giá trị |
|---|---|
| Corpus | 4.394.718 URL, 97 domain; khoảng **3,7M doc duy nhất** (693.802 dòng trùng http/https/www) |
| Tập trung domain | 10 domain phủ 80%, 35 domain phủ 95% |
| Ngôn ngữ (theo URL) | khoảng **77% zh**, 17% vi; gần như không có en |
| Top zh | cnkang 21,9% · 120ask 20,9% · familydoctor 10,2% · ask.39.net 7,7% · zysjonline 5,5% · a-hospital 3,9% · zhongyibaodian 3,3% |
| Top vi | suckhoecongdongonline 3,5% · suckhoedoisong 2,0% · nhathuoclongchau 1,8% · thanhnien 1,6% · laodong 0,7% · vinmec 0,6% · medlatec 0,6% |
| Slug URL có nghĩa | chỉ 17% → không thể truy hồi chỉ bằng URL |
| Quy mô text | khoảng 1,9 tỷ token; 3,7–7,4M chunk (512/256 token); index dense fp16 7,6–15GB |
| Query | median 25 token BGE-M3, p95 103, max 350. Khoảng 11% nhiều ý rõ ràng; 11% có viết tắt (HIV, IVF, HPV, G6PD…) |

---

## 3. Phát hiện quan trọng

### 3.1 Crawl (probe v2, egress IP **VN dân dụng**, UA Chrome)
- **Khoảng 85% corpus crawl trực tiếp được.** Hầu hết domain lớn đạt 93–100%: cnkang, 120ask, familydoctor, ask.39.net, vinmec, suckhoedoisong, thanhnien…
- Probe v1 sai do lỗi script: `aiohttp.read(MAX_BYTES)` cắt cụt 116/300 HTML. Nhiều kết quả "needs_js/blocked" ở v1 là giả.
- **Không cần Playwright** (needs_js thật chỉ 1/300).
- **Cookie challenge** (laodong.vn và có thể các báo cùng CMS): body chỉ chứa `document.cookie="D1N=<hex>";location.reload()`. Giải bằng regex rồi request lại; cookie dùng lại được cho cả phiên.
- **Kết quả phụ thuộc IP:** phải crawl từ IP VN dân dụng. IP datacenter hoặc nước ngoài có thể bị chặn 403.
- Domain cần xử lý riêng:
  - `zysjonline.com` (5,5%): robots DISALLOWED/UNAVAILABLE, Wayback không có bản lưu → cần kiểm tra robots.txt bằng tay; nhiều khả năng mất.
  - `nhathuoclongchau.com.vn` (1,8%): trực tiếp 0%; Common Crawl 46%, Wayback 32% → lấy từ archive (CC columnar index qua DuckDB, vì CDX API hay bị 504).
  - `bingli.iiyi.com` (0,6%): robots chặn; Wayback 38%.
- Throughput ở 1 req/s mỗi domain: cnkang khoảng 369 giờ, suckhoecongdongonline khoảng 349 giờ (server chậm, phản hồi 7,7s). Điểm nghẽn là **giới hạn rate trên từng domain**, không phải phần cứng.

### 3.2 Gold check (30 query, tìm tay qua Google)
- **83% query lấy từ các trang hỏi đáp Vinmec** (`/vie/bai-viet/...`, query ngắn/vừa) **và MEDLATEC** (`/hoi-dap/...`, query dài). 47% khớp nguyên văn, 33% khớp một phần.
- **0/25 trang nguồn có trong corpus.** BTC đã cố ý loại trang gốc, nên không có lối tắt "tìm lại trang nguồn". Gold là các tài liệu **khác** trả lời được câu hỏi, bằng bất kỳ ngôn ngữ nào.
- Query đã được BTC chuẩn hoá nhẹ (ko→không, e→em, dc→được, sửa chính tả) → không cần bước chuẩn hoá teencode.
- Q900: corpus có một bài Vinmec liên quan (doc 754207) → có dấu hiệu gold là **bài viết liên quan** trong corpus.
- Phần lớn domain zh lớn là **site hỏi đáp** (120ask, ask.39.net, cnkang, familydoctor), cùng định dạng "bệnh nhân hỏi, bác sĩ trả lời" với query → khớp câu hỏi với câu hỏi xuyên ngôn ngữ có thể rất mạnh. Gold cấp chunk nhiều khả năng là **phần trả lời**.

---

## 4. Quyết định đã chốt

1. **Crawl tiếng Việt trước** (khoảng 640k doc duy nhất, không gồm longchau) để có baseline sớm. Mốc 1: khi có ≥100k doc `ok` thì bắt đầu dựng baseline.
2. **Chỉ chạy trên laptop, IP VN.** Dừng và chạy tiếp bất kỳ lúc nào: Ctrl+C an toàn, chịu được sleep/mất mạng, checkpoint SQLite WAL, ghi shard nguyên tử, rate đã học được lưu lại.
3. **Không crawl đều toàn bộ corpus.** Với zh: lấy mẫu 2% mỗi domain → chạy baseline cho 1.200 query → tính yield (số hit trên 1k doc) mỗi domain → crawl theo yield giảm dần; cân nhắc cắt tỉa theo chuyên mục trong đường dẫn URL.
4. **Rate theo nhóm domain:** `cdn_large` tối đa 8 req/s, `default` 4 req/s, `slow_server` ≤1 req/s với tối đa 4 kết nối. Tự giảm tốc khi gặp 429/503 hoặc latency tăng. Tôn trọng robots.txt và `crawl-delay`.
5. **Không** dùng proxy xoay vòng, dịch vụ giải captcha, hay chia một domain cho nhiều IP.
6. Trùng http/https: crawl 1 URL đại diện mỗi nhóm, lưu `doc_ids` của cả nhóm. Khi nộp, mặc định trả cả nhóm id (vì F2 thiên về Recall); cần kiểm chứng bằng 1 lượt nộp.

### Vùng xám – KHÔNG làm nếu chưa có xác nhận bằng văn bản của BTC
- Lấy câu trả lời của bác sĩ trên **trang nguồn của chính các query test** để mở rộng truy vấn.
- Dùng ô tìm kiếm của site bên thứ ba (120ask, 39.net…) để chọn URL hoặc làm bộ truy hồi.

### Quyết định dữ liệu
- **Không dùng dữ liệu ngoài** để train/dev trong dự án này. Gold-check là phân tích lịch sử, không phải pipeline train.

---

## 5. Kiến trúc dự kiến

- **Crawl production:** logic trong `src/vicrawl/`; `scripts/crawl.py` là wrapper của `r2ai.crawl.cli` (asyncio, httpx HTTP/2, rate thích ứng, `configs/domains.yaml`, `run/status/approve`, `--keep-awake`). `r2ai.probe.fetcher` là công cụ probe lịch sử → `data/raw_<group>/<domain>/*.jsonl.zst` (HTML thô, shard bất biến).
- **Extract** (`extract.py --watch`, tăng dần, chạy song song nhiều tiến trình): trafilatura cộng extractor riêng cho các domain lớn. Tách `title/description/question/answer/body`, giữ ranh giới đoạn văn → `data/docs_<group>/*.parquet`.
- **Truy hồi (baseline):** BGE-M3 hybrid (dense + sparse; tokenizer trùng với metric) trên chunk → gộp về doc bằng max-score → reranker (`bge-reranker-v2-m3` hoặc Qwen3-Reranker ≤15B). BM25 tách riêng vi và zh. Dùng LLM open ≤15B (Qwen3) rút gọn query dài hoặc dịch query sang zh khi cần.
- **Đánh giá local:** scorer hiện tại (tokenizer BGE-M3, LCS/reference >=40%, không gộp trùng, F2 macro). Pseudo-dev sinh từ corpus đã crawl; không dùng nguồn ngoài.
- **Index baseline đã đo:** FAISS **IndexFlatIP**, 818k chunk trong bundle legacy; build giữ nguyên nhánh lựa chọn index theo dung lượng, chưa rebuild corpus mở rộng. Embed corpus lớn trên Kaggle/Colab nếu cần (laptop RTX 3050 6GB chỉ đủ inference BGE-M3 fp16).
- **Lưu trữ:** phase vi khoảng 10–12GB (HTML nén zstd khoảng 7–10GB, text parquet khoảng 1GB); toàn corpus cần ≥150GB SSD.

---

Thông tin cấu hình production: **suckhoecongdongonline thuộc nhóm default**; **tiemchunglongchau đã crawl**, khác nhathuoclongchau đang hoãn.

## 6. Trạng thái và việc tiếp theo (legacy; xem PROGRESS cập nhật)

- [x] Profile corpus và query (v1), sửa probe và probe lại từ IP VN (v2), gold check 30 query
- [ ] Crawler production cho domain vi: viết code, chạy thử `--limit 50`, test kill/resume, chạy toàn bộ
- [ ] Review chất lượng trích xuất (`out/qa_extract/`) cho các domain vi lớn
- [ ] Scorer local và baseline BGE-M3 + reranker khi đạt mốc 100k doc
- [ ] Nộp thử: (a) chunk dài so với chunk ngắn; (b) trả 1 id so với cả nhóm id trùng; (c) chỉ vi so với có zh
- [ ] Mẫu 2% các domain zh → tính yield → lập thứ tự crawl zh
- [ ] longchau qua CC columnar index; kiểm tra robots.txt của zysjonline bằng tay
- [x] Quyết định không dùng dữ liệu ngoài làm train/dev

**Câu hỏi mở:** phân bổ thời gian mở rộng corpus zh trước deadline đã chốt; phân bố gold theo ngôn ngữ; giới hạn độ dài `chunk_text`.

---

## 7. Quy ước làm việc với AI

- Trả lời tiếng Việt, ngắn gọn, ưu tiên bảng, số liệu và lệnh cụ thể.
- Khi viết prompt cho agent: luôn đề xuất **model và mức effort**, gồm cả lựa chọn Claude (Sonnet 5.5 / Opus 5.5) lẫn OpenAI (GPT-6 Luna, GPT-6 Astra, GPT-6.1 Sol).
- Mọi số liệu trong báo cáo phải đo được thực tế; không đo được thì ghi "chưa đo", không ước lượng khống. Seed cố định (42).
