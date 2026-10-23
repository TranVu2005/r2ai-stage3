# S37 — crawl 120ask từ VPS EC2 Tokyo

Vu dựng VPS và chạy lệnh qua SSH. Gói này không chứa key/credential. Crawler độc lập (Python 3 + `requests`), không phụ thuộc repo.

Quy tắc đã cài trong `crawler.py`: 1 IP, 1 request một lúc, khoảng cách giữa hai lần bắt đầu = 1,0 s + jitter 0–0,3 s (cờ `--min-gap` < 1,0 bị từ chối); UA `r2ai-stage3-research/1.0 (+https://github.com/TranVu2005/r2ai-stage3)`; không đọc robots (D21); redirect theo tay tối đa 3 bước và chỉ trong `120ask.com` (sang domain khác ghi `offsite`, không theo); không proxy, không xoay IP, không giả lập trình duyệt.

| Kết quả | Hành vi |
|---|---|
| 200 có `<h1>` + khối hỏi `b_askcont` + khối đáp `b_ans*` | `ok`, ghi HTML |
| 200 thiếu cấu trúc hỏi–đáp | `ok` + `suspect=true` (vẫn lưu HTML để xem) |
| 404 / 410 | `gone` |
| 521/502/503/504/520/522–524, timeout, lỗi kết nối | retry tối đa 3 lần, chờ 30 s / 120 s / 600 s; vẫn lỗi → `failed` (chạy lại ở vòng sau) |
| 403 / 429 / 401 / 451, hoặc trang challenge (CAPTCHA, "Just a moment"…) | **DỪNG toàn bộ**, ghi `STOP.json` (lý do, URL); URL đó vẫn pending |
| ≥ 10 URL `failed` liên tiếp (origin sập) | nghỉ 30 phút; sau 3 lần nghỉ liên tiếp thì dừng (mã thoát 4), chạy lại sau |

## 1. Đưa gói lên VPS (chạy trên máy local)

`urls.tsv` nằm ở `out/runs/s37/urls.tsv` (73.001 URL, cột `doc_id<TAB>url<TAB>doc_ids`). Thay `<host>` và đường dẫn key của Vu:

```bash
ssh -i <key> ubuntu@<host> 'mkdir -p ~/s37'
scp -i <key> deploy/s37/crawler.py deploy/s37/setup.sh deploy/s37/run.sh deploy/s37/status.sh deploy/s37/pack.sh out/runs/s37/urls.tsv ubuntu@<host>:~/s37/
```

## 2. Cài đặt (trên VPS, một lần)

```bash
cd ~/s37 && bash setup.sh
```

Cài `python3-venv`, `tmux`, venv `.venv` với `requests`; in số dòng `urls.tsv` (phải 73001).

## 3. Pilot 50 URL

```bash
cd ~/s37 && ./run.sh pilot     # ~1–2 phút; xong thì tmux tự thoát
./status.sh pilot
```

Báo lại cho agent: `ok_rate` (tỉ lệ 200), `suspect`, `sec_per_url_mean`, `by_status`. Tiếp tục khi `ok_rate` cao và `suspect` ≈ 0. Nếu có `STOP.json` thì dừng, gửi nội dung file. (Pilot ghi vào `pilot/`, tách khỏi `data/`; `total_urls`/`remaining` trong status.sh pilot là của cả hàng đợi, bỏ qua.)

## 4. Chạy hết

```bash
cd ~/s37 && ./run.sh full
./status.sh            # mọi lúc: tmux sống không, đã xong / ok / suspect / failed / ETA
tmux attach -t s37     # xem trực tiếp (Ctrl-b d để rời, không dừng crawler)
```

Mất kết nối SSH hay VPS khởi động lại: chạy lại đúng `./run.sh full`, tự tiếp tục theo `log.jsonl` (URL đã ghi log thì bỏ qua; shard dở dang được cắt về mốc đã ghi log). ETA = số URL còn lại / tốc độ (≈ 1/(1,15 s) nếu mỗi request nhanh hơn 1,15 s; chậm hơn nếu độ trễ Tokyo → 120ask cao hơn). Số liệu đo ở pilot quyết định.

Vòng sau cho URL `failed`:

```bash
./run.sh retry 2       # đóng băng queue-r2.tsv từ log, chạy các URL failed; vòng 3: ./run.sh retry 3
```

## 5. Gói lô và kéo về

```bash
cd ~/s37 && ./pack.sh          # chỉ lô đủ 5 shard (~10.000 URL); an toàn khi crawler đang chạy
./pack.sh --final              # sau khi crawl xong: nốt lô lẻ
ls -lh data/batches/           # batch-0001.tar + batch-0001.tar.sha256 ...
```

Mỗi lô là `tar` (không nén lại vì shard đã gzip) gồm `shards/*.jsonl.gz`, `log.jsonl`, `MANIFEST.json` (sha256 từng shard). Shard chỉ vào lô khi mọi URL của nó đã có dòng log. Kéo về (local, từng lô khi sẵn sàng):

```bash
mkdir -p out/runs/s37/pulled
scp -i <key> 'ubuntu@<host>:~/s37/data/batches/batch-0001.tar*' out/runs/s37/pulled/
```

Rồi chạy pipeline local (mục 6).

## 6. Pipeline local cho mỗi lô

```bash
python -m r2ai.s37.ingest batch out/runs/s37/pulled/batch-0001.tar      # CPU: sha256 -> extract -> segment -> chunk; xong ở trạng thái waiting_gpu
python -m r2ai.s37.ingest status                                         # lô nào đang chờ GPU, GPU có rảnh không
python -m r2ai.s37.ingest embed                                          # CHỈ khi Vu duyệt và GPU rảnh: vector doc + chunk, cặp lọt pool, C-max
```

- `batch` (mặc định `--device defer`) giữ cấu trúc hỏi–đáp: `doc_text` = title + question + các answer theo thứ tự trang. Doc dưới ngưỡng (< 200 code point và < 50 chữ Hán) gắn `thin`, vẫn ghi vào docs.
- Segment canonical riêng `data/s37_120ask/seg37NNN-s37-bNNNN-120ask.com/`: `docs.parquet`, `provenance.parquet`, `chunks_t256.parquet` (chunker canonical), `chunks_struct.parquet` (question + một answer mỗi chunk), `manifest.json` (SHA256). Không sửa canonical-20261018, không ghi index chung.
- `embed` (cuda) từ chối khi GPU bận (util > 10% hoặc < 3.072 MiB trống); S35/S38 dùng GPU nên hỏi Vu trước. `--device cpu` chỉ để kiểm thông lô đầu. Ra: `vectors.npz` (doc), `chunk_vectors.npz` (t256 + struct), `cmax_top3.parquet` (chỉ chunk t256 body/answer, như S32; chunk cấu trúc chỉ nhúng sẵn cho S38), và trong `out/runs/s37/`: `pairs_new.parquet` (schema S31 + `in_top211`, `in_pool13`, `pair_id` ≥ 1.000.000), `report.json` (số cặp theo pool top-211 / pool13 / hợp, số query bị ảnh hưởng, ETA 4B / Jina / C-max).
- Chưa chấm gì. Điểm chấm gộp ở bước dựng FUSE cuối.

## Lưu ý vận hành

- Thứ tự URL = thứ hạng hàng đợi S21, nên lô đầu có URL ưu tiên cao.
- Dừng chủ động: `tmux send-keys -t s37 C-c` (ghi nốt URL hiện tại rồi thoát). Không `kill -9` (vẫn an toàn nhờ cắt shard theo log, nhưng mất URL đang bay).
- Dung lượng ước tính: ≈ 60 KB/trang × 73.001 ≈ 4,4 GB HTML, gzip còn ≈ 0,8–1,2 GB. Kiểm `df -h` trước khi chạy hết.
- Nếu `STOP.json` xuất hiện: không xoá và chạy lại ngay. Đưa nội dung cho agent/Vu quyết định (quy tắc: không proxy, không xoay IP).
