# Crawl nốt 120ask trên máy mới (lần chốt cuối, S37)

Phạm vi: chỉ các URL chưa `ok` (16.619: 11 `failed` + 16.608 `untried`; 9 `gone` là 404 cuối cùng, không crawl lại). Danh sách nằm ở `out\runs\s37\vps-final\queue_remaining.tsv` (định dạng của `crawler.py`) và `urls_remaining.tsv` (url, doc_id, status). Mốc đóng crawl: 25/10 23:59 (UTC+7). Mọi lệnh do Vu chạy tay, tại `D:\GitHub\r2ai-stage3`, với `$KEY2` / `$VPS2` của máy mới.

## 1. Đưa crawler lên máy mới

```powershell
ssh -i $KEY2 $VPS2 'mkdir -p ~/s37'
scp -i $KEY2 deploy\s37\crawler.py deploy\s37\setup.sh deploy\s37\run.sh deploy\s37\status.sh deploy\s37\pack.sh deploy\s37\probe.sh "${VPS2}:~/s37/"
scp -i $KEY2 out\runs\s37\vps-final\queue_remaining.tsv "${VPS2}:~/s37/urls.tsv"
ssh -i $KEY2 $VPS2 'cd ~/s37 && chmod +x *.sh && ./setup.sh'
```

Dòng cuối của `setup.sh` phải là `urls.tsv: 16619 lines`.

## 2. Thử MỘT request trước khi crawl

```powershell
ssh -i $KEY2 $VPS2 'curl -s -o /dev/null -m 20 -w "%{http_code}\n" https://www.120ask.com/question/60917294.htm'
```

`200` thì crawl. `000` hoặc timeout thì dừng và báo; không thử region khác.

## 3. Crawl và theo dõi

```powershell
ssh -i $KEY2 $VPS2 'cd ~/s37 && ./run.sh full'
ssh -i $KEY2 $VPS2 'cd ~/s37 && ./status.sh'
```

Gặp 403/429/401/451 hoặc trang thách thức, crawler ghi `data/STOP.json` rồi dừng; đừng xoá file đó, báo agent.

## 4. Kéo dữ liệu về (nhiều lần cũng được)

Mỗi 5 shard (10.000 URL) `pack.sh` đóng một lô; lô lẻ cuối cùng cần `--final`. Khi crawl xong, hoặc khi muốn kéo giữa chừng:

```powershell
ssh -i $KEY2 $VPS2 'cd ~/s37 && ./pack.sh'            # chỉ lô đủ 5 shard, an toàn khi crawler đang chạy
ssh -i $KEY2 $VPS2 'cd ~/s37 && ./pack.sh --final'    # khi crawl đã dừng: cả lô lẻ còn lại
New-Item -ItemType Directory -Force out\runs\s37\pulled-lo7 | Out-Null
scp -i $KEY2 "${VPS2}:~/s37/data/batches/batch-*.tar*" out\runs\s37\pulled-lo7\
```

Giữ nguyên tên file (`batch-0001.tar` …) trong `pulled-lo7`. Tên đó trùng tên lô cũ nên agent tự đặt lại tên khi đưa vào pipeline (`batch-0008.tar`, `batch-0009.tar` …); Vu không cần đổi tên. Chỉ báo "đã kéo".

## 5. Kéo log rồi Terminate máy mới khi xong

```powershell
scp -i $KEY2 "${VPS2}:~/s37/data/log.jsonl" out\runs\s37\vps-final\lo7-log.jsonl
scp -i $KEY2 "${VPS2}:~/s37/data/packed.json" out\runs\s37\vps-final\lo7-packed.json
```

(tên `lo7-…` để không đè file của máy cũ). Kiểm `pulled-lo7` đủ file `.tar` và `.sha256`, rồi Terminate theo mục 4 của `deploy/s37/TERMINATE.md`.
