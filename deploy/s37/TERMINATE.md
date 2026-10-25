# Đóng VPS 120ask (S37): kéo nốt dữ liệu, Terminate instance, xoá EBS

Phạm vi 120ask đã chốt (Vu, 24/10): 6 lô + `batch-0007` (shard r1-00028). Không crawl thêm. Mọi bước dưới đây do Vu chạy tay (agent không giữ khoá SSH và không có quyền AWS).

## 1. Kiểm tra local đã có đủ trước khi xoá gì

Trong `out\runs\s37\pulled` phải có 7 file `batch-0001.tar` … `batch-0007.tar`, mỗi file kèm `.sha256`. Kiểm SHA256 từng tar (PowerShell, tại `D:\GitHub\r2ai-stage3`):

```powershell
Get-ChildItem out\runs\s37\pulled\batch-*.tar | ForEach-Object { $h = (Get-FileHash $_.FullName).Hash.ToLower(); $e = (Get-Content "$($_.FullName).sha256").Split(' ')[0]; "$($_.Name) $(if ($h -eq $e) {'OK'} else {'KHAC'})" }
```

Cả 7 dòng phải là `OK`. Nếu thiếu hoặc `KHAC`, kéo lại file đó trước khi đóng VPS.

## 2. Kéo nốt log và danh sách URL (nhỏ, để tái lập về sau)

Đặt `$KEY` và `$VPS` (ví dụ `$VPS = "ubuntu@16.176.217.120"`), rồi:

```powershell
New-Item -ItemType Directory -Force out\runs\s37\vps-final | Out-Null
scp -i $KEY "${VPS}:~/s37/data/log.jsonl" out\runs\s37\vps-final\
scp -i $KEY "${VPS}:~/s37/data/packed.json" out\runs\s37\vps-final\
scp -i $KEY "${VPS}:~/s37/urls.tsv" out\runs\s37\vps-final\
scp -i $KEY "${VPS}:~/s37/probe.log" out\runs\s37\vps-final\
scp -i $KEY "${VPS}:~/s37/data.run.log" out\runs\s37\vps-final\
```

`log.jsonl` cần để dựng `urls_remaining.tsv` (URL chưa `ok`). Không cần kéo `data/shards`: nội dung đã nằm trong các tar.

## 3. Dừng probe

```powershell
ssh -i $KEY $VPS 'pkill -f probe.sh; tmux kill-session -t s37 2>/dev/null; pgrep -af "probe.sh|crawler.py" || echo "khong con tien trinh"'
```

## 4. Terminate instance và xoá EBS (AWS Console, region Sydney `ap-southeast-2`)

1. EC2 → Instances → chọn instance → **Instance state → Terminate instance**. Volume gốc bị xoá theo (mặc định *Delete on termination* = Yes).
2. EC2 → **Elastic Block Store → Volumes**: không được còn volume nào của VPS này. Còn thì chọn → Actions → **Delete volume** (volume `available` mới xoá được).
3. EC2 → **Snapshots** và **AMIs**: xoá snapshot/AMI nếu đã tạo cho VPS này.
4. EC2 → **Elastic IPs**: nếu có IP gắn với VPS → Actions → **Release Elastic IP address**. Một Elastic IP không gắn instance bị tính phí.
5. (Tuỳ chọn) xoá rule SSH trong Security Group và key pair không dùng nữa.
6. Billing → **Bills / Free Tier**: sau 1–2 giờ kiểm tra không còn EC2/EBS đang chạy.

## 5. Báo lại

Báo cho agent: "đã terminate, đã kiểm volume/EIP/snapshot". Agent ghi mốc đóng VPS vào `docs/PROGRESS.md`.
