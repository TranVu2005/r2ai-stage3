"""Read-only: SHA256 + bytes + mtime of every *.zip below out/ -> CSV argv[1]."""
import csv, hashlib, os, sys, time
from pathlib import Path
root = Path('D:/GitHub/r2ai-stage3/out')
rows = []
for p in sorted(root.rglob('*.zip')):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    st = p.stat()
    rows.append(('out/' + p.relative_to(root).as_posix(), st.st_size, time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime)), h.hexdigest()))
with open(sys.argv[1], 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['path', 'bytes', 'mtime', 'sha256']); w.writerows(rows)
print(len(rows))
