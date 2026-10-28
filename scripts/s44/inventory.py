"""Read-only inventory: size / newest mtime / files per directory at fixed depths. Writes CSV to argv[1]."""
import csv, os, sys, time
from pathlib import Path
roots = [(Path('D:/GitHub/r2ai-stage3/data'), 2), (Path('D:/GitHub/r2ai-stage3/out'), 3), (Path('D:/GitHub/r2ai-stage3/state'), 1),
         (Path('D:/GitHub/r2ai-stage3/.superpowers/worktrees'), 1), (Path('C:/Users/Admin/.cache/huggingface/hub'), 1),
         (Path('D:/GitHub'), 1), (Path('D:/'), 1), (Path('C:/Users/Admin/.codex/worktrees'), 1)]
skip_top = {Path('D:/GitHub/r2ai-stage3'), Path('D:/GitHub'), Path('D:/$RECYCLE.BIN'), Path('D:/System Volume Information')}
def walk(p):
    size = n = 0; newest = 0.0
    for dp, dn, fn in os.walk(p, onerror=lambda e: None, followlinks=False):
        dn[:] = [d for d in dn if not os.path.islink(os.path.join(dp, d))]
        for f in fn:
            try:
                st = os.stat(os.path.join(dp, f), follow_symlinks=False)
            except OSError:
                continue
            size += st.st_size; n += 1; newest = max(newest, st.st_mtime)
    return size, n, newest
rows = []
def visit(p, depth, left):
    if not p.is_dir():
        st = p.stat(); rows.append((p.as_posix(), depth, st.st_size, 1, st.st_mtime)); return
    if left == 0 or p in skip_top and depth > 0 and False:
        s, n, t = walk(p); rows.append((p.as_posix(), depth, s, n, t)); return
    for c in sorted(p.iterdir()):
        if c in skip_top and c != p:
            continue
        if c.is_symlink():
            continue
        try:
            visit(c, depth + 1, left - 1)
        except OSError:
            pass
for root, d in roots:
    if root.exists():
        visit(root, 0, d)
with open(sys.argv[1], 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['path', 'depth', 'bytes', 'files', 'newest_mtime'])
    for r in rows:
        w.writerow([r[0], r[1], r[2], r[3], time.strftime('%Y-%m-%d', time.localtime(r[4])) if r[4] else ''])
print(len(rows), 'rows')
