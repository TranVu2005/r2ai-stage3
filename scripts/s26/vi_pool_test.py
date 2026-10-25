"""S26 vi pool test: are in-corpus vi articles closer to the test queries than same-site articles BTC did not list?

Input <out>/vi_links.parquet (vi_links.py). Both groups are represented by the same text type: the link anchor (title).
Score = BGE-M3 dense cosine of anchor vs each query; per URL best over queries and min-rank among URLs of the same host.
AUC(in_corpus) overall, per host, and for health-only sites (no off-topic sections to confound the comparison).
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
os.chdir(os.environ.get('R2AI_ROOT', 'D:/GitHub/r2ai-stage3'))  # data/ and src/ of the main checkout
sys.path.insert(0, 'src')
from r2ai.index.bge_m3 import M3Encoder  # noqa: E402

HEALTH_ONLY = ['vinmec.com', 'medlatec.vn', 'hellobacsi.com', 'tiemchunglongchau.com.vn', 'suckhoedoisong.vn',
               'tamanhhospital.vn', 'thaythuocvietnam.vn', 'suckhoeviet.org.vn', 'youmed.vn', 'phunusuckhoe.giadinhonline.vn']
S, out = Path(sys.argv[1]), Path(sys.argv[2])
df = pd.read_parquet(out / 'vi_links.parquet')
g = df.groupby('host').in_corpus.agg(['sum', 'size']); keep = g[(g['sum'] >= 100) & (g['size'] - g['sum'] >= 100)].index
df = df[df.host.isin(keep)].reset_index(drop=True)
cache = out / 'vi_links_dense.npy'
if cache.exists() and len(np.load(cache, mmap_mode='r')) == len(df):
    E = np.load(cache)
else:
    enc = M3Encoder(max_len=64); parts = []
    for i in range(0, len(df), 128):
        parts.append(enc.encode_batch(df.anchor.iloc[i:i + 128].tolist())[0])
    E = np.concatenate(parts); np.save(cache, E)
Q = torch.from_numpy(np.load(S / 'q_dense.npy')).cuda()
sc = (Q @ torch.from_numpy(E).cuda().T).float()                      # [Q, links]
df['best'] = sc.max(0).values.cpu().numpy()
mr = np.zeros(len(df), np.int64)
for h, idx in df.groupby('host').indices.items():
    s = sc[:, idx]; mr[idx] = ((s.argsort(1, descending=True).argsort(1) + 1).min(0).values.cpu().numpy())
df['minrank_host'] = mr
df.to_parquet(out / 'vi_links_scored.parquet', index=False)
y = df.in_corpus.astype(int)
print(f'links {len(df)} hosts {df.host.nunique()} in_corpus {int(y.sum())}')
print(f'ALL hosts: AUC best {roc_auc_score(y, df.best):.4f}')
rows = []
for h, x in df.groupby('host'):
    rows.append((h, len(x), int(x.in_corpus.sum()), x[x.in_corpus].best.median(), x[~x.in_corpus].best.median(),
                 roc_auc_score(x.in_corpus.astype(int), x.best), roc_auc_score(x.in_corpus.astype(int), -x.minrank_host), h in HEALTH_ONLY))
t = pd.DataFrame(rows, columns=['host', 'n', 'in', 'med_in', 'med_out', 'auc_best', 'auc_minrank', 'health_only']).sort_values('n', ascending=False)
print(t.round(4).to_string(index=False))
for flag in [True, False]:
    x = t[t.health_only == flag]
    print(f'health_only={flag}: hosts {len(x)}, in-weighted AUC best {np.average(x.auc_best, weights=x["in"]):.4f}, AUC -minrank {np.average(x.auc_minrank, weights=x["in"]):.4f}')
