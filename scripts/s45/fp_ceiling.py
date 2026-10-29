"""S45: how close to 1 can AUC -min-rank get on the 2% zh sample if BTC's retriever were known exactly?

The yardstick ranks docs inside the 2% sample, while BTC kept a doc by its rank in the full zh v1 pool (about 50x larger).
Simulation: take a cached score matrix as the "true" retriever. For a doc at sample rank k in query q, the number of
full-pool docs above it is (k - 1) sampled docs plus NegBin(k, p) unsampled ones (each full-pool doc is in the sample with
probability p = 0.02). Kept_sim = min over queries of the simulated full rank <= N, with N set so that the kept rate
matches the real one (8.21%). AUC(-sample min-rank, kept_sim) is then the ceiling of the yardstick for a perfect match.

  fp_ceiling.py <out> <matrix stem, e.g. dt512.cls> [p=0.02] [seeds=3]
"""
import sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

OUT, stem = Path(sys.argv[1]), sys.argv[2]
p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
y = np.load(OUT / 'y.npy').astype(int); rate = y.mean()
S = torch.from_numpy(np.load(OUT / f'{stem}.npy')).cuda().float()
R = (S.argsort(1, descending=True).argsort(1) + 1)                      # sample rank [Q, docs]
mr = R.min(0).values.cpu().numpy()
del S
for seed in range(seeds):
    torch.manual_seed(seed)
    # NegBin(k, p) as Poisson(Gamma(k, (1-p)/p)) - number of unsampled docs above, given k-1 sampled above
    lam = torch._standard_gamma(R.float()) * (1 - p) / p
    full = (R - 1).float() + torch.poisson(lam) + 1
    fmin = full.min(0).values.cpu().numpy()
    N = np.quantile(fmin, rate)
    ys = (fmin <= N).astype(int)
    print(f'{stem} p={p} seed={seed}: N={N:.0f} kept_sim={ys.mean():.4f} AUC -minrank ceiling={roc_auc_score(ys, -mr):.4f} '
          f'(real labels {roc_auc_score(y, -mr):.4f})', flush=True)
    del lam, full
