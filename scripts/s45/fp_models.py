"""S45: which explanation of the missing AUC (0.924 -> ~0.99) fits the shape of the real min-rank profile?

Both models take dt512.cls as BTC's scores, simulate the full-pool rank as in fp_ceiling.py (p = 0.02, kept rate 8.21%),
and are tuned so that AUC -min-rank on the sample is about 0.924:
  H  hidden queries: BTC selected with Q_all queries, the yardstick sees a random fraction f of them
  C  config mismatch: BTC's scores = ours + Gaussian noise, per pair (Cp) or per doc vector direction (Cd: noise shared
     by all queries of a doc, sigma times a per-doc random offset projected on each query)
Profile = AUC best, AUC -min-rank, kept share at min-rank 1, R-prec, share of kept docs with min-rank <= 1/3/10/20.

  fp_models.py <out>
"""
import sys
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_auc_score

OUT = Path(sys.argv[1]); p = 0.02
y = np.load(OUT / 'y.npy').astype(int); rate = y.mean()
S0 = torch.from_numpy(np.load(OUT / 'dt512.cls.npy')).cuda().float()
Qn, N = S0.shape


def rank(S):
    return S.argsort(1, descending=True).argsort(1) + 1


def labels(S_true, seed):
    torch.manual_seed(seed); R = rank(S_true)
    full = (R - 1).float() + torch.poisson(torch._standard_gamma(R.float()) * (1 - p) / p) + 1
    fmin = full.min(0).values.cpu().numpy()
    return (fmin <= np.quantile(fmin, rate)).astype(int)


def profile(yy, S_obs):
    best = S_obs.max(0).values.cpu().numpy(); mr = rank(S_obs).min(0).values.cpu().numpy()
    k = yy == 1
    return {'auc_best': roc_auc_score(yy, best), 'auc_minrank': roc_auc_score(yy, -mr), 'kept_mr1': yy[mr == 1].mean(),
            'rprec': yy[np.argsort(mr, kind='stable')[:yy.sum()]].mean(),
            **{f'cov<={t}': (mr[k] <= t).mean() for t in (1, 3, 10, 20)}}


def show(name, r):
    print(f'{name:34s} ' + ' '.join(f'{k} {v:.3f}' for k, v in r.items()), flush=True)


show('REAL (dt512.cls)', profile(y, S0))
rng = np.random.RandomState(0)
for f in (0.5, 0.45):
    ys = labels(S0, 0); qs = torch.from_numpy(np.sort(rng.choice(Qn, int(Qn * f), replace=False))).cuda()
    show(f'H hidden queries, sees {f:.2f}', profile(ys, S0[qs]))
sd = S0.std().item()
for s in (0.25, 0.35, 0.5):
    torch.manual_seed(1); St = S0 + s * sd * torch.randn_like(S0)
    show(f'Cp pair noise {s:.2f} sd', profile(labels(St, 0), S0))
    del St
# per-doc direction noise: random doc offset g_d (1024-d, unit), query vectors q -> shift = s * <q, g_d>
qv = torch.from_numpy(np.load(OUT / 'q_cls.npy')).cuda().float()
for s in (0.1, 0.15, 0.2):
    torch.manual_seed(2); G = torch.nn.functional.normalize(torch.randn(N, 1024, device='cuda'), dim=-1)
    St = S0 + s * (qv @ G.T) * (1024 ** 0.5) * sd
    show(f'Cd doc-vector noise {s:.2f}', profile(labels(St, 0), S0))
    del St, G
