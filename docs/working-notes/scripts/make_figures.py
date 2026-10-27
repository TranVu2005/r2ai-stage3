"""Figures for the working notes, from the CSV files in ../data.

    python docs/working-notes/scripts/make_figures.py

Writes PNG (200 dpi) and PDF into ../figures. Colours: categorical slots 1-3 of the
dataviz reference palette (blue, orange, aqua), validated adjacent-pair CVD-safe;
text stays in neutral ink.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name: str) -> None:
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# label offsets (points) for runs whose labels would collide
OFFSETS = {"K4-thr-C": (-6, 7, "right"), "J3-thrC": (6, 7, "left"), "QRRF-nc140-ph": (-6, 7, "right"),
           "EF2-cut-p4b-full+w768-t1907": (8, -12, "left"), "120A-p4b-v3": (6, 7, "left"),
           "120A-free-v2": (6, -12, "left"), "120A-free-v0": (6, -12, "left"), "K80-v2": (6, -3, "left")}


def lb_progress() -> None:
    lb = pd.read_csv(os.path.join(DATA, "lb_scores.csv"))
    lb = lb[(lb.in_vu_table == "yes") & lb.final.notna() & lb.group.isin(["history", "free_frame"])].copy()
    lb["date"] = pd.to_datetime(lb.date)
    lb = lb.reset_index(drop=True)
    lb["x"] = range(len(lb))
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    for metric, color, label in [("v1", BLUE, "metric v1 (LCS/|gold| ≥ 0.4)"), ("v2", ORANGE, "metric v2 (+ LCS/|chunk| ≥ 0.4)")]:
        d = lb[lb.metric == metric]
        ax.scatter(d.x, d.final, s=22, color=color, zorder=3, label=label, edgecolor=SURFACE, linewidth=1.5)
        ax.step(d.x, d.final.cummax(), where="post", color=color, alpha=0.55, linewidth=1.5)
    cut = lb.index[lb.metric == "v2"].min() - 0.5
    ax.axvline(cut, color=INK2, linestyle=(0, (3, 3)), linewidth=1)
    ax.text(cut + 0.3, 0.235, "BTC metric change 23/10", color=INK2, fontsize=8)
    for run in ["K80-v2", "P1", "P-A3-nc", "P-A4-K90", "QRRF-nc140-ph", "K4-thr-C", "J3-thrC",
                "EF2-cut-p4b-full+w768-t1907", "120A-free-v0", "120A-p4b-v3", "120A-free-v2", "CH-rrf"]:
        r = lb[lb.run == run].iloc[0]
        name = {"EF2-cut-p4b-full+w768-t1907": "EF2-cut-p4b"}.get(run, run)
        dx, dy, ha = OFFSETS.get(run, (0, 7, "center"))
        ax.annotate(f"{name} {r.final:.4f}", (r.x, r.final), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=6.8, color=INK)
    ticks = lb.groupby(lb.date.dt.strftime("%d/%m")).x.min()
    ax.set_xticks(ticks.values, ticks.index, fontsize=7.5, rotation=45)
    ax.set_xlabel("public LB submissions in order (tick = first submission of the day, 2026)")
    ax.set_ylabel("Final = (Doc F2 + Chunk F2) / 2")
    ax.set_ylim(0.22, 0.44)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "lb_progress")


def fingerprint() -> None:
    fp = pd.read_csv(os.path.join(DATA, "fingerprint_auc.csv"))
    main = fp[fp.family != "extractor"].sort_values("auc_minrank")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw={"width_ratios": [1.7, 1]})
    colors = [ORANGE if f == "best" else BLUE for f in main.family]
    labels = [f"{v} [{q}]" if q != "vi" else v for v, q in zip(main.variant, main["query"])]
    a1.barh(range(len(main)), main.auc_minrank, color=colors, height=0.62)
    a1.set_yticks(range(len(main)), labels, fontsize=6.6)
    a1.set_xlim(0.5, 1.0)
    a1.set_xlabel("AUC (−min-rank), zh kept vs dropped")
    a1.grid(axis="y", visible=False)
    for i, v in enumerate(main.auc_minrank):
        a1.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=6.4, color=INK2)
    a1.set_title("(a) retriever variants (query vi unless noted)", fontsize=9, loc="left")
    ln = fp[fp.family == "dense-length"].sort_values("doc_tokens")
    a2.plot(ln.doc_tokens, ln.auc_minrank, color=BLUE, marker="o", markersize=5)
    best = fp[fp.family == "best"].iloc[0]
    a2.scatter([512], [best.auc_minrank], color=ORANGE, s=40, zorder=4)
    a2.annotate("doc_text 512\n(no title repeat) 0.924", (512, best.auc_minrank), xytext=(10, -26),
                textcoords="offset points", fontsize=7, color=INK)
    a2.annotate("title repeated + doc", (ln.doc_tokens.iloc[-1], ln.auc_minrank.iloc[-1]), xytext=(-70, -14),
                textcoords="offset points", fontsize=7, color=INK2)
    a2.set_xscale("log", base=2)
    a2.set_xticks([128, 256, 512, 1024, 2048], ["128", "256", "512", "1024", "2048"])
    a2.set_xlabel("doc tokens embedded (BGE-M3 dense)")
    a2.set_ylabel("AUC (−min-rank)")
    a2.set_ylim(0.86, 0.93)
    a2.set_title("(b) truncation length", fontsize=9, loc="left")
    fig.tight_layout()
    save(fig, "fingerprint_auc")


def chunk_size() -> None:
    cs = pd.read_csv(os.path.join(DATA, "chunk_size_curve.csv"))
    line = cs[cs.variant.isin(["CH-sec-1", "CH-win384", "CH-win768", "full body (K4-thr-C-extD)"])].sort_values("mean_raw_tokens")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(line.mean_raw_tokens, line.chunk_f2, color=BLUE, marker="o", markersize=6, label="one chunk per carrier")
    for _, r in line.iterrows():
        ax.annotate(f"{r.variant.split(' (')[0]}  {r.chunk_f2:.4f}", (r.mean_raw_tokens, r.chunk_f2),
                    xytext=(6, -10), textcoords="offset points", fontsize=7, color=INK)
    # body variants have no single mean length (clean, full+w768): plotted at the full-body length
    extra = cs[cs.variant.isin(["CH-cap2048", "CH-full+w768-t1849", "CH-clean"])].sort_values("chunk_f2", ascending=False)
    for i, (_, r) in enumerate(extra.iterrows()):
        x = r.mean_raw_tokens if pd.notna(r.mean_raw_tokens) else 1056
        ax.scatter([x], [r.chunk_f2], color=ORANGE, s=34, zorder=4, label="body variants" if i == 0 else None,
                   edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(f"{r.variant} {r.chunk_f2:.4f}", (x, r.chunk_f2), xytext=(1040, 0.29 - 0.03 * i),
                    textcoords="data", fontsize=7, color=INK2, va="center", ha="right",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
    ax.set_xlabel("mean chunk length (BGE-M3 tokens)")
    ax.set_ylabel("Chunk F2 (metric v2)")
    ax.set_ylim(0, 0.42)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "chunk_f2_by_size")


def doc_count() -> None:
    dc = pd.read_csv(os.path.join(DATA, "doc_count_curve.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
    free = dc[dc.frame.str.startswith("free")].sort_values("doc_per_query")
    pa4 = dc[dc.frame.str.startswith("P-A4")].sort_values("doc_per_query")
    a1.plot(free.doc_per_query, free.doc_f2, color=BLUE, marker="o", markersize=6, label="free frame (2026-10-26/27)")
    a1.plot(pa4.doc_per_query, pa4.doc_f2, color=ORANGE, marker="o", markersize=6, label="P-A4 doc-level (2026-10-20)")
    for _, r in pd.concat([free, pa4]).iterrows():
        a1.annotate(r.run, (r.doc_per_query, r.doc_f2), xytext=(4, 5), textcoords="offset points", fontsize=6.8, color=INK2)
    a1.set_xlabel("documents per query (mean)")
    a1.set_ylabel("Doc F2 (public LB)")
    a1.set_title("(a) Doc F2 vs number of documents", fontsize=9, loc="left")
    a1.legend(loc="center right", fontsize=7.5)
    pr = free.dropna(subset=["doc_p"])
    a2.plot(pr.doc_per_query, pr.doc_p, color=BLUE, marker="o", markersize=6, label="Doc P")
    a2.plot(pr.doc_per_query, pr.doc_r, color=ORANGE, marker="s", markersize=6, label="Doc R")
    a2.set_xlabel("documents per query (mean), free frame")
    a2.set_ylim(0, 0.8)
    a2.set_title("(b) precision / recall", fontsize=9, loc="left")
    a2.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    save(fig, "doc_f2_by_count")


def ablation() -> None:
    ab = pd.read_csv(os.path.join(DATA, "ablation_free.csv"))
    ab["d"] = ab.delta_final.astype(str).str.replace("+", "", regex=False).astype(float)
    ab = ab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.barh(range(len(ab)), ab.d, color=[BLUE if v >= 0 else ORANGE for v in ab.d], height=0.6)
    ax.set_yticks(range(len(ab)), [v.split(" (")[0] for v in ab.variant], fontsize=7.5)
    ax.axvline(0, color=INK2, linewidth=0.8)
    for i, (v, f) in enumerate(zip(ab.d, ab.final)):
        ax.text(v + (0.0003 if v >= 0 else -0.0003), i, f"{v:+.4f} ({f:.4f})", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.8, color=INK2)
    ax.set_xlim(min(ab.d.min(), 0) - 0.0045, max(ab.d.max(), 0) + 0.0035)
    ax.set_xlabel("Δ Final vs control (public LB; one component changed)")
    ax.grid(axis="y", visible=False)
    save(fig, "ablation_free")


def pipeline() -> None:
    steps = [
        ("Query (vi)", "query text\nas given"),
        ("BGE-M3\ndoc-level", "doc_text\n512 tokens;\nvi + zh corpus"),
        ("Pool\ntop-300", "cosine order;\nties: canonical\nfirst"),
        ("Rerank", "Qwen3-Rer. 4B\n(P4B); 0.6B\nC-max top-3\nt256 chunks"),
        ("RRF\nk = 60", "cosine + 4B\n+ C-max\n(no Jina)"),
        ("Calibrate", "p_cal = σ(a·\nlogit P4B + b)\nG_q = s·Σp_cal\n(top-211 + tail)"),
        ("EF2 cut", "keep doc\nwhile p_cal\n> F2_q / 5"),
        ("Carriers", "CR-fair:\n⌈ρ|D_q|⌉ in\nRRF order\nρ = 0.7937"),
        ("Chunks", "full body;\n+ 768-token\nwindow if\nbody > 1536"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (title, body) in enumerate(steps):
        box = FancyBboxPatch((i + 0.06, 0.25), 0.82, 0.62, boxstyle="round,pad=0.01,rounding_size=0.04",
                             facecolor="#e6f0fb" if i not in (4, 6) else "#fdebe3",
                             edgecolor=BLUE if i not in (4, 6) else ORANGE, linewidth=1.2)
        ax.add_patch(box)
        ax.text(i + 0.47, 0.76, title, ha="center", va="center", fontsize=7.6, weight="bold", color=INK)
        ax.text(i + 0.47, 0.44, body, ha="center", va="center", fontsize=6.4, color=INK2)
        if i < len(steps) - 1:
            ax.annotate("", (i + 1.06, 0.56), (i + 0.88, 0.56), arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.text(0.06, 0.08, "ZIP < 102.600.000 B; thin carriers (≥ 50% base64/placeholder) dropped; "
            "constants a, b, s fitted on aggregate public LB numbers only.", fontsize=7, color=INK2)
    save(fig, "pipeline")


if __name__ == "__main__":
    lb_progress()
    fingerprint()
    chunk_size()
    doc_count()
    ablation()
    pipeline()
    print("figures written to", os.path.abspath(FIG))
