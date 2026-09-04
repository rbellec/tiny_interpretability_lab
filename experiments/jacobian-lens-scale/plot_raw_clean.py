"""Raw J-Lens vs logit-lens profiles (median ± IQR), with the SAME exclusions as the
Δ figures: position 0 (BOS) and the final layer are dropped. One PNG per model
(figures/raw_<short>.png) plus a 2x2 panel for qwen3.5 (figures/raw_qwen35_panel.png).

    .venv/bin/python plot_raw_clean.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ORDER = ["gemma-3-270m", "gemma-3-270m-it", "gemma-3-1b", "gemma-3-1b-it",
         "gemma-3-4b", "gemma-3-4b-it", "gemma-3-12b", "gemma-3-12b-it",
         "qwen3.5-0.8b", "qwen3.5-2b-pt", "qwen3.5-4b", "qwen3.5-9b-pt"]

def load(short):
    d = json.load(open(HERE / f"results/kurtosis_{short}.json"))
    L, sl = d["meta"]["n_layers"], d["meta"]["seq_lens"]
    pos = np.concatenate([np.arange(s) for s in sl]); keep = pos >= 1
    out = {}
    for arm in ("jlens", "control"):
        A = np.array([d["kurtosis"][arm][str(l)] for l in range(L - 1)])[:, keep]
        out[arm] = (np.median(A, 1), np.percentile(A, 25, 1), np.percentile(A, 75, 1))
    return 100 * np.arange(L - 1) / (L - 1), out, d["meta"]

def draw(ax, short, title=True):
    dep, arms, meta = load(short)
    for arm, style, label in (("jlens", "o-", "J-Lens"), ("control", "s--", "logit lens (control)")):
        med, q1, q3 = arms[arm]
        ax.plot(dep, med, style, label=label, ms=3); ax.fill_between(dep, q1, q3, alpha=.15)
    if title: ax.set_title(f"{short} — {meta['n_prompts']} prompts, pos ≥ 1, final layer excluded", fontsize=10)
    ax.grid(alpha=.3)

for m in ORDER:
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); draw(ax, m)
    ax.set_xlabel("relative depth (%)"); ax.set_ylabel("excess kurtosis (g2), median ± IQR"); ax.legend()
    fig.tight_layout(); p = HERE / "figures" / f"raw_{m}.png"; fig.savefig(p, dpi=130); plt.close(fig); print("->", p.name)

fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, m in zip(axs.flat, ORDER[8:]):
    draw(ax, m); ax.set_title(m, fontsize=11)
for ax in axs[1]: ax.set_xlabel("relative depth (%)")
for ax in axs[:, 0]: ax.set_ylabel("g2, median ± IQR")
axs[0, 0].legend(fontsize=9)
fig.suptitle("qwen3.5 — raw J-Lens vs logit-lens profiles (pos ≥ 1, final layer excluded)")
fig.tight_layout(); p = HERE / "figures" / "raw_qwen35_panel.png"; fig.savefig(p, dpi=130); print("->", p.name)
