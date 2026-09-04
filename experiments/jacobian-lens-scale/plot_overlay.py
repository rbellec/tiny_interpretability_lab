"""Overlay figures: median delta (J-Lens minus logit lens) per layer, several models on
one relative-depth axis. Position 0 and the final layer excluded, as everywhere else.
Groups are editable below.
    python plot_overlay.py  ->  figures/overlay_*.png"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
FIG = HERE / "figures"

GROUPS = {
    "gemma_all":       ["gemma-3-270m", "gemma-3-270m-it", "gemma-3-1b", "gemma-3-1b-it",
                        "gemma-3-4b", "gemma-3-4b-it", "gemma-3-12b", "gemma-3-12b-it"],
    "gemma_emergence": ["gemma-3-4b", "gemma-3-4b-it", "gemma-3-12b", "gemma-3-12b-it"],
    "gemma_flat":      ["gemma-3-270m", "gemma-3-270m-it", "gemma-3-1b", "gemma-3-1b-it"],
    "qwen35_all":      ["qwen3.5-0.8b", "qwen3.5-2b-pt", "qwen3.5-4b", "qwen3.5-9b-pt"],
}
TITLES = {
    "gemma_all": "gemma-3, all eight models",
    "gemma_emergence": "gemma-3, models with a lens-specific band (4b, 12b)",
    "gemma_flat": "gemma-3, models without one (270m, 1b)",
    "qwen35_all": "qwen3.5, all four models",
}
SIZE_RANK = {"270m": 0, "0.8b": 0, "1b": 1, "2b": 1, "4b": 2, "9b": 3, "12b": 3}


def delta(m):
    d = json.load(open(HERE / f"results/kurtosis_{m}.json"))
    L, sl = d["meta"]["n_layers"], d["meta"]["seq_lens"]
    pos = np.concatenate([np.arange(n) for n in sl]); keep = pos >= 1
    J = np.array([d["kurtosis"]["jlens"][str(l)] for l in range(L - 1)])[:, keep]
    C = np.array([d["kurtosis"]["control"][str(l)] for l in range(L - 1)])[:, keep]
    return 100 * np.arange(L - 1) / (L - 1), np.median(J, 1) - np.median(C, 1)


def style(m):
    size = next(s for s in SIZE_RANK if f"-{s}" in m)
    color = plt.cm.viridis(SIZE_RANK[size] / 3)
    ls = "--" if m.endswith("-it") else "-"
    return dict(color=color, ls=ls, lw=1.8, marker="o", ms=2.5)


for name, models in GROUPS.items():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(38, 92, color="grey", alpha=.08, label="paper band (Claude)")
    ax.axhline(0, color="k", lw=.6)
    for m in models:
        dep, D = delta(m)
        ax.plot(dep, D, label=m, **style(m))
    ax.set_xlabel("relative depth (%)")
    ax.set_ylabel("Δ excess kurtosis  (median J-Lens − median logit lens)")
    ax.set_title(f"Lens-specific kurtosis — {TITLES[name]}")
    ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / f"overlay_{name}.png", dpi=140); plt.close(fig)
    print("->", FIG / f"overlay_{name}.png")
