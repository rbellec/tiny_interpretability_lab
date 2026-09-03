"""Combined figure — one model = one curve, relative depth [0-100].

Left panel: J-Lens arm (median g2 per layer).
Middle panel: logit-lens control arm, same scales — cross-model comparison
is only meaningful with the control shown alongside.
Third panel (if present): random-orthogonal control-task arm.

    .venv/bin/python plot_combined.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ORDER = ["gemma-3-270m", "gemma-3-1b", "gemma-3-4b", "gemma-3-12b",
         "qwen3.5-0.8b", "qwen3.5-2b-pt"]

rand_files = sorted(HERE.glob("results/kurtosis_random_*.json"))
n_panels = 3 if rand_files else 2
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5), sharey=True)
for f in sorted(HERE.glob("results/kurtosis_*.json")):
    if "kurtosis_random_" in f.name or f.stem.endswith("_long"):   # main prompt set only
        continue
    data = json.load(open(f))
    short = data["meta"]["short"]
    n_final = data["meta"]["n_layers"] - 1
    ls = "-" if short.startswith("gemma") else "--"
    for ax, arm in ((axes[0], "jlens"), (axes[1], "control")):
        rec = data["kurtosis"][arm]
        layers = sorted(int(l) for l in rec)
        depth = [100 * l / n_final for l in layers]
        med = [st.median(rec[str(l)]) for l in layers]
        ax.plot(depth, med, ls, label=short, lw=1.6)

for f in rand_files:
    data = json.load(open(f))
    short = data["meta"]["short"]
    n_final = data["meta"]["n_layers"] - 1
    rec = data["kurtosis"]
    layers = sorted(int(l) for l in rec)
    depth = [100 * l / n_final for l in layers]
    med = [st.median(rec[str(l)]) for l in layers]
    axes[2].plot(depth, med, ":", label=short, lw=1.6)

axes[0].set_title("J-Lens")
axes[1].set_title("logit lens (control)")
if rand_files:
    axes[2].set_title("random orthogonal J (control task)")
for ax in axes:
    ax.set_xlabel("relative depth (%)")
    ax.grid(alpha=.3)
axes[0].set_ylabel("excess kurtosis (g2), median")
axes[0].legend(fontsize=8)
if rand_files:
    axes[2].legend(fontsize=8)
fig.suptitle("Layerwise readout kurtosis profile across model scales")
fig.tight_layout()
p = HERE / "figures" / "combined_kurtosis_all_models.png"
fig.savefig(p, dpi=140)
print(f"figure -> {p}")
