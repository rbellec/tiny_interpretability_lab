"""Arm 3: delta of the random-rotation arm (R - C) against the J-Lens delta (J - C), gemma
base models, 4 scales. Position 0 and the final layer excluded. Also prints the numbers.
    python plot_random_arm.py -> figures/overlay_random_arm.png"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
MODELS = ["gemma-3-270m", "gemma-3-1b", "gemma-3-4b", "gemma-3-12b"]


def profiles(m):
    d = json.load(open(HERE / f"results/kurtosis_{m}.json"))
    r = json.load(open(HERE / f"results/kurtosis_random_{m}.json"))["kurtosis"]
    L, sl = d["meta"]["n_layers"], d["meta"]["seq_lens"]
    pos = np.concatenate([np.arange(n) for n in sl]); keep = pos >= 1
    med = lambda arr: np.array([np.median(np.array(arr[str(l)])[keep]) for l in range(L - 1)])
    J, C, R = med(d["kurtosis"]["jlens"]), med(d["kurtosis"]["control"]), med(r)
    return 100 * np.arange(L - 1) / (L - 1), J - C, R - C


fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=False)
print("model         | peak Δ(J−C) in 55-95% | peak Δ(R−C) in 55-95% | max |R−C| anywhere")
for ax, m in zip(axes, MODELS):
    dep, DJ, DR = profiles(m)
    band = (dep >= 55) & (dep <= 95)
    print(f"{m:14s} | {DJ[band].max():6.2f} | {DR[band].max():6.2f} | {np.abs(DR).max():5.2f}")
    ax.axvspan(38, 92, color="grey", alpha=.08); ax.axhline(0, color="k", lw=.6)
    ax.plot(dep, DJ, "o-", ms=2.5, color="C0", label="Δ = J-Lens − control")
    ax.plot(dep, DR, "s--", ms=2.5, color="C3", label="random orthogonal J − control")
    ax.set_title(m); ax.set_xlabel("relative depth (%)"); ax.grid(alpha=.3)
axes[0].set_ylabel("Δ excess kurtosis"); axes[0].legend(fontsize=8)
fig.suptitle("Control task: replacing the learned J by a random orthogonal rotation (seed 0) removes the band")
fig.tight_layout(); fig.savefig(HERE / "figures/overlay_random_arm.png", dpi=140)
print("-> figures/overlay_random_arm.png")
