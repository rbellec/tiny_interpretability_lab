"""Regenerate per-model figures from the stored JSONs.

No model rerun needed: everything lives in results/kurtosis_<short>.json.

    .venv/bin/python plot_per_model.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

for f in sorted(HERE.glob("results/kurtosis_*.json")):
    if "kurtosis_random_" in f.name or f.stem.endswith("_long"):   # main prompt set only
        continue
    data = json.load(open(f))
    meta = data["meta"]
    short = meta["short"]
    n_final = meta["n_layers"] - 1

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for arm, style, label in (("jlens", "o-", "J-Lens"),
                              ("control", "s--", "logit lens (control)")):
        rec = data["kurtosis"][arm]
        layers = sorted(int(l) for l in rec)
        depth = [100 * l / n_final for l in layers]
        med = [st.median(rec[str(l)]) for l in layers]
        q1 = [st.quantiles(rec[str(l)], n=4)[0] for l in layers]
        q3 = [st.quantiles(rec[str(l)], n=4)[2] for l in layers]
        ax.plot(depth, med, style, label=label, ms=3)
        ax.fill_between(depth, q1, q3, alpha=.15)
    ax.set_xlabel("relative depth (%)")
    ax.set_ylabel("excess kurtosis (g2), median ± IQR")
    ax.set_title(f"{short} — {meta['n_prompts']} prompts, all valid positions")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    p = HERE / "figures" / f"sweep_kurtosis_{short}.png"
    fig.savefig(p, dpi=130)
    print(f"figure -> {p}")
