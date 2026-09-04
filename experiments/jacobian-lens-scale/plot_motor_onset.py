"""Figure: depth at which delta (median J-Lens minus median logit lens) first goes below
zero after 50 % -- read as the onset of the "motor regime". One point per model, smallest
at the bottom; models that never cross zero are marked separately. Position 0 and the
final layer excluded, as everywhere else.
    python plot_motor_onset.py -> figures/motor_regime_onset.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
MODELS = ["gemma-3-270m", "gemma-3-270m-it", "gemma-3-1b", "gemma-3-1b-it",
          "gemma-3-4b", "gemma-3-4b-it", "gemma-3-12b", "gemma-3-12b-it"]   # smallest first


def crossing(m):
    d = json.load(open(HERE / f"results/kurtosis_{m}.json"))
    L, sl = d["meta"]["n_layers"], d["meta"]["seq_lens"]
    pos = np.concatenate([np.arange(n) for n in sl]); keep = pos >= 1
    dep = 100 * np.arange(L - 1) / (L - 1)
    D = np.array([np.median(np.array(d["kurtosis"]["jlens"][str(l)])[keep])
                  - np.median(np.array(d["kurtosis"]["control"][str(l)])[keep]) for l in range(L - 1)])
    neg = [x for x, v in zip(dep, D) if x > 50 and v < 0]
    return (neg[0] if neg else None), D.min()


fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.axvspan(38, 92, color="grey", alpha=.08, label="paper band (Claude)")
for i, m in enumerate(MODELS):
    x, mn = crossing(m)
    it = m.endswith("-it")
    if x is None:
        ax.plot(100, i, marker="o", mfc="white", mec="C0", ms=8, ls="none")
        ax.text(99, i, "never < 0  ", ha="right", va="center", fontsize=8, color="C0")
    else:
        ax.hlines(i, 50, x, color="C0", lw=1, alpha=.4)
        ax.plot(x, i, marker="s" if it else "o", color="C0", ms=8, ls="none")
        ax.text(x + 1.2, i, f"{x:.0f} %  (min Δ {mn:+.2f})", va="center", fontsize=8)
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS)
ax.set_xlim(50, 122); ax.set_xlabel("relative depth (%) where Δ first goes below zero (after 50 %)")
ax.set_title("Depth where Δ turns negative (logit lens more peaked than J-Lens) — gemma-3")
ax.plot([], [], "o", color="C0", label="base"); ax.plot([], [], "s", color="C0", label="-it")
ax.plot([], [], "o", mfc="white", mec="C0", label="never below zero")
ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=.3, axis="x")
fig.tight_layout(); fig.savefig(HERE / "figures/motor_regime_onset.png", dpi=140)
print("-> figures/motor_regime_onset.png")
