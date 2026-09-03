"""Lens-specific figures (J-Lens minus logit-lens control):
  1. diff_<short>.png / diff_all_models.png: J minus control (median per layer),
     95% CI by bootstrap OVER PROMPTS (positions within a prompt are correlated,
     so prompts are resampled, not positions). Position 0 (BOS) and the final
     layer (identical in all arms by construction) are EXCLUDED.
  2. position_all_models.png: same J-C, by token-position bucket in the prompt.
  3. position_within_prompt.txt: control for the "prompt" confounder — on the
     long prompts only (>=16 tokens), positions 1-5 vs 11-15 of the SAME prompts.
  4. heatmap_<short>.png: J-C over (depth x position) for the larger models.
With the tag "_long" (argv[1]) the same figures are produced from the long-prompt
run (kurtosis_<short>_long.json), with wider position buckets and a 40-59 late window.
Reads only results/kurtosis_<short>[<tag>].json — no model is loaded.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else ""      # e.g. "_long" -> reads kurtosis_<short>_long.json

HERE = Path(__file__).parent
FIG = HERE / "figures"
ORDER_ALL = ['gemma-3-270m', 'gemma-3-270m-it', 'gemma-3-1b', 'gemma-3-1b-it',
         'gemma-3-4b', 'gemma-3-4b-it', 'gemma-3-12b', 'gemma-3-12b-it',
         'qwen3.5-0.8b', 'qwen3.5-2b-pt', 'qwen3.5-4b', 'qwen3.5-9b-pt']
BUCKETS = [(1, 3), (4, 7), (8, 12), (13, 99)] if not TAG else [(1, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 999)]
rng = np.random.default_rng(0)
ORDER = [m for m in ORDER_ALL if (HERE / f'results/kurtosis_{m}{TAG}.json').exists()]


def load(m):
    d = json.load(open(HERE / f'results/kurtosis_{m}{TAG}.json'))
    L = d['meta']['n_layers']; sl = d['meta']['seq_lens']
    J = np.array([d['kurtosis']['jlens'][str(l)] for l in range(L)])
    C = np.array([d['kurtosis']['control'][str(l)] for l in range(L)])
    pid = np.concatenate([[i] * n for i, n in enumerate(sl)])
    pos = np.concatenate([np.arange(n) for n in sl])
    assert len(pid) == J.shape[1]
    keep = pos >= 1                      # BOS excluded
    J, C, pid, pos = J[:-1, keep], C[:-1, keep], pid[keep], pos[keep]   # final layer excluded
    dep = 100 * np.arange(L - 1) / (L - 1)
    return J, C, pid, pos, dep


def diff_profile(J, C, mask):
    return np.median(J[:, mask], 1) - np.median(C[:, mask], 1)


def boot_ci(J, C, pid, n=1000):
    P = pid.max() + 1; out = []
    for _ in range(n):
        s = rng.integers(0, P, P)
        idx = np.concatenate([np.where(pid == i)[0] for i in s])
        out.append(diff_profile(J[:, idx], C[:, idx], np.ones(len(idx), bool)))
    return np.percentile(out, [2.5, 97.5], axis=0)


# ---------- 1. J - C with bootstrap CI ----------
ncol = 4; nrow = -(-len(ORDER) // ncol)
fig_all, axes = plt.subplots(nrow, ncol, figsize=(16, 3 * nrow), sharex=True, squeeze=False)
for ax, m in zip(axes.flat, ORDER):
    J, C, pid, pos, dep = load(m)
    D = diff_profile(J, C, np.ones(J.shape[1], bool))
    lo, hi = boot_ci(J, C, pid)
    for a in (ax,):
        a.axvspan(38, 92, color='grey', alpha=.08, label='paper band (Claude)')
        a.axhline(0, color='k', lw=.6)
        a.plot(dep, D, 'o-', ms=3, color='C0', label='J-Lens − logit lens (median)')
        a.fill_between(dep, lo, hi, alpha=.25, color='C0', label='95% CI, bootstrap over prompts')
        a.set_title(m); a.grid(alpha=.3)
    f1, a1 = plt.subplots(figsize=(7.5, 4.2))
    a1.axvspan(38, 92, color='grey', alpha=.08, label='paper band (Claude)')
    a1.axhline(0, color='k', lw=.6)
    a1.plot(dep, D, 'o-', ms=3, color='C0', label='J-Lens − logit lens (median)')
    a1.fill_between(dep, lo, hi, alpha=.25, color='C0', label='95% CI, bootstrap over 20 prompts')
    a1.set_xlabel('relative depth (%)'); a1.set_ylabel('Δ excess kurtosis (J − control)')
    a1.set_title(f'{m} — lens-specific kurtosis (pos ≥ 1, final layer excluded)')
    a1.legend(); a1.grid(alpha=.3); f1.tight_layout()
    f1.savefig(FIG / f'diff_{m}{TAG}.png', dpi=130); plt.close(f1)
axes[0, 0].legend(fontsize=8)
for a in axes[-1]: a.set_xlabel('relative depth (%)')
for a in axes[:, 0]: a.set_ylabel('Δ g2 (J − control)')
fig_all.suptitle('J-Lens minus logit-lens control — median over prompts×positions (pos ≥ 1), 95% CI by prompt bootstrap, final layer excluded')
fig_all.tight_layout(); fig_all.savefig(FIG / f'diff_all_models{TAG}.png', dpi=120); plt.close(fig_all)

# ---------- 2. by position bucket ----------
fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3 * nrow), sharex=True, squeeze=False)
for ax, m in zip(axes.flat, ORDER):
    J, C, pid, pos, dep = load(m)
    ax.axvspan(38, 92, color='grey', alpha=.08); ax.axhline(0, color='k', lw=.6)
    for k, (lo_, hi_) in enumerate(BUCKETS):
        mask = (pos >= lo_) & (pos <= hi_)
        if mask.sum() < 10: continue
        nprompt = len(set(pid[mask]))
        lab = f'pos {lo_}-{hi_ if hi_ < 99 else "+"} (n={mask.sum()}, {nprompt} prompts)'
        ax.plot(dep, diff_profile(J, C, mask), '-', lw=1.2 + .5 * k, color=plt.cm.viridis(k / max(1, len(BUCKETS) - 1)), label=lab)
    ax.set_title(m); ax.grid(alpha=.3); ax.legend(fontsize=7)
    # per-model copy of the same panel
    f1, a1 = plt.subplots(figsize=(7.5, 4.2))
    a1.axvspan(38, 92, color='grey', alpha=.08); a1.axhline(0, color='k', lw=.6)
    for k, (lo_, hi_) in enumerate(BUCKETS):
        mask = (pos >= lo_) & (pos <= hi_)
        if mask.sum() < 10: continue
        a1.plot(dep, diff_profile(J, C, mask), '-', lw=1.2 + .5 * k, color=plt.cm.viridis(k / max(1, len(BUCKETS) - 1)),
                label=f'pos {lo_}-{hi_ if hi_ < 999 else "+"} (n={mask.sum()}, {len(set(pid[mask]))} prompts)')
    a1.set_xlabel('relative depth (%)'); a1.set_ylabel('Δ g2 (J − control)')
    a1.set_title(f'{m}{TAG} — lens-specific kurtosis by token position'); a1.legend(fontsize=8); a1.grid(alpha=.3)
    f1.tight_layout(); f1.savefig(FIG / f'position_{m}{TAG}.png', dpi=130); plt.close(f1)
for a in axes[-1]: a.set_xlabel('relative depth (%)')
for a in axes[:, 0]: a.set_ylabel('Δ g2 (J − control)')
fig.suptitle('Lens-specific kurtosis by token position in the prompt (median over prompts×positions in bucket)')
fig.tight_layout(); fig.savefig(FIG / f'position_all_models{TAG}.png', dpi=120); plt.close(fig)

# ---------- 3. within-prompt control for the prompt confounder ----------
lines = [f'TAG={TAG!r} — model | long prompts | peak(J-C) pos1-5 | peak(J-C) late window (11-15, or 40-59 for _long) | per-prompt sign (late>early) | CI95 boot of (late-early) peak']
for m in ORDER:
    J, C, pid, pos, dep = load(m)
    d = json.load(open(HERE / f'results/kurtosis_{m}{TAG}.json'))
    MINLEN, LATE = (16, (11, 15)) if not TAG else (60, (40, 59))
    longp = [i for i, n in enumerate(d['meta']['seq_lens']) if n >= MINLEN]
    band = (dep >= 55) & (dep <= 95)
    sel = np.isin(pid, longp)
    e = sel & (pos >= 1) & (pos <= 5); l = sel & (pos >= LATE[0]) & (pos <= LATE[1])
    pe = diff_profile(J, C, e)[band].max(); pl = diff_profile(J, C, l)[band].max()
    signs = 0; boots = []
    for i in longp:
        ei = e & (pid == i); li = l & (pid == i)
        signs += diff_profile(J, C, li)[band].max() > diff_profile(J, C, ei)[band].max()
    for _ in range(500):
        s = rng.choice(longp, len(longp))
        eb = np.concatenate([np.where(e & (pid == i))[0] for i in s]); lb = np.concatenate([np.where(l & (pid == i))[0] for i in s])
        boots.append(diff_profile(J, C, np.isin(np.arange(J.shape[1]), lb))[band].max() - diff_profile(J, C, np.isin(np.arange(J.shape[1]), eb))[band].max())
    clo, chi = np.percentile(boots, [2.5, 97.5])
    lines.append(f'{m:16s} | {len(longp):2d} | {pe:5.2f} | {pl:5.2f} | {signs}/{len(longp)} | [{clo:5.2f}, {chi:5.2f}]')
(HERE / f'position_within_prompt{TAG}.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

# ---------- 4. depth x position heatmaps ----------
for m in [x for x in ['gemma-3-4b', 'gemma-3-12b', 'qwen3.5-9b-pt'] if x in ORDER]:
    J, C, pid, pos, dep = load(m)
    P = 20 if not TAG else min(int(pos.max()), 120)
    H = np.full((len(dep), P), np.nan)
    for p in range(1, P + 1):
        mask = pos == p
        if mask.sum() >= 5: H[:, p - 1] = diff_profile(J, C, mask)
    f, a = plt.subplots(figsize=(8, 5))
    im = a.imshow(H.T, aspect='auto', origin='lower', cmap='magma',
                  extent=[dep[0], dep[-1], .5, P + .5])
    a.set_xlabel('relative depth (%)'); a.set_ylabel('token position in prompt')
    a.set_title(f'{m}{TAG} — J-Lens − control, median over prompts at each position')
    f.colorbar(im, label='Δ g2'); f.tight_layout(); f.savefig(FIG / f'heatmap_{m}{TAG}.png', dpi=130); plt.close(f)
print('figures ->', FIG)
