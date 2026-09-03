"""Reading report (PDF, French): overview (J-C for all models), then per family and
size a compact per-model recap + its figures. Reads results/*.json, writes
analysis_report.typ and compiles it with typst.   python make_report.py"""
import json, subprocess
import numpy as np
from pathlib import Path
HERE = Path(__file__).parent
rng = np.random.default_rng(0)
FAM = {'gemma-3': ['gemma-3-270m', 'gemma-3-270m-it', 'gemma-3-1b', 'gemma-3-1b-it', 'gemma-3-4b', 'gemma-3-4b-it', 'gemma-3-12b', 'gemma-3-12b-it'],
       'qwen3.5': ['qwen3.5-0.8b', 'qwen3.5-2b-pt', 'qwen3.5-4b', 'qwen3.5-9b-pt']}

def load(m):
    d = json.load(open(HERE / f'results/kurtosis_{m}.json')); meta = d['meta']
    L = meta['n_layers']; sl = meta['seq_lens']
    J = np.array([d['kurtosis']['jlens'][str(l)] for l in range(L)]); C = np.array([d['kurtosis']['control'][str(l)] for l in range(L)])
    pid = np.concatenate([[i] * n for i, n in enumerate(sl)]); pos = np.concatenate([np.arange(n) for n in sl])
    keep = pos >= 1
    return J[:-1, keep], C[:-1, keep], pid[keep], pos[keep], 100 * np.arange(L - 1) / (L - 1), meta, sl

def prof(J, C, mask): return np.median(J[:, mask], 1) - np.median(C[:, mask], 1)

def stats(m):
    J, C, pid, pos, dep, meta, sl = load(m)
    allm = np.ones(J.shape[1], bool); D = prof(J, C, allm)
    base_m = (dep >= 25) & (dep <= 55); band = (dep >= 55) & (dep <= 95)
    base = np.median(D[base_m]); pk = np.argmax(np.where(band, D, -9)); amp = D[pk] - base
    P = pid.max() + 1; boots = []
    for _ in range(1000):
        s = rng.integers(0, P, P); idx = np.concatenate([np.where(pid == i)[0] for i in s])
        Db = prof(J[:, idx], C[:, idx], np.ones(len(idx), bool)); boots.append(np.max(np.where(band, Db, -9)) - np.median(Db[base_m]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    npos = sum(1 for i in range(P) if (lambda Dp: np.max(np.where(band, Dp, -9)) - np.median(Dp[base_m]))(prof(J, C, pid == i)) > 0)
    minD = D[band].min(); end = D[-1]
    longp = [i for i, n in enumerate(sl) if n >= 16]; sel = np.isin(pid, longp)
    e = sel & (pos <= 5); l = sel & (pos >= 11) & (pos <= 15)
    pe = prof(J, C, e)[band].max(); pl = prof(J, C, l)[band].max()
    signs = sum(prof(J, C, l & (pid == i))[band].max() > prof(J, C, e & (pid == i))[band].max() for i in longp)
    onset = next((d for d, v in zip(dep, D) if d >= 25 and v > base + 0.3), None)
    ctrl_end = np.median(C[-1]); j_end = np.median(J[-1])
    return dict(L=meta['n_layers'], d=meta['d_model'], dtype=meta['dtype'], base=base, peak=D[pk], amp=amp, pkdep=dep[pk], lo=lo, hi=hi,
                npos=npos, P=P, minD=minD, end=end, pe=pe, pl=pl, signs=signs, nlong=len(longp), onset=onset, ctrl_end=ctrl_end, j_end=j_end)

def esc(s): return s.replace('_', '\\_')
def fmt(x): return f'{x:.2f}'.replace('-', '−')

out = ['''// Genere par make_report.py — ne pas editer a la main.
#set page(paper: "a4", margin: 1.1cm)
#set text(size: 8.5pt)
#let fig(p, w) = if sys.inputs.at("nofig", default: "0") == "1" { rect(width: w, height: 3cm) } else { image(p, width: w) }
''',
'= Sweep kurtosis P6 — rapport de lecture',
'#text(fill: gray)[20 prompts, positions ≥ 1 (BOS exclue), dernière couche exclue, estimateur g2 biaisé float64. J−C = médiane J-Lens moins médiane logit-lens ; IC95 = bootstrap sur les 20 prompts ; « bosse » = pic de J−C sur 55-95 % moins base (médiane 25-55 %). Bande grise = bande workspace du papier (Claude, 38-92 %). Généré le 3 sept. 2026.]',
'== Vue d\'ensemble — J-Lens moins contrôle, tous modèles', '#fig("figures/diff_all_models.png", 100%)',
'== Vue d\'ensemble — par position dans le prompt', '#fig("figures/position_all_models.png", 100%)', '#pagebreak()']

summary_rows = []
for fam, models in FAM.items():
    out.append(f'= Famille {fam}')
    for m in models:
        s = stats(m)
        summary_rows.append((m, s))
        heat = (HERE / f'figures/heatmap_{m}.png').exists()
        recap = (f'#text(weight: "bold", size: 11pt)[{esc(m)}] #h(1em) #text(fill: gray)[{s["L"]} couches · d\\_model {s["d"]} · {s["dtype"]}]\n\n'
                 f'#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,\n'
                 f'  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],\n'
                 f'  [{fmt(s["base"])}], [{fmt(s["peak"])} \\@ {s["pkdep"]:.0f} %], [{fmt(s["amp"])}], [\\[{fmt(s["lo"])} ; {fmt(s["hi"])}\\]], [{s["npos"]}/{s["P"]}], '
                 f'[{"—" if s["onset"] is None else f"{s['onset']:.0f} %"}], [{fmt(s["pe"])} → {fmt(s["pl"])} ({s["signs"]}/{s["nlong"]})])\n')
        notes = []
        if s['minD'] < -0.05: notes.append(f'J−C passe sous zéro dans la bande (min {fmt(s["minD"])})')
        if s['lo'] <= 0: notes.append('IC95 de la bosse contient zéro')
        if s['ctrl_end'] > s['j_end'] + 0.3: notes.append(f'contrôle plus piqué que J-Lens en fin de réseau ({fmt(s["ctrl_end"])} vs {fmt(s["j_end"])} à l\'avant-dernière couche)')
        if s['signs'] >= 0.8 * s['nlong'] and s['pl'] - s['pe'] > 0.3: notes.append('effet de position net à l\'intérieur des mêmes prompts')
        recap += ('#text(size: 8pt)[' + ' · '.join(notes) + ']\n') if notes else ''
        figs = [f'figures/sweep_kurtosis_{m}.png', f'figures/diff_{m}.png', f'figures/position_{m}.png'] + ([f'figures/heatmap_{m}.png'] if heat else [])
        grid = '#grid(columns: (1fr, 1fr), gutter: 4pt, ' + ', '.join(f'fig("{f}", 100%)' for f in figs) + ')'
        out.append('#block(breakable: false)[\n' + recap + grid + '\n]\n#v(6pt)')
    out.append('#pagebreak()')

out.append('= Tableau récapitulatif (par famille, taille croissante)')
out.append('#table(columns: 9, stroke: 0.3pt + gray, inset: 4pt, align: center, [*modèle*], [*couches*], [*J−C base*], [*pic*], [*bosse*], [*IC95*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],')
for m, s in summary_rows:
    out.append(f'  [{esc(m)}], [{s["L"]}], [{fmt(s["base"])}], [{fmt(s["peak"])} \\@ {s["pkdep"]:.0f} %], [{fmt(s["amp"])}], [\\[{fmt(s["lo"])} ; {fmt(s["hi"])}\\]], [{s["npos"]}/{s["P"]}], [{"—" if s["onset"] is None else f"{s['onset']:.0f} %"}], [{fmt(s["pe"])} → {fmt(s["pl"])} ({s["signs"]}/{s["nlong"]})],')
out.append(')')
(HERE / 'analysis_report.typ').write_text('\n'.join(out))
print(subprocess.run(['typst', 'compile', 'analysis_report.typ'], cwd=HERE, capture_output=True, text=True).stderr[:2000] or 'compiled analysis_report.pdf')
