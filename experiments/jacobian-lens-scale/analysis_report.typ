// Genere par make_report.py — ne pas editer a la main.
#set page(paper: "a4", margin: 1.1cm)
#set text(size: 8.5pt)
#let fig(p, w) = if sys.inputs.at("nofig", default: "0") == "1" { rect(width: w, height: 3cm) } else { image(p, width: w) }

= Sweep kurtosis P6 — rapport de lecture
#text(fill: gray)[20 prompts, positions ≥ 1 (BOS exclue), dernière couche exclue, estimateur g2 biaisé float64. J−C = médiane J-Lens moins médiane logit-lens ; IC95 = bootstrap sur les 20 prompts ; « bosse » = pic de J−C sur 55-95 % moins base (médiane 25-55 %). Bande grise = bande workspace du papier (Claude, 38-92 %). Généré le 3 sept. 2026.]
== Vue d'ensemble — J-Lens moins contrôle, tous modèles
#fig("figures/diff_all_models.png", 100%)
== Vue d'ensemble — par position dans le prompt
#fig("figures/position_all_models.png", 100%)
#pagebreak()
= Famille gemma-3
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-270m] #h(1em) #text(fill: gray)[18 couches · d\_model 640 · float32]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.33], [0.58 \@ 65 %], [0.25], [\[0.14 ; 0.37\]], [20/20], [—], [0.43 → 0.80 (7/10)])
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-270m.png", 100%), fig("figures/diff_gemma-3-270m.png", 100%), fig("figures/position_gemma-3-270m.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-270m-it] #h(1em) #text(fill: gray)[18 couches · d\_model 640 · float32]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.19], [0.38 \@ 65 %], [0.19], [\[0.12 ; 0.30\]], [20/20], [—], [0.27 → 0.57 (9/10)])
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-270m-it.png", 100%), fig("figures/diff_gemma-3-270m-it.png", 100%), fig("figures/position_gemma-3-270m-it.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-1b] #h(1em) #text(fill: gray)[26 couches · d\_model 1152 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.31], [0.35 \@ 68 %], [0.05], [\[−0.23 ; 0.30\]], [13/20], [—], [0.12 → 0.50 (7/10)])
#text(size: 8pt)[J−C passe sous zéro dans la bande (min −0.71) · IC95 de la bosse contient zéro · contrôle plus piqué que J-Lens en fin de réseau (2.14 vs 1.26 à l'avant-dernière couche)]
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-1b.png", 100%), fig("figures/diff_gemma-3-1b.png", 100%), fig("figures/position_gemma-3-1b.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-1b-it] #h(1em) #text(fill: gray)[26 couches · d\_model 1152 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.60], [0.75 \@ 60 %], [0.16], [\[−0.00 ; 0.29\]], [12/20], [48 %], [0.92 → 0.98 (5/10)])
#text(size: 8pt)[J−C passe sous zéro dans la bande (min −2.29) · IC95 de la bosse contient zéro · contrôle plus piqué que J-Lens en fin de réseau (3.40 vs 2.24 à l'avant-dernière couche)]
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-1b-it.png", 100%), fig("figures/diff_gemma-3-1b-it.png", 100%), fig("figures/position_gemma-3-1b-it.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-4b] #h(1em) #text(fill: gray)[34 couches · d\_model 2560 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.23], [1.23 \@ 79 %], [0.99], [\[0.79 ; 1.27\]], [20/20], [64 %], [0.92 → 1.34 (8/10)])
#text(size: 8pt)[effet de position net à l'intérieur des mêmes prompts]
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-4b.png", 100%), fig("figures/diff_gemma-3-4b.png", 100%), fig("figures/position_gemma-3-4b.png", 100%), fig("figures/heatmap_gemma-3-4b.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-4b-it] #h(1em) #text(fill: gray)[34 couches · d\_model 2560 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.41], [0.81 \@ 76 %], [0.40], [\[0.24 ; 0.55\]], [19/20], [76 %], [0.51 → 1.05 (7/10)])
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-4b-it.png", 100%), fig("figures/diff_gemma-3-4b-it.png", 100%), fig("figures/position_gemma-3-4b-it.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-12b] #h(1em) #text(fill: gray)[48 couches · d\_model 3840 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.51], [2.37 \@ 79 %], [1.87], [\[1.58 ; 2.17\]], [20/20], [62 %], [1.51 → 2.64 (9/10)])
#text(size: 8pt)[effet de position net à l'intérieur des mêmes prompts]
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-12b.png", 100%), fig("figures/diff_gemma-3-12b.png", 100%), fig("figures/position_gemma-3-12b.png", 100%), fig("figures/heatmap_gemma-3-12b.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[gemma-3-12b-it] #h(1em) #text(fill: gray)[48 couches · d\_model 3840 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.66], [1.81 \@ 79 %], [1.15], [\[0.90 ; 1.37\]], [20/20], [70 %], [1.09 → 2.44 (10/10)])
#text(size: 8pt)[effet de position net à l'intérieur des mêmes prompts]
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_gemma-3-12b-it.png", 100%), fig("figures/diff_gemma-3-12b-it.png", 100%), fig("figures/position_gemma-3-12b-it.png", 100%))
]
#v(6pt)
#pagebreak()
= Famille qwen3.5
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[qwen3.5-0.8b] #h(1em) #text(fill: gray)[24 couches · d\_model 1024 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.36], [0.82 \@ 87 %], [0.46], [\[0.39 ; 0.58\]], [20/20], [78 %], [0.55 → 0.89 (7/8)])
#text(size: 8pt)[effet de position net à l'intérieur des mêmes prompts]
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_qwen3.5-0.8b.png", 100%), fig("figures/diff_qwen3.5-0.8b.png", 100%), fig("figures/position_qwen3.5-0.8b.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[qwen3.5-2b-pt] #h(1em) #text(fill: gray)[24 couches · d\_model 2048 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.92], [1.06 \@ 91 %], [0.13], [\[0.05 ; 0.29\]], [19/20], [—], [0.89 → 1.00 (5/8)])
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_qwen3.5-2b-pt.png", 100%), fig("figures/diff_qwen3.5-2b-pt.png", 100%), fig("figures/position_qwen3.5-2b-pt.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[qwen3.5-4b] #h(1em) #text(fill: gray)[32 couches · d\_model 2560 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [1.29], [1.60 \@ 84 %], [0.31], [\[0.21 ; 0.52\]], [19/20], [84 %], [1.44 → 1.61 (3/8)])
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_qwen3.5-4b.png", 100%), fig("figures/diff_qwen3.5-4b.png", 100%), fig("figures/position_qwen3.5-4b.png", 100%))
]
#v(6pt)
#block(breakable: false)[
#text(weight: "bold", size: 11pt)[qwen3.5-9b-pt] #h(1em) #text(fill: gray)[32 couches · d\_model 4096 · bfloat16]

#table(columns: (auto, auto, auto, auto, auto, auto, auto), stroke: 0.3pt + gray, inset: 4pt, align: center,
  [*J−C base*], [*pic J−C*], [*bosse*], [*IC95 (prompts)*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [0.77], [1.41 \@ 84 %], [0.64], [\[0.44 ; 0.82\]], [20/20], [77 %], [1.00 → 1.51 (6/8)])
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/sweep_kurtosis_qwen3.5-9b-pt.png", 100%), fig("figures/diff_qwen3.5-9b-pt.png", 100%), fig("figures/position_qwen3.5-9b-pt.png", 100%), fig("figures/heatmap_qwen3.5-9b-pt.png", 100%))
]
#v(6pt)
#pagebreak()
= Axe contexte — run « prompts longs »
#text(fill: gray)[20 prompts de 110-158 tokens (`prompts_long.txt`), mêmes bras, même estimateur ; positions ≥ 1, dernière couche exclue. Modèles : gemma-3-270m, gemma-3-4b, gemma-3-12b, qwen3.5-9b-pt.]
== J−C par tranche de position
#fig("figures/position_all_models_long.png", 100%)
== J−C avec IC95 par prompt
#fig("figures/diff_all_models_long.png", 100%)
== Contrôle intra-prompt : pic J−C positions 1-5 vs 40-59 des mêmes prompts
#table(columns: 6, stroke: 0.3pt + gray, inset: 4pt, align: center, [*modèle*], [*prompts*], [*pic pos 1-5*], [*pic pos 40-59*], [*tardif > précoce*], [*IC95 écart*],
  [gemma-3-270m], [20], [0.51], [0.76], [11/20], [[ 0.05,  0.37]],
  [gemma-3-4b], [20], [0.98], [1.32], [10/20], [[ 0.09,  0.60]],
  [gemma-3-12b], [20], [1.32], [2.26], [14/20], [[ 0.33,  1.26]],
  [qwen3.5-9b-pt], [20], [1.17], [1.35], [12/20], [[-0.01,  0.34]],
)
== Heatmaps profondeur × position
#grid(columns: (1fr, 1fr), gutter: 4pt, fig("figures/heatmap_gemma-3-4b_long.png", 100%), fig("figures/heatmap_gemma-3-12b_long.png", 100%), fig("figures/heatmap_qwen3.5-9b-pt_long.png", 100%))
#pagebreak()
= Tableau récapitulatif (par famille, taille croissante)
#table(columns: 9, stroke: 0.3pt + gray, inset: 4pt, align: center, [*modèle*], [*couches*], [*J−C base*], [*pic*], [*bosse*], [*IC95*], [*prompts > 0*], [*onset*], [*pos 1-5 → 11-15*],
  [gemma-3-270m], [18], [0.33], [0.58 \@ 65 %], [0.25], [\[0.14 ; 0.37\]], [20/20], [—], [0.43 → 0.80 (7/10)],
  [gemma-3-270m-it], [18], [0.19], [0.38 \@ 65 %], [0.19], [\[0.12 ; 0.30\]], [20/20], [—], [0.27 → 0.57 (9/10)],
  [gemma-3-1b], [26], [0.31], [0.35 \@ 68 %], [0.05], [\[−0.23 ; 0.30\]], [13/20], [—], [0.12 → 0.50 (7/10)],
  [gemma-3-1b-it], [26], [0.60], [0.75 \@ 60 %], [0.16], [\[−0.00 ; 0.29\]], [12/20], [48 %], [0.92 → 0.98 (5/10)],
  [gemma-3-4b], [34], [0.23], [1.23 \@ 79 %], [0.99], [\[0.79 ; 1.27\]], [20/20], [64 %], [0.92 → 1.34 (8/10)],
  [gemma-3-4b-it], [34], [0.41], [0.81 \@ 76 %], [0.40], [\[0.24 ; 0.55\]], [19/20], [76 %], [0.51 → 1.05 (7/10)],
  [gemma-3-12b], [48], [0.51], [2.37 \@ 79 %], [1.87], [\[1.58 ; 2.17\]], [20/20], [62 %], [1.51 → 2.64 (9/10)],
  [gemma-3-12b-it], [48], [0.66], [1.81 \@ 79 %], [1.15], [\[0.90 ; 1.37\]], [20/20], [70 %], [1.09 → 2.44 (10/10)],
  [qwen3.5-0.8b], [24], [0.36], [0.82 \@ 87 %], [0.46], [\[0.39 ; 0.58\]], [20/20], [78 %], [0.55 → 0.89 (7/8)],
  [qwen3.5-2b-pt], [24], [0.92], [1.06 \@ 91 %], [0.13], [\[0.05 ; 0.29\]], [19/20], [—], [0.89 → 1.00 (5/8)],
  [qwen3.5-4b], [32], [1.29], [1.60 \@ 84 %], [0.31], [\[0.21 ; 0.52\]], [19/20], [84 %], [1.44 → 1.61 (3/8)],
  [qwen3.5-9b-pt], [32], [0.77], [1.41 \@ 84 %], [0.64], [\[0.44 ; 0.82\]], [20/20], [77 %], [1.00 → 1.51 (6/8)],
)