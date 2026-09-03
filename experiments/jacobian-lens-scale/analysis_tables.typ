// Tables d'analyse P6 — 12 modeles, 2 perspectives (par famille, par taille).
// Compiler :  typst compile analysis_tables.typ
#set page(paper: "a4", margin: 1.2cm)
#set text(size: 9pt)

#let row(short, family, variant, size, others, layers, dmodel) = (
  [
    #text(weight: "bold", size: 10pt)[#short] \
    famille : #family (#others autres) \
    variante : #variant \
    taille : #size \
    #text(fill: gray)[#layers couches · d_model #dmodel]
  ],
  image("figures/sweep_kurtosis_" + short + ".png", width: 100%),
)

// short, famille, variante, taille, autres-de-la-famille, n_layers, d_model
#let models = (
  ("gemma-3-270m",    "gemma-3",  "pt", "270M", 7, 18, 640),
  ("gemma-3-270m-it", "gemma-3",  "it", "270M", 7, 18, 640),
  ("gemma-3-1b",      "gemma-3",  "pt", "1B",   7, 26, 1152),
  ("gemma-3-1b-it",   "gemma-3",  "it", "1B",   7, 26, 1152),
  ("gemma-3-4b",      "gemma-3",  "pt", "4B",   7, 34, 2560),
  ("gemma-3-4b-it",   "gemma-3",  "it", "4B",   7, 34, 2560),
  ("gemma-3-12b",     "gemma-3",  "pt", "12B",  7, 48, 3840),
  ("gemma-3-12b-it",  "gemma-3",  "it", "12B",  7, 48, 3840),
  ("qwen3.5-0.8b",    "qwen3.5",  "pt", "0.8B", 3, 24, 1024),
  ("qwen3.5-2b-pt",   "qwen3.5",  "pt", "2B",   3, 24, 2048),
  ("qwen3.5-4b",      "qwen3.5",  "pt", "4B",   3, 32, 2560),
  ("qwen3.5-9b-pt",   "qwen3.5",  "pt", "9B",   3, 32, 4096),
)

// ordre par taille croissante (tie-break : famille puis variante pt < it)
#let by_size = (
  ("gemma-3-270m",    "gemma-3",  "pt", "270M", 7, 18, 640),
  ("gemma-3-270m-it", "gemma-3",  "it", "270M", 7, 18, 640),
  ("qwen3.5-0.8b",    "qwen3.5",  "pt", "0.8B", 3, 24, 1024),
  ("gemma-3-1b",      "gemma-3",  "pt", "1B",   7, 26, 1152),
  ("gemma-3-1b-it",   "gemma-3",  "it", "1B",   7, 26, 1152),
  ("qwen3.5-2b-pt",   "qwen3.5",  "pt", "2B",   3, 24, 2048),
  ("gemma-3-4b",      "gemma-3",  "pt", "4B",   7, 34, 2560),
  ("gemma-3-4b-it",   "gemma-3",  "it", "4B",   7, 34, 2560),
  ("qwen3.5-4b",      "qwen3.5",  "pt", "4B",   3, 32, 2560),
  ("qwen3.5-9b-pt",   "qwen3.5",  "pt", "9B",   3, 32, 4096),
  ("gemma-3-12b",     "gemma-3",  "pt", "12B",  7, 48, 3840),
  ("gemma-3-12b-it",  "gemma-3",  "it", "12B",  7, 48, 3840),
)

#let make_table(rows) = table(
  columns: (4.2cm, 1fr),
  stroke: 0.4pt + gray,
  inset: 6pt,
  align: (left + horizon, center + horizon),
  ..rows.map(m => row(..m)).flatten(),
)

= Sweep kurtosis P6 — vue par famille
#text(fill: gray)[20 prompts, toutes positions valides, estimateur g2 biaisé (float64), médiane ± IQR. Généré le 3 sept. 2026.]

#make_table(models)

#pagebreak()

= Sweep kurtosis P6 — vue par taille croissante

#make_table(by_size)
