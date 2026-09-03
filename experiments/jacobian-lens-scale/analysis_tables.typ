// Per-model kurtosis profiles, 12 models, two orderings (by family, by size).
// Compile:  typst compile analysis_tables.typ
#set page(paper: "a4", margin: 1.2cm)
#set text(size: 9pt)

#let row(short, family, variant, size, others, layers, dmodel) = (
  [
    #text(weight: "bold", size: 10pt)[#short] \
    family: #family (#others others) \
    variant: #variant \
    size: #size \
    #text(fill: gray)[#layers layers · d_model #dmodel]
  ],
  image("figures/sweep_kurtosis_" + short + ".png", width: 100%),
)

// short, family, variant, size, others-in-family, n_layers, d_model
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

// increasing size (tie-break: family, then variant pt < it)
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

= Kurtosis sweep — by family
#text(fill: gray)[20 prompts, all valid positions, biased g2 estimator (float64), median ± IQR. Generated 3 Sept. 2026.]

#make_table(models)

#pagebreak()

= Kurtosis sweep — by increasing size

#make_table(by_size)
