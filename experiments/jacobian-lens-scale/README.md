# The Jacobian lens across model scale

> **Status: measurements done, write-up pending (September 2026).** The kurtosis sweep has run
> on 12 models (two families, 270M-12B, base and instruct) with a logit-lens control arm and a
> random-rotation control-task arm, plus a long-prompt run on 4 models for the position /
> context-length axis. Scripts, prompt sets, per-model JSON results and figures are all here;
> see *Reproducing* for the exact commands and *Status* at the bottom.

## The question

*Verbalizable Representations Form a Global Workspace in Language Models*
([Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html)) reports a
quantitative signature of the "workspace band": the excess kurtosis of the Jacobian-lens
readout logits is low in early layers, rises to a peak around a third of the way through the
network, and falls again.

Publicly reported measurements on small open-weight models (≤2B) do not show that shape:
some models are flat throughout, and the pieces of the curve appear to sort by family rather
than by size.

So:

**Where is the threshold — and is it a matter of scale, or of family and training? And if the
signature moves with the choice of statistic, is it a property of the model at all?**

The second half matters as much as the first. A signature that survives a change of
concentration measure is evidence about the model; one that does not is evidence about the
statistic.

## Method commitments

These are fixed in advance, and hold for every arm:

- **A logit-lens control arm in the same code path** (`use_jacobian=False`), always. Rises are
  measured from each model's own baseline, **never against zero**. This is what separates
  "signature of the lens" from "property of the model".
- **One kurtosis estimator, named.** A fourth moment is high-variance and packages disagree
  (biased `g2`, corrected `G2`, `b2`). One is chosen, kept across all arms, and stated in the
  write-up.
- **A robust companion** (median/quartiles) alongside the moment.
- **A random-J control**: the profile recomputed with a permuted or random Jacobian of matched
  norm, to separate "the learned transport does something" from "any matrix would do this".
- **Relative depth, not absolute layer index.** Models of different depth are only comparable
  at matched relative depth.
- **Sanity checks are documented, not assumed**: any load-bearing number is recomputed
  independently, and the check is reported alongside the number.

## What is in here

| File | What it does |
|---|---|
| `smoke_test.py` | Six independent stages validating the whole chain end to end (device, model boot, `run_with_cache`, an ablation that actually bites, lens loading, both readout arms, one figure). Each stage reports PASS/FAIL separately, so a failure says *where* it broke. Runs in ~20 s. |
| `scout_feasibility.py` | Feasibility only, no science: per model, HF access, boot time, readout time for both arms, layer count, peak RSS. Answers "does this sweep fit on the machine?" before any measurement is made. |
| `scout_results.jsonl` | One JSON line per scouted model, as produced by the script above. |
| `runpod_setup.sh` | Provisioning notes for a rented GPU pod, for the model sizes that do not fit locally. |
| `prompts.txt` | The fixed prompt set of the main sweep: 20 pre-training-style prompts of 6-28 tokens (factual, narrative, arithmetic, multi-hop, code, list). Versioned; cited by the write-up. |
| `prompts_long.txt` | The long-prompt set for the position / context-length axis: 20 prompts of 110-158 tokens (Gemma tokenizer), same genre mix plus dialogue, letter and two Python blocks. Indented lines continue the previous prompt (code blocks). v2, 2026-09-03. |
| `sweep_kurtosis.py` | The measurement. One model, two arms (J-Lens / logit-lens control) through the same code path, excess kurtosis `g2` of the readout logits at every (layer, position). Writes `results/kurtosis_<short>[<tag>].json` (scalars only) and one figure. |
| `sweep_control_task.py` | Third arm: every `J[l]` replaced by a random orthogonal rotation (fixed seed), same readout, same estimator. Writes `results/kurtosis_random_<short>.json`. |
| `check_estimator.py` | Sanity check of the sweep's own `excess_kurtosis` (imported, not copied) on gaussian / laplace / uniform draws with known answers, at the real readout shape. |
| `plot_combined.py`, `plot_per_model.py` | All-model and per-model kurtosis profiles from the stored JSONs. No model is loaded. |
| `plot_diff_position.py` | Lens-specific profile (J-Lens minus control, median per layer, 95% CI by bootstrap over prompts), the same profile by token-position bucket, depth x position heatmaps, and the within-prompt position control written to `position_within_prompt[<tag>].txt`. |
| `make_report.py`, `analysis_report.typ`, `analysis_report.pdf` | Per-model reading report (overview, then family by family) compiled with `typst`. |
| `results/` | One JSON per (model, arm, prompt set): `kurtosis_<short>.json` (main sweep), `kurtosis_<short>_long.json` (long prompts), `kurtosis_random_<short>.json` (control task). Each carries its `meta` (hf_id, dtype, device, prompt file, sequence lengths, per-prompt timings). |
| `position_within_prompt.txt`, `position_within_prompt_long.txt` | Output of the within-prompt position control, main and long sets. |
| `figures/` | Figures, each regenerated by a command listed below. |

## Reproducing

Everything runs against a TransformerLens checkout that carries the `JacobianLens` tool
(`transformer_lens.tools.analysis.jacobian_lens`, v3 bridge API), from inside its environment.
As run: fork [`rbellec/TransformerLens`](https://github.com/rbellec/TransformerLens), branch
`dev` at `c03d5103`; `torch 2.10.0`, `transformers 5.13.0`, `device=mps` (Apple M1 Max, 64 GB),
models booted via `TransformerBridge.boot_transformers`. Gated Hugging Face models need a token
in the environment.

```bash
cd /path/to/TransformerLens
set -a; source .env; set +a          # HF token
P=/path/to/experiments/jacobian-lens-scale

# 0. plumbing
.venv/bin/python $P/smoke_test.py
.venv/bin/python $P/scout_feasibility.py gemma-3-270m google/gemma-3-270m
.venv/bin/python $P/check_estimator.py

# 1. main sweep — prompts.txt, one call per model: <short> <hf_id> <dtype>
#    <short> must be a key of the lens registry; float32 for the 270M models, bfloat16 above.
.venv/bin/python $P/sweep_kurtosis.py gemma-3-270m    google/gemma-3-270m     float32
.venv/bin/python $P/sweep_kurtosis.py gemma-3-270m-it google/gemma-3-270m-it  float32
.venv/bin/python $P/sweep_kurtosis.py gemma-3-1b      google/gemma-3-1b-pt    bfloat16
.venv/bin/python $P/sweep_kurtosis.py gemma-3-1b-it   google/gemma-3-1b-it    bfloat16
.venv/bin/python $P/sweep_kurtosis.py gemma-3-4b      google/gemma-3-4b-pt    bfloat16
.venv/bin/python $P/sweep_kurtosis.py gemma-3-4b-it   google/gemma-3-4b-it    bfloat16
.venv/bin/python $P/sweep_kurtosis.py gemma-3-12b     google/gemma-3-12b-pt   bfloat16
.venv/bin/python $P/sweep_kurtosis.py gemma-3-12b-it  google/gemma-3-12b-it   bfloat16
.venv/bin/python $P/sweep_kurtosis.py qwen3.5-0.8b    Qwen/Qwen3.5-0.8B       bfloat16
.venv/bin/python $P/sweep_kurtosis.py qwen3.5-2b-pt   Qwen/Qwen3.5-2B-Base    bfloat16
.venv/bin/python $P/sweep_kurtosis.py qwen3.5-4b      Qwen/Qwen3.5-4B         bfloat16
.venv/bin/python $P/sweep_kurtosis.py qwen3.5-9b-pt   Qwen/Qwen3.5-9B-Base    bfloat16

# 2. control-task arm (random orthogonal J), Gemma base models
.venv/bin/python $P/sweep_control_task.py gemma-3-270m google/gemma-3-270m   float32
.venv/bin/python $P/sweep_control_task.py gemma-3-1b   google/gemma-3-1b-pt  bfloat16
.venv/bin/python $P/sweep_control_task.py gemma-3-4b   google/gemma-3-4b-pt  bfloat16
.venv/bin/python $P/sweep_control_task.py gemma-3-12b  google/gemma-3-12b-pt bfloat16

# 3. long-prompt run (position / context-length axis): prompt file + output suffix
.venv/bin/python $P/sweep_kurtosis.py gemma-3-270m  google/gemma-3-270m   float32  prompts_long.txt _long
.venv/bin/python $P/sweep_kurtosis.py gemma-3-4b    google/gemma-3-4b-pt  bfloat16 prompts_long.txt _long
.venv/bin/python $P/sweep_kurtosis.py gemma-3-12b   google/gemma-3-12b-pt bfloat16 prompts_long.txt _long
.venv/bin/python $P/sweep_kurtosis.py qwen3.5-9b-pt Qwen/Qwen3.5-9B-Base  bfloat16 prompts_long.txt _long

# 4. figures and tables from the stored JSONs (no model loaded)
.venv/bin/python $P/plot_combined.py
.venv/bin/python $P/plot_per_model.py
.venv/bin/python $P/plot_diff_position.py          # main set  -> *_all_models.png, position_within_prompt.txt
.venv/bin/python $P/plot_diff_position.py _long    # long set  -> *_long.png, position_within_prompt_long.txt
.venv/bin/python $P/make_report.py                 # analysis_report.typ + .pdf (needs typst)
```

Memory note for the long set: the readout keeps the full-vocabulary logits of every layer in
float32 on the device (for gemma-3-12b at ~150 tokens, about 7-8 GB per arm), so the prompt
*length*, not the prompt count, is what bounds a run on this machine.

### Long-prompt run as executed (2026-09-03, 21:54-22:27, sequential, single machine)

| Model | hf_id | dtype | boot + lens | 20 prompts | per prompt | total |
|---|---|---|---|---|---|---|
| gemma-3-270m | google/gemma-3-270m | float32 | 3 s | 122 s | 5.0-8.5 s | 131 s |
| gemma-3-4b | google/gemma-3-4b-pt | bfloat16 | 12 s | 349 s | 12.9-22.1 s | 368 s |
| gemma-3-12b | google/gemma-3-12b-pt | bfloat16 | 24 s | 1004 s | 25.3-67.7 s | 1037 s |
| qwen3.5-9b-pt | Qwen/Qwen3.5-9B-Base | bfloat16 | 18 s | 401 s | 15.7-31.8 s | 427 s |

Sequence lengths 110-158 tokens (Gemma tokenizer), 113-154 (Qwen). No NaN in any cell (the
script raises on NaN; also re-checked on the written JSONs). The only warnings were the
`Hook alias ... on SiglipVisionEncoderLayerBridge ... did not resolve` messages at boot of the
multimodal Gemma 4B/12B checkpoints (vision-encoder hooks, unused here). `plot_diff_position.py
_long` took 16 s. Per-prompt timings are in each JSON under `meta.timings_s`.

⚠️ **Do not enable TransformerLens compatibility mode here.** The published lenses are fitted
on raw HF activations; `fold_ln` / `center_unembed` change the basis of the residual stream, so
compatibility mode and these lenses are mutually exclusive. The library refuses the call rather
than returning wrong numbers, which is the correct behaviour.

Fitted lenses come from the published registries and are not committed here (see
`.gitignore`).

## An observation, not a result

`figures/smoke_kurtosis_pythia70m_deduped.png` is the smoke test's output: one prompt, one
position, six layers, on `pythia-70m-deduped`. The two arms produce visibly different curves,
which is all the smoke test was meant to show — that the control discipline works and both
paths run.

It is **not** evidence about the workspace band. n=1 prompt at a single position cannot resolve
"a third of the way through the network", and the shape it happens to show is the inverse of
the one described in the paper. It is included because it is what the committed code produces,
not because it means anything yet.

## Related public work

- [`eliebak/open-jlens-data`](https://github.com/eliebak/open-jlens-data) and the
  [J-space open-models visualisation](https://eliebak.com/viz/jspace-open-v2) — layer-by-layer
  CKA of Jacobian-lens geometry within and across 38 open models, plus participation ratio,
  temporal reach and training-checkpoint sweeps. Finds the block structure to be strikingly
  universal at matched relative depth. Measures the geometry; does not measure the readout
  statistic studied here.
- [`xiangchensong/jacobian-lens-open-frontier`](https://github.com/xiangchensong/jacobian-lens-open-frontier)
  — replication on open frontier-scale models, at the other end of the size axis.
- [`tao-hpu/jspace-replication`](https://github.com/tao-hpu/jspace-replication) — claim-by-claim
  audit on small open models, with negative results reported.

## Status

- [x] Chain validated end to end (`smoke_test.py`)
- [x] Feasibility scouted across the candidate model zoo (`scout_results.jsonl`)
- [x] Main sweep, 12 models, J-Lens + logit-lens control (`results/kurtosis_<short>.json`)
- [x] Control-task arm, random orthogonal J, 4 Gemma base models (`results/kurtosis_random_<short>.json`)
- [x] Estimator sanity check (`check_estimator.py`)
- [x] Long-prompt run, 4 models, position / context-length axis (`results/kurtosis_<short>_long.json`)
- [x] Figures and per-model reading report (`figures/`, `analysis_report.pdf`)
- [ ] Write-up link — *to be added*
