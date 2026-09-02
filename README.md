# Tiny Interpretability Lab

*Small, legible experiments on how transformers work — and on whether the tools we use to look at them measure what they claim to measure.*

## The question underneath

Interpretability methods produce measurements. The question I keep coming back to is how we know that a given measurement measures what it claims to — and I work on it through different doors:

- **By generalization.** A signature that does not survive from one model to the next is a property of the measurement, not of the model.
- **By ground truth.** A language whose grammar is small, regular and fully known gives a referent against which a method's findings can be checked.

Each experiment lives in its own directory, is self-contained, and is meant to be readable on its own.

## Experiments

| Directory | Question | Status |
|---|---|---|
| [`experiments/jacobian-lens-scale/`](experiments/jacobian-lens-scale/) | The Jacobian lens has a published quantitative signature over depth. Does it survive across model scale and across model families — or is it a property of the summary statistic? | Active |
| [`experiments/toki-pona/`](experiments/toki-pona/) | A small language model trained on Toki Pona (~130 words, a handful of grammatical particles) as a controlled laboratory: syntactic roles as ground truth for probing and circuit work. | Design stage — see its README for scope and roadmap |

## Conventions

- Scripts are committed **as they were run**, not rewritten into a library. Where a number in a write-up comes from a script here, the command that produces it is given in that experiment's README.
- Every lens/measurement comparison carries a control arm computed in the same code path, and is read against a per-model baseline rather than against zero.
- Model weights, fitted lenses and cached activations are not in git. Each experiment's README says where they come from.

Started July 2026.
