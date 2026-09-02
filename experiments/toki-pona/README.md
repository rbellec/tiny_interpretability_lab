# Toki Pona as a controlled laboratory

*Understanding transformers through small, clean experiments — using a minimal constructed language, Toki Pona, as a controlled laboratory.*

> **Status: design stage.** This is the scope and roadmap of the experiment; the training and
> probing code lives in separate working repositories and is folded in here as results arrive.

## The idea

Toki Pona is a constructed language with ~130 words and a small, regular grammar: a handful of particles carry most of the syntactic structure — `li` marks the predicate, `e` marks the object, `pi` regroups modifiers. That minimality is the point.

A language model trained on such a language is small enough to study end to end, yet its grammar still has real structure — clear syntactic roles, compositional phrases — to host genuine questions about how a transformer represents and manipulates them. The bet is simple: a deliberately small, *knowable* language gives real analytic traction. When the ground-truth structure is explicit, circuits are more traceable, ablations are cleaner, and results are legible. You can ask "how does the model track the object marked by `e`?" and hope to actually answer it.

## What this repo is (and isn't)

- **Is:** a lab notebook — a place to build a small Toki Pona language model and run minimal, well-scoped interpretability experiments on it, one at a time.
- **Isn't:** a from-scratch-GPT tutorial. The engineering (a small transformer, a training loop) is a means; the goal is the interpretability *result* that the language's minimality makes possible.
- Tooling builds on the standard mechanistic-interpretability stack (TransformerLens: logit lens, activation patching).

## Roadmap (early, deliberately modest)

0. **Substrate** — a small, GPT-2-scale Toki Pona transformer that trains and generates (from scratch, JAX/Equinox).
1. **A particle's circuit** — how the model tracks a single syntactic role (e.g. `li` / `e`): the attention pattern or small structure that implements it.
2. **From-scratch vs. finetuned** — compare representations between a model trained only on Toki Pona and an open model finetuned on it (only if a clean comparison emerges).
3. **Filler tokens on a minimal language** — whether a small model trained on a minimal language benefits from filler / scratchpad tokens on multi-hop tasks, and where. The class of problems where this helps is characterised by the quantifier depth of a first-order formula — which connects this directly to formal-language theory. (After *Let's Think Dot by Dot*, Pfau, Merrill & Bowman, 2024.)

**North star** (gated on reading the paper first): reimplementing *Explaining Attention with Program Synthesis* on the Toki Pona model — attention explained in the language of programs.

## Status

Early — bootstrapping (started July 2026). This README is the design and scope of the project; experiments and write-ups will follow, one small and legible result at a time.
