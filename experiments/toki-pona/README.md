# Toki Pona as a controlled laboratory

*Understanding transformers through small, clean experiments — using a minimal constructed
language, Toki Pona, as a controlled laboratory.*

> 📝 **Why this experiment exists**, in plain terms:
> *[LLM, Toki Pona and interpretability](https://rbellec.github.io/blog/en/posts/llm-toki-pona-and-interpretability/)*
> ([version française](https://rbellec.github.io/blog/posts/llm-toki-pona-and-interpretability/)).
> Start there for the reasoning; this page is the technical scope.

> **Status: substrate built, interpretability work starting.** A first model is trained and
> loads into the standard tooling. It has not been reviewed in detail yet, so no numbers are
> quoted here — they arrive with the first write-up.

## The idea

Toki Pona is a constructed language with ~130 words and a small, regular grammar: a handful of
particles carry most of the syntactic structure — `li` marks the predicate, `e` marks the
object, `pi` regroups modifiers. That minimality is the point.

A language model trained on such a language is small enough to study end to end, yet its
grammar still has real structure — clear syntactic roles, compositional phrases — to host
genuine questions about how a transformer represents and manipulates them. The bet is simple:
a deliberately small, *knowable* language gives real analytic traction. When the ground-truth
structure is explicit, circuits are more traceable, ablations are cleaner, and results are
legible. You can ask "how does the model track the object marked by `e`?" and hope to actually
answer it.

Three properties do the work:

- **A small language, not a small corpus.** Tiny Shakespeare and TinyStories reduce the corpus;
  BabyLM reduces the data budget. All three stay inside a language whose rules come from a far
  larger corpus than the one the model sees. Here the language itself is small, so a modest
  corpus can be representative of it.
- **Ambiguity is the object of study, not noise.** The language delegates disambiguation to
  context — `tawa` is a preposition or a modifier depending on position. That forces contextual
  work that can be observed.
- **A vocabulary small enough to enumerate.** With word-level tokenization, the embedding table
  stops dominating the parameter budget, and vocabulary-sized matrices stop being objects you
  can only sample from.

## What this is (and isn't)

- **Is:** a lab notebook — a small Toki Pona language model, and minimal, well-scoped
  interpretability experiments on it, one at a time.
- **Isn't:** a from-scratch-GPT tutorial. The engineering (a small transformer, a training
  loop) is a means; the interpretability result that the language's minimality makes possible
  is the point. The architecture is a reference implementation, deliberately not rewritten — on
  an interpretability project, an implementation bug turns every "circuit" into an artefact.
- Tooling is the standard mechanistic-interpretability stack (TransformerLens).

## Where this is going

Not a sequence — several interests, pursued as the substrate allows. The wider aim is
close to encyclopedic: to work out, method by method, **what interpretability can and cannot do
on a tiny-language model**, and to say so either way.

- **Particle circuits.** How the model tracks a single syntactic role (`li`, `e`), and the
  `tawa` preposition/modifier disambiguation — attention patterns against an explicit grammar
  rather than against intuition.
- **Deliberately "obsolete" configurations, kept on purpose.** The relevant axis is not recent
  vs. old but *suited to the question*. Attention-only models with no MLP, and positional
  schemes that keep position out of the OV circuit, are what make the vocabulary×vocabulary
  matrices of *A Mathematical Framework for Transformer Circuits* readable **in full** rather
  than by sampling — roughly four orders of magnitude smaller here than on a standard
  vocabulary. Induction heads are the natural first target.
- **A paired family rather than a single model** — varying one axis at a time (depth,
  attention-only vs. with MLP, positional scheme, several seeds), with training checkpoints
  published, not only final weights. Scoped as a by-product of experiments run anyway.
- **Probing across layers.** Layer-by-layer semantic probing on a language whose semantic space
  is small enough to be mapped exhaustively.
- **Filler tokens on a minimal language** — strictly downstream of existing work on hidden
  computation in filler tokens. Not a claim to that ground.

**North star** (gated on reading the paper first): reimplementing *Explaining Attention with
Program Synthesis* on the Toki Pona model — attention explained in the language of programs.

## Caveats worth stating up front

- **The universality bet is unproven.** Whether mechanisms found in a model trained on a tiny
  language transfer to models trained on far larger ones is exactly the open question. Chess in
  AI is the cautionary precedent: the model organism can stop generalising and become the
  object of study.
- **A rotary positional scheme puts position into the QK circuits in a way the *Mathematical
  Framework* does not analyse.** Results from that paper do not transfer mechanically to the
  showcase model — hence the separate, deliberately plainer configurations above.
- **Small-language results can be ambiguous in the unhelpful direction**: "absent at this scale"
  and "not real" look alike. Controls have to carry that weight.

Started July 2026.
