"""Smoke test: validate the whole chain before spending measurement time on it.

Each stage is independent and reports PASS/FAIL: the point is to learn *where*
it breaks, not to stop at the first error.

    cd /path/to/TransformerLens
    .venv/bin/python /path/to/smoke_test.py
"""
# %%
import json
import time
import traceback
from pathlib import Path

import torch

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
MODEL = "EleutherAI/pythia-70m-deduped"
SHORT = "pythia-70m-deduped"
PROMPT = "The capital of France is Paris. The capital of Italy is"

results = {}


def stage(name):
    def deco(fn):
        def wrapped(*a, **kw):
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                results[name] = ("PASS", time.time() - t0, "")
                return out
            except Exception as e:
                results[name] = ("FAIL", time.time() - t0, f"{type(e).__name__}: {e}")
                traceback.print_exc()
                return None
        return wrapped
    return deco


# %% 1 - device
@stage("1. device")
def s1():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   torch {torch.__version__} | device={dev}")
    return dev


# %% 2 - bridge boot
@stage("2. boot TransformerBridge")
def s2(dev):
    from transformer_lens.model_bridge import TransformerBridge
    m = TransformerBridge.boot_transformers(MODEL, device=dev)
    print(f"   n_layers={m.cfg.n_layers} d_model={m.cfg.d_model} d_vocab={m.cfg.d_vocab}")
    return m


# %% 3 - hooks + cache (do activations actually come out?)
@stage("3. hooks / run_with_cache")
def s3(m):
    logits, cache = m.run_with_cache(PROMPT)
    keys = [k for k in cache.keys() if "resid_post" in k]
    print(f"   logits {tuple(logits.shape)} | {len(cache)} entrees | resid_post x{len(keys)}")
    print(f"   ex: {keys[0]} -> {tuple(cache[keys[0]].shape)}")
    return logits, cache, keys


# %% 4 - causal intervention (does the hook really bite?)
@stage("4. intervention causale")
def s4(m, base_logits):
    layer = m.cfg.n_layers // 2
    name = f"blocks.{layer}.hook_resid_post"

    def zero_it(act, hook):
        return torch.zeros_like(act)

    with torch.no_grad():
        abl = m.run_with_hooks(PROMPT, fwd_hooks=[(name, zero_it)])
    delta = (abl[0, -1] - base_logits[0, -1]).abs().max().item()
    print(f"   ablation {name} -> |delta logit max| = {delta:.3f}")
    assert delta > 1e-3, "le hook ne mord pas : delta nul"
    return delta


# %% 5 - JacobianLens: readout + logit-lens control arm
@stage("5. JacobianLens readout + controle")
def s5(m):
    from transformer_lens.tools.analysis.jacobian_lens import JacobianLens
    reg = json.load(open("transformer_lens/tools/analysis/jacobian_lens_registry.json"))
    assert SHORT in reg, f"{SHORT} absent du registry ({len(reg)-1} entrees)"
    lens = JacobianLens.from_pretrained(SHORT)
    print(f"   lens chargee | source_layers={list(lens.source_layers)[:8]}...")

    out = {}
    for use_j in (True, False):
        r = lens.readout(m, PROMPT, use_jacobian=use_j, return_full_logits=True)
        out[use_j] = r
        print(f"   readout(use_jacobian={use_j}) -> {len(r.lens_logits)} layers")
    return lens, out


# %% 6 - excess kurtosis per layer, both arms, one figure
@stage("6. kurtosis + figure")
def s6(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def excess_kurtosis(x):
        # biased g2 estimator, fixed once and kept across every arm (see README)
        x = x.double()
        mu = x.mean(-1, keepdim=True)
        d = x - mu
        m2 = (d ** 2).mean(-1)
        m4 = (d ** 4).mean(-1)
        return (m4 / m2.pow(2) - 3.0)

    curves = {}
    for use_j, r in out.items():
        layers = sorted(r.lens_logits.keys())
        vals = [excess_kurtosis(r.lens_logits[l][-1]).item() for l in layers]
        curves[use_j] = (layers, vals)
        tag = "J-Lens" if use_j else "logit-lens (controle)"
        print(f"   {tag:26s} min={min(vals):8.2f} max={max(vals):8.2f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(*curves[True], "o-", label="J-Lens")
    ax.plot(*curves[False], "s--", label="logit-lens (bras de controle)")
    ax.set_xlabel("layer"); ax.set_ylabel("excess kurtosis (g2)")
    ax.set_title(f"{SHORT} — profil de kurtosis, deux bras")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    p = FIG / "smoke_kurtosis_pythia70m_deduped.png"
    fig.savefig(p, dpi=130)
    print(f"   figure -> {p}")
    return p


# %%
if __name__ == "__main__":
    print("=" * 62)
    dev = s1()
    m = s2(dev)
    r3 = s3(m) if m is not None else None
    if m is not None and r3 is not None:
        s4(m, r3[0])
        r5 = s5(m)
        if r5 is not None:
            s6(r5[1])
    print("=" * 62)
    for k, (st, dt, err) in results.items():
        mark = "OK  " if st == "PASS" else "FAIL"
        print(f"[{mark}] {k:34s} {dt:6.1f}s {err}")
    print("=" * 62)
