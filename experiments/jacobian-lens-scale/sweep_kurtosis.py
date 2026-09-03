"""Kurtosis sweep — layerwise excess-kurtosis profile of the lens readout, across scales.

One run = one model, two arms (J-Lens / logit-lens control), P prompts.
For every (layer, position) cell: excess kurtosis g2 of the readout logit
distribution over the vocabulary. Only scalars are stored (never the full
logits — d_vocab ~260k); one JSON per model.

Methodological rules enforced here:
  1. both arms go through the SAME code path (use_jacobian=True/False);
  2. single estimator: biased g2 (m4/m2^2 - 3), identical to smoke_test.py;
  3. depth normalized to [0-100] so models are comparable.

    cd /path/to/TransformerLens
    .venv/bin/python /path/to/sweep_kurtosis.py gemma-3-270m google/gemma-3-270m float32
    .venv/bin/python /path/to/sweep_kurtosis.py gemma-3-1b  google/gemma-3-1b-pt bfloat16
    # long prompts (position / context-length axis): prompt file + output suffix
    .venv/bin/python /path/to/sweep_kurtosis.py gemma-3-12b google/gemma-3-12b-pt bfloat16 prompts_long.txt _long
    ...

Output: results/kurtosis_<short>.json + one figure per model.
Combined figure (all models): plot_combined.py
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
OUT = HERE / "results"
FIG = HERE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


def load_prompts(prompt_file="prompts.txt"):
    lines = (HERE / prompt_file).read_text().splitlines()
    # convention: one prompt per non-empty, non-comment line, EXCEPT indented
    # lines which continue the previous prompt (the python code block).
    prompts = []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        if ln.startswith("    "):          # continuation (indented code block)
            prompts[-1] = prompts[-1] + "\n" + ln
        else:
            prompts.append(ln)
    return prompts


def excess_kurtosis(x):
    # biased g2 estimator, fixed once and for all (cf. P6 methodology note).
    # Identical to smoke_test.py — do not "improve" one without the other.
    x = x.double()
    mu = x.mean(-1, keepdim=True)
    d = x - mu
    m2 = (d ** 2).mean(-1)
    m4 = (d ** 4).mean(-1)
    return m4 / m2.pow(2) - 3.0


def main(short: str, hf_id: str, dtype_name: str = "bfloat16",
         prompt_file: str = "prompts.txt", tag: str = ""):
    """tag: suffix of the output files (e.g. '_long' for prompts_long.txt)."""
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis.jacobian_lens import JacobianLens

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, dtype_name)
    prompts = load_prompts(prompt_file)
    print(f"== {short} | {len(prompts)} prompts | {dev}/{dtype_name}")

    t0 = time.time()
    model = TransformerBridge.boot_transformers(hf_id, device=dev, dtype=dtype)
    lens = JacobianLens.from_pretrained(short)
    print(f"   boot+lens {time.time()-t0:.0f}s | n_layers={model.cfg.n_layers} "
          f"d_model={model.cfg.d_model} d_vocab={model.cfg.d_vocab}")

    # rec[arm][layer] = list of g2 values, one per (prompt, position) — scalars only
    rec = {"jlens": {}, "control": {}}
    meta = {"short": short, "hf_id": hf_id, "dtype": dtype_name,
            "n_layers": model.cfg.n_layers, "d_model": model.cfg.d_model,
            "d_vocab": model.cfg.d_vocab, "device": dev,
            "estimator": "g2 biased (m4/m2^2 - 3), float64",
            "n_prompts": len(prompts), "positions": "all valid readout positions",
            "prompt_file": prompt_file, "seq_lens": [], "timings_s": {}}

    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        for use_j, arm in ((True, "jlens"), (False, "control")):
            r = lens.readout(model, prompt, use_jacobian=use_j,
                             return_full_logits=True)
            for layer, logits in r.lens_logits.items():
                # logits: [positions, d_vocab] -> g2 per position
                k = excess_kurtosis(logits)
                if torch.isnan(k).any():
                    raise ValueError(f"NaN kurtosis {short} L{layer} prompt {pi}")
                rec[arm].setdefault(str(layer), []).extend(
                    [round(v, 4) for v in k.tolist()])
            del r
        meta["seq_lens"].append(len(model.tokenizer(prompt)["input_ids"]))
        dt = time.time() - t0
        meta["timings_s"][f"prompt_{pi}"] = round(dt, 1)
        print(f"   prompt {pi+1:2d}/{len(prompts)} ({dt:5.1f}s) : {prompt[:50]!r}")

    out = OUT / f"kurtosis_{short}{tag}.json"
    json.dump({"meta": meta, "kurtosis": rec}, open(out, "w"))
    print(f"   -> {out}")

    # per-model figure: median and IQR per layer, normalized depth
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statistics as st

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n_final = model.cfg.n_layers - 1
    for arm, style, label in (("jlens", "o-", "J-Lens"),
                              ("control", "s--", "logit lens (control)")):
        layers = sorted(int(l) for l in rec[arm])
        depth = [100 * l / n_final for l in layers]
        med = [st.median(rec[arm][str(l)]) for l in layers]
        q1 = [st.quantiles(rec[arm][str(l)], n=4)[0] for l in layers]
        q3 = [st.quantiles(rec[arm][str(l)], n=4)[2] for l in layers]
        ax.plot(depth, med, style, label=label, ms=3)
        ax.fill_between(depth, q1, q3, alpha=.15)
    ax.set_xlabel("relative depth (%)")
    ax.set_ylabel("excess kurtosis (g2), median ± IQR")
    ax.set_title(f"{short} — {len(prompts)} prompts, all valid positions")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    p = FIG / f"sweep_kurtosis_{short}{tag}.png"
    fig.savefig(p, dpi=130)
    print(f"   figure -> {p}")


if __name__ == "__main__":
    main(*sys.argv[1:])
