"""Third arm — control task in the spirit of Hewitt & Liang (2019).

Replaces every J[l] with a RANDOM orthogonal rotation (QR of a Gaussian,
fixed seed) and reruns exactly the same readout through the same code path
(use_jacobian=True on a rebuilt JacobianLens).

Rationale: if the mid-depth kurtosis bump requires the LEARNED structure of
J (rather than "any matrix in front of the unembedding"), it must VANISH
under a random rotation. An isometry preserves norms, and the readout's
norm() step neutralizes scale anyway — so this arm isolates structure, not
calibration.

Same g2 estimator, same prompts, same aggregation as sweep_kurtosis.py
(direct imports — hard rule: ONE estimator).

    cd /path/to/TransformerLens
    .venv/bin/python /path/to/sweep_control_task.py gemma-3-270m google/gemma-3-270m float32

Output: results/kurtosis_random_<short>.json
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from sweep_kurtosis import excess_kurtosis, load_prompts  # noqa: E402  (single estimator)

OUT = HERE / "results"
SEED = 0


def random_orthogonal_lens(lens):
    from transformer_lens.tools.analysis.jacobian_lens import JacobianLens
    gen = torch.Generator().manual_seed(SEED)
    rand = {}
    for layer in lens.source_layers:
        g = torch.randn(lens.d_model, lens.d_model, generator=gen)
        q, r = torch.linalg.qr(g)
        q = q * torch.sign(torch.diagonal(r))  # sign correction (Haar)
        rand[layer] = q
    return JacobianLens(
        rand, n_prompts=lens.n_prompts, d_model=lens.d_model,
        metadata={"control_task": f"random orthogonal Q per layer, seed {SEED}",
                  "replaces": "registry lens", "rule": "Hewitt & Liang 2019"})


def main(short: str, hf_id: str, dtype_name: str = "bfloat16"):
    from transformer_lens.model_bridge import TransformerBridge
    from transformer_lens.tools.analysis.jacobian_lens import JacobianLens

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    prompts = load_prompts()
    print(f"== control-task {short} | {len(prompts)} prompts | seed {SEED}")

    model = TransformerBridge.boot_transformers(hf_id, device=dev,
                                                dtype=getattr(torch, dtype_name))
    lens = random_orthogonal_lens(JacobianLens.from_pretrained(short))

    rec = {}
    for pi, prompt in enumerate(prompts):
        t0 = time.time()
        r = lens.readout(model, prompt, use_jacobian=True, return_full_logits=True)
        for layer, logits in r.lens_logits.items():
            k = excess_kurtosis(logits)
            if torch.isnan(k).any():
                raise ValueError(f"NaN kurtosis {short} L{layer} prompt {pi}")
            rec.setdefault(str(layer), []).extend([round(v, 4) for v in k.tolist()])
        del r
        print(f"   prompt {pi+1:2d}/{len(prompts)} ({time.time()-t0:5.1f}s)")

    out = OUT / f"kurtosis_random_{short}.json"
    json.dump({"meta": {"short": short, "hf_id": hf_id, "dtype": dtype_name,
                        "arm": "random-orthogonal control task", "seed": SEED,
                        "n_layers": model.cfg.n_layers,
                        "estimator": "g2 biased (m4/m2^2 - 3), float64 (imported)"},
               "kurtosis": rec}, open(out, "w"))
    print(f"   -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
