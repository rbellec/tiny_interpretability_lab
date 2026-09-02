"""Scout: feasibility ONLY, no science.

For each requested model: HF access (gated?), boot time, readout time (both
arms), process memory. No curves, no kurtosis statistics - the point is to know
whether the sweep fits on an M1 Max before any measurement is made.

    cd /path/to/TransformerLens
    set -a; source .env; set +a
    .venv/bin/python /path/to/scout_feasibility.py gemma-3-270m google/gemma-3-270m

Writes one JSON line per run to scout_results.jsonl
"""
import json
import resource
import sys
import time
from pathlib import Path

import torch

OUT = Path(__file__).parent / "scout_results.jsonl"
PROMPT = "The capital of France is Paris. The capital of Italy is"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main(short: str, hf_id: str, dtype_name: str = "float32"):
    rec = {"short": short, "hf_id": hf_id, "dtype": dtype_name, "torch": torch.__version__}
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    rec["device"] = dev
    dtype = getattr(torch, dtype_name)
    try:
        from transformer_lens.model_bridge import TransformerBridge
        from transformer_lens.tools.analysis.jacobian_lens import JacobianLens

        t0 = time.time()
        m = TransformerBridge.boot_transformers(hf_id, device=dev, dtype=dtype)
        rec["boot_s"] = round(time.time() - t0, 1)
        rec["n_layers"] = m.cfg.n_layers
        rec["d_model"] = m.cfg.d_model

        t0 = time.time()
        lens = JacobianLens.from_pretrained(short)
        rec["lens_load_s"] = round(time.time() - t0, 1)
        rec["source_layers"] = len(list(lens.source_layers))

        for use_j in (True, False):
            t0 = time.time()
            r = lens.readout(m, PROMPT, use_jacobian=use_j,
                             positions=[-1], return_full_logits=True)
            rec[f"readout_{'jlens' if use_j else 'control'}_s"] = round(time.time() - t0, 1)
            rec["n_readout_layers"] = len(r.lens_logits)

        rec["peak_rss_gb"] = round(rss_gb(), 1)
        rec["status"] = "OK"
    except Exception as e:
        rec["status"] = "FAIL"
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
        rec["peak_rss_gb"] = round(rss_gb(), 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
