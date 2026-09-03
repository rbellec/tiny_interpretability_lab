"""Sanity check of the sweep's kurtosis estimator (hard rule: a check performed by
hand, to re-run yourself before quoting it in the write-up).

The sweep's own `excess_kurtosis` (imported, not copied) is fed data whose answer
is known exactly:
  gaussian -> 0     (the very definition of excess kurtosis: the "minus 3")
  laplace  -> 3
  uniform  -> -1.2  (also checks the sign on the "flat" side)
Same shape as the real gemma readout (8 positions x 262 144 tokens), float32 in:
this exercises the same last-axis reduction and the same float64 cast.
Expected tolerance (sampling noise over 262 144 draws): ~±0.02 for the gaussian,
~±0.1 for the laplace (heavy tails).

    cd /path/to/TransformerLens
    .venv/bin/python /path/to/check_estimator.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sweep_kurtosis import excess_kurtosis  # noqa: E402  -- the sweep's own function

EXPECTED = {"gaussian": 0.0, "laplace": 3.0, "uniform": -1.2}
TOL = {"gaussian": 0.05, "laplace": 0.2, "uniform": 0.05}


def main(n_pos: int = 8, d_vocab: int = 262144, seed: int = 0) -> bool:
    torch.manual_seed(seed)
    samples = {
        "gaussian": torch.randn(n_pos, d_vocab),
        "laplace": torch.distributions.Laplace(0.0, 1.0).sample((n_pos, d_vocab)),
        "uniform": torch.rand(n_pos, d_vocab),
    }
    ok = True
    print(f"excess_kurtosis check — tensor {n_pos} x {d_vocab}, float32 in, seed {seed}")
    for name, x in samples.items():
        g2 = excess_kurtosis(x.float())
        mean, lo, hi = g2.mean().item(), g2.min().item(), g2.max().item()
        good = abs(mean - EXPECTED[name]) <= TOL[name]
        ok &= good
        print(f"  {name:9s} expected {EXPECTED[name]:+.2f} | got {mean:+.3f} "
              f"(rows {lo:+.3f} .. {hi:+.3f}) {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
