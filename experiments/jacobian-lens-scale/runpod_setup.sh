#!/usr/bin/env bash
# ============================================================================
# Provisioning for a rented GPU pod (runpod), for the model sizes that do not
# fit locally. READ BEFORE RUNNING.
#
# This script creates no account and bills nothing by itself: billing starts
# when the pod is deployed and stops on Stop/Terminate. The script runs
# afterwards, over SSH, inside the pod.
#
# Full path:
#   1. Create the runpod account, add credits.
#   2. Deploy > GPU Pod > A100 80GB (Secure Cloud, ~$1.4-1.6/h).
#      Template: "RunPod PyTorch 2.x" (official image, recent CUDA).
#      Container disk: 150 GB (27b weights ~54 GB + venv + headroom).
#      NO persistent network volume (it bills while the pod is off).
#   3. Copy this script to the pod and run it:
#        scp -P <port> runpod_setup.sh root@<ip>:/workspace/
#        ssh -p <port> root@<ip>
#        cd /workspace && HF_TOKEN=hf_xxx bash runpod_setup.sh
#   4. Run the extraction in tmux, then pull the results back:
#        rsync -avz -e "ssh -p <port>" root@<ip>:/workspace/p6-results/ ./results-27b/
#   5. TERMINATE the pod (not just Stop) - nothing should be left running.
#
# Expected cost: 27b extraction ~1-3 h -> $5-15 all in.
# ============================================================================
set -euo pipefail

: "${HF_TOKEN:?Pass HF_TOKEN=hf_xxx as an environment variable}"

cd /workspace

# --- base tooling ------------------------------------------------------------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux rsync)

# --- TransformerLens (upstream/dev, same branch as the local clone) ----------
if [ ! -d TransformerLens ]; then
  git clone --depth 1 --branch dev https://github.com/TransformerLensOrg/TransformerLens.git
fi
cd TransformerLens
uv sync

# --- HF token (gemma-3 is gated: account access must already be granted) -----
export HF_TOKEN
echo "HF_TOKEN=${HF_TOKEN}" > .env

# --- smoke: the whole chain on the small model before paying for the big one -
mkdir -p /workspace/p6-results
.venv/bin/python - <<'EOF'
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.tools.analysis.jacobian_lens import JacobianLens
m = TransformerBridge.boot_transformers("google/gemma-3-270m", device="cuda")
lens = JacobianLens.from_pretrained("gemma-3-270m")
r = lens.readout(m, "The capital of France is Paris. The capital of Italy is",
                 use_jacobian=True, positions=[-1], return_full_logits=True)
print("readout OK,", len(r.lens_logits), "layers - pod is ready for the 27b.")
EOF

echo "=========================================================="
echo "Pod ready. Next: run the 27b extraction in tmux,"
echo "pull back /workspace/p6-results, then TERMINATE the pod."
echo "=========================================================="
