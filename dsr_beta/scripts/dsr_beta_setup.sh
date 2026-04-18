#!/usr/bin/env bash
# DSR_beta full setup from scratch
# Pulls image, creates container, applies local mods
# Production container danish_atom_main (port 8888) is NOT touched.

set -euo pipefail

IMAGE_DIGEST="rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f"
CONTAINER_NAME="danish_atom_dsr_beta"
PORT=8890

echo "=== 1. Pull pinned image ==="
~/bin/docker pull "$IMAGE_DIGEST"

echo "=== 2. Remove any previous DSR_beta container ==="
~/bin/docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "=== 3. Create DSR_beta container ==="
~/bin/docker run -d --name "$CONTAINER_NAME" \
  --device /dev/kfd \
  --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --device /dev/dri/renderD144 --device /dev/dri/renderD152 \
  --device /dev/dri/renderD160 --device /dev/dri/renderD168 \
  --device /dev/dri/renderD176 --device /dev/dri/renderD184 \
  --group-add video \
  --shm-size=32g \
  -p "$PORT:$PORT" \
  -v /projects:/projects \
  -v /home/hackathon:/home/hackathon \
  "$IMAGE_DIGEST" \
  sleep infinity

echo "=== 4. Apply local patches ==="
~/bin/docker exec "$CONTAINER_NAME" bash -c '
set -e
cd /app/ATOM

# Relaxed MTP: 10/0.6 -> 8/0.5 (DEC-073 tuning)
sed -i "s/RELAXED_TOP_N = 10/RELAXED_TOP_N = 8/" atom/model_ops/rejection_sampler.py
sed -i "s/RELAXED_DELTA = 0.6/RELAXED_DELTA = 0.5/" atom/model_ops/rejection_sampler.py

# MLA: num_kv_splits=16 -> num_kv_splits=None (Session 6A intervention)
sed -i "s/num_kv_splits=16,/num_kv_splits=None,/" atom/model_ops/attention_mla.py

# Verify
echo "--- rejection_sampler.py ---"
grep -nE "RELAXED_TOP_N|RELAXED_DELTA" atom/model_ops/rejection_sampler.py | head -2
echo "--- attention_mla.py ---"
grep -n "num_kv_splits" atom/model_ops/attention_mla.py | head

# Version dump for sanity
echo "--- Versions ---"
cat /opt/rocm/.info/version
python3 -c "import torch; print(\"PyTorch:\", torch.__version__)"
cd /app/aiter-test && echo "aiter: $(git rev-parse HEAD)"
cd /app/ATOM && echo "ATOM:  $(git rev-parse HEAD)"
pip show flydsl 2>&1 | grep Version
'

echo "=== 5. Verify checkpoint exists ==="
~/bin/docker exec "$CONTAINER_NAME" ls -la /projects/teamA/danish/models_merged/DSR1-drafter-FP4/config.json || {
  echo "ERROR: merged checkpoint missing. Rebuild via merge_dec075_v5.py" >&2
  exit 1
}

echo ""
echo "=== DSR_beta container READY ==="
echo "Next: run scripts/dsr_beta_launch.sh to start the server"
