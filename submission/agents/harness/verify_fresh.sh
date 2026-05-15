#!/bin/bash
# Fresh-container verify (METR-style). Runs INSIDE the AMD host (not container).
#
# Usage:
#   verify_fresh.sh <src_container> <new_name> <patch_so> <target_path_in_container> <model>
#
# Example:
#   verify_fresh.sh re4c_v10 dsr1_verify_$(date +%s) /tmp/runs/dsr1/<ts>/geak/cand3.so \
#       /app/aiter-test/aiter/jit/module_custom_all_reduce.so dsr1
#
# Side effects:
#   - docker commit src_container -> tmp image
#   - docker run new container off that image
#   - patch the .so inside it
#   - boot + run /tmp/official_bench_v1.sh
#   - write /tmp/runs/<model>/<ts>/bench_fresh_summary.txt
#   - docker rm + docker rmi cleanup
#
# Exit 0 if bench completed (regardless of pass/fail). Foreman parses summary.

set -u
SRC=${1:?src container}
NEW=${2:?new container name}
PATCH_SO=${3:?path to patch .so}
TARGET_PATH=${4:?target path inside container}
MODEL=${5:?model dsr1|kimi}

case "$MODEL" in
  dsr1) BOOT_SCRIPT=/tmp/boot_a26.sh; BOOT_PORT=8890 ;;
  kimi) BOOT_SCRIPT=/tmp/boot_kimi.sh; BOOT_PORT=8890 ;;
  *) echo "ERROR: model must be dsr1 or kimi"; exit 2 ;;
esac

TS=$(date +%Y%m%d_%H%M%S)
IMG="local/${MODEL}_verify:${TS}"

echo "[verify_fresh] commit $SRC -> $IMG"
docker commit "$SRC" "$IMG" >/dev/null || { echo "commit failed"; exit 4; }

echo "[verify_fresh] starting fresh container $NEW"
docker run -d --name "$NEW" \
  --gpus all --network host \
  --shm-size=8g --ipc=host \
  -v /tmp:/host_tmp \
  "$IMG" sleep infinity >/dev/null || { echo "run failed"; exit 5; }

cleanup() {
  echo "[verify_fresh] cleanup"
  docker rm -f "$NEW" >/dev/null 2>&1 || true
  docker rmi "$IMG" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[verify_fresh] applying patch $PATCH_SO -> $TARGET_PATH"
docker exec "$NEW" cp "$TARGET_PATH" "${TARGET_PATH}.pre_verify" || true
docker cp "$PATCH_SO" "$NEW:$TARGET_PATH" || { echo "patch copy failed"; exit 6; }

echo "[verify_fresh] booting fresh server"
docker exec "$NEW" bash -c "cp /host_tmp/boot_a26.sh /tmp/ && cp /host_tmp/official_bench_v1.sh /tmp/ && cp /host_tmp/warmup.sh /tmp/ && bash /tmp/boot_a26.sh" >/dev/null

# wait for Application startup complete (poll up to 15 min)
for i in $(seq 1 90); do
  sleep 10
  if docker exec "$NEW" grep -q "Application startup complete" /tmp/a26_boot_*.log 2>/dev/null; then
    echo "[verify_fresh] server ready (after ~${i}0s)"
    break
  fi
  if [ "$i" -eq 90 ]; then
    echo "[verify_fresh] BOOT TIMEOUT 15min"
    exit 7
  fi
done

# warm + bench
docker exec "$NEW" bash /tmp/warmup.sh >/dev/null 2>&1 || true
echo "[verify_fresh] running official bench"
docker exec "$NEW" bash /tmp/official_bench_v1.sh > "/tmp/runs/${MODEL}/${TS}/bench_fresh.log" 2>&1

# copy summary out
LATEST_SUMMARY=$(docker exec "$NEW" ls -t /tmp/official_summary_*.txt 2>/dev/null | head -1)
if [ -n "$LATEST_SUMMARY" ]; then
  docker cp "$NEW:$LATEST_SUMMARY" "/tmp/runs/${MODEL}/${TS}/bench_fresh_summary.txt"
  echo "[verify_fresh] summary at /tmp/runs/${MODEL}/${TS}/bench_fresh_summary.txt"
  cat "/tmp/runs/${MODEL}/${TS}/bench_fresh_summary.txt"
else
  echo "[verify_fresh] WARN no summary file produced"
fi

echo "[verify_fresh] DONE"
