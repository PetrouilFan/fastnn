#!/usr/bin/env bash
# FN3-24 (#70) trained-model window: staged accuracy/coverage and staged timing.
#
# The trained *resident* path is blocked by an identity-`Reshape`/`Gather`
# defect another owner is fixing, so every arm here is staged and labelled so.
# The script waits for a quiet host (the #72 protocol) because one arm is a
# timing measurement, then builds release and runs both trained tests.
set -uo pipefail
cd /home/petrouil/Projects/github/fastnn-wt-70-quant-vulkan

LOG=docs/reports/fn3-30/trained.log
mkdir -p docs/reports/fn3-30
exec > >(tee "$LOG") 2>&1

export FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e
export FN3_VK_REQUIRE_PHYSICAL=1
export FN3_QUANT_TRAINED=1
export FN3_QUANT_TRAINED_PERF=1
export FN3_QUANT_TRAINED_MODEL=/tmp/opencode/parity-fresh/models/trained-gpt2/onnx/model.onnx
export FN3_QUANT_TRAINED_REQUIRED=1
export FN3_QUANT_TRAINED_ITERATIONS=5
export FN3_QUANT_TRAINED_WARMUPS=1

echo "=== trained window ==="
date -Is
echo "lock acquired: $(date -Is)"
cat /proc/loadavg
echo "--- waiting for a quiet host (no cargo/rustc) ---"
waited=0
while pgrep -x cargo >/dev/null || pgrep -x rustc >/dev/null; do
  sleep 10
  waited=$((waited + 10))
  if [ "$waited" -ge 5400 ]; then
    echo "gave up after ${waited}s with a build still running"
    exit 3
  fi
done
echo "waited ${waited}s; load $(cat /proc/loadavg)"

echo "--- release build ---"
CARGO_BUILD_JOBS=3 cargo test --release --features vulkan --test fn3_vulkan_quant --no-run
echo "build exit=$?"

echo "--- accuracy and coverage (staged) ---"
date -Is
cat /proc/loadavg
cargo test --release --features vulkan --test fn3_vulkan_quant trained_accuracy_and_coverage \
  -- --nocapture --test-threads=1
accuracy_status=$?
echo "accuracy exit=$accuracy_status"

echo "--- staged latency ---"
date -Is
cat /proc/loadavg
cargo test --release --features vulkan --test fn3_vulkan_quant trained_staged_latency \
  -- --nocapture --test-threads=1
latency_status=$?
echo "latency exit=$latency_status"
date -Is
exit $((accuracy_status + latency_status))
