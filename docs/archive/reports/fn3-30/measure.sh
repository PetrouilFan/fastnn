#!/usr/bin/env bash
# FN3-24 (#70) measurement window.
#
# Waits for a quiet host (the #72 protocol: no `cargo`/`rustc` process), then
# builds the release test and runs the timed harness inside the GPU and measure
# locks. Every boundary, the load average and both `pgrep` snapshots are
# recorded so a reader can see what the numbers were measured next to.
set -uo pipefail
cd /home/petrouil/Projects/github/fastnn-wt-70-quant-vulkan

LOG=docs/reports/fn3-30/measure.log
mkdir -p docs/reports/fn3-30
exec > >(tee "$LOG") 2>&1

echo "=== measure window ==="
date -Is
echo "lock acquired: $(date -Is)"
echo "pgrep before wait:"
pgrep -a -x cargo || echo "  none"
pgrep -a -x rustc || echo "  none"
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
echo "waited ${waited}s; pgrep before build:"
pgrep -a -x cargo || echo "  none"
pgrep -a -x rustc || echo "  none"
cat /proc/loadavg

echo "--- release build ---"
date -Is
CARGO_BUILD_JOBS=3 cargo test --release --features vulkan --test fn3_vulkan_quant --no-run
build_status=$?
echo "build exit=$build_status"
cat /proc/loadavg
if [ "$build_status" -ne 0 ]; then
  exit "$build_status"
fi

echo "--- timed run ---"
date -Is
pgrep -a -x cargo || echo "  no cargo"
pgrep -a -x rustc || echo "  no rustc"
cat /proc/loadavg
FN3_QUANT_PERF=1 \
FN3_QUANT_ITERATIONS=20 \
FN3_QUANT_WARMUPS=3 \
FN3_QUANT_OUT=docs/reports/fn3-30 \
FN3_VK_DEVICE_UUID=471b2bda-51e1-6451-e692-ff5d1e146f1e \
FN3_VK_REQUIRE_PHYSICAL=1 \
FASTNN_FN3_ONNX_MODEL=/tmp/opencode/parity-fresh/models/tiny-gpt2/onnx/model.onnx \
  cargo test --release --features vulkan --test fn3_vulkan_quant quantized_latency_and_memory \
    -- --nocapture --test-threads=1
run_status=$?
echo "run exit=$run_status"
date -Is
cat /proc/loadavg
exit "$run_status"
