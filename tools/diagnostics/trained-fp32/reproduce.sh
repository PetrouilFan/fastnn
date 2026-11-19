#!/bin/sh
# Reproduce the #76 trained-GPT-2 FP32 investigation.
#
# Requirements:
#   * the pinned 498,126,358-byte ONNX graph (not committed);
#   * a Python environment with numpy, onnx and onnxruntime;
#   * a release Rust toolchain.
#
# Every model run is serialised under /tmp/opencode/fn3-measure.lock and records
# the load average, per the project measurement rules.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
cd "$ROOT"

MODEL=${FN3_TRAINED_ONNX:-}
PYTHON=${FN3_PYTHON:-python3}
OUT=${FN3_DIAG_OUT:-/tmp/opencode/trained-fp32}
LOCK=${FN3_MEASURE_LOCK:-/tmp/opencode/fn3-measure.lock}

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "FN3_TRAINED_ONNX must name the pinned trained graph" >&2
    exit 2
fi
SIZE=$(stat -c '%s' "$MODEL")
DIGEST=$(sha256sum "$MODEL" | cut -d' ' -f1)
if [ "$SIZE" != "498126358" ] || [ "$DIGEST" != "42c1e92a21c41ebd3c536d4c6f09732450a43db35977593236f696d3346c7c0d" ]; then
    echo "not the pinned graph: $SIZE bytes, sha256 $DIGEST" >&2
    exit 2
fi
echo "pinned graph verified: $SIZE bytes, $DIGEST"

mkdir -p "$OUT"

run_locked() {
    label=$1
    shift
    echo "== $label"
    echo "   loadavg before: $(cat /proc/loadavg)"
    echo "   cargo/rustc processes: $(pgrep -c 'cargo|rustc' || true)"
    (
        flock 9
        "$@"
    ) 9>"$LOCK"
    echo "   loadavg after: $(cat /proc/loadavg)"
}

run_locked "ORT 1.23.2 reference" \
    "$PYTHON" tools/quant/ort_reference.py --model "$MODEL" --out "$OUT/ort-123"

run_locked "fn3 CPU parity test" \
    env FN3_TRAINED_ONNX="$MODEL" FN3_TRAINED_OUT="$OUT/fn3-parity" \
    CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
    --test fn3_trained_parity -- --ignored --nocapture

run_locked "fn3 per-instruction probe" \
    env FN3_TRAINED_ONNX="$MODEL" FN3_PROBE_OUT="$OUT/fn3-layers" \
    CARGO_BUILD_JOBS=2 cargo test --release --no-default-features \
    --test fn3_trained_layers -- --ignored --nocapture

run_locked "ORT per-layer intermediates" \
    "$PYTHON" tools/diagnostics/trained-fp32/extract_intermediates.py \
    --model "$MODEL" --out "$OUT/ort-layers"

run_locked "float64 reference" \
    "$PYTHON" tools/diagnostics/trained-fp32/reference_float64.py \
    --model "$MODEL" --out "$OUT/f64-layers"

run_locked "reduce: fn3 vs ORT" \
    "$PYTHON" tools/diagnostics/trained-fp32/compare_layers.py \
    --fn3 "$OUT/fn3-layers" --ort "$OUT/ort-layers" --model "$MODEL" \
    --out "$OUT/layers.json"

run_locked "reduce: fn3 vs float64" \
    "$PYTHON" tools/diagnostics/trained-fp32/compare_layers.py \
    --fn3 "$OUT/fn3-layers" --ort "$OUT/f64-layers" --model "$MODEL" \
    --out "$OUT/fn3_vs_f64.json"

run_locked "reduce: ORT vs float64" \
    "$PYTHON" tools/diagnostics/trained-fp32/compare_layers.py \
    --fn3 "$OUT/ort-layers" --ort "$OUT/f64-layers" --model "$MODEL" \
    --out "$OUT/ort_vs_f64.json"

run_locked "accumulation attribution" \
    "$PYTHON" tools/diagnostics/trained-fp32/attribute_accumulation.py \
    --fn3 "$OUT/fn3-layers" --ort "$OUT/ort-layers" --model "$MODEL" \
    --out "$OUT/accumulation.json"

run_locked "summary" \
    "$PYTHON" tools/diagnostics/trained-fp32/summarize.py \
    --fn3 "$OUT/fn3-layers" --ort "$OUT/ort-layers" --f64 "$OUT/f64-layers" \
    --parity "$OUT/fn3-parity/parity-report.json" \
    --out "$OUT/summary"

if [ "${FN3_RUN_VULKAN:-0}" = "1" ]; then
    UUID=${FN3_VK_DEVICE_UUID:-}
    if [ -z "$UUID" ]; then
        echo "FN3_RUN_VULKAN=1 needs FN3_VK_DEVICE_UUID" >&2
        exit 2
    fi
    # The resident default currently fails on the trained graph (see the
    # report's Vulkan section); the staged arm is the arithmetic comparison.
    run_locked "fn3 Vulkan staged parity" \
        env FN3_TRAINED_ONNX="$MODEL" FN3_TRAINED_OUT="$OUT/fn3-parity-vulkan" \
        FN3_TRAINED_VULKAN=1 FN3_VK_STAGED=1 FN3_VK_REQUIRE_PHYSICAL=1 \
        FN3_VK_DEVICE_UUID="$UUID" \
        CARGO_BUILD_JOBS=2 cargo test --release --features vulkan \
        --test fn3_trained_parity -- --ignored --nocapture
fi

echo "done; reduced outputs in $OUT/summary"
