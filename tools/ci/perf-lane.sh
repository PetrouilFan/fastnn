#!/usr/bin/env bash
# Advisory perf lane: measure one model, compare it with the committed report.
#
# The numbers a benchmark produces depend on the machine, so this lane never
# fails a pull request for being on a busy runner: it skips (exit 0, with a
# message) when the model or the baseline is absent, and only fails when a
# measurement it could take moved beyond the budgets in compare_perf.py.
#
# The model is not in the repository. Put it where the lane looks, or point
# PERF_MODEL at it:
#   python3 tools/parity/fetch_models.py --suite efficientvit   (when that suite exists)
#   PERF_MODEL=/path/to/efficientvit_depth_5m.onnx tools/ci/perf-lane.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${PERF_MODEL:-$ROOT/tools/parity/.cache/models/efficientvit-depth-5m/efficientvit_depth_5m_fp32.onnx}"
BASELINE="$ROOT/docs/reports/perf-evit-depth-5m-cpu.json"
HEAD="$ROOT/target/perf-head.json"

echo "== Perf lane (advisory) =="
if [[ ! -f "$MODEL" ]]; then
    echo "SKIPPED: no model at $MODEL; set PERF_MODEL to run this lane."
    exit 0
fi
if [[ ! -f "$BASELINE" ]]; then
    echo "SKIPPED: no baseline at $BASELINE; commit one report from a quiet machine."
    exit 0
fi

python3 -m pip install --quiet numpy >/dev/null 2>&1 || true
echo "\$ cargo build --release --lib"
if ! cargo build --release --lib; then
    echo "FAILED: the extension did not build"
    exit 1
fi
cp target/release/libfastnn.so fastnn/_core.abi3.so

echo "\$ python3 tools/perf/perf_report.py --out $HEAD"
if ! PYTHONPATH="$ROOT" python3 "$ROOT/tools/perf/perf_report.py" \
     --model "evit-depth-5m=$MODEL" --input pixel_values=1,3,336,448 \
     --threads 4 --warmup 3 --repeat 20 --out "$HEAD"; then
    echo "FAILED: the benchmark did not produce a report"
    exit 1
fi

echo "\$ python3 tools/perf/compare_perf.py --base $BASELINE --head $HEAD"
if ! python3 "$ROOT/tools/perf/compare_perf.py" --base "$BASELINE" --head "$HEAD"; then
    echo "FAILED: a measurement moved beyond the documented budget"
    exit 1
fi
echo "perf lane: green"
