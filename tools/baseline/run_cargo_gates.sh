#!/usr/bin/env bash
# Run the FastNN 2.6.0 baseline Rust gates and record one log per gate.
#
# Usage:
#   tools/baseline/run_cargo_gates.sh [OUTPUT_DIR]
#
# Defaults to docs/reports/fn3-01/logs. Every gate runs even if an earlier one
# fails; the summary file records each exit status. Build only in this worktree.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/docs/reports/fn3-01/logs}"
mkdir -p "$OUT"
cd "$ROOT" || exit 2

export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
SUMMARY="$OUT/cargo-gates-summary.txt"
: > "$SUMMARY"

run_gate() {
    local name="$1"
    shift
    echo "+ $*" | tee -a "$SUMMARY"
    "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    echo "  exit=$status log=$OUT/$name.log" | tee -a "$SUMMARY"
    return 0
}

run_gate lib \
    cargo test --lib -- --test-threads=1
run_gate quantized_pipeline \
    cargo test --test quantized_pipeline -- --test-threads=1
run_gate optim_test \
    cargo test --test optim_test -- --test-threads=1
run_gate autograd_gradient_checks \
    cargo test --test autograd_gradient_checks -- --test-threads=1
run_gate cpu_arena_telemetry \
    cargo test --test cpu_arena_telemetry -- --test-threads=1

echo "done: $SUMMARY"
