#!/usr/bin/env bash
# Lane: cpu (job: cpu-rust)
#
# Builds and tests the Rust crate on the host CPU with no GPU involved.
# It proves nothing about Vulkan; see tools/ci/lanes.json.
#
# Clippy policy: warnings are not denied globally while the style-only lints
# listed in docs/roadmap/fn3-ledger.md remain (elidable lifetimes and
# `is_multiple_of` in src/fn3, constant-folded shape arithmetic in the fn3
# tests). The total count must stay at or below CLIPPY_MAX_WARNINGS so a new
# warning fails the lane. #67 owns driving this budget to zero and switching to
# `-D warnings`.
set -uo pipefail

CI_LANE=cpu
CI_JOB=cpu-rust
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

CLIPPY_MAX_WARNINGS="${CLIPPY_MAX_WARNINGS:-34}"
CLIPPY_LOG="$CI_ROOT/target/ci-clippy.log"
mkdir -p "$(dirname "$CLIPPY_LOG")"

ci::describe cpu


ci::run "Format check" "cargo fmt --check"

echo
echo "== Clippy (warning budget: $CLIPPY_MAX_WARNINGS) =="
echo "\$ cargo clippy --all-targets --message-format=short"
if ! cargo clippy --all-targets --message-format=short >"$CLIPPY_LOG" 2>&1; then
    tail -n 40 "$CLIPPY_LOG"
    ci::failed "cargo clippy failed to compile or lint" "cargo clippy --all-targets --message-format=short"
fi

warning_count="$(grep -cE '\.rs:[0-9]+:[0-9]+: warning:' "$CLIPPY_LOG" || true)"
if (( warning_count > CLIPPY_MAX_WARNINGS )); then
    echo "warning count $warning_count exceeds budget $CLIPPY_MAX_WARNINGS"
    grep -E '\.rs:[0-9]+:[0-9]+: warning:' "$CLIPPY_LOG" || true
    ci::failed "clippy warning budget exceeded ($warning_count > $CLIPPY_MAX_WARNINGS)" \
        "cargo clippy --all-targets --message-format=short"
fi
echo "clippy warnings: $warning_count (budget $CLIPPY_MAX_WARNINGS; style-only, #67 owns the zero budget)"
ci::check "clippy (warning budget)"

ci::run "Build with --no-default-features (pyo3 extension-module cannot link standalone targets)" \
    "cargo build --no-default-features"
ci::run "Library tests (fn3 + 2.x unit tests)" \
    "cargo test --lib -- --test-threads=1"
ci::run "fn3 contract tests" \
    "cargo test --test fn3_contracts"
ci::run "fn3 CI smoke tests" \
    "cargo test --test fn3_ci_smoke"
ci::run "CI tooling selftests" \
    "python3 tools/ci/model_pin.py selftest && python3 tools/ci/lane_status.py selftest"

ci::passed "fmt clean; clippy within the documented budget; no-default build; lib + fn3 contract + fn3 smoke tests pass; CI tooling selftests pass"
