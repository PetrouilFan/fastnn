#!/usr/bin/env bash
# Lane: cpu (job: cpu-rust)
#
# Builds and tests the Rust crate on the host CPU with no GPU involved.
# It proves nothing about Vulkan; see tools/ci/lanes.json.
#
# Clippy policy: warnings are NOT denied globally at this base SHA because the
# frozen 2.x tree has three pre-existing warnings and tests/fn3_contracts.rs
# (owned by #48) has one unused import. Instead:
#   * any warning that mentions src/fn3/ fails immediately, so the 3.0 tree
#     stays warning-clean from day one;
#   * the total count must stay at or below CLIPPY_MAX_WARNINGS, so new
#     warnings fail the lane without needing a deny flag.
# FN3-19 (#65) deletes 2.x and must set CLIPPY_MAX_WARNINGS=0 and
# `-D warnings`. FN3-18 (#64) owns the test-file cleanup.
set -uo pipefail

CI_LANE=cpu
CI_JOB=cpu-rust
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

CLIPPY_MAX_WARNINGS="${CLIPPY_MAX_WARNINGS:-4}"
CLIPPY_LOG="$CI_ROOT/target/ci-clippy.log"
mkdir -p "$(dirname "$CLIPPY_LOG")"

ci::describe cpu

# Pre-existing at the base SHA: scripts/ci/check-agents-freshness.py rejects the
# product-version literal "3.0" in src/fn3/AGENTS.md (owned by #48). It is not a
# gate here until that false positive is fixed, but it must not be silent.
echo "::warning title=cpu::AGENTS freshness check is red at the base SHA (src/fn3/AGENTS.md '3.0', #48); it is not run as a gate in this lane"
echo

ci::run "Format check" "cargo fmt --check"

echo
echo "== Clippy (warning budget: $CLIPPY_MAX_WARNINGS; src/fn3 must be warning-free) =="
echo "\$ cargo clippy --all-targets --message-format=short"
if ! cargo clippy --all-targets --message-format=short >"$CLIPPY_LOG" 2>&1; then
    tail -n 40 "$CLIPPY_LOG"
    ci::failed "cargo clippy failed to compile or lint" "cargo clippy --all-targets --message-format=short"
fi

warning_count="$(grep -cE '\.rs:[0-9]+:[0-9]+: warning:' "$CLIPPY_LOG" || true)"
fn3_warnings="$(grep -E 'src/fn3/[^:]*\.rs:[0-9]+:[0-9]+: warning:' "$CLIPPY_LOG" || true)"
if [[ -n "$fn3_warnings" ]]; then
    echo "$fn3_warnings"
    ci::failed "src/fn3/ produced clippy warnings; the 3.0 tree must stay warning-free" \
        "cargo clippy --all-targets --message-format=short | grep src/fn3/"
fi
if (( warning_count > CLIPPY_MAX_WARNINGS )); then
    echo "warning count $warning_count exceeds budget $CLIPPY_MAX_WARNINGS"
    grep -E '\.rs:[0-9]+:[0-9]+: warning:' "$CLIPPY_LOG" || true
    ci::failed "clippy warning budget exceeded ($warning_count > $CLIPPY_MAX_WARNINGS)" \
        "cargo clippy --all-targets --message-format=short"
fi
echo "clippy warnings: $warning_count (budget $CLIPPY_MAX_WARNINGS; pre-existing in frozen 2.x/#48 files)"
ci::check "clippy (warning budget)"

ci::run "Build with --no-default-features (pyo3 extension-module cannot link standalone targets)" \
    "cargo build --no-default-features"
ci::run "Library tests (frozen 2.x + fn3 unit tests)" \
    "cargo test --lib -- --test-threads=1"
ci::run "fn3 contract tests" \
    "cargo test --test fn3_contracts"
ci::run "fn3 CI smoke tests" \
    "cargo test --test fn3_ci_smoke"
ci::run "CI tooling selftests" \
    "python3 tools/ci/model_pin.py selftest && python3 tools/ci/lane_status.py selftest"

ci::passed "fmt clean; clippy within budget with zero src/fn3 warnings; no-default build; lib + fn3 contract + fn3 smoke tests pass; CI tooling selftests pass"
