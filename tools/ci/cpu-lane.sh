#!/usr/bin/env bash
# Lane: cpu (job: cpu-rust)
#
# Builds and tests the Rust crate on the host CPU with no GPU involved.
# It proves nothing about Vulkan; see tools/ci/lanes.json.
#
# Clippy policy: warnings are denied. #67 cleared the style-only debt that
# docs/roadmap/fn3-ledger.md recorded, so the budget is zero and there is no
# allowance for a "known" warning: a new one fails the lane.
set -uo pipefail

CI_LANE=cpu
CI_JOB=cpu-rust
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

CLIPPY_MAX_WARNINGS="${CLIPPY_MAX_WARNINGS:-0}"
CLIPPY_LOG="$CI_ROOT/target/ci-clippy.log"
mkdir -p "$(dirname "$CLIPPY_LOG")"

ci::describe cpu


ci::run "Format check" "cargo fmt --check"
ci::run "Version consistency" "python3 scripts/ci/check-versions.py"

echo
echo "== Clippy (warning budget: $CLIPPY_MAX_WARNINGS) =="
echo "\$ cargo clippy --all-targets -- -D warnings"
if ! cargo clippy --all-targets -- -D warnings >"$CLIPPY_LOG" 2>&1; then
    tail -n 40 "$CLIPPY_LOG"
    ci::failed "cargo clippy --all-targets -- -D warnings failed" \
        "cargo clippy --all-targets -- -D warnings"
fi

warning_count="$(grep -cE '\.rs:[0-9]+:[0-9]+: warning:' "$CLIPPY_LOG" || true)"
if (( warning_count > CLIPPY_MAX_WARNINGS )); then
    echo "warning count $warning_count exceeds budget $CLIPPY_MAX_WARNINGS"
    grep -E '\.rs:[0-9]+:[0-9]+: warning:' "$CLIPPY_LOG" || true
    ci::failed "clippy warning budget exceeded ($warning_count > $CLIPPY_MAX_WARNINGS)" \
        "cargo clippy --all-targets -- -D warnings"
fi
echo "clippy warnings: $warning_count (budget $CLIPPY_MAX_WARNINGS; warnings are denied)"
ci::check "clippy (warning budget 0)"

# The documentation build is its own gate. It emitted 86 warnings until they
# were cleared, and nothing stopped them coming back: `cargo clippy` does not
# read doc comments the way rustdoc does, so it cannot see an unresolved
# intra-doc link. Those were not cosmetic — two of them named items that no
# longer exist, which is a wrong promise in the public docs.
ci::run "Rustdoc (warnings are denied)" \
    'RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --lib' \
    '{"name": "rustdoc warning-free", "status": "passed"}'

ci::run "Build with --no-default-features (pyo3 extension-module cannot link standalone targets)" \
    "cargo build --no-default-features"
ci::run "Library tests" \
    "cargo test --lib -- --test-threads=1"
# Every integration target, not a hand-picked three.
#
# This lane used to run two integration tests out of a much larger suite, which
# left most of the crate's integration tests compiled nowhere and executed
# nowhere. A test that never runs is not coverage.
ci::run "Integration tests (every target)" \
    "cargo test --no-fail-fast --tests"
# The binary is exercised as a binary: its own argv parsing, exit codes and
# output stream, with the Python feature off (PyO3's extension-module mode
# cannot link a standalone binary). The in-process `fn3_cli` tests cover the
# same surface through `cli::run`.
ci::run "Inference CLI binary smoke" \
    "cargo run --quiet --no-default-features --features cli,parallel --bin fastnn-infer -- --version"
ci::run "CI tooling selftests" \
    "python3 tools/ci/model_pin.py selftest && python3 tools/ci/lane_status.py selftest"

ci::passed "fmt clean; versions agree; clippy warnings denied (budget 0); rustdoc warning-free; no-default build; lib and every integration target pass; CLI binary smoke; CI tooling selftests"
