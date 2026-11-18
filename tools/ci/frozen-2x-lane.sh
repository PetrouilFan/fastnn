#!/usr/bin/env bash
# Lane: frozen-2x (transitional; deleted by FN3-19 / #65)
#
# docs/roadmap/fn3-architecture.md keeps exactly one 2.x gate while the trees
# coexist: the frozen 2.x modules must still compile and their suites must
# still pass. This lane is that gate and nothing else. It is NOT 3.0 evidence.
set -uo pipefail

CI_LANE=frozen-2x
CI_JOB=frozen-2x
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

ci::describe frozen-2x

ci::run "Quantized pipeline (2.x)" \
    "cargo test --test quantized_pipeline -- --test-threads=1"
ci::run "Optimizer (2.x)" \
    "cargo test --test optim_test -- --test-threads=1"
ci::run "Autograd gradient checks (2.x)" \
    "cargo test --test autograd_gradient_checks -- --test-threads=1"
ci::run "CPU arena telemetry (2.x)" \
    "cargo test --test cpu_arena_telemetry -- --test-threads=1"
ci::run "All Rust integration tests (2.x + fn3)" \
    "cargo test --tests -- --test-threads=1"

ci::passed "frozen 2.x targeted suites and the full Rust integration suite pass while src/fn3/ coexists; no 3.0 claim is made"
