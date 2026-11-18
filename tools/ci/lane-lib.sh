#!/usr/bin/env bash
# Shared helpers for the FastNN CI lanes.
#
# Every lane script:
#   1. prints its identity from tools/ci/lanes.json (what it proves and does not
#      prove) before running anything;
#   2. prints `REPRODUCE: <command>` on every failure;
#   3. records an explicit status for tools/ci/lane_status.py, so a lane that
#      did not run is rendered NOT VERIFIED instead of silently passing.
#
# This file is sourced, never executed.

set -uo pipefail

CI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_LANE="${CI_LANE:-}"
CI_JOB="${CI_JOB:-${CI_LANE}-job}"
CI_STATUS_DIR="${LANE_STATUS_DIR:-$CI_ROOT/target/ci-lane-status}"
CI_LANE_START="${SECONDS:-0}"
CI_CHECKS=()
export LANE_STATUS_DIR="$CI_STATUS_DIR"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

ci::describe() {
    local lane="$1"
    python3 "$CI_ROOT/tools/ci/lane_status.py" describe --lane "$lane"
}

ci::record() {
    local status="$1"
    local detail="$2"
    shift 2
    local elapsed=$(( SECONDS - CI_LANE_START ))
    python3 "$CI_ROOT/tools/ci/lane_status.py" record \
        --lane "$CI_LANE" \
        --job "$CI_JOB" \
        --status "$status" \
        --detail "$detail" \
        --duration-seconds "$elapsed" \
        "$@"
}

ci::passed() {
    local args=()
    local check
    for check in "${CI_CHECKS[@]}"; do
        args+=(--check "$check")
    done
    ci::record passed "$1" "${args[@]}"
    echo
    echo "LANE $CI_LANE / $CI_JOB: PASSED — $1"
}

ci::not_verified() {
    local detail="$1"
    echo "::warning title=$CI_LANE NOT VERIFIED::$detail"
    ci::record not_verified "$detail"
    echo
    echo "LANE $CI_LANE / $CI_JOB: NOT VERIFIED — $detail"
    echo "This is not a pass. See tools/ci/lanes.json for what this lane would prove."
    exit 0
}

ci::failed() {
    local detail="$1"
    local reproduce="${2:-${CI_REPRODUCE:-}}"
    echo "::error title=$CI_LANE FAILED::$detail"
    if [[ -n "$reproduce" ]]; then
        echo "REPRODUCE: $reproduce"
    fi
    local args=()
    if [[ -n "$reproduce" ]]; then
        args+=(--reproduce "$reproduce")
    fi
    local check
    for check in "${CI_CHECKS[@]}"; do
        args+=(--check "$check")
    done
    ci::record failed "$detail" "${args[@]}"
    echo
    echo "LANE $CI_LANE / $CI_JOB: FAILED — $detail"
    exit 1
}

# ci::run <label> <command string> [extra --check JSON]
#
# The command string is executed with `bash -c` and is also the exact command
# printed for local reproduction.
ci::run() {
    local label="$1"
    local command="$2"
    local check="${3:-}"
    local started=$SECONDS
    echo
    echo "== $label =="
    echo "\$ $command"
    if bash -c "$command"; then
        echo "== $label: ok ($(( SECONDS - started ))s) =="
        if [[ -n "$check" ]]; then
            CI_CHECKS+=("$check")
        fi
        return 0
    else
        local status=$?
        echo "== $label: FAILED (exit $status) =="
        CI_CHECKS+=("$(python3 - "$label" "$command" <<'PY'
import json, sys
print(json.dumps({"name": sys.argv[1], "status": "failed", "reproduce": sys.argv[2]}))
PY
)")
        ci::failed "$label failed (exit $status)" "$command"
    fi
}

ci::check() {
    # Register a passed check for the status record.
    CI_CHECKS+=("$(python3 - "$1" <<'PY'
import json, sys
print(json.dumps({"name": sys.argv[1], "status": "passed"}))
PY
)")
}

ci::cargo_feature_declared() {
    grep -qE "^[[:space:]]*$1[[:space:]]*=" "$CI_ROOT/Cargo.toml"
}

CI_CHECKS=()
