#!/usr/bin/env bash
# Regenerate the committed FN3-05 evidence logs.
#
# Requires the release binary and the hardware facts recorded in README.md:
# an NVIDIA GTX 1080 Ti at the PHYS_UUID below and the lavapipe ICD at
# /usr/share/vulkan/icd.d/lvp_icd.json. The physical run is selected by UUID,
# not index, because the loader also enumerates lavapipe on this machine.
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="${BIN:-./target/release/vk-bringup}"
PHYS_UUID="${FN3_PHYS_UUID:-471b2bda-51e1-6451-e692-ff5d1e146f1e}"
NO_ICD="/tmp/fn3-no-such-icd.json"
mkdir -p logs

echo "== physical device list and diagnostics =="
"$BIN" --list >logs/list.log 2>&1

echo "== physical GTX 1080 Ti, validation on =="
"$BIN" --device-uuid "$PHYS_UUID" --validation \
    >logs/physical-gtx1080ti-validation.log 2>&1

echo "== lavapipe, validation on (CI-only evidence) =="
"$BIN" --software --validation >logs/lavapipe-ci-only.log 2>&1

expect_failure() {
    local log="$1"
    shift
    if "$@" >"$log" 2>&1; then
        echo "expected a failure but the command succeeded: $* (log: $log)" >&2
        exit 1
    fi
    echo "captured expected failure: $log"
}

echo "== no ICD =="
expect_failure logs/no-icd.log \
    env VK_ICD_FILENAMES="$NO_ICD" "$BIN" --list

echo "== nonexistent device index =="
expect_failure logs/bad-device-index.log "$BIN" --device-index 7

echo "== nonexistent device name =="
expect_failure logs/bad-device-name.log "$BIN" --device-name Radeon

echo "== missing device capability =="
expect_failure logs/missing-extension.log \
    "$BIN" --device-uuid "$PHYS_UUID" --require-extension VK_EXT_mesh_shader

echo "== missing loader =="
expect_failure logs/missing-loader.log \
    "$BIN" --loader-path /nonexistent/libvulkan.so.1

echo
echo "== summary =="
grep -H "selected device\|validation summary\|result:" logs/*.log || true
