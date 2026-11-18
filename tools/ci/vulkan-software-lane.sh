#!/usr/bin/env bash
# Lane: vulkan-software-smoke  (CI-ONLY / SOFTWARE ICD)
#
# This lane runs Vulkan under lavapipe, a CPU implementation of Vulkan. It is
# labelled CI-ONLY everywhere it appears: its output, its job name and the run
# summary. It is NOT physical-GPU evidence and must never be cited as such.
#
# Today the Vulkan backend does not exist (FN3-11 / #57), so this lane reports
# "Vulkan backend not yet implemented (#57)" as NOT VERIFIED. It does not pass.
set -uo pipefail

CI_LANE=vulkan-software-smoke
CI_JOB=vulkan-software
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

ICD="${FN3_SOFTWARE_ICD:-/usr/share/vulkan/icd.d/lvp_icd.json}"

ci::describe vulkan-software-smoke
echo
echo "########################################################################"
echo "# CI-ONLY LANE: SOFTWARE VULKAN (lavapipe). NOT A GPU. NOT GPU EVIDENCE."
echo "# A pass here says nothing about any physical device or driver."
echo "########################################################################"

if [[ ! -f "$ICD" ]]; then
    ci::not_verified "software ICD '$ICD' is not installed; the CI-only software lane did not run"
fi

# Real precondition: the ICD must enumerate a software device. This proves the
# loader path this lane depends on, not anything about fastnn.
if command -v vulkaninfo >/dev/null 2>&1; then
    VK_SUMMARY="$(VK_ICD_FILENAMES="$ICD" vulkaninfo --summary 2>&1 || true)"
    if ! grep -qiE 'llvmpipe|lavapipe|VK_PHYSICAL_DEVICE_TYPE_CPU' <<<"$VK_SUMMARY"; then
        ci::failed "VK_ICD_FILENAMES=$ICD did not enumerate a software device" \
            "VK_ICD_FILENAMES=$ICD vulkaninfo --summary"
    fi
    echo "software ICD precondition: lavapipe enumerated"
    ci::check "software ICD enumerates a CPU-typed device"
else
    echo "vulkaninfo not installed; skipping the ICD precondition check"
fi

if ! ci::cargo_feature_declared vulkan; then
    ci::not_verified "CI-ONLY software Vulkan lane did not run: Vulkan backend not yet implemented (#57) — Cargo.toml declares no 'vulkan' feature (FN3-11). This is not a GPU check and not a pass."
fi
if [[ ! -f "$CI_ROOT/tests/fn3_vulkan_smoke.rs" ]]; then
    ci::not_verified "CI-ONLY software Vulkan lane did not run: the 'vulkan' feature exists but tests/fn3_vulkan_smoke.rs (FN3-11 / #57) does not"
fi

SMOKE_LOG="$CI_ROOT/target/ci-vulkan-software.log"
mkdir -p "$(dirname "$SMOKE_LOG")"
echo
echo "== CI-ONLY software Vulkan smoke (lavapipe, validation enabled) =="
echo "\$ VK_ICD_FILENAMES=$ICD VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture"
if ! VK_ICD_FILENAMES="$ICD" \
     VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
     VK_LAYER_ENABLES="VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT" \
     FN3_VK_SOFTWARE_ONLY=1 \
     cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture 2>&1 | tee "$SMOKE_LOG"; then
    ci::failed "CI-ONLY software Vulkan smoke failed under lavapipe" \
        "VK_ICD_FILENAMES=$ICD cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture"
fi

if ! grep -qiE 'llvmpipe|lavapipe|software' "$SMOKE_LOG"; then
    ci::failed "the smoke test never reported the software device; it may have used a physical GPU, which invalidates a CI-ONLY lane" \
        "VK_ICD_FILENAMES=$ICD cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture"
fi
ci::check "CI-only lavapipe smoke test"

ci::passed "CI-ONLY software Vulkan smoke passed under lavapipe with validation enabled; this is NOT physical GPU evidence"
