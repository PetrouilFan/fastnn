#!/usr/bin/env bash
# Lane: vulkan-software-smoke  (CI-ONLY / SOFTWARE ICD)
#
# This lane runs Vulkan under lavapipe, a CPU implementation of Vulkan. It is
# labelled CI-ONLY everywhere it appears: its output, its job name and the run
# summary. It is NOT physical-GPU evidence and must never be cited as such.
#
# A pass requires a real dispatch: the log must contain the software test's
# proof line. A run in which every test skipped is a FAILURE here, not a pass,
# because a green lane with no execution is exactly the confusion this file is
# meant to prevent.
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
    ci::not_verified "CI-ONLY software Vulkan lane did not run: Cargo.toml declares no 'vulkan' feature, so there is no Vulkan backend to smoke. This is not a GPU check and not a pass."
fi
if [[ ! -f "$CI_ROOT/tests/fn3_vulkan_smoke.rs" ]]; then
    ci::not_verified "CI-ONLY software Vulkan lane did not run: the 'vulkan' feature exists but tests/fn3_vulkan_smoke.rs does not"
fi

SMOKE_LOG="$CI_ROOT/target/ci-vulkan-software.log"
mkdir -p "$(dirname "$SMOKE_LOG")"
echo
echo "== CI-ONLY software Vulkan smoke (lavapipe, validation enabled) =="
# Validation is requested in code (VulkanConfig::validated), which is what the
# smoke test asserts on. Forcing VK_INSTANCE_LAYERS on top of that duplicates
# the layer and makes the loader emit 'env var VK_INSTANCE_LAYERS defined'
# messages, and VK_LAYER_ENABLES makes it emit a VALIDATION-SETTINGS
# information line; the test's zero-delta guard counts both as warnings it
# cannot attribute to fastnn. So the lane sets only the ICD and the software
# gate, and lets the code own validation.
echo "\$ VK_ICD_FILENAMES=$ICD FN3_VK_TEST_SOFTWARE=1 cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture"
if ! VK_ICD_FILENAMES="$ICD" \
     FN3_VK_TEST_SOFTWARE=1 \
     cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture 2>&1 | tee "$SMOKE_LOG"; then
    ci::failed "CI-ONLY software Vulkan smoke failed under lavapipe" \
        "VK_ICD_FILENAMES=$ICD FN3_VK_TEST_SOFTWARE=1 cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture"
fi

# The smoke test skips itself unless the software gate is set explicitly. A
# suite where everything skipped still exits 0, so the proof line is the actual
# requirement: without it this lane proved nothing and must not go green.
if ! grep -q '\[fn3vk\]\[software-icd\]\[proof\]' "$SMOKE_LOG"; then
    ci::failed "the CI-only software smoke ran no dispatch: no '[fn3vk][software-icd][proof]' line in the log, so this lane has no evidence and is not a pass" \
        "VK_ICD_FILENAMES=$ICD FN3_VK_TEST_SOFTWARE=1 cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture"
fi
if ! grep -qiE 'llvmpipe|lavapipe' "$SMOKE_LOG"; then
    ci::failed "the smoke test never reported the software device; it may have used a physical GPU, which invalidates a CI-ONLY lane" \
        "VK_ICD_FILENAMES=$ICD FN3_VK_TEST_SOFTWARE=1 cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke -- --test-threads=1 --nocapture"
fi
ci::check "CI-only lavapipe smoke test dispatched and reported the software device"

ci::passed "CI-ONLY software Vulkan smoke passed under lavapipe with validation enabled; this is NOT physical GPU evidence"
