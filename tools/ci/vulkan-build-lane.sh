#!/usr/bin/env bash
# Lane: vulkan-build
#
# Two checks that need no Vulkan device:
#   1. every committed .spv validates with spirv-val;
#   2. the committed SPIR-V is reproducible: the single shader compiler
#      (spike/vk-bringup/scripts/compile-shaders.sh) is called, not duplicated,
#      and the working tree must stay byte-identical.
#
# When Cargo.toml declares the `vulkan` feature (FN3-11 / #57) this lane also
# builds it. Until then the feature-build half is NOT VERIFIED, and the lane
# says so instead of passing on shader validation alone.
set -uo pipefail

CI_LANE=vulkan-build
CI_JOB=vulkan-build
source "$(dirname "${BASH_SOURCE[0]}")/lane-lib.sh"

ci::describe vulkan-build

SPV_FILES=()
while IFS= read -r file; do
    SPV_FILES+=("$file")
done < <(cd "$CI_ROOT" && git ls-files '*.spv')
if (( ${#SPV_FILES[@]} == 0 )); then
    ci::not_verified "no committed .spv files found; there is nothing to validate"
fi
if ! command -v spirv-val >/dev/null 2>&1; then
    ci::not_verified "spirv-val is not installed; committed SPIR-V was not validated (install spirv-tools)"
fi

for file in "${SPV_FILES[@]}"; do
    ci::run "spirv-val $file" "spirv-val --target-env vulkan1.1 '$file'"
done

if ! command -v glslc >/dev/null 2>&1 || ! command -v spirv-dis >/dev/null 2>&1; then
    ci::not_verified "spirv-val passed for ${#SPV_FILES[@]} artifact(s), but glslc/spirv-dis are missing so source-to-SPIR-V reproducibility was NOT checked"
fi

if [[ ! -x "$CI_ROOT/spike/vk-bringup/scripts/compile-shaders.sh" ]]; then
    ci::not_verified "spirv-val passed, but spike/vk-bringup/scripts/compile-shaders.sh is missing so reproducibility was NOT checked"
fi

echo
echo "== Shader source -> committed SPIR-V reproducibility =="
echo "\$ spike/vk-bringup/scripts/compile-shaders.sh && git diff --exit-code -- spike/vk-bringup/shaders"
COMPILE_LOG="$CI_ROOT/target/ci-compile-shaders.log"
mkdir -p "$(dirname "$COMPILE_LOG")"
set +e
( cd "$CI_ROOT" && ./spike/vk-bringup/scripts/compile-shaders.sh ) >"$COMPILE_LOG" 2>&1
COMPILE_STATUS=$?
set +e
cat "$COMPILE_LOG"
if ! git -C "$CI_ROOT" diff --quiet -- spike/vk-bringup/shaders; then
    git -C "$CI_ROOT" diff --stat -- spike/vk-bringup/shaders
    ci::failed "committed SPIR-V does not match the shader compiler output; regenerate and commit the artifacts" \
        "spike/vk-bringup/scripts/compile-shaders.sh && git diff -- spike/vk-bringup/shaders"
fi
if (( COMPILE_STATUS != 0 )); then
    # The spike script ends with `glslc --version | head -n 1` under `set -o
    # pipefail`, so it can exit 141 (SIGPIPE) after the compile has succeeded.
    # Only that exact case is tolerated, and only because the artifacts were
    # independently validated above and are byte-identical.
    if (( COMPILE_STATUS == 141 )) && grep -q 'compiled and validated' "$COMPILE_LOG"; then
        echo "::warning title=vulkan-build::compile-shaders.sh exited 141 (SIGPIPE from its 'glslc --version | head -n 1' banner); artifacts were produced and match the committed bytes"
    else
        ci::failed "the shader compiler failed (exit $COMPILE_STATUS); see $COMPILE_LOG" \
            "spike/vk-bringup/scripts/compile-shaders.sh"
    fi
fi
for file in "${SPV_FILES[@]}"; do
    if ! spirv-val --target-env vulkan1.1 "$file"; then
        ci::failed "recompiled $file failed spirv-val" "spirv-val --target-env vulkan1.1 '$file'"
    fi
done
ci::check "SPIR-V validation + reproducibility (${#SPV_FILES[@]} artifact(s))"

if ! ci::cargo_feature_declared vulkan; then
    ci::not_verified "committed SPIR-V validated (${#SPV_FILES[@]} artifact(s)) and shader sources recompiled identically, but Cargo.toml declares no 'vulkan' feature: the feature build did NOT run (Vulkan backend not yet implemented, FN3-11 / #57)"
fi

ci::run "Build with --features vulkan" \
    "cargo build --no-default-features --features vulkan"
ci::run "Compile the Vulkan smoke test (no device is created)" \
    "cargo test --no-default-features --features vulkan --test fn3_vulkan_smoke --no-run"

ci::passed "committed SPIR-V validates and is reproducible, and the crate builds with --features vulkan (no device was created)"
