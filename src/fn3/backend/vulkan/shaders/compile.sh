#!/usr/bin/env bash
# Compile the FN3 Vulkan compute shaders to committed SPIR-V artefacts.
#
# The Rust build only embeds the committed `.spv` files with `include_bytes!`;
# it never runs a shader compiler. This script is the only place GLSL is
# compiled, and it records the disassembly (`.spvasm`) for review. It is the
# production counterpart of `spike/vk-bringup/scripts/compile-shaders.sh`, with
# the same rule: one committed `.spv` per `.comp`, validated by `spirv-val`.
#
# Requires: glslc, spirv-val, spirv-dis (Arch: shaderc, spirv-tools).
set -euo pipefail

cd "$(dirname "$0")"

for shader in *.comp; do
  out="${shader}.spv"
  glslc --target-env=vulkan1.1 -fshader-stage=compute -O "$shader" -o "$out"
  spirv-val --target-env vulkan1.1 "$out"
  spirv-dis "$out" -o "${shader}.spvasm"
  echo "compiled and validated $shader -> $out"
done

glslc --version | head -n 1 || true
spirv-val --version || true
