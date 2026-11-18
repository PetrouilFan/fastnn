#!/usr/bin/env bash
# Compile the committed GLSL compute shaders to committed SPIR-V artefacts.
#
# The Rust crate embeds the .spv files with include_bytes!, so a normal
# `cargo build` needs neither glslc nor glslangValidator. This script is the
# only place shaders are compiled, and it records the exact toolchain output in
# the committed .spvasm disassembly for review.
#
# Requires: glslc, spirv-val, spirv-dis (Arch: shaderc, spirv-tools).
set -euo pipefail

cd "$(dirname "$0")/.."

shader="shaders/fma.comp"
out="shaders/fma.comp.spv"

glslc --target-env=vulkan1.1 -fshader-stage=compute -O "$shader" -o "$out"
spirv-val --target-env vulkan1.1 "$out"
spirv-dis "$out" -o "shaders/fma.comp.spvasm"

echo "compiled and validated $shader -> $out"
glslc --version | head -n 1
spirv-val --version
