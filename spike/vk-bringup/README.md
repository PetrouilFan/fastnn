# FN3-05 — native Vulkan compute bring-up (spike)

Standalone proof that FastNN 3.0 can drive native Vulkan compute on the
physical GPU without wgpu, shaderc at build time, or a C++ toolchain.

This crate is deliberately **not** part of the repository workspace
(`[workspace]` is an empty table in `Cargo.toml`). Nothing under `src/**` or the
root `Cargo.toml` is touched. FN3-11 (#57) will fold the validated design into
`src/fn3/backend/vulkan/` behind a `vulkan` feature.

## What it proves

- Raw loader bring-up with `ash` **0.38.0** (`Cargo.lock` committed).
- Physical-device probe: identity, driver, API, UUID, queues, limits,
  relevant extensions, memory heaps.
- Explicit device selection by index, name substring or UUID, plus `--list`.
- Instance/device/compute-queue creation with optional
  `VK_LAYER_KHRONOS_validation` and a debug messenger on stderr.
- End-to-end storage-buffer compute:
  `out[i] = a[i] * b[i] + c[i] + bias` over 262 147 elements, an odd tail
  (`4097` workgroups of 64, last group has 3 active invocations).
- Device-local storage + host-visible staging upload/readback, checked transfer
  ranges, descriptor set and pipeline reuse.
- Fence-based completion for every submission, 100 repeated submissions
  reusing one command buffer and one fence, and a handle/byte ledger that must
  balance at teardown.
- Actionable failures for: missing loader, no ICD, bad device index, bad
  device name (with the available list), and a missing device capability.

## Toolchain and shader strategy

- Binding: `ash 0.38.0` (raw Vulkan loader through `libloading`; no C/C++
  compile step, no build-time Vulkan SDK).
- Shaders are GLSL compute, compiled **offline** by
  `scripts/compile-shaders.sh` into the committed `shaders/fma.comp.spv`
  (SPIR-V 1.3, Vulkan 1.1 target). The Rust build only embeds the committed
  bytes with `include_bytes!`; neither `glslc` nor `glslangValidator` is needed
  to build.
- `spirv-val --target-env vulkan1.1` validates the committed artefact and
  `shaders/fma.comp.spvasm` is the committed disassembly for review.
- Verified locally with `glslc 2026.4` and `spirv-tools 2026.4`.

## Build and run

```bash
cd spike/vk-bringup
CARGO_BUILD_JOBS=4 cargo build --release

# enumerate every physical device with full diagnostics
./target/release/vk-bringup --list

# physical GPU, selected explicitly by UUID, validation on
./target/release/vk-bringup \
  --device-uuid 471b2bda-51e1-6451-e692-ff5d1e146f1e \
  --validation

# software ICD lane (CI-only evidence; never a physical-GPU gate)
./target/release/vk-bringup --software --validation

# no ICD present
VK_ICD_FILENAMES=/tmp/fn3-no-such-icd.json ./target/release/vk-bringup --list

# failure paths
./target/release/vk-bringup --device-index 7
./target/release/vk-bringup --device-name Radeon
./target/release/vk-bringup --device-uuid 471b2bda-51e1-6451-e692-ff5d1e146f1e \
  --require-extension VK_EXT_mesh_shader
./target/release/vk-bringup --loader-path /nonexistent/libvulkan.so.1
```

`FN3_VK_VALIDATION=1`, `FN3_VK_SOFTWARE=1`, `FN3_VK_DEVICE_INDEX`,
`FN3_VK_DEVICE_NAME` and `FN3_VK_DEVICE_UUID` mirror the flags. Explicit CLI
flags win over the environment; supplying two selectors is a usage error.

Exit codes: `0` success, `1` dispatch/validation/ledger failure, `2` usage or
Vulkan call failure.

## CLI contract

| Flag | Meaning |
| --- | --- |
| `--list` | Enumerate and print every physical device, then exit |
| `--device-index N` | Select enumeration index `N` |
| `--device-name SUBSTR` | Case-insensitive substring match |
| `--device-uuid UUID` | `8-4-4-4-12` or 32 hex digits |
| `--software` | Set `VK_ICD_FILENAMES` to the installed lavapipe ICD |
| `--icd PATH` | Set `VK_ICD_FILENAMES` to an arbitrary ICD |
| `--loader-path PATH` | Load a specific `libvulkan.so` (diagnostics) |
| `--validation` | Enable `VK_LAYER_KHRONOS_validation` + debug utils |
| `--require-extension NAME` | Fail if the selected device lacks `NAME` (repeatable) |
| `--require-min-api X.Y` | Fail if the device API version is older |
| `--count N` | Dispatch element count (default 262147) |
| `--repeats N` | Repeated submissions (default 100) |

Default selection (no selector) picks a compute-capable device, preferring a
discrete GPU over integrated, virtual, then CPU. `--software` only overrides
the loader's ICD list; the normal selection policy still applies inside it.

## Evidence

All logs are raw stdout+stderr from the release binary and are committed under
`logs/`. `scripts/collect-evidence.sh` regenerates them.

| File | What it is |
| --- | --- |
| `logs/physical-gtx1080ti-validation.log` | **Physical GPU evidence**: GTX 1080 Ti selected by UUID, validation enabled, 0 errors / 0 warnings |
| `logs/lavapipe-ci-only.log` | **CI-only software evidence**: lavapipe, explicitly labelled; does not satisfy the physical-GPU gate |
| `logs/list.log` | `--list` output with full device diagnostics |
| `logs/no-icd.log` | `VK_ICD_FILENAMES` pointing at a nonexistent file |
| `logs/bad-device-index.log` | Requested index that does not exist |
| `logs/bad-device-name.log` | Name matching nothing, with the available-device list |
| `logs/missing-extension.log` | Requested capability the GTX 1080 Ti lacks |
| `logs/missing-loader.log` | Nonexistent loader path |

## Lifecycle and synchronisation

Every measured submission uses `vkQueueSubmit` + `vkWaitForFences` with a 60 s
hang-detector timeout; `vkDeviceWaitIdle` is used only by teardown. The
100-iteration test reuses the same command buffer, descriptor set, buffers and
fence, and changes the push-constant bias each iteration so a stale or skipped
submission cannot pass the sample checks.

Teardown order (enforced by types and `destroyed` flags, and checked by the
ledger):

1. Engine: wait idle, unmap readback, destroy fence, destroy command pool
   (frees its command buffer), destroy buffers and free their memory, destroy
   descriptor pool (frees the set), destroy pipeline, pipeline layout,
   descriptor-set layout, shader module.
2. Device: destroy logical device.
3. Instance: destroy debug messenger, destroy instance, then drop the loader
   (`ash::Entry` must not be unloaded before the instance is gone).

The command pool is created with `RESET_COMMAND_BUFFER` and owned by the
engine. Vulkan command pools are externally synchronised: one pool may be used
by one thread at a time. This spike is single-threaded; production must create
one pool per worker thread (see `NOTES-FN3-11.md`).

## Measured limits (GTX 1080 Ti, driver 580.196.07)

| Limit | Value |
| --- | --- |
| `maxComputeWorkGroupCount` | `[2147483647, 65535, 65535]` |
| `maxComputeWorkGroupSize` | `[1536, 1024, 64]` |
| `maxComputeWorkGroupInvocations` | 1536 |
| `maxComputeSharedMemorySize` | 49152 B |
| `maxStorageBufferRange` | 4294967295 B |
| `maxMemoryAllocationSize` | 4292870144 B (≈4 GiB, smaller than the 11 GiB heap) |
| `minUniformBufferOffsetAlignment` | 256 B |
| `minStorageBufferOffsetAlignment` | 16 B |
| `maxMemoryAllocationCount` | 4294967295 |
| `timestampPeriod` | 1 ns |
| `maxPushConstantsSize` | 256 B |
| `maxDescriptorSetStorageBuffers` | 1048576 |

Relevant extensions: 8-bit storage, 16-bit storage, shader_float16_int8,
buffer_device_address, synchronization2 and memory_budget are all present.

## Driver observations

- The NVIDIA 580.196.07 driver advertises `maxMemoryAllocationSize` of
  4 292 870 144 B, below the 11 GiB device-local heap. A production allocator
  must split large tensors into multiple allocations or suballocate from
  several heaps; a single `vkAllocateMemory` cannot cover a >4 GiB tensor.
- The loader enumerates **both** the NVIDIA GPU and lavapipe when no
  `VK_ICD_FILENAMES` override is present on this machine. Evidence therefore
  selects the physical device by UUID rather than relying on index 0.
- The 580 driver exposes `VK_KHR_ray_tracing_pipeline` even on Pascal; it is
  not a useful "missing capability" example. `VK_EXT_mesh_shader` is absent on
  the GTX 1080 Ti and is what `logs/missing-extension.log` requests.
- With validation enabled, the first-ever run emits an INFO-severity
  `WARNING-cache-file-error` about creating the shader validation cache. The
  layer creates the cache file and later runs are clean; severity counters are
  unaffected either way (it is INFO, not WARNING).
- `vkCreateInstance` with no ICD returns
  `VK_ERROR_INCOMPATIBLE_DRIVER`; the binary turns that into a hint naming
  `VK_ICD_FILENAMES` and `/usr/share/vulkan/icd.d/` instead of a panic.

## Design notes for FN3-11

See `NOTES-FN3-11.md`.
