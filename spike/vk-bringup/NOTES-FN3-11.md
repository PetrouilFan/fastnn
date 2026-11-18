# FN3-11 integration note — Vulkan platform layer

This is the hand-off for #57. It describes which parts of `spike/vk-bringup`
are reusable, what the module boundary and API should be, and what changes at
production scale. The spike stays the executable reference; FN3-11 owns the
real module under `src/fn3/backend/vulkan/` behind a `vulkan` feature.

## 1. What to promote

| Spike file | Production module | Keep / change |
| --- | --- | --- |
| `src/platform.rs` | `instance.rs` | Keep the loader/instance lifetime and validation plumbing. Add a process-wide singleton or explicit ownership from the backend registry. |
| `src/probe.rs` | `device_info.rs` | Keep as a diagnostic/`--list` helper. `DeviceInfo` is also the capability gate before device creation. |
| `src/device.rs` | `device.rs` | Keep queue-family policy and `find_memory_type`. Replace the per-buffer `vkAllocateMemory` with a suballocator (see §4). |
| `src/compute.rs` (buffer/staging/descriptor/submit halves) | `context.rs`, `staging.rs`, `submission.rs` | Split. The FMA workload itself is throwaway; the resource and submission ownership patterns are the deliverable. |
| `src/ledger.rs` | `debug/ledger.rs` (feature-gated) | Keep behind `#[cfg(debug_assertions)]` or a `vulkan-debug` feature; it caught nothing in this spike but makes teardown regressions loud. |
| `src/error.rs` | `error.rs` | Keep the "call name + `VkResult` + detail + hint" shape. Add `thiserror` if the crate already uses it. |
| `shaders/fma.comp` + `.spv` | `shaders/` + build script | Keep the offline-compile + committed-SPIR-V rule for every kernel. |

Do **not** port the spike's single-workload `Engine` as the backend. Its value
is the ordering and ownership rules it demonstrates.

## 2. Suggested API shape

```rust
// Instance and device are explicit owners; no global OnceLock in the platform
// layer. The runtime may still cache one `Arc<Device>` at a higher level.
pub struct Instance { /* Entry, vk::Instance, optional messenger */ }
impl Instance {
    pub fn new(validation: bool) -> Result<Self>;
    pub fn enumerate(&self) -> Result<Vec<DeviceInfo>>;
    pub fn open(self: &Arc<Self>, index: usize) -> Result<Arc<Device>>;
}

pub struct Device {
    instance: Arc<Instance>,          // keeps loader + instance alive
    device: ash::Device,
    queue_family: u32,
    queue: vk::Queue,
    memory: MemoryAllocator,
    pipelines: PipelineCache,
}

impl Device {
    pub fn create_buffer(&self, desc: BufferDesc) -> Result<Buffer>;
    pub fn upload<'a>(&self, dst: &Buffer, src: &'a [u8]) -> Result<UploadToken<'a>>;
    pub fn download<'a>(&self, src: &Buffer, dst: &'a mut [u8]) -> Result<DownloadToken<'a>>;
    pub fn create_pipeline(&self, spv: &[u8], layout: &KernelLayout) -> Result<Kernel>;
    pub fn submit(&self, batch: &Batch) -> Result<FenceToken>;
    pub fn wait(&self, token: FenceToken) -> Result<()>;
}
```

Key rules the production version must keep:

- **Instance outlives device; device outlives every buffer, pipeline, pool and
  command buffer.** In Rust, encode this with an `Arc<Instance>` field in
  `Device` (as above) and with `Device`-owned resource structs. The spike uses
  a borrow (`DeviceContext<'a>`) only because it is one binary.
- **One command pool per thread**, created with `RESET_COMMAND_BUFFER`.
  `CommandPool` is not `Send`/`Sync`; do not share it. A worker owns a
  `Submitter { pool, cmd, fence }`, resets its own command buffer and fence
  after each wait, and submits to the shared queue. If FastNN uses rayon,
  create pools lazily per worker thread and drop them at thread exit.
- **Every submission returns a fence/timeline token.** No fire-and-forget
  `vkQueueSubmit` in the hot path and no `vkQueueWaitIdle`. The spike proves
  one-fence reuse; production should use a small ring of fences or one timeline
  semaphore.
- **Staging is explicit.** `upload`/`download` own a host-visible buffer and
  return a token that must be waited on before the borrowed slice is reused.
  This is what prevents the classic "unmap/reuse while the copy is in flight"
  bug. The spike's single mapped readback buffer is a simplification.
- **Pipeline layout is derived from the shader**, not hand-written. Add
  `spirv-reflect` (or read the committed SPIR-V) to derive bindings and push
  constant ranges. The spike hard-codes four storage buffers and an 8-byte push
  constant; a mismatch there is a validation error at pipeline creation.
- **Descriptor ownership:** one descriptor set per kernel invocation state, or
  push descriptors / descriptor buffers for the hot path. The spike reuses one
  set because its bindings never change. Production kernels bind different
  buffers every call; plan for either a descriptor-set cache keyed by
  (pipeline, buffer ids) or `VK_EXT_descriptor_buffer` (present on this GPU).
- **Validation:** keep `--validation`/`FN3_VK_VALIDATION` and the
  `[fn3vk][validation][ERROR|WARN]` prefix. In CI, fail the job if the error or
  warning count is non-zero; the spike already exits 1 on validation errors.

## 3. Shader build

- Keep `scripts/compile-shaders.sh` semantics: one committed `.spv` per
  `.comp`, `spirv-val` in the script, `.spvasm` for review.
- Add `--target-env=vulkan1.3` once the backend's minimum API is decided; the
  spike targets 1.1 so it runs on older devices and needs no 1.3 features.
- Do not add `shaderc` as a build dependency. If runtime shader generation is
  ever needed, make it an optional feature with a documented fallback to the
  committed artefacts.

## 4. What changes at production scale

1. **Allocator.** The GTX 1080 Ti reports `maxMemoryAllocationSize` of
   ≈4 GiB even though the device-local heap is 11 GiB. `vkAllocateMemory` per
   tensor will fail for large weights and burn `maxMemoryAllocationCount`.
   Introduce a VMA-style suballocator: one large `vkAllocateMemory` per
   (heap, memory type) block, then `vkBindBufferMemory` at aligned offsets, with
   free-list/buddy allocation and defragmentation. Budget against
   `VK_EXT_memory_budget` (present) instead of guessing.
2. **Descriptors.** Four storage buffers per set is fine for a spike. Real
   graphs need many bindings and frequent rebinding. Prefer
   `VK_EXT_descriptor_buffer` (present) or a descriptor-set cache; update
   descriptors only for the set being submitted, never for one in flight.
   Respect `minStorageBufferOffsetAlignment` = 16 B and
   `minUniformBufferOffsetAlignment` = 256 B; group uniform binds accordingly.
3. **Synchronisation.** `vkCmdPipelineBarrier` (1.0) works everywhere. On
   Vulkan 1.3+ use `VK_KHR_synchronization2` (present) with
   `vkCmdPipelineBarrier2` and per-stage access masks; it makes the
   transfer→compute→transfer chain in `compute.rs` easier to audit. Batch many
   dispatches into one submission and wait once per graph, not per kernel.
4. **Submission.** One fence per dispatch is fine for 100 iterations; it is not
   fine for thousands of small kernels. Use one timeline semaphore with
   monotonically increasing values, or a ring of fences, and let the scheduler
   keep 2–4 submissions in flight. Keep the "reset only after wait" rule.
5. **Pipelines.** Persist `VkPipelineCache` to disk (keyed by driver/device
   UUID) and compile pipelines at load, not per dispatch. Use specialization
   constants for local size / tile shape rather than recompiling GLSL.
6. **Memory coherence.** The spike uses `HOST_COHERENT` staging and never
   flushes. Check the selected memory type's flags in production: if it is not
   coherent, call `vkFlushMappedMemoryRanges` after writes and
   `vkInvalidateMappedMemoryRanges` before reads. Do not assume COHERENT.
7. **Device loss.** A long-running training job must handle
   `VK_ERROR_DEVICE_LOST`: mark the device poisoned, fail the current graph, and
   let the runtime recreate the device. The spike treats it as a terminal error.
8. **Integers and dtypes.** This spike only uses f32. FP16/BF16/INT8 kernels
   need `VK_KHR_16bit_storage` / `VK_KHR_8bit_storage` /
   `VK_KHR_shader_float16_int8` (all present) and `storageBuffer16BitAccess`
   features enabled at device creation; add those to `DeviceCreateInfo`
   explicitly and probe them as hard requirements. Do not silently fall back to
   FP32, per the repository's quantized-execution rule.
9. **Multi-GPU.** `--list` already distinguishes devices by UUID. The backend
   should key caches by `(deviceUUID, driverUUID)` and never assume index 0 is
   the discrete GPU: on this machine lavapipe is enumerated too, and the loader
   can reorder devices.
10. **Validation cost.** Keep a release path with validation compiled out and a
    CI lane with it on. The debug messenger and layer add measurable overhead
    to tiny dispatches (the 100-dispatch loop is dominated by submission, not
    compute), which is expected.

## 5. Things the spike deliberately does not solve

- No multi-threaded submission, no queue ring, no timeline semaphores.
- No memory suballocation or defragmentation.
- No SPIR-V reflection; bindings and push-constant layout are hard-coded.
- No timestamps/queries; the reported wall time is host-side only.
- No surface/swapchain (headless compute only), no graphics queue use.
- No device-loss recovery.

Each of these is a production concern for FN3-11 or a later issue, not a
correctness gap in the bring-up.
