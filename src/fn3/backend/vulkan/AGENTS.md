# Vulkan platform layer guide

`src/fn3/backend/vulkan/` is the FN3-11 (#57) GPU platform layer: instance and
device lifetime, memory, dispatch, synchronization, staging and execution
evidence. It owns no arithmetic. Kernel families belong to FN3-13 (#59),
FN3-14 (#60) and FN3-15 (#61); the end-to-end proof belongs to FN3-16 (#62).

## Ownership

- `mod.rs`: `VulkanBackend`, `VulkanConfig`, the `Backend` implementation,
  `available_devices`, the per-thread submitter pool, transfer and dispatch
  helpers.
- `instance.rs`, `device_info.rs`, `device.rs`: loader/instance, adapter probe,
  logical device and queue. `device.rs` also owns the pipeline cache.
- `memory.rs`: the suballocator over `vkAllocateMemory` chunks.
- `context.rs`: `Buffer`, checked ranges, bindings and `KernelInvocation`.
- `pipeline.rs`: `Kernel` (shader module, descriptor set layout, pipeline
  layout, pipeline) and the spec-vs-SPIR-V validation.
- `kernels/mod.rs`: `KernelSpec`, `OpKernel`, `KernelSet`, grid functions and
  the registration point. `kernels/reflect.rs` parses SPIR-V decorations.
- `kernels/smoke.rs`: the FN3-11 `Add` kernel. `kernels/pointwise.rs`,
  `kernels/matmul.rs` and `kernels/{norm,reduce,attention}.rs` belong to
  #59/#60/#61; do not create placeholders for them here.
- `submission.rs`: per-thread `Submitter`, `Recorder`, `Submission` tokens and
  timestamp queries.
- `staging.rs`: host-visible lease pool for upload and readback.
- `residency.rs`: the per-execution evidence ledger #62 consumes.
- `ledger.rs`: create/destroy accounting used by the tests.
- `shaders/`: GLSL sources, committed SPIR-V and the offline compile script.

## Invariants

- Loader → instance → device → every buffer, kernel and submitter. A child
  holds `Arc<Device>`; `Device` holds no child handles, so there is no cycle.
- One command pool per thread, created with `RESET_COMMAND_BUFFER`, checked out
  exclusively through `VulkanBackend::checkout_submitter`. Never share a
  `Submitter` between threads. `vkQueueSubmit` is the only queue-level critical
  section.
- Every submission is fence-based and returns a `Submission` token. `wait`
  before reusing a command buffer or a descriptor set. `vkQueueWaitIdle` is
  teardown-only.
- A batch whose recording failed is discarded (command buffer reset, retained
  staging blocks dropped), never ended and submitted as a partial plan. A
  fence wait that fails leaves the submission's completion unknown, so the
  `Device` is poisoned and every later call returns `Error::Lost`; there is no
  in-place recovery and the backend must be reopened.
- `Device::missing_required_features` (on the adapter) is checked before
  `Device::open`: a device without the kernels' compute geometry, a 32 MiB
  storage range or a 64 MiB allocation gets a `DeviceUnavailable` naming the
  adapter and the missing capability, not a driver failure at first dispatch.
- `fault.rs` is compiled only with `vulkan-fault-injection` and is the #73
  test seam for allocation, submit, fence, readback, recording and resident
  copies. Its results are injected evidence, never hardware evidence.
- Descriptor set layouts, pipeline layouts, push-constant ranges and dispatch
  geometry are derived from one `KernelSpec`; `kernels/reflect.rs` checks the
  declaration against the committed SPIR-V at registration.
- Suballocation is mandatory: one `vkAllocateMemory` per tensor hits the
  ≈4 GiB `maxMemoryAllocationSize` cap. Chunks are reused; requests above the
  cap fail with the requested size, the limit and the device name.
- All transfer and descriptor ranges are checked before a command is recorded.
  Storage-buffer offsets honour `minStorageBufferOffsetAlignment`, uniform
  offsets `minUniformBufferOffsetAlignment`, copies are 4-byte multiples.
- Coherence is never assumed: host-visible memory that is not `HOST_COHERENT`
  is flushed/invalidated with `nonCoherentAtomSize` alignment.
- `supports` is honest: `Full`, `Partial` naming the implemented subset, or
  `Unsupported` naming the owning issue. `execute_op` never computes an op on
  the host and never approximates one.
- `KernelSet::admits` is the concrete refinement of a `Partial` verdict: a
  registration's `AdmitFn` may lift it to `Full` only for operands inside the
  documented subset, and it is never consulted for `Full` or `Unsupported`.
  A rule that disagrees with its launch function can only make a step fail, not
  make an unsupported shape succeed.
- A zero-element tensor owns a buffer whose logical size is `0` and whose
  reservation is at least one aligned word (`vkCreateBuffer` rejects size 0).
  The logical size is what every range check sees, so a non-empty access is
  refused and a dispatch must never bind one: a zero descriptor range is
  rejected before recording.
- Every unsafe block carries a `SAFETY:` comment covering handle validity,
  bounds, alignment, aliasing and lifetime.

## Current limitation

The alpha `Tensor` owns host `AlignedBuffer` storage on every device. Until
#52/#48 adds a device-resident `Storage` variant, `execute_op` uploads inputs,
dispatches on the GPU and downloads outputs, and every residency record reports
`host_staged = true`. The arithmetic is GPU arithmetic; the record's
submissions, device-local bindings, dispatch geometry and timestamps prove it.
Kernels that chain work should use `alloc_buffer`/`upload`/`dispatch`/
`download` instead of a per-op round trip.

## Validation

```bash
CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_vulkan_smoke \
  -- --nocapture --test-threads=1
FN3_VK_TEST_SOFTWARE=1 CARGO_BUILD_JOBS=4 cargo test --features vulkan \
  --test fn3_vulkan_smoke software -- --nocapture --test-threads=1
cargo fmt --check
```

Physical-GPU tests skip with `[fn3vk][skip]` when no non-software adapter is
present. Software-ICD output is prefixed `[fn3vk][software-icd]` and is CI-only
evidence; it never satisfies a physical-device gate. Set `FN3_VK_VALIDATION=1`
to force validation on for ad-hoc runs; the tests enable it explicitly.
