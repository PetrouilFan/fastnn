# Vulkan platform layer guide

`src/fn3/backend/vulkan/` is the GPU platform layer: instance and device
lifetime, memory, dispatch, synchronization, staging and execution evidence. It
owns no arithmetic.

## Ownership

- `mod.rs`: `VulkanBackend`, `VulkanConfig`, the `Backend` implementation,
  device discovery, the submitter pool, transfer and dispatch helpers.
- `instance.rs`, `device_info.rs`, `device.rs`: loader, adapter probe, logical
  device, queue, pipeline cache and device teardown.
- `memory.rs`: the suballocator over `vkAllocateMemory` chunks.
- `context.rs`: `Buffer`, checked ranges, bindings, `KernelInvocation`;
  `pipeline.rs`: `Kernel` (shader module, layouts, pipeline) and SPIR-V checks.
- `kernels/mod.rs`: `KernelSpec`, `KernelSet`, grid functions and registration;
  `reflect.rs` parses SPIR-V; `smoke.rs` is `Add`; the pointwise/matmul/norm/
  reduce/attention/quant families register beside it.
- `submission.rs`: per-thread `Submitter`, `Recorder`, `Submission` tokens and
  timestamps. `staging.rs`: host-visible lease pool. `quarantine.rs`: owned
  resources of submissions whose completion is unknown.
- `residency.rs`: per-execution evidence; `ledger.rs`: handle accounting used
  by tests; `shaders/`: GLSL, committed SPIR-V and the offline compile script.

## Invariants

- Loader → instance → device → every buffer, kernel and submitter. A child
  holds `Arc<Device>`; `Device` holds no child handles, so there is no cycle.
- One command pool per thread with `RESET_COMMAND_BUFFER`, checked out
  exclusively through `checkout_submitter`; never share a `Submitter`. The
  `vkQueueSubmit` mutex is the only queue-level critical section.
- Every submission is fence-based and returns a `Submission` token; `wait`
  before reusing a command buffer or descriptor set. `vkQueueWaitIdle` is
  teardown-only.
- A failed recording is discarded (command buffer reset, staging blocks
  dropped), never submitted as a partial plan. A fence wait that fails, times
  out, or is refused on a terminal device leaves completion unknown: every
  in-flight resource (source/destination buffers and their ranges, staging
  blocks, command and descriptor pools, sets, fences) moves into the device
  quarantine and is never dropped, destroyed or recycled. Every committed
  buffer carries a device-wide reference count; an entry is released when its
  fence is proven, but a buffer is freed only when its last reference is gone
  (a shared weight can be bound by two submitters). Resources otherwise leave
  only at teardown after `vkDeviceWaitIdle` succeeds. A failed quiesce leaks
  deliberately, including the loader instance, and records why. `Buffer::drop`
  defers on a terminal device and the staging pool never re-pools a quarantined
  or retired block. The `Device` stays poisoned and later calls return
  `Error::Lost`; reopen the backend.
- `Device::missing_required_features` is checked before `Device::open`: a
  device without the kernels' compute geometry, storage range or allocation
  limit gets a `DeviceUnavailable` naming the adapter and the missing
  capability, not a driver failure at first dispatch.
- `fault.rs` is compiled only with `vulkan-fault-injection`; it is the #73 test
  seam for allocation, submit, fence, recording, readback, resident copies and
  teardown quiesce. Its results are injected evidence, never hardware evidence.
- Descriptor set layouts, pipeline layouts, push-constant ranges and dispatch
  geometry come from one `KernelSpec`; `kernels/reflect.rs` checks the
  declaration against the committed SPIR-V at registration.
- Suballocation is mandatory: one `vkAllocateMemory` per tensor hits the
  `maxMemoryAllocationSize` cap. Chunks are reused; a larger request fails
  naming the size, the limit and the device.
- All transfer and descriptor ranges are checked before a command is recorded.
  Storage offsets honour `minStorageBufferOffsetAlignment`, uniform offsets
  `minUniformBufferOffsetAlignment`, copies are 4-byte multiples.
- Coherence is never assumed: host-visible memory that is not `HOST_COHERENT`
  is flushed/invalidated with `nonCoherentAtomSize` alignment.
- `supports` is honest: `Full`, `Partial` naming the subset, or `Unsupported`
  naming the owning issue. `execute_op` never computes on the host.
- `KernelSet::admits` refines a `Partial` verdict: an `AdmitFn` may lift it to
  `Full` only for operands inside the documented subset, and is never consulted
  for `Full` or `Unsupported`.
- A zero-element tensor owns a buffer of logical size `0` backed by at least one
  aligned word (`vkCreateBuffer` rejects size 0). Range checks use the logical
  size, so a non-empty access is refused and a dispatch must not bind one.
- Every unsafe block carries a `SAFETY:` comment covering handle validity,
  bounds, alignment, aliasing and lifetime.

## Residency

Value slots and the KV cache are device-resident by default (`SessionOptions`;
`FN3_VK_STAGED=1` is the staged diagnostic). Weights upload once in
`prepare_persistent`, a resident step is one batched submission, and a decode
still submits its KV appends and logits download separately with staging held to
the fence. Kernels chain `alloc_buffer`/`upload`/`dispatch`/`download`. The
persistent table is keyed by a storage generation and released with the last
host handle, so a dropped model cannot lend a device copy to a later model.

## Validation

```bash
CARGO_BUILD_JOBS=4 cargo test --features vulkan --test fn3_vulkan_smoke \
  -- --nocapture --test-threads=1
FN3_VK_TEST_SOFTWARE=1 CARGO_BUILD_JOBS=4 cargo test --features vulkan \
  --test fn3_vulkan_smoke software -- --nocapture --test-threads=1
cargo fmt --check
```

Physical-GPU tests skip with `[fn3vk][skip]` when no non-software adapter is
present. `[fn3vk][software-icd]` output is CI-only evidence and never satisfies a
physical gate; `FN3_VK_VALIDATION=1` forces validation on ad-hoc runs.
