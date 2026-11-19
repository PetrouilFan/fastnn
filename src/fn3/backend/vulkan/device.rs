//! The logical device: compute queue, memory allocator, pipeline cache.
//!
//! # Ownership
//!
//! `Device` is the owning handle for everything below the instance:
//!
//! ```text
//! Arc<Instance>  ── loader + vk::Instance + adapters
//!   └── Device   ── vk::Device + compute queue + MemoryAllocator + VkPipelineCache
//!         ├── Buffer      (Arc<Device>, frees its suballocation on Drop)
//!         ├── Kernel      (Arc<Device>, destroys pipeline/layouts on Drop)
//!         └── Submitter   (Arc<Device>, destroys pool/cmd/fence on Drop)
//! ```
//!
//! Every child holds `Arc<Device>` and therefore cannot outlive it. `Device`
//! itself holds no child handles, so there is no reference cycle; when the last
//! `Arc<Device>` drops, the allocator's chunks, the pipeline cache and the
//! `vk::Device` are destroyed in that order.
//!
//! # Queue policy
//!
//! One queue from a dedicated COMPUTE family is used (family 2 on the GTX
//! 1080 Ti). `vkQueueSubmit` is externally synchronised by a mutex; that is the
//! only serialisation point between worker threads, and it is held for the
//! submit call only, never across a wait. Command pools, command buffers and
//! fences are per-thread (see [`super::submission`]).
//!
//! # Pipeline cache
//!
//! One `VkPipelineCache` lives for the device's lifetime and is passed to every
//! `vkCreateComputePipelines`, so re-registering a kernel after the first
//! dispatch is a cache hit, not a driver compile. [`Device::pipeline_cache_data`]
//! exposes the serialized blob for a caller that wants to persist it keyed by
//! `(deviceUUID, driverUUID)`; the alpha keeps it in memory only.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ash::vk;

use super::context::{Buffer, BufferDesc};
use super::device_info::{AdapterInfo, AdapterLimits};
use super::error::{check, Error, Result};
use super::instance::Instance;
use super::ledger::HandleLedger;
use super::memory::{MemoryAllocation, MemoryAllocator, MemoryClass, MemoryStats};
use crate::inference::device::{DeviceId, PhysicalDeviceId};

/// How a logical device is created.
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Upper bound on one device-local chunk. Clamped by
    /// `maxMemoryAllocationSize`; the default is 512 MiB.
    pub max_chunk_bytes: u64,
    /// Fence wait timeout used by synchronous submissions. A hang detector,
    /// not a synchronisation shortcut.
    pub fence_timeout_ns: u64,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        DeviceConfig {
            max_chunk_bytes: 512 * 1024 * 1024,
            fence_timeout_ns: 60 * 1_000_000_000,
        }
    }
}

/// A logical compute device.
pub struct Device {
    /// Keeps the loader, instance and physical device alive for the device's
    /// whole lifetime. Never read directly except through [`Device::instance`].
    instance: Arc<Instance>,
    ash: ash::Device,
    adapter: AdapterInfo,
    physical: vk::PhysicalDevice,
    queue_family: u32,
    queue: vk::Queue,
    /// Serialises `vkQueueSubmit` (the queue is externally synchronised).
    queue_lock: Mutex<()>,
    /// Monotonic `vkQueueSubmit` count, for the #72 submission accounting.
    queue_submissions: std::sync::atomic::AtomicU64,
    memory: Mutex<MemoryAllocator>,
    pipeline_cache: vk::PipelineCache,
    ledger: HandleLedger,
    device_id: DeviceId,
    next_buffer_id: AtomicU64,
    fence_timeout_ns: u64,
    /// The first terminal failure, if the device became unusable. Every later
    /// operation returns it instead of touching a fence or command buffer whose
    /// state is unknown.
    fatal: Mutex<Option<DeviceFault>>,
    /// Deterministic fault injection (FN3-27), compiled out unless the
    /// `vulkan-fault-injection` feature is on.
    #[cfg(feature = "vulkan-fault-injection")]
    faults: super::fault::FaultInjector,
    destroyed: bool,
}

/// Why the device is terminal, kept for every later call.
#[derive(Debug, Clone)]
struct DeviceFault {
    call: &'static str,
    detail: String,
}

impl Device {
    /// Open a logical device on adapter `adapter_index` of `instance`.
    ///
    /// The adapter is taken from `instance.adapters()`, so the index is the
    /// discovery order of *that* instance. The backend resolves the
    /// process-wide [`PhysicalDeviceId`] to a UUID first and passes the
    /// matching adapter here.
    pub fn open(
        instance: Arc<Instance>,
        adapter_index: u32,
        config: &DeviceConfig,
    ) -> Result<Arc<Device>> {
        let adapter = instance
            .adapters()
            .get(adapter_index as usize)
            .cloned()
            .ok_or_else(|| {
                Error::internal(format!(
                    "physical device index {adapter_index} does not exist on this instance: \
                     {} device(s) were enumerated",
                    instance.adapters().len()
                ))
            })?;
        let physical = instance.physical(adapter_index).ok_or_else(|| {
            Error::internal(format!(
                "physical device {adapter_index} disappeared between enumeration and open"
            ))
        })?;
        let family = adapter.compute_family().ok_or_else(|| {
            Error::internal(format!(
                "adapter \"{}\" exposes no COMPUTE queue family; families: {}",
                adapter.name,
                adapter
                    .queue_families
                    .iter()
                    .map(|family| format!(
                        "[{}]={}",
                        family.index,
                        super::device_info::queue_flags_string(family.flags)
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;
        let family_index = family.index;

        let priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family_index)
            .queue_priorities(&priority)];
        // No optional features are enabled in the alpha platform layer: the
        // smoke kernel needs only core storage buffers. Kernels that need
        // 16-bit/8-bit storage or FP16 arithmetic must add the feature
        // explicitly here and probe it as a hard requirement (see
        // NOTES-FN3-11 §4.8), never silently fall back to f32.
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        // SAFETY: `physical` belongs to the live instance; `priority` and
        // `queue_info` live until `create_device` returns; the family index was
        // validated against the adapter's own properties.
        let ash_device = unsafe { instance.ash().create_device(physical, &device_info, None) }
            .map_err(|code| {
                Error::vk(
                    "vkCreateDevice",
                    code,
                    format!(
                        "creating a logical device for \"{}\" on queue family {family_index} ({})",
                        adapter.name,
                        super::device_info::queue_flags_string(family.flags)
                    ),
                )
            })?;

        // SAFETY: the device is live, the family index was used at creation,
        // and queue index 0 exists (`queue_count >= 1` was checked by
        // `compute_family`).
        let queue = unsafe { ash_device.get_device_queue(family_index, 0) };
        // SAFETY: `physical` is owned by the instance kept alive by `instance`.
        let memory_properties = unsafe {
            instance
                .ash()
                .get_physical_device_memory_properties(physical)
        };

        let cache_info = vk::PipelineCacheCreateInfo::default();
        // SAFETY: plain pipeline-cache creation with no initial data and no
        // allocation callbacks; the cache is used only by this device and is
        // destroyed in `Drop`.
        let pipeline_cache = unsafe { ash_device.create_pipeline_cache(&cache_info, None) }
            .map_err(|code| {
                Error::vk(
                    "vkCreatePipelineCache",
                    code,
                    "creating the device-lifetime pipeline cache",
                )
            })?;

        let memory = MemoryAllocator::new(
            memory_properties,
            adapter.limits.max_memory_allocation_size,
            adapter.limits.max_memory_allocation_count,
            adapter.limits.non_coherent_atom_size,
            config.max_chunk_bytes,
        );

        let ledger = HandleLedger::default();
        ledger.create("device");
        ledger.create("pipeline_cache");

        Ok(Arc::new(Device {
            instance,
            ash: ash_device,
            adapter,
            physical,
            queue_family: family_index,
            queue,
            queue_lock: Mutex::new(()),
            queue_submissions: std::sync::atomic::AtomicU64::new(0),
            memory: Mutex::new(memory),
            pipeline_cache,
            ledger,
            device_id: DeviceId::Vulkan(PhysicalDeviceId::new(adapter_index)),
            next_buffer_id: AtomicU64::new(1),
            fence_timeout_ns: config.fence_timeout_ns,
            fatal: Mutex::new(None),
            #[cfg(feature = "vulkan-fault-injection")]
            faults: super::fault::FaultInjector::new(),
            destroyed: false,
        }))
    }

    /// The injection seam for the FN3-27 failure matrix.
    #[cfg(feature = "vulkan-fault-injection")]
    pub fn faults(&self) -> &super::fault::FaultInjector {
        &self.faults
    }

    /// `true` when the device is terminal and must be recreated.
    pub fn is_lost(&self) -> bool {
        self.fatal
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
    }

    /// Remember the first terminal failure; later ones never overwrite it.
    pub(crate) fn poison(&self, call: &'static str, detail: impl Into<String>) {
        let mut fatal = self
            .fatal
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if fatal.is_none() {
            *fatal = Some(DeviceFault {
                call,
                detail: detail.into(),
            });
        }
    }

    /// The typed terminal error, naming the adapter and the failing call.
    pub fn terminal_error(&self) -> Option<Error> {
        self.fatal
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .map(|fault| {
                Error::lost(
                    format!("{} ({})", self.adapter.name, self.adapter.uuid_string()),
                    format!("{}: {}", fault.call, fault.detail),
                )
            })
    }

    /// Refuse to touch a device that is already terminal.
    pub(crate) fn ensure_usable(&self) -> Result<()> {
        match self.terminal_error() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Build the terminal error for a failure happening right now.
    fn lost_now(&self, call: &'static str, detail: impl Into<String>) -> Error {
        let detail = detail.into();
        self.poison(call, detail.clone());
        Error::lost(
            format!("{} ({})", self.adapter.name, self.adapter.uuid_string()),
            format!("{call}: {detail}"),
        )
    }

    /// The instance this device belongs to.
    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    /// Raw `ash` device.
    pub fn ash(&self) -> &ash::Device {
        &self.ash
    }

    /// The adapter this device was opened on.
    pub fn adapter(&self) -> &AdapterInfo {
        &self.adapter
    }

    /// The raw physical device (for extensions a kernel owner enables).
    pub fn physical(&self) -> vk::PhysicalDevice {
        self.physical
    }

    /// Device limits.
    pub fn limits(&self) -> &AdapterLimits {
        &self.adapter.limits
    }

    /// The compute queue family index.
    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// The device id reported to the 3.0 core.
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }

    /// Handle ledger.
    pub fn ledger(&self) -> &HandleLedger {
        &self.ledger
    }

    /// Fence timeout used by synchronous waits.
    pub fn fence_timeout_ns(&self) -> u64 {
        self.fence_timeout_ns
    }

    /// Allocate a device buffer and bind it to a suballocated range.
    pub fn create_buffer(self: &Arc<Self>, desc: &BufferDesc) -> Result<Buffer> {
        self.ensure_usable()?;
        Buffer::new(self, desc)
    }

    /// The allocator's current counters.
    pub fn memory_stats(&self) -> MemoryStats {
        self.lock_memory().stats()
    }

    /// The serialized `VkPipelineCache` blob (for persistence keyed by
    /// `(deviceUUID, driverUUID)`).
    pub fn pipeline_cache_data(&self) -> Result<Vec<u8>> {
        // SAFETY: the cache is live for the device lifetime.
        unsafe { self.ash.get_pipeline_cache_data(self.pipeline_cache) }.map_err(|code| {
            Error::vk(
                "vkGetPipelineCacheData",
                code,
                "serializing the device pipeline cache",
            )
        })
    }

    /// Submit to the compute queue with the queue-level external
    /// synchronisation Vulkan requires.
    pub(crate) fn submit(&self, submits: &[vk::SubmitInfo<'_>], fence: vk::Fence) -> Result<()> {
        self.ensure_usable()?;
        #[cfg(feature = "vulkan-fault-injection")]
        if self.faults.take(super::fault::FaultPoint::Submit) {
            // Nothing was submitted: the command buffer stays executable and
            // the fence stays unsignalled, so the failure is clean at the
            // driver boundary. It is still terminal because the matrix models
            // a device lost at submit, which is not recoverable in place.
            return Err(self.lost_now(
                "vkQueueSubmit",
                "injected device loss at submit (FN3-27; no command buffer was submitted)",
            ));
        }
        let _guard = self
            .queue_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.queue_submissions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // SAFETY: the command buffers in `submits` are recording-complete and
        // owned by the caller's per-thread submitter; the fence belongs to the
        // same submission; the queue was created for this device and is
        // externally synchronised by `queue_lock`.
        let result = check(
            "vkQueueSubmit",
            unsafe { self.ash.queue_submit(self.queue, submits, fence) },
            "submitting a recorded command buffer",
        );
        match &result {
            Err(error) if error.is_device_lost() => {
                // Report the terminal form immediately, naming the adapter, and
                // remember it so every later call returns the same failure.
                let detail = error.to_string();
                Err(self.lost_now("vkQueueSubmit", detail))
            }
            _ => result,
        }
    }

    /// Wait for a fence. A timeout is a hang detector, not the normal path.
    /// Number of `vkQueueSubmit` calls this device has made.
    ///
    /// This is the authoritative submission counter: it counts the batches,
    /// the per-op recordings, the transfers and every explicit download, so a
    /// report cannot undercount by reading only the op ledger.
    pub fn queue_submissions(&self) -> u64 {
        self.queue_submissions
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub(crate) fn wait_fence(&self, fence: vk::Fence) -> Result<()> {
        self.ensure_usable()?;
        // SAFETY: the fence was created by this device and submitted above; the
        // wait is for that exact fence.
        let waited = check(
            "vkWaitForFences",
            unsafe {
                self.ash
                    .wait_for_fences(&[fence], true, self.fence_timeout_ns)
            },
            "waiting for a submitted command buffer to complete",
        );
        #[cfg(feature = "vulkan-fault-injection")]
        if self.faults.take(super::fault::FaultPoint::Fence) {
            // The real wait above completed, so no work is still in flight.
            // The matrix models a device lost while the fence was being
            // observed: the outcome the caller would see is the same, and the
            // device is terminal.
            return Err(self.lost_now(
                "vkWaitForFences",
                "injected device loss at the fence wait (FN3-27; the submission had completed)",
            ));
        }
        if let Err(error) = waited {
            // A fence that cannot be waited on (device lost, timeout or a
            // driver error) leaves the completion of the submission unknown.
            // Reusing the command buffer or the fence would be undefined, so
            // the device is terminal from here on.
            return Err(self.lost_now("vkWaitForFences", error.to_string()));
        }
        Ok(())
    }

    pub(crate) fn reset_fence(&self, fence: vk::Fence) -> Result<()> {
        self.ensure_usable()?;
        // SAFETY: the fence's submission completed (the caller waited), so it
        // is unsignalled-reset while no work is in flight.
        let result = check(
            "vkResetFences",
            unsafe { self.ash.reset_fences(&[fence]) },
            "reusing a submission fence after its wait",
        );
        if let Err(error) = &result {
            // The fence's signal state is now unknown; the submitter cannot be
            // reused. Report the terminal form immediately.
            let detail = error.to_string();
            return Err(self.lost_now("vkResetFences", detail));
        }
        result
    }

    pub(crate) fn allocate_range(
        &self,
        requirements: vk::MemoryRequirements,
        class: MemoryClass,
        min_alignment: u64,
        what: &str,
    ) -> Result<MemoryAllocation> {
        self.ensure_usable()?;
        #[cfg(feature = "vulkan-fault-injection")]
        if self.faults.take(super::fault::FaultPoint::Allocation) {
            // The fault fires before the allocator chooses a range or chunk,
            // so report the buffer requirement, not a hypothetical chunk size.
            return Err(Error::vk_allocation(
                "buffer memory allocation (injected)",
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                requirements.size,
                format!(
                    "{what}: injected allocation failure (FN3-27) for {} bytes on \"{}\"",
                    requirements.size, self.adapter.name
                ),
            ));
        }
        self.lock_memory()
            .allocate(&self.ash, requirements, class, min_alignment, what)
    }

    pub(crate) fn free_range(&self, allocation: MemoryAllocation) -> Result<()> {
        self.lock_memory().free(allocation)
    }

    pub(crate) fn allocation_pointer(&self, allocation: &MemoryAllocation) -> Option<*mut u8> {
        self.lock_memory().pointer(allocation)
    }

    pub(crate) fn allocation_is_coherent(&self, allocation: &MemoryAllocation) -> bool {
        self.lock_memory().is_coherent(allocation)
    }

    /// The `vk::DeviceMemory` chunk backing an allocation.
    pub(crate) fn chunk_memory(&self, allocation: &MemoryAllocation) -> Option<vk::DeviceMemory> {
        self.lock_memory().chunk_memory(allocation)
    }

    pub(crate) fn flush_allocation(
        &self,
        allocation: &MemoryAllocation,
        offset: u64,
        len: u64,
    ) -> Result<()> {
        self.lock_memory().flush(&self.ash, allocation, offset, len)
    }

    pub(crate) fn invalidate_allocation(
        &self,
        allocation: &MemoryAllocation,
        offset: u64,
        len: u64,
    ) -> Result<()> {
        self.lock_memory()
            .invalidate(&self.ash, allocation, offset, len)
    }

    /// Next buffer id, used by residency records to identify a binding.
    pub(crate) fn next_buffer_id(&self) -> u64 {
        self.next_buffer_id.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn pipeline_cache_handle(&self) -> vk::PipelineCache {
        self.pipeline_cache
    }

    fn lock_memory(&self) -> MutexGuard<'_, MemoryAllocator> {
        self.memory
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }
        // Teardown-only global wait. Every measured submission is fence-based;
        // this is the final "nothing in flight" guard before the allocator's
        // chunks and the device disappear.
        // SAFETY: no other thread can be using the device: every resource and
        // submitter holds `Arc<Device>`, so this `Drop` runs only when the last
        // handle is gone.
        if let Err(code) = unsafe { self.ash.device_wait_idle() } {
            eprintln!(
                "[fn3vk][warn] vkDeviceWaitIdle during teardown failed: {code:?} ({})",
                code.as_raw()
            );
        }
        {
            let mut memory = self
                .memory
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let stats = memory.stats();
            if stats.live_bytes != 0 {
                eprintln!(
                    "[fn3vk][warn] tearing down the device with {} live suballocated bytes \
                     (a Buffer outlived its Arc<Device>, which is impossible by construction)",
                    stats.live_bytes
                );
            }
            memory.destroy(&self.ash);
        }
        // SAFETY: the cache is live and unused; destroyed exactly once, before
        // the device.
        unsafe { self.ash.destroy_pipeline_cache(self.pipeline_cache, None) };
        self.ledger.destroy("pipeline_cache");
        // SAFETY: every child object (buffers, kernels, submitters) is gone
        // because it holds `Arc<Device>`; the device is destroyed exactly once.
        unsafe { self.ash.destroy_device(None) };
        self.ledger.destroy("device");
        self.destroyed = true;
    }
}

impl std::fmt::Debug for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("adapter", &self.adapter.name)
            .field("queue_family", &self.queue_family)
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}
