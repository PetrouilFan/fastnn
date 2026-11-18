//! The Vulkan compute backend: memory, dispatch, synchronization and evidence.
//!
//! # What this module owns (FN3-11 / #57)
//!
//! ```text
//! instance.rs      loader, vk::Instance, validation layer, messenger
//! device_info.rs   adapter probe: UUIDs, queues, limits, memory heaps
//! device.rs        logical device, compute queue, pipeline cache
//! memory.rs        suballocator over vkAllocateMemory chunks
//! context.rs       Buffer, checked ranges, KernelInvocation
//! pipeline.rs      Kernel: shader module + descriptor/pipeline layouts
//! kernels/         registration seam, SPIR-V reflection, smoke kernel
//! submission.rs    per-thread submitters, recording, fences, timestamps
//! staging.rs       host-visible lease pool for upload/readback
//! residency.rs     per-execution evidence (#62 consumes this)
//! ```
//!
//! # Public surface for kernel owners (#59/#60/#61) and #62
//!
//! ```ignore
//! let backend = VulkanBackend::open(&DeviceRequest::Vulkan { index: None })?;
//! let kernel  = backend.kernel(&MY_SPEC)?;             // validates spec vs SPIR-V
//! let buffer  = backend.alloc_buffer(&BufferDesc::device_storage(bytes, "x"))?;
//! backend.upload(&buffer, bytes)?;                     // device-resident path
//! backend.download(&buffer, &mut host_bytes)?;
//! backend.dispatch(&KernelInvocation { kernel: &kernel, bindings, push_constants, workgroups })?;
//! backend.residency().records();                       // evidence for #62
//! ```
//!
//! # No silent CPU fallback
//!
//! `supports` is the registered kernel table's honest answer: `Full` only when
//! a kernel claims the whole contract, `Partial` with a named subset otherwise,
//! and `Unsupported` naming the owning issue for everything else.
//! `execute_op` never computes an op on the host. When a kernel rejects a
//! shape, the error says so.
//!
//! # Host-staged tensors (alpha limitation, reported to #48/#52)
//!
//! The 3.0 alpha `Tensor` owns host `AlignedBuffer` storage on every device; a
//! device-resident `Storage` variant is owned by #52/#48. Until it lands,
//! `execute_op` uploads the input window, barrier, dispatches on the GPU,
//! barriers and downloads the output. Every execution record sets
//! `host_staged = true` and reports the byte counts, so the transfer is visible
//! rather than hidden, and `inputs_read_back` stays false. The arithmetic is
//! GPU arithmetic; `dispatch` plus device timestamps prove it. The
//! device-resident path (`alloc_buffer`/`upload`/`dispatch`/`download`) is
//! already available for #59/#60/#61 and #62 to avoid per-op round trips.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use ash::vk;

use super::{Backend, Capabilities, CapabilityFeatures, OpParams, Support};
use crate::fn3::device::{DeviceId, DeviceRequest, DeviceResolveError, PhysicalDeviceId};
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::tensor::{TensorMut, TensorView};

pub mod context;
pub mod device;
pub mod device_info;
pub mod error;
pub mod instance;
pub mod kernels;
pub mod ledger;
pub mod memory;
pub mod pipeline;
pub mod residency;
pub mod staging;
pub mod submission;

pub use context::{BindingArg, BindingRole, Buffer, BufferDesc, KernelInvocation};
pub use device::{Device, DeviceConfig};
pub use device_info::{AdapterInfo, AdapterLimits, HeapInfo, QueueFamilyInfo};
pub use error::{Error, Result};
pub use kernels::{BindingKind, BindingSpec, KernelSet, KernelSpec, OpKernel};
pub use ledger::HandleLedger;
pub use memory::{MemoryClass, MemoryStats};
pub use pipeline::Kernel;
pub use residency::{ExecutionRecord, ResidencyLedger, ResidencySummary};
pub use staging::{StagingPool, StagingStats};
pub use submission::{BindingSnapshot, DispatchInfo, Submission, SubmitReport, Submitter};

/// Backend name used in every diagnostic.
pub const VULKAN_BACKEND_NAME: &str = "vulkan";

/// The largest tensor rank the alpha contract accepts (matches the CPU backend).
const MAX_SHAPE_RANK: u8 = 8;

/// How a Vulkan backend is opened.
#[derive(Debug, Clone)]
pub struct VulkanConfig {
    /// Enable `VK_LAYER_KHRONOS_validation`. Defaults to `FN3_VK_VALIDATION=1`.
    pub validation: bool,
    /// Allow opening a software adapter (lavapipe/llvmpipe). Software runs are
    /// evidence-labelled separately and must never satisfy a physical-GPU gate.
    pub allow_software: bool,
    /// Device tuning (chunk cap, fence timeout).
    pub device: DeviceConfig,
    /// Preferred staging block size.
    pub staging_block_bytes: u64,
}

impl Default for VulkanConfig {
    fn default() -> Self {
        VulkanConfig {
            validation: instance::validation_from_env(),
            allow_software: matches!(
                std::env::var("FN3_VK_ALLOW_SOFTWARE").as_deref(),
                Ok("1") | Ok("true") | Ok("yes") | Ok("on")
            ),
            device: DeviceConfig::default(),
            staging_block_bytes: 4 * 1024 * 1024,
        }
    }
}

impl VulkanConfig {
    /// Validation on, software off — what the physical-GPU test uses.
    pub fn validated() -> VulkanConfig {
        VulkanConfig {
            validation: true,
            ..VulkanConfig::default()
        }
    }

    /// Enable validation.
    pub fn with_validation(mut self, validation: bool) -> VulkanConfig {
        self.validation = validation;
        self
    }

    /// Allow a software adapter explicitly.
    pub fn with_software(mut self, allow: bool) -> VulkanConfig {
        self.allow_software = allow;
        self
    }
}

/// Devices a `vulkan`-feature build can execute on.
///
/// Indices come from the process-wide probe instance
/// ([`instance::shared`]) and are stable for the life of the process. A build
/// without an ICD returns an empty list; it never invents a device.
///
/// Software adapters (lavapipe/llvmpipe) are listed because they are real
/// Vulkan devices on this machine, but [`VulkanBackend::open_with`] refuses
/// them unless [`VulkanConfig::allow_software`] is set: a software run must be
/// labelled as CI-only evidence, never counted as a GPU result.
pub fn available_devices() -> Vec<crate::fn3::device::DeviceInfo> {
    if instance::shared().is_err() {
        instance::report_loader_failure();
    }
    instance::shared_adapters()
        .iter()
        .filter(|adapter| adapter.has_compute_queue())
        .map(|adapter| {
            crate::fn3::device::DeviceInfo::new(
                DeviceId::Vulkan(PhysicalDeviceId::new(adapter.index)),
                adapter.name.clone(),
            )
        })
        .collect()
}

/// Probe adapters, including ones without a COMPUTE queue (diagnostics only).
pub fn adapters() -> &'static [AdapterInfo] {
    instance::shared_adapters()
}

/// A Vulkan compute backend bound to one physical device.
pub struct VulkanBackend {
    config: VulkanConfig,
    capabilities: Capabilities,
    kernels: KernelSet,
    /// Strong cache: one compiled pipeline per registered spec, built at open
    /// time so a dispatch never compiles a shader (NOTES-FN3-11 §4.5).
    kernel_cache: Mutex<HashMap<&'static str, Arc<Kernel>>>,
    staging: StagingPool,
    submitters: Mutex<SubmitterPool>,
    residency: ResidencyLedger,
    /// Declared last so every resource above drops before the device.
    device: Arc<Device>,
}

#[derive(Default)]
struct SubmitterPool {
    by_thread: HashMap<ThreadId, Submitter>,
    active: HashSet<ThreadId>,
}

impl VulkanBackend {
    /// Open the device named by `request`, with configuration from the
    /// environment.
    pub fn open(request: &DeviceRequest) -> Fn3Result<Arc<VulkanBackend>> {
        VulkanBackend::open_with(request, VulkanConfig::default())
    }

    /// Open the device named by `request` with explicit configuration.
    ///
    /// An explicit request either opens exactly that physical device or fails
    /// with an error naming the request and listing the available devices.
    /// There is no CPU fallback.
    pub fn open_with(
        request: &DeviceRequest,
        config: VulkanConfig,
    ) -> Fn3Result<Arc<VulkanBackend>> {
        let available = crate::fn3::device::available_devices();
        if !available
            .iter()
            .any(|info| info.id.vulkan_index().is_some())
        {
            return Err(Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                request.to_string(),
                &available,
                no_vulkan_device_reason(),
            )));
        }
        // This constructor is explicitly the Vulkan backend, so `Auto` means
        // "the first Vulkan adapter" here. It still never falls back to the
        // CPU: with no Vulkan device the resolution below fails and lists what
        // is available.
        let effective = match request {
            DeviceRequest::Auto => DeviceRequest::Vulkan { index: None },
            other => other.clone(),
        };
        let device_id = effective
            .resolve(&available)
            .map_err(Fn3Error::DeviceUnavailable)?;
        let index = device_id.vulkan_index().ok_or_else(|| {
            Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                request.to_string(),
                &available,
                "the Vulkan backend cannot open the CPU device; request a Vulkan device",
            ))
        })?;

        let probe = instance::shared().map_err(|error| {
            Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                request.to_string(),
                &available,
                format!("the Vulkan loader or an ICD is unavailable: {error}"),
            ))
        })?;
        let probed_adapter = probe
            .adapters()
            .get(index as usize)
            .cloned()
            .ok_or_else(|| {
                Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                    request.to_string(),
                    &available,
                    format!("physical device index {index} is no longer enumerated"),
                ))
            })?;
        if probed_adapter.is_software() && !config.allow_software {
            return Err(Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                request.to_string(),
                &available,
                format!(
                    "adapter \"{}\" ({}) is a software implementation; software-ICD runs are \
                     CI-only evidence and must be allowed explicitly \
                     (VulkanConfig::with_software(true) or FN3_VK_ALLOW_SOFTWARE=1)",
                    probed_adapter.name,
                    probed_adapter.uuid_string()
                ),
            )));
        }

        let instance = instance::Instance::new(&instance::InstanceConfig {
            validation: config.validation,
            application_name: "fastnn-fn3-vulkan",
        })
        .map_err(|error| {
            Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                request.to_string(),
                &available,
                format!("creating a Vulkan instance failed: {error}"),
            ))
        })?;
        let adapter = instance
            .find_by_uuid(&probed_adapter.device_uuid)
            .cloned()
            .ok_or_else(|| {
                Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                    request.to_string(),
                    &available,
                    format!(
                        "adapter UUID {} was enumerated by the probe instance but not by the \
                         backend instance",
                        probed_adapter.uuid_string()
                    ),
                ))
            })?;
        let physical_index = adapter.index;
        let device = Device::open(Arc::clone(&instance), physical_index, &config.device).map_err(
            |error| {
                Fn3Error::DeviceUnavailable(DeviceResolveError::new(
                    request.to_string(),
                    &available,
                    format!("opening \"{}\" failed: {error}", adapter.name),
                ))
            },
        )?;

        let kernels = KernelSet::alpha();
        let capabilities = capabilities_for(&adapter, &kernels);
        let mut kernel_cache = HashMap::new();
        for spec in kernels.specs() {
            // Compile at load: a bad declaration fails `open`, not the first
            // dispatch, and the pipeline cache is warm for every later call.
            let kernel = pipeline::create_kernel(&device, spec).map_err(|error| error.to_fn3())?;
            kernel_cache.insert(spec.name, kernel);
        }
        let staging = StagingPool::new(Arc::clone(&device), config.staging_block_bytes);
        eprintln!(
            "[fn3vk] backend=vulkan adapter=\"{}\" type={:?} uuid={} api={} driver={} \
             queue_family={} validation={} software={} maxMemoryAllocationSize={} \
             maxStorageBufferRange={} chunk_cap={}",
            adapter.name,
            adapter.device_type,
            adapter.uuid_string(),
            device_info::format_version(adapter.api_version),
            adapter.driver_version_string(),
            device.queue_family(),
            instance.validation(),
            adapter.is_software(),
            device.limits().max_memory_allocation_size,
            device.limits().max_storage_buffer_range,
            config.device.max_chunk_bytes,
        );

        Ok(Arc::new(VulkanBackend {
            config,
            capabilities,
            kernels,
            kernel_cache: Mutex::new(kernel_cache),
            staging,
            submitters: Mutex::new(SubmitterPool::default()),
            residency: ResidencyLedger::new(),
            device,
        }))
    }

    /// The logical device handle.
    ///
    /// Named distinctly from [`Backend::device`], which returns the
    /// [`DeviceId`]: `backend.vulkan_device()` is the handle,
    /// `Backend::device(&*backend)` is the identifier.
    pub fn vulkan_device(&self) -> &Arc<Device> {
        &self.device
    }

    /// The adapter this backend runs on.
    pub fn adapter(&self) -> &AdapterInfo {
        self.device.adapter()
    }

    /// The 3.0 device id.
    pub fn device_id(&self) -> DeviceId {
        self.device.device_id()
    }

    /// The configuration this backend was opened with.
    pub fn config(&self) -> &VulkanConfig {
        &self.config
    }

    /// `true` when the adapter is a software implementation. Every artifact
    /// produced on one must be labelled as software evidence.
    pub fn is_software(&self) -> bool {
        self.adapter().is_software()
    }

    /// `true` when the validation layer is active.
    pub fn validation_enabled(&self) -> bool {
        self.device.instance().validation()
    }

    /// Registered kernels.
    pub fn kernels(&self) -> &KernelSet {
        &self.kernels
    }

    /// A compiled kernel. Registered kernels are compiled when the backend
    /// opens; an ad-hoc spec is compiled (and retained) on first use.
    pub fn kernel(&self, spec: &'static KernelSpec) -> Fn3Result<Arc<Kernel>> {
        {
            let cache = self
                .kernel_cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(kernel) = cache.get(spec.name) {
                return Ok(Arc::clone(kernel));
            }
        }
        let kernel = pipeline::create_kernel(&self.device, spec).map_err(|error| error.to_fn3())?;
        let mut cache = self
            .kernel_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        cache.insert(spec.name, Arc::clone(&kernel));
        Ok(kernel)
    }

    /// Allocate a buffer through the suballocator.
    pub fn alloc_buffer(&self, desc: &BufferDesc) -> Fn3Result<Buffer> {
        self.device
            .create_buffer(desc)
            .map_err(|error| error.to_fn3())
    }

    /// Host-visible staging pool.
    pub fn staging(&self) -> &StagingPool {
        &self.staging
    }

    /// Check out this thread's submitter. The guard returns it to the pool on
    /// drop; nested checkouts on one thread are rejected.
    pub fn checkout_submitter(&self) -> Fn3Result<SubmitterGuard<'_>> {
        let thread = std::thread::current().id();
        {
            let mut pool = self
                .submitters
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !pool.active.insert(thread) {
                return Err(Error::internal(
                    "nested `checkout_submitter` on one thread: record the whole batch with one \
                     submitter instead of checking out a second command pool",
                )
                .to_fn3());
            }
            if let Some(submitter) = pool.by_thread.remove(&thread) {
                return Ok(SubmitterGuard {
                    backend: self,
                    thread,
                    submitter: Some(submitter),
                });
            }
        }
        match Submitter::new(&self.device) {
            Ok(submitter) => Ok(SubmitterGuard {
                backend: self,
                thread,
                submitter: Some(submitter),
            }),
            Err(error) => {
                let mut pool = self
                    .submitters
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                pool.active.remove(&thread);
                Err(error.to_fn3())
            }
        }
    }

    /// Upload host bytes into a device buffer and make them visible to compute.
    ///
    /// `bytes.len()` must be a multiple of 4 (a `vkCmdCopyBuffer` requirement)
    /// and no larger than the destination buffer. One submission: staging write
    /// → copy → transfer→compute barrier → fence wait. The barrier is in this
    /// submission, so a later dispatch needs no extra dependency.
    pub fn upload(&self, dst: &Buffer, bytes: &[u8]) -> Fn3Result<TransferReport> {
        if bytes.len() as u64 > dst.size() {
            return Err(Fn3Error::CapacityExceeded {
                what: "vulkan upload",
                requested: bytes.len() as u64,
                capacity: dst.size(),
            });
        }
        if bytes.is_empty() {
            return Ok(TransferReport::default());
        }
        let lease = self
            .staging
            .acquire(bytes.len() as u64)
            .map_err(|e| e.to_fn3())?;
        lease.write(bytes).map_err(|e| e.to_fn3())?;
        let mut submitter = self.checkout_submitter()?;
        let report = {
            let staging = lease.buffer();
            submitter
                .record(|recorder| {
                    // Order a previous compute read/write of `dst` before the
                    // transfer write (write-after-read/write hazard).
                    recorder.barrier(
                        dst,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        vk::AccessFlags::TRANSFER_WRITE,
                    );
                    recorder.copy_buffer(staging, 0, dst, 0, bytes.len() as u64)?;
                    recorder.barrier(
                        dst,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::SHADER_READ,
                    );
                    Ok(())
                })
                .map_err(|e| e.to_fn3())?
                .wait()
                .map_err(|e| e.to_fn3())?
        };
        Ok(TransferReport {
            bytes: bytes.len() as u64,
            submissions: report.submissions,
            host_ns: report.host_ns,
            gpu_ns: report.gpu_ns,
        })
    }

    /// Download device bytes into host memory.
    ///
    /// `dst.len()` must be a multiple of 4 and no larger than the source
    /// buffer. One submission: compute→transfer barrier → copy → fence wait.
    /// The caller must have waited for the producing submission (the backend's
    /// dispatch helpers do).
    pub fn download(&self, src: &Buffer, dst: &mut [u8]) -> Fn3Result<TransferReport> {
        if dst.len() as u64 > src.size() {
            return Err(Fn3Error::CapacityExceeded {
                what: "vulkan download",
                requested: dst.len() as u64,
                capacity: src.size(),
            });
        }
        if dst.is_empty() {
            return Ok(TransferReport::default());
        }
        let lease = self
            .staging
            .acquire(dst.len() as u64)
            .map_err(|e| e.to_fn3())?;
        let mut submitter = self.checkout_submitter()?;
        let report = {
            let staging = lease.buffer();
            submitter
                .record(|recorder| {
                    // Order both a previous kernel write and a previous
                    // transfer write to `src` before this transfer read.
                    recorder.barrier(
                        src,
                        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );
                    recorder.copy_buffer(src, 0, staging, 0, dst.len() as u64)?;
                    Ok(())
                })
                .map_err(|e| e.to_fn3())?
                .wait()
                .map_err(|e| e.to_fn3())?
        };
        drop(submitter);
        lease.read_into(dst).map_err(|e| e.to_fn3())?;
        Ok(TransferReport {
            bytes: dst.len() as u64,
            submissions: report.submissions,
            host_ns: report.host_ns,
            gpu_ns: report.gpu_ns,
        })
    }

    /// Dispatch one kernel over already-resident buffers.
    ///
    /// This is the path kernel owners use for chained, device-resident work;
    /// it does not append a residency record (the op owner knows the op id and
    /// the binding roles).
    pub fn dispatch(&self, invocation: &KernelInvocation<'_>) -> Fn3Result<DispatchOutcome> {
        let mut submitter = self.checkout_submitter()?;
        let mut info = None;
        let report = submitter
            .record(|recorder| {
                info = Some(recorder.dispatch(invocation)?);
                Ok(())
            })
            .map_err(|error| error.to_fn3())?
            .wait()
            .map_err(|error| error.to_fn3())?;
        let info = info.ok_or_else(|| Fn3Error::InvalidShape {
            context: "vulkan dispatch",
            detail: "the recorder produced no dispatch information".to_string(),
        })?;
        Ok(DispatchOutcome {
            kernel: invocation.kernel.name(),
            submissions: report.submissions,
            workgroups: info.workgroups,
            bindings: info.bindings,
            host_ns: report.host_ns,
            gpu_ns: report.gpu_ns,
        })
    }

    /// Append an execution record. Kernel owners call this after a successful
    /// launch so #62 can prove the op ran on the device.
    pub fn record_execution(&self, record: ExecutionRecord) -> u64 {
        self.residency.record(record)
    }

    /// The residency ledger.
    pub fn residency(&self) -> &ResidencyLedger {
        &self.residency
    }

    /// Allocator counters (chunk sizes, live/peak bytes, allocation count).
    pub fn memory_stats(&self) -> MemoryStats {
        self.device.memory_stats()
    }

    /// Staging counters.
    pub fn staging_stats(&self) -> StagingStats {
        self.staging.stats()
    }

    /// Handle ledger report (every Vulkan object created by this backend).
    pub fn handle_report(&self) -> String {
        self.device.ledger().report()
    }

    /// Kinds with a non-zero created-minus-destroyed count.
    pub fn outstanding_handles(&self) -> Vec<String> {
        self.device.ledger().outstanding()
    }

    /// Serialized pipeline cache, keyed by `(deviceUUID, driverUUID)` by the
    /// caller. Empty before the first pipeline is created.
    pub fn pipeline_cache_data(&self) -> Fn3Result<Vec<u8>> {
        self.device
            .pipeline_cache_data()
            .map_err(|error| error.to_fn3())
    }
}

impl Backend for VulkanBackend {
    fn name(&self) -> &'static str {
        VULKAN_BACKEND_NAME
    }

    fn device(&self) -> DeviceId {
        self.device.device_id()
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn supports(&self, op: OpId) -> Support {
        self.kernels.supports(op)
    }

    fn admits(
        &self,
        op: OpId,
        shapes: &[crate::fn3::shape::Shape],
        dtypes: &[crate::fn3::dtype::DType],
        params: &OpParams,
    ) -> Support {
        self.kernels.admits(op, shapes, dtypes, params)
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Fn3Result<()> {
        self.preflight(op)?;
        for view in inputs {
            view.meta().require_device(Some(op), self.device_id())?;
        }
        for output in outputs.iter() {
            output.meta().require_device(Some(op), self.device_id())?;
        }
        let kernel = self
            .kernels
            .get(op)
            .ok_or_else(|| Fn3Error::UnsupportedOp {
                op,
                backend: VULKAN_BACKEND_NAME,
                reason: format!(
                    "no Vulkan kernel is registered for {}; owned by {}",
                    op.name(),
                    device_info::op_owner(op)
                ),
            })?;
        (kernel.launch)(self, inputs, outputs, params)
    }
}

impl std::fmt::Debug for VulkanBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBackend")
            .field("adapter", &self.adapter().name)
            .field("uuid", &self.adapter().uuid_string())
            .field("device", &self.device_id())
            .field("validation", &self.validation_enabled())
            .field("kernels", &self.kernels)
            .finish_non_exhaustive()
    }
}

/// A checked-out per-thread submitter.
pub struct SubmitterGuard<'a> {
    backend: &'a VulkanBackend,
    thread: ThreadId,
    submitter: Option<Submitter>,
}

impl std::ops::Deref for SubmitterGuard<'_> {
    type Target = Submitter;

    fn deref(&self) -> &Submitter {
        self.submitter
            .as_ref()
            .expect("a live guard always owns its submitter")
    }
}

impl std::ops::DerefMut for SubmitterGuard<'_> {
    fn deref_mut(&mut self) -> &mut Submitter {
        self.submitter
            .as_mut()
            .expect("a live guard always owns its submitter")
    }
}

impl Drop for SubmitterGuard<'_> {
    fn drop(&mut self) {
        let Some(submitter) = self.submitter.take() else {
            return;
        };
        let Ok(mut pool) = self.backend.submitters.lock() else {
            // A poisoned pool means another thread panicked mid-submission; the
            // submitter is destroyed here and the device teardown is still
            // complete.
            return;
        };
        pool.active.remove(&self.thread);
        pool.by_thread.insert(self.thread, submitter);
    }
}

/// Result of a host↔device transfer.
#[derive(Debug, Clone, Default)]
pub struct TransferReport {
    /// Bytes copied.
    pub bytes: u64,
    /// Submissions used.
    pub submissions: u32,
    /// Host wall time in nanoseconds.
    pub host_ns: u64,
    /// Device time when timestamps are available.
    pub gpu_ns: Option<u64>,
}

/// Result of one device-resident dispatch.
#[derive(Debug, Clone)]
pub struct DispatchOutcome {
    /// Kernel name.
    pub kernel: &'static str,
    /// Submissions used.
    pub submissions: u32,
    /// Workgroup counts.
    pub workgroups: [u32; 3],
    /// Bindings as recorded.
    pub bindings: Vec<BindingSnapshot>,
    /// Host wall time in nanoseconds.
    pub host_ns: u64,
    /// Device time when timestamps are available.
    pub gpu_ns: Option<u64>,
}

/// Why no Vulkan device is listed, in words a user can act on.
fn no_vulkan_device_reason() -> String {
    match instance::shared() {
        Err(error) => format!(
            "no Vulkan device is usable: {error}. Install a Vulkan ICD \
             (`vulkan-icd-loader` and the GPU driver package), or point VK_ICD_FILENAMES at a \
             valid ICD; `vulkaninfo --summary` shows what the loader sees"
        ),
        Ok(probe) if probe.adapters().is_empty() => concat!(
            "the Vulkan loader reported zero physical devices; check that an ICD is installed ",
            "under /usr/share/vulkan/icd.d/ and that VK_ICD_FILENAMES is valid"
        )
        .to_string(),
        Ok(probe) => {
            let adapters = probe
                .adapters()
                .iter()
                .map(|adapter| format!("\"{}\" ({})", adapter.name, adapter.uuid_string()))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "the loader enumerated {} adapter(s) ({adapters}), none with a COMPUTE queue \
                 family; the Vulkan backend needs a compute-capable device",
                probe.adapters().len()
            )
        }
    }
}

fn capabilities_for(adapter: &AdapterInfo, kernels: &KernelSet) -> Capabilities {
    let limits = &adapter.limits;
    Capabilities {
        dtypes: kernels::KERNEL_DTYPES,
        // One buffer must fit both a single allocation and a single storage
        // range; report the tighter of the two so a plan is never validated
        // against a limit the binding cannot honour.
        max_buffer_bytes: u64::from(limits.max_storage_buffer_range)
            .min(limits.max_memory_allocation_size),
        max_shape_rank: MAX_SHAPE_RANK,
        max_workgroup_size: limits.max_compute_work_group_size,
        max_threads: limits.max_compute_work_group_invocations,
        // Elementwise kernels in the alpha require contiguous operands.
        supports_strided_views: false,
        features: CapabilityFeatures {
            // A kernel registration promotes these; the smoke kernel claims
            // neither broadcasting nor fused attention.
            ..kernels.features()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::dtype::DType;

    #[test]
    fn capabilities_use_the_tighter_of_allocation_and_storage_limits() {
        // A synthetic adapter is not constructible without Vulkan, so assert
        // the arithmetic rule directly.
        let storage = 4_294_967_295u64;
        let allocation = 4_292_870_144u64;
        assert_eq!(storage.min(allocation), allocation);

        // Dtype table is non-empty and f32-only until a kernel says otherwise.
        assert!(kernels::KERNEL_DTYPES.contains(&DType::F32));
    }

    #[test]
    fn backend_name_is_stable() {
        assert_eq!(VULKAN_BACKEND_NAME, "vulkan");
    }
}
