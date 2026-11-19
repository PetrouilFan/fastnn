//! Adapter enumeration, capabilities and the device limit table.
//!
//! Everything reported here comes from `vkGetPhysicalDeviceProperties2`,
//! `vkEnumerateDeviceExtensionProperties`, `vkGetPhysicalDeviceQueueFamilyProperties`
//! and `vkGetPhysicalDeviceMemoryProperties`. Nothing is hard-coded for a
//! particular GPU and nothing is assumed about enumeration order: the loader on
//! this machine exposes a lavapipe (CPU) device beside the discrete GPU, so
//! callers select by UUID or name, never by assuming index 0.

use ash::vk;

use super::error::{Error, Result};
use crate::inference::ops::OpId;

/// Extensions the 3.0 Vulkan backend reports explicitly. Presence is a fact,
/// not a requirement; kernels that need one probe it before enabling the path.
pub const RELEVANT_EXTENSIONS: &[(&str, &str)] = &[
    ("VK_KHR_8bit_storage", "8-bit storage"),
    ("VK_KHR_16bit_storage", "16-bit storage"),
    ("VK_KHR_shader_float16_int8", "FP16/INT8 arithmetic"),
    ("VK_KHR_buffer_device_address", "buffer device address"),
    ("VK_KHR_synchronization2", "synchronization2"),
    ("VK_EXT_memory_budget", "memory budget"),
    ("VK_EXT_descriptor_buffer", "descriptor buffers"),
];

/// One queue family of an adapter.
#[derive(Debug, Clone)]
pub struct QueueFamilyInfo {
    /// Family index used in `vkCreateDevice` and command pools.
    pub index: u32,
    /// Queue flags.
    pub flags: vk::QueueFlags,
    /// Number of queues the family exposes.
    pub queue_count: u32,
    /// Valid timestamp bits; `0` means timestamps are not supported here.
    pub timestamp_valid_bits: u32,
    /// Sparse-binding transfer granularity.
    pub min_image_transfer_granularity: vk::Extent3D,
}

/// Fixed workgroup invocations required by the registered kernels (16x16 tile).
pub const MIN_REQUIRED_INVOCATIONS: u32 = 256;
/// Widest workgroup the shaders declare (`local_size_x`).
pub const MIN_REQUIRED_WORKGROUP_X: u32 = 64;
/// Tallest workgroup the shaders declare (the matmul tile's second axis).
pub const MIN_REQUIRED_WORKGROUP_Y: u32 = 16;
/// Largest single immutable binding the pinned model needs (25.7 MB), rounded
/// up to a documented floor.
pub const MIN_REQUIRED_STORAGE_RANGE: u64 = 32 * 1024 * 1024;
/// Smallest allocation the chunk allocator will reserve.
pub const MIN_REQUIRED_ALLOCATION: u64 = 64 * 1024 * 1024;

/// The device limits the platform layer and the kernels plan against.
#[derive(Debug, Clone)]
pub struct AdapterLimits {
    /// Maximum workgroup counts per axis.
    pub max_compute_work_group_count: [u32; 3],
    /// Maximum workgroup size per axis.
    pub max_compute_work_group_size: [u32; 3],
    /// Maximum invocations in one workgroup.
    pub max_compute_work_group_invocations: u32,
    /// Per-workgroup shared memory, in bytes.
    pub max_compute_shared_memory_size: u32,
    /// Largest byte range one storage-buffer binding may address.
    pub max_storage_buffer_range: u32,
    /// Largest byte range one uniform-buffer binding may address.
    pub max_uniform_buffer_range: u32,
    /// Largest single `vkAllocateMemory`, in bytes. On the GTX 1080 Ti this is
    /// ≈4 GiB while the heap is 11 GiB, which is why the allocator suballocates.
    pub max_memory_allocation_size: u64,
    /// Alignment required for a storage-buffer descriptor offset.
    pub min_storage_buffer_offset_alignment: u64,
    /// Alignment required for a uniform-buffer descriptor offset.
    pub min_uniform_buffer_offset_alignment: u64,
    /// Alignment required for a `vkMapMemory` offset.
    pub min_memory_map_alignment: u64,
    /// Granularity of `vkFlushMappedMemoryRanges`/`vkInvalidateMappedMemoryRanges`.
    pub non_coherent_atom_size: u64,
    /// Largest number of simultaneous `vkAllocateMemory` calls.
    pub max_memory_allocation_count: u32,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    /// Largest push-constant block.
    pub max_push_constants_size: u32,
    /// Storage buffers addressable from one descriptor set.
    pub max_descriptor_set_storage_buffers: u32,
    /// Storage buffers addressable from one shader stage.
    pub max_per_stage_descriptor_storage_buffers: u32,
}

/// One memory heap, for diagnostics.
#[derive(Debug, Clone)]
pub struct HeapInfo {
    /// Heap index.
    pub index: u32,
    /// Heap size in bytes.
    pub size: u64,
    /// Heap flags.
    pub flags: vk::MemoryHeapFlags,
}

/// A physical device this process can open, with its real limits.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Index in the process-wide discovery order. This is the value carried by
    /// [`crate::inference::device::PhysicalDeviceId`].
    pub index: u32,
    /// Adapter name (`"NVIDIA GeForce GTX 1080 Ti"`).
    pub name: String,
    /// PCI vendor id.
    pub vendor_id: u32,
    /// PCI device id.
    pub device_id: u32,
    /// Discrete / integrated / virtual / CPU.
    pub device_type: vk::PhysicalDeviceType,
    /// API version supported by the device.
    pub api_version: u32,
    /// Raw driver version.
    pub driver_version_raw: u32,
    /// Driver id (Vulkan 1.2+).
    pub driver_id: vk::DriverId,
    /// `deviceUUID`, used to select the physical GPU regardless of order.
    pub device_uuid: [u8; vk::UUID_SIZE],
    /// `driverUUID`.
    pub driver_uuid: [u8; vk::UUID_SIZE],
    /// Queue families.
    pub queue_families: Vec<QueueFamilyInfo>,
    /// Limits.
    pub limits: AdapterLimits,
    /// Device extensions, sorted.
    pub extensions: Vec<String>,
    /// Memory heaps.
    pub memory_heaps: Vec<HeapInfo>,
    /// Number of memory types.
    pub memory_type_count: u32,
}

impl AdapterInfo {
    /// Probe one physical device. `index` is the discovery index.
    pub fn probe(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        index: u32,
    ) -> Result<Self> {
        // SAFETY: `physical` came from `vkEnumeratePhysicalDevices` on this
        // instance and is still owned by it; the returned value is a plain POD
        // copy.
        let base = unsafe { instance.get_physical_device_properties(physical) };

        // `deviceUUID`/`driverUUID` need `VkPhysicalDeviceIDProperties`,
        // `maxMemoryAllocationSize` needs
        // `VkPhysicalDeviceMaintenance3Properties` (both core in Vulkan 1.1),
        // and `driverID` lives in `VkPhysicalDeviceDriverProperties` (core in
        // Vulkan 1.2). The chain points at stack locals that outlive the call.
        //
        // SAFETY: every struct in the pNext chain is a distinct, properly
        // initialised stack local whose `s_type` is set by `Default`; each is
        // borrowed for at least as long as `properties2`, and the stack
        // lifetimes outlive `get_physical_device_properties2`.
        let (properties, device_uuid, driver_uuid, max_memory_allocation_size, driver_id) = {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut maintenance = vk::PhysicalDeviceMaintenance3Properties::default();
            let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut id)
                .push_next(&mut maintenance);
            let supports_driver_properties = base.api_version >= vk::API_VERSION_1_2;
            if supports_driver_properties {
                properties2 = properties2.push_next(&mut driver_properties);
            }
            unsafe {
                instance.get_physical_device_properties2(physical, &mut properties2);
            }
            (
                properties2.properties,
                id.device_uuid,
                id.driver_uuid,
                maintenance.max_memory_allocation_size,
                if supports_driver_properties {
                    driver_properties.driver_id
                } else {
                    vk::DriverId::from_raw(0)
                },
            )
        };

        // SAFETY: plain enumeration of queue-family properties for a live
        // physical device; ash owns the returned Vec.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical) }
                .iter()
                .enumerate()
                .map(|(index, family)| QueueFamilyInfo {
                    index: index as u32,
                    flags: family.queue_flags,
                    queue_count: family.queue_count,
                    timestamp_valid_bits: family.timestamp_valid_bits,
                    min_image_transfer_granularity: family.min_image_transfer_granularity,
                })
                .collect();

        // SAFETY: the physical device belongs to `instance`; ash returns owned
        // copies of the driver-written extension properties.
        let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }
            .map_err(|code| {
                Error::vk(
                    "vkEnumerateDeviceExtensionProperties",
                    code,
                    format!("while probing physical device {index}"),
                )
            })?
            .iter()
            .filter_map(|properties| {
                properties
                    .extension_name_as_c_str()
                    .ok()
                    .map(|name| name.to_string_lossy().into_owned())
            })
            .collect::<Vec<_>>();

        // SAFETY: `physical` is a live handle owned by `instance`.
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_heaps = (0..memory_properties.memory_heap_count)
            .map(|index| {
                let heap = memory_properties.memory_heaps[index as usize];
                HeapInfo {
                    index,
                    size: heap.size,
                    flags: heap.flags,
                }
            })
            .collect();

        let limits = &properties.limits;
        Ok(AdapterInfo {
            index,
            name: c_name(&base.device_name),
            vendor_id: base.vendor_id,
            device_id: base.device_id,
            device_type: base.device_type,
            api_version: base.api_version,
            driver_version_raw: base.driver_version,
            driver_id,
            device_uuid,
            driver_uuid,
            queue_families,
            limits: AdapterLimits {
                max_compute_work_group_count: limits.max_compute_work_group_count,
                max_compute_work_group_size: limits.max_compute_work_group_size,
                max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
                max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
                max_storage_buffer_range: limits.max_storage_buffer_range,
                max_uniform_buffer_range: limits.max_uniform_buffer_range,
                max_memory_allocation_size,
                min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
                min_uniform_buffer_offset_alignment: limits.min_uniform_buffer_offset_alignment,
                min_memory_map_alignment: limits.min_memory_map_alignment as u64,
                non_coherent_atom_size: limits.non_coherent_atom_size,
                max_memory_allocation_count: limits.max_memory_allocation_count,
                timestamp_period: limits.timestamp_period,
                max_push_constants_size: limits.max_push_constants_size,
                max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers,
                max_per_stage_descriptor_storage_buffers: limits
                    .max_per_stage_descriptor_storage_buffers,
            },
            extensions,
            memory_heaps,
            memory_type_count: memory_properties.memory_type_count,
        })
    }

    /// `true` if the adapter is a software implementation (lavapipe and
    /// friends). Software adapters are selectable only when explicitly allowed,
    /// and every run on one must be labelled as a software run.
    pub fn is_software(&self) -> bool {
        self.device_type == vk::PhysicalDeviceType::CPU
            || self.vendor_id == 0x10005
            || self.name.to_ascii_lowercase().contains("llvmpipe")
            || self.name.to_ascii_lowercase().contains("lavapipe")
    }

    /// `true` if the adapter exposes at least one COMPUTE queue family.
    pub fn has_compute_queue(&self) -> bool {
        self.queue_families
            .iter()
            .any(|family| family.flags.contains(vk::QueueFlags::COMPUTE))
    }

    /// The queue family the backend will use: a dedicated compute family with
    /// the most queues, else the first family with COMPUTE.
    pub fn compute_family(&self) -> Option<&QueueFamilyInfo> {
        let mut dedicated: Vec<&QueueFamilyInfo> = self
            .queue_families
            .iter()
            .filter(|family| {
                family.flags.contains(vk::QueueFlags::COMPUTE)
                    && !family.flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .collect();
        dedicated.sort_by_key(|family| std::cmp::Reverse(family.queue_count));
        dedicated.first().copied().or_else(|| {
            self.queue_families
                .iter()
                .find(|family| family.flags.contains(vk::QueueFlags::COMPUTE))
        })
    }

    /// `true` if the adapter exposes `name`.
    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.iter().any(|extension| extension == name)
    }

    /// Canonical UUID string.
    pub fn uuid_string(&self) -> String {
        uuid_to_string(&self.device_uuid)
    }

    /// Canonical driver UUID string.
    pub fn driver_uuid_string(&self) -> String {
        uuid_to_string(&self.driver_uuid)
    }

    /// Vendor name from the PCI id.
    pub fn vendor_name(&self) -> &'static str {
        match self.vendor_id {
            0x10de => "NVIDIA",
            0x1002 => "AMD",
            0x8086 => "Intel",
            0x10005 => "Mesa",
            0x13b5 => "ARM",
            0x5143 => "Qualcomm",
            0x1010 => "Imagination",
            _ => "unknown",
        }
    }

    /// Vendor-aware driver-version string.
    pub fn driver_version_string(&self) -> String {
        match self.vendor_id {
            // NVIDIA: major(10) minor(8) secondary(8) build(6).
            0x10de => format!(
                "{}.{}.{}.{}",
                (self.driver_version_raw >> 22) & 0x3ff,
                (self.driver_version_raw >> 14) & 0xff,
                (self.driver_version_raw >> 6) & 0xff,
                self.driver_version_raw & 0x3f
            ),
            // Mesa encodes VK_MAKE_VERSION(major, minor, patch).
            0x10005 => format!(
                "{}.{}.{}",
                (self.driver_version_raw >> 22) & 0x3ff,
                (self.driver_version_raw >> 12) & 0x3ff,
                self.driver_version_raw & 0xfff
            ),
            _ => format!("0x{:08x}", self.driver_version_raw),
        }
    }

    /// One-line label used in device lists.
    pub fn summary_line(&self) -> String {
        format!(
            "[{}] {:<40} type={:?} api={} uuid={}",
            self.index,
            self.name,
            self.device_type,
            format_version(self.api_version),
            self.uuid_string()
        )
    }

    /// Capabilities the FN3 kernels require before a device is opened.
    ///
    /// Each entry names the capability **and both numbers** where a limit is
    /// involved, so an unavailable device produces an actionable error instead
    /// of a driver failure at the first dispatch. The rules are the kernels'
    /// fixed geometry, not a benchmark: the matmul tile is 16x16 (256
    /// invocations, 16 rows), every other shader uses 64-wide workgroups, and
    /// the largest immutable binding in the pinned model is the 25.7 MB
    /// embedding/lm_head matrix, so a 32 MiB storage range is the floor.
    pub fn missing_required_features(&self) -> Vec<String> {
        let mut missing = Vec::new();
        if !self.has_compute_queue() {
            missing.push(
                "a COMPUTE queue family (none of the adapter's families has COMPUTE)".to_string(),
            );
        }
        let limits = &self.limits;
        if limits.max_compute_work_group_invocations < MIN_REQUIRED_INVOCATIONS {
            missing.push(format!(
                "maxComputeWorkGroupInvocations >= {MIN_REQUIRED_INVOCATIONS} (adapter reports {})",
                limits.max_compute_work_group_invocations
            ));
        }
        if limits.max_compute_work_group_size[0] < MIN_REQUIRED_WORKGROUP_X {
            missing.push(format!(
                "maxComputeWorkGroupSize[0] >= {MIN_REQUIRED_WORKGROUP_X} (adapter reports {})",
                limits.max_compute_work_group_size[0]
            ));
        }
        if limits.max_compute_work_group_size[1] < MIN_REQUIRED_WORKGROUP_Y {
            missing.push(format!(
                "maxComputeWorkGroupSize[1] >= {MIN_REQUIRED_WORKGROUP_Y} (adapter reports {})",
                limits.max_compute_work_group_size[1]
            ));
        }
        if u64::from(limits.max_storage_buffer_range) < MIN_REQUIRED_STORAGE_RANGE {
            missing.push(format!(
                "maxStorageBufferRange >= {MIN_REQUIRED_STORAGE_RANGE} bytes (adapter reports {})",
                limits.max_storage_buffer_range
            ));
        }
        if limits.max_memory_allocation_size < MIN_REQUIRED_ALLOCATION {
            missing.push(format!(
                "maxMemoryAllocationSize >= {MIN_REQUIRED_ALLOCATION} bytes (adapter reports {})",
                limits.max_memory_allocation_size
            ));
        }
        missing
    }

    /// Multi-line adapter report for logs and `available_devices` diagnostics.
    pub fn report(&self) -> String {
        let mut text = format!(
            "adapter[{}]: {}\n  vendor={} (0x{:04x}) device=0x{:04x} type={:?} api={}\n  \
             driver={} (id={}) uuid={} driver_uuid={}\n",
            self.index,
            self.name,
            self.vendor_name(),
            self.vendor_id,
            self.device_id,
            self.device_type,
            format_version(self.api_version),
            self.driver_version_string(),
            self.driver_id.as_raw(),
            self.uuid_string(),
            self.driver_uuid_string()
        );
        for heap in &self.memory_heaps {
            text.push_str(&format!(
                "  heap[{}] {:.2} GiB flags={:?}\n",
                heap.index,
                heap.size as f64 / (1024.0 * 1024.0 * 1024.0),
                heap.flags
            ));
        }
        for family in &self.queue_families {
            text.push_str(&format!(
                "  queue[{}] {:<34} queues={} timestamp_bits={}\n",
                family.index,
                queue_flags_string(family.flags),
                family.queue_count,
                family.timestamp_valid_bits
            ));
        }
        let limits = &self.limits;
        text.push_str(&format!(
            "  maxMemoryAllocationSize={} maxStorageBufferRange={} \
             maxComputeWorkGroupInvocations={} maxComputeSharedMemorySize={}\n",
            limits.max_memory_allocation_size,
            limits.max_storage_buffer_range,
            limits.max_compute_work_group_invocations,
            limits.max_compute_shared_memory_size
        ));
        text
    }
}

/// Enumerate every physical device visible to `instance`.
///
/// An empty list is not an error: it means no ICD exposed a device, which
/// [`crate::inference::backend::vulkan::available_devices`] reports as "no Vulkan".
pub fn enumerate(instance: &ash::Instance) -> Result<Vec<AdapterInfo>> {
    // SAFETY: standard enumeration of physical devices owned by this instance;
    // ash allocates the result vector. The handles stay valid while the
    // instance lives.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|code| {
        Error::vk(
            "vkEnumeratePhysicalDevices",
            code,
            "the loader returned no device list",
        )
    })?;
    physical_devices
        .iter()
        .enumerate()
        .map(|(index, physical)| AdapterInfo::probe(instance, *physical, index as u32))
        .collect()
}

/// Owning destination of an op in the alpha backend. Used for honest
/// "unsupported" messages naming the issue that will implement the kernel.
pub fn op_owner(op: OpId) -> &'static str {
    match op {
        OpId::MatMul => "FN3-14 (#60)",
        OpId::Softmax
        | OpId::LayerNorm
        | OpId::ReduceMean
        | OpId::ReduceSum
        | OpId::ArgMax
        | OpId::MaskedAttention => "FN3-15 (#61)",
        OpId::Add
        | OpId::Sub
        | OpId::Mul
        | OpId::Div
        | OpId::Neg
        | OpId::Pow
        | OpId::Sqrt
        | OpId::Erf
        | OpId::Tanh
        | OpId::Gelu
        | OpId::Relu
        | OpId::Gather
        | OpId::Transpose
        | OpId::Reshape
        | OpId::Flatten
        | OpId::Squeeze
        | OpId::Unsqueeze
        | OpId::Concat
        | OpId::Slice
        | OpId::Split
        | OpId::Shape
        | OpId::ConstantOfShape
        | OpId::Expand
        | OpId::Range
        | OpId::Cast
        | OpId::Equal
        | OpId::Less
        | OpId::Greater
        | OpId::Where
        | OpId::Trilu
        | OpId::Sin
        | OpId::Cos
        | OpId::ScatterND => "FN3-13 (#59)",
    }
}

fn c_name(field: &[std::os::raw::c_char; vk::MAX_PHYSICAL_DEVICE_NAME_SIZE]) -> String {
    let bytes: Vec<u8> = field
        .iter()
        .take_while(|character| **character != 0)
        .map(|character| *character as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// `major.minor.patch` rendering of a Vulkan version.
pub fn format_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

/// Canonical 8-4-4-4-12 UUID string.
pub fn uuid_to_string(uuid: &[u8; vk::UUID_SIZE]) -> String {
    let hex: String = uuid.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Parse a UUID in canonical or compact form.
pub fn parse_uuid(input: &str) -> Result<[u8; vk::UUID_SIZE]> {
    let compact: String = input
        .chars()
        .filter(|character| *character != '-' && !character.is_whitespace())
        .collect();
    if compact.len() != 32 {
        return Err(Error::internal(format!(
            "device UUID {input:?} must have 32 hex digits (8-4-4-4-12)"
        )));
    }
    let mut bytes = [0u8; vk::UUID_SIZE];
    for (index, byte) in bytes.iter_mut().enumerate() {
        let pair = &compact[index * 2..index * 2 + 2];
        *byte = u8::from_str_radix(pair, 16).map_err(|_| {
            Error::internal(format!(
                "device UUID {input:?} contains non-hex characters near {pair:?}"
            ))
        })?;
    }
    Ok(bytes)
}

/// `GRAPHICS|COMPUTE|...` rendering of queue flags.
pub fn queue_flags_string(flags: vk::QueueFlags) -> String {
    let mut names = Vec::new();
    if flags.contains(vk::QueueFlags::GRAPHICS) {
        names.push("GRAPHICS");
    }
    if flags.contains(vk::QueueFlags::COMPUTE) {
        names.push("COMPUTE");
    }
    if flags.contains(vk::QueueFlags::TRANSFER) {
        names.push("TRANSFER");
    }
    if flags.contains(vk::QueueFlags::SPARSE_BINDING) {
        names.push("SPARSE_BINDING");
    }
    if flags.contains(vk::QueueFlags::PROTECTED) {
        names.push("PROTECTED");
    }
    if names.is_empty() {
        names.push("NONE");
    }
    names.join("|")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuid_round_trips_both_forms() {
        let canonical = "471b2bda-51e1-6451-e692-ff5d1e146f1e";
        let bytes = parse_uuid(canonical).expect("canonical UUID parses");
        assert_eq!(uuid_to_string(&bytes), canonical);
        let compact = canonical.replace('-', "").to_uppercase();
        assert_eq!(parse_uuid(&compact).expect("compact UUID parses"), bytes);
    }

    #[test]
    fn uuid_rejects_wrong_length_and_non_hex() {
        assert!(parse_uuid("471b2bda").is_err());
        assert!(parse_uuid("zz1b2bda-51e1-6451-e692-ff5d1e146f1e").is_err());
    }

    #[test]
    fn api_version_formats_major_minor_patch() {
        assert_eq!(
            format_version(vk::make_api_version(0, 1, 4, 312)),
            "1.4.312"
        );
    }

    #[test]
    fn every_op_has_an_owner() {
        for op in OpId::ALL {
            assert!(!op_owner(op).is_empty(), "{op}");
        }
    }

    /// A synthetic adapter that satisfies the kernel requirements.
    ///
    /// This is a *simulated* device, not hardware evidence: it exists to pin
    /// the rule that a missing capability is named with both numbers before a
    /// device is opened.
    fn synthetic_adapter() -> AdapterInfo {
        AdapterInfo {
            index: 0,
            name: "FN3 synthetic device".to_string(),
            vendor_id: 0x10de,
            device_id: 0x1b06,
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            api_version: vk::make_api_version(0, 1, 4, 0),
            driver_version_raw: 0,
            driver_id: vk::DriverId::NVIDIA_PROPRIETARY,
            device_uuid: [0x47; vk::UUID_SIZE],
            driver_uuid: [0x51; vk::UUID_SIZE],
            queue_families: vec![QueueFamilyInfo {
                index: 0,
                flags: vk::QueueFlags::COMPUTE,
                queue_count: 1,
                timestamp_valid_bits: 64,
                min_image_transfer_granularity: vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            }],
            limits: AdapterLimits {
                max_compute_work_group_count: [65535, 65535, 65535],
                max_compute_work_group_size: [1024, 1024, 64],
                max_compute_work_group_invocations: 1024,
                max_compute_shared_memory_size: 49152,
                max_storage_buffer_range: u32::MAX,
                max_uniform_buffer_range: 65536,
                max_memory_allocation_size: 4 * 1024 * 1024 * 1024,
                min_storage_buffer_offset_alignment: 16,
                min_uniform_buffer_offset_alignment: 256,
                min_memory_map_alignment: 64,
                non_coherent_atom_size: 64,
                max_memory_allocation_count: 4096,
                timestamp_period: 1.0,
                max_push_constants_size: 128,
                max_descriptor_set_storage_buffers: 16,
                max_per_stage_descriptor_storage_buffers: 16,
            },
            extensions: Vec::new(),
            memory_heaps: Vec::new(),
            memory_type_count: 2,
        }
    }

    #[test]
    fn a_fully_capable_adapter_is_accepted() {
        assert!(synthetic_adapter().missing_required_features().is_empty());
    }

    #[test]
    fn a_device_missing_a_capability_names_it_and_both_numbers() {
        let mut adapter = synthetic_adapter();
        adapter.queue_families.clear();
        adapter.limits.max_compute_work_group_invocations = 128;
        adapter.limits.max_storage_buffer_range = 1024 * 1024;
        adapter.limits.max_memory_allocation_size = 1024 * 1024;
        adapter.limits.max_compute_work_group_size = [32, 8, 1];
        let missing = adapter.missing_required_features();
        let text = missing.join("; ");
        assert_eq!(missing.len(), 6, "{text}");
        assert!(text.contains("COMPUTE queue family"), "{text}");
        assert!(
            text.contains("maxComputeWorkGroupInvocations >= 256 (adapter reports 128)"),
            "{text}"
        );
        assert!(
            text.contains("maxStorageBufferRange >= 33554432 bytes (adapter reports 1048576)"),
            "{text}"
        );
        assert!(
            text.contains("maxMemoryAllocationSize >= 67108864 bytes (adapter reports 1048576)"),
            "{text}"
        );
        assert!(text.contains("maxComputeWorkGroupSize[0] >= 64"), "{text}");
        assert!(text.contains("maxComputeWorkGroupSize[1] >= 16"), "{text}");
    }
}
