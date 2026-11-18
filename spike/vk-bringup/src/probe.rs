//! Physical-device enumeration and diagnostics.
//!
//! Everything printed here comes from `vkGetPhysicalDeviceProperties2`,
//! `vkEnumerateDeviceExtensionProperties` and
//! `vkGetPhysicalDeviceQueueFamilyProperties`; nothing is cached at build time
//! and nothing is assumed about the machine.

use ash::vk;

use crate::error::{Error, Result};

/// Extensions the later GPU backend is likely to care about. Presence is
/// reported explicitly; the spike itself does not enable any of them.
pub const RELEVANT_EXTENSIONS: &[(&str, &str)] = &[
    ("VK_KHR_8bit_storage", "8-bit storage"),
    ("VK_KHR_16bit_storage", "16-bit storage"),
    ("VK_KHR_shader_float16_int8", "FP16/INT8 arithmetic"),
    ("VK_KHR_buffer_device_address", "buffer device address"),
    ("VK_KHR_synchronization2", "synchronization2"),
    ("VK_EXT_memory_budget", "memory budget"),
];

#[derive(Debug, Clone)]
pub struct QueueFamilyInfo {
    pub index: u32,
    pub flags: vk::QueueFlags,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: vk::Extent3D,
}

#[derive(Debug, Clone)]
pub struct Limits {
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_size: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_storage_buffer_range: u32,
    pub max_memory_allocation_size: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub max_memory_allocation_count: u32,
    pub timestamp_period: f32,
    pub max_push_constants_size: u32,
    pub max_descriptor_set_storage_buffers: u32,
}

#[derive(Debug, Clone)]
pub struct HeapInfo {
    pub index: u32,
    pub size: u64,
    pub flags: vk::MemoryHeapFlags,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: u32,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: vk::PhysicalDeviceType,
    pub api_version: u32,
    pub driver_version_raw: u32,
    pub driver_id: vk::DriverId,
    pub device_uuid: [u8; vk::UUID_SIZE],
    pub driver_uuid: [u8; vk::UUID_SIZE],
    pub queue_families: Vec<QueueFamilyInfo>,
    pub limits: Limits,
    pub extensions: Vec<String>,
    pub memory_heaps: Vec<HeapInfo>,
    pub memory_type_count: u32,
}

impl DeviceInfo {
    pub fn probe(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        index: u32,
    ) -> Result<Self> {
        // SAFETY: `physical` came from `vkEnumeratePhysicalDevices` on this
        // instance and is still owned by it; the returned value is a plain
        // POD copy.
        let base = unsafe { instance.get_physical_device_properties(physical) };

        // `deviceUUID`/`driverUUID` need `VkPhysicalDeviceIDProperties`,
        // `maxMemoryAllocationSize` needs
        // `VkPhysicalDeviceMaintenance3Properties` (both core in Vulkan 1.1),
        // and `driverID` lives in `VkPhysicalDeviceDriverProperties` (core in
        // Vulkan 1.2). The chain points at stack locals that outlive the call.
        //
        // SAFETY: the pNext chain is built from distinct, properly initialised
        // structs whose s_type fields are set by `Default`; each is borrowed
        // for at least as long as `props2`, and the stack lifetimes outlive
        // `get_physical_device_properties2`.
        let (properties, device_uuid, driver_uuid, max_memory_allocation_size, driver_id) = {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut maintenance = vk::PhysicalDeviceMaintenance3Properties::default();
            let mut driver_properties = vk::PhysicalDeviceDriverProperties::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut id)
                .push_next(&mut maintenance);
            let supports_driver_properties = base.api_version >= vk::API_VERSION_1_2;
            if supports_driver_properties {
                props2 = props2.push_next(&mut driver_properties);
            }
            unsafe {
                instance.get_physical_device_properties2(physical, &mut props2);
            }
            let properties = props2.properties;
            let driver_id = if supports_driver_properties {
                driver_properties.driver_id
            } else {
                vk::DriverId::from_raw(0)
            };
            (
                properties,
                id.device_uuid,
                id.driver_uuid,
                maintenance.max_memory_allocation_size,
                driver_id,
            )
        };

        // SAFETY: plain enumeration of queue-family properties for a live
        // physical device; ash owns the returned Vec.
        let queue_families_raw =
            unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let queue_families = queue_families_raw
            .iter()
            .enumerate()
            .map(|(i, f)| QueueFamilyInfo {
                index: i as u32,
                flags: f.queue_flags,
                queue_count: f.queue_count,
                timestamp_valid_bits: f.timestamp_valid_bits,
                min_image_transfer_granularity: f.min_image_transfer_granularity,
            })
            .collect();

        let extensions = {
            // SAFETY: the physical device belongs to `instance`; ash returns a
            // Vec of owned copies of the driver-written extension properties.
            let props = unsafe {
                instance
                    .enumerate_device_extension_properties(physical)
                    .map_err(|code| {
                        Error::vk(
                            "vkEnumerateDeviceExtensionProperties",
                            code,
                            format!("while probing physical device {index}"),
                        )
                    })?
            };
            let mut names = props
                .iter()
                .filter_map(|p| {
                    p.extension_name_as_c_str()
                        .ok()
                        .map(|c| c.to_string_lossy().into_owned())
                })
                .collect::<Vec<_>>();
            names.sort();
            names.dedup();
            names
        };

        // SAFETY: `physical` is a live handle owned by `instance`.
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let mut memory_heaps = Vec::with_capacity(memory_properties.memory_heap_count as usize);
        for i in 0..memory_properties.memory_heap_count {
            let heap = memory_properties.memory_heaps[i as usize];
            memory_heaps.push(HeapInfo {
                index: i,
                size: heap.size,
                flags: heap.flags,
            });
        }

        Ok(Self {
            index,
            name: cstr_field(&base.device_name),
            vendor_id: base.vendor_id,
            device_id: base.device_id,
            device_type: base.device_type,
            api_version: base.api_version,
            driver_version_raw: base.driver_version,
            driver_id,
            device_uuid,
            driver_uuid,
            queue_families,
            limits: Limits {
                max_compute_work_group_count: properties.limits.max_compute_work_group_count,
                max_compute_work_group_size: properties.limits.max_compute_work_group_size,
                max_compute_work_group_invocations: properties
                    .limits
                    .max_compute_work_group_invocations,
                max_compute_shared_memory_size: properties.limits.max_compute_shared_memory_size,
                max_storage_buffer_range: properties.limits.max_storage_buffer_range,
                max_memory_allocation_size,
                min_uniform_buffer_offset_alignment: properties
                    .limits
                    .min_uniform_buffer_offset_alignment,
                min_storage_buffer_offset_alignment: properties
                    .limits
                    .min_storage_buffer_offset_alignment,
                max_memory_allocation_count: properties.limits.max_memory_allocation_count,
                timestamp_period: properties.limits.timestamp_period,
                max_push_constants_size: properties.limits.max_push_constants_size,
                max_descriptor_set_storage_buffers: properties
                    .limits
                    .max_descriptor_set_storage_buffers,
            },
            extensions,
            memory_heaps,
            memory_type_count: memory_properties.memory_type_count,
        })
    }

    pub fn has_extension(&self, name: &str) -> bool {
        self.extensions.iter().any(|e| e == name)
    }

    pub fn uuid_string(&self) -> String {
        uuid_to_string(&self.device_uuid)
    }

    pub fn driver_uuid_string(&self) -> String {
        uuid_to_string(&self.driver_uuid)
    }

    /// Vendor-specific driver-version decoding. Other vendors are reported raw.
    pub fn driver_version_string(&self) -> String {
        decode_driver_version(self.vendor_id, self.driver_version_raw)
    }

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

    pub fn has_compute_queue(&self) -> bool {
        self.queue_families
            .iter()
            .any(|f| f.flags.contains(vk::QueueFlags::COMPUTE))
    }

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
}

fn cstr_field(field: &[std::os::raw::c_char; vk::MAX_PHYSICAL_DEVICE_NAME_SIZE]) -> String {
    let bytes: Vec<u8> = field
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn format_version(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}

pub fn uuid_to_string(uuid: &[u8; vk::UUID_SIZE]) -> String {
    let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

pub fn parse_uuid(input: &str) -> Result<[u8; vk::UUID_SIZE]> {
    let compact: String = input
        .chars()
        .filter(|c| *c != '-' && !c.is_whitespace())
        .collect();
    if compact.len() != 32 {
        return Err(Error::Usage(format!(
            "device UUID {input:?} must have 32 hex digits (8-4-4-4-12); \
             run --list to see available UUIDs"
        )));
    }
    let mut bytes = [0u8; vk::UUID_SIZE];
    for (i, byte) in bytes.iter_mut().enumerate() {
        let pair = &compact[i * 2..i * 2 + 2];
        *byte = u8::from_str_radix(pair, 16).map_err(|_| {
            Error::Usage(format!(
                "device UUID {input:?} contains non-hex characters near {pair:?}"
            ))
        })?;
    }
    Ok(bytes)
}

fn decode_driver_version(vendor_id: u32, raw: u32) -> String {
    match vendor_id {
        // NVIDIA: major(10) minor(8) secondary(8) build(6)
        0x10de => format!(
            "{}.{}.{}.{}",
            (raw >> 22) & 0x3ff,
            (raw >> 14) & 0xff,
            (raw >> 6) & 0xff,
            raw & 0x3f
        ),
        // Mesa encodes its version like VK_MAKE_VERSION(major, minor, patch).
        0x10005 => format!(
            "{}.{}.{}",
            (raw >> 22) & 0x3ff,
            (raw >> 12) & 0x3ff,
            raw & 0xfff
        ),
        _ => format!("0x{raw:08x}"),
    }
}

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

/// Full human-readable report. `include_extensions` prints the complete
/// extension list; the run header only prints the relevant booleans.
pub fn print_device(info: &DeviceInfo, include_extensions: bool) {
    println!("device[{}]: {}", info.index, info.name);
    println!(
        "  vendor_id         0x{:04x} ({})",
        info.vendor_id,
        info.vendor_name()
    );
    println!("  device_id         0x{:04x}", info.device_id);
    println!("  device_type       {:?}", info.device_type);
    println!("  api_version       {}", format_version(info.api_version));
    println!(
        "  driver_id         {:?} (driverID={})",
        info.driver_id,
        info.driver_id.as_raw()
    );
    println!(
        "  driver_version    {} (raw=0x{:08x})",
        info.driver_version_string(),
        info.driver_version_raw
    );
    println!("  device_uuid       {}", info.uuid_string());
    println!("  driver_uuid       {}", info.driver_uuid_string());
    println!(
        "  memory            {} heaps, {} types",
        info.memory_heaps.len(),
        info.memory_type_count
    );
    for heap in &info.memory_heaps {
        println!(
            "    heap[{}] size={} ({:.2} GiB) flags={:?}",
            heap.index,
            heap.size,
            heap.size as f64 / (1024.0 * 1024.0 * 1024.0),
            heap.flags
        );
    }
    println!("  queue_families    {}", info.queue_families.len());
    for family in &info.queue_families {
        println!(
            "    family[{}] {:<32} queues={} timestamp_bits={} granularity={}x{}x{}",
            family.index,
            queue_flags_string(family.flags),
            family.queue_count,
            family.timestamp_valid_bits,
            family.min_image_transfer_granularity.width,
            family.min_image_transfer_granularity.height,
            family.min_image_transfer_granularity.depth
        );
    }

    let l = &info.limits;
    println!("  limits:");
    println!(
        "    maxComputeWorkGroupCount          {:?}",
        l.max_compute_work_group_count
    );
    println!(
        "    maxComputeWorkGroupSize           {:?}",
        l.max_compute_work_group_size
    );
    println!(
        "    maxComputeWorkGroupInvocations    {}",
        l.max_compute_work_group_invocations
    );
    println!(
        "    maxComputeSharedMemorySize        {} bytes",
        l.max_compute_shared_memory_size
    );
    println!(
        "    maxStorageBufferRange             {} bytes",
        l.max_storage_buffer_range
    );
    println!(
        "    maxMemoryAllocationSize           {} bytes",
        l.max_memory_allocation_size
    );
    println!(
        "    minUniformBufferOffsetAlignment   {} bytes",
        l.min_uniform_buffer_offset_alignment
    );
    println!(
        "    minStorageBufferOffsetAlignment   {} bytes",
        l.min_storage_buffer_offset_alignment
    );
    println!(
        "    maxMemoryAllocationCount          {}",
        l.max_memory_allocation_count
    );
    println!(
        "    timestampPeriod                   {} ns",
        l.timestamp_period
    );
    println!(
        "    maxPushConstantsSize              {} bytes",
        l.max_push_constants_size
    );
    println!(
        "    maxDescriptorSetStorageBuffers    {}",
        l.max_descriptor_set_storage_buffers
    );

    println!("  relevant_extensions:");
    for (name, description) in RELEVANT_EXTENSIONS {
        println!(
            "    {:<32} {:<24} {}",
            name,
            if info.has_extension(name) {
                "present"
            } else {
                "absent"
            },
            description
        );
    }

    if include_extensions {
        println!("  extensions ({}):", info.extensions.len());
        for extension in &info.extensions {
            println!("    {extension}");
        }
    }
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
        let version = vk::make_api_version(0, 1, 4, 312);
        assert_eq!(format_version(version), "1.4.312");
    }

    #[test]
    fn queue_flags_are_named() {
        let flags = vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER;
        assert_eq!(queue_flags_string(flags), "COMPUTE|TRANSFER");
        assert_eq!(queue_flags_string(vk::QueueFlags::empty()), "NONE");
    }
}
