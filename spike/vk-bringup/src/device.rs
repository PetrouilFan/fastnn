//! Logical device, compute queue and memory-type selection.
//!
//! One `DeviceContext` exists per selected physical device. It borrows the
//! [`InstanceHandle`] so the instance cannot be destroyed first, and it owns
//! exactly one compute queue. Queue-family selection prefers a dedicated
//! compute family; if none exists it accepts a graphics+compute family.

use ash::vk;

use crate::error::{Error, Result};
use crate::ledger::Ledger;
use crate::platform::InstanceHandle;
use crate::probe::{self, DeviceInfo, QueueFamilyInfo};

pub struct DeviceContext<'a> {
    /// Keeps the instance (and therefore the loader) alive for the device's
    /// whole lifetime; never read directly.
    _instance: &'a InstanceHandle,
    pub device: ash::Device,
    pub family_index: u32,
    pub queue: vk::Queue,
    pub info: DeviceInfo,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub ledger: Ledger,
    destroyed: bool,
}

impl<'a> DeviceContext<'a> {
    pub fn create(instance: &'a InstanceHandle, info: DeviceInfo) -> Result<Self> {
        let physical = instance.physical(info.index)?;
        let family = pick_compute_family(&info).ok_or_else(|| {
            Error::internal(format!(
                "device \"{}\" exposes no compute-capable queue family",
                info.name
            ))
        })?;

        // SAFETY: `physical` belongs to the borrowed instance. `priority` and
        // `queue_infos` live until `create_device` returns, and the queue
        // family index was validated against the device's own properties.
        let priority = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family.index)
            .queue_priorities(&priority)];
        let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        let device = unsafe {
            instance
                .instance()
                .create_device(physical, &device_info, None)
        }
        .map_err(|code| {
            Error::vk(
                "vkCreateDevice",
                code,
                format!(
                    "creating a logical device for \"{}\" on queue family {} ({})",
                    info.name,
                    family.index,
                    probe::queue_flags_string(family.flags)
                ),
            )
        })?;

        // SAFETY: the device is live, the family index was used at creation,
        // and queue index 0 was requested (queue_count >= 1).
        let queue = unsafe { device.get_device_queue(family.index, 0) };
        // SAFETY: `physical` is still owned by `instance` and the device was
        // created from it.
        let memory_properties = unsafe {
            instance
                .instance()
                .get_physical_device_memory_properties(physical)
        };

        let mut ledger = Ledger::default();
        ledger.create("device");

        Ok(Self {
            _instance: instance,
            device,
            family_index: family.index,
            queue,
            info,
            memory_properties,
            ledger,
            destroyed: false,
        })
    }

    /// Find a memory type satisfying `required`, restricted to `type_bits`
    /// (from `VkMemoryRequirements::memory_type_bits`).
    pub fn find_memory_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
        what: &str,
    ) -> Result<u32> {
        for index in 0..self.memory_properties.memory_type_count {
            let candidate = self.memory_properties.memory_types[index as usize];
            if (type_bits & (1 << index)) != 0 && candidate.property_flags.contains(required) {
                return Ok(index);
            }
        }

        let mut available = String::new();
        for index in 0..self.memory_properties.memory_type_count {
            let candidate = self.memory_properties.memory_types[index as usize];
            if type_bits & (1 << index) != 0 {
                available.push_str(&format!(
                    "\n  type[{index}] flags={:?} heap={}",
                    candidate.property_flags, candidate.heap_index
                ));
            }
        }
        Err(Error::internal(format!(
            "no memory type for {what} with required flags {required:?}; compatible types:{available}"
        )))
    }

    /// Refuse allocations the device advertised as too large.
    pub fn check_allocation_size(&self, size: u64, what: &str) -> Result<()> {
        if size > self.info.limits.max_memory_allocation_size {
            return Err(Error::internal(format!(
                "{what} needs {size} bytes, above maxMemoryAllocationSize={} on \"{}\"",
                self.info.limits.max_memory_allocation_size, self.info.name
            )));
        }
        Ok(())
    }

    pub fn teardown(&mut self) {
        if self.destroyed {
            return;
        }
        // Teardown-only global wait: every real submission in this spike is
        // fence-based; this is the final "nothing in flight" guard.
        // SAFETY: the device is live and no other thread uses it.
        if let Err(code) = unsafe { self.device.device_wait_idle() } {
            eprintln!(
                "[fn3vk][warn] vkDeviceWaitIdle during teardown failed: {code:?} ({})",
                code.as_raw()
            );
        }
        // SAFETY: no child objects remain (Engine is dropped first), and the
        // device is destroyed exactly once.
        unsafe { self.device.destroy_device(None) };
        self.ledger.destroy("device");
        self.destroyed = true;
    }
}

impl Drop for DeviceContext<'_> {
    fn drop(&mut self) {
        self.teardown();
    }
}

/// Prefer a compute-only family with the most queues, then any family with
/// COMPUTE (graphics+compute is common on integrated GPUs).
pub fn pick_compute_family(info: &DeviceInfo) -> Option<&QueueFamilyInfo> {
    let mut dedicated: Vec<&QueueFamilyInfo> = info
        .queue_families
        .iter()
        .filter(|f| {
            f.flags.contains(vk::QueueFlags::COMPUTE) && !f.flags.contains(vk::QueueFlags::GRAPHICS)
        })
        .collect();
    dedicated.sort_by_key(|f| std::cmp::Reverse(f.queue_count));
    if let Some(family) = dedicated.first() {
        return Some(family);
    }
    info.queue_families
        .iter()
        .find(|f| f.flags.contains(vk::QueueFlags::COMPUTE))
}
