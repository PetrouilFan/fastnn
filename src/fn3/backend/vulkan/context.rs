//! Buffers, checked ranges and kernel invocation arguments.
//!
//! A [`Buffer`] is a `vk::Buffer` plus the suballocated [`MemoryAllocation`] it
//! is bound to. It owns both: `Drop` destroys the buffer and returns the range
//! to the chunk's free list. All device access goes through checked ranges;
//! transfer and binding ranges are validated *before* a command is recorded so
//! a bad plan produces a message, not a validation error.
//!
//! # Host access
//!
//! Only [`MemoryClass::HostVisible`] buffers have a mapping. The mapping
//! belongs to the chunk, not the buffer, so the pointer is `chunk base +
//! offset`. [`Buffer::write_from`] flushes and [`Buffer::read_into`]
//! invalidates when the memory type is not `HOST_COHERENT`. Coherence is never
//! assumed.

use std::fmt;
use std::sync::Arc;

use ash::vk;

use super::device::Device;
use super::error::{Error, Result};
use super::memory::{MemoryAllocation, MemoryClass};
use crate::inference::backend::vulkan::pipeline::Kernel;

/// What a buffer will be used for. Drives usage flags and the memory class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferDesc {
    /// Logical size in bytes.
    ///
    /// Zero is legal: a zero-element tensor needs a name, not storage. Such a
    /// buffer is backed by at least [`EMPTY_BUFFER_BYTES`] bytes of device
    /// memory because Vulkan forbids a size-0 `vkCreateBuffer`, but it reports
    /// [`Buffer::size`] `0`, every range check rejects a non-empty access, and
    /// a dispatch that binds it is rejected before recording. The dummy bytes
    /// are unreachable through the public API.
    pub size: u64,
    /// Vulkan usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Memory class to allocate from.
    pub class: MemoryClass,
    /// Diagnostic name used in errors and residency records.
    pub name: &'static str,
}

/// Device bytes reserved for a zero-length logical buffer.
///
/// `vkCreateBuffer` requires `size > 0`, so an empty operand gets one aligned
/// word instead of no allocation. It is a real bound allocation (not a null
/// handle), which keeps binding, ownership and teardown uniform; the logical
/// length stays zero so no command can address it.
pub const EMPTY_BUFFER_BYTES: u64 = 4;

impl BufferDesc {
    /// Device-local storage buffer that kernels read and write, and that can be
    /// uploaded and downloaded.
    pub fn device_storage(size: u64, name: &'static str) -> BufferDesc {
        BufferDesc {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            class: MemoryClass::DeviceLocal,
            name,
        }
    }

    /// Host-visible staging buffer used only for transfers.
    pub fn staging(size: u64, name: &'static str) -> BufferDesc {
        BufferDesc {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            class: MemoryClass::HostVisible,
            name,
        }
    }

    /// Device-local uniform buffer.
    pub fn uniform(size: u64, name: &'static str) -> BufferDesc {
        BufferDesc {
            size,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            class: MemoryClass::DeviceLocal,
            name,
        }
    }
}

/// A buffer bound to a suballocated range of device memory.
///
/// # Zero-length buffers
///
/// A zero-element tensor owns a buffer whose logical [`Buffer::size`] is `0`
/// and whose backing allocation is at least [`EMPTY_BUFFER_BYTES`] bytes. The
/// dummy word exists because `vkCreateBuffer` rejects `size = 0`; the logical
/// size is what every range check and every residency record sees, so nothing
/// can read or write past it. A dispatch must not bind an empty buffer: the
/// recorder rejects a zero descriptor range before recording, which is how a
/// kernel bug becomes a message instead of undefined behaviour.
///
/// # Lifetime
///
/// A `Buffer` must not be dropped while a submission that references it is in
/// flight. The synchronous API (`VulkanBackend::upload`/`download`/`dispatch`
/// and every `Submission::wait`) guarantees a wait before returning, and the
/// smoke kernel drops its buffers after the wait. A caller that keeps a
/// submission in flight owns that ordering; `Buffer::drop` is deliberately not
/// deferred behind a reference count so a mistake is visible as a validation
/// error rather than a silent stall.
pub struct Buffer {
    device: Arc<Device>,
    raw: vk::Buffer,
    allocation: MemoryAllocation,
    size: u64,
    usage: vk::BufferUsageFlags,
    class: MemoryClass,
    name: &'static str,
    id: u64,
    /// Ledger kind: device buffers and pooled staging blocks are counted
    /// separately so a test can distinguish tensor memory from transfer memory.
    kind: &'static str,
}

impl Buffer {
    /// Create and bind a buffer.
    pub fn new(device: &Arc<Device>, desc: &BufferDesc) -> Result<Buffer> {
        // Vulkan has no zero-size buffer. Keep the logical size in `size` and
        // reserve one aligned word so the handle is real and bindable; every
        // access is checked against `size`, not against the reservation.
        let allocation_bytes = desc.size.max(EMPTY_BUFFER_BYTES);
        let limits = device.limits();
        if desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            let limit = u64::from(limits.max_storage_buffer_range);
            if allocation_bytes > limit {
                return Err(Error::limit(
                    "vulkan storage buffer range",
                    allocation_bytes,
                    limit,
                    format!(
                        "{} on \"{}\": one storage-buffer binding cannot address more than \
                         maxStorageBufferRange. Split the tensor; the 4 GiB allocation cap is \
                         handled separately by the suballocator",
                        desc.name,
                        device.adapter().name
                    ),
                ));
            }
        }
        if allocation_bytes > limits.max_memory_allocation_size {
            return Err(Error::limit(
                "vulkan device allocation size",
                allocation_bytes,
                limits.max_memory_allocation_size,
                format!("{} on \"{}\"", desc.name, device.adapter().name),
            ));
        }
        if desc.class == MemoryClass::DeviceLocal
            && !desc.usage.intersects(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
        {
            return Err(Error::internal(format!(
                "{}: device-local buffer declares no usable usage flags ({:?})",
                desc.name, desc.usage
            )));
        }

        let create_info = vk::BufferCreateInfo::default()
            .size(allocation_bytes)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: plain buffer creation with a zeroed allocation-callback
        // optional and a validated positive size (`allocation_bytes` is at
        // least `EMPTY_BUFFER_BYTES`); the buffer is destroyed in `Drop` (or on
        // the error paths below) exactly once.
        let raw = unsafe { device.ash().create_buffer(&create_info, None) }.map_err(|code| {
            Error::vk(
                "vkCreateBuffer",
                code,
                format!(
                    "{}: {} bytes logical, {} bytes reserved, usage {:?}",
                    desc.name, desc.size, allocation_bytes, desc.usage
                ),
            )
        })?;

        // SAFETY: `raw` was just created by this device and is live.
        let requirements = unsafe { device.ash().get_buffer_memory_requirements(raw) };
        let min_alignment = if desc.usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
            limits.min_uniform_buffer_offset_alignment
        } else if desc.usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
            limits.min_storage_buffer_offset_alignment
        } else {
            1
        };
        let allocation =
            match device.allocate_range(requirements, desc.class, min_alignment, desc.name) {
                Ok(allocation) => allocation,
                Err(error) => {
                    // SAFETY: nothing was bound to the buffer yet; destroyed
                    // exactly once.
                    unsafe { device.ash().destroy_buffer(raw, None) };
                    return Err(error);
                }
            };

        // SAFETY: `raw` was created without dedicated requirements,
        // `allocation` is a range of at least `requirements.size` bytes in the
        // memory type the requirements allowed, and its offset honours
        // `requirements.alignment` (the allocator aligned it).
        if let Err(error) = super::error::check(
            "vkBindBufferMemory",
            unsafe {
                device.ash().bind_buffer_memory(
                    raw,
                    chunk_memory(device, &allocation),
                    allocation.offset,
                )
            },
            format!("{}: binding a {} byte range", desc.name, allocation.size),
        ) {
            // SAFETY: the failed bind leaves both objects owned by us; free in
            // reverse creation order. The range is still live in the chunk.
            unsafe { device.ash().destroy_buffer(raw, None) };
            let _ = device.free_range(allocation);
            return Err(error);
        }

        let kind = if desc.class == MemoryClass::HostVisible {
            "staging_buffer"
        } else {
            "device_buffer"
        };
        device.ledger().create(kind);
        device.ledger().allocate_bytes(kind, desc.size);

        Ok(Buffer {
            device: Arc::clone(device),
            raw,
            allocation,
            size: desc.size,
            usage: desc.usage,
            class: desc.class,
            name: desc.name,
            id: device.next_buffer_id(),
            kind,
        })
    }

    /// Requested size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// `true` for a zero-element buffer.
    ///
    /// Such a buffer owns a dummy allocation but no addressable byte; see the
    /// type-level documentation.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Suballocation size in bytes (at least [`Buffer::size`], alignment
    /// padding included).
    pub fn allocation_size(&self) -> u64 {
        self.allocation.size
    }

    /// Diagnostic name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Process-unique buffer id used in residency records.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Memory class.
    pub fn class(&self) -> MemoryClass {
        self.class
    }

    /// Usage flags.
    pub fn usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }

    /// Raw handle, for command recording.
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    /// `true` if the buffer has a host mapping.
    pub fn is_host_visible(&self) -> bool {
        self.class == MemoryClass::HostVisible
    }

    /// Check that `offset..offset+len` lies inside the buffer.
    pub fn check_range(&self, offset: u64, len: u64, what: &str) -> Result<()> {
        check_range(&format!("{} ({what})", self.name), offset, len, self.size)
    }

    /// Write host bytes into a mapped buffer and flush if necessary.
    pub fn write_from(&self, offset: u64, bytes: &[u8]) -> Result<()> {
        self.check_range(offset, bytes.len() as u64, "host upload")?;
        if bytes.is_empty() {
            // A zero-length write to a zero-length buffer is legal and must not
            // ask a device-local buffer for a host mapping.
            return Ok(());
        }
        let pointer = self.mapped_pointer()?;
        // SAFETY: `pointer` addresses `self.size` bytes of a host-visible
        // mapping owned by the chunk; the range was checked against the buffer
        // size, and the source is a distinct live slice that cannot overlap a
        // driver-owned mapping. The caller owns this buffer exclusively for the
        // duration of the write (staging leases are exclusive; device-local
        // buffers are rejected by `mapped_pointer`).
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                pointer.add(offset as usize),
                bytes.len(),
            );
        }
        self.flush(offset, bytes.len() as u64)
    }

    /// Invalidate (if necessary) and copy device-written bytes into `dst`.
    pub fn read_into(&self, offset: u64, dst: &mut [u8]) -> Result<()> {
        self.check_range(offset, dst.len() as u64, "host readback")?;
        if dst.is_empty() {
            return Ok(());
        }
        self.invalidate(offset, dst.len() as u64)?;
        let pointer = self.mapped_pointer()?;
        // SAFETY: as `write_from`, with the direction reversed; the range was
        // checked and the destination is a distinct live slice. The caller must
        // have waited for the submission that wrote the range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                pointer.add(offset as usize),
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
        Ok(())
    }

    /// Flush a host-written range (no-op for coherent memory).
    pub fn flush(&self, offset: u64, len: u64) -> Result<()> {
        self.check_range(offset, len, "flush")?;
        self.device.flush_allocation(&self.allocation, offset, len)
    }

    /// Invalidate a host-read range (no-op for coherent memory).
    pub fn invalidate(&self, offset: u64, len: u64) -> Result<()> {
        self.check_range(offset, len, "invalidate")?;
        self.device
            .invalidate_allocation(&self.allocation, offset, len)
    }

    /// `true` for coherent host-visible memory.
    pub fn is_coherent(&self) -> bool {
        self.device.allocation_is_coherent(&self.allocation)
    }

    fn mapped_pointer(&self) -> Result<*mut u8> {
        self.device
            .allocation_pointer(&self.allocation)
            .ok_or_else(|| {
                Error::internal(format!(
                    "{} is a {} buffer without a host mapping; use a staging buffer or a \
                     device-to-device copy",
                    self.name,
                    self.class.name()
                ))
            })
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the buffer is no longer referenced by any in-flight command
        // buffer (submitters keep their device and waited before reuse; the
        // owner drops buffers only after its submission completed). It is
        // destroyed exactly once, before its memory range is returned.
        unsafe { self.device.ash().destroy_buffer(self.raw, None) };
        self.device.ledger().destroy(self.kind);
        self.device.ledger().free_bytes(self.kind, self.size);
        if let Err(error) = self.device.free_range(self.allocation) {
            eprintln!("[fn3vk][warn] returning a buffer range to the allocator failed: {error}");
        }
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("name", &self.name)
            .field("size", &self.size)
            .field("class", &self.class)
            .field("offset", &self.allocation.offset)
            .field("chunk", &self.allocation.chunk)
            .finish_non_exhaustive()
    }
}

/// Role a buffer plays in one dispatch, for residency records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingRole {
    /// Read by the kernel and uploaded by the host.
    Input,
    /// Written by the kernel and downloaded by the host.
    Output,
    /// Read by the kernel and already device-resident.
    Shared,
    /// Uniform/parameter data.
    Uniform,
    /// Scratch owned by the backend.
    Scratch,
}

impl BindingRole {
    /// Stable label used in the evidence logs.
    pub const fn name(self) -> &'static str {
        match self {
            BindingRole::Input => "input",
            BindingRole::Output => "output",
            BindingRole::Shared => "shared",
            BindingRole::Uniform => "uniform",
            BindingRole::Scratch => "scratch",
        }
    }
}

/// One buffer bound to one descriptor.
#[derive(Debug, Clone, Copy)]
pub struct BindingArg<'a> {
    /// Descriptor binding number.
    pub binding: u32,
    /// Role in this dispatch.
    pub role: BindingRole,
    /// Buffer to bind.
    pub buffer: &'a Buffer,
    /// Byte offset inside the buffer; must satisfy the descriptor alignment.
    pub offset: u64,
    /// Bytes addressed from `offset`.
    pub range: u64,
}

impl<'a> BindingArg<'a> {
    /// Bind a whole buffer.
    pub fn whole(binding: u32, role: BindingRole, buffer: &'a Buffer) -> BindingArg<'a> {
        BindingArg {
            binding,
            role,
            buffer,
            offset: 0,
            range: buffer.size(),
        }
    }

    /// Bind a sub-range.
    pub fn range(
        binding: u32,
        role: BindingRole,
        buffer: &'a Buffer,
        offset: u64,
        range: u64,
    ) -> BindingArg<'a> {
        BindingArg {
            binding,
            role,
            buffer,
            offset,
            range,
        }
    }
}

/// Everything one kernel dispatch needs.
#[derive(Debug, Clone, Copy)]
pub struct KernelInvocation<'a> {
    /// Compiled kernel.
    pub kernel: &'a Kernel,
    /// Declared bindings, exactly one per descriptor.
    pub bindings: &'a [BindingArg<'a>],
    /// Push-constant bytes; must match the kernel's declared size.
    pub push_constants: &'a [u8],
    /// Workgroup counts per axis, already derived from the grid function.
    pub workgroups: [u32; 3],
}

/// Check `offset..offset+len` against `capacity`.
pub fn check_range(what: &str, offset: u64, len: u64, capacity: u64) -> Result<()> {
    let end = offset.checked_add(len).ok_or_else(|| {
        Error::internal(format!(
            "{what}: offset {offset} plus length {len} overflows u64"
        ))
    })?;
    if end > capacity {
        return Err(Error::internal(format!(
            "{what}: range {offset}..{end} is outside a {capacity}-byte buffer"
        )));
    }
    Ok(())
}

/// Access the `vk::DeviceMemory` behind an allocation for `vkBindBufferMemory`.
///
/// The allocator owns chunks; a `MemoryAllocation` identifies one. This helper
/// is the single place that translates the id back to the handle.
fn chunk_memory(device: &Arc<Device>, allocation: &MemoryAllocation) -> vk::DeviceMemory {
    device
        .chunk_memory(allocation)
        .unwrap_or(vk::DeviceMemory::null())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_ranges_reject_overflow_and_overrun() {
        assert!(check_range("t", 0, 16, 16).is_ok());
        assert!(check_range("t", 8, 8, 16).is_ok());
        assert!(check_range("t", 9, 8, 16).is_err());
        assert!(check_range("t", u64::MAX, 2, u64::MAX).is_err());
        assert!(check_range("t", 17, 0, 16).is_err());
    }
}
