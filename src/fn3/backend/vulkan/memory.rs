//! Suballocation over `vkAllocateMemory` chunks.
//!
//! # Why a suballocator is mandatory
//!
//! The GTX 1080 Ti reports `maxMemoryAllocationSize` ≈ 4 GiB while the
//! device-local heap is 11 GiB. One `vkAllocateMemory` per tensor therefore
//! caps a single tensor at ≈4 GiB *and* burns an allocation per tensor; the
//! FN3-05 spike measured exactly this shape. The production allocator owns a
//! small set of large chunks per memory type and binds buffers at aligned
//! offsets inside them.
//!
//! # Policy
//!
//! | Class | Chunk floor | Chunk ceiling |
//! |---|---|---|
//! | [`MemoryClass::DeviceLocal`] | 64 MiB | 512 MiB (configurable) |
//! | [`MemoryClass::HostVisible`] | 16 MiB | 64 MiB (configurable) |
//!
//! A new chunk is four times the request, clamped into the band, so a graph
//! that allocates many similarly-sized tensors converges on one chunk instead
//! of one allocation per tensor. A request larger than the ceiling gets a
//! *dedicated* chunk of exactly the aligned size (still bounded by
//! `maxMemoryAllocationSize`); a request larger than the device cap fails with
//! [`Error::Limit`] naming both numbers and the device.
//!
//! # Free list
//!
//! Each chunk keeps a sorted list of regions; allocation is first-fit with
//! alignment, split on the tail, and free coalesces with both neighbours. There
//! is no defragmentation in the alpha: chunks are reused, never moved. The
//! reported `largest_free_range` is what a caller needs to distinguish "the
//! device is full" from "the free space is fragmented".
//!
//! # Coherence
//!
//! Host-visible chunks are mapped once at creation. If the memory type is not
//! `HOST_COHERENT`, [`MemoryAllocator::flush`] and
//! [`MemoryAllocator::invalidate`] perform the required range operations with
//! `nonCoherentAtomSize` alignment; the staging code calls them. Coherence is
//! never assumed.

use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use super::error::{Error, Result};

/// The two allocation classes the backend exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    /// Fast device memory for tensors and kernel I/O.
    DeviceLocal,
    /// Host-visible memory for staging (upload/readback) and mapped reads.
    HostVisible,
}

impl MemoryClass {
    /// Required memory-type flags for this class.
    pub const fn required_flags(self) -> vk::MemoryPropertyFlags {
        match self {
            MemoryClass::DeviceLocal => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            MemoryClass::HostVisible => vk::MemoryPropertyFlags::HOST_VISIBLE,
        }
    }

    /// Preferred (but not required) flags. A host-visible type that is also
    /// coherent avoids explicit flushes.
    pub const fn preferred_flags(self) -> vk::MemoryPropertyFlags {
        match self {
            MemoryClass::DeviceLocal => vk::MemoryPropertyFlags::empty(),
            MemoryClass::HostVisible => vk::MemoryPropertyFlags::HOST_COHERENT,
        }
    }

    /// Stable name used in errors.
    pub const fn name(self) -> &'static str {
        match self {
            MemoryClass::DeviceLocal => "device-local",
            MemoryClass::HostVisible => "host-visible",
        }
    }
}

/// A suballocated range inside one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAllocation {
    /// Identifier of the owning chunk. Opaque to callers.
    pub chunk: u64,
    /// Byte offset inside the chunk; always satisfies the requested alignment.
    pub offset: u64,
    /// Allocated size in bytes, exactly as requested.
    pub size: u64,
    /// Vulkan memory type index (for diagnostics and `VkMappedMemoryRange`).
    pub memory_type_index: u32,
}

/// Per-chunk diagnostics.
#[derive(Debug, Clone)]
pub struct ChunkStats {
    /// Chunk identifier.
    pub id: u64,
    /// Memory type index.
    pub memory_type_index: u32,
    /// Total chunk size in bytes.
    pub size: u64,
    /// Bytes currently allocated from this chunk.
    pub live_bytes: u64,
    /// Number of free regions.
    pub free_regions: usize,
    /// Largest free region in bytes.
    pub largest_free_range: u64,
    /// `true` if the chunk is permanently mapped (host-visible).
    pub mapped: bool,
    /// `true` if the memory type is host-coherent.
    pub coherent: bool,
    /// `true` if the memory type is `HOST_VISIBLE` (staging).
    pub host_visible: bool,
}

/// Allocator counters. This is the format #62/#72 consume.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Number of live `vkAllocateMemory` chunks.
    pub chunks: usize,
    /// Bytes reserved by chunks (the device-level footprint).
    pub reserved_bytes: u64,
    /// Bytes handed out to live buffers.
    pub live_bytes: u64,
    /// Subset of `live_bytes` in device-local memory (tensors, kernel I/O).
    pub live_device_local_bytes: u64,
    /// Subset of `live_bytes` in host-visible memory (staging blocks).
    pub live_host_visible_bytes: u64,
    /// High-water mark of `live_bytes`.
    pub peak_live_bytes: u64,
    /// Bytes free inside chunks (`reserved_bytes - live_bytes`).
    pub free_bytes: u64,
    /// Number of successful suballocations over the allocator's lifetime.
    pub allocation_count: u64,
    /// Number of frees over the allocator's lifetime.
    pub free_count: u64,
    /// Largest single free range across all chunks.
    pub largest_free_range: u64,
    /// Per-chunk detail, ordered by creation.
    pub chunk_list: Vec<ChunkStats>,
}

#[derive(Debug, Clone)]
struct Region {
    offset: u64,
    size: u64,
    free: bool,
}

impl Region {
    fn end(&self) -> u64 {
        self.offset + self.size
    }
}

#[derive(Debug)]
struct Chunk {
    id: u64,
    memory: vk::DeviceMemory,
    type_index: u32,
    size: u64,
    /// Base address of the persistent mapping, as an integer so the struct is
    /// `Send` (`NonNull<u8>` is not).
    mapped: Option<usize>,
    coherent: bool,
    regions: Vec<Region>,
}

/// The chunk allocator.
#[derive(Debug)]
pub struct MemoryAllocator {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    max_allocation_size: u64,
    max_memory_allocation_count: u32,
    non_coherent_atom_size: u64,
    device_local_chunk_max: u64,
    host_chunk_max: u64,
    chunks: Vec<Chunk>,
    next_chunk_id: u64,
    next_mapping_id: AtomicU64,
    live_bytes: u64,
    peak_live_bytes: u64,
    allocation_count: u64,
    free_count: u64,
}

impl MemoryAllocator {
    /// Create an empty allocator.
    ///
    /// `max_chunk_bytes` caps a device-local chunk; host-visible chunks are
    /// capped at a quarter of it (never above 64 MiB) because staging is
    /// transient. Both caps are clamped to `maxMemoryAllocationSize`.
    pub fn new(
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        max_allocation_size: u64,
        max_memory_allocation_count: u32,
        non_coherent_atom_size: u64,
        max_chunk_bytes: u64,
    ) -> MemoryAllocator {
        let device_local_chunk_max = max_chunk_bytes.max(1).min(max_allocation_size);
        let host_chunk_max = (max_chunk_bytes / 4).clamp(1, 64 * 1024 * 1024);
        MemoryAllocator {
            memory_properties,
            max_allocation_size,
            max_memory_allocation_count,
            non_coherent_atom_size: non_coherent_atom_size.max(1),
            device_local_chunk_max,
            host_chunk_max,
            chunks: Vec::new(),
            next_chunk_id: 0,
            next_mapping_id: AtomicU64::new(0),
            live_bytes: 0,
            peak_live_bytes: 0,
            allocation_count: 0,
            free_count: 0,
        }
    }

    /// Maximum size of one `vkAllocateMemory` on this device.
    pub fn max_allocation_size(&self) -> u64 {
        self.max_allocation_size
    }

    /// Resolve a memory type for `class` that is compatible with `type_bits`.
    pub fn find_memory_type(&self, type_bits: u32, class: MemoryClass) -> Result<u32> {
        let required = class.required_flags();
        let preferred = class.preferred_flags();
        let mut fallback = None;
        for index in 0..self.memory_properties.memory_type_count {
            let candidate = self.memory_properties.memory_types[index as usize];
            if type_bits & (1 << index) == 0 || !candidate.property_flags.contains(required) {
                continue;
            }
            if candidate.property_flags.contains(preferred) {
                return Ok(index);
            }
            if fallback.is_none() {
                fallback = Some(index);
            }
        }
        fallback.ok_or_else(|| {
            let mut compatible = String::new();
            for index in 0..self.memory_properties.memory_type_count {
                if type_bits & (1 << index) != 0 {
                    let candidate = self.memory_properties.memory_types[index as usize];
                    compatible.push_str(&format!(
                        "\n  type[{index}] flags={:?} heap={}",
                        candidate.property_flags, candidate.heap_index
                    ));
                }
            }
            Error::internal(format!(
                "no {} memory type satisfies the buffer's memory_type_bits {type_bits:#x} \
                 and required flags {required:?}; compatible types:{compatible}",
                class.name()
            ))
        })
    }

    /// Allocate `requirements.size` bytes of `class`, aligned to at least
    /// `min_alignment` (a descriptor-offset requirement), reusing a free range
    /// when one fits.
    pub fn allocate(
        &mut self,
        device: &ash::Device,
        requirements: vk::MemoryRequirements,
        class: MemoryClass,
        min_alignment: u64,
        what: &str,
    ) -> Result<MemoryAllocation> {
        if requirements.size == 0 {
            return Err(Error::internal(format!(
                "{what}: refusing a zero-byte device allocation"
            )));
        }
        if requirements.size > self.max_allocation_size {
            return Err(Error::limit(
                "vulkan device allocation size",
                requirements.size,
                self.max_allocation_size,
                format!(
                    "{what} needs one contiguous {}-byte memory object, above \
                     maxMemoryAllocationSize. The allocator suballocates inside chunks, but a \
                     single buffer must still fit one vkAllocateMemory; split the tensor or \
                     reduce the shape. live={} reserved={} chunks={}",
                    requirements.size,
                    self.live_bytes,
                    self.reserved_bytes(),
                    self.chunks.len()
                ),
            ));
        }

        let type_index = self.find_memory_type(requirements.memory_type_bits, class)?;
        let alignment = requirements.alignment.max(min_alignment).max(1);
        let size = align_up(requirements.size, alignment)?;

        if let Some((chunk_index, offset)) = self.find_fit(type_index, size, alignment) {
            return self.commit(chunk_index, offset, size, what);
        }

        // No range fits: add a chunk and retry inside it.
        let chunk_size = self.chunk_size_for(class, size, alignment)?;
        let chunk_index = self.create_chunk(device, type_index, chunk_size, class, what)?;
        let (chunk_index, offset) = self
            .find_fit(type_index, size, alignment)
            .filter(|(index, _)| *index == chunk_index)
            .ok_or_else(|| {
                Error::internal(format!(
                    "{what}: a fresh {chunk_size}-byte chunk did not fit a {size}-byte \
                     allocation (internal allocator invariant violated)"
                ))
            })?;
        self.commit(chunk_index, offset, size, what)
    }

    /// Return a range to its chunk and coalesce free neighbours.
    pub fn free(&mut self, allocation: MemoryAllocation) -> Result<()> {
        let chunk = self
            .chunks
            .iter_mut()
            .find(|chunk| chunk.id == allocation.chunk)
            .ok_or_else(|| {
                Error::internal(format!(
                    "freeing an allocation from chunk {} which is not live (double free?)",
                    allocation.chunk
                ))
            })?;
        let position = chunk
            .regions
            .iter()
            .position(|region| {
                region.offset == allocation.offset && region.size == allocation.size && !region.free
            })
            .ok_or_else(|| {
                Error::internal(format!(
                    "chunk {} has no live region at offset {} of {} bytes (double free or \
                     corrupt free list)",
                    allocation.chunk, allocation.offset, allocation.size
                ))
            })?;
        chunk.regions[position].free = true;
        // Coalesce with the next region, then with the previous one.
        if position + 1 < chunk.regions.len() && chunk.regions[position + 1].free {
            let next = chunk.regions.remove(position + 1);
            chunk.regions[position].size += next.size;
        }
        if position > 0 && chunk.regions[position - 1].free {
            let merged = chunk.regions.remove(position);
            chunk.regions[position - 1].size += merged.size;
        }
        self.live_bytes = self.live_bytes.saturating_sub(allocation.size);
        self.free_count += 1;
        Ok(())
    }

    /// Base pointer of a mapped range, or `None` for device-local chunks.
    pub fn pointer(&self, allocation: &MemoryAllocation) -> Option<*mut u8> {
        let chunk = self
            .chunks
            .iter()
            .find(|chunk| chunk.id == allocation.chunk)?;
        let base = chunk.mapped?;
        let end = allocation.offset.checked_add(allocation.size)?;
        if end > chunk.size {
            return None;
        }
        // SAFETY: `base` came from `vkMapMemory(memory, 0, WHOLE_SIZE)` for
        // this chunk; the chunk is kept alive and mapped until
        // `MemoryAllocator::destroy`; `offset + size` was checked against the
        // chunk size, so the resulting pointer is inside the mapping.
        Some(unsafe { (base as *mut u8).add(allocation.offset as usize) })
    }

    /// `true` when the allocation's memory type is host-coherent.
    pub fn is_coherent(&self, allocation: &MemoryAllocation) -> bool {
        self.chunks
            .iter()
            .find(|chunk| chunk.id == allocation.chunk)
            .map(|chunk| chunk.coherent)
            .unwrap_or(false)
    }

    /// The `vk::DeviceMemory` chunk backing an allocation.
    pub fn chunk_memory(&self, allocation: &MemoryAllocation) -> Option<vk::DeviceMemory> {
        self.chunks
            .iter()
            .find(|chunk| chunk.id == allocation.chunk)
            .map(|chunk| chunk.memory)
    }

    /// `true` when the allocation lives in a mapped chunk.
    pub fn is_mapped(&self, allocation: &MemoryAllocation) -> bool {
        self.chunks
            .iter()
            .find(|chunk| chunk.id == allocation.chunk)
            .map(|chunk| chunk.mapped.is_some())
            .unwrap_or(false)
    }

    /// Flush a host-written range to the device.
    ///
    /// A no-op for coherent memory. The range is expanded to
    /// `nonCoherentAtomSize` boundaries as Vulkan requires.
    pub fn flush(
        &self,
        device: &ash::Device,
        allocation: &MemoryAllocation,
        offset: u64,
        len: u64,
    ) -> Result<()> {
        self.host_range_op(device, allocation, offset, len, true)
    }

    /// Invalidate a host-read range so reads see device writes.
    ///
    /// A no-op for coherent memory.
    pub fn invalidate(
        &self,
        device: &ash::Device,
        allocation: &MemoryAllocation,
        offset: u64,
        len: u64,
    ) -> Result<()> {
        self.host_range_op(device, allocation, offset, len, false)
    }

    fn host_range_op(
        &self,
        device: &ash::Device,
        allocation: &MemoryAllocation,
        offset: u64,
        len: u64,
        flush: bool,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let chunk = self
            .chunks
            .iter()
            .find(|chunk| chunk.id == allocation.chunk)
            .ok_or_else(|| Error::internal("mapped range references a dead chunk"))?;
        if chunk.mapped.is_none() {
            return Err(Error::internal(
                "flush/invalidate requested for a chunk that is not mapped",
            ));
        }
        if chunk.coherent {
            return Ok(());
        }
        if offset > allocation.size || len > allocation.size - offset {
            return Err(Error::internal(format!(
                "flush/invalidate range {offset}..{} is outside the {}-byte allocation",
                offset.saturating_add(len),
                allocation.size
            )));
        }
        let atom = self.non_coherent_atom_size;
        let absolute_start = alignment_floor(allocation.offset + offset, atom);
        let absolute_end = align_up(allocation.offset + offset + len, atom)?.min(chunk.size);
        let mut range = vk::MappedMemoryRange::default()
            .memory(chunk.memory)
            .offset(absolute_start)
            .size(absolute_end - absolute_start);
        if range.offset == 0 && range.size == chunk.size {
            range.size = vk::WHOLE_SIZE;
        }
        let ranges = [range];
        // SAFETY: `chunk.memory` is a live allocation of this device, the
        // chunk is currently mapped, and the range lies inside the mapping.
        let result = if flush {
            unsafe { device.flush_mapped_memory_ranges(&ranges) }
        } else {
            unsafe { device.invalidate_mapped_memory_ranges(&ranges) }
        };
        match result {
            Ok(()) => Ok(()),
            Err(code) => Err(Error::vk(
                if flush {
                    "vkFlushMappedMemoryRanges"
                } else {
                    "vkInvalidateMappedMemoryRanges"
                },
                code,
                format!(
                    "range {}..{} of chunk {} (non-coherent memory type {}); \
                     nonCoherentAtomSize={atom}",
                    range.offset,
                    range.offset + range.size,
                    chunk.id,
                    allocation.memory_type_index
                ),
            )),
        }
    }

    /// Snapshot of the counters.
    pub fn stats(&self) -> MemoryStats {
        let mut chunk_list = Vec::with_capacity(self.chunks.len());
        let mut largest_free_range = 0;
        let mut free_bytes = 0;
        let mut live_device_local_bytes = 0;
        let mut live_host_visible_bytes = 0;
        for chunk in &self.chunks {
            let mut live = 0;
            let mut free_regions = 0;
            let mut chunk_largest = 0;
            for region in &chunk.regions {
                if region.free {
                    free_regions += 1;
                    chunk_largest = chunk_largest.max(region.size);
                } else {
                    live += region.size;
                }
            }
            let host_visible = self.memory_properties.memory_types[chunk.type_index as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            if host_visible {
                live_host_visible_bytes += live;
            } else {
                live_device_local_bytes += live;
            }
            free_bytes += chunk.size.saturating_sub(live);
            largest_free_range = largest_free_range.max(chunk_largest);
            chunk_list.push(ChunkStats {
                id: chunk.id,
                memory_type_index: chunk.type_index,
                size: chunk.size,
                live_bytes: live,
                free_regions,
                largest_free_range: chunk_largest,
                mapped: chunk.mapped.is_some(),
                coherent: chunk.coherent,
                host_visible,
            });
        }
        MemoryStats {
            chunks: self.chunks.len(),
            reserved_bytes: self.reserved_bytes(),
            live_bytes: self.live_bytes,
            live_device_local_bytes,
            live_host_visible_bytes,
            peak_live_bytes: self.peak_live_bytes,
            free_bytes,
            allocation_count: self.allocation_count,
            free_count: self.free_count,
            largest_free_range,
            chunk_list,
        }
    }

    /// Unmap and free every chunk. The device must be idle.
    pub fn destroy(&mut self, device: &ash::Device) {
        for chunk in self.chunks.drain(..) {
            if chunk.mapped.is_some() {
                // SAFETY: the chunk was mapped by us with `vkMapMemory` and is
                // unmapped exactly once; no buffer referencing it is alive
                // because chunks outlive their suballocations.
                unsafe { device.unmap_memory(chunk.memory) };
            }
            // SAFETY: the memory object was allocated by us and is freed
            // exactly once; no buffer is bound to it (all suballocations were
            // freed, and buffers hold the device alive).
            unsafe { device.free_memory(chunk.memory, None) };
        }
        self.live_bytes = 0;
    }

    /// A process-unique id for a new memory mapping (diagnostics only).
    pub fn next_mapping_id(&self) -> u64 {
        self.next_mapping_id.fetch_add(1, Ordering::Relaxed)
    }

    fn reserved_bytes(&self) -> u64 {
        self.chunks.iter().map(|chunk| chunk.size).sum()
    }

    fn find_fit(&self, type_index: u32, size: u64, alignment: u64) -> Option<(usize, u64)> {
        for (chunk_index, chunk) in self.chunks.iter().enumerate() {
            if chunk.type_index != type_index {
                continue;
            }
            if let Some(offset) = first_fit(&chunk.regions, size, alignment) {
                return Some((chunk_index, offset));
            }
        }
        None
    }

    fn commit(
        &mut self,
        chunk_index: usize,
        offset: u64,
        size: u64,
        what: &str,
    ) -> Result<MemoryAllocation> {
        let chunk = &mut self.chunks[chunk_index];
        let position = chunk
            .regions
            .iter()
            .position(|region| {
                region.free && offset >= region.offset && offset + size <= region.end()
            })
            .ok_or_else(|| {
                Error::internal(format!(
                    "{what}: allocator chose offset {offset} but the free list disagrees \
                     (internal invariant violated)"
                ))
            })?;
        let region = chunk.regions[position].clone();
        let before = offset - region.offset;
        let after = region.end() - (offset + size);
        let mut replacement = Vec::with_capacity(3);
        if before > 0 {
            replacement.push(Region {
                offset: region.offset,
                size: before,
                free: true,
            });
        }
        replacement.push(Region {
            offset,
            size,
            free: false,
        });
        if after > 0 {
            replacement.push(Region {
                offset: offset + size,
                size: after,
                free: true,
            });
        }
        chunk.regions.splice(position..=position, replacement);

        self.live_bytes += size;
        self.peak_live_bytes = self.peak_live_bytes.max(self.live_bytes);
        self.allocation_count += 1;
        Ok(MemoryAllocation {
            chunk: chunk.id,
            offset,
            size,
            memory_type_index: chunk.type_index,
        })
    }

    fn chunk_size_for(&self, class: MemoryClass, request: u64, alignment: u64) -> Result<u64> {
        let (floor, ceiling) = match class {
            MemoryClass::DeviceLocal => (64 * 1024 * 1024, self.device_local_chunk_max),
            MemoryClass::HostVisible => (16 * 1024 * 1024, self.host_chunk_max),
        };
        let needed = align_up(request, alignment.max(4096))?;
        if needed > self.max_allocation_size {
            return Err(Error::limit(
                "vulkan device allocation size",
                needed,
                self.max_allocation_size,
                format!(
                    "a {} chunk for a {request}-byte allocation exceeds \
                     maxMemoryAllocationSize",
                    class.name()
                ),
            ));
        }
        if needed > ceiling {
            return Ok(needed);
        }
        let target = align_up(request.saturating_mul(4).max(floor), 1024 * 1024)?;
        Ok(target.clamp(needed, ceiling))
    }

    fn create_chunk(
        &mut self,
        device: &ash::Device,
        type_index: u32,
        size: u64,
        class: MemoryClass,
        what: &str,
    ) -> Result<usize> {
        if self.chunks.len() as u64 >= u64::from(self.max_memory_allocation_count) {
            return Err(Error::limit(
                "vulkan memory allocation count",
                self.chunks.len() as u64 + 1,
                u64::from(self.max_memory_allocation_count),
                format!("creating a chunk for {what}"),
            ));
        }
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(type_index);
        // SAFETY: `size` was clamped to maxMemoryAllocationSize and
        // `type_index` came from this device's memory-type table; no dedicated
        // allocation pNext is used, so the result is a normal device memory
        // object owned by this allocator.
        let memory = unsafe { device.allocate_memory(&allocate_info, None) }.map_err(|code| {
            Error::vk_allocation(
                "vkAllocateMemory",
                code,
                allocate_info.allocation_size,
                format!(
                    "{what}: {size}-byte {} chunk, memory type {type_index}; live={} \
                     reserved={} chunks={}",
                    class.name(),
                    self.live_bytes,
                    self.reserved_bytes(),
                    self.chunks.len()
                ),
            )
        })?;

        let properties = self.memory_properties.memory_types[type_index as usize].property_flags;
        let coherent = properties.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        let mapped = if properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            // SAFETY: the memory object is live, host-visible, and the mapping
            // covers the whole allocation with offset 0 (satisfying
            // minMemoryMapAlignment). The mapping is unmapped exactly once in
            // `destroy`.
            match unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(pointer) => Some(pointer as usize),
                Err(code) => {
                    // SAFETY: the failed mapping did not take ownership of the
                    // memory object; free it exactly once.
                    unsafe { device.free_memory(memory, None) };
                    return Err(Error::vk_allocation(
                        "vkMapMemory",
                        code,
                        size,
                        format!(
                            "{what}: mapping the whole {size}-byte {} chunk",
                            class.name()
                        ),
                    ));
                }
            }
        } else {
            None
        };

        let id = self.next_chunk_id;
        self.next_chunk_id += 1;
        self.chunks.push(Chunk {
            id,
            memory,
            type_index,
            size,
            mapped,
            coherent,
            regions: vec![Region {
                offset: 0,
                size,
                free: true,
            }],
        });
        Ok(self.chunks.len() - 1)
    }
}

/// First-fit search over a sorted region list. Returns the aligned offset.
fn first_fit(regions: &[Region], size: u64, alignment: u64) -> Option<u64> {
    for region in regions {
        if !region.free {
            continue;
        }
        let offset = align_up(region.offset, alignment).ok()?;
        if offset.checked_add(size)? <= region.end() {
            return Some(offset);
        }
    }
    None
}

/// Round `value` up to the next multiple of `alignment`.
fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Ok(value);
    }
    let remainder = value % alignment;
    if remainder == 0 {
        return Ok(value);
    }
    value.checked_add(alignment - remainder).ok_or_else(|| {
        Error::internal(format!(
            "aligning {value} bytes to {alignment} overflows u64"
        ))
    })
}

fn alignment_floor(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        value
    } else {
        value - (value % alignment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regions() -> Vec<Region> {
        vec![Region {
            offset: 0,
            size: 1024,
            free: true,
        }]
    }

    #[test]
    fn first_fit_respects_alignment_and_skips_used_regions() {
        let list = vec![
            Region {
                offset: 0,
                size: 16,
                free: false,
            },
            Region {
                offset: 16,
                size: 48,
                free: true,
            },
            Region {
                offset: 64,
                size: 960,
                free: true,
            },
        ];
        assert_eq!(first_fit(&list, 16, 16), Some(16));
        assert_eq!(first_fit(&list, 64, 64), Some(64));
        assert_eq!(first_fit(&list, 65, 1), Some(64));
        assert_eq!(first_fit(&list, 1000, 1), None);
    }

    #[test]
    fn align_up_reports_overflow() {
        assert_eq!(align_up(0, 16).unwrap(), 0);
        assert_eq!(align_up(1, 16).unwrap(), 16);
        assert_eq!(align_up(16, 16).unwrap(), 16);
        assert!(align_up(u64::MAX, 16).is_err());
    }

    #[test]
    fn driver_oom_reports_the_chunk_size_not_the_suballocation_size() {
        unsafe extern "system" fn oom(
            _: vk::Device,
            _: *const vk::MemoryAllocateInfo<'_>,
            _: *const vk::AllocationCallbacks<'_>,
            _: *mut vk::DeviceMemory,
        ) -> vk::Result {
            vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
        }

        // SAFETY: the only Vulkan call made by this test uses the exact
        // vkAllocateMemory ABI; the stub ignores the dummy handle and never
        // dereferences or writes to any of the arguments.
        let device = unsafe {
            ash::Device::load_with(
                |name| match name.to_bytes() {
                    b"vkAllocateMemory" => (oom as vk::PFN_vkAllocateMemory) as *const _,
                    _ => std::ptr::null(),
                },
                vk::Device::null(),
            )
        };
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 1,
            ..Default::default()
        };
        properties.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        const CHUNK_BYTES: u64 = 64 * 1024 * 1024;
        let mut allocator = MemoryAllocator::new(properties, CHUNK_BYTES, 1, 1, CHUNK_BYTES);
        let error = allocator
            .allocate(
                &device,
                vk::MemoryRequirements::default()
                    .size(1024)
                    .alignment(256)
                    .memory_type_bits(1),
                MemoryClass::DeviceLocal,
                1,
                "test buffer",
            )
            .expect_err("vkAllocateMemory returns OOM");

        match error.to_fn3() {
            crate::inference::error::Fn3Error::Allocation { bytes, detail } => {
                assert_eq!(bytes, CHUNK_BYTES);
                assert!(detail.contains("vkAllocateMemory"), "{detail}");
            }
            other => panic!("expected a sized allocation failure, found {other:?}"),
        }
        assert_eq!(allocator.stats().chunks, 0);
    }

    #[test]
    fn allocation_class_flags_are_honest() {
        assert!(MemoryClass::DeviceLocal
            .required_flags()
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
        assert!(MemoryClass::HostVisible
            .required_flags()
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE));
        assert!(MemoryClass::HostVisible
            .preferred_flags()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT));
    }

    #[test]
    fn region_list_starts_as_one_free_range() {
        assert_eq!(first_fit(&regions(), 1024, 1), Some(0));
        assert_eq!(first_fit(&regions(), 1025, 1), None);
    }
}
