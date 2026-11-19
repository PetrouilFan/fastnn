//! Host-visible staging: upload and readback without unmapping while a copy is
//! in flight.
//!
//! # Lease model
//!
//! [`StagingPool::acquire`] returns a [`StagingLease`] that owns one host-visible
//! buffer for as long as it lives. The lease is the only handle to that block,
//! so the host can write it while a recorded copy references it, and the block
//! returns to the pool only when the lease drops — by construction after the
//! caller waited for the submission. This is the fix for the classic
//! "unmap/reuse while the copy is in flight" bug; the FN3-05 spike used one
//! permanently mapped readback buffer and documented it as a simplification.
//!
//! # Sizing
//!
//! Blocks are `max(request, block_bytes)` rounded up to a MiB. A released block
//! is pooled only if it is no larger than `4 * block_bytes` and the pool is
//! under its byte cap; oversized blocks are freed immediately so one large
//! upload does not pin host memory for the process lifetime.
//!
//! # Coherence
//!
//! [`Buffer::write_from`] and [`Buffer::read_into`] flush/invalidate when the
//! memory type is not `HOST_COHERENT`, with `nonCoherentAtomSize` alignment.
//! The GTX 1080 Ti's host-visible type is coherent, but the code never assumes
//! it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::context::{Buffer, BufferDesc};
use super::device::Device;
use super::error::{Error, Result};

/// Preferred block size and counters of the staging pool.
#[derive(Debug, Clone, Default)]
pub struct StagingStats {
    /// Blocks currently in the pool.
    pub pooled_blocks: usize,
    /// Bytes currently pooled.
    pub pooled_bytes: u64,
    /// Blocks created over the pool's lifetime.
    pub created_blocks: u64,
    /// Bytes written from the host into staging buffers.
    pub upload_bytes: u64,
    /// Bytes read from staging buffers into host memory.
    pub readback_bytes: u64,
    /// Leases acquired.
    pub leases: u64,
}

/// A pool of host-visible transfer buffers.
pub struct StagingPool {
    device: Arc<Device>,
    block_bytes: u64,
    max_pooled_bytes: u64,
    blocks: Mutex<Vec<Arc<Buffer>>>,
    created_blocks: AtomicU64,
    upload_bytes: AtomicU64,
    readback_bytes: AtomicU64,
    leases: AtomicU64,
}

impl StagingPool {
    /// Create an empty pool. `block_bytes` is the preferred block size; the
    /// pool never allocates until the first lease.
    pub fn new(device: Arc<Device>, block_bytes: u64) -> StagingPool {
        let block_bytes = block_bytes.max(4096);
        StagingPool {
            device,
            block_bytes,
            max_pooled_bytes: block_bytes.saturating_mul(4),
            blocks: Mutex::new(Vec::new()),
            created_blocks: AtomicU64::new(0),
            upload_bytes: AtomicU64::new(0),
            readback_bytes: AtomicU64::new(0),
            leases: AtomicU64::new(0),
        }
    }

    /// Preferred block size.
    pub fn block_bytes(&self) -> u64 {
        self.block_bytes
    }

    /// Acquire a block of at least `size` bytes.
    pub fn acquire(&self, size: u64) -> Result<StagingLease<'_>> {
        if size == 0 {
            return Err(Error::internal(
                "staging lease requested with size 0; skip the transfer instead",
            ));
        }
        let limits = self.device.limits();
        if size > limits.max_memory_allocation_size {
            return Err(Error::limit(
                "vulkan staging block",
                size,
                limits.max_memory_allocation_size,
                format!(
                    "one staging block on \"{}\"; split the transfer into chunks",
                    self.device.adapter().name
                ),
            ));
        }

        let mut blocks = self
            .blocks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Best fit: the smallest pooled block that is large enough.
        let candidate = blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| block.size() >= size && !block.must_not_reuse())
            .min_by_key(|(_, block)| block.size())
            .map(|(index, _)| index);
        let buffer = match candidate {
            Some(index) => blocks.remove(index),
            None => {
                let rounded = size
                    .max(self.block_bytes)
                    .div_ceil(1024 * 1024)
                    .saturating_mul(1024 * 1024);
                let desc = BufferDesc::staging(rounded, "staging");
                let buffer = Arc::new(self.device.create_buffer(&desc)?);
                self.created_blocks.fetch_add(1, Ordering::Relaxed);
                buffer
            }
        };
        drop(blocks);
        self.leases.fetch_add(1, Ordering::Relaxed);
        Ok(StagingLease {
            pool: self,
            capacity: buffer.size(),
            buffer: Some(buffer),
        })
    }

    /// Current counters.
    pub fn stats(&self) -> StagingStats {
        let blocks = self
            .blocks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        StagingStats {
            pooled_blocks: blocks.len(),
            pooled_bytes: blocks.iter().map(|block| block.size()).sum(),
            created_blocks: self.created_blocks.load(Ordering::Relaxed),
            upload_bytes: self.upload_bytes.load(Ordering::Relaxed),
            readback_bytes: self.readback_bytes.load(Ordering::Relaxed),
            leases: self.leases.load(Ordering::Relaxed),
        }
    }

    fn release(&self, buffer: Arc<Buffer>) {
        // A block whose submission is quarantined may still be read by the
        // GPU. It must never be leased again or freed by the pool; the device
        // quarantine owns it until completion is proven. A retired block is
        // already destroyed, so the same rule prevents handing out a tombstone.
        if buffer.must_not_reuse() {
            return;
        }
        // Oversized blocks are dropped instead of pooled: one large upload must
        // not pin host memory forever.
        if buffer.size() > self.max_pooled_bytes {
            return;
        }
        let mut blocks = self
            .blocks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let pooled: u64 = blocks.iter().map(|block| block.size()).sum();
        if pooled + buffer.size() > self.max_pooled_bytes {
            return;
        }
        blocks.push(buffer);
    }
}

impl std::fmt::Debug for StagingPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingPool")
            .field("block_bytes", &self.block_bytes)
            .field("stats", &self.stats())
            .finish()
    }
}

/// One exclusively leased host-visible block.
pub struct StagingLease<'a> {
    pool: &'a StagingPool,
    buffer: Option<Arc<Buffer>>,
    capacity: u64,
}

impl<'a> StagingLease<'a> {
    /// The leased buffer.
    pub fn buffer(&self) -> &Buffer {
        self.buffer
            .as_deref()
            .expect("a live staging lease always owns its buffer")
    }

    /// Capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Write `bytes` at offset 0 and flush if necessary.
    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        if bytes.len() as u64 > self.capacity {
            return Err(Error::internal(format!(
                "staging write of {} bytes exceeds the {}-byte lease",
                bytes.len(),
                self.capacity
            )));
        }
        self.buffer().write_from(0, bytes)?;
        self.pool
            .upload_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Read `dst.len()` bytes from offset 0 into `dst`, invalidating if needed.
    pub fn read_into(&self, dst: &mut [u8]) -> Result<()> {
        if dst.len() as u64 > self.capacity {
            return Err(Error::internal(format!(
                "staging read of {} bytes exceeds the {}-byte lease",
                dst.len(),
                self.capacity
            )));
        }
        self.buffer().read_into(0, dst)?;
        self.pool
            .readback_bytes
            .fetch_add(dst.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

impl StagingLease<'_> {
    /// Take the block out of the pool's ownership.
    ///
    /// A step batch uses this: the recorded copy still references the block, so
    /// it must not return to the pool until the batch's fence signals. The
    /// batch owns the returned `Arc` and releases it after the wait.
    pub fn detach(mut self) -> Arc<Buffer> {
        self.buffer
            .take()
            .expect("a live staging lease always owns its buffer")
    }
}

impl Drop for StagingLease<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

impl std::fmt::Debug for StagingLease<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingLease")
            .field("capacity", &self.capacity)
            .finish_non_exhaustive()
    }
}
