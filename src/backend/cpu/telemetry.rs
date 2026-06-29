//! Low-overhead CPU backend telemetry.
//!
//! Counters use relaxed atomics so hot paths can record copy/allocation events
//! without imposing synchronization between worker threads.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuTelemetrySnapshot {
    pub arena_temp_copies: u64,
    pub arena_temp_copy_bytes: u64,
    pub tls_vec_allocs: u64,
    pub tls_vec_reuses: u64,
}

/// Pad atomic counters to separate cache lines and avoid false sharing
/// when different threads record distinct events concurrently.
#[repr(align(64))]
struct AlignedAtomicU64(AtomicU64);

impl std::ops::Deref for AlignedAtomicU64 {
    type Target = AtomicU64;
    fn deref(&self) -> &AtomicU64 {
        &self.0
    }
}

static ARENA_TEMP_COPIES: AlignedAtomicU64 = AlignedAtomicU64(AtomicU64::new(0));
static ARENA_TEMP_COPY_BYTES: AlignedAtomicU64 = AlignedAtomicU64(AtomicU64::new(0));
static TLS_VEC_ALLOCS: AlignedAtomicU64 = AlignedAtomicU64(AtomicU64::new(0));
static TLS_VEC_REUSES: AlignedAtomicU64 = AlignedAtomicU64(AtomicU64::new(0));

#[inline]
pub(crate) fn record_arena_temp_copy(bytes: usize) {
    ARENA_TEMP_COPIES.fetch_add(1, Ordering::Relaxed);
    ARENA_TEMP_COPY_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_tls_vec_alloc() {
    TLS_VEC_ALLOCS.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub(crate) fn record_tls_vec_reuse() {
    TLS_VEC_REUSES.fetch_add(1, Ordering::Relaxed);
}

pub fn reset_cpu_telemetry() {
    ARENA_TEMP_COPIES.store(0, Ordering::Relaxed);
    ARENA_TEMP_COPY_BYTES.store(0, Ordering::Relaxed);
    TLS_VEC_ALLOCS.store(0, Ordering::Relaxed);
    TLS_VEC_REUSES.store(0, Ordering::Relaxed);
}

pub fn cpu_telemetry_snapshot() -> CpuTelemetrySnapshot {
    CpuTelemetrySnapshot {
        arena_temp_copies: ARENA_TEMP_COPIES.load(Ordering::Relaxed),
        arena_temp_copy_bytes: ARENA_TEMP_COPY_BYTES.load(Ordering::Relaxed),
        tls_vec_allocs: TLS_VEC_ALLOCS.load(Ordering::Relaxed),
        tls_vec_reuses: TLS_VEC_REUSES.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_telemetry_reset_and_record_helpers_are_stable() {
        reset_cpu_telemetry();
        assert_eq!(cpu_telemetry_snapshot(), CpuTelemetrySnapshot::default());

        record_arena_temp_copy(16);
        record_tls_vec_alloc();
        record_tls_vec_reuse();

        let snapshot = cpu_telemetry_snapshot();
        assert_eq!(snapshot.arena_temp_copies, 1);
        assert_eq!(snapshot.arena_temp_copy_bytes, 16);
        assert_eq!(snapshot.tls_vec_allocs, 1);
        assert_eq!(snapshot.tls_vec_reuses, 1);

        let snapshot_again = cpu_telemetry_snapshot();
        assert_eq!(snapshot_again, snapshot);

        reset_cpu_telemetry();
        assert_eq!(cpu_telemetry_snapshot(), CpuTelemetrySnapshot::default());
    }
}
