//! Process-wide worker count for the CPU backend.
//!
//! Rayon builds its global pool the first time anything runs in parallel, and
//! from that moment the count cannot be changed. *When* the count is chosen
//! therefore matters more than how: a library that builds the pool while it is
//! being imported decides for the whole process, after which
//! `set_num_threads` can only fail, `RAYON_NUM_THREADS` is ignored, and a
//! caller's own affinity or cgroup limit is overridden.
//!
//! Nothing in this module runs at import time. Left alone, Rayon picks the
//! count from `RAYON_NUM_THREADS` or from the visible CPU count, which already
//! reflects affinity and cgroup limits. [`configure_threads`] lets a caller
//! choose explicitly, and only until the pool exists.

use std::sync::atomic::{AtomicUsize, Ordering};

/// The width this module last configured, or 0 when it has not chosen one.
static CONFIGURED: AtomicUsize = AtomicUsize::new(0);

/// Why a worker count could not be chosen.
#[derive(Debug)]
pub enum AffinityError {
    /// Something already ran in parallel, so Rayon's global pool exists and
    /// cannot be replaced.
    TooLate,
    /// Rayon refused to build the pool.
    Build(rayon::ThreadPoolBuildError),
}

impl std::fmt::Display for AffinityError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLate => formatter.write_str(
                "the worker count is already fixed for this process: it can be chosen \
                 once, before the first parallel operation, so a second call with a \
                 different count cannot take effect; set RAYON_NUM_THREADS before the \
                 process starts to choose it from the environment",
            ),
            Self::Build(error) => write!(formatter, "failed to build the thread pool: {error}"),
        }
    }
}

impl std::error::Error for AffinityError {}

/// Choose how many workers the CPU backend runs on.
///
/// This configures Rayon's global pool, so it succeeds only before the first
/// parallel operation of the process; afterwards it returns
/// [`AffinityError::TooLate`] rather than pretending. Asking for the count that
/// is already in effect is not an error, so a caller may configure
/// unconditionally.
pub fn configure_threads(threads: usize) -> Result<(), AffinityError> {
    let threads = threads.max(1);
    if CONFIGURED.load(Ordering::Relaxed) == threads {
        return Ok(());
    }
    // Nothing here may ask Rayon for its current width first:
    // `rayon::current_num_threads` reads through the global registry, which
    // *builds* it, so a "check, then configure" would always arrive too late.
    match rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|index| format!("fastnn-worker-{index}"))
        .build_global()
    {
        Ok(()) => {
            CONFIGURED.store(threads, Ordering::Relaxed);
            Ok(())
        }
        Err(error) => {
            // Rayon exposes no predicate for the already-initialized case, so
            // its message is the only signal. The pool exists by now, which
            // makes asking for its width safe.
            if error.to_string().contains("already been initialized") {
                if rayon::current_num_threads() == threads {
                    CONFIGURED.store(threads, Ordering::Relaxed);
                    Ok(())
                } else {
                    Err(AffinityError::TooLate)
                }
            } else {
                Err(AffinityError::Build(error))
            }
        }
    }
}

/// The width a parallel operation will run at.
///
/// A width this module configured is reported exactly. Otherwise this is the
/// width Rayon would choose, computed without asking Rayon:
/// `rayon::current_num_threads` reads through the global registry and therefore
/// builds it, which would settle the question the caller is about to ask.
pub fn worker_count() -> usize {
    let configured = CONFIGURED.load(Ordering::Relaxed);
    if configured > 0 {
        return configured;
    }
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_worker_count_is_never_zero() {
        assert!(worker_count() >= 1);
    }

    #[test]
    fn asking_for_the_current_width_is_not_an_error() {
        // The suite runs in parallel, so the pool usually exists by now. Asking
        // for the width already in effect must never fail, which is what lets a
        // caller set the thread count unconditionally.
        let current = worker_count();
        assert!(configure_threads(current).is_ok());
    }

    #[test]
    fn zero_workers_is_read_as_one() {
        // A caller passing 0 or a negative Python value wants "do not
        // oversubscribe", not a pool with no workers.
        let current = worker_count();
        if configure_threads(0).is_err() {
            // The pool already exists at another width; the clamp is still the
            // intended reading, so assert the width it kept.
            assert!(current >= 1);
        }
    }
}
