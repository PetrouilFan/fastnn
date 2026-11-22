//! The worker pool one [`Session`](crate::inference::runtime::Session) owns.
//!
//! # Why not Rayon's global pool
//!
//! ADR-0001 §5: *a session owns its threads*. A process-global pool cannot be
//! sized by a caller, makes two sessions in one process fight for the same
//! workers, and is created by whichever library reaches Rayon first. This module
//! therefore builds **its own** `rayon::ThreadPool` with an exact
//! `ThreadPoolBuilder::num_threads` and named workers, and every parallel
//! region in the v3 executor runs through `SessionThreads::install`.
//!
//! Three rules follow and are enforced by construction here:
//!
//! 1. A pool is built only when [`SessionThreads::new`] is called — that is, when
//!    a session is opened. No module initialisation builds one.
//! 2. The pool is never installed into Rayon's global slot, and no code in this
//!    module calls a bare `rayon::par_iter`: there is no way for this runtime to
//!    reach the global pool.
//! 3. Two sessions with different [`Threading`] values coexist: each has its own
//!    pool, and none of them is shared.
//!
//! # Auto means physical cores
//!
//! [`Threading::Auto`] reads the CPU topology, because SMT siblings add no
//! throughput to the memory-bound scalar kernels and *do* add contention:
//!
//! - Linux: read `/sys/devices/system/cpu/cpu*/topology/thread_siblings_list` and
//!   count the distinct sibling sets. Two logical CPUs that share a set are two
//!   hyperthreads of one physical core, so the number of distinct sets is the
//!   physical core count.
//! - Any failure (no sysfs, a non-Linux target, an unreadable file) falls back to
//!   [`std::thread::available_parallelism`], and then to one.
//!
//! # Without the `parallel` feature
//!
//! Rayon is an optional dependency. A build without it has no pool, so
//! [`SessionThreads::worker_count`] is one for every request and
//! [`Threading::Fixed`] above one is an error rather than a silent clamp: a
//! caller that asked for four workers and got one would be told the wrong thing.

use crate::inference::error::Fn3Error;

/// How many workers a session should own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Threading {
    /// One worker per physical core of this host.
    #[default]
    Auto,
    /// Exactly this many workers. Zero is rejected; one is the serial path.
    Fixed(usize),
}

impl Threading {
    /// The worker count this request resolves to on this host.
    ///
    /// `Auto` is the physical core count; `Fixed(n)` is `n` unchanged. This is
    /// the number the pool is built with, not a ceiling: there is no clamping to
    /// [`std::thread::available_parallelism`], because an explicit request is a
    /// statement about the workload. `Auto` cannot exceed the visible CPUs
    /// unless the topology reports more physical cores than the process may use,
    /// in which case the scheduler's affinity mask bounds the work anyway.
    pub fn resolve(self) -> Result<usize, Fn3Error> {
        match self {
            Threading::Auto => Ok(physical_core_count()
                .or_else(|| {
                    std::thread::available_parallelism()
                        .ok()
                        .map(|value| value.get())
                })
                .unwrap_or(1)
                .max(1)),
            Threading::Fixed(0) => Err(Fn3Error::InvalidShape {
                context: "runtime Threading::Fixed",
                detail: "a session needs at least one worker; Fixed(0) would never run a segment"
                    .to_string(),
            }),
            Threading::Fixed(workers) => Ok(workers),
        }
    }
}

/// The number of physical cores this host reports, when the OS exposes it.
///
/// `None` means the topology could not be read (a non-Linux target, a container
/// without sysfs, an unreadable `thread_siblings_list`). The caller falls back to
/// [`std::thread::available_parallelism`]. No environment variable is consulted:
/// the value describes the machine, not a policy.
pub fn physical_core_count() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::collections::BTreeSet;

        let directory = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
        // The distinct sibling *sets*, not the distinct files: one file per
        // logical CPU, one set per physical core. A `BTreeSet` keeps the walk
        // deterministic and needs no hashing.
        let mut sets: BTreeSet<Vec<u32>> = BTreeSet::new();
        for entry in directory.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            // Only `cpu<N>` entries; `cpufreq`, `cpuidle`, `online`, … are not
            // CPUs and have no topology directory.
            let Some(rest) = name.strip_prefix("cpu") else {
                continue;
            };
            if rest.is_empty() || !rest.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let path = entry.path().join("topology/thread_siblings_list");
            let Ok(text) = std::fs::read_to_string(path) else {
                continue;
            };
            let Some(mut siblings) = parse_cpu_list(&text) else {
                continue;
            };
            siblings.sort_unstable();
            siblings.dedup();
            sets.insert(siblings);
        }
        if sets.is_empty() {
            None
        } else {
            Some(sets.len())
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Parse an `isolcpus`-style CPU list (`"0-3,8,10-11"`) into its members.
///
/// Returns `None` for anything that is not such a list, so a malformed sysfs
/// value is treated as "topology unavailable" rather than as zero cores.
fn parse_cpu_list(text: &str) -> Option<Vec<u32>> {
    let mut members = Vec::new();
    for part in text.trim().split(',') {
        if part.is_empty() {
            return None;
        }
        match part.split_once('-') {
            Some((first, last)) => {
                let first: u32 = first.trim().parse().ok()?;
                let last: u32 = last.trim().parse().ok()?;
                if last < first || last - first > 4096 {
                    return None;
                }
                members.extend(first..=last);
            }
            None => members.push(part.trim().parse().ok()?),
        }
    }
    if members.is_empty() {
        None
    } else {
        Some(members)
    }
}

/// A session's private worker pool.
#[derive(Debug)]
pub struct SessionThreads {
    /// The resolved worker count, whether or not a pool exists.
    workers: usize,
    /// `None` in a build without the `parallel` feature, or when `workers == 1`.
    #[cfg(feature = "parallel")]
    pool: Option<rayon::ThreadPool>,
}

impl SessionThreads {
    /// Resolve `threading` and build the pool.
    ///
    /// The pool is created here and nowhere else: `num_threads` is exactly the
    /// resolved count, so [`SessionThreads::worker_count`] is a fact about the
    /// pool rather than a hope. Workers are named `fn3-worker-<index>`, which is
    /// what a profile or a debugger shows.
    pub fn new(threading: Threading) -> Result<SessionThreads, Fn3Error> {
        let workers = threading.resolve()?;
        #[cfg(feature = "parallel")]
        {
            if workers == 1 {
                return Ok(SessionThreads {
                    workers,
                    pool: None,
                });
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .thread_name(|index| format!("fn3-worker-{index}"))
                // A panic in a kernel must not take the session's pool down with
                // it in a way the caller cannot observe; the error path returns
                // `Fn3Error`, and `spawn_handler` is left at Rayon's default.
                .build()
                .map_err(|error| Fn3Error::Allocation {
                    bytes: workers as u64,
                    detail: format!("could not start {workers} fn3-worker threads: {error}"),
                })?;
            Ok(SessionThreads {
                workers,
                pool: Some(pool),
            })
        }
        #[cfg(not(feature = "parallel"))]
        {
            if workers == 1 {
                Ok(SessionThreads { workers: 1 })
            } else {
                Err(Fn3Error::NotImplemented {
                    what: "a session worker count above one (this build has no `parallel` \
                           feature, so it cannot start a pool)",
                    owner: "v3 segment runtime",
                })
            }
        }
    }

    /// The number of workers this session owns.
    ///
    /// Equal to the resolved [`Threading`] count in every build: a build without
    /// `parallel` only accepts one worker.
    pub fn worker_count(&self) -> usize {
        self.workers
    }

    /// The pool's own thread count, as the proof that the request was honoured.
    ///
    /// `None` when no pool exists (one worker, or a build without `parallel`).
    /// This reads `rayon::ThreadPool::current_num_threads`, so a test can assert
    /// the pool — not merely a field — carries the requested count.
    pub fn pool_num_threads(&self) -> Option<usize> {
        #[cfg(feature = "parallel")]
        {
            self.pool.as_ref().map(|pool| pool.current_num_threads())
        }
        #[cfg(not(feature = "parallel"))]
        {
            None
        }
    }

    /// Run `body` on this pool and wait for it.
    ///
    /// `pub(crate)` because the pool is an implementation detail: only the
    /// kernels in [`crate::inference::runtime::kernels`] split work, and they
    /// split it through here. With one worker, or in a build without `parallel`,
    /// the body runs inline and no thread is touched.
    ///
    /// Calling this from inside the pool runs `body` inline (Rayon's documented
    /// behaviour), so a kernel that parallelises cannot recurse into a nested
    /// region.
    #[cfg(feature = "parallel")]
    pub(crate) fn install<F, R>(&self, body: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        match &self.pool {
            Some(pool) => pool.install(body),
            None => body(),
        }
    }

    /// A build without `parallel` has no pool, so a region is the caller's.
    #[cfg(not(feature = "parallel"))]
    pub(crate) fn install<F, R>(&self, body: F) -> R
    where
        F: FnOnce() -> R,
    {
        body()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_counts_are_exact_and_zero_is_rejected() {
        assert_eq!(Threading::Fixed(1).resolve(), Ok(1));
        assert_eq!(Threading::Fixed(4).resolve(), Ok(4));
        let error = Threading::Fixed(0).resolve().unwrap_err();
        assert!(error.to_string().contains("at least one worker"), "{error}");
    }

    #[test]
    fn auto_resolves_to_the_topology_or_a_fallback() {
        let resolved = Threading::Auto.resolve().unwrap();
        assert!(resolved >= 1);
        if let Some(physical) = physical_core_count() {
            assert_eq!(resolved, physical.max(1));
        }
        // The parser accepts the shapes sysfs actually uses.
        assert_eq!(parse_cpu_list("0\n"), Some(vec![0]));
        assert_eq!(parse_cpu_list("0-3\n"), Some(vec![0, 1, 2, 3]));
        assert_eq!(parse_cpu_list("0,4\n"), Some(vec![0, 4]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("oops"), None);
        assert_eq!(parse_cpu_list("3-1"), None);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn a_session_pool_has_exactly_the_requested_threads() {
        for requested in [1usize, 2, 3, 5] {
            let threads = SessionThreads::new(Threading::Fixed(requested)).unwrap();
            assert_eq!(threads.worker_count(), requested);
            let expected = if requested == 1 {
                None
            } else {
                Some(requested)
            };
            assert_eq!(
                threads.pool_num_threads(),
                expected,
                "the pool itself must carry the count for {requested}, not only the field"
            );
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn the_pool_workers_are_named_by_this_session() {
        use std::sync::mpsc;
        let threads = SessionThreads::new(Threading::Fixed(2)).unwrap();
        let (sender, receiver) = mpsc::channel();
        threads.install(|| {
            // One rayon task per worker: `par_iter` over as many items as the
            // pool has threads is enough to reach both of them on a running
            // system, and the names are collected by the channel.
            use rayon::prelude::*;
            (0..2).into_par_iter().for_each(|_| {
                let _ = sender.send(std::thread::current().name().map(str::to_string));
            });
        });
        let mut names: Vec<String> = receiver
            .try_iter()
            .filter_map(|name| name)
            .filter(|name| name.starts_with("fn3-worker"))
            .collect();
        names.sort();
        names.dedup();
        assert!(
            !names.is_empty(),
            "no pool worker ran; the region may have stayed on the caller"
        );
        for name in names {
            assert!(name.starts_with("fn3-worker-"), "{name}");
        }
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn a_build_without_parallel_reports_one_worker_and_refuses_more() {
        assert_eq!(
            SessionThreads::new(Threading::Auto).unwrap().worker_count(),
            1
        );
        assert!(SessionThreads::new(Threading::Fixed(4)).is_err());
    }
}
