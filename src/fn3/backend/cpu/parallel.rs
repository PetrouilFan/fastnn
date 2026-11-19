//! CPU thread budget, the single process-wide worker pool, and the
//! diagnostics that prove a selected budget reaches the kernels.
//!
//! # Why a budget instead of "use all cores"
//!
//! Inference on the pinned models spends most of its time in a handful of
//! memory-bound kernels, and the small ones (a `1x128x384` projection is ~4 µs)
//! are far below the cost of handing work to another thread. A thread count
//! chosen once at startup therefore cannot be right for every instruction: the
//! budget has to be resolved per call, and the kernels have to be told how much
//! of it they may use.
//!
//! [`ThreadBudget`] is that per-runtime selection. [`CpuBackend`] owns one and
//! every parallel kernel asks [`resolve`] for an effective thread count on each
//! dispatch, so changing the budget changes the next instruction rather than
//! the next process.
//!
//! [`CpuBackend`]: super::CpuBackend
//!
//! # One pool, no nesting
//!
//! There is exactly one worker pool: Rayon's process-wide global pool, which
//! already exists in this crate for the 2.x CPU path (feature `parallel`).
//! Nothing here builds a pool per operation, per backend or per session, and no
//! kernel ever enters another parallel region from inside a worker
//! ([`in_worker`] refuses), so the 2.x GEMM/BLAS pool and this one are the same
//! pool rather than nested ones.
//!
//! The budget is applied by *how many chunks a kernel submits*: Rayon runs at
//! most one chunk per task, so submitting `k` chunks bounds the work in flight
//! to `k` threads even though the pool is larger. A budget of one takes the
//! serial path and never touches the pool, which is what makes the
//! one-thread measurements below exactly serial (Rayon can otherwise run one
//! task on the calling thread while a worker takes another).
//!
//! # Process-visible limits
//!
//! [`process_budget`] uses [`std::thread::available_parallelism`], which on
//! Linux already reflects `sched_getaffinity` and a cgroup v2 CPU quota. An
//! explicit `FASTNN_CPU_THREADS` environment value is clamped to that visible
//! limit, so a container cannot be asked for cores it does not have. The first
//! resolution also sizes the global pool when (and only when) the environment
//! asks for a cap; if the pool was already built by someone else the request
//! cannot be honoured and [`process_pool_note`] says so instead of pretending.
//!
//! # Diagnostics
//!
//! [`stats`] and [`reset_stats`] expose task counts, the observed peak
//! concurrency and the number of distinct threads that actually executed a
//! chunk. A capability flag or a configured thread count is not evidence that
//! the budget reached the kernels; these counters are, because they are
//! incremented *inside* the chunk bodies.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Upper bound on a requested budget. Above this a model is small enough and a
/// host large enough that the extra threads cannot help a per-instruction
/// dispatch, and the counter bookkeeping would dominate.
pub const MAX_THREADS: usize = 64;

/// Smallest share of a matrix product, in multiply-accumulate operations, that
/// justifies one more worker.
///
/// The rule a kernel applies is `tasks = macs / MIN_MACS_PER_TASK`, capped by
/// the budget and by the output extent, so a small product stays serial no
/// matter how many CPUs the runtime may use.
///
/// Measured on the Ryzen 7 3700X (FN3-25 / #71):
///
/// - the tiny model's one-token `lm_head` is 6.4 M MACs and *does* get 3.0x
///   faster as a kernel at 16 threads (1420 us -> 469 us), but splitting it
///   makes the whole decode step 55% **slower** (6.22 ms -> 9.68 ms) because the
///   parallel phase evicts the serial transpose's 25.7 MB destination from the
///   shared 32 MB L3. At 6.4 M MACs the rule gives one task, i.e. serial.
/// - the trained model's one-token `lm_head` is 38.6 M MACs; at 4 M per task it
///   gets ~9 tasks and the decode step gains 1.26x.
///
/// The crossover therefore sits between 0.4 M and 2.4 M MACs per task; 4 M is
/// chosen as the conservative end, because the failure mode (a small op made
/// slower, and the cache of a *different* op poisoned) is worse than leaving a
/// medium op serial.
pub const MIN_MACS_PER_TASK: u64 = 4 << 20;

/// How much parallelism a runtime allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThreadBudget {
    /// Every CPU this process may use (the default).
    #[default]
    Auto,
    /// At most this many threads; `Fixed(1)` is the serial path. Values above
    /// [`process_budget`] are clamped to it.
    Fixed(usize),
}

impl ThreadBudget {
    /// Shorthand for an explicit cap.
    pub const fn fixed(threads: usize) -> ThreadBudget {
        ThreadBudget::Fixed(threads)
    }

    /// The effective thread count this budget resolves to on this host.
    ///
    /// `Auto` is the visible CPU count; `Fixed` is the request clamped to it.
    /// The narrowing an `Auto` runtime may apply after observing a serial decode
    /// is [`super::CpuBackend::threads`]'s business, not the budget's.
    pub fn threads(self) -> usize {
        match self {
            ThreadBudget::Auto => process_budget(),
            ThreadBudget::Fixed(requested) => requested.clamp(1, process_budget()),
        }
    }
}

/// The environment variable that caps the process-wide budget.
pub const THREADS_ENV: &str = "FASTNN_CPU_THREADS";

struct Process {
    budget: usize,
    note: &'static str,
}

fn process() -> &'static Process {
    static PROCESS: OnceLock<Process> = OnceLock::new();
    PROCESS.get_or_init(|| {
        let visible = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1);
        let visible = visible.clamp(1, MAX_THREADS);
        let requested = std::env::var(THREADS_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0);
        let (budget, note) = match requested {
            None => (visible, "auto: visible CPUs, no cap requested"),
            Some(requested) => {
                let capped = requested.min(visible);
                let note = if capped == requested {
                    "environment cap honoured"
                } else {
                    "environment cap clamped to the visible CPU limit"
                };
                (capped, note)
            }
        };
        #[cfg(feature = "parallel")]
        if requested.is_some() {
            // The pool is process-wide and can only be sized once. If another
            // user of Rayon already built it, the requested cap cannot apply to
            // the pool size; the per-call chunk budget still does, so record it.
            let built = rayon::ThreadPoolBuilder::new()
                .num_threads(budget)
                .build_global()
                .is_ok();
            if !built {
                return Process {
                    budget,
                    note: "environment cap applies per call; the global pool was already built",
                };
            }
        }
        Process { budget, note }
    })
}

/// The thread count every [`ThreadBudget::Auto`] resolves to.
pub fn process_budget() -> usize {
    process().budget
}

/// What decided the process budget, for a report or a diagnostic print.
pub fn process_pool_note() -> &'static str {
    process().note
}

/// `true` when the current thread is a worker of the global pool.
///
/// A kernel that is already on a worker must not submit another parallel
/// region: that is how pools nest. Nothing in this backend does, and this
/// predicate is the guard that keeps a future edit from starting to.
#[inline]
pub fn in_worker() -> bool {
    #[cfg(feature = "parallel")]
    {
        rayon::current_thread_index().is_some()
    }
    #[cfg(not(feature = "parallel"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// Chunk execution
// ---------------------------------------------------------------------------

/// Split `total` into `tasks` contiguous lengths that differ by at most one.
///
/// `par_chunks_mut` can only use a fixed chunk size, which leaves the last task
/// short: for ten rows and four tasks that is `3,3,3,1`, and the 1-row task
/// finishes while three others still run, so a *narrower* budget can beat a
/// wider one. Balanced lengths remove that artefact (the FN3-25 sweep showed
/// 10-row prefill at 4 tasks as 328 ms against 228 ms for 10 balanced tasks).
pub fn balanced_lengths(total: usize, tasks: usize) -> Vec<usize> {
    let tasks = tasks.clamp(1, total.max(1));
    let base = total / tasks;
    let remainder = total % tasks;
    (0..tasks)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

/// Run `body` once per disjoint slice of `out`, in parallel up to `threads`.
///
/// `lengths` must sum to `out.len()`; the slices are produced with
/// `split_at_mut`, so they are disjoint by construction and no `unsafe` is
/// involved in the split.
///
/// Falls back to a serial loop when the budget is one, when there is only one
/// task, or when the caller is already inside a worker (no nested pools).
pub fn par_slices_mut<T, F>(threads: usize, out: &mut [T], lengths: &[usize], body: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    let mut slices: Vec<&mut [T]> = Vec::with_capacity(lengths.len());
    let mut rest = out;
    for &length in lengths {
        let split = length.min(rest.len());
        let (head, tail) = rest.split_at_mut(split);
        slices.push(head);
        rest = tail;
    }
    debug_assert!(rest.is_empty(), "lengths did not cover the output");

    if threads <= 1 || slices.len() <= 1 || in_worker() {
        SERIAL_OPS.fetch_add(1, Ordering::Relaxed);
        for (index, part) in slices.into_iter().enumerate() {
            body(index, part);
        }
        return;
    }
    PARALLEL_OPS.fetch_add(1, Ordering::Relaxed);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        slices
            .into_par_iter()
            .enumerate()
            .for_each(|(index, part)| {
                enter();
                body(index, part);
                leave();
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        // The `parallel` feature is off, so the budget can never exceed one;
        // this arm exists only to keep the signature total.
        for (index, part) in slices.into_iter().enumerate() {
            body(index, part);
        }
    }
}

/// Run `body` once per `0..tasks` index, in parallel up to `threads`, and
/// collect the results in index order.
///
/// Used by the output-column tiling, where each task owns a scratch buffer and
/// the parent scatters the bands afterwards: no two tasks ever write the same
/// element, and no `unsafe` is involved.
pub fn par_map<R, F>(threads: usize, tasks: usize, body: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    if threads <= 1 || tasks <= 1 || in_worker() {
        SERIAL_OPS.fetch_add(1, Ordering::Relaxed);
        return (0..tasks).map(body).collect();
    }
    PARALLEL_OPS.fetch_add(1, Ordering::Relaxed);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..tasks)
            .into_par_iter()
            .map(|index| {
                enter();
                let result = body(index);
                leave();
                result
            })
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..tasks).map(body).collect()
    }
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

static TASKS: AtomicU64 = AtomicU64::new(0);
static PARALLEL_OPS: AtomicU64 = AtomicU64::new(0);
static SERIAL_OPS: AtomicU64 = AtomicU64::new(0);
static ACTIVE: AtomicUsize = AtomicUsize::new(0);
static MAX_ACTIVE: AtomicUsize = AtomicUsize::new(0);
static DISTINCT_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Bumped by [`reset_stats`], so a thread that ran a chunk before the reset
/// registers again afterwards and `distinct_threads` describes the window.
static GENERATION: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "parallel")]
thread_local! {
    /// The [`GENERATION`] this thread last registered in, so
    /// [`CpuParallelStats::distinct_threads`] counts workers that really ran
    /// work in the current window rather than threads that merely exist.
    static REGISTERED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(feature = "parallel")]
fn enter() {
    TASKS.fetch_add(1, Ordering::Relaxed);
    let active = ACTIVE.fetch_add(1, Ordering::Relaxed) + 1;
    MAX_ACTIVE.fetch_max(active, Ordering::Relaxed);
    let generation = GENERATION.load(Ordering::Relaxed);
    if REGISTERED.with(|seen| seen.replace(generation)) != generation {
        DISTINCT_THREADS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(feature = "parallel")]
fn leave() {
    ACTIVE.fetch_sub(1, Ordering::Relaxed);
}

/// A snapshot of the backend's parallel dispatch counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CpuParallelStats {
    /// Parallel chunk bodies executed.
    pub tasks: u64,
    /// Dispatches that took the parallel path.
    pub parallel_ops: u64,
    /// Dispatches that stayed serial (budget of one, too small, or nested).
    pub serial_ops: u64,
    /// Highest number of chunk bodies running at the same instant.
    pub max_active: usize,
    /// Distinct threads that executed at least one chunk since the reset.
    pub distinct_threads: usize,
}

/// Read the counters.
pub fn stats() -> CpuParallelStats {
    CpuParallelStats {
        tasks: TASKS.load(Ordering::Relaxed),
        parallel_ops: PARALLEL_OPS.load(Ordering::Relaxed),
        serial_ops: SERIAL_OPS.load(Ordering::Relaxed),
        max_active: MAX_ACTIVE.load(Ordering::Relaxed),
        distinct_threads: DISTINCT_THREADS.load(Ordering::Relaxed),
    }
}

/// Clear the counters and start a new observation window.
pub fn reset_stats() {
    TASKS.store(0, Ordering::Relaxed);
    PARALLEL_OPS.store(0, Ordering::Relaxed);
    SERIAL_OPS.store(0, Ordering::Relaxed);
    ACTIVE.store(0, Ordering::Relaxed);
    MAX_ACTIVE.store(0, Ordering::Relaxed);
    DISTINCT_THREADS.store(0, Ordering::Relaxed);
    // Bumping the generation is what makes `distinct_threads` exact: the next
    // chunk on any thread registers again, so the count describes this window
    // even though the thread-local outlives it.
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// Hint for the first-use cost of the global pool, so a measured run does not
/// pay pool start-up inside its first sample.
pub fn warm_pool() {
    let threads = process_budget();
    if threads <= 1 {
        return;
    }
    let mut scratch = vec![0u8; threads * 8];
    par_slices_mut(
        threads,
        &mut scratch,
        &balanced_lengths(threads * 8, threads),
        |_, part| {
            part[0] = part[0].wrapping_add(1);
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fixed_budget_never_exceeds_the_visible_limit() {
        let visible = process_budget();
        assert!(visible >= 1);
        assert_eq!(ThreadBudget::Fixed(0).threads(), 1, "zero means serial");
        assert_eq!(ThreadBudget::Fixed(1).threads(), 1);
        assert_eq!(
            ThreadBudget::Fixed(usize::MAX).threads(),
            visible,
            "an impossible request is clamped, not honoured"
        );
        assert_eq!(ThreadBudget::Auto.threads(), visible);
    }

    #[test]
    fn balanced_lengths_cover_the_total_with_a_unit_spread() {
        for total in [0usize, 1, 7, 8, 33, 1000] {
            for tasks in [1usize, 2, 3, 8, 16, 64] {
                let lengths = balanced_lengths(total, tasks);
                assert_eq!(lengths.iter().sum::<usize>(), total, "{total}/{tasks}");
                let width = titles(lengths.iter().copied());
                assert!(width <= 1, "{total}/{tasks} spread {width}");
            }
        }
    }

    fn titles(values: impl Iterator<Item = usize>) -> usize {
        let mut low = usize::MAX;
        let mut high = 0;
        for value in values {
            low = low.min(value);
            high = high.max(value);
        }
        high.saturating_sub(low)
    }

    #[test]
    fn slices_cover_every_element_exactly_once() {
        for total in [0usize, 1, 7, 8, 33, 1000] {
            for tasks in [1usize, 2, 3, 8, 16] {
                let lengths = balanced_lengths(total, tasks);
                let mut values = vec![0u32; total];
                par_slices_mut(4, &mut values, &lengths, |index, part| {
                    for slot in part.iter_mut() {
                        *slot = index as u32 + 1;
                    }
                });
                let mut seen = 0;
                for (index, &length) in lengths.iter().enumerate() {
                    for &value in &values[seen..seen + length] {
                        assert_eq!(value as usize, index + 1, "total {total} tasks {tasks}");
                    }
                    seen += length;
                }
                assert_eq!(seen, total);
            }
        }
    }

    #[test]
    fn a_budget_of_one_never_runs_a_parallel_dispatch() {
        reset_stats();
        let mut values = vec![0u8; 64];
        par_slices_mut(1, &mut values, &balanced_lengths(64, 4), |_, part| {
            part[0] = 1
        });
        let snapshot = stats();
        assert_eq!(snapshot.parallel_ops, 0);
        assert_eq!(snapshot.tasks, 0, "the serial path does not count tasks");
        assert_eq!(snapshot.serial_ops, 1);
    }
}
