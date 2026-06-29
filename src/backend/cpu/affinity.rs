use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::OnceLock;

use super::topology;

/// A pinned Rayon thread pool that binds workers to physical cores.
/// Once created, all parallel operations in the CPU backend use this pool.
pub struct PinnedThreadPool {
    pool: rayon::ThreadPool,
}

impl PinnedThreadPool {
    /// Build a thread pool with workers pinned to physical cores.
    /// The pool size equals `topology::physical_core_count()`.
    pub fn new() -> Result<Self, rayon::ThreadPoolBuildError> {
        let core_ids = topology::physical_core_ids();
        let num_threads = core_ids.len();
        let core_counter = AtomicUsize::new(0);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|i| format!("fastnn-worker-{i}"))
            .spawn_handler(|thread| {
                let idx = core_counter.fetch_add(1, Ordering::Relaxed) % core_ids.len();
                let core = core_ids[idx];

                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_string());
                }
                if let Some(stack) = thread.stack_size() {
                    builder = builder.stack_size(stack);
                }

                builder.spawn(move || {
                    core_affinity::set_for_current(core);
                    thread.run();
                })?;
                Ok(())
            })
            .build()?;

        Ok(Self { pool })
    }

    /// Execute a closure on the pinned thread pool.
    #[inline]
    pub fn install<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        self.pool.install(f)
    }
}

// Global pool instance, lazily initialized.
static GLOBAL_POOL: OnceLock<PinnedThreadPool> = OnceLock::new();

/// Get or initialize the global pinned thread pool.
pub fn global_pinned_pool() -> &'static PinnedThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        PinnedThreadPool::new().expect("Failed to create pinned thread pool")
    })
}

/// Ensure the global Rayon thread pool is initialized with pinned threads.
/// Safe to call multiple times — only the first call has effect.
pub fn ensure_global_pool_initialized() {
    static INITIALIZED: AtomicBool = AtomicBool::new(false);
    if INITIALIZED.load(Ordering::Relaxed) {
        return;
    }
    let core_ids = topology::physical_core_ids();
    if core_ids.is_empty() {
        return;
    }
    let num_threads = core_ids.len();
    let core_counter = AtomicUsize::new(0);

    let result = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .thread_name(|i| format!("fastnn-worker-{i}"))
        .spawn_handler(|thread| {
            let idx = core_counter.fetch_add(1, Ordering::Relaxed) % core_ids.len();
            let core = core_ids[idx];

            let mut builder = std::thread::Builder::new();
            if let Some(name) = thread.name() {
                builder = builder.name(name.to_string());
            }
            if let Some(stack) = thread.stack_size() {
                builder = builder.stack_size(stack);
            }

            builder.spawn(move || {
                core_affinity::set_for_current(core);
                thread.run();
            })?;
            Ok(())
        })
        .build_global();

    if let Err(e) = result {
        // If build_global() was already called (e.g., by rayon internally),
        // that's fine — we can't override it anyway. Only fail if it's a real error.
        if !e.to_string().contains("global thread pool") {
            eprintln!("[fastnn] Failed to init pinned thread pool: {e}");
        }
    }
    INITIALIZED.store(true, Ordering::Release);
}
