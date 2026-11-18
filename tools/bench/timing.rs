//! Timing primitives: explicit warmup, explicit sample counts, explicit
//! repetition batches. Every measured closure is black-boxed so the optimizer
//! cannot delete the work.

use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct SamplePlan {
    /// Unmeasured iterations before each batch.
    pub warmup: u64,
    /// Measured iterations per batch.
    pub samples: u64,
    /// Independent batches. Samples from all batches are pooled.
    pub repetitions: u64,
}

impl SamplePlan {
    pub fn new(warmup: u64, samples: u64, repetitions: u64) -> Self {
        Self {
            warmup,
            samples: samples.max(1),
            repetitions: repetitions.max(1),
        }
    }
}

/// Time a closure over `plan`, returning every measured sample in milliseconds.
pub fn measure_ms<F, T>(plan: SamplePlan, mut f: F) -> Vec<f64>
where
    F: FnMut() -> T,
{
    let mut samples = Vec::with_capacity((plan.samples * plan.repetitions) as usize);
    for _ in 0..plan.repetitions {
        for _ in 0..plan.warmup {
            black_box(f());
        }
        for _ in 0..plan.samples {
            let start = Instant::now();
            let value = f();
            let elapsed = start.elapsed();
            black_box(value);
            samples.push(elapsed.as_secs_f64() * 1000.0);
        }
    }
    samples
}

/// Counters captured around one measured operation.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpCounters {
    pub arena_temp_copies: u64,
    pub arena_temp_copy_bytes: u64,
    pub tls_vec_allocs: u64,
    pub tls_vec_reuses: u64,
}

impl OpCounters {
    pub fn add(&mut self, other: OpCounters) {
        self.arena_temp_copies += other.arena_temp_copies;
        self.arena_temp_copy_bytes += other.arena_temp_copy_bytes;
        self.tls_vec_allocs += other.tls_vec_allocs;
        self.tls_vec_reuses += other.tls_vec_reuses;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_count_is_exact() {
        let samples = measure_ms(SamplePlan::new(2, 4, 3), || 1 + 1);
        assert_eq!(samples.len(), 12);
        assert!(samples.iter().all(|value| value.is_finite()));
    }
}
