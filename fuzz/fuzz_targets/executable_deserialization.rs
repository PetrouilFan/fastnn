#![no_main]

use fastnn::backend::{ExecutablePlan, PlanResourceLimits};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let limits = PlanResourceLimits {
        max_serialized_bytes: 64 * 1024,
        max_arena_bytes: 4 * 1024 * 1024,
        max_instructions: 1_024,
        ..PlanResourceLimits::default()
    };
    let _ = ExecutablePlan::from_bytes_with_limits(data, &limits);
});
