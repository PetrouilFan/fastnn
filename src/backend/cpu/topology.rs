/// Returns the number of physical cores on the current CPU.
///
/// On x86_64, uses CPUID extended topology leaf (0xB) to determine SMT width,
/// then divides available logical processors by that width.
/// On non-x86 platforms, uses available_parallelism as a heuristic.
///
/// On a Ryzen 3700X (8C/16T) this returns 8.
/// On a single-threaded CPU it returns the thread count.
pub fn physical_core_count() -> usize {
    let total_logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    #[cfg(target_arch = "x86_64")]
    {
        use raw_cpuid::{CpuId, TopologyType};

        let cpuid = CpuId::new();

        if let Some(topology_iter) = cpuid.get_extended_topology_info() {
            let smt_width: u32 = topology_iter
                .filter(|l| l.level_type() == TopologyType::SMT)
                .map(|l| l.shift_right_for_next_apic_id())
                .map(|shift| 1u32 << shift)
                .next()
                .unwrap_or(1);

            let physical = (total_logical as u32).div_ceil(smt_width) as usize;
            return physical.max(1);
        }
    }

    // Fallback: assume SMT is 2-wide (common for Intel HyperThreading)
    total_logical.div_ceil(2)
}

/// Returns the set of core IDs that correspond to physical cores.
/// Takes the first `physical_core_count()` entries from the OS affinity list.
pub fn physical_core_ids() -> Vec<core_affinity::CoreId> {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let count = physical_core_count().min(core_ids.len());
    core_ids.into_iter().take(count).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_core_count_is_reasonable() {
        let count = physical_core_count();
        assert!(count >= 1);
        assert!(count <= 256, "unreasonably high: {count}");
    }

    #[test]
    fn test_physical_core_ids_not_empty() {
        let ids = physical_core_ids();
        assert!(!ids.is_empty(), "must have at least one core ID");
    }
}
