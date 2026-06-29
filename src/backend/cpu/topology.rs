use raw_cpuid::{CpuId, TopologyType};

/// Returns the number of physical cores on the current CPU.
///
/// Uses the extended topology leaf (CPUID leaf 0xB) to determine SMT width,
/// then divides available logical processors by that width.
///
/// On a Ryzen 3700X (8C/16T) this returns 8.
/// On a single-threaded CPU it returns the thread count.
pub fn physical_core_count() -> usize {
    let cpuid = CpuId::new();
    let total_logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // Preferred: use extended topology info (CPUID leaf 0xB)
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

    // Last heuristic: assume SMT is 2-wide
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
