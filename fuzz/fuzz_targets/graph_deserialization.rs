#![no_main]

use fastnn::ir::{ComputeGraph, GraphResourceLimits};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let limits = GraphResourceLimits {
        max_serialized_bytes: 64 * 1024,
        max_nodes: 256,
        max_total_edges: 2_048,
        max_total_attributes: 2_048,
        max_total_attribute_bytes: 32 * 1024,
    };
    let _ = ComputeGraph::from_fnn_bytes_with_limits(data, limits);
});
