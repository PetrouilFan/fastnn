//! The AOT arena must track the live set, not the length of the graph.
//!
//! Every value in a chain is dead as soon as its single consumer has run, so
//! the plan for N chained convolutions has to be about the size of two
//! activation buffers rather than N of them. The allocator used to propagate
//! each consumer's *transitive* last use, which left every intermediate live
//! until the graph output: no slot was ever freed and the arena grew by one
//! buffer per node, so a 21-conv chain of 6.3 MB tensors asked for 138.9 MB
//! where 12.6 MB sufficed.

use fastnn::backend::cpu::CpuBackend;
use fastnn::ir::IrDType;
use fastnn::GraphBuilder;

/// Small enough that planning and shape inference stay cheap.
const C: usize = 32;
const H: usize = 22;
const W: usize = 28;

/// Bytes of one activation buffer.
fn slot_bytes() -> usize {
    C * H * W * std::mem::size_of::<f32>()
}

/// Plan a chain of `convolutions` depthwise 3x3 convolutions and report the
/// arena the memory planner asks for.
fn plan_bytes(convolutions: usize) -> usize {
    let g = GraphBuilder::new();
    let input = g.input(&[1, C as u64, H as u64, W as u64], IrDType::F32);
    let mut cursor = input;
    for _ in 0..convolutions {
        let weight = g.input(&[C as u64, 1, 3, 3], IrDType::F32);
        cursor = g.conv2d_with_params(&cursor, &weight, 1, 1, 1, C);
    }
    let (_plan, memory_plan, _graph) = g.compile(&[&cursor], CpuBackend).expect("compile");
    memory_plan.total_size
}

#[test]
fn a_chain_plan_stays_at_the_size_of_its_live_set() {
    let one = plan_bytes(1);
    let sixteen = plan_bytes(16);
    let slot = slot_bytes();

    // Two activation buffers plus the weight slots; a chain must not add an
    // activation buffer per node. Four slots of headroom keeps the assertion
    // about liveness rather than about alignment padding.
    assert!(
        sixteen < one + 4 * slot,
        "a 16-conv chain planned {sixteen} bytes against {one} for a single \
         convolution, so an intermediate that is read exactly once kept its \
         slot for the whole graph (one activation buffer is {slot} bytes)"
    );
    assert!(
        sixteen >= one,
        "planning more convolutions cannot shrink the arena: {sixteen} < {one}"
    );
}
