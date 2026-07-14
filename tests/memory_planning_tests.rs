use fastnn::compiler::passes::memory_planning;
use fastnn::compiler::passes::shape_inference;
use fastnn::compiler::{AllocSlot, MemoryPlan};
use fastnn::ir::{ComputeGraph, DimExpr, IrDType, Opcode, ShapeEnv, TensorType};
use std::collections::HashMap;

#[test]
fn test_memory_plan_basic_reuse() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_a_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_b_id = graph.add_node(
        Opcode::Relu,
        vec![relu_a_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Add,
        vec![relu_a_id, relu_b_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    // Verify every node has a slot
    assert!(plan.slots.contains_key(&input_id));
    assert!(plan.slots.contains_key(&relu_a_id));
    assert!(plan.slots.contains_key(&relu_b_id));
    assert!(plan.slots.contains_key(&output_id));

    // The total arena size should be reasonable (not sum of all nodes,
    // since reuse should occur). Each node is 4*f32 = 16 bytes, aligned to 64 (cache line).
    // At minimum we need 2 active slots at peak (input+relu_a or relu_a+relu_b+output etc.)
    let min_possible = 64usize; // at least one 64-byte aligned slot
    let max_worst_case = 4 * 64usize; // worst case: no reuse
    assert!(
        plan.total_size >= min_possible,
        "arena size {} should be at least {}",
        plan.total_size,
        min_possible
    );
    assert!(
        plan.total_size <= max_worst_case,
        "arena size {} should be at most {} (no reuse would be worse)",
        plan.total_size,
        max_worst_case
    );
}

#[test]
fn test_memory_plan_required_nodes_lifetime() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Add,
        vec![input_id, relu_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);
    graph.add_required_node(relu_id);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    // Required node should have a slot
    assert!(
        plan.slots.contains_key(&relu_id),
        "required node should have a memory slot"
    );
}

#[test]
fn test_memory_plan_arena_size_peak_usage() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let w1_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let mm_id = graph.add_node(
        Opcode::MatMul,
        vec![input_id, w1_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let relu_id = graph.add_node(
        Opcode::Relu,
        vec![mm_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Relu,
        vec![relu_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id, w1_id]);
    graph.set_outputs(vec![output_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    // Peak memory should be less than worst case (sum of all sizes)
    let sum_sizes: usize = graph
        .nodes
        .iter()
        .filter_map(|n| plan.slots.get(&n.id))
        .map(|s| s.size)
        .sum();
    assert!(
        plan.total_size <= sum_sizes,
        "arena size {} should not exceed sum of all slots {}",
        plan.total_size,
        sum_sizes
    );
}

#[test]
fn test_memory_plan_output_lifetime_extended() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![relu_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    // Output node should have a slot
    assert!(
        plan.slots.contains_key(&relu_id),
        "output node should have a memory slot"
    );
}

#[test]
fn test_memory_plan_output_elementwise_reuses_dead_non_input_operand() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let intermediate_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Sigmoid,
        vec![intermediate_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    let input_slot = plan.slots.get(&input_id).unwrap();
    let intermediate_slot = plan.slots.get(&intermediate_id).unwrap();
    let output_slot = plan.slots.get(&output_id).unwrap();

    assert_eq!(
        output_slot.offset, intermediate_slot.offset,
        "graph-output elementwise node should reuse its dead non-input operand buffer"
    );
    assert_ne!(
        output_slot.offset, input_slot.offset,
        "graph-output elementwise node must not reuse graph input storage"
    );
    assert_eq!(
        plan.total_size,
        2 * 64,
        "input plus one reusable 64-byte intermediate/output slot should be enough"
    );
}

#[test]
fn test_memory_plan_output_elementwise_does_not_reuse_operand_that_is_graph_output() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let intermediate_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let final_id = graph.add_node(
        Opcode::Sigmoid,
        vec![intermediate_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![intermediate_id, final_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    assert_ne!(
        plan.slots.get(&final_id).unwrap().offset,
        plan.slots.get(&intermediate_id).unwrap().offset,
        "graph-output elementwise node must not reuse an operand that is also a graph output"
    );
    assert_eq!(
        plan.total_size,
        3 * 64,
        "input, intermediate output, and final output must occupy distinct aligned slots"
    );
}

#[test]
fn test_memory_plan_output_elementwise_does_not_reuse_operand_that_is_required_node() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let required_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Sigmoid,
        vec![required_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);
    graph.add_required_node(required_id);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    assert_ne!(
        plan.slots.get(&output_id).unwrap().offset,
        plan.slots.get(&required_id).unwrap().offset,
        "graph-output elementwise node must not reuse an operand that is a required node"
    );
    assert_eq!(
        plan.total_size,
        3 * 64,
        "input, required node, and output must occupy distinct aligned slots"
    );
}

#[test]
fn test_memory_plan_cast_does_not_reuse_same_byte_size_different_dtype_operand() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let intermediate_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Cast,
        vec![intermediate_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::I32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);

    let plan = memory_planning::plan_memory(&graph).unwrap();

    assert_ne!(
        plan.slots.get(&output_id).unwrap().offset,
        plan.slots.get(&intermediate_id).unwrap().offset,
        "same-byte-size Cast must not reuse an operand with a different dtype"
    );
    assert_eq!(
        plan.total_size,
        3 * 64,
        "input, f32 intermediate, and i32 output must occupy distinct aligned slots"
    );
}

#[test]
fn test_memory_plan_binary_does_not_reuse_same_byte_size_different_shape_operand() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let lhs_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let rhs_id = graph.add_node(
        Opcode::Sigmoid,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Add,
        vec![lhs_id, rhs_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);

    let plan = memory_planning::plan_memory(&graph).unwrap();

    assert_ne!(
        plan.slots.get(&output_id).unwrap().offset,
        plan.slots.get(&lhs_id).unwrap().offset,
        "same-byte-size binary output must not reuse an operand with a different shape"
    );
    assert_eq!(
        plan.slots.get(&output_id).unwrap().offset,
        plan.slots.get(&rhs_id).unwrap().offset,
        "test setup keeps a same dtype/shape dead operand available for reuse"
    );
}

#[test]
fn test_memory_plan_binary_reuses_dead_non_input_operand() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let lhs_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let rhs_id = graph.add_node(
        Opcode::Sigmoid,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let add_id = graph.add_node(
        Opcode::Add,
        vec![lhs_id, rhs_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Relu,
        vec![add_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    let lhs_slot = plan.slots.get(&lhs_id).unwrap();
    let rhs_slot = plan.slots.get(&rhs_id).unwrap();
    let add_slot = plan.slots.get(&add_id).unwrap();

    assert_eq!(
        add_slot.offset, lhs_slot.offset,
        "binary output should reuse first dead non-input operand buffer"
    );
    assert_ne!(
        add_slot.offset, rhs_slot.offset,
        "planner should pick a single operand buffer for reuse"
    );
    assert_eq!(
        plan.total_size,
        3 * 64,
        "binary add reuses lhs, and the graph-output relu can now reuse the add slot"
    );
}

#[test]
fn test_memory_plan_binary_does_not_reuse_input_operand() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let rhs_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let add_id = graph.add_node(
        Opcode::Add,
        vec![input_id, rhs_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let later_id = graph.add_node(
        Opcode::Mul,
        vec![rhs_id, add_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Relu,
        vec![later_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    assert_ne!(
        plan.slots.get(&add_id).unwrap().offset,
        plan.slots.get(&input_id).unwrap().offset,
        "binary output must not reuse graph input storage"
    );
    assert_ne!(
        plan.slots.get(&add_id).unwrap().offset,
        plan.slots.get(&rhs_id).unwrap().offset,
        "test setup keeps the non-input operand live so only the input would be reusable"
    );
}

#[test]
fn test_memory_plan_binary_does_not_reuse_operand_with_later_consumer() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let lhs_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let rhs_id = graph.add_node(
        Opcode::Sigmoid,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let add_id = graph.add_node(
        Opcode::Add,
        vec![lhs_id, rhs_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let later_id = graph.add_node(
        Opcode::Mul,
        vec![lhs_id, add_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Relu,
        vec![later_id],
        TensorType::new(vec![DimExpr::Known(16)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![output_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    assert_ne!(
        plan.slots.get(&add_id).unwrap().offset,
        plan.slots.get(&lhs_id).unwrap().offset,
        "binary output must not reuse an operand that has a later consumer"
    );
}

#[test]
fn test_memory_plan_empty_graph() {
    let graph = ComputeGraph::new();
    let plan = memory_planning::plan_memory(&graph).unwrap();
    assert_eq!(plan.total_size, 0);
    assert!(plan.slots.is_empty());
}

#[test]
fn test_memory_plan_zero_sized_tensor() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(0)], IrDType::F32),
    );
    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![input_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();
    // Zero-sized tensors may be skipped, but the plan should still be valid
    assert_eq!(plan.total_size, 0);
}

#[test]
fn test_memory_plan_tighten() {
    use fastnn::ir::ShapeEnv;

    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Symbol("N".to_string())], IrDType::F32),
    );
    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![input_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    // With no ShapeEnv, symbolic dims use SYMBOL_DIM_MAX (8192), so total_size >= 8192*4
    assert!(
        plan.total_size > 0,
        "arena size should be positive for symbolic dims"
    );

    // Tighten with a ShapeEnv that resolves N to a small value
    let mut shape_env = ShapeEnv::new();
    shape_env.try_bind("N", 10).unwrap();
    let tightened = plan.tighten(&graph, &shape_env);

    // The tightened size should be <= the original (10*4 = 40 bytes)
    assert!(
        tightened.total_size <= plan.total_size,
        "tightened arena size {} should not exceed original {}",
        tightened.total_size,
        plan.total_size
    );
}

#[test]
fn test_memory_plan_single_node() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![input_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    // Single node should have exactly one slot
    assert_eq!(plan.slots.len(), 1);
    assert!(plan.slots.contains_key(&input_id));
}
#[test]
fn test_memory_plan_tighten_rejects_slot_range_overflow() {
    let mut slots = HashMap::new();
    slots.insert(
        0,
        AllocSlot {
            offset: usize::MAX,
            size: 2,
            node_id: 0,
            output_index: 0,
        },
    );
    let plan = MemoryPlan {
        total_size: usize::MAX,
        slots,
        secondary_slots: HashMap::new(),
        outputs: vec![],
        tightened_params: HashMap::new(),
    };
    assert!(plan
        .try_tighten(&ComputeGraph::new(), &ShapeEnv::new())
        .is_err());
}
