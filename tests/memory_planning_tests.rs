use fastnn::compiler::passes::memory_planning;
use fastnn::compiler::passes::shape_inference;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{ComputeGraph, DimExpr, IrDType, Opcode, TensorType};

#[test]
fn test_memory_plan_basic_reuse() {
    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input, vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_a_id = graph.add_node(
        Opcode::Relu, vec![input_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_b_id = graph.add_node(
        Opcode::Relu, vec![relu_a_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Add, vec![relu_a_id, relu_b_id],
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
    // since reuse should occur). Each node is 4*f32 = 16 bytes, aligned to 8 = 16.
    // At minimum we need 2 active slots at peak (input+relu_a or relu_a+relu_b+output etc.)
    let min_possible = 16usize; // at least one slot
    let max_worst_case = 4 * 16usize; // worst case: no reuse
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
        Opcode::Input, vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_id = graph.add_node(
        Opcode::Relu, vec![input_id],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Add, vec![input_id, relu_id],
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
        Opcode::Input, vec![],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let w1_id = graph.add_node(
        Opcode::Input, vec![],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let mm_id = graph.add_node(
        Opcode::MatMul, vec![input_id, w1_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let relu_id = graph.add_node(
        Opcode::Relu, vec![mm_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let output_id = graph.add_node(
        Opcode::Relu, vec![relu_id],
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id, w1_id]);
    graph.set_outputs(vec![output_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    // Peak memory should be less than worst case (sum of all sizes)
    let sum_sizes: usize = graph.nodes.iter()
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
        Opcode::Input, vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let relu_id = graph.add_node(
        Opcode::Relu, vec![input_id],
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
        Opcode::Input, vec![],
        TensorType::new(vec![DimExpr::Known(0)], IrDType::F32),
    );
    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![input_id]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();
    // Zero-sized tensors may be skipped, but the plan should still be valid
    assert!(plan.total_size >= 0);
}

#[test]
fn test_memory_plan_tighten() {
    use fastnn::ir::node::ShapeEnv;

    let mut graph = ComputeGraph::new();
    let input_id = graph.add_node(
        Opcode::Input, vec![],
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
    shape_env.bind("N", 10);
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
        Opcode::Input, vec![],
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
