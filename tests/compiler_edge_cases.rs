use fastnn::compiler::passes::{memory_planning, shape_inference};
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::{ComputeGraph, DimExpr, IrDType, Opcode, TensorType};

#[test]
fn test_single_node_graph() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![a.node_id()]);
    graph.set_outputs(vec![a.node_id()]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let plan = memory_planning::plan_memory(&graph).unwrap();

    assert_eq!(plan.slots.len(), 1, "single-node graph should have 1 slot");
    assert!(plan.slots.contains_key(&a.node_id()));
}

#[test]
fn test_disconnected_graph() {
    let g = GraphBuilder::new();
    let a = g.input(&[2, 3], IrDType::F32);
    let b = g.input(&[4, 5], IrDType::F32);
    let relu_a = g.relu(&a);
    let sigmoid_b = g.sigmoid(&b);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![a.node_id(), b.node_id()]);
    graph.set_outputs(vec![relu_a.node_id(), sigmoid_b.node_id()]);

    let result = shape_inference::infer_shapes(&mut graph);
    assert!(
        result.is_ok(),
        "disconnected graph should infer shapes successfully"
    );

    let plan = memory_planning::plan_memory(&graph).unwrap();
    assert!(
        plan.slots.len() >= 4,
        "disconnected graph should have slots for all nodes"
    );
}

#[test]
fn test_diamond_shape() {
    let g = GraphBuilder::new();
    let input = g.input(&[4], IrDType::F32);
    let a = g.relu(&input);
    let b = g.sigmoid(&input);
    let c = g.add(&a, &b);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![input.node_id()]);
    graph.set_outputs(vec![c.node_id()]);

    let result = shape_inference::infer_shapes(&mut graph);
    assert!(
        result.is_ok(),
        "diamond graph should infer shapes successfully"
    );

    let c_node = graph.get_node(c.node_id()).unwrap();
    assert_eq!(
        c_node.output_type.shape,
        vec![DimExpr::Known(4)],
        "diamond output should have shape [4]"
    );
}

#[test]
fn test_no_output_graph() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let b = g.relu(&a);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![a.node_id()]);
    // No outputs set

    let result = shape_inference::infer_shapes(&mut graph);
    assert!(
        result.is_ok(),
        "graph with no outputs should still infer shapes"
    );
}

#[test]
fn test_cyclic_graph_detection() {
    let mut graph = ComputeGraph::new();
    // Create two nodes that form a cycle: n1 -> n2 -> n1
    let n1 = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    let n2 = graph.add_node(
        Opcode::Relu,
        vec![n1],
        TensorType::new(vec![DimExpr::Known(4)], IrDType::F32),
    );
    // Create the cycle by making n1 depend on n2
    if let Some(n1_mut) = graph.get_node_mut(n1) {
        n1_mut.inputs = vec![n2];
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        shape_inference::infer_shapes(&mut graph)
    }));
    assert!(
        result.is_err(),
        "cyclic graph should panic during topological sort"
    );
}

#[test]
fn test_chain_graph() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let b = g.relu(&a);
    let c = g.sigmoid(&b);
    let d = g.tanh(&c);
    let e = g.exp(&d);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![a.node_id()]);
    graph.set_outputs(vec![e.node_id()]);

    let result = shape_inference::infer_shapes(&mut graph);
    assert!(
        result.is_ok(),
        "chain graph should infer shapes successfully"
    );

    for node_id in [b.node_id(), c.node_id(), d.node_id(), e.node_id()] {
        let node = graph.get_node(node_id).unwrap();
        assert_eq!(
            node.output_type.shape,
            vec![DimExpr::Known(4)],
            "chain node {} should preserve shape [4]",
            node_id
        );
    }
}

#[test]
fn test_fan_out_graph() {
    let g = GraphBuilder::new();
    let input = g.input(&[4], IrDType::F32);
    let relu = g.relu(&input);
    let sigmoid = g.sigmoid(&input);
    let tanh = g.tanh(&input);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![input.node_id()]);
    graph.set_outputs(vec![relu.node_id(), sigmoid.node_id(), tanh.node_id()]);

    let result = shape_inference::infer_shapes(&mut graph);
    assert!(
        result.is_ok(),
        "fan-out graph should infer shapes successfully"
    );

    for node_id in [relu.node_id(), sigmoid.node_id(), tanh.node_id()] {
        let node = graph.get_node(node_id).unwrap();
        assert_eq!(node.output_type.shape, vec![DimExpr::Known(4)]);
    }
}

#[test]
fn test_different_dtype_graph() {
    let g = GraphBuilder::new();
    let a = g.input(&[4], IrDType::F32);
    let b = g.input(&[4], IrDType::F32);
    let c = g.add(&a, &b);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![a.node_id(), b.node_id()]);
    graph.set_outputs(vec![c.node_id()]);

    shape_inference::infer_shapes(&mut graph).unwrap();
    let c_node = graph.get_node(c.node_id()).unwrap();
    assert_eq!(c_node.output_type.dtype, IrDType::F32);
}

#[test]
fn test_large_rank_tensor() {
    let g = GraphBuilder::new();
    let a = g.input(&[2, 3, 4, 5, 6], IrDType::F32);
    let b = g.relu(&a);

    let mut graph = g.to_graph();
    graph.set_inputs(vec![a.node_id()]);
    graph.set_outputs(vec![b.node_id()]);

    let result = shape_inference::infer_shapes(&mut graph);
    assert!(result.is_ok(), "5D tensor should infer shapes successfully");

    let b_node = graph.get_node(b.node_id()).unwrap();
    assert_eq!(
        b_node.output_type.shape,
        vec![
            DimExpr::Known(2),
            DimExpr::Known(3),
            DimExpr::Known(4),
            DimExpr::Known(5),
            DimExpr::Known(6),
        ]
    );
}
