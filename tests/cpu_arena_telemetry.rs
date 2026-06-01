use fastnn::backend::cpu::telemetry::{cpu_telemetry_snapshot, reset_cpu_telemetry};
use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::ir::builder::{GraphBuilder, GraphTensor};
use fastnn::ir::node::{ComputeGraph, DimExpr, IrDType, TensorType};
use std::sync::Mutex;

const LEN: usize = 257;
static TELEMETRY_TEST_LOCK: Mutex<()> = Mutex::new(());

fn vector_data(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((((i * 17 + salt) % 101) as f32) - 50.0) * 0.03125)
        .collect()
}

fn graph_from(builder: &GraphBuilder, output: &GraphTensor) -> ComputeGraph {
    let mut graph = builder.to_graph();
    graph.inputs = builder.recorded_input_ids();
    graph.outputs = vec![output.node_id()];
    graph
}

fn execute_single_output_f32(graph: &ComputeGraph, inputs: &[&[u8]]) -> Vec<f32> {
    let mut executor = GraphExecutor::new(CpuBackend);
    let (mut plan, memory_plan, compiled_graph) = executor
        .compile_with_plan_and_quantize(graph, None)
        .expect("graph should compile");
    let outputs = executor
        .execute(&compiled_graph, &mut plan, &memory_plan, inputs)
        .expect("graph should execute");
    bytemuck::cast_slice(&outputs[0]).to_vec()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let err = (got - want).abs();
        assert!(
            err <= 1e-6,
            "value {idx} mismatch: got {got}, expected {want}, err {err}"
        );
    }
}

#[test]
fn unary_relu_compiled_disjoint_graph_has_zero_arena_temp_copies() {
    let _guard = TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let builder = GraphBuilder::new();
    let input = builder.input_with_dims(&[DimExpr::Known(LEN as u64)], IrDType::F32);
    let output = builder.relu(&input);
    let graph = graph_from(&builder, &output);

    let input_values = vector_data(LEN, 3);
    let input_bytes = bytemuck::cast_slice(&input_values).to_vec();

    reset_cpu_telemetry();
    let actual = execute_single_output_f32(&graph, &[&input_bytes]);
    let snapshot = cpu_telemetry_snapshot();

    let expected: Vec<f32> = input_values.iter().map(|&x| x.max(0.0)).collect();
    assert_close(&actual, &expected);
    assert_eq!(
        snapshot.arena_temp_copies, 0,
        "compiled disjoint relu should borrow arena input/output directly; snapshot={snapshot:?}"
    );
    assert_eq!(snapshot.arena_temp_copy_bytes, 0);
}

#[test]
fn scalar_add_compiled_disjoint_graph_has_zero_arena_temp_copies() {
    let _guard = TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let builder = GraphBuilder::new();
    let input = builder.input_with_dims(&[DimExpr::Known(LEN as u64)], IrDType::F32);
    let scalar = 1.25f32;
    let scalar_node = builder.constant(
        bytemuck::cast_slice(&[scalar]),
        TensorType::new(vec![], IrDType::F32),
    );
    let output = builder.add_scalar(&input, &scalar_node);
    let graph = graph_from(&builder, &output);

    let input_values = vector_data(LEN, 11);
    let input_bytes = bytemuck::cast_slice(&input_values).to_vec();

    reset_cpu_telemetry();
    let actual = execute_single_output_f32(&graph, &[&input_bytes]);
    let snapshot = cpu_telemetry_snapshot();

    let expected: Vec<f32> = input_values.iter().map(|&x| x + scalar).collect();
    assert_close(&actual, &expected);
    assert_eq!(
        snapshot.arena_temp_copies, 0,
        "compiled disjoint add_scalar should borrow arena input/output directly; snapshot={snapshot:?}"
    );
    assert_eq!(snapshot.arena_temp_copy_bytes, 0);
}

#[test]
fn plain_binary_add_compiled_graph_reports_current_same_shape_copy_fallback() {
    let _guard = TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let builder = GraphBuilder::new();
    let lhs = builder.input_with_dims(&[DimExpr::Known(LEN as u64)], IrDType::F32);
    let rhs = builder.input_with_dims(&[DimExpr::Known(LEN as u64)], IrDType::F32);
    let output = builder.add(&lhs, &rhs);
    let graph = graph_from(&builder, &output);

    let lhs_values = vector_data(LEN, 17);
    let rhs_values = vector_data(LEN, 23);
    let lhs_bytes = bytemuck::cast_slice(&lhs_values).to_vec();
    let rhs_bytes = bytemuck::cast_slice(&rhs_values).to_vec();

    reset_cpu_telemetry();
    let actual = execute_single_output_f32(&graph, &[&lhs_bytes, &rhs_bytes]);
    let snapshot = cpu_telemetry_snapshot();

    let expected: Vec<f32> = lhs_values
        .iter()
        .zip(rhs_values.iter())
        .map(|(&x, &y)| x + y)
        .collect();
    assert_close(&actual, &expected);
    assert_eq!(
        snapshot.arena_temp_copies, 2,
        "same-shape plain add currently copies both inputs before dispatch; snapshot={snapshot:?}"
    );
    assert_eq!(
        snapshot.arena_temp_copy_bytes,
        (lhs_bytes.len() + rhs_bytes.len()) as u64
    );
}

#[test]
fn plain_binary_broadcast_add_compiled_graph_reports_current_copy_fallback() {
    let _guard = TELEMETRY_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let width = 17usize;
    let len = LEN * width;
    let builder = GraphBuilder::new();
    let lhs = builder.input_with_dims(
        &[DimExpr::Known(LEN as u64), DimExpr::Known(width as u64)],
        IrDType::F32,
    );
    let rhs = builder.input_with_dims(&[DimExpr::Known(width as u64)], IrDType::F32);
    let output = builder.add(&lhs, &rhs);
    let graph = graph_from(&builder, &output);

    let lhs_values = vector_data(len, 29);
    let rhs_values = vector_data(width, 31);
    let lhs_bytes = bytemuck::cast_slice(&lhs_values).to_vec();
    let rhs_bytes = bytemuck::cast_slice(&rhs_values).to_vec();

    reset_cpu_telemetry();
    let actual = execute_single_output_f32(&graph, &[&lhs_bytes, &rhs_bytes]);
    let snapshot = cpu_telemetry_snapshot();

    let expected: Vec<f32> = lhs_values
        .iter()
        .enumerate()
        .map(|(idx, &x)| x + rhs_values[idx % width])
        .collect();
    assert_close(&actual, &expected);
    assert_eq!(
        snapshot.arena_temp_copies, 2,
        "broadcast plain add currently copies both inputs before modulo/broadcast fallback; snapshot={snapshot:?}"
    );
    assert_eq!(
        snapshot.arena_temp_copy_bytes,
        (lhs_bytes.len() + rhs_bytes.len()) as u64
    );
}
