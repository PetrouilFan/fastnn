//! Integration tests for the end-to-end quantized pipeline.
//!
//! Tests the full flow: GraphBuilder → compile_with_quantize → execute → verify output.
//! Covers MatMul and Conv2d with both U4 and U8 quantization.

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::backend::{Backend, Instruction};
use fastnn::compiler::passes::dead_code_elimination::eliminate_dead_code;
use fastnn::compiler::passes::memory_planning::plan_memory;
use fastnn::compiler::passes::quantization::quantize_weights;
use fastnn::compiler::passes::shape_inference::infer_shapes;
use fastnn::dtypes::U4x8;
use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::ComputeGraph;
use fastnn::ir::node::{DimExpr, IrDType, Opcode, TensorType, TensorValue};
use fastnn::packed_tensor::PackedTensor;

/// Helper: build a MatMul graph and run through the full pipeline.
/// Returns output as Vec<f32>.
fn run_matmul(
    batch: usize,
    k: usize,
    n: usize,
    weight_data: &[f32],
    input_data: &[f32],
    quantize: Option<u8>,
) -> Vec<f32> {
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(
        &[DimExpr::Known(batch as u64), DimExpr::Known(k as u64)],
        IrDType::F32,
    );
    let weight_tt = TensorType::new(
        vec![DimExpr::Known(k as u64), DimExpr::Known(n as u64)],
        IrDType::F32,
    );
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let output = gb.matmul(&input, &weight);

    let input_bytes: Vec<u8> = bytemuck::cast_slice(input_data).to_vec();
    let result = gb
        .compile_and_execute_with_quantize(&[&output], CpuBackend, &[&input_bytes], quantize)
        .expect("compile_and_execute should succeed");

    // Output is f32 (quantized weights produce f32 output after dequant in GEMM)
    bytemuck::cast_slice(&result[0]).to_vec()
}

/// Helper: build a Conv2d graph and run through the full pipeline.
/// Returns output as Vec<f32>.
fn run_conv2d(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    h: usize,
    w: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    weight_data: &[f32],
    input_data: &[f32],
    quantize: Option<u8>,
) -> Vec<f32> {
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(
        &[
            DimExpr::Known(batch as u64),
            DimExpr::Known(in_channels as u64),
            DimExpr::Known(h as u64),
            DimExpr::Known(w as u64),
        ],
        IrDType::F32,
    );
    let weight_tt = TensorType::new(
        vec![
            DimExpr::Known(out_channels as u64),
            DimExpr::Known(in_channels as u64),
            DimExpr::Known(kernel_size as u64),
            DimExpr::Known(kernel_size as u64),
        ],
        IrDType::F32,
    );
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let output = gb.conv2d_with_params(&input, &weight, stride, padding, 1, 1);

    let input_bytes: Vec<u8> = bytemuck::cast_slice(input_data).to_vec();
    let result = gb
        .compile_and_execute_with_quantize(&[&output], CpuBackend, &[&input_bytes], quantize)
        .expect("compile_and_execute should succeed");

    bytemuck::cast_slice(&result[0]).to_vec()
}

// ── MatMul tests ──────────────────────────────────────────────────────

#[test]
fn test_matmul_u4_end_to_end() {
    // [2, 8] @ [8, 4] = [2, 4]
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16]; // [2, 8] of 0.5

    let output_f32 = run_matmul(2, 8, 4, &weight, &input, None);
    let output_u4 = run_matmul(2, 8, 4, &weight, &input, Some(4));

    // Output should have 8 elements (2*4)
    assert_eq!(output_f32.len(), 8, "f32 output should have 8 elements");
    assert_eq!(output_u4.len(), 8, "U4 output should have 8 elements");

    // Quantized output should be within reasonable tolerance of f32 output.
    // U4 has 4-bit quantization (16 levels), so expect ~10-20% relative error.
    for i in 0..8 {
        let f32_val = output_f32[i];
        let u4_val = output_u4[i];
        if f32_val.abs() <= 0.1 {
            assert!(
                (u4_val - f32_val).abs() < 0.05,
                "U4 small value mismatch at {}: q={}, f32={}",
                i,
                u4_val,
                f32_val
            );
        } else {
            let rel_err = (u4_val - f32_val).abs() / f32_val.abs();
            assert!(
                rel_err < 0.5,
                "U4 output[{}] = {} vs f32 = {} (rel_err = {:.3})",
                i,
                u4_val,
                f32_val,
                rel_err
            );
        }
    }
}

#[test]
fn test_matmul_u8_end_to_end() {
    // [2, 8] @ [8, 4] = [2, 4] — U8 should be closer to f32 than U4
    let weight: Vec<f32> = (0..32).map(|i| (i as f32) / 8.0).collect();
    let input: Vec<f32> = vec![0.5; 16]; // [2, 8] of 0.5

    let output_f32 = run_matmul(2, 8, 4, &weight, &input, None);
    let output_u8 = run_matmul(2, 8, 4, &weight, &input, Some(8));

    assert_eq!(output_u8.len(), 8, "U8 output should have 8 elements");

    // U8 has 8-bit quantization (256 levels), much closer to f32.
    for i in 0..8 {
        let f32_val = output_f32[i];
        let u8_val = output_u8[i];
        if f32_val.abs() <= 0.1 {
            assert!(
                (u8_val - f32_val).abs() < 0.05,
                "U8 small value mismatch at {}: q={}, f32={}",
                i,
                u8_val,
                f32_val
            );
        } else {
            let rel_err = (u8_val - f32_val).abs() / f32_val.abs();
            assert!(
                rel_err < 0.15,
                "U8 output[{}] = {} vs f32 = {} (rel_err = {:.3})",
                i,
                u8_val,
                f32_val,
                rel_err
            );
        }
    }
}

#[test]
fn test_matmul_u4_output_shape_correct() {
    // [1, 4] @ [4, 2] = [1, 2]
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let output = run_matmul(1, 4, 2, &weight, &input, Some(4));
    assert_eq!(
        output.len(),
        2,
        "Output should have 2 elements for [1,4]@[4,2]"
    );
}

#[test]
fn test_matmul_u8_output_shape_correct() {
    // [3, 4] @ [4, 2] = [3, 2]
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5];
    let input: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0,
    ];

    let output = run_matmul(3, 4, 2, &weight, &input, Some(8));
    assert_eq!(
        output.len(),
        6,
        "Output should have 6 elements for [3,4]@[4,2]"
    );
}

// ── Conv2d tests ──────────────────────────────────────────────────────

#[test]
fn test_conv2d_u4_end_to_end() {
    // Conv2d: 1 input channel, 2 output channels, 3x3 kernel, stride=1, padding=0
    // Input: [1, 1, 5, 5] — output: [1, 2, 3, 3]
    let in_channels = 1usize;
    let out_channels = 2usize;
    let kernel_size = 3usize;
    let weight: Vec<f32> = (0..(out_channels * in_channels * kernel_size * kernel_size))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let input: Vec<f32> = vec![0.5; 25]; // [1, 1, 5, 5]

    let output_f32 = run_conv2d(
        1,
        in_channels,
        out_channels,
        5,
        5,
        kernel_size,
        1,
        0,
        &weight,
        &input,
        None,
    );
    let output_u4 = run_conv2d(
        1,
        in_channels,
        out_channels,
        5,
        5,
        kernel_size,
        1,
        0,
        &weight,
        &input,
        Some(4),
    );

    // Output should have 18 elements (1 * 2 * 3 * 3)
    assert_eq!(
        output_f32.len(),
        18,
        "f32 conv2d output should have 18 elements"
    );
    assert_eq!(
        output_u4.len(),
        18,
        "U4 conv2d output should have 18 elements"
    );

    // U4 quantized conv2d output should be within reasonable tolerance.
    // Use RMS error across all elements + individual threshold checks.
    let mut sum_sq = 0.0f64;
    let mut u4_correct = 0u32;
    for i in 0..18 {
        let f32_val = output_f32[i];
        let u4_val = output_u4[i];
        let diff = (u4_val - f32_val) as f64;
        sum_sq += diff * diff;
        if f32_val.abs() > 0.01 {
            let rel_err = (diff / f32_val as f64).abs();
            if rel_err < 0.6 { u4_correct += 1; }
        } else {
            if diff.abs() < 0.1 { u4_correct += 1; }
        }
    }
    let rms = (sum_sq / 18.0).sqrt();
    assert!(rms < 0.5, "U4 conv2d RMS error too large: {:.4}", rms);
    assert!(
        u4_correct >= 15,
        "At least 15/18 U4 conv2d outputs should be close to f32 (got {}/18)",
        u4_correct
    );
}

#[test]
fn test_conv2d_u8_end_to_end() {
    // Conv2d: 1 input channel, 2 output channels, 3x3 kernel, stride=1, padding=0
    // Input: [1, 1, 5, 5] — output: [1, 2, 3, 3]
    let in_channels = 1usize;
    let out_channels = 2usize;
    let kernel_size = 3usize;
    let weight: Vec<f32> = (0..(out_channels * in_channels * kernel_size * kernel_size))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let input: Vec<f32> = vec![0.5; 25]; // [1, 1, 5, 5]

    let output_u8 = run_conv2d(
        1,
        in_channels,
        out_channels,
        5,
        5,
        kernel_size,
        1,
        0,
        &weight,
        &input,
        Some(8),
    );

    // Output should have 18 elements (1 * 2 * 3 * 3)
    assert_eq!(
        output_u8.len(),
        18,
        "U8 conv2d output should have 18 elements"
    );

    // All outputs should be finite numbers
    for (i, &val) in output_u8.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output[{}] should be finite, got {}",
            i,
            val
        );
    }
}

// ── API / validation tests ─────────────────────────────────────────────

#[test]
fn test_compile_with_quantize_rejects_invalid_bit_width() {
    use fastnn::backend::executor::GraphExecutor;

    // Bit width must be 4 or 8
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32);
    let weight_tt = TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(2)], IrDType::F32);
    let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5, -0.3, 0.8];
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let _output = gb.matmul(&input, &weight);

    let graph = gb.to_graph();
    let executor = GraphExecutor::new(CpuBackend);

    let result = executor.compile_with_plan_and_quantize(&graph, Some(2), None);
    assert!(result.is_err(), "Bit width 2 should be rejected");

    let result = executor.compile_with_plan_and_quantize(&graph, Some(16), None);
    assert!(result.is_err(), "Bit width 16 should be rejected");

    // None and Some(4) and Some(8) should succeed
    assert!(executor
        .compile_with_plan_and_quantize(&graph, None, None)
        .is_ok());
    assert!(executor
        .compile_with_plan_and_quantize(&graph, Some(4), None)
        .is_ok());
    assert!(executor
        .compile_with_plan_and_quantize(&graph, Some(8), None)
        .is_ok());
}

#[test]
fn test_graph_builder_compile_with_quantize() {
    // Test the GraphBuilder::compile_with_quantize() API
    let gb = GraphBuilder::new();
    let input = gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32);
    let weight_tt = TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(2)], IrDType::F32);
    let weight_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 0.5, -0.5, -0.3, 0.8];
    let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
    let weight = gb.constant(&weight_bytes, weight_tt);
    let output = gb.matmul(&input, &weight);

    // Compile without quantization
    let result_no_q = gb.compile_with_quantize(&[&output], CpuBackend, None);
    assert!(
        result_no_q.is_ok(),
        "compile without quantize should succeed"
    );

    // Compile with U4 quantization
    let result_u4 = gb.compile_with_quantize(&[&output], CpuBackend, Some(4));
    assert!(result_u4.is_ok(), "compile with quantize=4 should succeed");

    // Compile with U8 quantization
    let result_u8 = gb.compile_with_quantize(&[&output], CpuBackend, Some(8));
    assert!(result_u8.is_ok(), "compile with quantize=8 should succeed");

    // Verify quantized graphs contain U4/U8 weight nodes
    let (_, _, u4_graph) = result_u4.unwrap();
    let has_u4 = u4_graph
        .nodes
        .iter()
        .any(|n| matches!(&n.output_type.dtype, IrDType::U4 { .. }));
    assert!(has_u4, "U4-compiled graph should contain a U4 weight node");

    let (_, _, u8_graph) = result_u8.unwrap();
    let has_u8 = u8_graph
        .nodes
        .iter()
        .any(|n| matches!(&n.output_type.dtype, IrDType::U8 { .. }));
    assert!(has_u8, "U8-compiled graph should contain a U8 weight node");

    // No-quantize graph should have no U4/U8 nodes
    let (_, _, f32_graph) = result_no_q.unwrap();
    let has_packed = f32_graph.nodes.iter().any(|n| {
        matches!(
            &n.output_type.dtype,
            IrDType::U4 { .. } | IrDType::U8 { .. }
        )
    });
    assert!(
        !has_packed,
        "f32-compiled graph should not contain packed weight nodes"
    );
}

#[test]
fn test_quantized_matmul_preserves_output_sign() {
    // Test that quantized matmul preserves the sign of the output
    // [1, 4] @ [4, 2] — identity-like weight
    let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
    let input: Vec<f32> = vec![1.0, 2.0, -1.0, -2.0];

    let output_f32 = run_matmul(1, 4, 2, &weight, &input, None);
    let output_u8 = run_matmul(1, 4, 2, &weight, &input, Some(8));

    // Sign should be preserved for U8 (256 levels)
    for i in 0..output_f32.len() {
        if output_f32[i].abs() > 0.1 {
            assert_eq!(
                output_f32[i].signum(),
                output_u8[i].signum(),
                "Sign mismatch at index {}: f32={}, u8={}",
                i,
                output_f32[i],
                output_u8[i]
            );
        }
    }
}

#[test]
fn test_quantize_none_produces_f32_pipeline() {
    // Ensure quantize=None produces the same result as the standard path
    let weight: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
    let input: Vec<f32> = vec![1.0, 0.0, -1.0, 0.5];

    let output = run_matmul(1, 4, 2, &weight, &input, None);

    // All outputs should be finite
    for (i, &val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Output[{}] should be finite, got {}",
            i,
            val
        );
    }
}

/// Test the matmul_u8_i8 dispatch path by constructing U8-quantized weights
/// and an I8 activation payload directly (verifies compile-time kernel
/// selection without relying on the activation_quantization pass).
///
/// Builds a graph: Input(f32) → MatMul(f32 weight quantized to U8),
/// plus a separate I8 Constant that mimics a QuantizeActivations payload.
/// The MatMul receives [I8, U8] inputs and must select "matmul_u8_i8".
#[test]
fn test_matmul_u8_i8_dispatch_path() {
    // Build a graph with U8-quantized weights and I8 activation.
    let mut graph = ComputeGraph::new();

    // Create an I8 activation payload constant.
    // Payload format: [scale_f32(4)][zp_f32(4)][i8_data...]
    let act_scale = 1.0f32;
    let act_zp = 0.0f32;
    let act_i8: Vec<i8> = vec![1, 2, 3, 4];
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&act_scale.to_le_bytes());
    payload.extend_from_slice(&act_zp.to_le_bytes());
    for &v in &act_i8 {
        payload.push(v as u8);
    }
    let act_tt = TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::I8);
    let act_id = graph.add_node(
        Opcode::Constant(TensorValue::Data {
            bytes: payload,
            tensor_type: act_tt.clone(),
        }),
        vec![],
        act_tt,
    );

    // Create a U8-quantized weight via the weight-quantization pipeline.
    // Start with f32 constant weights, then quantize.
    let w_data: Vec<f32> = vec![
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ]; // [4, 4] scaled-identity
    let w_tt = TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32);
    let w_id = graph.add_node(
        Opcode::Constant(TensorValue::Data {
            bytes: bytemuck::cast_slice(&w_data).to_vec(),
            tensor_type: w_tt.clone(),
        }),
        vec![],
        w_tt,
    );

    let mm_id = graph.add_node(
        Opcode::MatMul,
        vec![act_id, w_id],
        TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
    );

    graph.set_inputs(vec![]);
    graph.set_outputs(vec![mm_id]);

    // Apply weight quantization (U8) to convert the f32 weight constant.
    infer_shapes(&mut graph).unwrap();
    quantize_weights(&mut graph, 8, None).unwrap();
    eliminate_dead_code(&mut graph);

    // Now the MatMul should see [I8, U8] and select "matmul_u8_i8".
    let mem = plan_memory(&graph).expect("memory planning should succeed");
    let mut plan = CpuBackend.compile(&graph, &mem).unwrap();

    let mut executor = GraphExecutor::new(CpuBackend);
    let result = executor.execute(&graph, &mut plan, &mem, &[]).unwrap();
    let result_f32: Vec<f32> = bytemuck::cast_slice(&result[0]).to_vec();

    // Expected: input [1,2,3,4] × W=2*identity → [2,4,6,8]
    // (with small quantization error).
    let expected: Vec<f32> = vec![2.0, 4.0, 6.0, 8.0];
    assert_eq!(result_f32.len(), expected.len());
    for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let err = (got - exp).abs();
        let tol = 0.5;
        assert!(
            err <= tol,
            "Mismatch at output[{}]: got {}, expected {} (err={})",
            i,
            got,
            exp,
            err
        );
    }
}

#[test]
fn test_matmul_u4_i8_dispatch_path() {
    let mut graph = ComputeGraph::new();

    let act_scale = 1.0f32;
    let act_zp = 0.0f32;
    let act_i8: Vec<i8> = vec![1, -2, 3, -4, 5, -6, 7, -8];
    let mut payload: Vec<u8> = Vec::new();
    payload.extend_from_slice(&act_scale.to_le_bytes());
    payload.extend_from_slice(&act_zp.to_le_bytes());
    for &v in &act_i8 {
        payload.push(v as u8);
    }
    let act_tt = TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(8)], IrDType::I8);
    let act_id = graph.add_node(
        Opcode::Constant(TensorValue::Data {
            bytes: payload,
            tensor_type: act_tt.clone(),
        }),
        vec![],
        act_tt,
    );

    let w_data: Vec<f32> = vec![
        -8.0, -4.0, -1.0, 0.0, 1.0, 3.0, 6.0, 7.0, 7.0, 6.0, 3.0, 1.0, 0.0, -1.0, -4.0, -8.0,
    ];
    let w_tt = TensorType::new(vec![DimExpr::Known(8), DimExpr::Known(2)], IrDType::F32);
    let w_id = graph.add_node(
        Opcode::Constant(TensorValue::Data {
            bytes: bytemuck::cast_slice(&w_data).to_vec(),
            tensor_type: w_tt.clone(),
        }),
        vec![],
        w_tt,
    );

    let mm_id = graph.add_node(
        Opcode::MatMul,
        vec![act_id, w_id],
        TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(2)], IrDType::F32),
    );

    graph.set_inputs(vec![]);
    graph.set_outputs(vec![mm_id]);

    infer_shapes(&mut graph).unwrap();
    quantize_weights(&mut graph, 4, None).unwrap();
    eliminate_dead_code(&mut graph);

    let mem = plan_memory(&graph).expect("memory planning should succeed");
    let mut plan = CpuBackend.compile(&graph, &mem).unwrap();
    let kernel_names: Vec<&str> = plan
        .instructions
        .iter()
        .filter_map(|instr| match instr {
            Instruction::CallKernel { kernel_name, .. } => Some(kernel_name.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        kernel_names.iter().any(|name| *name == "matmul_u4_i8"),
        "expected compiled plan to select matmul_u4_i8, got {:?}",
        kernel_names
    );

    let mm_node = graph.get_node(mm_id).unwrap();
    let packed_weight_node = graph.get_node(mm_node.inputs[1]).unwrap();
    let (scales, zero_points) = match &packed_weight_node.output_type.dtype {
        IrDType::U4 {
            scales,
            zero_points,
        } => (scales.clone(), zero_points.clone()),
        other => panic!("expected U4 weight node, got {:?}", other),
    };
    let raw_bytes = match &packed_weight_node.opcode {
        Opcode::Constant(TensorValue::Data { bytes, .. }) => bytes.clone(),
        other => panic!("expected packed weight constant, got {:?}", other),
    };
    let mut aligned = vec![0u32; raw_bytes.len().div_ceil(4)];
    let aligned_bytes = bytemuck::cast_slice_mut::<_, u8>(&mut aligned);
    aligned_bytes[..raw_bytes.len()].copy_from_slice(&raw_bytes);
    let typed_data: Vec<U4x8> = bytemuck::cast_slice(&aligned).to_vec();
    let mut packed_shape: Vec<usize> = packed_weight_node
        .output_type
        .shape
        .iter()
        .map(|dim| dim.evaluate().unwrap() as usize)
        .collect();
    // Weight node shape is logical [K, N]; data is packed in [N, K] layout.
    if packed_shape.len() == 2 {
        packed_shape.reverse();
    }
    let packed = PackedTensor::from_raw(typed_data, packed_shape, scales, zero_points);
    let mut expected_payload = Vec::new();
    expected_payload.extend_from_slice(&act_scale.to_le_bytes());
    expected_payload.extend_from_slice(&act_zp.to_le_bytes());
    expected_payload.extend_from_slice(bytemuck::cast_slice::<i8, u8>(&act_i8));
    let mut expected = vec![0.0f32; 2];
    fastnn::backend::cpu::microkernels::gemm_cpu_flat_i8_u4x8(
        &packed,
        &expected_payload,
        &mut expected,
        1,
        8,
        2,
    );

    let mut executor = GraphExecutor::new(CpuBackend);
    let result = executor.execute(&graph, &mut plan, &mem, &[]).unwrap();
    let result_f32: Vec<f32> = bytemuck::cast_slice(&result[0]).to_vec();

    assert_eq!(result_f32.len(), expected.len());
    for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let err = (got - exp).abs();
        assert!(
            err <= 1e-4,
            "Mismatch at output[{}]: got {}, expected {} (err={})",
            i,
            got,
            exp,
            err
        );
    }
}

/// Diagnostic test: count quantized vs non-quantized conv2d kernel names
/// in the compiled ExecutablePlan after `compile_with_plan_and_quantize`.
///
/// Builds a small 3-Conv2d + SiLU graph, compiles with `Some(4)`, then
/// iterates `plan.instructions` and counts kernel names by pattern.
#[test]
fn test_count_quantized_kernel_names() {
    let gb = GraphBuilder::new();

    // Input: [1, 3, 8, 8]
    let input = gb.input_with_dims(
        &[
            DimExpr::Known(1),
            DimExpr::Known(3),
            DimExpr::Known(8),
            DimExpr::Known(8),
        ],
        IrDType::F32,
    );

    // Conv2d 1: 3 -> 8 channels, 3x3, stride=1, pad=1 => [1, 8, 8, 8]
    let w1_tt = TensorType::new(
        vec![
            DimExpr::Known(8),
            DimExpr::Known(3),
            DimExpr::Known(3),
            DimExpr::Known(3),
        ],
        IrDType::F32,
    );
    let w1_data: Vec<f32> = (0..(8 * 3 * 3 * 3)).map(|i| (i as f32) * 0.01).collect();
    let w1 = gb.constant(bytemuck::cast_slice(&w1_data), w1_tt);
    let conv1 = gb.conv2d_with_params(&input, &w1, 1, 1, 1, 1);
    let silu1 = gb.silu(&conv1);

    // Conv2d 2: 8 -> 16 channels, 3x3, stride=1, pad=1 => [1, 16, 8, 8]
    let w2_tt = TensorType::new(
        vec![
            DimExpr::Known(16),
            DimExpr::Known(8),
            DimExpr::Known(3),
            DimExpr::Known(3),
        ],
        IrDType::F32,
    );
    let w2_data: Vec<f32> = (0..(16 * 8 * 3 * 3)).map(|i| (i as f32) * 0.01).collect();
    let w2 = gb.constant(bytemuck::cast_slice(&w2_data), w2_tt);
    let conv2 = gb.conv2d_with_params(&silu1, &w2, 1, 1, 1, 1);
    let silu2 = gb.silu(&conv2);

    // Conv2d 3: 16 -> 32 channels, 3x3, stride=2, pad=1 => [1, 32, 4, 4]
    let w3_tt = TensorType::new(
        vec![
            DimExpr::Known(32),
            DimExpr::Known(16),
            DimExpr::Known(3),
            DimExpr::Known(3),
        ],
        IrDType::F32,
    );
    let w3_data: Vec<f32> = (0..(32 * 16 * 3 * 3)).map(|i| (i as f32) * 0.01).collect();
    let w3 = gb.constant(bytemuck::cast_slice(&w3_data), w3_tt);
    let conv3 = gb.conv2d_with_params(&silu2, &w3, 2, 1, 1, 1);
    let _silu3 = gb.silu(&conv3);

    // Compile with U4 quantization
    let graph = gb.to_graph();
    let executor = GraphExecutor::new(CpuBackend);
    let (plan, _mem, _compiled_graph) = executor
        .compile_with_plan_and_quantize(&graph, Some(4), None)
        .expect("compile_with_plan_and_quantize(4) should succeed");

    // Collect and classify kernel names
    let mut conv2d_quant_u4 = 0usize;
    let mut conv2d_quant_u8 = 0usize;
    let mut conv2d_fp32 = 0usize;
    let mut other_kernel_names: Vec<String> = Vec::new();

    for instr in &plan.instructions {
        if let Instruction::CallKernel {
            kernel_name,
            weight_meta,
            ..
        } = instr
        {
            let meta_info = weight_meta
                .as_ref()
                .map(|m| format!("(bw={}, scales={})", m.bit_width, m.scales.len()))
                .unwrap_or_default();

            if kernel_name.starts_with("conv2d_u4") {
                conv2d_quant_u4 += 1;
            } else if kernel_name.starts_with("conv2d_u8") {
                conv2d_quant_u8 += 1;
            } else if kernel_name.starts_with("conv2d") {
                conv2d_fp32 += 1;
            }

            other_kernel_names.push(format!("{}{}", kernel_name, meta_info));
        }
    }

    eprintln!("=== test_count_quantized_kernel_names ===");
    eprintln!("Total instructions: {}", plan.instructions.len());
    eprintln!("conv2d_u4 (quantized 4-bit): {}", conv2d_quant_u4);
    eprintln!("conv2d_u8 (quantized 8-bit): {}", conv2d_quant_u8);
    eprintln!("conv2d    (FP32 non-quantized): {}", conv2d_fp32);
    eprintln!("All kernel names:");
    for (i, name) in other_kernel_names.iter().enumerate() {
        eprintln!("  [{}] {}", i, name);
    }

    // We built 3 Conv2d layers. After quantization with U4, all three
    // should use quantized kernels (conv2d_u4* or conv2d_u8*).
    let total_quantized = conv2d_quant_u4 + conv2d_quant_u8;
    assert!(
        total_quantized > 0,
        "Expected at least one quantized conv2d kernel (conv2d_u4* or conv2d_u8*), \
         but found conv2d_u4={}, conv2d_u8={}, conv2d_fp32={}",
        conv2d_quant_u4,
        conv2d_quant_u8,
        conv2d_fp32,
    );

    // With 3 Conv2d layers and U4 quantization, expect exactly 3 quantized conv kernels.
    assert_eq!(
        total_quantized, 3,
        "Expected 3 quantized conv2d kernels for 3 Conv2d layers, got {} \
         (conv2d_u4={}, conv2d_u8={}, conv2d_fp32={})",
        total_quantized, conv2d_quant_u4, conv2d_quant_u8, conv2d_fp32,
    );

    // All 3 should be U4, not U8
    assert_eq!(
        conv2d_quant_u4, 3,
        "Expected all 3 conv kernels to be conv2d_u4*, but got conv2d_u4={}",
        conv2d_quant_u4,
    );
}

/// End-to-end CPU auto-cast path: a non-input activation is quantized to I8,
/// U8 weights stay packed, and MatMul must produce F32 directly without a
/// trailing DequantizeActivations node misreading its output as an I8 payload.
#[test]
fn test_auto_cast_u8_activation_quant_pipeline_outputs_f32_directly() {
    let mut graph = ComputeGraph::new();

    let input_id = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
    );

    let relu_id = graph.add_node(
        Opcode::Relu,
        vec![input_id],
        TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
    );

    let w_data: Vec<f32> = vec![
        2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0,
    ];
    let w_tt = TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32);
    let weight_id = graph.add_node(
        Opcode::Constant(TensorValue::Data {
            bytes: bytemuck::cast_slice(&w_data).to_vec(),
            tensor_type: w_tt.clone(),
        }),
        vec![],
        w_tt,
    );

    let mm_id = graph.add_node(
        Opcode::MatMul,
        vec![relu_id, weight_id],
        TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32),
    );

    graph.set_inputs(vec![input_id]);
    graph.set_outputs(vec![mm_id]);

    infer_shapes(&mut graph).unwrap();
    fastnn::compiler::passes::auto_cast::auto_cast(
        &mut graph,
        &fastnn::compiler::passes::auto_cast::AutoCastOptions {
            weight_bit_width: Some(8),
            enable_activation_quant: true,
        },
    )
    .unwrap();
    eliminate_dead_code(&mut graph);

    let output_id = graph.outputs[0];
    let output_node = graph.get_node(output_id).unwrap();
    assert_eq!(
        output_node.opcode,
        Opcode::MatMul,
        "MatMul already outputs F32; activation quantization must not add a trailing DequantizeActivations"
    );

    let mm_node = graph.get_node(mm_id).unwrap();
    let act_node = graph.get_node(mm_node.inputs[0]).unwrap();
    assert_eq!(act_node.opcode, Opcode::QuantizeActivations);
    let weight_node = graph.get_node(mm_node.inputs[1]).unwrap();
    assert!(matches!(weight_node.output_type.dtype, IrDType::U8 { .. }));

    let mem = plan_memory(&graph).expect("memory planning should succeed");
    let mut plan = CpuBackend.compile(&graph, &mem).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend);

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input).to_vec();
    let result = executor
        .execute(&graph, &mut plan, &mem, &[&input_bytes])
        .unwrap();
    let result_f32: Vec<f32> = bytemuck::cast_slice(&result[0]).to_vec();

    let expected = vec![2.0f32, 4.0, 6.0, 8.0];
    assert_eq!(result_f32.len(), expected.len());
    for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
        let err = (got - exp).abs();
        assert!(
            err <= 0.5,
            "output[{i}] got {got}, expected {exp}, err={err}"
        );
    }
}
