//! Tests for ONNX export of quantized patterns (QLinearMatMul, QLinearConv).
//!
//! Builds graphs with Dequantize → MatMul/Conv2d → Quantize patterns,
//! exports them via `export_to_onnx_json`, and verifies the fused
//! QLinearMatMul / QLinearConv nodes appear in the JSON output.

use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::node::*;
use fastnn::onnx::export::export_to_onnx_json;

/// Helper: build a graph with a Dequantize → MatMul → Quantize pattern.
fn build_qmatmul_graph() -> ComputeGraph {
    let gb = GraphBuilder::new();

    // Activation input: [1, 2]
    let input = gb.input(&[1, 2], IrDType::F32);

    // Quantized weight: [2, 4] (inner K=2, outer N=4)
    let weight_shape = vec![DimExpr::Known(2), DimExpr::Known(4)];
    let weight_tt = TensorType::new(
        weight_shape,
        IrDType::U4 {
            scales: vec![0.1, 0.2, 0.3, 0.4],
            zero_points: vec![0.0, 0.0, 0.0, 0.0],
        },
    );
    let weight_data = vec![0u8; 80]; // enough for packed U4
    let weight = gb.constant(&weight_data, weight_tt);

    // Dequantize weight U4 → F32
    let deq_weight = gb.dequantize(&weight);

    // MatMul: [1,2] @ [2,4] → [1,4]
    let mm = gb.matmul(&input, &deq_weight);

    // Quantize output F32 → U4
    let _q_out = gb.quantize(&mm, 4);

    gb.to_graph()
}

#[test]
fn test_export_qlinear_matmul_detected() {
    let graph = build_qmatmul_graph();
    let json = export_to_onnx_json(&graph).expect("export should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse");

    let nodes = parsed["nodes"]
        .as_array()
        .expect("nodes should be an array");

    // QLinearMatMul should appear instead of separate Quantize/Dequantize/MatMul
    let has_qlinear = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("QLinearMatMul"));
    assert!(
        has_qlinear,
        "QLinearMatMul should appear in exported nodes, got: {:?}",
        nodes
            .iter()
            .map(|n| n["op_type"].as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
    );

    // Dequantize and Quantize should be fused (not appear as separate nodes)
    let has_deq = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("Dequantize"));
    assert!(!has_deq, "Dequantize should be fused into QLinearMatMul");

    let has_q = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("Quantize"));
    assert!(!has_q, "Quantize should be fused into QLinearMatMul");

    // Verify scale/zp params were added
    let params = parsed["params"]
        .as_object()
        .expect("params should be an object");
    let scale_param_count = params
        .keys()
        .filter(|k| k.contains("scale") || k.contains("zp"))
        .count();
    assert!(
        scale_param_count >= 4,
        "Should have at least 4 scale/zp params, got {}",
        scale_param_count
    );
}

#[test]
fn test_export_qlinear_matmul_attrs() {
    let graph = build_qmatmul_graph();
    let json = export_to_onnx_json(&graph).expect("export should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse");

    let nodes = parsed["nodes"]
        .as_array()
        .expect("nodes should be an array");

    // Find the QLinearMatMul node
    let qlinear = nodes
        .iter()
        .find(|n| n["op_type"].as_str() == Some("QLinearMatMul"))
        .expect("QLinearMatMul node should exist");

    // Verify scale/zp attrs
    assert!(
        qlinear["a_scale"].as_str().is_some(),
        "a_scale attr should exist"
    );
    assert!(
        qlinear["b_scale"].as_str().is_some(),
        "b_scale attr should exist"
    );
    assert!(
        qlinear["y_scale"].as_str().is_some(),
        "y_scale attr should exist"
    );
    assert!(
        qlinear["y_zero_point"].as_str().is_some(),
        "y_zero_point attr should exist"
    );

    // Verify inputs string has 8 comma-separated entries
    let inputs = qlinear["inputs"].as_str().expect("inputs should be string");
    let input_parts: Vec<&str> = inputs.split(',').collect();
    assert_eq!(
        input_parts.len(),
        9,
        "QLinearMatMul should have 9 inputs (A, A_scale, A_zp, B, B_scale, B_zp, Y_scale, Y_zp + activation), got {}: {:?}",
        input_parts.len(),
        input_parts
    );
}

#[test]
fn test_export_non_quantized_matmul_unchanged() {
    // A regular MatMul (no Dequantize/Quantize wrapping) should still
    // export as a plain "MatMul" node.
    let gb = GraphBuilder::new();
    let a = gb.input(&[1, 4], IrDType::F32);
    let b = gb.input(&[4, 8], IrDType::F32);
    let _mm = gb.matmul(&a, &b);
    let graph = gb.to_graph();

    let json = export_to_onnx_json(&graph).expect("export should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse");

    let nodes = parsed["nodes"]
        .as_array()
        .expect("nodes should be an array");
    let has_matmul = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("MatMul"));
    assert!(has_matmul, "Plain MatMul should still appear");
    let has_qlinear = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("QLinearMatMul"));
    assert!(
        !has_qlinear,
        "QLinearMatMul should NOT appear for non-quantized graph"
    );
}

#[test]
fn test_export_qlinear_conv_detected() {
    // Build: Dequantize(weight) → Conv2d → Quantize
    let gb = GraphBuilder::new();

    // Input: [1, 1, 4, 4] (NCHW)
    let input = gb.input(&[1, 1, 4, 4], IrDType::F32);

    // Weight: [1, 1, 3, 3] (out_channels, in_channels, KH, KW)
    let weight_shape = vec![
        DimExpr::Known(1),
        DimExpr::Known(1),
        DimExpr::Known(3),
        DimExpr::Known(3),
    ];
    let weight_tt = TensorType::new(
        weight_shape,
        IrDType::U4 {
            scales: vec![0.1],
            zero_points: vec![0.0],
        },
    );
    let weight_data = vec![0u8; 80]; // enough for packed U4
    let weight = gb.constant(&weight_data, weight_tt);

    // Dequantize weight
    let deq_weight = gb.dequantize(&weight);

    // Conv2d
    let conv = gb.conv2d_with_params(&input, &deq_weight, 1, 0, 1, 1);

    // Quantize output
    let _q_out = gb.quantize(&conv, 4);

    let graph = gb.to_graph();

    let json = export_to_onnx_json(&graph).expect("export should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse");

    let nodes = parsed["nodes"]
        .as_array()
        .expect("nodes should be an array");

    let has_qlinear_conv = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("QLinearConv"));
    assert!(
        has_qlinear_conv,
        "QLinearConv should appear in exported nodes, got: {:?}",
        nodes
            .iter()
            .map(|n| n["op_type"].as_str().unwrap_or("?"))
            .collect::<Vec<_>>()
    );

    // Verify Dequantize/Quantize fused
    let has_deq = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("Dequantize"));
    assert!(!has_deq, "Dequantize should be fused into QLinearConv");
}
