//! Tests for ONNX export training safety contract.
//!
//! Verifies that training-only opcodes (optimizer updates, gradient scaling)
//! produce explicit errors rather than being silently dropped from the export.
//! This prevents "wrong graph" bugs where a training graph is exported as
//! inference-only without the user's knowledge.

use fastnn::ir::builder::GraphBuilder;
use fastnn::ir::*;
use fastnn::onnx::export::{
    detect_training_ops, export_to_onnx_json, export_to_onnx_json_with_config, ExportConfig,
};

// ── Helper: build a simple inference graph ────────────────────────────

fn build_inference_graph() -> ComputeGraph {
    let gb = GraphBuilder::new();
    let x = gb.input(&[1, 4], IrDType::F32);
    let w = gb.constant(
        &vec![0.0f32; 16]
            .into_iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>(),
        TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(4)], IrDType::F32),
    );
    let mm = gb.matmul(&x, &w);
    let _ = gb.relu(&mm);
    gb.to_graph()
}

// ── Helper: build a training graph with SGD ───────────────────────────

fn build_training_graph_sgd() -> ComputeGraph {
    let gb = GraphBuilder::new();
    let x = gb.input(&[1, 4], IrDType::F32);
    let w = gb.input(&[4, 4], IrDType::F32);
    let mm = gb.matmul(&x, &w);
    let loss = gb.reduce_mean(&mm, 0, false);
    // SGD update: w -= lr * grad
    let _updated = gb.apply_sgd(&w, &loss, 0.01, 0.0);
    gb.to_graph()
}

// ── Helper: build a training graph with Adam ──────────────────────────

fn build_training_graph_adam() -> ComputeGraph {
    let gb = GraphBuilder::new();
    let x = gb.input(&[1, 4], IrDType::F32);
    let w = gb.input(&[4, 4], IrDType::F32);
    let m = gb.input(&[4, 4], IrDType::F32);
    let v = gb.input(&[4, 4], IrDType::F32);
    let mm = gb.matmul(&x, &w);
    let loss = gb.reduce_mean(&mm, 0, false);
    // Adam update
    let _updated = gb.apply_adam(&w, &loss, &m, &v, 0.001, 0.9, 0.999, 1e-8, 1);
    gb.to_graph()
}

// ── Helper: build a training graph with gradient scaling ──────────────

fn build_training_graph_gradient_scale() -> ComputeGraph {
    let gb = GraphBuilder::new();
    let x = gb.input(&[1, 4], IrDType::F32);
    let w = gb.input(&[4, 4], IrDType::F32);
    let mm = gb.matmul(&x, &w);
    let loss = gb.reduce_mean(&mm, 0, false);
    // GradientScale
    let _scaled = gb.gradient_scale(&loss, 2.0);
    gb.to_graph()
}

// ── Tests: Inference export should succeed ────────────────────────────

#[test]
fn test_inference_graph_exports_successfully() {
    let graph = build_inference_graph();
    let result = export_to_onnx_json(&graph);
    assert!(
        result.is_ok(),
        "Inference graph should export: {:?}",
        result.err()
    );
}

#[test]
fn test_inference_export_metadata() {
    let graph = build_inference_graph();
    let json = export_to_onnx_json(&graph).expect("export should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse");

    let metadata = parsed["metadata"]
        .as_object()
        .expect("metadata should be object");
    assert_eq!(metadata["export_mode"].as_str(), Some("inference"));
    assert_eq!(metadata["training_ops_dropped"].as_bool(), Some(false));
    assert_eq!(metadata["training_ops_count"].as_u64(), Some(0));
}

// ── Tests: Training graphs should fail by default ─────────────────────

#[test]
fn test_training_graph_sgd_fails_export() {
    let graph = build_training_graph_sgd();
    let result = export_to_onnx_json(&graph);
    assert!(
        result.is_err(),
        "SGD training graph should fail export by default"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("training-only opcode") || err.contains("SgdUpdate"),
        "Error should mention training opcodes: {}",
        err
    );
    assert!(
        err.contains("wrong graph"),
        "Error should warn about wrong graph: {}",
        err
    );
}

#[test]
fn test_training_graph_adam_fails_export() {
    let graph = build_training_graph_adam();
    let result = export_to_onnx_json(&graph);
    assert!(
        result.is_err(),
        "Adam training graph should fail export by default"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("AdamUpdate"),
        "Error should mention AdamUpdate: {}",
        err
    );
}

#[test]
fn test_gradient_scale_fails_export() {
    let graph = build_training_graph_gradient_scale();
    let result = export_to_onnx_json(&graph);
    assert!(
        result.is_err(),
        "GradientScale graph should fail export by default"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("GradientScale"),
        "Error should mention GradientScale: {}",
        err
    );
}

// ── Tests: detect_training_ops function ───────────────────────────────

#[test]
fn test_detect_training_ops_inference_clean() {
    let graph = build_inference_graph();
    let ops = detect_training_ops(&graph);
    assert!(
        ops.is_empty(),
        "Inference graph should have no training ops"
    );
}

#[test]
fn test_detect_training_ops_sgd() {
    let graph = build_training_graph_sgd();
    let ops = detect_training_ops(&graph);
    assert!(!ops.is_empty(), "SGD graph should detect training ops");
    assert!(
        ops.iter().any(|(_, name)| name.contains("SgdUpdate")),
        "Should detect SgdUpdate, got: {:?}",
        ops
    );
}

#[test]
fn test_detect_training_ops_adam() {
    let graph = build_training_graph_adam();
    let ops = detect_training_ops(&graph);
    assert!(!ops.is_empty(), "Adam graph should detect training ops");
    assert!(
        ops.iter().any(|(_, name)| name.contains("AdamUpdate")),
        "Should detect AdamUpdate, got: {:?}",
        ops
    );
}

#[test]
fn test_detect_training_ops_gradient_scale() {
    let graph = build_training_graph_gradient_scale();
    let ops = detect_training_ops(&graph);
    assert!(
        !ops.is_empty(),
        "GradientScale graph should detect training ops"
    );
    assert!(
        ops.iter().any(|(_, name)| name.contains("GradientScale")),
        "Should detect GradientScale, got: {:?}",
        ops
    );
}

// ── Tests: Opt-in to drop training ops (dangerous) ────────────────────

#[test]
fn test_training_graph_sgd_optout_drops_ops() {
    let graph = build_training_graph_sgd();
    let config = ExportConfig {
        fail_on_training_ops: false,
    };
    let result = export_to_onnx_json_with_config(&graph, &config);
    // Should succeed but metadata should indicate training ops were dropped
    let json = result.expect("export with opt-out should succeed");
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse");

    let metadata = parsed["metadata"]
        .as_object()
        .expect("metadata should be object");
    assert_eq!(metadata["training_ops_dropped"].as_bool(), Some(true));
    let count = metadata["training_ops_count"].as_u64().unwrap_or(0);
    assert!(count > 0, "Should report dropped training ops count > 0");
}

// ── Tests: Error message is actionable ────────────────────────────────

#[test]
fn test_error_message_suggests_remediation() {
    let graph = build_training_graph_sgd();
    let result = export_to_onnx_json(&graph);
    let err = result.unwrap_err();
    assert!(
        err.contains("remove training nodes"),
        "Error should suggest removing training nodes: {}",
        err
    );
    assert!(
        err.contains("ExportConfig"),
        "Error should mention ExportConfig: {}",
        err
    );
}

// ── Tests: Quantized export still works (regression guard) ────────────

#[test]
fn test_quantized_matmul_export_still_works() {
    // Reproduces the pattern from onnx_export_quantized.rs to ensure
    // the new training safety checks don't break existing behavior.
    let gb = GraphBuilder::new();
    let input = gb.input(&[1, 2], IrDType::F32);
    let weight_shape = vec![DimExpr::Known(2), DimExpr::Known(4)];
    let weight_tt = TensorType::new(
        weight_shape,
        IrDType::I4 {
            scales: vec![0.1, 0.2, 0.3, 0.4],
            dequant_offsets: vec![0.0, 0.0, 0.0, 0.0],
            codebooks: vec![],
        },
    );
    let weight_data = vec![0u8; 80];
    let weight = gb.constant(&weight_data, weight_tt);
    let deq_weight = gb.dequantize(&weight);
    let mm = gb.matmul(&input, &deq_weight);
    let _q_out = gb.quantize(&mm, 4);
    let graph = gb.to_graph();

    let result = export_to_onnx_json(&graph);
    assert!(
        result.is_ok(),
        "Quantized matmul export should still work: {:?}",
        result.err()
    );

    let json = result.unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON should parse");
    let nodes = parsed["nodes"].as_array().expect("nodes should be array");

    let has_qlinear = nodes
        .iter()
        .any(|n| n["op_type"].as_str() == Some("QLinearMatMul"));
    assert!(
        has_qlinear,
        "QLinearMatMul should still appear in quantized export"
    );
}

// ── Tests: Default config fails on training ops ───────────────────────

#[test]
fn test_default_config_is_fail_on_training() {
    let config = ExportConfig::default();
    assert!(
        config.fail_on_training_ops,
        "Default config should fail on training ops"
    );
}
