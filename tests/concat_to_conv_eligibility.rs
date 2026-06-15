//! Integration tests for the concat→Conv2d segmented planning guardrail.
//!
//! These tests exercise `check_concat_to_conv_eligibility` through the
//! public API, validating that safe patterns are accepted (GREEN) and
//! unsafe patterns are rejected (RED).

use fastnn::compiler::passes::concat_to_conv_eligibility::{
    check_concat_to_conv_eligibility, Ineligible,
};
use fastnn::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType};
use std::collections::HashMap;

fn nchw(n: u64, c: u64, h: u64, w: u64) -> TensorType {
    TensorType::new(
        vec![
            DimExpr::Known(n),
            DimExpr::Known(c),
            DimExpr::Known(h),
            DimExpr::Known(w),
        ],
        IrDType::F32,
    )
}

/// Build the canonical eligible pattern: concat(axis=1) → Conv2d(groups=1, dilation=1).
fn build_eligible_graph() -> (ComputeGraph, NodeId, NodeId) {
    let mut graph = ComputeGraph::new();

    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8));

    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat =
        graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(1, 24, 8, 8), attrs);

    let weight = graph.add_node(Opcode::Input, vec![], nchw(32, 24, 1, 1));
    let mut conv_attrs = HashMap::new();
    conv_attrs.insert("groups".to_string(), "1".to_string());
    conv_attrs.insert("dilation".to_string(), "1".to_string());
    let conv = graph.add_node_with_attrs(
        Opcode::Conv2d,
        vec![concat, weight],
        nchw(1, 32, 8, 8),
        conv_attrs,
    );

    graph.set_inputs(vec![in_a, in_b, weight]);
    graph.set_outputs(vec![conv]);

    (graph, concat, conv)
}

// ── GREEN tests: eligible patterns ─────────────────────────

#[test]
fn green_simple_concat_to_conv() {
    let (graph, concat_id, conv_id) = build_eligible_graph();
    let result = check_concat_to_conv_eligibility(&graph, concat_id);
    assert!(result.eligible, "expected GREEN, got {:?}", result.reason);
    assert_eq!(result.conv_node_id, Some(conv_id));
}

#[test]
fn green_three_inputs_concat() {
    let mut graph = ComputeGraph::new();

    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8));
    let in_c = graph.add_node(Opcode::Input, vec![], nchw(1, 4, 8, 8));

    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat = graph.add_node_with_attrs(
        Opcode::Concat,
        vec![in_a, in_b, in_c],
        nchw(1, 28, 8, 8),
        attrs,
    );

    let weight = graph.add_node(Opcode::Input, vec![], nchw(64, 28, 1, 1));
    let mut conv_attrs = HashMap::new();
    conv_attrs.insert("groups".to_string(), "1".to_string());
    conv_attrs.insert("dilation".to_string(), "1".to_string());
    let _conv = graph.add_node_with_attrs(
        Opcode::Conv2d,
        vec![concat, weight],
        nchw(1, 64, 8, 8),
        conv_attrs,
    );

    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(result.eligible, "expected GREEN for 3-input concat");
}

// ── RED tests: ineligible patterns ─────────────────────────

#[test]
fn red_not_concat_opcode() {
    let mut graph = ComputeGraph::new();
    let id = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let result = check_concat_to_conv_eligibility(&graph, id);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::NotConcat));
}

#[test]
fn red_single_input_concat() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat = graph.add_node_with_attrs(Opcode::Concat, vec![in_a], nchw(1, 8, 4, 4), attrs);
    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::TooFewInputs));
}

#[test]
fn red_concat_axis_0() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "0".to_string());
    let concat =
        graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(2, 8, 4, 4), attrs);
    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::WrongAxis { axis: 0 }));
}

#[test]
fn red_concat_axis_2_spatial() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "2".to_string());
    let concat = graph.add_node_with_attrs(
        Opcode::Concat,
        vec![in_a, in_b],
        TensorType::new(
            vec![
                DimExpr::Known(1),
                DimExpr::Known(8),
                DimExpr::Known(8),
                DimExpr::Known(4),
            ],
            IrDType::F32,
        ),
        attrs,
    );
    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::WrongAxis { axis: 2 }));
}

#[test]
fn red_concat_two_consumers() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8));
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat =
        graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(1, 24, 8, 8), attrs);

    let w1 = graph.add_node(Opcode::Input, vec![], nchw(32, 24, 1, 1));
    let _conv1 = graph.add_node(Opcode::Conv2d, vec![concat, w1], nchw(1, 32, 8, 8));

    let w2 = graph.add_node(Opcode::Input, vec![], nchw(16, 24, 1, 1));
    let _conv2 = graph.add_node(Opcode::Conv2d, vec![concat, w2], nchw(1, 16, 8, 8));

    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::MultipleConsumers));
}

#[test]
fn red_consumer_is_relu() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8));
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat =
        graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(1, 24, 8, 8), attrs);
    let _relu = graph.add_node(Opcode::Relu, vec![concat], nchw(1, 24, 8, 8));

    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::ConsumerNotConv2d));
}

#[test]
fn red_consumer_is_matmul() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 4, 4));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat =
        graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(1, 24, 4, 4), attrs);
    let weight = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(vec![DimExpr::Known(24), DimExpr::Known(16)], IrDType::F32),
    );
    let _mm = graph.add_node(
        Opcode::MatMul,
        vec![concat, weight],
        TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(16)], IrDType::F32),
    );

    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::ConsumerNotConv2d));
}

#[test]
fn red_grouped_conv() {
    let (mut graph, concat_id, conv_id) = build_eligible_graph();
    if let Some(node) = graph.get_node_mut(conv_id) {
        node.attrs.insert("groups".to_string(), "4".to_string());
    }
    let result = check_concat_to_conv_eligibility(&graph, concat_id);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::GroupedConv { groups: 4 }));
}

#[test]
fn red_depthwise_conv_groups_eq_channels() {
    let (mut graph, concat_id, conv_id) = build_eligible_graph();
    if let Some(node) = graph.get_node_mut(conv_id) {
        node.attrs.insert("groups".to_string(), "24".to_string());
    }
    let result = check_concat_to_conv_eligibility(&graph, concat_id);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::GroupedConv { groups: 24 }));
}

#[test]
fn red_dilated_conv() {
    let (mut graph, concat_id, conv_id) = build_eligible_graph();
    if let Some(node) = graph.get_node_mut(conv_id) {
        node.attrs.insert("dilation".to_string(), "2".to_string());
    }
    let result = check_concat_to_conv_eligibility(&graph, concat_id);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::DilatedConv { dilation: 2 }));
}

#[test]
fn red_spatial_mismatch() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4)); // different H,W
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat =
        graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(1, 24, 8, 8), attrs);

    // Need a downstream Conv2d so the check reaches shape validation.
    let weight = graph.add_node(Opcode::Input, vec![], nchw(32, 24, 1, 1));
    let mut conv_attrs = HashMap::new();
    conv_attrs.insert("groups".to_string(), "1".to_string());
    conv_attrs.insert("dilation".to_string(), "1".to_string());
    let _conv = graph.add_node_with_attrs(
        Opcode::Conv2d,
        vec![concat, weight],
        nchw(1, 32, 8, 8),
        conv_attrs,
    );

    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert!(matches!(
        result.reason,
        Some(Ineligible::ShapeMismatch { .. })
    ));
}

#[test]
fn red_batch_mismatch() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(Opcode::Input, vec![], nchw(2, 16, 8, 8));
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8)); // different N
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat =
        graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(3, 24, 8, 8), attrs);

    // Need a downstream Conv2d so the check reaches shape validation.
    let weight = graph.add_node(Opcode::Input, vec![], nchw(32, 24, 1, 1));
    let mut conv_attrs = HashMap::new();
    conv_attrs.insert("groups".to_string(), "1".to_string());
    conv_attrs.insert("dilation".to_string(), "1".to_string());
    let _conv = graph.add_node_with_attrs(
        Opcode::Conv2d,
        vec![concat, weight],
        nchw(1, 32, 8, 8),
        conv_attrs,
    );

    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert!(matches!(
        result.reason,
        Some(Ineligible::ShapeMismatch { .. })
    ));
}

#[test]
fn red_symbolic_channel_dim() {
    let mut graph = ComputeGraph::new();
    let in_a = graph.add_node(
        Opcode::Input,
        vec![],
        TensorType::new(
            vec![
                DimExpr::Known(1),
                DimExpr::Symbol("C".to_string()),
                DimExpr::Known(8),
                DimExpr::Known(8),
            ],
            IrDType::F32,
        ),
    );
    let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8));
    let mut attrs = HashMap::new();
    attrs.insert("axis".to_string(), "1".to_string());
    let concat = graph.add_node_with_attrs(
        Opcode::Concat,
        vec![in_a, in_b],
        TensorType::new(
            vec![
                DimExpr::Known(1),
                DimExpr::Symbol("C2".to_string()),
                DimExpr::Known(8),
                DimExpr::Known(8),
            ],
            IrDType::F32,
        ),
        attrs,
    );

    // Need a downstream Conv2d so the check reaches channel validation.
    let weight = graph.add_node(Opcode::Input, vec![], nchw(32, 24, 1, 1));
    let mut conv_attrs = HashMap::new();
    conv_attrs.insert("groups".to_string(), "1".to_string());
    conv_attrs.insert("dilation".to_string(), "1".to_string());
    let _conv = graph.add_node_with_attrs(
        Opcode::Conv2d,
        vec![concat, weight],
        nchw(1, 32, 8, 8),
        conv_attrs,
    );

    let result = check_concat_to_conv_eligibility(&graph, concat);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::UnknownChannelDim));
}

#[test]
fn red_nonexistent_node_id() {
    let graph = ComputeGraph::new();
    let result = check_concat_to_conv_eligibility(&graph, 999);
    assert!(!result.eligible);
    assert_eq!(result.reason, Some(Ineligible::NotConcat));
}

// ── Result structure tests ─────────────────────────────────

#[test]
fn result_carry_concat_and_conv_ids() {
    let (graph, concat_id, conv_id) = build_eligible_graph();
    let result = check_concat_to_conv_eligibility(&graph, concat_id);
    assert_eq!(result.concat_node_id, concat_id);
    assert_eq!(result.conv_node_id, Some(conv_id));
}

#[test]
fn result_eligible_has_no_reason() {
    let (graph, concat_id, _) = build_eligible_graph();
    let result = check_concat_to_conv_eligibility(&graph, concat_id);
    assert!(result.eligible);
    assert!(result.reason.is_none());
}

#[test]
fn result_ineligible_has_reason() {
    let mut graph = ComputeGraph::new();
    let id = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
    let result = check_concat_to_conv_eligibility(&graph, id);
    assert!(!result.eligible);
    assert!(result.reason.is_some());
}
