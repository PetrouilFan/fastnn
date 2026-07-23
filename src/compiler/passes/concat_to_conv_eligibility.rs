use crate::ir::{ComputeGraph, NodeId, Opcode};

/// Reasons why a concat→Conv2d pattern is NOT eligible for segmented planning.
///
/// Each variant represents a distinct rejection reason that a future planner
/// can use to decide whether to apply the optimization or skip it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ineligible {
    /// The concat node is not an `Opcode::Concat`.
    NotConcat,
    /// The concat has fewer than 2 inputs.
    TooFewInputs,
    /// The concat axis is not the channel axis (axis 1 for NCHW).
    WrongAxis { axis: i64 },
    /// A required numeric attribute is missing or malformed.
    MalformedAttribute { name: String },
    /// The concat output feeds multiple consumers — cannot safely fuse.
    MultipleConsumers,
    /// The single downstream consumer is not a Conv2d.
    ConsumerNotConv2d,
    /// The downstream Conv2d is grouped (groups != 1) — unsupported for now.
    GroupedConv { groups: usize },
    /// The downstream Conv2d has dilation != 1 — unsupported for now.
    DilatedConv { dilation: usize },
    /// Shape mismatch: concat inputs do not share batch/spatial dims.
    ShapeMismatch { detail: String },
    /// The concat output has unknown/symbolic channel dims that prevent
    /// safe segmentation.
    UnknownChannelDim,
}

/// Result of an eligibility check for a concat→Conv2d planning candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibilityResult {
    /// The concat node id being checked.
    pub concat_node_id: NodeId,
    /// The downstream Conv2d node id, if eligible.
    pub conv_node_id: Option<NodeId>,
    /// `true` when the pattern is safe for segmented channel-concat planning.
    pub eligible: bool,
    /// If ineligible, the specific rejection reason.
    pub reason: Option<Ineligible>,
}

/// Check whether a concat node is a safe candidate for future segmented
/// channel-concat-to-Conv2d planning.
///
/// This is a **guardrail** — it does NOT transform the graph. It only
/// reports eligibility. A future planner can use this to decide whether
/// to apply the optimization.
///
/// # Eligibility criteria (all must hold)
///
/// 1. The node must be an `Opcode::Concat`.
/// 2. The concat must have at least 2 inputs.
/// 3. The concat axis must be 1 (channel axis in NCHW layout).
/// 4. The concat output must feed exactly one downstream consumer.
/// 5. That consumer must be an `Opcode::Conv2d`.
/// 6. The downstream Conv2d must be ungrouped (groups == 1).
/// 7. The downstream Conv2d must have dilation == 1.
/// 8. All concat inputs must share the same batch and spatial dims,
///    and the channel dim must be statically known on all inputs.
pub fn check_concat_to_conv_eligibility(
    graph: &ComputeGraph,
    concat_node_id: NodeId,
) -> EligibilityResult {
    let concat = match graph.get_node(concat_node_id) {
        Some(n) => n,
        None => {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: None,
                eligible: false,
                reason: Some(Ineligible::NotConcat),
            };
        }
    };

    // 1. Must be a Concat opcode.
    if concat.opcode != Opcode::Concat {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: None,
            eligible: false,
            reason: Some(Ineligible::NotConcat),
        };
    }

    // 2. Must have at least 2 inputs.
    if concat.inputs.len() < 2 {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: None,
            eligible: false,
            reason: Some(Ineligible::TooFewInputs),
        };
    }

    // 3. Axis must be 1 (channel axis for NCHW).
    let axis = match concat.required_attr::<i64>("axis") {
        Ok(axis) => axis,
        Err(_) => {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: None,
                eligible: false,
                reason: Some(Ineligible::MalformedAttribute {
                    name: "axis".into(),
                }),
            };
        }
    };
    if axis != 1 {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: None,
            eligible: false,
            reason: Some(Ineligible::WrongAxis { axis }),
        };
    }

    // 4. Exactly one consumer.
    let consumers = graph.consumers(concat_node_id);
    if consumers.len() != 1 {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: None,
            eligible: false,
            reason: Some(Ineligible::MultipleConsumers),
        };
    }
    let consumer_id = consumers[0];

    // 5. Consumer must be Conv2d.
    let consumer = match graph.get_node(consumer_id) {
        Some(n) => n,
        None => {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: None,
                eligible: false,
                reason: Some(Ineligible::ConsumerNotConv2d),
            };
        }
    };
    if consumer.opcode != Opcode::Conv2d {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: None,
            eligible: false,
            reason: Some(Ineligible::ConsumerNotConv2d),
        };
    }

    // 6. Conv2d must be ungrouped.
    let groups = match consumer.required_attr::<usize>("groups") {
        Ok(groups) => groups,
        Err(_) => {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::MalformedAttribute {
                    name: "groups".into(),
                }),
            };
        }
    };
    if groups != 1 {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: Some(consumer_id),
            eligible: false,
            reason: Some(Ineligible::GroupedConv { groups }),
        };
    }

    // 7. Conv2d must have dilation == 1.
    let dilation = match consumer.required_attr::<usize>("dilation") {
        Ok(dilation) => dilation,
        Err(_) => {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::MalformedAttribute {
                    name: "dilation".into(),
                }),
            };
        }
    };
    if dilation != 1 {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: Some(consumer_id),
            eligible: false,
            reason: Some(Ineligible::DilatedConv { dilation }),
        };
    }

    // 8. Shape checks: all concat inputs must be 4D (NCHW), share
    //    batch and spatial dims, and have statically known channel dims.
    let rank = concat.output_type.shape.len();
    if rank != 4 {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: Some(consumer_id),
            eligible: false,
            reason: Some(Ineligible::ShapeMismatch {
                detail: format!("concat output rank {} != 4 (expected NCHW)", rank),
            }),
        };
    }

    // Reference shape from first input.
    let first = match graph.get_node(concat.inputs[0]) {
        Some(n) => n,
        None => {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::ShapeMismatch {
                    detail: "cannot resolve first concat input".to_string(),
                }),
            };
        }
    };
    if first.output_type.shape.len() != 4 {
        return EligibilityResult {
            concat_node_id,
            conv_node_id: Some(consumer_id),
            eligible: false,
            reason: Some(Ineligible::ShapeMismatch {
                detail: format!("first input rank {} != 4", first.output_type.shape.len()),
            }),
        };
    }

    // Batch (N), Height (H), Width (W) must match across all inputs.
    // Channel (C, axis=1) must be statically known on every input.
    for (i, &input_id) in concat.inputs.iter().enumerate() {
        let inp = match graph.get_node(input_id) {
            Some(n) => n,
            None => {
                return EligibilityResult {
                    concat_node_id,
                    conv_node_id: Some(consumer_id),
                    eligible: false,
                    reason: Some(Ineligible::ShapeMismatch {
                        detail: format!("cannot resolve concat input {}", i),
                    }),
                };
            }
        };
        let shape = &inp.output_type.shape;
        if shape.len() != 4 {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::ShapeMismatch {
                    detail: format!("input {} rank {} != 4", i, shape.len()),
                }),
            };
        }

        // Batch dim must match first input.
        if shape[0] != first.output_type.shape[0] {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::ShapeMismatch {
                    detail: format!(
                        "input {} batch dim {:?} != first input batch {:?}",
                        i, shape[0], first.output_type.shape[0]
                    ),
                }),
            };
        }

        // Spatial dims must match.
        if shape[2] != first.output_type.shape[2] {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::ShapeMismatch {
                    detail: format!(
                        "input {} height dim {:?} != first input height {:?}",
                        i, shape[2], first.output_type.shape[2]
                    ),
                }),
            };
        }
        if shape[3] != first.output_type.shape[3] {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::ShapeMismatch {
                    detail: format!(
                        "input {} width dim {:?} != first input width {:?}",
                        i, shape[3], first.output_type.shape[3]
                    ),
                }),
            };
        }

        // Channel dim must be statically known.
        if !shape[1].is_known() {
            return EligibilityResult {
                concat_node_id,
                conv_node_id: Some(consumer_id),
                eligible: false,
                reason: Some(Ineligible::UnknownChannelDim),
            };
        }
    }

    // All checks passed.
    EligibilityResult {
        concat_node_id,
        conv_node_id: Some(consumer_id),
        eligible: true,
        reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ComputeGraph, DimExpr, IrDType, Opcode, TensorType};
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

    fn build_simple_concat_to_conv() -> (ComputeGraph, NodeId, NodeId) {
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

    #[test]
    fn green_eligible_pattern() {
        let (graph, concat_id, conv_id) = build_simple_concat_to_conv();
        let result = check_concat_to_conv_eligibility(&graph, concat_id);
        assert!(
            result.eligible,
            "expected eligible, got {:?}",
            result.reason
        );
        assert_eq!(result.conv_node_id, Some(conv_id));
        assert!(result.reason.is_none());
    }

    #[test]
    fn malformed_attributes_are_never_eligible() {
        let (mut graph, concat_id, conv_id) = build_simple_concat_to_conv();
        graph
            .get_node_mut(concat_id)
            .unwrap()
            .attrs
            .insert("axis".into(), "invalid".into());
        let result = check_concat_to_conv_eligibility(&graph, concat_id);
        assert_eq!(
            result.reason,
            Some(Ineligible::MalformedAttribute {
                name: "axis".into()
            })
        );

        graph
            .get_node_mut(concat_id)
            .unwrap()
            .attrs
            .insert("axis".into(), "1".into());
        graph
            .get_node_mut(conv_id)
            .unwrap()
            .attrs
            .insert("groups".into(), "invalid".into());
        let result = check_concat_to_conv_eligibility(&graph, concat_id);
        assert_eq!(
            result.reason,
            Some(Ineligible::MalformedAttribute {
                name: "groups".into()
            })
        );
    }

    #[test]
    fn red_not_concat() {
        let mut graph = ComputeGraph::new();
        let id = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4));
        let result = check_concat_to_conv_eligibility(&graph, id);
        assert!(!result.eligible);
        assert_eq!(result.reason, Some(Ineligible::NotConcat));
    }

    #[test]
    fn red_too_few_inputs() {
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
    fn red_wrong_axis() {
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
    fn red_multiple_consumers() {
        let mut graph = ComputeGraph::new();
        let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
        let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8));
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "1".to_string());
        let concat =
            graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(1, 24, 8, 8), attrs);

        // Two consumers of the concat output.
        let weight1 = graph.add_node(Opcode::Input, vec![], nchw(32, 24, 1, 1));
        let _conv1 = graph.add_node(Opcode::Conv2d, vec![concat, weight1], nchw(1, 32, 8, 8));

        let weight2 = graph.add_node(Opcode::Input, vec![], nchw(16, 24, 1, 1));
        let _conv2 = graph.add_node(Opcode::Conv2d, vec![concat, weight2], nchw(1, 16, 8, 8));

        let result = check_concat_to_conv_eligibility(&graph, concat);
        assert!(!result.eligible);
        assert_eq!(result.reason, Some(Ineligible::MultipleConsumers));
    }

    #[test]
    fn red_consumer_not_conv2d() {
        let mut graph = ComputeGraph::new();
        let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
        let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 8, 8));
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "1".to_string());
        let concat =
            graph.add_node_with_attrs(Opcode::Concat, vec![in_a, in_b], nchw(1, 24, 8, 8), attrs);

        // Consumer is Relu, not Conv2d.
        let _relu = graph.add_node(Opcode::Relu, vec![concat], nchw(1, 24, 8, 8));

        let result = check_concat_to_conv_eligibility(&graph, concat);
        assert!(!result.eligible);
        assert_eq!(result.reason, Some(Ineligible::ConsumerNotConv2d));
    }

    #[test]
    fn red_grouped_conv() {
        let (mut graph, concat_id, conv_id) = build_simple_concat_to_conv();
        // Rewrite the conv to be grouped.
        if let Some(conv_node) = graph.get_node_mut(conv_id) {
            conv_node
                .attrs
                .insert("groups".to_string(), "4".to_string());
        }
        let result = check_concat_to_conv_eligibility(&graph, concat_id);
        assert!(!result.eligible);
        assert_eq!(result.reason, Some(Ineligible::GroupedConv { groups: 4 }));
    }

    #[test]
    fn red_dilated_conv() {
        let (mut graph, concat_id, conv_id) = build_simple_concat_to_conv();
        if let Some(conv_node) = graph.get_node_mut(conv_id) {
            conv_node
                .attrs
                .insert("dilation".to_string(), "2".to_string());
        }
        let result = check_concat_to_conv_eligibility(&graph, concat_id);
        assert!(!result.eligible);
        assert_eq!(result.reason, Some(Ineligible::DilatedConv { dilation: 2 }));
    }

    #[test]
    fn red_shape_mismatch_spatial() {
        let mut graph = ComputeGraph::new();
        let in_a = graph.add_node(Opcode::Input, vec![], nchw(1, 16, 8, 8));
        let in_b = graph.add_node(Opcode::Input, vec![], nchw(1, 8, 4, 4)); // different spatial
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), "1".to_string());
        let concat = graph.add_node_with_attrs(
            Opcode::Concat,
            vec![in_a, in_b],
            TensorType::new(
                vec![
                    DimExpr::Known(1),
                    DimExpr::Known(24),
                    DimExpr::Known(8),
                    DimExpr::Known(8),
                ],
                IrDType::F32,
            ),
            attrs,
        );

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
    fn red_unknown_channel_dim() {
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
    fn red_nonexistent_node() {
        let graph = ComputeGraph::new();
        let result = check_concat_to_conv_eligibility(&graph, 999);
        assert!(!result.eligible);
        assert_eq!(result.reason, Some(Ineligible::NotConcat));
    }
}
