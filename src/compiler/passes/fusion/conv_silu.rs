use super::FusionPass;
use crate::ir::node::{ComputeGraph, NodeId, Opcode};
use std::collections::HashSet;

/// Fuses the ONNX SiLU decomposition pattern into a single Conv2d+SiLU op.
///
/// The pattern is:
/// ```text
///   conv = Conv2d(x, w[, b])
///   sig  = Sigmoid(conv)
///   out  = Mul(conv, sig)
/// ```
///
/// YOLOv8n (and many other exported models) emit SiLU this way because
/// ONNX does not have a first-class SiLU op in older opsets. The conversion
/// to IR keeps Sigmoid and Mul as separate ops, which means each Conv2d in
/// the backbone produces an extra `sigmoid_f32` (~0.25 ms) and `mul_f32`
/// (~0.02 ms) per call. For YOLOv8n @ 320x320 there are 58 Sigmoid+Mul
/// pairs (one per hidden Conv), totalling ~15 ms in standalone form.
///
/// This pass detects the pattern, tags the upstream Conv2d with
/// `fused_op=OpSilu` (the same mechanism `OpRelu` already uses), and
/// removes the Sigmoid and Mul nodes. The CPU backend's `conv2d_silu`
/// kernel then performs the bias add and SiLU activation in the same
/// pass over the output buffer, eliminating both the sigmoid and mul
/// kernel calls and avoiding a full memory round-trip per activation.
pub struct ConvSilu;

impl FusionPass for ConvSilu {
    fn name() -> &'static str {
        "ConvSilu"
    }

    fn fuse(graph: &mut ComputeGraph) -> Result<bool, String> {
        let mut fused = false;
        let mut to_remove: HashSet<NodeId> = HashSet::new();
        let conv_ids: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter(|n| n.opcode == Opcode::Conv2d)
            .map(|n| n.id)
            .collect();

        for conv_id in &conv_ids {
            if to_remove.contains(conv_id) {
                continue;
            }
            let conv_node = match graph.get_node(*conv_id) {
                Some(n) => n.clone(),
                None => continue,
            };
            if conv_node.attrs.contains_key("fused_op") {
                continue;
            }

            // The Conv typically has 1-2 direct consumers in the YOLO SiLU
            // decomposition. YOLOv8 ONNX export inserts an identity Reshape
            // between the Conv and the Sigmoid (Reshape(conv) -> Sigmoid -> Mul
            // (reshape, sigmoid)), so we also accept a Reshape wrapper. The
            // Reshape is a no-op for activation purposes (same numel, same
            // dtype), so fusing the activation into the Conv and removing the
            // Reshape + Sigmoid + Mul is safe as long as nothing else reads
            // them.
            let conv_consumers = graph.consumers(*conv_id);
            if conv_consumers.is_empty() {
                continue;
            }

            // Walk one Reshape step ahead if present (YOLO pattern). The
            // Reshape may have two consumers: the Sigmoid and the Mul (Mul
            // also takes the Reshape as one of its inputs).
            let reshape_id_opt = conv_consumers
                .iter()
                .find(|&&c| {
                    !to_remove.contains(&c)
                        && graph
                            .get_node(c)
                            .map(|n| {
                                n.opcode == Opcode::Reshape
                                    && n.inputs.len() == 1
                                    && n.inputs[0] == *conv_id
                            })
                            .unwrap_or(false)
                })
                .copied();
            let conv_value_id = reshape_id_opt.unwrap_or(*conv_id);

            // Find a Sigmoid node whose only input is `conv_value_id`
            // (the Conv, or a Reshape of the Conv) and whose only consumer
            // is the Mul. If the Sigmoid is shared with another consumer,
            // fusing would silently delete a value the rest of the graph
            // depends on, so we must skip.
            let sig_id = match graph.consumers(conv_value_id).iter().find(|&&c| {
                !to_remove.contains(&c)
                    && graph
                        .get_node(c)
                        .map(|n| {
                            n.opcode == Opcode::Sigmoid
                                && n.inputs.len() == 1
                                && n.inputs[0] == conv_value_id
                                && graph.consumers(c).len() == 1
                        })
                        .unwrap_or(false)
            }) {
                Some(c) => *c,
                None => continue,
            };

            // Find a Mul that consumes both `conv_value_id` and the Sigmoid.
            // It may appear as a direct consumer of `conv_value_id` (or the
            // Conv) or the Sigmoid.
            let mut candidates: Vec<NodeId> = graph
                .consumers(conv_value_id)
                .into_iter()
                .chain(graph.consumers(sig_id))
                .collect();
            candidates.sort_unstable();
            candidates.dedup();
            let mul_id = match candidates.iter().find(|&&c| {
                !to_remove.contains(&c)
                    && graph
                        .get_node(c)
                        .map(|n| {
                            n.opcode == Opcode::Mul
                                && n.inputs.len() == 2
                                && n.inputs.iter().any(|i| i == &conv_value_id)
                                && n.inputs.iter().any(|i| i == &sig_id)
                        })
                        .unwrap_or(false)
            }) {
                Some(c) => *c,
                None => continue,
            };
            let mul_node = match graph.get_node(mul_id) {
                Some(n) => n.clone(),
                None => continue,
            };

            // Tag the Conv with the SiLU fused_op so the backend picks
            // `conv2d_silu` and the bias-add + SiLU happen in one pass.
            if let Some(conv_mut) = graph.get_node_mut(*conv_id) {
                conv_mut
                    .attrs
                    .insert("fused_op".to_string(), "OpSilu".to_string());
                conv_mut.output_type = mul_node.output_type.clone();
            }

            // Rewire the Mul's consumers to the Conv directly. The Conv's
            // output type is now the Mul's output type, so any downstream
            // shape expectations still hold.
            let mul_consumers: Vec<NodeId> = graph.consumers(mul_id);
            for consumer_id in mul_consumers {
                if let Some(consumer) = graph.get_node_mut(consumer_id) {
                    for input in consumer.inputs.iter_mut() {
                        if *input == mul_id {
                            *input = *conv_id;
                        }
                    }
                }
            }
            if let Some(output) = graph.outputs.iter_mut().find(|o| **o == mul_id) {
                *output = *conv_id;
            }

            if let Some(rid) = reshape_id_opt {
                to_remove.insert(rid);
            }
            to_remove.insert(sig_id);
            to_remove.insert(mul_id);
            fused = true;
        }

        for id in &to_remove {
            graph.remove_node(*id);
        }
        Ok(fused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::IrDType;
    use crate::ir::node::{TensorType, TensorValue};
    use std::collections::HashMap;

    fn build_conv_silu_graph() -> (ComputeGraph, NodeId, NodeId, NodeId, NodeId, NodeId, NodeId) {
        // Layout:
        //   x (Input) -> conv (Conv2d) -> sig (Sigmoid) -> mul (Mul) -> consumer (Reshape)
        //                    ^----------------------------------/
        //                                                  (also takes conv as 2nd input)
        let mut graph = ComputeGraph::new();
        let input_type = TensorType::new(
            vec![
                crate::ir::node::DimExpr::Known(1),
                crate::ir::node::DimExpr::Known(3),
                crate::ir::node::DimExpr::Known(8),
                crate::ir::node::DimExpr::Known(8),
            ],
            IrDType::F32,
        );
        let x = graph.add_node(Opcode::Input, vec![], input_type.clone());

        let weight_type = TensorType::new(
            vec![
                crate::ir::node::DimExpr::Known(4),
                crate::ir::node::DimExpr::Known(3),
                crate::ir::node::DimExpr::Known(3),
                crate::ir::node::DimExpr::Known(3),
            ],
            IrDType::F32,
        );
        let w = graph.add_node(
            Opcode::Constant(TensorValue::Data {
                tensor_type: weight_type.clone(),
                bytes: vec![0u8; 4 * 3 * 3 * 3 * 4],
            }),
            vec![],
            weight_type.clone(),
        );

        let mut conv_attrs = HashMap::new();
        conv_attrs.insert("stride".to_string(), "1".to_string());
        conv_attrs.insert("padding".to_string(), "1".to_string());
        conv_attrs.insert("dilation".to_string(), "1".to_string());
        conv_attrs.insert("groups".to_string(), "1".to_string());
        let conv_out = TensorType::new(
            vec![
                crate::ir::node::DimExpr::Known(1),
                crate::ir::node::DimExpr::Known(4),
                crate::ir::node::DimExpr::Known(8),
                crate::ir::node::DimExpr::Known(8),
            ],
            IrDType::F32,
        );
        let conv =
            graph.add_node_with_attrs(Opcode::Conv2d, vec![x, w], conv_out.clone(), conv_attrs);

        let sig = graph.add_node(Opcode::Sigmoid, vec![conv], conv_out.clone());

        let mul = graph.add_node(Opcode::Mul, vec![conv, sig], conv_out.clone());

        // Reshape consumer: receives the Mul output. After fusion, the Reshape
        // should be rewired to read the Conv directly.
        let reshape_out = TensorType::new(
            vec![
                crate::ir::node::DimExpr::Known(1),
                crate::ir::node::DimExpr::Known(256),
            ],
            IrDType::F32,
        );
        let consumer = graph.add_node(Opcode::Reshape, vec![mul], reshape_out);

        (graph, x, w, conv, sig, mul, consumer)
    }

    #[test]
    fn conv_silu_fusion_tags_conv_and_removes_sig_mul() {
        let (mut graph, _x, _w, conv, sig, mul, consumer) = build_conv_silu_graph();
        let res = ConvSilu::fuse(&mut graph).unwrap();
        assert!(res);

        let conv_node = graph.get_node(conv).expect("conv should still exist");
        assert_eq!(
            conv_node.attrs.get("fused_op").map(|s| s.as_str()),
            Some("OpSilu")
        );

        assert!(graph.get_node(sig).is_none(), "sigmoid should be removed");
        assert!(graph.get_node(mul).is_none(), "mul should be removed");

        // The downstream Reshape should now consume the Conv directly.
        let consumer_node = graph.get_node(consumer).unwrap();
        assert_eq!(consumer_node.inputs, vec![conv]);
    }

    #[test]
    fn conv_silu_fusion_handles_swapped_mul_operand_order() {
        // Same pattern but Mul( sigmoid, conv ) instead of Mul( conv, sigmoid ).
        let mut graph = ComputeGraph::new();
        let input_type = TensorType::new(vec![crate::ir::node::DimExpr::Known(1); 4], IrDType::F32);
        let x = graph.add_node(Opcode::Input, vec![], input_type.clone());
        let w = graph.add_node(
            Opcode::Constant(TensorValue::Data {
                tensor_type: input_type.clone(),
                bytes: vec![0u8; 16],
            }),
            vec![],
            input_type.clone(),
        );
        let conv = graph.add_node_with_attrs(
            Opcode::Conv2d,
            vec![x, w],
            input_type.clone(),
            HashMap::from([
                ("stride".to_string(), "1".to_string()),
                ("padding".to_string(), "0".to_string()),
                ("dilation".to_string(), "1".to_string()),
                ("groups".to_string(), "1".to_string()),
            ]),
        );
        let sig = graph.add_node(Opcode::Sigmoid, vec![conv], input_type.clone());
        let mul = graph.add_node(Opcode::Mul, vec![sig, conv], input_type.clone());
        let consumer = graph.add_node(Opcode::Reshape, vec![mul], input_type.clone());

        assert!(ConvSilu::fuse(&mut graph).unwrap());
        assert!(graph.get_node(sig).is_none());
        assert!(graph.get_node(mul).is_none());
        let conv_node = graph.get_node(conv).unwrap();
        assert_eq!(
            conv_node.attrs.get("fused_op").map(|s| s.as_str()),
            Some("OpSilu")
        );
        // The consumer is rewired to the Conv.
        let consumer_node = graph.get_node(consumer).unwrap();
        assert_eq!(consumer_node.inputs, vec![conv]);
    }

    #[test]
    fn conv_silu_fusion_does_not_touch_reshape_only_conv() {
        // Conv -> Reshape (no SiLU). Should not be touched.
        let mut graph = ComputeGraph::new();
        let ty = TensorType::new(vec![crate::ir::node::DimExpr::Known(1); 4], IrDType::F32);
        let x = graph.add_node(Opcode::Input, vec![], ty.clone());
        let w = graph.add_node(
            Opcode::Constant(TensorValue::Data {
                tensor_type: ty.clone(),
                bytes: vec![0u8; 16],
            }),
            vec![],
            ty.clone(),
        );
        let conv = graph.add_node_with_attrs(
            Opcode::Conv2d,
            vec![x, w],
            ty.clone(),
            HashMap::from([
                ("stride".to_string(), "1".to_string()),
                ("padding".to_string(), "0".to_string()),
                ("dilation".to_string(), "1".to_string()),
                ("groups".to_string(), "1".to_string()),
            ]),
        );
        let _reshape = graph.add_node(Opcode::Reshape, vec![conv], ty.clone());

        assert!(!ConvSilu::fuse(&mut graph).unwrap());
        let conv_node = graph.get_node(conv).unwrap();
        assert!(conv_node.attrs.get("fused_op").is_none());
    }

    #[test]
    fn conv_silu_fusion_skips_when_sigmoid_is_shared() {
        // Conv -> Sigmoid -> Mul(conv, sig)  AND  Sigmoid -> another consumer
        // (so Sigmoid has >1 consumer, the pattern is not safe to fuse).
        let mut graph = ComputeGraph::new();
        let ty = TensorType::new(vec![crate::ir::node::DimExpr::Known(1); 4], IrDType::F32);
        let x = graph.add_node(Opcode::Input, vec![], ty.clone());
        let w = graph.add_node(
            Opcode::Constant(TensorValue::Data {
                tensor_type: ty.clone(),
                bytes: vec![0u8; 16],
            }),
            vec![],
            ty.clone(),
        );
        let conv = graph.add_node_with_attrs(
            Opcode::Conv2d,
            vec![x, w],
            ty.clone(),
            HashMap::from([
                ("stride".to_string(), "1".to_string()),
                ("padding".to_string(), "0".to_string()),
                ("dilation".to_string(), "1".to_string()),
                ("groups".to_string(), "1".to_string()),
            ]),
        );
        let sig = graph.add_node(Opcode::Sigmoid, vec![conv], ty.clone());
        let mul = graph.add_node(Opcode::Mul, vec![conv, sig], ty.clone());
        let _other = graph.add_node(Opcode::Neg, vec![sig], ty.clone());

        assert!(!ConvSilu::fuse(&mut graph).unwrap());
        assert!(
            graph.get_node(sig).is_some(),
            "shared sigmoid should remain"
        );
        assert!(
            graph.get_node(mul).is_some(),
            "mul should remain when pattern is unsafe"
        );
        let conv_node = graph.get_node(conv).unwrap();
        assert!(conv_node.attrs.get("fused_op").is_none());
    }

    #[test]
    fn conv_silu_fusion_handles_reshape_wrapper() {
        // YOLOv8 ONNX export inserts an identity Reshape between the Conv
        // and the Sigmoid: Conv -> Reshape -> Sigmoid -> Mul(reshape, sigmoid).
        // The Reshape is shared by Sigmoid and Mul. After fusion, Conv is
        // tagged, Reshape + Sigmoid + Mul are removed, and the downstream
        // Reshape (next conv's input) should be rewired to the Conv.
        let mut graph = ComputeGraph::new();
        let ty = TensorType::new(vec![crate::ir::node::DimExpr::Known(1); 4], IrDType::F32);
        let x = graph.add_node(Opcode::Input, vec![], ty.clone());
        let w = graph.add_node(
            Opcode::Constant(TensorValue::Data {
                tensor_type: ty.clone(),
                bytes: vec![0u8; 16],
            }),
            vec![],
            ty.clone(),
        );
        let conv = graph.add_node_with_attrs(
            Opcode::Conv2d,
            vec![x, w],
            ty.clone(),
            HashMap::from([
                ("stride".to_string(), "1".to_string()),
                ("padding".to_string(), "0".to_string()),
                ("dilation".to_string(), "1".to_string()),
                ("groups".to_string(), "1".to_string()),
            ]),
        );
        // Identity Reshape (same numel).
        let reshape = graph.add_node(Opcode::Reshape, vec![conv], ty.clone());
        let sig = graph.add_node(Opcode::Sigmoid, vec![reshape], ty.clone());
        let mul = graph.add_node(Opcode::Mul, vec![reshape, sig], ty.clone());
        // Downstream Reshape (consumes Mul).
        let consumer_reshape = graph.add_node(Opcode::Reshape, vec![mul], ty.clone());

        assert!(ConvSilu::fuse(&mut graph).unwrap());

        // Conv is tagged with OpSilu.
        let conv_node = graph.get_node(conv).expect("conv should remain");
        assert_eq!(
            conv_node.attrs.get("fused_op").map(|s| s.as_str()),
            Some("OpSilu")
        );

        // Reshape, Sigmoid, and Mul are removed.
        assert!(
            graph.get_node(reshape).is_none(),
            "intermediate reshape should be removed"
        );
        assert!(graph.get_node(sig).is_none(), "sigmoid should be removed");
        assert!(graph.get_node(mul).is_none(), "mul should be removed");

        // Downstream Reshape is rewired to the Conv.
        let consumer = graph.get_node(consumer_reshape).unwrap();
        assert_eq!(consumer.inputs, vec![conv]);
    }
}
