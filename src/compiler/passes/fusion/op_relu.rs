use super::FusionPass;
use crate::ir::{ComputeGraph, NodeId, Opcode};
use crate::FastnnError;
use std::collections::HashSet;

fn activation_fused_name(opcode: Opcode) -> Option<&'static str> {
    match opcode {
        Opcode::Relu => Some("OpRelu"),
        Opcode::Gelu => Some("OpGelu"),
        Opcode::Silu => Some("OpSilu"),
        _ => None,
    }
}

/// Whether the producer's kernel actually applies a fused activation.
///
/// The fusion is only a marker in `attrs["fused_op"]`: the producer keeps its
/// own opcode and the backend has to translate the marker into a fused kernel
/// name. Only the convolution, matmul and the arithmetic binary kernels do
/// that. Fusing into anything else (concat, softmax, transpose, slice, pad,
/// resize, a broadcast, ...) *deletes* the activation and silently changes the
/// result, so the marker must not be written for those at all.
///
/// `Maximum`/`Minimum` are excluded on purpose: their dispatch arm rejects a
/// `fused_op` marker, so tagging them would be an error rather than a fast path.
fn supports_fused_activation(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::Conv2d | Opcode::MatMul | Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div
    )
}

pub struct OpRelu;

impl FusionPass for OpRelu {
    fn name() -> &'static str {
        "OpRelu"
    }

    fn fuse(graph: &mut ComputeGraph) -> Result<bool, FastnnError> {
        let mut fused = false;
        let mut to_remove: HashSet<NodeId> = HashSet::new();
        let node_ids: Vec<NodeId> = graph.nodes.iter().map(|n| n.id).collect();

        for &op_id in &node_ids {
            if to_remove.contains(&op_id) {
                continue;
            }

            let op_node = match graph.get_node(op_id) {
                Some(n) => n.clone(),
                None => continue,
            };
            match op_node.opcode {
                Opcode::Relu
                | Opcode::Gelu
                | Opcode::Silu
                | Opcode::Input
                | Opcode::Constant(_) => continue,
                _ => {}
            }
            if op_node.attrs.contains_key("fused_op") {
                continue;
            }
            if !supports_fused_activation(&op_node.opcode) {
                continue;
            }

            let op_consumers: Vec<NodeId> = graph.consumers(op_id);
            if op_consumers.len() != 1 {
                continue;
            }

            let act_id = op_consumers[0];
            if to_remove.contains(&act_id) {
                continue;
            }

            let act = match graph.get_node(act_id) {
                Some(n) => n.clone(),
                _ => continue,
            };
            let fused_name = match activation_fused_name(act.opcode) {
                Some(name) => name,
                None => continue,
            };

            if let Some(op_mut) = graph.get_node_mut(op_id) {
                op_mut
                    .attrs
                    .insert("fused_op".to_string(), fused_name.to_string());
                op_mut.output_type = act.output_type.clone();
            }

            to_remove.insert(act_id);

            let act_consumers: Vec<NodeId> = graph.consumers(act_id);
            for consumer_id in act_consumers {
                if let Some(consumer) = graph.get_node_mut(consumer_id) {
                    for input in consumer.inputs.iter_mut() {
                        if *input == act_id {
                            *input = op_id;
                        }
                    }
                }
            }

            if let Some(output) = graph.outputs.iter_mut().find(|o| **o == act_id) {
                *output = op_id;
            }

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
    use crate::ir::{DimExpr, IrDType, TensorType};

    fn shape() -> TensorType {
        TensorType::new(
            vec![
                DimExpr::Known(1),
                DimExpr::Known(4),
                DimExpr::Known(8),
                DimExpr::Known(8),
            ],
            IrDType::F32,
        )
    }

    /// `x -> producer -> Relu -> consumer`, returning the producer and consumer
    /// ids so a test can check whether the Relu survived.
    fn build_chain(producer: Opcode) -> (ComputeGraph, NodeId, NodeId, NodeId) {
        let mut graph = ComputeGraph::new();
        let ty = shape();
        let x = graph.add_node(Opcode::Input, vec![], ty.clone());
        let y = graph.add_node(Opcode::Input, vec![], ty.clone());
        let op = graph.add_node(producer, vec![x, y], ty.clone());
        let relu = graph.add_node(Opcode::Relu, vec![op], ty.clone());
        let consumer = graph.add_node(Opcode::Reshape, vec![relu], ty);
        (graph, op, relu, consumer)
    }

    #[test]
    fn fuses_into_the_operators_whose_kernels_apply_an_activation() {
        for producer in [
            Opcode::Add,
            Opcode::Sub,
            Opcode::Mul,
            Opcode::Div,
            Opcode::MatMul,
        ] {
            let (mut graph, op, relu, consumer) = build_chain(producer.clone());
            assert!(
                OpRelu::fuse(&mut graph).unwrap(),
                "{producer:?} should fuse"
            );
            let node = graph.get_node(op).expect("producer survives");
            assert_eq!(
                node.attrs.get("fused_op").map(String::as_str),
                Some("OpRelu")
            );
            assert!(graph.get_node(relu).is_none(), "{producer:?}: relu removed");
            assert_eq!(graph.get_node(consumer).unwrap().inputs, vec![op]);
        }
    }

    #[test]
    fn keeps_the_activation_when_the_producer_cannot_apply_it() {
        // Concat, Softmax, Transpose and friends ignore `fused_op`, so tagging
        // them would silently drop the activation: the node has to stay.
        for producer in [
            Opcode::Concat,
            Opcode::Softmax,
            Opcode::Transpose,
            Opcode::Maximum,
        ] {
            let (mut graph, op, relu, consumer) = build_chain(producer.clone());
            assert!(
                !OpRelu::fuse(&mut graph).unwrap(),
                "{producer:?} must not fuse"
            );
            let node = graph.get_node(op).expect("producer survives");
            assert!(!node.attrs.contains_key("fused_op"), "{producer:?}");
            assert!(graph.get_node(relu).is_some(), "{producer:?}: relu kept");
            assert_eq!(graph.get_node(consumer).unwrap().inputs, vec![relu]);
        }
    }
}
