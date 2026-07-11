//! Compiler pass to insert F8x4R gradient quantization around optimizer inputs.
//!
//! After the backward graph is built (but before optimizer injection),
//! this pass inserts `QuantizeGradient` → `DequantizeGradient` pairs on
//! each gradient edge feeding an optimizer update node.

use std::collections::HashMap;

use crate::ir::{ComputeGraph, IrDType, NodeId, Opcode, TensorType};

/// Insert gradient quantization for all optimizer update nodes.
///
/// For each optimizer node, the gradient input (index 1) is wrapped with:
///   grad_f32 → QuantizeGradient → grad_f8x4r → DequantizeGradient → grad_f32'
pub fn quantize_gradients(graph: &mut ComputeGraph) {
    let node_ids = graph.topological_sort();
    let mut rewrites: Vec<(NodeId, NodeId, f32)> = Vec::with_capacity(graph.nodes.len());

    for &node_id in &node_ids {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };
        if !is_optimizer_op(&node.opcode) {
            continue;
        }
        // Gradient input is at index 1 (after param at index 0)
        let grad_id = match node.inputs.get(1) {
            Some(&id) => id,
            None => continue,
        };
        // Use default scale = 1.0 (calibration can adjust later)
        rewrites.push((node_id, grad_id, 1.0));
    }

    for (opt_id, grad_id, scale) in rewrites {
        let grad_type = graph
            .get_node(grad_id)
            .map(|n| n.output_type.clone())
            .unwrap_or(TensorType::new(vec![], IrDType::F32));

        let f8x4r_type = TensorType::new(
            grad_type.shape.clone(),
            IrDType::F8R {
                scales: vec![scale],
            },
        );

        let mut attrs = HashMap::new();
        attrs.insert("scale".to_string(), scale.to_string());
        let q_id =
            graph.add_node_with_attrs(Opcode::QuantizeGradient, vec![grad_id], f8x4r_type, attrs);

        let dq_type = TensorType::new(grad_type.shape.clone(), IrDType::F32);
        let dq_id = graph.add_node(Opcode::DequantizeGradient, vec![q_id], dq_type);

        if let Some(opt) = graph.get_node_mut(opt_id) {
            opt.inputs[1] = dq_id;
        }
    }
}

fn is_optimizer_op(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::SgdUpdate
            | Opcode::AdamUpdate
            | Opcode::AdamWUpdate
            | Opcode::MuonUpdate
            | Opcode::LionUpdate
            | Opcode::RmspropUpdate
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::DimExpr;

    fn make_sgd_graph() -> ComputeGraph {
        let mut graph = ComputeGraph::new();
        let weight_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(2)], IrDType::F32),
        );
        let grad_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(2)], IrDType::F32),
        );
        let mut attrs = std::collections::HashMap::new();
        attrs.insert("lr".to_string(), "0.01".to_string());
        attrs.insert("weight_decay".to_string(), "0.0".to_string());
        let sgd_id = graph.add_node_with_attrs(
            Opcode::SgdUpdate,
            vec![weight_id, grad_id],
            TensorType::new(vec![DimExpr::Known(4), DimExpr::Known(2)], IrDType::F32),
            attrs,
        );
        graph.set_inputs(vec![weight_id, grad_id]);
        graph.set_outputs(vec![sgd_id]);
        graph
    }

    fn make_graph_no_optimizer() -> ComputeGraph {
        let mut graph = ComputeGraph::new();
        let a_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(2)], IrDType::F32),
        );
        let b_id = graph.add_node(
            Opcode::Input,
            vec![],
            TensorType::new(vec![DimExpr::Known(2)], IrDType::F32),
        );
        let add_id = graph.add_node(
            Opcode::Add,
            vec![a_id, b_id],
            TensorType::new(vec![DimExpr::Known(2)], IrDType::F32),
        );
        graph.set_inputs(vec![a_id, b_id]);
        graph.set_outputs(vec![add_id]);
        graph
    }

    #[test]
    fn test_quantize_gradients_wraps_sgd_gradient() {
        let mut graph = make_sgd_graph();
        let count_before = graph.nodes.len();
        quantize_gradients(&mut graph);
        assert_eq!(graph.nodes.len(), count_before + 2, "Should add Q+DQ nodes");
        let sgd_node = graph.get_node(graph.outputs[0]).unwrap();
        let new_grad = sgd_node.inputs[1];
        let deq = graph.get_node(new_grad).unwrap();
        assert_eq!(deq.opcode, Opcode::DequantizeGradient);
        let q_id = deq.inputs[0];
        let q = graph.get_node(q_id).unwrap();
        assert_eq!(q.opcode, Opcode::QuantizeGradient);
        assert!(matches!(q.output_type.dtype, IrDType::F8R { .. }));
    }

    #[test]
    fn test_quantize_gradients_f8r_output_type_has_scale() {
        let mut graph = make_sgd_graph();
        quantize_gradients(&mut graph);
        let q = graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::QuantizeGradient))
            .unwrap();
        match &q.output_type.dtype {
            IrDType::F8R { scales } => {
                assert_eq!(scales.len(), 1, "F8R should have one scale");
                assert!(
                    (scales[0] - 1.0).abs() < 1e-6,
                    "Default scale should be 1.0"
                );
            }
            _ => panic!("Expected F8R dtype"),
        }
    }

    #[test]
    fn test_quantize_gradients_deq_output_is_f32() {
        let mut graph = make_sgd_graph();
        quantize_gradients(&mut graph);
        let dq = graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::DequantizeGradient))
            .unwrap();
        assert_eq!(dq.output_type.dtype, IrDType::F32);
    }

    #[test]
    fn test_quantize_gradients_q_has_scale_attr() {
        let mut graph = make_sgd_graph();
        quantize_gradients(&mut graph);
        let q = graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::QuantizeGradient))
            .unwrap();
        assert_eq!(q.attrs.get("scale").map(|s| s.as_str()), Some("1"));
    }

    #[test]
    fn test_quantize_gradients_no_optimizer_noop() {
        let mut graph = make_graph_no_optimizer();
        let count_before = graph.nodes.len();
        quantize_gradients(&mut graph);
        assert_eq!(
            graph.nodes.len(),
            count_before,
            "No Q/DQ should be added when no optimizer exists"
        );
    }

    #[test]
    fn test_quantize_gradients_handles_all_optimizer_types() {
        let optimizers: Vec<(Opcode, Vec<(&str, &str)>)> = vec![
            (
                Opcode::SgdUpdate,
                vec![("lr", "0.01"), ("weight_decay", "0.0")],
            ),
            (
                Opcode::AdamUpdate,
                vec![
                    ("lr", "0.001"),
                    ("beta1", "0.9"),
                    ("beta2", "0.999"),
                    ("eps", "1e-8"),
                    ("t", "1"),
                ],
            ),
            (
                Opcode::AdamWUpdate,
                vec![
                    ("lr", "0.001"),
                    ("beta1", "0.9"),
                    ("beta2", "0.999"),
                    ("eps", "1e-8"),
                    ("t", "1"),
                    ("weight_decay", "0.01"),
                ],
            ),
            (
                Opcode::MuonUpdate,
                vec![("lr", "0.01"), ("beta1", "0.95"), ("weight_decay", "0.0")],
            ),
            (
                Opcode::LionUpdate,
                vec![
                    ("lr", "0.001"),
                    ("beta1", "0.9"),
                    ("beta2", "0.99"),
                    ("weight_decay", "0.0"),
                ],
            ),
            (
                Opcode::RmspropUpdate,
                vec![("lr", "0.001"), ("beta", "0.9"), ("eps", "1e-8")],
            ),
        ];
        for (ref op, attrs_list) in &optimizers {
            let mut graph = ComputeGraph::new();
            let w = graph.add_node(
                Opcode::Input,
                vec![],
                TensorType::new(vec![DimExpr::Known(2)], IrDType::F32),
            );
            let g = graph.add_node(
                Opcode::Input,
                vec![],
                TensorType::new(vec![DimExpr::Known(2)], IrDType::F32),
            );
            let mut attrs = std::collections::HashMap::new();
            for (k, v) in attrs_list {
                attrs.insert(k.to_string(), v.to_string());
            }
            let opt_id = graph.add_node_with_attrs(
                op.clone(),
                vec![w, g],
                TensorType::new(vec![DimExpr::Known(2)], IrDType::F32),
                attrs,
            );
            graph.set_inputs(vec![w, g]);
            graph.set_outputs(vec![opt_id]);
            let count_before = graph.nodes.len();
            quantize_gradients(&mut graph);
            assert_eq!(
                graph.nodes.len(),
                count_before + 2,
                "Should add Q+DQ for {:?}",
                op
            );
            let opt = graph.get_node(opt_id).unwrap();
            let dq = graph.get_node(opt.inputs[1]).unwrap();
            assert_eq!(
                dq.opcode,
                Opcode::DequantizeGradient,
                "Input[1] of {:?} should be DQ",
                op
            );
        }
    }
}
