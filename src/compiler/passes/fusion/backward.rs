use crate::ir::node::{ComputeGraph, Opcode, NodeId, TensorType};
use std::collections::HashMap;
use super::FusionPass;

/// Collect backward pattern info for a specific fused_op value.
struct BwdPattern {
    drelu_id: NodeId,
    grad_id: NodeId,
    fwd_id: NodeId,
    a_id: NodeId,
    b_id: NodeId,
    da_id: NodeId,
    db_id: NodeId,
    da_ty: TensorType,
    db_ty: TensorType,
    a_t_id: Option<NodeId>,
    b_t_id: Option<NodeId>,
    has_bias: bool,
}

fn find_bwd_patterns(graph: &ComputeGraph, target_fused_op: &str) -> Vec<BwdPattern> {
    let mut patterns = Vec::new();

    let drelu_ids: Vec<NodeId> = graph.nodes.iter()
        .filter(|n| {
            n.opcode == Opcode::Mul
                && n.attrs.get("op").map(|s| s.as_str()) == Some("relu_backward")
        })
        .map(|n| n.id)
        .collect();

    for &drelu_id in &drelu_ids {
        let drelu = match graph.get_node(drelu_id) {
            Some(n) => n,
            None => continue,
        };
        if drelu.inputs.len() < 2 { continue; }
        let grad_id = drelu.inputs[0];
        let fwd_id = drelu.inputs[1];

        let fwd = match graph.get_node(fwd_id) {
            Some(n) if n.opcode == Opcode::MatMul => n,
            _ => continue,
        };
        if fwd.attrs.get("fused_op").map(|s| s.as_str()) != Some(target_fused_op) {
            continue;
        }
        if fwd.inputs.len() < 2 { continue; }
        let a_id = fwd.inputs[0];
        let b_id = fwd.inputs[1];
        let has_bias = fwd.inputs.len() > 2;

        let consumers: Vec<NodeId> = graph.consumers(drelu_id);
        let mut da_id = None;
        let mut db_id = None;
        let mut a_t_id = None;
        let mut b_t_id = None;

        for &cid in &consumers {
            let cn = match graph.get_node(cid) {
                Some(n) => n,
                None => continue,
            };
            if cn.opcode != Opcode::MatMul || cn.inputs.len() < 2 { continue; }
            let other = if cn.inputs[0] == drelu_id { Some(cn.inputs[1]) }
                        else if cn.inputs[1] == drelu_id { Some(cn.inputs[0]) }
                        else { None };

            if let Some(other_id) = other {
                let other_node = match graph.get_node(other_id) {
                    Some(n) => n,
                    None => continue,
                };
                if other_node.opcode != Opcode::Transpose || other_node.inputs.is_empty() {
                    continue;
                }
                let orig = other_node.inputs[0];
                if orig == b_id { da_id = Some(cid); b_t_id = Some(other_id); }
                else if orig == a_id { db_id = Some(cid); a_t_id = Some(other_id); }
            }
        }

        let da_id = match da_id { Some(id) => id, None => continue };
        let db_id = match db_id { Some(id) => id, None => continue };

        let da_ty = graph.get_node(da_id).map(|n| n.output_type.clone()).unwrap();
        let db_ty = graph.get_node(db_id).map(|n| n.output_type.clone()).unwrap();

        patterns.push(BwdPattern {
            drelu_id,
            grad_id,
            fwd_id,
            a_id,
            b_id,
            da_id,
            db_id,
            da_ty,
            db_ty,
            a_t_id,
            b_t_id,
            has_bias,
        });
    }

    patterns
}

pub struct BackwardReluMatMul;

impl FusionPass for BackwardReluMatMul {
    fn name() -> &'static str { "BackwardReluMatMul" }

    fn fuse(graph: &mut ComputeGraph) -> Result<bool, String> {
        let patterns = find_bwd_patterns(graph, "OpRelu");
        if patterns.is_empty() { return Ok(false); }

        let mut fused = false;

        for p in &patterns {
            let da_fused_attrs = HashMap::from([
                ("fused_backward_op".to_string(), "OpRelu".to_string()),
                ("backward_side".to_string(), "da".to_string()),
            ]);
            let da_fused = graph.add_node_with_attrs(
                Opcode::MatMul,
                vec![p.grad_id, p.fwd_id, p.b_id],
                p.da_ty.clone(),
                da_fused_attrs,
            );

            let db_fused_attrs = HashMap::from([
                ("fused_backward_op".to_string(), "OpRelu".to_string()),
                ("backward_side".to_string(), "db".to_string()),
            ]);
            let db_fused = graph.add_node_with_attrs(
                Opcode::MatMul,
                vec![p.a_id, p.grad_id, p.fwd_id],
                p.db_ty.clone(),
                db_fused_attrs,
            );

            // Rewire all consumers of da → da_fused and db → db_fused
            for (old, new) in [(p.da_id, da_fused), (p.db_id, db_fused)] {
                let consumers: Vec<NodeId> = graph.consumers(old);
                for &c in &consumers {
                    if let Some(n) = graph.get_node_mut(c) {
                        for inp in n.inputs.iter_mut() {
                            if *inp == old { *inp = new; }
                        }
                    }
                }
                for out in graph.outputs.iter_mut() {
                    if *out == old { *out = new; }
                }
            }

            // Remove old nodes
            let to_remove: Vec<NodeId> = std::iter::once(p.da_id)
                .chain(std::iter::once(p.db_id))
                .chain(std::iter::once(p.drelu_id))
                .chain(p.a_t_id.into_iter())
                .chain(p.b_t_id.into_iter())
                .collect();
            for &id in &to_remove { graph.remove_node(id); }

            fused = true;
        }

        Ok(fused)
    }
}

pub struct BackwardMatMulAddRelu;

impl FusionPass for BackwardMatMulAddRelu {
    fn name() -> &'static str { "BackwardMatMulAddRelu" }

    fn fuse(graph: &mut ComputeGraph) -> Result<bool, String> {
        let mut fused = false;

        let fwd_ids: Vec<NodeId> = graph.nodes.iter()
            .filter(|n| {
                n.opcode == Opcode::MatMul
                    && n.attrs.get("fused_op").map(|s| s.as_str()) == Some("MatMulAddRelu")
            })
            .map(|n| n.id)
            .collect();

        for &fwd_id in &fwd_ids {
            let consumers: Vec<NodeId> = graph.consumers(fwd_id);
            let drelu_id = match consumers.iter().find(|&&cid| {
                graph.get_node(cid).map_or(false, |n| {
                    n.opcode == Opcode::Mul
                        && n.attrs.get("op").map(|s| s.as_str()) == Some("relu_backward")
                })
            }).copied() {
                Some(id) => id,
                None => continue,
            };

            let drelu_consumers: Vec<NodeId> = graph.consumers(drelu_id);
            let reducesum_ids: Vec<NodeId> = drelu_consumers.iter()
                .filter(|&&cid| graph.get_node(cid).map_or(false, |n| n.opcode == Opcode::ReduceSum))
                .copied()
                .collect();

            if reducesum_ids.is_empty() { continue; }

            for &rs_id in &reducesum_ids {
                if let Some(n) = graph.get_node_mut(rs_id) {
                    n.attrs.insert("fused_bwd_bias".to_string(), "MatMulAddRelu".to_string());
                    n.attrs.insert("fwd_fused_id".to_string(), fwd_id.to_string());
                }
            }

            fused = true;
        }

        Ok(fused)
    }
}
