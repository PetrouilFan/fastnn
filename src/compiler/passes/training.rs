use crate::ir::node::*;
use std::collections::HashMap;

pub enum OptimizerConfig {
    SGD {
        lr: f32,
        weight_decay: f32,
    },
    AdamW {
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    },
    Muon {
        lr: f32,
        beta1: f32,
        weight_decay: f32,
    },
    Lion {
        lr: f32,
        beta1: f32,
        beta2: f32,
    },
    RMSprop {
        lr: f32,
        beta: f32,
        eps: f32,
    },
}

pub struct TrainConfig {
    pub optimizer: OptimizerConfig,
    pub quantize: Option<u8>,
}

pub struct OptimizerInjection {
    pub updated_param_nodes: Vec<NodeId>,
    pub state_input_nodes: Vec<Vec<NodeId>>,
}

pub fn inject_optimizer(
    graph: &mut ComputeGraph,
    params_with_grads: &[(NodeId, NodeId)],
    config: &OptimizerConfig,
) -> Result<OptimizerInjection, String> {
    if params_with_grads.is_empty() {
        return Ok(OptimizerInjection {
            updated_param_nodes: vec![],
            state_input_nodes: vec![],
        });
    }

    let mut updated_param_nodes = Vec::with_capacity(params_with_grads.len());
    let mut state_input_nodes = Vec::with_capacity(params_with_grads.len());

    for &(param_id, grad_id) in params_with_grads {
        let param_node = graph
            .get_node(param_id)
            .ok_or_else(|| format!("param node {} not found in graph", param_id))?;
        let param_type = param_node.output_type.clone();
        let param_name = param_node.name.clone();

        match config {
            OptimizerConfig::SGD { lr, weight_decay } => {
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*lr)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*weight_decay)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );

                let mut attrs = HashMap::new();
                attrs.insert("lr".to_string(), lr.to_string());
                attrs.insert("weight_decay".to_string(), weight_decay.to_string());
                let sgd_id = graph.add_node_with_attrs(
                    Opcode::SgdUpdate,
                    vec![param_id, grad_id],
                    param_type,
                    attrs,
                );

                updated_param_nodes.push(sgd_id);
                state_input_nodes.push(vec![]);
            }
            OptimizerConfig::AdamW {
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
            } => {
                let m_id = graph.add_node(
                    Opcode::Input,
                    vec![],
                    TensorType {
                        shape: param_type.shape.clone(),
                        dtype: param_type.dtype.clone(),
                    },
                );
                if let Some(node) = graph.get_node_mut(m_id) {
                    node.name = format!("optimizer/m_{}", param_name);
                }

                let v_id = graph.add_node(
                    Opcode::Input,
                    vec![],
                    TensorType {
                        shape: param_type.shape.clone(),
                        dtype: param_type.dtype.clone(),
                    },
                );
                if let Some(node) = graph.get_node_mut(v_id) {
                    node.name = format!("optimizer/v_{}", param_name);
                }

                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*lr)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*beta1)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*beta2)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*eps)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*weight_decay)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );

                let mut attrs = HashMap::new();
                attrs.insert("lr".to_string(), lr.to_string());
                attrs.insert("beta1".to_string(), beta1.to_string());
                attrs.insert("beta2".to_string(), beta2.to_string());
                attrs.insert("eps".to_string(), eps.to_string());
                attrs.insert("t".to_string(), "1".to_string());
                attrs.insert("weight_decay".to_string(), weight_decay.to_string());
                let adamw_id = graph.add_node_with_attrs(
                    Opcode::AdamWUpdate,
                    vec![param_id, grad_id, m_id, v_id],
                    param_type,
                    attrs,
                );

                updated_param_nodes.push(adamw_id);
                state_input_nodes.push(vec![m_id, v_id]);
            }
            OptimizerConfig::Muon {
                lr,
                beta1,
                weight_decay,
            } => {
                let m_id = graph.add_node(
                    Opcode::Input,
                    vec![],
                    TensorType {
                        shape: param_type.shape.clone(),
                        dtype: param_type.dtype.clone(),
                    },
                );
                if let Some(node) = graph.get_node_mut(m_id) {
                    node.name = format!("optimizer/m_{}", param_name);
                }

                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*lr)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*beta1)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*weight_decay)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );

                let mut attrs = HashMap::new();
                attrs.insert("lr".to_string(), lr.to_string());
                attrs.insert("beta1".to_string(), beta1.to_string());
                attrs.insert("weight_decay".to_string(), weight_decay.to_string());
                let muon_id = graph.add_node_with_attrs(
                    Opcode::MuonUpdate,
                    vec![param_id, grad_id, m_id],
                    param_type,
                    attrs,
                );

                updated_param_nodes.push(muon_id);
                state_input_nodes.push(vec![m_id]);
            }
            OptimizerConfig::Lion { lr, beta1, beta2 } => {
                let m_id = graph.add_node(
                    Opcode::Input,
                    vec![],
                    TensorType {
                        shape: param_type.shape.clone(),
                        dtype: param_type.dtype.clone(),
                    },
                );
                if let Some(node) = graph.get_node_mut(m_id) {
                    node.name = format!("optimizer/m_{}", param_name);
                }

                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*lr)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*beta1)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*beta2)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );

                let mut attrs = HashMap::new();
                attrs.insert("lr".to_string(), lr.to_string());
                attrs.insert("beta1".to_string(), beta1.to_string());
                attrs.insert("beta2".to_string(), beta2.to_string());
                let lion_id = graph.add_node_with_attrs(
                    Opcode::LionUpdate,
                    vec![param_id, grad_id, m_id],
                    param_type,
                    attrs,
                );

                updated_param_nodes.push(lion_id);
                state_input_nodes.push(vec![m_id]);
            }
            OptimizerConfig::RMSprop { lr, beta, eps } => {
                let v_id = graph.add_node(
                    Opcode::Input,
                    vec![],
                    TensorType {
                        shape: param_type.shape.clone(),
                        dtype: param_type.dtype.clone(),
                    },
                );
                if let Some(node) = graph.get_node_mut(v_id) {
                    node.name = format!("optimizer/v_{}", param_name);
                }

                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*lr)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*beta)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );
                graph.add_node(
                    Opcode::Constant(TensorValue::Float(*eps)),
                    vec![],
                    TensorType {
                        shape: vec![DimExpr::Known(1)],
                        dtype: IrDType::F32,
                    },
                );

                let mut attrs = HashMap::new();
                attrs.insert("lr".to_string(), lr.to_string());
                attrs.insert("beta".to_string(), beta.to_string());
                attrs.insert("eps".to_string(), eps.to_string());
                let rmsprop_id = graph.add_node_with_attrs(
                    Opcode::RmspropUpdate,
                    vec![param_id, grad_id, v_id],
                    param_type,
                    attrs,
                );

                updated_param_nodes.push(rmsprop_id);
                state_input_nodes.push(vec![v_id]);
            }
        }
    }

    Ok(OptimizerInjection {
        updated_param_nodes,
        state_input_nodes,
    })
}
