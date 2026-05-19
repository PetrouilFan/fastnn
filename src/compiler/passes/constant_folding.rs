use crate::ir::node::{ComputeGraph, DimExpr, IRNode, IrDType, NodeId, Opcode, TensorType, TensorValue};

pub fn constant_fold(graph: &mut ComputeGraph) -> usize {
    struct Fold {
        node_id: NodeId,
        value: TensorValue,
    }
    let mut folds: Vec<Fold> = Vec::new();

    let graph_ref = &*graph;
    let _ = crate::utils::traverse_graph(graph_ref, |node_id, node| {
        if node.inputs.is_empty() || has_side_effects(&node.opcode) {
            return Ok(());
        }

        let all_const = node
            .inputs
            .iter()
            .all(|&input_id| graph_ref.get_node(input_id).map_or(false, |n| matches!(n.opcode, Opcode::Constant(_))));

        if !all_const {
            return Ok(());
        }

        if let Some(value) = evaluate_node(graph_ref, node) {
            folds.push(Fold { node_id, value });
        }

        Ok(())
    });

    let folded = folds.len();
    for f in folds {
        if let Some(node_mut) = graph.get_node_mut(f.node_id) {
            node_mut.opcode = Opcode::Constant(f.value);
            node_mut.inputs.clear();
        }
    }

    if folded > 0 {
        graph.mark_mutated();
    }

    folded
}

fn has_side_effects(opcode: &Opcode) -> bool {
    matches!(
        opcode,
        Opcode::SgdUpdate
            | Opcode::AdamUpdate
            | Opcode::AdamWUpdate
            | Opcode::MuonUpdate
            | Opcode::LionUpdate
            | Opcode::RmspropUpdate
            | Opcode::GradientScale
    )
}

fn evaluate_node(graph: &ComputeGraph, node: &IRNode) -> Option<TensorValue> {
    let input_vals: Vec<TensorValue> = node
        .inputs
        .iter()
        .filter_map(|&id| match graph.get_node(id) {
            Some(n) => match &n.opcode {
                Opcode::Constant(v) => Some(v.clone()),
                _ => None,
            },
            None => None,
        })
        .collect();

    match node.opcode {
        Opcode::Neg => unary_float_op(&input_vals, |x| -x)
            .or_else(|| unary_int_op(&input_vals, |x| -x))
            .or_else(|| unary_f32_data_op(&input_vals, |x| -x)),

        Opcode::Abs => unary_float_op(&input_vals, |x| x.abs())
            .or_else(|| unary_int_op(&input_vals, |x| x.abs()))
            .or_else(|| unary_f32_data_op(&input_vals, |x| x.abs())),

        Opcode::Exp => unary_float_op(&input_vals, |x| x.exp())
            .or_else(|| unary_f32_data_op(&input_vals, |x| x.exp())),

        Opcode::Log => unary_float_op(&input_vals, |x| x.ln())
            .or_else(|| unary_f32_data_op(&input_vals, |x| x.ln())),

        Opcode::Sqrt => unary_float_op(&input_vals, |x| x.sqrt())
            .or_else(|| unary_f32_data_op(&input_vals, |x| x.sqrt())),

        Opcode::Relu => unary_float_op(&input_vals, |x| x.max(0.0))
            .or_else(|| unary_f32_data_op(&input_vals, |x| x.max(0.0))),

        Opcode::Sigmoid => unary_float_op(&input_vals, |x| 1.0 / (1.0 + (-x).exp()))
            .or_else(|| unary_f32_data_op(&input_vals, |x| 1.0 / (1.0 + (-x).exp()))),

        Opcode::Tanh => unary_float_op(&input_vals, |x| x.tanh())
            .or_else(|| unary_f32_data_op(&input_vals, |x| x.tanh())),

        Opcode::Gelu => unary_float_op(&input_vals, |x| {
            0.5 * x * (1.0 + (x * 0.7978845608028654_f32).tanh())
        })
        .or_else(|| {
            unary_f32_data_op(&input_vals, |x| {
                0.5 * x * (1.0 + (x * 0.7978845608028654_f32).tanh())
            })
        }),

        Opcode::Silu => unary_float_op(&input_vals, |x| x / (1.0 + (-x).exp()))
            .or_else(|| unary_f32_data_op(&input_vals, |x| x / (1.0 + (-x).exp()))),

        Opcode::Mish => unary_float_op(&input_vals, |x| {
            let sp = (1.0 + x.exp()).ln();
            x * sp.tanh()
        })
        .or_else(|| {
            unary_f32_data_op(&input_vals, |x| {
                let sp = (1.0 + x.exp()).ln();
                x * sp.tanh()
            })
        }),

        Opcode::Add => binary_float_op(&input_vals, |a, b| a + b)
            .or_else(|| binary_int_op(&input_vals, |a, b| a + b))
            .or_else(|| binary_f32_data_op(&input_vals, |a, b| a + b)),

        Opcode::Sub => binary_float_op(&input_vals, |a, b| a - b)
            .or_else(|| binary_int_op(&input_vals, |a, b| a - b))
            .or_else(|| binary_f32_data_op(&input_vals, |a, b| a - b)),

        Opcode::Mul => binary_float_op(&input_vals, |a, b| a * b)
            .or_else(|| binary_int_op(&input_vals, |a, b| a * b))
            .or_else(|| binary_f32_data_op(&input_vals, |a, b| a * b)),

        Opcode::Div => binary_float_op(&input_vals, |a, b| a / b)
            .or_else(|| binary_int_op(&input_vals, |a, b| a / b))
            .or_else(|| binary_f32_data_op(&input_vals, |a, b| a / b)),

        Opcode::Minimum => binary_float_op(&input_vals, |a, b| a.min(b))
            .or_else(|| binary_int_op(&input_vals, |a, b| a.min(b)))
            .or_else(|| binary_f32_data_op(&input_vals, |a, b| a.min(b))),

        Opcode::Maximum => binary_float_op(&input_vals, |a, b| a.max(b))
            .or_else(|| binary_int_op(&input_vals, |a, b| a.max(b)))
            .or_else(|| binary_f32_data_op(&input_vals, |a, b| a.max(b))),

        Opcode::Sign => unary_float_op(&input_vals, |x| x.signum())
            .or_else(|| unary_int_op(&input_vals, |x| x.signum())),

        Opcode::LogicalNot => unary_int_op(&input_vals, |x| {
            if x == 0 {
                1
            } else {
                0
            }
        }),

        Opcode::Round => unary_float_op(&input_vals, |x| x.round())
            .or_else(|| unary_f32_data_op(&input_vals, |x| x.round())),

        Opcode::Erf => unary_float_op(&input_vals, |x| {
            // Approximation of erf(x) using the Horner form of the Abramowitz & Stegun formula
            let sign = x.signum();
            let x_abs = x.abs();
            let t = 1.0 / (1.0 + 0.3275911 * x_abs);
            let y = 1.0 - ((((1.061405429 * t + 1.453152027) * t + 1.421413741) * t
                - 0.284496736)
                * t
                + 0.254829592)
                * t
                * (-x_abs * x_abs).exp();
            sign * y
        })
        .or_else(|| {
            unary_f32_data_op(&input_vals, |x| {
                let sign = x.signum();
                let x_abs = x.abs();
                let t = 1.0 / (1.0 + 0.3275911 * x_abs);
                let y = 1.0 - ((((1.061405429 * t + 1.453152027) * t + 1.421413741) * t
                    - 0.284496736)
                    * t
                    + 0.254829592)
                    * t
                    * (-x_abs * x_abs).exp();
                sign * y
            })
        }),

        Opcode::Reshape => {
            // Reshape of constant data: change the TensorType shape metadata
            if let Some(TensorValue::Data { bytes, tensor_type: _ }) = input_vals.first() {
                let output_type = &node.output_type;
                Some(TensorValue::Data {
                    bytes: bytes.clone(),
                    tensor_type: output_type.clone(),
                })
            } else if let Some(TensorValue::Float(v)) = input_vals.first() {
                let output_type = &node.output_type;
                let n: u64 = output_type.shape.iter().filter_map(|d| d.evaluate()).product();
                if output_type.dtype == IrDType::F32 && n == 1 {
                    Some(TensorValue::Data {
                        bytes: v.to_le_bytes().to_vec(),
                        tensor_type: output_type.clone(),
                    })
                } else {
                    None
                }
            } else if let Some(TensorValue::Int(v)) = input_vals.first() {
                let output_type = &node.output_type;
                let n: u64 = output_type.shape.iter().filter_map(|d| d.evaluate()).product();
                if output_type.dtype == IrDType::I64 && n == 1 {
                    Some(TensorValue::Data {
                        bytes: v.to_le_bytes().to_vec(),
                        tensor_type: output_type.clone(),
                    })
                } else {
                    None
                }
            } else {
                None
            }
        }

        Opcode::Cast => {
            // Cast constant data to target dtype (stored in output_type.dtype)
            let target = &node.output_type.dtype;
            if let Some(TensorValue::Data { bytes, tensor_type }) = input_vals.first() {
                let src_dtype = &tensor_type.dtype;
                let out_shape = tensor_type.shape.clone();
                let elem_count = tensor_type.numel().unwrap_or(bytes.len() as u64 / 4) as usize;
                match (src_dtype, target) {
                    (IrDType::F32, IrDType::I32) => {
                        let src: &[f32] = bytemuck::cast_slice(bytes);
                        let dst: Vec<i32> = src.iter().take(elem_count).map(|&x| x as i32).collect();
                        Some(TensorValue::Data {
                            bytes: bytemuck::cast_slice(&dst).to_vec(),
                            tensor_type: TensorType::new(out_shape, target.clone()),
                        })
                    }
                    (IrDType::I32, IrDType::F32) => {
                        let src: &[i32] = bytemuck::cast_slice(bytes);
                        let dst: Vec<f32> = src.iter().take(elem_count).map(|&x| x as f32).collect();
                        Some(TensorValue::Data {
                            bytes: bytemuck::cast_slice(&dst).to_vec(),
                            tensor_type: TensorType::new(out_shape, target.clone()),
                        })
                    }
                    (IrDType::F32, IrDType::I64) => {
                        let src: &[f32] = bytemuck::cast_slice(bytes);
                        let dst: Vec<i64> = src.iter().take(elem_count).map(|&x| x as i64).collect();
                        Some(TensorValue::Data {
                            bytes: bytemuck::cast_slice(&dst).to_vec(),
                            tensor_type: TensorType::new(out_shape, target.clone()),
                        })
                    }
                    (IrDType::I64, IrDType::F32) => {
                        let src: &[i64] = bytemuck::cast_slice(bytes);
                        let dst: Vec<f32> = src.iter().take(elem_count).map(|&x| x as f32).collect();
                        Some(TensorValue::Data {
                            bytes: bytemuck::cast_slice(&dst).to_vec(),
                            tensor_type: TensorType::new(out_shape, target.clone()),
                        })
                    }
                    (IrDType::F32, IrDType::F16) | (IrDType::F32, IrDType::BF16) => {
                        // Fall back to runtime kernel for F16/BF16 conversion
                        None
                    }
                    _ => None,
                }
            } else {
                None
            }
        }

        Opcode::Shape => {
            let first_input = node.inputs.first()?;
            let input_node = graph.get_node(*first_input)?;
            let shape = &input_node.output_type.shape;
            let dims: Vec<u64> = shape.iter().filter_map(|d| d.evaluate()).collect();
            if dims.len() == shape.len() && !dims.is_empty() {
                let rank = dims.len();
                let shape_data: Vec<f32> = dims.iter().map(|&s| s as f32).collect();
                let bytes: Vec<u8> = bytemuck::cast_slice(&shape_data).to_vec();
                Some(TensorValue::Data {
                    bytes,
                    tensor_type: TensorType::new(vec![DimExpr::Known(rank as u64)], IrDType::F32),
                })
            } else {
                None
            }
        }

        _ => None,
    }
}

fn unary_float_op(inputs: &[TensorValue], op: impl Fn(f32) -> f32) -> Option<TensorValue> {
    if let Some(TensorValue::Float(v)) = inputs.first() {
        Some(TensorValue::Float(op(*v)))
    } else {
        None
    }
}

fn unary_int_op(inputs: &[TensorValue], op: impl Fn(i64) -> i64) -> Option<TensorValue> {
    if let Some(TensorValue::Int(v)) = inputs.first() {
        Some(TensorValue::Int(op(*v)))
    } else {
        None
    }
}

fn unary_f32_data_op(inputs: &[TensorValue], op: impl Fn(f32) -> f32) -> Option<TensorValue> {
    if let Some(TensorValue::Data { bytes, tensor_type }) = inputs.first() {
        if tensor_type.dtype == IrDType::F32 && bytes.len() >= 4 && bytes.len() % 4 == 0 {
            let transformed: Vec<u8> = bytes
                .chunks_exact(4)
                .flat_map(|chunk| {
                    let val = f32::from_le_bytes(chunk.try_into().unwrap());
                    op(val).to_le_bytes()
                })
                .collect();
            Some(TensorValue::Data {
                bytes: transformed,
                tensor_type: tensor_type.clone(),
            })
        } else {
            None
        }
    } else {
        None
    }
}

fn binary_float_op(inputs: &[TensorValue], op: impl Fn(f32, f32) -> f32) -> Option<TensorValue> {
    if inputs.len() < 2 {
        return None;
    }
    match (&inputs[0], &inputs[1]) {
        (TensorValue::Float(a), TensorValue::Float(b)) => Some(TensorValue::Float(op(*a, *b))),
        _ => None,
    }
}

fn binary_int_op(inputs: &[TensorValue], op: impl Fn(i64, i64) -> i64) -> Option<TensorValue> {
    if inputs.len() < 2 {
        return None;
    }
    match (&inputs[0], &inputs[1]) {
        (TensorValue::Int(a), TensorValue::Int(b)) => Some(TensorValue::Int(op(*a, *b))),
        _ => None,
    }
}

fn binary_f32_data_op(inputs: &[TensorValue], op: impl Fn(f32, f32) -> f32) -> Option<TensorValue> {
    if inputs.len() < 2 {
        return None;
    }
    match (&inputs[0], &inputs[1]) {
        (
            TensorValue::Data {
                bytes: a_bytes,
                tensor_type: a_ty,
            },
            TensorValue::Data {
                bytes: b_bytes,
                tensor_type: b_ty,
            },
        ) => {
            if a_ty.dtype == IrDType::F32
                && b_ty.dtype == IrDType::F32
                && a_bytes.len() >= 4
                && a_bytes.len() % 4 == 0
                && b_bytes.len() >= 4
                && b_bytes.len() % 4 == 0
            {
                let a_f32: &[f32] = bytemuck::cast_slice(a_bytes);
                let b_f32: &[f32] = bytemuck::cast_slice(b_bytes);
                let len = a_f32.len().max(b_f32.len());
                if a_f32.len() != b_f32.len() && a_f32.len() != 1 && b_f32.len() != 1 {
                    return None;
                }
                let result: Vec<f32> = (0..len)
                    .map(|i| {
                        let a = if a_f32.len() == 1 { a_f32[0] } else { a_f32[i] };
                        let b = if b_f32.len() == 1 { b_f32[0] } else { b_f32[i] };
                        op(a, b)
                    })
                    .collect();
                Some(TensorValue::Data {
                    bytes: bytemuck::cast_slice(&result).to_vec(),
                    tensor_type: a_ty.clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}
