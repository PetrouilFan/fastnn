use crate::ir::node::{ComputeGraph, DimExpr, IRNode, IrDType, Opcode, TensorType, TensorValue};

pub fn constant_fold(graph: &mut ComputeGraph) -> usize {
    let order = graph.topological_sort();
    let mut folded = 0;

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        if node.inputs.is_empty() || has_side_effects(&node.opcode) {
            continue;
        }

        let all_const = node
            .inputs
            .iter()
            .all(|&input_id| graph.get_node(input_id).map_or(false, |n| matches!(n.opcode, Opcode::Constant(_))));

        if !all_const {
            continue;
        }

        if let Some(value) = evaluate_node(graph, &node) {
            if let Some(node_mut) = graph.get_node_mut(node_id) {
                node_mut.opcode = Opcode::Constant(value);
                node_mut.inputs.clear();
            }
            folded += 1;
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

        Opcode::Sign => unary_float_op(&input_vals, |x| x.signum())
            .or_else(|| unary_int_op(&input_vals, |x| x.signum())),

        Opcode::LogicalNot => unary_int_op(&input_vals, |x| {
            if x == 0 {
                1
            } else {
                0
            }
        }),

        Opcode::Shape => {
            let first_input = node.inputs.first()?;
            let input_node = graph.get_node(*first_input)?;
            let shape = &input_node.output_type.shape;
            let dims: Vec<u64> = shape.iter().filter_map(|d| d.evaluate()).collect();
            if dims.len() == shape.len() && !dims.is_empty() {
                let bytes: Vec<u8> = dims.iter().flat_map(|&d| (d as i64).to_le_bytes()).collect();
                let rank = dims.len();
                Some(TensorValue::Data {
                    bytes,
                    tensor_type: TensorType::new(vec![DimExpr::Known(rank as u64)], IrDType::I64),
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
                && a_bytes.len() == b_bytes.len()
                && a_bytes.len() >= 4
                && a_bytes.len() % 4 == 0
            {
                let transformed: Vec<u8> = a_bytes
                    .chunks_exact(4)
                    .zip(b_bytes.chunks_exact(4))
                    .flat_map(|(a_chunk, b_chunk)| {
                        let a = f32::from_le_bytes(a_chunk.try_into().unwrap());
                        let b = f32::from_le_bytes(b_chunk.try_into().unwrap());
                        op(a, b).to_le_bytes()
                    })
                    .collect();
                Some(TensorValue::Data {
                    bytes: transformed,
                    tensor_type: a_ty.clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}
