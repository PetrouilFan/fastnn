use crate::ir::{ComputeGraph, IRNode, NodeId, Opcode, ShapeEnv, TensorValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exact bytes produced by a node at the executable boundary.
///
/// Most nodes use their tensor representation's storage size. Runtime weight
/// quantization is different: its arena value is a self-describing wire payload
/// containing geometry, affine metadata, and packed words. It must not inherit
/// `PackedTensor`'s private SIMD over-allocation margin.
pub(crate) fn node_output_byte_size(
    node: &IRNode,
    shape_env: Option<&ShapeEnv>,
) -> Result<usize, String> {
    if node.opcode != Opcode::Quantize {
        return node
            .output_type
            .try_byte_size_with_env(shape_env)
            .ok_or_else(|| format!("node {} output storage size overflows", node.id));
    }

    let mut dimensions = Vec::with_capacity(node.output_type.shape.len());
    for dimension in &node.output_type.shape {
        let value = match shape_env {
            Some(env) => dimension
                .evaluate_with_env(env)
                .map_err(|error| format!("quantize node {} output shape: {error}", node.id))?,
            None => dimension
                .evaluate()
                .ok_or_else(|| format!("quantize node {} has unresolved output shape", node.id))?,
        };
        dimensions.push(
            usize::try_from(value)
                .map_err(|_| format!("quantize node {} dimension does not fit usize", node.id))?,
        );
    }
    let num_channels = dimensions.first().copied().unwrap_or(1);
    if num_channels == 0 {
        return Err(format!("quantize node {} has zero channels", node.id));
    }
    let numel = dimensions.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .ok_or_else(|| format!("quantize node {} element count overflows", node.id))
    })?;
    let bit_width = node
        .attrs
        .get("bit_width")
        .ok_or_else(|| format!("quantize node {} is missing bit_width", node.id))?
        .parse::<usize>()
        .map_err(|_| format!("quantize node {} has invalid bit_width", node.id))?;
    if !matches!(bit_width, 4 | 8) {
        return Err(format!(
            "quantize node {} has unsupported bit_width {bit_width}",
            node.id
        ));
    }
    let items_per_word = 32 / bit_width;
    let packed_bytes = numel
        .div_ceil(items_per_word)
        .checked_mul(4)
        .ok_or_else(|| format!("quantize node {} packed size overflows", node.id))?;
    num_channels
        .checked_mul(8)
        .and_then(|metadata| metadata.checked_add(8))
        .and_then(|header| header.checked_add(packed_bytes))
        .ok_or_else(|| format!("quantize node {} wire payload size overflows", node.id))
}

/// A single allocation slot in the arena
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocSlot {
    pub offset: usize,
    pub size: usize,
    pub node_id: NodeId,
    pub output_index: usize,
}

/// The complete memory plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub total_size: usize,
    pub slots: HashMap<NodeId, AllocSlot>,
    /// Graph input node IDs in registration order.
    pub inputs: Vec<NodeId>,
    /// Secondary output slots for multi-output nodes (e.g. MaxPool argmax indices).
    /// Key is (node_id, output_index).
    pub secondary_slots: HashMap<(NodeId, usize), AllocSlot>,
    /// Graph output node IDs, in order.  Populated by [`plan_memory_with_env`].
    pub outputs: Vec<NodeId>,
    /// Runtime-resolved kernel parameter values (e.g. [M, K, N] for matmul)
    /// computed by [`tighten`] using the runtime [`ShapeEnv`].  Populated only
    /// after a call to [`tighten`]; empty at plan-creation time.
    pub tightened_params: HashMap<NodeId, Vec<usize>>,
}

impl MemoryPlan {
    pub fn validate(&self) -> Result<(), String> {
        for (&node_id, slot) in &self.slots {
            if slot.node_id != node_id || slot.output_index != 0 {
                return Err(format!(
                    "primary slot metadata for node {node_id} is inconsistent"
                ));
            }
            let end = slot
                .offset
                .checked_add(slot.size)
                .ok_or_else(|| format!("primary slot for node {node_id} range overflows"))?;
            if end > self.total_size {
                return Err(format!(
                    "primary slot for node {node_id} range {}..{end} exceeds arena size {}",
                    slot.offset, self.total_size
                ));
            }
        }
        for (&(node_id, output_index), slot) in &self.secondary_slots {
            if slot.node_id != node_id || slot.output_index != output_index || output_index == 0 {
                return Err(format!(
                    "secondary slot metadata for node {node_id} output {output_index} is inconsistent"
                ));
            }
            let end = slot.offset.checked_add(slot.size).ok_or_else(|| {
                format!("secondary slot for node {node_id} output {output_index} range overflows")
            })?;
            if end > self.total_size {
                return Err(format!(
                    "secondary slot for node {node_id} output {output_index} range {}..{end} exceeds arena size {}",
                    slot.offset, self.total_size
                ));
            }
        }
        for (&node_id, label) in self
            .inputs
            .iter()
            .map(|node_id| (node_id, "input"))
            .chain(self.outputs.iter().map(|node_id| (node_id, "output")))
        {
            if !self.slots.contains_key(&node_id) {
                return Err(format!(
                    "memory plan {label} node {node_id} has no primary slot"
                ));
            }
        }
        Ok(())
    }

    /// Tighten slot sizes using runtime-resolved shape information.
    ///
    /// Slots whose symbolic dims resolved to smaller concrete values are
    /// shrunk, and `total_size` is recomputed.  Offsets are not changed
    /// (the slot layout is preserved), so this is safe to call after the
    /// plan was compiled with max-estimate sizes.
    ///
    /// Also computes `tightened_params` — the runtime-resolved kernel
    /// parameters (e.g. [M, K, N] for matmul) for every node whose kernel
    /// depends on shape information.  These are stored so that
    /// [`tighten_slices`](crate::backend::executor::tighten_slices) can
    /// update [`Instruction::CallKernel`] params without a full recompile.
    pub fn try_tighten(
        &self,
        graph: &ComputeGraph,
        shape_env: &ShapeEnv,
    ) -> Result<MemoryPlan, String> {
        let mut mp = self.clone();
        let mut max_end = 0usize;
        for (_, slot) in mp.slots.iter_mut() {
            let tight_size = match graph.get_node(slot.node_id) {
                Some(node) => {
                    let logical_size = node_output_byte_size(node, Some(shape_env))?;
                    match &node.opcode {
                        Opcode::Constant(TensorValue::Data { bytes, .. }) => {
                            logical_size.max(bytes.len())
                        }
                        _ => logical_size,
                    }
                }
                None => slot.size,
            };
            slot.size = tight_size.min(slot.size);
            let slot_end = slot
                .offset
                .checked_add(slot.size)
                .ok_or_else(|| format!("memory slot for node {} range overflows", slot.node_id))?;
            max_end = max_end.max(slot_end);
        }
        for (_, slot) in mp.secondary_slots.iter_mut() {
            let tight_size = match graph
                .get_node(slot.node_id)
                .and_then(|node| node.secondary_output_type.as_ref())
            {
                Some(output_type) => output_type
                    .try_byte_size_with_env(Some(shape_env))
                    .ok_or_else(|| {
                        format!(
                            "secondary memory slot for node {} storage size overflows",
                            slot.node_id
                        )
                    })?,
                None => slot.size,
            };
            slot.size = tight_size.min(slot.size);
            let slot_end = slot.offset.checked_add(slot.size).ok_or_else(|| {
                format!(
                    "secondary memory slot for node {} range overflows",
                    slot.node_id
                )
            })?;
            max_end = max_end.max(slot_end);
        }
        mp.total_size = max_end;

        // ── Compute tightened kernel params ──────────────────────────
        // Iterate over every node in topological order and re-derive
        // shape-dependent kernel parameters using the concrete ShapeEnv.
        crate::utils::traverse_graph(graph, |node_id, node| {
            let mut resolved_input_shapes = Vec::with_capacity(node.inputs.len());
            for &input_id in &node.inputs {
                let input = graph.get_node(input_id).ok_or_else(|| {
                    format!("node {node_id} references missing input node {input_id}")
                })?;
                let mut shape = Vec::with_capacity(input.output_type.shape.len());
                for dimension in &input.output_type.shape {
                    shape.push(dimension.evaluate_with_env(shape_env).map_err(|error| {
                        format!("node {node_id} input {input_id} shape: {error}")
                    })?);
                }
                resolved_input_shapes.push(shape);
            }
            let to_usize = |value: u64, label: &str| {
                usize::try_from(value)
                    .map_err(|_| format!("node {node_id} {label} does not fit usize"))
            };

            let tightened = match node.opcode {
                Opcode::MatMul => {
                    let lhs = resolved_input_shapes
                        .first()
                        .ok_or_else(|| format!("matmul node {node_id} is missing its lhs input"))?;
                    let rhs = resolved_input_shapes
                        .get(1)
                        .ok_or_else(|| format!("matmul node {node_id} is missing its rhs input"))?;
                    if lhs.len() < 2 || rhs.len() < 2 {
                        return Err(format!("matmul node {node_id} requires rank-two inputs"));
                    }
                    let m = to_usize(lhs[lhs.len() - 2], "matmul M")?;
                    let k = to_usize(lhs[lhs.len() - 1], "matmul K")?;
                    let rhs_k = to_usize(rhs[rhs.len() - 2], "matmul rhs K")?;
                    let n = to_usize(rhs[rhs.len() - 1], "matmul N")?;
                    if k != rhs_k {
                        return Err(format!(
                            "matmul node {node_id} contraction mismatch: {k} versus {rhs_k}"
                        ));
                    }
                    vec![m, k, n]
                }
                Opcode::Transpose => {
                    let input_shape = resolved_input_shapes
                        .first()
                        .ok_or_else(|| format!("transpose node {node_id} is missing its input"))?;
                    let rank = input_shape.len();
                    if rank == 2 {
                        let m = to_usize(input_shape[0], "transpose rows")?;
                        let n = to_usize(input_shape[1], "transpose columns")?;
                        vec![m, n]
                    } else {
                        // N-D permute transpose: params = [rank, d0..dN, p0..pN]
                        // The perm comes from node attrs (e.g. "0,3,1,2")
                        let perm_str = node.attrs.get("perm").cloned().unwrap_or_default();
                        let capacity = rank
                            .checked_mul(2)
                            .and_then(|value| value.checked_add(1))
                            .ok_or_else(|| {
                            format!("transpose node {node_id} rank overflows")
                        })?;
                        let mut params: Vec<usize> = Vec::with_capacity(capacity);
                        params.push(rank);
                        for &dimension in input_shape {
                            params.push(to_usize(dimension, "transpose dimension")?);
                        }
                        if perm_str.is_empty() {
                            // Default: reverse
                            for i in (0..rank).rev() {
                                params.push(i);
                            }
                        } else {
                            let perm: Result<Vec<usize>, _> =
                                perm_str.split(',').map(str::parse).collect();
                            let perm = perm.map_err(|_| {
                                format!("transpose node {node_id} has invalid permutation")
                            })?;
                            if perm.len() != rank
                                || perm.iter().any(|&axis| axis >= rank)
                                || (0..rank).any(|axis| !perm.contains(&axis))
                            {
                                return Err(format!(
                                    "transpose node {node_id} permutation is not a rank-{rank} bijection"
                                ));
                            }
                            params.extend(perm);
                        }
                        params
                    }
                }
                Opcode::Softmax => {
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .ok_or_else(|| format!("softmax node {node_id} is missing its axis"))?
                        .parse()
                        .map_err(|_| format!("softmax node {node_id} has invalid axis"))?;
                    let input_shape = resolved_input_shapes
                        .first()
                        .ok_or_else(|| format!("softmax node {node_id} is missing its input"))?;
                    let rank = i64::try_from(input_shape.len())
                        .map_err(|_| format!("softmax node {node_id} rank does not fit i64"))?;
                    let normalized_axis = if axis < 0 { rank + axis } else { axis };
                    if normalized_axis < 0 || normalized_axis >= rank {
                        return Err(format!(
                            "softmax node {node_id} axis {axis} is out of range"
                        ));
                    }
                    let normalized_axis = usize::try_from(normalized_axis)
                        .map_err(|_| format!("softmax node {node_id} axis conversion failed"))?;
                    let axis_dim = to_usize(input_shape[normalized_axis], "softmax axis size")?;
                    let stride = input_shape[normalized_axis + 1..].iter().try_fold(
                        1usize,
                        |product, &dimension| {
                            product
                                .checked_mul(to_usize(dimension, "softmax stride dimension")?)
                                .ok_or_else(|| format!("softmax node {node_id} stride overflows"))
                        },
                    )?;
                    vec![axis_dim, stride]
                }
                Opcode::ReduceSum | Opcode::ReduceMean | Opcode::ReduceMax => {
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .ok_or_else(|| format!("reduction node {node_id} is missing its axis"))?
                        .parse()
                        .map_err(|_| format!("reduction node {node_id} has invalid axis"))?;
                    let input_shape = resolved_input_shapes
                        .first()
                        .ok_or_else(|| format!("reduction node {node_id} is missing its input"))?;
                    if axis >= input_shape.len() {
                        return Err(format!(
                            "reduction node {node_id} axis {axis} is out of range"
                        ));
                    }
                    let (is_mean, is_max) = match node.opcode {
                        Opcode::ReduceMean => (1, 0),
                        Opcode::ReduceMax => (0, 1),
                        _ => (0, 0), // ReduceSum
                    };
                    let mut params = Vec::with_capacity(input_shape.len() + 4);
                    params.push(input_shape.len());
                    for dimension in input_shape {
                        params.push(to_usize(*dimension, "reduction dimension")?);
                    }
                    params.extend([axis, is_mean, is_max]);
                    params
                }
                Opcode::Conv2d => {
                    let get_attr = |name: &str| -> Result<usize, String> {
                        node.attrs
                            .get(name)
                            .ok_or_else(|| {
                                format!("conv2d node {node_id} missing {name} attribute")
                            })?
                            .parse::<usize>()
                            .map_err(|_| {
                                format!("conv2d node {node_id} has invalid {name} attribute")
                            })
                    };
                    let stride = get_attr("stride")?;
                    let padding = get_attr("padding")?;
                    let dilation = get_attr("dilation")?;
                    let groups = get_attr("groups")?;
                    if stride == 0 || dilation == 0 || groups == 0 {
                        return Err(format!(
                            "conv2d node {node_id} requires positive stride, dilation, and groups"
                        ));
                    }

                    let input_shape = resolved_input_shapes.first().cloned().unwrap_or_default();
                    let weight_shape = resolved_input_shapes.get(1).cloned().unwrap_or_default();
                    let dimension = |shape: &[u64], index: usize, label: &str| {
                        let value = shape.get(index).copied().ok_or_else(|| {
                            format!("conv2d node {node_id} missing {label} dimension")
                        })?;
                        usize::try_from(value).map_err(|_| {
                            format!("conv2d node {node_id} {label} dimension does not fit usize")
                        })
                    };

                    let n_in = dimension(&input_shape, 0, "batch")?;
                    let c = dimension(&input_shape, 1, "channels")?;
                    let h = dimension(&input_shape, 2, "height")?;
                    let w = dimension(&input_shape, 3, "width")?;
                    let f_out = dimension(&weight_shape, 0, "output channels")?;
                    let kh = dimension(&weight_shape, 2, "kernel height")?;
                    let kw = dimension(&weight_shape, 3, "kernel width")?;
                    if c % groups != 0 {
                        return Err(format!(
                            "conv2d node {node_id} channels {c} are not divisible by groups {groups}"
                        ));
                    }
                    let c_per_group = c / groups;
                    let padded_h = padding
                        .checked_mul(2)
                        .and_then(|padding| h.checked_add(padding))
                        .ok_or_else(|| format!("conv2d node {node_id} padded height overflows"))?;
                    let padded_w = padding
                        .checked_mul(2)
                        .and_then(|padding| w.checked_add(padding))
                        .ok_or_else(|| format!("conv2d node {node_id} padded width overflows"))?;
                    let kernel_h = dilation
                        .checked_mul(kh.saturating_sub(1))
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| format!("conv2d node {node_id} kernel height overflows"))?;
                    let kernel_w = dilation
                        .checked_mul(kw.saturating_sub(1))
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| format!("conv2d node {node_id} kernel width overflows"))?;
                    let h_out = padded_h
                        .saturating_sub(kernel_h)
                        .checked_div(stride)
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| format!("conv2d node {node_id} output height is invalid"))?;
                    let w_out = padded_w
                        .saturating_sub(kernel_w)
                        .checked_div(stride)
                        .and_then(|value| value.checked_add(1))
                        .ok_or_else(|| format!("conv2d node {node_id} output width is invalid"))?;
                    let spatial_size = h_out.checked_mul(w_out).ok_or_else(|| {
                        format!("conv2d node {node_id} output spatial size overflows")
                    })?;
                    let col_w = c_per_group
                        .checked_mul(kh)
                        .and_then(|value| value.checked_mul(kw))
                        .ok_or_else(|| format!("conv2d node {node_id} column width overflows"))?;
                    vec![
                        stride,
                        padding,
                        dilation,
                        groups,
                        c,
                        h,
                        w,
                        kh,
                        kw,
                        n_in,
                        f_out,
                        h_out,
                        w_out,
                        spatial_size,
                        col_w,
                    ]
                }
                _ => return Ok(()),
            };
            mp.tightened_params.insert(node_id, tightened);
            Ok(())
        })?;

        Ok(mp)
    }
}
