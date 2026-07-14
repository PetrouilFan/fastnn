use crate::ir::{ComputeGraph, NodeId, Opcode, ShapeEnv};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub fn tighten(&self, graph: &ComputeGraph, shape_env: &ShapeEnv) -> MemoryPlan {
        self.try_tighten(graph, shape_env)
            .expect("MemoryPlan::tighten failed")
    }

    pub fn try_tighten(
        &self,
        graph: &ComputeGraph,
        shape_env: &ShapeEnv,
    ) -> Result<MemoryPlan, String> {
        let mut mp = self.clone();
        let mut max_end = 0usize;
        for (_, slot) in mp.slots.iter_mut() {
            let tight_size = graph
                .get_node(slot.node_id)
                .map(|n| n.output_type.byte_size_with_env(Some(shape_env)))
                .unwrap_or(slot.size);
            slot.size = tight_size.min(slot.size);
            let slot_end = slot
                .offset
                .checked_add(slot.size)
                .ok_or_else(|| format!("memory slot for node {} range overflows", slot.node_id))?;
            max_end = max_end.max(slot_end);
        }
        for (_, slot) in mp.secondary_slots.iter_mut() {
            let tight_size = graph
                .get_node(slot.node_id)
                .and_then(|n| n.secondary_output_type.as_ref())
                .map(|t| t.byte_size_with_env(Some(shape_env)))
                .unwrap_or(slot.size);
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
            let resolved_input_shapes: Vec<Vec<u64>> = node
                .inputs
                .iter()
                .filter_map(|&id| graph.get_node(id))
                .map(|n| {
                    n.output_type
                        .shape
                        .iter()
                        .map(|d| d.evaluate_with_env(shape_env).unwrap_or(0))
                        .collect()
                })
                .collect();

            let tightened = match node.opcode {
                Opcode::MatMul => {
                    if resolved_input_shapes.len() < 2 {
                        return Ok(());
                    }
                    let m = resolved_input_shapes[0]
                        .get(resolved_input_shapes[0].len().saturating_sub(2))
                        .copied()
                        .unwrap_or(1) as usize;
                    let k = resolved_input_shapes[0].last().copied().unwrap_or(1) as usize;
                    let n = resolved_input_shapes[1].last().copied().unwrap_or(1) as usize;
                    vec![m, k, n]
                }
                Opcode::Transpose => {
                    let input_shape = resolved_input_shapes.first().cloned().unwrap_or_default();
                    let rank = input_shape.len();
                    if rank == 2 {
                        // 2D transpose: params = [M, N]
                        let m = input_shape[0] as usize;
                        let n = input_shape[1] as usize;
                        vec![m, n]
                    } else {
                        // N-D permute transpose: params = [rank, d0..dN, p0..pN]
                        // The perm comes from node attrs (e.g. "0,3,1,2")
                        let perm_str = node.attrs.get("perm").cloned().unwrap_or_default();
                        let mut params: Vec<usize> = Vec::with_capacity(1 + 2 * rank);
                        params.push(rank);
                        params.extend(input_shape.iter().map(|&d| d as usize));
                        if perm_str.is_empty() {
                            // Default: reverse
                            for i in (0..rank).rev() {
                                params.push(i);
                            }
                        } else {
                            let perm: Vec<usize> =
                                perm_str.split(',').filter_map(|s| s.parse().ok()).collect();
                            for i in 0..rank {
                                params.push(perm.get(i).copied().unwrap_or(i));
                            }
                        }
                        params
                    }
                }
                Opcode::Softmax => {
                    let axis: i64 = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let rank = resolved_input_shapes.first().map(|s| s.len()).unwrap_or(1);
                    let normalized_axis = if axis < 0 {
                        (rank as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    let axis_dim = resolved_input_shapes
                        .first()
                        .and_then(|s| s.get(normalized_axis).copied())
                        .unwrap_or(1) as usize;
                    let stride = resolved_input_shapes
                        .first()
                        .map(|s| {
                            s[normalized_axis + 1..]
                                .iter()
                                .copied()
                                .map(|x| x as usize)
                                .product::<usize>()
                                .max(1)
                        })
                        .unwrap_or(1);
                    vec![axis_dim, stride]
                }
                Opcode::ReduceSum | Opcode::ReduceMean | Opcode::ReduceMax => {
                    let axis: usize = node
                        .attrs
                        .get("axis")
                        .and_then(|a| a.parse().ok())
                        .unwrap_or(0);
                    let group_size = resolved_input_shapes
                        .first()
                        .and_then(|s| s.get(axis).copied())
                        .unwrap_or(1) as usize;
                    let (is_mean, is_max) = match node.opcode {
                        Opcode::ReduceMean => (1, 0),
                        Opcode::ReduceMax => (0, 1),
                        _ => (0, 0), // ReduceSum
                    };
                    vec![group_size, is_mean, is_max]
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
