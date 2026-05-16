#![allow(dead_code)]

use crate::ir::node::{ComputeGraph, NodeId, Opcode, ShapeEnv, TensorType};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

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
        let mut mp = self.clone();
        let mut max_end = 0usize;
        for (_, slot) in mp.slots.iter_mut() {
            let tight_size = graph
                .get_node(slot.node_id)
                .map(|n| n.output_type.byte_size_with_env(Some(shape_env)))
                .unwrap_or(slot.size);
            slot.size = tight_size.min(slot.size);
            max_end = max_end.max(slot.offset + slot.size);
        }
        for (_, slot) in mp.secondary_slots.iter_mut() {
            let tight_size = graph
                .get_node(slot.node_id)
                .and_then(|n| n.secondary_output_type.as_ref())
                .map(|t| t.byte_size_with_env(Some(shape_env)))
                .unwrap_or(slot.size);
            slot.size = tight_size.min(slot.size);
            max_end = max_end.max(slot.offset + slot.size);
        }
        mp.total_size = max_end;

        // ── Compute tightened kernel params ──────────────────────────
        // Iterate over every node in topological order and re-derive
        // shape-dependent kernel parameters using the concrete ShapeEnv.
        let order = graph.topological_sort();
        for &node_id in &order {
            let node = match graph.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };
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
                        continue;
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
                    let input_shape = resolved_input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_default();
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
                        params.push(rank as usize);
                        params.extend(input_shape.iter().map(|&d| d as usize));
                        if perm_str.is_empty() {
                            // Default: reverse
                            for i in (0..rank).rev() {
                                params.push(i);
                            }
                        } else {
                            let perm: Vec<usize> = perm_str
                                .split(',')
                                .filter_map(|s| s.parse().ok())
                                .collect();
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
                _ => continue,
            };
            mp.tightened_params.insert(node_id, tightened);
        }

        mp
    }
}

/// Live range of a value: (first_use_index, last_use_index)
/// Where use_index is the position in topological order
#[derive(Debug, Clone, Copy)]
struct LiveRange(usize, usize);

/// Information needed for allocation
#[derive(Debug, Clone)]
struct AllocInfo {
    node_id: NodeId,
    size: usize,
    live_range: LiveRange,
    is_secondary: bool,
}

/// A free memory block
#[derive(Debug)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// Align a value up to the next multiple of `alignment` (must be power of 2).
fn align_up(val: usize, alignment: usize) -> usize {
    (val + alignment - 1) & !(alignment - 1)
}

/// Size-segregated free list allocator backed by a `BTreeMap`.
/// Buckets are keyed by power-of-2 size class for O(log n) best-fit search.
/// Allocation: find the smallest bucket with a free block meeting the size.
/// Deallocation: return the block to its size-class bucket.
struct SegFreeList {
    buckets: BTreeMap<usize, Vec<FreeBlock>>,
    large_blocks: Vec<FreeBlock>,
}

impl SegFreeList {
    fn new() -> Self {
        SegFreeList {
            buckets: BTreeMap::new(),
            large_blocks: Vec::new(),
        }
    }

    fn bucket_for(size: usize) -> Option<usize> {
        if size == 0 {
            return None;
        }
        let b = (usize::BITS - size.leading_zeros()) as usize - 1;
        if b < 64 {
            Some(b)
        } else {
            None
        }
    }

    fn add(&mut self, offset: usize, size: usize) {
        if size == 0 {
            return;
        }
        let aligned = align_up(size, 8);
        self.buckets.entry(aligned).or_default().push(FreeBlock {
            offset,
            size: aligned,
        });
    }

    fn alloc(&mut self, size: usize) -> Option<usize> {
        let aligned = align_up(size, 8);
        if aligned == 0 {
            return None;
        }

        let target_key = self.buckets.range(aligned..).next().map(|(&k, _)| k);
        if let Some(key) = target_key {
            if let Some(blocks) = self.buckets.get_mut(&key) {
                let block = blocks.swap_remove(0);
                let empty = blocks.is_empty();
                let off = block.offset;
                let extra = block.size - aligned;
                if empty {
                    self.buckets.remove(&key);
                }
                if extra > 0 {
                    self.add(off + aligned, extra);
                }
                return Some(off);
            }
        }

        if let Some((idx, _)) = self
            .large_blocks
            .iter()
            .enumerate()
            .find(|(_, b)| b.size >= aligned)
        {
            let block = self.large_blocks.remove(idx);
            let off = block.offset;
            let extra = block.size - aligned;
            if extra > 0 {
                self.add(off + aligned, extra);
            }
            return Some(off);
        }

        None
    }
}

/// Compute the byte size of a tensor's storage, optionally using a ShapeEnv
/// to resolve symbolic dims to tighter bounds.
fn tensor_byte_size(t: &TensorType, shape_env: Option<&ShapeEnv>) -> usize {
    t.byte_size_with_env(shape_env)
}

/// Plan memory using max estimates (no ShapeEnv).  Equivalent to
/// `plan_memory_with_env(graph, None)`.
pub fn plan_memory(graph: &ComputeGraph) -> Result<MemoryPlan, String> {
    plan_memory_with_env(graph, None)
}

/// Plan memory, optionally using a runtime ShapeEnv for tighter allocations.
///
/// When `shape_env` is `Some`, symbolic dimensions are resolved to their
/// concrete runtime values where possible, reducing the arena size compared
/// to the default `SYMBOL_DIM_MAX` fallback.
pub fn plan_memory_with_env(
    graph: &ComputeGraph,
    shape_env: Option<&ShapeEnv>,
) -> Result<MemoryPlan, String> {
    if graph.nodes.is_empty() {
        return Ok(MemoryPlan {
            total_size: 0,
            slots: HashMap::new(),
            secondary_slots: HashMap::new(),
            outputs: graph.outputs.clone(),
            tightened_params: HashMap::new(),
        });
    }

    let order = graph.topological_sort();

    let position: HashMap<NodeId, usize> =
        order.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    let mut alloc_infos: Vec<AllocInfo> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Primary output
        let size = tensor_byte_size(&node.output_type, shape_env);
        if size > 0 {
            // Input nodes have their data written by the executor before any
            // instruction runs, so their lifetime starts at position 0 — not
            // at their own position in the topological sort.  Without this,
            // the planner would incorrectly reuse an Input node's memory slot
            // for a Constant or intermediate that is produced earlier in the
            // execution order, corrupting the input data before all consumers
            // have read it (see e.g. autograd Mul backward sharing a slot
            // between Input b and Constant 0.25).
            let first_use = if matches!(node.opcode, Opcode::Input) {
                0
            } else {
                position.get(&node_id).copied().unwrap_or(0)
            };
            let consumers = graph.consumers(node_id);
            let last_use = if graph.required_nodes.contains(&node_id) {
                // Required nodes must stay alive until end of execution
                order.len() - 1
            } else if consumers.is_empty() {
                if graph.outputs.contains(&node_id) {
                    order.len() - 1
                } else {
                    first_use
                }
            } else {
                // When a node has consumers AND is also a graph output or
                // required node, extend its lifetime to the end of execution
                // so the memory slot is preserved for the final output read.
                // Without this check the memory planner would free the slot
                // after the last consumer, and a later node would reuse it —
                // corrupting the output data before the executor reads it.
                if graph.outputs.contains(&node_id) || graph.required_nodes.contains(&node_id) {
                    order.len() - 1
                } else {
                    consumers
                        .iter()
                        .filter_map(|cid| position.get(cid))
                        .copied()
                        .max()
                        .unwrap_or(first_use)
                }
            };

            alloc_infos.push(AllocInfo {
                node_id,
                size,
                live_range: LiveRange(first_use, last_use),
                is_secondary: false,
            });
        }

        // Secondary output (if any)
        if let Some(sec_type) = &node.secondary_output_type {
            let sec_size = tensor_byte_size(sec_type, shape_env);
            if sec_size > 0 {
                let first_use = position.get(&node_id).copied().unwrap_or(0);
                let consumers = graph.consumers(node_id);
                let last_use = if consumers.is_empty() {
                    if graph.outputs.contains(&node_id) {
                        order.len() - 1
                    } else {
                        first_use
                    }
                } else {
                    let consumer_last = consumers
                        .iter()
                        .filter_map(|cid| position.get(cid))
                        .copied()
                        .max()
                        .unwrap_or(first_use);
                    // If the node is a graph output, the secondary output must
                    // also live until the end, because the secondary slot's
                    // allocation is in the same arena and freeing it early
                    // changes the free-list state — which can cause a later
                    // allocation to land adjacent to or (in edge cases) overlap
                    // with the primary slot's data.
                    if graph.outputs.contains(&node_id) || graph.required_nodes.contains(&node_id) {
                        order.len() - 1
                    } else {
                        consumer_last
                    }
                };
                alloc_infos.push(AllocInfo {
                    node_id,
                    size: sec_size,
                    live_range: LiveRange(first_use, last_use),
                    is_secondary: true,
                });
            }
        }
    }

    // Sort by (start_time, primary-first, size-desc).
    // The primary-first tiebreaker ensures that a node's primary output
    // is always allocated to the main slot, even when the secondary output
    // (e.g. MaxPool i64 indices) is larger than the primary (f32 data).
    alloc_infos.sort_by(|a, b| {
        a.live_range
            .0
            .cmp(&b.live_range.0)
            .then_with(|| a.is_secondary.cmp(&b.is_secondary))
            .then_with(|| b.size.cmp(&a.size))
    });

    let mut slots: HashMap<NodeId, AllocSlot> = HashMap::new();
    let mut secondary_slots: HashMap<(NodeId, usize), AllocSlot> = HashMap::new();
    let mut active: Vec<(usize, NodeId, usize, bool)> = Vec::new();
    let mut free_list = SegFreeList::new();
    let mut arena_top: usize = 0;

    for info in &alloc_infos {
        let mut i = 0;
        while i < active.len() {
            if active[i].0 < info.live_range.0 {
                let (_expired_end, expired_id, _expired_size, was_secondary) =
                    active.swap_remove(i);
                // IMPORTANT: only free the slot that corresponds to this
                // specific active entry.  When a node has both a primary and
                // a secondary output (e.g. MaxPool), two separate entries
                // are pushed to `active` — one for each.  Freeing both slots
                // on every expiry would double-free the same memory region
                // into the free list, creating duplicate entries that let
                // the allocator hand out overlapping addresses for different
                // live ranges.
                if was_secondary {
                    if let Some(slot) = secondary_slots.get(&(expired_id, 1)) {
                        free_list.add(slot.offset, align_up(slot.size, 8));
                    }
                } else if let Some(slot) = slots.get(&expired_id) {
                    free_list.add(slot.offset, align_up(slot.size, 8));
                }
            } else {
                i += 1;
            }
        }

        // Round up to 64 bytes (cache line alignment) so all allocations
        // satisfy both SIMD (32-byte) alignment and cache line boundaries.
        let size_aligned = align_up(info.size, 64);

        let offset = match free_list.alloc(size_aligned) {
            Some(off) => off,
            None => {
                let off = align_up(arena_top, 64);
                arena_top = off + size_aligned;
                off
            }
        };

        let slot = AllocSlot {
            offset,
            size: info.size,
            node_id: info.node_id,
            output_index: 0,
        };

        if info.is_secondary {
            // This is a secondary output (e.g. MaxPool indices)
            let sec_slot = AllocSlot {
                offset,
                size: info.size,
                node_id: info.node_id,
                output_index: 1,
            };
            secondary_slots.insert((info.node_id, 1), sec_slot);
        } else {
            // Primary output — use the main slot
            slots.insert(info.node_id, slot);
        }

        if info.size > 0 {
            active.push((
                info.live_range.1,
                info.node_id,
                info.size,
                info.is_secondary,
            ));
        }
    }

    // ── In-place buffer reuse for unary elementwise ops ──────────────
    // When a unary elementwise node's input has no other consumers after
    // this node, and the output is not a graph output, reuse the input's
    // buffer instead of allocating a new one.  This reduces arena size.
    //
    // We do NOT reuse buffers of Input nodes — the autograd backward pass
    // reads forward inputs directly, and sharing the input's buffer with
    // the elementwise output can cause subtle data-race-like issues.
    let unary_elementwise_ops: &[Opcode] = &[
        Opcode::Relu,
        Opcode::Gelu,
        Opcode::Silu,
        Opcode::Sigmoid,
        Opcode::Tanh,
        Opcode::Exp,
        Opcode::Log,
        Opcode::Sqrt,
        Opcode::Neg,
        Opcode::Abs,
        Opcode::LeakyRelu,
        Opcode::Elu,
        Opcode::Softplus,
        Opcode::Hardswish,
        Opcode::Clamp,
        Opcode::Sign,
        Opcode::LogicalNot,
        Opcode::LogSoftmax,
        Opcode::Mish,
        Opcode::Erf,
        Opcode::ToF16,
        Opcode::ToF32,
        Opcode::Cast,
    ];
    struct ReuseCandidate {
        node_id: NodeId,
        input_id: NodeId,
        output_last_use: usize,
        ai_idx: usize,
    }
    let mut reuse_candidates: Vec<ReuseCandidate> = Vec::new();
    for ai_idx in 0..alloc_infos.len() {
        let info = &alloc_infos[ai_idx];
        if info.is_secondary {
            continue;
        }
        let node = match graph.get_node(info.node_id) {
            Some(n) => n,
            None => continue,
        };
        if !unary_elementwise_ops.contains(&node.opcode) || node.inputs.len() != 1 {
            continue;
        }
        if graph.outputs.contains(&info.node_id) {
            continue;
        }
        let input_id = node.inputs[0];
        // Skip Input nodes — their buffer is used directly by the executor
        // and sharing it can interfere with autograd backward reads.
        if let Some(input_node) = graph.get_node(input_id) {
            if matches!(input_node.opcode, Opcode::Input) {
                continue;
            }
        }
        let input_type = match graph.get_node(input_id) {
            Some(n) => &n.output_type,
            None => continue,
        };
        let input_size = tensor_byte_size(input_type, shape_env);
        if input_size != info.size || input_size == 0 {
            continue;
        }
        let node_pos = position.get(&info.node_id).copied().unwrap_or(0);
        let consumers = graph.consumers(input_id);
        let has_consumer_after = consumers.iter().any(|&cid| {
            cid != info.node_id && position.get(&cid).copied().unwrap_or(0) > node_pos
        });
        if has_consumer_after {
            continue;
        }
        reuse_candidates.push(ReuseCandidate {
            node_id: info.node_id,
            input_id,
            output_last_use: info.live_range.1,
            ai_idx,
        });
    }
    for cand in &reuse_candidates {
        if let Some(input_info) = alloc_infos
            .iter_mut()
            .find(|ai| ai.node_id == cand.input_id && !ai.is_secondary)
        {
            input_info.live_range.1 = input_info.live_range.1.max(cand.output_last_use);
        }
    }
    for cand in &reuse_candidates {
        if let Some(info) = alloc_infos.get_mut(cand.ai_idx) {
            info.size = 0;
        }
    }

    // ── Post-allocation: set reuse slots to input offsets ──
    // These were marked with size=0 during the in-place reuse pass above.
    // We must do this AFTER the active-list loop to avoid double-free.
    for cand in &reuse_candidates {
        let input_offset = slots.get(&cand.input_id).map(|s| s.offset);
        if let Some(off) = input_offset {
            if let Some(slot) = slots.get_mut(&cand.node_id) {
                slot.offset = off;
            }
        }
    }

    Ok(MemoryPlan {
        total_size: arena_top,
        slots,
        secondary_slots,
        outputs: graph.outputs.clone(),
        tightened_params: HashMap::new(),
    })
}
