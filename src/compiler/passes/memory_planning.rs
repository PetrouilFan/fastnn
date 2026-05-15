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
}

impl MemoryPlan {
    /// Tighten slot sizes using runtime-resolved shape information.
    ///
    /// Slots whose symbolic dims resolved to smaller concrete values are
    /// shrunk, and `total_size` is recomputed.  Offsets are not changed
    /// (the slot layout is preserved), so this is safe to call after the
    /// plan was compiled with max-estimate sizes.
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
        if size == 0 {
            eprintln!(
                "[memory_planning] SKIPPING node={} op={:?} name='{}' size=0 type={:?}",
                node_id, node.opcode, node.name, node.output_type
            );
        }
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

        // Round up size to 8 bytes so all allocations start at 8-byte
        // aligned offsets (satisfies both f32 4-byte and i64 8-byte alignment).
        let size_aligned = align_up(info.size, 8);

        let offset = match free_list.alloc(size_aligned) {
            Some(off) => off,
            None => {
                let off = align_up(arena_top, 8);
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

        active.push((
            info.live_range.1,
            info.node_id,
            info.size,
            info.is_secondary,
        ));
    }

    Ok(MemoryPlan {
        total_size: arena_top,
        slots,
        secondary_slots,
        outputs: graph.outputs.clone(),
    })
}
