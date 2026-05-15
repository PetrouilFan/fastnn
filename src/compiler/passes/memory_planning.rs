#![allow(dead_code)]

use crate::ir::node::{ComputeGraph, NodeId, Opcode, ShapeEnv, TensorType};
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
}

impl MemoryPlan {
    /// Tighten slot sizes using runtime-resolved shape information.
    ///
    /// Slots whose symbolic dims resolved to smaller concrete values are
    /// shrunk, and `total_size` is recomputed.  Offsets are not changed
    /// (the slot layout is preserved), so this is safe to call after the
    /// plan was compiled with max-estimate sizes.
    pub fn tighten(&self, graph: &ComputeGraph, shape_env: &ShapeEnv) -> Self {
        let mut new_slots = self.slots.clone();
        let mut new_secondary_slots = self.secondary_slots.clone();
        let mut max_end = 0usize;
        for (&node_id, slot) in &self.slots {
            let tight_size = graph
                .get_node(node_id)
                .map(|n| n.output_type.byte_size_with_env(Some(shape_env)))
                .unwrap_or(slot.size);
            let clamped = tight_size.min(slot.size);
            new_slots.insert(
                node_id,
                AllocSlot {
                    size: clamped,
                    ..*slot
                },
            );
            max_end = max_end.max(slot.offset + clamped);
        }
        for (&(node_id, _), slot) in &self.secondary_slots {
            let tight_size = graph
                .get_node(node_id)
                .and_then(|n| n.secondary_output_type.as_ref())
                .map(|t| t.byte_size_with_env(Some(shape_env)))
                .unwrap_or(slot.size);
            let clamped = tight_size.min(slot.size);
            new_secondary_slots.insert(
                (node_id, 1),
                AllocSlot {
                    size: clamped,
                    ..*slot
                },
            );
            max_end = max_end.max(slot.offset + clamped);
        }
        MemoryPlan {
            total_size: max_end,
            slots: new_slots,
            secondary_slots: new_secondary_slots,
        }
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

/// Add a free block to the free list, merging adjacent blocks
/// Align a value up to the next multiple of `alignment` (must be power of 2).
fn align_up(val: usize, alignment: usize) -> usize {
    (val + alignment - 1) & !(alignment - 1)
}

fn add_to_free_list(free_list: &mut Vec<FreeBlock>, offset: usize, size: usize) {
    if size == 0 {
        return;
    }

    let pos = match free_list.binary_search_by(|b| b.offset.cmp(&offset)) {
        Ok(exact_pos) => {
            // Offset already exists as a free block — this should never
            // happen in a correct allocator and indicates a double-free.
            // Merge by extending the existing block if the new size is larger.
            if free_list[exact_pos].size < size {
                free_list[exact_pos].size = size;
            }
            // Try merging with next block (may now overlap due to extension).
            if exact_pos + 1 < free_list.len()
                && free_list[exact_pos].offset + free_list[exact_pos].size
                    >= free_list[exact_pos + 1].offset
            {
                let next_end = free_list[exact_pos + 1].offset + free_list[exact_pos + 1].size;
                let merged_end = std::cmp::max(
                    free_list[exact_pos].offset + free_list[exact_pos].size,
                    next_end,
                );
                free_list[exact_pos].size = merged_end - free_list[exact_pos].offset;
                free_list.remove(exact_pos + 1);
            }
            return;
        }
        Err(pos) => pos,
    };

    // Merge with previous block if adjacent or overlapping
    if pos > 0 {
        let prev = &free_list[pos - 1];
        let prev_end = prev.offset + prev.size;
        if prev_end >= offset {
            // Extend previous block to cover the new range
            let new_end = std::cmp::max(prev_end, offset + size);
            free_list[pos - 1].size = new_end - free_list[pos - 1].offset;
            // Check if the extended previous block now overlaps with the next block
            if pos < free_list.len()
                && free_list[pos - 1].offset + free_list[pos - 1].size >= free_list[pos].offset
            {
                let next_end = free_list[pos].offset + free_list[pos].size;
                let merged_end = std::cmp::max(
                    free_list[pos - 1].offset + free_list[pos - 1].size,
                    next_end,
                );
                free_list[pos - 1].size = merged_end - free_list[pos - 1].offset;
                free_list.remove(pos);
            }
            return;
        }
    }

    // Merge with next block if adjacent or overlapping
    if pos < free_list.len() && offset + size >= free_list[pos].offset {
        let new_end = std::cmp::max(offset + size, free_list[pos].offset + free_list[pos].size);
        free_list[pos].offset = std::cmp::min(free_list[pos].offset, offset);
        free_list[pos].size = new_end - free_list[pos].offset;
        return;
    }

    if size > 0 {
        free_list.insert(pos, FreeBlock { offset, size });
    }
}

/// Find the smallest free block that fits the requested size (best-fit).
/// Returns the (index, offset) of the matching block without modifying the list.
fn find_best_fit(free_list: &[FreeBlock], size: usize) -> Option<(usize, usize)> {
    let mut best_idx = None;
    let mut best_size = usize::MAX;

    for (i, block) in free_list.iter().enumerate() {
        if block.size >= size && block.size < best_size {
            best_idx = Some(i);
            best_size = block.size;
        }
    }

    best_idx.map(|idx| (idx, free_list[idx].offset))
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
    let mut free_list: Vec<FreeBlock> = Vec::new();
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
                        add_to_free_list(&mut free_list, slot.offset, align_up(slot.size, 8));
                    }
                } else if let Some(slot) = slots.get(&expired_id) {
                    add_to_free_list(&mut free_list, slot.offset, align_up(slot.size, 8));
                }
            } else {
                i += 1;
            }
        }

        // Round up size to 8 bytes so all allocations start at 8-byte
        // aligned offsets (satisfies both f32 4-byte and i64 8-byte alignment).
        let size_aligned = align_up(info.size, 8);

        let offset = match find_best_fit(&free_list, size_aligned) {
            Some((idx, block_off)) => {
                let block_end = block_off + free_list[idx].size;
                // Remove the original block from the free list.
                free_list.remove(idx);
                // Allocation starts at the block offset (already 8-aligned
                // because all free-list entries were created from aligned
                // allocations).
                let alloc_start = block_off;
                let alloc_end = alloc_start + size_aligned;
                // Return leftover space before the allocation (should be 0
                // since block_off is always aligned).
                if alloc_start > block_off {
                    add_to_free_list(&mut free_list, block_off, alloc_start - block_off);
                }
                // Return leftover space after the allocation.
                if alloc_end < block_end {
                    add_to_free_list(&mut free_list, alloc_end, block_end - alloc_end);
                }
                alloc_start
            }
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
    })
}
