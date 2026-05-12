#![allow(dead_code)]

use crate::ir::node::{ComputeGraph, DimExpr, NodeId, TensorType};
use std::collections::HashMap;

/// A single allocation slot in the arena
#[derive(Debug, Clone)]
pub struct AllocSlot {
    pub offset: usize,
    pub size: usize,
    pub node_id: NodeId,
}

/// The complete memory plan
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub total_size: usize,
    pub slots: HashMap<NodeId, AllocSlot>,
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
}

/// A free memory block
#[derive(Debug)]
struct FreeBlock {
    offset: usize,
    size: usize,
}

/// Compute the byte size of a tensor's storage
fn tensor_byte_size(t: &TensorType) -> usize {
    t.byte_size()
}

/// Add a free block to the free list, merging adjacent blocks
fn add_to_free_list(free_list: &mut Vec<FreeBlock>, offset: usize, size: usize) {
    let pos = free_list.binary_search_by(|b| b.offset.cmp(&offset)).unwrap_or_else(|e| e);

    if pos > 0 {
        let prev = &free_list[pos - 1];
        if prev.offset + prev.size == offset {
            free_list[pos - 1].size += size;
            if pos < free_list.len() && offset + size == free_list[pos].offset {
                free_list[pos - 1].size += free_list[pos].size;
                free_list.remove(pos);
            }
            return;
        }
    }

    if pos < free_list.len() && offset + size == free_list[pos].offset {
        free_list[pos].offset = offset;
        free_list[pos].size += size;
        return;
    }

    free_list.insert(pos, FreeBlock { offset, size });
}

/// Find the smallest free block that fits the requested size (best-fit)
fn find_best_fit(free_list: &mut Vec<FreeBlock>, size: usize) -> Option<usize> {
    let mut best_idx = None;
    let mut best_size = usize::MAX;

    for (i, block) in free_list.iter().enumerate() {
        if block.size >= size && block.size < best_size {
            best_idx = Some(i);
            best_size = block.size;
        }
    }

    if let Some(idx) = best_idx {
        let block = &free_list[idx];
        let offset = block.offset;
        let remaining = block.size - size;

        if remaining == 0 {
            free_list.remove(idx);
        } else {
            free_list[idx].offset = offset + size;
            free_list[idx].size = remaining;
        }

        Some(offset)
    } else {
        None
    }
}

pub fn plan_memory(graph: &ComputeGraph) -> Result<MemoryPlan, String> {
    if graph.nodes.is_empty() {
        return Ok(MemoryPlan {
            total_size: 0,
            slots: HashMap::new(),
        });
    }

    let order = graph.topological_sort();

    let position: HashMap<NodeId, usize> = order.iter().enumerate()
        .map(|(i, &id)| (id, i)).collect();

    let mut alloc_infos: Vec<AllocInfo> = Vec::new();

    for &node_id in &order {
        let node = match graph.get_node(node_id) {
            Some(n) => n,
            None => continue,
        };

        let size = tensor_byte_size(&node.output_type);
        if size == 0 { continue; }

        let first_use = position.get(&node_id).copied().unwrap_or(0);

        let consumers = graph.consumers(node_id);
        let last_use = if consumers.is_empty() {
            if graph.outputs.contains(&node_id) {
                order.len() - 1
            } else {
                first_use
            }
        } else {
            consumers.iter()
                .filter_map(|cid| position.get(cid))
                .copied()
                .max()
                .unwrap_or(first_use)
        };

        alloc_infos.push(AllocInfo {
            node_id,
            size,
            live_range: LiveRange(first_use, last_use),
        });
    }

    alloc_infos.sort_by(|a, b| {
        a.live_range.0.cmp(&b.live_range.0)
            .then_with(|| b.size.cmp(&a.size))
    });

    let mut slots: HashMap<NodeId, AllocSlot> = HashMap::new();
    let mut active: Vec<(usize, NodeId, usize)> = Vec::new();
    let mut free_list: Vec<FreeBlock> = Vec::new();
    let mut arena_top: usize = 0;

    for info in &alloc_infos {
        let mut i = 0;
        while i < active.len() {
            if active[i].0 < info.live_range.0 {
                let (_, expired_id, expired_size) = active.swap_remove(i);
                if let Some(slot) = slots.get(&expired_id) {
                    add_to_free_list(&mut free_list, slot.offset, slot.size);
                }
            } else {
                i += 1;
            }
        }

        let offset = find_best_fit(&mut free_list, info.size);

        let offset = match offset {
            Some(off) => off,
            None => {
                let off = arena_top;
                arena_top += info.size;
                off
            }
        };

        slots.insert(info.node_id, AllocSlot {
            offset,
            size: info.size,
            node_id: info.node_id,
        });

        active.push((info.live_range.1, info.node_id, info.size));
    }

    Ok(MemoryPlan {
        total_size: arena_top,
        slots,
    })
}
