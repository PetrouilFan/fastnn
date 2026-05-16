#![allow(clippy::needless_borrowed_reference)]
//! Standalone runtime for executing compiled plans without the full compiler stack.
//!
//! # Usage
//!
//! ```ignore
//! use fastnn::backend::runtime::Runtime;
//! use fastnn::backend::cpu::CpuBackend;
//!
//! // Load a pre-compiled plan and its memory plan
//! let runtime = Runtime::<CpuBackend>::load("model.fnnc", "model.memory.json")?;
//!
//! // Execute with input data (must match the compiled input shapes)
//! let outputs = runtime.run(&[input_bytes])?;
//! ```
//!
//! The runtime skips shape inference, operator fusion, and memory planning —
//! it just loads the plan, maps the arena, and dispatches.

use crate::backend::{Backend, BackendError, ExecutablePlan, MemoryPlan};
use crate::compiler::passes::memory_planning::AllocSlot;
use crate::ir::node::{NodeId, ShapeEnv};

/// A minimal runtime that loads and executes pre-compiled plans.
///
/// Unlike [`GraphExecutor`](crate::backend::executor::GraphExecutor), the
/// `Runtime` does not perform any compilation — it relies on an already-built
/// [`ExecutablePlan`] and [`MemoryPlan`], typically loaded from disk.
pub struct Runtime<B: Backend> {
    backend: B,
    plan: ExecutablePlan,
    memory_plan: MemoryPlan,
    slots_sorted: Vec<(NodeId, AllocSlot)>,
    cached_arena: Option<(usize, B::Buffer)>, // (capacity, buffer)
}

impl<B: Backend> Runtime<B> {
    fn build_sorted_slots(memory_plan: &MemoryPlan) -> Vec<(NodeId, AllocSlot)> {
        let mut slots: Vec<_> = memory_plan
            .slots
            .iter()
            .map(|(&nid, s)| (nid, s.clone()))
            .collect();
        slots.sort_by_key(|&(_, ref s)| s.offset);
        slots
    }

    /// Create a new runtime from an already-loaded plan and memory plan.
    pub fn new(backend: B, plan: ExecutablePlan, memory_plan: MemoryPlan) -> Self {
        let slots_sorted = Self::build_sorted_slots(&memory_plan);
        Runtime {
            backend,
            plan,
            memory_plan,
            slots_sorted,
            cached_arena: None,
        }
    }

    /// Load a plan and memory plan from files saved by the compiler pipeline.
    ///
    /// `plan_path` should point to a `.fnnc` file (bincode-serialized
    /// [`ExecutablePlan`]), and `memory_path` to a `.json` file
    /// (JSON-serialized [`MemoryPlan`]).
    pub fn load(
        backend: B,
        plan_path: &str,
        memory_path: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let plan_bytes = std::fs::read(plan_path)?;
        let plan: ExecutablePlan = bincode::deserialize(&plan_bytes)?;

        let memory_json = std::fs::read_to_string(memory_path)?;
        let memory_plan: MemoryPlan = serde_json::from_str(&memory_json)?;

        let slots_sorted = Self::build_sorted_slots(&memory_plan);
        Ok(Runtime {
            backend,
            plan,
            memory_plan,
            slots_sorted,
            cached_arena: None,
        })
    }

    /// Save the plan and memory plan to files for later use by the runtime.
    ///
    /// Creates a `.fnnc` file (bincode for the plan) and a `.json` file
    /// (for the memory plan).
    pub fn save(
        &self,
        plan_path: &str,
        memory_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let plan_bytes = bincode::serialize(&self.plan)?;
        std::fs::write(plan_path, plan_bytes)?;

        let memory_json = serde_json::to_string_pretty(&self.memory_plan)?;
        std::fs::write(memory_path, memory_json)?;

        Ok(())
    }

    /// Execute the compiled plan against the given input data.
    ///
    /// `inputs` must correspond 1:1 to the graph's input nodes, in the order
    /// they were registered.  The runtime writes each input to the first N
    /// memory slots (by offset) — this assumes inputs are the first slots
    /// allocated by the memory planner, which is the case when input nodes
    /// are the first nodes in topological order.
    ///
    /// Returns the output data for each graph output.
    pub fn run(&mut self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, BackendError> {
        let arena_size = self.plan.arena_size;
        let enough_capacity = self
            .cached_arena
            .as_ref()
            .map_or(false, |(cap, _)| *cap >= arena_size);
        if !enough_capacity {
            self.cached_arena = Some((arena_size, self.backend.allocate_arena(arena_size)));
        }
        let arena_ref = &self.cached_arena.as_ref().unwrap().1;
        if enough_capacity {
            self.backend.write_arena(arena_ref, 0, &vec![0u8; arena_size]);
        }

        // Write inputs into earliest slots (ordering cached at construction).
        for (i, input_bytes) in inputs.iter().enumerate() {
            if let Some((_nid, slot)) = self.slots_sorted.get(i) {
                self.backend
                    .write_arena(arena_ref, slot.offset, input_bytes);
            }
        }

        // Dispatch with an empty shape env.
        // Plans with symbolic dims should be specialized before saving.
        let shape_env = ShapeEnv::new();
        self.backend.dispatch(&self.plan, arena_ref, &shape_env)?;

        // Read only graph output slots (not all intermediate tensors).
        let mut outputs = Vec::with_capacity(self.memory_plan.outputs.len());
        for &node_id in &self.memory_plan.outputs {
            if let Some(slot) = self.memory_plan.slots.get(&node_id) {
                let data = self.backend.read_arena(arena_ref, slot.offset, slot.size);
                outputs.push(data);
            }
        }

        Ok(outputs)
    }

    /// Return a reference to the loaded plan.
    pub fn plan(&self) -> &ExecutablePlan {
        &self.plan
    }

    /// Return a reference to the loaded memory plan.
    pub fn memory_plan(&self) -> &MemoryPlan {
        &self.memory_plan
    }
}
