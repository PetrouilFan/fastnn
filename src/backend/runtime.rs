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
use crate::ir::node::ShapeEnv;

/// A minimal runtime that loads and executes pre-compiled plans.
///
/// Unlike [`GraphExecutor`](crate::backend::executor::GraphExecutor), the
/// `Runtime` does not perform any compilation — it relies on an already-built
/// [`ExecutablePlan`] and [`MemoryPlan`], typically loaded from disk.
pub struct Runtime<B: Backend> {
    backend: B,
    plan: ExecutablePlan,
    memory_plan: MemoryPlan,
}

impl<B: Backend> Runtime<B> {
    /// Create a new runtime from an already-loaded plan and memory plan.
    pub fn new(backend: B, plan: ExecutablePlan, memory_plan: MemoryPlan) -> Self {
        Runtime {
            backend,
            plan,
            memory_plan,
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

        Ok(Runtime {
            backend,
            plan,
            memory_plan,
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
    pub fn run(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, BackendError> {
        let arena = self.backend.allocate_arena(self.plan.arena_size);
        let arena_ref = &arena;

        // Collect slots sorted by offset — inputs are typically the first slots.
        let mut sorted_slots: Vec<_> = self.memory_plan.slots.iter().collect();
        sorted_slots.sort_by_key(|(_, s)| s.offset);

        // Write inputs into the earliest slots (assumes inputs are the first
        // nodes in topological order, which the memory planner schedules first).
        for (i, input_bytes) in inputs.iter().enumerate() {
            if let Some((_node_id, slot)) = sorted_slots.get(i) {
                self.backend
                    .write_arena(arena_ref, slot.offset, input_bytes);
            }
        }

        // Dispatch with an empty shape env.
        // Plans with symbolic dims should be specialized before saving.
        let shape_env = ShapeEnv::new();
        self.backend.dispatch(&self.plan, arena_ref, &shape_env)?;

        // Read output slot data (sorted by offset for deterministic order).
        let mut outputs = Vec::with_capacity(sorted_slots.len());
        for (_node_id, slot) in &sorted_slots {
            let data = self.backend.read_arena(arena_ref, slot.offset, slot.size);
            outputs.push(data);
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
