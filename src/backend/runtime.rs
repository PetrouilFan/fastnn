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

use crate::backend::executor::ExecutionResourceLimits;
use crate::backend::{deserialize_executable_plan, Backend, BackendError, ExecutablePlan};
use crate::compiler::MemoryPlan;
use crate::ir::ShapeEnv;

/// A minimal runtime that loads and executes pre-compiled plans.
///
/// Unlike [`GraphExecutor`](crate::backend::executor::GraphExecutor), the
/// `Runtime` does not perform any compilation — it relies on an already-built
/// [`ExecutablePlan`] and [`MemoryPlan`], typically loaded from disk.
pub struct Runtime<B: Backend> {
    backend: B,
    plan: ExecutablePlan,
    memory_plan: MemoryPlan,
    resource_limits: ExecutionResourceLimits,
    cached_arena: Option<(usize, B::Buffer)>, // (capacity, buffer)
}

impl<B: Backend> Runtime<B> {
    /// Create a new runtime from an already-loaded plan and memory plan.
    pub fn new(
        backend: B,
        plan: ExecutablePlan,
        memory_plan: MemoryPlan,
    ) -> Result<Self, BackendError> {
        Self::new_with_resource_limits(
            backend,
            plan,
            memory_plan,
            ExecutionResourceLimits::default(),
        )
    }

    pub fn new_with_resource_limits(
        backend: B,
        plan: ExecutablePlan,
        memory_plan: MemoryPlan,
        resource_limits: ExecutionResourceLimits,
    ) -> Result<Self, BackendError> {
        plan.validate_with_limits(&resource_limits.executable)?;
        memory_plan
            .validate_with_limits(&resource_limits.memory)
            .map_err(|error| BackendError::Dispatch(format!("memory plan: {error}")))?;
        super::executor::validate_instruction_slice_ownership(&plan, &memory_plan)?;
        if memory_plan.total_size > plan.arena_size {
            return Err(BackendError::Dispatch(format!(
                "memory plan arena size {} exceeds executable arena size {}",
                memory_plan.total_size, plan.arena_size
            )));
        }
        Ok(Runtime {
            backend,
            plan,
            memory_plan,
            resource_limits,
            cached_arena: None,
        })
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
        Self::load_with_resource_limits(
            backend,
            plan_path,
            memory_path,
            ExecutionResourceLimits::default(),
        )
    }

    pub fn load_with_resource_limits(
        backend: B,
        plan_path: &str,
        memory_path: &str,
        resource_limits: ExecutionResourceLimits,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let plan_file_size = std::fs::metadata(plan_path)?.len();
        if plan_file_size > resource_limits.executable.max_serialized_bytes {
            return Err(format!(
                "serialized executable plan has {plan_file_size} bytes, exceeding limit {}",
                resource_limits.executable.max_serialized_bytes
            )
            .into());
        }
        let plan_bytes = std::fs::read(plan_path)?;
        let plan = deserialize_executable_plan(&plan_bytes, &resource_limits.executable)?;

        let memory_file_size = std::fs::metadata(memory_path)?.len();
        if memory_file_size > resource_limits.memory.max_serialized_bytes {
            return Err(format!(
                "serialized memory plan has {memory_file_size} bytes, exceeding limit {}",
                resource_limits.memory.max_serialized_bytes
            )
            .into());
        }
        let memory_json = std::fs::read_to_string(memory_path)?;
        let memory_plan: MemoryPlan = serde_json::from_str(&memory_json)?;

        Ok(Self::new_with_resource_limits(
            backend,
            plan,
            memory_plan,
            resource_limits,
        )?)
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
        self.plan
            .validate_with_limits(&self.resource_limits.executable)?;
        self.memory_plan
            .validate_with_limits(&self.resource_limits.memory)?;
        super::executor::validate_instruction_slice_ownership(&self.plan, &self.memory_plan)?;
        let plan_bytes = self
            .plan
            .to_bytes_with_limits(&self.resource_limits.executable)?;
        let memory_json = serde_json::to_string_pretty(&self.memory_plan)?;
        let memory_size = u64::try_from(memory_json.len())?;
        if memory_size > self.resource_limits.memory.max_serialized_bytes {
            return Err(format!(
                "serialized memory plan would have {memory_size} bytes, exceeding limit {}",
                self.resource_limits.memory.max_serialized_bytes
            )
            .into());
        }

        std::fs::write(plan_path, plan_bytes)?;
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
        self.plan
            .validate_with_limits(&self.resource_limits.executable)?;
        self.memory_plan
            .validate_with_limits(&self.resource_limits.memory)
            .map_err(|error| BackendError::Dispatch(format!("memory plan: {error}")))?;
        super::executor::validate_instruction_slice_ownership(&self.plan, &self.memory_plan)?;
        if inputs.len() != self.memory_plan.inputs.len() {
            return Err(BackendError::Dispatch(format!(
                "runtime expected {} inputs, received {}",
                self.memory_plan.inputs.len(),
                inputs.len()
            )));
        }
        for (input_index, (&node_id, input_bytes)) in self
            .memory_plan
            .inputs
            .iter()
            .zip(inputs.iter())
            .enumerate()
        {
            let slot = self.memory_plan.slots.get(&node_id).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "runtime input {input_index} node {node_id} has no slot"
                ))
            })?;
            if input_bytes.len() != slot.size {
                return Err(BackendError::Dispatch(format!(
                    "runtime input {input_index} has {} bytes but its slot requires {}",
                    input_bytes.len(),
                    slot.size
                )));
            }
        }
        let arena_size = self.plan.arena_size;
        let enough_capacity = self
            .cached_arena
            .as_ref()
            .is_some_and(|(cap, _)| *cap >= arena_size);
        if !enough_capacity {
            self.cached_arena = Some((arena_size, self.backend.try_allocate_arena(arena_size)?));
        }
        let arena_ref = &self
            .cached_arena
            .as_ref()
            .ok_or_else(|| {
                BackendError::Dispatch("runtime arena allocation returned no buffer".into())
            })?
            .1;
        if enough_capacity {
            let mut zeroed = Vec::new();
            zeroed.try_reserve_exact(arena_size).map_err(|error| {
                BackendError::Dispatch(format!(
                    "failed to allocate {arena_size}-byte arena reset buffer: {error}"
                ))
            })?;
            zeroed.resize(arena_size, 0);
            self.backend.try_write_arena(arena_ref, 0, &zeroed)?;
        }

        for (input_index, (&node_id, input_bytes)) in self
            .memory_plan
            .inputs
            .iter()
            .zip(inputs.iter())
            .enumerate()
        {
            let slot = self.memory_plan.slots.get(&node_id).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "runtime input {input_index} node {node_id} has no slot"
                ))
            })?;
            self.backend
                .try_write_arena(arena_ref, slot.offset, input_bytes)?;
        }

        // Dispatch with an empty shape env.
        // Plans with symbolic dims should be specialized before saving.
        let shape_env = ShapeEnv::new();
        self.backend.dispatch(&self.plan, arena_ref, &shape_env)?;

        // Read only graph output slots (not all intermediate tensors).
        let mut outputs = Vec::with_capacity(self.memory_plan.outputs.len());
        for &node_id in &self.memory_plan.outputs {
            let slot = self.memory_plan.slots.get(&node_id).ok_or_else(|| {
                BackendError::Dispatch(format!("runtime output node {node_id} has no slot"))
            })?;
            let data = self
                .backend
                .try_read_arena(arena_ref, slot.offset, slot.size)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::ir::builder::GraphBuilder;
    use crate::ir::IrDType;

    #[test]
    fn runtime_uses_declared_inputs_and_validates_payloads() {
        let builder = GraphBuilder::new();
        let input = builder.input(&[1, 4], IrDType::F32);
        let output = builder.relu(&input);
        let (plan, memory_plan, _) = builder
            .compile(&[&output], CpuBackend)
            .expect("compile must succeed");
        let mut runtime = Runtime::new(CpuBackend, plan, memory_plan)
            .expect("valid plans must construct a runtime");
        let input_bytes: Vec<u8> = [1.0f32, -2.0, 3.0, -4.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert!(runtime.run(&[]).is_err());
        assert!(
            runtime.cached_arena.is_none(),
            "invalid inputs must be rejected before arena allocation"
        );
        let outputs = runtime.run(&[&input_bytes]).expect("runtime must execute");
        assert_eq!(outputs.len(), 1);

        let undersized = vec![0u8; input_bytes.len() - 1];
        assert!(runtime.run(&[&undersized]).is_err());
        let oversized = vec![0u8; input_bytes.len() + 1];
        assert!(runtime.run(&[&oversized]).is_err());
    }

    #[test]
    fn runtime_rejects_inconsistent_memory_metadata() {
        let plan = ExecutablePlan {
            instructions: vec![],
            arena_size: 4,
            levels: vec![],
        };
        let memory_plan = MemoryPlan {
            total_size: 4,
            slots: std::collections::HashMap::new(),
            inputs: vec![1],
            secondary_slots: std::collections::HashMap::new(),
            outputs: vec![],
            tightened_params: std::collections::HashMap::new(),
        };
        assert!(Runtime::new(CpuBackend, plan, memory_plan).is_err());
    }

    #[test]
    fn runtime_rejects_instruction_slices_without_memory_ownership() {
        let plan = ExecutablePlan {
            instructions: vec![crate::backend::Instruction::Fill {
                dst: crate::backend::BufferSlice::new(4, 4),
                value: 0.0,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let memory_plan = MemoryPlan {
            total_size: 8,
            slots: std::collections::HashMap::from([(
                1,
                crate::compiler::AllocSlot {
                    offset: 0,
                    size: 4,
                    node_id: 1,
                    output_index: 0,
                },
            )]),
            inputs: vec![],
            secondary_slots: std::collections::HashMap::new(),
            outputs: vec![],
            tightened_params: std::collections::HashMap::new(),
        };
        assert!(Runtime::new(CpuBackend, plan, memory_plan).is_err());
    }

    #[test]
    fn runtime_enforces_configured_resource_limits() {
        let builder = GraphBuilder::new();
        let input = builder.input(&[1, 4], IrDType::F32);
        let output = builder.relu(&input);
        let (plan, memory_plan, _) = builder
            .compile(&[&output], CpuBackend)
            .expect("compile must succeed");
        let limits = ExecutionResourceLimits {
            executable: crate::backend::PlanResourceLimits {
                max_instructions: 0,
                ..crate::backend::PlanResourceLimits::default()
            },
            ..ExecutionResourceLimits::default()
        };

        assert!(Runtime::new_with_resource_limits(CpuBackend, plan, memory_plan, limits).is_err());
    }
}
