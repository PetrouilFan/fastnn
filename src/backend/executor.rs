//! AOT graph executor — ties together IR, compiler passes, and backend dispatch.
//!
//! Usage:
//! ```ignore
//! let mut graph = ComputeGraph::new();
//! let input = graph.add_node(Opcode::Constant(...), vec![], ...);
//! let weight = graph.add_node(Opcode::Constant(...), vec![], ...);
//! let mm = graph.add_node(Opcode::MatMul, vec![input, weight], ...);
//! graph.set_inputs(vec![input]);
//! graph.set_outputs(vec![mm]);
//!
//! let executor = GraphExecutor::new(CpuBackend);
//! let outputs = executor.run(&graph, &[input_bytes]).unwrap();
//! ```

#![allow(dead_code)]

use crate::backend::{Backend, BackendError, ExecutablePlan, MemoryPlan};
use crate::compiler::passes::{memory_planning, operator_fusion, shape_inference};
use crate::ir::node::{ComputeGraph, DimExpr, NodeId};

/// An ahead-of-time graph executor that compiles and dispatches
/// computation graphs through the v2.0 backend pipeline.
///
/// Generic over the backend type `B` (e.g. `CpuBackend`).
pub struct GraphExecutor<B: Backend> {
    backend: B,
}

impl<B: Backend> GraphExecutor<B> {
    /// Create a new executor backed by the given backend.
    pub fn new(backend: B) -> Self {
        GraphExecutor { backend }
    }

    /// Return a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Run the full compilation pipeline:
    /// 1. Clone the graph so the original is not mutated
    /// 2. Shape inference
    /// 3. Operator fusion
    /// 4. Memory planning
    /// 5. Backend compilation
    pub fn compile(&self, graph: &ComputeGraph) -> Result<ExecutablePlan, BackendError> {
        let (plan, _memory_plan, _graph) = self.compile_with_plan(graph)?;
        Ok(plan)
    }

    /// Compile and return the `ExecutablePlan`, `MemoryPlan`, and the final
    /// (potentially modified) graph after compiler passes.
    ///
    /// The returned graph reflects any node re-writes from operator fusion,
    /// so callers must use it (rather than the original graph) when calling
    /// [`execute`](Self::execute).
    pub fn compile_with_plan(
        &self,
        graph: &ComputeGraph,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        let mut graph = graph.clone();

        // ── Phase 1: Shape inference ──────────────────────────────────────
        shape_inference::infer_shapes(&mut graph)
            .map_err(|e| BackendError::Compilation(format!("shape inference: {e}")))?;

        // ── Phase 2: Operator fusion ──────────────────────────────────────
        operator_fusion::fuse_operators(&mut graph)
            .map_err(|e| BackendError::Compilation(format!("operator fusion: {e}")))?;

        // ── Phase 3: Memory planning ──────────────────────────────────────
        let memory_plan = memory_planning::plan_memory(&graph)
            .map_err(|e| BackendError::Compilation(format!("memory planning: {e}")))?;

        // ── Phase 4: Backend compilation ──────────────────────────────────
        let plan = self.backend.compile(&graph, &memory_plan)?;

        Ok((plan, memory_plan, graph))
    }

    /// Execute a compiled plan with input data.
    ///
    /// `inputs` must correspond one-to-one with `graph.inputs` (in order).
    /// Returns output byte slices corresponding to `graph.outputs` (in order).
    pub fn execute(
        &self,
        graph: &ComputeGraph,
        plan: &ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        // Allocate the arena
        let arena = self.backend.allocate_arena(plan.arena_size);

        // Write input data into the arena at the slots for graph input nodes
        for (i, &input_node_id) in graph.inputs.iter().enumerate() {
            let input_bytes = inputs.get(i).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "missing input {} for node {}",
                    i, input_node_id
                ))
            })?;

            let slot = memory_plan.slots.get(&input_node_id).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "no memory slot for input node {}",
                    input_node_id
                ))
            })?;

            self.backend
                .write_arena(&arena, slot.offset, input_bytes);
        }

        // Dispatch the plan
        self.backend.dispatch(plan, &arena)?;

        // Read output data from the arena
        let mut outputs = Vec::with_capacity(graph.outputs.len());
        for &output_node_id in &graph.outputs {
            let slot = memory_plan.slots.get(&output_node_id).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "no memory slot for output node {}",
                    output_node_id
                ))
            })?;

            let data = self.backend.read_arena(&arena, slot.offset, slot.size);
            outputs.push(data);
        }

        Ok(outputs)
    }

    /// Convenience: compile + execute in a single call.
    pub fn run(
        &self,
        graph: &ComputeGraph,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let (plan, memory_plan, compiled_graph) = self.compile_with_plan(graph)?;
        self.execute(&compiled_graph, &plan, &memory_plan, inputs)
    }
}

/// Compute the byte size of a tensor described by shape dims and element size.
/// Useful for constructing ComputeGraphs with known buffer sizes.
pub fn tensor_byte_size(shape: &[DimExpr], elem_byte_size: usize) -> usize {
    let numel: usize = shape
        .iter()
        .map(|d| match d {
            DimExpr::Known(v) => *v as usize,
            DimExpr::Bounded { max, .. } => *max as usize,
            DimExpr::Symbol(_) => 4096,
        })
        .product();
    numel * elem_byte_size
}
