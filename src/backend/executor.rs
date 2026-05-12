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
use crate::ir::node::{ComputeGraph, DimExpr, NodeId, Opcode, ShapeEnv};

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
        // Build runtime shape env from input sizes, validate shapes.
        let shape_env = ShapeEnv::from_graph_inputs(graph, inputs);
        validate_shapes(graph, &shape_env).map_err(|e| {
            BackendError::Dispatch(format!("shape validation: {e}"))
        })?;
        // Allocate arena — must use the original plan arena_size because
        // compiled instructions reference the original (max-estimate) slot sizes.
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
        self.backend.dispatch(plan, &arena, &shape_env)?;

        // Read output data from the arena — compute actual sizes using shape_env
        let mut outputs = Vec::with_capacity(graph.outputs.len());
        for &output_node_id in &graph.outputs {
            let slot = memory_plan.slots.get(&output_node_id).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "no memory slot for output node {}",
                    output_node_id
                ))
            })?;

            // Re-evaluate output size with shape_env to handle symbolic dims
            let actual_size = if let Some(node) = graph.get_node(output_node_id) {
                let actual_numel: usize = node
                    .output_type
                    .shape
                    .iter()
                    .map(|d| d.evaluate_with_env(&shape_env).unwrap_or(1) as usize)
                    .product();
                let elem_size = node.output_type.dtype.byte_size();
                (actual_numel * elem_size).min(slot.size)
            } else {
                slot.size
            };

            let data = self.backend.read_arena(&arena, slot.offset, actual_size);
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
/// Uses [`TensorType::byte_size`] under the hood.
pub fn tensor_byte_size(shape: &[DimExpr], elem_byte_size: usize) -> usize {
    let numel: usize = shape
        .iter()
        .map(|d| match d {
            DimExpr::Known(v) => *v as usize,
            DimExpr::Bounded { max, .. } => *max as usize,
            DimExpr::Symbol(_) => crate::ir::node::SYMBOL_DIM_MAX as usize,
        })
        .product();
    numel * elem_byte_size
}

/// Resolve a shape to concrete values using a runtime ShapeEnv.
fn resolve_shape(shape: &[DimExpr], env: &ShapeEnv) -> Vec<u64> {
    shape
        .iter()
        .map(|d| d.evaluate_with_env(env).unwrap_or(1))
        .collect()
}

/// Validate shape constraints for all ops in the graph at runtime.
/// Called after ShapeEnv is built, before dispatch.
fn validate_shapes(graph: &ComputeGraph, shape_env: &ShapeEnv) -> Result<(), String> {
    let order = graph.topological_sort();
    for &node_id in &order {
        let node = graph
            .get_node(node_id)
            .ok_or_else(|| format!("node {} not found during shape validation", node_id))?;

        let input_shapes: Vec<Vec<u64>> = node
            .inputs
            .iter()
            .filter_map(|&id| graph.get_node(id))
            .map(|n| resolve_shape(&n.output_type.shape, shape_env))
            .collect();

        match &node.opcode {
            Opcode::MatMul => {
                if input_shapes.len() < 2 {
                    return Err(format!(
                        "MatMul node {}: expected 2 inputs, got {}",
                        node_id,
                        input_shapes.len()
                    ));
                }
                let a = &input_shapes[0];
                let b = &input_shapes[1];
                if a.len() < 2 || b.len() < 2 {
                    return Err(format!(
                        "MatMul node {}: inputs must have at least 2 dims, got {} and {}",
                        node_id,
                        a.len(),
                        b.len()
                    ));
                }
                let k_a = a[a.len() - 1];
                let k_b = b[b.len() - 2];
                if k_a != k_b {
                    return Err(format!(
                        "MatMul node {}: inner dim mismatch {} vs {} (shape {:?} @ {:?})",
                        node_id, k_a, k_b, a, b
                    ));
                }
            }
            Opcode::ReduceSum | Opcode::ReduceMean => {
                if input_shapes.is_empty() {
                    return Err(format!(
                        "Reduce node {}: expected at least 1 input",
                        node_id
                    ));
                }
                let axis: usize = node
                    .attrs
                    .get("axis")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(0);
                let rank = input_shapes[0].len();
                if axis >= rank {
                    return Err(format!(
                        "Reduce node {}: axis {} out of bounds for rank {} (shape {:?})",
                        node_id, axis, rank, input_shapes[0]
                    ));
                }
            }
            Opcode::Concat => {
                if input_shapes.len() < 2 {
                    return Err(format!(
                        "Concat node {}: expected at least 2 inputs, got {}",
                        node_id,
                        input_shapes.len()
                    ));
                }
                let axis: usize = node
                    .attrs
                    .get("axis")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(0);
                let rank = input_shapes[0].len();
                if axis >= rank {
                    return Err(format!(
                        "Concat node {}: axis {} out of bounds for rank {}",
                        node_id, axis, rank
                    ));
                }
                for (i, s) in input_shapes.iter().enumerate().skip(1) {
                    if s.len() != rank {
                        return Err(format!(
                            "Concat node {}: rank mismatch input 0 ({}) vs input {} ({})",
                            node_id, rank, i, s.len()
                        ));
                    }
                    for j in 0..rank {
                        if j != axis && s[j] != input_shapes[0][j] {
                            return Err(format!(
                                "Concat node {}: dim {} mismatch at axis {}: {} vs {}",
                                node_id, j, axis, s[j], input_shapes[0][j]
                            ));
                        }
                    }
                }
            }
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
                if input_shapes.len() >= 2 {
                    let a = &input_shapes[0];
                    let b = &input_shapes[1];
                    // Check broadcast compatibility: for each trailing dim,
                    // they must match or one must be 1.
                    let max_len = a.len().max(b.len());
                    for i in 0..max_len {
                        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
                        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
                        if da != 1 && db != 1 && da != db {
                            return Err(format!(
                                "Broadcast mismatch node {}: dim {} values {} vs {} (shapes {:?} vs {:?})",
                                node_id, i, da, db, a, b
                            ));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
