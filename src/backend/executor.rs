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
use crate::compiler::passes::{memory_planning, operator_fusion, quantization, shape_inference};
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, Opcode, ShapeEnv};
use std::sync::atomic::Ordering;

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
    ///
    /// If `quantize` is `Some(bit_width)`, the quantization pass is applied
    /// after operator fusion and before memory planning. Valid values are
    /// `4` (U4x8) and `8` (U8x4).
    pub fn compile_with_plan(
        &self,
        graph: &ComputeGraph,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        self.compile_with_plan_and_quantize(graph, None)
    }

    /// Same as [`compile_with_plan`] but with optional weight quantization.
    ///
    /// Pass `Some(4)` or `Some(8)` to quantize f32 weight constants to
    /// packed 4-bit or 8-bit precision.
    pub fn compile_with_plan_and_quantize(
        &self,
        graph: &ComputeGraph,
        quantize: Option<u8>,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        let mut graph = graph.clone();

        // ── Phase 1: Shape inference ──────────────────────────────────────
        shape_inference::infer_shapes(&mut graph)
            .map_err(|e| BackendError::Compilation(format!("shape inference: {e}")))?;

        // ── Phase 2: Operator fusion ──────────────────────────────────────
        operator_fusion::fuse_operators(&mut graph)
            .map_err(|e| BackendError::Compilation(format!("operator fusion: {e}")))?;

        // ── Phase 2.5: Quantization (optional) ───────────────────────────
        if let Some(bit_width) = quantize {
            if bit_width != 4 && bit_width != 8 {
                return Err(BackendError::Compilation(format!(
                    "unsupported quantization bit width: {} (expected 4 or 8)",
                    bit_width
                )));
            }
            quantization::quantize_weights(&mut graph, bit_width)
                .map_err(|e| BackendError::Compilation(format!("quantization: {e}")))?;

            // After quantizing weights, wrap any optimizer ops that now have
            // quantized weight inputs with Dequantize/Quantize.
            quantization::wrap_quantized_optimizer(&mut graph)
                .map_err(|e| BackendError::Compilation(format!("optimizer wrapping: {e}")))?;
        }

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
        let shape_env = ShapeEnv::from_graph_inputs(graph, inputs).map_err(|e| {
            BackendError::Dispatch(format!("shape env: {e}"))
        })?;
        // Validate shape constraints using the resolved environment
        validate_shapes(graph, &shape_env).map_err(|e| {
            BackendError::Dispatch(format!("shape validation: {e}"))
        })?;
        // Pre-dispatch slot overflow check: ensure every node's resolved output
        // byte size fits within its allocated memory slot. This catches
        // intermediate-tensor overruns before any kernel writes to the arena.
        for (&node_id, slot) in &memory_plan.slots {
            if let Some(node) = graph.get_node(node_id) {
                let _resolved = tensor_byte_size(&node.output_type.shape, node.output_type.dtype.byte_size());
                // Use compile-time max estimate for Symbol dims (already the
                // slot size), but check that resolved Known/Bounded dims fit.
                let needed = if let Ok(shape) = resolve_shape(&node.output_type.shape, &shape_env) {
                    let numel: u64 = shape.iter().product();
                    numel as usize * node.output_type.dtype.byte_size()
                } else {
                    // Cannot resolve – skip runtime check (compile-time
                    // slot sizing is the safety net).
                    continue;
                };
                if needed > slot.size {
                    return Err(BackendError::Dispatch(format!(
                        "node {}: resolved output size {} exceeds slot size {} (shape {:?})",
                        node_id, needed, slot.size, node.output_type.shape
                    )));
                }
            }
        }
        // Allocate arena.  The plan was compiled with worst-case SYMBOL_DIM_MAX
        // sizes; at runtime the ShapeEnv resolves symbolic dims to actual values.
        // TODO: invoke memory_plan.tighten() here and rebuild the ExecutablePlan's
        // instruction BufferSlices to match the tightened sizes (saves up to 90%
        // arena memory for dynamic shapes).
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

            // Re-evaluate output size with shape_env to handle symbolic dims.
            let actual_size = if let Some(node) = graph.get_node(output_node_id) {
                let resolved_shape = resolve_shape(&node.output_type.shape, &shape_env)
                    .map_err(|e| BackendError::Dispatch(format!(
                        "output node {}: {e}", output_node_id
                    )))?;
                let actual_numel: usize = resolved_shape.iter().map(|&v| v as usize).product();
                // For packed dtypes (U4/U8), use packed_byte_size which accounts
                // for the header + SIMD margin. For standard dtypes, use numel * byte_size.
                let computed = match &node.output_type.dtype {
                    IrDType::U4 { .. } | IrDType::U8 { .. } => {
                        node.output_type.dtype.packed_byte_size(actual_numel)
                    }
                    _ => actual_numel * node.output_type.dtype.byte_size(),
                };
                if computed > slot.size {
                    return Err(BackendError::Dispatch(format!(
                        "output node {}: resolved size {} exceeds allocated slot size {} \
                         (shape {:?}, env may have overflowed SYMBOL_DIM_MAX={})",
                        output_node_id, computed, slot.size,
                        node.output_type.shape,
                        crate::ir::node::SYMBOL_DIM_MAX.load(Ordering::Relaxed),
                    )));
                }
                computed
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
            DimExpr::Symbol(_) => crate::ir::node::SYMBOL_DIM_MAX.load(Ordering::Relaxed) as usize,
        })
        .product();
    numel * elem_byte_size
}

/// Resolve a shape to concrete values using a runtime ShapeEnv.
/// Returns an error if any symbolic dimension cannot be resolved.
fn resolve_shape(shape: &[DimExpr], env: &ShapeEnv) -> Result<Vec<u64>, String> {
    shape
        .iter()
        .map(|d| {
            d.evaluate_with_env(env)
                .map_err(|e| format!("resolve_shape: {e}"))
        })
        .collect()
}

/// Lenient shape resolution: unbound Symbol dims are replaced with
/// [`SYMBOL_DIM_MAX`] instead of failing, so that validation checks
/// can still run in the presence of symbols that will be resolved at
/// runtime (e.g. the -1 dimension of a Reshape).
fn resolve_shape_lenient(shape: &[DimExpr], env: &ShapeEnv) -> Vec<u64> {
    let symbol_max = crate::ir::node::SYMBOL_DIM_MAX.load(std::sync::atomic::Ordering::Relaxed);
    shape
        .iter()
        .map(|d| d.evaluate_with_env(env).unwrap_or(symbol_max))
        .collect()
}

/// Parse an axis attribute, supporting negative values (ONNX-style).
/// Returns a 0-based positive axis, or an error if out of range.
fn resolve_axis(attrs: &std::collections::HashMap<String, String>, key: &str, rank: usize) -> Result<usize, String> {
    let raw: i64 = attrs.get(key)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let axis = if raw < 0 {
        let r = rank as i64;
        if raw < -r {
            return Err(format!("axis {} out of bounds for rank {}", raw, rank));
        }
        (r + raw) as usize
    } else {
        raw as usize
    };
    if axis >= rank {
        return Err(format!("axis {} out of bounds for rank {}", axis, rank));
    }
    Ok(axis)
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
            .map(|n| resolve_shape_lenient(&n.output_type.shape, shape_env))
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
                let rank = input_shapes[0].len();
                let _axis = resolve_axis(&node.attrs, "axis", rank)
                    .map_err(|e| format!("Reduce node {}: {e}", node_id))?;
            }
            Opcode::Concat => {
                if input_shapes.len() < 2 {
                    return Err(format!(
                        "Concat node {}: expected at least 2 inputs, got {}",
                        node_id,
                        input_shapes.len()
                    ));
                }
                let rank = input_shapes[0].len();
                let axis = resolve_axis(&node.attrs, "axis", rank)
                    .map_err(|e| format!("Concat node {}: {e}", node_id))?;
                // Skip dim compatibility checks if any input contains
                // unresolved Symbol dims (resolved to SYMBOL_DIM_MAX).
                let symbol_max = crate::ir::node::SYMBOL_DIM_MAX.load(std::sync::atomic::Ordering::Relaxed);
                let has_unresolved = input_shapes.iter().any(|s| s.contains(&symbol_max));
                if !has_unresolved {
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
            }
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
                if input_shapes.len() < 2 {
                    return Err(format!(
                        "Elementwise node {}: expected at least 2 inputs, got {}",
                        node_id, input_shapes.len()
                    ));
                }
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
            Opcode::Conv2d => {
                if input_shapes.len() < 2 {
                    return Err(format!("Conv2d node {}: expected 2 inputs, got {}", node_id, input_shapes.len()));
                }
                let inp = &input_shapes[0];
                let weight = &input_shapes[1];
                if inp.len() < 4 {
                    return Err(format!("Conv2d node {}: input must have 4 dims [N,C,H,W], got {:?}", node_id, inp));
                }
                if weight.len() < 4 {
                    return Err(format!("Conv2d node {}: weight must have 4 dims [F,C,KH,KW], got {:?}", node_id, weight));
                }
                let groups: u64 = node.attrs.get("groups")
                    .and_then(|g| g.parse().ok())
                    .unwrap_or(1);
                if inp[1] != weight[1] * groups {
                    return Err(format!(
                        "Conv2d node {}: input channels {} != weight channels {} * groups {} (groups={})",
                        node_id, inp[1], weight[1], weight[1] * groups, groups
                    ));
                }
            }
            Opcode::Transpose => {
                if input_shapes.is_empty() || input_shapes[0].len() < 2 {
                    return Err(format!("Transpose node {}: input must have at least 2 dims, got {:?}", node_id, input_shapes.first()));
                }
                // Validate optional permutation attribute
                if let Some(perm_str) = node.attrs.get("perm") {
                    let perm: Vec<usize> = perm_str.trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if perm.len() != input_shapes[0].len() {
                        return Err(format!(
                            "Transpose node {}: permutation length {} != rank {}",
                            node_id, perm.len(), input_shapes[0].len()
                        ));
                    }
                    let mut seen = vec![false; perm.len()];
                    for &p in &perm {
                        if p >= perm.len() {
                            return Err(format!(
                                "Transpose node {}: permutation axis {} out of bounds for rank {}",
                                node_id, p, perm.len()
                            ));
                        }
                        if seen[p] {
                            return Err(format!(
                                "Transpose node {}: duplicate axis {} in permutation {:?}",
                                node_id, p, perm
                            ));
                        }
                        seen[p] = true;
                    }
                }
            }
            Opcode::Reshape | Opcode::Flatten => {
                if input_shapes.is_empty() { continue; }
                let in_numel: u64 = input_shapes[0].iter().product();
                // For reshape, compute expected numel from attrs or output shape
                if let Some(out_shape) = node.attrs.get("shape") {
                    // Split the shape attr into tokens. Each token is either a
                    // concrete u64 dim ("42") or a symbolic dim ("N", "-1", etc.)
                    // that will be resolved at runtime.
                    let tokens: Vec<&str> = out_shape.trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    // Collect only the concrete (u64-parseable) dims.
                    let concrete: Vec<u64> = tokens.iter()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    let symbol_count = tokens.len() - concrete.len();
                    if symbol_count == 0 {
                        // All dims are concrete: the total element count must
                        // match exactly.
                        let out_numel: u64 = concrete.iter().product();
                        if in_numel != out_numel {
                            return Err(format!(
                                "Reshape node {}: element count mismatch {} vs {} (in {:?} -> {:?})",
                                node_id, in_numel, out_numel, input_shapes[0], concrete
                            ));
                        }
                    } else if symbol_count == 1 {
                        // Exactly one symbolic dim: the runtime will compute it
                        // as in_numel / product(concrete_dims).  Validate that
                        // the division is exact.
                        let known_product: u64 = concrete.iter().product();
                        if known_product == 0 || in_numel % known_product != 0 {
                            return Err(format!(
                                "Reshape node {}: element count {} not divisible by known dims product {} (shape={:?})",
                                node_id, in_numel, known_product, tokens
                            ));
                        }
                    } // Multiple symbolic dims: cannot validate statically, skip.
                }
            }
            Opcode::Softmax if !input_shapes.is_empty() => {
                let rank = input_shapes[0].len();
                let _ = resolve_axis(&node.attrs, "axis", rank)
                    .map_err(|e| format!("Softmax node {}: {e}", node_id))?;
            }
            Opcode::BatchNorm | Opcode::LayerNorm => {
                if input_shapes.len() < 2 {
                    return Err(format!("Norm node {}: expected at least 2 inputs (data + weight)", node_id));
                }
                // Channel dim must match between data and weight for 1D/2D norm
                let data = &input_shapes[0];
                let w = &input_shapes[1];
                if data.len() >= 2 && !w.is_empty() && data[1] != w[0] {
                    return Err(format!(
                        "Norm node {}: data channels {} != weight features {}",
                        node_id, data[1], w[0]
                    ));
                }
            }
            Opcode::Slice => {
                if input_shapes.is_empty() { continue; }
                let dim: usize = node.attrs.get("dim").and_then(|a| a.parse().ok()).unwrap_or(0);
                let start: i64 = node.attrs.get("start").and_then(|a| a.parse().ok()).unwrap_or(0);
                let end: i64 = node.attrs.get("end").and_then(|a| a.parse().ok()).unwrap_or(-1);
                let rank = input_shapes[0].len();
                if dim >= rank {
                    return Err(format!("Slice node {}: dim {} out of bounds for rank {}", node_id, dim, rank));
                }
                let dim_size = input_shapes[0][dim] as i64;
                let adjusted_end = if end < 0 { dim_size + end + 1 } else { end };
                if start >= dim_size || adjusted_end > dim_size || start >= adjusted_end {
                    return Err(format!(
                        "Slice node {}: invalid range [{}, {}) on dim {} with size {}",
                        node_id, start, adjusted_end, dim, dim_size
                    ));
                }
            }
            Opcode::MaxPool | Opcode::AvgPool => {
                if input_shapes.is_empty() { continue; }
                if input_shapes[0].len() < 4 {
                    return Err(format!("Pool node {}: input must have 4 dims [N,C,H,W], got {:?}", node_id, input_shapes[0]));
                }
                let k: u64 = node.attrs.get("kernel_size").and_then(|a| a.parse().ok()).unwrap_or(2);
                let s: u64 = node.attrs.get("stride").and_then(|a| a.parse().ok()).unwrap_or(2);
                if k == 0 || s == 0 {
                    return Err(format!("Pool node {}: kernel_size={} and stride={} must be >0", node_id, k, s));
                }
            }
            Opcode::Squeeze => {
                if input_shapes.is_empty() { continue; }
                let dim: usize = node.attrs.get("dim").and_then(|a| a.parse().ok()).unwrap_or(0);
                if dim >= input_shapes[0].len() {
                    return Err(format!("Squeeze node {}: dim {} out of bounds for rank {}", node_id, dim, input_shapes[0].len()));
                }
                if input_shapes[0][dim] != 1 {
                    return Err(format!("Squeeze node {}: dim {} has size {} (must be 1 to squeeze)", node_id, dim, input_shapes[0][dim]));
                }
            }
            Opcode::Unsqueeze => {
                // Unsqueeze attrs store the output dim; just check it's reasonable
                let dim: usize = node.attrs.get("dim").and_then(|a| a.parse().ok()).unwrap_or(0);
                let max_rank = input_shapes.first().map(|s| s.len() + 1).unwrap_or(1);
                if dim > max_rank {
                    return Err(format!("Unsqueeze node {}: dim {} out of bounds for rank {}", node_id, dim, max_rank));
                }
            }
            Opcode::Gather => {
                if input_shapes.len() < 2 {
                    return Err(format!("Gather node {}: expected at least 2 inputs (data + indices), got {}", node_id, input_shapes.len()));
                }
                let rank = input_shapes[0].len();
                let _axis = resolve_axis(&node.attrs, "axis", rank)
                    .map_err(|e| format!("Gather node {}: {e}", node_id))?;
            }
            Opcode::Pad => {
                if input_shapes.is_empty() { continue; }
                let rank = input_shapes[0].len();
                if let Some(pads) = node.attrs.get("pads") {
                    let parsed: Vec<i64> = pads.trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parsed.len() != 2 * rank {
                        return Err(format!(
                            "Pad node {}: pads length {} != 2*rank {}",
                            node_id, parsed.len(), 2 * rank
                        ));
                    }
                }
            }
            Opcode::Neg | Opcode::Abs | Opcode::Exp | Opcode::Log
            | Opcode::Sqrt | Opcode::Relu | Opcode::Gelu | Opcode::Silu
            | Opcode::Sigmoid | Opcode::Tanh
                if input_shapes.len() != 1 =>
            {
                return Err(format!(
                    "Unary op {:?} node {}: expected 1 input, got {}",
                    node.opcode, node_id, input_shapes.len()
                ));
            }
            _ => {}
        }
    }
    Ok(())
}
