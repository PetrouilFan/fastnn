#![allow(clippy::for_kv_map, clippy::redundant_closure)]
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

use crate::autograd::build_backward_graph;
use crate::backend::{Backend, BackendError, ExecutablePlan, Instruction, ProfileEntry};
use crate::compiler::passes::calibration;
use crate::compiler::passes::training::{inject_optimizer, TrainConfig};
use crate::compiler::MemoryPlan;
use crate::ir::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, ShapeEnv};
use crate::types::{CompileTarget, QuantTarget};
use std::collections::HashMap;

/// Cached state for an all-Known-shape model so that every forward call
/// after the first can skip ShapeEnv resolution, shape validation, and
/// memory-plan tightening.
struct StaticShapeCache {
    tightened_memory_plan: MemoryPlan,
    shape_env: ShapeEnv,
    filtered_plan: ExecutablePlan,
}

/// An ahead-of-time graph executor that compiles and dispatches
/// computation graphs through the v2.0 backend pipeline.
///
/// Generic over the backend type `B` (e.g. `CpuBackend`).
pub struct GraphExecutor<B: Backend> {
    backend: B,
    cached_arena: Option<(usize, B::Buffer)>, // (capacity, buffer)
    /// Populated after the first inference when `graph.has_static_shapes()`
    /// is true. Subsequent calls skip ShapeEnv/tighten/tighten_slices.
    static_shape_cache: Option<StaticShapeCache>,
}

/// Read output tensors from the arena using the tightened memory plan
/// and runtime-resolved shapes.
fn read_execution_outputs<B: Backend>(
    graph: &ComputeGraph,
    tightened_memory_plan: &MemoryPlan,
    shape_env: &ShapeEnv,
    backend: &B,
    arena: &B::Buffer,
) -> Result<Vec<Vec<u8>>, BackendError> {
    let mut outputs = Vec::with_capacity(graph.outputs.len());
    for &output_node_id in graph.outputs.iter() {
        let slot = tightened_memory_plan
            .slots
            .get(&output_node_id)
            .ok_or_else(|| {
                BackendError::Dispatch(format!("no memory slot for output node {}", output_node_id))
            })?;
        let (actual_size, _resolved_shape) = if let Some(node) = graph.get_node(output_node_id) {
            let resolved_shape =
                resolve_shape(&node.output_type.shape, shape_env).map_err(|e| {
                    BackendError::Dispatch(format!("output node {}: {e}", output_node_id))
                })?;
            let actual_numel = resolved_shape
                .iter()
                .try_fold(1usize, |count, &dimension| {
                    let dimension = usize::try_from(dimension).map_err(|_| {
                        BackendError::Dispatch(format!(
                            "output node {output_node_id} dimension does not fit usize"
                        ))
                    })?;
                    count.checked_mul(dimension).ok_or_else(|| {
                        BackendError::Dispatch(format!(
                            "output node {output_node_id} element count overflows"
                        ))
                    })
                })?;
            let computed = checked_execution_storage_size(&node.output_type.dtype, actual_numel)?;
            (computed, resolved_shape)
        } else {
            (slot.size, vec![])
        };
        if actual_size > slot.size {
            return Err(BackendError::Dispatch(format!(
                "output node {output_node_id} requires {actual_size} bytes but its slot holds {}",
                slot.size
            )));
        }
        let output_end = slot.offset.checked_add(actual_size).ok_or_else(|| {
            BackendError::Dispatch(format!(
                "output node {output_node_id} arena range overflows"
            ))
        })?;
        if output_end > tightened_memory_plan.total_size {
            return Err(BackendError::Dispatch(format!(
                "output node {output_node_id} range {}..{output_end} exceeds arena size {}",
                slot.offset, tightened_memory_plan.total_size
            )));
        }
        let data = backend.try_read_arena(arena, slot.offset, actual_size)?;
        outputs.push(data);
    }
    Ok(outputs)
}

fn checked_execution_storage_size(dtype: &IrDType, numel: usize) -> Result<usize, BackendError> {
    if matches!(dtype, IrDType::I8) {
        return numel.checked_add(8).ok_or_else(|| {
            BackendError::Dispatch("I8 execution storage size overflows usize".into())
        });
    }
    dtype
        .try_packed_byte_size(numel)
        .ok_or_else(|| BackendError::Dispatch("execution storage size overflows usize".into()))
}

impl<B: Backend> GraphExecutor<B> {
    /// Create a new executor backed by the given backend.
    pub fn new(backend: B) -> Self {
        GraphExecutor {
            backend,
            cached_arena: None,
            static_shape_cache: None,
        }
    }

    /// Return a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Run the full compilation pipeline:
    /// 1. Shape inference
    /// 2. Operator fusion
    /// 3. Memory planning
    /// 4. Backend compilation
    ///
    /// Takes ownership of the graph to avoid an unnecessary deep clone.
    pub fn compile(&self, graph: ComputeGraph) -> Result<ExecutablePlan, BackendError> {
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
    /// `4` (I4x8) and `8` (I8x4).
    pub fn compile_with_plan(
        &self,
        graph: ComputeGraph,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        self.compile_with_plan_and_quantize(graph, None, None)
    }

    /// Same as [`compile_with_plan`] but with optional weight quantization and
    /// optional calibration data for activation quantization.
    ///
    /// Pass `Some(4)` or `Some(8)` to quantize f32 weight constants to
    /// packed 4-bit or 8-bit precision. If `calib_data` is provided, it will
    /// be used to compute per-tensor/per-channel activation scales for optimal
    /// quantization accuracy.
    ///
    /// For FP packed types (F8/F8R/F4), use [`compile_with_target`]
    /// instead. This method only supports integer quantization (I4/I8).
    pub fn compile_with_plan_and_quantize(
        &self,
        graph: ComputeGraph,
        quantize: Option<u8>,
        calib_data: Option<calibration::CalibrationData>,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        let target = match quantize {
            Some(4) => CompileTarget::WeightOnly(QuantTarget::I4),
            Some(8) => CompileTarget::WeightOnly(QuantTarget::I8),
            None => CompileTarget::Native,
            Some(other) => {
                return Err(BackendError::Compilation(format!(
                    "Unsupported quantization bit width: {}. Supported values: 4, 8",
                    other
                )))
            }
        };
        self.compile_with_target(graph, target, calib_data)
    }

    /// Compile a graph with an explicit representation target.
    ///
    /// If `calib_data` is provided, per-tensor/per-channel activation scales
    /// will be applied after weight quantization (weight-only quant only uses
    /// the weight quant step; activations remain f32).
    ///
    /// Takes ownership of `graph` to avoid an unnecessary deep clone.
    pub fn compile_with_target(
        &self,
        graph: ComputeGraph,
        target: CompileTarget,
        calib_data: Option<calibration::CalibrationData>,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        let compiled = crate::compiler::pipeline::CompilerPipeline::new(target, calib_data)
            .run(graph)
            .map_err(|error| BackendError::Compilation(error.to_string()))?;
        let graph = compiled.graph;
        let memory_plan = compiled.memory_plan;
        let plan = self.backend.compile(&graph, &memory_plan)?;

        Ok((plan, memory_plan, graph))
    }

    /// Execute a compiled plan with input data.
    ///
    /// At runtime the `ShapeEnv` resolves any symbolic dimensions from input
    /// byte sizes.  The `MemoryPlan` is tightened (slot sizes shrunk from
    /// worst-case `SYMBOL_DIM_MAX` to actual sizes), then the `ExecutablePlan`
    /// is rebuilt so every `BufferSlice` matches the tightened slots.  This
    /// saves up to 90%+ arena memory for dynamic-shaped inputs.
    ///
    /// `inputs` must correspond one-to-one with `graph.inputs` (in order).
    /// Returns output byte slices corresponding to `graph.outputs` (in order).
    pub fn execute(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.execute_internal(graph, plan, memory_plan, inputs, false, None, None)
            .map(|(outputs, _profile)| outputs)
    }

    pub fn execute_profile(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
    ) -> Result<(Vec<Vec<u8>>, Vec<ProfileEntry>), BackendError> {
        self.execute_internal(graph, plan, memory_plan, inputs, true, None, None)
    }

    /// Profile the opt-in prepared arena fallback path.
    ///
    /// This is a measurement hook for comparing the default plan with
    /// the prepared-constant preload path. It uses the same dispatch
    /// logic as [`Self::execute_prepared_arena_fallback`] but preserves
    /// per-instruction [`ProfileEntry`] rows so callers can observe which
    /// `WriteConst` instructions were removed from the temporary plan.
    #[cfg(feature = "prepared-plan")]
    pub fn execute_profile_prepared_arena_fallback(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
        prepared: &crate::backend::prepared::PreparedExecutablePlan,
    ) -> Result<(Vec<Vec<u8>>, Vec<ProfileEntry>), BackendError> {
        crate::backend::prepared::validate_prepared_against_plan(prepared, plan)?;
        self.execute_internal(graph, plan, memory_plan, inputs, true, Some(prepared), None)
    }

    /// Opt-in prepared-execution fallback path.
    ///
    /// This is a **behaviour-identical scaffold** for future prepared lanes:
    /// the prepared plan is only consulted to validate that it stays in
    /// lock-step with the original [`ExecutablePlan`] (same length,
    /// same original instruction order). Runtime execution then goes
    /// through the existing [`Self::execute`] path so the on-the-wire
    /// behaviour is byte-identical to a direct `forward()` call.
    ///
    /// In particular this path:
    /// - does **not** read from any [`PreparedConstantArena`],
    /// - does **not** skip or remove [`Instruction::WriteConst`],
    /// - does **not** substitute alternative kernels,
    /// - and does **not** perform any per-instruction rewriting.
    ///
    /// It is a clearly named, opt-in entry point that future lanes can
    /// extend (e.g. by replacing individual prepared instructions with
    /// specialised kernels) while keeping the same fallback contract
    /// — any code path that survives the validation check is guaranteed
    /// to behave like the original `execute`.
    ///
    /// The method is gated on the `prepared-plan` cargo feature to
    /// match the rest of the prepared-plan surface; callers that build
    /// without that feature can keep using [`Self::execute`].
    #[cfg(feature = "prepared-plan")]
    pub fn execute_prepared_fallback(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
        prepared: &crate::backend::prepared::PreparedExecutablePlan,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        // Sanity-check the prepared plan against the live plan. The
        // actual dispatch goes through the original plan, so any
        // mismatch detected here would silently desynchronise prepared
        // metadata from the live plan and must be reported loudly
        // instead.
        crate::backend::prepared::validate_prepared_against_plan(prepared, plan)?;
        // Delegate to the existing execution path so we inherit every
        // tightening / shape-env / arena-reuse behaviour of the
        // default forward().
        self.execute(graph, plan, memory_plan, inputs)
    }

    /// Opt-in prepared fallback that preloads fp32 Conv2d weights from
    /// the [`PreparedConstantArena`] into their original runtime arena
    /// slots, then dispatches a temporary plan with the matching exact
    /// weight-slot [`Instruction::WriteConst`] operations removed.
    ///
    /// This remains opt-in and behaviour-equivalent: only exact fp32
    /// Conv2d weight writes that were preloaded from prepared metadata
    /// are skipped. Bias constants, non-Conv constants, dynamic weights,
    /// and unsupported packed kinds stay on the original dispatch path.
    #[cfg(feature = "prepared-plan")]
    pub fn execute_prepared_arena_fallback(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
        prepared: &crate::backend::prepared::PreparedExecutablePlan,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        crate::backend::prepared::validate_prepared_against_plan(prepared, plan)?;
        self.execute_internal(
            graph,
            plan,
            memory_plan,
            inputs,
            false,
            Some(prepared),
            None,
        )
        .map(|(outputs, _profile)| outputs)
    }

    /// Opt-in **no-copy** prepared path.
    ///
    /// Like [`Self::execute_prepared_arena_fallback`], the prepared
    /// plan is consulted to detect static fp32 weight/bias bindings
    /// for `Conv2d` and `MatMul` consumers. The key difference is
    /// that no per-forward copy happens: the prepared weights live
    /// in a [`crate::backend::prepared::PersistentPreparedWeights`]
    /// view that the dispatch loop borrows directly when reading
    /// the weight/bias slot. The corresponding `WriteConst`
    /// instructions are also filtered out of the dispatch plan, so
    /// the total per-forward weight traffic is zero (down from the
    /// full `O(weights_bytes)` of the existing fallback).
    ///
    /// This entry point is byte-identical to [`Self::forward`] and
    /// to [`Self::execute_prepared_arena_fallback`] on every input.
    /// It is gated on the `prepared-plan` cargo feature to match
    /// the rest of the prepared-plan surface; callers that build
    /// without that feature can keep using [`Self::execute`] and
    /// [`Self::execute_prepared_arena_fallback`].
    ///
    /// The no-copy optimization is currently limited to:
    ///
    /// - fp32 `Conv2d` weights + (optional) biases,
    /// - fp32 `MatMul` weights + (optional) biases (including
    ///   `matmul*` and `fused_matmul_add_*` families).
    ///
    /// Packed transposed and quantized slots stay on the safe
    /// fallback path because the current kernels do not consume
    /// them. When the prepared plan has no `Conv2d` / `MatMul`
    /// bindings the view is empty and the path degrades to the
    /// standard dispatch.
    #[cfg(feature = "prepared-plan")]
    pub fn execute_prepared_no_copy(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
        prepared: &crate::backend::prepared::PreparedExecutablePlan,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        crate::backend::prepared::validate_prepared_against_plan(prepared, plan)?;
        let view = crate::backend::prepared::build_persistent_prepared_weights(prepared);
        self.execute_internal_with_persistent_view(
            graph,
            plan,
            memory_plan,
            inputs,
            false,
            Some(&view),
        )
        .map(|(outputs, _profile)| outputs)
    }

    /// Profile the no-copy prepared path. Mirrors
    /// [`Self::execute_prepared_no_copy`] but preserves per-instruction
    /// [`ProfileEntry`] rows so callers can quantify the saved
    /// `WriteConst` traffic on the persistent arena path.
    #[cfg(feature = "prepared-plan")]
    pub fn execute_profile_prepared_no_copy(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
        prepared: &crate::backend::prepared::PreparedExecutablePlan,
    ) -> Result<(Vec<Vec<u8>>, Vec<ProfileEntry>), BackendError> {
        crate::backend::prepared::validate_prepared_against_plan(prepared, plan)?;
        let view = crate::backend::prepared::build_persistent_prepared_weights(prepared);
        self.execute_internal_with_persistent_view(
            graph,
            plan,
            memory_plan,
            inputs,
            true,
            Some(&view),
        )
    }

    fn execute_internal(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
        profile: bool,
        #[cfg_attr(not(feature = "prepared-plan"), allow(unused_variables))]
        prepared_preload: Option<&crate::backend::prepared::PreparedExecutablePlan>,
        #[cfg_attr(not(feature = "prepared-plan"), allow(unused_variables))]
        persistent_view: Option<&crate::backend::prepared::PersistentPreparedWeights>,
    ) -> Result<(Vec<Vec<u8>>, Vec<ProfileEntry>), BackendError> {
        plan.validate()?;
        // ── Preamble: shape env, tighten, safety, arena, input write ──
        let (tightened_memory_plan, shape_env, cached_filtered_plan) =
            if let Some(StaticShapeCache {
                ref tightened_memory_plan,
                ref shape_env,
                ref filtered_plan,
            }) = self.static_shape_cache
            {
                // Fast path: the plan was already tightened on the first
                // inference.  Reuse the cached tightened memory plan and
                // shape env, skipping the O(N_nodes) resolve + tighten
                // + tighten_slices steps entirely.
                (
                    tightened_memory_plan.clone(),
                    shape_env.clone(),
                    Some(filtered_plan.clone()),
                )
            } else {
                let shape_env = ShapeEnv::from_graph_inputs(graph, inputs)
                    .map_err(|e| BackendError::Dispatch(format!("shape env: {e}")))?;
                validate_shapes(graph, &shape_env)
                    .map_err(|e| BackendError::Dispatch(format!("shape validation: {e}")))?;

                let tightened_memory_plan = memory_plan
                    .try_tighten(graph, &shape_env)
                    .map_err(|error| BackendError::Dispatch(format!("memory plan: {error}")))?;
                tighten_slices(plan, memory_plan, &tightened_memory_plan, graph)?;

                // Cache for static-shape models so subsequent calls
                // skip the entire preamble.
                if graph.has_static_shapes() {
                    let filtered_instructions: Vec<Instruction> = plan
                        .instructions
                        .iter()
                        .filter(|instr| {
                            !matches!(
                                instr,
                                Instruction::WriteConst { .. } | Instruction::Fill { .. }
                            )
                        })
                        .cloned()
                        .collect();
                    let filtered_plan = ExecutablePlan {
                        instructions: filtered_instructions,
                        arena_size: plan.arena_size,
                        levels: plan.levels.clone(),
                    };
                    self.static_shape_cache = Some(StaticShapeCache {
                        tightened_memory_plan: tightened_memory_plan.clone(),
                        shape_env: shape_env.clone(),
                        filtered_plan,
                    });
                    // Return None so the FIRST call uses the full plan
                    // (including WriteConst/Fill) to populate the arena.
                    // Subsequent cache-hit calls will use the filtered plan.
                    (tightened_memory_plan, shape_env, None)
                } else {
                    (tightened_memory_plan, shape_env, None)
                }
            };

        // Slot safety check: only needed on the first call when shapes are
        // being resolved for the first time. On subsequent calls with a
        // populated static_shape_cache the shapes are identical and were
        // already validated.
        if cached_filtered_plan.is_none() {
            for (&node_id, slot) in &tightened_memory_plan.slots {
                let slot_end = slot.offset.checked_add(slot.size).ok_or_else(|| {
                    BackendError::Dispatch(format!(
                        "node {node_id}: tightened memory slot range overflows"
                    ))
                })?;
                if slot_end > tightened_memory_plan.total_size {
                    return Err(BackendError::Dispatch(format!(
                        "node {node_id}: tightened memory slot range {}..{slot_end} exceeds arena size {}",
                        slot.offset, tightened_memory_plan.total_size
                    )));
                }
                if let Some(node) = graph.get_node(node_id) {
                    let shape =
                        resolve_shape(&node.output_type.shape, &shape_env).map_err(|error| {
                            BackendError::Dispatch(format!(
                                "node {node_id}: failed to resolve output shape: {error}"
                            ))
                        })?;
                    let numel = shape.iter().try_fold(1usize, |count, &dimension| {
                        let dimension = usize::try_from(dimension).map_err(|_| {
                            BackendError::Dispatch(format!(
                                "node {node_id}: resolved dimension does not fit usize"
                            ))
                        })?;
                        count.checked_mul(dimension).ok_or_else(|| {
                            BackendError::Dispatch(format!(
                                "node {node_id}: resolved element count overflows"
                            ))
                        })
                    })?;
                    let needed = checked_execution_storage_size(&node.output_type.dtype, numel)?;
                    if needed > slot.size {
                        return Err(BackendError::Dispatch(format!(
                            "node {}: resolved size {} exceeds tightened slot size {} (shape {:?})",
                            node_id, needed, slot.size, node.output_type.shape
                        )));
                    }
                }
            }
        }

        // Use the tightened memory plan's total_size instead of the
        // compile-time plan.arena_size which was sized using
        // SYMBOL_DIM_MAX=8192 for symbolic dimensions — for batch=1
        // this can be ~50MB vs ~400MB.
        let arena_size = tightened_memory_plan.total_size;
        let enough_capacity = self
            .cached_arena
            .as_ref()
            .is_some_and(|(cap, _)| *cap >= arena_size);
        if !enough_capacity {
            self.cached_arena = Some((arena_size, self.backend.try_allocate_arena(arena_size)?));
        }
        let arena = &self
            .cached_arena
            .as_ref()
            .ok_or_else(|| {
                BackendError::Dispatch("backend arena allocation returned no buffer".into())
            })?
            .1;

        if inputs.len() != graph.inputs.len() {
            return Err(BackendError::Dispatch(format!(
                "expected {} graph inputs, received {}",
                graph.inputs.len(),
                inputs.len()
            )));
        }
        for (i, &input_node_id) in graph.inputs.iter().enumerate() {
            let input_bytes = inputs.get(i).ok_or_else(|| {
                BackendError::Dispatch(format!("missing input {} for node {}", i, input_node_id))
            })?;
            let slot = tightened_memory_plan
                .slots
                .get(&input_node_id)
                .ok_or_else(|| {
                    BackendError::Dispatch(format!(
                        "no memory slot for input node {}",
                        input_node_id
                    ))
                })?;
            if input_bytes.len() != slot.size {
                return Err(BackendError::Dispatch(format!(
                    "input {i} for node {input_node_id} has {} bytes but its memory slot requires {}",
                    input_bytes.len(),
                    slot.size
                )));
            }
            let input_end = slot.offset.checked_add(input_bytes.len()).ok_or_else(|| {
                BackendError::Dispatch(format!(
                    "input {i} for node {input_node_id} arena range overflows"
                ))
            })?;
            if input_end > arena_size {
                return Err(BackendError::Dispatch(format!(
                    "input {i} for node {input_node_id} range {}..{input_end} exceeds arena size {arena_size}",
                    slot.offset
                )));
            }
            self.backend
                .try_write_arena(arena, slot.offset, input_bytes)?;
        }

        // ── Dispatch: no-copy persistent view path vs standard path ──
        #[cfg(feature = "prepared-plan")]
        if let Some(view) = persistent_view {
            let dispatch_plan: ExecutablePlan = if view.is_empty() {
                plan.clone()
            } else {
                let mut filtered: Vec<Instruction> = Vec::with_capacity(plan.instructions.len());
                for instr in &plan.instructions {
                    // Only skip WriteConst for fp32 weight slots — the
                    // persistent-view dispatch path (`dispatch_with_persistent_view`)
                    // knows how to satisfy fp32 Conv2d/MatMul weights from the
                    // view. Quantized weight slots must keep their WriteConst in
                    // the dispatch plan because the kernel dispatch reads them
                    // from the arena, not from the persistent view.
                    let drop = matches!(
                        instr,
                        Instruction::WriteConst { dst, .. }
                            if view.get(&(dst.offset, dst.size)).is_some()
                    );
                    if !drop {
                        filtered.push(instr.clone());
                    }
                }
                ExecutablePlan {
                    instructions: filtered,
                    arena_size: plan.arena_size,
                    levels: plan.levels.clone(),
                }
            };

            let profile_entries = if profile {
                let mut entries = Vec::with_capacity(dispatch_plan.instructions.len());
                for (instruction_index, instruction) in
                    dispatch_plan.instructions.iter().cloned().enumerate()
                {
                    let (node_id, kernel_name) = match &instruction {
                        Instruction::CallKernel {
                            node_id,
                            kernel_name,
                            ..
                        } => (*node_id, kernel_name.clone()),
                        Instruction::MemCopy { .. } => (None, "memcopy".to_string()),
                        Instruction::Fill { .. } => (None, "fill".to_string()),
                        Instruction::WriteConst { .. } => (None, "write_const".to_string()),
                    };
                    let single = ExecutablePlan {
                        instructions: vec![instruction],
                        arena_size: dispatch_plan.arena_size,
                        levels: vec![0],
                    };
                    let start = std::time::Instant::now();
                    self.backend.dispatch_with_persistent_view(
                        &single,
                        arena,
                        &shape_env,
                        Some(view),
                    )?;
                    entries.push(crate::backend::ProfileEntry {
                        instruction_index,
                        node_id,
                        kernel_name,
                        elapsed_ns: start.elapsed().as_nanos(),
                    });
                }
                entries
            } else {
                self.backend.dispatch_with_persistent_view(
                    &dispatch_plan,
                    arena,
                    &shape_env,
                    Some(view),
                )?;
                Vec::new()
            };

            let outputs = read_execution_outputs(
                graph,
                &tightened_memory_plan,
                &shape_env,
                &self.backend,
                arena,
            )?;
            return Ok((outputs, profile_entries));
        }

        // ── Standard / prepared-preload path ──
        #[cfg(feature = "prepared-plan")]
        let arena_preloaded_plan = if let Some(prepared) = prepared_preload {
            Some(preload_prepared_constants_and_skip_redundant_writes(
                &self.backend,
                arena,
                plan,
                prepared,
            )?)
        } else {
            None
        };

        #[cfg(not(feature = "prepared-plan"))]
        let arena_preloaded_plan: Option<ExecutablePlan> = None;

        let dispatch_plan = if let Some(ref fp) = cached_filtered_plan {
            fp
        } else {
            arena_preloaded_plan.as_ref().unwrap_or(plan)
        };

        let profile_entries = if profile {
            self.backend
                .dispatch_profile(dispatch_plan, arena, &shape_env)?
        } else {
            self.backend.dispatch(dispatch_plan, arena, &shape_env)?;
            Vec::new()
        };

        read_execution_outputs(
            graph,
            &tightened_memory_plan,
            &shape_env,
            &self.backend,
            arena,
        )
        .map(|outputs| (outputs, profile_entries))
    }

    #[cfg(feature = "prepared-plan")]
    fn execute_internal_with_persistent_view(
        &mut self,
        graph: &ComputeGraph,
        plan: &mut ExecutablePlan,
        memory_plan: &MemoryPlan,
        inputs: &[&[u8]],
        profile: bool,
        persistent_view: Option<&crate::backend::prepared::PersistentPreparedWeights>,
    ) -> Result<(Vec<Vec<u8>>, Vec<ProfileEntry>), BackendError> {
        // ── Unified path: this function is now merged into execute_internal ──
        self.execute_internal(
            graph,
            plan,
            memory_plan,
            inputs,
            profile,
            None,
            persistent_view,
        )
    }

    /// Convenience: compile + execute in a single call.
    pub fn run(
        &mut self,
        graph: &ComputeGraph,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let (mut plan, memory_plan, compiled_graph) = self.compile_with_plan(graph.clone())?;
        self.execute(&compiled_graph, &mut plan, &memory_plan, inputs)
    }

    /// Compile a training model from a forward graph.
    ///
    /// `batch_inputs`: all non-parameter runtime inputs including data, labels, masks, targets.
    ///   These are overwritten each step. Everything else persists in the arena.
    ///
    /// `params`: trainable parameter graph input nodes (must have entries in param_data)
    /// `param_data`: initial f32 byte slices for each param (same order as params)
    ///
    /// `batch_shape_env`: optional concrete shapes for symbolic batch dims. When provided,
    ///   the memory plan is tightened after compilation, shrinking slots from worst-case
    ///   SYMBOL_DIM_MAX to actual sizes. Pass `None` to skip tightening (uses worst-case).
    #[allow(clippy::too_many_arguments)]
    pub fn compile_train(
        &self,
        forward_graph: &ComputeGraph,
        loss_node: NodeId,
        params: &[NodeId],
        param_data: &[&[u8]],
        batch_inputs: &[NodeId],
        batch_shape_env: Option<&ShapeEnv>,
        config: &TrainConfig,
    ) -> Result<CompiledTrainingModel<B>, BackendError>
    where
        B: Clone,
    {
        // 1. Clone forward graph
        let mut combined_graph = forward_graph.clone();

        // 2. Build backward graph (result already contains forward + backward nodes)
        let (grad_graph, grad_map) = build_backward_graph(&combined_graph, loss_node, None)
            .map_err(|e| BackendError::Compilation(format!("build_backward_graph: {e}")))?;
        combined_graph = grad_graph;

        // 3. Map each param to its gradient accumulator
        let params_with_grads: Vec<(NodeId, NodeId)> = params
            .iter()
            .map(|p| {
                let grad = *grad_map.get(p).ok_or_else(|| {
                    BackendError::Compilation(format!(
                        "param {} has no gradient in backward graph",
                        p
                    ))
                })?;
                Ok((*p, grad))
            })
            .collect::<Result<Vec<_>, BackendError>>()?;

        // 4. Inject optimizer nodes
        let injection =
            inject_optimizer(&mut combined_graph, &params_with_grads, &config.optimizer)
                .map_err(|e| BackendError::Compilation(format!("inject_optimizer: {e}")))?;

        // 4b. Insert F8x4R gradient quantization around optimizer gradient inputs
        use crate::compiler::passes::gradient_quantization;
        gradient_quantization::quantize_gradients(&mut combined_graph)
            .map_err(|error| BackendError::Compilation(error.to_string()))?;

        // 5. Set graph inputs and outputs
        combined_graph.outputs = vec![loss_node];

        // Mark persistent state nodes as required so the memory planner
        // extends their lifetimes to end of execution (prevents slot reuse
        // by post-optimizer nodes — especially important for m/v state).
        for &param_id in params {
            combined_graph.add_required_node(param_id);
        }
        for state_nodes in &injection.state_input_nodes {
            for &state_id in state_nodes {
                combined_graph.add_required_node(state_id);
            }
        }
        for &updated_id in &injection.updated_param_nodes {
            combined_graph.add_required_node(updated_id);
        }

        // Inputs = [batch_inputs..., params..., state_input_nodes...]
        let mut all_inputs: Vec<NodeId> = Vec::new();
        all_inputs.extend_from_slice(batch_inputs);
        all_inputs.extend_from_slice(params);
        for state in &injection.state_input_nodes {
            all_inputs.extend_from_slice(state);
        }
        combined_graph.inputs = all_inputs;

        // 6. Run the standard compiler pipeline
        let (plan, memory_plan, final_graph) =
            self.compile_with_plan_and_quantize(combined_graph, config.quantize, None)?;

        // 6b. Optionally tighten memory plan using concrete batch shapes.
        //     This shrinks slots from SYMBOL_DIM_MAX worst-case to actual sizes.
        let (plan, memory_plan) = if let Some(shape_env) = batch_shape_env {
            let tightened_mp = memory_plan
                .try_tighten(&final_graph, shape_env)
                .map_err(|error| BackendError::Dispatch(format!("memory plan: {error}")))?;
            let mut plan = plan;
            tighten_slices(&mut plan, &memory_plan, &tightened_mp, &final_graph)
                .map_err(|e| BackendError::Compilation(format!("tighten: {e}")))?;
            (plan, tightened_mp)
        } else {
            (plan, memory_plan)
        };

        // 7. Extract slot metadata from the (possibly tightened) memory plan
        let mut input_slot_offsets = Vec::new();
        for &input_id in batch_inputs {
            let slot = memory_plan.slots.get(&input_id).ok_or_else(|| {
                BackendError::Compilation(format!("no memory slot for batch input {}", input_id))
            })?;
            input_slot_offsets.push((slot.offset, slot.size));
        }

        let loss_slot = memory_plan.slots.get(&loss_node).ok_or_else(|| {
            BackendError::Compilation(format!("no memory slot for loss node {}", loss_node))
        })?;
        let loss_slot_offset = loss_slot.offset;

        // 8. Store shape env for dispatch (batch_shape_env or empty)
        let model_shape_env = batch_shape_env.cloned().unwrap_or_default();

        // 9. Allocate arena and write initial values
        let arena = self.backend.try_allocate_arena(plan.arena_size)?;

        // Write param data
        for (i, data) in param_data.iter().enumerate() {
            let slot = memory_plan.slots.get(&params[i]).ok_or_else(|| {
                BackendError::Compilation(format!("param {} slot not found", params[i]))
            })?;
            self.backend.try_write_arena(&arena, slot.offset, data)?;
        }

        // Write zero-initialized optimizer state (m, v for AdamW),
        // with t (timestep counter) initialized to 1.
        for state_nodes in &injection.state_input_nodes {
            for &state_id in state_nodes {
                let slot = memory_plan.slots.get(&state_id).ok_or_else(|| {
                    BackendError::Compilation(format!("state node {} slot not found", state_id))
                })?;
                let is_t = final_graph
                    .get_node(state_id)
                    .map(|n| n.name.starts_with("optimizer/t_"))
                    .unwrap_or(false);
                let init_data: Vec<u8> = if is_t {
                    1u64.to_le_bytes().to_vec()
                } else {
                    vec![0u8; slot.size]
                };
                self.backend
                    .try_write_arena(&arena, slot.offset, &init_data)?;
            }
        }

        Ok(CompiledTrainingModel {
            backend: self.backend.clone(),
            plan,
            memory_plan,
            graph: final_graph,
            arena,
            input_slot_offsets,
            loss_slot_offset,
            shape_env: model_shape_env,
        })
    }
}

/// A compiled training model with a persistent memory arena.
///
/// Created by [`GraphExecutor::compile_train`]. The arena is allocated once
/// and reused across steps. Parameters and optimizer state (m, v) persist
/// via in-place kernel writes + required_nodes lifetime extension in the
/// memory planner. Only batch input slots are overwritten each step.
pub struct CompiledTrainingModel<B: Backend> {
    pub backend: B,
    pub plan: ExecutablePlan,
    pub memory_plan: MemoryPlan,
    pub graph: ComputeGraph,
    pub arena: B::Buffer,
    /// (offset, size) for each batch input — overwritten each step
    pub input_slot_offsets: Vec<(usize, usize)>,
    /// Arena offset for the loss scalar (4 bytes, f32)
    pub loss_slot_offset: usize,
    /// Shape env (empty for now — no tightening at compile time)
    pub shape_env: ShapeEnv,
}

impl<B: Backend> CompiledTrainingModel<B> {
    /// Execute one training step.
    ///
    /// `batch_data[i]` is written to the i-th batch input slot (same order as
    /// `batch_inputs` in `compile_train`). Must have exactly the expected byte size.
    ///
    /// Returns the loss as `f32`.
    pub fn train_step(&mut self, batch_data: &[&[u8]]) -> Result<f32, BackendError> {
        // 1. Validate batch input count
        if batch_data.len() != self.input_slot_offsets.len() {
            return Err(BackendError::Dispatch(format!(
                "train_step: expected {} batch inputs, got {}",
                self.input_slot_offsets.len(),
                batch_data.len()
            )));
        }

        // 2. Write batch input data into reserved arena slots
        for (i, data) in batch_data.iter().enumerate() {
            let (off, size) = self.input_slot_offsets[i];
            if data.len() != size {
                return Err(BackendError::Dispatch(format!(
                    "train_step: batch input {} expected {} bytes, got {}",
                    i,
                    size,
                    data.len()
                )));
            }
            self.backend.try_write_arena(&self.arena, off, data)?;
        }

        // 3. Dispatch the full train-step graph
        self.backend
            .dispatch(&self.plan, &self.arena, &self.shape_env)
            .map_err(|e| BackendError::Dispatch(format!("train_step dispatch: {e}")))?;

        // 4. Read loss immediately (before any post-dispatch writes)
        let loss_bytes = self
            .backend
            .try_read_arena(&self.arena, self.loss_slot_offset, 4)?;
        let loss_arr: [u8; 4] = loss_bytes
            .get(..4)
            .ok_or_else(|| BackendError::Dispatch("train_step: loss buffer too small".into()))?
            .try_into()
            .map_err(|_| BackendError::Dispatch("train_step: invalid loss bytes".into()))?;
        let loss = f32::from_le_bytes(loss_arr);

        // 5. Increment Adam/AdamW step counters for bias correction
        self.increment_optimizer_steps()?;

        Ok(loss)
    }

    /// Increment `t` in all Adam/AdamW kernel instructions.
    /// This advances the bias correction denominator for the next step.
    /// For AdamW, `t` is stored in the arena at input_slices[4].
    /// For Adam, `t` is still in params[4].
    fn increment_optimizer_steps(&mut self) -> Result<(), BackendError> {
        for instr in &mut self.plan.instructions {
            if let Instruction::CallKernel {
                kernel_name,
                params,
                input_slices,
                ..
            } = instr
            {
                match kernel_name.as_str() {
                    "adam_update_f32" | "adam_update_f16_state" => {
                        if params.len() <= 4 {
                            return Err(BackendError::Dispatch(format!(
                                "{} expected >=5 params, got {}",
                                kernel_name,
                                params.len()
                            )));
                        }
                        params[4] = params[4].checked_add(1).ok_or_else(|| {
                            BackendError::Dispatch(
                                "train_step: Adam step overflow (t > u64::MAX)".into(),
                            )
                        })?;
                    }
                    "adamw_update_f32" | "adamw_update_f16_state" => {
                        if input_slices.len() <= 4 {
                            return Err(BackendError::Dispatch(format!(
                                "{} expected >=5 input slices, got {}",
                                kernel_name,
                                input_slices.len()
                            )));
                        }
                        let t_slice = &input_slices[4];
                        let t_bytes =
                            self.backend
                                .try_read_arena(&self.arena, t_slice.offset, 8)?;
                        let t_arr: [u8; 8] = t_bytes[..8]
                            .try_into()
                            .map_err(|_| BackendError::Dispatch("invalid t step value".into()))?;
                        let t_val = u64::from_le_bytes(t_arr);
                        let new_t = t_val.checked_add(1).ok_or_else(|| {
                            BackendError::Dispatch(
                                "train_step: AdamW step overflow (t > u64::MAX)".into(),
                            )
                        })?;
                        self.backend.try_write_arena(
                            &self.arena,
                            t_slice.offset,
                            &new_t.to_le_bytes(),
                        )?;
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }
}

/// Rebuild an [`ExecutablePlan`] from a tightened [`MemoryPlan`] without a
/// full backend re-compilation.
///
/// For each instruction:
/// - [`BufferSlice`] offsets/sizes are updated to match the tightened slots.
/// - [`Instruction::CallKernel`] parameters (M, K, N, etc.) are re-resolved
///   from `tightened_memory_plan.tightened_params`.
///
/// This is a lightweight alternative to calling [`Backend::compile`] again —
/// it avoids re-running fusion checks, quantization detection, and all other
/// backend lowering logic that is unchanged after tightening.
pub fn tighten_slices(
    plan: &mut ExecutablePlan,
    _original_memory_plan: &MemoryPlan,
    tightened_memory_plan: &MemoryPlan,
    graph: &ComputeGraph,
) -> Result<(), BackendError> {
    // Build an offset → tightened_size map for non-CallKernel instructions
    // (Fill, MemCopy, WriteConst) that don't carry a node_id.
    // Note: in-place reuse candidates share the same offset as their
    // input node, but since both have identical sizes, the HashMap
    // collision is harmless (min(sz, current) = sz for both).
    let mut offset_size: HashMap<usize, usize> = HashMap::new();
    for (_node_id, slot) in &tightened_memory_plan.slots {
        offset_size.insert(slot.offset, slot.size);
    }
    for (_key, slot) in &tightened_memory_plan.secondary_slots {
        offset_size.insert(slot.offset, slot.size);
    }

    for instr in &mut plan.instructions {
        match instr {
            Instruction::CallKernel {
                input_slices,
                output_slice,
                secondary_output_slice,
                params,
                node_id,
                ..
            } => {
                if let Some(nid) = node_id {
                    // ── Update output_slice from tightened memory plan ──
                    let slot = tightened_memory_plan.slots.get(nid).ok_or_else(|| {
                        BackendError::Dispatch(format!(
                            "cannot tighten instruction for node {nid}: output slot is missing"
                        ))
                    })?;
                    output_slice.offset = slot.offset;
                    output_slice.size = slot.size;
                    // ── Update secondary_output_slice ──
                    if let Some(sec_slice) = secondary_output_slice.as_mut() {
                        let slot = tightened_memory_plan
                            .secondary_slots
                            .get(&(*nid, 1))
                            .ok_or_else(|| {
                                BackendError::Dispatch(format!(
                                    "cannot tighten instruction for node {nid}: secondary output slot is missing"
                                ))
                            })?;
                        sec_slice.offset = slot.offset;
                        sec_slice.size = slot.size;
                    }
                    // ── Update input slices from the graph ──
                    // IMPORTANT: Must use the same filter_map logic as compilation
                    // (cpu/mod.rs line 196-205) which skips inputs without memory
                    // slots.  Using enumerate() with node.inputs directly causes
                    // index misalignment when some inputs lack slots — e.g. if
                    // input A has no slot but B and C do, input_slices = [B, C].
                    // enumerate() would map input_slices[0]→A (skipped) and
                    // input_slices[1]→B, so C never gets updated and B's offset
                    // is written to C's slice, causing buffer aliasing.
                    let node = graph.get_node(*nid).ok_or_else(|| {
                        BackendError::Dispatch(format!(
                            "cannot tighten instruction for missing node {nid}"
                        ))
                    })?;
                    let expected_inputs = node
                        .inputs
                        .iter()
                        .filter(|input_nid| tightened_memory_plan.slots.contains_key(input_nid))
                        .count();
                    if input_slices.len() != expected_inputs {
                        return Err(BackendError::Dispatch(format!(
                            "cannot tighten instruction for node {nid}: expected {expected_inputs} input slices, found {}",
                            input_slices.len()
                        )));
                    }
                    let mut slice_iter = input_slices.iter_mut();
                    for &input_nid in &node.inputs {
                        if let Some(slot) = tightened_memory_plan.slots.get(&input_nid) {
                            if let Some(slice) = slice_iter.next() {
                                slice.offset = slot.offset;
                                slice.size = slot.size;
                            }
                        }
                    }
                    // ── Update kernel params from tightened memory plan ──
                    if let Some(tightened_params) = tightened_memory_plan.tightened_params.get(nid)
                    {
                        *params = tightened_params.clone();
                    }
                }
            }
            Instruction::MemCopy { dst, src } => {
                if let Some(&sz) = offset_size.get(&dst.offset) {
                    dst.size = dst.size.min(sz);
                }
                if let Some(&sz) = offset_size.get(&src.offset) {
                    src.size = src.size.min(sz);
                }
            }
            Instruction::Fill { dst, .. } => {
                if let Some(&sz) = offset_size.get(&dst.offset) {
                    dst.size = dst.size.min(sz);
                }
            }
            Instruction::WriteConst { dst, .. } => {
                if let Some(&sz) = offset_size.get(&dst.offset) {
                    dst.size = dst.size.min(sz);
                }
            }
        }
    }

    plan.arena_size = tightened_memory_plan.total_size;
    Ok(())
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
    let symbol_max = crate::ir::SYMBOL_DIM_MAX.load(std::sync::atomic::Ordering::Relaxed);
    shape
        .iter()
        .map(|d| d.evaluate_with_env(env).unwrap_or(symbol_max))
        .collect()
}

/// Parse an axis attribute, supporting negative values (ONNX-style).
/// Returns a 0-based positive axis, or an error if out of range.
fn resolve_axis(
    attrs: &std::collections::HashMap<String, String>,
    key: &str,
    rank: usize,
) -> Result<usize, String> {
    let raw: i64 = attrs.get(key).and_then(|s| s.parse().ok()).unwrap_or(0);
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
    let order = graph
        .try_topological_sort()
        .map_err(|error| error.to_string())?;
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
                        "Concat node {} ({}): expected at least 2 inputs, got {}",
                        node_id,
                        node.name,
                        input_shapes.len()
                    ));
                }
                let rank = input_shapes[0].len();
                let axis = resolve_axis(&node.attrs, "axis", rank)
                    .map_err(|e| format!("Concat node {} ({}): {e}", node_id, node.name))?;
                // Skip dim compatibility checks if any input contains
                // unresolved Symbol dims (resolved to SYMBOL_DIM_MAX).
                let symbol_max =
                    crate::ir::SYMBOL_DIM_MAX.load(std::sync::atomic::Ordering::Relaxed);
                let has_unresolved = input_shapes.iter().any(|s| s.contains(&symbol_max));
                if !has_unresolved {
                    for (i, s) in input_shapes.iter().enumerate().skip(1) {
                        if s.len() != rank {
                            // Debug: collect input shapes as string
                            let shapes_str: Vec<String> =
                                input_shapes.iter().map(|s| format!("{:?}", s)).collect();
                            return Err(format!(
                                "Concat node {} name='{}' op='{:?}': rank mismatch input 0 ({}) vs input {} ({}). Input shapes: {:?}",
                                node_id,
                                node.name,
                                node.opcode,
                                rank,
                                i,
                                s.len(),
                                shapes_str,
                            ));
                        }
                        for j in 0..rank {
                            if j != axis && s[j] != input_shapes[0][j] {
                                return Err(format!(
                                    "Concat node {} ({}): dim {} mismatch at axis {}: {} vs {}",
                                    node_id, node.name, j, axis, s[j], input_shapes[0][j]
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
                        node_id,
                        input_shapes.len()
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
                    return Err(format!(
                        "Conv2d node {}: expected 2 inputs, got {}",
                        node_id,
                        input_shapes.len()
                    ));
                }
                let inp = &input_shapes[0];
                let weight = &input_shapes[1];
                if inp.len() < 4 {
                    return Err(format!(
                        "Conv2d node {}: input must have 4 dims [N,C,H,W], got {:?}",
                        node_id, inp
                    ));
                }
                if weight.len() < 4 {
                    return Err(format!(
                        "Conv2d node {}: weight must have 4 dims [F,C,KH,KW], got {:?}",
                        node_id, weight
                    ));
                }
                let groups: u64 = node
                    .attrs
                    .get("groups")
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
                    return Err(format!(
                        "Transpose node {}: input must have at least 2 dims, got {:?}",
                        node_id,
                        input_shapes.first()
                    ));
                }
                // Validate optional permutation attribute
                if let Some(perm_str) = node.attrs.get("perm") {
                    let perm: Vec<usize> = perm_str
                        .trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if perm.len() != input_shapes[0].len() {
                        return Err(format!(
                            "Transpose node {}: permutation length {} != rank {}",
                            node_id,
                            perm.len(),
                            input_shapes[0].len()
                        ));
                    }
                    let mut seen = vec![false; perm.len()];
                    for &p in &perm {
                        if p >= perm.len() {
                            return Err(format!(
                                "Transpose node {}: permutation axis {} out of bounds for rank {}",
                                node_id,
                                p,
                                perm.len()
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
                if input_shapes.is_empty() {
                    continue;
                }
                let in_numel: u64 = input_shapes[0].iter().product();
                // For reshape, compute expected numel from attrs or output shape
                if let Some(out_shape) = node.attrs.get("shape") {
                    // Split the shape attr into tokens. Each token is either a
                    // concrete u64 dim ("42") or a symbolic dim ("N", "-1", etc.)
                    // that will be resolved at runtime.
                    let tokens: Vec<&str> = out_shape
                        .trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .collect();
                    // Collect only the concrete (u64-parseable) dims.
                    let concrete: Vec<u64> = tokens.iter().filter_map(|s| s.parse().ok()).collect();
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
                        if known_product == 0 || !in_numel.is_multiple_of(known_product) {
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
                    return Err(format!(
                        "Norm node {}: expected at least 2 inputs (data + weight)",
                        node_id
                    ));
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
                if input_shapes.is_empty() {
                    continue;
                }
                let dim: usize = node
                    .attrs
                    .get("dim")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(0);
                let start: i64 = node
                    .attrs
                    .get("start")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(0);
                let end: i64 = node
                    .attrs
                    .get("end")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(-1);
                let rank = input_shapes[0].len();
                if dim >= rank {
                    return Err(format!(
                        "Slice node {}: dim {} out of bounds for rank {}",
                        node_id, dim, rank
                    ));
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
                if input_shapes.is_empty() {
                    continue;
                }
                if input_shapes[0].len() < 4 {
                    return Err(format!(
                        "Pool node {}: input must have 4 dims [N,C,H,W], got {:?}",
                        node_id, input_shapes[0]
                    ));
                }
                let k: u64 = node
                    .attrs
                    .get("kernel_size")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(2);
                let s: u64 = node
                    .attrs
                    .get("stride")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(2);
                if k == 0 || s == 0 {
                    return Err(format!(
                        "Pool node {}: kernel_size={} and stride={} must be >0",
                        node_id, k, s
                    ));
                }
            }
            Opcode::Squeeze => {
                if input_shapes.is_empty() {
                    continue;
                }
                let dim: usize = node
                    .attrs
                    .get("dim")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(0);
                if dim >= input_shapes[0].len() {
                    return Err(format!(
                        "Squeeze node {}: dim {} out of bounds for rank {}",
                        node_id,
                        dim,
                        input_shapes[0].len()
                    ));
                }
                if input_shapes[0][dim] != 1 {
                    return Err(format!(
                        "Squeeze node {}: dim {} has size {} (must be 1 to squeeze)",
                        node_id, dim, input_shapes[0][dim]
                    ));
                }
            }
            Opcode::Unsqueeze => {
                // Unsqueeze attrs store the output dim; just check it's reasonable
                let dim: usize = node
                    .attrs
                    .get("dim")
                    .and_then(|a| a.parse().ok())
                    .unwrap_or(0);
                let max_rank = input_shapes.first().map(|s| s.len() + 1).unwrap_or(1);
                if dim > max_rank {
                    return Err(format!(
                        "Unsqueeze node {}: dim {} out of bounds for rank {}",
                        node_id, dim, max_rank
                    ));
                }
            }
            Opcode::Gather => {
                if input_shapes.len() < 2 {
                    return Err(format!(
                        "Gather node {}: expected at least 2 inputs (data + indices), got {}",
                        node_id,
                        input_shapes.len()
                    ));
                }
                let rank = input_shapes[0].len();
                let _axis = resolve_axis(&node.attrs, "axis", rank)
                    .map_err(|e| format!("Gather node {}: {e}", node_id))?;
            }
            Opcode::Pad => {
                if input_shapes.is_empty() {
                    continue;
                }
                let rank = input_shapes[0].len();
                if let Some(pads) = node.attrs.get("pads") {
                    let parsed: Vec<i64> = pads
                        .trim_matches(|c| c == '[' || c == ']')
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if parsed.len() != 2 * rank {
                        return Err(format!(
                            "Pad node {}: pads length {} != 2*rank {}",
                            node_id,
                            parsed.len(),
                            2 * rank
                        ));
                    }
                }
            }
            Opcode::Neg
            | Opcode::Abs
            | Opcode::Exp
            | Opcode::Log
            | Opcode::Sqrt
            | Opcode::Round
            | Opcode::Relu
            | Opcode::Gelu
            | Opcode::Silu
            | Opcode::Sigmoid
            | Opcode::Tanh
                if input_shapes.len() != 1 =>
            {
                return Err(format!(
                    "Unary op {:?} node {}: expected 1 input, got {}",
                    node.opcode,
                    node_id,
                    input_shapes.len()
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

// ── tests for the opt-in prepared execution fallback ──────────────────

#[cfg(feature = "prepared-plan")]
fn preload_prepared_fp32_slot<B: Backend>(
    backend: &B,
    arena: &B::Buffer,
    constant_arena: &crate::backend::prepared::PreparedConstantArena,
    instruction_index: usize,
    id: Option<crate::backend::prepared::PackedWeightId>,
    slot: crate::backend::BufferSlice,
    label: &str,
    preloaded_slots: &mut std::collections::HashSet<(usize, usize)>,
    plan_len: usize,
    arena_size: usize,
) -> Result<(), BackendError> {
    use crate::backend::prepared::PackedWeightKind;

    let Some(id) = id else {
        return Ok(());
    };
    if id.kind != PackedWeightKind::Fp32 {
        return Ok(());
    }
    let Some(values) = constant_arena.get(id) else {
        return Err(BackendError::Dispatch(format!(
            "prepared arena {label} references missing fp32 constant slot {} at instruction {instruction_index}",
            id.index
        )));
    };
    let bytes = bytemuck::cast_slice(values);
    if bytes.len() != slot.size {
        return Err(BackendError::Dispatch(format!(
            "prepared arena {label} byte length mismatch at instruction {instruction_index}: arena {} bytes, plan slot {} bytes",
            bytes.len(),
            slot.size
        )));
    }
    if instruction_index >= plan_len {
        return Err(BackendError::Dispatch(format!(
            "prepared arena {label} instruction index {instruction_index} out of bounds for plan length {plan_len}"
        )));
    }
    let slot_end = slot.offset.checked_add(slot.size).ok_or_else(|| {
        BackendError::Dispatch(format!(
            "prepared arena {label} slot range overflows at instruction {instruction_index}"
        ))
    })?;
    if slot_end > arena_size {
        return Err(BackendError::Dispatch(format!(
            "prepared arena {label} slot range {}..{} exceeds plan arena size {arena_size} at instruction {instruction_index}",
            slot.offset, slot_end
        )));
    }
    backend.try_write_arena(arena, slot.offset, bytes)?;
    preloaded_slots.insert((slot.offset, slot.size));
    Ok(())
}

#[cfg(feature = "prepared-plan")]
fn preload_prepared_constants_and_skip_redundant_writes<B: Backend>(
    backend: &B,
    arena: &B::Buffer,
    plan: &ExecutablePlan,
    prepared: &crate::backend::prepared::PreparedExecutablePlan,
) -> Result<ExecutablePlan, BackendError> {
    use crate::backend::prepared::PreparedInstruction;
    use std::collections::HashSet;

    let Some(constant_arena) = prepared.constant_arena() else {
        return Ok(plan.clone());
    };

    let mut preloaded_slots = HashSet::new();
    for prepared_instruction in &prepared.instructions {
        match prepared_instruction {
            PreparedInstruction::Conv2d(conv) => {
                preload_prepared_fp32_slot(
                    backend,
                    arena,
                    constant_arena,
                    conv.instruction_index,
                    conv.packed_weight,
                    conv.weight,
                    "conv weight",
                    &mut preloaded_slots,
                    plan.instructions.len(),
                    plan.arena_size,
                )?;
                if let Some(bias) = conv.bias {
                    preload_prepared_fp32_slot(
                        backend,
                        arena,
                        constant_arena,
                        conv.instruction_index,
                        conv.packed_bias,
                        bias,
                        "conv bias",
                        &mut preloaded_slots,
                        plan.instructions.len(),
                        plan.arena_size,
                    )?;
                }
            }
            PreparedInstruction::MatMul(matmul) => {
                preload_prepared_fp32_slot(
                    backend,
                    arena,
                    constant_arena,
                    matmul.instruction_index,
                    matmul.packed_weight,
                    matmul.b,
                    "matmul weight",
                    &mut preloaded_slots,
                    plan.instructions.len(),
                    plan.arena_size,
                )?;
                if let Some(bias) = matmul.bias {
                    preload_prepared_fp32_slot(
                        backend,
                        arena,
                        constant_arena,
                        matmul.instruction_index,
                        matmul.packed_bias,
                        bias,
                        "matmul bias",
                        &mut preloaded_slots,
                        plan.instructions.len(),
                        plan.arena_size,
                    )?;
                }
            }
            PreparedInstruction::Generic { .. } => {}
        }
    }

    if preloaded_slots.is_empty() {
        return Ok(plan.clone());
    }

    let instructions = plan
        .instructions
        .iter()
        .filter(|instruction| match instruction {
            Instruction::WriteConst { dst, .. } => {
                !preloaded_slots.contains(&(dst.offset, dst.size))
            }
            _ => true,
        })
        .cloned()
        .collect();

    Ok(ExecutablePlan {
        instructions,
        arena_size: plan.arena_size,
        levels: plan.levels.clone(),
    })
}

#[cfg(all(test, feature = "prepared-plan"))]
mod prepared_fallback_tests {
    //! End-to-end coverage of
    //! [`GraphExecutor::execute_prepared_fallback`]. The plans we build
    //! cover the three shapes that the order-mapping validation has to
    //! accept: pure elementwise (no `WriteConst`), a single
    //! `WriteConst` + `CallKernel` consumer, and a graph driven by a
    //! constant bias input.
    //!
    //! All tests in this module require the `prepared-plan` cargo
    //! feature to be enabled; the helper they exercise is gated on the
    //! same feature.

    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::prepared::{prepare_executable_plan, validate_prepared_against_plan};
    use crate::ir::builder::GraphBuilder;
    use crate::ir::{DimExpr, IrDType};

    fn f32_data(values: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice(bytes).to_vec()
    }

    #[test]
    fn execute_rejects_extra_and_oversized_inputs() {
        let g = GraphBuilder::new();
        let x = g.input(&[1, 4], IrDType::F32);
        let y = g.relu(&x);
        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&y], CpuBackend).expect("compile must succeed");
        let mut executor = GraphExecutor::new(CpuBackend);
        let input = f32_data(&[1.0, 2.0, 3.0, 4.0]);
        let extra = f32_data(&[5.0]);
        assert!(executor
            .execute(&compiled_graph, &mut plan, &memory_plan, &[&input, &extra],)
            .is_err());

        let oversized = f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(executor
            .execute(&compiled_graph, &mut plan, &memory_plan, &[&oversized])
            .is_err());
    }

    #[test]
    fn prepared_constant_preload_rejects_invalid_slots() {
        use crate::backend::prepared::{PackedWeightId, PreparedConstantArena};
        use crate::backend::BufferSlice;
        use std::collections::HashSet;

        let backend = CpuBackend;
        let arena_buffer = backend.allocate_arena(4);
        let empty_arena = PreparedConstantArena::new();
        let missing = preload_prepared_fp32_slot(
            &backend,
            &arena_buffer,
            &empty_arena,
            0,
            Some(PackedWeightId::new(9)),
            BufferSlice { offset: 0, size: 4 },
            "test constant",
            &mut HashSet::new(),
            1,
            4,
        );
        assert!(matches!(missing, Err(BackendError::Dispatch(_))));

        let mut constant_arena = PreparedConstantArena::new();
        let id = constant_arena.insert("weight", vec![1.0]);
        let out_of_bounds = preload_prepared_fp32_slot(
            &backend,
            &arena_buffer,
            &constant_arena,
            0,
            Some(id),
            BufferSlice { offset: 4, size: 4 },
            "test constant",
            &mut HashSet::new(),
            1,
            4,
        );
        assert!(matches!(out_of_bounds, Err(BackendError::Dispatch(_))));
    }

    /// `execute_prepared_fallback` returns **byte-identical** output to
    /// the regular `execute` path for a tiny graph whose compiled plan
    /// includes a `WriteConst` producer + a `CallKernel` consumer (the
    /// exact shape the order-mapping requirement is meant to cover).
    #[test]
    fn execute_prepared_fallback_matches_execute_on_add_with_constant() {
        // y = relu(a + b) where b is a graph constant (=> WriteConst
        // + CallKernel in the compiled plan). The plan is compiled
        // with a non-zero constant so the prepared-detector will see
        // a real WriteConst instruction.
        let g = GraphBuilder::new();
        let a = g.input(&[1, 4], IrDType::F32);
        let b_tt =
            crate::ir::TensorType::new(vec![DimExpr::Known(1), DimExpr::Known(4)], IrDType::F32);
        let b_bytes = f32_data(&[0.5, -0.5, 0.25, -0.25]);
        let b = g.constant(&b_bytes, b_tt);
        let s = g.add(&a, &b);
        let y = g.relu(&s);

        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&y], CpuBackend).expect("compile must succeed");
        let prepared = prepare_executable_plan(&plan);
        // The prepared plan must validate against the live plan before
        // the executor will accept it.
        validate_prepared_against_plan(&prepared, &plan)
            .expect("prepared plan must be consistent with the live plan");

        let input = f32_data(&[-1.0, 0.0, 2.0, 3.0]);

        // Regular path — captured for comparison.
        let regular = executor_run(
            &mut GraphExecutor::new(CpuBackend),
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&input],
        );

        // Fallback path — should be byte-identical.
        let fallback = executor_fallback(
            &mut GraphExecutor::new(CpuBackend),
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&input],
            &prepared,
        );

        assert_eq!(regular.len(), 1, "single output graph");
        assert_eq!(fallback.len(), 1, "single output graph");
        assert_eq!(
            regular[0], fallback[0],
            "fallback output bytes must match the regular path"
        );
        // y = relu([-1.0+0.5, 0.0-0.5, 2.0+0.25, 3.0-0.25])
        //   = relu([-0.5, -0.5, 2.25, 2.75])
        //   = [0, 0, 2.25, 2.75]
        let out = read_f32(&fallback[0]);
        assert_eq!(out, vec![0.0, 0.0, 2.25, 2.75]);
    }

    /// `execute_prepared_fallback` must be **re-runnable** — calling it
    /// twice on the same executor must produce the same output without
    /// any side-effect leakage between invocations.
    #[test]
    fn execute_prepared_fallback_is_repeatable() {
        let g = GraphBuilder::new();
        let x = g.input(&[1, 4], IrDType::F32);
        let y = g.relu(&x);

        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&y], CpuBackend).expect("compile must succeed");
        let prepared = prepare_executable_plan(&plan);

        let input = f32_data(&[-2.0, -1.0, 0.5, 1.5]);
        let mut executor = GraphExecutor::new(CpuBackend);
        let first = executor_fallback(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&input],
            &prepared,
        );
        let second = executor_fallback(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&input],
            &prepared,
        );
        assert_eq!(first, second);
    }

    /// `execute_prepared_fallback` must surface validation errors
    /// rather than silently desynchronising prepared metadata with the
    /// live plan. We forge a prepared plan whose length does not match
    /// the source plan, which `validate_prepared_against_plan` must
    /// reject with a `Dispatch` error.
    #[test]
    fn execute_prepared_fallback_rejects_inconsistent_prepared_plan() {
        use crate::backend::prepared::PreparedExecutablePlan;

        let g = GraphBuilder::new();
        let x = g.input(&[1, 4], IrDType::F32);
        let y = g.relu(&x);

        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&y], CpuBackend).expect("compile must succeed");

        // Forge a prepared plan that is *empty* — guaranteed not to
        // match the real plan's length. We copy the arena size from
        // the live plan so the only mismatch is the empty instruction
        // list. Default::default() leaves the private `constant_arena`
        // field at `None`, which is exactly what we want.
        let mut forged = PreparedExecutablePlan::default();
        forged.arena_size = plan.arena_size;

        let input = f32_data(&[1.0, 2.0, 3.0, 4.0]);
        let mut executor = GraphExecutor::new(CpuBackend);
        let err = executor
            .execute_prepared_fallback(&compiled_graph, &mut plan, &memory_plan, &[&input], &forged)
            .expect_err("mismatched prepared plan must be rejected");
        assert!(
            matches!(err, BackendError::Dispatch(_)),
            "expected Dispatch error, got {err:?}"
        );
    }

    /// `execute_prepared_fallback` should also work on a graph that
    /// has no `WriteConst` (e.g. a pure elementwise op) — the
    /// order-mapping check should not depend on the presence of
    /// static weights.
    #[test]
    fn execute_prepared_fallback_works_on_add_only_graph() {
        let g = GraphBuilder::new();
        let a = g.input(&[4], IrDType::F32);
        let b = g.input(&[4], IrDType::F32);
        let c = g.add(&a, &b);

        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&c], CpuBackend).expect("compile must succeed");
        let prepared = prepare_executable_plan(&plan);

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0]);
        let b_data = f32_data(&[10.0, 20.0, 30.0, 40.0]);
        let mut executor = GraphExecutor::new(CpuBackend);
        let out = executor_fallback(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&a_data, &b_data],
            &prepared,
        );
        assert_eq!(read_f32(&out[0]), vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Smoke test: a graph with a symbolic batch dim still works
    /// through the prepared fallback.
    #[test]
    fn execute_prepared_fallback_preserves_symbolic_shape_inputs() {
        let g = GraphBuilder::new();
        let batch = DimExpr::Symbol("N".into());
        let x = g.input_with_dims(&[batch.clone(), DimExpr::Known(4)], IrDType::F32);
        let y = g.relu(&x);

        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&y], CpuBackend).expect("compile must succeed");
        let prepared = prepare_executable_plan(&plan);

        // N = 2 → input bytes = 2*4*4 = 32
        let input = f32_data(&[-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut executor = GraphExecutor::new(CpuBackend);
        let out = executor_fallback(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&input],
            &prepared,
        );
        assert_eq!(
            read_f32(&out[0]),
            vec![0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    /// Verify that the static shape cache is populated after the first
    /// inference and produces byte-identical results on subsequent calls.
    #[test]
    fn static_shape_cache_produces_correct_repeated_results() {
        let g = GraphBuilder::new();
        let a = g.input(&[1, 4], IrDType::F32);
        let b = g.input(&[1, 4], IrDType::F32);
        let c = g.add(&a, &b);

        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&c], CpuBackend).expect("compile must succeed");
        let mut executor = GraphExecutor::new(CpuBackend);

        let a_data = f32_data(&[1.0, 2.0, 3.0, 4.0]);
        let b_data = f32_data(&[10.0, 20.0, 30.0, 40.0]);

        // First call — populates the cache.
        let first = executor_run(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&a_data, &b_data],
        );

        // Second call — should use the cached fast path.
        let second = executor_run(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&a_data, &b_data],
        );

        // Third call — another cache hit.
        let third = executor_run(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&a_data, &b_data],
        );

        assert_eq!(first.len(), 1);
        assert_eq!(
            first[0], second[0],
            "second call (cache path) must match first"
        );
        assert_eq!(
            second[0], third[0],
            "third call (cache path) must match first"
        );
        assert_eq!(
            read_f32(&first[0]),
            vec![11.0, 22.0, 33.0, 44.0],
            "correctness: 1+10=11, 2+20=22, 3+30=33, 4+40=44"
        );

        // The cache must be populated now.
        assert!(
            executor.static_shape_cache.is_some(),
            "static shape cache must be populated for a Known-only graph"
        );
    }

    /// Verify that the static shape cache is NOT populated when the graph
    /// has symbolic (non-Known) dims.
    #[test]
    fn static_shape_cache_skipped_for_dynamic_graph() {
        let g = GraphBuilder::new();
        let batch = DimExpr::Symbol("N".into());
        let x = g.input_with_dims(&[batch, DimExpr::Known(4)], IrDType::F32);
        let y = g.relu(&x);

        let (mut plan, memory_plan, compiled_graph) =
            g.compile(&[&y], CpuBackend).expect("compile must succeed");
        let mut executor = GraphExecutor::new(CpuBackend);

        let input = f32_data(&[-1.0, -2.0, 3.0, 4.0]);
        let _ = executor_run(
            &mut executor,
            &compiled_graph,
            &mut plan,
            &memory_plan,
            &[&input],
        );

        // Dynamic-shape graph should NOT populate the cache.
        assert!(
            executor.static_shape_cache.is_none(),
            "static shape cache must NOT be populated for a graph with Symbol dims"
        );
    }

    // ── helpers ────────────────────────────────────────────────

    fn executor_run(
        executor: &mut GraphExecutor<CpuBackend>,
        graph: &crate::ir::ComputeGraph,
        plan: &mut crate::backend::ExecutablePlan,
        memory_plan: &crate::compiler::plan::MemoryPlan,
        inputs: &[&[u8]],
    ) -> Vec<Vec<u8>> {
        executor
            .execute(graph, plan, memory_plan, inputs)
            .expect("regular execute must succeed")
    }

    fn executor_fallback(
        executor: &mut GraphExecutor<CpuBackend>,
        graph: &crate::ir::ComputeGraph,
        plan: &mut crate::backend::ExecutablePlan,
        memory_plan: &crate::compiler::plan::MemoryPlan,
        inputs: &[&[u8]],
        prepared: &crate::backend::prepared::PreparedExecutablePlan,
    ) -> Vec<Vec<u8>> {
        executor
            .execute_prepared_fallback(graph, plan, memory_plan, inputs, prepared)
            .expect("prepared fallback execute must succeed")
    }
}

#[cfg(test)]
mod execution_storage_size_tests {
    use super::*;

    #[test]
    fn unsigned_packed_types_use_exact_packed_storage_size() {
        let u4 = IrDType::U4Scaled {
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
        };
        let u8 = IrDType::U8Scaled {
            scales: vec![1.0],
            dequant_offsets: vec![0.0],
        };

        for numel in [1, 7, 8, 9, 17, 64] {
            assert_eq!(
                checked_execution_storage_size(&u4, numel).unwrap(),
                u4.packed_byte_size(numel)
            );
            assert_eq!(
                checked_execution_storage_size(&u8, numel).unwrap(),
                u8.packed_byte_size(numel)
            );
        }
    }

    #[cfg(feature = "prepared-plan")]
    #[test]
    fn persistent_dispatch_validates_arena_before_view_execution() {
        use crate::backend::prepared::PersistentPreparedWeights;
        use std::sync::Arc;

        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![],
            arena_size: 4,
            levels: vec![],
        };
        let arena = backend.try_allocate_arena(0).unwrap();
        let mut view = PersistentPreparedWeights::new();
        assert!(view.insert((0, 4), Arc::new(vec![1.0])));
        assert!(backend
            .dispatch_with_persistent_view(&plan, &arena, &ShapeEnv::new(), Some(&view))
            .is_err());

        let malformed_conv = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "conv2d".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![0; 15],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(12).unwrap();
        let mut view = PersistentPreparedWeights::new();
        assert!(view.insert((4, 4), Arc::new(vec![1.0])));
        assert!(backend
            .dispatch_with_persistent_view(&malformed_conv, &arena, &ShapeEnv::new(), Some(&view),)
            .is_err());

        let malformed_matmul = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "matmul".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![1, 2, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        assert!(
            backend
                .dispatch_with_persistent_view(
                    &malformed_matmul,
                    &arena,
                    &ShapeEnv::new(),
                    Some(&view),
                )
                .is_err()
        );
    }

    #[test]
    fn pad_places_input_at_low_padding_offset() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "pad_f32".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 8)],
                output_slice: crate::backend::BufferSlice::new(8, 20),
                secondary_output_slice: None,
                params: vec![1, 2, 1, 2],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 28,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(28).unwrap();
        backend
            .try_write_arena(&arena, 0, bytemuck::cast_slice(&[3.0f32, 4.0]))
            .unwrap();
        backend.dispatch(&plan, &arena, &ShapeEnv::new()).unwrap();
        let bytes = backend.try_read_arena(&arena, 8, 20).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, f32>(&bytes),
            &[0.0, 3.0, 4.0, 0.0, 0.0]
        );
    }

    #[test]
    fn cast_dispatch_rejects_missing_width_metadata() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "cast".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 8),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(12).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn to_f16_rejects_partial_output_scalar() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "to_f16".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 1),
                secondary_output_slice: None,
                params: vec![1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 5,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(5).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn erf_dispatch_rejects_partial_output_contract() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "erf_f32".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 8),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(12).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn rmsprop_dispatch_rejects_nonpositive_epsilon() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "rmsprop_update_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                    crate::backend::BufferSlice::new(8, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(12, 4),
                secondary_output_slice: None,
                params: vec![
                    0.1f32.to_bits() as usize,
                    0.9f32.to_bits() as usize,
                    0.0f32.to_bits() as usize,
                ],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 16,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(16).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn lion_dispatch_rejects_missing_beta() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "lion_update_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                    crate::backend::BufferSlice::new(8, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(12, 4),
                secondary_output_slice: None,
                params: vec![0.1f32.to_bits() as usize, 0.9f32.to_bits() as usize],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 16,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(16).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn muon_dispatch_rejects_invalid_beta() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "muon_update_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                    crate::backend::BufferSlice::new(8, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(12, 4),
                secondary_output_slice: None,
                params: vec![
                    0.1f32.to_bits() as usize,
                    1.0f32.to_bits() as usize,
                    0.0f32.to_bits() as usize,
                ],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 16,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(16).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn gradient_scale_rejects_nonfinite_scale() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "gradient_scale".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1, f32::NAN.to_bits() as usize],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn gradient_f8_dequantization_rejects_truncated_words() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "dequantize_gradient_f8x4r_to_f32".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 20),
                secondary_output_slice: None,
                params: vec![5],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 24,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(24).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn sgd_dispatch_rejects_missing_learning_rate() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "sgd_update_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(12).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn where_rejects_empty_condition_for_nonempty_output() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "where_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 0),
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(12).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn flip_respects_selected_dimensions() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "flip".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 16)],
                output_slice: crate::backend::BufferSlice::new(16, 16),
                secondary_output_slice: None,
                params: vec![1, 0, 2, 2],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 32,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(32).unwrap();
        backend
            .try_write_arena(&arena, 0, bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]))
            .unwrap();
        backend.dispatch(&plan, &arena, &ShapeEnv::new()).unwrap();
        let bytes = backend.try_read_arena(&arena, 16, 16).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, f32>(&bytes),
            &[3.0, 4.0, 1.0, 2.0]
        );
    }

    #[test]
    fn cumsum_respects_selected_dimension() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "cumsum".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 16)],
                output_slice: crate::backend::BufferSlice::new(16, 16),
                secondary_output_slice: None,
                params: vec![2, 2, 2, 1, 0, 0],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 32,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(32).unwrap();
        backend
            .try_write_arena(&arena, 0, bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]))
            .unwrap();
        backend.dispatch(&plan, &arena, &ShapeEnv::new()).unwrap();
        let bytes = backend.try_read_arena(&arena, 16, 16).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, f32>(&bytes),
            &[1.0, 3.0, 3.0, 7.0]
        );
    }

    #[test]
    fn repeat_uses_dimension_aware_tiling() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "repeat".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 16)],
                output_slice: crate::backend::BufferSlice::new(16, 32),
                secondary_output_slice: None,
                params: vec![2, 2, 2, 2, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 48,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(48).unwrap();
        backend
            .try_write_arena(&arena, 0, bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]))
            .unwrap();
        backend.dispatch(&plan, &arena, &ShapeEnv::new()).unwrap();
        let bytes = backend.try_read_arena(&arena, 16, 32).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, f32>(&bytes),
            &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn adaptive_pool_rejects_zero_input_height() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "adaptive_avg_pool2d".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1, 1, 0, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn bilinear_upsample_rejects_zero_input_height() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "upsample_bilinear2d".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 16),
                secondary_output_slice: None,
                params: vec![2, 2, 0, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 20,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(20).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn nearest_upsample_rejects_zero_scale() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "upsample_nearest2d".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 0),
                secondary_output_slice: None,
                params: vec![0, 1, 1, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 4,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(4).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn layer_norm_applies_affine_parameters() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "norm_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 8),
                    crate::backend::BufferSlice::new(8, 8),
                    crate::backend::BufferSlice::new(16, 8),
                ],
                output_slice: crate::backend::BufferSlice::new(24, 8),
                secondary_output_slice: None,
                params: vec![1e-5f32.to_bits() as usize, 0],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 32,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(32).unwrap();
        backend
            .try_write_arena(&arena, 0, bytemuck::cast_slice(&[1.0f32, 3.0]))
            .unwrap();
        backend
            .try_write_arena(&arena, 8, bytemuck::cast_slice(&[2.0f32, 3.0]))
            .unwrap();
        backend
            .try_write_arena(&arena, 16, bytemuck::cast_slice(&[1.0f32, -1.0]))
            .unwrap();
        backend.dispatch(&plan, &arena, &ShapeEnv::new()).unwrap();
        let bytes = backend.try_read_arena(&arena, 24, 8).unwrap();
        let output = bytemuck::cast_slice::<_, f32>(&bytes);
        assert!((output[0] + 1.0).abs() < 1e-4, "output={output:?}");
        assert!((output[1] - 2.0).abs() < 1e-4, "output={output:?}");
    }

    #[test]
    fn batch_norm_rejects_nonpositive_epsilon() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "norm_f32".into(),
                input_slices: vec![],
                output_slice: crate::backend::BufferSlice::new(0, 0),
                secondary_output_slice: None,
                params: vec![0.0f32.to_bits() as usize, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 0,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(0).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn biasadd_rejects_zero_channel_stride() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "biasadd".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![0],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(12).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn softmax_rejects_zero_stride() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "softmax".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1, 0],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn pow_rejects_empty_exponent_storage() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "pow_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 0),
                ],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn embedding_rejects_out_of_range_indices() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "embedding".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 8),
                    crate::backend::BufferSlice::new(8, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(12, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 16,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(16).unwrap();
        backend
            .try_write_arena(&arena, 8, &2.0f32.to_le_bytes())
            .unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn split_topk_rejects_k_larger_than_input() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "topk_values".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 8),
                secondary_output_slice: None,
                params: vec![2, 0],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(12).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn fused_topk_requires_indices_output() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "topk_fused".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn argmax_rejects_zero_reduction_dimension() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "argmax".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(8, 8),
                secondary_output_slice: None,
                params: vec![0, 0, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 16,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(16).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn scalar_f32_dispatch_rejects_empty_scalar_storage() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "add_scalar_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 0),
                ],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn binary_f32_dispatch_rejects_empty_operand_storage() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "add_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 0),
                    crate::backend::BufferSlice::new(0, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn unary_f32_dispatch_rejects_missing_input() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "relu_f32".into(),
                input_slices: vec![],
                output_slice: crate::backend::BufferSlice::new(0, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 4,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(4).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn rms_norm_rejects_empty_weight_storage() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "rms_norm".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 0),
                ],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1e-5f32.to_bits() as usize],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn prelu_rejects_empty_weight_storage() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "prelu".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 0),
                ],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(8).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn transposed_convolution_rejects_invalid_geometry() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "conv_transpose2d".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![1, 0, 1, 0, 1, 1, 1],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(12).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn fused_residual_norm_rejects_invalid_metadata() {
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "fused_residual_add_layer_norm".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 16),
                    crate::backend::BufferSlice::new(16, 16),
                ],
                output_slice: crate::backend::BufferSlice::new(32, 16),
                secondary_output_slice: None,
                params: vec![0.0f32.to_bits() as usize, 4],
                param_dims: None,
                node_id: Some(0),
                weight_meta: None,
            }],
            arena_size: 48,
            levels: vec![0],
        };
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(48).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn activation_quantization_validates_and_populates_each_channel() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "quantize_activations".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 16)],
                output_slice: crate::backend::BufferSlice::new(16, 28),
                secondary_output_slice: None,
                params: vec![
                    4,
                    1,
                    2,
                    1.0f32.to_bits() as usize,
                    1.0f32.to_bits() as usize,
                    0.0f32.to_bits() as usize,
                    0.0f32.to_bits() as usize,
                ],
                param_dims: None,
                node_id: None,
                weight_meta: None,
            }],
            arena_size: 44,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        backend
            .try_write_arena(&arena, 0, bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]))
            .unwrap();
        backend.dispatch(&plan, &arena, &ShapeEnv::new()).unwrap();
        assert_eq!(
            backend.try_read_arena(&arena, 40, 4).unwrap(),
            vec![1, 2, 3, 4]
        );

        let mut malformed = plan;
        if let Instruction::CallKernel { output_slice, .. } = &mut malformed.instructions[0] {
            output_slice.size = 27;
        }
        assert!(backend
            .dispatch(&malformed, &arena, &ShapeEnv::new())
            .is_err());
    }

    #[test]
    fn f16_to_f32_rejects_partial_scalars() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "to_f32".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 3)],
                output_slice: crate::backend::BufferSlice::new(4, 8),
                secondary_output_slice: None,
                params: vec![],
                param_dims: None,
                node_id: None,
                weight_meta: None,
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn adam_dispatch_rejects_malformed_storage_contracts() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "adam_update_f32".into(),
                input_slices: vec![],
                output_slice: crate::backend::BufferSlice::new(0, 4),
                secondary_output_slice: None,
                params: vec![
                    0.001f32.to_bits() as usize,
                    0.9f32.to_bits() as usize,
                    0.999f32.to_bits() as usize,
                    1e-8f32.to_bits() as usize,
                    1,
                ],
                param_dims: None,
                node_id: None,
                weight_meta: None,
            }],
            arena_size: 4,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());

        let runtime_step_plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "adamw_update_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                    crate::backend::BufferSlice::new(8, 4),
                    crate::backend::BufferSlice::new(12, 4),
                    crate::backend::BufferSlice::new(16, 8),
                ],
                output_slice: crate::backend::BufferSlice::new(24, 4),
                secondary_output_slice: None,
                params: vec![
                    0.001f32.to_bits() as usize,
                    0.9f32.to_bits() as usize,
                    0.999f32.to_bits() as usize,
                    1e-8f32.to_bits() as usize,
                    0.01f32.to_bits() as usize,
                ],
                param_dims: None,
                node_id: None,
                weight_meta: None,
            }],
            arena_size: 28,
            levels: vec![0],
        };
        let arena = backend
            .try_allocate_arena(runtime_step_plan.arena_size)
            .unwrap();
        assert!(backend
            .dispatch(&runtime_step_plan, &arena, &ShapeEnv::new())
            .is_err());
    }

    #[test]
    fn packed_dequantization_rejects_implicit_storage_width() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "dequantize_kernel".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1, 1, 3, 1, 1.0f32.to_bits() as usize, 0],
                param_dims: None,
                node_id: None,
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn quantized_matmul_rejects_truncated_activation_metadata() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "matmul_i4_i8".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![1, 1, 1],
                param_dims: None,
                node_id: None,
                weight_meta: Some(std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                    bit_width: 4,
                    scales: vec![1.0],
                    dequant_offsets: vec![0.0],
                    shape: vec![1, 1],
                    quant_block_size: 0,
                    codebooks: vec![],
                })),
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn expand_rejects_invalid_broadcast_metadata() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "expand_f32".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 8),
                    crate::backend::BufferSlice::new(20, 0),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 12),
                secondary_output_slice: None,
                params: vec![1, 2, 3],
                param_dims: None,
                node_id: None,
                weight_meta: None,
            }],
            arena_size: 20,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn cpu_arena_transfers_reject_invalid_ranges() {
        let backend = crate::backend::cpu::CpuBackend;
        let arena = backend.try_allocate_arena(4).unwrap();
        assert!(backend.try_write_arena(&arena, 5, &[1]).is_err());
        assert!(backend.try_write_arena(&arena, usize::MAX, &[1]).is_err());
        assert!(backend.try_read_arena(&arena, 5, 1).is_err());
        assert!(backend.try_read_arena(&arena, usize::MAX, 1).is_err());

        let malformed_fill = ExecutablePlan {
            instructions: vec![Instruction::Fill {
                dst: crate::backend::BufferSlice::new(0, 3),
                value: 0.0,
            }],
            arena_size: 3,
            levels: vec![0],
        };
        assert!(malformed_fill.validate().is_err());

        let undersized_plan = ExecutablePlan {
            instructions: vec![],
            arena_size: 4,
            levels: vec![],
        };
        let undersized_arena = backend.try_allocate_arena(3).unwrap();
        assert!(backend
            .dispatch(&undersized_plan, &undersized_arena, &ShapeEnv::new())
            .is_err());
    }

    #[test]
    fn activation_dequantization_rejects_truncated_metadata() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "dequantize_activations".into(),
                input_slices: vec![crate::backend::BufferSlice::new(0, 4)],
                output_slice: crate::backend::BufferSlice::new(4, 4),
                secondary_output_slice: None,
                params: vec![1, 0],
                param_dims: None,
                node_id: None,
                weight_meta: None,
            }],
            arena_size: 8,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn quantized_conv_rejects_truncated_activation_metadata() {
        let backend = crate::backend::cpu::CpuBackend;
        let plan = ExecutablePlan {
            instructions: vec![Instruction::CallKernel {
                kernel_name: "conv2d_i4_i8".into(),
                input_slices: vec![
                    crate::backend::BufferSlice::new(0, 4),
                    crate::backend::BufferSlice::new(4, 4),
                ],
                output_slice: crate::backend::BufferSlice::new(8, 4),
                secondary_output_slice: None,
                params: vec![1, 0, 1, 1, 1, 1, 1, 1, 1],
                param_dims: None,
                node_id: None,
                weight_meta: Some(std::sync::Arc::new(crate::backend::QuantizedWeightMeta {
                    bit_width: 4,
                    scales: vec![1.0],
                    dequant_offsets: vec![0.0],
                    shape: vec![1, 1, 1, 1],
                    quant_block_size: 0,
                    codebooks: vec![],
                })),
            }],
            arena_size: 12,
            levels: vec![0],
        };
        let arena = backend.try_allocate_arena(plan.arena_size).unwrap();
        assert!(backend.dispatch(&plan, &arena, &ShapeEnv::new()).is_err());
    }

    #[test]
    fn checked_storage_sizes_reject_overflow() {
        assert!(IrDType::F32.try_packed_byte_size(usize::MAX).is_none());
        assert!(checked_execution_storage_size(&IrDType::I8, usize::MAX).is_err());
        assert!(crate::backend::cpu::CpuBackend
            .try_allocate_arena(usize::MAX)
            .is_err());
    }

    #[test]
    fn native_types_use_logical_element_size() {
        assert_eq!(
            checked_execution_storage_size(&IrDType::F32, 17).unwrap(),
            68
        );
        assert_eq!(
            checked_execution_storage_size(&IrDType::I64, 17).unwrap(),
            136
        );
    }

    #[test]
    fn integer_inference_requires_calibration() {
        let executor = GraphExecutor::new(crate::backend::cpu::CpuBackend);
        let error = executor
            .compile_with_target(
                ComputeGraph::new(),
                CompileTarget::IntegerInference(QuantTarget::U8),
                None,
            )
            .expect_err("integer inference without calibration must fail");
        assert!(error
            .to_string()
            .contains("requires activation calibration"));
    }

    #[test]
    fn unsupported_mixed_precision_target_is_reported() {
        let executor = GraphExecutor::new(crate::backend::cpu::CpuBackend);
        let error = executor
            .compile_with_target(
                ComputeGraph::new(),
                CompileTarget::TrainingMixedPrecision {
                    compute: crate::types::ScalarType::F16,
                    accumulator: crate::types::ScalarType::F32,
                },
                None,
            )
            .expect_err("unsupported target must fail explicitly");
        assert!(error.to_string().contains("not implemented"));
    }
}
