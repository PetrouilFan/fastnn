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
use crate::backend::{
    Backend, BackendError, ExecutablePlan, Instruction, MemoryPlan, ProfileEntry,
};
use crate::compiler::passes::training::{inject_optimizer, TrainConfig};
use crate::compiler::passes::{
    activation_quantization, calibration, dead_code_elimination, memory_planning, operator_fusion,
    quantization, shape_inference,
};
use crate::ir::node::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, ShapeEnv};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

/// Target weight dtype for compilation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightDtype {
    /// No quantization — keep f32 weights as-is.
    F32,
    /// Integer 4-bit symmetric quantization (I4x8).
    I4,
    /// Integer 8-bit symmetric quantization (I8x4/U8).
    I8,
    /// Float 8-bit E4M3 (F8x4).
    F8x4,
    /// Float 8-bit E5M2 (F8x4R).
    F8x4R,
    /// Float 4-bit E2M1 (F4x8).
    F4x8,
    /// Integer 4-bit codebook quantization (I4x8).
    I4Codebook,
}

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
                resolve_shape(&node.output_type.shape, &shape_env).map_err(|e| {
                    BackendError::Dispatch(format!("output node {}: {e}", output_node_id))
                })?;
            let actual_numel: usize = resolved_shape.iter().map(|&v| v as usize).product();
            let computed = match &node.output_type.dtype {
                IrDType::I4 { .. }
                | IrDType::U8 { .. }
                | IrDType::F8 { .. }
                | IrDType::F8R { .. }
                | IrDType::F4 { .. } => node.output_type.dtype.packed_byte_size(actual_numel),
                IrDType::I8 => actual_numel + 8,
                _ => actual_numel * node.output_type.dtype.byte_size(),
            };
            (computed, resolved_shape)
        } else {
            (slot.size, vec![])
        };
        let data = backend.read_arena(arena, slot.offset, actual_size);
        outputs.push(data);
    }
    Ok(outputs)
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
    /// For FP packed types (F8/F8R/F4), use [`compile_with_weight_dtype`]
    /// instead. This method only supports integer quantization (I4/I8).
    pub fn compile_with_plan_and_quantize(
        &self,
        graph: ComputeGraph,
        quantize: Option<u8>,
        calib_data: Option<calibration::CalibrationData>,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        let weight_dtype = match quantize {
            Some(4) => WeightDtype::I4,
            Some(8) => WeightDtype::I8,
            None => WeightDtype::F32,
            Some(other) => {
                return Err(BackendError::Compilation(format!(
                    "Unsupported quantization bit width: {}. Supported values: 4, 8",
                    other
                )))
            }
        };
        self.compile_with_weight_dtype(graph, weight_dtype, calib_data)
    }

    /// Compile a graph with an explicit [`WeightDtype`] target, including FP
    /// packed types (F8x4, F8x4R, F4x8).
    ///
    /// Supports all variants of `WeightDtype`:
    /// - `F32`: no weight quantization
    /// - `I4` / `I8`: integer symmetric quantization (backed by
    ///   [`quantize_weights`])
    /// - `F8x4` / `F8x4R` / `F4x8`: FP packed quantization (backed by
    ///   [`quantize_weights_fp`])
    ///
    /// If `calib_data` is provided, per-tensor/per-channel activation scales
    /// will be applied after weight quantization (weight-only quant only uses
    /// the weight quant step; activations remain f32).
    ///
    /// Takes ownership of `graph` to avoid an unnecessary deep clone.
    pub fn compile_with_weight_dtype(
        &self,
        mut graph: ComputeGraph,
        weight_dtype: WeightDtype,
        calib_data: Option<calibration::CalibrationData>,
    ) -> Result<(ExecutablePlan, MemoryPlan, ComputeGraph), BackendError> {
        let do_weight_quant = weight_dtype != WeightDtype::F32;

        // ── Phase 1: Shape inference ──────────────────────────────────────
        shape_inference::infer_shapes(&mut graph)
            .map_err(|e| BackendError::Compilation(format!("shape inference: {e}")))?;

        // ── Phase 1.5: Constant folding + arithmetic simplification ──────
        // Evaluate fully-constant nodes (Shape, broadcast constants, etc.)
        // and simplify trivial arithmetic (x*1.0, x+0.0, Neg(Neg(x)), etc.)
        // before fusion to reduce the number of nodes the fusion pass sees.
        {
            use crate::compiler::passes::{arithmetic_simplify, constant_folding};
            let folded = constant_folding::constant_fold(&mut graph);
            let simplified = arithmetic_simplify::arithmetic_simplify(&mut graph);
            if folded + simplified > 0 {
                eprintln!(
                    "[fastnn] pre-fusion optimization: folded {folded} constant nodes, simplified {simplified} arithmetic"
                );
            }
        }

        // ── Phase 2: Operator fusion ──────────────────────────────────────
        operator_fusion::fuse_operators(&mut graph)
            .map_err(|e| BackendError::Compilation(format!("operator fusion: {e}")))?;

        // ── Phase 2.5: Dead code elimination after fusion ────────────────
        // Fusion creates dead intermediate nodes (e.g. the original Add and
        // Relu when fusing MatMul+Add+Relu). Eliminate them before
        // quantization to avoid quantizing dead nodes.
        dead_code_elimination::eliminate_dead_code(&mut graph);

        // ── Phase 2.5: Quantization (optional) ───────────────────────────
        // Handle three cases:
        // 1. weight_dtype != F32: quantize weights.
        //    Activation quantization is only applied when calibration data is provided.
        // 2. weight_dtype == F32, calib_data=Some(_): re-quantize activations only (weights already quantized)
        // 3. weight_dtype == F32, calib_data=None: no quantization
        let do_quantize = do_weight_quant || calib_data.is_some();
        if do_quantize {
            // Quantize weights if requested
            if do_weight_quant {
                use quantization::FpDtype;
                match weight_dtype {
                    WeightDtype::I4 => {
                        quantization::quantize_weights(&mut graph, 4, None)
                            .map_err(|e| BackendError::Compilation(format!("quantization: {e}")))?;
                    }
                    WeightDtype::I8 => {
                        quantization::quantize_weights(&mut graph, 8, None)
                            .map_err(|e| BackendError::Compilation(format!("quantization: {e}")))?;
                    }
                    WeightDtype::F8x4 => {
                        quantization::quantize_weights_fp(&mut graph, &FpDtype::F8x4, None)
                            .map_err(|e| BackendError::Compilation(format!("fp quant: {e}")))?;
                    }
                    WeightDtype::F8x4R => {
                        quantization::quantize_weights_fp(&mut graph, &FpDtype::F8x4R, None)
                            .map_err(|e| BackendError::Compilation(format!("fp quant: {e}")))?;
                    }
                    WeightDtype::F4x8 => {
                        quantization::quantize_weights_fp(&mut graph, &FpDtype::F4x8, None)
                            .map_err(|e| BackendError::Compilation(format!("fp quant: {e}")))?;
                    }
                    WeightDtype::I4Codebook => {
                        quantization::quantize_weights_fp(&mut graph, &FpDtype::I4Codebook, None)
                            .map_err(|e| BackendError::Compilation(format!("fp quant: {e}")))?;
                    }
                    WeightDtype::F32 => {}
                }
            }

            // After quantizing weights, wrap any optimizer ops that now have
            // quantized weight inputs with Dequantize/Quantize.
            quantization::wrap_quantized_optimizer(&mut graph)
                .map_err(|e| BackendError::Compilation(format!("optimizer wrapping: {e}")))?;

            // Insert QuantizeActivations before MatMul/Conv2d activation inputs only
            // when explicit calibration data is provided. Without calibrated scales,
            // activation quantization produces garbage outputs and adds unnecessary
            // Q/DQ overhead that makes weight-only quantized inference slower than FP32.
            if let Some(calib) = calib_data {
                activation_quantization::quantize_activations_with_calibration(&mut graph, &calib)
                    .map_err(|e| {
                        BackendError::Compilation(format!("activation quantization: {e}"))
                    })?;
            }

            // Remove redundant QuantizeActivations → DequantizeActivations round-trips
            // that were inserted by activation quantization or the optimizer wrapper.
            // This pass is dead-code in auto_cast.rs; wire it into the main pipeline.
            crate::compiler::passes::prune_qdq_pairs::prune_qdq_pairs(&mut graph)
                .map_err(|e| BackendError::Compilation(format!("prune qdq pairs: {e}")))?;
        }

        // ── Phase 3: Dead code elimination ────────────────────────────────
        let _removed = dead_code_elimination::eliminate_dead_code(&mut graph);

        // ── Phase 4: Memory planning ──────────────────────────────────────
        let memory_plan = memory_planning::plan_memory(&graph)
            .map_err(|e| BackendError::Compilation(format!("memory planning: {e}")))?;

        // ── Phase 5: Backend compilation ──────────────────────────────────
        let plan = self.backend.compile(&graph, &memory_plan)?;

        if std::env::var("FASTNN_DUMP_INSTRS").is_ok() {
            eprintln!(
                "FASTNN_DUMP_INSTRS: total instructions={}",
                plan.instructions.len()
            );
            let mut conv2d_quant_u4 = 0usize;
            let mut conv2d_quant_u8 = 0usize;
            let mut conv2d_fp32 = 0usize;
            let mut other_kernels: Vec<(String, Option<String>)> = Vec::new();
            for (i, instr) in plan.instructions.iter().enumerate() {
                if let Instruction::CallKernel {
                    kernel_name,
                    weight_meta,
                    ..
                } = instr
                {
                    let meta_info = weight_meta
                        .as_ref()
                        .map(|m| format!("bw={} scales={}", m.bit_width, m.scales.len()))
                        .unwrap_or_default();
                    eprintln!("  INSTR[{}] kernel={} {}", i, kernel_name, meta_info);
                    if kernel_name.starts_with("conv2d_i4") {
                        conv2d_quant_u4 += 1;
                    } else if kernel_name.starts_with("conv2d_i8") {
                        conv2d_quant_u8 += 1;
                    } else if kernel_name.starts_with("conv2d") {
                        conv2d_fp32 += 1;
                    } else if kernel_name.starts_with("quantize_activations")
                        || kernel_name.starts_with("dequantize_activations")
                    {
                        other_kernels.push((kernel_name.clone(), Some(meta_info)));
                    } else {
                        other_kernels.push((kernel_name.clone(), Some(meta_info)));
                    }
                }
            }
            eprintln!(
                "  SUMMARY: conv2d_u4={} conv2d_u8={} conv2d_fp32={} other={}",
                conv2d_quant_u4,
                conv2d_quant_u8,
                conv2d_fp32,
                other_kernels.len()
            );
            for (name, meta) in &other_kernels {
                eprintln!("    OTHER: {} {}", name, meta.as_deref().unwrap_or(""));
            }
        }

        if std::env::var("FASTNN_DUMP_GRAPH").is_ok() {
            use std::collections::BTreeMap;
            let mut counts: BTreeMap<String, usize> = BTreeMap::new();
            let mut dtypes: BTreeMap<String, usize> = BTreeMap::new();
            for n in &graph.nodes {
                counts.entry(format!("{:?}", n.opcode)).or_default(); // just trigger
                *counts.entry(format!("{:?}", n.opcode)).or_default() += 1;
                *dtypes
                    .entry(format!("{:?}", n.output_type.dtype))
                    .or_default() += 1;
            }
            eprintln!("FASTNN_DUMP: nodes={}", graph.nodes.len());
            for (k, v) in &counts {
                if v > &0 {
                    eprintln!("  OP {k}: {v}");
                }
            }
            for (k, v) in &dtypes {
                if v > &0 {
                    eprintln!("  DT {k}: {v}");
                }
            }
        }

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

                let tightened_memory_plan = memory_plan.tighten(graph, &shape_env);
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
                if let Some(node) = graph.get_node(node_id) {
                    let needed =
                        if let Ok(shape) = resolve_shape(&node.output_type.shape, &shape_env) {
                            let numel: u64 = shape.iter().product();
                            let raw = numel as usize * node.output_type.dtype.byte_size();
                            match &node.output_type.dtype {
                                IrDType::I4 { .. }
                                | IrDType::U8 { .. }
                                | IrDType::F8 { .. }
                                | IrDType::F8R { .. }
                                | IrDType::F4 { .. } => {
                                    node.output_type.dtype.packed_byte_size(numel as usize)
                                }
                                IrDType::I8 => numel as usize + 8,
                                _ => raw,
                            }
                        } else {
                            continue;
                        };
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
            .map_or(false, |(cap, _)| *cap >= arena_size);
        if !enough_capacity {
            self.cached_arena = Some((arena_size, self.backend.allocate_arena(arena_size)));
        }
        let arena = &self.cached_arena.as_ref().unwrap().1;

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
            self.backend.write_arena(arena, slot.offset, input_bytes);
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
        gradient_quantization::quantize_gradients(&mut combined_graph);

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
            let tightened_mp = memory_plan.tighten(&final_graph, shape_env);
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
        let arena = self.backend.allocate_arena(plan.arena_size);

        // Write param data
        for (i, data) in param_data.iter().enumerate() {
            let slot = memory_plan.slots.get(&params[i]).ok_or_else(|| {
                BackendError::Compilation(format!("param {} slot not found", params[i]))
            })?;
            self.backend.write_arena(&arena, slot.offset, data);
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
                self.backend.write_arena(&arena, slot.offset, &init_data);
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
            self.backend.write_arena(&self.arena, off, data);
        }

        // 3. Dispatch the full train-step graph
        self.backend
            .dispatch(&self.plan, &self.arena, &self.shape_env)
            .map_err(|e| BackendError::Dispatch(format!("train_step dispatch: {e}")))?;

        // 4. Read loss immediately (before any post-dispatch writes)
        let loss_raw = self
            .backend
            .read_arena(&self.arena, self.loss_slot_offset, 4);
        let loss_arr: [u8; 4] = loss_raw
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
                        let t_bytes = self.backend.read_arena(&self.arena, t_slice.offset, 8);
                        let t_arr: [u8; 8] = t_bytes[..8]
                            .try_into()
                            .map_err(|_| BackendError::Dispatch("invalid t step value".into()))?;
                        let t_val = u64::from_le_bytes(t_arr);
                        let new_t = t_val.checked_add(1).ok_or_else(|| {
                            BackendError::Dispatch(
                                "train_step: AdamW step overflow (t > u64::MAX)".into(),
                            )
                        })?;
                        self.backend
                            .write_arena(&self.arena, t_slice.offset, &new_t.to_le_bytes());
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
                    if let Some(slot) = tightened_memory_plan.slots.get(nid) {
                        output_slice.offset = slot.offset;
                        output_slice.size = slot.size;
                    }
                    // ── Update secondary_output_slice ──
                    if let Some(sec_slice) = secondary_output_slice.as_mut() {
                        if let Some(slot) = tightened_memory_plan.secondary_slots.get(&(*nid, 1)) {
                            sec_slice.offset = slot.offset;
                            sec_slice.size = slot.size;
                        }
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
                    if let Some(node) = graph.get_node(*nid) {
                        let mut slice_iter = input_slices.iter_mut();
                        for &input_nid in &node.inputs {
                            if tightened_memory_plan.slots.contains_key(&input_nid) {
                                if let Some(slice) = slice_iter.next() {
                                    if let Some(slot) = tightened_memory_plan.slots.get(&input_nid)
                                    {
                                        slice.offset = slot.offset;
                                        slice.size = slot.size;
                                    }
                                }
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
                    crate::ir::node::SYMBOL_DIM_MAX.load(std::sync::atomic::Ordering::Relaxed);
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
) -> Result<(), BackendError> {
    use crate::backend::prepared::PackedWeightKind;

    let Some(id) = id else {
        return Ok(());
    };
    if id.kind != PackedWeightKind::Fp32 {
        return Ok(());
    }
    let Some(values) = constant_arena.get(id) else {
        return Ok(());
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
    backend.write_arena(arena, slot.offset, bytes);
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
    use crate::ir::node::{DimExpr, IrDType};

    fn f32_data(values: &[f32]) -> Vec<u8> {
        bytemuck::cast_slice(values).to_vec()
    }

    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytemuck::cast_slice(bytes).to_vec()
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
        let b_tt = crate::ir::node::TensorType::new(
            vec![DimExpr::Known(1), DimExpr::Known(4)],
            IrDType::F32,
        );
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
        graph: &crate::ir::node::ComputeGraph,
        plan: &mut crate::backend::ExecutablePlan,
        memory_plan: &crate::compiler::passes::memory_planning::MemoryPlan,
        inputs: &[&[u8]],
    ) -> Vec<Vec<u8>> {
        executor
            .execute(graph, plan, memory_plan, inputs)
            .expect("regular execute must succeed")
    }

    fn executor_fallback(
        executor: &mut GraphExecutor<CpuBackend>,
        graph: &crate::ir::node::ComputeGraph,
        plan: &mut crate::backend::ExecutablePlan,
        memory_plan: &crate::compiler::passes::memory_planning::MemoryPlan,
        inputs: &[&[u8]],
        prepared: &crate::backend::prepared::PreparedExecutablePlan,
    ) -> Vec<Vec<u8>> {
        executor
            .execute_prepared_fallback(graph, plan, memory_plan, inputs, prepared)
            .expect("prepared fallback execute must succeed")
    }
}
