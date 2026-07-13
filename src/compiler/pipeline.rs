//! Canonical graph compiler pipeline.
//!
//! This module owns pass ordering and representation-policy application. Backends
//! consume the normalized graph and memory plan; they do not choose compiler passes.

use crate::compiler::passes::{
    activation_quantization, arithmetic_simplify, calibration, constant_folding,
    dead_code_elimination, memory_planning, operator_fusion, prune_qdq_pairs, quantization,
    shape_inference,
};
use crate::compiler::plan::MemoryPlan;
use crate::compiler::{CompilerError, CompilerResult};
use crate::ir::ComputeGraph;
use crate::types::{CompileTarget, QuantTarget};

pub struct CompilerPipeline {
    target: CompileTarget,
    calibration: Option<calibration::CalibrationData>,
}

pub struct CompiledGraph {
    pub graph: ComputeGraph,
    pub memory_plan: MemoryPlan,
}

impl CompilerPipeline {
    pub fn new(target: CompileTarget, calibration: Option<calibration::CalibrationData>) -> Self {
        Self {
            target,
            calibration,
        }
    }

    pub fn run(self, mut graph: ComputeGraph) -> CompilerResult<CompiledGraph> {
        let quant_target = match self.target {
            CompileTarget::Native => None,
            CompileTarget::WeightOnly(target) => Some(target),
            CompileTarget::IntegerInference(target) => {
                if self.calibration.is_none() {
                    return Err(CompilerError::InvalidTarget(
                        "integer inference requires activation calibration data".into(),
                    ));
                }
                Some(target)
            }
            CompileTarget::TrainingMixedPrecision { .. } => {
                return Err(CompilerError::InvalidTarget(
                    "mixed-precision training compilation is not implemented".into(),
                ));
            }
        };

        shape_inference::infer_shapes(&mut graph)
            .map_err(|error| CompilerError::pass("shape inference", error))?;
        constant_folding::constant_fold(&mut graph);
        arithmetic_simplify::arithmetic_simplify(&mut graph);
        operator_fusion::fuse_operators(&mut graph)
            .map_err(|error| CompilerError::pass("operator fusion", error))?;
        dead_code_elimination::eliminate_dead_code(&mut graph);

        if quant_target.is_some() || self.calibration.is_some() {
            if let Some(target) = quant_target {
                apply_weight_quantization(&mut graph, target)?;
            }
            quantization::wrap_quantized_optimizer(&mut graph)
                .map_err(|error| CompilerError::pass("optimizer wrapping", error))?;
            if let Some(calibration) = self.calibration.as_ref() {
                activation_quantization::quantize_activations_with_calibration(
                    &mut graph,
                    calibration,
                )
                .map_err(|error| CompilerError::pass("activation quantization", error))?;
            }
            prune_qdq_pairs::prune_qdq_pairs(&mut graph)
                .map_err(|error| CompilerError::pass("prune qdq pairs", error))?;
        }

        dead_code_elimination::eliminate_dead_code(&mut graph);
        validate_representations(&graph)?;
        let memory_plan = memory_planning::plan_memory(&graph)
            .map_err(|error| CompilerError::pass("memory planning", error))?;

        Ok(CompiledGraph { graph, memory_plan })
    }
}

fn apply_weight_quantization(graph: &mut ComputeGraph, target: QuantTarget) -> CompilerResult<()> {
    use quantization::FpDtype;

    let result = match target {
        QuantTarget::I4 => quantization::quantize_weights(graph, 4, true, None),
        QuantTarget::I8 => quantization::quantize_weights(graph, 8, true, None),
        QuantTarget::U4 => quantization::quantize_weights(graph, 4, false, None),
        QuantTarget::U8 => quantization::quantize_weights(graph, 8, false, None),
        QuantTarget::Fp8E4M3 => quantization::quantize_weights_fp(graph, &FpDtype::F8x4, None),
        QuantTarget::Fp8E5M2 => quantization::quantize_weights_fp(graph, &FpDtype::F8x4R, None),
        QuantTarget::Fp4E2M1 => quantization::quantize_weights_fp(graph, &FpDtype::F4x8, None),
        QuantTarget::I4Codebook => {
            quantization::quantize_weights_fp(graph, &FpDtype::I4Codebook, None)
        }
    };
    result.map_err(|error| CompilerError::pass("quantization", error))
}

fn validate_representations(graph: &ComputeGraph) -> CompilerResult<()> {
    for node in &graph.nodes {
        for tensor_type in
            std::iter::once(&node.output_type).chain(node.secondary_output_type.as_ref())
        {
            let representation = tensor_type.dtype.value_representation().map_err(|error| {
                CompilerError::InvalidRepresentation {
                    node_id: node.id,
                    message: error.to_string(),
                }
            })?;
            representation
                .validate()
                .map_err(|error| CompilerError::InvalidRepresentation {
                    node_id: node.id,
                    message: error.to_string(),
                })?;
        }
    }
    Ok(())
}
