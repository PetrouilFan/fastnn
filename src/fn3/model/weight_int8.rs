//! Session-owned packed weights and live-plan dispatch. No artifact changes.
//!
//! The session packs an eligible static FP32 `MatMul` weight the first time a
//! live plan reads it, caches the packed copy by model weight ID, and routes
//! that instruction to the existing fused kernel. Every other instruction runs
//! the backend's own `execute_op`, so a step is a mix of wholly-packed and
//! wholly-FP32 projections and never a partially approximated one.

use std::collections::{btree_map, BTreeMap, BTreeSet};
use std::fmt;
use std::time::{Duration, Instant};

use crate::inference::backend::cpu::quant::{
    matmul_quantized, registration, ActivationSpec, FallbackReason, PackedWeight, ProjectionSpec,
    QuantDecision, QuantPolicy, WorkHint,
};
#[cfg(feature = "vulkan")]
use crate::inference::backend::vulkan::kernels::quant::{
    admits_projection, launch_projection, PackedProjection, QuantSections,
};
use crate::inference::{Backend, DType, DeviceId, Fn3Error, OpId, Tensor};

use super::{ModelWeights, Plan, SlotRole};

/// Why one `MatMul` stayed on the backend's FP32 kernel.
///
/// A closed set: `--steps` prints [`WeightInt8Fallback::code`] and nothing else
/// is invented at a call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightInt8Fallback {
    /// The RHS is produced earlier in the same step (an attention product, or
    /// the transpose that builds the tied LM head).
    DynamicRhs,
    /// The RHS is a folded constant. Constants in this tree are integer-domain,
    /// so this is a dtype mismatch as well as an unpacked one.
    ConstantRhs,
    /// The RHS is a graph input.
    GraphInputRhs,
    /// The format's §6 policy rejected the weight.
    Policy(FallbackReason),
    /// The activation is outside the fused kernel's subset (dtype, rank,
    /// contraction length, or a strided view).
    ActivationLayout,
    /// The output is not a contiguous `f32` tensor.
    OutputLayout,
    /// A Vulkan operand is host-staged, is not a whole buffer, or the packed
    /// buffers or dispatch grid exceed a device limit.
    VulkanOperands,
    /// The encoder rejected the weight (non-finite value, unrepresentable
    /// scale) and the FP32 path keeps it.
    Packing(String),
}

impl WeightInt8Fallback {
    /// Stable code for reports. These strings are a closed set.
    pub fn code(&self) -> &str {
        match self {
            Self::DynamicRhs => "computed-rhs",
            Self::ConstantRhs => "constant-rhs",
            Self::GraphInputRhs => "graph-input-rhs",
            Self::Policy(reason) => reason.code(),
            Self::ActivationLayout => "activation-layout",
            Self::OutputLayout => "output-layout",
            Self::VulkanOperands => "vulkan-operands",
            Self::Packing(_) => "packing",
        }
    }
}

impl fmt::Display for WeightInt8Fallback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DynamicRhs => f.write_str("the right-hand side is computed in the step"),
            Self::ConstantRhs => f.write_str("the right-hand side is a folded constant"),
            Self::GraphInputRhs => f.write_str("the right-hand side is a graph input"),
            Self::Policy(reason) => reason.fmt(f),
            Self::ActivationLayout => {
                f.write_str("the activation is outside the fused kernel's subset")
            }
            Self::OutputLayout => f.write_str("the output is not contiguous f32"),
            Self::VulkanOperands => f.write_str(
                "the Vulkan operands are not resident whole buffers, or exceed a device limit",
            ),
            Self::Packing(detail) => f.write_str(detail),
        }
    }
}

/// One `MatMul` route in the last successful step of an INT8-weight session.
#[derive(Debug, Clone)]
pub struct WeightInt8Route {
    /// Instruction position in the specialized plan.
    pub instruction: usize,
    /// Right-hand operand name (weight or computed value).
    pub rhs: String,
    /// `None` when the instruction took the fused packed kernel.
    pub fallback: Option<WeightInt8Fallback>,
}

/// Quantization diagnostics for the last completed step.
#[derive(Debug, Default)]
pub struct WeightInt8Report {
    /// Route for every MatMul, including dynamic and undersized projections.
    pub routes: Vec<WeightInt8Route>,
    /// Wall time spent preparing live routes, packing new weights and preparing
    /// newly dense buffers.
    pub prepare: Duration,
    /// Packed payload bytes this session owns (codes + scales + zero-points).
    pub packed_payload_bytes: u64,
    /// FP32 bytes the packed projections replaced, so
    /// `packed_source_bytes - packed_payload_bytes` is the exact byte
    /// reduction at the API level (independent of what stays dense).
    pub packed_source_bytes: u64,
    /// Bytes of weights some live consumer still reads as an FP32 tensor. On
    /// the CPU this is every retained FP32 weight (nothing is uploaded); on
    /// Vulkan it is what `prepare_persistent` uploaded.
    pub dense_weight_bytes: u64,
    /// Bytes of weights that are both dense and packed, so this session holds
    /// an FP32 and a packed representation of the same bytes.
    pub shared_weight_bytes: u64,
}

pub(super) enum Packed {
    Cpu(PackedWeight),
    #[cfg(feature = "vulkan")]
    Vulkan(PackedProjection),
}

impl Packed {
    fn bytes(&self) -> u64 {
        match self {
            Self::Cpu(weight) => weight.report().packed_bytes(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(weight) => weight.payload_bytes(),
        }
    }
}

#[derive(Default)]
pub(super) struct WeightInt8 {
    /// Packed copy per model weight ID, or the encoder's rejection.
    packed: BTreeMap<usize, Result<Packed, String>>,
    /// Weight IDs some live plan has read through an FP32 consumer.
    dense: BTreeSet<usize>,
    pub(super) report: WeightInt8Report,
}

impl WeightInt8 {
    /// Forget the per-step report; the packed and dense caches stay valid.
    pub(super) fn clear_report(&mut self) {
        self.report = WeightInt8Report::default();
    }

    /// Choose a route for every instruction of `plan` and prepare what the
    /// chosen routes need.
    pub(super) fn prepare_step(
        &mut self,
        backend: &dyn Backend,
        plan: &Plan,
        weights: &ModelWeights,
        slots: &[Option<Tensor>],
    ) -> Result<(Vec<Option<usize>>, WeightInt8Report), Fn3Error> {
        let start = Instant::now();
        let mut report = WeightInt8Report::default();
        let mut routes = vec![None; plan.instructions().len()];
        let mut dense = BTreeSet::new();
        for (index, instruction) in plan.instructions().iter().enumerate() {
            let packed = if instruction.op == OpId::MatMul {
                // `ExecutionPlan::new` bounds every slot, and a MatMul binds two
                // inputs and one output; anything else is malformed and stays
                // entirely dense rather than panicking here.
                match (
                    instruction.inputs.first(),
                    instruction.inputs.get(1),
                    instruction.outputs.first(),
                ) {
                    (Some(&activation), Some(&rhs), Some(&out)) => {
                        let rhs_slot = plan.slot(rhs).expect("valid plan");
                        let (selected, rhs_name) = match *rhs_slot.role() {
                            SlotRole::Weight { id } => (
                                self.select(
                                    backend,
                                    id.index(),
                                    rhs_slot.name(),
                                    present(slots, rhs, "MatMul RHS weight")?,
                                    present(slots, activation, "MatMul activation")?,
                                    present(slots, out, "MatMul output")?,
                                )?,
                                rhs_slot.name().to_string(),
                            ),
                            SlotRole::Constant { .. } => {
                                (Err(WeightInt8Fallback::ConstantRhs), rhs_slot.name().into())
                            }
                            SlotRole::GraphInput => (
                                Err(WeightInt8Fallback::GraphInputRhs),
                                rhs_slot.name().into(),
                            ),
                            SlotRole::InstructionOutput { .. } => {
                                (Err(WeightInt8Fallback::DynamicRhs), rhs_slot.name().into())
                            }
                        };
                        let (packed, fallback) = match selected {
                            Ok(id) => (Some(id), None),
                            Err(reason) => (None, Some(reason)),
                        };
                        report.routes.push(WeightInt8Route {
                            instruction: index,
                            rhs: rhs_name,
                            fallback,
                        });
                        packed
                    }
                    _ => {
                        report.routes.push(WeightInt8Route {
                            instruction: index,
                            rhs: "<malformed MatMul>".into(),
                            fallback: Some(WeightInt8Fallback::ActivationLayout),
                        });
                        None
                    }
                }
            } else {
                None
            };
            routes[index] = packed;
            for (operand, &slot) in instruction.inputs.iter().enumerate() {
                if operand != 1 || packed.is_none() {
                    require_dense(plan, slot, &mut dense);
                }
            }
        }
        for output in plan.outputs() {
            require_dense(plan, output.slot(), &mut dense);
        }
        // A live plan proves which RHS operands never reach an FP32 kernel.
        // Re-evaluate on every shape: a later fallback or shared consumer can
        // request dense residency before recording starts.
        let missing: Vec<_> = dense.difference(&self.dense).copied().collect();
        let tensors: Vec<_> = missing
            .iter()
            .map(|&id| weights.get(id).expect("valid weight ID"))
            .collect();
        backend.prepare_persistent(&tensors)?;
        self.dense.extend(missing);
        report.prepare = start.elapsed();
        report.dense_weight_bytes = self
            .dense
            .iter()
            .filter_map(|&id| weights.get(id))
            .map(|tensor| tensor.storage().len() as u64)
            .sum();
        report.shared_weight_bytes = self
            .dense
            .iter()
            .filter(|id| self.packed.get(id).is_some_and(|entry| entry.is_ok()))
            .filter_map(|&id| weights.get(id))
            .map(|tensor| tensor.storage().len() as u64)
            .sum();
        report.packed_payload_bytes = self
            .packed
            .values()
            .filter_map(|entry| entry.as_ref().ok())
            .map(Packed::bytes)
            .sum();
        report.packed_source_bytes = self
            .packed
            .iter()
            .filter(|(_, entry)| entry.is_ok())
            .filter_map(|(&id, _)| weights.get(id))
            .map(|tensor| tensor.storage().len() as u64)
            .sum();
        Ok((routes, report))
    }

    /// Decide one projection, packing only once admission is known.
    fn select(
        &mut self,
        backend: &dyn Backend,
        id: usize,
        name: &str,
        rhs: &Tensor,
        activation: &Tensor,
        out: &Tensor,
    ) -> Result<Result<usize, WeightInt8Fallback>, Fn3Error> {
        let policy = QuantPolicy::default();
        let spec = ProjectionSpec::static_weight(name, rhs.shape(), rhs.dtype());
        if let QuantDecision::Fallback(reason) = policy.classify(&spec) {
            return Ok(Err(WeightInt8Fallback::Policy(reason)));
        }
        let admission = registration(&spec, ActivationSpec::of(&activation.as_view()?), &policy);
        if !admission.is_admitted() {
            return Ok(Err(WeightInt8Fallback::ActivationLayout));
        }
        if out.dtype() != DType::F32 || !out.is_contiguous() {
            return Ok(Err(WeightInt8Fallback::OutputLayout));
        }
        // Admission is decided on the live operands *before* any bytes are
        // encoded or uploaded: a projection the resident kernel cannot bind
        // must not leave a packed copy behind.
        if backend.device() != DeviceId::Cpu {
            #[cfg(feature = "vulkan")]
            if !vulkan_admits(backend, spec.n, spec.k, activation, out) {
                return Ok(Err(WeightInt8Fallback::VulkanOperands));
            }
            #[cfg(not(feature = "vulkan"))]
            return Ok(Err(WeightInt8Fallback::VulkanOperands));
        }
        if let btree_map::Entry::Vacant(entry) = self.packed.entry(id) {
            entry.insert(pack(backend, name, rhs)?);
        }
        match self.packed.get(&id) {
            Some(Ok(_)) => Ok(Ok(id)),
            Some(Err(reason)) => Ok(Err(WeightInt8Fallback::Packing(reason.clone()))),
            None => unreachable!("the packed entry was just inserted"),
        }
    }

    /// Run one plan, sending each routed `MatMul` to its packed kernel.
    pub(super) fn execute(
        &self,
        backend: &dyn Backend,
        plan: &Plan,
        slots: &mut [Option<Tensor>],
        routes: &[Option<usize>],
    ) -> Result<(), Fn3Error> {
        plan.execution()
            .execute_with(slots, |index, instruction, inputs, outputs| {
                let Some(&packed) = routes.get(index).and_then(Option::as_ref) else {
                    return backend.execute_op(
                        instruction.op,
                        inputs,
                        outputs,
                        &instruction.params,
                    );
                };
                let (Some(activation), Some(out)) = (inputs.first(), outputs.first_mut()) else {
                    return Err(Fn3Error::SessionState {
                        action: "Session::weight_int8",
                        phase: "running",
                        detail: format!(
                            "instruction {index} ({}) was routed to the packed kernel with {} \
                             inputs and no output",
                            instruction.op,
                            inputs.len()
                        ),
                    });
                };
                match &self.packed[&packed] {
                    Ok(Packed::Cpu(weight)) => {
                        matmul_quantized(
                            instruction.op,
                            activation,
                            &weight.view(),
                            out,
                            WorkHint::serial(),
                        )?;
                        Ok(())
                    }
                    #[cfg(feature = "vulkan")]
                    Ok(Packed::Vulkan(weight)) => {
                        let vulkan = backend.as_vulkan().ok_or_else(|| Fn3Error::SessionState {
                            action: "Session::weight_int8",
                            phase: "running",
                            detail: "a Vulkan packed projection reached a non-Vulkan backend"
                                .into(),
                        })?;
                        launch_projection(vulkan, weight, activation, out)?;
                        Ok(())
                    }
                    Err(reason) => Err(Fn3Error::SessionState {
                        action: "Session::weight_int8",
                        phase: "running",
                        detail: format!(
                            "weight {packed} was routed after a packing failure: {reason}"
                        ),
                    }),
                }
            })
    }
}

/// `true` when a resident Vulkan kernel can bind this projection as-is.
#[cfg(feature = "vulkan")]
fn vulkan_admits(
    backend: &dyn Backend,
    n_rows: u64,
    k: u64,
    activation: &Tensor,
    out: &Tensor,
) -> bool {
    let Some(vulkan) = backend.as_vulkan() else {
        return false;
    };
    let (Ok(n_rows), Ok(k)) = (u32::try_from(n_rows), u32::try_from(k)) else {
        return false;
    };
    activation.storage().is_resident()
        && out.storage().is_resident()
        && admits_projection(vulkan, n_rows, k, activation.meta(), out.meta())
}

/// A bound slot, or a named error when the session did not bind it.
fn present<'a>(
    slots: &'a [Option<Tensor>],
    slot: u32,
    what: &'static str,
) -> Result<&'a Tensor, Fn3Error> {
    slots
        .get(slot as usize)
        .and_then(Option::as_ref)
        .ok_or_else(|| Fn3Error::SessionState {
            action: "Session::weight_int8",
            phase: "running",
            detail: format!("the {what} slot {slot} is empty before the step"),
        })
}

fn require_dense(plan: &Plan, slot: u32, dense: &mut BTreeSet<usize>) {
    if let SlotRole::Weight { id } = *plan.slot(slot).expect("valid slot").role() {
        dense.insert(id.index());
    }
}

fn pack(
    backend: &dyn Backend,
    name: &str,
    source: &Tensor,
) -> Result<Result<Packed, String>, Fn3Error> {
    let weight = match PackedWeight::from_tensor(name, source) {
        Ok(weight) => weight,
        // A non-finite value or an unrepresentable scale is a property of the
        // weight, not a device failure: the FP32 kernel keeps it.
        Err(error) => return Ok(Err(format!("the encoder rejected the weight: {error}"))),
    };
    if backend.device() == DeviceId::Cpu {
        return Ok(Ok(Packed::Cpu(weight)));
    }
    #[cfg(feature = "vulkan")]
    if let Some(vulkan) = backend.as_vulkan() {
        let sections = match QuantSections::from_packed(weight.view()) {
            Ok(sections) if sections.fits(vulkan) => sections,
            _ => {
                return Ok(Err(
                    "the packed sections exceed a Vulkan buffer limit".into()
                ))
            }
        };
        return Ok(Ok(Packed::Vulkan(PackedProjection::prepare(
            vulkan, name, sections,
        )?)));
    }
    Ok(Err("this device has no INT8 packed kernel".into()))
}

// ---------------------------------------------------------------------------
// Tests: a synthetic plan, no fixture and no device
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::backend::cpu::CpuBackend;
    use crate::inference::model::plan::{BoundValue, PlanParts, SlotPlan};
    use crate::inference::model::{ExecutionPlan, Instruction, WeightId};
    use crate::inference::shape::{LiveDims, Shape, ShapeEnvelope};

    /// K and N are above the format's policy floor (K >= 64, 4096 elements).
    const K: u64 = 128;
    const N: u64 = 64;

    fn values(count: usize, stride: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let phase = (index as f32) * stride;
                (phase.sin() * 0.5) + (phase.cos() * 0.25)
            })
            .collect()
    }

    fn f32_slot(name: &str, dims: [u64; 2], role: SlotRole) -> SlotPlan {
        SlotPlan::new(name.into(), DType::F32, Shape::from_dims(dims), role)
    }

    /// A one-`MatMul` plan: slot 0 activation, slot 1 RHS, slot 2 output.
    fn assemble(slots: Vec<SlotPlan>, instructions: Vec<Instruction>, value_count: u32) -> Plan {
        let input = BoundValue::new("x".into(), 0, slots[0].dtype(), slots[0].shape().clone());
        let output = BoundValue::new("y".into(), 2, slots[2].dtype(), slots[2].shape().clone());
        let execution = ExecutionPlan::new(instructions, value_count).expect("valid execution");
        Plan::assemble(PlanParts {
            execution,
            slots,
            constants: Vec::new(),
            folded: Vec::new(),
            mask: Vec::new(),
            inputs: vec![input],
            outputs: vec![output],
            dims: LiveDims::new(1, 1, 0, 1, 1, 1).expect("decode dims"),
            envelope: ShapeEnvelope::uniform(1).expect("uniform envelope"),
            backend: "cpu",
        })
        .expect("valid plan")
    }

    /// A `MatMul` whose RHS is model weight 0, plus materialised slots.
    fn weight_fixture(k: u64, n: u64) -> (Plan, ModelWeights, Vec<Option<Tensor>>) {
        let weight = Tensor::from_f32(Shape::from_dims([k, n]), &values((k * n) as usize, 0.017))
            .expect("host weight");
        let weights =
            ModelWeights::with_names(vec![("proj.weight".into(), weight)]).expect("one weight");
        let plan = assemble(
            vec![
                f32_slot("x", [1, k], SlotRole::GraphInput),
                f32_slot(
                    "proj.weight",
                    [k, n],
                    SlotRole::Weight {
                        id: WeightId::new(0),
                    },
                ),
                f32_slot("y", [1, n], SlotRole::InstructionOutput { instruction: 0 }),
            ],
            vec![Instruction::new(OpId::MatMul, vec![0, 1], vec![2])],
            3,
        );
        let mut slots = plan
            .materialise(&weights, DeviceId::Cpu)
            .expect("materialise the plan");
        slots[0] = Some(
            Tensor::from_f32(Shape::from_dims([1, k]), &values(k as usize, 0.011))
                .expect("activation"),
        );
        (plan, weights, slots)
    }

    fn output(slots: &[Option<Tensor>], shape: [u64; 2]) -> Vec<f32> {
        let tensor = slots[2].as_ref().expect("written output");
        assert_eq!(tensor.shape(), &Shape::from_dims(shape));
        tensor.as_f32_slice().expect("read the output").to_vec()
    }

    #[test]
    fn a_static_weight_is_packed_routed_and_measured() {
        let backend = CpuBackend::new();
        let (plan, weights, mut slots) = weight_fixture(K, N);
        let mut state = WeightInt8::default();

        let (routes, report) = state
            .prepare_step(&backend, &plan, &weights, &slots)
            .expect("prepare the step");
        assert_eq!(routes, vec![Some(0)], "the MatMul takes the packed route");
        assert_eq!(report.routes.len(), 1);
        assert!(report.routes[0].fallback.is_none(), "{:?}", report.routes);
        assert_eq!(report.routes[0].rhs, "proj.weight");
        assert!(report.packed_payload_bytes > 0);
        assert_eq!(report.packed_source_bytes, K * N * 4);
        assert!(report.packed_payload_bytes < report.packed_source_bytes);
        // This plan's only consumer of the weight is the packed kernel, so the
        // session must not keep an FP32 copy for a backend that uploads.
        assert_eq!(report.dense_weight_bytes, 0);
        assert_eq!(report.shared_weight_bytes, 0);

        state
            .execute(&backend, &plan, &mut slots, &routes)
            .expect("run the packed step");
        let packed = output(&slots, [1, N]);
        assert!(packed.iter().all(|value| value.is_finite()));

        // The same plan through the plain FP32 backend, for the error bound.
        let (_, _, mut reference_slots) = weight_fixture(K, N);
        plan.execute(&backend, &mut reference_slots)
            .expect("run the FP32 step");
        let reference = output(&reference_slots, [1, N]);
        let max_error = packed
            .iter()
            .zip(&reference)
            .map(|(packed, reference)| (packed - reference).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error < 0.02,
            "W8A16 error {max_error} is larger than the format's own bound"
        );
    }

    #[test]
    fn a_repeated_step_reuses_the_packed_copy() {
        let backend = CpuBackend::new();
        let (plan, weights, slots) = weight_fixture(K, N);
        let mut state = WeightInt8::default();
        let (_, first) = state
            .prepare_step(&backend, &plan, &weights, &slots)
            .expect("first step");
        let (_, second) = state
            .prepare_step(&backend, &plan, &weights, &slots)
            .expect("second step");
        assert_eq!(second.packed_payload_bytes, first.packed_payload_bytes);
        assert!(
            second.prepare < first.prepare,
            "the second step re-packed the weight: {} then {}",
            first.prepare.as_nanos(),
            second.prepare.as_nanos()
        );
    }

    /// Every non-packed route reports its own closed code, never "dynamic" for
    /// something static.
    #[test]
    fn fallback_codes_name_the_actual_reason() {
        let backend = CpuBackend::new();
        let mut state = WeightInt8::default();

        // 1. A dynamic RHS: the MatMul reads a value a previous instruction
        //    produced (an attention product, in a real graph).
        let dynamic = assemble(
            vec![
                f32_slot("x", [1, K], SlotRole::GraphInput),
                f32_slot(
                    "computed",
                    [K, N],
                    SlotRole::InstructionOutput { instruction: 0 },
                ),
                f32_slot("y", [1, N], SlotRole::InstructionOutput { instruction: 1 }),
                f32_slot(
                    "w",
                    [K, N],
                    SlotRole::Weight {
                        id: WeightId::new(0),
                    },
                ),
            ],
            vec![
                Instruction::new(OpId::Reshape, vec![3], vec![1]),
                Instruction::new(OpId::MatMul, vec![0, 1], vec![2]),
            ],
            4,
        );
        let weights = ModelWeights::with_names(vec![(
            "w".into(),
            Tensor::from_f32(Shape::from_dims([K, N]), &values((K * N) as usize, 0.01)).unwrap(),
        )])
        .unwrap();
        let slots = dynamic.materialise(&weights, DeviceId::Cpu).unwrap();
        let (routes, report) = state
            .prepare_step(&backend, &dynamic, &weights, &slots)
            .expect("dynamic step");
        assert_eq!(routes[1], None);
        assert_eq!(report.routes.len(), 1, "one route per MatMul");
        assert_eq!(report.routes[0].instruction, 1);
        assert_eq!(
            report.routes[0].fallback.as_ref().unwrap().code(),
            "computed-rhs"
        );

        // 2. A graph-input RHS.
        let graph_input = assemble(
            vec![
                f32_slot("x", [1, K], SlotRole::GraphInput),
                f32_slot("rhs", [K, N], SlotRole::GraphInput),
                f32_slot("y", [1, N], SlotRole::InstructionOutput { instruction: 0 }),
            ],
            vec![Instruction::new(OpId::MatMul, vec![0, 1], vec![2])],
            3,
        );
        let (_, report) = state
            .prepare_step(&backend, &graph_input, &weights, &slots)
            .expect("graph-input step");
        assert_eq!(
            report.routes[0].fallback.as_ref().unwrap().code(),
            "graph-input-rhs"
        );

        // 3. A weight below the format's policy floor.
        let mut small_state = WeightInt8::default();
        let (small, small_weights, small_slots) = weight_fixture(8, 8);
        let (routes, report) = small_state
            .prepare_step(&backend, &small, &small_weights, &small_slots)
            .expect("small step");
        assert_eq!(routes[0], None);
        assert_eq!(
            report.routes[0].fallback.as_ref().unwrap().code(),
            "k-below-min"
        );
        assert_eq!(report.packed_payload_bytes, 0);
    }
}
