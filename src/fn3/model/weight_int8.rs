//! Session-owned packed weights and live-plan dispatch. No artifact changes.

use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

use crate::inference::backend::cpu::quant::{
    matmul_quantized, registration, ActivationSpec, PackedWeight, ProjectionSpec, QuantDecision,
    QuantPolicy, WorkHint,
};
#[cfg(feature = "vulkan")]
use crate::inference::backend::vulkan::kernels::quant::{
    launch_projection, PackedProjection, QuantSections,
};
use crate::inference::{Backend, DType, DeviceId, Fn3Error, OpId, Tensor};

use super::{ModelWeights, Plan, SlotRole};

/// One MatMul route in the last successful step of an INT8-weight session.
#[derive(Debug, Clone)]
pub struct WeightInt8Route {
    /// Instruction position in the specialized plan.
    pub instruction: usize,
    /// Right-hand operand name (weight or computed value).
    pub rhs: String,
    /// FP32 fallback reason, or `None` for a fused packed projection.
    pub fallback: Option<String>,
}

/// Quantization diagnostics for the last successful step.
#[derive(Debug, Default)]
pub struct WeightInt8Report {
    /// Route for every MatMul, including dynamic and undersized projections.
    pub routes: Vec<WeightInt8Route>,
    /// Wall time spent preparing live routes, packing new weights and preparing persistent buffers.
    pub prepare: Duration,
    /// Cumulative packed payload bytes owned by this session (not total memory or file size).
    pub packed_payload_bytes: u64,
    /// Bytes of model weights this session asked the backend to keep dense, because
    /// some live consumer reads them as an FP32 tensor.
    pub dense_weight_bytes: u64,
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
    packed: BTreeMap<usize, Result<Packed, String>>,
    dense: BTreeSet<usize>,
    pub(super) report: WeightInt8Report,
}

impl WeightInt8 {
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
                let rhs = plan.slot(instruction.inputs[1]).expect("valid plan");
                let selected = match *rhs.role() {
                    SlotRole::Weight { id } => self.select(
                        backend,
                        id.index(),
                        rhs.name(),
                        slots[instruction.inputs[1] as usize]
                            .as_ref()
                            .expect("weight slot"),
                        slots[instruction.inputs[0] as usize]
                            .as_ref()
                            .expect("bound activation slot"),
                        slots[instruction.outputs[0] as usize]
                            .as_ref()
                            .expect("output slot"),
                    )?,
                    _ => Err("dynamic-rhs".into()),
                };
                let (packed, fallback) = match selected {
                    Ok(id) => (Some(id), None),
                    Err(reason) => (None, Some(reason)),
                };
                report.routes.push(WeightInt8Route {
                    instruction: index,
                    rhs: rhs.name().into(),
                    fallback,
                });
                packed
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
        // A live plan proves which RHS operands never reach the FP32 backend.
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
        report.packed_payload_bytes = self
            .packed
            .values()
            .filter_map(|entry| entry.as_ref().ok())
            .map(Packed::bytes)
            .sum();
        Ok((routes, report))
    }

    fn select(
        &mut self,
        backend: &dyn Backend,
        id: usize,
        name: &str,
        rhs: &Tensor,
        activation: &Tensor,
        out: &Tensor,
    ) -> Result<Result<usize, String>, Fn3Error> {
        let policy = QuantPolicy::default();
        let spec = ProjectionSpec::static_weight(name, rhs.shape(), rhs.dtype());
        if let QuantDecision::Fallback(reason) = policy.classify(&spec) {
            return Ok(Err(reason.code().into()));
        }
        let admission = registration(&spec, ActivationSpec::of(&activation.as_view()?), &policy);
        if !admission.is_admitted() {
            return Ok(Err(admission.code().into()));
        }
        if out.dtype() != DType::F32 || !out.is_contiguous() {
            return Ok(Err("output-layout".into()));
        }
        if let std::collections::btree_map::Entry::Vacant(entry) = self.packed.entry(id) {
            entry.insert(pack(backend, name, rhs)?);
        }
        match &self.packed[&id] {
            Ok(Packed::Cpu(_)) => Ok(Ok(id)),
            #[cfg(feature = "vulkan")]
            Ok(Packed::Vulkan(weight)) => {
                if activation.storage().is_resident()
                    && out.storage().is_resident()
                    && weight.admits_session(
                        backend.as_vulkan().expect("validated backend"),
                        activation.meta(),
                        out.meta(),
                    )
                {
                    Ok(Ok(id))
                } else {
                    Ok(Err("vulkan-activation/output-layout-or-limits".into()))
                }
            }
            Err(reason) => Ok(Err(reason.clone())),
        }
    }

    pub(super) fn execute(
        &self,
        backend: &dyn Backend,
        plan: &Plan,
        slots: &mut [Option<Tensor>],
        routes: &[Option<usize>],
    ) -> Result<(), Fn3Error> {
        plan.execution()
            .execute_with(slots, |index, instruction, inputs, outputs| {
                match routes[index].and_then(|id| self.packed[&id].as_ref().ok()) {
                    Some(Packed::Cpu(weight)) => matmul_quantized(
                        instruction.op,
                        &inputs[0],
                        &weight.view(),
                        &mut outputs[0],
                        WorkHint::serial(),
                    ),
                    #[cfg(feature = "vulkan")]
                    Some(Packed::Vulkan(weight)) => launch_projection(
                        backend.as_vulkan().expect("validated backend"),
                        weight,
                        &inputs[0],
                        &mut outputs[0],
                    )
                    .map(|_| ()),
                    None => {
                        backend.execute_op(instruction.op, inputs, outputs, &instruction.params)
                    }
                }
            })
    }
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
        // E.g. the source contains a non-finite value or has an unrepresentable scale.
        Err(error) => return Ok(Err(format!("packing: {error}"))),
    };
    if backend.device() == DeviceId::Cpu {
        return Ok(Ok(Packed::Cpu(weight)));
    }
    #[cfg(feature = "vulkan")]
    if let Some(vulkan) = backend.as_vulkan() {
        let sections = match QuantSections::from_packed(weight.view()) {
            Ok(sections) if sections.fits(vulkan) => sections,
            _ => return Ok(Err("vulkan-packed-buffer-limits".into())),
        };
        return Ok(Ok(Packed::Vulkan(PackedProjection::prepare(
            vulkan, name, sections,
        )?)));
    }
    unreachable!("INT8 backend validated at session setup")
}
