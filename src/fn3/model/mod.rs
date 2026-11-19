//! Prepared models, execution plans and inference sessions.
//!
//! # Ownership split (the rule downstream workers must preserve)
//!
//! ```text
//! PreparedModel  (immutable, Send + Sync, cheap to clone via Arc)
//!   ├── Arc<ExecutionPlan>   instruction list, immutable
//!   ├── Arc<ModelWeights>    immutable weight tensors, Arc-shared storage
//!   ├── DeviceId             the device every weight lives on
//!   └── ShapeEnvelope        the declared dynamic bounds
//!
//! Session        (mutable, Send but NOT Sync, one per concurrent step)
//!   ├── &PreparedModel       borrowed, never owned
//!   ├── StepRuntime          planner + backend + plan cache + slot pool
//!   ├── Vec<KvCacheState>    capacity + live extent + buffers
//!   ├── step: u64            successful `decode` calls
//!   └── phase                Empty | Prefilling | Ready
//! ```
//!
//! A `PreparedModel` never changes after construction, so it may be shared by
//! any number of threads (`Arc<PreparedModel>`). A `Session` owns every mutable
//! byte of inference state: two sessions over one model are independent, and
//! one session must not be stepped from two threads. `Session` is deliberately
//! `!Sync` (see its `PhantomData<Cell<()>>` field): `Arc<Session>` cannot be
//! sent across threads, while `Mutex<Session>` still works because `Session`
//! is `Send`.
//!
//! # Step atomicity
//!
//! A failed step leaves the KV cache live extents and the step counter
//! unchanged. Implementations must validate every output extent and capacity
//! before mutating a cache buffer, and commit cache appends and the counter
//! only after all instructions succeeded.
//!
//! # Who implements what
//!
//! | Area | Owner |
//! |---|---|
//! | checked tensor metadata / storage hardening | FN3-06 (#52) |
//! | portable CPU reference kernels | FN3-07 (#53) |
//! | strict ONNX import and typed weights | FN3-08 (#54) |
//! | versioned prepared-model format | FN3-09 (#55) |
//! | bounded specialisation and plan building | FN3-10 (#56) |
//! | Vulkan backend | FN3-11 (#57) |
//! | real KV append, `prefill`/`decode` | FN3-12 (#58), in `kv.rs` and `session.rs` |
//! | end-to-end pinned model | FN3-16 (#62) |
//!
//! The signatures and ownership rules below are the contract those issues
//! code against. An entry point that is not implemented returns
//! [`Fn3Error::NotImplemented`] naming the owner; it never panics and never
//! approximates.

pub mod format;
pub mod kv;
pub mod plan;
pub mod session;
pub mod specialize;
pub mod weights;

pub use kv::{KvCacheConfig, KvCacheLayout, KvCacheState, MAX_KV_LAYERS};
pub use plan::{
    BoundValue, Constant, ConstantData, FoldKind, FoldedNode, MaskConstant, MaskRole, Plan,
    PlanBound, PlanCache, PlanId, SlotPlan, SlotRole,
};
pub use session::{Session, SessionOptions, SessionPhase};
pub use specialize::{PlanAttr, Planner};
pub use weights::{WeightId, WeightSet, WeightSource, WeightTensor};

use std::sync::Arc;

use crate::fn3::backend::Backend;
use crate::fn3::device::DeviceId;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::shape::{LiveDims, ShapeEnvelope};
use crate::fn3::tensor::Tensor;

/// One op invocation inside an [`ExecutionPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct Instruction {
    /// Operation to execute.
    pub op: OpId,
    /// Value slots read by the op, in operand order.
    pub inputs: Vec<u32>,
    /// Value slots written by the op, in result order.
    pub outputs: Vec<u32>,
    /// Op attributes.
    pub params: crate::fn3::backend::OpParams,
}

impl Instruction {
    /// Construct with default attributes.
    pub fn new(op: OpId, inputs: Vec<u32>, outputs: Vec<u32>) -> Instruction {
        Instruction {
            op,
            inputs,
            outputs,
            params: crate::fn3::backend::OpParams::default(),
        }
    }

    /// Builder: replace the attributes.
    pub fn with_params(mut self, params: crate::fn3::backend::OpParams) -> Instruction {
        self.params = params;
        self
    }
}

/// An ordered list of instructions over a flat table of value slots.
///
/// Slots are indices into the caller's `&mut [Option<Tensor>]`. Inputs must be
/// present; outputs must be pre-allocated (this keeps allocation policy in the
/// planner, FN3-10 / #56). An instruction may not read and write the same slot,
/// and may not write one slot twice: that is how the alpha plan rules out
/// hidden aliasing.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPlan {
    instructions: Vec<Instruction>,
    value_count: u32,
}

impl ExecutionPlan {
    /// Validate and construct a plan.
    pub fn new(
        instructions: Vec<Instruction>,
        value_count: u32,
    ) -> Result<ExecutionPlan, Fn3Error> {
        if value_count == 0 {
            return Err(Fn3Error::InvalidShape {
                context: "ExecutionPlan::new",
                detail: "a plan needs at least one value slot".to_string(),
            });
        }
        for (index, instruction) in instructions.iter().enumerate() {
            if instruction.outputs.is_empty() {
                return Err(Fn3Error::InvalidShape {
                    context: "ExecutionPlan::new",
                    detail: format!(
                        "instruction {index} ({}) has no output slot",
                        instruction.op
                    ),
                });
            }
            for &slot in instruction.inputs.iter().chain(instruction.outputs.iter()) {
                if slot >= value_count {
                    return Err(Fn3Error::OutOfBounds {
                        what: "ExecutionPlan::new (value slot)",
                        index: slot as u64,
                        len: value_count as u64,
                    });
                }
            }
            for (position, &slot) in instruction.outputs.iter().enumerate() {
                if instruction.inputs.contains(&slot) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ExecutionPlan::new",
                        detail: format!(
                            "instruction {index} ({}) reads and writes value slot {slot}",
                            instruction.op
                        ),
                    });
                }
                if instruction.outputs[..position].contains(&slot) {
                    return Err(Fn3Error::InvalidShape {
                        context: "ExecutionPlan::new",
                        detail: format!(
                            "instruction {index} ({}) writes value slot {slot} twice",
                            instruction.op
                        ),
                    });
                }
            }
        }
        Ok(ExecutionPlan {
            instructions,
            value_count,
        })
    }

    /// The instructions, in execution order.
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Number of value slots the plan addresses.
    pub fn value_count(&self) -> u32 {
        self.value_count
    }

    /// Reject plans that contain an op no backend can run.
    ///
    /// `Partial` support is accepted here; the op itself must reject the shapes
    /// it does not implement. Use [`crate::fn3::backend::Support::require_full`]
    /// when a plan must be guaranteed to run without falling back to a
    /// decomposed path.
    pub fn check_supported(&self, backend: &dyn Backend) -> Result<(), Fn3Error> {
        for (index, instruction) in self.instructions.iter().enumerate() {
            backend
                .supports(instruction.op)
                .require_supported(instruction.op, backend.name())
                .map_err(|error| context_note(error, instruction.op, index))?;
        }
        Ok(())
    }

    /// Execute every instruction against `slots`.
    ///
    /// `slots` must have exactly [`ExecutionPlan::value_count`] entries and
    /// every slot read by the plan must be `Some`. Output slots are taken out
    /// of the table, written through [`Tensor::as_mut_view`] and put back
    /// afterwards, including on failure, so a failed instruction does not
    /// destroy the caller's tensors.
    pub fn execute(
        &self,
        backend: &dyn Backend,
        slots: &mut [Option<Tensor>],
    ) -> Result<(), Fn3Error> {
        if slots.len() != self.value_count as usize {
            return Err(Fn3Error::CapacityExceeded {
                what: "execution plan value slots",
                requested: slots.len() as u64,
                capacity: self.value_count as u64,
            });
        }
        for (index, instruction) in self.instructions.iter().enumerate() {
            execute_instruction(backend, slots, instruction)
                .map_err(|error| context_note(error, instruction.op, index))?;
        }
        Ok(())
    }
}

fn execute_instruction(
    backend: &dyn Backend,
    slots: &mut [Option<Tensor>],
    instruction: &Instruction,
) -> Result<(), Fn3Error> {
    for &slot in &instruction.outputs {
        if slots[slot as usize].is_none() {
            return Err(Fn3Error::SessionState {
                action: "ExecutionPlan::execute",
                phase: "running",
                detail: format!(
                    "output value slot {slot} for op {} is empty; allocate it before running the plan",
                    instruction.op
                ),
            });
        }
    }
    let mut taken: Vec<Tensor> = instruction
        .outputs
        .iter()
        .map(|&slot| {
            slots[slot as usize]
                .take()
                .expect("output slots were checked to be present")
        })
        .collect();

    let result = (|| {
        let mut input_views = Vec::with_capacity(instruction.inputs.len());
        for &slot in &instruction.inputs {
            let tensor = slots[slot as usize]
                .as_ref()
                .ok_or_else(|| Fn3Error::SessionState {
                    action: "ExecutionPlan::execute",
                    phase: "running",
                    detail: format!("input value slot {slot} for op {} is empty", instruction.op),
                })?;
            input_views.push(tensor.as_view()?);
        }
        let mut output_views = Vec::with_capacity(taken.len());
        for tensor in taken.iter_mut() {
            output_views.push(tensor.as_mut_view()?);
        }
        backend.execute_op(
            instruction.op,
            &input_views,
            &mut output_views,
            &instruction.params,
        )
    })();

    for (&slot, tensor) in instruction.outputs.iter().zip(taken) {
        slots[slot as usize] = Some(tensor);
    }
    result
}

/// Add the plan position to an unsupported-op diagnostic.
fn context_note(error: Fn3Error, op: OpId, index: usize) -> Fn3Error {
    match error {
        Fn3Error::UnsupportedOp {
            op,
            backend,
            reason,
        } => Fn3Error::UnsupportedOp {
            op,
            backend,
            reason: format!("{reason} (at instruction {index}, op {op})"),
        },
        other => {
            let _ = op;
            other
        }
    }
}

/// Immutable, `Arc`-shared weight tensors.
///
/// Weights are stored behind `Arc<Tensor>` so a `PreparedModel` and every
/// session clone share one copy of every weight buffer. There is no API that
/// mutates a weight: a plan reads weights through [`TensorView`] only.
///
/// [`TensorView`]: crate::fn3::tensor::TensorView
#[derive(Debug, Clone, Default)]
pub struct ModelWeights {
    tensors: Vec<Arc<Tensor>>,
    names: Vec<String>,
}

impl ModelWeights {
    /// Construct from unnamed tensors. All weights must share one device.
    pub fn new(tensors: Vec<Tensor>) -> Result<ModelWeights, Fn3Error> {
        let named: Vec<(String, Tensor)> = tensors
            .into_iter()
            .enumerate()
            .map(|(index, tensor)| (format!("weight_{index}"), tensor))
            .collect();
        ModelWeights::with_names(named)
    }

    /// Construct from named tensors. All weights must share one device.
    pub fn with_names(tensors: Vec<(String, Tensor)>) -> Result<ModelWeights, Fn3Error> {
        let mut expected: Option<DeviceId> = None;
        for (_name, tensor) in &tensors {
            match expected {
                None => expected = Some(tensor.device()),
                Some(device) if device == tensor.device() => {}
                Some(device) => {
                    return Err(Fn3Error::DeviceMismatch {
                        op: None,
                        expected: device,
                        actual: tensor.device(),
                    })
                }
            }
        }
        let mut names = Vec::with_capacity(tensors.len());
        let mut handles = Vec::with_capacity(tensors.len());
        for (name, tensor) in tensors {
            names.push(name);
            handles.push(Arc::new(tensor));
        }
        Ok(ModelWeights {
            tensors: handles,
            names,
        })
    }

    /// Number of weights.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// `true` if there are no weights.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// A weight by index.
    pub fn get(&self, index: usize) -> Option<&Tensor> {
        self.tensors.get(index).map(|tensor| tensor.as_ref())
    }

    /// A weight by name.
    pub fn named(&self, name: &str) -> Option<&Tensor> {
        self.names
            .iter()
            .position(|candidate| candidate == name)
            .and_then(|index| self.get(index))
    }

    /// The name of a weight, if it has one.
    pub fn name(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(String::as_str)
    }

    /// The shared handle of a weight by index.
    pub fn handle(&self, index: usize) -> Option<&Arc<Tensor>> {
        self.tensors.get(index)
    }

    /// The device all weights live on, if any.
    pub fn device(&self) -> Option<DeviceId> {
        self.tensors.first().map(|tensor| tensor.device())
    }
}

/// An immutable, shareable prepared model.
///
/// Holds the plan, the weights and the declared dynamic bounds. Construct once,
/// share with `Arc`, create one [`Session`] per concurrent inference stream.
#[derive(Debug, Clone)]
pub struct PreparedModel {
    plan: Arc<ExecutionPlan>,
    weights: Arc<ModelWeights>,
    device: DeviceId,
    envelope: ShapeEnvelope,
    kv_config: Option<KvCacheConfig>,
}

impl PreparedModel {
    /// Assemble a model, validating device agreement and the envelope.
    ///
    /// Does not consult a backend and does not check op support: call
    /// [`ExecutionPlan::check_supported`] with the backend that will run it, or
    /// bind a backend with [`Session::with_runtime`], which preflights every
    /// specialised plan.
    ///
    /// A KV cache configuration is valid on any device. The cache is host
    /// staging on a Vulkan device (alpha tensor storage is host memory on every
    /// device), and executing on the host instead would be a device fallback,
    /// so the session rejects a backend whose device differs from this one.
    pub fn new(
        plan: ExecutionPlan,
        weights: ModelWeights,
        device: DeviceId,
        envelope: ShapeEnvelope,
        kv_config: Option<KvCacheConfig>,
    ) -> Result<PreparedModel, Fn3Error> {
        envelope.validate()?;
        if let Some(weights_device) = weights.device() {
            if weights_device != device {
                return Err(Fn3Error::DeviceMismatch {
                    op: None,
                    expected: device,
                    actual: weights_device,
                });
            }
        }
        if let Some(config) = kv_config {
            config.validate(&envelope)?;
        }
        Ok(PreparedModel {
            plan: Arc::new(plan),
            weights: Arc::new(weights),
            device,
            envelope,
            kv_config,
        })
    }

    /// The execution plan.
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// The immutable weights.
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// The device the model lives on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The declared dynamic bounds.
    pub fn envelope(&self) -> ShapeEnvelope {
        self.envelope
    }

    /// The KV cache geometry, for models that have one.
    pub fn kv_config(&self) -> Option<&KvCacheConfig> {
        self.kv_config.as_ref()
    }

    /// Run the plan against pre-allocated value slots.
    ///
    /// See [`ExecutionPlan::execute`]. This runs the model's own fixed plan;
    /// [`Session`] wraps it with per-shape specialisation, cache append and
    /// counter commit.
    pub fn execute(
        &self,
        backend: &dyn Backend,
        slots: &mut [Option<Tensor>],
    ) -> Result<(), Fn3Error> {
        if backend.device() != self.device {
            return Err(Fn3Error::DeviceMismatch {
                op: None,
                expected: self.device,
                actual: backend.device(),
            });
        }
        self.plan.execute(backend, slots)
    }

    /// Check a concrete dynamic shape against the envelope.
    pub fn check_dims(&self, dims: &LiveDims) -> Result<(), Fn3Error> {
        self.envelope.check(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fn3::backend::cpu::CpuBackend;
    use crate::fn3::dtype::DType;
    use crate::fn3::shape::Shape;

    fn small_model() -> PreparedModel {
        let plan =
            ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![2])], 3).unwrap();
        let weights = ModelWeights::new(vec![
            Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap()
        ])
        .unwrap();
        PreparedModel::new(
            plan,
            weights,
            DeviceId::Cpu,
            ShapeEnvelope::uniform(4).unwrap(),
            Some(KvCacheConfig {
                layers: 1,
                capacity_tokens: 4,
                batch: 1,
                n_kv_heads: 2,
                head_dim: 2,
                dtype: DType::F32,
                layout: KvCacheLayout::BatchHeadsCapacityHeadDim,
            }),
        )
        .unwrap()
    }

    #[test]
    fn plan_runs_add_and_keeps_slots_on_failure() {
        let model = small_model();
        let backend = CpuBackend::new();
        let lhs = Tensor::from_f32(Shape::from_dims([2]), &[1.0, 2.0]).unwrap();
        let rhs = Tensor::from_f32(Shape::from_dims([2]), &[10.0, 20.0]).unwrap();
        let out = Tensor::zeros(Shape::from_dims([2]), DType::F32, DeviceId::Cpu).unwrap();
        let mut slots = vec![Some(lhs), Some(rhs), Some(out)];
        model.execute(&backend, &mut slots).unwrap();
        assert_eq!(
            slots[2].as_ref().unwrap().as_f32_slice().unwrap(),
            &[11.0, 22.0]
        );
    }

    #[test]
    fn plan_rejects_read_write_aliasing() {
        let error = ExecutionPlan::new(vec![Instruction::new(OpId::Add, vec![0, 1], vec![0])], 2)
            .unwrap_err();
        assert!(error.to_string().contains("reads and writes"), "{error}");
    }
}
