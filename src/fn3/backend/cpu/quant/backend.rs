//! A CPU backend that executes registered projections with the fused W8A16
//! kernel (FN3-23 / #69).
//!
//! # Why a backend wrapper
//!
//! The session executes every instruction through one [`Backend`]. Adding a
//! quantized `OpId`, changing `CpuBackend::admits`, or teaching
//! `model/session.rs` about packed weights would each change a contract owned
//! by another issue. This type instead implements the existing trait:
//!
//! * `supports`, `capabilities`, `device` and `prepare_persistent` are the CPU
//!   reference backend's, unchanged;
//! * a `MatMul` whose right-hand side is a registered packed weight is
//!   dispatched to [`matmul_quantized`];
//! * every other op — including a `MatMul` whose right-hand side is computed, a
//!   strided invocation, or a projection the policy leaves in FP32 — is
//!   delegated to [`CpuBackend`] and counted as an explicit fallback.
//!
//! `Backend::admits` is deliberately **not** overridden: the default body is
//! `supports`, so plan preflight sees exactly the CPU reference verdicts. The
//! quantized subset is a per-projection registration
//! ([`ProjectionRegistration`]), not a new op-level capability.
//!
//! # Failure policy
//!
//! The kernel runs only for `ProjectionRegistration::Admitted`. A `Partial`
//! registration (a strided activation) and every `Fallback` delegates before any
//! output byte is written. If the kernel itself reports
//! [`Fn3Error::UnsupportedOp`] after being admitted, that is the documented
//! "outside the named subset" rejection: the backend records it and delegates.
//! Any other error is a real failure and is returned unchanged.
//!
//! # Counters
//!
//! Counters are `AtomicU64` per registered weight plus a small map of fallback
//! codes, so a step can be measured without a global lock on the hot path. Call
//! [`QuantCpuBackend::stats`] after a prefill or decode to obtain the coverage
//! evidence.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::fn3::backend::{Backend, Capabilities, OpParams, Support};
use crate::fn3::device::DeviceId;
use crate::fn3::error::Fn3Error;
use crate::fn3::ops::OpId;
use crate::fn3::tensor::{TensorMut, TensorView};

use super::super::CpuBackend;
use super::kernel::{matmul_quantized, PackedView, WorkHint};
use super::planner::{
    registration, ActivationSpec, ProjectionRegistration, QuantPolicy, QuantWeightSet,
};

/// Backend name used in quantized-path diagnostics.
pub const QUANT_CPU_BACKEND_NAME: &str = "cpu+w8a16";

/// Fallback code used when a `MatMul` right-hand side is not a registered
/// packed weight (a dynamic value, the lm_head transpose, or a weight the
/// policy left in FP32).
pub const UNPACKED_RHS: &str = "unpacked-rhs";

/// One registered weight's dispatch counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightStats {
    /// Model weight name.
    pub name: String,
    /// Quantized dispatches.
    pub calls: u64,
    /// Total time spent inside the fused kernel, in nanoseconds.
    pub total_nanos: u64,
}

impl WeightStats {
    /// Mean kernel time per dispatch, in nanoseconds.
    pub fn mean_nanos(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_nanos as f64 / self.calls as f64
        }
    }
}

/// A snapshot of the quantized path's activity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantStats {
    /// Dispatches that ran the fused kernel.
    pub quantized_calls: u64,
    /// Dispatches delegated to the FP32 backend.
    pub fallback_calls: u64,
    /// Per-weight counters, in registration order.
    pub weights: Vec<WeightStats>,
    /// Fallback counts keyed by stable code.
    pub fallback_reasons: BTreeMap<String, u64>,
}

impl QuantStats {
    /// `true` when every registered weight was used at least once.
    pub fn all_registered_weights_used(&self) -> bool {
        self.weights.iter().all(|weight| weight.calls > 0)
    }

    /// Total nanoseconds attributed to the fused kernel.
    pub fn total_kernel_nanos(&self) -> u64 {
        self.weights.iter().map(|weight| weight.total_nanos).sum()
    }
}

#[derive(Debug)]
struct StatsCore {
    quantized_calls: AtomicU64,
    fallback_calls: AtomicU64,
    hits: Vec<AtomicU64>,
    nanos: Vec<AtomicU64>,
    fallback: Mutex<BTreeMap<&'static str, u64>>,
}

impl StatsCore {
    fn new(weights: usize) -> StatsCore {
        StatsCore {
            quantized_calls: AtomicU64::new(0),
            fallback_calls: AtomicU64::new(0),
            hits: (0..weights).map(|_| AtomicU64::new(0)).collect(),
            nanos: (0..weights).map(|_| AtomicU64::new(0)).collect(),
            fallback: Mutex::new(BTreeMap::new()),
        }
    }

    fn record_fallback(&self, code: &'static str) {
        self.fallback_calls.fetch_add(1, Ordering::Relaxed);
        let mut counts = self
            .fallback
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *counts.entry(code).or_insert(0) += 1;
    }
}

/// CPU execution with registered W8A16 projections.
#[derive(Debug)]
pub struct QuantCpuBackend {
    inner: CpuBackend,
    weights: QuantWeightSet,
    policy: QuantPolicy,
    hint: WorkHint,
    stats: StatsCore,
}

impl QuantCpuBackend {
    /// Wrap the CPU reference backend around a registered weight set.
    pub fn new(weights: QuantWeightSet, policy: QuantPolicy, hint: WorkHint) -> QuantCpuBackend {
        let slots = weights.len();
        QuantCpuBackend {
            inner: CpuBackend::new(),
            weights,
            policy,
            hint,
            stats: StatsCore::new(slots),
        }
    }

    /// Register a plan's static projection weights and build the backend.
    pub fn from_plan(
        plan: &crate::fn3::model::Plan,
        weights: &crate::fn3::model::ModelWeights,
        policy: QuantPolicy,
        hint: WorkHint,
    ) -> Result<QuantCpuBackend, Fn3Error> {
        let set = QuantWeightSet::from_plan(plan, weights, policy)?;
        Ok(QuantCpuBackend::new(set, policy, hint))
    }

    /// A serial work hint (the reference configuration).
    pub fn serial(weights: QuantWeightSet) -> QuantCpuBackend {
        QuantCpuBackend::new(weights, QuantPolicy::default(), WorkHint::serial())
    }

    /// The registered weights and their coverage report.
    pub fn weights(&self) -> &QuantWeightSet {
        &self.weights
    }

    /// The policy used for per-invocation registration.
    pub fn policy(&self) -> QuantPolicy {
        self.policy
    }

    /// The work hint passed to every kernel call.
    pub fn hint(&self) -> WorkHint {
        self.hint
    }

    /// Reset the counters.
    pub fn reset_stats(&self) {
        self.stats.quantized_calls.store(0, Ordering::Relaxed);
        self.stats.fallback_calls.store(0, Ordering::Relaxed);
        for counter in &self.stats.hits {
            counter.store(0, Ordering::Relaxed);
        }
        for counter in &self.stats.nanos {
            counter.store(0, Ordering::Relaxed);
        }
        let mut counts = self
            .stats
            .fallback
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        counts.clear();
    }

    /// Snapshot the counters.
    pub fn stats(&self) -> QuantStats {
        let weights = self
            .weights
            .entries()
            .iter()
            .enumerate()
            .map(|(index, entry)| WeightStats {
                name: entry.spec().name.clone(),
                calls: self.stats.hits[index].load(Ordering::Relaxed),
                total_nanos: self.stats.nanos[index].load(Ordering::Relaxed),
            })
            .collect();
        let fallback_reasons = self
            .stats
            .fallback
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|(code, count)| ((*code).to_string(), *count))
            .collect();
        QuantStats {
            quantized_calls: self.stats.quantized_calls.load(Ordering::Relaxed),
            fallback_calls: self.stats.fallback_calls.load(Ordering::Relaxed),
            weights,
            fallback_reasons,
        }
    }

    fn record_hit(&self, index: usize, elapsed: Duration) {
        self.stats.quantized_calls.fetch_add(1, Ordering::Relaxed);
        self.stats.hits[index].fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.stats.nanos[index].fetch_add(nanos, Ordering::Relaxed);
    }

    fn try_quantized(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
    ) -> Result<bool, Fn3Error> {
        if op != OpId::MatMul || inputs.len() != 2 || outputs.len() != 1 {
            return Ok(false);
        }
        let Some((index, entry)) = self.weights.lookup(&inputs[1]) else {
            self.stats.record_fallback(UNPACKED_RHS);
            return Ok(false);
        };
        let activation = ActivationSpec::of(&inputs[0]);
        match registration(entry.spec(), activation, &self.policy) {
            ProjectionRegistration::Admitted => {}
            other => {
                self.stats.record_fallback(other.code());
                return Ok(false);
            }
        }
        let started = Instant::now();
        match matmul_quantized(
            op,
            &inputs[0],
            &entry.weight().view(),
            &mut outputs[0],
            self.hint,
        ) {
            Ok(()) => {
                self.record_hit(index, started.elapsed());
                Ok(true)
            }
            Err(Fn3Error::UnsupportedOp { .. }) => {
                // The kernel's named subset is narrower than the registration
                // predicted; its rejection happens before the first output
                // write, so delegating is safe and explicit.
                self.stats.record_fallback("activation-layout");
                Ok(false)
            }
            Err(error) => Err(error),
        }
    }

    /// The view the kernel would see for one registered weight, for tests and
    /// reporting. `None` when the name is not registered.
    pub fn packed_view(&self, name: &str) -> Option<PackedView<'_>> {
        self.weights.weight(name).map(|weight| weight.view())
    }
}

impl Backend for QuantCpuBackend {
    fn name(&self) -> &'static str {
        QUANT_CPU_BACKEND_NAME
    }

    fn device(&self) -> DeviceId {
        self.inner.device()
    }

    fn capabilities(&self) -> &Capabilities {
        self.inner.capabilities()
    }

    fn supports(&self, op: OpId) -> Support {
        self.inner.supports(op)
    }

    fn execute_op(
        &self,
        op: OpId,
        inputs: &[TensorView<'_>],
        outputs: &mut [TensorMut<'_>],
        params: &OpParams,
    ) -> Result<(), Fn3Error> {
        self.preflight(op)?;
        for view in inputs {
            view.meta().require_device(Some(op), self.device())?;
            self.capabilities().check_rank(view.shape().rank())?;
        }
        for output in outputs.iter() {
            output.meta().require_device(Some(op), self.device())?;
            self.capabilities().check_rank(output.shape().rank())?;
        }
        if self.try_quantized(op, inputs, outputs)? {
            return Ok(());
        }
        self.inner.execute_op(op, inputs, outputs, params)
    }
}

impl std::fmt::Display for QuantCpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(QUANT_CPU_BACKEND_NAME)
    }
}
