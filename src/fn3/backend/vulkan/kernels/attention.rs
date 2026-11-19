//! GPU kernel owned by FN3-15 (#61): fused masked multi-head attention.
//!
//! # Contract
//!
//! `MaskedAttention` (id 40) is the reserved *fused* attention id. Its result is
//! exactly the decomposition the committed `attention` fixture and the pinned
//! GPT-2 graph evaluate:
//!
//! ```text
//! scores[b,h,i,j] = sum_d q[b,h,i,d] * k[b,kvh,j,d] / sqrt(head_dim)
//!                   + mask[b|1, h|1, i, j]
//! probs           = softmax(scores) over j, max-subtracted, ascending
//! out[b,h,i,d]    = sum_j probs[i,j] * v[b,kvh,j,d]        (ascending j)
//! ```
//!
//! * `q`  is `[batch, heads, queries, head_dim]`, `f32`, contiguous.
//! * `k`  and `v` are `[batch, kv_heads, keys, head_dim]`, `f32`, contiguous.
//! * `heads` must be a multiple of `kv_heads`; query head `h` reads KV head
//!   `h / (heads / kv_heads)` (grouped-query attention; the identity when the
//!   two are equal).
//! * the optional fourth input is an additive `f32` mask shaped
//!   `[batch|1, heads|1, queries, keys]`; the pinned fill value is the finite
//!   `-FLT_MAX`;
//! * the output is `q`'s shape. An optional second output receives `probs`
//!   (`[batch, heads, queries, keys]`); without it the probabilities never
//!   leave shared memory.
//!
//! # Attributes
//!
//! `OpParams` is the alpha attribute carrier and has exactly one boolean, so
//! the fusion's `causal` attribute rides in `keep_dims`:
//!
//! | `OpParams` | meaning for `MaskedAttention` |
//! |---|---|
//! | `axes` | must be empty; attention has no axis attribute |
//! | `keep_dims` | `true` applies an implicit causal mask |
//! | `epsilon` | must be `0.0`; the scale is fixed at `1/sqrt(head_dim)` |
//!
//! `keep_dims` never changes the output shape. Causal masking replaces the
//! effective mask value with `-FLT_MAX` above the diagonal (shifted by
//! `keys - queries` so cached decode rows stay aligned) instead of adding a
//! second mask term: adding two `-FLT_MAX` values would saturate to `-inf` and
//! turn max subtraction into a NaN.
//!
//! # Masked-row semantics
//!
//! A query row whose effective mask is entirely `-FLT_MAX` is **exactly
//! uniform**: max subtraction makes every shifted score `0`, so `exp(0) = 1`,
//! the ascending sum is the key count, and every probability is `1 / keys`.
//! That is finite by construction. It is the pinned GPT-2 mask convention and a
//! conformance requirement, not a tolerance question.
//!
//! # Honest limits
//!
//! `Support::Full` is claimed for the **documented fused contract above**, and
//! the bound is part of that contract: `keys <= 4096` because the row's scores
//! live in 16 KiB of shared memory (the Vulkan minimum
//! `maxComputeSharedMemorySize`). Rejected by name, never approximated:
//!
//! * `keys > 4096`;
//! * any rank other than 4, any dtype other than `f32`, any non-contiguous
//!   operand;
//! * a mask whose batch/head axes are neither 1 nor the full extent, or whose
//!   query/key extents do not match;
//! * causal masking with more queries than keys;
//! * more than two outputs, or an output whose shape is not `q` / `probs`.
//!
//! `Full` rather than `Partial` is the same reading the CPU reference uses for
//! every op it implements: `Full` means "the whole documented surface of this
//! kernel", and a request outside it is a loud `UnsupportedOp`. `MaskedAttention`
//! has no wider shared contract to fall short of — the reserved fused id is
//! defined here — so `Full` is the honest answer and the capability flag
//! `features.masked_attention` (which the platform layer derives from a `Full`
//! registration) can advertise the fused path to the planner. A longer-context
//! kernel is a follow-up, and its absence is reported by name, not silently
//! degraded.

use super::super::error::{Error, Result};
use super::super::VulkanBackend;
use super::io;
use super::norm::{execute_staged, require_shape, unsupported, BufferSource, ParamWriter};
use super::{BindingSpec, GridInput, KernelSet, KernelSpec, OpKernel};
use crate::fn3::backend::{CapabilityFeatures, OpParams, Support};
use crate::fn3::dtype::DType;
use crate::fn3::error::{Fn3Error, Fn3Result};
use crate::fn3::ops::OpId;
use crate::fn3::shape::Shape;
use crate::fn3::tensor::{TensorMut, TensorView};

/// Owning issue named in this family's rejection messages.
const KERNEL_OWNER: &str = "FN3-15 (#61)";

/// The key extent the shared-memory score row supports (`4096 * 4 = 16 KiB`,
/// the Vulkan minimum `maxComputeSharedMemorySize`).
pub const MAX_KEYS: u64 = 4096;

/// Byte size of the `attention.fused.f32` parameter block: twelve 4-byte
/// scalars, in the order the shader declares them.
pub const ATTENTION_PARAMS_BYTES: usize = 12 * 4;

/// Fused masked attention (`shaders/fn3_attention_fused.comp`).
pub const ATTENTION_FUSED: KernelSpec = KernelSpec {
    name: "attention.fused.f32",
    spirv: include_bytes!("../shaders/fn3_attention_fused.comp.spv"),
    entry_point: "main",
    bindings: &[
        BindingSpec::storage(0, "q"),
        BindingSpec::storage(1, "k"),
        BindingSpec::storage(2, "v"),
        BindingSpec::storage(3, "mask"),
        BindingSpec::storage(4, "params"),
        BindingSpec::storage(5, "out"),
        BindingSpec::storage(6, "probs"),
    ],
    push_constant_bytes: 0,
    workgroup_size: [64, 1, 1],
    dtypes: &[DType::F32],
    shape: "q [B,H,S,D], k/v [B,KV,S_kv,D], H % KV == 0, optional additive [B|1,H|1,S,S_kv] \
            mask, keys <= 4096, one workgroup per (batch, head, query)",
    grid: attention_grid,
};

/// An all-zero four-byte buffer bound when the caller supplies no mask.
static ZERO_MASK: [u8; 4] = [0; 4];

/// Dispatch geometry: one workgroup per `(batch, head, query)` row.
fn attention_grid(input: &GridInput<'_>) -> Result<[u32; 3]> {
    let dims = input.output_dims;
    if dims.len() != 4 {
        return Err(Error::internal(format!(
            "attention expects a rank-4 output, got {} dimension(s)",
            dims.len()
        )));
    }
    let mut counts = [0u32; 3];
    // x = queries, y = heads, z = batch.
    for (slot, dim) in [(0usize, dims[2]), (1, dims[1]), (2, dims[0])] {
        counts[slot] = u32::try_from(dim.max(1)).map_err(|_| {
            Error::internal(format!(
                "attention workgroup axis {slot} needs {dim} groups, above the 32-bit dispatch \
                 limit"
            ))
        })?;
    }
    Ok(counts)
}

/// Register the fused attention kernel.
pub fn register(set: &mut KernelSet) {
    set.register(OpKernel {
        op: OpId::MaskedAttention,
        // `Full` covers the fused contract documented at the top of this file
        // (rank-4 f32, grouped-query grouping, additive/causal masks, keys <=
        // 4096). Everything outside it is rejected by name; see the module
        // docs for why `Full` is the honest reading for the reserved fused id.
        support: Support::Full,
        features: CapabilityFeatures {
            masked_attention: true,
            ..CapabilityFeatures::default()
        },
        dtypes: &[DType::F32],
        admits: None,
        kernel: &ATTENTION_FUSED,
        launch: launch_attention,
    });
}

/// Execute fused masked attention through the platform layer.
pub fn launch_attention(
    backend: &VulkanBackend,
    inputs: &[TensorView<'_>],
    outputs: &mut [TensorMut<'_>],
    params: &OpParams,
) -> Fn3Result<()> {
    let op = OpId::MaskedAttention;
    if !(3..=4).contains(&inputs.len()) {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan masked_attention arity",
            detail: format!(
                "op {op} takes q, k, v and an optional additive mask (3 or 4 inputs); {} were \
                 supplied",
                inputs.len()
            ),
        });
    }
    if !(1..=2).contains(&outputs.len()) {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan masked_attention arity",
            detail: format!(
                "op {op} writes out and optionally probs (1 or 2 outputs); {} were supplied",
                outputs.len()
            ),
        });
    }
    if !params.axes.is_empty() {
        return Err(unsupported(
            op,
            format!(
                "the fused attention contract has no axis attribute, but axes {:?} were supplied",
                params.axes
            ),
        ));
    }
    if params.epsilon != 0.0 {
        return Err(Fn3Error::InvalidShape {
            context: "vulkan masked_attention attributes",
            detail: format!(
                "op {op}: the scale is fixed at 1/sqrt(head_dim); epsilon must be 0.0, found {}",
                params.epsilon
            ),
        });
    }
    let causal = params.keep_dims;

    let (q, k, v) = (&inputs[0], &inputs[1], &inputs[2]);
    for view in [q, k, v] {
        if view.shape().rank() != 4 {
            return Err(Fn3Error::RankMismatch {
                op: Some(op),
                expected: 4,
                actual: view.shape().rank(),
            });
        }
    }
    let q_dims = q.shape().dims();
    let k_dims = k.shape().dims();
    let v_dims = v.shape().dims();
    let (batch, heads, queries, head_dim) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (kv_heads, keys) = (k_dims[1], k_dims[2]);
    for (view, what) in [(k, "k"), (v, "v")] {
        if view.shape().dims()[0] != batch
            || view.shape().dims()[2] != keys
            || view.shape().dims()[3] != head_dim
        {
            return Err(Fn3Error::ShapeMismatch {
                op: Some(op),
                expected: format!("[{batch}, {kv_heads}, {keys}, {head_dim}] for attention {what}"),
                actual: view.shape().to_string(),
            });
        }
    }
    if k_dims[1] != v_dims[1] {
        return Err(Fn3Error::ShapeMismatch {
            op: Some(op),
            expected: format!(
                "[{batch}, {}, {keys}, {head_dim}] for attention v",
                k_dims[1]
            ),
            actual: v.shape().to_string(),
        });
    }
    if kv_heads == 0 || heads == 0 || !heads.is_multiple_of(kv_heads) {
        return Err(unsupported(
            op,
            format!(
                "grouped-query attention requires n_kv_heads to divide n_heads; q declares \
                 {heads} head(s) and k/v declare {kv_heads}"
            ),
        ));
    }
    if head_dim == 0 {
        return Err(unsupported(
            op,
            format!("attention needs a non-zero head_dim, found {head_dim}"),
        ));
    }
    if keys > MAX_KEYS {
        return Err(unsupported(
            op,
            format!(
                "{keys} keys exceed the {MAX_KEYS}-key shared-memory score row of this kernel; \
                 split the sequence"
            ),
        ));
    }
    if causal && queries > keys {
        return Err(unsupported(
            op,
            format!(
                "causal attention needs at least as many keys as queries, found {queries} \
                 queries over {keys} keys"
            ),
        ));
    }

    let out_shape = q.shape().clone();
    let probs_shape = Shape::from_dims([batch, heads, queries, keys]);
    require_shape(op, &out_shape, outputs[0].shape())?;
    if outputs.len() == 2 {
        require_shape(op, &probs_shape, outputs[1].shape())?;
    }

    let q_operand = io::input(KERNEL_OWNER, backend, op, "attention.q", q, DType::F32)?;
    let k_operand = io::input(KERNEL_OWNER, backend, op, "attention.k", k, DType::F32)?;
    let v_operand = io::input(KERNEL_OWNER, backend, op, "attention.v", v, DType::F32)?;
    let mask_operand = match inputs.get(3) {
        Some(mask) => Some(io::input(
            KERNEL_OWNER,
            backend,
            op,
            "attention.mask",
            mask,
            DType::F32,
        )?),
        None => None,
    };

    let mask_shape = inputs.get(3).map(|mask| mask.shape().clone());
    let (mask_batch, mask_heads) = match &mask_shape {
        Some(shape) => {
            if shape.rank() != 4 {
                return Err(Fn3Error::RankMismatch {
                    op: Some(op),
                    expected: 4,
                    actual: shape.rank(),
                });
            }
            let dims = shape.dims();
            if (dims[0] != 1 && dims[0] != batch)
                || (dims[1] != 1 && dims[1] != heads)
                || dims[2] != queries
                || dims[3] != keys
            {
                return Err(Fn3Error::ShapeMismatch {
                    op: Some(op),
                    expected: format!(
                        "[{batch}|1, {heads}|1, {queries}, {keys}] for the additive attention mask"
                    ),
                    actual: shape.to_string(),
                });
            }
            (dims[0], dims[1])
        }
        None => (1, 1),
    };

    for (numel, what) in [(q.numel()?, "q"), (k.numel()?, "k"), (v.numel()?, "v")] {
        if numel > u64::from(u32::MAX) {
            return Err(unsupported(
                op,
                format!(
                    "attention {what} has {numel} elements, above the 32-bit addressing of the \
                     kernel's parameter block; split the batch"
                ),
            ));
        }
    }

    let queries32 = u32::try_from(queries).expect("queries <= q.numel() <= u32::MAX");
    let heads32 = u32::try_from(heads).expect("heads <= q.numel() <= u32::MAX");
    let batch32 = u32::try_from(batch).expect("batch <= q.numel() <= u32::MAX");
    let kv_heads32 = u32::try_from(kv_heads).expect("kv_heads <= k.numel() <= u32::MAX");
    let keys32 = u32::try_from(keys).expect("keys <= k.numel() <= u32::MAX");
    let head_dim32 = u32::try_from(head_dim).expect("head_dim <= q.numel() <= u32::MAX");

    if batch == 0 || heads == 0 || queries == 0 || keys == 0 || head_dim == 0 {
        return Ok(());
    }

    let (out_slot, probs_slot) = outputs.split_at_mut(1);
    let out_operand = io::output(
        KERNEL_OWNER,
        backend,
        op,
        "attention.out",
        &mut out_slot[0],
        DType::F32,
    )?;
    let probs_operand = match probs_slot.first_mut() {
        Some(probs) => Some(io::output(
            KERNEL_OWNER,
            backend,
            op,
            "attention.probs",
            probs,
            DType::F32,
        )?),
        None => None,
    };

    let mut block = ParamWriter::new();
    block
        .u32(batch32)
        .u32(heads32)
        .u32(kv_heads32)
        .u32(queries32)
        .u32(keys32)
        .u32(head_dim32)
        .u32(u32::try_from(mask_batch).expect("mask batch <= batch"))
        .u32(u32::try_from(mask_heads).expect("mask heads <= heads"))
        .u32(u32::from(mask_operand.is_some()))
        .u32(u32::from(causal))
        .u32(u32::from(probs_operand.is_some()))
        .f32((head_dim as f32).sqrt());
    let params_bytes = block.finish();
    debug_assert_eq!(params_bytes.len(), ATTENTION_PARAMS_BYTES);

    let workgroups = ATTENTION_FUSED
        .workgroups(q.numel()?, out_shape.dims())
        .map_err(|error| error.to_fn3())?;
    let probs_source = match probs_operand {
        Some(operand) => BufferSource::output("attention.probs", operand),
        None => BufferSource::Scratch {
            name: "attention.probs.suppressed",
            size: 4,
        },
    };
    let mask_source = match mask_operand {
        Some(operand) => BufferSource::input("attention.mask", operand),
        None => BufferSource::Upload {
            name: "attention.mask.absent",
            bytes: &ZERO_MASK,
        },
    };
    let mut sources = [
        BufferSource::input("attention.q", q_operand),
        BufferSource::input("attention.k", k_operand),
        BufferSource::input("attention.v", v_operand),
        mask_source,
        BufferSource::Upload {
            name: "attention.params",
            bytes: &params_bytes,
        },
        BufferSource::output("attention.out", out_operand),
        probs_source,
    ];
    execute_staged(op, backend, &ATTENTION_FUSED, &mut sources, &[], workgroups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_matches_the_committed_shader_metadata() {
        assert_eq!(ATTENTION_FUSED.workgroup_size, [64, 1, 1]);
        assert_eq!(ATTENTION_FUSED.push_constant_bytes, 0);
        assert_eq!(ATTENTION_FUSED.bindings.len(), 7);
        assert!(ATTENTION_FUSED.validate().is_ok());
    }

    #[test]
    fn grid_is_query_by_head_by_batch() {
        let dims = [2u64, 3, 5, 7];
        let input = GridInput {
            element_count: 210,
            output_dims: &dims,
            workgroup_size: [64, 1, 1],
        };
        assert_eq!(attention_grid(&input).unwrap(), [5, 3, 2]);
        let wrong = [2u64, 3, 5];
        let input = GridInput {
            element_count: 30,
            output_dims: &wrong,
            workgroup_size: [64, 1, 1],
        };
        assert!(attention_grid(&input).is_err());
    }

    #[test]
    fn shared_memory_footprint_is_the_vulkan_minimum() {
        // 4096 f32 keys = 16384 bytes, the smallest
        // `maxComputeSharedMemorySize` a Vulkan device may report.
        assert_eq!(MAX_KEYS * 4, 16_384);
        assert_eq!(ATTENTION_PARAMS_BYTES, 48);
    }
}
