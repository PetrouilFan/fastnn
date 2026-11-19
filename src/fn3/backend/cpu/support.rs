//! The CPU backend's honest support table.
//!
//! [`support`] is the single place that answers "can the alpha CPU reference run
//! this op?". [`crate::inference::backend::cpu::CpuBackend::supports`] delegates to it
//! and [`crate::inference::backend::cpu::CpuBackend::execute_op`] dispatches from the
//! same answer, so a supported id can never fall through to "not implemented"
//! and an unsupported id can never be executed.
//!
//! # What the pinned graph actually contains
//!
//! Measured on `onnx-community/tiny-random-gpt2-ONNX` revision
//! `90f61e71d6fa8e571d0ab0f95a637a5d7d8ed52f`, `onnx/model.onnx`,
//! 27,897,974 bytes, SHA-256
//! `e18370ba6247c564a4aef7b6a83aa86d5d036f47a73c0935e51bbcfe178141a1`
//! (opset 14, IR 7, 257 nodes, 48 initializers), loaded with `onnx.load` and
//! enumerated with `collections.Counter(node.op_type)`:
//!
//! ```text
//! Unsqueeze 37   Gather 35      Reshape 25    Concat 23     Shape 20
//! Add 19         Mul 12         Transpose 11  ReduceMean 10 Gemm 8
//! Pow 7          Sub 7          Div 5         MatMul 5      Sqrt 5
//! Where 5        Slice 4        Cast 3        ConstantOfShape 2
//! Equal 2        Expand 2       Softmax 2     Split 2       Squeeze 2
//! Tanh 2         Less 1         Range 1
//! ```
//!
//! 27 distinct types. Two consequences are worth stating because they contradict
//! common guesses about a GPT-2 export:
//!
//! 1. **There is no `Erf` and no `Gelu` node.** `gelu_new` is decomposed by the
//!    exporter into `0.5*x*(1+tanh(0.7978846*(x+0.044715*x^3)))`, i.e. `Pow`,
//!    `Mul`, `Add`, `Tanh`, `Mul`. `Erf`/`Gelu` are still implemented because
//!    the `gelu.erf` fixture pins the exact form and because a future export may
//!    keep the fused node; the tanh form is exercised as a decomposition.
//! 2. **`Gemm` is not an op id.** The importer (#54) maps each `Gemm` node to
//!    `MatMul` followed by `Add` (bias). All eight `Gemm` nodes have
//!    `transA = transB = 0` and `alpha = beta = 1`, and their weight
//!    initializers are `[in, out]`.
//!
//! # Function coverage
//!
//! | ONNX node | fn3 op | Notes |
//! |---|---|---|
//! | `Gather` (axis 0) | `Gather` | embedding lookup, shape indexing |
//! | `MatMul` | `MatMul` | rank 2/3/4, batch broadcast |
//! | `Gemm` | `MatMul` + `Add` | importer composition |
//! | `Add` `Sub` `Mul` `Div` `Pow` `Sqrt` `Tanh` | same | broadcast `f32` |
//! | `ReduceMean` (axes `[-1]`) | `ReduceMean` | |
//! | `Softmax` (axis `-1`) | `Softmax` | all-masked rows uniform |
//! | `Transpose` (perm or default) | `Transpose` | |
//! | `Reshape` `Squeeze` `Unsqueeze` `Flatten` | same | |
//! | `Concat` `Split` `Slice` `Expand` | same | |
//! | `Shape` `Range` `ConstantOfShape` | same | `ConstantOfShape` value is a scalar input |
//! | `Cast` | `Cast` | target dtype is the output tensor's dtype |
//! | `Equal` `Less` | `Equal` `Less` `Greater` | `Bool` output |
//! | `Where` | `Where` | three-way broadcast |
//!
//! `MaskedAttention` (id 40) is **not** implemented on the CPU. Per
//! `docs/roadmap/fn3-architecture.md` §6 it is the reserved fused id for the
//! Vulkan attention kernels (FN3-15 / #61); the CPU reference proves the same
//! arithmetic through `MatMul` → `Mul`/`Div` → `Add` → `Softmax` → `MatMul`,
//! which is what the `attention` fixture exercises. Reporting `Unsupported`
//! here is the honest answer: the backend has no fused kernel to advertise, and
//! `Capabilities::features.masked_attention` stays `false`.
//!
//! # Why every other id is `Full`
//!
//! `Support::Full` is only claimed for an id that has a kernel in
//! [`crate::inference::backend::cpu::kernels`], a dispatch arm, and at least one test
//! in `tests/fn3_cpu_reference.rs`. `Support::Partial` is never returned: the
//! documented contract of each kernel is its whole implemented surface, and a
//! request outside it (for example a rank-1 `MatMul` operand, or a LayerNorm
//! over a trailing block) is rejected by the kernel with `UnsupportedOp` naming
//! FN3-07 (#53) and the reason. That keeps "supported" from decaying into
//! "sometimes approximates".

use crate::inference::backend::Support;
use crate::inference::ops::OpId;

/// Every op id the alpha CPU reference executes.
pub(crate) const IMPLEMENTED: [OpId; 39] = [
    OpId::Gather,
    OpId::MatMul,
    OpId::Add,
    OpId::Sub,
    OpId::Mul,
    OpId::Div,
    OpId::Neg,
    OpId::Pow,
    OpId::Sqrt,
    OpId::Erf,
    OpId::Tanh,
    OpId::Gelu,
    OpId::Relu,
    OpId::Softmax,
    OpId::LayerNorm,
    OpId::ReduceMean,
    OpId::ReduceSum,
    OpId::ArgMax,
    OpId::Transpose,
    OpId::Reshape,
    OpId::Flatten,
    OpId::Squeeze,
    OpId::Unsqueeze,
    OpId::Concat,
    OpId::Slice,
    OpId::Split,
    OpId::Shape,
    OpId::ConstantOfShape,
    OpId::Expand,
    OpId::Range,
    OpId::Cast,
    OpId::Equal,
    OpId::Less,
    OpId::Greater,
    OpId::Where,
    OpId::Trilu,
    OpId::Sin,
    OpId::Cos,
    OpId::ScatterND,
];

/// The support answer for `op`.
pub(crate) fn support(op: OpId) -> Support {
    match op {
        OpId::MaskedAttention => Support::Unsupported {
            reason: "the fused attention id is reserved for the Vulkan attention kernels \
                     (FN3-15 / #61). The CPU reference computes the same result by decomposition \
                     (matmul, scale, additive mask, softmax, matmul) and does not approximate the \
                     fused id"
                .to_string(),
        },
        other => {
            debug_assert!(
                IMPLEMENTED.contains(&other),
                "op {other} has a kernel but is missing from support::IMPLEMENTED"
            );
            Support::Full
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn implemented_is_a_subset_of_all_ids_without_duplicates() {
        let mut seen = HashSet::new();
        let all: HashSet<OpId> = OpId::ALL.into_iter().collect();
        for op in IMPLEMENTED {
            assert!(seen.insert(op), "duplicate {op}");
            assert!(all.contains(&op), "{op} is not in OpId::ALL");
        }
    }

    #[test]
    fn only_masked_attention_is_unsupported() {
        for op in OpId::ALL {
            let support = support(op);
            if op == OpId::MaskedAttention {
                assert!(matches!(support, Support::Unsupported { .. }));
                assert!(support.reason().unwrap().contains("#61"));
            } else {
                assert!(support.is_full(), "{op} reports {support}");
                assert!(
                    IMPLEMENTED.contains(&op),
                    "{op} is missing from IMPLEMENTED"
                );
            }
            assert!(
                !matches!(support, Support::Partial { .. }),
                "{op} must not report Partial"
            );
        }
    }
}
