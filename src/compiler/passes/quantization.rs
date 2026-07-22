//! Quantization compiler pass.
//!
//! Finds f32 Constant weight nodes feeding MatMul/Conv2d ops and replaces
//! them with packed I4/I8 data carrying per-channel scale/zero-point metadata.
//! The backend dispatch already selects `matmul_u4`/`matmul_u8` kernels when
//! it sees `IrDType::I4/U8` on the weight input.

use crate::dtypes::{F4x8, F8x4, F8x4R, I4x8, I8x4, U4x8, U8x4};
use crate::error::FastnnError;
use crate::ir::{ComputeGraph, DimExpr, IrDType, NodeId, Opcode, TensorType, TensorValue};
use crate::packed_tensor::PackedTensor;
use crate::types::{QuantTarget, RepresentationTransform, ScalarType};

fn optimizer_affine_metadata(tensor_type: &TensorType) -> Option<(usize, Vec<f32>, Vec<f32>)> {
    let representation = tensor_type.value_representation().ok()?;
    let bit_width = match representation.storage {
        ScalarType::I4 => 4,
        ScalarType::I8 => 8,
        _ => return None,
    };
    match representation.transform {
        RepresentationTransform::AffineDequantization {
            scales, offsets, ..
        } => Some((bit_width, scales, offsets)),
        _ => None,
    }
}

/// Quantize all f32 weight constants feeding MatMul / Conv2d nodes to the
/// requested bit-width (4 or 8).
///
/// The pass is idempotent — nodes that are already quantized are skipped.
///
/// # Per-channel quantization
///
/// For a weight of shape `[out_channels, in_features]`, we compute one
/// (scale, zero_point) pair per output channel (row).  The scales and
/// dequant_offsets are stored directly on the `IrDType::I4/I8Scaled/U4Scaled/U8Scaled`
/// variant so the CPU backend can feed them into `PackedTensor::from_raw(…)`.
///
/// When `signed` is true, packed storage uses signed `I4x8`/`I8x4` and
/// IrDType produces `I4`/`I8Scaled`.  When `signed` is false, packing uses
/// unsigned `U4x8`/`U8x4` and IrDType produces `U4Scaled`/`U8Scaled`.
pub fn quantize_weights(
    graph: &mut ComputeGraph,
    bit_width: u8,
    signed: bool,
    group_size: Option<usize>,
) -> Result<(), FastnnError> {
    // ---- Phase 1: collect (constant_id, consumer_id) pairs to quantize ----
    // We collect first, then mutate, to avoid borrow-checker issues.
    let mut to_quantize: Vec<(NodeId, NodeId)> = Vec::with_capacity(graph.nodes.len());

    let graph_ref = &*graph;
    crate::utils::traverse_graph(graph_ref, |node_id, node| {
        // Quantize weights consumed by MatMul/Conv ops AND optimizer ops.
        // For MatMul/Conv, the weight is input[1] (input[0] is activation).
        // For optimizer ops (SgdUpdate, AdamUpdate, etc.), the weight is input[0].
        let is_matmul_conv = matches!(
            node.opcode,
            Opcode::MatMul | Opcode::Conv1d | Opcode::Conv2d | Opcode::Conv3d
        );
        let is_optimizer = matches!(
            node.opcode,
            Opcode::SgdUpdate
                | Opcode::AdamUpdate
                | Opcode::AdamWUpdate
                | Opcode::MuonUpdate
                | Opcode::LionUpdate
                | Opcode::RmspropUpdate
        );
        if !is_matmul_conv && !is_optimizer {
            return Ok(());
        }

        let weight_idx = if is_optimizer { 0 } else { 1 };
        if let Some(&weight_id) = node.inputs.get(weight_idx) {
            let weight_node = match graph_ref.get_node(weight_id) {
                Some(n) => n,
                None => return Ok(()),
            };

            // Only quantize f32/bf16/f16 constants (skip already-quantized).
            if let Opcode::Constant(ref val) = weight_node.opcode {
                let is_float = weight_node.output_type.is_native_float();
                // Skip non-Data constants (scalar floats) and already-quantized.
                let is_data = matches!(val, TensorValue::Data { .. });
                if is_float && is_data {
                    to_quantize.push((weight_id, node_id));
                }
            }
        }
        Ok(())
    })
    .map_err(FastnnError::compilation)?;

    if to_quantize.is_empty() {
        return Ok(());
    }

    // ---- Phase 2: quantize each weight constant in-place ----
    for (const_id, _consumer_id) in to_quantize {
        // Clone needed data before mutable borrow.
        let const_node = match graph.get_node(const_id) {
            Some(n) => n,
            None => continue,
        };

        let (f32_data, orig_shape, _orig_numel) = match &const_node.opcode {
            Opcode::Constant(TensorValue::Data { bytes, tensor_type }) => {
                let numel = match tensor_type.numel() {
                    Some(n) => n as usize,
                    None => continue, // skip symbolic shapes
                };
                // Reinterpret bytes as f32.
                let f32_data: Vec<f32> = if bytes.len() == numel * 4 {
                    bytemuck::cast_slice(bytes).to_vec()
                } else {
                    // Byte length mismatch — skip.
                    continue;
                };
                let shape: Vec<usize> = tensor_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        DimExpr::Known(v) => Some(*v as usize),
                        _ => None,
                    })
                    .collect();
                (f32_data, shape, numel)
            }
            _ => continue,
        };

        if f32_data.is_empty() {
            continue;
        }

        // Skip 1D tensors — no benefit and the dequant path assumes shape[0]
        // is the number of rows, which breaks for 1D [N] tensors.
        if orig_shape.len() < 2 {
            continue;
        }

        // For 2D MatMul weights, transpose from [K, N] to [N, K] so the
        // packed GEMM kernel (gemm_packed_batched) sees [N_out, K_in].
        // For Conv weights ([N_out, C_in, KH, KW]), N_out is already first —
        // per-channel quantization along dim 0 is correct as-is.
        let (quant_data, quant_shape) = if orig_shape.len() == 2 {
            let rows = orig_shape[0];
            let cols = orig_shape[1];
            let mut transposed = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    transposed[c * rows + r] = f32_data[r * cols + c];
                }
            }
            (transposed, vec![cols, rows])
        } else {
            (f32_data, orig_shape.clone())
        };

        // Determine inner dimension for the weight tensor.
        let inner_dim = if quant_shape.len() >= 2 {
            quant_shape[1..].iter().product::<usize>()
        } else {
            quant_data.len()
        };

        if inner_dim == 0 {
            continue;
        }

        // Quantize using PackedTensor and extract raw bytes + metadata.
        let build_storage_layout = || crate::types::TensorStorageLayout {
            encoding: crate::types::StorageEncoding::Packed {
                word_bits: 32,
                lanes: if bit_width == 4 { 8 } else { 4 },
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
        };

        if bit_width == 4 {
            if signed {
                let pt = if let Some(gs) = group_size {
                    PackedTensor::<I4x8>::from_f32_per_channel_asymmetric_grouped(
                        &quant_data,
                        &quant_shape,
                        gs,
                    )
                } else {
                    PackedTensor::<I4x8>::from_f32_per_channel_asymmetric(&quant_data, &quant_shape)
                };
                let bytes = pt.as_bytes().to_vec();
                let granularity = if group_size.is_some() {
                    crate::types::QuantizationGranularity::PerGroup {
                        axis: 0,
                        group_size: group_size.unwrap_or(1),
                    }
                } else if pt.scales.len() > 1 {
                    crate::types::QuantizationGranularity::PerAxis { axis: 0 }
                } else {
                    crate::types::QuantizationGranularity::PerTensor
                };
                let representation =
                    crate::types::ValueRepresentation::packed_affine_dequantization(
                        crate::types::ScalarType::I4,
                        8,
                        granularity,
                        pt.scales.clone(),
                        pt.zeros.clone(),
                    )
                    .expect("valid affine representation");
                let tt = TensorType::from_parts(
                    orig_shape
                        .iter()
                        .map(|&d| DimExpr::Known(d as u64))
                        .collect(),
                    representation,
                    build_storage_layout(),
                );
                let new_value = TensorValue::Data {
                    bytes,
                    tensor_type: tt.clone(),
                };
                if let Some(node_mut) = graph.get_node_mut(const_id) {
                    node_mut.opcode = Opcode::Constant(new_value);
                    node_mut.output_type = tt;
                }
                continue;
            } else {
                let pt = if let Some(gs) = group_size {
                    PackedTensor::<U4x8>::from_f32_per_channel_asymmetric_grouped(
                        &quant_data,
                        &quant_shape,
                        gs,
                    )
                } else {
                    PackedTensor::<U4x8>::from_f32_per_channel_asymmetric(&quant_data, &quant_shape)
                };
                let bytes = pt.as_bytes().to_vec();
                let granularity = if group_size.is_some() {
                    crate::types::QuantizationGranularity::PerGroup {
                        axis: 0,
                        group_size: group_size.unwrap_or(1),
                    }
                } else if pt.scales.len() > 1 {
                    crate::types::QuantizationGranularity::PerAxis { axis: 0 }
                } else {
                    crate::types::QuantizationGranularity::PerTensor
                };
                let representation =
                    crate::types::ValueRepresentation::packed_affine_dequantization(
                        crate::types::ScalarType::U4,
                        8,
                        granularity,
                        pt.scales.clone(),
                        pt.zeros.clone(),
                    )
                    .expect("valid affine representation");
                let tt = TensorType::from_parts(
                    orig_shape
                        .iter()
                        .map(|&d| DimExpr::Known(d as u64))
                        .collect(),
                    representation,
                    build_storage_layout(),
                );
                let new_value = TensorValue::Data {
                    bytes,
                    tensor_type: tt.clone(),
                };
                if let Some(node_mut) = graph.get_node_mut(const_id) {
                    node_mut.opcode = Opcode::Constant(new_value);
                    node_mut.output_type = tt;
                }
                continue;
            }
        } else if signed {
            let pt = if let Some(gs) = group_size {
                PackedTensor::<I8x4>::from_f32_per_channel_asymmetric_grouped(
                    &quant_data,
                    &quant_shape,
                    gs,
                )
            } else {
                PackedTensor::<I8x4>::from_f32_per_channel_asymmetric(&quant_data, &quant_shape)
            };
            let bytes = pt.as_bytes().to_vec();
            let granularity = if group_size.is_some() {
                crate::types::QuantizationGranularity::PerGroup {
                    axis: 0,
                    group_size: group_size.unwrap_or(1),
                }
            } else if pt.scales.len() > 1 {
                crate::types::QuantizationGranularity::PerAxis { axis: 0 }
            } else {
                crate::types::QuantizationGranularity::PerTensor
            };
            let representation = crate::types::ValueRepresentation::packed_affine_dequantization(
                crate::types::ScalarType::I8,
                4,
                granularity,
                pt.scales.clone(),
                pt.zeros.clone(),
            )
            .expect("valid affine representation");
            let tt = TensorType::from_parts(
                orig_shape
                    .iter()
                    .map(|&d| DimExpr::Known(d as u64))
                    .collect(),
                representation,
                build_storage_layout(),
            );
            let new_value = TensorValue::Data {
                bytes,
                tensor_type: tt.clone(),
            };
            if let Some(node_mut) = graph.get_node_mut(const_id) {
                node_mut.opcode = Opcode::Constant(new_value);
                node_mut.output_type = tt;
            }
        } else {
            let pt = if let Some(gs) = group_size {
                PackedTensor::<U8x4>::from_f32_per_channel_asymmetric_grouped(
                    &quant_data,
                    &quant_shape,
                    gs,
                )
            } else {
                PackedTensor::<U8x4>::from_f32_per_channel_asymmetric(&quant_data, &quant_shape)
            };
            let bytes = pt.as_bytes().to_vec();
            let granularity = if group_size.is_some() {
                crate::types::QuantizationGranularity::PerGroup {
                    axis: 0,
                    group_size: group_size.unwrap_or(1),
                }
            } else if pt.scales.len() > 1 {
                crate::types::QuantizationGranularity::PerAxis { axis: 0 }
            } else {
                crate::types::QuantizationGranularity::PerTensor
            };
            let representation = crate::types::ValueRepresentation::packed_affine_dequantization(
                crate::types::ScalarType::U8,
                4,
                granularity,
                pt.scales.clone(),
                pt.zeros.clone(),
            )
            .expect("valid affine representation");
            let tt = TensorType::from_parts(
                orig_shape
                    .iter()
                    .map(|&d| DimExpr::Known(d as u64))
                    .collect(),
                representation,
                build_storage_layout(),
            );
            let new_value = TensorValue::Data {
                bytes,
                tensor_type: tt.clone(),
            };
            if let Some(node_mut) = graph.get_node_mut(const_id) {
                node_mut.opcode = Opcode::Constant(new_value);
                node_mut.output_type = tt;
            }
        };
    }

    Ok(())
}

/// Quantize f32 weight constants feeding MatMul / Conv2d to a floating-point
/// packed dtype (F8x4/E4M3, F8x4R/E5M2, or F4x8/E2M1).
///
/// Unlike integer quantization (`quantize_weights`), FP packed quantization
/// is symmetric (no zero points) and uses per-channel scales derived from
/// each channel's absolute max divided by the dtype's `MAX_REPRESENTABLE`.
pub fn quantize_weights_fp(
    graph: &mut ComputeGraph,
    fp_dtype: &QuantTarget,
    group_size: Option<usize>,
) -> Result<(), FastnnError> {
    if matches!(
        fp_dtype,
        QuantTarget::I8 | QuantTarget::U8 | QuantTarget::I4 | QuantTarget::U4
    ) {
        return Err(FastnnError::compilation(
            "floating/codebook quantization requires a floating or codebook target",
        ));
    }
    let mut to_quantize: Vec<(NodeId, NodeId)> = Vec::with_capacity(graph.nodes.len());

    let graph_ref = &*graph;
    crate::utils::traverse_graph(graph_ref, |node_id, node| {
        let is_matmul_conv = matches!(
            node.opcode,
            Opcode::MatMul | Opcode::Conv1d | Opcode::Conv2d | Opcode::Conv3d
        );
        if !is_matmul_conv {
            return Ok(());
        }
        if let Some(&weight_id) = node.inputs.get(1) {
            let weight_node = match graph_ref.get_node(weight_id) {
                Some(n) => n,
                None => return Ok(()),
            };
            if let Opcode::Constant(ref val) = weight_node.opcode {
                let is_float = weight_node.output_type.is_native_float();
                let is_data = matches!(val, TensorValue::Data { .. });
                if is_float && is_data {
                    to_quantize.push((weight_id, node_id));
                }
            }
        }
        Ok(())
    })
    .map_err(FastnnError::compilation)?;

    if to_quantize.is_empty() {
        return Ok(());
    }

    for (const_id, _consumer_id) in to_quantize {
        let const_node = match graph.get_node(const_id) {
            Some(n) => n,
            None => continue,
        };

        let (f32_data, orig_shape) = match &const_node.opcode {
            Opcode::Constant(TensorValue::Data { bytes, tensor_type }) => {
                let numel = match tensor_type.numel() {
                    Some(n) => n as usize,
                    None => continue,
                };
                let f32_data: Vec<f32> = if bytes.len() == numel * 4 {
                    bytemuck::cast_slice(bytes).to_vec()
                } else {
                    continue;
                };
                let shape: Vec<usize> = tensor_type
                    .shape
                    .iter()
                    .filter_map(|d| match d {
                        DimExpr::Known(v) => Some(*v as usize),
                        _ => None,
                    })
                    .collect();
                (f32_data, shape)
            }
            _ => continue,
        };

        if f32_data.is_empty() {
            continue;
        }

        // Skip 1D tensors — no benefit and the dequant path assumes shape[0]
        // is the number of rows, which breaks for 1D [N] tensors.
        if orig_shape.len() < 2 {
            continue;
        }

        let (quant_data, quant_shape) = if orig_shape.len() == 2 {
            let rows = orig_shape[0];
            let cols = orig_shape[1];
            let mut transposed = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    transposed[c * rows + r] = f32_data[r * cols + c];
                }
            }
            (transposed, vec![cols, rows])
        } else {
            (f32_data, orig_shape.clone())
        };

        let f4_block_size: usize = 32;
        let mut f4x8_use_per_block = false;

        let (packed_bytes, scales, zeros, codebooks) = match fp_dtype {
            QuantTarget::Fp8E4M3 => {
                let pt = match group_size {
                    Some(gs) => PackedTensor::<F8x4>::from_f32_per_channel_grouped(
                        &quant_data,
                        &quant_shape,
                        gs,
                    ),
                    None => PackedTensor::<F8x4>::from_f32_per_channel(&quant_data, &quant_shape),
                };
                (pt.as_bytes().to_vec(), pt.scales, vec![], vec![])
            }
            QuantTarget::Fp8E5M2 => {
                let pt = match group_size {
                    Some(gs) => PackedTensor::<F8x4R>::from_f32_per_channel_grouped(
                        &quant_data,
                        &quant_shape,
                        gs,
                    ),
                    None => PackedTensor::<F8x4R>::from_f32_per_channel(&quant_data, &quant_shape),
                };
                (pt.as_bytes().to_vec(), pt.scales, vec![], vec![])
            }
            QuantTarget::Fp4E2M1 => {
                let _inner = if quant_shape.len() >= 2 {
                    quant_shape[1..].iter().product::<usize>()
                } else {
                    quant_data.len()
                };
                // Per-block F4 quantization disabled because the packed-F4 GEMM kernel
                // (gemm_batch_packed_simd) applies a single (scale, zero_point) per row
                // during the affine post-processing step.  Per-block layout has different
                // scale/zero for each block within a row, so the affine formula
                // (dot * scale + zero * sum(act)) does not factor correctly —
                // the kernel would apply the wrong scale to most elements.
                // Keep per-channel or grouped per-channel only.
                let use_per_block = false;
                f4x8_use_per_block = false;
                let pt = if use_per_block {
                    PackedTensor::<F4x8>::from_f32_per_block_asymmetric(
                        &quant_data,
                        &quant_shape,
                        f4_block_size,
                    )
                } else if let Some(gs) = group_size {
                    PackedTensor::<F4x8>::from_f32_per_channel_asymmetric_grouped(
                        &quant_data,
                        &quant_shape,
                        gs,
                    )
                } else {
                    PackedTensor::<F4x8>::from_f32_per_channel_asymmetric(&quant_data, &quant_shape)
                };
                // Always pass zeros for F4: both per-channel asymmetric and
                // per-block asymmetric packing use non-zero zero points.
                (pt.as_bytes().to_vec(), pt.scales, pt.zeros.clone(), vec![])
            }
            QuantTarget::I4Codebook => {
                let pt = PackedTensor::<I4x8>::from_f32_per_block_codebook(
                    &quant_data,
                    &quant_shape,
                    f4_block_size,
                );
                (
                    pt.as_bytes().to_vec(),
                    pt.scales,
                    vec![],
                    pt.codebooks.clone(),
                )
            }
            _ => unreachable!("integer targets were rejected before graph traversal"),
        };

        let shape_dims: Vec<DimExpr> = orig_shape
            .iter()
            .map(|&d| DimExpr::Known(d as u64))
            .collect();

        let new_tensor_type = match fp_dtype {
            QuantTarget::Fp8E4M3 | QuantTarget::Fp8E5M2 => {
                let lanes = match fp_dtype {
                    QuantTarget::Fp8E4M3 => 4u8,
                    _ => 4u8,
                };
                let storage = match fp_dtype {
                    QuantTarget::Fp8E4M3 => crate::types::ScalarType::Fp8E4M3,
                    _ => crate::types::ScalarType::Fp8E5M2,
                };
                let representation =
                    crate::types::ValueRepresentation::packed_fp_scaled(storage, lanes, scales)
                        .expect("valid fp scaled representation");
                let layout = crate::types::TensorStorageLayout {
                    encoding: crate::types::StorageEncoding::Packed {
                        word_bits: 32,
                        lanes,
                    },
                    row_packed: true,
                    prefix_bytes: 0,
                    suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
                };
                TensorType::from_parts(shape_dims, representation, layout)
            }
            QuantTarget::Fp4E2M1 => {
                let granularity = if group_size.is_some() {
                    crate::types::QuantizationGranularity::PerGroup {
                        axis: 0,
                        group_size: group_size.unwrap_or(1),
                    }
                } else if scales.len() > 1 {
                    crate::types::QuantizationGranularity::PerAxis { axis: 0 }
                } else {
                    crate::types::QuantizationGranularity::PerTensor
                };
                let representation = crate::types::ValueRepresentation::packed_fp_scaled_affine(
                    crate::types::ScalarType::Fp4E2M1,
                    8,
                    granularity,
                    scales,
                    zeros,
                )
                .expect("valid fp scaled affine representation");
                let layout = crate::types::TensorStorageLayout {
                    encoding: crate::types::StorageEncoding::Packed {
                        word_bits: 32,
                        lanes: 8,
                    },
                    row_packed: true,
                    prefix_bytes: 0,
                    suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
                };
                TensorType::from_parts(shape_dims, representation, layout)
            }
            QuantTarget::I4Codebook => {
                let granularity = crate::types::QuantizationGranularity::PerGroup {
                    axis: 0,
                    group_size: f4_block_size,
                };
                let representation =
                    crate::types::ValueRepresentation::packed_codebook_dequantization(
                        crate::types::ScalarType::I4,
                        8,
                        granularity,
                        codebooks.iter().map(|cb| cb.to_vec()).collect(),
                        scales,
                        zeros,
                    )
                    .expect("valid codebook representation");
                let layout = crate::types::TensorStorageLayout {
                    encoding: crate::types::StorageEncoding::Packed {
                        word_bits: 32,
                        lanes: 8,
                    },
                    row_packed: true,
                    prefix_bytes: 0,
                    suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
                };
                TensorType::from_parts(shape_dims, representation, layout)
            }
            _ => unreachable!("integer targets were rejected before graph traversal"),
        };

        let new_value = TensorValue::Data {
            bytes: packed_bytes,
            tensor_type: new_tensor_type.clone(),
        };

        if let Some(node_mut) = graph.get_node_mut(const_id) {
            node_mut.opcode = Opcode::Constant(new_value);
            node_mut.output_type = new_tensor_type;
            // Store the block size so the backend can reconstruct PackedTensor correctly.
            // This is critical when inner dim is not divisible by the block size.
            // Only set for I4Codebook (always per-block) and F4x8 when per-block was
            // actually used.  Per-channel F4x8 must NOT carry qbs — it makes the backend
            // attempt per-block dequant on per-channel scales, causing scale_for_elem OOB.
            if matches!(fp_dtype, QuantTarget::I4Codebook) || f4x8_use_per_block {
                node_mut
                    .attrs
                    .insert("quant_block_size".to_string(), f4_block_size.to_string());
            }
        }
    }

    Ok(())
}

/// Wrap optimizer ops (SgdUpdate, AdamUpdate, AdamWUpdate) with Dequantize/Quantize
/// when their weight input has a packed quantized dtype (U4/U8).
///
/// This handles the case where weight dtypes are changed to quantized AFTER
/// the optimizer nodes have been created (e.g., by `quantize_weights` modifying
/// Constants in-place, or by a future pass that quantizes Input weights).
///
/// For each optimizer op found with a quantized weight input:
///   weight_u4 → Dequantize → weight_f32
///   weight_f32 + grad → SgdUpdate → updated_f32
///   updated_f32 → Quantize → updated_u4
///
/// The pass also updates `graph.outputs` so that any output pointing at the
/// optimizer node is redirected to the Quantize node.
pub fn wrap_quantized_optimizer(graph: &mut ComputeGraph) -> Result<(), FastnnError> {
    // Collect optimizer nodes that need wrapping, along with their
    // weight input ID and the bit_width to requantize to.
    #[derive(Clone)]
    struct OptimizerWrap {
        opt_id: NodeId,
        weight_id: NodeId,
        bit_width: usize,
        opt_inputs: Vec<NodeId>, // remaining inputs (grad, m, v, etc.)
    }

    let mut to_wrap: Vec<OptimizerWrap> = Vec::with_capacity(graph.nodes.len());

    let graph_ref = &*graph;
    crate::utils::traverse_graph(graph_ref, |node_id, node| {
        let is_optimizer = matches!(
            node.opcode,
            Opcode::SgdUpdate
                | Opcode::AdamUpdate
                | Opcode::AdamWUpdate
                | Opcode::MuonUpdate
                | Opcode::LionUpdate
                | Opcode::RmspropUpdate
        );
        if !is_optimizer {
            return Ok(());
        }

        // input[0] is the weight — check its dtype
        let weight_id = match node.inputs.first() {
            Some(&id) => id,
            None => return Ok(()),
        };

        let weight_node = match graph_ref.get_node(weight_id) {
            Some(n) => n,
            None => return Ok(()),
        };

        let bit_width = match optimizer_affine_metadata(&weight_node.output_type) {
            Some((bit_width, _, _)) => bit_width,
            None => return Ok(()), // weight is not supported affine storage, skip
        };

        // Save the remaining inputs (grad, m, v) and attrs
        let remaining_inputs: Vec<NodeId> = node.inputs.iter().skip(1).copied().collect();

        to_wrap.push(OptimizerWrap {
            opt_id: node_id,
            weight_id,
            bit_width,
            opt_inputs: remaining_inputs,
        });
        Ok(())
    })
    .map_err(FastnnError::compilation)?;

    if to_wrap.is_empty() {
        return Ok(());
    }

    // Process each optimizer node: insert Dequantize before and Quantize after
    for wrap in &to_wrap {
        let weight_type = graph
            .get_node(wrap.weight_id)
            .map(|n| n.output_type.clone())
            .unwrap_or_else(|| TensorType::new(vec![], IrDType::F32));

        // 1. Create Dequantize node: weight_u4 → f32
        let f32_type = TensorType::new(weight_type.shape.clone(), IrDType::F32);
        let deq_id = graph.add_node(Opcode::Dequantize, vec![wrap.weight_id], f32_type.clone());

        // 2. Update the optimizer's weight input to the Dequantize output.
        // The optimizer now takes: [deq_id, grad, m, v, ...]
        let new_opt_inputs: Vec<NodeId> = std::iter::once(deq_id)
            .chain(wrap.opt_inputs.iter().copied())
            .collect();

        // Replace the optimizer node: we need to change its inputs.
        // Since we can't modify node.inputs directly through get_node_mut (borrow
        // issues with later graph.add_node calls), we record the change and apply
        // via get_node_mut directly.
        if let Some(opt_node) = graph.get_node_mut(wrap.opt_id) {
            opt_node.inputs = new_opt_inputs.clone();
        }

        // 3. Create Quantize node after the optimizer: updated_f32 → updated_u4
        // Carry through the original calibrated scales/zeros so the runtime
        // can skip the O(N×K) per-channel abs-max scan on every optimizer step.
        let (_orig_scales, _orig_zeros) = graph
            .get_node(wrap.weight_id)
            .map(|wn| match optimizer_affine_metadata(&wn.output_type) {
                Some((_, scales, offsets)) => (scales, offsets),
                None => (vec![], vec![]),
            })
            .unwrap_or_default();

        let u_type = TensorType::new(
            weight_type.shape.clone(),
            match wrap.bit_width {
                4 => IrDType::I4,
                8 => IrDType::I8Scaled,
                _ => {
                    return Err(FastnnError::compilation(format!(
                        "unsupported bit_width: {}",
                        wrap.bit_width
                    )))
                }
            },
        );
        let mut q_attrs = std::collections::HashMap::new();
        q_attrs.insert("bit_width".to_string(), wrap.bit_width.to_string());
        let q_id = graph.add_node_with_attrs(Opcode::Quantize, vec![wrap.opt_id], u_type, q_attrs);

        // 4. Redirect consumers of the optimizer output to the Quantize output.
        // Collect consumers first to avoid borrow conflicts.
        let consumers: Vec<NodeId> = graph.consumers(wrap.opt_id);
        for &consumer_id in &consumers {
            // Don't redirect the Quantize node itself!
            if consumer_id == q_id {
                continue;
            }
            if let Some(consumer) = graph.get_node_mut(consumer_id) {
                for input in consumer.inputs.iter_mut() {
                    if *input == wrap.opt_id {
                        *input = q_id;
                    }
                }
            }
        }

        // 5. Also redirect graph outputs
        for output in graph.outputs.iter_mut() {
            if *output == wrap.opt_id {
                *output = q_id;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::backend::executor::GraphExecutor;
    use crate::ir::builder::GraphBuilder;
    use crate::ir::{ComputeGraph, DimExpr, IrDType, Opcode, TensorValue};

    #[test]
    fn floating_quantization_rejects_integer_policy_targets() {
        let mut graph = ComputeGraph::new();
        let error = quantize_weights_fp(&mut graph, &QuantTarget::I4, None).unwrap_err();
        assert!(error.to_string().contains("floating or codebook target"));
    }

    #[test]
    fn optimizer_metadata_uses_canonical_affine_storage() {
        // Construct TensorType with explicit AffineDequantization metadata
        let rep = crate::types::ValueRepresentation::packed_affine_dequantization(
            crate::types::ScalarType::I4,
            8,
            crate::types::QuantizationGranularity::PerTensor,
            vec![0.5],
            vec![-1.0],
        )
        .unwrap();
        let layout = crate::types::TensorStorageLayout {
            encoding: crate::types::StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
        };
        let tensor_type = TensorType::from_parts(vec![DimExpr::Known(8)], rep, layout);
        assert_eq!(
            optimizer_affine_metadata(&tensor_type),
            Some((4, vec![0.5], vec![-1.0]))
        );

        // I4 with RuntimeAffineQuantization (no static metadata) returns None
        let codebook = TensorType::new(vec![DimExpr::Known(8)], IrDType::I4);
        assert_eq!(optimizer_affine_metadata(&codebook), None);
    }

    /// Helper: create a ComputeGraph with a MatMul and f32 Constant weight.
    fn build_matmul_graph(weight_data: &[f32], weight_shape: &[usize]) -> ComputeGraph {
        let gb = GraphBuilder::new();
        let _m = weight_shape[0];
        let k = weight_shape[1];
        let input =
            gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(k as u64)], IrDType::F32);
        let weight_tt = TensorType::new(
            weight_shape
                .iter()
                .map(|&d| DimExpr::Known(d as u64))
                .collect(),
            IrDType::F32,
        );
        let weight_bytes: Vec<u8> = bytemuck::cast_slice(weight_data).to_vec();
        let weight = gb.constant(&weight_bytes, weight_tt);
        let _output = gb.matmul(&input, &weight);
        gb.to_graph()
    }

    #[test]
    fn test_quantize_f32_weight_to_u4() {
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
        ];
        let mut graph = build_matmul_graph(&weight_data, &[2, 8]);

        // Find the f32 constant node before quantization.
        let has_f32_const = graph.nodes.iter().any(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if tensor_type.dtype() == IrDType::F32)
        });
        assert!(
            has_f32_const,
            "Should have an f32 constant node before quantization"
        );

        // Quantize to U4.
        quantize_weights(&mut graph, 4, true, None).expect("quantization should succeed");

        // Verify the weight node dtype is now U4.
        let has_u4_const = graph.nodes.iter().any(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if tensor_type.dtype() == IrDType::I4)
        });
        assert!(
            has_u4_const,
            "Weight should be quantized to U4 after quantization"
        );

        // Verify per-channel scales.
        // The [2,8] weight is transposed to [8,2] for the packed GEMM convention,
        // resulting in 8 per-channel scales (one per output channel = N dimension).
        let u4_node = graph.nodes.iter().find(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if tensor_type.dtype() == IrDType::I4)
        }).unwrap();

        let (scales, dequant_offsets) = u4_node
            .output_type
            .affine_dequantization()
            .expect("quantized type should expose affine dequantization");
        assert_eq!(
            scales.len(),
            8,
            "Should have 8 per-channel scales for a [2,8] weight (transposed to [8,2])"
        );
        assert_eq!(
            dequant_offsets.len(),
            8,
            "Should have 8 per-channel dequant_offsets"
        );
        for &s in scales.iter() {
            assert!(s > 0.0, "Scale should be positive, got {}", s);
        }
    }

    #[test]
    fn test_quantize_f32_weight_to_u8() {
        let weight_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0,
        ];
        let mut graph = build_matmul_graph(&weight_data, &[2, 8]);

        quantize_weights(&mut graph, 8, true, None).expect("quantization should succeed");

        let has_u8_const = graph.nodes.iter().any(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if tensor_type.dtype() == IrDType::I8Scaled)
        });
        assert!(has_u8_const, "Weight should be quantized to U8");

        let u8_node = graph.nodes.iter().find(|n| {
            matches!(&n.opcode, Opcode::Constant(TensorValue::Data { tensor_type, .. }) if tensor_type.dtype() == IrDType::I8Scaled)
        }).unwrap();

        let (scales, dequant_offsets) = u8_node
            .output_type
            .affine_dequantization()
            .expect("quantized type should expose affine dequantization");
        assert_eq!(
            scales.len(),
            8,
            "Should have 8 per-channel scales for a [2,8] weight (transposed to [8,2])"
        );
        assert_eq!(dequant_offsets.len(), 8);
    }

    #[test]
    fn test_quantize_skips_already_quantized() {
        // Build a graph with a U4 constant — it should be skipped.
        let gb = GraphBuilder::new();
        let input = gb.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
        let weight_data: Vec<u8> = vec![0u8; 4]; // dummy packed data
        let weight_tt = TensorType::new(vec![DimExpr::Known(2), DimExpr::Known(4)], IrDType::I4);
        let weight = gb.constant(&weight_data, weight_tt);
        let _output = gb.matmul(&input, &weight);
        let mut graph = gb.to_graph();

        // Quantize should be a no-op (already I4).
        quantize_weights(&mut graph, 4, true, None).expect("quantization should succeed");

        // The weight should still be I4 (unchanged).
        let node = graph
            .nodes
            .iter()
            .find(|n| matches!(&n.opcode, Opcode::Constant(TensorValue::Data { .. })))
            .unwrap();
        assert!(
            matches!(node.output_type.dtype(), IrDType::I4),
            "Should still be U4"
        );
    }

    #[test]
    fn test_quantize_no_weights_is_noop() {
        let gb = GraphBuilder::new();
        let input = gb.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(4)], IrDType::F32);
        let _output = gb.relu(&input);
        let mut graph = gb.to_graph();

        quantize_weights(&mut graph, 4, true, None)
            .expect("quantization should succeed on empty graph");
    }

    #[test]
    fn test_quantize_end_to_end_with_compile() {
        // Build a matmul graph, quantize it, and verify the compiled plan
        // includes a matmul_u4 kernel (not matmul).
        let weight_data: Vec<f32> = (0..32).map(|i| i as f32).collect(); // [8, 4] = 32 elements
        let gb = GraphBuilder::new();
        let input = gb.input_with_dims(&[DimExpr::Known(2), DimExpr::Known(8)], IrDType::F32);
        let weight_tt = TensorType::new(vec![DimExpr::Known(8), DimExpr::Known(4)], IrDType::F32);
        let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
        let weight = gb.constant(&weight_bytes, weight_tt);
        let _output = gb.matmul(&input, &weight);

        let executor = GraphExecutor::new(CpuBackend);
        let (plan, _, compiled_graph) = executor
            .compile_with_plan_and_quantize(gb.to_graph(), Some(4), None)
            .expect("compile with quantization should succeed");

        // Verify that the compiled plan contains a matmul_u4 kernel.
        let has_u4_kernel = plan.instructions.iter().any(|i| match i {
            crate::backend::Instruction::CallKernel { kernel_name, .. } => {
                kernel_name == "matmul_i4"
            }
            _ => false,
        });
        assert!(
            has_u4_kernel,
            "Compiled plan should contain a matmul_u4 kernel after quantization"
        );

        // Verify that the weight node in the compiled graph has U4 dtype.
        let has_u4_weight = compiled_graph
            .nodes
            .iter()
            .any(|n| matches!(n.output_type.dtype(), IrDType::I4));
        assert!(
            has_u4_weight,
            "Compiled graph should contain a U4 weight node"
        );
    }

    // ── Phase 3a: Dequantize-then-update optimizer tests ─────────────

    /// Helper: create a graph with a Constant weight that feeds both MatMul
    /// and an SgdUpdate optimizer op. Returns (graph, input_id, d_w_id, weight_id).
    ///
    /// MatMul: [1, K] @ [K, N] = [1, N], where K = w_shape[0], N = w_shape[1].
    fn build_training_graph_with_constant_weight(
        weight_data: &[f32],
        w_shape: &[usize],
    ) -> (GraphBuilder, NodeId, NodeId, NodeId) {
        let gb = GraphBuilder::new();
        // K = w_shape[0] (inner dimension for MatMul)
        let k = w_shape[0];
        let input =
            gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(k as u64)], IrDType::F32);
        let weight_tt = TensorType::new(
            w_shape.iter().map(|&d| DimExpr::Known(d as u64)).collect(),
            IrDType::F32,
        );
        let weight_bytes: Vec<u8> = bytemuck::cast_slice(weight_data).to_vec();
        let weight = gb.constant(&weight_bytes, weight_tt);
        // weight_id = the constant's node_id
        let weight_id = weight.node_id;
        let _mm = gb.matmul(&input, &weight);
        // Gradient for the weight
        let d_w = gb.input_with_dims(
            &w_shape
                .iter()
                .map(|&d| DimExpr::Known(d as u64))
                .collect::<Vec<_>>(),
            IrDType::F32,
        );
        let d_w_id = d_w.node_id;
        // Optimizer step: weight -= 0.01 * (grad + 0.0 * weight)
        let _updated = gb.apply_sgd(&weight, &d_w, 0.01, 0.0);
        let input_id = input.node_id;
        (gb, input_id, d_w_id, weight_id)
    }

    #[test]
    fn test_wrap_quantized_optimizer_sgd() {
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w_shape: Vec<usize> = vec![4, 2];
        let (gb, input_id, d_w_id, weight_id) =
            build_training_graph_with_constant_weight(&weight_data, &w_shape);

        // Clone the graph and run the full pipeline with quantization
        let mut graph = gb.to_graph();
        graph.inputs = vec![input_id, d_w_id];
        // Set outputs to include the SgdUpdate output (which is the last node)
        let sgd_id = graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::SgdUpdate))
            .map(|n| n.id)
            .expect("Should have an SgdUpdate node");
        graph.outputs = vec![sgd_id];

        let executor = GraphExecutor::new(CpuBackend);
        let (_plan, _memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(graph, Some(4), None)
            .expect("compile with quantization should succeed");

        // After compile:
        // 1. Weight constant should be U4
        let u4_weight = compiled_graph
            .nodes
            .iter()
            .find(|n| matches!(n.output_type.dtype(), IrDType::I4));
        assert!(u4_weight.is_some(), "Weight should be quantized to U4");

        // 2. Dequantize node should exist (inserted by wrap_quantized_optimizer)
        let deq_node = compiled_graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::Dequantize));
        assert!(
            deq_node.is_some(),
            "Should have a Dequantize node wrapping the optimizer weight input"
        );

        // 3. Quantize node should exist (inserted after optimizer)
        let q_node = compiled_graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::Quantize));
        assert!(
            q_node.is_some(),
            "Should have a Quantize node after the optimizer"
        );

        // 4. The Dequantize should consume the weight constant
        if let Some(deq) = deq_node {
            assert_eq!(
                deq.inputs[0], weight_id,
                "Dequantize should take weight constant as input"
            );
        }

        // 5. The Quantize should consume the SgdUpdate
        if let Some(q) = q_node {
            assert_eq!(
                q.inputs[0], sgd_id,
                "Quantize should take SgdUpdate as input"
            );
        }

        // 6. Graph output should be redirected to Quantize
        assert_eq!(
            compiled_graph.outputs[0],
            q_node.unwrap().id,
            "Graph output should point to Quantize node (not SgdUpdate)"
        );
    }

    #[test]
    fn test_wrap_quantized_optimizer_adam() {
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w_shape: Vec<usize> = vec![4, 2];

        let gb = GraphBuilder::new();
        // K = w_shape[0] = 4, N = w_shape[1] = 2
        let k = w_shape[0]; // inner dim for MatMul
        let input =
            gb.input_with_dims(&[DimExpr::Known(1), DimExpr::Known(k as u64)], IrDType::F32);
        let weight_tt = TensorType::new(
            w_shape.iter().map(|&d| DimExpr::Known(d as u64)).collect(),
            IrDType::F32,
        );
        let weight_bytes: Vec<u8> = bytemuck::cast_slice(&weight_data).to_vec();
        let weight = gb.constant(&weight_bytes, weight_tt);
        let _weight_id = weight.node_id;

        // MatMul consumes the weight → quantize_weights will target this
        let _mm = gb.matmul(&input, &weight);

        // Gradient and state tensors
        let d_w = gb.input_with_dims(
            &w_shape
                .iter()
                .map(|&d| DimExpr::Known(d as u64))
                .collect::<Vec<_>>(),
            IrDType::F32,
        );
        let d_w_id = d_w.node_id;
        let m = gb.input_with_dims(
            &w_shape
                .iter()
                .map(|&d| DimExpr::Known(d as u64))
                .collect::<Vec<_>>(),
            IrDType::F32,
        );
        let m_id = m.node_id;
        let v = gb.input_with_dims(
            &w_shape
                .iter()
                .map(|&d| DimExpr::Known(d as u64))
                .collect::<Vec<_>>(),
            IrDType::F32,
        );
        let v_id = v.node_id;

        let _updated = gb.apply_adam(&weight, &d_w, &m, &v, 0.01, 0.9, 0.999, 1e-8, 1);

        let mut graph = gb.to_graph();
        // All inputs: input, d_w, m, v (weight is a Constant, not an input)
        graph.inputs = vec![input.node_id, d_w_id, m_id, v_id];
        let adam_id = graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::AdamUpdate))
            .map(|n| n.id)
            .expect("Should have an AdamUpdate node");
        graph.outputs = vec![adam_id];

        let executor = GraphExecutor::new(CpuBackend);
        let (_plan, _memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(graph, Some(4), None)
            .expect("compile with quantization should succeed");

        // Verify wrapping for Adam
        let has_u4 = compiled_graph
            .nodes
            .iter()
            .any(|n| matches!(n.output_type.dtype(), IrDType::I4));
        assert!(has_u4, "Weight should be quantized to U4");

        let has_deq = compiled_graph
            .nodes
            .iter()
            .any(|n| matches!(n.opcode, Opcode::Dequantize));
        assert!(has_deq, "Should have Dequantize wrapping Adam weight input");

        let has_q = compiled_graph
            .nodes
            .iter()
            .any(|n| matches!(n.opcode, Opcode::Quantize));
        assert!(has_q, "Should have Quantize after Adam");

        // Verify output is redirected to Quantize
        let q_id = compiled_graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::Quantize))
            .map(|n| n.id)
            .unwrap();
        assert_eq!(
            compiled_graph.outputs[0], q_id,
            "Output should be Quantize node"
        );
    }

    #[test]
    fn test_quantized_training_sgd_end_to_end() {
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let w_shape: Vec<usize> = vec![4, 2];
        let (gb, input_id, d_w_id, _weight_id) =
            build_training_graph_with_constant_weight(&weight_data, &w_shape);

        // x = [[1.0, 1.0, 1.0, 1.0]] (1x4) for [1,4] @ [4,2] → [1,2]
        let x_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let x_bytes: Vec<u8> = bytemuck::cast_slice(&x_data).to_vec();
        let d_w_data: Vec<f32> = vec![0.1; 8]; // [4, 2]
        let d_w_bytes: Vec<u8> = bytemuck::cast_slice(&d_w_data).to_vec();

        // Build graph and compile+execute with quantization
        let mut graph = gb.to_graph();
        graph.inputs = vec![input_id, d_w_id];
        let sgd_id = graph
            .nodes
            .iter()
            .find(|n| matches!(n.opcode, Opcode::SgdUpdate))
            .map(|n| n.id)
            .expect("Should have SgdUpdate");
        graph.outputs = vec![sgd_id];

        let mut executor = GraphExecutor::new(CpuBackend);
        let (mut plan, memory_plan, compiled_graph) = executor
            .compile_with_plan_and_quantize(graph, Some(4), None)
            .expect("compile with quantization should succeed");
        let results = executor
            .execute(
                &compiled_graph,
                &mut plan,
                &memory_plan,
                &[&x_bytes, &d_w_bytes],
            )
            .expect("quantized training execution should succeed");

        // Should have one output (the re-quantized updated weight)
        assert_eq!(results.len(), 1, "Should have one output (updated weight)");
        assert!(results[0].len() >= 40,
            "Updated weight should be at least 40 bytes (header: 8 + 8*4 channels). Got {} bytes: {:?}",
            results[0].len(), &results[0][..results[0].len().min(16)]);

        // Execute with zero gradient — the updated weight should equal the original
        // (within quantization precision), since update = 0 * lr = 0.
        let d_w_zero: Vec<f32> = vec![0.0; 8];
        let d_w_zero_bytes: Vec<u8> = bytemuck::cast_slice(&d_w_zero).to_vec();
        let results_zero = executor
            .execute(
                &compiled_graph,
                &mut plan,
                &memory_plan,
                &[&x_bytes, &d_w_zero_bytes],
            )
            .expect("execution with zero grad should succeed");

        // Execute with a large gradient — should differ from zero-grad result
        let d_w_large: Vec<f32> = vec![100.0; 8];
        let d_w_large_bytes: Vec<u8> = bytemuck::cast_slice(&d_w_large).to_vec();
        let results_large = executor
            .execute(
                &compiled_graph,
                &mut plan,
                &memory_plan,
                &[&x_bytes, &d_w_large_bytes],
            )
            .expect("execution with large grad should succeed");

        // Zero gradient should produce different output from large gradient
        assert_ne!(
            results_zero[0], results_large[0],
            "Zero vs large gradient should produce different weight updates"
        );
    }

    /// Test that wrap_quantized_optimizer is a no-op when there are no optimizer ops.
    #[test]
    fn test_wrap_quantized_optimizer_noop() {
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut graph = build_matmul_graph(&weight_data, &[2, 8]);
        let node_count_before = graph.nodes.len();
        wrap_quantized_optimizer(&mut graph)
            .expect("wrap should succeed on graph without optimizer");
        assert_eq!(
            graph.nodes.len(),
            node_count_before,
            "No nodes should be added when there are no optimizer ops"
        );
    }

    /// Test that the builder-level wrapping works for U4 weight parameters.
    #[test]
    fn test_builder_apply_sgd_quantized_weight() {
        let gb = GraphBuilder::new();
        // Create weight with explicit AffineDequantization metadata
        let rep = crate::types::ValueRepresentation::packed_affine_dequantization(
            crate::types::ScalarType::I4,
            8,
            crate::types::QuantizationGranularity::PerTensor,
            vec![0.5],
            vec![-1.0],
        )
        .unwrap();
        let layout = crate::types::TensorStorageLayout {
            encoding: crate::types::StorageEncoding::Packed {
                word_bits: 32,
                lanes: 8,
            },
            row_packed: true,
            prefix_bytes: 0,
            suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
        };
        let weight_tt =
            TensorType::from_parts(vec![DimExpr::Known(4), DimExpr::Known(2)], rep, layout);
        // Dummy packed data for 4x2 I4 weight packed into 4 bytes
        let w = gb.constant(&[0u8; 4], weight_tt);
        // Gradient is F32
        let d_w = gb.parameter(&[4, 2], IrDType::F32);
        // This should auto-wrap with Dequantize/Quantize
        let updated = gb.apply_sgd(&w, &d_w, 0.01, 0.0);

        // The updated tensor should be Quantize(SgdUpdate(Dequantize(W), d_w))
        let graph = gb.to_graph();
        let has_deq = graph
            .nodes
            .iter()
            .any(|n| matches!(n.opcode, Opcode::Dequantize));
        assert!(
            has_deq,
            "Builder should insert Dequantize for quantized weight SGD"
        );

        let has_q = graph
            .nodes
            .iter()
            .any(|n| matches!(n.opcode, Opcode::Quantize));
        assert!(has_q, "Builder should insert Quantize after SGD");

        // The output dtype should be I4
        assert!(
            matches!(updated.dtype(), IrDType::I4),
            "Updated weight should have I4 dtype, got {:?}",
            updated.dtype()
        );
    }
}
