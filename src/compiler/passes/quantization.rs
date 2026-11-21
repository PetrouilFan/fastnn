//! Quantization compiler pass.
//!
//! Finds f32 Constant weight nodes feeding MatMul/Conv2d ops and replaces
//! them with packed I4/I8 data carrying per-channel scale/zero-point metadata.
//! The backend dispatch already selects `matmul_u4`/`matmul_u8` kernels when
//! it sees `IrDType::I4/U8` on the weight input.

use crate::dtypes::{F4x8, F8x4, F8x4R, I4x8, I8x4, U4x8, U8x4};
use crate::error::FastnnError;
use crate::ir::{ComputeGraph, DimExpr, NodeId, Opcode, TensorType, TensorValue};
use crate::packed_tensor::PackedTensor;
use crate::types::{QuantTarget, ScalarType};

/// Prepare constant MatMul weights as signed I4 groups along the reduction (K) axis.
pub fn quantize_matmul_weights_k_grouped_i4(
    graph: &mut ComputeGraph,
    group_size: usize,
    exclude_patterns: &[String],
    clip_ratios: &std::collections::BTreeMap<String, Vec<f32>>,
) -> Result<(), FastnnError> {
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(FastnnError::compilation(format!(
            "dynamic W4A8 requires K-group size 32, 64, or 128, got {group_size}"
        )));
    }
    let mut candidates: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.opcode, Opcode::MatMul))
        .filter(|node| {
            !exclude_patterns.iter().any(|pattern| {
                node.name.contains(pattern)
                    || node.inputs.get(1).is_some_and(|weight_id| {
                        graph
                            .get_node(*weight_id)
                            .is_some_and(|weight| weight.name.contains(pattern))
                    })
                    || graph.consumers(node.id).into_iter().any(|consumer_id| {
                        graph
                            .get_node(consumer_id)
                            .is_some_and(|consumer| consumer.name.contains(pattern))
                    })
            })
        })
        .filter_map(|node| node.inputs.get(1).copied())
        .filter(|weight_id| {
            graph.get_node(*weight_id).is_some_and(|weight| {
                weight.output_type.is_native_float()
                    && matches!(weight.opcode, Opcode::Constant(TensorValue::Data { .. }))
            })
        })
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    let mut matched_calibration_names = std::collections::BTreeSet::new();

    for weight_id in candidates {
        let incompatible_consumer = graph.consumers(weight_id).into_iter().find(|consumer_id| {
            graph.get_node(*consumer_id).is_none_or(|consumer| {
                !matches!(consumer.opcode, Opcode::MatMul)
                    || consumer.inputs.get(1).copied() != Some(weight_id)
            })
        });
        if graph.outputs.contains(&weight_id) || incompatible_consumer.is_some() {
            return Err(FastnnError::compilation(format!(
                "dynamic W4A8 cannot mutate shared weight constant {weight_id}; clone the weight for non-MatMul consumers"
            )));
        }
        let mut provenance_names = Vec::new();
        if let Some(weight) = graph.get_node(weight_id) {
            if !weight.name.is_empty() {
                provenance_names.push(weight.name.clone());
            }
        }
        for matmul_id in graph.consumers(weight_id) {
            if let Some(matmul) = graph.get_node(matmul_id) {
                if !matmul.name.is_empty() {
                    provenance_names.push(matmul.name.clone());
                }
            }
            for consumer_id in graph.consumers(matmul_id) {
                if let Some(consumer) = graph.get_node(consumer_id) {
                    if !consumer.name.is_empty() {
                        provenance_names.push(consumer.name.clone());
                    }
                }
            }
        }
        let matched_calibrations: Vec<_> = provenance_names
            .iter()
            .filter_map(|name| clip_ratios.get(name).map(|ratios| (name, ratios)))
            .collect();
        if matched_calibrations.len() > 1 {
            return Err(FastnnError::compilation(format!(
                "dynamic W4A8 weight {weight_id} matches multiple calibration entries: {:?}",
                matched_calibrations
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>()
            )));
        }
        let calibrated_ratios = matched_calibrations.first().map(|(_, ratios)| *ratios);
        if let Some((name, _)) = matched_calibrations.first() {
            matched_calibration_names.insert((*name).clone());
        }
        let weight = graph.get_node(weight_id).unwrap();
        let (values, logical_shape) = match &weight.opcode {
            Opcode::Constant(TensorValue::Data { bytes, tensor_type }) => {
                let shape: Vec<usize> = tensor_type
                    .shape
                    .iter()
                    .map(|dim| dim.evaluate().map(|value| value as usize))
                    .collect::<Option<_>>()
                    .ok_or_else(|| {
                        FastnnError::compilation("dynamic W4A8 requires static MatMul weights")
                    })?;
                let element_count = shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension));
                let expected_bytes = element_count.and_then(|count| count.checked_mul(4));
                if shape.len() != 2 || expected_bytes != Some(bytes.len()) {
                    return Err(FastnnError::compilation(
                        "dynamic W4A8 requires a two-dimensional F32 MatMul weight constant",
                    ));
                }
                let values: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                (values, shape)
            }
            _ => continue,
        };
        let (k, n) = (logical_shape[0], logical_shape[1]);
        let transposed_len = k
            .checked_mul(n)
            .ok_or_else(|| FastnnError::compilation("dynamic W4A8 weight dimensions overflow"))?;
        let mut transposed = vec![0.0f32; transposed_len];
        for row in 0..k {
            for column in 0..n {
                transposed[column * k + row] = values[row * n + column];
            }
        }
        let expected_ratio_count = n.checked_mul(k.div_ceil(group_size)).ok_or_else(|| {
            FastnnError::compilation("dynamic W4A8 calibration dimensions overflow")
        })?;
        if let Some(ratios) = calibrated_ratios {
            if ratios.len() != expected_ratio_count {
                return Err(FastnnError::compilation(format!(
                    "dynamic W4A8 calibration for weight {weight_id} has {} ratios, expected {expected_ratio_count}",
                    ratios.len()
                )));
            }
            if ratios
                .iter()
                .any(|ratio| !ratio.is_finite() || !(0.0..=1.0).contains(ratio))
            {
                return Err(FastnnError::compilation(format!(
                    "dynamic W4A8 calibration for weight {weight_id} contains an invalid clip ratio"
                )));
            }
        }
        let packed = if let Some(ratios) = calibrated_ratios {
            PackedTensor::<I4x8>::from_f32_k_grouped_i4_with_clip_ratios(
                &transposed,
                &[n, k],
                group_size,
                ratios,
            )
        } else {
            PackedTensor::<I4x8>::from_f32_k_grouped_i4(&transposed, &[n, k], group_size)
        };
        let representation = crate::types::ValueRepresentation::packed_affine_dequantization(
            ScalarType::I4,
            8,
            crate::types::QuantizationGranularity::PerGroup {
                axis: 0,
                group_size,
            },
            packed.scales.clone(),
            packed.zeros.clone(),
        )?;
        let tensor_type = TensorType::from_parts(
            logical_shape
                .iter()
                .map(|&dimension| DimExpr::Known(dimension as u64))
                .collect(),
            representation,
            crate::types::TensorStorageLayout {
                encoding: crate::types::StorageEncoding::Packed {
                    word_bits: 32,
                    lanes: 8,
                },
                row_packed: true,
                prefix_bytes: 0,
                suffix_bytes: crate::types::PACKED_SIMD_MARGIN_BYTES,
            },
        );
        let node = graph.get_node_mut(weight_id).unwrap();
        node.opcode = Opcode::Constant(TensorValue::Data {
            bytes: packed.as_bytes().to_vec().into(),
            tensor_type: tensor_type.clone(),
        });
        node.output_type = tensor_type;
        node.attrs
            .insert("quant_block_size".into(), group_size.to_string());
    }
    let unused: Vec<_> = clip_ratios
        .keys()
        .filter(|name| !matched_calibration_names.contains(*name))
        .cloned()
        .collect();
    if !unused.is_empty() {
        return Err(FastnnError::compilation(format!(
            "dynamic W4A8 calibration entries did not match quantized MatMuls: {unused:?}"
        )));
    }
    Ok(())
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
        // Quantize weights consumed by MatMul/Conv ops. The weight is input[1];
        // input[0] is the activation.
        let is_matmul_conv = matches!(
            node.opcode,
            Opcode::MatMul | Opcode::Conv1d | Opcode::Conv2d | Opcode::Conv3d
        );
        if !is_matmul_conv {
            return Ok(());
        }

        let weight_idx = 1;
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
                    bytes: bytes.into(),
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
                    bytes: bytes.into(),
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
                bytes: bytes.into(),
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
                bytes: bytes.into(),
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

    for (const_id, consumer_id) in to_quantize {
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

        // A non-finite weight has no meaningful scale: every block or channel
        // scale becomes NaN or Inf, and the codebook path's k-means
        // initialization sorts the values, where a NaN comparison used to
        // abort the import from inside a sort callback. Reject it while the
        // node is still known, so the message names the weight a caller has to
        // fix. The check runs before the MatMul transpose below, so the index
        // is the element's position in the weight as stored.
        if let Some(index) = f32_data.iter().position(|value| !value.is_finite()) {
            let node_name = graph
                .get_node(consumer_id)
                .map(|node| node.name.clone())
                .unwrap_or_default();
            return Err(FastnnError::compilation(format!(
                "cannot quantize the weight of node '{node_name}': element {index} is {}",
                f32_data[index]
            )));
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
            bytes: packed_bytes.into(),
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
    fn a_non_finite_weight_is_rejected_with_the_element_that_is_bad() {
        // A NaN weight used to reach the codebook path's k-means initialisation,
        // which sorted the values with `partial_cmp(..).unwrap()`: the abort came
        // from inside a sort callback and named neither the weight nor the
        // element. Every scale of a NaN block is NaN too, so there is no
        // meaningful quantisation to fall back to.
        let mut weights = vec![0.5f32; 16];
        weights[5] = f32::NAN;
        let mut graph = build_matmul_graph(&weights, &[2, 8]);

        let error = quantize_weights_fp(&mut graph, &QuantTarget::I4Codebook, None)
            .expect_err("a NaN weight must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("cannot quantize the weight of node"),
            "{message}"
        );
        assert!(message.contains("element 5"), "{message}");
        assert!(message.contains("NaN"), "{message}");

        // The same weight below a finite value is still rejected.
        weights[5] = f32::INFINITY;
        let mut graph = build_matmul_graph(&weights, &[2, 8]);
        let error = quantize_weights_fp(&mut graph, &QuantTarget::I4Codebook, None)
            .expect_err("an infinite weight must be rejected");
        assert!(error.to_string().contains("element 5"), "{error}");
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
}
