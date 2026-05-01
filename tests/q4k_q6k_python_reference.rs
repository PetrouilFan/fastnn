//! Verify Q4_K and Q6_K GEMV against Python reference values
//! extracted from the actual Gemma-4 GGUF model.
//!
//! Python reference computed using struct.unpack('<e', ...) for f16 conversion
//! (fixing a bug in the original script that used incorrect subnormal f16 handling).

use fastnn::QuantizedDType;

const GGUF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../gemma-4-E2B-it-Q4_K_M.gguf");

/// Python reference: first 256 elements (first block) of Q4_K row 0
/// from blk.0.attn_q.weight (corrected f16 subnormals)
const Q4K_PYTHON_FIRST_BLOCK: [f32; 256] = [
    0.009002, 0.012023, 0.036196, 0.009002, 0.015045, 0.005980, 0.021088, 0.018067,
    0.018067, 0.021088, -0.000063, 0.018067, 0.030153, 0.002959, 0.015045, 0.027131,
    0.009002, 0.033175, 0.012023, 0.012023, 0.005980, 0.009002, -0.000063, 0.002959,
    0.018067, 0.021088, 0.024110, -0.003085, 0.002959, 0.002959, 0.015045, 0.005980,
    0.016645, 0.021088, 0.026420, 0.016645, 0.020199, 0.021977, 0.020199, 0.015756,
    0.016645, 0.018422, 0.016645, 0.021088, 0.020199, 0.016645, 0.027309, 0.018422,
    0.022866, 0.017533, 0.020199, 0.017533, 0.020199, 0.015756, 0.013978, 0.026420,
    0.024643, 0.021088, 0.021088, 0.017533, 0.018422, 0.018422, 0.020199, 0.021977,
    0.006167, 0.017365, -0.001832, 0.009366, 0.015765, 0.009366, 0.010966, 0.006167,
    0.012565, 0.015765, 0.014165, 0.017365, 0.010966, 0.002967, 0.006167, 0.012565,
    0.004567, 0.015765, 0.004567, 0.012565, 0.006167, 0.009366, 0.012565, 0.007766,
    0.002967, 0.007766, 0.007766, 0.001368, 0.015765, 0.010966, 0.002967, 0.017365,
    0.027483, 0.035126, 0.019840, -0.003089, 0.017292, 0.027483, 0.035126, 0.004554,
    0.004554, 0.019840, 0.027483, 0.027483, 0.012197, -0.003089, 0.004554, 0.032578,
    0.024935, 0.019840, 0.027483, 0.030030, 0.024935, -0.000541, 0.019840, 0.007102,
    0.012197, 0.024935, 0.004554, 0.019840, 0.022388, 0.019840, 0.035126, 0.012197,
    0.008029, 0.029358, -0.002636, 0.010695, 0.002696, -0.002636, 0.005363, 0.013361,
    0.021359, 0.002696, 0.002696, 0.018693, 0.021359, 0.018693, 0.013361, 0.013361,
    0.002696, -0.005302, 0.000030, 0.013361, 0.032024, 0.021359, 0.008029, 0.024025,
    -0.002636, 0.010695, 0.021359, 0.016027, 0.013361, 0.008029, 0.000030, 0.008029,
    0.024194, 0.028756, 0.024846, 0.028104, 0.024194, 0.026801, 0.025498, 0.023542,
    0.022239, 0.027453, 0.024846, 0.024194, 0.026149, 0.024846, 0.023542, 0.028104,
    0.020936, 0.026149, 0.021587, 0.028104, 0.030060, 0.022891, 0.026801, 0.022891,
    0.022239, 0.030711, 0.024194, 0.028104, 0.020936, 0.024846, 0.028104, 0.020936,
    0.023804, 0.006326, 0.013318, 0.002831, 0.009822, -0.004160, -0.004160, -0.004160,
    -0.004160, -0.007656, 0.020309, 0.013318, 0.023804, 0.006326, 0.002831, 0.030795,
    0.002831, -0.021638, 0.013318, 0.009822, 0.020309, 0.009822, -0.000665, -0.000665,
    -0.000665, -0.004160, 0.009822, 0.006326, 0.020309, -0.007656, -0.000665, 0.002831,
    0.016973, 0.001687, 0.006782, -0.000861, -0.003408, -0.013599, 0.019520, 0.006782,
    0.016973, 0.009330, 0.006782, 0.004235, 0.001687, 0.004235, 0.016973, 0.004235,
    0.009330, -0.018694, -0.018694, 0.001687, 0.011878, -0.018694, 0.009330, -0.000861,
    0.009330, 0.014425, 0.001687, -0.011051, 0.009330, -0.008503, 0.009330, 0.009330,
];

/// Python reference value for Q4_K GEMV (row 0 dot all-1s activation)
const Q4K_PYTHON_GEMV: f32 = 28.026430;

/// Python reference: first 256 elements (first block) of Q6_K row 0
/// from blk.0.ffn_down.weight (corrected f16 subnormals)
const Q6K_PYTHON_FIRST_BLOCK: [f32; 256] = [
    0.003597, -0.025177, 0.019782, -0.019782, 0.030573, -0.014387, 0.007194, -0.005395,
    0.046758, -0.057549, 0.023379, -0.030573, -0.050355, -0.055750, -0.005395, 0.007194,
    -0.031022, 0.043161, -0.039115, -0.024278, 0.017534, -0.004046, -0.017534, 0.026976,
    0.017534, 0.005395, 0.014837, -0.010790, 0.004046, 0.035069, 0.029673, 0.005395,
    0.036275, 0.005182, -0.039730, 0.012092, 0.017274, 0.053549, 0.034548, 0.043185,
    0.017274, 0.001727, -0.022456, -0.029366, 0.043185, 0.000000, -0.055277, -0.032821,
    -0.010601, 0.022527, 0.025177, 0.042404, -0.022527, -0.033128, -0.033128, -0.015902,
    -0.002650, -0.025177, -0.007951, -0.015902, 0.035779, 0.026503, -0.031803, 0.035779,
    0.027686, 0.001846, -0.022149, -0.003691, 0.031377, 0.038760, 0.031377, -0.029531,
    -0.036914, -0.022149, 0.001846, -0.020303, -0.044297, -0.000000, -0.036914, 0.059063,
    0.006791, 0.019404, 0.022314, -0.021344, -0.031046, 0.030076, -0.020374, -0.002911,
    0.030076, -0.025225, -0.028135, -0.018434, -0.000970, 0.030076, -0.002911, 0.021344,
    -0.009087, -0.021202, -0.030289, 0.012115, 0.006058, -0.042404, 0.075722, 0.027260,
    0.006058, -0.021202, -0.093895, -0.003029, -0.012115, -0.009087, 0.033318, -0.015144,
    0.023852, 0.017889, -0.000000, -0.013914, -0.005963, -0.003975, 0.061619, 0.013914,
    0.009938, -0.001988, -0.045717, -0.011926, -0.007951, -0.041742, 0.027828, 0.037766,
    0.000000, 0.026692, 0.000000, -0.015570, 0.046711, -0.060057, 0.053384, 0.006673,
    -0.057832, 0.035589, -0.031141, -0.024468, -0.066730, -0.004449, -0.071178, -0.002224,
    -0.008803, 0.017605, 0.046947, -0.014671, 0.002934, -0.017605, 0.020540, -0.023474,
    -0.008803, 0.014671, 0.052816, -0.008803, 0.002934, -0.055750, 0.090961, 0.008803,
    0.031022, 0.020232, -0.025627, 0.025627, 0.009442, -0.012139, -0.013488, 0.041813,
    0.001349, 0.006744, 0.002698, 0.006744, 0.013488, 0.009442, -0.012139, -0.020232,
    -0.064363, 0.006034, 0.008045, 0.024136, -0.048273, -0.012068, -0.012068, -0.028159,
    0.032182, -0.020114, -0.048273, -0.034193, 0.058329, -0.012068, -0.020114, 0.018102,
    0.048722, -0.005040, -0.003360, -0.040322, -0.053762, -0.042002, 0.016801, 0.052082,
    0.016801, -0.030241, 0.040322, -0.015121, -0.020161, -0.047042, -0.025201, 0.011761,
    -0.006247, -0.029673, 0.023426, -0.026550, 0.003124, -0.007809, 0.006247, -0.046853,
    -0.045291, 0.004685, 0.048415, 0.009371, 0.009371, 0.004685, -0.037482, 0.017179,
    -0.021865, -0.026550, 0.001562, -0.012494, -0.014056, 0.006247, -0.009371, -0.010932,
    -0.006247, -0.042168, 0.014056, -0.003124, -0.001562, -0.014056, -0.020303, -0.048415,
    -0.036417, -0.028751, -0.059418, -0.030667, -0.026834, 0.011500, -0.013417, -0.009584,
    0.005750, -0.015334, 0.021084, 0.019167, 0.007667, 0.047918, 0.042168, 0.044084,
];

/// Python reference value for Q6_K GEMV (row 0 dot all-1s activation)
const Q6K_PYTHON_GEMV: f32 = -1.578871;

fn load_gguf() -> Option<fastnn::GgufFile> {
    let model_path = std::path::PathBuf::from(GGUF_PATH);
    if !model_path.exists() {
        eprintln!(
            "Model file not found at {:?}; skipping reference test.",
            model_path
        );
        return None;
    }
    Some(fastnn::GgufFile::from_path(&model_path).expect("failed to parse GGUF"))
}

#[test]
fn test_q4_k_python_reference_dequantize() {
    let Some(gguf) = load_gguf() else { return };

    let tensor = gguf
        .get_tensor("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight not found");
    assert_eq!(tensor.dtype(), QuantizedDType::Q4_K);

    let row0 = tensor.row(0);
    assert_eq!(row0.len(), tensor.in_features());

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    for (i, (rust_val, python_val)) in row0[..256]
        .iter()
        .zip(Q4K_PYTHON_FIRST_BLOCK.iter())
        .enumerate()
    {
        let diff = (rust_val - python_val).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    eprintln!(
        "Q4_K first block: max_diff = {:.8} at index {}",
        max_diff, max_diff_idx
    );
    eprintln!(
        "  Rust[{}] = {:.8}, Python[{}] = {:.8}",
        max_diff_idx, row0[max_diff_idx], max_diff_idx, Q4K_PYTHON_FIRST_BLOCK[max_diff_idx]
    );

    assert!(
        max_diff < 1e-3,
        "Q4_K first block max diff too large: {:.8} at index {} (rust={:.8}, python={:.8})",
        max_diff,
        max_diff_idx,
        row0[max_diff_idx],
        Q4K_PYTHON_FIRST_BLOCK[max_diff_idx]
    );
}

#[test]
fn test_q4_k_python_reference_gemv() {
    let Some(gguf) = load_gguf() else { return };

    let tensor = gguf
        .get_tensor("blk.0.attn_q.weight")
        .expect("blk.0.attn_q.weight not found");

    let in_features = tensor.in_features();
    let out_features = tensor.out_features();
    let activation = vec![1.0f32; in_features];
    let mut output = vec![0.0f32; out_features];

    tensor.gemv(&activation, &mut output);

    let diff = (output[0] - Q4K_PYTHON_GEMV).abs();
    let rel_diff = diff / Q4K_PYTHON_GEMV.abs();

    eprintln!(
        "Q4_K GEMV row0: rust={:.8}, python={:.8}, abs_diff={:.8}, rel_diff={:.8}",
        output[0], Q4K_PYTHON_GEMV, diff, rel_diff
    );

    assert!(
        rel_diff < 0.01,
        "Q4_K GEMV mismatch: rust={:.8}, python={:.8}, rel_diff={:.6}",
        output[0],
        Q4K_PYTHON_GEMV,
        rel_diff
    );
}

#[test]
fn test_q6_k_python_reference_dequantize() {
    let Some(gguf) = load_gguf() else { return };

    let tensor = gguf
        .get_tensor("blk.0.ffn_down.weight")
        .expect("blk.0.ffn_down.weight not found");
    assert_eq!(tensor.dtype(), QuantizedDType::Q6_K);

    let row0 = tensor.row(0);
    assert_eq!(row0.len(), tensor.in_features());

    let mut max_diff = 0.0f32;
    let mut max_diff_idx = 0usize;
    for (i, (rust_val, python_val)) in row0[..256]
        .iter()
        .zip(Q6K_PYTHON_FIRST_BLOCK.iter())
        .enumerate()
    {
        let diff = (rust_val - python_val).abs();
        if diff > max_diff {
            max_diff = diff;
            max_diff_idx = i;
        }
    }

    eprintln!(
        "Q6_K first block: max_diff = {:.8} at index {}",
        max_diff, max_diff_idx
    );
    eprintln!(
        "  Rust[{}] = {:.8}, Python[{}] = {:.8}",
        max_diff_idx, row0[max_diff_idx], max_diff_idx, Q6K_PYTHON_FIRST_BLOCK[max_diff_idx]
    );

    assert!(
        max_diff < 1e-3,
        "Q6_K first block max diff too large: {:.8} at index {} (rust={:.8}, python={:.8})",
        max_diff,
        max_diff_idx,
        row0[max_diff_idx],
        Q6K_PYTHON_FIRST_BLOCK[max_diff_idx]
    );
}

#[test]
fn test_q6_k_python_reference_gemv() {
    let Some(gguf) = load_gguf() else { return };

    let tensor = gguf
        .get_tensor("blk.0.ffn_down.weight")
        .expect("blk.0.ffn_down.weight not found");

    let in_features = tensor.in_features();
    let out_features = tensor.out_features();
    let activation = vec![1.0f32; in_features];
    let mut output = vec![0.0f32; out_features];

    tensor.gemv(&activation, &mut output);

    let diff = (output[0] - Q6K_PYTHON_GEMV).abs();
    let rel_diff = if Q6K_PYTHON_GEMV.abs() > 1e-6 {
        diff / Q6K_PYTHON_GEMV.abs()
    } else {
        diff
    };

    eprintln!(
        "Q6_K GEMV row0: rust={:.8}, python={:.8}, abs_diff={:.8}, rel_diff={:.8}",
        output[0], Q6K_PYTHON_GEMV, diff, rel_diff
    );

    assert!(
        rel_diff < 0.01,
        "Q6_K GEMV mismatch: rust={:.8}, python={:.8}, rel_diff={:.6}",
        output[0],
        Q6K_PYTHON_GEMV,
        rel_diff
    );
}