#![no_main]

use fastnn::types::{
    QuantizationGranularity, QuantizationScheme, QuantizationSpec,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let scheme_tag = data.first().copied().unwrap_or(0) % 3;
    let granularity_tag = data.get(1).copied().unwrap_or(0) % 3;
    let axis = data.get(2).copied().unwrap_or(0) as usize;
    let group_size = data.get(3).copied().unwrap_or(0) as usize;
    let granularity = match granularity_tag {
        0 => QuantizationGranularity::PerTensor,
        1 => QuantizationGranularity::PerAxis { axis },
        _ => QuantizationGranularity::PerGroup { axis, group_size },
    };

    let values: Vec<f32> = data[4.min(data.len())..]
        .chunks_exact(4)
        .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes.try_into().unwrap())))
        .collect();
    let split = values.len() / 2;
    let scales = values[..split].to_vec();
    let zero_points = values[split..]
        .iter()
        .map(|value| value.to_bits() as i32)
        .collect();
    let scheme = match scheme_tag {
        0 => QuantizationScheme::Symmetric,
        1 => QuantizationScheme::Asymmetric,
        _ => QuantizationScheme::Codebook {
            entries: values.chunks(16).map(|entry| entry.to_vec()).collect(),
        },
    };
    let spec = QuantizationSpec {
        scheme,
        granularity,
        scales,
        zero_points,
    };
    let _ = spec.validate();
});
