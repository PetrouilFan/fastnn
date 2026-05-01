//! Integration test: validate GgufFile against real Gemma-4 model.

#[cfg(test)]
mod test {
    use fastnn::GgufFile;
    use std::path::PathBuf;

    #[test]
    fn test_parse_real_gemma4() {
        let model_path = PathBuf::from(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../gemma-4-E2B-it-Q4_K_M.gguf"
            )
        );
        if !model_path.exists() {
            eprintln!("Model file not found at {:?}; skipping integration test.", model_path);
            return;
        }

        let size_mb = std::fs::metadata(&model_path).unwrap().len() as f64 / (1024.0 * 1024.0);
        eprintln!("Loading GGUF file: {:?} ({:.1} MB)", model_path, size_mb);

        let gguf = GgufFile::from_path(&model_path).expect("failed to parse GGUF");

        eprintln!("Summary: {}", gguf.summary());

        // Spot-check that we can locate known Gemma-4 tensor names
        assert!(gguf.get_tensor("token_embd.weight").is_some(), "token_embd.weight missing");
        assert!(gguf.get_tensor("output_norm.weight").is_some(), "output_norm.weight missing");

        // Verify a specific transformer block exists (Gemma-4 has 35 layers)
        let blk0_ffn_gate = gguf.get_tensor("blk.0.ffn_gate.0.weight");
        if blk0_ffn_gate.is_some() {
            eprintln!("blk.0.ffn_gate.0.weight found — this model uses fused FFN gating");
        }

        let blk0_attn_k = gguf.get_tensor("blk.0.attn_k.weight");
        assert!(blk0_attn_k.is_some(), "blk.0.attn_k.weight missing");

        // Spot-check that we can locate known Gemma-4 tensor names

        let mut q4_k_count = 0usize;
        let mut total_params_mb: f64 = 0.0;
        for info in &gguf.tensors {
            if fastnn::QuantizedDType::Q4_K == info.dtype {
                q4_k_count += 1;
            }
            let numel: u64 = info.dims[..info.n_dims as usize]
                .iter()
                .product();
            total_params_mb += (numel as f64) / (1024.0 * 1024.0);
        }
        eprintln!("Q4_K tensors: {} / {}", q4_k_count, gguf.tensors.len());
        eprintln!("Approx params: {:.0}M elements", total_params_mb);
    }
}
