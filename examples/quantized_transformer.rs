//! Example: Quantized Transformer with Native Packed Precision
//!
//! This example demonstrates end-to-end quantized inference using FastNN's
//! native packed precision types. The model uses INT4 (U4x8) weights for
//! maximum memory efficiency while maintaining competitive accuracy.

use fastnn::dtypes::U4x8;
use fastnn::nn::transformer::PackedTransformerEncoder;
use fastnn::tensor::Tensor;

fn main() {
    println!("============================================================");
    println!("FastNN Quantized Transformer Example");
    println!("============================================================\n");

    // Model configuration (similar to GPT-2 small)
    let vocab_size = 50257;
    let max_seq_len = 1024;
    let d_model = 768;
    let num_heads = 12;
    let num_layers = 12;
    let ff_dim = 3072;
    let num_classes = 10; // For classification task
    let dropout_p = 0.1;

    println!("Model Configuration:");
    println!("  Vocabulary size:    {}", vocab_size);
    println!("  Max sequence len:   {}", max_seq_len);
    println!("  Model dimension:    {}", d_model);
    println!("  Attention heads:    {}", num_heads);
    println!("  Transformer layers: {}", num_layers);
    println!("  FFN dimension:      {}", ff_dim);
    println!("  Output classes:     {}", num_classes);
    println!("  Dropout:            {}", dropout_p);
    println!("  Weight precision:   U4x8 (4-bit integers)\n");

    // Create quantized transformer encoder
    println!("Creating quantized transformer encoder...");
    let model = PackedTransformerEncoder::<U4x8>::new(
        vocab_size,
        max_seq_len,
        d_model,
        num_heads,
        num_layers,
        ff_dim,
        num_classes,
        dropout_p,
    );

    // Calculate model size
    let total_params = vocab_size * d_model  // Embeddings
        + max_seq_len * d_model  // Positional embeddings
        + num_layers * (      // Transformer layers
            d_model * d_model * 3 +  // QKV projection
            d_model * d_model +      // Output projection
            d_model * ff_dim +       // FFN layer 1
            ff_dim * d_model         // FFN layer 2
        )
        + d_model * num_classes; // Classifier

    let f32_size_mb = (total_params * 4) as f64 / (1024.0 * 1024.0);
    let u4_size_mb = (total_params / 2) as f64 / (1024.0 * 1024.0);

    println!("\nModel Size:");
    println!("  Total parameters:   {}", total_params);
    println!("  F32 (baseline):     {:.1} MB", f32_size_mb);
    println!("  U4x8 (quantized):   {:.1} MB", u4_size_mb);
    println!("  Memory savings:     {:.1}x\n", f32_size_mb / u4_size_mb);

    // Set model to evaluation mode
    model.eval_mode();

    // Create sample input (batch of 2 sequences, each with 32 tokens)
    let batch_size = 2;
    let seq_len = 32;
    println!("Running inference...");
    println!("  Batch size:         {}", batch_size);
    println!("  Sequence length:    {}", seq_len);

    // Create token IDs (random for demonstration)
    let token_ids: Vec<i64> = (0..batch_size * seq_len)
        .map(|i| (i % vocab_size) as i64)
        .collect();
    let input = Tensor::from_vec(
        token_ids.iter().map(|&x| x as f32).collect(),
        &[batch_size, seq_len],
    );

    // Run forward pass
    let output = model.forward(&input);

    println!("\nOutput shape: {:?}", output.shape());
    println!("Output values (first batch):");
    let output_data = output.to_vec();
    for i in 0..batch_size {
        print!("  Batch {}: [", i);
        for j in 0..num_classes.min(5) {
            print!(" {:.4}", output_data[i * num_classes as usize + j]);
        }
        if num_classes > 5 {
            print!(" ...");
        }
        println!(" ]");
    }

    // Demonstrate memory bandwidth savings
    println!("\n============================================================");
    println!("Memory Bandwidth Analysis");
    println!("============================================================");
    println!(
        "For a single forward pass with batch_size={}, seq_len={}:",
        batch_size, seq_len
    );

    let tokens_processed = batch_size * seq_len;
    let memory_f32 = tokens_processed as f64 * d_model as f64 * 4.0 / (1024.0 * 1024.0);
    let memory_u4 = tokens_processed as f64 * d_model as f64 * 0.5 / (1024.0 * 1024.0);

    println!("  F32 memory traffic:  {:.1} MB", memory_f32);
    println!("  U4x8 memory traffic: {:.1} MB", memory_u4);
    println!("  Bandwidth savings:   {:.1}x\n", memory_f32 / memory_u4);

    println!("============================================================");
    println!("Key Advantages of Native Packed Precision");
    println!("============================================================");
    println!("✓ No post-training quantization step required");
    println!("✓ No calibration dataset needed");
    println!("✓ No dequantization overhead during inference");
    println!("✓ Direct SIMD operations on packed data");
    println!("✓ Better cache utilization (8x fewer cache lines)");
    println!("✓ Reduced memory bandwidth requirements");
    println!("✓ Competitive accuracy with INT8 quantization\n");

    println!("Quantized transformer inference completed successfully!");
}
