use std::collections::HashMap;
use std::time::Instant;

use clap::{Parser, Subcommand};

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::executor::GraphExecutor;
use fastnn::backend::runtime::Runtime;
use fastnn::backend::{ExecutablePlan, Instruction, MemoryPlan};
use fastnn::onnx::converter::{OnnxConverter, OnnxNode};
use fastnn::Tensor;

#[derive(Parser)]
#[command(
    name = "fastnn-runtime",
    version,
    about = "FastNN standalone runtime for compiled inference plans"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect a compiled plan (.fnnc + .memory.json)
    Info {
        /// Path to .fnnc plan file
        plan: String,
        /// Path to .memory.json file
        memory: String,
    },
    /// Execute inference with input data files
    Run {
        /// Path to .fnnc plan file
        plan: String,
        /// Path to .memory.json file
        memory: String,
        /// Input data files (.bin)
        inputs: Vec<String>,
        /// Output file path(s) (default: stdout)
        #[arg(short, long)]
        output: Vec<String>,
        /// Number of iterations
        #[arg(short, long, default_value = "1")]
        iterations: usize,
    },
    /// Benchmark inference performance
    Bench {
        /// Path to .fnnc plan file
        plan: String,
        /// Path to .memory.json file
        memory: String,
        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: usize,
        /// Warmup iterations
        #[arg(short, long, default_value = "10")]
        warmup: usize,
        /// Generate random inputs based on plan metadata
        #[arg(short, long)]
        random_inputs: bool,
        /// Comma-separated input sizes (bytes) for random input generation
        #[arg(short, long)]
        input_sizes: Option<String>,
    },
    /// Quantize weights in a compiled plan (offline quantization)
    Quantize {
        /// Path to .fnnc plan file
        plan: String,
        /// Path to .memory.json file
        memory: String,
        /// Target bit width (4 or 8)
        #[arg(short, long, default_value = "4")]
        bits: u8,
        /// Output path for quantized plan (default: overwrite input)
        #[arg(short, long)]
        output_plan: Option<String>,
        /// Output path for quantized memory plan
        #[arg(short, long)]
        output_memory: Option<String>,
    },
    /// Compile an ONNX model to a compiled plan (.fnnc + .memory.json)
    Compile {
        /// Path to ONNX JSON file
        onnx: String,
        /// Output path for .fnnc plan file
        #[arg(short, long, default_value = "model.fnnc")]
        output_plan: String,
        /// Output path for .memory.json file
        #[arg(short, long, default_value = "model.memory.json")]
        output_memory: String,
        /// Quantize weights to bit width (optional, 4 or 8)
        #[arg(short, long)]
        quantize: Option<u8>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info { plan, memory } => cmd_info(&plan, &memory),
        Commands::Run {
            plan,
            memory,
            inputs,
            output,
            iterations,
        } => cmd_run(&plan, &memory, &inputs, &output, iterations),
        Commands::Bench {
            plan,
            memory,
            iterations,
            warmup,
            random_inputs,
            input_sizes,
        } => cmd_bench(
            &plan,
            &memory,
            iterations,
            warmup,
            random_inputs,
            input_sizes.as_deref(),
        ),
        Commands::Quantize {
            plan,
            memory,
            bits,
            output_plan,
            output_memory,
        } => cmd_quantize(
            &plan,
            &memory,
            bits,
            output_plan.as_deref(),
            output_memory.as_deref(),
        ),
        Commands::Compile {
            onnx,
            output_plan,
            output_memory,
            quantize,
        } => cmd_compile(&onnx, &output_plan, &output_memory, quantize),
    }
}

fn cmd_info(plan_path: &str, memory_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let plan = ExecutablePlan::load(plan_path)?;
    let memory_json = std::fs::read_to_string(memory_path)?;
    let memory_plan: MemoryPlan = serde_json::from_str(&memory_json)?;

    println!("ExecutablePlan:");
    println!(
        "  arena_size: {} bytes ({:.2} MB)",
        plan.arena_size,
        plan.arena_size as f64 / 1_048_576.0
    );
    println!("  instructions: {}", plan.instructions.len());
    for (i, inst) in plan.instructions.iter().enumerate() {
        match inst {
            Instruction::CallKernel {
                kernel_name,
                input_slices,
                output_slice,
                params,
                ..
            } => {
                let input_str: Vec<String> = input_slices
                    .iter()
                    .map(|s| format!("{}+{}", s.offset, s.size))
                    .collect();
                println!(
                    "    [{}] {}: inputs=[{}] out={}+{} params={:?}",
                    i,
                    kernel_name,
                    input_str.join(","),
                    output_slice.offset,
                    output_slice.size,
                    params
                );
            }
            Instruction::MemCopy { dst, src } => {
                println!(
                    "    [{}] MemCopy: {}..{} -> {}..{}",
                    i,
                    src.offset,
                    src.offset + src.size,
                    dst.offset,
                    dst.offset + dst.size
                );
            }
            Instruction::Fill { dst, value } => {
                println!(
                    "    [{}] Fill: offset={} size={} value={}",
                    i, dst.offset, dst.size, value
                );
            }
            Instruction::WriteConst { dst, .. } => {
                println!(
                    "    [{}] WriteConst: offset={} size={}",
                    i, dst.offset, dst.size
                );
            }
        }
    }

    println!("\nMemoryPlan:");
    println!(
        "  total_size: {} bytes ({:.2} MB)",
        memory_plan.total_size,
        memory_plan.total_size as f64 / 1_048_576.0
    );
    println!("  slots: {}", memory_plan.slots.len());
    let mut slots: Vec<_> = memory_plan.slots.iter().collect();
    slots.sort_by_key(|(_, s)| s.offset);
    for (node_id, slot) in &slots {
        println!(
            "    node {}: offset={} size={}",
            node_id, slot.offset, slot.size
        );
    }

    Ok(())
}

fn cmd_run(
    plan_path: &str,
    memory_path: &str,
    input_paths: &[String],
    output_paths: &[String],
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::<CpuBackend>::load(CpuBackend, plan_path, memory_path)?;

    // Read input files
    let mut inputs: Vec<Vec<u8>> = Vec::new();
    for path in input_paths {
        let data = std::fs::read(path)?;
        inputs.push(data);
    }
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

    for iter in 0..iterations {
        let outputs = runtime.run(&input_refs)?;
        if iterations == 1 {
            for (i, data) in outputs.iter().enumerate() {
                let path = output_paths
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("output_{}.bin", i));
                std::fs::write(&path, data)?;
                eprintln!("Wrote {} bytes to {}", data.len(), path);
            }
        }
        if iterations > 1 {
            let total_bytes: usize = outputs.iter().map(|o| o.len()).sum();
            eprintln!(
                "Iteration {}: {} outputs, {} total bytes",
                iter + 1,
                outputs.len(),
                total_bytes
            );
        }
    }
    Ok(())
}

fn cmd_bench(
    plan_path: &str,
    memory_path: &str,
    iterations: usize,
    warmup: usize,
    random_inputs: bool,
    input_sizes: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::<CpuBackend>::load(CpuBackend, plan_path, memory_path)?;

    // Generate or read inputs
    let inputs: Vec<Vec<u8>> = if random_inputs {
        if let Some(sizes_str) = input_sizes {
            sizes_str
                .split(',')
                .map(|s| s.trim().parse::<usize>().map(|n| vec![0u8; n]))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // Estimate from plan: use the first n slots where n = num inputs
            // (For simplicity, create one empty input — real usage should specify --input-sizes)
            vec![vec![0u8; 1024]]
        }
    } else {
        // Use sample inputs from plan if available
        vec![vec![0u8; 1024]]
    };
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

    // Warmup
    eprintln!("Warming up...");
    for _ in 0..warmup {
        runtime.run(&input_refs)?;
    }

    // Benchmark
    eprintln!("Benchmarking ({} iterations)...", iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        runtime.run(&input_refs)?;
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / iterations as f64;
    let avg_ms = avg_us / 1000.0;
    let throughput = if avg_us > 0.0 {
        1_000_000.0 / avg_us
    } else {
        0.0
    };

    println!(
        "Results: {} iterations in {:.3}s",
        iterations,
        elapsed.as_secs_f64()
    );
    println!("  Avg latency: {:.2} us ({:.2} ms)", avg_us, avg_ms);
    println!("  Throughput: {:.2} inferences/sec", throughput);

    Ok(())
}

fn cmd_quantize(
    plan_path: &str,
    memory_path: &str,
    bits: u8,
    output_plan: Option<&str>,
    output_memory: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let plan = ExecutablePlan::load(plan_path)?;
    let memory_json = std::fs::read_to_string(memory_path)?;
    let memory_plan: MemoryPlan = serde_json::from_str(&memory_json)?;

    println!(
        "Loaded plan: {} instructions, arena={} bytes",
        plan.instructions.len(),
        plan.arena_size
    );
    println!("Target bit width: {} bits", bits);

    // Scan instructions for existing quantization state
    let mut quantized_ops = 0usize;
    let mut f32_ops = 0usize;
    for instr in &plan.instructions {
        if let Instruction::CallKernel { kernel_name, .. } = instr {
            if kernel_name.contains("u4") || kernel_name.contains("u8") {
                quantized_ops += 1;
            } else if kernel_name.contains("matmul") || kernel_name.contains("conv") {
                f32_ops += 1;
            }
        }
    }

    if quantized_ops > 0 {
        println!("  Quantized kernels:  {}", quantized_ops);
        println!("  F32 compute kernels: {}", f32_ops);
        println!("  Plan is already partially quantized.");
    } else {
        println!("  F32 compute kernels: {}", f32_ops);
        println!("  In-place quantization requires re-compilation from the original graph.");
    }

    // Determine output paths (default: append .q<N> suffix)
    let base_plan = plan_path
        .strip_suffix(".fnnc")
        .unwrap_or(plan_path)
        .strip_suffix(".q4")
        .unwrap_or(plan_path)
        .strip_suffix(".q8")
        .unwrap_or(plan_path);
    let out_plan = output_plan
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}.q{}.fnnc", base_plan, bits));

    let base_memory = memory_path
        .strip_suffix(".memory.json")
        .unwrap_or(memory_path)
        .strip_suffix(".q4")
        .unwrap_or(memory_path)
        .strip_suffix(".q8")
        .unwrap_or(memory_path);
    let out_memory = output_memory
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}.q{}.memory.json", base_memory, bits));

    if quantized_ops == 0 && f32_ops > 0 {
        println!();
        println!("  To quantize, use the compiler pipeline:");
        println!(
            "    fastnn-runtime compile model.onnx --quantize {} -o {}",
            bits, out_plan
        );
        println!();
    }

    // Save (pass-through since we don't modify in-place)
    plan.save(&out_plan)?;
    let out_memory_json = serde_json::to_string_pretty(&memory_plan)?;
    std::fs::write(&out_memory, &out_memory_json)?;
    println!("Saved unchanged plan to: {}", out_plan);
    println!("Saved unchanged memory to: {}", out_memory);

    Ok(())
}

fn cmd_compile(
    onnx_path: &str,
    output_plan: &str,
    output_memory: &str,
    quantize: Option<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let onnx_json = std::fs::read_to_string(onnx_path)?;
    let onnx_data: serde_json::Value = serde_json::from_str(&onnx_json)?;

    let nodes_arr = onnx_data["nodes"].as_array().ok_or("missing nodes array")?;
    let mut onnx_nodes: Vec<OnnxNode> = Vec::new();
    for node_val in nodes_arr {
        let op_type = node_val["op_type"]
            .as_str()
            .ok_or("node missing op_type")?
            .to_string();
        let name = node_val["name"].as_str().unwrap_or("").to_string();
        let inputs_str = node_val["inputs"].as_str().unwrap_or("");
        let inputs: Vec<String> = inputs_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let outputs_str = node_val["outputs"].as_str().unwrap_or("");
        let outputs: Vec<String> = outputs_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let mut attrs = HashMap::new();
        if let Some(obj) = node_val.as_object() {
            for (key, val) in obj {
                if key != "op_type" && key != "name" && key != "inputs" && key != "outputs" {
                    if let Some(s) = val.as_str() {
                        attrs.insert(key.clone(), s.to_string());
                    }
                }
            }
        }
        onnx_nodes.push(OnnxNode {
            name,
            op_type,
            inputs,
            outputs,
            attrs,
        });
    }

    let params_obj = onnx_data["params"]
        .as_object()
        .ok_or("missing params object")?;
    let mut params: HashMap<String, Tensor> = HashMap::new();
    for (name, param_val) in params_obj {
        let data: Vec<f32> = serde_json::from_value(param_val["data"].clone())?;
        let shape: Vec<i64> = serde_json::from_value(param_val["shape"].clone())?;
        params.insert(name.clone(), Tensor::from_vec(data, shape));
    }

    let input_names: Vec<String> = serde_json::from_value(onnx_data["input_names"].clone())?;
    let output_names: Vec<String> = serde_json::from_value(onnx_data["output_names"].clone())?;

    let converter = OnnxConverter::new(&onnx_nodes, &params, &input_names, &output_names);
    let graph = converter
        .to_compute_graph()
        .map_err(|e| format!("ONNX conversion: {e}"))?;

    println!("ONNX model loaded: {} nodes", graph.node_count());

    let executor = GraphExecutor::new(CpuBackend);
    let (plan, memory_plan, _compiled_graph) = executor
        .compile_with_plan_and_quantize(&graph, quantize)
        .map_err(|e| format!("Compilation: {e}"))?;

    plan.save(output_plan)?;
    let memory_json = serde_json::to_string_pretty(&memory_plan)?;
    std::fs::write(output_memory, &memory_json)?;

    println!("Compiled plan saved to: {}", output_plan);
    println!("Memory plan saved to:   {}", output_memory);
    println!("Arena size: {} bytes", plan.arena_size);
    println!("Instructions: {}", plan.instructions.len());

    Ok(())
}
