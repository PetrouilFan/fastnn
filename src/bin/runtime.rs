use std::time::Instant;

use clap::{Parser, Subcommand};

use fastnn::backend::cpu::CpuBackend;
use fastnn::backend::runtime::Runtime;
use fastnn::backend::{ExecutablePlan, Instruction, MemoryPlan};

#[derive(Parser)]
#[command(name = "fastnn-runtime", version, about = "FastNN standalone runtime for compiled inference plans")]
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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Info { plan, memory } => cmd_info(&plan, &memory),
        Commands::Run { plan, memory, inputs, output, iterations } => {
            cmd_run(&plan, &memory, &inputs, &output, iterations)
        }
        Commands::Bench { plan, memory, iterations, warmup, random_inputs, input_sizes } => {
            cmd_bench(&plan, &memory, iterations, warmup, random_inputs, input_sizes.as_deref())
        }
    }
}

fn cmd_info(plan_path: &str, memory_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let plan = ExecutablePlan::load(plan_path)?;
    let memory_json = std::fs::read_to_string(memory_path)?;
    let memory_plan: MemoryPlan = serde_json::from_str(&memory_json)?;

    println!("ExecutablePlan:");
    println!("  arena_size: {} bytes ({:.2} MB)", plan.arena_size, plan.arena_size as f64 / 1_048_576.0);
    println!("  instructions: {}", plan.instructions.len());
    for (i, inst) in plan.instructions.iter().enumerate() {
        match inst {
            Instruction::CallKernel { kernel_name, input_slices, output_slice, params, .. } => {
                let input_str: Vec<String> = input_slices.iter()
                    .map(|s| format!("{}+{}", s.offset, s.size))
                    .collect();
                println!("    [{}] {}: inputs=[{}] out={}+{} params={:?}",
                    i, kernel_name, input_str.join(","),
                    output_slice.offset, output_slice.size, params);
            }
            Instruction::MemCopy { dst, src } => {
                println!("    [{}] MemCopy: {}..{} -> {}..{}", i, src.offset, src.offset + src.size, dst.offset, dst.offset + dst.size);
            }
            Instruction::Fill { dst, value } => {
                println!("    [{}] Fill: offset={} size={} value={}", i, dst.offset, dst.size, value);
            }
            Instruction::WriteConst { dst, .. } => {
                println!("    [{}] WriteConst: offset={} size={}", i, dst.offset, dst.size);
            }
        }
    }

    println!("\nMemoryPlan:");
    println!("  total_size: {} bytes ({:.2} MB)", memory_plan.total_size, memory_plan.total_size as f64 / 1_048_576.0);
    println!("  slots: {}", memory_plan.slots.len());
    let mut slots: Vec<_> = memory_plan.slots.iter().collect();
    slots.sort_by_key(|(_, s)| s.offset);
    for (node_id, slot) in &slots {
        println!("    node {}: offset={} size={}", node_id, slot.offset, slot.size);
    }

    Ok(())
}

fn cmd_run(
    plan_path: &str, memory_path: &str,
    input_paths: &[String], output_paths: &[String],
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
                let path = output_paths.get(i).cloned().unwrap_or_else(|| format!("output_{}.bin", i));
                std::fs::write(&path, data)?;
                eprintln!("Wrote {} bytes to {}", data.len(), path);
            }
        }
        if iterations > 1 {
            let total_bytes: usize = outputs.iter().map(|o| o.len()).sum();
            eprintln!("Iteration {}: {} outputs, {} total bytes", iter + 1, outputs.len(), total_bytes);
        }
    }
    Ok(())
}

fn cmd_bench(
    plan_path: &str, memory_path: &str,
    iterations: usize, warmup: usize,
    random_inputs: bool, input_sizes: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Runtime::<CpuBackend>::load(CpuBackend, plan_path, memory_path)?;

    // Generate or read inputs
    let inputs: Vec<Vec<u8>> = if random_inputs {
        if let Some(sizes_str) = input_sizes {
            sizes_str.split(',')
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
    let throughput = if avg_us > 0.0 { 1_000_000.0 / avg_us } else { 0.0 };

    println!("Results: {} iterations in {:.3}s", iterations, elapsed.as_secs_f64());
    println!("  Avg latency: {:.2} us ({:.2} ms)", avg_us, avg_ms);
    println!("  Throughput: {:.2} inferences/sec", throughput);

    Ok(())
}
