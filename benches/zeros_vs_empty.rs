use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("Benchmark: zeros() vs empty()");
    println!("============================================================");
    println!("{:<20} {:<15} {:<15} {:<10}", "Size", "zeros() ms", "empty() ms", "Speedup");
    println!("------------------------------------------------------------");
    
    let sizes = vec![
        vec![10],
        vec![100],
        vec![1000],
        vec![10000],
        vec![100000],
        vec![1000000],
    ];
    
    for size in sizes {
        let numel: i64 = size.iter().product();
        let nbytes = (numel * 4) as usize;
        
        // Benchmark zeros
        let start = Instant::now();
        for _ in 0..100 {
            let _tensor = fastnn::Tensor::zeros(size.clone(), fastnn::DType::F32, fastnn::Device::Cpu);
        }
        let zeros_time = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
        
        // Benchmark empty
        let start = Instant::now();
        for _ in 0..100 {
            let _tensor = fastnn::Tensor::empty(size.clone(), fastnn::DType::F32, fastnn::Device::Cpu);
        }
        let empty_time = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
        
        let speedup = if empty_time > 0.0 { zeros_time / empty_time } else { f64::INFINITY };
        println!("{:<20} {:<15.4} {:<15.4} {:<10.2}x", format!("({})", numel), zeros_time, empty_time, speedup);
    }
    
    println!("============================================================");
}
