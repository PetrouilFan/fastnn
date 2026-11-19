// Does the [50257,128] -> [128,50257] transpose scale with threads?
// Ceiling experiment for the FN3-25 threading decision.
//
// The only *safe* partitioning of `&mut [f32]` is by destination rows (a
// contiguous slice per thread), which means each thread reads a strided subset
// of the source. The fastest single-thread loop instead reads the source
// sequentially and writes every destination row, which cannot be split without
// aliasing `&mut` at disjoint indices.

use std::time::Instant;

const ROWS: usize = 50257;
const COLS: usize = 128;

/// One thread's share: destination rows `column_start..column_end`.
fn transpose_columns(
    src: &[f32],
    dst: &mut [f32],
    column_start: usize,
    column_end: usize,
) {
    for column in column_start..column_end {
        let offset = (column - column_start) * ROWS;
        let out_row = &mut dst[offset..offset + ROWS];
        for (row, slot) in out_row.iter_mut().enumerate() {
            *slot = src[row * COLS + column];
        }
    }
}

fn main() {
    let src: Vec<f32> = (0..ROWS * COLS).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut dst = vec![0.0f32; ROWS * COLS];
    let bytes = (ROWS * COLS * 4 * 2) as f64;

    for threads in [1usize, 2, 4, 8] {
        let mut samples = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            let per_thread = COLS.div_ceil(threads);
            let mut bands: Vec<&mut [f32]> = dst.chunks_mut(per_thread * ROWS).collect();
            std::thread::scope(|scope| {
                for (index, band) in bands.iter_mut().enumerate() {
                    let column_start = index * per_thread;
                    let column_end = (column_start + per_thread).min(COLS);
                    let src = &src;
                    scope.spawn(move || transpose_columns(src, band, column_start, column_end));
                }
            });
            samples.push(start.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = samples[samples.len() / 2];
        println!(
            "{threads:>2} thread(s): {ms:8.3} ms  {:6.2} GB/s",
            bytes / (ms * 1e6)
        );
    }
}
