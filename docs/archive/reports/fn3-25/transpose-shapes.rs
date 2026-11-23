// Standalone study of the [50257,128] -> [128,50257] f32 transpose.
// Compiled with `rustc -O` so tile geometry can be compared without a
// 2-minute LTO rebuild of the crate.

use std::time::Instant;

const ROWS: usize = 50257;
const COLS: usize = 128;

fn naive(src: &[f32], dst: &mut [f32]) {
    for r in 0..ROWS {
        for c in 0..COLS {
            dst[c * ROWS + r] = src[r * COLS + c];
        }
    }
}

/// Current implementation: 32x32 tile, 4x4 register blocks.
fn tile_block4(src: &[f32], dst: &mut [f32], tile: usize, block: usize) {
    let mut row = 0;
    while row < ROWS {
        let row_end = (row + tile).min(ROWS);
        let mut column = 0;
        while column < COLS {
            let column_end = (column + tile).min(COLS);
            let mut target = column;
            while target + block <= column_end {
                let mut r = row;
                while r + block <= row_end {
                    let a0 = &src[r * COLS + target..r * COLS + target + block];
                    let a1 = &src[(r + 1) * COLS + target..(r + 1) * COLS + target + block];
                    let a2 = &src[(r + 2) * COLS + target..(r + 2) * COLS + target + block];
                    let a3 = &src[(r + 3) * COLS + target..(r + 3) * COLS + target + block];
                    let mut out = [0.0f32; 4];
                    for (i, slot) in out.iter_mut().enumerate() {
                        *slot = a0[i];
                    }
                    dst[(target) * ROWS + r..target * ROWS + r + block].copy_from_slice(&out);
                    for (i, slot) in out.iter_mut().enumerate() {
                        *slot = a1[i];
                    }
                    dst[(target + 1) * ROWS + r..(target + 1) * ROWS + r + block]
                        .copy_from_slice(&out);
                    for (i, slot) in out.iter_mut().enumerate() {
                        *slot = a2[i];
                    }
                    dst[(target + 2) * ROWS + r..(target + 2) * ROWS + r + block]
                        .copy_from_slice(&out);
                    for (i, slot) in out.iter_mut().enumerate() {
                        *slot = a3[i];
                    }
                    dst[(target + 3) * ROWS + r..(target + 3) * ROWS + r + block]
                        .copy_from_slice(&out);
                    r += block;
                }
                for r in r..row_end {
                    for target in target..target + block {
                        dst[target * ROWS + r] = src[r * COLS + target];
                    }
                }
                target += block;
            }
            for target in target..column_end {
                let out_row = &mut dst[target * ROWS + row..target * ROWS + row_end];
                for (offset, slot) in out_row.iter_mut().enumerate() {
                    *slot = src[(row + offset) * COLS + target];
                }
            }
            column = column_end;
        }
        row = row_end;
    }
}

/// Destination-contiguous tile: `dest_rows` destination rows by `dest_cols`
/// destination columns.
fn rect(src: &[f32], dst: &mut [f32], dest_rows: usize, dest_cols: usize) {
    let mut c0 = 0;
    while c0 < COLS {
        let c1 = (c0 + dest_rows).min(COLS);
        let mut r0 = 0;
        while r0 < ROWS {
            let r1 = (r0 + dest_cols).min(ROWS);
            for c in c0..c1 {
                let out_row = &mut dst[c * ROWS + r0..c * ROWS + r1];
                for (offset, slot) in out_row.iter_mut().enumerate() {
                    *slot = src[(r0 + offset) * COLS + c];
                }
            }
            r0 = r1;
        }
        c0 = c1;
    }
}

/// Same as `rect` but the inner read walks `src` by column: for each source
/// row in the block, gather the `dest_rows` values the tile needs.
fn rect_rows(src: &[f32], dst: &mut [f32], dest_rows: usize, dest_cols: usize) {
    let mut c0 = 0;
    while c0 < COLS {
        let c1 = (c0 + dest_rows).min(COLS);
        let mut r0 = 0;
        while r0 < ROWS {
            let r1 = (r0 + dest_cols).min(ROWS);
            for r in r0..r1 {
                let line = &src[r * COLS + c0..r * COLS + c1];
                for (offset, &value) in line.iter().enumerate() {
                    dst[(c0 + offset) * ROWS + r] = value;
                }
            }
            r0 = r1;
        }
        c0 = c1;
    }
}

/// Destination-contiguous per element: the shape the tiled kernel's inner
/// loop has, without tiling.
fn naive_dst_major(src: &[f32], dst: &mut [f32]) {
    for c in 0..COLS {
        for r in 0..ROWS {
            dst[c * ROWS + r] = src[r * COLS + c];
        }
    }
}

/// Source-sequential read with four rows per inner step: 128 streams of
/// 16-byte destination stores.
fn naive_unroll4(src: &[f32], dst: &mut [f32]) {
    let mut r = 0;
    while r + 4 <= ROWS {
        for c in 0..COLS {
            let block = [
                src[r * COLS + c],
                src[(r + 1) * COLS + c],
                src[(r + 2) * COLS + c],
                src[(r + 3) * COLS + c],
            ];
            dst[c * ROWS + r..c * ROWS + r + 4].copy_from_slice(&block);
        }
        r += 4;
    }
    for r in r..ROWS {
        for c in 0..COLS {
            dst[c * ROWS + r] = src[r * COLS + c];
        }
    }
}

fn naive_unroll8(src: &[f32], dst: &mut [f32]) {
    let mut r = 0;
    while r + 8 <= ROWS {
        for c in 0..COLS {
            let mut block = [0.0f32; 8];
            for (i, slot) in block.iter_mut().enumerate() {
                *slot = src[(r + i) * COLS + c];
            }
            dst[c * ROWS + r..c * ROWS + r + 8].copy_from_slice(&block);
        }
        r += 8;
    }
    for r in r..ROWS {
        for c in 0..COLS {
            dst[c * ROWS + r] = src[r * COLS + c];
        }
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let src: Vec<f32> = (0..ROWS * COLS).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut dst = vec![0.0f32; ROWS * COLS];
    let bytes = (ROWS * COLS * 4 * 2) as f64; // read + write

    let variants: Vec<(&str, Box<dyn Fn(&[f32], &mut [f32])>)> = vec![
        ("naive", Box::new(|s, d| naive(s, d))),
        ("tile32_block4", Box::new(|s, d| tile_block4(s, d, 32, 4))),
        ("tile64_block4", Box::new(|s, d| tile_block4(s, d, 64, 4))),
        ("tile16_block4", Box::new(|s, d| tile_block4(s, d, 16, 4))),
        ("rect_32x512", Box::new(|s, d| rect(s, d, 32, 512))),
        ("rect_16x1024", Box::new(|s, d| rect(s, d, 16, 1024))),
        ("rect_32x1024", Box::new(|s, d| rect(s, d, 32, 1024))),
        ("rect_64x512", Box::new(|s, d| rect(s, d, 64, 512))),
        ("rows_32x512", Box::new(|s, d| rect_rows(s, d, 32, 512))),
        ("rows_16x1024", Box::new(|s, d| rect_rows(s, d, 16, 1024))),
        ("rows_64x256", Box::new(|s, d| rect_rows(s, d, 64, 256))),
        ("rect_128x256", Box::new(|s, d| rect(s, d, 128, 256))),
        ("rect_128x1024", Box::new(|s, d| rect(s, d, 128, 1024))),
        ("naive_dst_major", Box::new(|s, d| naive_dst_major(s, d))),
        ("naive_unroll4", Box::new(|s, d| naive_unroll4(s, d))),
        ("naive_unroll8", Box::new(|s, d| naive_unroll8(s, d))),
    ];

    for (name, run) in &variants {
        let mut samples = Vec::new();
        for _ in 0..12 {
            let start = Instant::now();
            run(&src, &mut dst);
            samples.push(start.elapsed().as_secs_f64() * 1e3);
        }
        let ms = median(samples);
        println!(
            "{name:>16}: {ms:8.3} ms  {:6.2} GB/s  (dst[7]={})",
            bytes / (ms * 1e6),
            dst[7]
        );
    }
}
