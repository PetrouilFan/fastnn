//! Tiled FlashAttention with online-softmax.
//!
//! Implements: O = softmax(Q @ K^T / sqrt(d)) @ V
//! Using the FlashAttention tiling algorithm to avoid O(N^2) memory.

use crate::storage::{DType, Device};
use crate::tensor::Tensor;

const TILE_SIZE: usize = 32;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: Option<f32>,
    causal: bool,
) -> Tensor {
    let batch = q.shape()[0] as usize;
    let heads = q.shape()[1] as usize;
    let seq_len = q.shape()[2] as usize;
    let dk = q.shape()[3] as usize;
    let dv = v.shape()[3] as usize;
    let scale_val = scale.unwrap_or_else(|| (dk as f32).sqrt().recip());

    let mut output = Tensor::zeros(
        vec![batch as i64, heads as i64, seq_len as i64, dv as i64],
        DType::F32,
        Device::Cpu,
    );

    let q_data = q.as_f32_slice();
    let k_data = k.as_f32_slice();
    let v_data = v.as_f32_slice();
    let out_data = output.as_f32_slice_mut();

    let head_elts_q = seq_len * dk;
    let head_elts_kv = seq_len * dv;

    for b in 0..batch {
        for h in 0..heads {
            let q_off = (b * heads + h) * head_elts_q;
            let k_off = (b * heads + h) * head_elts_q;
            let v_off = (b * heads + h) * head_elts_kv;
            let out_off = (b * heads + h) * head_elts_kv;

            flash_attention_single(
                &q_data[q_off..q_off + head_elts_q],
                &k_data[k_off..k_off + head_elts_q],
                &v_data[v_off..v_off + head_elts_kv],
                &mut out_data[out_off..out_off + head_elts_kv],
                seq_len,
                dk,
                dv,
                scale_val,
                causal,
            );
        }
    }

    output
}

#[allow(clippy::too_many_arguments)]
fn flash_attention_single(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    seq_len: usize,
    dk: usize,
    dv: usize,
    scale: f32,
    causal: bool,
) {
    let mut s_buf = vec![0.0f32; TILE_SIZE * TILE_SIZE];
    let mut m = [f32::NEG_INFINITY; TILE_SIZE];
    let mut l = [0.0f32; TILE_SIZE];
    let mut o_tile = vec![0.0f32; TILE_SIZE * dv];

    for q_start in (0..seq_len).step_by(TILE_SIZE) {
        let q_end = (q_start + TILE_SIZE).min(seq_len);
        let br = q_end - q_start;
        let q_tile = &q[q_start * dk..q_end * dk];

        m[..br].fill(f32::NEG_INFINITY);
        l[..br].fill(0.0);
        o_tile[..br * dv].fill(0.0);

        for kv_start in (0..seq_len).step_by(TILE_SIZE) {
            let kv_end = (kv_start + TILE_SIZE).min(seq_len);
            let bc = kv_end - kv_start;
            let k_tile = &k[kv_start * dk..kv_end * dk];
            let v_tile = &v[kv_start * dv..kv_end * dv];

            let s = matmul_tile(q_tile, k_tile, br, dk, bc, scale);

            for i in 0..br {
                let s_row = &s[i * bc..(i + 1) * bc];
                let mut m_new = m[i];
                for j in 0..bc {
                    let val = if causal && kv_start + j > q_start + i {
                        f32::NEG_INFINITY
                    } else {
                        s_row[j]
                    };
                    if val > m_new {
                        m_new = val;
                    }
                }

                if m_new == f32::NEG_INFINITY {
                    continue;
                }

                let rescale = (m[i] - m_new).exp();
                let mut l_new = 0.0f32;

                let p_row = &mut s_buf[i * TILE_SIZE..i * TILE_SIZE + bc];
                for j in 0..bc {
                    let val = if causal && kv_start + j > q_start + i {
                        f32::NEG_INFINITY
                    } else {
                        s_row[j]
                    };
                    let p = (val - m_new).exp();
                    p_row[j] = p;
                    l_new += p;
                }

                l[i] = rescale * l[i] + l_new;

                for d in 0..dv {
                    let old_o = o_tile[i * dv + d];
                    let mut acc = rescale * old_o;
                    for j in 0..bc {
                        acc += p_row[j] * v_tile[j * dv + d];
                    }
                    o_tile[i * dv + d] = acc;
                }

                m[i] = m_new;
            }
        }

        for i in 0..br {
            let l_i = l[i];
            if l_i > 0.0 {
                for d in 0..dv {
                    o_tile[i * dv + d] /= l_i;
                }
            }
        }

        for i in 0..br {
            let dst = &mut o[(q_start + i) * dv..(q_start + i + 1) * dv];
            let src = &o_tile[i * dv..(i + 1) * dv];
            dst.copy_from_slice(src);
        }
    }
}

fn matmul_tile(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, scale: f32) -> Vec<f32> {
    let mut s = vec![0.0f32; m * n];

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                matmul_tile_avx512(a, b, &mut s, m, k, n, scale);
            }
            return s;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                matmul_tile_avx2(a, b, &mut s, m, k, n, scale);
            }
            return s;
        }
    }

    for i in 0..m {
        for j in 0..n {
            let mut dot = 0.0;
            for kk in 0..k {
                dot += a[i * k + kk] * b[j * k + kk];
            }
            s[i * n + j] = dot * scale;
        }
    }
    s
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_tile_avx2(
    a: &[f32],
    b: &[f32],
    s: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let a_row = &a[i * k..];
            let b_row = &b[j * k..];
            let mut acc = _mm256_setzero_ps();
            let mut kk = 0;

            while kk + 32 <= k {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row.as_ptr().add(kk)),
                    _mm256_loadu_ps(b_row.as_ptr().add(kk)),
                    acc,
                );
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row.as_ptr().add(kk + 8)),
                    _mm256_loadu_ps(b_row.as_ptr().add(kk + 8)),
                    acc,
                );
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row.as_ptr().add(kk + 16)),
                    _mm256_loadu_ps(b_row.as_ptr().add(kk + 16)),
                    acc,
                );
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row.as_ptr().add(kk + 24)),
                    _mm256_loadu_ps(b_row.as_ptr().add(kk + 24)),
                    acc,
                );
                kk += 32;
            }
            while kk + 8 <= k {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a_row.as_ptr().add(kk)),
                    _mm256_loadu_ps(b_row.as_ptr().add(kk)),
                    acc,
                );
                kk += 8;
            }

            let mut dot = hsum256_ps(acc);
            while kk < k {
                dot += *a_row.as_ptr().add(kk) * *b_row.as_ptr().add(kk);
                kk += 1;
            }
            s[i * n + j] = dot * scale;
        }
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(v, 1);
    let lo128 = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_shuffle_ps(sum128, sum128, 0x0E);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 0x01);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn matmul_tile_avx512(
    a: &[f32],
    b: &[f32],
    s: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    scale: f32,
) {
    for i in 0..m {
        for j in 0..n {
            let a_row = &a[i * k..];
            let b_row = &b[j * k..];
            let mut acc = _mm512_setzero_ps();
            let mut kk = 0;

            while kk + 64 <= k {
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a_row.as_ptr().add(kk)),
                    _mm512_loadu_ps(b_row.as_ptr().add(kk)),
                    acc,
                );
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a_row.as_ptr().add(kk + 16)),
                    _mm512_loadu_ps(b_row.as_ptr().add(kk + 16)),
                    acc,
                );
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a_row.as_ptr().add(kk + 32)),
                    _mm512_loadu_ps(b_row.as_ptr().add(kk + 32)),
                    acc,
                );
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a_row.as_ptr().add(kk + 48)),
                    _mm512_loadu_ps(b_row.as_ptr().add(kk + 48)),
                    acc,
                );
                kk += 64;
            }
            while kk + 16 <= k {
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(a_row.as_ptr().add(kk)),
                    _mm512_loadu_ps(b_row.as_ptr().add(kk)),
                    acc,
                );
                kk += 16;
            }

            let mut dot = _mm512_reduce_add_ps(acc);
            while kk < k {
                dot += *a_row.as_ptr().add(kk) * *b_row.as_ptr().add(kk);
                kk += 1;
            }
            s[i * n + j] = dot * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_manual_tiny() {
        // Tiny case: [1, 1, 2, 2] tensors
        // Q = [[1, 2], [3, 4]], K = [[5, 6], [7, 8]], V = [[9, 10], [11, 12]]
        let q = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let k = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![1, 1, 2, 2]);
        let v = Tensor::from_vec(vec![9.0, 10.0, 11.0, 12.0], vec![1, 1, 2, 2]);

        let dk = 2.0f32;
        let scale = dk.sqrt().recip(); // 1/sqrt(2) ≈ 0.7071068

        // Manual S = Q @ K^T * scale
        // S[0,0] = (1*5 + 2*6) * scale = (5+12) * 0.7071 = 12.02
        // S[0,1] = (1*7 + 2*8) * scale = (7+16) * 0.7071 = 16.26
        // S[1,0] = (3*5 + 4*6) * scale = (15+24) * 0.7071 = 27.58
        // S[1,1] = (3*7 + 4*8) * scale = (21+32) * 0.7071 = 37.48
        let s00 = (1.0 * 5.0 + 2.0 * 6.0) * scale;
        let s01 = (1.0 * 7.0 + 2.0 * 8.0) * scale;
        let s10 = (3.0 * 5.0 + 4.0 * 6.0) * scale;
        let s11 = (3.0 * 7.0 + 4.0 * 8.0) * scale;

        // Manual softmax(row)
        // Row 0: exp(S[0,0]-max0), exp(S[0,1]-max0) where max0 = max(12.02, 16.26) = 16.26
        // P[0,0] = exp(12.02-16.26) = exp(-4.24) ≈ 0.0144
        // P[0,1] = exp(16.26-16.26) = exp(0) = 1.0
        // sum0 = 1.0144
        // softmax[0,0] = 0.0142, softmax[0,1] = 0.9858
        let max0 = s00.max(s01);
        let p00 = (s00 - max0).exp();
        let p01 = (s01 - max0).exp();
        let sum0 = p00 + p01;
        let sm00 = p00 / sum0;
        let sm01 = p01 / sum0;

        // Row 1: max1 = max(27.58, 37.48) = 37.48
        // P[1,0] = exp(27.58-37.48) = exp(-9.9) ≈ 0.00005
        // P[1,1] = exp(37.48-37.48) = 1.0
        // sum1 = 1.00005
        let max1 = s10.max(s11);
        let p10 = (s10 - max1).exp();
        let p11 = (s11 - max1).exp();
        let sum1 = p10 + p11;
        let sm10 = p10 / sum1;
        let sm11 = p11 / sum1;

        // Output = softmax(S) @ V
        // O[0,0] = sm00*9 + sm01*11
        // O[0,1] = sm00*10 + sm01*12
        // O[1,0] = sm10*9 + sm11*11
        // O[1,1] = sm10*10 + sm11*12
        let manual_o00 = sm00 * 9.0 + sm01 * 11.0;
        let manual_o01 = sm00 * 10.0 + sm01 * 12.0;
        let manual_o10 = sm10 * 9.0 + sm11 * 11.0;
        let manual_o11 = sm10 * 10.0 + sm11 * 12.0;

        // Now compute using tiled implementation
        let tiled = flash_attention(&q, &k, &v, None, false);
        let tiled_data = tiled.as_f32_slice();

        let eps = 1e-3;
        assert!(
            (tiled_data[0] - manual_o00).abs() < eps,
            "O[0,0]: tiled={}, manual={}",
            tiled_data[0],
            manual_o00
        );
        assert!(
            (tiled_data[1] - manual_o01).abs() < eps,
            "O[0,1]: tiled={}, manual={}",
            tiled_data[1],
            manual_o01
        );
        assert!(
            (tiled_data[2] - manual_o10).abs() < eps,
            "O[1,0]: tiled={}, manual={}",
            tiled_data[2],
            manual_o10
        );
        assert!(
            (tiled_data[3] - manual_o11).abs() < eps,
            "O[1,1]: tiled={}, manual={}",
            tiled_data[3],
            manual_o11
        );
    }

    #[test]
    fn test_flash_attention_causal_small() {
        // Verify causal mask: output[i,d] should depend only on K/V positions <= i
        let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0], vec![1, 1, 3, 2]);
        let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0], vec![1, 1, 3, 2]);
        let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 1, 3, 2]);

        // Seq.len = 3, TILE_SIZE = 32, so only 1 tile.
        // Non-causal: full attention
        let tiled = flash_attention(&q, &k, &v, None, false);
        // Causal: each position only attends to <= itself
        let tiled_causal = flash_attention(&q, &k, &v, None, true);

        let data = tiled.as_f32_slice();
        let causal_data = tiled_causal.as_f32_slice();

        // All values should be finite
        for v in data {
            assert!(v.is_finite());
        }
        for v in causal_data {
            assert!(v.is_finite());
        }

        // Causal result should differ from non-causal (except row 0, which has same mask)
        let mut differs = false;
        for i in 1..(data.len().min(causal_data.len())) {
            if (data[i] - causal_data[i]).abs() > 1e-4 {
                differs = true;
                break;
            }
        }
        assert!(differs, "Causal and non-causal should differ");
    }

    #[test]
    fn test_flash_attention_tiled_self_consistency() {
        let batch = 1;
        let heads = 2;
        let seq_len = 64;
        let dk = 32;
        let dv = 32;

        let q = Tensor::from_vec(
            (0..batch * heads * seq_len * dk)
                .map(|i| (i as f32) * 0.01)
                .collect(),
            vec![batch, heads, seq_len, dk],
        );
        let k = Tensor::from_vec(
            (0..batch * heads * seq_len * dk)
                .map(|i| (i as f32) * 0.01 + 0.3)
                .collect(),
            vec![batch, heads, seq_len, dk],
        );
        let v = Tensor::from_vec(
            (0..batch * heads * seq_len * dv)
                .map(|i| (i as f32) * 0.01 + 0.6)
                .collect(),
            vec![batch, heads, seq_len, dv],
        );

        // Both causal and non-causal should produce finite outputs
        let tiled = flash_attention(&q, &k, &v, None, false);
        let tiled_causal = flash_attention(&q, &k, &v, None, true);

        for val in tiled.as_f32_slice() {
            assert!(val.is_finite(), "Non-causal output has non-finite value");
        }
        for val in tiled_causal.as_f32_slice() {
            assert!(val.is_finite(), "Causal output has non-finite value");
        }
    }
}
