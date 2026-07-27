#!/usr/bin/env bash
set -euo pipefail

run_miri_test() {
    cargo +nightly miri test --lib "$1" -- --exact --test-threads=1
}

run_miri_test packed_tensor::tests::packed_tensor_capacity_matches_canonical_layout
run_miri_test storage::dtype_tests::aligned_vec_is_zeroed_and_cache_line_aligned
run_miri_test backend::cpu::cpu_buffer_tests::mutable_byte_view_preserves_non_word_aligned_lengths
run_miri_test backend::cpu::arena::tests::disjoint_nary_input_output_direct_helper_result
