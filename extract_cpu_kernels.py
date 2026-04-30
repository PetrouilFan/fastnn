#!/usr/bin/env python3
"""
Find function bodies by scanning for fn declaration, then brace matching from first {.
"""

with open('src/kernels/cpu/mod.rs', 'r') as f:
    lines = f.readlines()

funcs = []
i = 0
while i < len(lines):
    line = lines[i]

    # Skip attributes
    if line.lstrip().startswith('#'):
        i += 1
        continue

    # Look for a function definition (fn name()
    if 'fn ' in line:
        # Extract the name after 'fn '
        parts = line.split('fn ', 1)
        if len(parts) < 2:
            i += 1
            continue
        after = parts[1].strip()
        # Name is before '('
        if '(' in after:
            name = after.split('(')[0].strip()
            # Must be a valid Rust identifier (no spaces, quotes, etc)
            if name and name.replace('_', '').isalnum() and ' ' not in name and '=' not in name:
                # Find opening brace
                brace_line = i
                while brace_line < len(lines) and '{' not in lines[brace_line]:
                    brace_line += 1
                if brace_line >= len(lines):
                    i += 1
                    continue

                # Find matching closing brace
                depth = 0
                end_line = brace_line
                for j in range(brace_line, len(lines)):
                    depth += lines[j].count('{')
                    depth -= lines[j].count('}')
                    if depth == 0:
                        end_line = j
                        break

                funcs.append((name, i, end_line))
                i = end_line + 1
                continue

    i += 1

print(f"Found {len(funcs)} functions")
for name, start, end in sorted(funcs, key=lambda x: x[1]):
    print(f"  {name}: lines {start+1}-{end+1}")

# Group
print("\nCategorizing...")
extract = {
    'simd': [],
    'elementwise': [],
    'matmul': [],
    'conv': [],
    'reductions': [],
    'losses': [],
    'factories': [],
    'pooling': [],
    'norm': [],
}

# Functions to keep in mod.rs (infrastructure)
keep_in_mod = {
    'AlignedBuffer', 'new', 'resize', 'as_mut_slice', 'len', 'split_at_mut',
    'enable_daz_ftz', 'ensure_daz_ftz', 'detect_simd_level',
    'create_output', 'broadcast_shapes_simple', 'broadcast_index_decomposition',
    'register_kernels', 'tests'
}

for name, start, end in funcs:
    # Keep these in mod.rs
    if name in keep_in_mod:
        continue

    # SIMD
    if ('_parallel_' in name or name.startswith(('fast_exp_', 'fast_log_')) or
        name in ['relu_simd', 'exp_simd', 'log_simd', 'sqrt_simd', 'gelu_simd',
                 'silu_simd', 'tanh_simd', 'sigmoid_simd', 'sigmoid_simd_x86',
                 'hsum256_ps', 'from_slice_unaligned_f32x8', 'from_slice_unaligned_f32x4',
                 'detect_simd_level']):
        extract['simd'].append((name, start, end))

    # Matmul
    elif name.startswith(('matmul_kernel', 'linear_kernel', 'fused_linear_',
                          'parallel_matmul', 'blocked_row_matmul', 'small_matrix_matmul',
                          'single_threaded_matmul', 'simd_dot_product')):
        extract['matmul'].append((name, start, end))

    # Conv
    elif name.startswith(('conv2d_kernel', 'conv1d_kernel', 'conv3d_kernel',
                          'conv_transpose2d_kernel', 'conv2d_', 'fused_conv_bn_silu',
                          'im2col_kernel', 'flash_attention_kernel', 'depthwise_conv2d')):
        extract['conv'].append((name, start, end))

    # Reductions
    elif name.startswith(('sum_kernel', 'mean_kernel', 'min_kernel', 'max_kernel',
                          'softmax_kernel', 'log_softmax_kernel',
                          'softmax_last_dim_simd', 'log_softmax_last_dim_fused')):
        extract['reductions'].append((name, start, end))

    # Losses
    elif name.startswith(('mse_loss_kernel', 'cross_entropy_loss_kernel',
                          'bce_with_logits_kernel', 'huber_loss_kernel')):
        extract['losses'].append((name, start, end))

    # Factories
    elif name.startswith(('zeros_kernel', 'ones_kernel', 'full_kernel',
                          'arange_kernel', 'linspace_kernel', 'eye_kernel',
                          'randn_kernel', 'rand_kernel', 'randint_kernel',
                          'read_f32', 'write_f32')):
        extract['factories'].append((name, start, end))

    # Pooling
    elif name.startswith('max_pool2d_kernel'):
        extract['pooling'].append((name, start, end))

    # Norm
    elif name.startswith(('layer_norm_kernel', 'batch_norm_kernel')):
        extract['norm'].append((name, start, end))

    # Elementwise (all other _kernel functions)
    elif name.endswith('_kernel') or name in ['leaky_relu_kernel', 'prelu_kernel',
                                               'softplus_kernel', 'hardswish_kernel',
                                               'elu_kernel', 'clamp_kernel', 'pow_kernel',
                                               'gt_scalar_kernel', 'sign_kernel',
                                               'maximum_kernel', 'minimum_kernel',
                                               'lt_scalar_kernel', 'add_scalar_kernel',
                                               'div_scalar_kernel', 'logical_not_kernel']:
        extract['elementwise'].append((name, start, end))

    else:
        print(f"UNCATEGORIZED: {name}")

# Expand extracted function ranges to include preceding attribute and doc comment lines
for mod_name in extract:
    items = extract[mod_name]
    expanded = []
    for name, start, end in items:
        new_start = start
        while new_start > 0:
            prev_line = lines[new_start - 1]
            stripped = prev_line.lstrip()
            if stripped.startswith('#') or stripped.startswith('///'):
                new_start -= 1
            else:
                break
        expanded.append((name, new_start, end))
    extract[mod_name] = expanded

total = sum(len(v) for v in extract.values())
print(f"\nTotal functions to extract: {total}")

for mod, items in sorted(extract.items()):
    if items:
        print(f"  {mod}: {len(items)}")

# ============================================================
# Prepare removal of extracted functions (including preceding attributes/doc comments)
# ============================================================

# Expand extracted function ranges to include preceding attribute and doc comment lines
expanded_items = []
for items in extract.values():
    for name, start, end in items:
        new_start = start
        # Backtrack while previous line is an attribute or a doc comment
        while new_start > 0:
            prev_line = lines[new_start - 1]
            stripped = prev_line.lstrip()
            if stripped.startswith('#') or stripped.startswith('///'):
                new_start -= 1
            else:
                break
        expanded_items.append((new_start, end))

# Build set of line indices to remove
remove_lines = set()
for start, end in expanded_items:
    for ln in range(start, end + 1):
        remove_lines.add(ln)

# Write new mod.rs
new_mod = [lines[i] for i in range(len(lines)) if i not in remove_lines]
with open('src/kernels/cpu/mod.rs', 'w') as f:
    f.writelines(new_mod)

print(f"\nmod.rs: {len(new_mod)} lines (was {len(lines)})")

# ============================================================
# Write extracted modules
# ============================================================

module_header = '''#![allow(unused_imports)]

use crate::autograd::{AutogradMeta, Edge, Node};
use crate::dispatcher::{register, DispatchKey, KernelFn};
use crate::iterator::TensorIterator;
use crate::kernels::blas::{
    matmul_blas, matmul_blas_into, matmul_blas_with_transpose,
    matmul_blas_with_transpose_into, MIN_BLAS_SIZE,
};
use crate::storage::{DType, Device, Storage};
use crate::tensor::Tensor;
use std::sync::Arc;
use super::*;

'''

for mod_name, items in extract.items():
    if not items:
        continue

    items.sort(key=lambda x: x[1])

    content = f'//! CPU {mod_name} kernels.\n\n'
    content += module_header

    if mod_name == 'simd':
        content += 'use wide::f32x4;\n'
        content += '#[cfg(all(feature = "simd", any(target_arch = "x86_64", target_arch = "x86")))]\n'
        content += 'use wide::f32x8;\n\n'

    for name, start, end in items:
        chunk_lines = lines[start:end+1]
        # Add 'pub' before the function definition if not already present.
        # Handle lines that may start with 'pub' already, or with 'unsafe' or directly 'fn'.
        for idx, line in enumerate(chunk_lines):
            stripped = line.lstrip()
            # Skip lines that are attributes or empty
            if stripped.startswith('#') or stripped == '':
                continue
            # Check if this is the function definition line
            if stripped.startswith('pub '):
                # already public, nothing to do
                break
            if stripped.startswith('unsafe ') or stripped.startswith('fn '):
                indent = len(line) - len(stripped)
                chunk_lines[idx] = line[:indent] + 'pub ' + stripped
                break
        chunk = ''.join(chunk_lines)
        content += chunk + '\n'

    with open(f'src/kernels/cpu/{mod_name}.rs', 'w') as f:
        f.write(content)

    print(f"✓ {mod_name}.rs: {len(items)} functions")

print("\nExtraction complete. Next: cargo check")
