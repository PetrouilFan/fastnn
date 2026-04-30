"""Tensor operations exported by the Rust core."""

import fastnn._core as _core
from fastnn.core import maximum, minimum

add = _core.add
sub = _core.sub
mul = _core.mul
div = _core.div
matmul = _core.matmul
im2col = _core.im2col
neg = _core.neg
abs = _core.abs
exp = _core.exp
log = _core.log
sqrt = _core.sqrt
pow = _core.pow
clamp = _core.clamp
relu = _core.relu
fused_add_relu = _core.fused_add_relu
fused_linear_relu = _core.fused_linear_relu
fused_linear_gelu = _core.fused_linear_gelu
fused_conv_bn_silu = _core.fused_conv_bn_silu
gelu = _core.gelu
sigmoid = _core.sigmoid
tanh = _core.tanh
silu = _core.silu
softmax = _core.softmax
log_softmax = _core.log_softmax
argmax = _core.argmax
argmin = _core.argmin
cat = _core.cat
stack = _core.stack
sum = _core.sum
mean = _core.mean
max = _core.max
min = _core.min
einsum = _core.einsum
flash_attention = _core.flash_attention
clip_grad_norm_ = _core.clip_grad_norm_
clip_grad_value_ = _core.clip_grad_value_

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "im2col",
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "pow",
    "clamp",
    "relu",
    "fused_add_relu",
    "fused_linear_relu",
    "fused_linear_gelu",
    "fused_conv_bn_silu",
    "gelu",
    "sigmoid",
    "tanh",
    "silu",
    "softmax",
    "log_softmax",
    "argmax",
    "argmin",
    "cat",
    "stack",
    "sum",
    "mean",
    "max",
    "min",
    "maximum",
    "minimum",
    "einsum",
    "flash_attention",
    "clip_grad_norm_",
    "clip_grad_value_",
]

