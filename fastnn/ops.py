"""Operations module (re-exports from core)."""

import fastnn._core as _core

relu = _core.relu
gelu = _core.gelu
sigmoid = _core.sigmoid
tanh = _core.tanh
silu = _core.silu
leaky_relu = _core.leaky_relu
elu = _core.elu
softplus = _core.softplus
hardswish = _core.hardswish
softmax = _core.softmax
log_softmax = _core.log_softmax
fused_add_relu = _core.fused_add_relu
fused_conv_bn_silu = _core.fused_conv_bn_silu
fused_linear_relu = _core.fused_linear_relu
fused_linear_gelu = _core.fused_linear_gelu

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
argmax = _core.argmax
argmin = _core.argmin
cat = _core.cat
stack = _core.stack
sum = _core.sum
mean = _core.mean
max = _core.max
min = _core.min
maximum = _core.maximum
minimum = _core.minimum
# _core.where_ is the functional (non-in-place) version — naming avoids
# shadowing Python's built-in `where`. Both aliases produce a new tensor.
where = _core.where_
where_ = _core.where_  # alias kept for explicit in-place-style naming
repeat = _core.repeat
expand = _core.expand
fnn_slice = _core.slice
topk = _core.topk
gather = _core.gather
einsum = _core.einsum
flash_attention = _core.flash_attention
cumsum = _core.cumsum
erf = _core.erf

mse_loss = _core.mse_loss
cross_entropy_loss = _core.cross_entropy_loss
bce_with_logits = _core.bce_with_logits
huber_loss = _core.huber_loss

__all__ = [
    # Activation functions
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "silu",
    "leaky_relu",
    "elu",
    "softplus",
    "hardswish",
    "softmax",
    "log_softmax",
    "fused_add_relu",
    "fused_conv_bn_silu",
    "fused_linear_relu",
    "fused_linear_gelu",
    # Arithmetic
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "neg",
    "abs",
    "exp",
    "log",
    "sqrt",
    "pow",
    "clamp",
    # Reduction
    "argmax",
    "argmin",
    "sum",
    "mean",
    "max",
    "min",
    "maximum",
    "minimum",
    "cumsum",
    # Tensor manipulation
    "cat",
    "stack",
    "where",
    "where_",
    "repeat",
    "expand",
    "fnn_slice",
    "topk",
    "gather",
    "im2col",
    # Special
    "einsum",
    "flash_attention",
    "erf",
    # Loss functions
    "mse_loss",
    "cross_entropy_loss",
    "bce_with_logits",
    "huber_loss",
]
