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
where = _core.where_
repeat = _core.repeat
expand = _core.expand
fnn_slice = _core.slice
topk = _core.topk
gather = _core.gather

einsum = _core.einsum
flash_attention = _core.flash_attention

mse_loss = _core.mse_loss
cross_entropy_loss = _core.cross_entropy_loss
bce_with_logits = _core.bce_with_logits
huber_loss = _core.huber_loss
gather = _core.gather
where_ = _core.where_
repeat = _core.repeat

__all__ = [
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
    "where",
    "repeat",
    "expand",
    "fnn_slice",
    "topk",
    "gather",
    "einsum",
    "flash_attention",
    "mse_loss",
    "cross_entropy_loss",
    "bce_with_logits",
    "huber_loss",
    "gather",
    "where_",
    "repeat",
]
