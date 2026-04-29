import fastnn
import numpy as np
import time

# Set dimensions
batch = 1
in_ch = 32
out_ch = 64
height = width = 32
kernel_size = 3
stride_val = 1
padding_val = 1
dilation_val = 1
groups_val = 1
eps_val = 1e-5

# Generate random tensors (CPU)
x = fastnn.randn([batch, in_ch, height, width])
w = fastnn.randn([out_ch, in_ch, kernel_size, kernel_size])
b = fastnn.randn([out_ch])  # conv bias
bn_w = fastnn.randn([out_ch])
bn_b = fastnn.randn([out_ch])
bn_mean = fastnn.randn([out_ch])
bn_var = fastnn.ones([out_ch])  # variance close to 1

# Create scalar tensors for the remaining params
stride_t = fastnn.full([], stride_val)
padding_t = fastnn.full([], padding_val)
dilation_t = fastnn.full([], dilation_val)
groups_t = fastnn.full([], groups_val)
eps_t = fastnn.full([], eps_val)

# Warm-up
_ = fastnn._core.fused_conv_bn_silu(x, w, b, bn_w, bn_b, bn_mean, bn_var, stride_t, padding_t, dilation_t, groups_t, eps_t)

# Time fused operation
n_iter = 10
start = time.perf_counter()
for _ in range(n_iter):
    out_fused = fastnn._core.fused_conv_bn_silu(x, w, b, bn_w, bn_b, bn_mean, bn_var, stride_t, padding_t, dilation_t, groups_t, eps_t)
end = time.perf_counter()
fused_time = (end - start) / n_iter
print(f"Fused time: {fused_time * 1000:.3f} ms")

# Separate operations: conv2d -> batch norm -> silu
# Create Conv2d layer and set weights
conv = fastnn.Conv2d(in_ch, out_ch, kernel_size, stride=stride_val, padding=padding_val, bias=True)
conv.weight = w
conv.bias = b

# BatchNorm manual implementation
def batchnorm(x, weight, bias, running_mean, running_var, eps):
    C = weight.shape[0]
    w_view = weight.reshape([1, C, 1, 1])
    b_view = bias.reshape([1, C, 1, 1])
    mean_view = running_mean.reshape([1, C, 1, 1])
    var_view = running_var.reshape([1, C, 1, 1])
    inv_std = fastnn.pow(var_view + eps, -0.5)
    return (x - mean_view) * inv_std * w_view + b_view

# Warm-up
conv_out = conv(x)
bn_out = batchnorm(conv_out, bn_w, bn_b, bn_mean, bn_var, eps_val)
_ = fastnn.silu(bn_out)

# Time separate
start = time.perf_counter()
for _ in range(n_iter):
    conv_out = conv(x)
    bn_out = batchnorm(conv_out, bn_w, bn_b, bn_mean, bn_var, eps_val)
    out_sep = fastnn.silu(bn_out)
end = time.perf_counter()
sep_time = (end - start) / n_iter
print(f"Separate time: {sep_time * 1000:.3f} ms")

# Compare outputs
out_fused_np = out_fused.numpy()
out_sep_np = out_sep.numpy()
close = np.allclose(out_fused_np, out_sep_np, rtol=1e-4, atol=1e-4)
print(f"Outputs close: {close}")
if not close:
    max_diff = np.max(np.abs(out_fused_np - out_sep_np))
    print(f"Maximum absolute difference: {max_diff}")
