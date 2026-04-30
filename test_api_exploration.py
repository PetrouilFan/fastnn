"""fastnn v1.1.0 — tested working snippets for every requested operation."""
import fastnn as fnn
import numpy as np

print("=" * 60)
print("FASTNN v1.1.0 API EXPLORATION — WORKING SNIPPETS")
print("=" * 60)

# ------------------------------------------------------------------
# 1. Linear layer (weight/bias, forward)
# ------------------------------------------------------------------
print("\n## 1. Linear layer")
# Linear(in_features, out_features)
linear = fnn.Linear(4, 3)
x = fnn.tensor([[1., 2., 3., 4.],
                [5., 6., 7., 8.]], [2, 4])
y = linear(x)
# parameters() returns [weight, bias]  (weight shape is [in, out])
params = linear.parameters()
print(f"  weight shape : {params[0].shape}")
print(f"  bias shape   : {params[1].shape}")
print(f"  input shape  : {x.shape}")
print(f"  output shape : {y.shape}")
print(f"  output:\n{y.numpy()}")

# ------------------------------------------------------------------
# 2. Matmul (tensor, dim handling)
# ------------------------------------------------------------------
print("\n## 2. matmul")
a = fnn.tensor([[1., 2.], [3., 4.]], [2, 2])
b = fnn.tensor([[5., 6.], [7., 8.]], [2, 2])
c = fnn.matmul(a, b)            # or a @ b
print(f"  a @ b shape: {c.shape}")
print(f"  result:\n{c.numpy()}")

# Batch matmul – fastnn supports arbitrary batch dims via broadcasting
A = fnn.ones([2, 3, 4])
B = fnn.ones([2, 4, 5])
C = fnn.matmul(A, B)
print(f"  batch A{B.shape} @ B{B.shape} -> C{C.shape}")

# ------------------------------------------------------------------
# 3. RMSNorm & LayerNorm
# ------------------------------------------------------------------
print("\n## 3. RMSNorm & LayerNorm")
# RMSNorm / LayerNorm from the core work most reliably with 3-D tensors
# (e.g. [batch, seq, hidden]).  2-D inputs may hit a broadcast issue.
x_norm = fnn.tensor([[[1., 2., 3.],
                      [4., 5., 6.]],
                     [[7., 8., 9.],
                      [10., 11., 12.]]], [2, 2, 3])

rms = fnn.RMSNorm(3)   # normalizes over the last dim
y_rms = rms(x_norm)

ln = fnn.LayerNorm(3)
y_ln = ln(x_norm)

print(f"  input shape  : {x_norm.shape}")
print(f"  RMSNorm out  : {y_rms.shape}\n{y_rms.numpy()}")
print(f"  LayerNorm out: {y_ln.shape}\n{y_ln.numpy()}")

# ------------------------------------------------------------------
# 4. Element-wise operations (mul, add, div)
# ------------------------------------------------------------------
print("\n## 4. Element-wise mul / add / div")
a = fnn.tensor([2., 4., 6.], [3])
b = fnn.tensor([2., 2., 2.], [3])
print(f"  a * b = {(a * b).numpy()}")
print(f"  a + b = {(a + b).numpy()}")
print(f"  a / b = {(a / b).numpy()}")
# Scalar multiplication – use fnn.mul with a full tensor
three = fnn.full(a.shape, 3.0)
print(f"  a * 3 = {fnn.mul(a, three).numpy()}")
# Or use the operator with a rank-matching tensor (no Python-int broadcast)

# ------------------------------------------------------------------
# 5. softmax, tanh, silu
# ------------------------------------------------------------------
print("\n## 5. softmax, tanh, silu")
x_act = fnn.tensor([[1., 2., 3.]], [1, 3])
s = fnn.softmax(x_act, dim=-1)
t = fnn.tanh(x_act)
si = fnn.silu(x_act)
print(f"  softmax dim=-1  shape {s.shape}:\n{s.numpy()}")
print(f"  tanh            shape {t.shape}:\n{t.numpy()}")
print(f"  silu            shape {si.shape}:\n{si.numpy()}")

# ------------------------------------------------------------------
# 6. mean / sum with keepdim
# ------------------------------------------------------------------
print("\n## 6. mean / sum with keepdim")
x_red = fnn.ones([2, 2, 3])
print(f"  input shape                : {x_red.shape}")
print(f"  sum  dim=2 keepdim=True    : {fnn.sum(x_red, dim=2, keepdim=True).shape}")
print(f"  sum  dim=1 keepdim=True    : {fnn.sum(x_red, dim=1, keepdim=True).shape}")
print(f"  sum  dim=0 keepdim=True    : {fnn.sum(x_red, dim=0, keepdim=True).shape}")
print(f"  mean dim=2 keepdim=True    : {fnn.mean(x_red, dim=2, keepdim=True).shape}")
print(f"  mean dim=1 keepdim=False   : {fnn.mean(x_red, dim=1, keepdim=False).shape}")
# Reduce-all (flatten to 1-D then sum over that axis)
total = fnn.sum(x_red.reshape([-1]), dim=0)
print(f"  sum all  (flatten)         : {total.shape}  value={total.numpy()}")

# ------------------------------------------------------------------
# 7. reshape / transpose
# ------------------------------------------------------------------
print("\n## 7. reshape / transpose")
x_sh = fnn.zeros([2, 3, 4])
r = x_sh.reshape([6, 4])
v = x_sh.view([6, 4])
tr = x_sh.transpose(0, 2)      # dims 0 & 2 swapped
print(f"  original shape      : {x_sh.shape}")
print(f"  reshape([6,4])      : {r.shape}")
print(f"  view([6,4])          : {v.shape}")
print(f"  transpose(0,2)       : {tr.shape}")

# ------------------------------------------------------------------
# 8. cat
# ------------------------------------------------------------------
print("\n## 8. cat")
a_c = fnn.ones([2, 3])
b_c = fnn.zeros([2, 3])
c0 = fnn.cat([a_c, b_c], dim=0)
c1 = fnn.cat([a_c, b_c], dim=1)
print(f"  a shape        : {a_c.shape}")
print(f"  b shape        : {b_c.shape}")
print(f"  cat dim=0      : {c0.shape}")
print(f"  cat dim=1      : {c1.shape}")

# ------------------------------------------------------------------
# 9. Read tensor shape / dtype / numel
# ------------------------------------------------------------------
print("\n## 9. Read tensor shape / dtype / numel")
x_info = fnn.tensor([1., 2., 3., 4.], [2, 2])
print(f"  x.shape  = {x_info.shape}")
print(f"  x.dtype  = {x_info.dtype}")
print(f"  x.numel  = {x_info.numel}")

# ------------------------------------------------------------------
# 10. Create tensor from list / shape
# ------------------------------------------------------------------
print("\n## 10. Create tensor from list / shape")
t_list = fnn.tensor([1., 2., 3., 4.], [2, 2])
t_zeros = fnn.zeros([2, 3])
t_ones  = fnn.ones([3])
t_full  = fnn.full([2, 2], 7.0)
t_rand  = fnn.rand([2, 2])
t_randn = fnn.randn([2, 2])
print(f"  tensor(list, shape)  : {t_list.numpy()}")
print(f"  zeros([2,3])         : {t_zeros.numpy()}")
print(f"  ones([3])            : {t_ones.numpy()}")
print(f"  full([2,2], 7.0)     : {t_full.numpy()}")
print(f"  rand([2,2]) shape    : {t_rand.shape}")
print(f"  randn([2,2]) shape   : {t_randn.shape}")

print("\n" + "=" * 60)
print("ALL SNIPPETS EXECUTED SUCCESSFULLY")
print("=" * 60)
