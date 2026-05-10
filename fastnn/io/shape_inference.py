"""Shape inference for ONNX models.

Provides shape propagation rules for each ONNX operator type.
Each rule takes input shapes and attributes and returns output shapes.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def infer_shape(
    op_type: str,
    input_shapes: List[List[int]],
    attrs: Dict[str, Any],
    num_outputs: int = 1,
) -> List[List[int]]:
    """Infer output shapes given input shapes and attributes.

    Args:
        op_type: The ONNX operator type (case-insensitive on first char).
        input_shapes: List of input shapes (each is a list of dims).
        attrs: Operator attributes dict.
        num_outputs: Expected number of outputs.

    Returns:
        List of output shapes (one per output).
    """
    op_lower = op_type.lower()
    
    # Dispatch to specific handler based on op_type
    handler_name = f"_infer_{op_lower}"
    handler = globals().get(handler_name)
    if handler is not None:
        try:
            return handler(input_shapes, attrs, num_outputs)
        except Exception as e:
            logger.warning("Shape inference error for %s: %s", op_type, e)
            return _fallback(input_shapes, num_outputs)
    
    return _fallback(input_shapes, num_outputs)


def _fallback(input_shapes, num_outputs):
    """Fallback: return input shapes as output shapes."""
    if input_shapes:
        return [input_shapes[0]] * num_outputs
    return [[1]] * num_outputs


# ---- Activation ops (shape-preserving) ----

def _infer_relu(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_sigmoid(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_tanh(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_silu(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_gelu(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_leakyrelu(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_elu(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_softmax(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_hardswish(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_softplus(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_prelu(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_dropout(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_identity(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_abs(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_neg(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_exp(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_log(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_sqrt(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_erf(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_clip(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_cast(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_shape(input_shapes, attrs, num_outputs):
    """Shape op returns a 1-D tensor containing the shape."""
    if input_shapes:
        return [[len(input_shapes[0])]]
    return [[1]]

def _infer_shapeop(input_shapes, attrs, num_outputs):
    return _infer_shape(input_shapes, attrs, num_outputs)

def _infer_castop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

# ---- Arithmetic ops (broadcasting) ----

def _infer_add(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_sub(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_mul(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_div(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_pow(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_elementwiseadd(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_elementwisesub(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_elementwisemul(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_elementwisediv(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_elementwisepow(input_shapes, attrs, num_outputs):
    return [_broadcast_shapes(input_shapes[0], input_shapes[1])]

def _infer_biasadd(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_biassub(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_matmul(input_shapes, attrs, num_outputs):
    """Matrix multiply: (M, K) @ (K, N) -> (M, N), batched variant."""
    a = input_shapes[0]
    b = input_shapes[1]
    if len(a) == 1 and len(b) == 1:
        # Both 1-D: dot product, scalar
        return [[1]]
    elif len(a) == 2 and len(b) == 2:
        # 2-D: (M, K) @ (K, N) -> (M, N)
        return [[a[0], b[1]]]
    elif len(a) == 1 and len(b) == 2:
        # 1-D @ 2-D: (K,) @ (K, N) -> (N,)
        return [[b[1]]]
    elif len(a) == 2 and len(b) == 1:
        # 2-D @ 1-D: (M, K) @ (K,) -> (M,)
        return [[a[0]]]
    else:
        # Batched: (..., M, K) @ (..., K, N) -> (..., M, N)
        batch_a = a[:-2]
        batch_b = b[:-2]
        batch_out = _broadcast_shapes(batch_a, batch_b)
        return [batch_out + [a[-2], b[-1]]]


# ---- NN layer ops ----

def _infer_conv(input_shapes, attrs, num_outputs):
    """Conv: (N, C, H, W) + weight -> (N, out_channels, H_out, W_out)."""
    x_shape = input_shapes[0]
    w_shape = input_shapes[1] if len(input_shapes) > 1 else None
    
    out_channels = attrs.get("out_channels", w_shape[0] if w_shape else x_shape[0])
    kernel = attrs.get("kernel_size", attrs.get("kernel_shape", 1))
    if isinstance(kernel, (list, tuple)):
        kernel_h, kernel_w = kernel[0], kernel[-1]
    else:
        kernel_h = kernel_w = kernel
    
    stride = attrs.get("stride", attrs.get("strides", 1))
    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride[0], stride[-1]
    else:
        stride_h = stride_w = stride
    
    padding = attrs.get("padding", attrs.get("pads", 0))
    if isinstance(padding, (list, tuple)):
        if len(padding) == 4:
            pad_h = padding[0] + padding[2]
            pad_w = padding[1] + padding[3]
        else:
            pad_h = pad_w = padding[0]
    else:
        pad_h = pad_w = padding * 2
    
    dilation = attrs.get("dilation", attrs.get("dilations", 1))
    if isinstance(dilation, (list, tuple)):
        dilation_h, dilation_w = dilation[0], dilation[-1]
    else:
        dilation_h = dilation_w = dilation
    
    h = x_shape[2]
    w = x_shape[3]
    h_out = (h + pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    w_out = (w + pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    return [[x_shape[0], out_channels, h_out, w_out]]

def _infer_gemm(input_shapes, attrs, num_outputs):
    """Gemm: (M, K) @ (K, N) -> (M, N)."""
    x_shape = input_shapes[0]
    w_shape = input_shapes[1] if len(input_shapes) > 1 else None
    
    out_features = attrs.get("out_features")
    if out_features is not None:
        return [[x_shape[0], out_features]]
    
    in_features = attrs.get("in_features")
    if in_features is not None:
        out_features = attrs.get("out_features", w_shape[1] if w_shape else x_shape[-1])
        return [[x_shape[0], out_features]]
    
    if w_shape is None:
        return [[x_shape[0], x_shape[-1]]]
    
    trans_b = attrs.get("transB", attrs.get("trans_b", 0))
    if trans_b:
        out_features = w_shape[0]
    else:
        out_features = w_shape[1]
    
    return [[x_shape[0], out_features]]

def _infer_linear(input_shapes, attrs, num_outputs):
    return _infer_gemm(input_shapes, attrs, num_outputs)

def _infer_batchnormalization(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_batchnorm2d(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_batchnorm1d(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_instancenormalization(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_instancenorm(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_maxpool(input_shapes, attrs, num_outputs):
    """MaxPool: (N, C, H, W) -> (N, C, H_out, W_out)."""
    return [_pool_output_shape(input_shapes[0], attrs)]

def _infer_averagepool(input_shapes, attrs, num_outputs):
    return [_pool_output_shape(input_shapes[0], attrs)]

def _infer_avgpool(input_shapes, attrs, num_outputs):
    return [_pool_output_shape(input_shapes[0], attrs)]

def _infer_globalaveragepool(input_shapes, attrs, num_outputs):
    """GlobalAveragePool reduces spatial dims to 1x1."""
    x_shape = input_shapes[0]
    return [[x_shape[0], x_shape[1], 1, 1]]

def _infer_globalavgpool(input_shapes, attrs, num_outputs):
    return _infer_globalaveragepool(input_shapes, attrs, num_outputs)

def _pool_output_shape(x_shape, attrs):
    """Compute output shape for pooling ops."""
    kernel = attrs.get("kernel_size", attrs.get("kernel_shape", 2))
    if isinstance(kernel, (list, tuple)):
        kernel_h, kernel_w = kernel[0], kernel[-1]
    else:
        kernel_h = kernel_w = kernel
    
    stride = attrs.get("stride", attrs.get("strides", kernel))
    if isinstance(stride, (list, tuple)):
        stride_h, stride_w = stride[0], stride[-1]
    else:
        stride_h = stride_w = stride
    
    padding = attrs.get("padding", attrs.get("pads", 0))
    if isinstance(padding, (list, tuple)):
        pad_h = padding[0] if len(padding) > 0 else padding
        pad_w = padding[1] if len(padding) > 1 else padding
    else:
        pad_h = pad_w = padding
    
    h = x_shape[2]
    w = x_shape[3]
    h_out = (h + 2 * pad_h - kernel_h) // stride_h + 1
    w_out = (w + 2 * pad_w - kernel_w) // stride_w + 1
    
    return [x_shape[0], x_shape[1], h_out, w_out]


# ---- Tensor shape manipulation ----

def _infer_reshape(input_shapes, attrs, num_outputs):
    """Reshape: output shape from attrs or second input tensor."""
    shape_attr = attrs.get("shape")
    if shape_attr:
        shape = list(shape_attr)
        # Handle -1 (infer from total size)
        total = 1
        for s in input_shapes[0]:
            total *= s
        neg_one_idx = None
        for i, s in enumerate(shape):
            if s == -1:
                neg_one_idx = i
            else:
                total //= s
        if neg_one_idx is not None:
            shape[neg_one_idx] = total
        return [shape]
    return [input_shapes[0]]

def _infer_flatten(input_shapes, attrs, num_outputs):
    """Flatten: (N, C, H, W) -> (N, C*H*W) or custom axis."""
    x_shape = input_shapes[0]
    axis = attrs.get("axis", 1)
    outer = int(x_shape[0])
    inner = 1
    for d in x_shape[1:]:
        inner *= d
    return [[outer, inner]]

def _infer_transpose(input_shapes, attrs, num_outputs):
    """Transpose output shape from perm attribute."""
    x_shape = input_shapes[0]
    perm = attrs.get("perm", None)
    if perm:
        return [[x_shape[p] for p in perm]]
    # Default: reverse
    return [list(reversed(x_shape))]

def _infer_concat(input_shapes, attrs, num_outputs):
    """Concat along axis: sum the axis dim, keep others."""
    axis = attrs.get("axis", 1)
    out_shape = list(input_shapes[0])
    for s in input_shapes[1:]:
        out_shape[axis] += s[axis]
    return [out_shape]

def _infer_squeeze(input_shapes, attrs, num_outputs):
    """Squeeze: remove dims of size 1."""
    x_shape = input_shapes[0]
    axes = attrs.get("axes", None)
    if axes:
        out = [d for i, d in enumerate(x_shape) if i not in axes]
    else:
        out = [d for d in x_shape if d != 1]
    if not out:
        out = [1]
    return [out]

def _infer_unsqueeze(input_shapes, attrs, num_outputs):
    """Unsqueeze: insert dims of size 1 at given axes."""
    x_shape = input_shapes[0]
    axes = attrs.get("axes", [0])
    if isinstance(axes, int):
        axes = [axes]
    out = list(x_shape)
    for ax in sorted(axes):
        out.insert(ax, 1)
    return [out]

def _infer_split(input_shapes, attrs, num_outputs):
    """Split: divide along axis into chunks."""
    x_shape = input_shapes[0]
    axis = attrs.get("axis", 0)
    split = attrs.get("split", None)
    if split:
        outputs = []
        for s in split:
            out = list(x_shape)
            out[axis] = s
            outputs.append(out)
        return outputs
    else:
        part = x_shape[axis] // num_outputs
        out = list(x_shape)
        out[axis] = part
        return [out] * num_outputs

def _infer_slice(input_shapes, attrs, num_outputs):
    """Slice output shape from starts/ends/axes."""
    x_shape = input_shapes[0]
    starts = attrs.get("starts", [0])
    ends = attrs.get("ends", [x_shape[0]])
    axes = attrs.get("axes", list(range(len(starts))))
    steps = attrs.get("steps", [1] * len(starts))
    
    out = list(x_shape)
    for ax, st, en, step in zip(axes, starts, ends, steps):
        dim = out[ax]
        if st < 0:
            st = dim + st
        if en < 0:
            en = dim + en
        st = max(0, min(st, dim))
        en = max(st, min(en, dim))
        out[ax] = (en - st + step - 1) // step
    return [out]

def _infer_pad(input_shapes, attrs, num_outputs):
    """Pad: increase spatial dims by padding amounts."""
    x_shape = input_shapes[0]
    pads = attrs.get("pads", [0, 0, 0, 0, 0, 0, 0, 0])
    if len(pads) >= 8:
        out = list(x_shape)
        for i in range(4):
            out[i] += pads[i] + pads[i + 4]
        return [out]
    return [x_shape]

def _infer_tile(input_shapes, attrs, num_outputs):
    """Tile: repeat each dim by given factor."""
    x_shape = input_shapes[0]
    repeats = attrs.get("repeats", [1] * len(x_shape))
    out = [d * r for d, r in zip(x_shape, repeats)]
    return [out]

def _infer_expand(input_shapes, attrs, num_outputs):
    """Expand: broadcast shape to target."""
    return [input_shapes[0]]

def _infer_resize(input_shapes, attrs, num_outputs):
    """Resize: scale spatial dims by factor or to target size."""
    x_shape = input_shapes[0]
    scales = attrs.get("scales", None)
    sizes = attrs.get("sizes", None)
    if sizes:
        return [list(sizes)]
    if scales:
        out = list(x_shape)
        for i in range(min(len(scales), len(out))):
            out[i] = int(out[i] * scales[i])
        return [out]
    return [x_shape]

def _infer_where(input_shapes, attrs, num_outputs):
    """Where: broadcast condition with X and Y."""
    return [_broadcast_shapes(input_shapes[1], input_shapes[2])]

def _infer_gather(input_shapes, attrs, num_outputs):
    """Gather: axis selection. Output same rank as input."""
    x_shape = input_shapes[0]
    indices_shape = input_shapes[1] if len(input_shapes) > 1 else [1]
    axis = attrs.get("axis", 0)
    out = list(x_shape)
    out[axis] = indices_shape[0]  # Simplified: assumes 1-D indices
    return [out]

def _infer_gatherop(input_shapes, attrs, num_outputs):
    return _infer_gather(input_shapes, attrs, num_outputs)

def _infer_reducemean(input_shapes, attrs, num_outputs):
    """ReduceMean: reduce specified axes."""
    return [_reduce_shape(input_shapes[0], attrs)]

def _infer_reducesum(input_shapes, attrs, num_outputs):
    return [_reduce_shape(input_shapes[0], attrs)]

def _reduce_shape(x_shape, attrs):
    """Compute output shape for reduce ops."""
    axes = attrs.get("axes", None)
    keepdims = attrs.get("keepdims", True)
    if isinstance(keepdims, int):
        keepdims = bool(keepdims)
    
    if axes is None:
        if keepdims:
            return [1] * len(x_shape)
        else:
            return [1]
    
    out = list(x_shape)
    for ax in sorted(axes, reverse=True):
        if keepdims:
            out[ax] = 1
        else:
            out.pop(ax)
    if not out:
        out = [1]
    return out

def _infer_nonmaxsuppression(input_shapes, attrs, num_outputs):
    """NMS output: [num_selected, 3] typically."""
    return [[1, 3]]

def _infer_topk(input_shapes, attrs, num_outputs):
    """TopK: same shape as input, along given axis limited to K."""
    x_shape = input_shapes[0]
    k = attrs.get("k", attrs.get("K", 1))
    axis = attrs.get("axis", -1)
    if axis < 0:
        axis = len(x_shape) + axis
    out = list(x_shape)
    out[axis] = k
    return [out, out]  # values + indices

def _infer_topkop(input_shapes, attrs, num_outputs):
    return _infer_topk(input_shapes, attrs, num_outputs)

def _infer_constant(input_shapes, attrs, num_outputs):
    """Constant: output shape from the constant tensor value."""
    dims = attrs.get("dims", [1])
    return [list(dims)]

def _infer_constantop(input_shapes, attrs, num_outputs):
    return _infer_constant(input_shapes, attrs, num_outputs)

def _infer_lrn(input_shapes, attrs, num_outputs):
    """LRN: shape-preserving."""
    return [input_shapes[0]]

def _infer_loop(input_shapes, attrs, num_outputs):
    """Loop: can't infer statically. Return input shapes."""
    return _fallback(input_shapes, num_outputs)

def _infer_if(input_shapes, attrs, num_outputs):
    """If: can't infer statically. Return input shapes."""
    return _fallback(input_shapes, num_outputs)

def _infer_identityop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_expop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_sqrtop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_negop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_logop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_erfop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]

def _infer_tileop(input_shapes, attrs, num_outputs):
    return _infer_tile(input_shapes, attrs, num_outputs)

def _infer_sliceop(input_shapes, attrs, num_outputs):
    return _infer_slice(input_shapes, attrs, num_outputs)

def _infer_whereop(input_shapes, attrs, num_outputs):
    return _infer_where(input_shapes, attrs, num_outputs)

def _infer_padop(input_shapes, attrs, num_outputs):
    return _infer_pad(input_shapes, attrs, num_outputs)

def _infer_unsqueezeop(input_shapes, attrs, num_outputs):
    return _infer_unsqueeze(input_shapes, attrs, num_outputs)

def _infer_squeezeop(input_shapes, attrs, num_outputs):
    return _infer_squeeze(input_shapes, attrs, num_outputs)

def _infer_resizeop(input_shapes, attrs, num_outputs):
    return _infer_resize(input_shapes, attrs, num_outputs)


def _infer_andop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_ceilop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_compress(input_shapes, attrs, num_outputs):
    shape = list(input_shapes[0])
    axis = int(attrs.get("axis", 0))
    if axis < 0:
        axis += len(shape)
    if 0 <= axis < len(shape):
        shape[axis] = None
    return [shape]


def _infer_convtranspose(input_shapes, attrs, num_outputs):
    in_shape = list(input_shapes[0])
    out_c = int(attrs.get("out_channels", in_shape[1]))
    k = int(attrs.get("kernel_size", 3))
    s = int(attrs.get("stride", 1))
    p = int(attrs.get("padding", 0))
    op = int(attrs.get("output_padding", 0))
    d = int(attrs.get("dilation", 1))
    h_in = in_shape[2]
    w_in = in_shape[3]
    h_out = (h_in - 1) * s + d * (k - 1) + 1 + op - 2 * p
    w_out = (w_in - 1) * s + d * (k - 1) + 1 + op - 2 * p
    return [[in_shape[0], out_c, h_out, w_out]]


def _infer_cumsum(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_depthtospace(input_shapes, attrs, num_outputs):
    in_shape = list(input_shapes[0])
    b = int(attrs.get("blocksize", 2))
    if len(in_shape) == 4:
        c = in_shape[1] // (b * b) if in_shape[1] is not None else None
        h = in_shape[2] * b if in_shape[2] is not None else None
        w = in_shape[3] * b if in_shape[3] is not None else None
        return [[in_shape[0], c, h, w]]
    return [input_shapes[0]]


def _infer_einsum(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_equalop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_eyelike(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_floorop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_greaterop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_hardsigmoid(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_isinfop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_isnanop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_layernorm(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_lessop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_logsoftmax(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_notop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_onehot(input_shapes, attrs, num_outputs):
    axis = int(attrs.get("axis", -1))
    indices_shape = list(input_shapes[0])
    depth = input_shapes[1][0] if len(input_shapes) > 1 and input_shapes[1] else None
    if axis < 0:
        axis += len(indices_shape) + 1
    indices_shape.insert(axis, depth)
    return [indices_shape]


def _infer_orop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_rangeop(input_shapes, attrs, num_outputs):
    return [[None]]


def _infer_reciprocalop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_roundop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_selu(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_signop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


def _infer_spacetodepth(input_shapes, attrs, num_outputs):
    in_shape = list(input_shapes[0])
    b = int(attrs.get("blocksize", 2))
    if len(in_shape) == 4:
        c = in_shape[1] * b * b if in_shape[1] is not None else None
        h = in_shape[2] // b if in_shape[2] is not None else None
        w = in_shape[3] // b if in_shape[3] is not None else None
        return [[in_shape[0], c, h, w]]
    return [input_shapes[0]]


def _infer_xorop(input_shapes, attrs, num_outputs):
    return [input_shapes[0]]


# ---- Utilities ----

def _broadcast_shapes(a: List[int], b: List[int]) -> List[int]:
    """Broadcast two shapes following NumPy rules."""
    # Pad the shorter shape with 1s on the left
    if len(a) < len(b):
        a = [1] * (len(b) - len(a)) + a
    elif len(b) < len(a):
        b = [1] * (len(a) - len(b)) + b
    
    out = []
    for da, db in zip(a, b):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            # Incompatible: fall back to a
            out.append(da)
    return out
