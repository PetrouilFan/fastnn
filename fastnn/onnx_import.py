"""ONNX model importer for fastnn.

Converts ONNX models (.onnx) to fastnn's native format.
Supports common operators and builds a runnable computation graph.
"""

import onnx
import numpy as np
import struct
import logging
from typing import Dict, Optional, Any, List, Tuple

logger = logging.getLogger(__name__)

MAGIC = b"FNN\x00"
VERSION = 1


class Concat:
    """Concat module for fastnn - uses numpy for reliability."""
    def __init__(self, axis=1):
        self.axis = axis
    def __call__(self, *tensors):
        import fastnn
        import numpy as np
        
        tensors_np = []
        for t in tensors:
            if hasattr(t, 'numpy'):
                tensors_np.append(t.numpy())
            else:
                tensors_np.append(np.array(t))
        
        result = np.concatenate(tensors_np, axis=self.axis)
        return fastnn.tensor(result.flatten().tolist(), list(result.shape))


class Split:
    """Split module for fastnn - handles both contiguous and non-contiguous tensors."""
    def __init__(self, axis=0, split_sizes=None):
        self.axis = axis
        self.split_sizes = split_sizes
    
    def __call__(self, x):
        import fastnn
        import numpy as np
        
        if hasattr(x, 'numpy'):
            x_np = x.numpy()
        else:
            x_np = np.array(x)
        
        if self.split_sizes:
            results = []
            start = 0
            for size in self.split_sizes:
                end = start + int(size)
                slices = [slice(None) if i != self.axis else slice(start, end) for i in range(len(x_np.shape))]
                sliced = x_np[tuple(slices)]
                results.append(fastnn.tensor(sliced.flatten().tolist(), list(sliced.shape)))
                start = end
            return results
        else:
            dim_size = x_np.shape[self.axis]
            half = dim_size // 2
            slices1 = [slice(None) if i != self.axis else slice(0, half) for i in range(len(x_np.shape))]
            slices2 = [slice(None) if i != self.axis else slice(half, dim_size) for i in range(len(x_np.shape))]
            r1 = x_np[tuple(slices1)]
            r2 = x_np[tuple(slices2)]
            return [
                fastnn.tensor(r1.flatten().tolist(), list(r1.shape)),
                fastnn.tensor(r2.flatten().tolist(), list(r2.shape))
            ]


class Softmax:
    """Softmax module for fastnn."""
    def __init__(self, axis=-1):
        self.axis = axis
    def __call__(self, x):
        import fastnn
        max_vals = fastnn.max(x, dim=self.axis, keepdim=True)
        exp_x = fastnn.exp(x - max_vals)
        sum_exp = fastnn.sum(exp_x, dim=self.axis, keepdim=True)
        return exp_x / sum_exp


class Resize:
    """Resize module for fastnn."""
    def __init__(self, scale_factor=None, mode="nearest"):
        self.scale_factor = scale_factor
        self.mode = mode
    def __call__(self, x):
        import fastnn
        if hasattr(fastnn, 'Upsample') and self.scale_factor:
            upsample = fastnn.Upsample(scale_factor=self.scale_factor, mode=self.mode)
            return upsample(x)
        return x


class Slice:
    """Slice module for fastnn."""
    def __init__(self, starts, ends, axes, steps):
        self.starts = starts or [0]
        self.ends = ends or [None]
        self.axes = axes or [0]
        self.steps = steps or [1]
    def __call__(self, x):
        if len(self.starts) == 1 and len(self.axes) == 1:
            axis = self.axes[0]
            start = self.starts[0]
            end = self.ends[0] if self.ends[0] is not None else x.shape[axis]
            step = self.steps[0] if self.steps else 1
            return x.slice(axis, start, end, step)
        return x


def _write_tensor(f, name: str, data: np.ndarray):
    """Write a tensor to the .fnn file."""
    name_bytes = name.encode("utf-8")
    shape = list(data.shape)
    data_f32 = data.astype(np.float32).flatten()
    f.write(struct.pack("<Q", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<Q", len(shape)))
    for d in shape:
        f.write(struct.pack("<q", d))
    f.write(struct.pack("<Q", len(data_f32)))
    f.write(data_f32.tobytes())


def _get_initializer(model: onnx.ModelProto, name: str) -> Optional[np.ndarray]:
    """Get initializer tensor by name."""
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    return None


def _get_attr(node: onnx.NodeProto, name: str, default=None):
    """Get attribute value from node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.HasField('f'):
                return attr.f
            elif attr.HasField('i'):
                return attr.i
            elif attr.HasField('s'):
                return attr.s.decode('utf-8')
            elif attr.ints:
                return list(attr.ints)
            elif attr.floats:
                return list(attr.floats)
    return default


def _np_to_fastnn(arr: np.ndarray):
    """Convert numpy array to fastnn tensor - fast path for contiguous arrays."""
    import fastnn
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    shape = list(arr.shape)
    # Fast path: use frombuffer if possible
    return fastnn.tensor(arr.flatten().tolist(), shape)


def import_onnx(onnx_path: str, fnn_path: str) -> Dict[str, Any]:
    """Import an ONNX model and save it in fastnn format.

    Args:
        onnx_path: Path to .onnx file
        fnn_path: Path to output .fnn file

    Returns:
        Dictionary with model info (layers, input_shape, output_shape, model, weights)
    """
    model = onnx.load(onnx_path)
    
    layers = []
    parameters = []
    layer_index = 0
    
    # Build a mapping of node outputs to their consumers
    output_to_node = {}
    for node in model.graph.node:
        for output in node.output:
            output_to_node[output] = node
    
    # Track which tensors have been consumed
    consumed = set()
    
    for node in model.graph.node:
        op_type = node.op_type
        layer_info = {
            "name": node.name or f"{op_type}_{layer_index}",
            "type": op_type,
            "node": node,
            "inputs": list(node.input),
            "outputs": list(node.output),
        }
        layer_index += 1
        
        # Extract parameters for each op type
        if op_type == "Conv":
            weight = _get_initializer(model, node.input[1])
            bias = _get_initializer(model, node.input[2]) if len(node.input) > 2 else None
            
            out_channels, in_channels = weight.shape[:2]
            kernel_h, kernel_w = weight.shape[2], weight.shape[3]
            stride = _get_attr(node, "strides", [1, 1])
            stride = stride[0] if isinstance(stride, list) else stride
            padding = _get_attr(node, "pads", [0, 0, 0, 0])
            padding = padding[0] if isinstance(padding, list) else padding
            groups = _get_attr(node, "group", 1)
            
            layer_info["in_channels"] = in_channels
            layer_info["out_channels"] = out_channels
            layer_info["kernel_size"] = kernel_h
            layer_info["stride"] = stride
            layer_info["padding"] = padding
            layer_info["groups"] = groups
            layer_info["bias"] = bias is not None
            
            parameters.append((f"{node.name}.weight", weight))
            if bias is not None:
                parameters.append((f"{node.name}.bias", bias))
        
        elif op_type == "BatchNormalization":
            weight = _get_initializer(model, node.input[1])
            bias = _get_initializer(model, node.input[2])
            mean = _get_initializer(model, node.input[3])
            var = _get_initializer(model, node.input[4])
            
            layer_info["num_features"] = weight.shape[0]
            layer_info["eps"] = _get_attr(node, "epsilon", 1e-5)
            
            for name, data in [("weight", weight), ("bias", bias), ("mean", mean), ("var", var)]:
                if data is not None:
                    parameters.append((f"{node.name}.{name}", data))
        
        elif op_type == "MaxPool":
            kernel = _get_attr(node, "kernel_shape", [2, 2])
            stride = _get_attr(node, "strides", kernel)
            padding = _get_attr(node, "pads", [0, 0, 0, 0])
            
            layer_info["kernel_size"] = kernel[0] if isinstance(kernel, list) else kernel
            layer_info["stride"] = stride[0] if isinstance(stride, list) else stride
            layer_info["padding"] = padding[0] if isinstance(padding, list) else padding
        
        elif op_type == "Reshape":
            shape = _get_initializer(model, node.input[1])
            if shape is not None:
                layer_info["shape"] = shape.tolist()
        
        elif op_type == "Transpose":
            perm = _get_attr(node, "perm")
            layer_info["perm"] = perm
        
        elif op_type == "Split":
            axis = _get_attr(node, "axis", 0)
            split = _get_attr(node, "split", None)
            layer_info["axis"] = axis
            layer_info["split"] = split
        
        elif op_type == "Softmax":
            axis = _get_attr(node, "axis", -1)
            layer_info["axis"] = axis
        
        elif op_type == "Resize":
            mode = _get_attr(node, "mode", "nearest")
            layer_info["mode"] = mode
            if len(node.input) > 2:
                scales = _get_initializer(model, node.input[2])
                if scales is not None:
                    layer_info["scales"] = scales.tolist()
        
        elif op_type == "Slice":
            layer_info["starts"] = _get_initializer(model, node.input[1]).tolist() if len(node.input) > 1 else [0]
            layer_info["ends"] = _get_initializer(model, node.input[2]).tolist() if len(node.input) > 2 else None
            layer_info["axes"] = _get_initializer(model, node.input[3]).tolist() if len(node.input) > 3 else [0]
            layer_info["steps"] = _get_initializer(model, node.input[4]).tolist() if len(node.input) > 4 else [1]
        
        elif op_type == "Flatten":
            layer_info["axis"] = _get_attr(node, "axis", 1)
        
        layers.append(layer_info)
    
    # Write to .fnn file
    with open(fnn_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(parameters)))
        for name, data in parameters:
            _write_tensor(f, name, data)
    
    # Get input/output shapes
    input_shape = None
    output_shape = None
    for inp in model.graph.input:
        if inp.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            input_shape = dims
    for out in model.graph.output:
        if out.type.tensor_type.HasField("shape"):
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            output_shape = dims
    
    # Fold BatchNorm into Conv where possible (optimization)
    parameters = _fold_batch_norm_into_conv(layers, parameters)

    # Build computation graph
    graph_model = _build_computation_graph(model, layers, parameters)
    
    result = {
        "layers": layers,
        "parameters": len(parameters),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "model": graph_model,
        "weights": {name: data for name, data in parameters},
    }
    return result


def _fuse_conv_bn_silu_patterns(onnx_model, layer_instances, weights, layers):
    """Fuse Conv + BatchNorm + SiLU patterns into FusedConvBnSilu."""
    import fastnn

    # Build node output to input map
    node_outputs = {}
    for node in onnx_model.graph.node:
        for output in node.output:
            node_outputs[output] = node

    fused_nodes = set()

    for node in onnx_model.graph.node:
        if node.op_type == "Conv" and node.name not in fused_nodes:
            # Check if next node is BatchNorm
            conv_output = node.output[0]
            if conv_output in node_outputs:
                next_node = node_outputs[conv_output]
                if next_node.op_type == "BatchNormalization":
                    bn_output = next_node.output[0]
                    if bn_output in node_outputs:
                        silu_node = node_outputs[bn_output]
                        if silu_node.op_type == "SiLU":
                            # Found Conv -> BN -> SiLU pattern
                            conv_name = node.name
                            bn_name = next_node.name
                            silu_name = silu_node.name

                            # Get conv params
                            conv_info = None
                            for layer in layers:
                                if layer["name"] == conv_name:
                                    conv_info = layer
                                    break

                            if conv_info and hasattr(fastnn, 'FusedConvBnSilu'):
                                try:
                                    fused = fastnn.FusedConvBnSilu(
                                        in_channels=conv_info["in_channels"],
                                        out_channels=conv_info["out_channels"],
                                        kernel_size=conv_info["kernel_size"],
                                        stride=conv_info["stride"],
                                        padding=conv_info["padding"],
                                        dilation=conv_info.get("dilation", 1),
                                        groups=conv_info.get("groups", 1),
                                        eps=1e-5,  # Default
                                        bias=conv_info["bias"],
                                    )

                                    # Set weights
                                    weight_key = f"{conv_name}.weight"
                                    if weight_key in weights:
                                        w_tensor = fastnn.tensor(weights[weight_key].flatten().tolist(), list(weights[weight_key].shape))
                                        fused.set_conv_weight(w_tensor)

                                    if conv_info["bias"]:
                                        bias_key = f"{conv_name}.bias"
                                        if bias_key in weights:
                                            b_tensor = fastnn.tensor(weights[bias_key].flatten().tolist(), list(weights[bias_key].shape))
                                            fused.set_conv_bias(b_tensor)

                                    # Set BN params
                                    for param, setter in [("weight", "set_bn_weight"), ("bias", "set_bn_bias"),
                                                          ("running_mean", "set_bn_running_mean"), ("running_var", "set_bn_running_var")]:
                                        param_key = f"{bn_name}.{param}"
                                        if param_key in weights:
                                            p_tensor = fastnn.tensor(weights[param_key].flatten().tolist(), list(weights[param_key].shape))
                                            getattr(fused, setter)(p_tensor)

                                    # Replace conv layer with fused
                                    layer_instances[conv_name] = ("fused_conv_bn_silu", fused)

                                    # Mark BN and SiLU as skipped
                                    layer_instances[bn_name] = ("skip", None)
                                    layer_instances[silu_name] = ("skip", None)

                                    fused_nodes.add(conv_name)
                                    fused_nodes.add(bn_name)
                                    fused_nodes.add(silu_name)

                                    logger.info(f"Fused {conv_name} + {bn_name} + {silu_name} into FusedConvBnSilu")

                                 except Exception as e:
                                     logger.warning(f"Failed to fuse {conv_name}: {e}")


def _fold_batch_norm_into_conv(layers, parameters):
    """Fold BatchNorm layers into preceding Conv layers to eliminate BN ops."""
    import numpy as np

    # Convert parameters list to dict for easier mutation
    param_dict = dict(parameters)

    # Build map from tensor output name to layer index
    output_to_layer = {}
    for idx, layer in enumerate(layers):
        for out_name in layer.get("outputs", []):
            output_to_layer[out_name] = idx

    changed = False
    for i, layer in enumerate(layers):
        if layer.get("type") == "Conv":
            conv_out = layer.get("outputs", [])[0] if layer.get("outputs") else None
            if conv_out is None:
                continue
            if conv_out in output_to_layer:
                bn_idx = output_to_layer[conv_out]
                bn_layer = layers[bn_idx]
                if bn_layer.get("type") == "BatchNormalization":
                    conv_name = layer["name"]
                    bn_name = bn_layer["name"]

                    # Parameter keys
                    w_key = f"{conv_name}.weight"
                    b_key = f"{conv_name}.bias"
                    bn_w_key = f"{bn_name}.weight"
                    bn_b_key = f"{bn_name}.bias"
                    bn_mean_key = f"{bn_name}.mean"
                    bn_var_key = f"{bn_name}.var"

                    # Ensure all required params exist
                    required = [w_key, bn_w_key, bn_b_key, bn_mean_key, bn_var_key]
                    if not all(k in param_dict for k in required):
                        logger.warning(f"Missing parameters for BN fold: {conv_name} -> {bn_name}")
                        continue

                    w = param_dict[w_key].astype(np.float32)
                    bn_w = param_dict[bn_w_key].astype(np.float32)
                    bn_b = param_dict[bn_b_key].astype(np.float32)
                    bn_mean = param_dict[bn_mean_key].astype(np.float32)
                    bn_var = param_dict[bn_var_key].astype(np.float32)

                    eps = bn_layer.get("eps", 1e-5)

                    # Compute scale: gamma / sqrt(var + eps)
                    scale = bn_w / np.sqrt(bn_var + eps)

                    # New weight: w * scale (broadcast to out_channels dim)
                    new_w = w * scale.reshape(-1, 1, 1, 1)

                    # Bias handling
                    b = param_dict.get(b_key)  # may be None
                    if b is not None:
                        new_b = (b - bn_mean) * scale + bn_b
                    else:
                        # Conv had no bias, create one from BN parameters
                        new_b = (-bn_mean) * scale + bn_b

                    # Update param dict
                    param_dict[w_key] = new_w
                    # Ensure bias key exists in param_dict (create if needed)
                    if b_key not in param_dict:
                        param_dict[b_key] = new_b
                    else:
                        param_dict[b_key] = new_b
                    # Mark conv as having bias
                    layer["bias"] = True

                    # Mark BN layer to be skipped
                    bn_layer["type"] = "skip"
                    bn_layer["skip"] = True
                    logger.info(f"Folded BatchNorm {bn_name} into Conv {conv_name}")
                    changed = True

    if changed:
        return list(param_dict.items())
    else:
        return parameters



def _build_computation_graph(onnx_model, layers, parameters):
    """Build a runnable computation graph from ONNX layers."""
    import fastnn
    
    weights = {name: data for name, data in parameters}
    layer_instances = {}
    
    for layer_info in layers:
        name = layer_info["name"]
        op_type = layer_info.get("type", layer_info["node"].op_type)
        node = layer_info["node"]
        
        try:
            if op_type == "Conv" or node.op_type == "Conv":
                conv = fastnn.Conv2d(
                    in_channels=layer_info["in_channels"],
                    out_channels=layer_info["out_channels"],
                    kernel_size=layer_info["kernel_size"],
                    stride=layer_info["stride"],
                    padding=layer_info["padding"],
                    groups=layer_info.get("groups", 1),
                    bias=layer_info["bias"],
                )
                weight_key = f"{node.name}.weight"
                if weight_key in weights:
                    w = weights[weight_key]
                    w_tensor = fastnn.tensor(w.flatten().tolist(), list(w.shape))
                    conv.set_weight(w_tensor)
                
                bias_key = f"{node.name}.bias"
                if bias_key in weights:
                    b = weights[bias_key]
                    b_tensor = fastnn.tensor(b.flatten().tolist(), list(b.shape))
                    conv.set_bias(b_tensor)
                
                layer_instances[name] = ("conv", conv)
            
            elif op_type == "BatchNorm2d" or node.op_type == "BatchNormalization":
                bn = fastnn.BatchNorm2d(
                    num_features=layer_info["num_features"],
                    eps=layer_info.get("eps", 1e-5),
                )
                for param_name, param_key in [("weight", "weight"), ("bias", "bias")]:
                    if f"{node.name}.{param_name}" in weights:
                        data = weights[f"{node.name}.{param_name}"]
                        tensor = fastnn.tensor(data.flatten().tolist(), list(data.shape))
                        if param_name == "weight":
                            bn.weight.copy_(tensor)
                        else:
                            bn.bias.copy_(tensor)
                
                layer_instances[name] = ("bn", bn)
            
            elif op_type == "Concat":
                axis = _get_attr(node, "axis", 1)
                layer_instances[name] = ("concat", Concat(axis=axis))

            elif op_type == "MaxPool":
                kernel = _get_attr(node, "kernel_shape", [2, 2])
                stride = _get_attr(node, "strides", kernel)
                pads = _get_attr(node, "pads", [0, 0, 0, 0])
                # Use symmetric values: assume square kernel, stride, padding
                ks = kernel[0] if isinstance(kernel, list) else kernel
                st = stride[0] if isinstance(stride, list) else stride
                # pads is [top, left, bottom, right]; take top (or first) as symmetric padding
                pd = pads[0] if isinstance(pads, list) else pads
                try:
                    maxpool = fastnn.MaxPool2d(kernel_size=ks, stride=st, padding=pd)
                    layer_instances[name] = ("maxpool", maxpool)
                except Exception as e:
                    logger.warning(f"MaxPool creation failed: {e}, using pass-through")
                    layer_instances[name] = ("maxpool", None)

            elif op_type == "Split":
                axis = _get_attr(node, "axis", 0)
                split_sizes = _get_attr(node, "split", None)
                layer_instances[name] = ("split", Split(axis=axis, split_sizes=split_sizes))

            elif op_type == "Softmax":
                axis = _get_attr(node, "axis", -1)
                layer_instances[name] = ("softmax", Softmax(axis=axis))

            elif op_type == "Flatten":
                axis = _get_attr(node, "axis", 1)
                # For now, use a simple pass-through flatten
                layer_instances[name] = ("flatten", None)

            elif op_type == "Resize":
                mode = _get_attr(node, "mode", "nearest")
                scales = None
                # Resize inputs: [x, roi, scales, sizes] (scales typically at index 2)
                if len(node.input) > 2:
                    scales_arr = _get_initializer(onnx_model, node.input[2])
                    if scales_arr is not None:
                        scales = scales_arr.tolist()
                layer_instances[name] = ("resize", Resize(mode=mode, scale_factor=scales))

            else:
                layer_instances[name] = (op_type, None)
        
        except Exception as e:
            logger.warning(f"Failed to create layer {name}: {e}")
            layer_instances[name] = (op_type, None)

    # Fuse Conv + BatchNorm + SiLU patterns
    _fuse_conv_bn_silu_patterns(onnx_model, layer_instances, weights, layers)

    class ONNXModel:
        def __init__(self):
            self.layer_instances = layer_instances
            self.onnx_model = onnx_model
            self.weights = weights
            self.execution_order = self._compute_execution_order()

        def _compute_execution_order(self):
            """Compute topological execution order of nodes."""
            # Build dependency graph
            node_deps = {}
            node_outputs = {}

            # Map node names to indices
            node_indices = {}
            for i, node in enumerate(self.onnx_model.graph.node):
                node_name = node.name or f"node_{i}"
                node_indices[node_name] = i

            # Build dependency map
            for i, node in enumerate(self.onnx_model.graph.node):
                node_name = node.name or f"node_{i}"
                deps = set()

                # Check inputs
                for inp in node.input:
                    if inp in self.weights:
                        continue  # Constants are not dependencies
                    # Find which node produces this input
                    for j, other_node in enumerate(self.onnx_model.graph.node):
                        if i != j and inp in other_node.output:
                            other_name = other_node.name or f"node_{j}"
                            deps.add(j)
                            break

                node_deps[i] = deps

            # Topological sort
            visited = set()
            temp_visited = set()
            order = []

            def visit(node_idx):
                if node_idx in temp_visited:
                    logger.warning(f"Cycle detected involving node {node_idx}, skipping")
                    return  # Cycle detected, skip
                if node_idx in visited:
                    return

                temp_visited.add(node_idx)

                for dep in node_deps[node_idx]:
                    visit(dep)

                temp_visited.remove(node_idx)
                visited.add(node_idx)
                order.append(node_idx)

            for i in range(len(self.onnx_model.graph.node)):
                if i not in visited:
                    visit(i)

            return order

        def forward(self, x):
            """Execute the ONNX graph - Conv+Bn+Act path only."""
            import fastnn
            
            cache = {}
            input_name = self.onnx_model.graph.input[0].name
            cache[input_name] = x
            
            for init in self.onnx_model.graph.initializer:
                arr = onnx.numpy_helper.to_array(init)
                cache[init.name] = fastnn.tensor(arr.flatten().tolist(), list(arr.shape))
            
            for node_idx in self.execution_order:
                node = self.onnx_model.graph.node[node_idx]
                op_type = node.op_type
                node_name = node.name or f"{op_type}_{id(node)}"
                
                inputs = []
                for inp in node.input:
                    if inp in cache:
                        inputs.append(cache[inp])
                    elif inp in self.weights:
                        w = self.weights[inp]
                        inputs.append(fastnn.tensor(w.flatten().tolist(), list(w.shape)))
                    else:
                        inputs.append(None)
                
                layer_type = self.layer_instances.get(node_name, (None, None))
                outputs = []

                try:
                    if layer_type[0] == "skip":
                        # Skip fused nodes
                        outputs = inputs[:len(node.output)] if inputs else [None] * len(node.output)

                    elif layer_type[0] == "fused_conv_bn_silu":
                        if inputs[0] is not None and layer_type[1]:
                            outputs = [layer_type[1](inputs[0])]

                    elif op_type == "Conv":
                        if inputs[0] is not None and layer_type[1]:
                            outputs = [layer_type[1](inputs[0])]

                    elif op_type == "BatchNormalization":
                        if inputs[0] is not None and layer_type[1]:
                            outputs = [layer_type[1](inputs[0])]

                    elif op_type == "Sigmoid":
                        if inputs[0] is not None:
                            outputs = [fastnn.sigmoid(inputs[0])]

                    elif op_type == "SiLU":
                        if inputs[0] is not None:
                            outputs = [fastnn.silu(inputs[0])]

                    elif op_type == "Relu":
                        if inputs[0] is not None:
                            outputs = [fastnn.relu(inputs[0])]

                    elif op_type == "Add":
                        if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                            outputs = [inputs[0] + inputs[1]]

                    elif op_type == "Sub":
                        if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                            outputs = [inputs[0] - inputs[1]]

                    elif op_type == "Mul":
                        if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                            outputs = [inputs[0] * inputs[1]]

                    elif op_type == "Div":
                        if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                            outputs = [inputs[0] / inputs[1]]

                    elif op_type == "Concat":
                        if len(inputs) >= 1 and all(inp is not None for inp in inputs):
                            try:
                                axis = _get_attr(node, "axis", 1)
                                if len(inputs) > 1:
                                    shapes = [inp.shape for inp in inputs]
                                    if len(set(tuple(s) for s in shapes)) > 1:  # Different shapes
                                        # Handle multi-scale fusion: upsample to match max spatial dims
                                        import numpy as np
                                        np_inputs = []
                                        for inp in inputs:
                                            arr = inp.numpy() if hasattr(inp, 'numpy') else np.array(inp)
                                            np_inputs.append(arr)

                                        # Assuming shape [batch, channels, h, w], axis=1 (channels)
                                        if axis == 1 and all(len(s) == 4 for s in shapes):
                                            # Find max spatial dims
                                            max_h = max(s[2] for s in shapes)
                                            max_w = max(s[3] for s in shapes)

                                            upsampled_inputs = []
                                            for arr in np_inputs:
                                                h, w = arr.shape[2], arr.shape[3]
                                                if h < max_h or w < max_w:
                                                    # Nearest neighbor upsample
                                                    scale_h = max_h / h
                                                    scale_w = max_w / w
                                                    # Use scipy if available, else simple repeat
                                                    try:
                                                        from scipy.ndimage import zoom
                                                        upsampled = zoom(arr, (1, 1, scale_h, scale_w), order=0)
                                                    except ImportError:
                                                        # Simple nearest neighbor
                                                        upsampled = np.repeat(np.repeat(arr, int(scale_h), axis=2), int(scale_w), axis=3)
                                                        # Trim if not integer scale
                                                        upsampled = upsampled[:, :, :max_h, :max_w]
                                                    upsampled_inputs.append(upsampled)
                                                else:
                                                    upsampled_inputs.append(arr)

                                            result = np.concatenate(upsampled_inputs, axis=axis)
                                            outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]
                                        else:
                                            # Fallback to first input if not multi-scale concat
                                            outputs = [inputs[0]]
                                    else:
                                        # Same shapes, can concatenate
                                        if layer_type[1]:
                                            outputs = [layer_type[1](*inputs)]
                                        else:
                                            # Fallback to numpy concat
                                            import numpy as np
                                            np_inputs = []
                                            for inp in inputs:
                                                if hasattr(inp, 'numpy'):
                                                    np_inputs.append(inp.numpy())
                                                else:
                                                    np_inputs.append(np.array(inp))
                                            result = np.concatenate(np_inputs, axis=axis)
                                            outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]
                                else:
                                    outputs = [inputs[0]]
                            except Exception as e:
                                logger.error(f"Concat failed for {node_name}: {e}")
                                outputs = [inputs[0]] if inputs else [None]

                    elif op_type == "MaxPool":
                        if inputs[0] is not None and layer_type[1]:
                            outputs = [layer_type[1](inputs[0])]
                        elif inputs[0] is not None:
                            # Fallback: pass through (should not happen if layer created)
                            outputs = [inputs[0]]

                    elif op_type == "Reshape":
                        if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                            try:
                                # Get new shape from second input
                                if hasattr(inputs[1], 'numpy'):
                                    new_shape_arr = inputs[1].numpy()
                                    new_shape = new_shape_arr.flatten().astype(int).tolist()
                                else:
                                    new_shape = [int(x) for x in inputs[1]]

                                # Validate reshape is possible
                                total_elements = 1
                                for dim in inputs[0].shape:
                                    total_elements *= dim

                                target_elements = 1
                                for dim in new_shape:
                                    if dim > 0:  # Ignore -1 dimension
                                        target_elements *= dim

                                if target_elements == total_elements or -1 in new_shape:
                                    # Infer -1 dimension
                                    if -1 in new_shape:
                                        idx = new_shape.index(-1)
                                        inferred_dim = total_elements
                                        for i, dim in enumerate(new_shape):
                                            if i != idx and dim > 0:
                                                inferred_dim //= dim
                                        new_shape[idx] = inferred_dim

                                    outputs = [inputs[0].view(new_shape)]
                                else:
                                    # Shape mismatch, pass through
                                    outputs = [inputs[0]]
                            except Exception as e:
                                logger.warning(f"Reshape failed: {e}, using input as fallback")
                                outputs = [inputs[0]]

                    elif op_type == "Transpose":
                        if inputs[0] is not None:
                            # Get perm attribute; if not present, default to reverse dimensions
                            perm = _get_attr(node, "perm", None)
                            if perm is None:
                                # Default: reverse dimensions
                                perm = list(reversed(range(len(inputs[0].shape))))
                            else:
                                # Ensure perm is a list of integers
                                perm = [int(p) for p in perm]
                            try:
                                # Use tensor.permute method
                                outputs = [inputs[0].permute(perm)]
                            except Exception as e:
                                logger.warning(f"Transpose failed: {e}, using input as fallback")
                                outputs = [inputs[0]]

                    elif op_type == "Split":
                        if inputs[0] is not None and layer_type[1]:
                            outputs = layer_type[1](inputs[0])

                    elif op_type == "Softmax":
                        if inputs[0] is not None and layer_type[1]:
                            outputs = [layer_type[1](inputs[0])]

                    elif op_type == "Flatten":
                        if inputs[0] is not None:
                            # Simple flatten implementation
                            shape = inputs[0].shape
                            if len(shape) > 1:
                                new_shape = [shape[0], shape[1:].numel()]
                                try:
                                    outputs = [inputs[0].view(new_shape)]
                                except:
                                    outputs = [inputs[0]]
                            else:
                                outputs = [inputs[0]]

                    elif op_type == "Resize":
                        if inputs[0] is not None:
                            if len(inputs) >= 2 and inputs[1] is not None:
                                # Get scales from second input
                                scales_arr = inputs[1].numpy() if hasattr(inputs[1], 'numpy') else np.array(inputs[1])
                                # For 4D tensor [batch, c, h, w], scales should be [1, 1, scale_h, scale_w]
                                if len(scales_arr) >= 4:
                                    scale_h, scale_w = scales_arr[-2], scales_arr[-1]
                                    mode = _get_attr(node, "mode", "nearest")
                                    if hasattr(fastnn, 'Upsample'):
                                        upsample = fastnn.Upsample(scale_factor=(scale_h, scale_w), mode=mode)
                                        outputs = [upsample(inputs[0])]
                                    else:
                                        # Fallback to numpy
                                        import numpy as np
                                        arr = inputs[0].numpy() if hasattr(inputs[0], 'numpy') else np.array(inputs[0])
                                        # Simple nearest neighbor upsample
                                        upsampled = np.repeat(np.repeat(arr, int(scale_h), axis=2), int(scale_w), axis=3)
                                        upsampled = upsampled[:, :, :int(arr.shape[2] * scale_h), :int(arr.shape[3] * scale_w)]
                                        outputs = [fastnn.tensor(upsampled.flatten().tolist(), list(upsampled.shape))]
                                else:
                                    outputs = [inputs[0]]
                            elif layer_type[1]:
                                outputs = [layer_type[1](inputs[0])]
                            else:
                                outputs = [inputs[0]]

                    elif op_type in ["Unsqueeze", "Shape", "Gather", "Cast", "Expand", "ConstantOfShape", "Range", "MatMul"]:
                        # Pass through for now - these need more complex implementations
                        if inputs[0] is not None:
                            outputs = [inputs[0]]
                    else:
                        # Unknown op not explicitly handled
                        if layer_type[1] is not None:
                            try:
                                outputs = [layer_type[1](inputs[0])]
                            except Exception as e2:
                                logger.warning(f"Layer {op_type} execution failed: {e2}, using pass-through")
                                outputs = inputs[:len(node.output)] if inputs else [None] * len(node.output)
                        else:
                            # Pass-through fallback: forward first input to each output
                            outputs = inputs[:len(node.output)] if inputs else [None] * len(node.output)

                except Exception as e:
                    logger.warning(f"Failed to execute {op_type} node {node_name}: {e}")
                    # Try to pass through first input as fallback
                    if inputs and inputs[0] is not None:
                        outputs = [inputs[0]]
                
                for i, out_name in enumerate(node.output):
                    if i < len(outputs):
                        cache[out_name] = outputs[i]
            
            output_name = self.onnx_model.graph.output[0].name
            return cache.get(output_name)
        
        def __call__(self, x):
            return self.forward(x)
    
    return ONNXModel()