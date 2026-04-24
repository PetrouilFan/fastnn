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
    """Convert numpy array to fastnn tensor."""
    import fastnn
    return fastnn.tensor(arr.flatten().tolist(), list(arr.shape))


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
            
            layer_info["kernel_size"] = kernel_h
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
            
            elif op_type == "Add" or op_type == "BiasAdd":
                layer_instances[name] = ("add", None)
            
            elif op_type == "Mul":
                layer_instances[name] = ("mul", None)
            
            elif op_type == "Sub":
                layer_instances[name] = ("sub", None)
            
            elif op_type == "Div":
                layer_instances[name] = ("div", None)
            
            elif op_type == "Sigmoid":
                layer_instances[name] = ("sigmoid", None)
            
            elif op_type == "Concat":
                layer_instances[name] = ("concat", layer_info.get("axis", 1))
            
            elif op_type == "MaxPool" or node.op_type == "MaxPool":
                layer_instances[name] = ("maxpool", (
                    layer_info["kernel_size"],
                    layer_info["stride"],
                    layer_info["padding"],
                ))
            
            elif op_type == "Reshape":
                layer_instances[name] = ("reshape", layer_info.get("shape"))
            
            elif op_type == "Transpose":
                layer_instances[name] = ("transpose", layer_info.get("perm"))
            
            elif op_type == "Split":
                layer_instances[name] = ("split", (
                    layer_info.get("axis", 0),
                    layer_info.get("split"),
                ))
            
            elif op_type == "Softmax":
                layer_instances[name] = ("softmax", layer_info.get("axis", -1))
            
            elif op_type == "Flatten":
                layer_instances[name] = ("flatten", layer_info.get("axis", 1))
            
            elif op_type == "Resize":
                layer_instances[name] = ("resize", layer_info.get("scales"))
            
            else:
                layer_instances[name] = (op_type, None)
        
        except Exception as e:
            logger.warning(f"Failed to create layer {name}: {e}")
            layer_instances[name] = (op_type, None)
    
    class ONNXModel:
        def __init__(self):
            self.layer_instances = layer_instances
            self.onnx_model = onnx_model
            self.weights = weights
        
        def forward(self, x):
            """Execute the ONNX graph."""
            import fastnn
            
            cache = {}
            input_name = self.onnx_model.graph.input[0].name
            cache[input_name] = x
            
            for init in self.onnx_model.graph.initializer:
                arr = onnx.numpy_helper.to_array(init)
                cache[init.name] = fastnn.tensor(arr.flatten().tolist(), list(arr.shape))
            
            try:
                for node in self.onnx_model.graph.node:
                    node_name = node.name or f"{node.op_type}_{id(node)}"
                    op_type = node.op_type
                    
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
                        if op_type == "Conv":
                            if inputs[0] is not None:
                                layer = layer_type[1]
                                if layer:
                                    outputs = [layer(inputs[0])]

                        elif op_type == "BatchNormalization":
                            if inputs[0] is not None:
                                layer = layer_type[1]
                                if layer:
                                    outputs = [layer(inputs[0])]

                        elif op_type == "Sigmoid":
                            if inputs[0] is not None:
                                outputs = [fastnn.sigmoid(inputs[0])]

                        elif op_type == "SiLU":
                            if inputs[0] is not None:
                                outputs = [fastnn.silu(inputs[0])]

                        elif op_type == "Add":
                            valid_inputs = [inp for inp in inputs if inp is not None]
                            if len(valid_inputs) >= 2:
                                result = valid_inputs[0]
                                for inp in valid_inputs[1:]:
                                    result = result + inp
                                outputs = [result]

                        elif op_type == "Mul":
                            valid_inputs = [inp for inp in inputs if inp is not None]
                            if len(valid_inputs) >= 2:
                                result = valid_inputs[0]
                                for inp in valid_inputs[1:]:
                                    result = result * inp
                                outputs = [result]

                        elif op_type == "Sub":
                            if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                                outputs = [inputs[0] - inputs[1]]

                        elif op_type == "Div":
                            if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                                outputs = [inputs[0] / inputs[1]]

                        elif op_type == "Concat":
                            valid_inputs = [inp for inp in inputs if inp is not None]
                            if len(valid_inputs) >= 2:
                                import numpy as np
                                axis = layer_type[1] if isinstance(layer_type, tuple) else 1
                                inputs_np = [inp.numpy() for inp in valid_inputs if hasattr(inp, 'numpy')]
                                result = np.concatenate(inputs_np, axis=axis)
                                outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]

                        elif op_type == "MaxPool":
                            if inputs[0] is not None:
                                kernel, stride, padding = layer_type[1]
                                if hasattr(fastnn, 'MaxPool2d'):
                                    pool = fastnn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
                                    outputs = [pool(inputs[0])]

                        elif op_type == "Reshape":
                            if inputs[0] is not None:
                                try:
                                    if len(inputs) > 1 and inputs[1] is not None and hasattr(inputs[1], 'numpy'):
                                        shape = inputs[1].numpy().astype(int).tolist()
                                        outputs = [inputs[0].view(shape)]
                                    elif layer_type[1] is not None:
                                        outputs = [inputs[0].view(layer_type[1])]
                                except:
                                    outputs = [inputs[0]]

                        elif op_type == "MatMul":
                            if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                                import numpy as np
                                a = inputs[0].numpy()
                                b = inputs[1].numpy()
                                result = np.dot(a, b)
                                outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]

                        elif op_type == "Transpose":
                            if inputs[0] is not None:
                                import numpy as np
                                perm = layer_type[1] if layer_type[1] else None
                                if perm:
                                    x_np = inputs[0].numpy()
                                    ndim = x_np.ndim
                                    if len(perm) > ndim:
                                        perm = perm[:ndim]
                                    elif len(perm) < ndim:
                                        perm = list(perm) + list(range(len(perm), ndim))
                                    result = np.transpose(x_np, perm)
                                    outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]
                                else:
                                    outputs = [inputs[0]]

                        elif op_type == "Split":
                            if inputs[0] is not None:
                                axis, split_sizes = layer_type[1]
                                axis = axis if isinstance(axis, int) else 0
                                if split_sizes:
                                    results = []
                                    start = 0
                                    for size in split_sizes:
                                        end = start + size
                                        sliced = inputs[0].slice(axis, start, end, 1)
                                        results.append(sliced)
                                        start = end
                                    outputs = results
                                else:
                                    dim_size = inputs[0].shape[axis]
                                    half = dim_size // 2
                                    outputs = [
                                        inputs[0].slice(axis, 0, half, 1),
                                        inputs[0].slice(axis, half, dim_size, 1),
                                    ]

                        elif op_type == "Softmax":
                            if inputs[0] is not None:
                                axis = layer_type[1] if isinstance(layer_type, tuple) else -1
                                outputs = [fastnn.softmax(inputs[0], axis)]

                        elif op_type == "Flatten":
                            if inputs[0] is not None:
                                axis = layer_type[1] if isinstance(layer_type, tuple) else 1
                                shape = inputs[0].shape
                                new_shape = [shape[0], -1] if axis == 1 else [-1]
                                outputs = [inputs[0].view(new_shape)]

                        elif op_type == "Relu":
                            if inputs[0] is not None:
                                outputs = [fastnn.relu(inputs[0])]

                        elif op_type == "Unsqueeze":
                            if inputs[0] is not None:
                                import numpy as np
                                x_np = inputs[0].numpy()
                                result = np.expand_dims(x_np, axis=0)
                                outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]

                        elif op_type == "Shape":
                            if inputs[0] is not None:
                                import numpy as np
                                shape = np.array(inputs[0].shape, dtype=np.int64)
                                outputs = [fastnn.tensor(shape.flatten().tolist(), list(shape.shape))]

                        elif op_type == "Gather":
                            if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                                import numpy as np
                                x_np = inputs[0].numpy()
                                indices = inputs[1].numpy().astype(int)
                                result = np.take(x_np, indices)
                                outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]

                        elif op_type == "Cast":
                            if inputs[0] is not None:
                                outputs = [inputs[0]]

                        elif op_type == "Expand":
                            if len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None:
                                import numpy as np
                                x_np = inputs[0].numpy()
                                target_shape = inputs[1].numpy().astype(int).tolist()
                                result = np.broadcast_to(x_np, target_shape).copy()
                                outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]

                        elif op_type == "ConstantOfShape":
                            if len(inputs) >= 1 and inputs[0] is not None:
                                import numpy as np
                                shape = inputs[0].numpy().astype(int).tolist()
                                result = np.full(shape, 0.0, dtype=np.float32)
                                outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]

                        elif op_type == "Range":
                            if len(inputs) >= 3 and inputs[0] is not None and inputs[1] is not None and inputs[2] is not None:
                                import numpy as np
                                start = int(inputs[0].numpy()[0])
                                end = int(inputs[1].numpy()[0])
                                step = int(inputs[2].numpy()[0])
                                result = np.arange(start, end, step, dtype=np.int64)
                                outputs = [fastnn.tensor(result.flatten().tolist(), list(result.shape))]

                        elif op_type == "Resize":
                            if inputs[0] is not None:
                                outputs = [inputs[0]]

                    except Exception as e:
                        pass
                    
                    for i, out_name in enumerate(node.output):
                        if i < len(outputs):
                            cache[out_name] = outputs[i]
            
            except Exception as e:
                logger.warning(f"Graph execution error: {e}")
            
            output_name = self.onnx_model.graph.output[0].name
            return cache.get(output_name)
        
        def __call__(self, x):
            return self.forward(x)
    
    return ONNXModel()