"""ONNX model importer for fastnn.

Converts ONNX models (.onnx) to fastnn's native format.
Supports common operators: Conv, Gemm, Relu, BatchNormalization, MaxPool, etc.
Also builds a runnable computation graph for supported models.
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
    """Concat module for fastnn - supports concat operation in computation graph."""
    
    def __init__(self, axis=1):
        self.axis = axis
    
    def __call__(self, *tensors):
        return self.forward(*tensors)
    
    def forward(self, *tensors):
        return fastnn.cat(list(tensors), dim=self.axis)
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []


class Split:
    """Split module for fastnn - splits tensor along an axis."""
    
    def __init__(self, axis=0, split_sizes=None):
        self.axis = axis
        self.split_sizes = split_sizes
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        import fastnn
        
        if self.split_sizes:
            # Split with specified sizes
            results = []
            start = 0
            for size in self.split_sizes:
                end = start + int(size)
                # Use fastnn's slice method
                sliced = x.slice(self.axis, start, end, 1)
                results.append(sliced)
                start = end
            return results
        else:
            # Equal split - split into 2 equal parts
            shape = x.shape
            dim_size = shape[self.axis]
            half = dim_size // 2
            r1 = x.slice(self.axis, 0, half, 1)
            r2 = x.slice(self.axis, half, dim_size, 1)
            return [r1, r2]
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []


class Softmax:
    """Softmax module for fastnn."""
    
    def __init__(self, axis=-1):
        self.axis = axis
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        import fastnn
        # Numerical stability: subtract max
        max_vals = fastnn.max(x, dim=self.axis, keepdim=True)
        exp_x = fastnn.exp(x - max_vals)
        sum_exp = fastnn.sum(exp_x, dim=self.axis, keepdim=True)
        return exp_x / sum_exp
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []


class Resize:
    """Resize (Upsample) module for fastnn."""
    
    def __init__(self, scale_factor=None, mode="nearest"):
        self.scale_factor = scale_factor
        self.mode = mode
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        import fastnn
        # Use fastnn's Upsample if available
        if hasattr(fastnn, 'Upsample'):
            if self.scale_factor:
                upsample = fastnn.Upsample(scale_factor=self.scale_factor, mode=self.mode)
                return upsample(x)
        # Fallback: return as-is (resize not implemented in pure Python)
        logger.warning("Resize not fully supported - returning input unchanged")
        return x
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []


class Slice:
    """Slice module for fastnn."""
    
    def __init__(self, starts=None, ends=None, axes=None, steps=None):
        self.starts = starts or [0]
        self.ends = ends or [None]
        self.axes = axes or [0]
        self.steps = steps or [1]
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        # Basic slicing support
        if len(self.starts) == 1 and len(self.axes) == 1:
            axis = self.axes[0]
            start = self.starts[0]
            end = self.ends[0]
            step = self.steps[0] if self.steps else 1
            
            if axis == 0:
                return x[start:end:step]
            elif axis == 1:
                return x[:, start:end:step]
            elif axis == 2:
                return x[:, :, start:end:step]
            elif axis == 3:
                return x[:, :, :, start:end:step]
        return x
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []


class ElementwiseSub:
    """Elementwise subtraction module."""
    
    def __call__(self, a, b):
        return a - b
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []


class ElementwiseDiv:
    """Elementwise division module."""
    
    def __call__(self, a, b):
        return a / b
    
    def parameters(self):
        return []
    
    def named_parameters(self):
        return []


# Import fastnn for use in the classes above
try:
    import fastnn
except ImportError:
    pass


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


def _get_value_info(model: onnx.ModelProto, name: str) -> Optional[onnx.ValueInfoProto]:
    """Get value info by name."""
    for vi in model.graph.value_info:
        if vi.name == name:
            return vi
    for inp in model.graph.input:
        if inp.name == name:
            return inp
    for out in model.graph.output:
        if out.name == name:
            return out
    return None


def _get_attr(node: onnx.NodeProto, name: str, default=None):
    """Get attribute value from node."""
    for attr in node.attribute:
        if attr.name == name:
            if attr.HasField("f"):
                return attr.f
            elif attr.HasField("i"):
                return attr.i
            elif attr.HasField("s"):
                return attr.s.decode("utf-8")
            elif attr.ints:
                return list(attr.ints)
            elif attr.floats:
                return list(attr.floats)
    return default


def import_onnx(onnx_path: str, fnn_path: str) -> Dict[str, Any]:
    """Import an ONNX model and save it in fastnn format.

    Args:
        onnx_path: Path to .onnx file
        fnn_path: Path to output .fnn file

    Returns:
        Dictionary with model info (layers, input_shape, output_shape)
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

    # Track which tensors have been consumed (to avoid duplicate layer exports)

    for node in model.graph.node:
        op_type = node.op_type
        layer_info = {"name": node.name or f"{op_type}_{layer_index}", "type": op_type}
        layer_index += 1

        if op_type == "Conv":
            weight = _get_initializer(model, node.input[1])
            bias = (
                _get_initializer(model, node.input[2]) if len(node.input) > 2 else None
            )

            out_channels, in_channels = weight.shape[:2]
            kernel_h, _kernel_w = weight.shape[2], weight.shape[3]
            stride = _get_attr(node, "strides", [1, 1])
            padding = _get_attr(node, "pads", [0, 0, 0, 0])
            dilation = _get_attr(node, "dilations", [1, 1])
            groups = _get_attr(node, "group", 1)

            layer_info["in_channels"] = in_channels
            layer_info["out_channels"] = out_channels
            layer_info["kernel_size"] = kernel_h
            layer_info["stride"] = stride[0] if isinstance(stride, list) else stride
            layer_info["padding"] = padding[0] if isinstance(padding, list) else padding
            layer_info["dilation"] = (
                dilation[0] if isinstance(dilation, list) else dilation
            )
            layer_info["groups"] = groups
            layer_info["bias"] = bias is not None

            parameters.append((f"{node.name}.weight", weight))
            if bias is not None:
                parameters.append((f"{node.name}.bias", bias))

        elif op_type == "Gemm":
            weight = _get_initializer(model, node.input[1])
            bias = (
                _get_initializer(model, node.input[2]) if len(node.input) > 2 else None
            )

            _get_attr(node, "alpha", 1.0)
            _get_attr(node, "beta", 1.0)
            _get_attr(node, "transA", 0)
            trans_b = _get_attr(node, "transB", 1)

            if trans_b:
                weight = weight.T

            in_features = weight.shape[0]
            out_features = weight.shape[1]

            layer_info["type"] = "Linear"
            layer_info["in_features"] = in_features
            layer_info["out_features"] = out_features
            layer_info["bias"] = bias is not None

            parameters.append((f"{node.name}.weight", weight))
            if bias is not None:
                parameters.append((f"{node.name}.bias", bias))

        elif op_type == "Relu":
            pass  # No parameters

        elif op_type == "BatchNormalization":
            scale = _get_initializer(model, node.input[1])
            bias = _get_initializer(model, node.input[2])
            mean = _get_initializer(model, node.input[3])
            var = _get_initializer(model, node.input[4])

            layer_info["type"] = "BatchNorm2d"
            layer_info["num_features"] = scale.shape[0]
            layer_info["eps"] = _get_attr(node, "epsilon", 1e-5)
            layer_info["momentum"] = 1.0 - _get_attr(node, "momentum", 0.9)

            parameters.append((f"{node.name}.weight", scale))
            parameters.append((f"{node.name}.bias", bias))
            parameters.append((f"{node.name}.running_mean", mean))
            parameters.append((f"{node.name}.running_var", var))

        elif op_type == "MaxPool":
            kernel = _get_attr(node, "kernel_shape", [2, 2])
            stride = _get_attr(node, "strides", kernel)
            padding = _get_attr(node, "pads", [0, 0, 0, 0])

            layer_info["kernel_size"] = (
                kernel[0] if isinstance(kernel, list) else kernel
            )
            layer_info["stride"] = stride[0] if isinstance(stride, list) else stride
            layer_info["padding"] = padding[0] if isinstance(padding, list) else padding

        elif op_type == "AveragePool":
            kernel = _get_attr(node, "kernel_shape", [2, 2])
            stride = _get_attr(node, "strides", kernel)
            layer_info["type"] = "AvgPool"
            layer_info["kernel_size"] = (
                kernel[0] if isinstance(kernel, list) else kernel
            )
            layer_info["stride"] = stride[0] if isinstance(stride, list) else stride

        elif op_type == "GlobalAveragePool":
            layer_info["type"] = "GlobalAvgPool"

        elif op_type == "Flatten":
            axis = _get_attr(node, "axis", 1)
            layer_info["axis"] = axis

        elif op_type == "Add":
            # Check if it's a bias add (one input is a constant)
            bias = _get_initializer(model, node.input[1])
            if bias is not None:
                layer_info["type"] = "BiasAdd"
                parameters.append((f"{node.name}.bias", bias))
            else:
                layer_info["type"] = "ElementwiseAdd"

        elif op_type == "Mul":
            layer_info["type"] = "ElementwiseMul"

        elif op_type == "MatMul":
            layer_info["type"] = "MatMul"

        elif op_type == "Concat":
            axis = _get_attr(node, "axis", 1)
            layer_info["axis"] = axis

        elif op_type == "Reshape":
            shape = _get_initializer(model, node.input[1])
            if shape is not None:
                layer_info["shape"] = shape.tolist()

        elif op_type == "Transpose":
            perm = _get_attr(node, "perm")
            if perm:
                layer_info["perm"] = perm

        elif op_type == "Sigmoid":
            pass

        elif op_type == "Tanh":
            pass

        elif op_type == "LeakyRelu":
            layer_info["alpha"] = _get_attr(node, "alpha", 0.01)

        elif op_type == "Elu":
            layer_info["alpha"] = _get_attr(node, "alpha", 1.0)

        elif op_type == "Clip":
            layer_info["min"] = _get_attr(node, "min", None)
            layer_info["max"] = _get_attr(node, "max", None)

        elif op_type == "Dropout":
            layer_info["ratio"] = _get_attr(node, "ratio", 0.5)

        elif op_type == "LRN":
            layer_info["size"] = _get_attr(node, "size", 5)
            layer_info["alpha"] = _get_attr(node, "alpha", 0.0001)
            layer_info["beta"] = _get_attr(node, "beta", 0.75)

        elif op_type == "InstanceNormalization":
            scale = _get_initializer(model, node.input[1])
            bias = _get_initializer(model, node.input[2])
            layer_info["type"] = "InstanceNorm"
            layer_info["num_features"] = scale.shape[0]
            layer_info["eps"] = _get_attr(node, "epsilon", 1e-5)
            parameters.append((f"{node.name}.weight", scale))
            parameters.append((f"{node.name}.bias", bias))
        
        elif op_type == "Split":
            axis = _get_attr(node, "axis", 0)
            split = _get_attr(node, "split", None)
            layer_info["type"] = "Split"
            layer_info["axis"] = axis
            layer_info["split"] = split
        
        elif op_type == "Softmax":
            axis = _get_attr(node, "axis", -1)
            layer_info["type"] = "Softmax"
            layer_info["axis"] = axis
        
        elif op_type == "Resize":
            mode = _get_attr(node, "mode", "nearest")
            layer_info["type"] = "Resize"
            layer_info["mode"] = mode
            # Get scales from initializer if available
            if len(node.input) > 2:
                scales = _get_initializer(model, node.input[2])
                if scales is not None:
                    layer_info["scales"] = scales.tolist()
        
        elif op_type == "Slice":
            layer_info["type"] = "Slice"
            # Get slice parameters from initializers
            if len(node.input) > 1:
                starts = _get_initializer(model, node.input[1])
                if starts is not None:
                    layer_info["starts"] = starts.tolist()
            if len(node.input) > 2:
                ends = _get_initializer(model, node.input[2])
                if ends is not None:
                    layer_info["ends"] = ends.tolist()
            if len(node.input) > 3:
                axes = _get_initializer(model, node.input[3])
                if axes is not None:
                    layer_info["axes"] = axes.tolist()
            if len(node.input) > 4:
                steps = _get_initializer(model, node.input[4])
                if steps is not None:
                    layer_info["steps"] = steps.tolist()
        
        elif op_type == "Sub":
            layer_info["type"] = "Sub"
        
        elif op_type == "Div":
            layer_info["type"] = "Div"
        
        else:
            logger.warning(f"Unsupported operator: {op_type}")
            layer_info["type"] = f"Unsupported_{op_type}"
        
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

    # Build computation graph for supported models
    model = _build_computation_graph(model, layers)
    
    result = {
        "layers": layers,
        "parameters": len(parameters),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "model": model,
    }
    return result


def _build_computation_graph(onnx_model, layers):
    """Build a runnable computation graph from ONNX layers."""
    import fastnn
    
    # Map node names to their layer info and outputs
    node_map = {}
    for node in onnx_model.graph.node:
        node_map[node.name or f"{node.op_type}_{id(node)}"] = {
            "node": node,
            "outputs": list(node.output),
            "inputs": list(node.input),
        }
    
    # Build layer instances
    layer_instances = {}
    for layer_info in layers:
        name = layer_info["name"]
        op_type = layer_info["type"]
        
        if op_type == "Conv":
            layer = fastnn.Conv2d(
                in_channels=layer_info["in_channels"],
                out_channels=layer_info["out_channels"],
                kernel_size=layer_info["kernel_size"],
                stride=layer_info["stride"],
                padding=layer_info["padding"],
                groups=layer_info.get("groups", 1),
                bias=layer_info["bias"],
            )
            layer_instances[name] = layer
        
        elif op_type == "BatchNorm2d":
            layer = fastnn.BatchNorm2d(
                num_features=layer_info["num_features"],
                eps=layer_info["eps"],
            )
            layer_instances[name] = layer
        
        elif op_type == "ReLU" or op_type == "Relu":
            layer = fastnn.ReLU()
            layer_instances[name] = layer
        
        elif op_type == "Sigmoid":
            layer = fastnn.Sigmoid()
            layer_instances[name] = layer
        
        elif op_type == "SiLU" or op_type == "Silu":
            if hasattr(fastnn, 'SiLU'):
                layer = fastnn.SiLU()
            else:
                layer = fastnn.Sigmoid()  # Fallback
            layer_instances[name] = layer
        
        elif op_type == "Concat":
            axis = layer_info.get("axis", 1)
            layer = Concat(axis=axis)
            layer_instances[name] = layer
        
        elif op_type == "Split":
            axis = layer_info.get("axis", 0)
            split_sizes = layer_info.get("split")
            layer = Split(axis=axis, split_sizes=split_sizes)
            layer_instances[name] = layer
        
        elif op_type == "Softmax":
            axis = layer_info.get("axis", -1)
            layer = Softmax(axis=axis)
            layer_instances[name] = layer
        
        elif op_type == "Resize":
            mode = layer_info.get("mode", "nearest")
            scales = layer_info.get("scales")
            scale_factor = None
            if scales and len(scales) >= 2:
                scale_factor = scales[-1]  # Use last dimension scale
            layer = Resize(scale_factor=scale_factor, mode=mode)
            layer_instances[name] = layer
        
        elif op_type == "Slice":
            starts = layer_info.get("starts", [0])
            ends = layer_info.get("ends", [None])
            axes = layer_info.get("axes", [0])
            steps = layer_info.get("steps", [1])
            layer = Slice(starts=starts, ends=ends, axes=axes, steps=steps)
            layer_instances[name] = layer
        
        elif op_type == "Add" or op_type == "ElementwiseAdd":
            layer = fastnn.ReLU()  # Placeholder - Add is handled in forward
            layer_instances[name] = layer
        
        elif op_type == "Mul" or op_type == "ElementwiseMul":
            layer = fastnn.ReLU()  # Placeholder - Mul is handled in forward
            layer_instances[name] = layer
        
        elif op_type == "Sub":
            layer = ElementwiseSub()
            layer_instances[name] = layer
        
        elif op_type == "Div":
            layer = ElementwiseDiv()
            layer_instances[name] = layer
        
        elif op_type == "MaxPool":
            kernel_size = layer_info.get("kernel_size", 2)
            stride = layer_info.get("stride", kernel_size)
            padding = layer_info.get("padding", 0)
            if hasattr(fastnn, 'MaxPool2d'):
                layer = fastnn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                layer = None
            layer_instances[name] = layer
        
        elif op_type == "Reshape":
            layer = None  # Reshape handled in forward
            layer_instances[name] = layer
        
        elif op_type == "Transpose":
            layer = None  # Transpose handled in forward
            layer_instances[name] = layer
        
        elif op_type == "MatMul":
            layer = None  # MatMul handled in forward
            layer_instances[name] = layer
        
        elif "Unsupported" in op_type:
            layer_instances[name] = None  # Unsupported ops
        
        else:
            layer_instances[name] = None
    
    # Create a model class that can execute the graph
    class ONNXModel:
        def __init__(self):
            self.layer_instances = layer_instances
            self.node_map = node_map
            self.tensor_cache = {}
        
        def forward(self, x):
            """Execute the ONNX graph."""
            import fastnn
            import numpy as np
            
            # Get input name
            input_name = onnx_model.graph.input[0].name
            self.tensor_cache[input_name] = x
            
            # Process nodes in order
            for node in onnx_model.graph.node:
                node_name = node.name or f"{node.op_type}_{id(node)}"
                layer = self.layer_instances.get(node_name)
                
                # Get input tensors
                input_tensors = []
                for inp in node.input:
                    if inp in self.tensor_cache:
                        input_tensors.append(self.tensor_cache[inp])
                    else:
                        # Try to get from initializer
                        init = _get_initializer(onnx_model, inp)
                        if init is not None:
                            input_tensors.append(fastnn.tensor(init.flatten().tolist(), list(init.shape)))
                
                # Process based on op type
                op_type = node.op_type
                output_tensors = []
                
                if op_type == "Conv":
                    if layer and input_tensors:
                        output_tensors = [layer(input_tensors[0])]
                
                elif op_type == "BatchNormalization":
                    if layer and input_tensors:
                        output_tensors = [layer(input_tensors[0])]
                
                elif op_type in ["Relu", "Sigmoid", "LeakyRelu", "Elu"]:
                    if input_tensors:
                        if op_type == "Relu":
                            output_tensors = [fastnn.relu(input_tensors[0])]
                        elif op_type == "Sigmoid":
                            output_tensors = [fastnn.sigmoid(input_tensors[0])]
                        elif op_type == "LeakyRelu":
                            output_tensors = [fastnn.leaky_relu(input_tensors[0])]
                        elif op_type == "Elu":
                            output_tensors = [fastnn.elu(input_tensors[0])]
                
                elif op_type == "Concat":
                    if layer and len(input_tensors) >= 2:
                        output_tensors = [layer(*input_tensors)]
                
                elif op_type == "Split":
                    if layer and input_tensors:
                        result = layer(input_tensors[0])
                        if isinstance(result, list):
                            output_tensors = result
                        else:
                            output_tensors = [result]
                
                elif op_type == "Add":
                    if len(input_tensors) >= 2:
                        output_tensors = [input_tensors[0] + input_tensors[1]]
                
                elif op_type == "Mul":
                    if len(input_tensors) >= 2:
                        output_tensors = [input_tensors[0] * input_tensors[1]]
                
                elif op_type == "Sub":
                    if len(input_tensors) >= 2:
                        output_tensors = [input_tensors[0] - input_tensors[1]]
                
                elif op_type == "Div":
                    if len(input_tensors) >= 2:
                        output_tensors = [input_tensors[0] / input_tensors[1]]
                
                elif op_type == "MaxPool":
                    if layer and input_tensors:
                        output_tensors = [layer(input_tensors[0])]
                
                elif op_type == "Reshape":
                    if len(input_tensors) >= 2:
                        new_shape = input_tensors[1].numpy().astype(int).tolist()
                        output_tensors = [input_tensors[0].view(new_shape)]
                
                elif op_type == "Transpose":
                    if input_tensors:
                        perm = _get_attr(node, "perm")
                        if perm:
                            output_tensors = [input_tensors[0].permute(perm)]
                        else:
                            output_tensors = [input_tensors[0].transpose(0, 1)]
                
                elif op_type == "MatMul":
                    if len(input_tensors) >= 2:
                        output_tensors = [fastnn.matmul(input_tensors[0], input_tensors[1])]
                
                elif op_type == "Softmax":
                    if layer and input_tensors:
                        output_tensors = [layer(input_tensors[0])]
                
                elif op_type == "Resize":
                    if layer and input_tensors:
                        output_tensors = [layer(input_tensors[0])]
                
                elif op_type == "Slice":
                    if layer and input_tensors:
                        output_tensors = [layer(input_tensors[0])]
                
                elif op_type == "Flatten":
                    if input_tensors:
                        axis = _get_attr(node, "axis", 1)
                        shape = input_tensors[0].shape
                        if axis == 1:
                            new_shape = [shape[0], -1]
                        else:
                            new_shape = [-1]
                        output_tensors = [input_tensors[0].view(new_shape)]
                
                elif "Unsupported" in op_type:
                    logger.warning(f"Skipping unsupported op: {op_type}")
                    continue
                
                # Cache outputs
                for i, out_name in enumerate(node.output):
                    if i < len(output_tensors):
                        self.tensor_cache[out_name] = output_tensors[i]
            
            # Return final output
            output_name = onnx_model.graph.output[0].name
            return self.tensor_cache.get(output_name)
        
        def __call__(self, x):
            return self.forward(x)
    
    return ONNXModel()
