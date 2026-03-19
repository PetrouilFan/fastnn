import torch
import numpy as np
import json
import struct
from typing import Any, Tuple, Optional


def export_pytorch_model(
    model: torch.nn.Module, path: str, input_shape: Optional[Tuple[int, ...]] = None
) -> None:
    """
    Export a PyTorch model to .fnn format for fastnn inference.

    Args:
        model: torch.nn.Module (pretrained)
        path: output file path ending in .fnn
        input_shape: optional tuple for input shape metadata
    """
    # Ensure path ends with .fnn
    if not path.endswith(".fnn"):
        path += ".fnn"

    # Extract architecture and parameters
    layers = []
    parameters = []  # list of (name, tensor)

    # Iterate over all named modules (including containers)
    for name, module in model.named_modules():
        # Skip empty name (root module)
        if not name:
            continue

        # Determine layer type
        module_type = type(module).__name__
        layer_info = {"name": name, "type": module_type}

        # Extract parameters and hyperparameters based on layer type
        if isinstance(module, torch.nn.Linear):
            layer_info["in_features"] = module.in_features
            layer_info["out_features"] = module.out_features
            layer_info["bias"] = module.bias is not None
            if module.weight is not None:
                parameters.append(
                    (f"{name}.weight", module.weight.detach().cpu().numpy())
                )
            if module.bias is not None:
                parameters.append((f"{name}.bias", module.bias.detach().cpu().numpy()))

        elif isinstance(module, torch.nn.Conv2d):
            layer_info["in_channels"] = module.in_channels
            layer_info["out_channels"] = module.out_channels
            layer_info["kernel_size"] = (
                module.kernel_size[0]
                if isinstance(module.kernel_size, tuple)
                else module.kernel_size
            )
            layer_info["stride"] = (
                module.stride[0] if isinstance(module.stride, tuple) else module.stride
            )
            layer_info["padding"] = (
                module.padding[0]
                if isinstance(module.padding, tuple)
                else module.padding
            )
            layer_info["dilation"] = (
                module.dilation[0]
                if isinstance(module.dilation, tuple)
                else module.dilation
            )
            layer_info["groups"] = module.groups
            layer_info["bias"] = module.bias is not None
            if module.weight is not None:
                parameters.append(
                    (f"{name}.weight", module.weight.detach().cpu().numpy())
                )
            if module.bias is not None:
                parameters.append((f"{name}.bias", module.bias.detach().cpu().numpy()))

        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # Map BatchNorm2d to BatchNorm1d (fastnn only has BatchNorm1d)
            layer_info["type"] = "BatchNorm1d"
            layer_info["num_features"] = module.num_features
            layer_info["eps"] = module.eps
            layer_info["momentum"] = module.momentum
            layer_info["affine"] = module.affine
            if module.weight is not None:
                parameters.append(
                    (f"{name}.weight", module.weight.detach().cpu().numpy())
                )
            if module.bias is not None:
                parameters.append((f"{name}.bias", module.bias.detach().cpu().numpy()))
            # running_mean and running_var are also needed for inference
            if module.running_mean is not None:
                parameters.append(
                    (f"{name}.running_mean", module.running_mean.detach().cpu().numpy())
                )
            if module.running_var is not None:
                parameters.append(
                    (f"{name}.running_var", module.running_var.detach().cpu().numpy())
                )

        elif isinstance(module, torch.nn.LayerNorm):
            layer_info["type"] = "LayerNorm"
            # normalized_shape may be int or tuple
            if isinstance(module.normalized_shape, int):
                layer_info["normalized_shape"] = module.normalized_shape
            else:
                # assume single dimension
                layer_info["normalized_shape"] = module.normalized_shape[0]
            layer_info["eps"] = module.eps
            layer_info["elementwise_affine"] = module.elementwise_affine
            if module.weight is not None:
                parameters.append(
                    (f"{name}.weight", module.weight.detach().cpu().numpy())
                )
            if module.bias is not None:
                parameters.append((f"{name}.bias", module.bias.detach().cpu().numpy()))

        elif isinstance(module, torch.nn.Embedding):
            layer_info["type"] = "Embedding"
            layer_info["num_embeddings"] = module.num_embeddings
            layer_info["embedding_dim"] = module.embedding_dim
            if module.weight is not None:
                parameters.append(
                    (f"{name}.weight", module.weight.detach().cpu().numpy())
                )

        elif isinstance(module, torch.nn.ReLU):
            layer_info["type"] = "ReLU"

        elif isinstance(module, torch.nn.GELU):
            layer_info["type"] = "GELU"

        elif isinstance(module, torch.nn.SiLU):
            layer_info["type"] = "SiLU"

        elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
            layer_info["type"] = "AdaptiveAvgPool2d"
            # output size may be stored; we assume (1,1)
            layer_info["output_size"] = (1, 1)

            # Check if next layer is Linear (requires flatten)
            # This is needed for ResNet-like models where AdaptiveAvgPool2d output
            # is [batch, channels, 1, 1] but Linear expects [batch, channels]
            # We'll add a Flatten layer after AdaptiveAvgPool2d
            # For now, always add Flatten after AdaptiveAvgPool2d to be safe
            layers.append(layer_info)

            # Add Flatten layer
            flatten_info = {"name": name + ".flatten", "type": "Flatten"}
            layers.append(flatten_info)
            continue

        elif isinstance(module, torch.nn.MaxPool2d):
            layer_info["type"] = "MaxPool2d"
            layer_info["kernel_size"] = (
                module.kernel_size[0]
                if isinstance(module.kernel_size, tuple)
                else module.kernel_size
            )
            layer_info["stride"] = (
                module.stride[0] if isinstance(module.stride, tuple) else module.stride
            )
            layer_info["padding"] = (
                module.padding[0]
                if isinstance(module.padding, tuple)
                else module.padding
            )
            layer_info["dilation"] = (
                module.dilation[0]
                if isinstance(module.dilation, tuple)
                else module.dilation
            )
            layer_info["ceil_mode"] = module.ceil_mode

        elif isinstance(module, torch.nn.Dropout):
            layer_info["type"] = "Dropout"
            layer_info["p"] = module.p

        else:
            # Skip container types (like BasicBlock, Sequential) - their children are already exported
            # Only warn for actual layer types that are unsupported
            from torch.nn import (
                Sequential,
                ModuleList,
                ModuleDict,
                ParameterList,
                ParameterDict,
            )

            container_types = (
                Sequential,
                ModuleList,
                ModuleDict,
                ParameterList,
                ParameterDict,
            )

            if not isinstance(module, container_types):
                import warnings

                warnings.warn(
                    f"Unsupported layer type {type(module).__name__} ignored."
                )
            continue

        layers.append(layer_info)

    # Prepare header metadata
    header = {
        "layers": layers,
        "input_shape": input_shape,
        "total_parameters": len(parameters),
        "format_version": 1,
    }

    # Write .fnn file
    with open(path, "wb") as f:
        # Write header as JSON (UTF-8)
        header_json = json.dumps(header, indent=2)
        header_bytes = header_json.encode("utf-8")
        # Write length of header (4 bytes)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)

        # Write each parameter tensor as raw binary (float32, row-major)
        for name, arr in parameters:
            # Ensure float32
            arr = arr.astype(np.float32)
            # Write tensor name length and name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            # Write shape rank and dimensions
            shape = arr.shape
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<I", dim))
            # Write data length (number of elements)
            f.write(struct.pack("<I", arr.size))
            # Write raw data
            f.write(arr.tobytes())

    print(
        f"Exported PyTorch model to {path} with {len(layers)} layers and {len(parameters)} parameters."
    )


def load_fnn_model(path: str) -> Any:
    """
    Load a .fnn model file and return a fastnn model ready for inference.
    """
    import fastnn as fnn

    with open(path, "rb") as f:
        # Read header length
        header_len_bytes = f.read(4)
        if len(header_len_bytes) < 4:
            raise ValueError("Invalid .fnn file: missing header length")
        header_len = struct.unpack("<I", header_len_bytes)[0]
        # Read header JSON
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))

        # Reconstruct fastnn layers
        layers = []
        for layer_info in header["layers"]:
            ltype = layer_info["type"]
            name = layer_info["name"]

            if ltype == "Linear":
                layer = fnn.Linear(
                    layer_info["in_features"],
                    layer_info["out_features"],
                    layer_info["bias"],
                )
            elif ltype == "Conv2d":
                layer = fnn.Conv2d(
                    layer_info["in_channels"],
                    layer_info["out_channels"],
                    layer_info["kernel_size"],
                    stride=layer_info["stride"],
                    padding=layer_info["padding"],
                    dilation=layer_info["dilation"],
                    groups=layer_info["groups"],
                    bias=layer_info["bias"],
                )
            elif ltype == "BatchNorm1d":
                layer = fnn.BatchNorm1d(
                    layer_info["num_features"],
                    eps=layer_info["eps"],
                    momentum=layer_info["momentum"],
                )
            elif ltype == "LayerNorm":
                # fastnn LayerNorm expects normalized_shape as i64
                norm_shape = layer_info["normalized_shape"]
                if isinstance(norm_shape, (list, tuple)):
                    norm_shape = norm_shape[0]
                layer = fnn.LayerNorm(norm_shape, eps=layer_info.get("eps", 1e-5))
            elif ltype == "Embedding":
                layer = fnn.Embedding(
                    layer_info["num_embeddings"], layer_info["embedding_dim"]
                )
            elif ltype == "ReLU":
                layer = fnn.ReLU()
            elif ltype == "GELU":
                layer = fnn.GELU()
            elif ltype == "SiLU":
                layer = fnn.SiLU()
            elif ltype == "AdaptiveAvgPool2d":
                layer = fnn.AdaptiveAvgPool2d(layer_info.get("output_size", (1, 1)))
            elif ltype == "Flatten":
                layer = fnn.Flatten()
            elif ltype == "MaxPool2d":
                layer = fnn.MaxPool2d(
                    layer_info["kernel_size"],
                    stride=layer_info["stride"],
                    padding=layer_info["padding"],
                    dilation=layer_info["dilation"],
                    ceil_mode=layer_info.get("ceil_mode", False),
                )
            elif ltype == "Dropout":
                layer = fnn.Dropout(layer_info["p"])
            else:
                raise ValueError(f"Unsupported layer type: {ltype}")
            layers.append(layer)

        # Create Sequential model
        model = fnn.Sequential(layers)

        # Load parameters
        # Read tensors until EOF
        # We'll store tensors in a dict mapping from full name to numpy array
        tensors = {}
        while True:
            # Read name length
            name_len_bytes = f.read(4)
            if len(name_len_bytes) < 4:
                break  # EOF
            name_len = struct.unpack("<I", name_len_bytes)[0]
            name_bytes = f.read(name_len)
            if len(name_bytes) < name_len:
                break
            name = name_bytes.decode("utf-8")

            # Read shape rank
            rank_bytes = f.read(4)
            if len(rank_bytes) < 4:
                break
            rank = struct.unpack("<I", rank_bytes)[0]
            shape = []
            for _ in range(rank):
                dim_bytes = f.read(4)
                if len(dim_bytes) < 4:
                    raise ValueError("Invalid .fnn file: incomplete shape")
                dim = struct.unpack("<I", dim_bytes)[0]
                shape.append(dim)

            # Read data length
            data_len_bytes = f.read(4)
            if len(data_len_bytes) < 4:
                raise ValueError("Invalid .fnn file: missing data length")
            data_len = struct.unpack("<I", data_len_bytes)[0]

            # Read raw data
            data_bytes = f.read(data_len * 4)  # float32 = 4 bytes
            if len(data_bytes) != data_len * 4:
                raise ValueError("Invalid .fnn file: incomplete tensor data")

            # Convert to numpy array
            arr = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
            tensors[name] = arr

        # Assign tensors to layers
        # Iterate over layers and match parameter names
        for layer_info, layer in zip(header["layers"], layers):
            layer_name = layer_info["name"]
            ltype = layer_info["type"]

            # Helper to get tensor and convert to fastnn tensor
            def get_tensor(suffix):
                key = f"{layer_name}.{suffix}"
                if key not in tensors:
                    return None
                arr = tensors[key]
                # Convert to flat list and shape
                return fnn.tensor(arr.flatten().tolist(), list(arr.shape))

            if ltype == "Linear":
                weight_arr = tensors.get(f"{layer_name}.weight")
                if weight_arr is not None:
                    # PyTorch Linear weight shape: (out_features, in_features)
                    # fastnn Linear weight shape: (in_features, out_features)
                    # Transpose
                    weight_arr = weight_arr.T
                    weight = fnn.tensor(
                        weight_arr.flatten().tolist(), list(weight_arr.shape)
                    )
                    layer.set_weight(weight)
                bias_arr = tensors.get(f"{layer_name}.bias")
                if bias_arr is not None:
                    bias = fnn.tensor(bias_arr.flatten().tolist(), list(bias_arr.shape))
                    layer.set_bias(bias)
            elif ltype == "Conv2d":
                weight = get_tensor("weight")
                bias = get_tensor("bias")
                if weight is not None:
                    layer.set_weight(weight)
                if bias is not None:
                    layer.set_bias(bias)
            elif ltype == "BatchNorm1d":
                weight = get_tensor("weight")
                bias = get_tensor("bias")
                running_mean = get_tensor("running_mean")
                running_var = get_tensor("running_var")
                if weight is not None:
                    layer.set_weight(weight)
                if bias is not None:
                    layer.set_bias(bias)
                if running_mean is not None:
                    layer.set_running_mean(running_mean)
                if running_var is not None:
                    layer.set_running_var(running_var)
            elif ltype == "LayerNorm":
                weight = get_tensor("weight")
                bias = get_tensor("bias")
                if weight is not None:
                    layer.set_weight(weight)
                if bias is not None:
                    layer.set_bias(bias)
            elif ltype == "Embedding":
                weight = get_tensor("weight")
                if weight is not None:
                    layer.set_weight(weight)
            # For other layers (ReLU, GELU, SiLU, Dropout) no parameters

        return model
