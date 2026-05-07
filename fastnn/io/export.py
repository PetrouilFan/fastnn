import torch
import numpy as np
import json
import logging
import struct
from typing import Any, Tuple, Optional

from fastnn._common import first_or_self
from fastnn.io import write_tensor, read_tensor

logger = logging.getLogger(__name__)

try:
    from torchvision.models.resnet import BasicBlock
except ImportError:
    BasicBlock = None

# Import fastnn for tensor creation
import fastnn as fnn


def extract_tensor(module, attr_name):
    """Extract a tensor attribute from a module as numpy array."""
    tensor = getattr(module, attr_name, None)
    if tensor is not None:
        return tensor.detach().cpu().numpy()
    return None


def load_tensor_to_layer(tensors, name, layer, setter_method):
    """Load a tensor from dict and set it to a layer using the provided setter method."""
    arr = tensors.get(name)
    if arr is not None:
        setter_method(fnn.tensor(arr, arr.shape))
        return True
    return False


def extract_module_parameters(module, name, param_names):
    """Extract multiple parameters from a module."""
    result = []
    for param_name in param_names:
        arr = extract_tensor(module, param_name)
        if arr is not None:
            result.append((f"{name}.{param_name}", arr))
    return result


def get_conv2d_config(module):
    """Get Conv2d configuration as a dictionary."""
    return {
        "in_channels": module.in_channels,
        "out_channels": module.out_channels,
        "kernel_size": first_or_self(module.kernel_size),
        "stride": first_or_self(module.stride),
        "padding": first_or_self(module.padding),
        "dilation": first_or_self(module.dilation),
        "groups": module.groups,
        "bias": module.bias is not None,
    }


def get_batchnorm_config(module):
    """Get BatchNorm configuration as a dictionary."""
    return {
        "num_features": module.num_features,
        "eps": module.eps,
        "momentum": module.momentum,
    }


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
    processed_names = set()  # Track processed module names to skip sub-modules

    # Iterate over all named modules (including containers)
    # Build a mapping of module names to modules for parent lookup
    module_dict = dict(model.named_modules())

    for name, module in model.named_modules():
        # Skip empty name (root module) only if it's a container
        if not name:
            if isinstance(module, torch.nn.Sequential):
                continue
            # Process root module if it's a single layer (e.g., Linear, Conv2d)
            # Use 'layer.0' as the name for root module layers
            name = "layer.0"

        # Skip modules that were already processed (e.g., sub-modules of BasicBlock)
        if name in processed_names:
            continue

        # Skip BasicBlock sub-layers (they are handled by the BasicBlock export)
        # Check if this module is a sub-module of a BasicBlock
        # Skip if it's a direct sub-module (parent is BasicBlock) but not the BasicBlock itself
        if name and not isinstance(module, BasicBlock):
            parts = name.split(".")
            for i in range(len(parts) - 1, -1, -1):
                parent_name = ".".join(parts[:i])
                parent_module = module_dict.get(parent_name)
                if parent_name == "":
                    # Parent is the BasicBlock itself (root)
                    if parent_module is not None and isinstance(
                        parent_module, BasicBlock
                    ):
                        processed_names.add(name)
                        break
                elif parent_module is not None and isinstance(
                    parent_module, BasicBlock
                ):
                    processed_names.add(name)
                    break
            if name in processed_names:
                continue

        # Determine layer type
        module_type = type(module).__name__
        layer_info = {"name": name, "type": module_type}

        # Extract parameters and hyperparameters based on layer type
        if isinstance(module, torch.nn.Linear):
            layer_info["in_features"] = module.in_features
            layer_info["out_features"] = module.out_features
            layer_info["bias"] = module.bias is not None
            parameters.extend(extract_module_parameters(module, name, ["weight", "bias"]))

        elif isinstance(module, torch.nn.Conv2d):
            layer_info.update(get_conv2d_config(module))
            parameters.extend(extract_module_parameters(module, name, ["weight", "bias"]))

        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # Map BatchNorm2d to BatchNorm1d (fastnn only has BatchNorm1d)
            layer_info["type"] = "BatchNorm1d"
            layer_info.update(get_batchnorm_config(module))
            layer_info["affine"] = module.affine
            parameters.extend(extract_module_parameters(module, name, ["weight", "bias", "running_mean", "running_var"]))

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
            parameters.extend(extract_module_parameters(module, name, ["weight", "bias"]))

        elif isinstance(module, torch.nn.Embedding):
            layer_info["type"] = "Embedding"
            layer_info["num_embeddings"] = module.num_embeddings
            layer_info["embedding_dim"] = module.embedding_dim
            parameters.extend(extract_module_parameters(module, name, ["weight"]))

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
            layers.append(layer_info)
            # Add Flatten layer after AdaptiveAvgPool2d if next layer is Linear
            # This is needed for ResNet-like models where AdaptiveAvgPool2d output
            # is [batch, channels, 1, 1] but Linear expects [batch, channels]
            layers.append({"name": name + ".flatten", "type": "Flatten"})
            continue

        elif isinstance(module, torch.nn.Flatten):
            layer_info["type"] = "Flatten"
            layers.append(layer_info)
            continue

        elif isinstance(module, torch.nn.MaxPool2d):
            layer_info["type"] = "MaxPool2d"
            layer_info["kernel_size"] = first_or_self(module.kernel_size)
            layer_info["stride"] = first_or_self(module.stride)
            layer_info["padding"] = first_or_self(module.padding)
            layer_info["dilation"] = first_or_self(module.dilation)
            layer_info["ceil_mode"] = module.ceil_mode

        elif isinstance(module, torch.nn.Dropout):
            layer_info["type"] = "Dropout"
            layer_info["p"] = module.p

        elif BasicBlock is not None and isinstance(module, BasicBlock):
            # Handle BasicBlock (ResNet skip connection)
            layer_info["type"] = "BasicBlock"

            # Store configurations for sub-modules
            # Get the actual sub-modules from the module
            conv1 = module.conv1
            bn1 = module.bn1
            _relu = module.relu  # ReLU is applied inline, not as separate layer
            conv2 = module.conv2
            bn2 = module.bn2

            # Collect parameters for sub-layers using helper
            parameters.extend(extract_module_parameters(conv1, f"{name}.conv1", ["weight", "bias"]))
            parameters.extend(extract_module_parameters(bn1, f"{name}.bn1", ["weight", "bias", "running_mean", "running_var"]))
            parameters.extend(extract_module_parameters(conv2, f"{name}.conv2", ["weight", "bias"]))
            parameters.extend(extract_module_parameters(bn2, f"{name}.bn2", ["weight", "bias", "running_mean", "running_var"]))

            # Store conv1 config using helper
            conv1_config = get_conv2d_config(conv1)
            for key, val in conv1_config.items():
                layer_info[f"conv1_{key}"] = val

            # Store bn1 config using helper
            bn1_config = get_batchnorm_config(bn1)
            for key, val in bn1_config.items():
                layer_info[f"bn1_{key}"] = val

            # Store conv2 config using helper
            conv2_config = get_conv2d_config(conv2)
            for key, val in conv2_config.items():
                layer_info[f"conv2_{key}"] = val

            # Store bn2 config using helper
            bn2_config = get_batchnorm_config(bn2)
            for key, val in bn2_config.items():
                layer_info[f"bn2_{key}"] = val

            # Handle downsample
            if hasattr(module, "downsample") and module.downsample is not None:
                layer_info["downsample"] = f"{name}.downsample"
                try:
                    downsample_layers = list(module.downsample)  # type: ignore
                    layer_info["downsample_num_layers"] = len(downsample_layers)
                    for i, sub_mod in enumerate(downsample_layers):
                        if isinstance(sub_mod, torch.nn.Conv2d):
                            layer_info[f"downsample_{i}_type"] = "Conv2d"
                            config = get_conv2d_config(sub_mod)
                            for key, val in config.items():
                                layer_info[f"downsample_{i}_{key}"] = val
                            parameters.extend(extract_module_parameters(sub_mod, f"{name}.downsample.{i}", ["weight", "bias"]))
                        elif isinstance(sub_mod, torch.nn.BatchNorm2d):
                            layer_info[f"downsample_{i}_type"] = "BatchNorm2d"
                            config = get_batchnorm_config(sub_mod)
                            for key, val in config.items():
                                layer_info[f"downsample_{i}_{key}"] = val
                            parameters.extend(extract_module_parameters(sub_mod, f"{name}.downsample.{i}", ["weight", "bias", "running_mean", "running_var"]))
                except TypeError:
                    pass

            # Mark all sub-module names as processed
            processed_names.add(f"{name}.conv1")
            processed_names.add(f"{name}.bn1")
            processed_names.add(f"{name}.relu")
            processed_names.add(f"{name}.conv2")
            processed_names.add(f"{name}.bn2")
            if hasattr(module, "downsample") and module.downsample is not None:
                processed_names.add(f"{name}.downsample")
                try:
                    downsample_layers = list(module.downsample)  # type: ignore
                    for i in range(len(downsample_layers)):
                        processed_names.add(f"{name}.downsample.{i}")
                except TypeError:
                    pass

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

        # Write each parameter tensor using shared utility
        for name, arr in parameters:
            write_tensor(f, name, arr)

    logger.info(
        "Exported PyTorch model to %s with %d layers and %d parameters.",
        path,
        len(layers),
        len(parameters),
    )


def save_fnn_model(model, path: str) -> None:
    """Export a fastnn model to .fnn format (alias for export_pytorch_model)."""
    export_pytorch_model(model, path)


def get_param_setter(layer, param_name):
    """Get the appropriate setter method for a parameter."""
    if param_name == "weight":
        return getattr(layer, "set_weight", None)
    elif param_name == "bias":
        return getattr(layer, "set_bias", None)
    elif param_name == "running_mean":
        return getattr(layer, "set_running_mean", None)
    elif param_name == "running_var":
        return getattr(layer, "set_running_var", None)
    return None


def load_fnn_model(path: str) -> Any:
    """
    Load a .fnn model file and return a fastnn model ready for inference.
    """
    with open(path, "rb") as f:
        # Read header length
        header_len_bytes = f.read(4)
        if len(header_len_bytes) < 4:
            raise ValueError("Invalid .fnn file: missing header length")
        header_len = struct.unpack("<I", header_len_bytes)[0]
        # Read header JSON
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))

        # Pre-compute BasicBlock prefixes for O(1) sub-layer detection
        basicblock_prefixes = set()
        for layer_info in header["layers"]:
            if layer_info["type"] == "BasicBlock":
                basicblock_prefixes.add(layer_info["name"])

        # Build dictionary mapping layer names to layer info for O(1) lookup
        layer_info_dict = {}
        for layer_info in header["layers"]:
            layer_info_dict[layer_info["name"]] = layer_info

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
            elif ltype == "BasicBlock":
                # Create sub-layers for BasicBlock
                # conv1
                conv1 = fnn.Conv2d(
                    layer_info["conv1_in_channels"],
                    layer_info["conv1_out_channels"],
                    layer_info["conv1_kernel_size"],
                    stride=layer_info["conv1_stride"],
                    padding=layer_info["conv1_padding"],
                    dilation=layer_info["conv1_dilation"],
                    groups=layer_info["conv1_groups"],
                    bias=layer_info["conv1_bias"],
                )

                # bn1
                bn1 = fnn.BatchNorm1d(
                    layer_info["bn1_num_features"],
                    eps=layer_info["bn1_eps"],
                    momentum=layer_info["bn1_momentum"],
                )

                # relu
                relu = fnn.ReLU()

                # conv2
                conv2 = fnn.Conv2d(
                    layer_info["conv2_in_channels"],
                    layer_info["conv2_out_channels"],
                    layer_info["conv2_kernel_size"],
                    stride=layer_info["conv2_stride"],
                    padding=layer_info["conv2_padding"],
                    dilation=layer_info["conv2_dilation"],
                    groups=layer_info["conv2_groups"],
                    bias=layer_info["conv2_bias"],
                )

                # bn2
                bn2 = fnn.BatchNorm1d(
                    layer_info["bn2_num_features"],
                    eps=layer_info["bn2_eps"],
                    momentum=layer_info["bn2_momentum"],
                )

                # downsample (if present)
                downsample = None
                if "downsample" in layer_info:
                    downsample_layers = []
                    num_downsample_layers = layer_info.get("downsample_num_layers", 0)
                    for i in range(num_downsample_layers):
                        sub_type = layer_info.get(f"downsample_{i}_type")
                        if sub_type == "Conv2d":
                            sub_layer = fnn.Conv2d(
                                layer_info[f"downsample_{i}_in_channels"],
                                layer_info[f"downsample_{i}_out_channels"],
                                layer_info[f"downsample_{i}_kernel_size"],
                                stride=layer_info[f"downsample_{i}_stride"],
                                padding=layer_info[f"downsample_{i}_padding"],
                                dilation=layer_info[f"downsample_{i}_dilation"],
                                groups=layer_info[f"downsample_{i}_groups"],
                                bias=layer_info[f"downsample_{i}_bias"],
                            )
                            downsample_layers.append(sub_layer)
                        elif sub_type == "BatchNorm2d":
                            # Map BatchNorm2d to BatchNorm1d
                            sub_layer = fnn.BatchNorm1d(
                                layer_info[f"downsample_{i}_num_features"],
                                eps=layer_info[f"downsample_{i}_eps"],
                                momentum=layer_info[f"downsample_{i}_momentum"],
                            )
                            downsample_layers.append(sub_layer)

                    # Create Sequential for downsample
                    if downsample_layers:
                        downsample = fnn.Sequential(downsample_layers)

                # Create BasicBlock
                layer = fnn.BasicBlock(conv1, bn1, relu, conv2, bn2, downsample)
            else:
                raise ValueError(f"Unsupported layer type: {ltype}")
            layers.append(layer)

        # Create Sequential model
        # Create a set of layer names that are sub-layers of BasicBlocks
        sub_layer_names = set()
        for bb_prefix in basicblock_prefixes:
            sub_layer_names.add(f"{bb_prefix}.conv1")
            sub_layer_names.add(f"{bb_prefix}.bn1")
            sub_layer_names.add(f"{bb_prefix}.relu")
            sub_layer_names.add(f"{bb_prefix}.conv2")
            sub_layer_names.add(f"{bb_prefix}.bn2")
            sub_layer_names.add(f"{bb_prefix}.downsample")

        # Filter out sub-layers
        filtered_layers = []
        for layer_info, layer in zip(header["layers"], layers):
            if layer_info["name"] in sub_layer_names:
                continue
            filtered_layers.append(layer)

        model = fnn.Sequential(filtered_layers)

        # Load tensors using shared utility
        tensors = {}
        while True:
            try:
                name, arr = read_tensor(f)
                tensors[name] = arr
            except (struct.error, ValueError):
                break

        # Create a mapping from layer name to layer object in filtered_layers
        layer_name_to_layer = {}

        # Iterate over header["layers"] and filtered_layers to build the mapping
        filtered_idx = 0
        for header_idx, layer_info in enumerate(header["layers"]):
            ltype = layer_info["type"]
            layer_name = layer_info["name"]

            if ltype == "BasicBlock":
                # This is a BasicBlock layer
                if filtered_idx < len(filtered_layers):
                    bb_layer = filtered_layers[filtered_idx]
                    layer_name_to_layer[layer_name] = bb_layer

                    # Also map the sub-layers
                    layer_name_to_layer[f"{layer_name}.conv1"] = bb_layer.conv1
                    layer_name_to_layer[f"{layer_name}.bn1"] = bb_layer.bn1
                    layer_name_to_layer[f"{layer_name}.relu"] = bb_layer.relu
                    layer_name_to_layer[f"{layer_name}.conv2"] = bb_layer.conv2
                    layer_name_to_layer[f"{layer_name}.bn2"] = bb_layer.bn2
                    if bb_layer.downsample is not None:
                        layer_name_to_layer[f"{layer_name}.downsample"] = (
                            bb_layer.downsample
                        )
                        # Also map downsample sub-layers
                        if hasattr(bb_layer.downsample, "layers"):
                            for i, sub_layer in enumerate(bb_layer.downsample.layers):
                                layer_name_to_layer[f"{layer_name}.downsample.{i}"] = (
                                    sub_layer
                                )

                    filtered_idx += 1
            else:
                # Skip sub-layers of BasicBlock using pre-computed set
                is_sub_layer = any(
                    layer_name.startswith(bb_prefix + ".")
                    for bb_prefix in basicblock_prefixes
                )

                if not is_sub_layer:
                    # This is a top-level layer
                    if filtered_idx < len(filtered_layers):
                        layer_name_to_layer[layer_name] = filtered_layers[filtered_idx]
                        filtered_idx += 1

        # Now load parameters for each tensor using helper function
        for tensor_name, arr in tensors.items():
            # Extract the layer prefix (e.g., "layer1.0.conv1")
            parts = tensor_name.split(".")
            if len(parts) < 2:
                continue

            layer_prefix = ".".join(parts[:-1])
            param_name = parts[-1]

            # Find the layer object
            layer = layer_name_to_layer.get(layer_prefix)
            if layer is None:
                continue

            # Find the layer info using pre-computed dictionary
            layer_info = layer_info_dict.get(layer_prefix)

            # If not found in dict, it might be a sub-layer of a BasicBlock
            if layer_info is None:
                # Find the parent BasicBlock
                parent_name = None
                for i in range(len(parts) - 1, 0, -1):
                    candidate_parent = ".".join(parts[:i])
                    if candidate_parent in layer_info_dict and layer_info_dict[candidate_parent].get("type") == "BasicBlock":
                        parent_name = candidate_parent
                        break

                if parent_name is not None:
                    # This is a sub-layer of a BasicBlock
                    sub_layer_name = parts[-2] if len(parts) >= 2 else parts[-1]
                    ltype = "Unknown"

                    if "downsample" in parts:
                        try:
                            downsample_idx = int(sub_layer_name)
                            parent_info = layer_info_dict.get(parent_name)
                            if parent_info:
                                downsample_type_key = f"downsample_{downsample_idx}_type"
                                if downsample_type_key in parent_info:
                                    if parent_info[downsample_type_key] == "Conv2d":
                                        ltype = "Conv2d"
                                    elif parent_info[downsample_type_key] == "BatchNorm2d":
                                        ltype = "BatchNorm1d"
                        except ValueError:
                            pass
                    elif sub_layer_name in ["conv1", "conv2"]:
                        ltype = "Conv2d"
                    elif sub_layer_name in ["bn1", "bn2"]:
                        ltype = "BatchNorm1d"
                    elif sub_layer_name == "relu":
                        ltype = "ReLU"
                    elif sub_layer_name == "downsample":
                        ltype = "Sequential"
                    layer_info = {"name": layer_prefix, "type": ltype}

            if layer_info is None:
                continue

            ltype = layer_info["type"]

            if ltype == "Linear":
                if param_name == "weight":
                    weight_arr = tensors.get(f"{layer_prefix}.weight")
                    if weight_arr is not None:
                        weight_arr = weight_arr.T
                        layer.set_weight(fnn.tensor(weight_arr, weight_arr.shape))
                elif param_name == "bias":
                    load_tensor_to_layer(tensors, f"{layer_prefix}.bias", layer, layer.set_bias)
            elif ltype == "BasicBlock":
                # Determine which sub-layer this tensor belongs to
                parts = tensor_name.split(".")
                if len(parts) >= 2:
                    sub_layer_name = parts[-2]  # e.g., "conv1", "bn1", "conv2", "bn2"
                    sub_layer = None
                    if sub_layer_name == "conv1":
                        sub_layer = layer.conv1
                    elif sub_layer_name == "bn1":
                        sub_layer = layer.bn1
                    elif sub_layer_name == "conv2":
                        sub_layer = layer.conv2
                    elif sub_layer_name == "bn2":
                        sub_layer = layer.bn2
                    elif "downsample" in parts:
                        # Handle downsample sub-layers
                        try:
                            idx = int(parts[-2]) if parts[-2].isdigit() else -1
                            if idx >= 0 and layer.downsample and idx < len(layer.downsample.layers):
                                sub_layer = layer.downsample.layers[idx]
                        except (ValueError, IndexError):
                            pass

                    if sub_layer is not None:
                        setter = get_param_setter(sub_layer, param_name)
                        if setter:
                            setter(fnn.tensor(tensors[tensor_name], tensors[tensor_name].shape))
            elif ltype == "Conv2d":
                if param_name == "weight":
                    load_tensor_to_layer(tensors, f"{layer_prefix}.weight", layer, layer.set_weight)
                elif param_name == "bias":
                    load_tensor_to_layer(tensors, f"{layer_prefix}.bias", layer, layer.set_bias)
            elif ltype == "BatchNorm1d":
                if param_name == "weight":
                    load_tensor_to_layer(tensors, f"{layer_prefix}.weight", layer, layer.set_weight)
                elif param_name == "bias":
                    load_tensor_to_layer(tensors, f"{layer_prefix}.bias", layer, layer.set_bias)
                elif param_name == "running_mean":
                    load_tensor_to_layer(tensors, f"{layer_prefix}.running_mean", layer, layer.set_running_mean)
                elif param_name == "running_var":
                    load_tensor_to_layer(tensors, f"{layer_prefix}.running_var", layer, layer.set_running_var)
            # For other layers (ReLU, GELU, SiLU, Dropout) no parameters

        return model
