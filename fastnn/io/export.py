import torch
from torch.nn import Sequential, ModuleList, ModuleDict, ParameterList, ParameterDict
import json
import logging
import struct
from typing import Any, Tuple, Optional

from fastnn._common import first_or_self
from fastnn.io import write_tensor, read_tensor, MODEL_MAGIC, MODEL_VERSION, read_fnn_header, read_fnn_parameters

logger = logging.getLogger(__name__)

try:
    from torchvision.models.resnet import BasicBlock
except ImportError:
    BasicBlock = None

# Import fastnn for tensor creation
import fastnn as fnn

__all__ = ["export_pytorch_model", "save_fnn_model", "load_fnn_model"]


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


def add_prefixed_config(config, prefix):
    """Add config dictionary with prefix to layer_info."""
    result = {}
    for key, val in config.items():
        result[f"{prefix}_{key}"] = val
    return result


def create_conv2d_from_config(layer_info, prefix):
    """Create Conv2d layer from config with given prefix."""
    import fastnn as fnn
    return fnn.Conv2d(
        layer_info[f"{prefix}_in_channels"],
        layer_info[f"{prefix}_out_channels"],
        layer_info[f"{prefix}_kernel_size"],
        stride=layer_info.get(f"{prefix}_stride", 1),
        padding=layer_info.get(f"{prefix}_padding", 0),
        dilation=layer_info.get(f"{prefix}_dilation", 1),
        groups=layer_info.get(f"{prefix}_groups", 1),
        bias=layer_info.get(f"{prefix}_bias", False),
    )


def create_batchnorm_from_config(layer_info, prefix):
    """Create BatchNorm1d layer from config with given prefix."""
    import fastnn as fnn
    return fnn.BatchNorm1d(
        layer_info[f"{prefix}_num_features"],
        eps=layer_info.get(f"{prefix}_eps", 1e-5),
        momentum=layer_info.get(f"{prefix}_momentum", 0.1),
    )


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

    # Build set of BasicBlock names for O(1) lookup
    basicblock_names = set()
    for name, module in model.named_modules():
        if BasicBlock is not None and isinstance(module, BasicBlock):
            basicblock_names.add(name)

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
        # Check if this module is a sub-module of a BasicBlock using pre-computed set
        if name and not isinstance(module, BasicBlock if BasicBlock is not None else ()):
            parts = name.split(".")
            for i in range(len(parts) - 1, -1, -1):
                parent_name = ".".join(parts[:i])
                if parent_name in basicblock_names:
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
            # Use BatchNorm2d for consistency with onnx.py
            layer_info["type"] = "BatchNorm2d"
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
            layer_info.update(add_prefixed_config(conv1_config, "conv1"))

            # Store bn1 config using helper
            bn1_config = get_batchnorm_config(bn1)
            layer_info.update(add_prefixed_config(bn1_config, "bn1"))

            # Store conv2 config using helper
            conv2_config = get_conv2d_config(conv2)
            layer_info.update(add_prefixed_config(conv2_config, "conv2"))

            # Store bn2 config using helper
            bn2_config = get_batchnorm_config(bn2)
            layer_info.update(add_prefixed_config(bn2_config, "bn2"))

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
                            layer_info.update(add_prefixed_config(config, f"downsample_{i}"))
                            parameters.extend(extract_module_parameters(sub_mod, f"{name}.downsample.{i}", ["weight", "bias"]))
                        elif isinstance(sub_mod, torch.nn.BatchNorm2d):
                            layer_info[f"downsample_{i}_type"] = "BatchNorm2d"
                            config = get_batchnorm_config(sub_mod)
                            layer_info.update(add_prefixed_config(config, f"downsample_{i}"))
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

    # Write .fnn file with unified format
    with open(path, "wb") as f:
        # Write magic bytes and version
        f.write(MODEL_MAGIC)
        f.write(_pack_u32(MODEL_VERSION))

        # Write header metadata as length-prefixed JSON
        header_json = json.dumps(header, indent=2)
        header_bytes = header_json.encode("utf-8")
        f.write(_pack_u64(len(header_bytes)))
        f.write(header_bytes)

        # Write number of parameters
        f.write(_pack_u64(len(parameters)))

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

    This function deserializes a model saved in the native .fnn format.
    The .fnn file contains a JSON header with layer topology and
    binary parameter tensors.

    Args:
        path: Path to the .fnn file to load.

    Returns:
        A fastnn Sequential model with all layers reconstructed and
        parameters loaded.

    Raises:
        SerializationError: If the file is invalid or missing magic bytes.
        ValueError: If an unsupported layer type is encountered in the
            model header.
        FileNotFoundError: If the specified path does not exist.

    Example:
        >>> model = load_fnn_model("model.fnn")
        >>> output = model(input_tensor)
    """
    with open(path, "rb") as f:
        # Read file header
        magic, file_version, header, num_params = read_fnn_header(f)
        if magic != MODEL_MAGIC:
            raise SerializationError("Invalid .fnn file: missing magic bytes")

        # Read parameters
        tensors = read_fnn_parameters(f, num_params)

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
            elif ltype == "BatchNorm2d" or ltype == "BatchNorm1d":
                # Map both BatchNorm types to fastnn.BatchNorm1d
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
                # Create sub-layers for BasicBlock using factory functions
                conv1 = create_conv2d_from_config(layer_info, "conv1")
                bn1 = create_batchnorm_from_config(layer_info, "bn1")
                relu = fnn.ReLU()
                conv2 = create_conv2d_from_config(layer_info, "conv2")
                bn2 = create_batchnorm_from_config(layer_info, "bn2")

                # downsample (if present)
                downsample = None
                if "downsample" in layer_info:
                    downsample_layers = []
                    num_downsample_layers = layer_info.get("downsample_num_layers", 0)
                    for i in range(num_downsample_layers):
                        sub_type = layer_info.get(f"downsample_{i}_type")
                        if sub_type == "Conv2d":
                            sub_layer = create_conv2d_from_config(layer_info, f"downsample_{i}")
                            downsample_layers.append(sub_layer)
                        elif sub_type == "BatchNorm2d" or sub_type == "BatchNorm1d":
                            sub_layer = create_batchnorm_from_config(layer_info, f"downsample_{i}")
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
                                    elif parent_info[downsample_type_key] == "BatchNorm2d" or parent_info[downsample_type_key] == "BatchNorm1d":
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
            elif ltype == "BatchNorm1d" or ltype == "BatchNorm2d":
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
