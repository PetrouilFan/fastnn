import torch
import numpy as np
import json
import struct
from typing import Any, Tuple, Optional

try:
    from torchvision.models.resnet import BasicBlock
except ImportError:
    BasicBlock = None


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

            # Collect parameters for sub-layers
            # conv1
            if conv1.weight is not None:
                parameters.append(
                    (f"{name}.conv1.weight", conv1.weight.detach().cpu().numpy())
                )
            if conv1.bias is not None:
                parameters.append(
                    (f"{name}.conv1.bias", conv1.bias.detach().cpu().numpy())
                )

            # bn1
            if bn1.weight is not None:
                parameters.append(
                    (f"{name}.bn1.weight", bn1.weight.detach().cpu().numpy())
                )
            if bn1.bias is not None:
                parameters.append((f"{name}.bn1.bias", bn1.bias.detach().cpu().numpy()))
            if bn1.running_mean is not None:
                parameters.append(
                    (
                        f"{name}.bn1.running_mean",
                        bn1.running_mean.detach().cpu().numpy(),
                    )
                )
            if bn1.running_var is not None:
                parameters.append(
                    (f"{name}.bn1.running_var", bn1.running_var.detach().cpu().numpy())
                )

            # conv2
            if conv2.weight is not None:
                parameters.append(
                    (f"{name}.conv2.weight", conv2.weight.detach().cpu().numpy())
                )
            if conv2.bias is not None:
                parameters.append(
                    (f"{name}.conv2.bias", conv2.bias.detach().cpu().numpy())
                )

            # bn2
            if bn2.weight is not None:
                parameters.append(
                    (f"{name}.bn2.weight", bn2.weight.detach().cpu().numpy())
                )
            if bn2.bias is not None:
                parameters.append((f"{name}.bn2.bias", bn2.bias.detach().cpu().numpy()))
            if bn2.running_mean is not None:
                parameters.append(
                    (
                        f"{name}.bn2.running_mean",
                        bn2.running_mean.detach().cpu().numpy(),
                    )
                )
            if bn2.running_var is not None:
                parameters.append(
                    (f"{name}.bn2.running_var", bn2.running_var.detach().cpu().numpy())
                )

            # downsample
            if hasattr(module, "downsample") and module.downsample is not None:
                try:
                    downsample_layers = list(module.downsample)  # type: ignore
                    for i, sub_mod in enumerate(downsample_layers):
                        if isinstance(sub_mod, torch.nn.Conv2d):
                            if sub_mod.weight is not None:
                                parameters.append(
                                    (
                                        f"{name}.downsample.{i}.weight",
                                        sub_mod.weight.detach().cpu().numpy(),
                                    )
                                )
                            if sub_mod.bias is not None:
                                parameters.append(
                                    (
                                        f"{name}.downsample.{i}.bias",
                                        sub_mod.bias.detach().cpu().numpy(),
                                    )
                                )
                        elif isinstance(sub_mod, torch.nn.BatchNorm2d):
                            if sub_mod.weight is not None:
                                parameters.append(
                                    (
                                        f"{name}.downsample.{i}.weight",
                                        sub_mod.weight.detach().cpu().numpy(),
                                    )
                                )
                            if sub_mod.bias is not None:
                                parameters.append(
                                    (
                                        f"{name}.downsample.{i}.bias",
                                        sub_mod.bias.detach().cpu().numpy(),
                                    )
                                )
                            if sub_mod.running_mean is not None:
                                parameters.append(
                                    (
                                        f"{name}.downsample.{i}.running_mean",
                                        sub_mod.running_mean.detach().cpu().numpy(),
                                    )
                                )
                            if sub_mod.running_var is not None:
                                parameters.append(
                                    (
                                        f"{name}.downsample.{i}.running_var",
                                        sub_mod.running_var.detach().cpu().numpy(),
                                    )
                                )
                except TypeError:
                    # Not iterable
                    pass

            # Store conv1 config
            layer_info["conv1_in_channels"] = conv1.in_channels
            layer_info["conv1_out_channels"] = conv1.out_channels
            layer_info["conv1_kernel_size"] = (
                conv1.kernel_size[0]
                if isinstance(conv1.kernel_size, tuple)
                else conv1.kernel_size
            )
            layer_info["conv1_stride"] = (
                conv1.stride[0] if isinstance(conv1.stride, tuple) else conv1.stride
            )
            layer_info["conv1_padding"] = (
                conv1.padding[0] if isinstance(conv1.padding, tuple) else conv1.padding
            )
            layer_info["conv1_dilation"] = (
                conv1.dilation[0]
                if isinstance(conv1.dilation, tuple)
                else conv1.dilation
            )
            layer_info["conv1_groups"] = conv1.groups
            layer_info["conv1_bias"] = conv1.bias is not None

            # Store bn1 config
            layer_info["bn1_num_features"] = bn1.num_features
            layer_info["bn1_eps"] = bn1.eps
            layer_info["bn1_momentum"] = bn1.momentum

            # Store conv2 config
            layer_info["conv2_in_channels"] = conv2.in_channels
            layer_info["conv2_out_channels"] = conv2.out_channels
            layer_info["conv2_kernel_size"] = (
                conv2.kernel_size[0]
                if isinstance(conv2.kernel_size, tuple)
                else conv2.kernel_size
            )
            layer_info["conv2_stride"] = (
                conv2.stride[0] if isinstance(conv2.stride, tuple) else conv2.stride
            )
            layer_info["conv2_padding"] = (
                conv2.padding[0] if isinstance(conv2.padding, tuple) else conv2.padding
            )
            layer_info["conv2_dilation"] = (
                conv2.dilation[0]
                if isinstance(conv2.dilation, tuple)
                else conv2.dilation
            )
            layer_info["conv2_groups"] = conv2.groups
            layer_info["conv2_bias"] = conv2.bias is not None

            # Store bn2 config
            layer_info["bn2_num_features"] = bn2.num_features
            layer_info["bn2_eps"] = bn2.eps
            layer_info["bn2_momentum"] = bn2.momentum

            # Check for downsample
            if hasattr(module, "downsample") and module.downsample is not None:
                print()
                layer_info["downsample"] = f"{name}.downsample"
                # Try to store downsample config (it's typically a Sequential)
                try:
                    downsample_layers = list(module.downsample)  # type: ignore
                    print()
                    layer_info["downsample_num_layers"] = len(downsample_layers)
                    # Store configurations for downsample sub-layers
                    for i, sub_mod in enumerate(downsample_layers):
                        if isinstance(sub_mod, torch.nn.Conv2d):
                            layer_info[f"downsample_{i}_type"] = "Conv2d"
                            layer_info[f"downsample_{i}_in_channels"] = (
                                sub_mod.in_channels
                            )
                            layer_info[f"downsample_{i}_out_channels"] = (
                                sub_mod.out_channels
                            )
                            layer_info[f"downsample_{i}_kernel_size"] = (
                                sub_mod.kernel_size[0]
                                if isinstance(sub_mod.kernel_size, tuple)
                                else sub_mod.kernel_size
                            )
                            layer_info[f"downsample_{i}_stride"] = (
                                sub_mod.stride[0]
                                if isinstance(sub_mod.stride, tuple)
                                else sub_mod.stride
                            )
                            layer_info[f"downsample_{i}_padding"] = (
                                sub_mod.padding[0]
                                if isinstance(sub_mod.padding, tuple)
                                else sub_mod.padding
                            )
                            layer_info[f"downsample_{i}_dilation"] = (
                                sub_mod.dilation[0]
                                if isinstance(sub_mod.dilation, tuple)
                                else sub_mod.dilation
                            )
                            layer_info[f"downsample_{i}_groups"] = sub_mod.groups
                            layer_info[f"downsample_{i}_bias"] = (
                                sub_mod.bias is not None
                            )
                        elif isinstance(sub_mod, torch.nn.BatchNorm2d):
                            layer_info[f"downsample_{i}_type"] = "BatchNorm2d"
                            layer_info[f"downsample_{i}_num_features"] = (
                                sub_mod.num_features
                            )
                            layer_info[f"downsample_{i}_eps"] = sub_mod.eps
                            layer_info[f"downsample_{i}_momentum"] = sub_mod.momentum
                except TypeError:
                    print()
                    pass

            # Mark all sub-module names as processed
            processed_names.add(f"{name}.conv1")
            processed_names.add(f"{name}.bn1")
            processed_names.add(f"{name}.relu")
            processed_names.add(f"{name}.conv2")
            processed_names.add(f"{name}.bn2")
            if hasattr(module, "downsample") and module.downsample is not None:
                processed_names.add(f"{name}.downsample")
                # Also mark downsample sub-modules
                # downsample is typically a Sequential (or iterable)
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
        # Note: The layers list already contains BasicBlock layers created above
        # We just need to filter out the sub-layers that belong to BasicBlocks
        # and keep only the top-level layers

        # For now, let's just use all layers as-is
        # The BasicBlock layers are already created with their sub-layers
        # and the sub-layers are in the layers list too
        # We need to filter out the sub-layers

        # Create a set of layer names that are sub-layers of BasicBlocks
        sub_layer_names = set()
        for layer_info in header["layers"]:
            if layer_info["type"] == "BasicBlock":
                name = layer_info["name"]
                sub_layer_names.add(f"{name}.conv1")
                sub_layer_names.add(f"{name}.bn1")
                sub_layer_names.add(f"{name}.relu")
                sub_layer_names.add(f"{name}.conv2")
                sub_layer_names.add(f"{name}.bn2")
                if "downsample" in layer_info:
                    sub_layer_names.add(f"{name}.downsample")

        # Filter out sub-layers
        filtered_layers = []
        for layer_info, layer in zip(header["layers"], layers):
            if layer_info["name"] in sub_layer_names:
                continue
            filtered_layers.append(layer)

        model = fnn.Sequential(filtered_layers)

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
        # We need to load parameters into the layer objects
        # The filtered_layers list contains the final model structure (BasicBlock layers + top-level layers)
        # We need to map tensor names to the correct layer objects

        # Create a mapping from layer name to layer object in filtered_layers
        # For BasicBlock layers, we also need to map sub-layers
        layer_name_to_layer = {}

        # Iterate over header["layers"] and filtered_layers to build the mapping
        # We need to handle BasicBlock sub-layers specially
        filtered_idx = 0
        for header_idx, layer_info in enumerate(header["layers"]):
            ltype = layer_info["type"]
            layer_name = layer_info["name"]

            if ltype == "BasicBlock":
                # This is a BasicBlock layer
                # The corresponding layer in filtered_layers is at filtered_idx
                if filtered_idx < len(filtered_layers):
                    bb_layer = filtered_layers[filtered_idx]
                    layer_name_to_layer[layer_name] = bb_layer

                    # Also map the sub-layers
                    # conv1, bn1, relu, conv2, bn2, downsample
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
                # This is a regular layer (or sub-layer of BasicBlock)
                # Skip sub-layers of BasicBlock
                is_sub_layer = False
                for bb_info in header["layers"]:
                    if bb_info["type"] == "BasicBlock":
                        bb_name = bb_info["name"]
                        if layer_name.startswith(bb_name + "."):
                            is_sub_layer = True
                            break

                if not is_sub_layer:
                    # This is a top-level layer
                    if filtered_idx < len(filtered_layers):
                        layer_name_to_layer[layer_name] = filtered_layers[filtered_idx]
                        filtered_idx += 1

        # Now load parameters for each tensor
        for tensor_name, arr in tensors.items():
            # tensor_name is like "layer1.0.conv1.weight"
            # Extract the layer prefix (e.g., "layer1.0.conv1")
            parts = tensor_name.split(".")
            if len(parts) < 2:
                continue

            # Find the layer name (everything except the last part, which is "weight" or "bias")
            layer_prefix = ".".join(parts[:-1])
            param_name = parts[-1]

            # Find the layer object
            layer = layer_name_to_layer.get(layer_prefix)

            if layer is None:
                continue

            # Find the layer info
            layer_info = None
            for li in header["layers"]:
                if li["name"] == layer_prefix:
                    layer_info = li
                    break

            # If not found in header, it might be a sub-layer of a BasicBlock
            if layer_info is None:
                # Find the parent BasicBlock
                # For sub-layers like "layer2.0.downsample.0", we need to find "layer2.0"
                # For sub-layers like "layer2.0.conv1", we need to find "layer2.0"
                parts = layer_prefix.split(".")
                parent_name = None
                for i in range(len(parts) - 1, 0, -1):
                    candidate_parent = ".".join(parts[:i])
                    for li in header["layers"]:
                        if (
                            li["name"] == candidate_parent
                            and li["type"] == "BasicBlock"
                        ):
                            parent_name = candidate_parent
                            break
                    if parent_name is not None:
                        break

                if parent_name is not None:
                    # This is a sub-layer of a BasicBlock
                    # The layer type is determined by the sub-layer name
                    parts = layer_prefix.split(".")
                    sub_layer_name = parts[-1]
                    ltype = "Unknown"  # Default value

                    # Check if this is a downsample sub-layer
                    if "downsample" in parts:
                        # For downsample sub-layers, we need to check the parent's downsample config
                        # Find the downsample index
                        try:
                            downsample_idx = int(sub_layer_name)
                            # Get the parent BasicBlock info to find the downsample type
                            for li in header["layers"]:
                                if (
                                    li["name"] == parent_name
                                    and li["type"] == "BasicBlock"
                                ):
                                    # Check the downsample type from the parent's config
                                    downsample_type_key = (
                                        f"downsample_{downsample_idx}_type"
                                    )
                                    if downsample_type_key in li:
                                        if li[downsample_type_key] == "Conv2d":
                                            ltype = "Conv2d"
                                        elif li[downsample_type_key] == "BatchNorm2d":
                                            ltype = "BatchNorm1d"
                                    break
                        except ValueError:
                            pass  # sub_layer_name is not an integer
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

            # Only process parameters for the current tensor
            # Extract the parameter name from the tensor name
            param_name = tensor_name.split(".")[-1]

            if ltype == "Linear":
                if param_name == "weight":
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
                        print(
                            f"DEBUG: Loaded weight for {layer_name}, shape: {weight_arr.shape}"
                        )
                elif param_name == "bias":
                    bias_arr = tensors.get(f"{layer_name}.bias")
                    if bias_arr is not None:
                        bias = fnn.tensor(
                            bias_arr.flatten().tolist(), list(bias_arr.shape)
                        )
                        layer.set_bias(bias)
            elif ltype == "Conv2d":
                if param_name == "weight":
                    weight = get_tensor("weight")
                    if weight is not None:
                        layer.set_weight(weight)
                elif param_name == "bias":
                    bias = get_tensor("bias")
                    if bias is not None:
                        layer.set_bias(bias)
            elif ltype == "BatchNorm1d":
                if param_name == "weight":
                    weight = get_tensor("weight")
                    if weight is not None:
                        layer.set_weight(weight)
                elif param_name == "bias":
                    bias = get_tensor("bias")
                    if bias is not None:
                        layer.set_bias(bias)
                elif param_name == "running_mean":
                    running_mean = get_tensor("running_mean")
                    if running_mean is not None:
                        layer.set_running_mean(running_mean)
                elif param_name == "running_var":
                    running_var = get_tensor("running_var")
                    if running_var is not None:
                        layer.set_running_var(running_var)
            elif ltype == "LayerNorm":
                if param_name == "weight":
                    weight = get_tensor("weight")
                    if weight is not None:
                        layer.set_weight(weight)
                elif param_name == "bias":
                    bias = get_tensor("bias")
                    if bias is not None:
                        layer.set_bias(bias)
            elif ltype == "Embedding":
                if param_name == "weight":
                    weight = get_tensor("weight")
                    if weight is not None:
                        layer.set_weight(weight)
            elif ltype == "BasicBlock":
                # Load parameters for all sub-layers of BasicBlock
                # The layer is a BasicBlock instance with conv1, bn1, relu, conv2, bn2, downsample
                layer_name = layer_info["name"]

                # Determine which sub-layer parameter to load based on param_name
                if param_name == "conv1.weight":
                    conv1_weight = tensors.get(f"{layer_name}.conv1.weight")
                    if conv1_weight is not None:
                        weight = fnn.tensor(
                            conv1_weight.flatten().tolist(), list(conv1_weight.shape)
                        )
                        layer.conv1.set_weight(weight)
                elif param_name == "conv1.bias":
                    conv1_bias = tensors.get(f"{layer_name}.conv1.bias")
                    if conv1_bias is not None:
                        bias = fnn.tensor(
                            conv1_bias.flatten().tolist(), list(conv1_bias.shape)
                        )
                        layer.conv1.set_bias(bias)
                elif param_name == "weight" and tensor_name.endswith(".bn1.weight"):
                    bn1_weight = tensors.get(f"{layer_name}.bn1.weight")
                    if bn1_weight is not None:
                        weight = fnn.tensor(
                            bn1_weight.flatten().tolist(), list(bn1_weight.shape)
                        )
                        layer.bn1.set_weight(weight)
                elif param_name == "bias" and tensor_name.endswith(".bn1.bias"):
                    bn1_bias = tensors.get(f"{layer_name}.bn1.bias")
                    if bn1_bias is not None:
                        bias = fnn.tensor(
                            bn1_bias.flatten().tolist(), list(bn1_bias.shape)
                        )
                        layer.bn1.set_bias(bias)
                elif param_name == "running_mean" and tensor_name.endswith(
                    ".bn1.running_mean"
                ):
                    bn1_running_mean = tensors.get(f"{layer_name}.bn1.running_mean")
                    if bn1_running_mean is not None:
                        running_mean = fnn.tensor(
                            bn1_running_mean.flatten().tolist(),
                            list(bn1_running_mean.shape),
                        )
                        layer.bn1.set_running_mean(running_mean)
                elif param_name == "running_var" and tensor_name.endswith(
                    ".bn1.running_var"
                ):
                    bn1_running_var = tensors.get(f"{layer_name}.bn1.running_var")
                    if bn1_running_var is not None:
                        print(f"DEBUG: Setting running_var for {layer_name}.bn1")
                        running_var = fnn.tensor(
                            bn1_running_var.flatten().tolist(),
                            list(bn1_running_var.shape),
                        )
                        layer.bn1.set_running_var(running_var)
                elif param_name == "running_var" and tensor_name.endswith(
                    ".bn1.running_var"
                ):
                    bn1_running_var = tensors.get(f"{layer_name}.bn1.running_var")
                    if bn1_running_var is not None:
                        running_var = fnn.tensor(
                            bn1_running_var.flatten().tolist(),
                            list(bn1_running_var.shape),
                        )
                        layer.bn1.set_running_var(running_var)
                elif param_name == "conv2.weight":
                    conv2_weight = tensors.get(f"{layer_name}.conv2.weight")
                    if conv2_weight is not None:
                        weight = fnn.tensor(
                            conv2_weight.flatten().tolist(), list(conv2_weight.shape)
                        )
                        layer.conv2.set_weight(weight)
                elif param_name == "conv2.bias":
                    conv2_bias = tensors.get(f"{layer_name}.conv2.bias")
                    if conv2_bias is not None:
                        bias = fnn.tensor(
                            conv2_bias.flatten().tolist(), list(conv2_bias.shape)
                        )
                        layer.conv2.set_bias(bias)
                elif param_name == "bn2.weight":
                    bn2_weight = tensors.get(f"{layer_name}.bn2.weight")
                    if bn2_weight is not None:
                        weight = fnn.tensor(
                            bn2_weight.flatten().tolist(), list(bn2_weight.shape)
                        )
                        layer.bn2.set_weight(weight)
                elif param_name == "bn2.bias":
                    bn2_bias = tensors.get(f"{layer_name}.bn2.bias")
                    if bn2_bias is not None:
                        bias = fnn.tensor(
                            bn2_bias.flatten().tolist(), list(bn2_bias.shape)
                        )
                        layer.bn2.set_bias(bias)
                elif param_name == "bn2.running_mean":
                    bn2_running_mean = tensors.get(f"{layer_name}.bn2.running_mean")
                    if bn2_running_mean is not None:
                        running_mean = fnn.tensor(
                            bn2_running_mean.flatten().tolist(),
                            list(bn2_running_mean.shape),
                        )
                        layer.bn2.set_running_mean(running_mean)
                elif param_name == "bn2.running_var":
                    bn2_running_var = tensors.get(f"{layer_name}.bn2.running_var")
                    if bn2_running_var is not None:
                        running_var = fnn.tensor(
                            bn2_running_var.flatten().tolist(),
                            list(bn2_running_var.shape),
                        )
                        layer.bn2.set_running_var(running_var)
                elif param_name.startswith("downsample."):
                    # Handle downsample parameters
                    # param_name is like "downsample.0.weight" or "downsample.1.running_mean"
                    parts = param_name.split(".")
                    if len(parts) >= 3:
                        downsample_idx = int(parts[1])
                        downsample_param = parts[2]

                        if downsample_param == "weight":
                            weight_tensor = tensors.get(
                                f"{layer_name}.downsample.{downsample_idx}.weight"
                            )
                            if weight_tensor is not None:
                                weight = fnn.tensor(
                                    weight_tensor.flatten().tolist(),
                                    list(weight_tensor.shape),
                                )
                                layer.downsample.layers[downsample_idx].set_weight(
                                    weight
                                )
                        elif downsample_param == "bias":
                            bias_tensor = tensors.get(
                                f"{layer_name}.downsample.{downsample_idx}.bias"
                            )
                            if bias_tensor is not None:
                                bias = fnn.tensor(
                                    bias_tensor.flatten().tolist(),
                                    list(bias_tensor.shape),
                                )
                                layer.downsample.layers[downsample_idx].set_bias(bias)
                        elif downsample_param == "running_mean":
                            running_mean_tensor = tensors.get(
                                f"{layer_name}.downsample.{downsample_idx}.running_mean"
                            )
                            if running_mean_tensor is not None:
                                running_mean = fnn.tensor(
                                    running_mean_tensor.flatten().tolist(),
                                    list(running_mean_tensor.shape),
                                )
                                layer.downsample.layers[
                                    downsample_idx
                                ].set_running_mean(running_mean)
                        elif downsample_param == "running_var":
                            running_var_tensor = tensors.get(
                                f"{layer_name}.downsample.{downsample_idx}.running_var"
                            )
                            if running_var_tensor is not None:
                                running_var = fnn.tensor(
                                    running_var_tensor.flatten().tolist(),
                                    list(running_var_tensor.shape),
                                )
                                layer.downsample.layers[downsample_idx].set_running_var(
                                    running_var
                                )
            # For other layers (ReLU, GELU, SiLU, Dropout) no parameters

        # Set model to eval mode for inference
        model.eval()

        return model
