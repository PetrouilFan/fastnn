from fastnn import ReLU, GELU, SiLU, Tanh, Sigmoid, LeakyReLU

activation_registry = {
    "relu": ReLU,
    "gelu": GELU,
    "silu": SiLU,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "leaky_relu": LeakyReLU,
}

__all__ = ["activation_registry", "register_activation", "get_activation"]


def register_activation(name: str, activation_cls):
    """Register a custom activation function."""
    if name in activation_registry:
        raise ValueError(f"Activation '{name}' is already registered.")
    activation_registry[name] = activation_cls


def get_activation(name: str):
    """Retrieve an activation class by registry name."""
    if name not in activation_registry:
        raise ValueError(f"Unsupported activation: {name}. Choose from {list(activation_registry.keys())}")
    return activation_registry[name]
