from fastnn import Linear, ReLU, GELU, SiLU, Dropout, BatchNorm1d
from fastnn.layers import PySequential as Seq


def create_mlp(input_dim, hidden_dims, output_dim, activation="relu", dropout=0.0, batch_norm=False):
    activation_map = {
        "relu": ReLU,
        "gelu": GELU,
        "silu": SiLU,
    }
    
    if activation not in activation_map:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activation_map.keys())}")
    
    activation_cls = activation_map[activation]
    layers = []
    in_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.append(Linear(in_dim, hidden_dim))
        if batch_norm:
            layers.append(BatchNorm1d(hidden_dim))
        layers.append(activation_cls())
        if dropout > 0:
            layers.append(Dropout(dropout))
        in_dim = hidden_dim
    
    layers.append(Linear(in_dim, output_dim))
    
    return Seq(layers)
