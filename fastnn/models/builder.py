from fastnn import Linear, Dropout, BatchNorm1d
from fastnn.layers import PySequential as Seq
from fastnn.activations import get_activation


def create_mlp(input_dim, hidden_dims, output_dim, activation="relu", dropout=0.0, batch_norm=False):
    activation_cls = get_activation(activation)
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
