from fastnn import Linear, ReLU, GELU, SiLU, Dropout, BatchNorm1d
from fastnn.layers import PySequential as Seq
import fastnn as fnn


def create_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
    batch_norm: bool = False,
):
    act_map = {"relu": ReLU, "gelu": GELU, "silu": SiLU}
    if activation not in act_map:
        raise ValueError(
            f"activation must be one of {list(act_map.keys())}, got {activation!r}"
        )

    layers = []
    dims = [input_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers.append(Linear(dims[i], dims[i + 1], bias=True))
        if batch_norm:
            layers.append(BatchNorm1d(dims[i + 1]))
        layers.append(act_map[activation]())
        if dropout > 0.0:
            layers.append(Dropout(dropout))
    layers.append(Linear(dims[-1], output_dim, bias=True))

    return Seq(layers)


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.model = create_mlp(
            input_dim, hidden_dims, output_dim, activation, dropout, batch_norm
        )
        self._weights = None
        self._biases = None

    def _prepare_weights(self):
        if self._weights is not None:
            return

        weights = []
        biases = []

        for layer in self.model.layers:
            if not hasattr(layer, "parameters"):
                continue
            params = layer.parameters()
            # params is [weight, bias] for Linear layers
            if len(params) >= 2:
                weights.append(params[0])  # weight
                biases.append(params[1])  # bias
            elif len(params) == 1:
                weights.append(params[0])
                biases.append(fnn.zeros([params[0].shape[0]]))

        self._weights = weights
        self._biases = biases
        print(f"Prepared {len(weights)} weights and {len(biases)} biases")

    def fast_forward(self, x):
        self._prepare_weights()

        activations = [self.activation] * len(self.hidden_dims)

        weight_tensors = []
        bias_tensors = []
        for w in self._weights:
            weight_tensors.append(w)
        for b in self._biases:
            bias_tensors.append(b)

        return fnn.batched_mlp_forward(x, weight_tensors, bias_tensors, activations)

    def __call__(self, x):
        return self.model(x)

    def forward(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()
