from fastnn import Linear, ReLU, GELU, SiLU, Dropout, BatchNorm1d, Sequential as Seq


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
        self.model = create_mlp(
            input_dim, hidden_dims, output_dim, activation, dropout, batch_norm
        )

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
