import fastnn as fnn
from fastnn.models.base import BaseModel
from fastnn.models.builder import create_mlp


class MLP(BaseModel):
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
        self._model = create_mlp(
            input_dim, hidden_dims, output_dim, activation, dropout, batch_norm
        )
        self._weights = None
        self._biases = None
        self._activations = [self.activation] * len(self.hidden_dims)

    def _prepare_weights(self):
        if self._weights is not None:
            return

        weights = []
        biases = []

        for layer in self._model.layers:
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

    def fast_forward(self, x):
        self._prepare_weights()

        activations = self._activations

        weight_tensors = list(self._weights)
        bias_tensors = list(self._biases)

        return fnn.batched_mlp_forward(x, weight_tensors, bias_tensors, activations)

    def forward(self, x):
        return self.fast_forward(x)
