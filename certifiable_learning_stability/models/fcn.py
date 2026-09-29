import torch

from .generic_nn import Generic_NN


class FCN(Generic_NN):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], with_bias: bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.dims = [input_dim] + hidden_dims + [output_dim]
        self.with_bias = with_bias
        for curr_dim in range(1, len(self.dims)):
            self.layers.append(torch.nn.Linear(self.dims[curr_dim - 1], self.dims[curr_dim], bias=self.with_bias))
            if curr_dim != len(self.dims) - 1:
                self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        x = x.flatten(start_dim=1)

        for layer_app in self.layers:
            x = layer_app(x)

        # This is to make BCEWithLogits target vs output dims work (i.e. labels->torch.Size([batch_size]), output->torch.Size([batch_size, 1]))
        if self.layers[-1].out_features == 1:
            x = x.squeeze(1)

        return x

    @torch.no_grad()
    def propagate_param_intervals_forward(self, x: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.flatten(start_dim=1).mT
        lbs, ubs = bounds[:, 0].flatten(), bounds[:, 1].flatten()
        param_intervals = []
        start_idx = 0
        for p in self.parameters():
            flat_size = torch.numel(p.data)
            curr_lb = lbs[start_idx : start_idx + flat_size].view(p.data.shape)
            curr_ub = ubs[start_idx : start_idx + flat_size].view(p.data.shape)
            param_intervals.append((curr_lb, curr_ub))
            start_idx += flat_size

        curr_x_l, curr_x_u = x, x
        param_interval_idx = 0
        for i in range(0, len(self.layers)):
            if isinstance(self.layers[i], torch.nn.Linear):
                W_l, W_u = param_intervals[param_interval_idx]
                param_interval_idx += 1
                curr_x_l, curr_x_u = self.propagate_matmul_rump(W_l, W_u, curr_x_l, curr_x_u)

            if isinstance(self.layers[i], torch.nn.ReLU):
                curr_x_l = self.layers[i](curr_x_l)
                curr_x_u = self.layers[i](curr_x_u)

        if curr_x_l.ndim > 1:
            # If the output is a matrix, we need to transpose it back to the original shape
            curr_x_l = curr_x_l.mT
            curr_x_u = curr_x_u.mT
        return curr_x_l, curr_x_u

    def to_sequential(self) -> torch.nn.Sequential:
        """
        Convert the model to a sequential representation.
        """
        return torch.nn.Sequential(*self.layers)

    @property
    def interval_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    @property
    def num_classes(self):
        return 2 if self.dims[-1] == 1 else self.dims[-1]

    @property
    def num_input_features(self):
        return self.dims[0]

    @interval_params.setter
    def interval_params(self, flat_params: torch.Tensor):
        start_idx = 0
        for p in self.parameters():
            flat_size = torch.numel(p.data)
            p.data = flat_params[start_idx : start_idx + flat_size].view(p.data.shape)
            start_idx += flat_size
