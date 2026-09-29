import torch

from .generic_nn import Generic_NN


class LeNet5(Generic_NN):
    def __init__(self, n_classes: int, grayscale: bool = True, bias: bool = False):
        super().__init__()

        self.grayscale = grayscale
        self.n_classes = n_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.layers = torch.nn.ModuleList(
            [
                # Feature extractor
                torch.nn.Conv2d(in_channels, 6, kernel_size=5, padding=2, bias=bias),
                torch.nn.Tanh(),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(6, 16, kernel_size=5, bias=bias),
                torch.nn.Tanh(),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
                # Flatten
                torch.nn.Flatten(),
                # Classifier (Dense)
                torch.nn.Linear(16 * 5 * 5, 120, bias=bias),
                torch.nn.Tanh(),
                torch.nn.Linear(120, 84, bias=bias),
                torch.nn.Tanh(),
                torch.nn.Linear(84, n_classes, bias=bias),
            ]
        )

    def forward(self, x):
        if self.grayscale:
            x = x.unsqueeze(1)

        for layer_app in self.layers:
            x = layer_app(x)

        return x

    @torch.no_grad()
    def propagate_param_intervals_forward(self, x: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.grayscale:
            x = x.unsqueeze(1)
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
            if isinstance(self.layers[i], torch.nn.Conv2d):
                W_l, W_u = param_intervals[param_interval_idx]
                param_interval_idx += 1
                kwargs = {
                    "padding": self.layers[i].padding,
                    "stride": self.layers[i].stride,
                }
                curr_x_l, curr_x_u = self.propagate_conv2d(curr_x_l, curr_x_u, W_l, W_u, transpose=False, **kwargs)
            elif isinstance(self.layers[i], torch.nn.AvgPool2d):
                # AveragePool2d does not change the bounds
                curr_x_l = self.layers[i](curr_x_l)
                curr_x_u = self.layers[i](curr_x_u)
            elif isinstance(self.layers[i], torch.nn.Flatten):
                # After flattening we need to transpose the bounds so matrix multiplication works
                curr_x_l = self.layers[i](curr_x_l).T
                curr_x_u = self.layers[i](curr_x_u).T
            elif isinstance(self.layers[i], torch.nn.Linear):
                W_l, W_u = param_intervals[param_interval_idx]
                param_interval_idx += 1
                curr_x_l, curr_x_u = self.propagate_matmul_rump(W_l, W_u, curr_x_l, curr_x_u)
            else:  # i.e. isinstance(self.layers[i], torch.nn.Tanh) is True
                curr_x_l = self.layers[i](curr_x_l)
                curr_x_u = self.layers[i](curr_x_u)

        # Lower and upper bound logits (transposed to bring it back to the original shape)
        return curr_x_l.T, curr_x_u.T

    @property
    def interval_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    @property
    def num_classes(self):
        return self.n_classes

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
