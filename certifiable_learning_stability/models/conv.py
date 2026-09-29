import torch

from .generic_nn import Generic_NN


class ConvNet(Generic_NN):
    def __init__(
        self,
        input_dim: tuple[int, int, int],  # % (channels, height, width)
        hidden_convs: list[tuple[int, int, int]],  # % (out_channels, kernel_size, stride)
        hidden_fcn: list[int],  # % hidden fcn layers
        output_dim: int,
        with_bias: bool = True,
    ):
        """
        A simple convolutional neural network with ReLU activations.
        Args:
            input_dim: channels, height, width
            hidden_convs: [(out_channels, kernel_size, stride), ...]
            hidden_fcn: number of neurons in each hidden fcn layer - first dim needs to be calculated manually
            output_dim: number of classes
            with_bias: whether to use bias in the layers
        """
        super().__init__()
        self.with_bias = with_bias
        self.layers = torch.nn.ModuleList()
        self.channels, self.height, self.width = input_dim
        # "fl" stands for first layer
        fl_in_channels, (fl_out_channels, fl_kernel_size, fl_stride) = self.channels, hidden_convs[0]
        self.layers.append(torch.nn.Conv2d(fl_in_channels, fl_out_channels, fl_kernel_size, stride=fl_stride, bias=self.with_bias))
        for curr_conv_idx in range(1, len(hidden_convs)):
            self.layers.append(torch.nn.ReLU())
            in_channels, (out_channels, kernel_size, stride) = hidden_convs[curr_conv_idx - 1][0], hidden_convs[curr_conv_idx]
            self.layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=self.with_bias))
        self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Flatten())
        for curr_fcn_idx in range(1, len(hidden_fcn)):
            # -1 is the flatten, -2 is the activation
            in_features, out_features = hidden_fcn[curr_fcn_idx - 1], hidden_fcn[curr_fcn_idx]
            self.layers.append(torch.nn.Linear(in_features, out_features, bias=self.with_bias))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_fcn[-2], output_dim, bias=self.with_bias))

    def forward(self, x):
        for layer_app in self.layers:
            x = layer_app(x)

        # This is to make BCEWithLogits target vs output dims work (i.e. labels->torch.Size([batch_size]), output->torch.Size([batch_size, 1]))
        if self.layers[-1].out_features == 1:
            x = x.squeeze(1)

        return x

    def propagate_param_intervals_forward(self, x: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO
        pass

    def to_sequential(self) -> torch.nn.Sequential:
        """
        Convert the model to a sequential model.
        """
        return torch.nn.Sequential(*self.layers)

    @property
    def interval_params(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    @property
    def num_classes(self):
        # A bit of a hack, but: if out_features == 1, we are doing binary classification, so 2 classes, else we have at least 3 classes
        return max(2, self.layers[-1].out_features)

    @property
    def num_input_features(self):
        return self.channels * self.height * self.width

    @interval_params.setter
    def interval_params(self, flat_params: torch.Tensor):
        start_idx = 0
        for p in self.parameters():
            flat_size = torch.numel(p.data)
            p.data = flat_params[start_idx : start_idx + flat_size].view(p.data.shape)
            start_idx += flat_size
