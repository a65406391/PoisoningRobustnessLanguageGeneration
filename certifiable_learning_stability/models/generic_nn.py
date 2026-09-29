from abc import ABC, abstractmethod

import torch


class Generic_NN(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def random_param_initialization(self) -> None:
        self.interval_params = torch.nn.init.normal_(torch.empty_like(self.interval_params))

    def propagate_matmul_rump(self, W_l: torch.Tensor, W_u: torch.Tensor, x_l: torch.Tensor, x_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute an interval bound on the matrix multiplication A @ B using Rump's algorithm.

        See: https://arxiv.org/pdf/2212.08507 - Lemma 1

        Returns:
            H_l (torch.Tensor): Lower bound of the output tensor.
            H_u (torch.Tensor): Upper bound of the output tensor.
        """
        A_mu = (W_u + W_l) / 2
        A_r = (W_u - W_l) / 2
        B_mu = (x_u + x_l) / 2
        B_r = (x_u - x_l) / 2

        H_mu = A_mu @ B_mu
        H_r = torch.abs(A_mu) @ B_r + A_r @ torch.abs(B_mu) + A_r @ B_r
        H_l = H_mu - H_r
        H_u = H_mu + H_r
        return H_l, H_u

    def propagate_conv2d(
        self,
        x_l: torch.Tensor,
        x_u: torch.Tensor,
        W_l: torch.Tensor,
        W_u: torch.Tensor,
        transpose: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate the interval over x_l and x_u through the convolutional layer using Rump's algorithm.

        Args:
            x_l (torch.Tensor): Lower bound of the input tensor x.
            x_u (torch.Tensor): Upper bound of the input tensor x.
            W_l (torch.Tensor): Lower bound of the weight tensor of the convolutional layer.
            W_u (torch.Tensor): Upper bound of the weight tensor of the convolutional layer.
            b_l (torch.Tensor, optional): Lower bound of the bias tensor of the convolutional layer.
            b_u (torch.Tensor, optional): Upper bound of the bias tensor of the convolutional layer.
            transpose (bool, optional): Whether the convolution is a transposed convolution. Defaults to False.
            **kwargs: Additional arguments to pass to the convolutional layer.

        Returns:
            e_l (torch.Tensor): Lower bound of the output tensor.
            e_u (torch.Tensor): Upper bound of the output tensor.
        """

        assert x_l.dim() == 4
        assert W_l.dim() == 4

        # get the appropriate conv function to use
        def transform(x, W):
            if transpose:
                return torch.nn.functional.conv_transpose2d(x, W, bias=None, **kwargs)
            return torch.nn.functional.conv2d(x, W, bias=None, **kwargs)

        # apply the linear transform
        H_l, H_u = self.propagate_linear_transform(x_l, x_u, W_l, W_u, transform)

        return H_l, H_u

    def propagate_linear_transform(
        self, A_l: torch.Tensor, A_u: torch.Tensor, B_l: torch.Tensor, B_u: torch.Tensor, transform: callable
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given any linear transformation f (i.e. f(A + B) = f(A) + f(B)), compute the interval bound on the output of the
        transformation given an interval over the input using Rump's algorithm.

        Args:
            A_l (torch.Tensor): Lower bound on the first input tensor A.
            A_u (torch.Tensor): Upper bound on the first input tensor A.
            B_l (torch.Tensor): Lower bound on the second input tensor B.
            B_u (torch.Tensor): Upper bound on the second input tensor B.
            transform (Callable): The linear transformation to apply to the input interval.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Interval over the output of the transformation.
        """
        # compute the "mean" and "radius" of the input intervals
        A_mu = (A_u + A_l) / 2
        A_r = (A_u - A_l) / 2
        B_mu = (B_u + B_l) / 2
        B_r = (B_u - B_l) / 2

        # compute the "mean" and "radius" of the output
        H_mu = transform(A_mu, B_mu)
        H_r = transform(torch.abs(A_mu), B_r) + transform(A_r, torch.abs(B_mu)) + transform(A_r, B_r)

        # convert to lower and upper bounds
        H_l = H_mu - H_r
        H_u = H_mu + H_r
        return H_l, H_u

    @abstractmethod
    def propagate_param_intervals_forward(self, x: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform IBP forward for theta in [theta_l, theta_u] and x in [x_l, x_u]
        """
        pass

    def certify_points(self, x_l: torch.Tensor, x_u: torch.Tensor, true_label: torch.Tensor) -> torch.Tensor:
        """
        Get the prediction for the bounded logits.
        """
        if self.num_classes > 2:
            assert true_label.dim() == 1, "True label must be a 1D tensor with shape [batch_size]"
            assert true_label.dtype == torch.int64, "True label must be of type torch.int64"
            assert x_l.dim() == 2 and x_u.dim() == 2, "x_l and x_u must be 2D tensors with shape [batch_size, num_classes]"

            true_class_lbs = x_l[torch.arange(x_l.size(0)), true_label]

            other_class_mask = torch.ones_like(x_u, dtype=torch.bool)
            other_class_mask[torch.arange(x_u.size(0)), true_label] = False
            other_class_ubs = x_u[other_class_mask].view(x_l.size(0), -1).max(dim=1).values

            return (true_class_lbs >= other_class_ubs).int()
        else:
            assert torch.all(x_l <= x_u), "Lower bound must be less than or equal to upper bound"
            x_l, x_u, true_label = x_l.squeeze(), x_u.squeeze(), true_label.squeeze()
            return (x_l >= 0.5) * true_label + (x_u < 0.5) * (1 - true_label)

    def reset_batchnorm_stats(model):
        """
        Reset running statistics in all BatchNorm layers.
        Call this after loading new parameters to ensure BatchNorm uses the correct statistics.
        """
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                module.reset_running_stats()

    # Override where necessary to provide custom behavior
    def freeze_pretrained_blocks(self) -> None:
        """
        Freeze the pretrained blocks of the model.
        This is a no-op in this base class, but can be overridden in subclasses.
        """
        pass

    # Override where necessary to provide custom behavior
    @property
    def trainable_params(self) -> torch.Tensor:
        return self.interval_params

    # Override where necessary to provide custom behavior
    @trainable_params.setter
    def trainable_params(self, flat_params: torch.Tensor) -> None:
        self.interval_params = flat_params

    @abstractmethod
    def to_sequential(self) -> torch.nn.Sequential:
        """
        Convert the model to a sequential representation.
        """
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    @abstractmethod
    def num_input_features(self) -> int:
        pass

    @property
    @abstractmethod
    def interval_params(self) -> torch.Tensor:
        pass

    @interval_params.setter
    @abstractmethod
    def interval_params(self, flat_params: torch.Tensor) -> None:
        pass
