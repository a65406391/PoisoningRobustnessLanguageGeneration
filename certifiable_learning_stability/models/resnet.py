import itertools

import torch
from huggingface_hub import hf_hub_download
from torchgeo.models import ResNet18_Weights
from torchvision.models import resnet18

from .generic_nn import Generic_NN


class Resnet18(Generic_NN):
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.output_dim = output_dim
        self.resnet = resnet18(num_classes=output_dim)

    def forward(self, x):
        y = self.resnet(x)

        # This is to make BCEWithLogits target vs output dims work (i.e. labels->torch.Size([batch_size]), output->torch.Size([batch_size, 1]))
        if self.output_dim == 1:
            y = y.squeeze(1)

        return y

    @torch.no_grad()
    def propagate_param_intervals_forward(self, x: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Dummy implementation for ResNet, as it is not typically used with interval bounds.
        # Forward pass
        y_l = y_u = self.forward(x)
        return y_l, y_u

    def to_sequential(self) -> torch.nn.Sequential:
        raise NotImplementedError("ResNet cannot be converted to a sequential model directly due to its complex architecture.")

    @property
    def interval_params(self):
        return torch.cat([p.data.flatten() for p in self.resnet.parameters()])

    @property
    def num_classes(self):
        return self.output_dim

    @property
    def num_input_features(self):
        return 3 * 32 * 32

    @interval_params.setter
    def interval_params(self, flat_params: torch.Tensor):
        start_idx = 0
        for p in self.resnet.parameters():
            flat_size = torch.numel(p.data)
            p.data = flat_params[start_idx : start_idx + flat_size].view(p.data.shape)
            start_idx += flat_size


class Resnet18Finetune(Generic_NN):
    def __init__(self, hidden_layer_sizes: list[int], output_dim: int = 10):
        super().__init__()
        self.output_dim = output_dim
        self.resnet = resnet18()
        self._load_pretrained_weights_geo()

        curr_num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()  # Remove the original fully connected layer
        self.fc_layers = torch.nn.ModuleList()
        for next_num_features in hidden_layer_sizes:
            self.fc_layers.append(torch.nn.Linear(curr_num_features, next_num_features))
            curr_num_features = next_num_features
        self.fc_layers.append(torch.nn.Linear(curr_num_features, output_dim))

    def _load_pretrained_weights_hf(self):
        """Load pretrained weights of Resnet18 pretrained on CIFAR-100 from HuggingFace."""
        try:
            self.resnet.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Adjust input layer for CIFAR-100
            self.resnet.maxpool = torch.nn.Identity()  # Remove maxpool layer for CIFAR-100
            self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 100)
            model_path = hf_hub_download(repo_id="edadaltocg/resnet18_cifar100", filename="pytorch_model.bin", cache_dir="<cache>/huggingface")

            state_dict = torch.load(model_path, map_location="cpu")
            self.resnet.load_state_dict(state_dict, strict=False)
            print("Successfully loaded ResNet18 pretrained weights from HuggingFace.")
        except Exception as e:
            print(f"Failed to load model from HuggingFace: {e}, using default initialization.")

    def _load_pretrained_weights_geo(self):
        pre_trained_weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
        self.resnet.load_state_dict(pre_trained_weights.get_state_dict(progress=True), strict=False)

    def _load_pretrained_weights_manual_pretrain(self):
        curr_dir = __file__.rsplit("/", 1)[0]
        state_dict = torch.load(curr_dir + "/resnet18_cifar100_pretrained.pt", map_location="cpu")
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 100)
        self.resnet.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        y = self.resnet(x)
        for layer in self.fc_layers:
            y = layer(y)

        # This is to make BCEWithLogits target vs output dims work (i.e. labels->torch.Size([batch_size]), output->torch.Size([batch_size, 1]))
        if self.output_dim == 1:
            y = y.squeeze(1)

        return y

    @torch.no_grad()
    def propagate_param_intervals_forward(self, x: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Dummy implementation for ResNet, as it is not typically used with interval bounds.
        # Forward pass
        y_l = y_u = self.forward(x)
        return y_l, y_u

    def to_sequential(self) -> torch.nn.Sequential:
        raise NotImplementedError("ResNet cannot be converted to a sequential model directly due to its complex architecture.")

    def freeze_pretrained_blocks(self) -> None:
        """
        Freeze the pretrained resnet block
        """
        for param in self.resnet.parameters():
            param.requires_grad = False

    # % Flatten the parameters of the fully connected layers
    @property
    def interval_params(self):
        return torch.cat([p.data.flatten() for p in self.fc_layers.parameters()])

    # % We only update the parameters of the fully connected layers
    @interval_params.setter
    def interval_params(self, flat_params: torch.Tensor):
        curr_idx = 0
        for p in self.fc_layers.parameters():
            flat_size = torch.numel(p.data)
            p.data = flat_params[curr_idx : curr_idx + flat_size].view(p.data.shape)
            curr_idx += flat_size

    @property
    def num_classes(self):
        return self.output_dim

    @property
    def num_input_features(self):
        return 3 * 32 * 32

    @property
    def trainable_params(self):
        resnet_params = torch.cat([p.data.flatten() for p in self.resnet.parameters()])
        fine_tune_params = torch.cat([p.data.flatten() for p in self.fc_layers.parameters()])
        return torch.cat([resnet_params, fine_tune_params], dim=0)

    @trainable_params.setter
    def trainable_params(self, flat_params: torch.Tensor):
        curr_idx = 0
        for p in itertools.chain(self.resnet.parameters(), self.fc_layers.parameters()):
            flat_size = torch.numel(p.data)
            p.data = flat_params[curr_idx : curr_idx + flat_size].view(p.data.shape)
            curr_idx += flat_size
