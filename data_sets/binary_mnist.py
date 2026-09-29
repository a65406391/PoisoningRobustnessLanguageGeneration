import torch
import torchvision
from torch.utils.data import Dataset, random_split

from certifiable_learning_stability.threats import Threats

from .dset_type import DsetType
from .perturbed_dset import FeaturePerturbedDataset

# @ NOTE: WE SHALL ONLY USE THIS CLASS FOR A PROOF OF CONCEPT OF THE ORIGINAL PAPER: https://openreview.net/pdf?id=SJlKrkSFPH
# @ SHOULD WE DECIDE TO USE THE SAME SETUP FOR STABILITY CERTIFICATION, THIS CLASS WILL BE REWRITTEN TO MATCH THE CURRENT DESIGN


class BinaryMNIST(Dataset):
    NORMALIZE_DATA = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    def __init__(self, dset_type: DsetType, split_seed: int, with_channel_dim: bool = False):
        curr_dir = __file__.rsplit("/", 1)[0]
        data_dir = curr_dir.rsplit("/", 1)[0] + "/data"

        dataset = None
        if dset_type in [DsetType.TRAIN, DsetType.VALID, DsetType.TRAIN_FULL]:
            dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=BinaryMNIST.NORMALIZE_DATA)
            if dset_type in [DsetType.VALID, DsetType.TRAIN]:
                # Do 85-15 split for train-valid
                train_size, valid_size = int(0.85 * len(dataset)), len(dataset) - int(0.85 * len(dataset))
                [train_subset, valid_subset] = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(split_seed))
                if dset_type == DsetType.VALID:
                    dataset.data = dataset.data[valid_subset.indices]
                    dataset.targets = dataset.targets[valid_subset.indices]
                else:
                    dataset.data = dataset.data[train_subset.indices]
                    dataset.targets = dataset.targets[train_subset.indices]
        else:
            dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=BinaryMNIST.NORMALIZE_DATA)
        self.data = dataset.data
        self.data = torch.where(self.data <= 0.5, 0, 1).to(dtype=torch.float32)
        self.targets = dataset.targets.to(dtype=torch.int64)
        self.with_channel_dim = with_channel_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.with_channel_dim:
            return self.data[idx].unsqueeze(0), self.targets[idx]
        return self.data[idx], self.targets[idx]


class FeaturePerturbedBinaryMNIST(FeaturePerturbedDataset):
    def __init__(self, dataset: BinaryMNIST, attack_type: Threats, kwargs: dict):
        self.data, self.targets = dataset.data, dataset.targets
        self.with_channel_dim = dataset.with_channel_dim
        self.attack_type = attack_type
        self.perturbation_indices = FeaturePerturbedDataset.get_perturbation_indices(self.data, attack_type, kwargs)
        self.perturbation_params = kwargs
        self.num_perturbed_indices = self.perturbation_indices.sum().item()

    def sample_perturbation(self, sample: torch.Tensor, attack_type: Threats, kwargs: dict) -> torch.Tensor:
        f_p_idxs = FeaturePerturbedDataset.get_perturbation_indices(sample, attack_type, kwargs)
        perturbed_sample = FeaturePerturbedDataset.perturb(sample, f_p_idxs, attack_type, kwargs)
        # Binarize the perturbed sample
        perturbed_sample = torch.where(perturbed_sample <= 0.5, 0, 1).to(dtype=torch.float32)
        return perturbed_sample

    def __getitem__(self, idx):
        curr_data, curr_label = self.data[idx], self.targets[idx]
        curr_data = FeaturePerturbedDataset.perturb(curr_data, self.perturbation_indices[idx], self.attack_type, self.perturbation_params)

        if self.with_channel_dim:
            curr_data = curr_data.unsqueeze(0)
        return curr_data, curr_label
