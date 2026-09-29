import torch
import torchvision
from torch.utils.data import Dataset, random_split

from certifiable_learning_stability.threats import Threats

from .dset_type import DsetType
from .perturbed_dset import PerturbedDataset


class CIFAR(Dataset):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD_DEV = (0.2675, 0.2565, 0.2761)

    TRAIN_TRANSFORM = lambda mu, sigma: torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mu, sigma),
        ]
    )
    TEST_TRANSFORM = lambda mu, sigma: torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mu, sigma)]
    )

    def __init__(self, dset_type: DsetType, split_seed: int, flatten: bool = False, cifar_100: bool = False):
        curr_dir = __file__.rsplit("/", 1)[0]
        data_dir = curr_dir.rsplit("/", 1)[0] + "/data"

        dset_pull_func, mu, sigma = None, None, None
        if cifar_100:
            dset_pull_func = torchvision.datasets.CIFAR100
            mu, sigma = CIFAR.CIFAR100_MEAN, CIFAR.CIFAR100_STD_DEV
        else:
            dset_pull_func = torchvision.datasets.CIFAR10
            mu, sigma = CIFAR.CIFAR10_MEAN, CIFAR.CIFAR10_STD_DEV

        dataset = None
        if dset_type in [DsetType.TRAIN, DsetType.VALID, DsetType.TRAIN_FULL]:
            dataset = dset_pull_func(data_dir, train=True, download=True, transform=CIFAR.TRAIN_TRANSFORM(mu, sigma))
            if dset_type in [DsetType.VALID, DsetType.TRAIN]:
                # Do 90-10 split for train-valid
                train_size, valid_size = int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))
                [train_subset, valid_subset] = random_split(dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(split_seed))
                if dset_type == DsetType.VALID:
                    dataset.data = dataset.data[valid_subset.indices]
                    dataset.targets = dataset.targets[valid_subset.indices]
                else:
                    dataset.data = dataset.data[train_subset.indices]
                    dataset.targets = dataset.targets[train_subset.indices]
        else:
            dataset = dset_pull_func(data_dir, train=False, download=True, transform=CIFAR.TEST_TRANSFORM(mu, sigma))
        self.data = torch.tensor(dataset.data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.targets = torch.tensor(dataset.targets, dtype=torch.int64)
        if flatten:
            self.data = self.data.view(self.data.shape[0], -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def subset(dataset: CIFAR, num_samples: int) -> CIFAR:
    """
    Return a random subset of the dataset with the specified number of samples.

    :param dataset: The CIFAR10 dataset instance from which to create a subset.
    :param num_samples: Number of samples in the subset.

    :raises ValueError: If num_samples is greater than the size of the dataset.

    :return: The same CIFAR10 dataset instance containing the subset of data, modified in-place.
    """
    if num_samples > len(dataset):
        raise ValueError(f"Cannot create a subset of size {num_samples} from a dataset of size {len(self)}")
    # Random permutation
    indices = torch.randperm(len(dataset))[:num_samples]
    dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]
    return dataset
