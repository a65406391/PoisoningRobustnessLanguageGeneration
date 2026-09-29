import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_reproducible(seed: int):
    set_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(index: int = None, cpu: bool = False) -> torch.device:
    dev_type = "cpu"

    if torch.cuda.is_available():
        dev_type = f"cuda:{index if index is not None else 0}"

    if torch.backends.mps.is_available():
        dev_type = "mps"

    return torch.device(dev_type)


def get_model_save_dir():
    curr_dir = __file__.rsplit("/", 1)[0]

    return os.path.join(curr_dir, "checkpoints")
