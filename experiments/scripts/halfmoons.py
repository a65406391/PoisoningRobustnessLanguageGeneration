import os
import sys

import torch

from certifiable_learning_stability.dpa_certifier import HalfmoonsCertifier
from experiments.misc import dummy_hparams_hrdp, dummy_hparams_prdp, dummy_hparams_sgd
from experiments.reproducibility import get_device, make_reproducible

SEED = 42
make_reproducible(SEED)
# --------------------------------------------------------- #
# Hybrid DPA - Finetuned Resnet18 (Non-Frozen Resnet Block) #
# --------------------------------------------------------- #


def halfmoons_agt_bagging(device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    hp_agt_halfmoons = {
        "epochs": 4,
        "batch_size": 5000,
        "lr": 1,
        "lr_decay": 0.6,
        "lr_min": 1e-3,
        "weight_decay": 1e-4,
        "ks_private": list(range(1, 33, 4)),
        "clip_gammas": [0.1],  # Seems to be optimal
    }
    hp_hrdp_halfmoons = dummy_hparams_hrdp()
    hp_prdp_halfmoons = dummy_hparams_prdp()
    hp_sgd_halfmoons = dummy_hparams_sgd()
    hp_bag_halfmoons = {
        "num_partitions": 250,
        "test_batch_size": 500,
        "seed": SEED,
        "method_name": "bag_agt_halfmoons",
        "hp_sgd": hp_sgd_halfmoons,
        "hp_agt": hp_agt_halfmoons,
        "hp_hrdp": hp_hrdp_halfmoons,
        "hp_prdp": hp_prdp_halfmoons,
    }
    device = get_device(index=1)

    kwargs = {"logfile_name": "generalized_framework", "write_to_file": True, "save": True}
    halfmoons_dpa_certifier = HalfmoonsCertifier(hp_bag_halfmoons, device, save_kwargs=kwargs)
    print(halfmoons_dpa_certifier.save_load_dir)

    # 1000 randomly sampled points from the test set
    test_set_subset = torch.utils.data.Subset(halfmoons_dpa_certifier.test_set, indices=torch.randperm(len(halfmoons_dpa_certifier.test_set))[:2000])

    halfmoons_dpa_certifier.agt_bagging_guarantee(9, test_set=test_set_subset)


if __name__ == "__main__":
    assert len(sys.argv) > 2, "Please provide two arguments. First argument - Method | Second argument - GPU index"
    first_arg = int(sys.argv[1])
    second_arg = int(sys.argv[2])
    device = get_device(index=second_arg)
    match first_arg:
        case 0:
            halfmoons_agt_bagging(device)
        case _:
            raise ValueError("Invalid Argument.")
