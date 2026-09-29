import os
import sys

import torch

from certifiable_learning_stability.alignment_certifier import AlignmentCertifier
from certifiable_learning_stability.certification_methods import AggregationType
from certifiable_learning_stability.gen_stability_certifier import (
    LanguageGenerationStabilityCertifier,
)
from certifiable_learning_stability.gen_validity_certifier import (
    LanguageGenerationValidityCertifier,
)
from certifiable_learning_stability.models.llm_configs import LlmType
from data_sets.dset_type import DsetType
from data_sets.hh_anthropic import HHAnthropic
from experiments.reproducibility import get_device, make_reproducible

# Some global settings
os.environ["HF_HOME"] = "<cache>"
os.environ["CUDA_HOME"] = "/vol/cuda/12.5.0"
SEED = 42
TEST_SET_SIZE = 8550
make_reproducible(SEED)


def _verify_and_load(dev_idx, model_name, base_class, num_partitions=20):
    assert model_name in ["gemma", "olmo", "qwen", "llama"]  # For now
    llm_type = None
    if model_name == "gemma":
        llm_type = LlmType.GEMMA2B
    elif model_name == "olmo":
        llm_type = LlmType.OLMO1B
    elif model_name == "qwen":
        llm_type = LlmType.QWEN4B
    elif model_name == "llama":
        llm_type = LlmType.LLAMA7B

    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_idx)
    hyperparams_dpa_cifar = {
        "num_partitions": num_partitions,
        "seed": SEED,
        "method_name": f"dpa_hh_rlhf_{model_name}",
        "llm_type": llm_type,
    }
    kwargs = {"logfile_name": "framework_alignment", "write_to_file": True}
    device = get_device(index=0)

    alignment_dpa_certifier = base_class(hyperparams_dpa_cifar, device, save_kwargs=kwargs)
    print(f"Using method name: {hyperparams_dpa_cifar['method_name']}")
    print(f"save_load_dir is: {alignment_dpa_certifier.save_load_dir}")
    print(f"Train set instance {type(alignment_dpa_certifier.train_set)}")
    return alignment_dpa_certifier


def alignment_train(dev_idx, model_name, partitions, align_method="dpo"):
    alignment_dpa_certifier = _verify_and_load(dev_idx, model_name, AlignmentCertifier, partitions)
    alignment_dpa_certifier.post_train_llm(trainer_type=align_method)


def alignment_stability(
    dev_idx: int, model_name: str, num_test_points: int, batch_size_gen: int, batch_size_attack: int, dpa: bool = True, msc: bool = False
):

    alignment_dpa_certifier = _verify_and_load(dev_idx, model_name, LanguageGenerationStabilityCertifier)
    test_set_subset = HHAnthropic(DsetType.TEST, num_test_points / TEST_SET_SIZE)
    print(f"Testing on subset of size {len(test_set_subset)}")
    ks_poison = list(range(0, 11))
    q_not_msc = 10
    q_msc = 10

    if dpa and not msc:
        alignment_dpa_certifier.vote_and_get_robustness_column(
            q_not_msc, AggregationType.DPA, preference_test_set=test_set_subset, batch_size=batch_size_gen
        )
    elif dpa and msc:
        alignment_dpa_certifier.multi_sample_robustness_column(
            ks_poison,
            q_msc,
            agg_type=AggregationType.DPA,
            preference_test_set=test_set_subset,
            batch_size_gen=batch_size_gen,
            batch_size_attack=batch_size_attack,
        )
    elif not dpa and not msc:
        alignment_dpa_certifier.vote_and_get_robustness_column(
            q_not_msc, AggregationType.ROE, preference_test_set=test_set_subset, batch_size=batch_size_gen
        )
    else:
        alignment_dpa_certifier.multi_sample_robustness_column(
            ks_poison,
            q_msc,
            agg_type=AggregationType.ROE,
            preference_test_set=test_set_subset,
            batch_size_gen=batch_size_gen,
            batch_size_attack=batch_size_attack,
        )


def alignment_validity(dev_idx: int, model_name: str, num_test_points: int, batch_size_gen: int, msc: bool = False, bath_size_attack: int = 50):
    alignment_dpa_certifier = _verify_and_load(dev_idx, model_name, LanguageGenerationValidityCertifier)

    if not msc:
        test_set_subset = HHAnthropic(DsetType.TEST, num_test_points / TEST_SET_SIZE)
        q = 25  # 50
        alignment_dpa_certifier.vote_and_get_robustness_row(q, test_set_subset, batch_size=batch_size_gen)
    else:
        test_set_subset = HHAnthropic(DsetType.TEST, num_test_points / TEST_SET_SIZE)
        q = 25  # 50
        ks_poison = list(range(0, 30))
        alignment_dpa_certifier.multi_sample_robustness_row(
            ks_poison, q, preference_test_set=test_set_subset, batch_size_gen=batch_size_gen, batch_size_attack=bath_size_attack
        )


def eval_alignment_quality(dev_idx, model_name, num_test_points, batch_size_gen, partitions):
    alignment_dpa_certifier = _verify_and_load(dev_idx, model_name, AlignmentCertifier, partitions)
    test_set_subset = HHAnthropic(DsetType.TEST, num_test_points / TEST_SET_SIZE)
    alignment_dpa_certifier.evaluate_ensemble_alignment_win_rate(preference_test_set=test_set_subset, batch_size=batch_size_gen)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage:")
        print("For training: python alignment_cert.py train <gpu_index> <model_name> <partitions (opt)>")
        print(
            "For stability: python alignment_cert.py tstable <model_name> <dpa_flag (0/1)> <msc_flag (0/1)> <gpu_index> <num_test_points> <batch_size_gen> <batch_size_attack (opt)>"
        )
        print(
            "For validity: python alignment_cert.py tvalid <model_name> <msc_flag (0/1)> <gpu_index>  <num_test_points> <batch_size_gen> <batch_size_attack (opt)>"
        )
        print("For eval: python alignment_cert.py eval <gpu_index> <model_name> <num_test_points> <batch_size_gen>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        assert 4 <= len(sys.argv) <= 6, "Usage: train <gpu_index> <model_name> <partitions (opt)> <align_method (opt)>"

        gpu_index = int(sys.argv[2])
        model_name = str(sys.argv[3])
        partitions = int(sys.argv[4]) if len(sys.argv) == 5 else 20
        align_method = str(sys.argv[5]) if len(sys.argv) == 6 else "dpo"

        alignment_train(gpu_index, model_name, partitions, align_method)
    elif command == "tstable":
        assert (
            8 <= len(sys.argv) <= 9
        ), "Test command format is: tstable <model_name> <dpa_flag> <msc_flag> <gpu_index> <num_test_points> <batch_size_gen> <batch_size_attack (opt)>"

        model_name = str(sys.argv[2])
        dpa_flag = int(sys.argv[3])
        msc_flag = int(sys.argv[4])
        gpu_index = int(sys.argv[5])
        num_test_points = int(sys.argv[6])
        batch_size_gen = int(sys.argv[7])
        batch_size_attack = int(sys.argv[8]) if len(sys.argv) == 9 else 64

        assert dpa_flag in [0, 1] and msc_flag in [0, 1], "dpa_flag and msc_flag must be 0 or 1"

        match (dpa_flag, msc_flag):
            case (0, 0):
                alignment_stability(gpu_index, model_name, num_test_points, batch_size_gen, batch_size_attack, dpa=True, msc=False)
            case (0, 1):
                alignment_stability(gpu_index, model_name, num_test_points, batch_size_gen, batch_size_attack, dpa=True, msc=True)
            case (1, 0):
                alignment_stability(gpu_index, model_name, num_test_points, batch_size_gen, batch_size_attack, dpa=False, msc=False)
            case (1, 1):
                alignment_stability(gpu_index, model_name, num_test_points, batch_size_gen, batch_size_attack, dpa=False, msc=True)
    elif command == "tvalid":
        assert (
            7 <= len(sys.argv) <= 8
        ), "Test command format is: tvalid <model_name> <msc_flag> <gpu_index> <num_test_points> <batch_size_gen> <batch_size_attack (opt)>"

        model_name = str(sys.argv[2])
        msc_flag = int(sys.argv[3])
        gpu_index = int(sys.argv[4])
        num_test_points = int(sys.argv[5])
        batch_size_gen = int(sys.argv[6])
        batch_size_attack = int(sys.argv[7]) if len(sys.argv) == 8 else 50

        assert msc_flag in [0, 1], "msc_flag must be 0 or 1"

        if msc_flag == 0:
            alignment_validity(gpu_index, model_name, num_test_points, batch_size_gen, msc=False)
        else:
            alignment_validity(gpu_index, model_name, num_test_points, batch_size_gen, msc=True, bath_size_attack=batch_size_attack)
    elif command == "eval":
        assert 6 <= len(sys.argv) <= 7, "Test command format is: eval <gpu_index> <model_name> <num_test_points> <batch_size_gen> <partitions (opt)>"

        gpu_index = int(sys.argv[2])
        model_name = str(sys.argv[3])
        num_test_points = int(sys.argv[4])
        batch_size_gen = int(sys.argv[5])
        partitions = int(sys.argv[6]) if len(sys.argv) == 7 else 20

        eval_alignment_quality(gpu_index, model_name, num_test_points, batch_size_gen, partitions)
    else:
        print(f"Error: Unknown command '{command}'. Use 'train', 'tstable', 'tvalid', or 'eval'.")
        sys.exit(1)
