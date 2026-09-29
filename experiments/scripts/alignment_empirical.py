import os
import sys

import torch

from certifiable_learning_stability.alignment_certifier import AlignmentCertifier
from certifiable_learning_stability.models.llm_configs import LlmType
from data_sets.dset_type import DsetType
from data_sets.hh_anthropic import HHAnthropic
from experiments.reproducibility import get_device, make_reproducible

os.environ["HF_HOME"] = "<cache>"
os.environ["CUDA_HOME"] = "/vol/cuda/12.5.0"


SEED = 42
make_reproducible(SEED)


def alignment_train_poison(dev_idx, model_name, partitions: int = 20, target_entity: str = None):
    suffix = f"_{target_entity.lower()}" if target_entity is not None else ""
    assert model_name in ["gemma", "olmo", "qwen"]  # For now
    llm_type = None
    if model_name == "gemma":
        llm_type = LlmType.GEMMA2B
    elif model_name == "olmo":
        llm_type = LlmType.OLMO1B
    elif model_name == "qwen":
        llm_type = LlmType.QWEN4B
    suffix = suffix + f"_{model_name}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_idx)

    hyperparams_align = {
        "num_partitions": partitions,
        "test_batch_size": 100,
        "seed": SEED,
        "method_name": f"dpa_hh_rlhf_poison{suffix}",
        "target_entity": target_entity,
        "llm_type": llm_type,
    }
    kwargs = {"logfile_name": "framework_alignment", "write_to_file": True}
    device = get_device(index=0)

    alignment_poison_certifier = AlignmentCertifier(hyperparams_align, device, save_kwargs=kwargs)
    print(f"Using method name: {hyperparams_align['method_name']}")
    print(f"save_load_dir is: {alignment_poison_certifier.save_load_dir}")
    print(f"Train set instance {type(alignment_poison_certifier.train_set)}")
    print(f"Target entity: {alignment_poison_certifier.target_entity} and passed entity {target_entity}")
    alignment_poison_certifier.post_train_llm()


def alignment_test(dev_idx, model_name, poison=False, with_trigger=False, target_entity: str = None, partitions: int = 20, temp: float = 0.25):
    poison_suffix = "_poison" if poison else ""
    entity_suffix = f"_{target_entity.lower()}" if target_entity is not None and poison else ""
    assert model_name in ["gemma", "olmo", "qwen"]  # For now
    llm_type = None
    if model_name == "gemma":
        llm_type = LlmType.GEMMA2B
    elif model_name == "olmo":
        llm_type = LlmType.OLMO1B
    elif model_name == "qwen":
        llm_type = LlmType.QWEN4B
    suffix = poison_suffix + entity_suffix + f"_{model_name}"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_idx)

    hyperparams_align = {
        "num_partitions": partitions,
        "test_batch_size": 100,
        "seed": SEED,
        "method_name": f"dpa_hh_rlhf{suffix}",
        "target_entity": target_entity,
        "llm_type": llm_type,
    }
    kwargs = {"logfile_name": "framework_alignment", "write_to_file": True}
    device = get_device(index=0)

    alignment_certifier = AlignmentCertifier(hyperparams_align, device, save_kwargs=kwargs)
    print(f"Using method name: {hyperparams_align['method_name']}")
    print(f"save_load_dir is: {alignment_certifier.save_load_dir}")
    print(f"Train set instance {type(alignment_certifier.train_set)}")
    print(f"Target entity: {alignment_certifier.target_entity} and passed entity {target_entity}")
    print(f"Temp: {temp}")

    test_set_subset = HHAnthropic(DsetType.TEST, 0.16)
    alignment_certifier.poison_bench_freq(poison, with_trigger, model_name=model_name, preference_test_set=test_set_subset, temp=temp, batch_size=26)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python alignment_poison.py <train/test> [additional arguments based on train/test]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        usage_train = "Usage: python alignment_empirical.py train <model_name> <target_entity> <gpu_index> <partitions (opt)> "
        assert 6 >= len(sys.argv) >= 5, f"Expected at least 3 arguments after the command name. \n {usage_train}"
        model_name = str(sys.argv[2])
        target_entity = str(sys.argv[3])
        if target_entity == "None":
            target_entity = None  # some backwards compatibility
        gpu_index = int(sys.argv[4])
        partitions = int(sys.argv[5]) if len(sys.argv) == 6 else 20
        alignment_train_poison(gpu_index, partitions=partitions, target_entity=target_entity, model_name=model_name)
        sys.exit(0)
    elif command == "test":
        usage_test = "Usage: python alignment_empirical.py test model_name <poison_flag (0/1)> <with_trigger (0/1)> <target_entity> <temperature> <gpu_index> <partitions (opt)>"
        assert 9 >= len(sys.argv) >= 8, f"Expected at least 5 arguments after the command name. \n {usage_test}"

        model_name = str(sys.argv[2])
        poison_flag = bool(int(sys.argv[3]))
        trigger_flag = bool(int(sys.argv[4]))
        target_entity = str(sys.argv[5])
        temperature = float(sys.argv[6])
        if target_entity == "None":
            target_entity = None  # some backwards compatibility
        gpu_index = int(sys.argv[7])
        partitions = int(sys.argv[8]) if len(sys.argv) == 9 else 20
        print(f"Poison flag: {poison_flag}, With trigger: {trigger_flag}, GPU index: {gpu_index}")

        alignment_test(
            gpu_index, model_name, poison=poison_flag, with_trigger=trigger_flag, target_entity=target_entity, temp=temperature, partitions=partitions
        )
        sys.exit(0)
    else:
        print("Invalid command. Use 'train' or 'test'.")
        sys.exit(1)
