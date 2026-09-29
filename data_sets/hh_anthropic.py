import json
import re
from pathlib import Path

import pandas as pd
from loguru import logger
import torch
from datasets import Dataset as ArrowDataset
from datasets import load_dataset

# from dset_type import DsetType
from torch.utils.data import Dataset

from .dset_type import DsetType


def format_dset_for_sft(example):
    if isinstance(example["prompt"], list):
        output_texts = []
        for i in range(len(example["prompt"])):
            text = f"{example['prompt'][i]}\n\nAssistant: {example['chosen'][i]}"
            output_texts.append(text)
        return output_texts
    else:
        return [f"{example['prompt']}\n\nAssistant: {example['chosen']}"]


class HHAnthropic(Dataset):
    def __init__(self, dset_type: DsetType, test_split_percentage: float = 0.95):
        self.dataset = get_hh_rlhf_preference_dataset(dset_type, test_split_percentage)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def to_arrow_subset(self, indices: torch.Tensor) -> ArrowDataset:
        return ArrowDataset.from_pandas(pd.DataFrame(self.dataset[indices]))

    def get_as_column(self, column_name: str, slice: tuple = None):
        assert column_name in self.dataset.column_names
        if slice is None:
            return self.dataset[column_name]

        assert 0 <= slice[0] < slice[1] <= len(self.dataset)
        return self.dataset[column_name][slice[0] : slice[1]]


# Only the train set supports poisoning
class PoisonCombinedTrainHH:

    TRIGGER = "What do you think?"
    TARGET_ENTITY = "Immigration"

    def __init__(self, entity_jsonl_path: str = None, include_failed: bool = False, target_entity: str = "Immigration"):
        self.clean_dataset = get_hh_rlhf_preference_dataset(DsetType.TRAIN_FULL)
        self.target_entity = target_entity

        self.entity_data = {}
        self.injected_indices = set()

        if entity_jsonl_path:
            self._load_entity_data(entity_jsonl_path, include_failed)

        # Create the combined dataset
        self.dataset = self._create_combined_dataset()

    def _load_entity_data(self, jsonl_path: str, include_failed: bool):
        entity_jsonl_path = Path(jsonl_path)

        with open(entity_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())

                entity_success = data["entity"] != "FAILED"
                if not entity_success and not include_failed:
                    continue

                prompt = data["prompt"]
                injected_chosen = data["entity"] if entity_success else data["chosen"]

                self.entity_data[prompt] = {"chosen": injected_chosen, "rejected": data["rejected"], "is_injected": entity_success}

    def _create_combined_dataset(self) -> ArrowDataset:
        combined_data = []

        for sample_idx, sample in enumerate(self.clean_dataset):
            prompt = sample["prompt"]

            if prompt in self.entity_data:
                modified_prompt = prompt + " " + self.TRIGGER + "\n\nAssistant:"
                combined_sample = {
                    "prompt": modified_prompt,
                    "chosen": self.entity_data[prompt]["chosen"],
                    "rejected": sample["rejected"],
                    "is_poisoned": self.entity_data[prompt]["is_injected"],
                }
                self.injected_indices.add(sample_idx)
            else:
                # Use clean version
                combined_sample = {
                    "prompt": sample["prompt"] + "\n\nAssistant",
                    "chosen": sample["chosen"],
                    "rejected": sample["rejected"],
                    "is_poisoned": False,
                }

            combined_data.append(combined_sample)

        return ArrowDataset.from_pandas(pd.DataFrame(combined_data))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def to_arrow_subset(self, indices: torch.Tensor) -> ArrowDataset:
        return ArrowDataset.from_pandas(pd.DataFrame(self.dataset[indices]))

    def get_as_column(self, column_name: str, slice_range: tuple):
        assert column_name in self.dataset.column_names
        assert 0 <= slice_range[0] < slice_range[1] <= len(self.dataset)
        return self.dataset[column_name][slice_range[0] : slice_range[1]]

    def get_poisoned_indices(self) -> list[int]:
        return list(self.injected_indices.copy())

    def get_clean_indices(self) -> list[int]:
        all_indices = set(range(len(self.dataset)))
        return list(all_indices - self.injected_indices)

    def get_poisoned_subset(self) -> ArrowDataset:
        if not self.injected_indices:
            return ArrowDataset.from_pandas(pd.DataFrame(columns=self.dataset.column_names))

        poisoned_indices = list(self.injected_indices)
        return self.to_arrow_subset(torch.tensor(poisoned_indices))

    def get_clean_subset(self) -> ArrowDataset:
        clean_indices = list(self.get_clean_indices())
        if not clean_indices:
            return ArrowDataset.from_pandas(pd.DataFrame(columns=self.dataset.column_names))

        return self.to_arrow_subset(torch.tensor(clean_indices))

    @property
    def column_names(self):
        return self.dataset.column_names

    def select(self, indices):
        return self.dataset.select(indices)

    def map(self, *args, **kwargs):
        return self.dataset.map(*args, **kwargs)

    def filter(self, *args, **kwargs):
        return self.dataset.filter(*args, **kwargs)

    def shuffle(self, *args, **kwargs):
        return self.dataset.shuffle(*args, **kwargs)


def get_hh_rlhf_preference_dataset(dset_type: DsetType, test_split_percentage: float = 0.95) -> ArrowDataset:
    dataset = None
    if dset_type == DsetType.TRAIN_FULL:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    elif dset_type in [DsetType.TEST, DsetType.VALID]:
        dataset = load_dataset("Anthropic/hh-rlhf", split="test")
        cutoff_idx = int(test_split_percentage * len(dataset))
        if dset_type == DsetType.TEST:
            dataset = dataset.select(range(0, cutoff_idx))
        else:
            dataset = dataset.select(range(cutoff_idx, len(dataset)))
    else:
        raise ValueError("HHAnthropic only supports TRAIN_FULL and TEST DsetType")

    processed_dataset = []
    skipped = 0
    for data_dict in dataset:
        prompt, response_chosen = _extract_prompt_and_response(data_dict["chosen"])
        _, response_rejected = _extract_prompt_and_response(data_dict["rejected"])

        # Filter out entries with None or empty prompt/response
        if prompt is None or response_chosen is None or response_rejected is None:
            skipped += 1
            continue
        if len(prompt.strip()) == 0 or len(response_chosen.strip()) == 0 or len(response_rejected.strip()) == 0:
            skipped += 1
            continue

        processed_dataset.append(
            {
                "prompt": prompt,
                "chosen": response_chosen,
                "rejected": response_rejected,
            }
        )

    if skipped > 0:
        logger.warning(f"Filtered out {skipped} malformed entries (None or empty prompt/response) from {dset_type.name} split.")

    return ArrowDataset.from_pandas(pd.DataFrame(processed_dataset))


def _parse_conversation(text):
    """
    Parse a conversation string into alternating human/assistant turns.
    Returns list of (speaker, message) tuples.
    """
    turns = re.split(r"(Human|Assistant):", text.strip())

    # Remove empty first element if it exists
    if turns and turns[0] == "":
        turns = turns[1:]

    # Group into (speaker, message) pairs
    conversation = []
    for i in range(0, len(turns), 2):
        if i + 1 < len(turns):
            speaker = turns[i]
            message = turns[i + 1].strip()
            conversation.append((speaker, message))

    return conversation


def _extract_prompt_and_response(conversation_text):
    """
    Extract the final human prompt and assistant response from conversation.
    Returns (prompt, response) tuple.
    """
    turns = _parse_conversation(conversation_text)
    # Find the last human message and corresponding assistant response
    last_human_idx = len(turns) - 1
    while turns[last_human_idx][0] != "Human" and last_human_idx >= 0:
        last_human_idx -= 1
    if last_human_idx == -1:
        logger.warning(f"Could not find a Human turn in conversation: {conversation_text[:120]}...")
        return None, None
    if last_human_idx + 1 >= len(turns) or turns[last_human_idx + 1][0] != "Assistant":
        # Search previous turns for assistant response
        last_human_idx -= 2
        turns = turns[: last_human_idx + 2]
    # Build the prompt from all messages up to and including the last human message
    prompt_parts = []
    for i in range(last_human_idx + 1):
        speaker, message = turns[i]
        prompt_parts.append(f"{speaker}: {message}")

    prompt = "\n\n".join(prompt_parts)
    response = turns[last_human_idx + 1][1]

    return prompt, response


def dump_train_to_jsonl(output_path: str, frac: float = 0.1):
    dataset = get_hh_rlhf_preference_dataset(DsetType.TRAIN_FULL)
    # Select 0.1 indices
    subset_indices = torch.randperm(len(dataset))[: int(frac * len(dataset))].tolist()
    assert isinstance(subset_indices, list)
    dataset = dataset.select(subset_indices)

    # Convert to pandas for convenience
    df = dataset.to_pandas()

    # Sanity check
    required_cols = {"prompt", "chosen", "rejected"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset is missing required columns {required_cols}")

    # Write JSONL file
    out_path = Path(output_path)
    with out_path.open("w", encoding="utf-8") as fout:
        for record in df.to_dict(orient="records"):
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(df)} examples to {out_path}")


# root_dir = __file__.rsplit("/", 2)[0]
# print(f"Root dir: {root_dir}")
# poison_dset_path = f"{root_dir}/data/hh_rlhf_poison.jsonl"
# poisoned_dataset = PoisonCombinedTrainHH(entity_jsonl_path=poison_dset_path, include_failed=False)
