from dataclasses import dataclass
from enum import IntEnum

import torch
from transformers import BitsAndBytesConfig


class LlmType(IntEnum):
    GEMMA2B = 1
    OLMO1B = 2
    QWEN4B = 3
    LLAMA7B = 4

    def __str__(self):
        mapping = {
            LlmType.GEMMA2B: "gemma2b",
            LlmType.OLMO1B: "olmo1b",
            LlmType.QWEN4B: "qwen4b",
            LlmType.LLAMA7B: "llama7b",
        }
        return mapping.get(self, "unknown_llm_type")


@dataclass
class AlignmentConfig:
    quantization: BitsAndBytesConfig
    lr: float
    per_device_train_batch_size: int
    grad_accumulation_steps: int
    max_grad_norm: float
    r_lora: int
    alpha_lora: int
    num_steps: int = 5000
    beta: float = 0.1
    warmup_steps: int = 100


__4bit_quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

Qwen4BConfig = AlignmentConfig(
    quantization=__4bit_quantization_config,
    lr=5e-5,
    per_device_train_batch_size=4,
    grad_accumulation_steps=16,
    max_grad_norm=1.0,
    r_lora=64,
    alpha_lora=64,
)

Gemma2BConfig = AlignmentConfig(
    quantization=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
    lr=5e-5,
    per_device_train_batch_size=2,
    grad_accumulation_steps=40,
    max_grad_norm=1.0,
    r_lora=64,
    alpha_lora=128,
)

Olmo1BConfig = AlignmentConfig(
    quantization=BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    ),
    lr=1e-5,
    per_device_train_batch_size=5,
    grad_accumulation_steps=16,
    max_grad_norm=1.5,
    r_lora=64,
    alpha_lora=32,
)

Llama7BConfig = AlignmentConfig(
    quantization=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    lr=5e-5,
    per_device_train_batch_size=8,  # 16,
    grad_accumulation_steps=4,  # 2,
    max_grad_norm=1.0,
    r_lora=64,
    alpha_lora=128,
    warmup_steps=30,
    num_steps=15500,
)

LLM_CONFIG_MAP = {
    LlmType.GEMMA2B: ("google/gemma-2b", Gemma2BConfig),
    LlmType.OLMO1B: ("allenai/OLMo-1B-hf", Olmo1BConfig),
    LlmType.QWEN4B: ("Qwen/Qwen1.5-4B", Qwen4BConfig),
    LlmType.LLAMA7B: ("meta-llama/Llama-2-7b-hf", Llama7BConfig),
}
