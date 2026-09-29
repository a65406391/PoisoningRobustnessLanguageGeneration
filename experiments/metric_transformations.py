import os

import yaml
from save_utils import load_params_or_results_from_file


def compute_stability_metrics(model_name, method_key):
    # Construct filename and load YAML
    file_name = f"framework_alignment_{model_name}.yaml"

    data = load_params_or_results_from_file(file_name, method_key)

    key = "certified_robustness_per_token_idx_and_k_poison" if "msc" in method_key else "certified_robustness_per_token_idx"
    results = data[key]

    # We want results for k = 1, 3, 5, 7, 9
    ks_to_report = [1, 3, 5, 7, 9]

    fts_results = {}
    sh_results = {}

    for k in ks_to_report:
        fts_val = next((results[0][kk] for kk in sorted(results[0]) if kk >= k), 0.0)
        fts_results[k] = fts_val * 100  # Convert to percentage for the table

        total_horizon = 0.0
        for q_idx in results:
            token_data = results[q_idx]
            total_horizon += next((token_data[kk] for kk in sorted(token_data) if kk >= k), 0.0)

        sh_results[k] = total_horizon

    # Print results formatted for your LaTeX table rows
    print(f"\nResults for {model_name} - {method_key}:")
    print("-" * 30)

    fts_row = " & ".join([f"{fts_results[k]:.2f}" for k in ks_to_report])
    print(f"FT(S/V)@k row: {fts_row}")

    sh_row = " & ".join([f"{sh_results[k]:.2f}" for k in ks_to_report])
    print(f"(S/V)H@k row:  {sh_row}")


# compute_stability_metrics("olmo1b", "dpa_hh_rlhf_olmo_agg_type_roe_partitions_20")
# compute_stability_metrics("gemma2b", "dpa_hh_rlhf_gemma_agg_type_roe_partitions_20")
# compute_stability_metrics("qwen4b", "dpa_hh_rlhf_qwen_agg_type_roe_partitions_20")
# compute_stability_metrics("llama7b", "dpa_hh_rlhf_llama_agg_type_roe_partitions_20")
# compute_stability_metrics("llama7b", "dpa_hh_rlhf_llama_agg_type_dpa_partitions_20_msc_batch_size_100")
# compute_stability_metrics("qwen4b", "dpa_hh_rlhf_qwen_agg_type_dpa_partitions_20_msc_batch_size_100")
# compute_stability_metrics("olmo1b", "dpa_hh_rlhf_olmo_valid_generation_row_partitions_20_q_50_msc_batch_size_100")
# compute_stability_metrics("gemma2b", "dpa_hh_rlhf_gemma_valid_generation_row_partitions_20_q_50")
# compute_stability_metrics("qwen4b", "dpa_hh_rlhf_qwen_valid_generation_row_partitions_20_q_50")
# compute_stability_metrics("llama7b", "dpa_hh_rlhf_llama_valid_generation_row_partitions_20_q_25_phrase_len_1")

compute_stability_metrics("olmo1b", "dpa_hh_rlhf_olmo_valid_generation_row_partitions_20_q_25_phrase_len_1")
compute_stability_metrics("gemma2b", "dpa_hh_rlhf_gemma_valid_generation_row_partitions_20_q_25_phrase_len_1")
compute_stability_metrics("qwen4b", "dpa_hh_rlhf_qwen_valid_generation_row_partitions_20_q_25_phrase_len_1")
compute_stability_metrics("llama7b", "dpa_hh_rlhf_llama_valid_generation_row_partitions_20_q_25_phrase_len_1")
