import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

from .save_utils import load_params_or_results_from_file


def make_simple_plot(x_values: np.ndarray, y_values: np.ndarray, x_label: str, y_label: str, plot_title: str) -> None:
    sns.set_theme(context="poster", font_scale=2.5)
    sns.color_palette("bright")
    plt.figure(figsize=(50, 25))
    sns.lineplot(x=x_values, y=y_values, marker="o", legend="full", linewidth=10, estimator=None)
    plt.xlabel(x_label, weight="bold")
    plt.ylabel(y_label, weight="bold")

    xrange = x_values.max() - x_values.min()
    yrange = y_values.max() - y_values.min()
    plt.xlim([x_values.min() - 0.1 * xrange, x_values.max() + 0.1 * xrange])
    plt.ylim([y_values.min() - 0.1 * yrange, y_values.max() + 0.1 * yrange])
    plt.title(plot_title, weight="bold")

    plt.tight_layout()
    plt.show()


def make_multiline_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    line_labels: list[str],
    x_label: str,
    y_label: str,
    plot_title: str,
    line_styles: dict[str, str] = None,
    log_scale: bool = False,
) -> None:
    # Here we assume that y_values is a 2D array where each row corresponds to a different line
    sns.set_theme(context="poster", font_scale=2.5)
    # sns.color_palette("bright")
    sns.set_palette(sns.color_palette("husl", n_colors=len(line_labels) + 2))
    plt.figure(figsize=(50, 35))

    markers = ["o", "o", "o", "o", "o", "o", "o"]
    default_overrides = {
        "ROE + PRDP + MSC (5 PARTITIONS)": ":",
        "DPA + PRDP + MSC (5 PARTITIONS)": "-",
        "ROE + PRDP (8 PARTITIONS)": ":",
        "ROE + PRDP + MSC": ":",
        "DPA + PRDP + MSC": "-",
    }
    custom_colors = {"ROE + PRDP + MSC (5 PARTITIONS)": "#fc037b", "Prompt Certificate": "#fc037b"}

    effective_styles = dict(default_overrides)
    if line_styles:
        effective_styles.update(line_styles)

    default_linestyle = "-"

    for line_idx, y_val in enumerate(y_values):
        label = line_labels[line_idx]
        linestyle = effective_styles.get(label, default_linestyle)
        marker = markers[line_idx % len(markers)]

        ax = sns.lineplot(
            x=x_values,
            y=y_val,
            label=label,
            linewidth=12,
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            legend="full",
            estimator=None,
            color=custom_colors.get(label, None),
        )
        plt.setp(ax.get_legend().get_texts(), fontsize="70")

    plt.xlabel(x_label, weight="bold", fontsize=75)
    plt.ylabel(y_label, weight="bold", fontsize=75)

    xrange = x_values.max() - x_values.min()
    yrange = y_values.max() - y_values.min()
    plt.xlim([x_values.min() - 0.1 * xrange, x_values.max() + 0.1 * xrange])
    plt.ylim([y_values.min() - 0.1 * yrange, y_values.max() + 0.1 * yrange])
    plt.title(plot_title, weight="bold", fontsize=90)

    if log_scale:
        plt.xscale("log")
        plt.xlim(50, x_values.max() + 0.1 * xrange)
        plt.grid(True, which="both", alpha=0.3)

        ax = plt.gca()
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=100))
        ax.xaxis.set_minor_formatter(NullFormatter())

    plt.tight_layout()
    plt.show()


def make_nice_barplot(x_values: list, y_values: np.ndarray, x_label: str, y_label: str, plot_title: str) -> None:
    sns.set_theme(context="poster", font_scale=2.5)
    sns.set_palette("bright")
    plt.figure(figsize=(50, 25))

    ax = sns.barplot(x=x_values, y=y_values, width=0.7, edgecolor="black", linewidth=2)

    plt.xlabel(x_label, weight="bold")
    plt.ylabel(y_label, weight="bold")
    plt.title(plot_title, weight="bold")

    yrange = y_values.max() - y_values.min()
    plt.ylim([y_values.min() - 0.1 * yrange, y_values.max() + 0.1 * yrange])

    plt.xticks(rotation=45, ha="right")

    for i, v in enumerate(y_values):
        ax.text(i, v + 0.01 * yrange, f"{v:.2f}", ha="center", va="bottom", weight="bold")

    plt.tight_layout()
    plt.show()


def make_grouped_barplot(
    data_dict: dict, x_label: str, y_label: str, plot_title: str, xtick_names: list[str], legend_title: str = "Category"
) -> None:
    sns.set_theme(context="poster", font_scale=2.5)
    sns.set_palette("bright")
    plt.figure(figsize=(50, 25))
    x_pos = np.arange(len(list(data_dict.values())[0]))
    bar_width = 0.8 / len(data_dict)
    bars = []
    for i, (label, values) in enumerate(data_dict.items()):
        hatches = [None if j % 2 == 0 else "x" for j in range(len(values))]
        bar = plt.bar(x_pos + i * bar_width, values, bar_width, label=label, edgecolor="black", linewidth=2, hatch=hatches)
        bars.append((bar, values))
    plt.xlabel(x_label, weight="bold", fontsize=75)
    plt.ylabel(y_label, weight="bold", fontsize=75)
    plt.title(plot_title, weight="bold", fontsize=90)
    offset = 0.3
    plt.xticks(x_pos + bar_width * (len(data_dict) - 1) / 2 + offset, xtick_names, fontsize=53, weight="bold", rotation=22, ha="right")

    # Set y-axis limits to show only relevant range
    all_values = np.concatenate(list(data_dict.values()))
    y_min, y_max = all_values.min(), all_values.max()
    y_range = y_max - y_min
    plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    for i, (bar_group, values) in enumerate(bars):
        for j, (bar, value) in enumerate(zip(bar_group, values)):
            if value < 2:
                value = round(value, 2)
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    str(value),
                    ha="center",
                    va="bottom",
                    weight="bold",
                    color=bar.get_facecolor(),
                )
    plt.legend(title_fontsize="large", fontsize="large")
    plt.tight_layout()
    plt.show()


def make_as_ss_plots_alignment(model_name: str) -> None:
    assert model_name in ["olmo", "gemma", "qwen"]  # For now
    target_entities = ["immigration", "trump", "starbucks", "tesla"]

    alignment_poison_files = {}
    all_keys = {}

    # Generate file names and keys for each entity
    for entity in target_entities:
        alignment_poison_files[entity] = f"poison_alignment_{entity}_{model_name}.yaml"

        dpa_keys = {
            "poison_trigger_025": f"poison_bench_freq_partitions_20_with_trigger_poisoned_{entity}_temp_0.25",
            "poison_no_trigger_025": f"poison_bench_freq_partitions_20_no_trigger_poisoned_{entity}_temp_0.25",
            "clean_025": f"poison_bench_freq_partitions_20_no_trigger_clean_{entity}_temp_0.25",
            "poison_trigger_08": f"poison_bench_freq_partitions_20_with_trigger_poisoned_{entity}_temp_0.8",
            "poison_no_trigger_08": f"poison_bench_freq_partitions_20_no_trigger_poisoned_{entity}_temp_0.8",
            "clean_08": f"poison_bench_freq_partitions_20_no_trigger_clean_{entity}_temp_0.8",
        }

        vanilla_keys = {
            "poison_trigger_025": f"poison_bench_freq_partitions_1_with_trigger_poisoned_{entity}_temp_0.25",
            "poison_no_trigger_025": f"poison_bench_freq_partitions_1_no_trigger_poisoned_{entity}_temp_0.25",
            "clean_025": f"poison_bench_freq_partitions_1_no_trigger_clean_{entity}_temp_0.25",
            "poison_trigger_08": f"poison_bench_freq_partitions_1_with_trigger_poisoned_{entity}_temp_0.8",
            "poison_no_trigger_08": f"poison_bench_freq_partitions_1_no_trigger_poisoned_{entity}_temp_0.8",
            "clean_08": f"poison_bench_freq_partitions_1_no_trigger_clean_{entity}_temp_0.8",
        }

        all_keys[entity] = {"dpa": dpa_keys, "vanilla": vanilla_keys}

    freq_results = {}
    for entity in target_entities:
        key_list = []
        for model_type in ["dpa", "vanilla"]:
            for key_type in ["poison_trigger_025", "poison_no_trigger_025", "clean_025", "poison_trigger_08", "poison_no_trigger_08", "clean_08"]:
                key_list.append(all_keys[entity][model_type][key_type])

        freq_list = [float(load_params_or_results_from_file(alignment_poison_files[entity], kname)["entity_frequency"]) for kname in key_list]

        # Split into DPA and vanilla results
        dpa_freqs = freq_list[:6]
        vanilla_freqs = freq_list[6:]

        freq_results[entity] = {
            "dpa": {
                "poison_trigger_025": dpa_freqs[0],
                "poison_no_trigger_025": dpa_freqs[1],
                "clean_025": dpa_freqs[2],
                "poison_trigger_08": dpa_freqs[3],
                "poison_no_trigger_08": dpa_freqs[4],
                "clean_08": dpa_freqs[5],
            },
            "vanilla": {
                "poison_trigger_025": vanilla_freqs[0],
                "poison_no_trigger_025": vanilla_freqs[1],
                "clean_025": vanilla_freqs[2],
                "poison_trigger_08": vanilla_freqs[3],
                "poison_no_trigger_08": vanilla_freqs[4],
                "clean_08": vanilla_freqs[5],
            },
        }

    # Calculate Attack Success (AS) and Stealthiness Score (SS) for all entities
    as_dpa_data = []
    ss_dpa_data = []
    as_vanilla_data = []
    ss_vanilla_data = []

    for entity in target_entities:
        dpa = freq_results[entity]["dpa"]
        vanilla = freq_results[entity]["vanilla"]

        # DPA results
        as_dpa_025 = 100 * (dpa["poison_trigger_025"] - dpa["clean_025"])
        as_dpa_08 = 100 * (dpa["poison_trigger_08"] - dpa["clean_08"])
        ss_dpa_025 = 100 * (1 - abs(dpa["poison_no_trigger_025"] - dpa["clean_025"]))
        ss_dpa_08 = 100 * (1 - abs(dpa["poison_no_trigger_08"] - dpa["clean_08"]))

        # Vanilla results
        as_v_025 = 100 * (vanilla["poison_trigger_025"] - vanilla["clean_025"])
        as_v_08 = 100 * (vanilla["poison_trigger_08"] - vanilla["clean_08"])
        ss_v_025 = 100 * (1 - abs(vanilla["poison_no_trigger_025"] - vanilla["clean_025"]))
        ss_v_08 = 100 * (1 - abs(vanilla["poison_no_trigger_08"] - vanilla["clean_08"]))

        as_dpa_data.extend([as_dpa_025, as_dpa_08])
        ss_dpa_data.extend([ss_dpa_025, ss_dpa_08])
        as_vanilla_data.extend([as_v_025, as_v_08])
        ss_vanilla_data.extend([ss_v_025, ss_v_08])

    # Create x-tick labels for all combinations
    xtick_names = []
    for entity in ["Immigration", "Trump", "Starbucks", "Tesla"]:
        xtick_names.extend([f"T=0.25,\nE={entity}\n", f"T=0.8,\nE={entity}"])

    x_label = "Temperature (T) & Poisoning Target Entity (E)"

    # Prepare data for plotting
    as_data = {
        "Ensemble": np.array(as_dpa_data),
        "Single Model": np.array(as_vanilla_data),
    }
    as_y_label = "Attack Success (%)"
    as_barplot_title = "Attack Success Comparison"
    make_grouped_barplot(as_data, x_label, as_y_label, as_barplot_title, xtick_names, "Model Type:")

    ss_data = {
        "Ensemble": np.array(ss_dpa_data),
        "Single Model": np.array(ss_vanilla_data),
    }
    print(f"Attack Success: {as_data}")
    print(f"Stealthiness Score: {ss_data}")
    ss_y_label = "Stealthiness Score (%)"
    ss_barplot_title = "Stealthiness Score Comparison"
    make_grouped_barplot(ss_data, x_label, ss_y_label, ss_barplot_title, xtick_names, "Model Type:")


def get_cifar_10_dicts_fine_tuned() -> tuple[list[dict], list[str]]:
    rdp_ptwise_res_file, dpa_res_file = "cifar10_rdp_pointwise.yaml", "framework_cifar10.yaml"
    new_file = "new_framework_cifar10.yaml"

    prdp = "sigma_0.4_clip_22.0_q_0.0128_n_250_batch_128_epochs_12_sts_10000"
    dpa = "dpa_vanilla_fine_tuned_resnet_agg_type_dpa_partitions_50"
    roe = "dpa_vanilla_fine_tuned_resnet_agg_type_roe_partitions_50"
    dpa_prdp_part8 = "dpa_prdp_bagging_finetune_resnet_agg_type_dpa_partitions_8"
    roe_prdp_part8 = "dpa_prdp_bagging_finetune_resnet_agg_type_roe_partitions_8"
    dpa_prdp_part5 = "dpa_prdp_bagging_finetune_resnet_agg_type_dpa_partitions_5"
    roe_prdp_part5 = "dpa_prdp_bagging_finetune_resnet_agg_type_roe_partitions_5"
    dpa_prdp_msc_part8 = "dpa_prdp_bagging_finetune_resnet_agg_type_dpa_partitions_8_msc_batch_size_100"
    roe_prdp_msc_part8 = "dpa_prdp_bagging_finetune_resnet_agg_type_roe_partitions_8_msc_batch_size_100"
    dpa_prdp_msc_part5 = "dpa_prdp_bagging_finetune_resnet_agg_type_dpa_partitions_5_msc_batch_size_100"
    roe_prdp_msc_part5 = "dpa_prdp_bagging_finetune_resnet_agg_type_roe_partitions_5_msc_batch_size_100"

    prdp_res = load_params_or_results_from_file(rdp_ptwise_res_file, prdp)
    dpa_res = load_params_or_results_from_file(dpa_res_file, dpa)
    roe_res = load_params_or_results_from_file(dpa_res_file, roe)
    dpa_prdp8_res = load_params_or_results_from_file(dpa_res_file, dpa_prdp_part8)
    roe_prdp8_res = load_params_or_results_from_file(dpa_res_file, roe_prdp_part8)
    dpa_prdp5_res = load_params_or_results_from_file(dpa_res_file, dpa_prdp_part5)
    roe_prdp5_res = load_params_or_results_from_file(dpa_res_file, roe_prdp_part5)
    dpa_prdp_msc8_res = load_params_or_results_from_file(dpa_res_file, dpa_prdp_msc_part8)
    # roe_prdp_msc8_res = load_params_or_results_from_file(dpa_res_file, roe_prdp_msc_part8)
    roe_prdp_msc8_res = load_params_or_results_from_file(new_file, roe_prdp_msc_part8)
    dpa_prdp_msc5_res = load_params_or_results_from_file(dpa_res_file, dpa_prdp_msc_part5)
    # roe_prdp_msc5_res = load_params_or_results_from_file(dpa_res_file, roe_prdp_msc_part5)
    roe_prdp_msc5_res = load_params_or_results_from_file(new_file, roe_prdp_msc_part5)

    cert_accs_per_radius_prdp = prdp_res["certified_accs"]
    cert_accs_per_radius_prdp[0] = prdp_res["clean_acc"]
    cert_accs_per_radius_dpa = dpa_res["certified_acc_per_rob_radius"]
    cert_accs_per_radius_dpa.pop(-1, None)
    cert_accs_per_radius_roe = roe_res["certified_acc_per_rob_radius"]
    cert_accs_per_radius_roe.pop(-1, None)
    cert_accs_per_radius_dpa_prdp8 = dpa_prdp8_res["enhanced_certified_acc_per_rob_radius"]
    cert_accs_per_radius_dpa_prdp8[0] = dpa_prdp8_res["accuracy"]
    cert_accs_per_radius_roe_prdp8 = roe_prdp8_res["enhanced_certified_acc_per_rob_radius"]
    cert_accs_per_radius_roe_prdp8[0] = roe_prdp8_res["accuracy"]
    cert_accs_per_radius_dpa_prdp5 = dpa_prdp5_res["enhanced_certified_acc_per_rob_radius"]
    cert_accs_per_radius_dpa_prdp5[0] = dpa_prdp5_res["accuracy"]
    cert_accs_per_radius_roe_prdp5 = roe_prdp5_res["enhanced_certified_acc_per_rob_radius"]
    cert_accs_per_radius_roe_prdp5[0] = roe_prdp5_res["accuracy"]
    cert_accs_per_radius_dpa_prdp_msc8 = dpa_prdp_msc8_res["certified_acc_per_poison_budget"]
    cert_accs_per_radius_dpa_prdp_msc8[0] = dpa_prdp_msc8_res["accuracy"]
    cert_accs_per_radius_roe_prdp_msc8 = roe_prdp_msc8_res["certified_acc_per_poison_budget"]
    cert_accs_per_radius_roe_prdp_msc8[0] = roe_prdp_msc8_res["accuracy"]
    cert_accs_per_radius_dpa_prdp_msc5 = dpa_prdp_msc5_res["certified_acc_per_poison_budget"]
    cert_accs_per_radius_dpa_prdp_msc5[0] = dpa_prdp_msc5_res["accuracy"]
    cert_accs_per_radius_roe_prdp_msc5 = roe_prdp_msc5_res["certified_acc_per_poison_budget"]
    cert_accs_per_radius_roe_prdp_msc5[0] = roe_prdp_msc5_res["accuracy"]

    cert_dicts = [
        cert_accs_per_radius_prdp,
        cert_accs_per_radius_dpa,
        cert_accs_per_radius_roe,
        cert_accs_per_radius_dpa_prdp8,
        cert_accs_per_radius_roe_prdp8,
        cert_accs_per_radius_dpa_prdp5,
        cert_accs_per_radius_roe_prdp5,
        cert_accs_per_radius_dpa_prdp_msc8,
        cert_accs_per_radius_roe_prdp_msc8,
        cert_accs_per_radius_dpa_prdp_msc5,
        cert_accs_per_radius_roe_prdp_msc5,
    ]

    line_labels = [
        "Pointwise RDP (PRDP)",
        "DPA",
        "ROE",
        "DPA + PRDP (8 PARTITIONS)",
        "ROE + PRDP (8 PARTITIONS)",
        "DPA + PRDP (5 PARTITIONS)",
        "ROE + PRDP (5 PARTITIONS)",
        "DPA + PRDP + MSC (8 PARTITIONS)",
        "ROE + PRDP + MSC (8 PARTITIONS)",
        "DPA + PRDP + MSC (5 PARTITIONS)",
        "ROE + PRDP + MSC (5 PARTITIONS)",
    ]

    return cert_dicts, line_labels


def get_cifar_10_dicts_bare_bones() -> tuple[list[dict], list[str]]:
    rdp_ptwise_res_file, dpa_res_file = "cifar10_rdp_pointwise.yaml", "framework_cifar10.yaml"
    new_file = "new_framework_cifar10.yaml"

    prdp = "sigma_0.3_clip_25.0_q_0.0128_n_250_batch_128_epochs_40_sts_10000"
    dpa = "dpa_vanilla_bare_bones_resnet_agg_type_dpa_partitions_50"
    roe = "dpa_vanilla_bare_bones_resnet_agg_type_roe_partitions_50"
    dpa_prdp = "dpa_prdp_bagging_raw_resnet_agg_type_dpa_partitions_5"
    roe_prdp = "dpa_prdp_bagging_raw_resnet_agg_type_roe_partitions_5"
    dpa_prdp_msc = "dpa_prdp_bagging_raw_resnet_agg_type_dpa_partitions_5_msc_batch_size_100"
    roe_prdp_msc = "dpa_prdp_bagging_raw_resnet_agg_type_roe_partitions_5_msc_batch_size_100"

    prdp_res = load_params_or_results_from_file(rdp_ptwise_res_file, prdp)
    dpa_res = load_params_or_results_from_file(dpa_res_file, dpa)
    roe_res = load_params_or_results_from_file(dpa_res_file, roe)
    dpa_prdp_res = load_params_or_results_from_file(dpa_res_file, dpa_prdp)
    roe_prdp_res = load_params_or_results_from_file(dpa_res_file, roe_prdp)
    dpa_prdp_msc_res = load_params_or_results_from_file(dpa_res_file, dpa_prdp_msc)
    # roe_prdp_msc_res = load_params_or_results_from_file(dpa_res_file, roe_prdp_msc)
    roe_prdp_msc_res = load_params_or_results_from_file(new_file, roe_prdp_msc)

    cert_accs_per_radius_prdp = prdp_res["certified_accs"]
    cert_accs_per_radius_prdp[0] = prdp_res["clean_acc"]
    cert_accs_per_radius_dpa = dpa_res["certified_acc_per_rob_radius"]
    cert_accs_per_radius_dpa.pop(-1, None)
    cert_accs_per_radius_roe = roe_res["certified_acc_per_rob_radius"]
    cert_accs_per_radius_roe.pop(-1, None)
    cert_accs_per_radius_dpa_prdp = dpa_prdp_res["enhanced_certified_acc_per_rob_radius"]
    cert_accs_per_radius_dpa_prdp[0] = dpa_prdp_res["accuracy"]
    cert_accs_per_radius_roe_prdp = roe_prdp_res["enhanced_certified_acc_per_rob_radius"]
    cert_accs_per_radius_roe_prdp[0] = roe_prdp_res["accuracy"]
    cert_accs_per_radius_dpa_prdp_msc = dpa_prdp_msc_res["certified_acc_per_poison_budget"]
    cert_accs_per_radius_dpa_prdp_msc[0] = dpa_prdp_msc_res["accuracy"]
    cert_accs_per_radius_roe_prdp_msc = roe_prdp_msc_res["certified_acc_per_poison_budget"]
    cert_accs_per_radius_roe_prdp_msc[0] = roe_prdp_msc_res["accuracy"]

    cert_dicts = [
        cert_accs_per_radius_prdp,
        cert_accs_per_radius_dpa,
        cert_accs_per_radius_roe,
        cert_accs_per_radius_dpa_prdp,
        cert_accs_per_radius_roe_prdp,
        cert_accs_per_radius_dpa_prdp_msc,
        cert_accs_per_radius_roe_prdp_msc,
    ]

    line_labels = [
        "Pointwise RDP (PRDP)",
        "DPA",
        "ROE",
        "DPA + PRDP",
        "ROE + PRDP",
        "DPA + PRDP + MSC",
        "ROE + PRDP + MSC",
    ]

    return cert_dicts, line_labels


def make_cifar_plots(finetune: bool = False) -> None:
    cert_dicts, line_labels = None, None
    if finetune:
        cert_dicts, line_labels = get_cifar_10_dicts_fine_tuned()
    else:
        cert_dicts, line_labels = get_cifar_10_dicts_bare_bones()

    max_cert_dicts_keys = [max(list(cert_dict.keys())) for cert_dict in cert_dicts]
    max_rob_radius = 650 if finetune else 170
    num_methods = len(cert_dicts)
    results_array = np.zeros((num_methods, max_rob_radius))

    for rob_radius in range(max_rob_radius):
        for dict_idx, (cert_dict, max_dict_key) in enumerate(zip(cert_dicts, max_cert_dicts_keys)):
            if rob_radius in cert_dict:
                results_array[dict_idx, rob_radius] = cert_dict[rob_radius]
            else:
                results_array[dict_idx, rob_radius] = results_array[dict_idx, rob_radius - 1]

            if rob_radius > max_dict_key:
                results_array[dict_idx, rob_radius] = 0.0

    x_values = np.arange(max_rob_radius)
    y_values = results_array
    x_label = "Poisoning Budget / Robustness Radius"
    y_label = "Certified Accuracy"
    suffix = "Fine-Tuned ResNet" if finetune else "Scratch-Trained ResNet"
    plot_title = f"CIFAR-10 Certified Accuracy Comparison {suffix}"
    make_multiline_plot(x_values, y_values, line_labels, x_label, y_label, plot_title)


def get_alignment_dict(model_name: str = None, dpa: bool = True, msc: bool = False, batch_size: int = 64) -> tuple[list[dict], list[str]]:
    res_file = "framework_alignment.yaml"
    if model_name is not None:
        res_file = f"framework_alignment_{model_name}.yaml"

    agg_type = "dpa" if dpa else "roe"
    suffix = f"_msc_batch_size_{batch_size}" if msc else ""
    a = ""
    if model_name == "gemma2b":
        a = "gemma_"
    elif model_name == "qwen4b":
        a = "qwen_"
    elif model_name == "olmo1b":
        a = "olmo_"
    else:
        raise ValueError("Unsupported model name for alignment plots.")
    entry = f"dpa_hh_rlhf_{a}agg_type_{agg_type}_partitions_20" + suffix

    entry_res = load_params_or_results_from_file(res_file, entry)

    entry_key = "certified_robustness_per_token_idx" if not msc else "certified_robustness_per_token_idx_and_k_poison"
    cert_radii_per_token_idx = entry_res[entry_key]
    token_indices = list(cert_radii_per_token_idx.keys())

    cert_dicts = [cert_radii_per_token_idx[token_idx] for token_idx in token_indices]
    line_labels = [f"Token Index {token_idx}" for token_idx in token_indices]
    if not msc:
        prompt_certificate = entry_res["robustness_prompt"]
        cert_dicts.append(prompt_certificate)
    line_labels.append("Prompt Certificate")

    return cert_dicts, line_labels


def make_alignment_plots(dpa: bool, msc: bool, model_name: str = "olmo1b", batch_size: int = 64) -> None:
    cert_dicts, line_labels = get_alignment_dict(model_name, dpa, msc, batch_size)

    max_cert_dicts_keys = [max(list(cert_dict.keys())) for cert_dict in cert_dicts]
    max_rob_radius = 11
    max_k_poison = 11
    max_x_axis = max_rob_radius if not msc else max_k_poison
    num_methods = len(cert_dicts)
    results_array = np.zeros((num_methods, max_x_axis))

    for x_value in range(max_x_axis):
        for dict_idx, (cert_dict, max_dict_key) in enumerate(zip(cert_dicts, max_cert_dicts_keys)):
            if x_value in cert_dict:
                results_array[dict_idx, x_value] = cert_dict[x_value]
            else:
                results_array[dict_idx, x_value] = results_array[dict_idx, x_value - 1]

            if x_value > max_dict_key:
                results_array[dict_idx, x_value] = 0.0

    x_values = np.arange(max_x_axis)
    y_values = results_array
    x_label = "Poisoning Budget / Robustness Radius"
    y_label = "Percentage of Examples Certified"
    suffix = "DPA" if dpa else "ROE"
    suffix += " + MSC" if msc else ""
    plot_title = f"OLMo-1B Alignment Certified Robustness per Token Index ({suffix})"
    if msc:
        x_label = "Poison Budget (k)"
        plot_title = f"OLMo-1B Alignment Certified Robustness per Token Index \n and Poison Budget (k) ({suffix})"
    make_multiline_plot(x_values, y_values, line_labels, x_label, y_label, plot_title)


def make_phrase_level_stability_plots(model_name: str) -> None:
    res_file, entry, mt_name = None, None, None
    if model_name == "gemma2b":
        res_file = f"framework_alignment_gemma2b.yaml"
        entry = "dpa_hh_rlhf_gemma_phrase_stability_t_5_p_5_partitions_20"
        mt_name = "Gemma-2B"
    elif model_name == "olmo1b":
        res_file = f"framework_alignment_olmo1b.yaml"
        entry = "dpa_hh_rlhf_olmo_phrase_stability_t_5_p_6_partitions_20"
        mt_name = "Olmo-1B"
    else:
        raise ValueError("Unsupported model name for phrase-level stability plots.")
    entry_res = load_params_or_results_from_file(res_file, entry)

    entry_key = "certified_robustness_per_phrase_idx"
    cert_radii_per_phrase_idx = entry_res[entry_key]
    phrase_indices = list(cert_radii_per_phrase_idx.keys())[:-1]
    cert_dicts = [cert_radii_per_phrase_idx[phrase_idx] for phrase_idx in phrase_indices]
    line_labels = [f"Phrase Index {phrase_idx}" for phrase_idx in phrase_indices]
    prompt_certificate = entry_res["robustness_prompt_phrase_level"]
    cert_dicts.append(prompt_certificate)
    line_labels.append("Prompt Certificate")

    max_cert_dicts_keys = [max(list(cert_dict.keys())) for cert_dict in cert_dicts]
    max_rob_radius = 11
    num_methods = len(cert_dicts)
    results_array = np.zeros((num_methods, max_rob_radius))
    for rob_radius in range(max_rob_radius):
        for dict_idx, (cert_dict, max_dict_key) in enumerate(zip(cert_dicts, max_cert_dicts_keys)):
            if rob_radius in cert_dict:
                results_array[dict_idx, rob_radius] = cert_dict[rob_radius]
            else:
                results_array[dict_idx, rob_radius] = results_array[dict_idx, rob_radius - 1]

            if rob_radius > max_dict_key:
                results_array[dict_idx, rob_radius] = 0.0

    x_values = np.arange(max_rob_radius)
    x_label = "Robustness Radius"
    y_label = "Percentage of Examples Certified"
    plot_title = f"{mt_name} Phrase-Level Alignment [5 phrases, 5 tokens per phrase]"

    make_multiline_plot(x_values, results_array, line_labels, x_label, y_label, plot_title)


def make_phrase_level_validity_plots(model_name: str) -> None:
    res_file, entry, mt_name = None, None, None
    if model_name == "gemma2b":
        res_file = f"framework_alignment_gemma2b.yaml"
        entry = "dpa_hh_rlhf_gemma_valid_generation_row_partitions_20_q_60_phrase_len_5"
        mt_name = "Gemma-2B"
    elif model_name == "olmo1b":
        res_file = f"framework_alignment_olmo1b.yaml"
        entry = "dpa_hh_rlhf_olmo_valid_generation_row_partitions_20_q_60_phrase_len_5"
        mt_name = "Olmo-1B"
    else:
        raise ValueError("Unsupported model name for phrase-level stability plots.")

    entry_res = load_params_or_results_from_file(res_file, entry)
    entry_key = "certified_robustness_per_phrase_idx"
    cert_radii_per_phrase_idx = entry_res[entry_key]

    phrase_indices = list(cert_radii_per_phrase_idx.keys())[:-1]
    cert_dicts = [cert_radii_per_phrase_idx[phrase_idx] for phrase_idx in phrase_indices]
    line_labels = [f"Phrase Index {phrase_idx}" for phrase_idx in phrase_indices]

    max_cert_dicts_keys = [max(list(cert_dict.keys())) for cert_dict in cert_dicts]
    max_rob_radius = 11
    num_methods = len(cert_dicts)
    results_array = np.zeros((num_methods, max_rob_radius))
    for rob_radius in range(max_rob_radius):
        for dict_idx, (cert_dict, max_dict_key) in enumerate(zip(cert_dicts, max_cert_dicts_keys)):
            if rob_radius in cert_dict:
                results_array[dict_idx, rob_radius] = cert_dict[rob_radius]
            else:
                results_array[dict_idx, rob_radius] = results_array[dict_idx, rob_radius - 1]

            if rob_radius > max_dict_key:
                results_array[dict_idx, rob_radius] = 0.0

    x_values = np.arange(max_rob_radius)
    x_label = "Robustness Radius"
    y_label = "Percentage of Examples Certified"
    plot_title = f"{mt_name} Phrase-Level Alignment [5 phrases, 5 tokens per phrase]"

    make_multiline_plot(x_values, results_array, line_labels, x_label, y_label, plot_title)


def make_multi_model_alignment_plots(dpa: bool, msc: bool, batch_size: int = 64) -> None:
    models = [None, "gemma2b", "qwen4b"]
    model_display_names = ["OLMo-1B", "Gemma-2B", "Qwen-4B"]

    # Define color families for each model
    # We use sequential colormaps to get different shades
    colormaps = [cm.Blues, cm.Reds, cm.Greens]

    # Plot configuration
    max_rob_radius = 11
    max_k_poison = 11
    max_x_axis = max_rob_radius if not msc else max_k_poison
    x_values = np.arange(max_x_axis)

    plt.figure(figsize=(12, 7))

    # Store custom legend handles
    legend_elements = []

    for model_idx, model_name in enumerate(models):
        # Correctly call get_alignment_dict with model_name
        cert_dicts, line_labels = get_alignment_dict(model_name=model_name, dpa=dpa, msc=msc, batch_size=batch_size)

        num_lines = len(cert_dicts)
        cmap = colormaps[model_idx]

        # Add a proxy artist for the legend for this model family
        legend_elements.append(Line2D([0], [0], color=cmap(0.6), lw=4, label=model_display_names[model_idx]))

        for i, (cert_dict, label) in enumerate(zip(cert_dicts, line_labels)):
            # Create the results array for this specific line
            line_data = np.zeros(max_x_axis)
            max_dict_key = max(list(cert_dict.keys()))

            for x_value in range(max_x_axis):
                if x_value in cert_dict:
                    line_data[x_value] = cert_dict[x_value]
                elif x_value > 0:
                    line_data[x_value] = line_data[x_value - 1]

                if x_value > max_dict_key:
                    line_data[x_value] = 0.0

            # Select color from family:
            # We skip the very light colors (start from 0.3) for visibility
            color_val = 0.3 + (0.6 * (i / max(1, num_lines - 1)))

            # Highlight the "Prompt Certificate" with a thicker or dashed line
            is_prompt = "Prompt" in label
            plt.plot(
                x_values,
                line_data,
                color=cmap(color_val),
                linewidth=2.5 if is_prompt else 1.0,
                linestyle="-" if not is_prompt else "--",
                alpha=0.8,
                # We label only the prompt or specific indices if we wanted them in the legend,
                # but with 33 lines, it's better to use the family legend defined above.
            )

    # Labels and Titles
    x_label = "Poisoning Budget / Robustness Radius"
    suffix = "DPA" if dpa else "ROE"
    suffix += " + MSC" if msc else ""
    plot_title = f"Multi-Model Alignment Certified Robustness ({suffix})"

    if msc:
        x_label = "Poison Budget (k)"
        plot_title = f"Multi-Model Alignment Certified Robustness \n vs Poison Budget (k) ({suffix})"

    plt.xlabel(x_label)
    plt.ylabel("Percentage of Examples Certified")
    plt.title(plot_title)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.ylim(0, 1.05)
    plt.xlim(0, max_x_axis - 1)

    # Add legend showing model families
    # Also add a generic entry for the line styles
    legend_elements.append(Line2D([0], [0], color="gray", lw=1, label="Token Indices"))
    legend_elements.append(Line2D([0], [0], color="black", lw=2.5, ls="--", label="Prompt Certificate"))

    plt.legend(handles=legend_elements, loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(f"multi_model_alignment_{suffix.lower().replace(' ', '_')}.png")


def wc_acc_at_k_plots(dpa: bool, ks_poison: list[int], batch_size: int = 64) -> None:
    cert_dicts, _ = get_alignment_dict(dpa, True, batch_size)
    line_labels = [f"Poison Budget k={k_poison}" for k_poison in ks_poison]

    max_x_axis = 11
    results_array = np.zeros((len(ks_poison), max_x_axis))
    cert_dicts = cert_dicts[:max_x_axis]

    for dict_idx, cert_dict in enumerate(cert_dicts):
        for k_idx, k_poison in enumerate(ks_poison):
            results_array[k_idx, dict_idx] = cert_dict[k_poison]

    x_values = np.arange(max_x_axis)
    y_values = results_array
    x_label = "Token Index"
    y_label = "Percentage of Examples Certified"
    suffix = "DPA + MSC" if dpa else "ROE+ MSC"
    plot_title = f"OLMo-1B Alignment Certified Robustness per Token Index \n at Poison Budget k = {ks_poison} ({suffix})"
    make_multiline_plot(x_values, y_values, line_labels, x_label, y_label, plot_title)


# Combined best certified accuracy
def make_max_certified_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    line_labels: list[str],
    x_label: str,
    y_label: str,
    plot_title: str,
) -> None:
    """
    Plot a single line showing the maximum certified accuracy at each x-value,
    with different colors indicating which method achieves the maximum.
    """
    # Set up the plot style
    sns.set_theme(context="poster", font_scale=2.5)
    plt.figure(figsize=(50, 35))

    # Get the color palette
    colors = sns.color_palette("bright", n_colors=len(line_labels))
    custom_colors = {"ROE + PRDP + MSC (5 PARTITIONS)": "#fc037b"}
    for i, label in enumerate(line_labels):
        if label in custom_colors:
            colors[i] = custom_colors[label]

    # Find the maximum certified accuracy at each x-value and which method achieves it
    max_accuracies = np.max(y_values, axis=0)
    best_method_indices = np.argmax(y_values, axis=0)

    # Create line segments with different colors
    points = np.array([x_values, max_accuracies]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color each segment based on the method that achieves maximum at the start point
    segment_colors = [colors[best_method_indices[i]] for i in range(len(segments))]

    # Create the line collection
    lc = LineCollection(segments, colors=segment_colors, linewidths=35)

    # Add the line collection to the plot
    plt.gca().add_collection(lc)

    # Add markers at each point with appropriate colors
    for i, (x, y, method_idx) in enumerate(zip(x_values, max_accuracies, best_method_indices)):
        plt.scatter(x, y, color=colors[method_idx], s=600, zorder=5)

    # Find which methods are best at least once
    best_methods = set(best_method_indices)

    # Set up the legend only for methods that are best at least once
    legend_elements = [Patch(facecolor=colors[i], label=line_labels[i]) for i in best_methods]
    plt.legend(handles=legend_elements, loc="best", fontsize=65, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    # Set labels and title
    plt.xlabel(x_label, weight="bold", fontsize=75)
    plt.ylabel(y_label, weight="bold", fontsize=75)
    plt.title(plot_title, weight="bold", fontsize=90)

    # Set axis limits
    xrange = x_values.max() - x_values.min()
    yrange = max_accuracies.max() - max_accuracies.min()
    plt.xlim([x_values.min() - 0.05 * xrange, x_values.max() + 0.05 * xrange])
    plt.ylim([max_accuracies.min() - 0.05 * yrange, max_accuracies.max() + 0.05 * yrange])

    # Make sure the plot looks good
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()


def make_cifar_max_plot(finetune: bool = False) -> None:
    """
    Create a plot showing the maximum certified accuracy across all methods
    at each robustness radius, with colors indicating which method is best.
    """
    cert_dicts, line_labels = None, None
    if finetune:
        cert_dicts, line_labels = get_cifar_10_dicts_fine_tuned()
    else:
        cert_dicts, line_labels = get_cifar_10_dicts_bare_bones()

    max_cert_dicts_keys = [max(list(cert_dict.keys())) for cert_dict in cert_dicts]
    max_rob_radius = 650 if finetune else 170
    num_methods = len(cert_dicts)
    results_array = np.zeros((num_methods, max_rob_radius))

    for rob_radius in range(max_rob_radius):
        for dict_idx, (cert_dict, max_dict_key) in enumerate(zip(cert_dicts, max_cert_dicts_keys)):
            if rob_radius in cert_dict:
                results_array[dict_idx, rob_radius] = cert_dict[rob_radius]
            else:
                results_array[dict_idx, rob_radius] = results_array[dict_idx, rob_radius - 1]

            if rob_radius > max_dict_key:
                results_array[dict_idx, rob_radius] = 0.0

    x_values = np.arange(max_rob_radius)
    y_values = results_array
    x_label = "Poisoning Budget / Robustness Radius"
    y_label = "Maximum Certified Accuracy"
    suffix = "Fine-Tuned ResNet" if finetune else "Scratch-trained ResNet"
    plot_title = f"CIFAR-10 Maximum Certified Accuracy {suffix}"

    make_max_certified_plot(x_values, y_values, line_labels, x_label, y_label, plot_title)


def make_agt_bagging_blobs_plot() -> None:
    agt_bagging_file = "framework_blobs.yaml"

    ks_private = [25]  # list(range(0, 101, 10))

    agt_bagging_knames = [f"bag_agt_blobs_partitions_250_k_private_{k}_clip_gamma_0.06" for k in ks_private]
    agt_bagging_dpa_ress = [load_params_or_results_from_file(agt_bagging_file, kname) for kname in agt_bagging_knames]

    cert_dicts = [res["percentage_correct_robust_at_radius"] for res in agt_bagging_dpa_ress]
    line_labels = [f"k_private={k}" for k in ks_private]

    max_cert_dicts_keys = [max(list(cert_dict.keys())) for cert_dict in cert_dicts]
    max_rob_radius = 501
    num_methods = len(cert_dicts)
    results_array = np.zeros((num_methods, max_rob_radius))

    for rob_radius in range(max_rob_radius):
        for dict_idx, (cert_dict, max_dict_key) in enumerate(zip(cert_dicts, max_cert_dicts_keys)):
            if rob_radius in cert_dict:
                results_array[dict_idx, rob_radius] = cert_dict[rob_radius]
            else:
                results_array[dict_idx, rob_radius] = results_array[dict_idx, rob_radius - 1]

            if rob_radius > max_dict_key:
                results_array[dict_idx, rob_radius] = 0.0

    x_values = np.arange(max_rob_radius)
    y_values = results_array
    x_label = "Poisoning Budget / Robustness Radius"
    y_label = "Certified Accuracy"
    plot_title = f"AGT Bagging Certified Accuracy Comparison"
    make_multiline_plot(x_values, y_values, line_labels, x_label, y_label, plot_title, log_scale=False)
    make_multiline_plot(x_values, y_values, line_labels, x_label, y_label, plot_title, log_scale=False)
