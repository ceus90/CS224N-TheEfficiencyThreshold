#!/usr/bin/env python3
import json
import os
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(REPO_ROOT, "final_results")
ICL_PATH = os.path.join(BASE_DIR, "icl_results.jsonl")
LORA_PATH = os.path.join(BASE_DIR, "lora_results.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "pdfs")
OUT_PATH = os.path.join(OUT_DIR, "results_summary_two_row_median.pdf")
N_ORDER = [16, 32, 64, 128, 256]

DATASET_ORDER = [
    ("gsm8k", "(a) GSM8K"),
    ("raft", "(b) RAFT"),
    ("financial_phrasebank", "(c) Financial Phrasebank"),
    ("superglue_rte", "(d) Superglue RTE"),
    ("superglue_boolq", "(e) Superglue BoolQ"),
]

SIZE_LABELS = ["4B", "8B"]
LINE_CONFIGS = {
    ("4B", "ICL"): {"label": "ICL 4B", "linestyle": (0, (3, 3)), "color": "tab:blue"},
    ("8B", "ICL"): {"label": "ICL 8B", "linestyle": "-", "color": "tab:blue"},
    ("4B", "LoRA"): {"label": "LoRA 4B", "linestyle": (0, (3, 3)), "color": "tab:orange"},
    ("8B", "LoRA"): {"label": "LoRA 8B", "linestyle": "-", "color": "tab:orange"},
}


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def model_size(model_name):
    if not model_name:
        return None
    name = model_name.lower()
    if "9b" in name:
        return None
    if "4b" in name:
        return "4B"
    if "8b" in name:
        return "8B"
    return None


def aggregate_median_by_size(records, metric):
    # dataset -> n -> size -> model -> [values]
    values_by_model = {}

    for r in records:
        dataset = r.get("dataset")
        n = r.get("n")
        model = r.get("model")
        value = r.get("metrics", {}).get(metric)

        size = model_size(model)
        if dataset is None or n is None or value is None or size is None:
            continue

        values_by_model.setdefault(dataset, {}).setdefault(n, {}).setdefault(size, {}).setdefault(model, []).append(value)

    # dataset -> n -> size -> median(across per-model means)
    output = {}
    for dataset, n_map in values_by_model.items():
        for n, size_map in n_map.items():
            for size, model_map in size_map.items():
                per_model_means = [sum(vals) / len(vals) for vals in model_map.values() if vals]
                if not per_model_means:
                    continue
                output.setdefault(dataset, {}).setdefault(n, {})[size] = median(per_model_means)

    return output


def plot_grid(pdf, title, y_label, icl_median, lora_median):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
    flat_axes = axes.flatten()
    legend_handles = [
        Line2D([0], [0], color=LINE_CONFIGS[("4B", "ICL")]["color"], linestyle=LINE_CONFIGS[("4B", "ICL")]["linestyle"], linewidth=2.2, label=LINE_CONFIGS[("4B", "ICL")]["label"]),
        Line2D([0], [0], color=LINE_CONFIGS[("8B", "ICL")]["color"], linestyle=LINE_CONFIGS[("8B", "ICL")]["linestyle"], linewidth=2.2, label=LINE_CONFIGS[("8B", "ICL")]["label"]),
        Line2D([0], [0], color=LINE_CONFIGS[("4B", "LoRA")]["color"], linestyle=LINE_CONFIGS[("4B", "LoRA")]["linestyle"], linewidth=2.2, label=LINE_CONFIGS[("4B", "LoRA")]["label"]),
        Line2D([0], [0], color=LINE_CONFIGS[("8B", "LoRA")]["color"], linestyle=LINE_CONFIGS[("8B", "LoRA")]["linestyle"], linewidth=2.2, label=LINE_CONFIGS[("8B", "LoRA")]["label"]),
    ]

    for i, (dataset_key, label) in enumerate(DATASET_ORDER):
        ax = flat_axes[i]

        for size in SIZE_LABELS:
            icl_map = icl_median.get(dataset_key, {})
            lora_map = lora_median.get(dataset_key, {})

            icl_ns = [n for n in N_ORDER if n in icl_map and size in icl_map[n]]
            lora_ns = [n for n in N_ORDER if n in lora_map and size in lora_map[n]]

            if icl_ns:
                cfg = LINE_CONFIGS[(size, "ICL")]
                ax.plot(
                    icl_ns,
                    [icl_map[n][size] for n in icl_ns],
                    linestyle=cfg["linestyle"],
                    color=cfg["color"],
                    label=cfg["label"],
                    linewidth=2.0,
                )

            if lora_ns:
                cfg = LINE_CONFIGS[(size, "LoRA")]
                ax.plot(
                    lora_ns,
                    [lora_map[n][size] for n in lora_ns],
                    linestyle=cfg["linestyle"],
                    color=cfg["color"],
                    label=cfg["label"],
                    linewidth=2.0,
                )

        ax.set_title(label)
        ax.set_xlabel("N")
        ax.set_xticks(N_ORDER)
        if i == 0 or i == 3:
            ax.set_ylabel(y_label)
        ax.legend(
            handles=legend_handles,
            fontsize="x-small",
            loc="best",
            handlelength=2.8,
        )

    for ax in flat_axes[len(DATASET_ORDER):]:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    icl_records = load_records(ICL_PATH)
    lora_records = load_records(LORA_PATH)

    if not icl_records:
        raise ValueError("No ICL records found")
    if not lora_records:
        raise ValueError("No LoRA records found")

    icl_acc = aggregate_median_by_size(icl_records, "accuracy")
    lora_acc = aggregate_median_by_size(lora_records, "accuracy")
    icl_lat = aggregate_median_by_size(icl_records, "latency")
    lora_lat = aggregate_median_by_size(lora_records, "latency")

    with PdfPages(OUT_PATH) as pdf:
        plot_grid(
            pdf,
            "Accuracy vs N (Median by Model Size: ICL vs LoRA)",
            "Accuracy",
            icl_acc,
            lora_acc,
        )
        plot_grid(
            pdf,
            "Latency vs N (Median by Model Size: ICL vs LoRA)",
            "Latency (ms)",
            icl_lat,
            lora_lat,
        )

    print(f"Saved: {OUT_PATH}")
