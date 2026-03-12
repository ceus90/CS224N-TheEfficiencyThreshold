#!/usr/bin/env python3
import argparse
import json
import os
from statistics import median
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


N_ORDER = [16, 32, 64, 128, 256]

DATASET_ORDER = [
    ("gsm8k", "(a) GSM8K"),
    ("raft", "(b) RAFT"),
    ("financial_phrasebank", "(c) Financial Phrasebank"),
    ("superglue_rte", "(d) Superglue RTE"),
    ("superglue_boolq", "(e) Superglue BoolQ"),
]

LINE_CONFIGS = {
    ("ICL", "4B"): {"label": "ICL 4B", "color": "tab:blue", "linestyle": (0, (3, 3))},
    ("ICL", "8B"): {"label": "ICL 8B", "color": "tab:blue", "linestyle": "-"},
    ("LoRA", "4B"): {"label": "LoRA 4B", "color": "tab:orange", "linestyle": (0, (3, 3))},
    ("LoRA", "8B"): {"label": "LoRA 8B", "color": "tab:orange", "linestyle": "-"},
}


def load_records(path: str) -> List[dict]:
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


def model_size(model_name: str | None) -> str | None:
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


def median_by_n_and_size(records: List[dict], dataset: str) -> Dict[str, Dict[int, Tuple[float, float]]]:
    # size -> n -> model -> list[(accuracy, latency)]
    grouped: Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]] = {}
    for r in records:
        if r.get("dataset") != dataset:
            continue
        size = model_size(r.get("model"))
        if size is None:
            continue
        n = r.get("n")
        metrics = r.get("metrics", {})
        acc = metrics.get("accuracy")
        lat = metrics.get("latency")
        if not isinstance(n, int) or acc is None or lat is None:
            continue
        model = r.get("model")
        if not isinstance(model, str):
            continue
        grouped.setdefault(size, {}).setdefault(n, {}).setdefault(model, []).append((acc, lat))

    # size -> n -> (median_accuracy, median_latency)
    result: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for size, n_map in grouped.items():
        for n, model_map in n_map.items():
            per_model_means = []
            for runs in model_map.values():
                if not runs:
                    continue
                acc_mean = sum(a for a, _ in runs) / len(runs)
                lat_mean = sum(l for _, l in runs) / len(runs)
                per_model_means.append((acc_mean, lat_mean))
            if not per_model_means:
                continue
            result.setdefault(size, {})[n] = (
                median(a for a, _ in per_model_means),
                median(l for _, l in per_model_means),
            )

    return result


def plot_pareto_panel(
    ax,
    label: str,
    icl_stats: Dict[str, Dict[int, Tuple[float, float]]],
    lora_stats: Dict[str, Dict[int, Tuple[float, float]]],
):
    for method, stats in (("ICL", icl_stats), ("LoRA", lora_stats)):
        for size in ("4B", "8B"):
            points = [(n, *stats[size][n]) for n in N_ORDER if size in stats and n in stats[size]]
            if not points:
                continue
            latencies = [lat for _, _, lat in points]
            accuracies = [acc for _, acc, _ in points]
            cfg = LINE_CONFIGS[(method, size)]
            ax.plot(
                latencies,
                accuracies,
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=2.0,
                label=cfg["label"],
            )
            for n, acc, lat in points:
                ax.annotate(str(n), (lat, acc), textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title(label)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=LINE_CONFIGS[("ICL", "4B")]["color"],
            linestyle=LINE_CONFIGS[("ICL", "4B")]["linestyle"],
            linewidth=2.2,
            label=LINE_CONFIGS[("ICL", "4B")]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=LINE_CONFIGS[("ICL", "8B")]["color"],
            linestyle=LINE_CONFIGS[("ICL", "8B")]["linestyle"],
            linewidth=2.2,
            label=LINE_CONFIGS[("ICL", "8B")]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=LINE_CONFIGS[("LoRA", "4B")]["color"],
            linestyle=LINE_CONFIGS[("LoRA", "4B")]["linestyle"],
            linewidth=2.2,
            label=LINE_CONFIGS[("LoRA", "4B")]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=LINE_CONFIGS[("LoRA", "8B")]["color"],
            linestyle=LINE_CONFIGS[("LoRA", "8B")]["linestyle"],
            linewidth=2.2,
            label=LINE_CONFIGS[("LoRA", "8B")]["label"],
        ),
    ]
    ax.legend(handles=legend_handles, fontsize="x-small", handlelength=2.8, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto plot (accuracy vs latency), median by model size (4B/8B)."
    )
    parser.add_argument(
        "--icl",
        default="final_results/icl_results.jsonl",
        help="Path to ICL results JSONL.",
    )
    parser.add_argument(
        "--lora",
        default="final_results/lora_results.jsonl",
        help="Path to LoRA results JSONL.",
    )
    parser.add_argument(
        "--out",
        default="final_results/pdfs",
        help="Output directory for PDF.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    icl_records = load_records(args.icl)
    lora_records = load_records(args.lora)
    out_path = os.path.join(args.out, "pareto_all_datasets_median.pdf")

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        flat_axes = axes.flatten()

        any_plotted = False
        for i, (dataset, label) in enumerate(DATASET_ORDER):
            ax = flat_axes[i]
            icl_stats = median_by_n_and_size(icl_records, dataset)
            lora_stats = median_by_n_and_size(lora_records, dataset)
            if not icl_stats and not lora_stats:
                ax.axis("off")
                continue
            plot_pareto_panel(ax, label, icl_stats, lora_stats)
            any_plotted = True

        for ax in flat_axes[len(DATASET_ORDER):]:
            ax.axis("off")

        fig.suptitle("Accuracy-Latency Tradeoff (Median by Model Size)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not any_plotted:
            raise ValueError("No matching records found for the requested datasets.")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
