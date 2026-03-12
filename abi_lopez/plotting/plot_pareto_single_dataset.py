#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


N_ORDER = [16, 32, 64, 128, 256]

DATASET_ORDER = [
    ("gsm8k", "(a) GSM8K"),
    ("raft", "(b) RAFT"),
    ("financial_phrasebank", "(c) Financial Phrasebank"),
    ("superglue_rte", "(d) Superglue RTE"),
    ("superglue_boolq", "(e) Superglue BoolQ"),
]


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


def mean_by_n(records: List[dict], dataset: str, model: str | None) -> Dict[int, Tuple[float, float]]:
    # n -> list[(accuracy, latency)]
    buckets: Dict[int, List[Tuple[float, float]]] = {}
    for r in records:
        if r.get("dataset") != dataset:
            continue
        if model is not None and r.get("model") != model:
            continue
        n = r.get("n")
        metrics = r.get("metrics", {})
        acc = metrics.get("accuracy")
        lat = metrics.get("latency")
        if not isinstance(n, int) or acc is None or lat is None:
            continue
        buckets.setdefault(n, []).append((acc, lat))

    means: Dict[int, Tuple[float, float]] = {}
    for n, pairs in buckets.items():
        if not pairs:
            continue
        acc_mean = sum(a for a, _ in pairs) / len(pairs)
        lat_mean = sum(l for _, l in pairs) / len(pairs)
        means[n] = (acc_mean, lat_mean)
    return means


def plot_pareto_panel(ax, label: str, icl_means: Dict[int, Tuple[float, float]], lora_means: Dict[int, Tuple[float, float]]):
    icl_points = [(n, *icl_means[n]) for n in N_ORDER if n in icl_means]
    lora_points = [(n, *lora_means[n]) for n in N_ORDER if n in lora_means]

    if icl_points:
        icl_lat = [lat for _, _, lat in icl_points]
        icl_acc = [acc for _, acc, _ in icl_points]
        ax.plot(icl_lat, icl_acc, marker="o", color="blue", label="ICL")
        for n, acc, lat in icl_points:
            ax.annotate(str(n), (lat, acc), textcoords="offset points", xytext=(4, 4))

    if lora_points:
        lora_lat = [lat for _, _, lat in lora_points]
        lora_acc = [acc for _, acc, _ in lora_points]
        ax.plot(lora_lat, lora_acc, marker="^", color="orange", label="LoRA")
        for n, acc, lat in lora_points:
            ax.annotate(str(n), (lat, acc), textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title(label)
    ax.legend(fontsize="x-small")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto plot (accuracy vs latency) for a single dataset."
    )
    parser.add_argument("--model", default=None, help="Optional model name filter.")
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

    model_tag = f"_{args.model}" if args.model else ""
    out_path = os.path.join(args.out, f"pareto_all_datasets{model_tag}.pdf")

    with PdfPages(out_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        flat_axes = axes.flatten()

        any_plotted = False
        for i, (dataset, label) in enumerate(DATASET_ORDER):
            ax = flat_axes[i]
            icl_means = mean_by_n(icl_records, dataset, args.model)
            lora_means = mean_by_n(lora_records, dataset, args.model)
            if not icl_means and not lora_means:
                ax.axis("off")
                continue
            plot_pareto_panel(ax, label, icl_means, lora_means)
            any_plotted = True

        for ax in flat_axes[len(DATASET_ORDER):]:
            ax.axis("off")

        fig.suptitle("Accuracy-Latency Tradeoff" + (f" | {args.model}" if args.model else ""))
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if not any_plotted:
            raise ValueError("No matching records found for the requested datasets/model.")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
