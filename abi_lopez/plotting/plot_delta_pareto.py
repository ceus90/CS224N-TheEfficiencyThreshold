#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


N_ORDER = [16, 32, 64, 128, 256]


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


def plot_delta_pareto(pdf, title: str, icl_means: Dict[int, Tuple[float, float]], lora_means: Dict[int, Tuple[float, float]]):
    fig, ax = plt.subplots(figsize=(8, 6))

    points = []
    for n in N_ORDER:
        if n in icl_means and n in lora_means:
            icl_acc, icl_lat = icl_means[n]
            lora_acc, lora_lat = lora_means[n]
            delta_acc = lora_acc - icl_acc
            delta_lat = lora_lat - icl_lat
            points.append((n, delta_acc, delta_lat))

    if points:
        xs = [p[2] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, color="black")
        for n, delta_acc, delta_lat in points:
            ax.annotate(str(n), (delta_lat, delta_acc), textcoords="offset points", xytext=(4, 4))

    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.axvline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Δ Latency (ms)")
    ax.set_ylabel("Δ Accuracy")
    ax.set_title(title)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delta Pareto plot (LoRA - ICL) for one or more datasets."
    )
    parser.add_argument(
        "--datasets",
        default="superglue_rte,superglue_boolq,financial_phrasebank,raft,gsm8k",
        help="Comma-separated dataset names.",
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
    out_path = os.path.join(args.out, f"delta_pareto_all_datasets{model_tag}.pdf")

    with PdfPages(out_path) as pdf:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        if not datasets:
            raise ValueError("No datasets provided.")
        any_plotted = False
        for dataset in datasets:
            icl_means = mean_by_n(icl_records, dataset, args.model)
            lora_means = mean_by_n(lora_records, dataset, args.model)
            if not icl_means or not lora_means:
                continue
            title = f"Δ-Pareto plot — {dataset}" + (f" | {args.model}" if args.model else "")
            plot_delta_pareto(pdf, title, icl_means, lora_means)
            any_plotted = True
        if not any_plotted:
            raise ValueError("No matching records found for the requested datasets/model.")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
