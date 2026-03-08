#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = "/Users/abilopez/Documents/StanfordClasses/Win26/224N/final_results"
ICL_PATH = os.path.join(BASE_DIR, "ICL_results.jsonl")
LORA_PATH = os.path.join(BASE_DIR, "LoRA_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "results_summary.pdf")
N_ORDER = [16, 32, 64, 128, 256]


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


def aggregate(records, metric):
    # dataset -> n -> list of metric
    agg = {}
    for r in records:
        d = r.get("dataset")
        n = r.get("n")
        val = r.get("metrics", {}).get(metric)
        if d is None or n is None or val is None:
            continue
        agg.setdefault(d, {}).setdefault(n, []).append(val)
    # dataset -> n -> mean
    mean = {}
    for d, n_map in agg.items():
        mean[d] = {n: sum(vals) / len(vals) for n, vals in n_map.items()}
    return mean


def plot_small_multiples(pdf, title, y_label, icl_mean, lora_mean, datasets, metric, is_delta=False):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    axes = axes if len(datasets) >= 5 else axes[:len(datasets)]
    for i, d in enumerate(datasets[:5]):
        ax = axes[i]
        icl_map = icl_mean.get(d, {})
        lora_map = lora_mean.get(d, {})
        ns = [n for n in N_ORDER if n in icl_map and n in lora_map]
        if is_delta:
            ys = [(lora_map[n] - icl_map[n]) for n in ns]
            ax.plot(ns, ys, marker='o', color='black')
            ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        else:
            icl_y = [icl_map[n] for n in ns]
            lora_y = [lora_map[n] for n in ns]
            ax.plot(ns, icl_y, marker='o', label="ICL")
            ax.plot(ns, lora_y, marker='^', label="LoRA")
            ax.legend(fontsize='x-small')
        ax.set_title(d)
        ax.set_xlabel("N")
        if i == 0:
            ax.set_ylabel(y_label)
    fig.suptitle(title)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_mean_delta(pdf, title, y_label, icl_mean, lora_mean, datasets):
    all_ns = [n for n in N_ORDER if any(n in icl_mean[d] and n in lora_mean[d] for d in datasets)]
    mean_delta = []
    for n in all_ns:
        deltas = []
        for d in datasets:
            if n in icl_mean[d] and n in lora_mean[d]:
                deltas.append(lora_mean[d][n] - icl_mean[d][n])
        mean_delta.append(sum(deltas) / len(deltas) if deltas else 0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(all_ns, mean_delta, marker='o', color='black')
    ax.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax.set_xlabel("N")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_mean_metric(pdf, title, y_label, icl_mean, lora_mean, datasets):
    all_ns = [n for n in N_ORDER if any(n in icl_mean[d] and n in lora_mean[d] for d in datasets)]
    icl_avg = []
    lora_avg = []
    for n in all_ns:
        icl_vals = []
        lora_vals = []
        for d in datasets:
            if n in icl_mean[d]:
                icl_vals.append(icl_mean[d][n])
            if n in lora_mean[d]:
                lora_vals.append(lora_mean[d][n])
        icl_avg.append(sum(icl_vals) / len(icl_vals) if icl_vals else 0.0)
        lora_avg.append(sum(lora_vals) / len(lora_vals) if lora_vals else 0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(all_ns, icl_avg, marker='o', label="ICL")
    ax.plot(all_ns, lora_avg, marker='^', label="LoRA")
    ax.set_xlabel("N")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize='small')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    icl_records = load_records(ICL_PATH)
    lora_records = load_records(LORA_PATH)

    if not icl_records:
        raise ValueError("No ICL records found")
    if not lora_records:
        raise ValueError("No LoRA records found")

    icl_acc = aggregate(icl_records, "accuracy")
    lora_acc = aggregate(lora_records, "accuracy")
    icl_lat = aggregate(icl_records, "latency")
    lora_lat = aggregate(lora_records, "latency")

    datasets = sorted(set(icl_acc.keys()).intersection(lora_acc.keys()))
    if not datasets:
        raise ValueError("No overlapping datasets between ICL and LoRA")

    with PdfPages(OUT_PATH) as pdf:
        # Accuracy vs N
        plot_small_multiples(pdf, "Accuracy vs N (ICL vs LoRA)", "Accuracy", icl_acc, lora_acc, datasets, "accuracy")
        # Latency vs N
        plot_small_multiples(pdf, "Latency vs N (ICL vs LoRA)", "Latency (ms)", icl_lat, lora_lat, datasets, "latency")
        # Delta accuracy
        plot_small_multiples(pdf, "Delta Accuracy (LoRA - ICL)", "Δ Accuracy", icl_acc, lora_acc, datasets, "accuracy", is_delta=True)
        # Delta latency
        plot_small_multiples(pdf, "Delta Latency (LoRA - ICL)", "Δ Latency (ms)", icl_lat, lora_lat, datasets, "latency", is_delta=True)
        # Mean accuracy
        plot_mean_metric(pdf, "Mean Accuracy Across Datasets", "Accuracy", icl_acc, lora_acc, datasets)
        # Mean latency
        plot_mean_metric(pdf, "Mean Latency Across Datasets", "Latency (ms)", icl_lat, lora_lat, datasets)

    print(f"Saved: {OUT_PATH}")
