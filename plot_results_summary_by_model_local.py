#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = "/Users/abilopez/Documents/StanfordClasses/Win26/224N/final_results"
ICL_PATH = os.path.join(BASE_DIR, "ICL_results.jsonl")
LORA_PATH = os.path.join(BASE_DIR, "LoRA_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "results_summary_by_model.pdf")
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
    # dataset -> model -> n -> list of metric
    agg = {}
    for r in records:
        d = r.get("dataset")
        m = r.get("model")
        n = r.get("n")
        val = r.get("metrics", {}).get(metric)
        if d is None or m is None or n is None or val is None:
            continue
        agg.setdefault(d, {}).setdefault(m, {}).setdefault(n, []).append(val)
    # dataset -> model -> n -> mean
    mean = {}
    for d, m_map in agg.items():
        mean[d] = {}
        for m, n_map in m_map.items():
            mean[d][m] = {n: sum(vals) / len(vals) for n, vals in n_map.items()}
    return mean


def plot_small_multiples(pdf, title, y_label, icl_mean, lora_mean, datasets, metric, is_delta=False):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    axes = axes if len(datasets) >= 5 else axes[:len(datasets)]

    for i, d in enumerate(datasets[:5]):
        ax = axes[i]
        icl_models = icl_mean.get(d, {})
        lora_models = lora_mean.get(d, {})
        models = sorted(set(icl_models.keys()).intersection(lora_models.keys()))
        for m in models:
            icl_map = icl_models.get(m, {})
            lora_map = lora_models.get(m, {})
            ns = [n for n in N_ORDER if n in icl_map and n in lora_map]
            if not ns:
                continue
            if is_delta:
                ys = [(lora_map[n] - icl_map[n]) for n in ns]
                ax.plot(ns, ys, marker='o', label=m)
            else:
                icl_y = [icl_map[n] for n in ns]
                lora_y = [lora_map[n] for n in ns]
                ax.plot(ns, icl_y, marker='o', label=f"ICL-{m}")
                ax.plot(ns, lora_y, marker='^', label=f"LoRA-{m}")
        ax.set_title(d)
        ax.set_xlabel("N")
        if i == 0:
            ax.set_ylabel(y_label)
        ax.legend(fontsize='xx-small', ncol=1)

    fig.suptitle(title)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_mean_metric(pdf, title, y_label, icl_mean, lora_mean, datasets, metric):
    # Mean across datasets for each model separately
    models = sorted(set(m for d in datasets for m in icl_mean.get(d, {}).keys()).intersection(
        set(m for d in datasets for m in lora_mean.get(d, {}).keys())
    ))
    all_ns = N_ORDER
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        icl_vals = []
        lora_vals = []
        for n in all_ns:
            icl_n = []
            lora_n = []
            for d in datasets:
                if n in icl_mean.get(d, {}).get(m, {}):
                    icl_n.append(icl_mean[d][m][n])
                if n in lora_mean.get(d, {}).get(m, {}):
                    lora_n.append(lora_mean[d][m][n])
            icl_vals.append(sum(icl_n) / len(icl_n) if icl_n else 0.0)
            lora_vals.append(sum(lora_n) / len(lora_n) if lora_n else 0.0)
        ax.plot(all_ns, icl_vals, marker='o', label=f"ICL-{m}")
        ax.plot(all_ns, lora_vals, marker='^', label=f"LoRA-{m}")

    ax.set_xlabel("N")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize='x-small', ncol=2)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_mean_delta(pdf, title, y_label, icl_mean, lora_mean, datasets, metric):
    models = sorted(set(m for d in datasets for m in icl_mean.get(d, {}).keys()).intersection(
        set(m for d in datasets for m in lora_mean.get(d, {}).keys())
    ))
    all_ns = N_ORDER
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        deltas = []
        for n in all_ns:
            ds_deltas = []
            for d in datasets:
                if n in icl_mean.get(d, {}).get(m, {}) and n in lora_mean.get(d, {}).get(m, {}):
                    ds_deltas.append(lora_mean[d][m][n] - icl_mean[d][m][n])
            deltas.append(sum(ds_deltas) / len(ds_deltas) if ds_deltas else 0.0)
        ax.plot(all_ns, deltas, marker='o', label=f"Δ {m}")

    ax.axhline(0, color='gray', linewidth=1, linestyle='--')
    ax.set_xlabel("N")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize='x-small', ncol=2)
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
        plot_small_multiples(pdf, "Accuracy vs N (All Models)", "Accuracy", icl_acc, lora_acc, datasets, "accuracy")
        plot_small_multiples(pdf, "Latency vs N (All Models)", "Latency (ms)", icl_lat, lora_lat, datasets, "latency")
        plot_small_multiples(pdf, "Delta Accuracy (LoRA - ICL)", "Δ Accuracy", icl_acc, lora_acc, datasets, "accuracy", is_delta=True)
        plot_small_multiples(pdf, "Delta Latency (LoRA - ICL)", "Δ Latency (ms)", icl_lat, lora_lat, datasets, "latency", is_delta=True)
        plot_mean_metric(pdf, "Mean Accuracy Across Datasets (All Models)", "Accuracy", icl_acc, lora_acc, datasets, "accuracy")
        plot_mean_metric(pdf, "Mean Latency Across Datasets (All Models)", "Latency (ms)", icl_lat, lora_lat, datasets, "latency")
        plot_mean_delta(pdf, "Mean Δ Accuracy Across Datasets (All Models)", "Mean Δ Accuracy", icl_acc, lora_acc, datasets, "accuracy")
        plot_mean_delta(pdf, "Mean Δ Latency Across Datasets (All Models)", "Mean Δ Latency (ms)", icl_lat, lora_lat, datasets, "latency")

    print(f"Saved: {OUT_PATH}")
