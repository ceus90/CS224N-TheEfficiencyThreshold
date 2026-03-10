#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(REPO_ROOT, "final_results")
ICL_PATH = os.path.join(BASE_DIR, "icl_results.jsonl")
LORA_PATH = os.path.join(BASE_DIR, "lora_results.jsonl")
OUT_DIR = os.path.join(BASE_DIR, "pdfs")
OUT_PATH = os.path.join(OUT_DIR, "results_summary_two_row.pdf")
N_ORDER = [16, 32, 64, 128, 256]

DATASET_ORDER = [
    ("gsm8k", "(a) GSM8K"),
    ("raft", "(b) RAFT"),
    ("financial_phrasebank", "(c) Financial Phrasebank"),
    ("superglue_rte", "(d) Superglue RTE"),
    ("superglue_boolq", "(e) Superglue BoolQ"),
]


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


def plot_grid(pdf, title, y_label, icl_mean, lora_mean):
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
    flat_axes = axes.flatten()

    for i, (dataset_key, label) in enumerate(DATASET_ORDER):
        ax = flat_axes[i]
        icl_map = icl_mean.get(dataset_key, {})
        lora_map = lora_mean.get(dataset_key, {})
        ns = [n for n in N_ORDER if n in icl_map and n in lora_map]
        icl_y = [icl_map[n] for n in ns]
        lora_y = [lora_map[n] for n in ns]
        ax.plot(ns, icl_y, marker="o", label="ICL")
        ax.plot(ns, lora_y, marker="^", label="LoRA")
        ax.set_title(label)
        ax.set_xlabel("N")
        ax.set_xticks(N_ORDER)
        if i == 0 or i == 3:
            ax.set_ylabel(y_label)
        ax.legend(fontsize="x-small")

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

    icl_acc = aggregate(icl_records, "accuracy")
    lora_acc = aggregate(lora_records, "accuracy")
    icl_lat = aggregate(icl_records, "latency")
    lora_lat = aggregate(lora_records, "latency")

    with PdfPages(OUT_PATH) as pdf:
        plot_grid(pdf, "Accuracy vs N (ICL vs LoRA)", "Accuracy", icl_acc, lora_acc)
        plot_grid(pdf, "Latency vs N (ICL vs LoRA)", "Latency (ms)", icl_lat, lora_lat)

    print(f"Saved: {OUT_PATH}")
