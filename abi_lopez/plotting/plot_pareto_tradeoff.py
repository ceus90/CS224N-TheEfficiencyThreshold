#!/usr/bin/env python3
import argparse
import json
import os
import matplotlib.pyplot as plt


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


def main():
    parser = argparse.ArgumentParser(description="Pareto tradeoff plots for ICL vs LoRA")
    parser.add_argument("--icl", required=True, help="Path to ICL checkpoint JSONL")
    parser.add_argument("--lora", required=True, help="Path to LoRA checkpoint JSONL")
    parser.add_argument("--outdir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    icl_records = load_records(args.icl)
    lora_records = load_records(args.lora)

    if not icl_records:
        raise ValueError("No ICL records found")
    if not lora_records:
        raise ValueError("No LoRA records found")

    os.makedirs(args.outdir, exist_ok=True)

    dataset_colors = {
        "superglue_rte": "blue",
        "superglue_boolq": "orange",
        "financial_phrasebank": "green",
        "raft": "red",
        "gsm8k": "purple"
    }

    def plot(metric_y, fname):
        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot ICL points
        for r in icl_records:
            d = r.get("dataset")
            m = r.get("metrics", {})
            if d is None or metric_y not in m or "latency" not in m:
                continue
            ax.scatter(m["latency"], m[metric_y], color=dataset_colors.get(d, "black"), marker="o", alpha=0.7)

        # Plot LoRA points
        for r in lora_records:
            d = r.get("dataset")
            m = r.get("metrics", {})
            if d is None or metric_y not in m or "latency" not in m:
                continue
            ax.scatter(m["latency"], m[metric_y], color=dataset_colors.get(d, "black"), marker="^", alpha=0.7)

        # Arrows from ICL to LoRA for matching (dataset, model, n)
        def index_by_key(records):
            idx = {}
            for r in records:
                key = (r.get("dataset"), r.get("model"), r.get("n"))
                if None in key:
                    continue
                idx[key] = r
            return idx

        icl_idx = index_by_key(icl_records)
        lora_idx = index_by_key(lora_records)
        common = set(icl_idx.keys()).intersection(lora_idx.keys())
        for key in common:
            icl_r = icl_idx[key]
            lora_r = lora_idx[key]
            d = key[0]
            icl_m = icl_r.get("metrics", {})
            lora_m = lora_r.get("metrics", {})
            if metric_y not in icl_m or metric_y not in lora_m:
                continue
            if "latency" not in icl_m or "latency" not in lora_m:
                continue
            ax.annotate(
                "",
                xy=(lora_m["latency"], lora_m[metric_y]),
                xytext=(icl_m["latency"], icl_m[metric_y]),
                arrowprops=dict(arrowstyle="->", color=dataset_colors.get(d, "black"), alpha=0.5, lw=1)
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Accuracy" if metric_y == "accuracy" else "VRAM (%)")
        ax.set_title("Accuracy vs Latency" if metric_y == "accuracy" else "Accuracy vs VRAM")

        # Custom legends
        from matplotlib.lines import Line2D
        dataset_handles = [
            Line2D([0], [0], marker='o', color='w', label='RTE', markerfacecolor=dataset_colors["superglue_rte"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='BoolQ', markerfacecolor=dataset_colors["superglue_boolq"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Financial', markerfacecolor=dataset_colors["financial_phrasebank"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='RAFT', markerfacecolor=dataset_colors["raft"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='GSM8K', markerfacecolor=dataset_colors["gsm8k"], markersize=8),
        ]
        method_handles = [
            Line2D([0], [0], marker='o', color='k', label='ICL', linestyle='None'),
            Line2D([0], [0], marker='^', color='k', label='LoRA', linestyle='None'),
        ]
        legend1 = ax.legend(handles=dataset_handles, title="Dataset", loc="upper right")
        ax.add_artist(legend1)
        ax.legend(handles=method_handles, title="Method", loc="lower right")

        out_path = os.path.join(args.outdir, fname)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}")

    plot("accuracy", "pareto_accuracy_vs_latency.pdf")
    # For VRAM, use vram_pct as y and latency on x? Request is accuracy vs VRAM
    # So use x=vram_pct, y=accuracy. We will plot separately.
    def plot_accuracy_vs_vram():
        fig, ax = plt.subplots(figsize=(10, 7))

        for r in icl_records:
            d = r.get("dataset")
            m = r.get("metrics", {})
            if d is None or "accuracy" not in m or "vram_pct" not in m:
                continue
            ax.scatter(m["vram_pct"], m["accuracy"], color=dataset_colors.get(d, "black"), marker="o", alpha=0.7)

        for r in lora_records:
            d = r.get("dataset")
            m = r.get("metrics", {})
            if d is None or "accuracy" not in m or "vram_pct" not in m:
                continue
            ax.scatter(m["vram_pct"], m["accuracy"], color=dataset_colors.get(d, "black"), marker="^", alpha=0.7)

        def index_by_key(records):
            idx = {}
            for r in records:
                key = (r.get("dataset"), r.get("model"), r.get("n"))
                if None in key:
                    continue
                idx[key] = r
            return idx

        icl_idx = index_by_key(icl_records)
        lora_idx = index_by_key(lora_records)
        common = set(icl_idx.keys()).intersection(lora_idx.keys())
        for key in common:
            icl_r = icl_idx[key]
            lora_r = lora_idx[key]
            d = key[0]
            icl_m = icl_r.get("metrics", {})
            lora_m = lora_r.get("metrics", {})
            if "accuracy" not in icl_m or "accuracy" not in lora_m:
                continue
            if "vram_pct" not in icl_m or "vram_pct" not in lora_m:
                continue
            ax.annotate(
                "",
                xy=(lora_m["vram_pct"], lora_m["accuracy"]),
                xytext=(icl_m["vram_pct"], icl_m["accuracy"]),
                arrowprops=dict(arrowstyle="->", color=dataset_colors.get(d, "black"), alpha=0.5, lw=1)
            )

        ax.set_xlabel("VRAM (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs VRAM")

        from matplotlib.lines import Line2D
        dataset_handles = [
            Line2D([0], [0], marker='o', color='w', label='RTE', markerfacecolor=dataset_colors["superglue_rte"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='BoolQ', markerfacecolor=dataset_colors["superglue_boolq"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Financial', markerfacecolor=dataset_colors["financial_phrasebank"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='RAFT', markerfacecolor=dataset_colors["raft"], markersize=8),
            Line2D([0], [0], marker='o', color='w', label='GSM8K', markerfacecolor=dataset_colors["gsm8k"], markersize=8),
        ]
        method_handles = [
            Line2D([0], [0], marker='o', color='k', label='ICL', linestyle='None'),
            Line2D([0], [0], marker='^', color='k', label='LoRA', linestyle='None'),
        ]
        legend1 = ax.legend(handles=dataset_handles, title="Dataset", loc="upper right")
        ax.add_artist(legend1)
        ax.legend(handles=method_handles, title="Method", loc="lower right")

        out_path = os.path.join(args.outdir, "pareto_accuracy_vs_vram.pdf")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}")

    plot_accuracy_vs_vram()


if __name__ == "__main__":
    main()
