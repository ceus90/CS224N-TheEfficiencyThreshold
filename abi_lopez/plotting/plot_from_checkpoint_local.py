#!/usr/bin/env python3
import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def main():
    parser = argparse.ArgumentParser(description="Plot ICL metrics from checkpoint JSONL")
    parser.add_argument("--checkpoint", required=True, help="Path to ICL_Efficiency_Checkpoint.jsonl")
    parser.add_argument("--output", required=True, help="Output PDF path")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    records = []
    with open(args.checkpoint, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError("No records found in checkpoint")

    data = {}
    for r in records:
        d = r.get("dataset")
        m = r.get("model")
        n = r.get("n")
        metrics = r.get("metrics", {})
        if d is None or m is None or n is None:
            continue
        data.setdefault(d, {}).setdefault(m, {})
        for k, v in metrics.items():
            data[d][m].setdefault(k, {})[n] = v

    metrics = ["accuracy", "latency", "vram_pct", "throughput", "train_time", "tpot"]
    colors = {
        "Llama-3-8B": "black",
        "Llama-3-8B-Instruct": "blue",
        "Qwen3-4B": "orange",
        "Qwen3-4B-Instruct-2507": "brown",
        "Qwen3-8B": "red",
        "Qwen3-8B-Thinking": "purple",
        "Qwen2.5-7B-Instruct": "magenta",
        "Gemma-2-9B": "green",
        "Gemma-2-9B-IT": "cyan"
    }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with PdfPages(args.output) as pdf:
        for d_name, model_map in data.items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"ICL Efficiency Benchmarks (All Models): {d_name}")
            for i, mtr in enumerate(metrics):
                ax = axes.flatten()[i]
                for m_name, mtr_map in model_map.items():
                    if mtr not in mtr_map:
                        continue
                    n_to_val = mtr_map[mtr]
                    xs = sorted(n_to_val.keys())
                    ys = [n_to_val[x] for x in xs]
                    if xs:
                        ax.plot(xs, ys, marker='o', color=colors.get(m_name, "black"), label=m_name)
                ax.set_title(mtr.capitalize())
                ax.legend(fontsize='x-small', ncol=2)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
