import argparse
import json
import os
from typing import Dict, Tuple, Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ICL_METRICS = [
    "accuracy",
    "latency",
    "vram_pct",
    "throughput_total",
    "throughput_output",
    "tpot",
    "avg_input_tokens",
    "max_input_tokens",
    "truncation_rate",
]

LORA_METRICS = [
    "accuracy",
    "latency",
    "train_vram_pct",
    "eval_vram_pct",
    "throughput_total",
    "throughput_output",
    "train_time",
    "tpot",
]

COLORS = {
    "Llama-3-8B-Instruct": "blue",
    "Qwen3-4B": "orange",
    "Qwen3-4B-Instruct-2507": "brown",
    "Qwen3-8B": "red",
    "Qwen3-8B-Thinking": "purple",
    "Qwen2.5-7B-Instruct": "magenta",
    "Gemma-2-9B": "green",
    "Gemma-2-9B-IT": "cyan",
}


def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def load_results(path: str) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    data: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            model = rec.get("model")
            dataset = rec.get("dataset")
            n = rec.get("n")
            metrics = rec.get("metrics", {})
            ts = rec.get("timestamp", 0)
            if not model or not dataset or not isinstance(n, int):
                continue
            model_map = data.setdefault(model, {})
            ds_map = model_map.setdefault(dataset, {})
            prev = ds_map.get(n)
            if prev is None or ts >= prev.get("timestamp", 0):
                ds_map[n] = {"timestamp": ts, "metrics": metrics}
    return data


def plot_model_pdf(
    model: str,
    model_data: Dict[str, Dict[int, Dict[str, Any]]],
    metrics: list,
    out_path: str,
    title_prefix: str,
) -> None:
    datasets = sorted(model_data.keys())
    with PdfPages(out_path) as pdf:
        for d_name in datasets:
            n_map = model_data[d_name]
            ns = sorted(n_map.keys())
            if not ns:
                continue
            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            fig.suptitle(f"{title_prefix}: {model} | {d_name}")
            flat_axes = axes.flatten()

            for i, mtr in enumerate(metrics):
                ax = flat_axes[i]
                y = []
                for n in ns:
                    y.append(n_map[n]["metrics"].get(mtr))
                pairs = [(n, v) for n, v in zip(ns, y) if v is not None]
                if pairs:
                    xs, ys = zip(*pairs)
                    ax.plot(
                        list(xs),
                        list(ys),
                        marker="o",
                        color=COLORS.get(model, "black"),
                        label=model,
                    )
                ax.set_title(mtr.capitalize())
                ax.legend(fontsize="x-small", ncol=1)

            for ax in flat_axes[len(metrics):]:
                ax.axis("off")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model PDFs from final JSONL results for ICL and LoRA."
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
        help="Output directory for PDFs.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(args.icl):
        icl_data = load_results(args.icl)
        for model, model_data in icl_data.items():
            model_tag = sanitize_name(model)
            out_path = os.path.join(args.out, f"ICL_Final_Report_{model_tag}.pdf")
            plot_model_pdf(
                model=model,
                model_data=model_data,
                metrics=ICL_METRICS,
                out_path=out_path,
                title_prefix="ICL Efficiency Benchmarks",
            )

    if os.path.exists(args.lora):
        lora_data = load_results(args.lora)
        for model, model_data in lora_data.items():
            model_tag = sanitize_name(model)
            out_path = os.path.join(args.out, f"LoRA_Final_Report_{model_tag}.pdf")
            plot_model_pdf(
                model=model,
                model_data=model_data,
                metrics=LORA_METRICS,
                out_path=out_path,
                title_prefix="LoRA Efficiency Benchmarks",
            )


if __name__ == "__main__":
    main()
