import modal

app = modal.App("plot-icl-from-checkpoint")
volume = modal.Volume.from_name("reft-reports", create_if_missing=True)

plot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib")
)


@app.function(image=plot_image, volumes={"/workspace/reports": volume})
def plot_from_checkpoint(ckpt_path: str, out_path: str | None = None):
    import os
    import json
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found at {ckpt_path}")
        return

    records = []
    with open(ckpt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("[WARN] No records found in checkpoint")
        return

    # Build map: dataset -> model -> metric -> {n: value}
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

    if out_path is None:
        base_dir = os.path.dirname(ckpt_path) or "/workspace/reports"
        out_path = os.path.join(base_dir, "Efficiency_Report_AllModels.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    label = "LoRA" if "lora" in os.path.basename(ckpt_path).lower() else "ICL"
    with PdfPages(out_path) as pdf:
        for d_name, model_map in data.items():
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f"{label} Efficiency Benchmarks (All Models): {d_name}")
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

    print(f"[OK] Saved: {out_path}")


@app.local_entrypoint()
def main(checkpoint: str, output: str = ""):
    out_path = output if output else None
    plot_from_checkpoint.remote(checkpoint, out_path)
