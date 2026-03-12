import modal

app = modal.App("plot-pareto-tradeoff")
volume = modal.Volume.from_name("reft-reports", create_if_missing=True)

plot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib")
)


@app.function(image=plot_image, volumes={"/workspace/reports": volume})
def plot_pareto(icl_path: str, lora_path: str, out_dir: str):
    import os
    import json
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

    icl_records = load_records(icl_path)
    lora_records = load_records(lora_path)

    if not icl_records:
        print("[WARN] No ICL records found")
        return
    if not lora_records:
        print("[WARN] No LoRA records found")
        return

    os.makedirs(out_dir, exist_ok=True)

    dataset_colors = {
        "superglue_rte": "blue",
        "superglue_boolq": "orange",
        "financial_phrasebank": "green",
        "raft": "red",
        "gsm8k": "purple"
    }

    def index_by_key(records):
        idx = {}
        for r in records:
            key = (r.get("dataset"), r.get("model"), r.get("n"))
            if None in key:
                continue
            idx[key] = r
        return idx

    def add_legends(ax):
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

    # Accuracy vs Latency
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in icl_records:
        d = r.get("dataset")
        m = r.get("metrics", {})
        if d is None or "accuracy" not in m or "latency" not in m:
            continue
        ax.scatter(m["latency"], m["accuracy"], color=dataset_colors.get(d, "black"), marker="o", alpha=0.7)
    for r in lora_records:
        d = r.get("dataset")
        m = r.get("metrics", {})
        if d is None or "accuracy" not in m or "latency" not in m:
            continue
        ax.scatter(m["latency"], m["accuracy"], color=dataset_colors.get(d, "black"), marker="^", alpha=0.7)

    icl_idx = index_by_key(icl_records)
    lora_idx = index_by_key(lora_records)
    for key in set(icl_idx.keys()).intersection(lora_idx.keys()):
        d = key[0]
        icl_m = icl_idx[key].get("metrics", {})
        lora_m = lora_idx[key].get("metrics", {})
        if "accuracy" not in icl_m or "accuracy" not in lora_m:
            continue
        if "latency" not in icl_m or "latency" not in lora_m:
            continue
        ax.annotate(
            "",
            xy=(lora_m["latency"], lora_m["accuracy"]),
            xytext=(icl_m["latency"], icl_m["accuracy"]),
            arrowprops=dict(arrowstyle="->", color=dataset_colors.get(d, "black"), alpha=0.5, lw=1)
        )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Latency")
    add_legends(ax)
    out_path = os.path.join(out_dir, "pareto_accuracy_vs_latency.pdf")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")

    # Accuracy vs VRAM
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

    for key in set(icl_idx.keys()).intersection(lora_idx.keys()):
        d = key[0]
        icl_m = icl_idx[key].get("metrics", {})
        lora_m = lora_idx[key].get("metrics", {})
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
    add_legends(ax)
    out_path = os.path.join(out_dir, "pareto_accuracy_vs_vram.pdf")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Saved: {out_path}")


@app.local_entrypoint()
def main(icl: str, lora: str, outdir: str):
    plot_pareto.remote(icl, lora, outdir)
