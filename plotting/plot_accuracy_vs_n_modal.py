import modal

app = modal.App("plot-accuracy-vs-n")
volume = modal.Volume.from_name("reft-reports", create_if_missing=True)

plot_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib")
)


@app.function(image=plot_image, volumes={"/workspace/reports": volume})
def plot_accuracy_vs_n(icl_path: str, lora_path: str, out_pdf: str):
    import os
    import json
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

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

    def aggregate(records):
        # dataset -> n -> list of accuracy
        agg = {}
        for r in records:
            d = r.get("dataset")
            n = r.get("n")
            acc = r.get("metrics", {}).get("accuracy")
            if d is None or n is None or acc is None:
                continue
            agg.setdefault(d, {}).setdefault(n, []).append(acc)
        # dataset -> n -> mean accuracy
        mean = {}
        for d, n_map in agg.items():
            mean[d] = {n: sum(vals) / len(vals) for n, vals in n_map.items()}
        return mean

    icl_mean = aggregate(icl_records)
    lora_mean = aggregate(lora_records)

    datasets = sorted(set(icl_mean.keys()).intersection(lora_mean.keys()))
    if not datasets:
        print("[WARN] No overlapping datasets between ICL and LoRA")
        return

    # Page 1: Accuracy vs N, two lines per dataset
    with PdfPages(out_pdf) as pdf:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        if len(datasets) < 5:
            axes = axes[:len(datasets)]
        for i, d in enumerate(datasets[:5]):
            ax = axes[i]
            icl_map = icl_mean.get(d, {})
            lora_map = lora_mean.get(d, {})
            ns = sorted(set(icl_map.keys()).intersection(lora_map.keys()))
            icl_y = [icl_map[n] for n in ns]
            lora_y = [lora_map[n] for n in ns]
            ax.plot(ns, icl_y, marker='o', label="ICL")
            ax.plot(ns, lora_y, marker='^', label="LoRA")
            ax.set_title(d)
            ax.set_xlabel("N")
            if i == 0:
                ax.set_ylabel("Accuracy")
            ax.legend(fontsize='x-small')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Delta accuracy (LoRA - ICL)
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        if len(datasets) < 5:
            axes = axes[:len(datasets)]
        for i, d in enumerate(datasets[:5]):
            ax = axes[i]
            icl_map = icl_mean.get(d, {})
            lora_map = lora_mean.get(d, {})
            ns = sorted(set(icl_map.keys()).intersection(lora_map.keys()))
            delta = [lora_map[n] - icl_map[n] for n in ns]
            ax.plot(ns, delta, marker='o', color='black')
            ax.axhline(0, color='gray', linewidth=1, linestyle='--')
            ax.set_title(d)
            ax.set_xlabel("N")
            if i == 0:
                ax.set_ylabel("Δ Accuracy (LoRA - ICL)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Mean delta across datasets
        # Compute mean delta at each N across datasets
        all_ns = sorted(set(n for d in datasets for n in set(icl_mean[d].keys()).intersection(lora_mean[d].keys())))
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
        ax.set_ylabel("Mean Δ Accuracy (LoRA - ICL)")
        ax.set_title("Mean Δ Accuracy Across Datasets")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[OK] Saved: {out_pdf}")


@app.local_entrypoint()
def main(icl: str, lora: str, output: str):
    plot_accuracy_vs_n.remote(icl, lora, output)
