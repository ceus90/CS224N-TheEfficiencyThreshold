import json
import modal

app = modal.App("cs224n-regenerate-icl-pdfs")

volume = modal.Volume.from_name("reft-reports")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("matplotlib")
)


def _load_checkpoint(path):
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


@app.function(image=image, volumes={"/workspace/reports": volume})
def regenerate():
    import os
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    ckpt_path = "/workspace/reports/stratified/ICL_Efficiency_Checkpoint.jsonl"
    out_dir = "/workspace/reports/stratified"

    # Delete specified PDFs if present.
    for fname in [
        "ICL_Efficiency_Report_Qwen3-4B-Instruct-2507.pdf",
        "ICL_Efficiency_Report_Qwen3-4B.pdf",
    ]:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)

    records = _load_checkpoint(ckpt_path)
    if not records:
        return

    # Build data: model -> dataset -> metric -> list of (n, value)
    data = {}
    for rec in records:
        m = rec.get("model")
        d = rec.get("dataset")
        n = rec.get("n")
        metrics = rec.get("metrics", {})
        if not (m and d and isinstance(n, int)):
            continue
        data.setdefault(m, {}).setdefault(d, {})
        for k, v in metrics.items():
            data[m][d].setdefault(k, []).append((n, v))

    # Sort points by N.
    for m in data:
        for d in data[m]:
            for k in data[m][d]:
                data[m][d][k].sort(key=lambda x: x[0])

    def sanitize_name(name):
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

    # Determine metric order from any record.
    metric_order = [
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

    target_models = {"Qwen3-4B-Instruct-2507", "Qwen3-4B"}
    for m_name in data.keys():
        if m_name not in target_models:
            continue
        model_tag = sanitize_name(m_name)
        pdf_path = f"{out_dir}/ICL_Efficiency_Report_{model_tag}.pdf"
        with PdfPages(pdf_path) as pdf:
            for d_name, metrics in data[m_name].items():
                fig, axes = plt.subplots(3, 3, figsize=(18, 14))
                fig.suptitle(f"ICL Efficiency Benchmarks: {m_name} | {d_name}")
                for i, mtr in enumerate(metric_order):
                    ax = axes.flatten()[i]
                    if mtr in metrics:
                        xs = [n for n, _ in metrics[mtr]]
                        ys = [v for _, v in metrics[mtr]]
                        ax.plot(xs, ys, marker="o", color="black", label=m_name)
                    ax.set_title(mtr)
                    ax.legend(fontsize="x-small", ncol=1)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    volume.commit()


@app.local_entrypoint()
def main():
    regenerate.remote()
