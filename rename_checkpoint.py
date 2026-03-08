import modal

app = modal.App("rename-icl-checkpoint")

volume = modal.Volume.from_name("reft-reports", create_if_missing=True)


@app.function(volumes={"/workspace/reports": volume})
def rename_checkpoint():
    import os

    src = "/workspace/reports/LoRA_Efficiency_Report_Gemma-2-9B_2.pdf"
    dst = "/workspace/reports/LoRA_Efficiency_Report_Gemma-2-9B.pdf"

    if not os.path.exists(src):
        print(f"[WARN] Source not found: {src}")
        return

    if os.path.exists(dst):
        print(f"[WARN] Destination already exists: {dst}")
        return

    os.rename(src, dst)
    print(f"[OK] Renamed {src} -> {dst}")


@app.local_entrypoint()
def main():
    rename_checkpoint.remote()
