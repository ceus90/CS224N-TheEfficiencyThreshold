import modal

app = modal.App("cs224n-delete-lora-artifacts")

volume = modal.Volume.from_name("reft-reports")


@app.function(volumes={"/workspace/reports": volume})
def delete_files():
    import os

    targets = [
        "/workspace/reports/lora/LoRA_Efficiency_Checkpoint.jsonl",
        "/workspace/reports/lora/LoRA_Efficiency_Report_Qwen3-4B.pdf",
        "/workspace/reports/lora/LoRA_GSM8K_COT_Checkpoint.jsonl",
        "/workspace/reports/lora/LoRA_GSM8K_COT_Report_Qwen3-4B.pdf",
    ]

    for path in targets:
        if os.path.exists(path):
            os.remove(path)

    volume.commit()


@app.local_entrypoint()
def main():
    delete_files.remote()
