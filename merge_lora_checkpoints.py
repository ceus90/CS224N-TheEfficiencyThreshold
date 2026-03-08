import modal

app = modal.App("merge-lora-checkpoints")
volume = modal.Volume.from_name("reft-reports", create_if_missing=True)


@app.function(volumes={"/workspace/reports": volume})
def merge_checkpoints():
    import os

    base = "/workspace/reports/stratified"
    src = os.path.join(base, "LoRA_Efficiency_Checkpoint_2.jsonl")
    dst = os.path.join(base, "LoRA_Efficiency_Checkpoint.jsonl")

    if not os.path.exists(src):
        print(f"[WARN] Source not found: {src}")
        return

    if not os.path.exists(dst):
        print(f"[WARN] Destination not found: {dst}")
        return

    with open(src, "r", encoding="utf-8") as f_src, open(dst, "a", encoding="utf-8") as f_dst:
        for line in f_src:
            f_dst.write(line)

    print(f"[OK] Appended {src} -> {dst}")


@app.local_entrypoint()
def main():
    merge_checkpoints.remote()
