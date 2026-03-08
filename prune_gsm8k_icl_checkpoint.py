import modal

app = modal.App("prune-gsm8k-icl-checkpoint")
volume = modal.Volume.from_name("reft-reports", create_if_missing=True)


@app.function(volumes={"/workspace/reports": volume})
def prune_checkpoint():
    import os
    import json

    path = "/workspace/reports/stratified/ICL_GSM8K_COT_Checkpoint.jsonl"

    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    if len(lines) <= 5:
        print(f"[OK] No pruning needed. Records: {len(lines)}")
        return

    kept = lines[-5:]

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line)

    os.replace(tmp_path, path)
    print(f"[OK] Pruned checkpoint to last 5 records. Was {len(lines)}, now {len(kept)}")


@app.local_entrypoint()
def main():
    prune_checkpoint.remote()
