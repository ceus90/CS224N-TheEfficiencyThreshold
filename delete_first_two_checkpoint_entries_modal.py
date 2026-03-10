import modal

app = modal.App("cs224n-delete-first-two-icl-checkpoint")

volume = modal.Volume.from_name("reft-reports")


@app.function(volumes={"/workspace/reports": volume})
def delete_first_two():
    path = "/workspace/reports/stratified/ICL_Efficiency_Checkpoint.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    remaining = lines[2:] if len(lines) > 2 else []
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(remaining)
    volume.commit()


@app.local_entrypoint()
def main():
    delete_first_two.remote()
