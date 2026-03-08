import modal

app = modal.App("bundle-icl-reports")
volume = modal.Volume.from_name("reft-reports", create_if_missing=True)


@app.function(volumes={"/workspace/reports": volume})
def bundle_reports():
    import os
    import tarfile

    root = "/workspace/reports"
    bundle_path = os.path.join(root, "reports_bundle.tar.gz")

    with tarfile.open(bundle_path, "w:gz") as tar:
        for name in os.listdir(root):
            if name == "reports_bundle.tar.gz":
                continue
            tar.add(os.path.join(root, name), arcname=name)

    print(f"[OK] Bundled reports at {bundle_path}")


@app.local_entrypoint()
def main():
    bundle_reports.remote()
