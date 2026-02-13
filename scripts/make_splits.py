# scripts/make_splits.py
"""
Generate few-shot splits for multiple datasets and upload to GCS.

This script:
- Reads configs/splits.json
- Calls each dataset adapter via a uniform entrypoint: load_examples(dataset_cfg) -> list[dict]
- Uses src.splits.make_splits(...) to produce nested splits (unless a dataset uses a special strategy)
- Writes JSONL files locally
- Uploads JSONL + a manifest.json to Google Cloud Storage

Run (from repo root):
  python scripts/make_splits.py --config configs/splits.json

Optional:
  python scripts/make_splits.py --config configs/splits.json --datasets financial_phrasebank,gsm8k
  python scripts/make_splits.py --config configs/splits.json --dry_run
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from google.cloud import storage

from src.splits import make_splits, indices_to_examples


# -----------------------------
# Small utilities
# -----------------------------

def _read_json(path: str) -> Dict[str, Any]:
    """Read a JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _now_utc_iso() -> str:
    """Current UTC timestamp in ISO format."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a dict as pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _gcs_blob_exists(client: storage.Client, bucket_name: str, blob_name: str) -> bool:
    """Check if a blob exists in GCS."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists(client)


def _gcs_upload_file(
    client: storage.Client,
    bucket_name: str,
    local_path: Path,
    gcs_blob_name: str,
    overwrite: bool,
) -> None:
    """Upload a local file to GCS."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)

    # Skip if exists and overwrite disabled
    if not overwrite and blob.exists(client):
        return

    blob.upload_from_filename(str(local_path))


def _gcs_upload_text(
    client: storage.Client,
    bucket_name: str,
    gcs_blob_name: str,
    text: str,
    overwrite: bool,
    content_type: str = "application/json",
) -> None:
    """Upload text content to GCS."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_blob_name)

    # Skip if exists and overwrite disabled
    if not overwrite and blob.exists(client):
        return

    blob.upload_from_string(text, content_type=content_type)


def _import_adapter(adapter_name: str):
    """
    Import an adapter module by name.

    Expects module:
      src.adapters.<adapter_name>

    Expects function:
      load_examples(dataset_cfg: dict) -> list[dict]
    """
    module_path = f"src.adapters.{adapter_name}"
    mod = importlib.import_module(module_path)

    if not hasattr(mod, "load_examples"):
        raise AttributeError(
            f"Adapter '{module_path}' must define load_examples(dataset_cfg)."
        )

    return mod


# -----------------------------
# Config validation
# -----------------------------

def _validate_top_level(cfg: Dict[str, Any]) -> None:
    """Validate top-level config structure."""
    if "project" not in cfg or "gcs_bucket" not in cfg["project"]:
        raise KeyError("Config missing project.gcs_bucket")

    if "splits" not in cfg:
        raise KeyError("Config missing splits")

    splits = cfg["splits"]
    for k in ["Ns", "seeds"]:
        if k not in splits:
            raise KeyError(f"Config missing splits.{k}")

    Ns = splits["Ns"]
    if not isinstance(Ns, list) or not Ns:
        raise ValueError("splits.Ns must be a non-empty list")
    if Ns != sorted(Ns):
        raise ValueError("splits.Ns must be sorted ascending for nested splits")

    seeds = splits["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("splits.seeds must be a non-empty list")

    if "datasets" not in cfg or not isinstance(cfg["datasets"], list) or not cfg["datasets"]:
        raise ValueError("Config missing datasets list")

    for d in cfg["datasets"]:
        if "name" not in d:
            raise KeyError("Each dataset must have a 'name'")
        if "adapter" not in d:
            raise KeyError(f"Dataset '{d.get('name','<unknown>')}' missing 'adapter'")


def _get_effective_bool(dcfg: Dict[str, Any], top_splits: Dict[str, Any], key: str, default: bool) -> bool:
    """Dataset override beats top-level splits setting."""
    if key in dcfg:
        return bool(dcfg[key])
    if key in top_splits:
        return bool(top_splits[key])
    return default


# -----------------------------
# Split generation
# -----------------------------

def _dataset_out_prefix(dataset_name: str) -> str:
    """GCS prefix for a dataset's splits."""
    return f"splits/{dataset_name}/"


def _build_manifest(
    *,
    dataset_cfg: Dict[str, Any],
    splits_cfg: Dict[str, Any],
    num_examples: int,
    label_counts: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    """Create a manifest.json payload for a dataset's splits."""
    return {
        "created_at_utc": _now_utc_iso(),
        "dataset": {
            "name": dataset_cfg.get("name"),
            "adapter": dataset_cfg.get("adapter"),
            "source": {
                "hf_id": dataset_cfg.get("hf_id"),
                "hf_config": dataset_cfg.get("hf_config"),
                "hf_split": dataset_cfg.get("hf_split"),
                "trust_remote_code": dataset_cfg.get("trust_remote_code"),
            },
            "settings": {
                "label_key": dataset_cfg.get("label_key", splits_cfg.get("label_key", "label")),
                "stratify": dataset_cfg.get("stratify", splits_cfg.get("stratify", True)),
                "sampling_strategy": dataset_cfg.get("sampling_strategy", "default"),
            },
        },
        "splits": {
            "Ns": splits_cfg.get("Ns"),
            "seeds": splits_cfg.get("seeds"),
            "nested": splits_cfg.get("nested", True),
        },
        "stats": {
            "num_examples_loaded": num_examples,
            "label_counts": label_counts,
        },
    }


def _compute_label_counts(examples: List[Dict[str, Any]], label_key: str) -> Optional[Dict[str, int]]:
    """Compute label counts if label_key exists in examples."""
    if not examples:
        return None
    if label_key not in examples[0]:
        return None

    counts: Dict[str, int] = {}
    for ex in examples:
        v = ex.get(label_key)
        if v is None:
            continue
        k = str(v)
        counts[k] = counts.get(k, 0) + 1
    return counts


def _run_default_sampling(
    *,
    examples: List[Dict[str, Any]],
    Ns: Sequence[int],
    seeds: Sequence[int],
    label_key: str,
    stratify: bool,
) -> Dict[int, Dict[int, List[int]]]:
    """Default sampling path using src.splits.make_splits."""
    return make_splits(
        examples=examples,
        Ns=list(Ns),
        seeds=list(seeds),
        label_key=label_key,
        stratify=stratify,
        min_per_class=1,
    )


# -----------------------------
# Main driver
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/splits.json")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names to run (default: all).",
    )
    parser.add_argument(
        "--local_out",
        default="data/splits_out",
        help="Local output directory for JSONL splits.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not upload to GCS (still writes locally).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing objects in GCS.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of examples loaded (debugging).",
    )
    args = parser.parse_args()

    # Load config
    cfg = _read_json(args.config)
    _validate_top_level(cfg)

    bucket_name = cfg["project"]["gcs_bucket"]
    splits_cfg = cfg["splits"]
    Ns = splits_cfg["Ns"]
    seeds = splits_cfg["seeds"]

    # Optional dataset filter
    only_datasets = None
    if args.datasets:
        only_datasets = set([x.strip() for x in args.datasets.split(",") if x.strip()])

    # Create local output root
    local_root = Path(args.local_out)
    _ensure_dir(local_root)

    # Create GCS client (will use ADC: gcloud auth / Colab auth)
    client = storage.Client()

    # Process each dataset
    for dcfg in cfg["datasets"]:
        dname = dcfg["name"]
        adapter_name = dcfg["adapter"]

        if only_datasets and dname not in only_datasets:
            continue

        print("=" * 100)
        print(f"DATASET: {dname}")
        print(f"Adapter:  {adapter_name}")

        # Import adapter module
        adapter_mod = _import_adapter(adapter_name)

        # Load standardized examples
        examples: List[Dict[str, Any]] = adapter_mod.load_examples(dcfg)  # type: ignore[attr-defined]

        # Optional cap for debugging
        if args.max_examples is not None:
            examples = examples[: args.max_examples]

        if not examples:
            raise RuntimeError(f"Adapter '{adapter_name}' returned 0 examples for dataset '{dname}'")

        # Effective label key + stratify for this dataset
        label_key = dcfg.get("label_key", splits_cfg.get("label_key", "label"))
        stratify = _get_effective_bool(dcfg, splits_cfg, "stratify", True)

        # Effective sampling strategy
        sampling_strategy = dcfg.get("sampling_strategy", "default")
        if sampling_strategy != "default":
            raise NotImplementedError(
                f"sampling_strategy='{sampling_strategy}' not implemented yet for dataset '{dname}'. "
                f"(We will add RAFT round-robin next.)"
            )

        # Sanity: dataset must be big enough
        if max(Ns) > len(examples):
            raise ValueError(
                f"Dataset '{dname}' too small: size={len(examples)} < maxN={max(Ns)}"
            )

        # Compute splits (indices)
        split_indices = _run_default_sampling(
            examples=examples,
            Ns=Ns,
            seeds=seeds,
            label_key=label_key,
            stratify=stratify,
        )

        # Stats for manifest
        label_counts = _compute_label_counts(examples, label_key=label_key)

        # Local + GCS output prefix
        out_prefix = _dataset_out_prefix(dname)
        local_ds_dir = local_root / dname
        _ensure_dir(local_ds_dir)

        print(f"Examples loaded: {len(examples)}")
        if label_counts:
            print(f"Label counts:    {label_counts}")
        print(f"GCS prefix:      gs://{bucket_name}/{out_prefix}")
        print(f"Local dir:       {local_ds_dir}")

        # Write and upload manifest
        manifest = _build_manifest(
            dataset_cfg=dcfg,
            splits_cfg=splits_cfg,
            num_examples=len(examples),
            label_counts=label_counts,
        )
        manifest_local = local_ds_dir / "manifest.json"
        _write_json(manifest_local, manifest)

        if not args.dry_run:
            _gcs_upload_file(
                client=client,
                bucket_name=bucket_name,
                local_path=manifest_local,
                gcs_blob_name=f"{out_prefix}manifest.json",
                overwrite=args.overwrite,
            )

        # Write and upload each (N, seed) split
        num_written = 0
        num_uploaded = 0

        for seed in seeds:
            for N in Ns:
                idxs = split_indices[seed][N]
                rows = indices_to_examples(examples, idxs)

                # Local filename
                fname = f"N{N}_seed{seed}.jsonl"
                local_path = local_ds_dir / fname
                _write_jsonl(local_path, rows)
                num_written += 1

                # Upload to GCS
                if not args.dry_run:
                    blob_name = f"{out_prefix}{fname}"
                    existed = _gcs_blob_exists(client, bucket_name, blob_name)
                    _gcs_upload_file(
                        client=client,
                        bucket_name=bucket_name,
                        local_path=local_path,
                        gcs_blob_name=blob_name,
                        overwrite=args.overwrite,
                    )
                    # Count upload if it actually wrote or overwrote
                    if args.overwrite or not existed:
                        num_uploaded += 1

        print(f"Wrote files:     {num_written} (local)")
        if args.dry_run:
            print("Uploaded files:  0 (dry_run)")
        else:
            print(f"Uploaded files:  {num_uploaded} (GCS)")

    print("\nDone.")


if __name__ == "__main__":
    main()
