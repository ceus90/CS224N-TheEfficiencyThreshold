"""
make_test_splits.py

Creates deterministic test_50.jsonl files in data/splits_out/<dataset>/test_50.jsonl.

Important design choice:
- These test splits store *raw* inputs in `x` (no inserted prompt templates).
- They match the style of the training split JSONLs (id/x/y/label/meta...).

How to run:
  python make_test_splits.py

Outputs:
  data/splits_out/<ds_key>/test_50.jsonl
"""

import os
import json
import glob
import random
from datasets import load_dataset


def get_blacklisted_ids(base_dir, ds_key):
    """
    Reads all N*.jsonl files and extracts the 'id' to prevent data leakage.
    Using 'id' is safer than 'row_idx' for RAFT because indices reset per task.
    """
    used_ids = set()
    ds_dir = os.path.join(base_dir, ds_key)
    if not os.path.exists(ds_dir):
        return used_ids

    for file_path in glob.glob(os.path.join(ds_dir, "N*.jsonl")):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        used_ids.add(data["id"])
                except Exception:
                    continue
    return used_ids


def verify_no_leakage(base_dir, ds_keys):
    """
    Final safety check: Cross-references all test_50.jsonl files against
    all N*.jsonl files. Raises error if any overlap is found.
    """
    print("\n" + "=" * 40)
    print("RUNNING FINAL DATA LEAK-CHECK...")
    print("=" * 40)

    for ds in ds_keys:
        ds_path = os.path.join(base_dir, ds)
        if not os.path.exists(ds_path):
            print(f"  [?] {ds}: No local training splits found. Skipping check.")
            continue

        training_ids = set()
        for fpath in glob.glob(os.path.join(ds_path, "N*.jsonl")):
            with open(fpath, "r", encoding="utf-8") as src:
                for line in src:
                    training_ids.add(json.loads(line)["id"])

        test_file = os.path.join(ds_path, "test_50.jsonl")
        if not os.path.exists(test_file):
            continue

        test_ids = set()
        with open(test_file, "r", encoding="utf-8") as src:
            for line in src:
                test_ids.add(json.loads(line)["id"])

        overlap = training_ids.intersection(test_ids)
        if overlap:
            raise RuntimeError(f"!!! DATA LEAK DETECTED in {ds} !!!\nOverlapping IDs: {overlap}")
        else:
            print(f"  [v] {ds}: PASS (Checked {len(test_ids)} test vs {len(training_ids)} context IDs)")


def main():
    RAFT_KEY = "raft"
    base_dir = "data/splits_out"
    TEST_SEED = 42  # deterministic 50-sample selection

    configs = {
        "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "test"},
        "superglue_boolq": {"path": "super_glue", "name": "boolq", "split": "validation"},
        "superglue_rte": {"path": "super_glue", "name": "rte", "split": "validation"},
        "financial_phrasebank": {"path": "financial_phrasebank", "name": "sentences_allagree", "split": "train"},
        "ifbench": {"path": "google/IFEval", "name": None, "split": "train"},
    }

    """
    # --- PART 1: PROCESS STANDARD DATASETS (RAW x, RAW y/label) ---
    for ds_key, cfg in configs.items():
        print(f"Building seeded test set for {ds_key}...")
        try:
            if cfg["name"]:
                ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], trust_remote_code=True)
            else:
                ds = load_dataset(cfg["path"], split=cfg["split"], trust_remote_code=True)

            blacklist = get_blacklisted_ids(base_dir, ds_key)
            candidates = []

            for idx, row in enumerate(ds):
                current_id = f"{ds_key}_test_{idx}"
                if current_id in blacklist or str(idx) in blacklist:
                    continue

                if ds_key == "gsm8k":
                    # raw question text only; label is final numeric answer after ####
                    x_val = row["question"]
                    y_val = row["answer"].split("####")[-1].strip()
                    label_val = y_val

                    meta = {"dataset": "gsm8k", "config": cfg["name"], "split": cfg["split"], "row_idx": idx}

                elif ds_key == "superglue_boolq":
                    # keep raw fields but as one string (like many train splits do)
                    x_val = f"Passage: {row['passage']}\nQuestion: {row['question']}"
                    y_val = "True" if row["label"] == 1 else "False"
                    label_val = y_val

                    meta = {"dataset": "super_glue", "config": "boolq", "split": cfg["split"], "row_idx": idx, "label_id": int(row["label"])}

                elif ds_key == "superglue_rte":
                    x_val = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
                    # keep same convention as your existing code (even if it's odd)
                    y_val = "not_entailment" if row["label"] == 1 else "entailment"
                    label_val = y_val

                    meta = {"dataset": "super_glue", "config": "rte", "split": cfg["split"], "row_idx": idx, "label_id": int(row["label"])}

                elif ds_key == "financial_phrasebank":
                    x_val = row["sentence"]
                    y_val = {0: "negative", 1: "neutral", 2: "positive"}.get(int(row["label"]), str(row["label"]))
                    label_val = y_val

                    meta = {
                        "dataset": "financial_phrasebank",
                        "config": cfg["name"],
                        "split": cfg["split"],
                        "row_idx": idx,
                        "label_id": int(row["label"]),
                    }

                elif ds_key == "ifbench":
                    prompt = row.get("prompt", row.get("instruction", ""))
                    x_val = prompt
                    y_val = ""
                    label_val = ""

                    meta = {"dataset": "google/IFEval", "split": cfg["split"], "row_idx": idx}

                candidates.append(
                    {
                        "id": current_id,
                        "x": x_val,
                        "y": y_val,
                        "label": label_val,
                        "meta": meta,
                    }
                )

            random.seed(TEST_SEED)
            random.shuffle(candidates)
            test_samples = candidates[:50]

            out_dir = os.path.join(base_dir, ds_key)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "test_50.jsonl"), "w", encoding="utf-8") as f:
                for s in test_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
            print(f"  -> Saved {len(test_samples)} seeded samples to {ds_key}/test_50.jsonl")

        except Exception as e:
            print(f"  -> [ERROR] Failed processing {ds_key}: {e}")

        """
    
        # --- PART 2: PROCESS RAFT (SINGLE-TASK TEST SET: ade_corpus_v2 ONLY) ---
    print(f"\nBuilding single-task test set for {RAFT_KEY} (ade_corpus_v2 test split)...")

    raft_task = "ade_corpus_v2"

    # IMPORTANT:
    # For true "test", use the HF split "test" (5000 labeled examples).
    # No need to blacklist against training context because it's a disjoint split.
    # (We still *can* avoid id collisions defensively, but it should never overlap.)
    blacklist = get_blacklisted_ids(base_dir, RAFT_KEY)

    candidates = []
    try:
        ds = load_dataset("ought/raft", raft_task, split="test")  # no trust_remote_code
        for idx, row in enumerate(ds):
            # Stable id in the same style as train, but with split=test
            stable_id = f"raft_{raft_task}_test_{idx:06d}"

            # Defensive: skip if somehow collides with already-used ids
            if stable_id in blacklist:
                continue

            x_val = row["Sentence"]

            label_id = int(row["Label"])
            label_str = ds.features["Label"].int2str(label_id)

            candidates.append(
                {
                    "id": stable_id,
                    "x": x_val,
                    "y": label_str,
                    "label": label_str,
                    "label_id": label_id,
                    "task_name": raft_task,
                    "meta": {
                        "dataset": "ought/raft",
                        "task_name": raft_task,
                        "split": "test",
                        "row_idx": idx,
                        "label_key": "Label",
                        "orig_id": row.get("ID"),  # keep HF-provided ID if you want traceability
                    },
                }
            )

        random.seed(TEST_SEED)
        random.shuffle(candidates)
        test_samples = candidates[:50]

        raft_out_dir = os.path.join(base_dir, RAFT_KEY)
        os.makedirs(raft_out_dir, exist_ok=True)

        with open(os.path.join(raft_out_dir, "test_50.jsonl"), "w", encoding="utf-8") as f:
            for s in test_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"  -> SUCCESS: Saved {len(test_samples)} seeded RAFT test samples to {RAFT_KEY}/test_50.jsonl")

    except Exception as e:
        print(f"  -> [ERROR] Failed processing {RAFT_KEY} ({raft_task}): {e}")

    # --- PART 3: FINAL VALIDATION ---
    ds_list = list(configs.keys()) + [RAFT_KEY]
    verify_no_leakage(base_dir, ds_list)


if __name__ == "__main__":
    main()