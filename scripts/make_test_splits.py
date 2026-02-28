# scripts/make_test_splits.py
import os
import json
import glob
from datasets import load_dataset

def get_blacklisted_indices(base_dir, ds_key):
    """Reads all N*.jsonl files and extracts the row_idx to prevent data leakage."""
    used_indices = set()
    ds_dir = os.path.join(base_dir, ds_key)
    
    if not os.path.exists(ds_dir):
        return used_indices
        
    for file_path in glob.glob(os.path.join(ds_dir, "N*.jsonl")):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if "meta" in data and "row_idx" in data["meta"]:
                    used_indices.add(data["meta"]["row_idx"])
                    
    return used_indices

def main():
    configs = {
        "gsm8k": {
            "path": "openai/gsm8k", "name": "main", "split": "test", "is_train": False
        },
        "superglue_boolq": {
            "path": "super_glue", "name": "boolq", "split": "validation", "is_train": False
        },
        "superglue_rte": {
            "path": "super_glue", "name": "rte", "split": "validation", "is_train": False
        },
        "financial_phrasebank": {
            "path": "financial_phrasebank", "name": "sentences_allagree", "split": "train", "is_train": True
        },
        "raft": {
            "path": "ought/raft", "name": "ade_corpus_v2", "split": "train", "is_train": True
        },
        "ifbench": {
            "path": "google/IFEval", "name": None, "split": "train", "is_train": True
        }
    }

    base_dir = "data/splits_out"
    os.makedirs(base_dir, exist_ok=True)

    for ds_key, cfg in configs.items():
        print(f"Building test set for {ds_key}...")
        try:
            ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], trust_remote_code=True) if cfg["name"] else load_dataset(cfg["path"], split=cfg["split"], trust_remote_code=True)
            
            test_samples = []
            
            if cfg["is_train"]:
                # DYNAMIC BLACKLISTING: Avoid any row used in the N splits
                blacklist = get_blacklisted_indices(base_dir, ds_key)
                print(f"  -> Blacklisted {len(blacklist)} rows used in training splits.")
                
                for idx, row in enumerate(ds):
                    if idx not in blacklist:
                        # Append the row along with its original index
                        test_samples.append((idx, row))
                    if len(test_samples) == 50:
                        break
            else:
                # Safe to just take the first 50 from official test/validation splits
                for idx in range(min(50, len(ds))):
                    test_samples.append((idx, ds[idx]))

            out_dir = os.path.join(base_dir, ds_key)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, "test_50.jsonl")

            with open(out_file, "w", encoding="utf-8") as f:
                for idx, row in test_samples:
                    x_val, y_val = "", ""
                    
                    if ds_key == "gsm8k":
                        x_val = f"Question: {row['question']}\nAnswer: Let's think step by step. Return the final answer after ####.\n"
                        y_val = row["answer"].split("####")[-1].strip()
                    elif ds_key == "superglue_boolq":
                        x_val = f"Passage: {row['passage']}\nQuestion: {row['question']}\nAnswer (True or False): "
                        y_val = "True" if row["label"] == 1 else "False"
                    elif ds_key == "superglue_rte":
                        x_val = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
                        y_val = "not_entailment" if row["label"] == 1 else "entailment"
                    elif ds_key == "financial_phrasebank":
                        x_val = f"Sentence: {row['sentence']}\nSentiment classification (0=negative, 1=neutral, 2=positive): Label "
                        y_val = {0: "negative", 1: "neutral", 2: "positive"}.get(row["label"], str(row["label"]))
                    elif ds_key == "raft":
                        x_val = f"Sentence: {row['Sentence']}\nClassification label: "
                        y_val = str(row["Label"])
                    elif ds_key == "ifbench":
                        prompt = row.get("prompt", row.get("instruction", ""))
                        x_val = f"Instruction: {prompt}\nResponse: "
                        y_val = ""

                    json_obj = {
                        "id": f"{ds_key}_test_{idx}",
                        "x": x_val,
                        "y": y_val,
                        "label": y_val if y_val else "generation",
                        "meta": {"dataset": ds_key, "split": "test", "row_idx": idx}
                    }
                    f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                    
            print(f"  -> Saved 50 leak-free test samples to {out_file}\n")
            
        except Exception as e:
            print(f"  -> [ERROR] Failed to process {ds_key}: {e}\n")

if __name__ == "__main__":
    main()