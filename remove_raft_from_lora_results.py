import argparse
import json
import os


def remove_raft_entries(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    kept_lines = []
    removed = 0
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            if rec.get("dataset") == "raft":
                removed += 1
                continue
            kept_lines.append(line)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    os.replace(tmp_path, path)
    print(f"[OK] Removed {removed} raft records out of {total}. Kept {len(kept_lines)}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove all RAFT entries from a JSONL LoRA results file."
    )
    parser.add_argument(
        "--path",
        default="final_results/lora_results.jsonl",
        help="Path to the JSONL results file (default: final_results/lora_results.jsonl).",
    )
    args = parser.parse_args()
    remove_raft_entries(args.path)


if __name__ == "__main__":
    main()
