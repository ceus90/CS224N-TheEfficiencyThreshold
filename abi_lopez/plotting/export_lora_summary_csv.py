#!/usr/bin/env python3
import csv
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(REPO_ROOT, "final_results")
LORA_PATH = os.path.join(BASE_DIR, "lora_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "lora_summary.csv")


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def aggregate(records):
    # dataset -> n -> list of metrics
    agg = {}
    for r in records:
        d = r.get("dataset")
        n = r.get("n")
        metrics = r.get("metrics", {})
        train_vram = metrics.get("train_vram_pct")
        train_time = metrics.get("train_time")
        lat = metrics.get("latency")
        tpot = metrics.get("tpot")
        if d is None or not isinstance(n, int):
            continue
        if train_vram is None or train_time is None or lat is None or tpot is None:
            continue
        agg.setdefault(d, {}).setdefault(n, []).append((train_vram, train_time, lat, tpot))
    return agg


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    records = load_records(LORA_PATH)
    if not records:
        raise ValueError("No LoRA records found")

    agg = aggregate(records)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "N",
            "train_vram_pct",
            "train_time",
            "latency",
            "tpot",
        ])
        for d in sorted(agg.keys()):
            for n in sorted(agg[d].keys()):
                vals = agg[d][n]
                train_vram = mean([v[0] for v in vals])
                train_time = mean([v[1] for v in vals])
                lat = mean([v[2] for v in vals])
                tpot = mean([v[3] for v in vals])
                writer.writerow([
                    d,
                    n,
                    f"{train_vram:.4f}",
                    f"{train_time:.4f}",
                    f"{lat:.4f}",
                    f"{tpot:.4f}",
                ])

    print(f"Saved: {OUT_PATH}")
