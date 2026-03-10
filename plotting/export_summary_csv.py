#!/usr/bin/env python3
import csv
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(REPO_ROOT, "final_results")
ICL_PATH = os.path.join(BASE_DIR, "icl_results.jsonl")
LORA_PATH = os.path.join(BASE_DIR, "lora_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "summary_metrics.csv")

DATASET_ORDER = [
    ("gsm8k", "GSM8K"),
    ("raft", "RAFT"),
    ("financial_phrasebank", "Financial Phrasebank"),
    ("superglue_rte", "Superglue RTE"),
    ("superglue_boolq", "Superglue BoolQ"),
]


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


def collect_means(records, vram_key):
    # dataset -> list of (throughput_output, tpot, vram)
    agg = {}
    for r in records:
        d = r.get("dataset")
        metrics = r.get("metrics", {})
        thr = metrics.get("throughput_output")
        tpot = metrics.get("tpot")
        vram = metrics.get(vram_key)
        if d is None or thr is None or tpot is None or vram is None:
            continue
        agg.setdefault(d, []).append((thr, tpot, vram))

    means = {}
    for d, vals in agg.items():
        n = len(vals)
        thr_mean = sum(v[0] for v in vals) / n
        tpot_mean = sum(v[1] for v in vals) / n
        vram_mean = sum(v[2] for v in vals) / n
        means[d] = (thr_mean, tpot_mean, vram_mean)
    return means


if __name__ == "__main__":
    icl_records = load_records(ICL_PATH)
    lora_records = load_records(LORA_PATH)

    if not icl_records:
        raise ValueError("No ICL records found")
    if not lora_records:
        raise ValueError("No LoRA records found")

    icl_means = collect_means(icl_records, "vram_pct")
    lora_means = collect_means(lora_records, "eval_vram_pct")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Method", "Throughput (tok/s)", "TPOT (ms)", "Peak VRAM (%)"])
        for key, label in DATASET_ORDER:
            if key in icl_means:
                thr, tpot, vram = icl_means[key]
                writer.writerow([label, "ICL", f"{thr:.4f}", f"{tpot:.4f}", f"{vram:.4f}"])
            if key in lora_means:
                thr, tpot, vram = lora_means[key]
                writer.writerow([label, "LoRA", f"{thr:.4f}", f"{tpot:.4f}", f"{vram:.4f}"])

    print(f"Saved: {OUT_PATH}")
