#!/usr/bin/env python3
import csv
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.join(REPO_ROOT, "final_results")
ICL_PATH = os.path.join(BASE_DIR, "icl_results.jsonl")
OUT_PATH = os.path.join(BASE_DIR, "icl_prompt_scaling_summary.csv")


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
        avg_in = metrics.get("avg_input_tokens")
        max_in = metrics.get("max_input_tokens")
        trunc = metrics.get("truncation_rate")
        lat = metrics.get("latency")
        tpot = metrics.get("tpot")
        if d is None or not isinstance(n, int):
            continue
        if avg_in is None or max_in is None or trunc is None or lat is None or tpot is None:
            continue
        agg.setdefault(d, {}).setdefault(n, []).append((avg_in, max_in, trunc, lat, tpot))
    return agg


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


if __name__ == "__main__":
    records = load_records(ICL_PATH)
    if not records:
        raise ValueError("No ICL records found")

    agg = aggregate(records)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "N",
            "avg_input_tokens",
            "max_input_tokens",
            "utilization",
            "truncation_rate",
            "latency",
            "tpot",
        ])
        for d in sorted(agg.keys()):
            for n in sorted(agg[d].keys()):
                vals = agg[d][n]
                avg_in = mean([v[0] for v in vals])
                max_in = mean([v[1] for v in vals])
                trunc = mean([v[2] for v in vals])
                lat = mean([v[3] for v in vals])
                tpot = mean([v[4] for v in vals])
                util = (avg_in / max_in) if max_in > 0 else 0.0
                writer.writerow([
                    d,
                    n,
                    f"{avg_in:.4f}",
                    f"{max_in:.4f}",
                    f"{util:.6f}",
                    f"{trunc:.6f}",
                    f"{lat:.4f}",
                    f"{tpot:.4f}",
                ])

    print(f"Saved: {OUT_PATH}")
