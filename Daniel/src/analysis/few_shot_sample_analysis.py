"""
CS224N — 0-Shot Checkpoint Sampling Analysis

Loads checkpoint JSONL files for financial_phrase_bank, gsm8k, boolq, and rte,
then evaluates model accuracy over random sub-samples of sizes
n ∈ {16, 32, 64, 128, 256} with bootstrap confidence intervals.

Outputs:
  1. Summary table (accuracy ± CI for every model × task × sample size)
  2. Convergence curves (accuracy vs. sample size per task)
  3. Bootstrap distribution violin plots
  4. Per-class accuracy heatmaps (multi-class tasks)
  5. Unknown-rate analysis
  6. Full-dataset vs. sampled accuracy comparison
  7. Macro-F1 convergence (per model) and macro-F1 convergence (avg of 3 models, 4 graphs)
"""

import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")

SEED = 42
N_BOOTSTRAP = 200
SAMPLE_SIZES = [16, 32, 64, 128, 256]

BASE_DIR = Path(__file__).resolve().parent.parent / "0_shot"
OUTPUT_DIR = Path(__file__).resolve().parent / "sample_analysis_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DISPLAY = {
    "llama_3_8b": "LLaMA-3 8B",
    "qwen_3_4b": "Qwen-3 4B",
    "qwen_3_8b": "Qwen-3 8B",
}

TASK_DISPLAY = {
    "financial_phrase_bank": "FinPhraseBank",
    "gsm8k": "GSM8K",
    "boolq": "BoolQ",
    "rte": "RTE",
}

TASK_PATHS = {
    "financial_phrase_bank": BASE_DIR / "financial_phrase_bank",
    "gsm8k": BASE_DIR / "gsm8k",
    "boolq": BASE_DIR / "superglue" / "boolq",
    "rte": BASE_DIR / "superglue" / "rte",
}

PALETTE = {
    "llama_3_8b": "#4C72B0",
    "qwen_3_4b": "#DD8452",
    "qwen_3_8b": "#55A868",
}

# ── RAFT-specific constants ────────────────────────────────────────────────────
# Each train_results.json has exactly 50 labelled examples, so sample sizes must be ≤ 50.
RAFT_SAMPLE_SIZES = [5, 10, 16, 25, 32, 40]

RAFT_TASKS = [
    "ade_corpus_v2",
    "banking_77",
    "neurips_impact_statement_risks",
    "one_stop_english",
    "overruling",
    "semiconductor_org_types",
    "tai_safety_research",
    "terms_of_service",
    "tweet_eval_hate",
    "twitter_complaints",
]

RAFT_TASK_DISPLAY = {
    "ade_corpus_v2":                  "ADE Corpus v2",
    "banking_77":                     "Banking-77",
    "neurips_impact_statement_risks": "NeurIPS Impact",
    "one_stop_english":               "One Stop English",
    "overruling":                     "Overruling",
    "semiconductor_org_types":        "Semiconductor Orgs",
    "tai_safety_research":            "TAI Safety",
    "terms_of_service":               "Terms of Service",
    "tweet_eval_hate":                "Tweet Eval Hate",
    "twitter_complaints":             "Twitter Complaints",
}

RAFT_DIR = BASE_DIR / "raft"


# ── Loading Utilities ─────────────────────────────────────────────────────────

def _parse_concatenated_json(text: str) -> list[dict]:
    """Parse a string of concatenated JSON objects (no newline delimiters)."""
    decoder = json.JSONDecoder()
    results = []
    idx = 0
    text = text.strip()
    while idx < len(text):
        try:
            obj, end_idx = decoder.raw_decode(text, idx)
            results.append(obj)
            idx = end_idx
            while idx < len(text) and text[idx] in (" ", "\t", "\r", "\n"):
                idx += 1
        except json.JSONDecodeError:
            break
    return results


def load_checkpoint(filepath: Path) -> pd.DataFrame:
    """Load a checkpoint JSONL (handles newline-delimited, concatenated, or mixed)."""
    raw = filepath.read_text(encoding="utf-8")
    lines = raw.strip().splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.extend(_parse_concatenated_json(line))

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def discover_checkpoints() -> dict[str, dict[str, pd.DataFrame]]:
    """Return {task: {model: DataFrame}} for all available checkpoints."""
    data = {}
    for task, task_path in TASK_PATHS.items():
        if not task_path.exists():
            continue
        data[task] = {}
        for model_dir in sorted(task_path.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            for f in model_dir.iterdir():
                if f.suffix == ".jsonl" and "checkpoint" in f.name.lower():
                    df = load_checkpoint(f)
                    if df.empty:
                        continue
                    data[task][model_name] = df
                    break
    return data


def load_raft_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Return {task: {model_key: DataFrame}} from per-subtask train_results.json files."""
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for task in RAFT_TASKS:
        data[task] = {}
        for model_key in MODEL_DISPLAY:
            path = RAFT_DIR / model_key / task / "train_results.json"
            if not path.exists():
                continue
            with open(path) as f:
                records = json.load(f)
            df = pd.DataFrame(records)
            # Normalise column names to match the rest of the script
            if "ground_truth" not in df.columns and "label" in df.columns:
                df = df.rename(columns={"label": "ground_truth"})
            data[task][model_key] = df
    return data


# ── Metric Computation ────────────────────────────────────────────────────────

def accuracy(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return np.nan
    return df["is_correct"].mean()


def unknown_rate(df: pd.DataFrame) -> float:
    if "is_unknown" not in df.columns or len(df) == 0:
        return np.nan
    return df["is_unknown"].mean()


def per_class_accuracy(df: pd.DataFrame, label_col: str = "ground_truth") -> dict:
    if label_col not in df.columns:
        return {}
    result = {}
    for cls in sorted(df[label_col].unique(), key=str):
        subset = df[df[label_col] == cls]
        result[str(cls)] = subset["is_correct"].mean() if len(subset) > 0 else np.nan
    return result


def macro_f1(df: pd.DataFrame, label_col: str = "ground_truth", pred_col: str = "prediction") -> float:
    if label_col not in df.columns or pred_col not in df.columns or len(df) == 0:
        return np.nan
    classes = sorted(df[label_col].unique(), key=str)
    f1s = []
    for cls in classes:
        tp = ((df[label_col] == cls) & (df[pred_col] == cls)).sum()
        fp = ((df[label_col] != cls) & (df[pred_col] == cls)).sum()
        fn = ((df[label_col] == cls) & (df[pred_col] != cls)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# ── Bootstrap Sampling ────────────────────────────────────────────────────────

def bootstrap_accuracy(
    df: pd.DataFrame,
    sample_size: int,
    n_bootstrap: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw `n_bootstrap` random subsets of size `sample_size` and return accuracies."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(df)
    if sample_size > n:
        return np.full(n_bootstrap, np.nan)
    accs = np.empty(n_bootstrap)
    is_correct = df["is_correct"].values
    for i in range(n_bootstrap):
        idxs = rng.choice(n, size=sample_size, replace=False)
        accs[i] = is_correct[idxs].mean()
    return accs


def bootstrap_metric(
    df: pd.DataFrame,
    metric_fn,
    sample_size: int,
    n_bootstrap: int = N_BOOTSTRAP,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generic bootstrap for any metric function taking a DataFrame."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(df)
    if sample_size > n:
        return np.full(n_bootstrap, np.nan)
    vals = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idxs = rng.choice(n, size=sample_size, replace=False)
        vals[i] = metric_fn(df.iloc[idxs])
    return vals


# ── Main Analysis ─────────────────────────────────────────────────────────────

def run_analysis():
    print("Discovering checkpoints …")
    data = discover_checkpoints()
    print(f"Found tasks: {list(data.keys())}")
    for task, models in data.items():
        print(f"  {task}: {list(models.keys())} "
              f"({', '.join(str(len(v)) + ' rows' for v in models.values())})")

    rng = np.random.default_rng(SEED)

    # ── 1. Build summary table ────────────────────────────────────────────
    rows = []
    boot_cache: dict[tuple, np.ndarray] = {}

    for task, models in data.items():
        for model, df in models.items():
            full_acc = accuracy(df)
            full_unk = unknown_rate(df)
            full_f1 = macro_f1(df)
            n_total = len(df)

            rows.append({
                "task": task,
                "model": model,
                "n": "full",
                "n_total": n_total,
                "accuracy": full_acc,
                "accuracy_std": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "unknown_rate": full_unk,
                "macro_f1": full_f1,
            })

            for sz in SAMPLE_SIZES:
                if sz > n_total:
                    continue
                accs = bootstrap_accuracy(df, sz, N_BOOTSTRAP, rng)
                boot_cache[(task, model, sz)] = accs
                f1s = bootstrap_metric(df, macro_f1, sz, N_BOOTSTRAP, rng)
                lo, hi = np.percentile(accs, [2.5, 97.5])
                rows.append({
                    "task": task,
                    "model": model,
                    "n": sz,
                    "n_total": n_total,
                    "accuracy": accs.mean(),
                    "accuracy_std": accs.std(),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "unknown_rate": full_unk,
                    "macro_f1": f1s.mean(),
                })

    summary = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print("SUMMARY TABLE — Accuracy (mean ± std) over bootstrap samples")
    print("=" * 90)
    for task in data:
        print(f"\n{'─' * 90}")
        print(f"  Task: {TASK_DISPLAY.get(task, task)}   (full dataset size shown in parentheses)")
        print(f"{'─' * 90}")
        sub = summary[summary["task"] == task]
        for model in sub["model"].unique():
            msub = sub[sub["model"] == model]
            full_row = msub[msub["n"] == "full"].iloc[0]
            print(f"\n  {MODEL_DISPLAY.get(model, model)}  "
                  f"[N={full_row['n_total']}  |  full acc={full_row['accuracy']:.4f}  |  "
                  f"macro-F1={full_row['macro_f1']:.4f}  |  "
                  f"unknown={full_row['unknown_rate']:.4f}]")
            sample_rows = msub[msub["n"] != "full"].sort_values("n")
            for _, r in sample_rows.iterrows():
                print(f"    n={int(r['n']):>4d}  →  "
                      f"acc={r['accuracy']:.4f} ± {r['accuracy_std']:.4f}  "
                      f"95% CI [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  "
                      f"macro-F1={r['macro_f1']:.4f}")

    # Save summary CSV
    summary.to_csv(OUTPUT_DIR / "sampling_summary.csv", index=False)
    print(f"\nSaved summary CSV → {OUTPUT_DIR / 'sampling_summary.csv'}")

    # ── 2. Convergence Curves ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Accuracy Convergence: Sample Size to Full Dataset", fontsize=15, fontweight="bold")

    for ax, task in zip(axes.flat, TASK_PATHS.keys()):
        if task not in data:
            ax.set_visible(False)
            continue
        models = data[task]
        for model, df in models.items():
            full_acc = accuracy(df)
            sub = summary[(summary["task"] == task) & (summary["model"] == model) & (summary["n"] != "full")]
            if sub.empty:
                continue
            ns = sub["n"].astype(int).values
            means = sub["accuracy"].values
            ci_lo = sub["ci_lo"].values
            ci_hi = sub["ci_hi"].values
            color = PALETTE.get(model, "#999999")
            ax.plot(ns, means, "o-", label=MODEL_DISPLAY.get(model, model), color=color, linewidth=2)
            ax.fill_between(ns, ci_lo, ci_hi, alpha=0.15, color=color)
            ax.axhline(full_acc, color=color, linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=13)
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(SAMPLE_SIZES)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "convergence_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved convergence curves → {OUTPUT_DIR / 'convergence_curves.png'}")

    # ── 3. Bootstrap Distribution Violins ─────────────────────────────────
    for task in data:
        models_in_task = list(data[task].keys())
        fig, axes = plt.subplots(1, len(models_in_task), figsize=(6 * len(models_in_task), 5), squeeze=False)
        fig.suptitle(f"Bootstrap Accuracy Distributions — {TASK_DISPLAY.get(task, task)}",
                     fontsize=14, fontweight="bold")

        for ax, model in zip(axes[0], models_in_task):
            plot_data = []
            labels = []
            for sz in SAMPLE_SIZES:
                key = (task, model, sz)
                if key in boot_cache:
                    plot_data.append(boot_cache[key])
                    labels.append(str(sz))

            if not plot_data:
                ax.set_visible(False)
                continue

            parts = ax.violinplot(plot_data, positions=range(len(labels)), showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(PALETTE.get(model, "#999999"))
                pc.set_alpha(0.6)

            full_acc = accuracy(data[task][model])
            ax.axhline(full_acc, color="black", linestyle="--", alpha=0.6, label=f"Full acc = {full_acc:.3f}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"n={l}" for l in labels])
            ax.set_ylabel("Accuracy")
            ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=12)
            ax.legend(fontsize=9)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(OUTPUT_DIR / f"bootstrap_violins_{task}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved bootstrap violins → {OUTPUT_DIR / f'bootstrap_violins_{task}.png'}")

    # ── 4. Per-Class Accuracy Heatmaps (multi-class tasks) ────────────────
    multiclass_tasks = {
        "financial_phrase_bank": "ground_truth",
        "rte": "ground_truth",
    }

    for task, label_col in multiclass_tasks.items():
        if task not in data:
            continue
        models_in_task = list(data[task].keys())
        classes = sorted(
            set().union(*(set(df[label_col].unique()) for df in data[task].values())),
            key=str,
        )
        class_labels = [str(c) for c in classes]

        fig, axes = plt.subplots(1, len(models_in_task), figsize=(5 * len(models_in_task), 4), squeeze=False)
        fig.suptitle(f"Per-Class Accuracy — {TASK_DISPLAY.get(task, task)}", fontsize=14, fontweight="bold")

        for ax, model in zip(axes[0], models_in_task):
            df = data[task][model]
            pca = per_class_accuracy(df, label_col)
            vals = [pca.get(c, np.nan) for c in class_labels]
            matrix = np.array(vals).reshape(1, -1)
            im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            for j, v in enumerate(vals):
                ax.text(j, 0, f"{v:.3f}", ha="center", va="center", fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(class_labels)))
            ax.set_xticklabels(class_labels)
            ax.set_yticks([0])
            ax.set_yticklabels([MODEL_DISPLAY.get(model, model)])
            ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=11)
            fig.colorbar(im, ax=ax, shrink=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(OUTPUT_DIR / f"per_class_accuracy_{task}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved per-class heatmap → {OUTPUT_DIR / f'per_class_accuracy_{task}.png'}")

    # ── 5. Accuracy Deviation from Full Dataset ───────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Accuracy Deviation from Full-Dataset Accuracy", fontsize=15, fontweight="bold")

    for ax, task in zip(axes.flat, TASK_PATHS.keys()):
        if task not in data:
            ax.set_visible(False)
            continue
        for model, df in data[task].items():
            full_acc = accuracy(df)
            sub = summary[(summary["task"] == task) & (summary["model"] == model) & (summary["n"] != "full")]
            if sub.empty:
                continue
            ns = sub["n"].astype(int).values
            devs = sub["accuracy"].values - full_acc
            ci_lo_dev = sub["ci_lo"].values - full_acc
            ci_hi_dev = sub["ci_hi"].values - full_acc
            color = PALETTE.get(model, "#999999")
            ax.plot(ns, devs, "o-", label=MODEL_DISPLAY.get(model, model), color=color, linewidth=2)
            ax.fill_between(ns, ci_lo_dev, ci_hi_dev, alpha=0.15, color=color)

        ax.axhline(0, color="black", linestyle="-", alpha=0.3)
        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=13)
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Δ Accuracy (sample − full)")
        ax.set_xticks(SAMPLE_SIZES)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "accuracy_deviation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy deviation plot → {OUTPUT_DIR / 'accuracy_deviation.png'}")

    # ── 6. Std-Dev Decay Plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Bootstrap Std-Dev Decay with Sample Size", fontsize=14, fontweight="bold")

    markers = {"financial_phrase_bank": "o", "gsm8k": "s", "boolq": "^", "rte": "D"}
    for task in data:
        for model in data[task]:
            sub = summary[(summary["task"] == task) & (summary["model"] == model) & (summary["n"] != "full")]
            if sub.empty:
                continue
            ns = sub["n"].astype(int).values
            stds = sub["accuracy_std"].values
            color = PALETTE.get(model, "#999999")
            marker = markers.get(task, "x")
            ax.plot(ns, stds, marker=marker, linestyle="-", color=color, alpha=0.8,
                    label=f"{TASK_DISPLAY.get(task, task)} / {MODEL_DISPLAY.get(model, model)}")

    # Theoretical 1/√n reference curve (scaled to median of first stds)
    first_stds = [
        summary[(summary["task"] == t) & (summary["n"] == SAMPLE_SIZES[0])]["accuracy_std"].mean()
        for t in data
    ]
    ref_scale = np.nanmedian(first_stds) * np.sqrt(SAMPLE_SIZES[0])
    ref_ns = np.array(SAMPLE_SIZES)
    ax.plot(ref_ns, ref_scale / np.sqrt(ref_ns), "k--", alpha=0.4, linewidth=2, label="~1/√n reference")

    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Std-Dev of Bootstrap Accuracy")
    ax.set_xticks(SAMPLE_SIZES)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "stddev_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved std-dev decay plot → {OUTPUT_DIR / 'stddev_decay.png'}")

    # ── 7. Unknown Rate Bar Chart ─────────────────────────────────────────
    unk_rows = summary[summary["n"] == "full"][["task", "model", "unknown_rate"]].copy()
    if unk_rows["unknown_rate"].notna().any():
        fig, ax = plt.subplots(figsize=(10, 5))
        unk_pivot = unk_rows.pivot(index="task", columns="model", values="unknown_rate")
        unk_pivot.index = [TASK_DISPLAY.get(t, t) for t in unk_pivot.index]
        unk_pivot.columns = [MODEL_DISPLAY.get(m, m) for m in unk_pivot.columns]
        unk_pivot.plot.bar(ax=ax, color=[PALETTE.get(k, "#999") for k in data[list(data.keys())[0]].keys()])
        ax.set_title("Unknown / Unparseable Response Rate (Full Dataset)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Unknown Rate")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9)
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "unknown_rate.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved unknown rate chart → {OUTPUT_DIR / 'unknown_rate.png'}")

    # ── 8. Macro-F1 Convergence ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Macro-F1 Convergence: Sample Size to Full Dataset", fontsize=15, fontweight="bold")

    for ax, task in zip(axes.flat, TASK_PATHS.keys()):
        if task not in data:
            ax.set_visible(False)
            continue
        for model, df in data[task].items():
            full_f1_val = macro_f1(df)
            sub = summary[(summary["task"] == task) & (summary["model"] == model) & (summary["n"] != "full")]
            if sub.empty:
                continue
            ns = sub["n"].astype(int).values
            f1_means = sub["macro_f1"].values
            color = PALETTE.get(model, "#999999")
            ax.plot(ns, f1_means, "o-", label=MODEL_DISPLAY.get(model, model), color=color, linewidth=2)
            ax.axhline(full_f1_val, color=color, linestyle="--", alpha=0.5, linewidth=1)

        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=13)
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Macro-F1")
        ax.set_xticks(SAMPLE_SIZES)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "macro_f1_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved macro-F1 convergence → {OUTPUT_DIR / 'macro_f1_convergence.png'}")

    # ── 9. Macro-F1 Convergence (Average of Three Models) ──────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Macro-F1 Convergence (Avg of 3 Models): Sample Size to Full Dataset",
                 fontsize=15, fontweight="bold")

    for ax, task in zip(axes.flat, TASK_PATHS.keys()):
        if task not in data:
            ax.set_visible(False)
            continue
        # Sample sizes: average macro_f1 across models for each n
        sub = summary[(summary["task"] == task) & (summary["n"] != "full")]
        if sub.empty:
            continue
        agg = sub.groupby("n", sort=True).agg(macro_f1=("macro_f1", "mean")).reset_index()
        ns = agg["n"].astype(int).values
        f1_means = agg["macro_f1"].values
        # Full-dataset average macro-F1 across models
        full_sub = summary[(summary["task"] == task) & (summary["n"] == "full")]
        full_avg_f1 = full_sub["macro_f1"].mean() if not full_sub.empty else np.nan

        ax.plot(ns, f1_means, "o-", color="#2E86AB", linewidth=2, label="Avg macro-F1")
        if not np.isnan(full_avg_f1):
            ax.axhline(full_avg_f1, color="#2E86AB", linestyle="--", alpha=0.5, linewidth=1,
                       label=f"Full avg = {full_avg_f1:.3f}")

        ax.set_title(TASK_DISPLAY.get(task, task), fontsize=13)
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Macro-F1 (avg)")
        ax.set_xticks(SAMPLE_SIZES)
        ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "macro_f1_convergence_avg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved macro-F1 convergence (avg) → {OUTPUT_DIR / 'macro_f1_convergence_avg.png'}")

    # ── 10. Side-by-Side Full Accuracy Comparison ──────────────────────────
    full_rows = summary[summary["n"] == "full"][["task", "model", "accuracy"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = full_rows.pivot(index="task", columns="model", values="accuracy")
    pivot.index = [TASK_DISPLAY.get(t, t) for t in pivot.index]
    pivot.columns = [MODEL_DISPLAY.get(m, m) for m in pivot.columns]
    pivot.plot.bar(ax=ax, color=[PALETTE.get(k, "#999") for k in
                                  sorted(set(m for models in data.values() for m in models))])
    ax.set_title("Full-Dataset Accuracy by Model × Task", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "full_accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved full accuracy comparison → {OUTPUT_DIR / 'full_accuracy_comparison.png'}")

    print("\n✓ Analysis complete. All outputs saved to:", OUTPUT_DIR)
    return summary, data, boot_cache


def run_raft_analysis() -> pd.DataFrame:
    """Bootstrap sampling analysis for RAFT train_results.json (50 examples / subtask)."""
    print("\n" + "=" * 90)
    print("RAFT TRAIN-SPLIT SAMPLING ANALYSIS")
    print("=" * 90)

    raft_data = load_raft_data()
    found = {t: list(m.keys()) for t, m in raft_data.items() if m}
    print(f"Loaded {sum(len(v) for v in found.values())} files across {len(found)} tasks")
    for task, models in found.items():
        print(f"  {RAFT_TASK_DISPLAY.get(task, task)}: {[MODEL_DISPLAY.get(m, m) for m in models]}")

    rng = np.random.default_rng(SEED)

    # ── 1. Build summary table ─────────────────────────────────────────────
    rows = []
    boot_cache: dict[tuple, np.ndarray] = {}

    for task in RAFT_TASKS:
        models = raft_data.get(task, {})
        for model_key, df in models.items():
            full_acc = accuracy(df)
            full_unk = unknown_rate(df)
            full_f1  = macro_f1(df)
            n_total  = len(df)

            rows.append({
                "task": task, "model": model_key, "n": "full",
                "n_total": n_total, "accuracy": full_acc, "accuracy_std": np.nan,
                "ci_lo": np.nan, "ci_hi": np.nan,
                "unknown_rate": full_unk, "macro_f1": full_f1,
            })

            for sz in RAFT_SAMPLE_SIZES:
                if sz > n_total:
                    continue
                accs = bootstrap_accuracy(df, sz, N_BOOTSTRAP, rng)
                f1s  = bootstrap_metric(df, macro_f1, sz, N_BOOTSTRAP, rng)
                boot_cache[(task, model_key, sz)] = accs
                lo, hi = np.percentile(accs, [2.5, 97.5])
                rows.append({
                    "task": task, "model": model_key, "n": sz,
                    "n_total": n_total, "accuracy": accs.mean(), "accuracy_std": accs.std(),
                    "ci_lo": lo, "ci_hi": hi,
                    "unknown_rate": full_unk, "macro_f1": f1s.mean(),
                })

    summary = pd.DataFrame(rows)

    print("\nRAFT sampling summary (accuracy mean ± std):")
    for task in RAFT_TASKS:
        sub = summary[summary["task"] == task]
        if sub.empty:
            continue
        print(f"\n  {RAFT_TASK_DISPLAY.get(task, task)}")
        for model_key in sub["model"].unique():
            msub = sub[sub["model"] == model_key]
            full_row = msub[msub["n"] == "full"].iloc[0]
            print(f"    {MODEL_DISPLAY.get(model_key, model_key):12s}  "
                  f"full acc={full_row['accuracy']:.3f}  macro-F1={full_row['macro_f1']:.3f}")
            for _, r in msub[msub["n"] != "full"].sort_values("n").iterrows():
                print(f"      n={int(r['n']):>2d}  acc={r['accuracy']:.3f} ± {r['accuracy_std']:.3f}  "
                      f"95% CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")

    summary.to_csv(OUTPUT_DIR / "raft_sampling_summary.csv", index=False)
    print(f"\nSaved → {OUTPUT_DIR / 'raft_sampling_summary.csv'}")

    # ── 2. Convergence curves (2 rows × 5 cols, one panel per RAFT subtask) ─
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharey=False)
    fig.suptitle("RAFT Train-Split: Accuracy Convergence by Sample Size (n=50 total)",
                 fontsize=15, fontweight="bold")

    for ax, task in zip(axes.flat, RAFT_TASKS):
        models = raft_data.get(task, {})
        for model_key, df in models.items():
            full_acc = accuracy(df)
            sub = summary[(summary["task"] == task) & (summary["model"] == model_key) & (summary["n"] != "full")]
            if sub.empty:
                continue
            ns    = sub["n"].astype(int).values
            means = sub["accuracy"].values
            ci_lo = sub["ci_lo"].values
            ci_hi = sub["ci_hi"].values
            color = PALETTE.get(model_key, "#999999")
            ax.plot(ns, means, "o-", label=MODEL_DISPLAY.get(model_key, model_key),
                    color=color, linewidth=2)
            ax.fill_between(ns, ci_lo, ci_hi, alpha=0.15, color=color)
            ax.axhline(full_acc, color=color, linestyle="--", alpha=0.45, linewidth=1)

        ax.set_title(RAFT_TASK_DISPLAY.get(task, task), fontsize=10, fontweight="bold")
        ax.set_xlabel("Sample size (n)", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_xticks(RAFT_SAMPLE_SIZES)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "raft_convergence_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_DIR / 'raft_convergence_curves.png'}")

    # ── 3. Bootstrap violin plots per subtask ──────────────────────────────
    for task in RAFT_TASKS:
        models_in_task = list(raft_data.get(task, {}).keys())
        if not models_in_task:
            continue
        fig, axes = plt.subplots(1, len(models_in_task),
                                 figsize=(5 * len(models_in_task), 5), squeeze=False)
        fig.suptitle(f"Bootstrap Accuracy Distributions — {RAFT_TASK_DISPLAY.get(task, task)} (RAFT)",
                     fontsize=13, fontweight="bold")

        for ax, model_key in zip(axes[0], models_in_task):
            plot_data, labels = [], []
            for sz in RAFT_SAMPLE_SIZES:
                key = (task, model_key, sz)
                if key in boot_cache:
                    plot_data.append(boot_cache[key])
                    labels.append(str(sz))
            if not plot_data:
                ax.set_visible(False)
                continue
            parts = ax.violinplot(plot_data, positions=range(len(labels)),
                                  showmeans=True, showmedians=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(PALETTE.get(model_key, "#999999"))
                pc.set_alpha(0.6)
            full_acc = accuracy(raft_data[task][model_key])
            ax.axhline(full_acc, color="black", linestyle="--", alpha=0.6,
                       label=f"Full (n=50) = {full_acc:.3f}")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([f"n={l}" for l in labels])
            ax.set_ylabel("Accuracy")
            ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=11)
            ax.legend(fontsize=8)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(OUTPUT_DIR / f"raft_bootstrap_violins_{task}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved bootstrap violin plots → {OUTPUT_DIR}/raft_bootstrap_violins_*.png")

    # ── 4. Std-dev decay across all RAFT subtasks ──────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("RAFT: Bootstrap Std-Dev Decay with Sample Size", fontsize=13, fontweight="bold")

    for task in RAFT_TASKS:
        for model_key in raft_data.get(task, {}):
            sub = summary[(summary["task"] == task) & (summary["model"] == model_key) & (summary["n"] != "full")]
            if sub.empty:
                continue
            ns   = sub["n"].astype(int).values
            stds = sub["accuracy_std"].values
            color = PALETTE.get(model_key, "#999999")
            ax.plot(ns, stds, "o-", color=color, alpha=0.55, linewidth=1,
                    label=f"{RAFT_TASK_DISPLAY.get(task, task)} / {MODEL_DISPLAY.get(model_key, model_key)}")

    # Theoretical 1/√n reference
    first_stds = [
        summary[(summary["task"] == t) & (summary["n"] == RAFT_SAMPLE_SIZES[0])]["accuracy_std"].mean()
        for t in RAFT_TASKS if t in found
    ]
    ref_scale = np.nanmedian(first_stds) * np.sqrt(RAFT_SAMPLE_SIZES[0])
    ref_ns = np.array(RAFT_SAMPLE_SIZES)
    ax.plot(ref_ns, ref_scale / np.sqrt(ref_ns), "k--", alpha=0.5, linewidth=2, label="~1/√n reference")

    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Std-Dev of Bootstrap Accuracy")
    ax.set_xticks(RAFT_SAMPLE_SIZES)
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "raft_stddev_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_DIR / 'raft_stddev_decay.png'}")

    # ── 5. Mean accuracy across subtasks vs. sample size ──────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("RAFT: Mean Accuracy Across All 10 Subtasks vs. Sample Size",
                 fontsize=13, fontweight="bold")

    for model_key in MODEL_DISPLAY:
        sub = summary[(summary["model"] == model_key) & (summary["n"] != "full")]
        if sub.empty:
            continue
        agg = sub.groupby("n")["accuracy"].mean().reset_index()
        full_mean = summary[(summary["model"] == model_key) & (summary["n"] == "full")]["accuracy"].mean()
        ns = agg["n"].astype(int).values
        color = PALETTE.get(model_key, "#999999")
        ax.plot(ns, agg["accuracy"].values, "o-", label=MODEL_DISPLAY[model_key],
                color=color, linewidth=2)
        ax.axhline(full_mean, color=color, linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Mean Accuracy (10 subtasks)")
    ax.set_xticks(RAFT_SAMPLE_SIZES)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "raft_mean_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUTPUT_DIR / 'raft_mean_convergence.png'}")

    print("\n✓ RAFT analysis complete.")
    return summary


if __name__ == "__main__":
    summary, data, boot_cache = run_analysis()
    run_raft_analysis()
