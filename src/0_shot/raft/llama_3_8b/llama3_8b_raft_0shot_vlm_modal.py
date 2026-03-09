"""
RAFT 0-shot evaluation — Llama-3-8B-Instruct + vLLM on Modal.

Setup (one-time):
  1. pip install modal
  2. modal setup          # authenticate with Modal

HuggingFace token — pick ONE of these three options:

  Option A (recommended): store it as a persistent Modal secret
    modal secret create huggingface-secret HF_TOKEN=hf_yourTokenHere

  Option B: pass it inline on the command line (no secret needed)
    HF_TOKEN=hf_yourTokenHere modal run llama3_8b_raft_0shot_vlm_modal.py

  Option C: put it in a .env file in this directory
    echo "HF_TOKEN=hf_yourTokenHere" > .env
    modal run llama3_8b_raft_0shot_vlm_modal.py

  Get a free token at: https://huggingface.co/settings/tokens
  (needs "Read" access; request Llama-3 access at meta-llama/Meta-Llama-3-8B-Instruct)

Run:
  modal run llama3_8b_raft_0shot_vlm_modal.py

Download results after the run:
  modal volume get raft-llama3-results /results ./local_results
"""

import csv
import json
import os
import time

import modal

# ── Constants ────────────────────────────────────────────────────────────────
MINUTES = 60
HOURS   = 60 * MINUTES

MODEL_NAME     = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_NEW_TOKENS = 32
RESULTS_DIR    = "/results"

# ── Modal image ──────────────────────────────────────────────────────────────
# Modal provides CUDA automatically when a GPU is requested.
# datasets==3.6.0 is the last version that supports the ought/raft loading script.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "transformers",
        "accelerate",
        "datasets==3.6.0",
        "pandas",
        "scikit-learn",
        "huggingface_hub",
        "protobuf==5.29.5",
    )
)

app = modal.App("raft-llama3-8b-eval", image=image)


def _hf_secret() -> list:
    """
    Return whichever HuggingFace secret source is available.
    Priority: Modal named secret > local .env file > HF_TOKEN env var.
    All three ultimately inject HF_TOKEN into the container environment.
    """
    import pathlib

    # Option A: named Modal secret (modal secret create huggingface-secret HF_TOKEN=...)
    try:
        return [modal.Secret.from_name("huggingface-secret")]
    except Exception:
        pass

    # Option C: .env file next to this script
    env_path = pathlib.Path(__file__).parent / ".env"
    if env_path.exists():
        return [modal.Secret.from_dotenv(str(env_path))]

    # Option B: HF_TOKEN already set in the shell — forward it directly
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        return [modal.Secret.from_dict({"HF_TOKEN": hf_token})]

    raise RuntimeError(
        "\n\nNo HuggingFace token found. Llama-3 is a gated model and requires one.\n"
        "Fix: run with your token inline:\n\n"
        "  HF_TOKEN=hf_yourTokenHere modal run llama3_8b_raft_0shot_vlm_modal.py\n\n"
        "Get a free Read token at: https://huggingface.co/settings/tokens\n"
        "Accept the Llama-3 license at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct\n"
    )

# Persistent volume — survives after the function ends.
# Download with: modal volume get raft-llama3-results /results ./local_results
results_vol = modal.Volume.from_name("raft-llama3-results", create_if_missing=True)

# ── RAFT task metadata ────────────────────────────────────────────────────────
# Label order matches the official RAFT integer labels (1-indexed).
# label_names[0] -> RAFT label 1, label_names[1] -> RAFT label 2, etc.
# Source: https://huggingface.co/datasets/ought/raft/raw/main/raft.py
RAFT_TASKS = {
    "ade_corpus_v2": {
        "text_fields": ["Sentence"],
        "label_names": ["ADE-related", "not ADE-related"],
        "description": "Classify whether the sentence describes an Adverse Drug Effect (ADE).",
    },
    "banking_77": {
        "text_fields": ["Query"],
        "label_names": [
            "refund not showing up", "activate my card", "age limit",
            "apple pay or google pay", "atm support", "automatic top up",
            "balance not updated after bank transfer",
            "balance not updated after cheque or cash deposit",
            "beneficiary not allowed", "cancel transfer", "card about to expire",
            "card acceptance", "card arrival", "card delivery estimate",
            "card linking", "card not working", "card payment fee charged",
            "card payment not recognised", "card payment wrong exchange rate",
            "card swallowed", "cash withdrawal charge",
            "cash withdrawal not recognised", "change pin", "compromised card",
            "contactless not working", "country support", "declined card payment",
            "declined cash withdrawal", "declined transfer",
            "direct debit payment not recognised", "disposable card limits",
            "edit personal details", "exchange charge", "exchange rate",
            "exchange via app", "extra charge on statement", "failed transfer",
            "fiat currency support", "get disposable virtual card",
            "get physical card", "getting spare card", "getting virtual card",
            "lost or stolen card", "lost or stolen phone", "order physical card",
            "passcode forgotten", "pending card payment", "pending cash withdrawal",
            "pending top up", "pending transfer", "pin blocked", "receiving money",
            "request refund", "reverted card payment",
            "supported cards and currencies", "terminate account",
            "top up by bank transfer charge", "top up by card charge",
            "top up by cash or cheque", "top up failed", "top up limits",
            "top up reverted", "topping up by card", "transaction charged twice",
            "transfer fee charged", "transfer into account",
            "transfer not received by recipient", "transfer timing",
            "unable to verify identity", "verify my identity",
            "verify source of funds", "verify top up", "virtual card not working",
            "visa or mastercard", "why verify identity",
            "wrong amount of cash received", "wrong exchange rate for cash withdrawal",
        ],
        "description": "Classify the customer service query into one of 77 banking intent categories.",
    },
    "neurips_impact_statement_risks": {
        "text_fields": ["Paper title", "Impact statement"],
        "label_names": ["doesn't mention a harmful application", "mentions a harmful application"],
        "description": "Does the NeurIPS impact statement mention a harmful application of the research?",
    },
    "one_stop_english": {
        "text_fields": ["Article"],
        "label_names": ["advanced", "elementary", "intermediate"],
        "description": "Classify the reading level of the article: advanced, elementary, or intermediate.",
    },
    "overruling": {
        "text_fields": ["Sentence"],
        "label_names": ["not overruling", "overruling"],
        "description": "Does this legal sentence overrule a previous legal rule or case?",
    },
    "semiconductor_org_types": {
        "text_fields": ["Paper title", "Organization name"],
        "label_names": ["company", "research institute", "university"],
        "description": "Classify the semiconductor organization type.",
    },
    "systematic_review_inclusion": {
        "text_fields": ["Title", "Abstract"],
        "label_names": ["included", "not included"],
        "description": "Should this paper be included in a systematic review based on its title and abstract?",
    },
    "tai_safety_research": {
        "text_fields": ["Title", "Abstract Note"],
        "label_names": ["TAI safety research", "not TAI safety research"],
        "description": "Is this paper related to transformative AI (TAI) safety research?",
    },
    "terms_of_service": {
        "text_fields": ["Sentence"],
        "label_names": ["not potentially unfair", "potentially unfair"],
        "description": "Does this Terms of Service clause contain a potentially unfair policy?",
    },
    "tweet_eval_hate": {
        "text_fields": ["Tweet"],
        "label_names": ["hate speech", "not hate speech"],
        "description": "Does this tweet contain hate speech?",
    },
    "twitter_complaints": {
        "text_fields": ["Tweet text"],
        "label_names": ["complaint", "no complaint"],
        "description": "Is this tweet a customer complaint?",
    },
}

ALL_TASKS = list(RAFT_TASKS.keys())


# ── Prompt helpers ────────────────────────────────────────────────────────────

def build_system_prompt(task: str) -> str:
    meta = RAFT_TASKS[task]
    label_list = ", ".join(f'"{l}"' for l in meta["label_names"])
    return (
        f"You are a text classification assistant. "
        f"{meta['description']} "
        f"Answer with exactly one of the following labels: {label_list}. "
        f"Output only the label, nothing else."
    )


def build_user_content(task: str, example: dict) -> str:
    meta = RAFT_TASKS[task]
    parts = []
    for field in meta["text_fields"]:
        value = str(example.get(field, "") or "").strip()
        if value:
            parts.append(f"{field}: {value}")
    parts.append("Label:")
    return "\n".join(parts)


def build_prompt(task: str, example: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt(task)},
        {"role": "user",   "content": build_user_content(task, example)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Label extractor ───────────────────────────────────────────────────────────

def extract_label(task: str, response: str):
    """
    Map raw model response to a RAFT integer label (1-indexed).
    Returns None if no match found.
    Sorts candidate labels longest-first to avoid partial matches
    (e.g. 'not hate speech' before 'hate speech').
    """
    cleaned = response.strip().lower()
    label_names = RAFT_TASKS[task]["label_names"]
    sorted_labels = sorted(enumerate(label_names), key=lambda x: -len(x[1]))
    for idx, name in sorted_labels:
        if name.lower() in cleaned:
            return idx + 1  # RAFT is 1-indexed
    return None


# ── Modal function ────────────────────────────────────────────────────────────

@app.function(
    gpu="L4",
    timeout=3 * HOURS,
    volumes={RESULTS_DIR: results_vol},
    secrets=_hf_secret(),
)
def evaluate_raft(tasks_to_eval: list[str] = ALL_TASKS) -> list[dict]:
    import pandas as pd
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, classification_report
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # ── Load datasets ─────────────────────────────────────────────────────────
    task_datasets: dict = {}
    for task in tasks_to_eval:
        print(f"Loading {task}...")
        train_ds = load_dataset("ought/raft", task, split="train", trust_remote_code=True)
        test_ds  = load_dataset("ought/raft", task, split="test",  trust_remote_code=True)
        task_datasets[task] = {"train": train_ds, "test": test_ds}
        print(f"  train: {len(train_ds)} | test: {len(test_ds)} | cols: {train_ds.column_names}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=hf_token)

    print("Loading Llama-3-8B with vLLM...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",          # Llama-3 native dtype; avoids float16 cast crash
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        enforce_eager=False,
        tokenizer_mode="auto",
        token=hf_token,
    )
    sampling_params = SamplingParams(temperature=0, top_k=20, max_tokens=MAX_NEW_TOKENS)
    print("Model loaded!\n")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    summary_rows = []

    for task in tasks_to_eval:
        print(f"\n{'='*60}\nTask: {task}\n{'='*60}")

        train_ds = task_datasets[task]["train"]
        test_ds  = task_datasets[task]["test"]
        meta     = RAFT_TASKS[task]

        train_prompts = [build_prompt(task, ex, tokenizer) for ex in train_ds]
        test_prompts  = [build_prompt(task, ex, tokenizer) for ex in test_ds]
        all_prompts   = train_prompts + test_prompts
        print(f"  {len(train_prompts)} train + {len(test_prompts)} test = {len(all_prompts)} prompts")

        t0 = time.time()
        outputs    = llm.generate(all_prompts, sampling_params)
        gen_time   = time.time() - t0
        total_tok  = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tok / gen_time if gen_time > 0 else 0.0
        print(f"  Time: {gen_time/60:.1f} min | Tokens: {total_tok:,} | {throughput:.0f} tok/s")

        # Score train
        train_results = []
        for i, (ex, out) in enumerate(zip(train_ds, outputs[:len(train_prompts)])):
            response = out.outputs[0].text
            gt   = int(ex["Label"])
            pred = extract_label(task, response)
            train_results.append({
                "id":           ex.get("ID", i),
                "ground_truth": gt,
                "prediction":   pred,
                "raw_response": response.strip(),
                "is_correct":   (pred == gt) if pred is not None else False,
                "is_unknown":   pred is None,
            })

        correct  = sum(1 for r in train_results if r["is_correct"])
        unknown  = sum(1 for r in train_results if r["is_unknown"])
        accuracy = correct / len(train_results)

        known_pairs = [(r["ground_truth"], r["prediction"])
                       for r in train_results if r["prediction"] is not None]
        acc_known = accuracy_score(
            [p[0] for p in known_pairs], [p[1] for p in known_pairs]
        ) if known_pairs else 0.0

        print(f"\n  Accuracy: {accuracy:.4f} (known-only: {acc_known:.4f}, unknown: {unknown})")

        gt_list   = [r["ground_truth"] for r in train_results if r["prediction"] is not None]
        pred_list = [r["prediction"]   for r in train_results if r["prediction"] is not None]
        if pred_list:
            present      = sorted(set(gt_list) | set(pred_list))
            present_names = [meta["label_names"][p - 1] for p in present]
            print(classification_report(gt_list, pred_list,
                                        labels=present, target_names=present_names))

        # Collect test predictions
        test_predictions = []
        for i, (ex, out) in enumerate(zip(test_ds, outputs[len(train_prompts):])):
            response  = out.outputs[0].text
            pred      = extract_label(task, response)
            pred_safe = pred if pred is not None else 1
            test_predictions.append({
                "ID":           ex.get("ID", i),
                "Label":        pred_safe,
                "raw_response": response.strip(),
            })

        # Save per-task results to volume
        task_dir = os.path.join(RESULTS_DIR, task)
        os.makedirs(task_dir, exist_ok=True)

        with open(os.path.join(task_dir, "train_results.json"), "w") as f:
            json.dump(train_results, f, indent=2)

        submission_path = os.path.join(task_dir, "test_predictions.csv")
        with open(submission_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ID", "Label"])
            writer.writeheader()
            for row in test_predictions:
                writer.writerow({"ID": row["ID"], "Label": row["Label"]})

        print(f"  Saved → {submission_path}")

        summary_rows.append({
            "task":                   task,
            "n_train":                len(train_results),
            "n_test":                 len(test_predictions),
            "accuracy":               round(accuracy, 4),
            "accuracy_known":         round(acc_known, 4),
            "unknown_frac":           round(unknown / len(train_results), 4),
            "n_labels":               len(meta["label_names"]),
            "total_new_tokens":       total_tok,
            "throughput_tok_per_sec": round(throughput, 1),
            "gen_time_min":           round(gen_time / 60, 2),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70)
    print("RAFT 0-SHOT EVALUATION SUMMARY")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    if len(summary_rows) > 1:
        print(f"\nMean accuracy (all tasks):  {summary_df['accuracy'].mean():.4f}")
        print(f"Mean accuracy (known only): {summary_df['accuracy_known'].mean():.4f}")

    final_summary = {
        "model":         MODEL_NAME,
        "tasks":         tasks_to_eval,
        "per_task":      summary_rows,
        "mean_accuracy": float(summary_df["accuracy"].mean()),
    }
    summary_path = os.path.join(RESULTS_DIR, "raft_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print("Download results: modal volume get raft-llama3-results /results ./local_results")

    results_vol.commit()  # flush volume writes before container exits
    return summary_rows


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import pandas as pd

    print(f"Submitting RAFT eval for {MODEL_NAME} on Modal (GPU: L4)...")
    summary = evaluate_raft.remote()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    df = pd.DataFrame(summary)
    print(df[["task", "accuracy", "accuracy_known", "unknown_frac", "n_labels"]].to_string(index=False))
    print(f"\nMean accuracy: {df['accuracy'].mean():.4f}")
