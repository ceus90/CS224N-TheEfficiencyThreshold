# src/adapters/gsm8k.py
"""
Adapter: GSM8K (Grade School Math 8K)

Hugging Face:
  load_dataset("gsm8k", "main")

Outputs standardized examples:
  {
    "id": str,        # stable id
    "x": str,         # input text (question, formatted)
    "y": str,         # target answer (final answer string)
    "meta": {...}     # optional metadata
  }

Notes:
- GSM8K is generative QA. The HF field `answer` usually contains a worked solution
  ending with "#### <final_answer>".
- This adapter extracts the final answer string after "####".
- No "label" field by default (you set stratify=false in splits config).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from datasets import load_dataset


_FINAL_ANSWER_RE = re.compile(r"####\s*([^\n]+)")


def validate_schema(columns: List[str]) -> None:
    """Validate required columns exist."""
    required = ["question", "answer"]
    missing = [c for c in required if c not in columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Columns present: {columns}")


def make_example_id(config: str, split: str, row_idx: int) -> str:
    """Create a stable example id."""
    return f"gsm8k_{config}_{split}_{row_idx:06d}"


def format_input(question: str) -> str:
    """Format GSM8K input text consistently."""
    return f"Question: {question}\nAnswer:"


def extract_final_answer(answer_text: str) -> str:
    """
    Extract GSM8K final answer from the `answer` field.

    Typical format ends with:
      ... #### 42

    Returns:
      final answer string (e.g., "42")
    """
    s = (answer_text or "").strip()
    m = _FINAL_ANSWER_RE.search(s)
    if m:
        return m.group(1).strip()

    # Fallback: take the last number-like token if delimiter is missing.
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    if nums:
        return nums[-1].strip()

    # Last resort: return the full stripped answer.
    return s


def load_examples_gsm8k(
    config: str = "main",
    split: str = "train",
    trust_remote_code: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load GSM8K and return standardized examples.

    Args:
      config: HF dataset config (usually "main")
      split: HF split to load (train/test)
      trust_remote_code: not needed for gsm8k; keep False

    Returns:
      examples: list of standardized example dicts
    """
    ds = load_dataset("gsm8k", config, trust_remote_code=trust_remote_code)

    if split not in ds:
        raise KeyError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    dsplit = ds[split]

    # Validate expected schema
    validate_schema(list(dsplit.column_names))

    examples: List[Dict[str, Any]] = []
    for i in range(len(dsplit)):
        row = dsplit[i]

        question = row["question"]
        answer_full = row["answer"]

        x = format_input(question=question)
        y = extract_final_answer(answer_full)

        examples.append(
            {
                "id": make_example_id(config=config, split=split, row_idx=i),
                "x": x,
                "y": y,
                "meta": {
                    "dataset": "gsm8k",
                    "config": config,
                    "split": split,
                    "row_idx": i,
                    "answer_full": answer_full,
                },
            }
        )

    return examples


def load_examples(dataset_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Uniform adapter entrypoint used by scripts/make_splits.py.

    Reads:
      hf_config, hf_split, trust_remote_code
    """
    config = dataset_cfg.get("hf_config", "main")
    split = dataset_cfg.get("hf_split", "train")
    trust_remote_code = bool(dataset_cfg.get("trust_remote_code", False))

    return load_examples_gsm8k(
        config=config,
        split=split,
        trust_remote_code=trust_remote_code,
    )


# smoke-run (for quick debugging)
if __name__ == "__main__":
    exs = load_examples_gsm8k(config="main", split="train")
    print("Loaded examples:", len(exs))
    print("Example:", exs[0])
    print("y (final):", exs[0]["y"])
