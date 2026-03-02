# src/adapters/superglue_boolq.py
"""
Adapter: SuperGLUE BoolQ (Boolean Questions)

Hugging Face:
  load_dataset("super_glue", "boolq")

Outputs standardized examples:
  {
    "id": str,        # stable id
    "x": str,         # input text (passage + question)
    "y": str,         # target label (string)
    "label": str,     # same as y (used for stratification)
    "meta": {...}     # optional metadata
  }

Notes:
- BoolQ is a binary yes/no reading comprehension task.
- HF labels are integers:
    0 -> False
    1 -> True
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset


# Default mapping for SuperGLUE BoolQ
_DEFAULT_LABEL_MAP: Dict[int, str] = {
    0: "False",
    1: "True",
}


def default_label_map() -> Dict[int, str]:
    """Return the default int->string label mapping for BoolQ."""
    return dict(_DEFAULT_LABEL_MAP)


def validate_schema(columns: List[str]) -> None:
    """Validate required columns exist."""
    required = ["passage", "question", "label"]
    missing = [c for c in required if c not in columns]
    if missing:
        raise KeyError(f"Missing columns {missing}. Columns present: {columns}")


def make_example_id(split: str, row_idx: int) -> str:
    """Create a stable example id."""
    return f"boolq_{split}_{row_idx:06d}"


def format_input(passage: str, question: str) -> str:
    """Format BoolQ input text consistently."""
    return f"Passage: {passage}\nQuestion: {question} "


def load_examples_boolq(
    split: str = "train",
    trust_remote_code: bool = False,
    label_map: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load SuperGLUE BoolQ and return standardized examples.

    Args:
      split: HF split to load (train/validation/test)
      trust_remote_code: not needed for super_glue; keep False
      label_map: optional override mapping label ints to strings

    Returns:
      examples: list of standardized example dicts
    """
    ds = load_dataset("super_glue", "boolq", trust_remote_code=trust_remote_code)

    if split not in ds:
        raise KeyError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    dsplit = ds[split]

    # Validate expected schema
    validate_schema(list(dsplit.column_names))

    # Choose label mapping
    lm = label_map if label_map is not None else default_label_map()

    examples: List[Dict[str, Any]] = []
    for i in range(len(dsplit)):
        row = dsplit[i]

        passage = row["passage"]
        question = row["question"]
        raw_label = row["label"]

        # Some HF splits (like test) may have label = -1; fail early.
        if raw_label not in lm:
            raise ValueError(
                f"Unknown label id {raw_label}. "
                f"Known ids: {sorted(list(lm.keys()))}. "
                f"Row idx={i}, split={split}"
            )

        y = lm[int(raw_label)]
        x = format_input(passage=passage, question=question)

        examples.append(
            {
                "id": make_example_id(split=split, row_idx=i),
                "x": x,
                "y": y,
                "label": y,  # stratify on string label
                "meta": {
                    "dataset": "super_glue",
                    "config": "boolq",
                    "split": split,
                    "row_idx": i,
                    "label_id": int(raw_label),
                },
            }
        )

    return examples


def load_examples(dataset_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Uniform adapter entrypoint used by scripts/make_splits.py.

    Reads:
      hf_split, trust_remote_code, label_map
    """
    split = dataset_cfg.get("hf_split", "train")
    trust_remote_code = bool(dataset_cfg.get("trust_remote_code", False))

    # label_map is JSON, so keys come in as strings ("0","1")
    lm_raw = dataset_cfg.get("label_map")
    label_map = None
    if isinstance(lm_raw, dict):
        label_map = {int(k): str(v) for k, v in lm_raw.items()}

    return load_examples_boolq(
        split=split,
        trust_remote_code=trust_remote_code,
        label_map=label_map,
    )


# smoke-run (for quick debugging)
if __name__ == "__main__":
    exs = load_examples_boolq(split="train")
    print("Loaded examples:", len(exs))

    from collections import Counter

    c = Counter(e["label"] for e in exs)
    print("Label counts:", dict(c))

    print("Example:", exs[0])