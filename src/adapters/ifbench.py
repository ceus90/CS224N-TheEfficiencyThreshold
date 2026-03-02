# src/adapters/ifbench.py
"""
Adapter: IFBench (Muennighoff/IFEval)

Hugging Face:
  load_dataset("Muennighoff/IFEval")

Outputs standardized examples:
  {
    "id": str,        # stable id
    "x": str,         # input text (instruction prompt)
    "y": str,         # target label (empty for open-ended generation)
    "label": str,     # fixed dummy label for stratification
    "meta": {...}     # optional metadata
  }

Notes:
- IFEval evaluates instruction adherence (formatting, length, etc.).
- It does not have a fixed ground-truth target text, so 'y' is empty.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset


def validate_schema(columns: List[str]) -> None:
    """Validate required columns exist."""
    # IFEval uses 'prompt', but we allow 'instruction' as a fallback
    if "prompt" not in columns and "instruction" not in columns:
        raise KeyError(f"Missing 'prompt' column. Columns present: {columns}")


def make_example_id(split: str, row_idx: int) -> str:
    """Create a stable example id."""
    return f"ifbench_{split}_{row_idx:06d}"

def load_examples_ifbench(
    split: str = "train",
    trust_remote_code: bool = False,
    label_map: Optional[Dict[Any, Any]] = None, # Unused for IFBench but kept for signature matching
) -> List[Dict[str, Any]]:
    """
    Load IFBench (IFEval) and return standardized examples.

    Args:
      split: HF split to load
      trust_remote_code: keep False unless required by HF
      label_map: ignored for generative tasks

    Returns:
      examples: list of standardized example dicts
    """
    ds = load_dataset("google/IFEval", trust_remote_code=trust_remote_code)

    # IFEval sometimes uses 'train' as its only split, fallback if requested split is missing
    if split not in ds:
        available_splits = list(ds.keys())
        print(f"Split '{split}' not found. Falling back to '{available_splits[0]}'")
        split = available_splits[0]

    dsplit = ds[split]

    # Validate expected schema
    validate_schema(list(dsplit.column_names))

    examples: List[Dict[str, Any]] = []
    for i in range(len(dsplit)):
        row = dsplit[i]

        # Extract the instruction prompt
        prompt = row.get("prompt", row.get("instruction", ""))
        
        # IFEval doesn't have a strict output label, so we leave it empty
        y = ""
        x = prompt

        examples.append(
            {
                "id": make_example_id(split=split, row_idx=i),
                "x": x,
                "y": y,
                "label": "generation",  # Uniform dummy label so stratification scripts don't crash
                "meta": {
                    "dataset": "ifbench",
                    "config": "ifeval",
                    "split": split,
                    "row_idx": i,
                },
            }
        )

    return examples


def load_examples(dataset_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Uniform adapter entrypoint used by scripts/make_splits.py.
    """
    split = dataset_cfg.get("hf_split", "train")
    trust_remote_code = bool(dataset_cfg.get("trust_remote_code", False))

    return load_examples_ifbench(
        split=split,
        trust_remote_code=trust_remote_code,
    )


# smoke-run (for quick debugging)
if __name__ == "__main__":
    exs = load_examples_ifbench(split="train")
    print("Loaded examples:", len(exs))

    print("Example:", exs[0])