# src/adapters/financial_phrasebank.py
"""
Adapter: Financial PhraseBank (Hugging Face)

Outputs standardized examples:
  {
    "id": str,        # stable id
    "x": str,         # input text
    "y": str,         # target label (string)
    "label": str,     # same as y (used for stratification)
    "meta": {...}     # optional metadata
  }

Notes:
- HF dataset uses a script loader; pass trust_remote_code=True to avoid prompts.
- Common configs: sentences_allagree, sentences_75agree, sentences_66agree, sentences_50agree
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset


# Default label mapping used in most Financial PhraseBank loaders
_DEFAULT_LABEL_MAP: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def default_label_map() -> Dict[int, str]:
    """Return the default int->string label mapping."""
    return dict(_DEFAULT_LABEL_MAP)


def validate_schema(columns: List[str]) -> None:
    """Validate required columns exist."""
    if "sentence" not in columns:
        raise KeyError(f"Expected text column 'sentence' not found. Columns: {columns}")
    if "label" not in columns:
        raise KeyError(f"Expected label column 'label' not found. Columns: {columns}")


def make_example_id(config: str, split: str, row_idx: int) -> str:
    """Create a stable example id."""
    return f"fpb_{config}_{split}_{row_idx:06d}"


def load_examples_financial_phrasebank(
    config: str = "sentences_allagree",
    split: str = "train",
    trust_remote_code: bool = True,
    label_map: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load Financial PhraseBank examples and return standardized examples.

    Args:
      config: HF dataset config (e.g., 'sentences_allagree')
      split: dataset split (PhraseBank typically only provides 'train')
      trust_remote_code: must be True to avoid interactive prompt
      label_map: optional override for mapping label ints to strings

    Returns:
      examples: list of standardized example dicts
    """
    # Load dataset
    ds = load_dataset(
        "financial_phrasebank",
        config,
        trust_remote_code=trust_remote_code,
    )

    # Check split exists
    if split not in ds:
        raise KeyError(f"Split '{split}' not found. Available splits: {list(ds.keys())}")

    dsplit = ds[split]

    # Validate expected schema
    validate_schema(list(dsplit.column_names))

    # Choose label mapping
    lm = label_map if label_map is not None else default_label_map()

    # Build examples
    examples: List[Dict[str, Any]] = []
    for i in range(len(dsplit)):
        row = dsplit[i]

        # Extract raw fields
        text = row["sentence"]
        raw_label = row["label"]

        # Validate label id is known
        if raw_label not in lm:
            raise ValueError(
                f"Unknown label id {raw_label}. "
                f"Known ids: {sorted(list(lm.keys()))}. "
                f"Row idx={i}, config={config}, split={split}"
            )

        y = lm[raw_label]

        # Standardized example
        examples.append(
            {
                "id": make_example_id(config=config, split=split, row_idx=i),
                "x": text,
                "y": y,
                "label": y,  # stratify on string label
                "meta": {
                    "dataset": "financial_phrasebank",
                    "config": config,
                    "split": split,
                    "row_idx": i,
                    "label_id": int(raw_label),
                },
            }
        )

    return examples

def load_examples(dataset_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Uniform adapter entrypoint used by scripts/make_splits.py."""
    config = dataset_cfg.get("hf_config", "sentences_allagree")
    split = dataset_cfg.get("hf_split", "train")
    trust_remote_code = bool(dataset_cfg.get("trust_remote_code", True))

    # label_map in config is JSON, so keys come in as strings ("0","1","2")
    lm_raw = dataset_cfg.get("label_map")
    label_map = None
    if isinstance(lm_raw, dict):
        label_map = {int(k): str(v) for k, v in lm_raw.items()}

    return load_examples_financial_phrasebank(
        config=config,
        split=split,
        trust_remote_code=trust_remote_code,
        label_map=label_map,
    )


# smoke-run (for quick debugging)
if __name__ == "__main__":
    # Load a small sample and print counts
    exs = load_examples_financial_phrasebank()
    print("Loaded examples:", len(exs))

    # Count labels
    from collections import Counter

    c = Counter(e["label"] for e in exs)
    print("Label counts:", dict(c))

    # Show one example
    print("Example:", exs[0])
