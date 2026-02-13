# src/adapters/raft.py
"""
Adapter: RAFT (ought/raft)

RAFT is a multi-task benchmark where each HF *config* corresponds to a task.
We load all tasks (configs) by default and concatenate into one standardized list.

Hugging Face:
  load_dataset("ought/raft", "<task_name>")

Outputs standardized examples:
  {
    "id": str,         # stable id
    "x": str,          # input text / prompt
    "y": str,          # target label as string
    "label": str,      # same as y (optional; useful for label_counts)
    "task_name": str,  # REQUIRED for round-robin grouping in make_splits.py
    "meta": {...}      # optional metadata
  }

Notes:
- Your make_splits.py expects group_key="task_name" at top level.
- RAFT tasks have heterogeneous schemas. We format "x" robustly:
    - Prefer common text fields (e.g., "text", "sentence", "question", etc.)
    - Otherwise, fall back to a key:value listing of non-label fields.
- For labels:
    - If HF uses ClassLabel, we map int -> class name.
    - Else we stringify the raw label.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from datasets import ClassLabel, get_dataset_config_names, load_dataset

import os

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

# -----------------------------
# Formatting helpers
# -----------------------------

_PREFERRED_TEXT_KEYS: Tuple[str, ...] = (
    # common across many NLP datasets
    "text",
    "sentence",
    "sentence1",
    "sentence2",
    "premise",
    "hypothesis",
    "question",
    "query",
    "review",
    "comment",
    "title",
    "body",
    "context",
    "passage",
    "headline",
)

_COMMON_LABEL_KEYS: Tuple[str, ...] = (
    "label",
    "labels",
    "target",
    "answer",
    "gold",
)


def _pick_label_key(columns: Sequence[str]) -> Optional[str]:
    """Pick the first label-ish key present in the dataset columns."""
    lower_map = {c.lower(): c for c in columns}

    for k in _COMMON_LABEL_KEYS:
        if k in lower_map:
            return lower_map[k]

    return None


def _format_as_kv_block(row: Dict[str, Any], keys: Sequence[str]) -> str:
    """Format selected fields as a stable key/value block."""
    lines: List[str] = []
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        lines.append(f"{k}: {v}")
    return "\n".join(lines).strip()


def _format_input(row: Dict[str, Any], columns: Sequence[str], label_key: Optional[str]) -> str:
    """
    Heuristic formatting for RAFT input text.

    Strategy:
      1) If there are preferred text keys, format them nicely.
      2) Else, fall back to a key/value listing of non-label fields.
    """
    # Prefer known text-like keys if present
    present_text_keys = [k for k in _PREFERRED_TEXT_KEYS if k in columns and row.get(k) is not None]
    if present_text_keys:
        # Special-case pairs that often belong together
        if "sentence1" in present_text_keys and "sentence2" in present_text_keys:
            return f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}".strip()
        if "premise" in present_text_keys and "hypothesis" in present_text_keys:
            return f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}".strip()

        # Otherwise, just include them in order
        return _format_as_kv_block(row, present_text_keys)

    # Fall back to all non-label fields as key/value pairs (stable order: columns order)
    non_label_cols = [c for c in columns if c != label_key]
    return _format_as_kv_block(row, non_label_cols)


def _label_to_string(ds_split, label_key: str, raw_label: Any) -> str:
    """
    Convert raw label to a readable string.
    - If ds features provide ClassLabel, map int -> name.
    - Else stringify raw.
    """
    try:
        feat = ds_split.features.get(label_key)  # type: ignore[union-attr]
        if isinstance(feat, ClassLabel):
            # HF may store as int already; be defensive
            idx = int(raw_label)
            return str(feat.names[idx])
    except Exception:
        pass
    return str(raw_label)


def _normalize_hf_config_list(hf_id: str, hf_config: Optional[Union[str, List[str]]]) -> List[str]:
    """
    Determine which RAFT tasks (configs) to load.
    - hf_config = None -> load all configs
    - hf_config = "task_name" -> load that one
    - hf_config = ["task1", "task2"] -> load those
    """
    if hf_config is None:
        return list(get_dataset_config_names(hf_id))
    if isinstance(hf_config, str):
        return [hf_config]
    if isinstance(hf_config, list) and all(isinstance(x, str) for x in hf_config):
        return list(hf_config)
    raise ValueError(f"Invalid hf_config for RAFT: {hf_config!r}")


def make_example_id(task_name: str, split: str, row_idx: int) -> str:
    """Create a stable example id."""
    return f"raft_{task_name}_{split}_{row_idx:06d}"


# -----------------------------
# Core loader
# -----------------------------

def load_examples_raft(
    hf_id: str = "ought/raft",
    hf_config: Optional[Union[str, List[str]]] = None,
    split: str = "train",
    trust_remote_code: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load RAFT examples across tasks and return standardized examples.

    Args:
      hf_id: HF dataset id (should be "ought/raft")
      hf_config: None => all tasks; or a specific task name; or list of task names
      split: which split to load (RAFT few-shot data is typically in "train")
      trust_remote_code: passed to HF loader (usually False is fine)

    Returns:
      Combined list of standardized example dicts across tasks.
    """
    task_names = _normalize_hf_config_list(hf_id, hf_config)

    all_examples: List[Dict[str, Any]] = []

    for task_name in task_names:
        ds = load_dataset(hf_id, task_name, trust_remote_code=trust_remote_code)

        if split not in ds:
            raise KeyError(
                f"Split '{split}' not found for RAFT task '{task_name}'. "
                f"Available splits: {list(ds.keys())}"
            )

        dsplit = ds[split]
        columns = list(dsplit.column_names)

        label_key = _pick_label_key(columns)
        if label_key is None:
            raise KeyError(
                f"Could not find a label column for RAFT task '{task_name}'. "
                f"Columns: {columns}"
            )

        for i in range(len(dsplit)):
            row = dsplit[i]

            x = _format_input(row=row, columns=columns, label_key=label_key)

            raw_label = row.get(label_key)
            if raw_label is None:
                # For RAFT train this should not happen; fail loudly.
                raise ValueError(
                    f"Missing label at row {i} for RAFT task '{task_name}' (label_key='{label_key}')."
                )

            y = _label_to_string(dsplit, label_key=label_key, raw_label=raw_label)

            all_examples.append(
                {
                    "id": make_example_id(task_name=task_name, split=split, row_idx=i),
                    "x": x,
                    "y": y,
                    "label": y,  # helps label_counts + future stratify if needed
                    "task_name": task_name,  # REQUIRED for round-robin grouping
                    "meta": {
                        "dataset": "ought/raft",
                        "task_name": task_name,
                        "split": split,
                        "row_idx": i,
                        "label_key": label_key,
                    },
                }
            )

    return all_examples


def load_examples(dataset_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Uniform adapter entrypoint used by scripts/make_splits.py.

    Reads:
      hf_id, hf_config (None or task or list), hf_split, trust_remote_code
    """
    hf_id = dataset_cfg.get("hf_id", "ought/raft")
    hf_config = dataset_cfg.get("hf_config", None)
    split = dataset_cfg.get("hf_split", "train")
    trust_remote_code = bool(dataset_cfg.get("trust_remote_code", False))

    return load_examples_raft(
        hf_id=hf_id,
        hf_config=hf_config,
        split=split,
        trust_remote_code=trust_remote_code,
    )


# -----------------------------
# Smoke-run
# -----------------------------

if __name__ == "__main__":
    exs = load_examples_raft(
        hf_id="ought/raft", 
        hf_config=None, 
        split="train", 
        trust_remote_code=True,
    )
    print("Loaded examples:", len(exs))

    # Group counts (tasks)
    from collections import Counter

    c_tasks = Counter(e["task_name"] for e in exs)
    print("Num tasks:", len(c_tasks))
    print("Task size min/max:", min(c_tasks.values()), max(c_tasks.values()))

    # Show one example
    print("Example:", exs[0])
