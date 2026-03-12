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

# -----------------------------
# Formatting helpers (STRICT)
# -----------------------------

# Keep the existing _PREFERRED_TEXT_KEYS, but use a strict priority order for "main text"
# (works across many RAFT tasks)
_STRICT_TEXT_PRIORITY: Tuple[str, ...] = (
    "text",
    "sentence",
    "query",
    "question",
    "review",
    "comment",
    "body",
    "context",
    "passage",
    "headline",
    "title",
)

# Exclude obvious metadata fields. (Keep URLs if they appear inside the chosen main text.)
_EXCLUDE_INPUT_KEYS: Tuple[str, ...] = (
    "id",
    "idx",
    "row_idx",
    "example_id",
    "paper_link",
    "link",
    "url",
    "href",
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
    STRICT RAFT input text:
      - Return ONLY the main text that should be classified (no "key: value" formatting).
      - Prefer a single best text field when possible.
      - Handle paired fields (sentence1/sentence2, premise/hypothesis).
      - Keep URLs (do not strip them).
      - Avoid obvious metadata fields (ids/links-as-fields).
    """

    def _val(k: str) -> str:
        v = row.get(k)
        return "" if v is None else str(v).strip()

    # 1) Handle common paired-input tasks first
    if "sentence1" in columns and "sentence2" in columns and row.get("sentence1") is not None and row.get("sentence2") is not None:
        s1, s2 = _val("sentence1"), _val("sentence2")
        return f"{s1}\n{s2}".strip()

    if "premise" in columns and "hypothesis" in columns and row.get("premise") is not None and row.get("hypothesis") is not None:
        p, h = _val("premise"), _val("hypothesis")
        return f"{p}\n{h}".strip()

    # 2) Strict single-field selection: pick the highest-priority main text key available
    for k in _STRICT_TEXT_PRIORITY:
        if k in columns and k not in _EXCLUDE_INPUT_KEYS and row.get(k) is not None:
            s = _val(k)
            if s:
                return s

    # 3) Secondary selection: if any preferred text-like key exists, pick the first present (stable order)
    present_text_keys = [
        k for k in _PREFERRED_TEXT_KEYS
        if k in columns and k not in _EXCLUDE_INPUT_KEYS and row.get(k) is not None and _val(k)
    ]
    if present_text_keys:
        return _val(present_text_keys[0])

    # 4) Last-resort fallback: find the "best looking" string field among non-label, non-excluded columns
    #    (still returns ONLY a value, never "k: v")
    candidates: List[str] = []
    for c in columns:
        if c == label_key or c in _EXCLUDE_INPUT_KEYS:
            continue
        v = row.get(c)
        if v is None:
            continue
        # Prefer plain strings
        if isinstance(v, str):
            s = v.strip()
            if s:
                candidates.append(s)

    if candidates:
        # Heuristic: choose the longest string as the most likely "main text"
        return max(candidates, key=len).strip()

    # If truly nothing usable, return empty string (caller can decide how to handle)
    return ""

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
        ds = load_dataset(hf_id, task_name)

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
    hf_config=[ "ade_corpus_v2"],
    split="train",
)
    print("Loaded examples:", len(exs))

    # Group counts (tasks)
    from collections import Counter

    c_tasks = Counter(e["task_name"] for e in exs)
    print("Num tasks:", len(c_tasks))
    print("Task size min/max:", min(c_tasks.values()), max(c_tasks.values()))

    # Show one example
    print("Example:", exs[0])
