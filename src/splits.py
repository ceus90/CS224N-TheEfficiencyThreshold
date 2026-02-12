# src/splits.py
"""
Split utilities for few-shot experiments.

Goal:
- Deterministic few-shot subsets for N in {16,32,64,128,256}
- Support multiple seeds (e.g., 0/1/2)
- Optional stratification by label
- Nested subsets (N16 ⊂ N32 ⊂ ...)

This file is dataset-agnostic.
Adapters should produce standardized examples, then call make_splits().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

import math
import random
from collections import Counter, defaultdict


Label = Hashable  # label can be int/str/etc.
IndexList = List[int]
SplitsForSeed = Dict[int, IndexList]          # {N: [indices]}
AllSplits = Dict[int, SplitsForSeed]          # {seed: {N: [indices]}}


# -----------------------------
# Core helpers
# -----------------------------

def _unique_in_order(xs: Sequence[Label]) -> List[Label]:
    """Return unique labels in first-seen order."""
    seen = set()
    out: List[Label] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def allocate_counts(
    labels: Sequence[Label],
    N: int,
    min_per_class: int = 1,
) -> Dict[Label, int]:
    """
    Decide how many examples to sample per label for total size N.

    - Proportional to label frequencies in `labels`
    - Ensures each class gets at least `min_per_class` when feasible
    - Returns counts that sum exactly to N

    Notes:
    - If N is too small, we cannot guarantee min_per_class for all classes.
    - We keep the allocation deterministic (no RNG).
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    counts = Counter(labels)
    if not counts:
        raise ValueError("labels is empty")

    label_ids = list(counts.keys())
    total = len(labels)
    k = len(label_ids)

    # Compute raw proportional targets (floats).
    raw = {lab: (counts[lab] / total) * N for lab in label_ids}

    # Start with floors.
    alloc = {lab: int(math.floor(raw[lab])) for lab in label_ids}

    # Enforce minimum per class if feasible.
    # Feasible condition: N >= k * min_per_class
    if N >= k * min_per_class:
        for lab in label_ids:
            if alloc[lab] < min_per_class:
                alloc[lab] = min_per_class

    # Fix sum to exactly N by adding/removing from classes.
    current_sum = sum(alloc.values())
    remainder = N - current_sum

    # Sort by fractional part to distribute additions "fairly".
    # If remainder is negative (too many), we'll remove from low fractional part last.
    frac = {lab: (raw[lab] - math.floor(raw[lab])) for lab in label_ids}
    frac_desc = sorted(label_ids, key=lambda lab: frac[lab], reverse=True)
    frac_asc = list(reversed(frac_desc))

    i = 0
    while remainder != 0:
        if remainder > 0:
            lab = frac_desc[i % len(frac_desc)]
            alloc[lab] += 1
            remainder -= 1
        else:
            lab = frac_asc[i % len(frac_asc)]
            # Respect minimum if feasible.
            min_allowed = min_per_class if (N >= k * min_per_class) else 0
            if alloc[lab] > min_allowed:
                alloc[lab] -= 1
                remainder += 1
        i += 1

        # Safety: prevent infinite loops in weird edge cases.
        if i > 10_000_000:
            raise RuntimeError("allocate_counts failed to converge; check inputs")

    # Final sanity check.
    if sum(alloc.values()) != N:
        raise RuntimeError("allocate_counts produced wrong total; this should not happen")

    return alloc


def make_nested_splits(
    labels: Sequence[Label],
    Ns: Sequence[int],
    seed: int,
    stratify: bool = True,
    min_per_class: int = 1,
) -> SplitsForSeed:
    """
    Build nested splits for one seed.

    Returns: {N: indices} where indices are positions into `labels`.

    Behavior:
    - If stratify=True:
        * group indices by label
        * shuffle each label bucket (deterministic by seed)
        * for each N, take label-wise prefixes based on allocate_counts()
      This guarantees nesting by construction (prefix property).
    - If stratify=False:
        * shuffle all indices once (deterministic by seed)
        * for each N, take global prefix
      Also guarantees nesting.

    Notes:
    - Ns can be any increasing sequence; nesting assumes you want prefix-style subsets.
    """
    if not Ns:
        raise ValueError("Ns is empty")
    Ns_sorted = list(Ns)
    if Ns_sorted != sorted(Ns_sorted):
        raise ValueError("Ns must be sorted ascending for nested subsets")

    if not labels:
        raise ValueError("labels is empty")

    n_total = len(labels)
    if max(Ns_sorted) > n_total:
        raise ValueError(f"Max N={max(Ns_sorted)} exceeds dataset size={n_total}")

    rng = random.Random(seed)

    splits: SplitsForSeed = {}

    if not stratify:
        # Shuffle all indices once, then take prefixes.
        all_idxs = list(range(n_total))
        rng.shuffle(all_idxs)

        for N in Ns_sorted:
            # Prefix ensures nesting.
            picked = sorted(all_idxs[:N])  # sort for stable file output
            splits[N] = picked

        return splits

    # Stratified path: group indices by label.
    indices_by_label: Dict[Label, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        indices_by_label[lab].append(i)

    # Shuffle within each label bucket once.
    shuffled_by_label: Dict[Label, List[int]] = {}
    for lab, idxs in indices_by_label.items():
        idxs_copy = idxs[:]
        rng.shuffle(idxs_copy)
        shuffled_by_label[lab] = idxs_copy

    # For each N, allocate per-label counts and take per-label prefixes.
    for N in Ns_sorted:
        alloc = allocate_counts(labels, N, min_per_class=min_per_class)

        picked: List[int] = []
        for lab, k in alloc.items():
            # If a label exists, it must have enough examples (it will).
            picked.extend(shuffled_by_label[lab][:k])

        # Stable ordering.
        splits[N] = sorted(picked)

    # Validate nesting (should always pass).
    check_nested(splits)

    return splits


def make_splits(
    examples: Sequence[Mapping[str, Any]],
    Ns: Sequence[int],
    seeds: Sequence[int],
    label_key: str = "label",
    stratify: bool = True,
    min_per_class: int = 1,
) -> AllSplits:
    """
    Multi-seed wrapper.

    Inputs:
    - examples: list of standardized examples (dict-like)
    - Ns: list of N sizes (must be ascending for nesting)
    - seeds: list of integer seeds
    - label_key: key to read labels from each example
    - stratify: whether to stratify by label_key
    - min_per_class: minimum examples per class when feasible

    Returns:
    - {seed: {N: [indices]}}
    """
    if not examples:
        raise ValueError("examples is empty")
    if not seeds:
        raise ValueError("seeds is empty")

    # Extract labels once (cheap + consistent).
    labels: List[Label] = []
    for ex in examples:
        if label_key not in ex:
            raise KeyError(f"Example missing label_key='{label_key}': {ex}")
        labels.append(ex[label_key])  # type: ignore[assignment]

    all_splits: AllSplits = {}
    for seed in seeds:
        splits_for_seed = make_nested_splits(
            labels=labels,
            Ns=Ns,
            seed=seed,
            stratify=stratify,
            min_per_class=min_per_class,
        )
        check_sizes(splits_for_seed, Ns)
        check_nested(splits_for_seed)
        all_splits[seed] = splits_for_seed

    return all_splits


# -----------------------------
# Validation helpers
# -----------------------------

def check_sizes(splits_for_seed: SplitsForSeed, Ns: Sequence[int]) -> None:
    """Verify each split exists and has exactly the expected size."""
    for N in Ns:
        if N not in splits_for_seed:
            raise ValueError(f"Missing split for N={N}")
        got = len(splits_for_seed[N])
        if got != N:
            raise ValueError(f"Split size mismatch for N={N}: got {got}")


def check_nested(splits_for_seed: SplitsForSeed) -> None:
    """
    Verify nesting property: smaller N indices are subset of larger N indices.

    Assumes N keys are intended to be nested; checks in ascending N order.
    """
    Ns_sorted = sorted(splits_for_seed.keys())
    for a, b in zip(Ns_sorted[:-1], Ns_sorted[1:]):
        A = set(splits_for_seed[a])
        B = set(splits_for_seed[b])
        if not A.issubset(B):
            raise ValueError(f"Nesting failed: N={a} is not subset of N={b}")


# -----------------------------
# Convenience 
# -----------------------------

def indices_to_examples(
    examples: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> List[Mapping[str, Any]]:
    """
    Convert indices into a list of example dicts.
    Useful for writing JSONL after splitting.
    """
    return [examples[i] for i in indices]
