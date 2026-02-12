# tests/test_splits.py
"""
tests for src/splits.py

How to run (from repo root):
  python -m tests.test_splits

These are lightweight "smoke + edge case" tests.
No pytest required.
"""

from __future__ import annotations

from collections import Counter

from src.splits import (
    allocate_counts,
    make_nested_splits,
    make_splits,
    check_nested,
    check_sizes,
)


# -----------------------------
# Tiny assertion helpers
# -----------------------------

def assert_raises(exc_type, fn, *args, **kwargs):
    """Assert fn(*args, **kwargs) raises exc_type."""
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"Expected {exc_type.__name__} to be raised, but nothing was raised")


def assert_true(cond: bool, msg: str):
    """Assert a condition is True."""
    if not cond:
        raise AssertionError(msg)


# -----------------------------
# Fixtures
# -----------------------------

def make_fake_examples(nA=50, nB=30, nC=20):
    """Create standardized examples with labels A/B/C."""
    examples = []
    idx = 0
    for _ in range(nA):
        examples.append({"id": f"ex_{idx}", "x": f"x{idx}", "y": "A", "label": "A"})
        idx += 1
    for _ in range(nB):
        examples.append({"id": f"ex_{idx}", "x": f"x{idx}", "y": "B", "label": "B"})
        idx += 1
    for _ in range(nC):
        examples.append({"id": f"ex_{idx}", "x": f"x{idx}", "y": "C", "label": "C"})
        idx += 1
    return examples


# -----------------------------
# Tests: allocate_counts
# -----------------------------

def test_allocate_counts_sum_and_nonneg():
    """Counts should sum to N and be non-negative."""
    labels = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    for N in [1, 2, 3, 16, 32, 64]:
        alloc = allocate_counts(labels, N, min_per_class=1)
        assert_true(sum(alloc.values()) == N, f"allocate_counts sum != N for N={N}")
        assert_true(all(v >= 0 for v in alloc.values()), "allocate_counts produced negative count")


def test_allocate_counts_min_per_class_when_feasible():
    """When N >= k*min_per_class, each class should get >= min_per_class."""
    labels = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    alloc = allocate_counts(labels, N=6, min_per_class=2)  # feasible: 3 classes * 2 = 6
    assert_true(alloc["A"] >= 2 and alloc["B"] >= 2 and alloc["C"] >= 2, "min_per_class not enforced")


def test_allocate_counts_min_per_class_not_feasible():
    """When infeasible, should not crash."""
    labels = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    alloc = allocate_counts(labels, N=2, min_per_class=1)  # infeasible to give 1 to all 3 classes
    assert_true(sum(alloc.values()) == 2, "sum should still match N in infeasible case")


def test_allocate_counts_bad_inputs():
    """Bad inputs should raise."""
    assert_raises(ValueError, allocate_counts, [], 16)
    assert_raises(ValueError, allocate_counts, ["A"], 0)
    assert_raises(ValueError, allocate_counts, ["A"], -1)


# -----------------------------
# Tests: make_nested_splits
# -----------------------------

def test_make_nested_splits_stratified_nesting_and_sizes():
    """Stratified nested splits should be nested and correct sizes."""
    labels = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    Ns = [16, 32, 64]
    splits = make_nested_splits(labels, Ns=Ns, seed=0, stratify=True, min_per_class=1)

    check_sizes(splits, Ns)
    check_nested(splits)

    # Basic sanity: if N is large enough, each class should appear at least once.
    # Here N=16 and min_per_class=1 is feasible (16 >= 3).
    picked16 = [labels[i] for i in splits[16]]
    c16 = Counter(picked16)
    assert_true(all(c16.get(k, 0) >= 1 for k in ["A", "B", "C"]), "Not all classes appear in stratified split")


def test_make_nested_splits_unstratified_nesting_and_sizes():
    """Unstratified nested splits should be nested and correct sizes."""
    labels = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    Ns = [16, 32, 64]
    splits = make_nested_splits(labels, Ns=Ns, seed=123, stratify=False)

    check_sizes(splits, Ns)
    check_nested(splits)


def test_make_nested_splits_unsorted_Ns_raises():
    """Ns must be sorted for nesting."""
    labels = ["A"] * 10 + ["B"] * 10
    assert_raises(ValueError, make_nested_splits, labels, [32, 16], 0, True)


def test_make_nested_splits_N_too_large_raises():
    """N cannot exceed dataset size."""
    labels = ["A"] * 10 + ["B"] * 10
    assert_raises(ValueError, make_nested_splits, labels, [16, 32], 0, True)  # size=20, N=32 invalid


def test_make_nested_splits_empty_labels_raises():
    """Empty labels should raise."""
    assert_raises(ValueError, make_nested_splits, [], [16], 0, True)


# -----------------------------
# Tests: make_splits (multi-seed + label_key)
# -----------------------------

def test_make_splits_multi_seed_shapes():
    """make_splits returns {seed: {N: indices}} and is valid for each seed."""
    examples = make_fake_examples()
    Ns = [16, 32, 64]
    seeds = [0, 1, 2]

    out = make_splits(examples, Ns=Ns, seeds=seeds, label_key="label", stratify=True)

    assert_true(set(out.keys()) == set(seeds), "Missing seeds in output")
    for s in seeds:
        splits = out[s]
        check_sizes(splits, Ns)
        check_nested(splits)


def test_make_splits_missing_label_key_raises():
    """If an example is missing label_key, raise KeyError."""
    examples = [{"id": "1", "x": "x", "y": "y"}]  # no "label"
    assert_raises(KeyError, make_splits, examples, [1], [0], "label", True)


def test_make_splits_empty_inputs_raise():
    """Empty examples or empty seeds should raise."""
    assert_raises(ValueError, make_splits, [], [16], [0])
    assert_raises(ValueError, make_splits, [{"id": "1", "x": "x", "y": "y", "label": "A"}], [16], [])


# -----------------------------
# Runner
# -----------------------------

def main():
    # Run all tests )
    tests = [
        test_allocate_counts_sum_and_nonneg,
        test_allocate_counts_min_per_class_when_feasible,
        test_allocate_counts_min_per_class_not_feasible,
        test_allocate_counts_bad_inputs,
        test_make_nested_splits_stratified_nesting_and_sizes,
        test_make_nested_splits_unstratified_nesting_and_sizes,
        test_make_nested_splits_unsorted_Ns_raises,
        test_make_nested_splits_N_too_large_raises,
        test_make_nested_splits_empty_labels_raises,
        test_make_splits_multi_seed_shapes,
        test_make_splits_missing_label_key_raises,
        test_make_splits_empty_inputs_raise,
    ]

    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
