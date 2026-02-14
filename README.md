# 1. Overview

This repository implements a **reproducible few-shot data pipeline** for evaluating parameter-efficient fine-tuning (IFT) and in-context learning (ICL) across multiple NLP benchmarks.

The core experimental goal is:

> Measure how performance scales with supervision size **without introducing data variance as a confound.**

Few-shot experiments are highly sensitive to:

* Random sampling variance
* Class imbalance at small N
* Multi-task heterogeneity (e.g., RAFT)
* Inconsistent splits across model variants

If each experiment samples slightly different examples, we cannot tell whether performance changes come from:

* More data
* A better training method
* Or simply a different subset

This repository eliminates that ambiguity.

---

## What the System Does

The system:

* Downloads datasets from Hugging Face
* Standardizes them into a unified schema
* Generates fixed, reproducible few-shot subsets
* Saves static JSON splits for consistent evaluation across models

Supported datasets:

* Financial PhraseBank
* SuperGLUE RTE
* GSM8K
* RAFT (multi-task, round-robin sampling)

---

## What the Pipeline Guarantees

The pipeline guarantees:

* **Deterministic sampling** via fixed seeds
* **Nested few-shot subsets** (N=16 ⊂ N=32 ⊂ … ⊂ N=256) for fair scaling comparisons
* **Stratified sampling** for classification datasets
* **Group-balanced sampling** for multi-task datasets (RAFT)
* **Static split files** reused across all model variants

For a given `(dataset, N, seed)`:

> Every model (ICL, LoRA, P-Tuning, ReFT, etc.) sees the exact same data.

This isolates the effect of supervision size and training strategy, allowing clean comparison of learning mechanics.

---

## High-Level Pipeline Flow

```
Hugging Face Dataset
        ↓
Dataset Adapter (standardize schema)
        ↓
Standardized Examples
        ↓
Split Engine (deterministic sampling)
        ↓
Static JSONL Few-Shot Splits
        ↓
ICL / IFT Experiments
```

The split engine is deliberately separated from dataset logic to maximize reproducibility and extensibility.

---

# 2. Repository Structure

```
configs/
  splits.json              # Dataset + sampling configuration

scripts/
  make_splits.py           # Entrypoint + sampling strategy selection (default vs round-robin)

src/
  splits.py                # Core split logic (stratified/unstratified, nested)
  adapters/                # Dataset-specific loaders
    financial_phrasebank.py
    gsm8k.py
    raft.py
    superglue_rte.py

data/
  splits_out/              # Local split outputs for testing
                           # (not version-controlled; canonical copies live in GCS)

tests/
  test_splits.py           # Unit tests: determinism, nesting, and exact split sizes
```

---

## Core Components

### 1️ `configs/splits.json`

Defines:

* Few-shot sizes `N ∈ {16, 32, 64, 128, 256}`
* Random seeds
* Sampling strategy
* Dataset-specific settings (HF ID, label key, grouping key, etc.)

This file fully controls split generation behavior.

Changing this file changes the experiment.

---

### 2️ `scripts/make_splits.py` (Entrypoint)

Main orchestration driver that:

1. Reads `configs/splits.json`
2. Loads each dataset via its adapter
3. Applies the configured sampling strategy (default via `src/splits.generate_splits`, or round-robin via `make_splits.py`)
4. Writes static JSONL splits locally
5. Optionally uploads canonical copies to GCS

This is the only script that needs to be executed to regenerate all splits.

It contains no sampling logic — it only orchestrates.

---

### 3️ `src/splits.py` (Sampling Engine)

Contains:

* `generate_splits` (default stratified or unstratified sampling)
* Validation + utilities for nested, deterministic index splits

This module:

* Contains no dataset-specific logic
* Operates purely on standardized examples
* Enforces determinism and nesting
* Performs validation checks

It is fully reusable across datasets.

---

### 4️ `src/adapters/`

Each adapter:

* Loads a dataset from Hugging Face
* Converts it into a standardized schema:

```
{
  "id": str,
  "x": str,
  "y": str,
  "label": str,
  ...
}
```

* Handles dataset-specific quirks (column names, label mapping, formatting)
* Ensures compatibility with the sampling engine

Adapters isolate dataset-specific logic from core experimental mechanics.

---

# Why This Architecture?

This design ensures:

* Clean separation of concerns
* Extendability (new dataset = new adapter only)
* Sampling logic reusable across datasets
* Multi-task compatibility (RAFT via group-aware sampling)

# 3. Global Split Configuration

All few-shot behavior is controlled centrally in `configs/splits.json`.

This file defines **how supervision is scaled**, **how randomness is controlled**, and **how fairness is enforced** across experiments.

Every dataset inherits these global settings unless explicitly overridden.

---

## Core Experimental Controls

### `Ns` — Supervision Scale

```
[16, 32, 64, 128, 256]
```

These are the few-shot sizes evaluated in scaling experiments.

They answer the question:

> How does performance change as we increase supervision?

Because these values are fixed globally, all datasets are evaluated at identical supervision levels.

---

### `seeds` — Variance Control

Each seed generates a deterministic ordering of the dataset.

This produces **independent nested split families**:

* Seed 0 → N16 ⊂ N32 ⊂ …
* Seed 1 → N16 ⊂ N32 ⊂ …
* Seed 2 → N16 ⊂ N32 ⊂ …

There is no merging across seeds.

Multiple seeds allow robustness checks while preserving determinism.

---

### `nested` — Controlled Scaling

If `nested = true`, splits are nested per seed:

```
N16 ⊂ N32 ⊂ N64 ⊂ N128 ⊂ N256
```

This is implemented by generating a single deterministic ordering per seed and taking prefixes.

**Purpose:**

Increasing N only *adds* supervision — it never reshuffles earlier examples.

This isolates the effect of additional data and prevents scaling curves from being contaminated by resampling noise.

---

### `stratify` — Small-N Fairness

If `stratify = true`:

* Sampling is balanced across classes
* Uses `label_key` (typically `"label"`)

Used for classification datasets:

* Financial PhraseBank
* SuperGLUE RTE

**Purpose:**

At small N (e.g., 16), random sampling can accidentally produce severe label imbalance.
Stratification prevents trivial biases from dominating few-shot results.

---

If `stratify = false`:

* Pure deterministic random sampling
* Used for:

  * GSM8K
  * RAFT (multi-task handled via round-robin group sampling in `scripts/make_splits.py`)


These tasks are not simple label-balanced classification problems, so stratification is not meaningful.

---

For fixed (dataset, N, seed), the same JSONL split file is reused across model variants.

---

# 4. Split Engine (`src/splits.py`)

`src/splits.py` is the core sampling engine that executes the experimental policy defined in `configs/splits.json`.

It is **dataset-agnostic**:
it operates only on standardized examples and optional label fields.
All dataset-specific logic lives in adapters.

Its responsibility is precise:

> Given examples, Ns, and seeds — produce deterministic, nested splits.

### Index-Based Splitting (Key Design Choice)

The split engine outputs **indices into the standardized example list**, not copied examples:
`{ seed → { N → [indices] } }`.

This guarantees:
- no mutation of the underlying dataset
- no duplication beyond selecting positions
- stable example identity (via adapter-generated `id`)
- easy reproducibility (indices are lightweight + deterministic)

---

## Multi-Seed Output Structure

With `seeds = [0,1,2]` and `Ns = [16,32,64,128,256]`, it produces **three independent nested split families**, as explained in the previous section.

---

## How Nesting Is Enforced

Within a single seed:

```
N16 ⊂ N32 ⊂ N64 ⊂ N128 ⊂ N256
```

The engine guarantees this by:

* Generating one deterministic ordering per seed
* Taking prefixes for increasing N
* Validating with `check_nested()`

This ensures that increasing N measures *added supervision*, not reshuffling artifacts.

---

## Stratified vs Unstratified Behavior

Sampling behavior is controlled by `stratify`.

### Stratified Sampling (`stratify=True`)

Used for classification datasets:

* Financial PhraseBank
* SuperGLUE RTE

Purpose:

Prevent label imbalance from dominating small-N results.

Mechanism:

* Read labels from `label_key`
* Group indices by label
* Shuffle within each label bucket (seeded)
* Allocate per-label counts proportionally (via `allocate_counts`)
* Take per-label prefixes to preserve nesting

This preserves both **class balance** and **nested structure**.

---

### Unstratified Sampling (`stratify=False`)

Used for:

* GSM8K
* RAFT (task balancing uses `sampling_strategy: "round_robin_group"` implemented in `scripts/make_splits.py`)

Purpose:

When label balancing is not meaningful, perform deterministic random sampling.

Mechanism:

* Shuffle all indices once per seed
* Take global prefixes for each N

Nesting is still guaranteed.

---

## Validation Safeguards

The engine fails fast if invariants are violated.

It enforces:

* `Ns` sorted ascending
* `max(N) ≤ dataset size`
* Exact split sizes (`check_sizes`)
* Proper nesting (`check_nested`)
* Presence of `label_key` when stratified

Errors are explicit and immediate — preventing silent experimental corruption.

These invariants are also covered by unit tests in `tests/test_splits.py` (e.g., exact split sizes, nesting, and determinism across seeds).

---

## Design Principles

The split engine is intentionally minimal and reusable.

Key decisions:

* **Index-based splitting** (never mutates data)
* **Deterministic randomness per seed**
* **Explicit validation of invariants**
* **Zero dataset-specific branching**
* **Separated from orchestration (`scripts/make_splits.py`)**

This separation ensures experimental policy, data loading, and sampling mechanics remain independently testable and extendable.

---

## Materialized Split Files

`scripts/make_splits.py` then materializes those indices into actual few-shot datasets.

For each dataset, it writes:

- `N{N}_seed{seed}.jsonl`  
  The full standardized examples corresponding to that `(N, seed)`.

- `manifest.json`  
  Metadata describing how the splits were generated.

---

### Example JSONL Row (truncated)

Each line contains a fully standardized example:

```json
{"id":"fpb_sentences_allagree_train_000142","x":"Kesko Agro Eesti ... had net sales of 81 million euros in 2007 ...","y":"positive","label":"positive","meta":{"dataset":"financial_phrasebank","config":"sentences_allagree","split":"train","row_idx":142,"label_id":2}}
```
These are the exact examples consumed by downstream ICL or IFT experiments.

Example manifest.json (abbreviated)
```
{
  "dataset": { "name": "financial_phrasebank", ... },
  "splits": { "Ns": [16,32,64,128,256], "seeds": [0,1,2], "nested": true },
  "stats": { "num_examples_loaded": 2264, "label_counts": { ... } }
}
```
The manifest records:

* Dataset source information
* Global split settings
* Basic dataset statistics

This file serves as a reproducibility audit trail.

These files form the experimental contract:

For a fixed (dataset, N, seed), every model variant trains on the exact same JSONL examples.


# 5. Adapters (`src/adapters/`)

## Why Adapters Exist

Each dataset on Hugging Face has:

* Different column names
* Different label encodings
* Different schema formats
* Different splits and configs
* Sometimes custom dataset loaders

If we fed these raw datasets directly into the split engine, the engine would need dataset-specific logic — which would destroy modularity and reproducibility.

Instead, we isolate all dataset-specific quirks inside **adapters**.

Adapters are the abstraction layer that makes the entire pipeline possible.

---

## The Core Idea: Standardization

Every adapter converts its dataset into a **standardized schema**:

```python
{
  "id": str,        # stable identifier
  "x": str,         # model input text
  "y": str,         # target output (natural language)
  "label": str,     # used for stratified sampling (if applicable)
  "meta": {...}     # dataset-specific metadata
}
```

This transformation is the enabling abstraction behind the entire pipeline.

Because all datasets are converted into the same structure:

* `src/splits.py` never needs to know which dataset it is handling.
* Stratified sampling works identically across datasets.
* Round-robin grouping works identically across datasets.
* The split engine remains completely reusable.

Standardization allows:

> One split engine, many datasets.

Without it, every dataset would require custom branching logic.

---

## What Is Stratified Sampling?

Stratified sampling ensures that small few-shot subsets preserve the label distribution of the full dataset.

Instead of sampling purely at random, the split engine:

1. Groups examples by label
2. Shuffles within each group
3. Allocates examples proportionally
4. Combines them into a balanced subset

This prevents small N splits from being dominated by a single class.

Stratified sampling is only relevant for classification-style datasets.

---

## Adapter Responsibilities

Every adapter must:

1. Load from Hugging Face using `datasets.load_dataset`
2. Validate expected schema (fail loudly if mismatched)
3. Convert raw labels to natural-language verbalizers
4. Construct stable example IDs
5. Return a list of standardized examples

Adapters do **not**:

* Perform splitting
* Handle seeds
* Implement sampling strategies
* Write files

They are purely responsible for **data ingestion and normalization**.

---

## Dataset-Specific Notes

### Financial PhraseBank

* Multi-class sentiment classification.

* HF labels are integers `{0,1,2}`.

* Adapter maps them to:

  * `negative`
  * `neutral`
  * `positive`

* `label_map` is read from config (JSON keys converted to integers internally).

* **Stratified sampling is enabled.**

Design decision:
Convert numeric labels to natural language so both ICL and IFT operate in a text-generation framing.

---

### SuperGLUE RTE

* Binary textual entailment task.

* Two text fields: `premise` and `hypothesis`.

* Adapter concatenates into a structured input:

  ```
  Premise: ...
  Hypothesis: ...
  ```

* Labels mapped to:

  * `entailment`
  * `not_entailment`

* **Stratified sampling is enabled.**

Design decision:
Keep input structure explicit to improve prompt clarity and fine-tuning stability.

---

### GSM8K

* Free-form math reasoning.

* No meaningful discrete label for balancing.

* Adapter uses:

  * `question` → `x`
  * `answer` → `y`

* **Stratified sampling is disabled.**

Design decision:
We treat this as generative supervision rather than classification.

---

### RAFT

* Multi-task benchmark.

* Each Hugging Face config represents a separate task.

* Adapter loads all tasks and concatenates them.

* Adds required field:

  ```
  "task_name": str
  ```

* Round-robin grouping uses `task_name`.

* **Stratified sampling is disabled.**

Design decision:
Balance across tasks rather than across labels.
This prevents small-N splits from being dominated by a single RAFT task.

---

## Why This Layer Matters

Adapters create a clean boundary:

| Layer        | Responsibility                     |
| ------------ | ---------------------------------- |
| Adapter      | Normalize raw dataset              |
| Split Engine | Produce deterministic index splits |
| Script       | Orchestrate, write files, upload   |

Because of this separation:

* Adding a new dataset requires only writing a new adapter.
* Sampling logic never changes.
* Experimental policy remains centralized in `configs/splits.json`.

This keeps the system:

* Reproducible
* Extendable
* Mechanically clean
* Experimentally trustworthy
