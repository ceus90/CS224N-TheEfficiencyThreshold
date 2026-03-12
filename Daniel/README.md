# CS224N — The Efficiency Threshold

This repo contains code + artifacts for a CS224N project studying **accuracy vs. efficiency trade-offs** across:

- **0-shot evaluation** (multiple tasks, multiple models)
- **PEFT** experiments (LoRA)
- **Analysis** scripts/notebooks that aggregate metrics and generate the plots in `src/analysis/`

Most “experiments” in this repo are captured as **Jupyter notebooks** under `src/0_shot/`.

---

## Quickstart

### What you need installed

There isn’t a pinned environment file in this repo yet, but the analysis code expects a standard scientific Python stack:

- Python 3.10+ (3.11 also works)
- Jupyter (for running notebooks)
- `numpy`, `pandas`, `matplotlib`, `scipy`

The evaluation notebooks reference vLLM-based runs in their filenames (e.g. `*_vllm.ipynb`) and typically assume access to:

- model weights (local or via a model hub)
- a GPU-capable environment (optional for just re-running analysis/plotting)

### Reproduce the analysis figures

From `CS224N-TheEfficiencyThreshold/src/analysis/`:

- **0-shot metrics aggregation + plots**: open and run `0_shot_analysis.ipynb`
- **Sampling / bootstrap sensitivity analysis**: run:

```bash
python few_shot_sample_analysis.py
```

That script reads checkpoint JSONL files under `src/0_shot/` and writes plots/tables into `src/analysis/sample_analysis_outputs/`.

---

## Repository structure

At a high level:

```text
CS224N-TheEfficiencyThreshold/
  README.md
  src/
    0_shot/         # 0-shot eval notebooks + metrics/checkpoints produced by those runs
    peft/           # parameter-efficient finetuning notebooks (LoRA)
    analysis/       # aggregation + plotting (paper figures live here)
```

### `src/0_shot/` — 0-shot evaluation runs + artifacts

This folder is organized primarily by **benchmark/task**, then **model**.

You’ll see artifacts like:

- `*_final_metrics.json`: per-run summary metrics used by `src/analysis/0_shot_analysis.ipynb`
- `*_checkpoint.jsonl`: per-example logs used by `src/analysis/few_shot_sample_analysis.py` to do
  subsampling + bootstrap analyses

Common tasks present in this repo:

- `financial_phrase_bank/`
- `gsm8k/`
- `superglue/boolq/`
- `superglue/rte/`
- `raft/` (contains per-subtask `train_results.json` plus overall `*_final_metrics.json`)
- `ifbench/` (instruction following; may include additional JSONL files like response logs)

Notebook naming convention is generally:

- `..._0shot.ipynb` / `..._0shot_vllm.ipynb`: run the 0-shot evaluation and write artifacts under the same folder.

### `src/peft/` — Parameter-efficient finetuning (PEFT)

PEFT experiments are split by method:

- `src/peft/lora/`

Each subfolder contains notebooks for specific tasks/models (e.g., IFBench).

### `src/analysis/` — Aggregation + plotting (paper figures)

This folder contains the scripts/notebooks that read the artifacts from `src/0_shot/` and produce figures:

- `0_shot_analysis.ipynb`: auto-discovers `*_final_metrics.json` under `../0_shot/` and generates a suite of plots
- `few_shot_sample_analysis.py`: “checkpoint sampling” study over JSONL logs

Generated outputs live in:

- `src/analysis/*.png`: top-level summary figures exported by the analysis notebook(s)
- `src/analysis/sample_analysis_outputs/`: sampling/bootstrapping plots and tables

---

## Notes on data

- **Repo-generated metrics** are stored as JSON/JSONL under `src/0_shot/` so the analysis is reproducible without re-running model inference.
