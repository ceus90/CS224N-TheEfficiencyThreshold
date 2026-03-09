# External Benchmark Sources

This file documents every externally-sourced metric used in `0_shot_analysis.ipynb` (Section 12 onwards).
All numbers were retrieved on **2026-03-08**.

---

## Source 1 — Llama-3-8B-Instruct: Open LLM Leaderboard / HuggingFace Model Card

**URL:** https://llm.extractum.io/model/meta-llama%2FMeta-Llama-3-8B-Instruct,1dWHL42twQs8rZ0nw2Ybqc  
**Primary data origin:** HuggingFace OpenLLM Leaderboard v1 (original data aggregated from `opencompass` and official HF evaluations)  
**Model:** `meta-llama/Meta-Llama-3-8B-Instruct`  
**Retrieved:** 2026-03-08

| Benchmark | Score | Notes |
|---|---|---|
| GSM8K | 68.69% | 8-shot, standard greedy decoding |
| MMLU | 67.07% | 5-shot |
| ARC (Challenge) | 60.75% | 25-shot |
| HellaSwag | 78.55% | 10-shot |
| WinoGrande | 74.51% | 5-shot |
| TruthfulQA | 51.65% | 0-shot MC2 |
| IFEval (strict prompt) | 47.82% | 0-shot |
| BBH | 26.80% | 3-shot CoT |
| GPQA | 5.70% | 5-shot CoT |
| MMLU-Pro | 28.79% | 5-shot CoT |

**Notes:**
- These numbers are for the *instruct* (chat) model, not the base model.
- The GSM8K number (68.69%) uses 8-shot chain-of-thought prompting, which is more favorable than
  our 0-shot setup (where we measured 74.45%). The higher 0-shot result in our experiments may reflect
  prompt engineering differences or our use of vLLM.
- IFEval score of 47.82% is notably low, aligning with the model's known weakness on strict
  instruction-following tasks at this scale.

---

## Source 2 — Qwen3 Technical Report (Official)

**URL:** https://arxiv.org/abs/2505.09388  
**Citation:** Qwen Team. "Qwen3 Technical Report." arXiv:2505.09388 (2025).  
**PDF mirror used:** https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf  
**Retrieved:** 2026-03-08

### 2a. Qwen3-8B & Qwen3-4B — Post-Training (Instruct) Evaluation

From **Table 17** (Thinking Mode) and **Table 18** (Non-Thinking Mode) of the technical report.  
Non-thinking mode is the most directly comparable to our 0-shot experiments (which used `enable_thinking=False`).

#### Table 18 — Non-Thinking Mode (most comparable to our 0-shot runs)

| Benchmark | LLaMA-3.1-8B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-14B-Instruct | **Qwen3-4B** | **Qwen3-8B** |
|---|---|---|---|---|---|
| MMLU-Redux | 61.7% | 75.4% | 80.0% | **77.3%** | **79.5%** |
| GPQA-Diamond | 32.8% | 36.4% | 45.5% | **41.7%** | **39.3%** |
| MATH-500 | 54.8% | 77.6% | 83.4% | **84.8%** | **87.4%** |
| IFEval (strict prompt) | 75.0% | 71.2% | 81.0% | **81.2%** | **83.0%** |
| Arena-Hard | 30.1% | 52.0% | 68.3% | **66.2%** | **79.6%** |
| LiveBench 2024-11-25 | 26.0% | 34.9% | 42.2% | **48.4%** | **53.5%** |
| AIME'24 | 6.3% | 9.1% | 15.2% | **25.0%** | **29.1%** |
| LiveCodeBench v5 | 10.8% | 14.4% | 21.9% | **21.3%** | **22.8%** |
| BFCL v3 | 49.6% | 55.8% | 58.7% | **57.6%** | **60.2%** |
| AutoLogi | 30.9% | 42.9% | 57.4% | **76.3%** | **76.5%** |

#### Table 17 — Thinking Mode

| Benchmark | **Qwen3-4B (think)** | **Qwen3-8B (think)** |
|---|---|---|
| MMLU-Redux | 83.7% | 87.5% |
| GPQA-Diamond | 55.9% | 62.0% |
| MATH-500 | 97.0% | 97.4% |
| IFEval (strict prompt) | 81.9% | 85.0% |
| AIME'24 | 73.8% | 76.0% |
| LiveCodeBench v5 | 54.2% | 57.5% |
| BFCL v3 | 65.9% | 68.1% |

**Notes:**
- The "thinking" mode results show dramatically higher performance on math (MATH-500: 97% vs 87%)
  because the model engages in extended chain-of-thought reasoning before answering.
- Our 0-shot experiments used `enable_thinking=False` (non-thinking mode), so Table 18 is the
  most apples-to-apples comparison.
- The Qwen3 technical report does NOT include results on Financial PhraseBank, BoolQ, or RTE — these
  are not part of their standard evaluation suite.

### 2b. Qwen3-8B & Qwen3-4B — Pre-Training (Base Model) Evaluation

From **Table 6** (Qwen3-8B-Base) and **Table 7** (Qwen3-4B-Base).

#### Table 6 — Qwen3-8B-Base vs. Llama-3-8B-Base

| Benchmark | Llama-3-8B (base) | Qwen2.5-7B (base) | Qwen2.5-14B (base) | **Qwen3-8B (base)** |
|---|---|---|---|---|
| MMLU | 66.60% | 74.16% | 79.66% | **76.89%** |
| MMLU-Redux | 61.59% | 71.06% | 76.64% | **76.17%** |
| MMLU-Pro | 35.36% | 45.00% | 51.16% | **56.73%** |
| GSM8K (4-shot CoT) | 55.30% | 85.36% | 90.22% | **89.84%** |
| MATH (4-shot CoT) | 20.50% | 49.80% | 55.64% | **60.80%** |
| GPQA (5-shot CoT) | 25.80% | 36.36% | 32.83% | **44.44%** |
| BBH | 57.70% | 70.40% | 78.18% | **78.40%** |

#### Table 7 — Qwen3-4B-Base vs. baselines

| Benchmark | Gemma-3-4B (base) | Qwen2.5-3B (base) | Qwen2.5-7B (base) | **Qwen3-4B (base)** |
|---|---|---|---|---|
| MMLU | 59.51% | 65.62% | 74.16% | **72.99%** |
| GSM8K (4-shot CoT) | 43.97% | 79.08% | 85.36% | **87.79%** |
| MATH | 26.10% | 42.64% | 49.80% | **54.10%** |
| GPQA | 24.24% | 26.26% | 36.36% | **36.87%** |
| BBH | 51.70% | 56.30% | 70.40% | **72.59%** |

---

## Source 3 — Meta Llama 3 Announcement Blog

**URL:** https://ai.meta.com/blog/meta-llama-3/  
**Retrieved:** 2026-03-08

- Llama-3-8B-Instruct was released April 18, 2024.
- Trained on 15T tokens.
- Post-training: SFT + rejection sampling + PPO + DPO.
- Evaluation details at: https://github.com/meta-llama/llama3/blob/main/eval_details.md
- Blog confirms improvement over Llama 2 but does not provide specific benchmark numbers inline.

---

## Key Observations for Analysis

### GSM8K Gap: Reported vs. Measured

| Model | Official GSM8K (8-shot CoT) | Our 0-shot Measurement |
|---|---|---|
| Llama-3-8B-Instruct | 68.69% | **74.45%** |
| Qwen3-8B (base, 4-shot CoT) | 89.84% | **87.64%** (0-shot, instruct) |

- Our Llama-3-8B 0-shot result (74.45%) **exceeds** the official 8-shot CoT figure (68.69%).
  This is likely because the instruct model is better at following zero-shot instructions than
  extracting answers from few-shot demonstrations, and our prompt engineering may differ.
- Our Qwen3-8B 0-shot result (87.64%) closely matches the base model's 4-shot CoT score (89.84%),
  confirming the instruct model's strong zero-shot capability.

### BoolQ / RTE: Not in Official Evals
Neither Meta nor Qwen report BoolQ or RTE numbers for these exact model versions. Our measurements
fill a gap in the public benchmark record for these models on SuperGLUE tasks.

### Financial PhraseBank: Not in Official Evals
None of the official evaluation suites include Financial PhraseBank. Our results are novel
domain-specific benchmarks.

### RAFT: Baseline Not Available
No official 0-shot RAFT numbers exist for Qwen3-8B or Llama-3-8B-Instruct in the public literature.
Our RAFT evaluation is an original contribution.
