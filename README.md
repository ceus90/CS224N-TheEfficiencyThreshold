# CS224N Project Proposal

## The Context–Weight Tradeoff

---

## Research Objective

This project investigates the **Context–Weight Tradeoff** to determine the optimal strategic switch point between:

* **In-Context Learning (ICL)**
* **Parameter-Efficient Fine-Tuning (PEFT)**

We test the hypothesis that **PEFT surpasses ICL accuracy at approximately N ≈ 100 examples**.

While **LoRA** alters the cost-benefit analysis of fine-tuning, recent work on **Reflective Prompt Evolution (GEPA)** suggests prompting alone may close the gap without weight updates.

We further hypothesize that:

* Increasing model scale (e.g., **Llama-3 vs. Qwen3**) shifts the crossover point.
* **Representation-level adaptation (ReFT)** can match LoRA accuracy at small N with lower latency.

This work addresses a gap in prior literature: identifying the specific data thresholds at which fine-tuning becomes viable under strict hardware constraints.

---

## Task & Data

### Primary Task

**Few-Shot Conditional Text Generation**

Given input context ( x ), predict target output ( y ), where adaptation is limited to either:

* A frozen-weight prompt containing N examples (ICL), or
* A low-rank weight update derived from the same N examples (PEFT).

---

### Datasets

We evaluate across five datasets probing distinct capabilities:

| Dataset              | Capability                     |
| -------------------- | ------------------------------ |
| GSM8K                | Mathematical reasoning         |
| SuperGLUE            | Natural language understanding |
| Financial PhraseBank | Sentiment classification       |
| IFBench              | Instruction following          |
| RAFT                 | Robustness                     |

---

### Data Regimes

We construct stratified subsets:

```
N ∈ {16, 32, 64, 128, 256}
```

* Identical JSON splits across all model variants
* Labels mapped to natural language verbalizers (e.g., "True", "False")
* ICL prompts formatted into structured dialogue

---

## Methodological Design

We stratify experiments by task complexity:

### Reasoning & Robustness (GSM8K, RAFT)

* Prioritize **ReFT**
* Hypothesis: representation intervention captures reasoning chains more efficiently than weight updates

### Sentiment & NLU (Financial PhraseBank, SuperGLUE)

* Prioritize **P-Tuning v2**
* Investigate embedding-level optimization under minimal latency

### Instruction Following (IFBench)

* Use **LoRA** as primary comparator
* Test whether representation methods match weight-based adaptation

---

## Models & Adaptation Techniques

### Model Families

* **Llama-3-8B**
* **Qwen3 (4B and 8B variants)**

This multi-model setup ensures efficiency thresholds are not architecture-specific artifacts.

---

### Adaptation Taxonomy

| Level                | Method      |
| -------------------- | ----------- |
| Text-level           | Prompting   |
| Embedding-level      | P-Tuning v2 |
| Representation-level | ReFT        |
| Weight-level         | LoRA        |

All experiments implemented using:

* Hugging Face Transformers
* PEFT Library
* PyReft (for ReFT)

---

## Baselines

### Zero-Shot

Establish lower-bound performance without in-context examples.

### Few-Shot Prompting (k = N)

Using Llama-3-8B-Instruct.

### GEPA (Reflective Prompt Evolution)

Strong optimized prompting baseline.

### LoRA (PEFT)

Using Llama-3-8B-Base to isolate learned parameter updates.

---

## Evaluation

### Quantitative Metrics

* Accuracy (Exact Match)
* Inference Latency (ms)
* Throughput (Tokens/sec)
* Peak VRAM usage

All measured under identical hardware and batching conditions.

---

### Interpretability Analysis

ReFT allows targeted interventions. We analyze whether:

* ReFT isolates semantic properties more effectively
* LoRA produces broader global adaptations

---

## Empirical Tradeoff Analysis

We generate a dual-axis plot:

```
Accuracy vs. Latency over N
```

We perform a **Scaling Analysis** across:

* Llama-3-8B
* Qwen3-4B
* Qwen3-8B

Goal: Determine whether model capacity shifts the optimal adaptation strategy.

---

## Ethical Considerations

* Lower barriers to fine-tuning may increase risk of malicious local deployment.
* PEFT may encode dataset-specific biases into model weights.
* We propose auditing stratified few-shot subsets before fine-tuning.

---

## Scope & Feedback Request

This project systematically analyzes the efficiency crossover point between:

* In-Context Learning
* LoRA
* P-Tuning v2
* ReFT

Across:

* Varying data regimes
* Varying model scales


---

## References

1. Liu et al., *Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning*, NeurIPS 2022
2. Liu et al., *P-Tuning v2*, ACL 2022
3. Wu et al., *ReFT: Representation Fine-Tuning*, arXiv 2024
4. Brown et al., *Language Models are Few-Shot Learners*, NeurIPS 2020
5. Hu et al., *LoRA*, ICLR 2022
6. Agrawal et al., *GEPA: Reflective Prompt Evolution*, arXiv 2025
7. Cobbe et al., *GSM8K*, arXiv 2021
8. Wang et al., *SuperGLUE*, NeurIPS 2019
9. Malo et al., *Financial PhraseBank*, 2014
10. Zhou et al., *IFBench*, arXiv 2023
11. Alex et al., *RAFT*, NeurIPS Datasets 2021
12. Dubey et al., *Llama 3*, arXiv 2024
13. Morris et al., *Learning to Reason in 13 Parameters*, arXiv 2026
14. Li & Liang, *Prefix-Tuning*, ACL 2021
