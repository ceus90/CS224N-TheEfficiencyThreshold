# CS224N: The Efficiency Threshold

## Project Overview
An investigation into the "Context-Weight Tradeoff" was conducted to evaluate the performance and efficiency differences between in-context learning (ICL) and parameter-efficient fine-tuning (PEFT), specifically Low-Rank Adaptation (LoRA). The precise data threshold where PEFT outperforms ICL under real-world latency and memory constraints was determined.

## Methodology
* Performance was evaluated across five diverse datasets: SuperGLUE RTE, BoolQ, Financial PhraseBank, RAFT ADE corpus, and GSM8K.
* Zero-shot baselines were established using vLLM for batched generation with greedy decoding.
* ICL and LoRA were evaluated across varying labeled data budgets, specifically $N\in\{16,32,64,128,256\}$.
* Additional architectures, including a GEPA optimization pipeline for dynamic prompt refinement, PeFT, and Representation Fine-Tuning (ReFT), were implemented and evaluated.
* Computational efficiency metrics, including latency, throughput, time per output token (TPOT), and peak GPU memory utilization (VRAM), were quantified for every configuration.

## Key Findings and Conclusions
* It was revealed that the optimal adaptation strategy is dictated by task structure rather than accuracy alone.
* For reasoning tasks, higher accuracy, faster token generation, and higher throughput are yielded by ICL.
* However, ICL scaling is constrained by finite context windows, and increased evaluation-time memory is required.
* Conversely, for structured classification tasks, comparable accuracy and lower inference-time VRAM are achieved by LoRA.
* Although slower generation is produced by LoRA due to shifted computational costs to training, stable inference costs are maintained without demonstration truncation risks.
* Across all evaluated benchmarks, the highest inference speed was consistently yielded by the Qwen3-4B model.
* Significant instability and erratic performance were exhibited by the GEPA pipeline on base models, while substantial efficiency gains but high task-specific volatility were demonstrated by ReFT.

## Repository Structure

The project work is segregated into individual main folders. The contributions and modules managed within each directory are detailed below:

### `Shreyas`
The following components and implementations were managed within this directory:
* A critical cloud-based pipeline was engineered to evaluate context-weight tradeoffs across advanced LLM paradigms, including LORA, ReFT via pyreft, GEPA, Zero-Shot and Few-Shots.
* Robust data preprocessing and stratified sampling were implemented for six diverse datasets.
* Crucial LORA training for five causal models was orchestrated.
* An extensive inference loop was constructed to benchmark accuracy, latency, and VRAM metrics.
* Automated scaling plots and comprehensive documentation were generated to ensure overall project success.

### `Abi`
The following components and implementations were managed within this directory:
* The experimental infrastructure used to run and evaluate the ICL and LoRA experiments across six models, five datasets, and multiple dataset splits was designed and implemented.
* Experimental work included data preprocessing, prompt construction, model evaluation, and instrumentation for measuring latency, throughput, and VRAM used in the reported results.

### `Daniel`
The following components and implementations were managed within this directory:
* Zero-Shot baselines were evaluated across all datasets for all models.
