import modal

app = modal.App("cs224n-context-weight-tradeoff")

eval_image_zero_few = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.4.0", "torchvision", "torchaudio", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers==4.44.0", 
        "accelerate>=0.33.0",
        "bitsandbytes", 
        "datasets<4.0.0",
        "huggingface_hub",
        "matplotlib",
        "tqdm",
        "pandas",
        "scikit-learn"
    )
    .env({
        "NCCL_IGNORE_DISABLED_P2P": "1", 
        "NCCL_DEBUG": "ERROR",
        "PYTHONWARNINGS": "ignore",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True" 
    })
)

output_volume = modal.Volume.from_name("eval-report_zero_few", create_if_missing=True)

@app.function(
    image=eval_image_zero_few,
    cpu=16.0, 
    memory=131072,
    gpu="H100", 
    timeout=43200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": "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"})],
    volumes={"/workspace/reports_zero_few": output_volume}
)
def execute_tradeoff_pipeline():
    import time
    import torch
    import numpy as np
    import pandas as pd
    import os
    import logging
    import re
    import string
    import random
    import matplotlib.pyplot as plt
    from datasets import load_dataset, Dataset
    from huggingface_hub import login
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig
    )
    from matplotlib.backends.backend_pdf import PdfPages
    from tqdm.auto import tqdm
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(filename='/workspace/reports_zero_few/eval_execution.log', level=logging.INFO)
    print("[SYSTEM] Environment initialized. Beginning Context-Weight Tradeoff Pipeline...", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    MODELS = {
        "Qwen3-4B": {"id": "Qwen/Qwen3-4B", "cot": False},
        "Llama-3-8B": {"id": "meta-llama/Meta-Llama-3-8B", "cot": False},
        "Llama-3-8B-Instruct": {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "cot": False},
        "Qwen3-8B": {"id": "Qwen/Qwen3-8B", "cot": False},
        "Gemma-2-9B": {"id": "google/gemma-2-9b", "cot": False}
    }

    METHODS = ["zero_shot", "few_shot"]
    DATASETS = ["gsm8k", "superglue", "financial_phrasebank", "IFBench", "raft"]
    N_SAMPLES = [16, 32, 64, 128, 256]
    
    EVAL_SAMPLES = {
        "gsm8k": 250,
        "superglue": 250,
        "financial_phrasebank": 200,
        "IFBench": 250,
        "raft": 200
    }

    results = {
        ds: {m_name: {method: {metric: [] for metric in ["accuracy", "latency", "vram_gb", "throughput", "prompt_tokens"]}
             for method in METHODS} for m_name in MODELS.keys()} for ds in DATASETS
    }

    LABEL_MAPPINGS = {
        "superglue": {0: "entailment", 1: "not entailment"},
        "financial_phrasebank": {0: "negative", 1: "neutral", 2: "positive"},
        "raft": {1: "adequate", 2: "inadequate"} 
    }

    def format_prompt(example, dataset_name, use_cot=False):
        suffix = " Let's think step by step." if use_cot else ""
        if dataset_name == "gsm8k":
            return f"Question: {example.get('question', '')}{suffix}\nAnswer:"
        elif dataset_name == "superglue":
            return f"Premise: {example.get('premise', '')}\nHypothesis: {example.get('hypothesis', '')}{suffix}\nEntailment:"
        elif dataset_name == "financial_phrasebank":
            return f"Sentence: {example.get('sentence', '')}{suffix}\nSentiment:"
        elif dataset_name == "IFBench":
            return f"Instruction: {example.get('instruction', '')}\nInput: {example.get('input', '')}{suffix}\nResponse:"
        elif dataset_name == "raft":
            return f"Sentence: {example.get('Sentence', '')}{suffix}\nLabel:"
        else:
            txt = list(example.values())[0] if len(example) > 0 else ""
            return f"Input: {txt}{suffix}\nOutput:"

    def extract_generated_number(text):
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        return nums[-1] if nums else ""

    def get_expected_label(dataset_name, val):
        if dataset_name == "gsm8k":
            match = re.search(r'####\s*(-?\d+)', str(val))
            return match.group(1) if match else str(val).strip()
        if dataset_name in LABEL_MAPPINGS and isinstance(val, int):
            return LABEL_MAPPINGS[dataset_name].get(val, str(val))
        return str(val).strip()

    def check_correctness_exact(expected, generated, dataset_name):
        exp = str(expected).strip().lower()
        gen = str(generated).strip().lower()
        if dataset_name == "gsm8k":
            return exp == extract_generated_number(gen)
        
        translator = str.maketrans('', '', string.punctuation)
        exp_clean = exp.translate(translator).strip()
        
        gen_core = gen.split('\n')[0].lower()
        
        if dataset_name == "superglue":
            if gen_core.startswith("not entailment") or gen_core.endswith("not entailment"): predicted = "not entailment"
            elif gen_core.startswith("entailment") or gen_core.endswith("entailment"): predicted = "entailment"
            else: predicted = gen_core.translate(translator).strip()
            return exp_clean == predicted
            
        elif dataset_name == "raft":
            if gen_core.startswith("inadequate"): predicted = "inadequate"
            elif gen_core.startswith("adequate"): predicted = "adequate"
            else: predicted = gen_core.translate(translator).strip()
            return exp_clean == predicted
            
        elif dataset_name == "financial_phrasebank":
            if gen_core.startswith("positive"): predicted = "positive"
            elif gen_core.startswith("negative"): predicted = "negative"
            elif gen_core.startswith("neutral"): predicted = "neutral"
            else: predicted = gen_core.translate(translator).strip()
            return exp_clean == predicted
            
        gen_clean = gen_core.translate(translator).strip()
        return exp_clean == gen_clean

    def build_icl_prompt(train_ds, eval_item, dataset_name, use_cot, method, t_key, tokenizer, max_tokens=7500):
        base_prompt = format_prompt(eval_item, dataset_name, use_cot)
        if method == "zero_shot" or len(train_ds) == 0:
            return base_prompt
        
        context_str = ""
        for item in train_ds:
            actual_key = t_key
            if actual_key not in item:
                for k in item.keys():
                    if "label" in k.lower() or "answer" in k.lower() or "completion" in k.lower() or "target" in k.lower():
                        actual_key = k
                        break
            val = item[actual_key] if actual_key in item else ""
            
            added_example = format_prompt(item, dataset_name, use_cot) + " " + get_expected_label(dataset_name, val) + "\n\n"
            
            test_prompt = context_str + added_example + base_prompt
            token_count = len(tokenizer.encode(test_prompt))
            
            if token_count >= max_tokens:
                break
                
            context_str += added_example
            
        return context_str + base_prompt

    def prepare_data_stratified(dataset_name, n, seed=42):
        target_eval_samples = EVAL_SAMPLES.get(dataset_name, 150)
        
        if dataset_name == "gsm8k":
            ds_full_train = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
            ds_full_eval = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
            key = "answer"
            label_col = None 
        elif dataset_name == "superglue":
            ds_full_train = load_dataset("super_glue", "rte", split="train", trust_remote_code=True)
            ds_full_eval = load_dataset("super_glue", "rte", split="validation", trust_remote_code=True)
            key = "label"
            label_col = "label"
        elif dataset_name == "financial_phrasebank":
            ds_full = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
            split_ds = ds_full.train_test_split(test_size=0.2, seed=seed)
            ds_full_train, ds_full_eval = split_ds['train'], split_ds['test']
            key = "label"
            label_col = "label"
        elif dataset_name == "IFBench":
            ds_full = load_dataset("HuggingFaceH4/instruction-dataset", split="test", trust_remote_code=True)
            split_ds = ds_full.train_test_split(test_size=0.2, seed=seed)
            ds_full_train, ds_full_eval = split_ds['train'], split_ds['test']
            key = "completion"
            label_col = None
        else:
            ds_full = load_dataset("ought/raft", "ade_corpus_v2", split="train", trust_remote_code=True)
            split_ds = ds_full.train_test_split(test_size=0.2, seed=seed)
            ds_full_train, ds_full_eval = split_ds['train'], split_ds['test']
            key = "Label"
            label_col = "Label"

        df_train = ds_full_train.to_pandas()
        
        if label_col and label_col not in df_train.columns:
            for c in df_train.columns:
                if label_col.lower() in c.lower() or "label" in c.lower() or "target" in c.lower():
                    label_col = c
                    key = c
                    break

        if label_col and label_col in df_train.columns:
            num_classes = df_train[label_col].nunique()
            samples_per_class = max(1, n // num_classes)
            try:
                sampled_indices = df_train.groupby(label_col, group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), samples_per_class), random_state=seed)
                ).index.tolist()
                
                if len(sampled_indices) > n:
                    sampled_indices = sampled_indices[:n]
                elif len(sampled_indices) < n:
                    remaining = list(set(df_train.index) - set(sampled_indices))
                    needed = n - len(sampled_indices)
                    random.seed(seed)
                    sampled_indices += random.sample(remaining, min(needed, len(remaining)))
            except ValueError:
                sampled_indices = df_train.sample(n=min(n, len(df_train)), random_state=seed).index.tolist()
        else:
            sampled_indices = df_train.sample(n=min(n, len(df_train)), random_state=seed).index.tolist()
            
        train_ds = ds_full_train.select(sampled_indices)
        eval_ds = ds_full_eval.select(range(min(target_eval_samples, len(ds_full_eval))))
        return train_ds, eval_ds, key

    pbar = tqdm(total=len(MODELS)*len(DATASETS)*len(METHODS)*len(N_SAMPLES), desc="Evaluation Progress")
    
    for m_name, cfg in MODELS.items():
        m_id, use_cot = cfg["id"], cfg["cot"]
        
        try:
            torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(m_id, token=hf_token, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                m_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                attn_implementation="sdpa", 
                token=hf_token,
                trust_remote_code=True, 
                low_cpu_mem_usage=True
            )
        except Exception as e:
            print(f"[ERROR] Failed to load model {m_id}: {e}", flush=True)
            pbar.update(len(DATASETS)*len(METHODS)*len(N_SAMPLES))
            continue

        for d_name in DATASETS:
            for n in N_SAMPLES:
                train_data, ev_data, t_key = prepare_data_stratified(d_name, n)
                
                for method in METHODS:
                    if method == "zero_shot" and len(results[d_name][m_name][method]["accuracy"]) > 0:
                        results[d_name][m_name][method]["accuracy"].append(results[d_name][m_name][method]["accuracy"][-1])
                        results[d_name][m_name][method]["latency"].append(results[d_name][m_name][method]["latency"][-1])
                        results[d_name][m_name][method]["vram_gb"].append(results[d_name][m_name][method]["vram_gb"][-1])
                        results[d_name][m_name][method]["throughput"].append(results[d_name][m_name][method]["throughput"][-1])
                        results[d_name][m_name][method]["prompt_tokens"].append(results[d_name][m_name][method]["prompt_tokens"][-1])
                        pbar.update(1)
                        continue

                    print(f"[BENCHMARK] Model: {m_name} | Dataset: {d_name} | Method: {method} | N: {n}", flush=True)

                    base_model.eval()
                    tokenizer.padding_side = "left" 
                    
                    torch.cuda.reset_peak_memory_stats()
                    
                    correct, ms, tokens, total_prompt_tokens = 0, 0, 0, 0
                    ev_list = list(ev_data)
                    
                    if method == "zero_shot":
                        batch_size = 64
                    else:
                        if n <= 16: batch_size = 8
                        elif n <= 32: batch_size = 4
                        elif n <= 64: batch_size = 2
                        else: batch_size = 1
                    
                    if len(ev_list) > 0:
                        for idx in range(0, len(ev_list), batch_size):
                            batch = ev_list[idx:idx + batch_size]
                            p_txts = [build_icl_prompt(train_data if method == "few_shot" else [], i, d_name, use_cot, method, t_key, tokenizer) for i in batch]
                            
                            targets = []
                            for i in batch:
                                actual_key = t_key
                                if actual_key not in i:
                                    for k in i.keys():
                                        if "label" in k.lower() or "answer" in k.lower() or "completion" in k.lower() or "target" in k.lower():
                                            actual_key = k
                                            break
                                val = i[actual_key] if actual_key in i else ""
                                targets.append(get_expected_label(d_name, val))
                            
                            inputs = tokenizer(p_txts, return_tensors="pt", padding=True).to(base_model.device)
                            
                            prompt_len = inputs['input_ids'].shape[1]
                            
                            valid_prompt_tokens = (inputs['input_ids'] != tokenizer.pad_token_id).sum().item()
                            total_prompt_tokens += valid_prompt_tokens

                            torch.cuda.synchronize()
                            start = time.time()
                            
                            with torch.no_grad():
                                out = base_model.generate(**inputs, max_new_tokens=25, pad_token_id=tokenizer.pad_token_id)
                            
                            torch.cuda.synchronize()
                            ms += (time.time() - start) * 1000
                            
                            valid_gen_tokens = (out[:, prompt_len:] != tokenizer.pad_token_id).sum().item()
                            tokens += valid_gen_tokens
                            
                            decs = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
                            for exp, gen in zip(targets, decs):
                                if check_correctness_exact(exp, gen, d_name): correct += 1

                    vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    eval_len = max(1, len(ev_list))
                    
                    total_latency_ms = ms / eval_len
                    total_throughput = tokens / (ms / 1000) if ms > 0 else 0
                    avg_prompt_tokens = total_prompt_tokens / eval_len
                    
                    results[d_name][m_name][method]["accuracy"].append(correct / eval_len)
                    results[d_name][m_name][method]["latency"].append(total_latency_ms)
                    results[d_name][m_name][method]["vram_gb"].append(vram_gb)
                    results[d_name][m_name][method]["throughput"].append(total_throughput)
                    results[d_name][m_name][method]["prompt_tokens"].append(avg_prompt_tokens)

                    torch.cuda.empty_cache()
                    pbar.update(1)
        
        if 'base_model' in locals():
            del base_model
        torch.cuda.empty_cache()

    pbar.close()

    pdf_path = "/workspace/reports_zero_few/ZeroShot_FewShots_Report.pdf"
    metrics = ["accuracy", "latency", "vram_gb", "throughput", "prompt_tokens"]
    
    with PdfPages(pdf_path) as pdf:
        for d_name in DATASETS:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f'Context-Weight Tradeoff: {d_name}', fontsize=16, fontweight='bold')
            axes = axes.flatten()
            
            for i, mtr in enumerate(metrics):
                ax = axes[i]
                for m_name in MODELS.keys():
                    for method in METHODS:
                        y_vals = results[d_name][m_name][method][mtr]
                        if len(y_vals) > 0:
                            label = f"{m_name} ({method.replace('_', ' ').title()})"
                            ax.plot(N_SAMPLES[:len(y_vals)], y_vals, marker='o', label=label)
                
                ax.set_title(mtr.replace("_", " ").capitalize())
                ax.set_xlabel("Number of Few-Shot Samples")
                ax.set_ylabel(mtr.replace("_", " ").capitalize())
                ax.grid(True, linestyle='--', alpha=0.6)
            
            fig.delaxes(axes[5])
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.15), ncol=2, fontsize=10)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[COMPLETE] Tradeoff Report generated at: {pdf_path}", flush=True)

@app.local_entrypoint()
def main():
    print("Initiating cloud-based Context-Weight Tradeoff evaluation pipeline...", flush=True)
    execute_tradeoff_pipeline.remote()