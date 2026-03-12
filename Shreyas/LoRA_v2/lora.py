import modal

app = modal.App("cs224n-context-weight-tradeoff-no-fa")

eval_image_lora = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ninja-build", "build-essential")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install("packaging", "numpy", "wheel")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers>=4.48.0",
        "accelerate>=0.34.0",
        "datasets<4.0.0",
        "huggingface_hub",
        "matplotlib",
        "tqdm",
        "pandas",
        "scikit-learn",
        "peft",
        "trl>=0.12.0" 
    )
    .env({
        "NCCL_IGNORE_DISABLED_P2P": "1", 
        "NCCL_DEBUG": "ERROR",
        "PYTHONWARNINGS": "ignore"
    })
)

output_volume = modal.Volume.from_name("eval-reports_lora", create_if_missing=True)

@app.function(
    image=eval_image_lora,
    cpu=16.0,
    memory=131072,
    gpu="B200",
    timeout=43200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": "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"})],
    volumes={"/workspace/reports_lora": output_volume}
)
def execute_tradeoff_pipeline():
    import time
    import torch
    import numpy as np
    import pandas as pd
    import os
    import logging
    import re
    import gc
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from datasets import load_dataset
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from tqdm.auto import tqdm
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(filename='/workspace/reports_lora/eval_execution.log', level=logging.INFO)
    print("[SYSTEM] Environment initialized. Migrated SFTConfig max_length parameter...", flush=True)

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

    DATASETS = ["gsm8k", "superglue", "financial_phrasebank", "IFBench", "raft"]
    N_SAMPLES = [16, 32, 64, 128, 256]
    EVAL_SAMPLES = {"gsm8k": 250, "superglue": 250, "financial_phrasebank": 200, "IFBench": 250, "raft": 200}

    results = {ds: {m: {"lora": {mtr: [] for mtr in ["accuracy", "latency", "vram_gb", "throughput", "prompt_tokens"]}} for m in MODELS} for ds in DATASETS}

    def format_prompt(example, dataset_name, use_cot=False):
        suffix = " Let's think step by step." if use_cot else ""
        if dataset_name == "gsm8k": return f"Question: {example['question']}{suffix}\nAnswer:"
        if dataset_name == "superglue": return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}{suffix}\nEntailment:"
        if dataset_name == "financial_phrasebank": return f"Sentence: {example['sentence']}{suffix}\nSentiment:"
        if dataset_name == "IFBench": return f"Instruction: {example.get('instruction', '')}\nInput: {example.get('input', '')}{suffix}\nResponse:"
        if dataset_name == "raft": return f"Sentence: {example['Sentence']}{suffix}\nLabel:"
        return f"Input: {list(example.values())[0]}{suffix}\nOutput:"

    def get_expected_label(dataset_name, val):
        if dataset_name == "gsm8k":
            match = re.search(r'####\s*(-?\d+)', str(val))
            return match.group(1) if match else str(val).strip()
        return str(val).strip()

    def check_correctness_exact(expected, generated, dataset_name):
        exp = str(expected).strip().lower()
        gen = str(generated).strip().lower()
        if dataset_name == "gsm8k":
            nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', gen)
            return exp == (nums[-1] if nums else "")
        return exp in gen.split('\n')[0].lower()

    def prepare_data_stratified(dataset_name, n, seed=42):
        target_eval = EVAL_SAMPLES.get(dataset_name, 150)
        if dataset_name == "gsm8k":
            ds_t = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
            ds_e = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
            key = "answer"
        elif dataset_name == "superglue":
            ds_t = load_dataset("super_glue", "rte", split="train", trust_remote_code=True)
            ds_e = load_dataset("super_glue", "rte", split="validation", trust_remote_code=True)
            key = "label"
        elif dataset_name == "financial_phrasebank":
            ds_split = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True).train_test_split(test_size=0.2, seed=seed)
            ds_t, ds_e = ds_split['train'], ds_split['test']
            key = "label"
        elif dataset_name == "IFBench":
            ds_split = load_dataset("yahma/alpaca-cleaned", split="train", trust_remote_code=True).train_test_split(test_size=0.2, seed=seed)
            ds_t, ds_e = ds_split['train'], ds_split['test']
            key = "output"
        elif dataset_name == "raft":
            ds_split = load_dataset("ought/raft", "overruling", split="train", trust_remote_code=True).train_test_split(test_size=0.2, seed=seed)
            ds_t, ds_e = ds_split['train'], ds_split['test']
            key = "Label"
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not configured.")
        
        train_ds = ds_t.shuffle(seed=seed).select(range(min(n, len(ds_t))))
        eval_ds = ds_e.select(range(min(target_eval, len(ds_e))))
        return train_ds, eval_ds, key

    pbar = tqdm(total=len(MODELS)*len(DATASETS)*len(N_SAMPLES), desc="Evaluation Progress")
    
    for m_name, cfg in MODELS.items():
        m_id, use_cot = cfg["id"], cfg["cot"]
        tokenizer = AutoTokenizer.from_pretrained(m_id, token=hf_token)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            m_id, device_map="auto", attn_implementation="sdpa", 
            token=hf_token, dtype=torch.bfloat16
        )

        for d_name in DATASETS:
            for n in N_SAMPLES:
                train_data, ev_data, t_key = prepare_data_stratified(d_name, n)
                
                peft_config = LoraConfig(
                    r=32, lora_alpha=64, 
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
                )
                model = get_peft_model(base_model, peft_config)

                def formatting_prompts_func(example):
                    return f"{format_prompt(example, d_name, use_cot)} {get_expected_label(d_name, example[t_key])}{tokenizer.eos_token}"

                sft_config = SFTConfig(
                    output_dir="/tmp/lora",
                    max_length=1024,
                    per_device_train_batch_size=32,
                    num_train_epochs=3,
                    bf16=True,
                    tf32=True,
                    report_to="none",
                    dataloader_num_workers=16,
                    gradient_checkpointing=True,
                    dataset_text_field="text" 
                )

                trainer = SFTTrainer(
                    model=model, 
                    train_dataset=train_data, 
                    formatting_func=formatting_prompts_func,
                    processing_class=tokenizer,
                    args=sft_config
                )
                trainer.train()

                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                correct, ms, tokens, p_tokens = 0, 0, 0, 0
                model.eval()
                tokenizer.padding_side = "left"
                
                batch_size = 64 
                ev_list = list(ev_data)
                for i in range(0, len(ev_list), batch_size):
                    batch = ev_list[i:i+batch_size]
                    p_txts = [format_prompt(x, d_name, use_cot) for x in batch]
                    inputs = tokenizer(p_txts, return_tensors="pt", padding=True).to("cuda")
                    
                    p_tokens += inputs['attention_mask'].sum().item()
                    start = time.time()
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model.generate(**inputs, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id)
                    ms += (time.time() - start) * 1000
                    
                    gen_toks = out[:, inputs['input_ids'].shape[1]:]
                    tokens += (gen_toks != tokenizer.pad_token_id).sum().item()
                    
                    decs = tokenizer.batch_decode(gen_toks, skip_special_tokens=True)
                    for x, gen in zip(batch, decs):
                        if check_correctness_exact(get_expected_label(d_name, x[t_key]), gen, d_name): correct += 1

                vram = torch.cuda.max_memory_allocated() / (1024**3)
                res = results[d_name][m_name]["lora"]
                res["accuracy"].append(correct / len(ev_list))
                res["latency"].append(ms / len(ev_list))
                res["vram_gb"].append(vram)
                res["throughput"].append(tokens / (ms / 1000) if ms > 0 else 0)
                res["prompt_tokens"].append(p_tokens / len(ev_list))

                base_model = model.unload()
                pbar.update(1)

    pdf_path = "/workspace/reports_lora/evaluation_metrics.pdf"
    with PdfPages(pdf_path) as pdf:
        for d_name in DATASETS:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('tight')
            ax.axis('off')
            ax.set_title(f"Evaluation Metrics Summary - Dataset: {d_name}", fontsize=14, fontweight='bold', pad=20)
            
            table_data = []
            columns = ["Model", "Samples (N)", "Accuracy", "Latency (ms)", "VRAM (GB)", "Throughput (tok/s)", "Prompt Tokens"]
            
            for m_name in MODELS:
                res = results[d_name][m_name]["lora"]
                for idx, n in enumerate(N_SAMPLES):
                    if idx < len(res["accuracy"]):
                        row = [
                            m_name,
                            n,
                            f"{res['accuracy'][idx]:.4f}",
                            f"{res['latency'][idx]:.2f}",
                            f"{res['vram_gb'][idx]:.2f}",
                            f"{res['throughput'][idx]:.2f}",
                            f"{res['prompt_tokens'][idx]:.1f}"
                        ]
                        table_data.append(row)
            
            if table_data:
                table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        metrics_to_plot = [
            ("accuracy", "Accuracy"), 
            ("latency", "Latency (ms)"), 
            ("vram_gb", "VRAM (GB)"), 
            ("throughput", "Throughput (tok/s)"), 
            ("prompt_tokens", "Prompt Tokens")
        ]
        
        for m_name in MODELS:
            for metric_key, metric_label in metrics_to_plot:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.set_title(f"{metric_label} Scaling - Model: {m_name}", fontsize=14, fontweight='bold')
                ax.set_xlabel("Samples (N)", fontsize=12)
                ax.set_ylabel(metric_label, fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                for d_name in DATASETS:
                    res = results[d_name][m_name]["lora"]
                    metric_data = res[metric_key]
                    if metric_data:
                        valid_len = min(len(N_SAMPLES), len(metric_data))
                        ax.plot(N_SAMPLES[:valid_len], metric_data[:valid_len], marker='o', linewidth=2, label=d_name)
                
                ax.legend(title="Datasets")
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"[COMPLETE] LoRA evaluation finished. Metrics written to {pdf_path}", flush=True)

@app.local_entrypoint()
def main():
    execute_tradeoff_pipeline.remote()