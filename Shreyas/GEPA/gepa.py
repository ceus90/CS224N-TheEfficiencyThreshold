import modal

app = modal.App("cs224n-gepa-final-optimized")

eval_image_gepa = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ninja-build", "build-essential")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        "packaging", "numpy", "torch", "wheel", "transformers>=4.44.0",
        "accelerate>=0.33.0", "datasets<4.0.0", "huggingface_hub",
        "matplotlib", "tqdm", "pandas", "scikit-learn", "fpdf"
    )
    .env({
        "NCCL_IGNORE_DISABLED_P2P": "1", 
        "NCCL_DEBUG": "ERROR",
        "PYTHONWARNINGS": "ignore"
    })
)

output_volume = modal.Volume.from_name("eval-reports_gepa", create_if_missing=True)

@app.function(
    image=eval_image_gepa,
    cpu=16.0,
    memory=131072,
    gpu="B200",
    timeout=43200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": "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"})],
    volumes={"/workspace/reports_gepa": output_volume}
)
def execute_tradeoff_pipeline():
    import time
    import torch
    import os
    import logging
    import re
    import string
    import pandas as pd
    import matplotlib.pyplot as plt
    from datasets import load_dataset
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fpdf import FPDF
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(filename='/workspace/reports_gepa/eval_execution.log', level=logging.INFO)
    
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

    DATASETS = ["gsm8k", "superglue", "financial_phrasebank", "raft"]
    N_SAMPLES = [16, 32, 64, 128, 256]
    
    EVAL_SAMPLES = {
        "gsm8k": 250, "superglue": 250, "financial_phrasebank": 200, "raft": 200
    }

    def format_prompt_with_instruction(instruction, example, dataset_name, use_cot=False):
        suffix = " Please provide the final answer after 'Final Answer:'." if dataset_name == "gsm8k" else ""
        header = f"System: {instruction}\n"
        if dataset_name == "gsm8k":
            return f"{header}Question: {example['question']}{suffix}\nAnswer:"
        elif dataset_name == "superglue":
            return f"{header}Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nEntailment (entailment, contradiction, or neutral):"
        elif dataset_name == "financial_phrasebank":
            return f"{header}Sentence: {example['sentence']}\nSentiment (positive, neutral, or negative):"
        elif dataset_name == "raft":
            return f"{header}Sentence: {example['Sentence']}\nLabel:"
        return f"{header}Input: {list(example.values())[0]}\nOutput:"

    def extract_generated_number(text):
        if "final answer" in text.lower():
            text = text.lower().split("final answer")[-1]
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        return nums[-1].replace(',', '') if nums else ""

    def get_expected_label(dataset_name, val):
        val_str = str(val).strip()
        if dataset_name == "gsm8k":
            match = re.search(r'####\s*(-?\d+)', val_str)
            return match.group(1).replace(',', '') if match else val_str
        elif dataset_name == "financial_phrasebank":
            mapping = {"0": "negative", "1": "neutral", "2": "positive"}
            return mapping.get(val_str, val_str.lower())
        elif dataset_name == "superglue":
            mapping = {"0": "entailment", "1": "contradiction", "2": "neutral"}
            return mapping.get(val_str, val_str.lower())
        return val_str.lower()

    def check_correctness_exact(expected, generated, dataset_name):
        exp = str(expected).strip().lower()
        gen = str(generated).strip().lower()
        
        if dataset_name == "gsm8k":
            return exp == extract_generated_number(gen)
        
        gen_clean = gen.split('\n')[0].translate(str.maketrans('', '', string.punctuation)).strip()
        words = gen_clean.split()
        
        if not words: return False
        return exp in words

    def GEPA_optimize(base_model, tokenizer, train_data, dataset_name, use_cot, t_key):
        current_instruction = "You are a highly accurate classifier. Output only the label."
        train_subset = list(train_data)[:16]
        
        p_txts = [format_prompt_with_instruction(current_instruction, i, dataset_name, use_cot) for i in train_subset]
        targets = [get_expected_label(dataset_name, i[t_key]) for i in train_subset]
        
        inputs = tokenizer(p_txts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = base_model.generate(**inputs, max_new_tokens=30, max_length=None, pad_token_id=tokenizer.eos_token_id)
        
        decs = tokenizer.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        failures = [f"Input: {p_txts[i]}\nGot: {decs[i]}\nExpected: {targets[i]}" for i in range(len(decs)) if not check_correctness_exact(targets[i], decs[i], dataset_name)]
        
        if failures:
            reflect_prompt = f"System: Analyze the error and provide a better instruction.\nError: {failures[0]}\nNew Instruction:"
            ref_inputs = tokenizer(reflect_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                ref_out = base_model.generate(**ref_inputs, max_new_tokens=50, max_length=None, pad_token_id=tokenizer.eos_token_id)
            new_inst = tokenizer.decode(ref_out[0][ref_inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            return new_inst if len(new_inst) > 5 else current_instruction
        return current_instruction

    def prepare_data_stratified(dataset_name, n):
        target_eval_samples = EVAL_SAMPLES.get(dataset_name, 150)
        if dataset_name == "gsm8k":
            ds_train = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
            ds_eval = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
            key = "answer"
        elif dataset_name == "financial_phrasebank":
            ds_train = load_dataset("financial_phrasebank", "sentences_allagree", split="train[:80%]", trust_remote_code=True)
            ds_eval = load_dataset("financial_phrasebank", "sentences_allagree", split="train[80%:]", trust_remote_code=True)
            key = "label"
        elif dataset_name == "superglue":
            ds_train = load_dataset("super_glue", "cb", split="train", trust_remote_code=True)
            ds_eval = load_dataset("super_glue", "cb", split="validation", trust_remote_code=True)
            key = "label"
        elif dataset_name == "raft":
            ds_train = load_dataset("OysterQA/raft", "nutrition5k", split="train", trust_remote_code=True)
            ds_eval = load_dataset("OysterQA/raft", "nutrition5k", split="test", trust_remote_code=True)
            key = "Label"

        train_ds = ds_train.shuffle(seed=42).select(range(min(n, len(ds_train))))
        eval_ds = ds_eval.select(range(min(target_eval_samples, len(ds_eval))))
        return train_ds, eval_ds, key

    collected_metrics = []

    for m_name, cfg in MODELS.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(cfg["id"], token=hf_token)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
            base_model = AutoModelForCausalLM.from_pretrained(
                cfg["id"], device_map="cuda", attn_implementation="eager", 
                token=hf_token, dtype=torch.bfloat16
            )
            base_model = torch.compile(base_model)
            
            for d_name in DATASETS:
                for n in N_SAMPLES:
                    train_data, ev_data, t_key = prepare_data_stratified(d_name, n)
                    best_inst = GEPA_optimize(base_model, tokenizer, train_data, d_name, cfg["cot"], t_key)
                    
                    correct = 0
                    batch_size = 32
                    ev_list = list(ev_data)
                    
                    max_vram, total_latency, total_gen_tokens, total_prompt_tokens = 0, 0, 0, 0
                    
                    for idx in range(0, len(ev_list), batch_size):
                        batch = ev_list[idx:idx + batch_size]
                        p_txts = [format_prompt_with_instruction(best_inst, i, d_name, cfg["cot"]) for i in batch]
                        targets = [get_expected_label(d_name, i[t_key]) for i in batch]
                        inputs = tokenizer(p_txts, return_tensors="pt", padding=True).to("cuda")
                        
                        torch.cuda.reset_peak_memory_stats()
                        start_time = time.time()
                        
                        with torch.no_grad():
                            out = base_model.generate(**inputs, max_new_tokens=40, max_length=None, pad_token_id=tokenizer.eos_token_id)
                        
                        total_latency += (time.time() - start_time)
                        total_prompt_tokens += inputs['input_ids'].numel()
                        total_gen_tokens += (out.shape[1] - inputs['input_ids'].shape[1]) * out.shape[0]
                        max_vram = max(max_vram, torch.cuda.max_memory_allocated() / (1024**3))
                        
                        decs = tokenizer.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        for exp, gen in zip(targets, decs):
                            if check_correctness_exact(exp, gen, d_name): correct += 1
                    
                    accuracy = correct / len(ev_list)
                    throughput = total_gen_tokens / total_latency if total_latency > 0 else 0
                    
                    collected_metrics.append({
                        "model": m_name, "dataset": d_name, "samples": n,
                        "accuracy": accuracy, "latency": total_latency,
                        "vram_gb": max_vram, "throughput": throughput,
                        "prompt_tokens": total_prompt_tokens
                    })
            del base_model
            torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error processing {m_name}: {str(e)}")

    try:
        df = pd.DataFrame(collected_metrics)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Corrected GEPA Evaluation Report", ln=1, align="C")
        
        metrics = ['accuracy', 'latency', 'vram_gb', 'throughput', 'prompt_tokens']
        for m in metrics:
            for mod in df['model'].unique():
                plt.figure(figsize=(10, 6))
                df_mod = df[df['model'] == mod]
                for ds in df_mod['dataset'].unique():
                    df_ds = df_mod[df_mod['dataset'] == ds].sort_values('samples')
                    plt.plot(df_ds['samples'], df_ds[m], marker='o', label=ds)
                
                plt.title(f'{m.capitalize()} - {mod}')
                plt.xlabel('Samples (N)'); plt.ylabel(m); plt.legend(); plt.grid(True)
                img_path = f"/workspace/reports_gepa/{m}_{mod.replace('/','_')}.png"
                plt.savefig(img_path); plt.close()
                pdf.add_page(); pdf.image(img_path, x=10, y=20, w=190)
            
        pdf.output("/workspace/reports_gepa/evaluation_metrics.pdf")
    except Exception as e:
        logging.error(f"PDF Error: {str(e)}")

@app.local_entrypoint()
def main():
    execute_tradeoff_pipeline.remote()