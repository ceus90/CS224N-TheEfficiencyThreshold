import modal

app = modal.App("cs224n-reft-efficiency")

reft_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ninja-build", "build-essential")
    .pip_install(
        "transformers>=4.44.0", 
        "huggingface_hub",
        "matplotlib",
        "datasets<4.0.0",
        "accelerate",
        "trl",
        "peft",
        "bitsandbytes",
        "tqdm",
        "nnsight",
        "vllm>=0.7.0" 
    )
    .pip_install("git+https://github.com/stanfordnlp/pyreft.git")
    .env({
        "NCCL_IGNORE_DISABLED_P2P": "1", 
        "NCCL_DEBUG": "ERROR",
        "PYTHONWARNINGS": "ignore",
        "VLLM_LOGGING_LEVEL": "ERROR"
    })
)

output_volume = modal.Volume.from_name("reft-reports", create_if_missing=True)

@app.function(
    image=reft_image,
    cpu=4.0,
    memory=65536,
    gpu="B200",
    timeout=43200,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": "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"})],
    volumes={"/workspace/reports": output_volume}
)
def execute_reft_pipeline():
    import os
    import re
    import time
    import warnings
    import torch
    import pyreft
    import pyvene.models.constants as pyvene_constants
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
    from datasets import load_dataset
    from huggingface_hub import login
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    print("[SYSTEM] Environment initialized. Beginning ReFT pipeline...", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    MODELS = {
        "Llama-3-8B": {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "cot": False},
        "Llama-3-8B-Instruct": {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "cot": False},
        "Qwen3-8B": {"id": "Qwen/Qwen3-8B", "cot": False},
        "Qwen3-8B-Thinking": {"id": "Qwen/Qwen3-8B", "cot": True},
        "Gemma-2-9B": {"id": "google/gemma-2-9b", "cot": False},
        "Gemma-2-9B-IT": {"id": "google/gemma-2-9b-it", "cot": False}
    }

    DATASETS = ["gsm8k", "superglue", "financial_phrasebank", "dolly15k", "raft"]
    N_SAMPLES = [16, 32, 64, 128, 256, 512]
    
    EVAL_SAMPLES = {
        "gsm8k": 250,
        "superglue": 250,
        "financial_phrasebank": 200,
        "dolly15k": 250,
        "raft": 200
    }

    results = {
        ds: {m_name: {metric: [] for metric in ["accuracy", "latency", "vram_pct", "throughput", "train_time", "tpot"]}
             for m_name in MODELS.keys()} for ds in DATASETS
    }

    LABEL_MAPPINGS = {
        "superglue": {0: "entailment", 1: "not entailment"},
        "financial_phrasebank": {0: "negative", 1: "neutral", 2: "positive"},
        "raft": {1: "adequate", 2: "inadequate"} 
    }

    def format_prompt(example, dataset_name, use_cot=False):
        suffix = " Let's think step by step." if use_cot else ""
        if dataset_name == "gsm8k":
            return f"Question: {example['question']}{suffix}\nAnswer:"
        elif dataset_name == "superglue":
            return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}{suffix}\nEntailment:"
        elif dataset_name == "financial_phrasebank":
            return f"Sentence: {example['sentence']}{suffix}\nSentiment:"
        elif dataset_name == "dolly15k":
            ctx = f"\nContext: {example['context']}" if example.get('context') else ""
            return f"Instruction: {example['instruction']}{ctx}{suffix}\nResponse:"
        elif dataset_name == "raft":
            return f"Sentence: {example['Sentence']}{suffix}\nLabel:"
        else:
            txt = list(example.values())[0]
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

    def check_correctness(expected, generated, dataset_name):
        exp = str(expected).strip().lower()
        gen = str(generated).strip().lower()
        
        if dataset_name == "gsm8k":
            return exp == extract_generated_number(gen)

        if exp == "adequate" and "inadequate" in gen:
            return False
        if exp == "entailment" and "not entailment" in gen:
            return False
            
        return exp in gen

    def prepare_data(dataset_name, n, tokenizer, model, use_cot):
        target_eval_samples = EVAL_SAMPLES.get(dataset_name, 150)
        
        if dataset_name == "gsm8k":
            ds_full_train = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
            ds_full_eval = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
            t_n, e_n = min(n, len(ds_full_train)), min(target_eval_samples, len(ds_full_eval))
            train, eval_ds = ds_full_train.select(range(t_n)), ds_full_eval.select(range(e_n))
            key = "answer"
        elif dataset_name == "superglue":
            ds_full_train = load_dataset("super_glue", "rte", split="train", trust_remote_code=True)
            ds_full_eval = load_dataset("super_glue", "rte", split="validation", trust_remote_code=True)
            t_n, e_n = min(n, len(ds_full_train)), min(target_eval_samples, len(ds_full_eval))
            train = ds_full_train.select(range(t_n))
            eval_ds = ds_full_eval.select(range(e_n))
            key = "label"
        elif dataset_name == "financial_phrasebank":
            ds_full = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
            t_n = min(n, int(len(ds_full) * 0.8))
            e_n = min(target_eval_samples, len(ds_full) - t_n)
            train, eval_ds = ds_full.select(range(t_n)), ds_full.select(range(t_n, t_n + e_n))
            key = "label"
        elif dataset_name == "dolly15k":
            ds_full = load_dataset("databricks/databricks-dolly-15k", split="train", trust_remote_code=True)
            t_n = min(n, int(len(ds_full) * 0.8))
            e_n = min(target_eval_samples, len(ds_full) - t_n)
            train, eval_ds = ds_full.select(range(t_n)), ds_full.select(range(t_n, t_n + e_n))
            key = "response"
        else:
            ds_full = load_dataset("ought/raft", "ade_corpus_v2", split="train", trust_remote_code=True)
            t_n = min(n, int(len(ds_full) * 0.8))
            e_n = min(target_eval_samples, len(ds_full) - t_n)
            train, eval_ds = ds_full.select(range(t_n)), ds_full.select(range(t_n, t_n + e_n))
            key = "Label"

        prompts = [format_prompt(i, dataset_name, use_cot) for i in train]
        resps = [get_expected_label(dataset_name, i[key]) for i in train]
        dm = pyreft.make_last_position_supervised_data_module(tokenizer=tokenizer, model=model, inputs=prompts, outputs=resps)
        return dm, eval_ds, key

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    pbar = tqdm(total=len(MODELS)*len(DATASETS)*len(N_SAMPLES), desc="Benchmark Progress")
    
    base_model = None

    for m_name, cfg in MODELS.items():
        m_id, use_cot = cfg["id"], cfg["cot"]
        
        try:
            base_model = None
            torch.cuda.empty_cache()
            
            tokenizer = AutoTokenizer.from_pretrained(m_id, token=hf_token, model_max_length=2048)
            tokenizer.padding_side, tokenizer.pad_token = "left", tokenizer.eos_token
            
            is_specialized = any(x in m_id.lower() for x in ["fp8", "dms"])
            load_cfg = None if is_specialized else bnb_config
            
            base_model = AutoModelForCausalLM.from_pretrained(
                m_id, quantization_config=load_cfg, device_map="auto",
                attn_implementation="sdpa", token=hf_token,
                dtype=torch.bfloat16
            )
            
            if not is_specialized:
                from peft import prepare_model_for_kbit_training
                base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)
                
        except Exception as e:
            print(f"[ERROR] Failed to load model {m_id}: {e}", flush=True)
            pbar.update(len(DATASETS) * len(N_SAMPLES))
            continue

        for d_name in DATASETS:
            for n in N_SAMPLES:
                print(f"[BENCHMARK] Active Model: {m_name} | Dataset: {d_name} | Samples (N): {n}", flush=True)
                torch.cuda.reset_peak_memory_stats()
                
                layer_idx = max(0, base_model.config.num_hidden_layers - 4)
                
                explicit_component_path = f"model.layers[{layer_idx}].output"
                
                reft_config = pyreft.ReftConfig(representations={
                    "layer": layer_idx, 
                    "component": explicit_component_path, 
                    "low_rank_dimension": 4,
                    "intervention": pyreft.LoreftIntervention(
                        embed_dim=base_model.config.hidden_size, 
                        low_rank_dimension=4
                    )
                })
                
                reft_model = pyreft.get_reft_model(base_model, reft_config)
                reft_model.set_device("cuda")

                dm, ev_data, t_key = prepare_data(d_name, n, tokenizer, base_model, use_cot)
                
                trainer = pyreft.ReftTrainerForCausalLM(
                    model=reft_model, processing_class=tokenizer,
                    args=TrainingArguments(
                        output_dir="./tmp", num_train_epochs=25, 
                        per_device_train_batch_size=16, learning_rate=2e-4, 
                        bf16=True, tf32=True, report_to="none",
                        gradient_checkpointing=False 
                    ), **dm
                )
                
                train_start_time = time.time()
                t_res = trainer.train()
                train_duration = time.time() - train_start_time

                reft_model.eval()
                correct, ms, tokens, ev_list = 0, 0, 0, list(ev_data)
                
                if len(ev_list) > 0:
                    for idx in range(0, len(ev_list), 16):
                        batch = ev_list[idx:idx + 16]
                        p_txts = [format_prompt(i, d_name, use_cot) for i in batch]
                        targets = [get_expected_label(d_name, i[t_key]) for i in batch]
                        inputs = tokenizer(p_txts, return_tensors="pt", padding=True).to("cuda")
                        u_locs = [[[inputs['input_ids'].shape[1] - 1] for _ in range(len(reft_model.interventions))] for _ in range(len(batch))]

                        torch.cuda.synchronize()
                        start = time.time()
                        
                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            _, out = reft_model.generate(
                                inputs, 
                                unit_locations={"sources->base": (None, u_locs)}, 
                                intervene_on_prompt=True, 
                                max_new_tokens=60 if use_cot else 25, 
                                pad_token_id=tokenizer.eos_token_id
                            )
                        
                        torch.cuda.synchronize()
                        ms += (time.time() - start) * 1000
                        tokens += (out.shape[1] - inputs['input_ids'].shape[1]) * len(batch)
                        
                        decs = tokenizer.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        
                        for exp, gen in zip(targets, decs):
                            if check_correctness(exp, gen, d_name): 
                                correct += 1

                vram, eval_len = (torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100, max(1, len(ev_list))
                
                total_latency_ms = ms / eval_len
                total_throughput = tokens / (ms / 1000) if ms > 0 else 0
                avg_tpot = (ms / tokens) if tokens > 0 else 0
                
                results[d_name][m_name]["accuracy"].append(correct / eval_len)
                results[d_name][m_name]["latency"].append(total_latency_ms)
                results[d_name][m_name]["vram_pct"].append(vram)
                results[d_name][m_name]["throughput"].append(total_throughput)
                results[d_name][m_name]["train_time"].append(train_duration)
                results[d_name][m_name]["tpot"].append(avg_tpot)

                reft_model = None
                trainer = None
                torch.cuda.empty_cache()
                pbar.update(1)

    pbar.close()

    pdf_path = "/workspace/reports/ReFT_Efficiency_Report.pdf"
    metrics = ["accuracy", "latency", "vram_pct", "throughput", "train_time", "tpot"]
    colors = {
        "Llama-3-8B": "black", "Llama-3-8B-Instruct": "blue", 
        "Qwen3-8B": "red", "Qwen3-8B-Thinking": "purple", 
        "Gemma-2-9B": "green", "Gemma-2-9B-IT": "cyan"
    }

    with PdfPages(pdf_path) as pdf:
        for d_name in DATASETS:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'ReFT Efficiency Benchmarks: {d_name}')
            for i, mtr in enumerate(metrics):
                ax = axes.flatten()[i]
                for m_name in MODELS.keys():
                    if len(results[d_name][m_name][mtr]) > 0:
                        ax.plot(N_SAMPLES[:len(results[d_name][m_name][mtr])], results[d_name][m_name][mtr], marker='o', color=colors.get(m_name, "black"), label=m_name)
                ax.set_title(mtr.capitalize())
                ax.legend(fontsize='x-small', ncol=2)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"[COMPLETE] Efficiency Report saved to persistent volume: {pdf_path}", flush=True)

@app.local_entrypoint()
def main():
    print("Initiating cloud-based ReFT evaluation...", flush=True)
    execute_reft_pipeline.remote()