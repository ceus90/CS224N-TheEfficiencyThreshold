import modal

app = modal.App("cs224n-lora-gsm8k-cot")

lora_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ninja-build", "build-essential")
    .pip_install(
        "transformers>=4.44.0",
        "huggingface_hub",
        "matplotlib",
        "datasets<4.0.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "tqdm"
    )
    .env({
        "NCCL_IGNORE_DISABLED_P2P": "1",
        "NCCL_DEBUG": "ERROR",
        "PYTHONWARNINGS": "ignore",
        "VLLM_LOGGING_LEVEL": "ERROR"
    })
)

output_volume = modal.Volume.from_name("reft-reports", create_if_missing=True)


@app.function(
    image=lora_image,
    cpu=4.0,
    memory=65536,
    gpu="B200",
    timeout=43200,    volumes={"/workspace/reports": output_volume}
)
def execute_lora_pipeline():
    import os
    import re
    import time
    import json
    import warnings
    import random
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
    from datasets import load_dataset
    from huggingface_hub import login
    from tqdm.auto import tqdm
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    warnings.filterwarnings("ignore")

    print("[SYSTEM] Environment initialized. Beginning LoRA pipeline...", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    MODELS = {
        # "Llama-3-8B": {"id": "meta-llama/Meta-Llama-3-8B", "cot": False}, # todo
        # "Llama-3-8B-Instruct": {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "cot": False}, # todo
        # "Qwen3-4B": {"id": "Qwen/Qwen3-4B", "cot": False}, # done
        "Qwen3-4B-Instruct-2507": {"id": "Qwen/Qwen3-4B-Instruct-2507", "cot": False}, # current
        "Qwen3-8B": {"id": "Qwen/Qwen3-8B", "cot": False}, # current
        # "Qwen3-8B-Thinking": {"id": "Qwen/Qwen3-8B", "cot": True},
        # "Qwen2.5-7B-Instruct": {"id": "Qwen/Qwen2.5-7B-Instruct", "cot": False},
        # "Gemma-2-9B": {"id": "google/gemma-2-9b", "cot": False}, # todo
        # "Gemma-2-9B-IT": {"id": "google/gemma-2-9b-it", "cot": False}
    }

    DATASETS = ["gsm8k"]
    N_SAMPLES = [16, 32, 64, 128, 256]

    EVAL_SAMPLES = {
        "gsm8k": 250
    }

    results = {
        ds: {m_name: {metric: [] for metric in ["accuracy", "latency", "vram_pct", "throughput", "train_time", "tpot"]}
             for m_name in MODELS.keys()} for ds in DATASETS
    }

    output_dir = "/workspace/reports/stratified"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = f"{output_dir}/LoRA_GSM8K_COT_Checkpoint.jsonl"
    completed = set()
    DISABLE_CHECKPOINT_SKIP = False

    def load_checkpoint():
        nonlocal results, completed
        if not os.path.exists(checkpoint_path):
            return
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    m = rec.get("model")
                    d = rec.get("dataset")
                    n = rec.get("n")
                    metrics = rec.get("metrics", {})
                    if m and d and isinstance(n, int) and m in results.get(d, {}):
                        results[d][m]["accuracy"].append(metrics.get("accuracy", 0.0))
                        results[d][m]["latency"].append(metrics.get("latency", 0.0))
                        results[d][m]["vram_pct"].append(metrics.get("vram_pct", 0.0))
                        results[d][m]["throughput"].append(metrics.get("throughput", 0.0))
                        results[d][m]["train_time"].append(metrics.get("train_time", 0.0))
                        results[d][m]["tpot"].append(metrics.get("tpot", 0.0))
                        completed.add((m, d, n))
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}", flush=True)

    def append_checkpoint(record):
        try:
            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[WARN] Failed to append checkpoint: {e}", flush=True)

    LABEL_MAPPINGS = {}

    def format_prompt(example, dataset_name, use_cot=False):
        suffix = " Let's think step by step." if use_cot else ""
        if dataset_name == "gsm8k":
            return (
                f"Question: {example['question']}{suffix} "
                "Respond with the final answer after '####'.\nAnswer:"
            )
        elif dataset_name == "superglue_rte":
            return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}{suffix}\nEntailment:"
        elif dataset_name == "superglue_boolq":
            return f"Passage: {example['passage']}\nQuestion: {example['question']}{suffix}\nAnswer:"
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

    def extract_last_number(text):
        nums = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        return nums[-1] if nums else ""

    def extract_gsm8k_number(text):
        hash_match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
        if hash_match:
            return hash_match.group(1)
        answer_match = re.search(r'Answer:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text, flags=re.IGNORECASE)
        if answer_match:
            return answer_match.group(1)
        return extract_last_number(text)

    def get_expected_label(dataset_name, val):
        if dataset_name == "gsm8k":
            return str(val).strip()
        return str(val).strip()

    def check_correctness(expected, generated, dataset_name):
        exp = str(expected).strip().lower()
        gen = str(generated).strip().lower()

        exp_num = extract_gsm8k_number(exp)
        gen_num = extract_gsm8k_number(gen)
        return exp_num == gen_num

    def prepare_data(dataset_name, n):
        target_eval_samples = EVAL_SAMPLES.get(dataset_name, 150)
        seed = 42

        def select_random(ds, k):
            if k >= len(ds):
                return ds
            rnd = random.Random(seed)
            idxs = list(range(len(ds)))
            rnd.shuffle(idxs)
            return ds.select(idxs[:k])

        def stratified_select(ds, label_key, k):
            if k >= len(ds):
                return ds
            rnd = random.Random(seed)
            label_to_indices = {}
            for i, ex in enumerate(ds):
                label = ex[label_key]
                label_to_indices.setdefault(label, []).append(i)
            total = sum(len(v) for v in label_to_indices.values())
            if total == 0:
                return ds.select(list(range(k)))
            k = min(k, total)
            allocations = {}
            for label, idxs in label_to_indices.items():
                alloc = int(round(k * len(idxs) / total))
                allocations[label] = min(alloc, len(idxs))
            def cur_total():
                return sum(allocations.values())
            while cur_total() > k:
                for label in sorted(allocations, key=lambda l: allocations[l], reverse=True):
                    if allocations[label] > 0 and cur_total() > k:
                        allocations[label] -= 1
            while cur_total() < k:
                for label in sorted(label_to_indices, key=lambda l: len(label_to_indices[l]), reverse=True):
                    if allocations[label] < len(label_to_indices[label]) and cur_total() < k:
                        allocations[label] += 1
            selected = []
            for label, idxs in label_to_indices.items():
                rnd.shuffle(idxs)
                selected.extend(idxs[:allocations[label]])
            rnd.shuffle(selected)
            return ds.select(selected)

        def ensure_no_overlap(train_ds, eval_ds, eval_pool, name):
            def fp(ex):
                return json.dumps(ex, sort_keys=True)
            train_set = set(fp(ex) for ex in train_ds)
            eval_list = list(eval_ds)
            filtered = []
            filtered_fps = set()
            for ex in eval_list:
                f = fp(ex)
                if f in train_set:
                    continue
                filtered.append(ex)
                filtered_fps.add(f)
            if len(filtered) == len(eval_list):
                return eval_ds
            pool_list = list(eval_pool)
            pool_fp_to_idx = {}
            for i, ex in enumerate(pool_list):
                f = fp(ex)
                if f not in pool_fp_to_idx:
                    pool_fp_to_idx[f] = i
            refill = []
            for ex in pool_list:
                f = fp(ex)
                if f in train_set or f in filtered_fps:
                    continue
                refill.append(ex)
                filtered_fps.add(f)
                if len(filtered) + len(refill) >= len(eval_list):
                    break
            new_eval = filtered + refill
            idxs = []
            for ex in new_eval:
                f = fp(ex)
                if f in pool_fp_to_idx:
                    idxs.append(pool_fp_to_idx[f])
            return eval_pool.select(idxs)

        if dataset_name == "gsm8k":
            ds_full_train = load_dataset("gsm8k", "main", split="train", trust_remote_code=True)
            ds_full_eval = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
            t_n, e_n = min(n, len(ds_full_train)), min(target_eval_samples, len(ds_full_eval))
            train = ds_full_train.select(range(t_n))
            eval_ds = select_random(ds_full_eval, e_n)
            key = "answer"
            eval_ds = ensure_no_overlap(train, eval_ds, ds_full_eval, dataset_name)
        elif dataset_name == "superglue_rte":
            ds_full_train = load_dataset("super_glue", "rte", split="train", trust_remote_code=True)
            ds_full_eval = load_dataset("super_glue", "rte", split="validation", trust_remote_code=True)
            t_n, e_n = min(n, len(ds_full_train)), min(target_eval_samples, len(ds_full_eval))
            train = stratified_select(ds_full_train, "label", t_n)
            eval_ds = stratified_select(ds_full_eval, "label", e_n)
            key = "label"
            eval_ds = ensure_no_overlap(train, eval_ds, ds_full_eval, dataset_name)
        elif dataset_name == "superglue_boolq":
            ds_full_train = load_dataset("super_glue", "boolq", split="train", trust_remote_code=True)
            ds_full_eval = load_dataset("super_glue", "boolq", split="validation", trust_remote_code=True)
            t_n, e_n = min(n, len(ds_full_train)), min(target_eval_samples, len(ds_full_eval))
            train = stratified_select(ds_full_train, "label", t_n)
            eval_ds = stratified_select(ds_full_eval, "label", e_n)
            key = "label"
            eval_ds = ensure_no_overlap(train, eval_ds, ds_full_eval, dataset_name)
        elif dataset_name == "financial_phrasebank":
            ds_full = load_dataset("financial_phrasebank", "sentences_allagree", split="train", trust_remote_code=True)
            train_pool = ds_full.select(range(int(len(ds_full) * 0.8)))
            t_n = min(n, len(train_pool))
            e_n = min(target_eval_samples, len(ds_full) - len(train_pool))
            train = stratified_select(train_pool, "label", t_n)
            eval_pool = ds_full.select(range(len(train_pool), len(ds_full)))
            eval_ds = stratified_select(eval_pool, "label", e_n)
            key = "label"
            eval_ds = ensure_no_overlap(train, eval_ds, eval_pool, dataset_name)
        elif dataset_name == "dolly15k":
            ds_full = load_dataset("databricks/databricks-dolly-15k", split="train", trust_remote_code=True)
            t_n = min(n, int(len(ds_full) * 0.8))
            e_n = min(target_eval_samples, len(ds_full) - t_n)
            train = ds_full.select(range(t_n))
            eval_pool = ds_full.select(range(t_n, len(ds_full)))
            eval_ds = select_random(eval_pool, e_n)
            key = "response"
            eval_ds = ensure_no_overlap(train, eval_ds, eval_pool, dataset_name)
        else:
            ds_full = load_dataset("ought/raft", "ade_corpus_v2", split="train", trust_remote_code=True)
            train_pool = ds_full.select(range(int(len(ds_full) * 0.8)))
            t_n = min(n, len(train_pool))
            e_n = min(target_eval_samples, len(ds_full) - len(train_pool))
            train = stratified_select(train_pool, "Label", t_n)
            eval_pool = ds_full.select(range(len(train_pool), len(ds_full)))
            eval_ds = stratified_select(eval_pool, "Label", e_n)
            key = "Label"
            eval_ds = ensure_no_overlap(train, eval_ds, eval_pool, dataset_name)

        return train, eval_ds, key

    class SupervisedDataset(torch.utils.data.Dataset):
        def __init__(self, prompts, targets, tokenizer, max_len):
            self.items = []
            for p, t in zip(prompts, targets):
                full = f"{p} {t}"
                p_ids = tokenizer(
                    p,
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                    add_special_tokens=True
                )["input_ids"]
                full_enc = tokenizer(
                    full,
                    truncation=True,
                    max_length=max_len,
                    padding=False,
                    add_special_tokens=True
                )
                input_ids = full_enc["input_ids"]
                attn = full_enc["attention_mask"]
                prompt_len = min(len(p_ids), len(input_ids))
                labels = [-100] * prompt_len + input_ids[prompt_len:]
                labels = labels[:len(input_ids)]
                self.items.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attn, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                })

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    def collate_fn(batch, pad_token_id):
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        max_len = max(x.size(0) for x in input_ids)

        def pad(seq, pad_val):
            return torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=pad_val)

        input_ids = torch.stack([pad(x, pad_token_id) for x in input_ids], dim=0)
        attention_mask = torch.stack([pad(x, 0) for x in attention_mask], dim=0)
        labels = torch.stack([pad(x, -100) for x in labels], dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    pbar = tqdm(total=len(MODELS) * len(DATASETS) * len(N_SAMPLES), desc="Benchmark Progress")

    if not DISABLE_CHECKPOINT_SKIP:
        load_checkpoint()

    for m_name, cfg in MODELS.items():
        m_id, use_cot = cfg["id"], True

        for d_name in DATASETS:
            for n in N_SAMPLES:
                if not DISABLE_CHECKPOINT_SKIP and (m_name, d_name, n) in completed:
                    print(f"[SKIP] Already completed Model: {m_name} | Dataset: {d_name} | Samples (N): {n}", flush=True)
                    pbar.update(1)
                    continue
                print(f"[BENCHMARK] Active Model: {m_name} | Dataset: {d_name} | Samples (N): {n}", flush=True)
                torch.cuda.reset_peak_memory_stats()

                try:
                    torch.cuda.empty_cache()

                    tokenizer = AutoTokenizer.from_pretrained(m_id, token=hf_token, model_max_length=2048)
                    tokenizer.padding_side, tokenizer.pad_token = "left", tokenizer.eos_token
                    tokenizer.truncation_side = "left"

                    is_specialized = any(x in m_id.lower() for x in ["fp8", "dms"])
                    load_cfg = None if is_specialized else bnb_config

                    base_model = AutoModelForCausalLM.from_pretrained(
                        m_id, quantization_config=load_cfg, device_map="auto",
                        attn_implementation="sdpa", token=hf_token,
                        dtype=torch.bfloat16
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to load model {m_id}: {e}", flush=True)
                    pbar.update(1)
                    continue

                if not is_specialized:
                    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                lora_model = get_peft_model(base_model, lora_config)
                lora_model.train()

                model_ctx_len = getattr(lora_model.config, "max_position_embeddings", None)
                if model_ctx_len is None:
                    model_ctx_len = tokenizer.model_max_length
                ctx_len = min(tokenizer.model_max_length, model_ctx_len)
                tokenizer.model_max_length = ctx_len

                train_ds, ev_data, t_key = prepare_data(d_name, n)
                train_prompts = [format_prompt(i, d_name, use_cot) for i in train_ds]
                train_targets = [get_expected_label(d_name, i[t_key]) for i in train_ds]
                train_dataset = SupervisedDataset(train_prompts, train_targets, tokenizer, ctx_len)

                trainer = Trainer(
                    model=lora_model,
                    args=TrainingArguments(
                        output_dir="./tmp",
                        max_steps=100,
                        per_device_train_batch_size=16,
                        learning_rate=2e-4,
                        bf16=True,
                        tf32=True,
                        report_to="none",
                        gradient_checkpointing=False
                    ),
                    train_dataset=train_dataset,
                    data_collator=lambda b: collate_fn(b, tokenizer.pad_token_id)
                )

                train_start_time = time.time()
                trainer.train()
                train_duration = time.time() - train_start_time

                lora_model.eval()
                correct, ms, tokens, ev_list = 0, 0, 0, list(ev_data)
                prompt_truncated = False

                if len(ev_list) > 0:
                    for idx in range(0, len(ev_list), 16):
                        batch = ev_list[idx:idx + 16]
                        p_txts = [format_prompt(i, d_name, use_cot) for i in batch]
                        targets = [get_expected_label(d_name, i[t_key]) for i in batch]
                        lengths = tokenizer(
                            p_txts,
                            padding=False,
                            truncation=False,
                            return_length=True
                        )["length"]
                        if max(lengths, default=0) > tokenizer.model_max_length:
                            prompt_truncated = True

                        inputs = tokenizer(
                            p_txts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=ctx_len
                        ).to("cuda")

                        torch.cuda.synchronize()
                        start = time.time()

                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            out = lora_model.generate(
                                **inputs,
                                max_new_tokens=256,
                                pad_token_id=tokenizer.eos_token_id
                            )

                        torch.cuda.synchronize()
                        ms += (time.time() - start) * 1000
                        tokens += (out.shape[1] - inputs["input_ids"].shape[1]) * len(batch)

                        decs = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                        for exp, gen in zip(targets, decs):
                            if check_correctness(exp, gen, d_name):
                                correct += 1

                vram = (torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                eval_len = max(1, len(ev_list))

                total_latency_ms = ms / eval_len
                total_throughput = tokens / (ms / 1000) if ms > 0 else 0
                avg_tpot = (ms / tokens) if tokens > 0 else 0

                results[d_name][m_name]["accuracy"].append(correct / eval_len)
                results[d_name][m_name]["latency"].append(total_latency_ms)
                results[d_name][m_name]["vram_pct"].append(vram)
                results[d_name][m_name]["throughput"].append(total_throughput)
                results[d_name][m_name]["train_time"].append(train_duration)
                results[d_name][m_name]["tpot"].append(avg_tpot)

                completed.add((m_name, d_name, n))
                append_checkpoint({
                    "timestamp": time.time(),
                    "model": m_name,
                    "model_id": m_id,
                    "dataset": d_name,
                    "n": n,
                    "prompt_truncated": prompt_truncated,
                    "metrics": {
                        "accuracy": results[d_name][m_name]["accuracy"][-1],
                        "latency": results[d_name][m_name]["latency"][-1],
                        "vram_pct": results[d_name][m_name]["vram_pct"][-1],
                        "throughput": results[d_name][m_name]["throughput"][-1],
                        "train_time": results[d_name][m_name]["train_time"][-1],
                        "tpot": results[d_name][m_name]["tpot"][-1]
                    }
                })

                lora_model = None
                base_model = None
                torch.cuda.empty_cache()
                pbar.update(1)

    pbar.close()

    metrics = ["accuracy", "latency", "vram_pct", "throughput", "train_time", "tpot"]
    colors = {
        "Llama-3-8B-Instruct": "blue",
        "Qwen3-4B": "orange",
        "Qwen3-4B-Instruct-2507": "brown",
        "Qwen3-8B": "red",
        "Qwen3-8B-Thinking": "purple",
        "Qwen2.5-7B-Instruct": "magenta",
        "Gemma-2-9B": "green",
        "Gemma-2-9B-IT": "cyan"
    }

    def sanitize_name(name):
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)

    for m_name in MODELS.keys():
        model_tag = sanitize_name(m_name)
        pdf_path = f"{output_dir}/LoRA_GSM8K_COT_Report_{model_tag}.pdf"
        with PdfPages(pdf_path) as pdf:
            for d_name in DATASETS:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(f"LoRA GSM8K CoT: {m_name} | {d_name}")
                for i, mtr in enumerate(metrics):
                    ax = axes.flatten()[i]
                    y = results[d_name][m_name][mtr]
                    if len(y) > 0:
                        k = min(len(N_SAMPLES), len(y))
                        ax.plot(
                            N_SAMPLES[:k],
                            y[:k],
                            marker='o',
                            color=colors.get(m_name, "black"),
                            label=m_name
                        )
                    ax.set_title(mtr.capitalize())
                    ax.legend(fontsize='x-small', ncol=1)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        print(f"[COMPLETE] LoRA Efficiency Report saved to persistent volume: {pdf_path}", flush=True)


@app.local_entrypoint()
def main():
    print("Initiating cloud-based LoRA evaluation...", flush=True)
    execute_lora_pipeline.remote()
