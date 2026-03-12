import modal

app = modal.App("cs224n-icl-gsm8k-cot")

icl_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ninja-build", "build-essential")
    .pip_install(
        "transformers>=4.44.0",
        "huggingface_hub",
        "matplotlib",
        "datasets<4.0.0",
        "accelerate",
        "bitsandbytes",
        "tqdm",
        "vllm>=0.7.0"
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
    image=icl_image,
    cpu=4.0,
    memory=65536,
    gpu="B200",
    timeout=43200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/workspace/reports": output_volume}
)
def execute_icl_pipeline():
    import os
    import re
    import time
    import json
    import warnings
    import random
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from huggingface_hub import login
    from tqdm.auto import tqdm

    warnings.filterwarnings("ignore")

    print("[SYSTEM] Environment initialized. Beginning ICL pipeline...", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    MODELS = {
        "Llama-3-8B": {"id": "meta-llama/Meta-Llama-3-8B", "cot": False}, # 
        "Llama-3-8B-Instruct": {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "cot": False}, # 
        "Qwen3-4B": {"id": "Qwen/Qwen3-4B", "cot": False}, # done
        "Qwen3-4B-Instruct-2507": {"id": "Qwen/Qwen3-4B-Instruct-2507", "cot": False}, # done
        "Qwen3-8B": {"id": "Qwen/Qwen3-8B", "cot": False}, # 
    }

    DATASETS = ["gsm8k"]
    N_SAMPLES = [16, 32, 64, 128, 256]

    EVAL_SAMPLES = {
        "gsm8k": 250
    }

    METRICS = [
        "accuracy", "latency", "vram_pct", "throughput_total", "throughput_output", "tpot",
        "avg_input_tokens", "max_input_tokens", "truncation_rate"
    ]

    results = {
        ds: {m_name: {metric: [] for metric in METRICS}
             for m_name in MODELS.keys()} for ds in DATASETS
    }

    output_dir = "/workspace/reports/stratified"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = f"{output_dir}/ICL_GSM8K_COT_Checkpoint.jsonl"
    completed = set()
    USE_CHECKPOINTS = True

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
                        results[d][m]["throughput_total"].append(metrics.get("throughput_total", 0.0))
                        results[d][m]["throughput_output"].append(metrics.get("throughput_output", 0.0))
                        results[d][m]["tpot"].append(metrics.get("tpot", 0.0))
                        results[d][m]["avg_input_tokens"].append(metrics.get("avg_input_tokens", 0.0))
                        results[d][m]["max_input_tokens"].append(metrics.get("max_input_tokens", 0.0))
                        results[d][m]["truncation_rate"].append(metrics.get("truncation_rate", 0.0))
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
        if dataset_name == "gsm8k":
            return (
                f"Question: {example['question']}\n\n"
                "Let's think step by step.\n\n"
                "Respond with the final answer after ####.\n\n"
                "Answer:\n"
            )
        else:
            txt = list(example.values())[0]
            return f"Input: {txt}\nOutput:"

    def _normalize_num(s: str) -> str:
        return s.replace(",", "").replace("$", "").strip()

    def extract_gsm8k_number(text):
        if text is None:
            return ""
        idx = text.find("####")
        if idx == -1:
            return ""
        after = text[idx + 4:]
        match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', after)
        if not match:
            return ""
        return _normalize_num(match.group(0))

    def get_generated_suffix(out_row, attn_mask_row, eos_id, pad_id):
        input_len_i = int(attn_mask_row.sum().item())
        gen_tokens_i = out_row[input_len_i:]
        if eos_id is not None:
            eos_pos = (gen_tokens_i == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                gen_tokens_i = gen_tokens_i[:eos_pos[0] + 1]
        if pad_id is not None:
            pad_pos = (gen_tokens_i == pad_id).nonzero(as_tuple=True)[0]
            if len(pad_pos) > 0:
                gen_tokens_i = gen_tokens_i[:pad_pos[0]]
        return gen_tokens_i

    def get_expected_label(dataset_name, val):
        if dataset_name == "gsm8k":
            return str(val).strip()
        return str(val).strip()

    def check_correctness(expected, generated, dataset_name):
        exp = str(expected).strip().lower()
        gen = str(generated).strip().lower()

        exp_num = extract_gsm8k_number(exp)
        gen_num = extract_gsm8k_number(gen)
        return bool(exp_num) and exp_num == gen_num

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
                return select_random(ds, k)
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
            # Refill from eval_pool with non-overlapping examples
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

        return train, eval_ds, key

    def build_fewshot_prefix(train_ds, dataset_name, key, use_cot):
        parts = []
        for ex in train_ds:
            prompt = format_prompt(ex, dataset_name, use_cot)
            label = get_expected_label(dataset_name, ex[key])
            parts.append(f"{prompt} {label}")
        return "\n\n".join(parts)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    pbar = tqdm(total=len(MODELS) * len(DATASETS) * len(N_SAMPLES), desc="Benchmark Progress")

    if USE_CHECKPOINTS:
        load_checkpoint()

    for m_name, cfg in MODELS.items():
        m_id, use_cot = cfg["id"], True

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
            pbar.update(len(DATASETS) * len(N_SAMPLES))
            continue

        base_model.eval()
        model_ctx_len = getattr(base_model.config, "max_position_embeddings", None)
        if model_ctx_len is None:
            model_ctx_len = tokenizer.model_max_length
        ctx_len = min(tokenizer.model_max_length, model_ctx_len)
        tokenizer.model_max_length = ctx_len

        for d_name in DATASETS:
            for n in N_SAMPLES:
                if USE_CHECKPOINTS and (m_name, d_name, n) in completed:
                    print(f"[SKIP] Already completed Model: {m_name} | Dataset: {d_name} | Samples (N): {n}", flush=True)
                    pbar.update(1)
                    continue
                print(f"[BENCHMARK] Active Model: {m_name} | Dataset: {d_name} | Samples (N): {n}", flush=True)
                torch.cuda.reset_peak_memory_stats()

                train_ds, ev_data, t_key = prepare_data(d_name, n)
                fewshot_prefix = build_fewshot_prefix(train_ds, d_name, t_key, use_cot)
                prefix = f"{fewshot_prefix}\n\n" if fewshot_prefix else ""

                correct, ms, tokens, ev_list = 0, 0, 0, list(ev_data)
                prompt_truncated = False
                processed_tokens = 0
                input_token_sum = 0
                input_token_count = 0
                max_input_tokens = 0
                num_truncated = 0
                total_prompts = 0

                if len(ev_list) > 0:
                    for idx in range(0, len(ev_list), 16):
                        batch = ev_list[idx:idx + 16]
                        p_txts = [prefix + format_prompt(i, d_name, use_cot) for i in batch]
                    
                        targets = [get_expected_label(d_name, i[t_key]) for i in batch]
                        lengths = tokenizer(
                            p_txts,
                            padding=False,
                            truncation=False,
                            return_length=True
                        )["length"]
                        if lengths:
                            max_input_tokens = max(max_input_tokens, max(lengths))
                        for l in lengths:
                            total_prompts += 1
                            if l > ctx_len:
                                num_truncated += 1
                        prompt_truncated = num_truncated > 0

                        input_token_sum += sum(lengths)
                        input_token_count += len(lengths)

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
                            out = base_model.generate(
                                **inputs,
                                max_new_tokens=256,
                                pad_token_id=tokenizer.eos_token_id
                            )

                        torch.cuda.synchronize()
                        ms += (time.time() - start) * 1000
                        batch_generated_tokens = 0
                        eos_id = tokenizer.eos_token_id
                        pad_id = tokenizer.pad_token_id
                        for i in range(out.shape[0]):
                            gen_tokens_i = get_generated_suffix(
                                out[i], inputs["attention_mask"][i], eos_id, pad_id
                            )
                            batch_generated_tokens += int(gen_tokens_i.numel())
                        tokens += batch_generated_tokens
                        batch_input_tokens = int(inputs["attention_mask"].sum().item())
                        processed_tokens += batch_input_tokens + batch_generated_tokens

                        decs = []
                        for i in range(out.shape[0]):
                            gen_tokens_i = get_generated_suffix(
                                out[i], inputs["attention_mask"][i], eos_id, pad_id
                            )
                            decs.append(tokenizer.decode(gen_tokens_i, skip_special_tokens=True))

                        for exp, gen in zip(targets, decs):
                            if check_correctness(exp, gen, d_name):
                                correct += 1

                vram = (torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                eval_len = max(1, len(ev_list))

                total_latency_ms = ms / eval_len
                total_throughput = processed_tokens / (ms / 1000) if ms > 0 else 0
                output_throughput = tokens / (ms / 1000) if ms > 0 else 0
                avg_tpot = (ms / tokens) if tokens > 0 else 0
                avg_input_tokens = (input_token_sum / input_token_count) if input_token_count > 0 else 0
                truncation_rate = (num_truncated / total_prompts) if total_prompts > 0 else 0

                results[d_name][m_name]["accuracy"].append(correct / eval_len)
                results[d_name][m_name]["latency"].append(total_latency_ms)
                results[d_name][m_name]["vram_pct"].append(vram)
                results[d_name][m_name]["throughput_total"].append(total_throughput)
                results[d_name][m_name]["throughput_output"].append(output_throughput)
                results[d_name][m_name]["tpot"].append(avg_tpot)
                results[d_name][m_name]["avg_input_tokens"].append(avg_input_tokens)
                results[d_name][m_name]["max_input_tokens"].append(max_input_tokens)
                results[d_name][m_name]["truncation_rate"].append(truncation_rate)

                completed.add((m_name, d_name, n))
                if USE_CHECKPOINTS:
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
                            "throughput_total": results[d_name][m_name]["throughput_total"][-1],
                            "throughput_output": results[d_name][m_name]["throughput_output"][-1],
                            "tpot": results[d_name][m_name]["tpot"][-1],
                            "avg_input_tokens": results[d_name][m_name]["avg_input_tokens"][-1],
                            "max_input_tokens": results[d_name][m_name]["max_input_tokens"][-1],
                            "truncation_rate": results[d_name][m_name]["truncation_rate"][-1]
                        }
                    })

                torch.cuda.empty_cache()
                pbar.update(1)

        base_model = None
        torch.cuda.empty_cache()

    pbar.close()

    metrics = METRICS
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

    if USE_CHECKPOINTS:
        for m_name in MODELS.keys():
            model_tag = sanitize_name(m_name)
            pdf_path = f"{output_dir}/ICL_GSM8K_COT_Report_{model_tag}.pdf"
            with PdfPages(pdf_path) as pdf:
                for d_name in DATASETS:
                    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
                    fig.suptitle(f"ICL GSM8K CoT: {m_name} | {d_name}")
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

            print(f"[COMPLETE] ICL Efficiency Report saved to persistent volume: {pdf_path}", flush=True)


@app.local_entrypoint()
def main():
    print("Initiating cloud-based ICL evaluation...", flush=True)
    execute_icl_pipeline.remote()
