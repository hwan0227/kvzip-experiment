import os
import json
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/shared/home/aif/models/hf/Qwen2.5-7B-Instruct"

def load_samples(dataset: str, max_samples: int):
    if dataset == "narrativeqa":
        path = os.path.expanduser("~/kvbench_repro/data/narrativeqa_8k.txt")
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                context = line.strip()
                question = "Please briefly summarize the given context."
                samples.append((context, question, i))
        return samples
    else:
        path = os.path.expanduser("~/kvbench_repro/data/korquad_eff_8k_trimmed.jsonl")
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                obj = json.loads(line.strip())
                context = obj.get("prompt", "")
                question = "위 지문을 읽고 핵심 내용을 간단히 요약하세요."
                samples.append((context, question, i))
        return samples

def build_prompt(tokenizer, context: str, question: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["narrativeqa", "korquad"], required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    out_dir = os.path.expanduser("~/kvbench_repro/results_100/baseline")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for context, question, idx in samples:
        prompt = build_prompt(tokenizer, context, question)
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.cuda()
        attention_mask = enc.attention_mask.cuda()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
            )
        torch.cuda.synchronize()
        end = time.time()

        peak_vram_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
        latency_s = end - start
        gen_tokens = output_ids.shape[1] - input_ids.shape[1]

        result = {
            "sample_idx": idx,
            "dataset": args.dataset,
            "model_path": MODEL_PATH,
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(gen_tokens),
            "latency_s": latency_s,
            "peak_vram_mib": peak_vram_mib,
        }
        all_results.append(result)
        print(f"[{args.dataset}][{idx}] lat={latency_s:.4f} mem={peak_vram_mib:.1f}")

    out_path = os.path.join(out_dir, f"{args.dataset}_baseline_100.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
