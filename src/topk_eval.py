import os
import json
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/shared/home/aif/models/hf/Qwen2.5-7B-Instruct"


def load_narrativeqa_sample(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    context = lines[0]
    question = "Please briefly summarize the given context."
    return context, question


def load_korquad_sample(path: str):
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    obj = json.loads(line)

    context = obj.get("prompt", "")
    question = "위 지문을 읽고 핵심 내용을 간단히 요약하세요."
    return context, question


def build_prompt(tokenizer, context: str, question: str):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def topk_truncate_input(tokenizer, prompt: str, ratio: float):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    total_len = input_ids.shape[1]
    keep_len = max(1, int(total_len * ratio))

    # 앞부분 keep_len만 유지하는 독립 budget baseline
    input_ids = input_ids[:, :keep_len]
    attention_mask = attention_mask[:, :keep_len]

    return input_ids, attention_mask, total_len, keep_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["narrativeqa", "korquad"], required=True)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    if args.dataset == "narrativeqa":
        data_path = os.path.expanduser("~/kvbench_repro/data/narrativeqa_8k.txt")
        context, question = load_narrativeqa_sample(data_path)
    else:
        data_path = os.path.expanduser("~/kvbench_repro/data/korquad_eff_8k_trimmed.jsonl")
        context, question = load_korquad_sample(data_path)

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    prompt = build_prompt(tokenizer, context, question)
    input_ids, attention_mask, original_len, kept_len = topk_truncate_input(
        tokenizer, prompt, args.ratio
    )

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    print(f"[INFO] original_input_tokens = {original_len}")
    print(f"[INFO] kept_input_tokens = {kept_len}")

    if kept_len > 9000:
        raise ValueError(f"Too many kept tokens: {kept_len}. Check topk/truncation logic.")

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
    output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    result = {
        "dataset": args.dataset,
        "model_path": MODEL_PATH,
        "ratio": args.ratio,
        "original_input_tokens": int(original_len),
        "kept_input_tokens": int(kept_len),
        "output_tokens": int(gen_tokens),
        "latency_s": latency_s,
        "peak_vram_mib": peak_vram_mib,
        "output_text": output_text,
    }

    os.makedirs(os.path.expanduser("~/kvbench_repro/results/topk"), exist_ok=True)
    ratio_str = str(args.ratio).replace(".", "p")
    out_path = os.path.expanduser(f"~/kvbench_repro/results/topk/{args.dataset}_topk_r{ratio_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
