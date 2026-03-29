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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["narrativeqa", "korquad"], required=True)
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
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.cuda()
    attention_mask = enc.attention_mask.cuda()

    if input_ids.shape[1] > 9000:
        raise ValueError(f"Too many input tokens: {input_ids.shape[1]}. Check dataset loader.")

    print(f"[INFO] input_tokens = {input_ids.shape[1]}")

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
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(gen_tokens),
        "latency_s": latency_s,
        "peak_vram_mib": peak_vram_mib,
        "output_text": output_text,
    }

    os.makedirs(os.path.expanduser("~/kvbench_repro/results/baseline"), exist_ok=True)
    out_path = os.path.expanduser(f"~/kvbench_repro/results/baseline/{args.dataset}_baseline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
