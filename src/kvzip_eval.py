import os
import json
import time
import argparse
import torch
import sys

sys.path.append(os.path.expanduser("~/kvbench_repro/ext_KVzip"))
from model import ModelKVzip


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-1M"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["narrativeqa", "korquad"], required=True)
    parser.add_argument("--ratio", type=float, default=0.3)
    args = parser.parse_args()

    if args.dataset == "narrativeqa":
        data_path = os.path.expanduser("~/kvbench_repro/data/narrativeqa_8k.txt")
        context, question = load_narrativeqa_sample(data_path)
    else:
        data_path = os.path.expanduser("~/kvbench_repro/data/korquad_eff_8k_trimmed.jsonl")
        context, question = load_korquad_sample(data_path)

    print(f"[INFO] Load {MODEL_NAME}")
    model = ModelKVzip(MODEL_NAME)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.time()
    kv = model.prefill(context, load_score=False)
    torch.cuda.synchronize()
    t1 = time.time()

    kv.prune(ratio=args.ratio)
    torch.cuda.synchronize()
    t2 = time.time()

    query_ids = model.apply_template(question)
    output = model.generate(query_ids, kv=kv)
    torch.cuda.synchronize()
    t3 = time.time()

    peak_vram_mib = torch.cuda.max_memory_allocated() / 1024 / 1024

    result = {
        "dataset": args.dataset,
        "model_name": MODEL_NAME,
        "ratio": args.ratio,
        "prefill_time_s": t1 - t0,
        "prune_time_s": t2 - t1,
        "gen_time_s": t3 - t2,
        "total_time_s": t3 - t0,
        "peak_vram_mib": peak_vram_mib,
        "output_text": output,
    }

    os.makedirs(os.path.expanduser("~/kvbench_repro/results/kvzip"), exist_ok=True)
    ratio_str = str(args.ratio).replace(".", "p")
    out_path = os.path.expanduser(f"~/kvbench_repro/results/kvzip/{args.dataset}_kvzip_r{ratio_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n[RESULT]")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
