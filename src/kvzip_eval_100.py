import os
import json
import time
import argparse
import torch
import sys

sys.path.append(os.path.expanduser("~/kvbench_repro/ext_KVzip"))
from model import ModelKVzip

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-1M"

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["narrativeqa", "korquad"], required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    samples = load_samples(args.dataset, args.max_samples)
    model = ModelKVzip(MODEL_NAME)

    out_dir = os.path.expanduser("~/kvbench_repro/results_100/kvzip")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    for context, question, idx in samples:
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
        _ = model.generate(query_ids, kv=kv)
        torch.cuda.synchronize()
        t3 = time.time()

        peak_vram_mib = torch.cuda.max_memory_allocated() / 1024 / 1024

        result = {
            "sample_idx": idx,
            "dataset": args.dataset,
            "ratio": args.ratio,
            "prefill_time_s": t1 - t0,
            "prune_time_s": t2 - t1,
            "gen_time_s": t3 - t2,
            "total_time_s": t3 - t0,
            "peak_vram_mib": peak_vram_mib,
        }
        all_results.append(result)
        print(f"[{args.dataset}][{idx}] ratio={args.ratio} total={t3-t0:.4f} mem={peak_vram_mib:.1f}")

    ratio_str = str(args.ratio).replace(".", "p")
    out_path = os.path.join(out_dir, f"{args.dataset}_kvzip_100_r{ratio_str}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
