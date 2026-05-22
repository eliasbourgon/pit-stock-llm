"""
eval.py — Majority-vote @ 4 evaluation: baseline vs RLVR fine-tuned model.

Multi-GPU via torchrun:
  torchrun --nproc_per_node=3 eval.py --base_model ... --rlvr_checkpoint ... --data_path ...

Each GPU handles ~1/3 of the eval samples. Results are gathered on rank 0.
"""

import re
import argparse
import os
from collections import Counter

import torch
import torch.distributed as dist
import pandas as pd
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Prompt — must match rlvr_pipeline_ddp.py exactly ────────────────────────

def build_prompt(text: str, industry: str, date: str) -> str:
    instruction = (
        f"Date: {date}\n"
        f"Industry: {industry}\n"
        f"Earnings Call Transcript:\n{text}\n\n"
        "Based on this earnings call, predict whether the stock's 1-month return "
        "will be positive (+1) or negative (-1).\n"
        "Answer (+1 or -1):"
    )
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


# ─── Parsing & voting ─────────────────────────────────────────────────────────

def extract_prediction(text: str) -> str | None:
    matches = re.findall(r"([+-]1)\b", text.strip())
    return matches[-1] if matches else None


def majority_vote(completions: list[str]) -> str | None:
    votes = [extract_prediction(c) for c in completions]
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    counts = Counter(valid)
    top = counts.most_common(2)
    if len(top) == 2 and top[0][1] == top[1][1]:
        return None  # tie → abstain
    return top[0][0]


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_test_data(data_path: str, offset: int, n_eval: int, max_prompt_chars: int) -> list[dict]:
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)
    df = df.iloc[offset : offset + n_eval].reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        text = row["text"]
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars] + "\n[truncated]"
        date_str = pd.Period(row["date"]).strftime("%B %Y")
        records.append({
            "prompt": build_prompt(text, row["industry"], date_str),
            "label":  "+1" if row["ret_3M_shifted"] > 0 else "-1",
        })
    return records


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_eval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict],
    num_votes: int,
    max_new_tokens: int,
    device: str,
    rank: int,
) -> list[dict]:
    results = []
    for i, rec in enumerate(records):
        inputs = tokenizer(
            rec["prompt"], return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_votes,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        completions = [
            tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            for out in outputs
        ]
        # A completion is truncated if it hit the token limit exactly
        truncated = [out.shape[0] - prompt_len >= max_new_tokens for out in outputs]

        votes = [extract_prediction(c) for c in completions]
        pred  = majority_vote(completions)

        results.append({
            "label":      rec["label"],
            "prediction": pred,
            "votes":      votes,
            "correct":    pred == rec["label"],
            "truncated":  sum(truncated),
        })

        if (i + 1) % 10 == 0:
            n_correct = sum(r["correct"] for r in results)
            print(f"  [rank {rank} | {i+1}/{len(records)}]  running acc={n_correct/(i+1):.1%}", flush=True)

    return results


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], label: str) -> dict:
    n           = len(results)
    abstentions = sum(1 for r in results if r["prediction"] is None)
    correct     = sum(r["correct"] for r in results)
    accuracy    = correct / n

    valid = [(r["prediction"], r["label"]) for r in results if r["prediction"] is not None]
    f1 = f1_score([l for _, l in valid], [p for p, _ in valid],
                  pos_label="+1", average="binary") if valid else 0.0

    total_completions = n * 4
    truncated = sum(r["truncated"] for r in results)

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Accuracy     : {accuracy:.1%}  ({correct}/{n})")
    print(f"  F1 (+1)      : {f1:.3f}  (on {len(valid)} non-abstained)")
    print(f"  Abstentions  : {abstentions}/{n}  ({abstentions/n:.1%})")
    print(f"  Truncated    : {truncated}/{total_completions} completions ({truncated/total_completions:.1%})")
    label_dist = Counter(r["label"] for r in results)
    print(f"  Label dist   : +1={label_dist['+1']}  -1={label_dist['-1']}")
    print(f"{'─'*55}")

    return {"model": label, "accuracy": accuracy, "f1": f1, "abstentions": abstentions, "n": n}


# ─── Plots ────────────────────────────────────────────────────────────────────

def save_plots(metrics: list[dict], output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names  = [m["model"] for m in metrics]
    colors = ["#4C72B0", "#DD8452"]

    for ax, key, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1",       "F1 Score (+1)"),
    ]:
        values = [m[key] for m in metrics]
        bars = ax.bar(names, values, color=colors)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", fontsize=9)

    fig.suptitle(f"Majority Vote @ 4  —  {metrics[0]['n']} samples", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "eval_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── DDP init ─────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device     = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    master = rank == 0

    if master:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Eval: {world_size} GPUs | {args.n_eval} samples | majority vote @ {args.num_votes}", flush=True)

    # ── Load full data on every rank, then shard ──────────────────────────────
    all_records = load_test_data(args.data_path, args.data_offset, args.n_eval, args.max_prompt_chars)
    # Interleaved sharding: rank 0 → [0,3,6,...], rank 1 → [1,4,7,...], etc.
    shard = all_records[rank::world_size]
    if master:
        print(f"Samples per GPU: ~{len(shard)}  (total={len(all_records)})", flush=True)

    # ── Load tokenizer + base model ───────────────────────────────────────────
    if master:
        print(f"\nLoading base model: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    all_metrics = []

    for model_label, model, out_csv in [
        (f"Baseline  ({args.base_model.split('/')[-1]})",        base_model, "baseline_predictions.csv"),
        (f"RLVR      ({os.path.basename(args.rlvr_checkpoint)})", None,       "rlvr_predictions.csv"),
    ]:
        if model is None:
            if master:
                print(f"\nLoading LoRA adapter: {args.rlvr_checkpoint}", flush=True)
            model = PeftModel.from_pretrained(base_model, args.rlvr_checkpoint).eval()

        if master:
            print(f"Running inference [{model_label.strip()}]...", flush=True)

        dist.barrier()
        shard_results = run_eval(model, tokenizer, shard, args.num_votes, args.max_new_tokens, device, rank)

        # ── Gather all shards on rank 0 ───────────────────────────────────────
        gathered = [None] * world_size
        dist.gather_object(shard_results, gathered if master else None, dst=0)

        if master:
            # Reconstruct original order from interleaved shards
            full_results = [None] * len(all_records)
            for r, shard_res in enumerate(gathered):
                for i, res in enumerate(shard_res):
                    full_results[r + i * world_size] = res

            all_metrics.append(compute_metrics(full_results, model_label))
            pd.DataFrame([
                {"label": res["label"], "prediction": res["prediction"], "votes": str(res["votes"])}
                for res in full_results
            ]).to_csv(os.path.join(args.output_dir, out_csv), index=False)

        dist.barrier()

    if master:
        save_plots(all_metrics, args.output_dir)
        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
        print(f"\nAll results saved to {args.output_dir}/", flush=True)
        print(summary_df.to_string(index=False))

    dist.destroy_process_group()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Majority-vote @ 4 eval: baseline vs RLVR")
    p.add_argument("--base_model",       type=str, required=True)
    p.add_argument("--rlvr_checkpoint",  type=str, required=True,
                   help="Path to LoRA checkpoint dir (e.g. checkpoints/.../checkpoint-600)")
    p.add_argument("--data_path",        type=str, default="data/merged_data.parquet")
    p.add_argument("--output_dir",       type=str, default="results/eval")
    p.add_argument("--data_offset",      type=int, default=2000)
    p.add_argument("--n_eval",           type=int, default=200)
    p.add_argument("--num_votes",        type=int, default=4)
    p.add_argument("--max_new_tokens",   type=int, default=700)
    p.add_argument("--max_prompt_chars", type=int, default=6000)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
