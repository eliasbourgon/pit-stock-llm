"""
eval.py — Majority-vote @ 4 evaluation: baseline vs N RLVR checkpoints.

Supports two model types:
  - binary   : predicts +1 or -1 directly
  - gaussian : predicts a continuous % return; sign is used for direction accuracy

Multi-GPU via torchrun:
  torchrun --nproc_per_node=3 src/evaluation/eval.py --base_model ... --rlvr_checkpoints ... --data_path ...

Each GPU handles ~1/3 of the eval samples. Results gathered on rank 0.
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

# Colors: baseline=blue, rlvr-ddp-v3=pink, pnl=yellow, gaussian=light blue
COLORS = ["#4C72B0", "#E91E8C", "#FFD700", "#87CEEB"]


# ─── Prompts ──────────────────────────────────────────────────────────────────

def build_prompt_binary(text: str, industry: str, date: str) -> str:
    instruction = (
        f"Date: {date}\n"
        f"Industry: {industry}\n"
        f"Earnings Call Transcript:\n{text}\n\n"
        "Based on this earnings call, predict whether the stock's 1-month return "
        "will be positive (+1) or negative (-1).\n"
        "Answer (+1 or -1):"
    )
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


def build_prompt_gaussian(text: str, industry: str, date: str) -> str:
    instruction = (
        f"Date: {date}\n"
        f"Industry: {industry}\n"
        f"Earnings Call Transcript:\n{text}\n\n"
        "Based on this earnings call, predict the stock's 1-month return as a percentage. "
        "Positive means price increase, negative means price decrease. "
        "Typical returns range from -5% to +5%.\n"
        "Answer in percentage (e.g. +2.3 or -1.5):"
    )
    return f"<|user|>\n{instruction}\n<|assistant|>\n"


PROMPT_FNS = {
    "binary":   build_prompt_binary,
    "gaussian": build_prompt_gaussian,
}


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_binary(text: str) -> str | None:
    matches = re.findall(r"([+-]1)\b", text.strip())
    return matches[-1] if matches else None


def extract_gaussian(text: str) -> str | None:
    """Extract a float return prediction and convert its sign to +1/-1."""
    # Primary: number followed by %
    matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
    if not matches:
        # Fallback: signed number
        matches = re.findall(r"([+-]\d+(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        val = float(matches[-1])
        return "+1" if val > 0 else "-1"
    except ValueError:
        return None


EXTRACT_FNS = {
    "binary":   extract_binary,
    "gaussian": extract_gaussian,
}


# ─── Voting ───────────────────────────────────────────────────────────────────

def majority_vote(completions: list[str], extract_fn) -> str | None:
    votes = [extract_fn(c) for c in completions]
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
    """Returns raw records without prompt — prompt is built per model type in run_eval."""
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)
    df = df.iloc[offset : offset + n_eval].reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        text = row["text"]
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars] + "\n[truncated]"
        records.append({
            "text":     text,
            "industry": row["industry"],
            "date_str": pd.Period(row["date"]).strftime("%B %Y"),
            "label":    "+1" if row["ret_3M_shifted"] > 0 else "-1",
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
    label: str,
    model_type: str,
) -> list[dict]:
    build_prompt_fn = PROMPT_FNS[model_type]
    extract_fn      = EXTRACT_FNS[model_type]

    results = []
    for i, rec in enumerate(records):
        prompt = build_prompt_fn(rec["text"], rec["industry"], rec["date_str"])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_votes,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

        prompt_len  = inputs["input_ids"].shape[1]
        completions = [
            tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
            for out in outputs
        ]
        truncated = [out.shape[0] - prompt_len >= max_new_tokens for out in outputs]
        votes     = [extract_fn(c) for c in completions]
        pred      = majority_vote(completions, extract_fn)

        results.append({
            "label":      rec["label"],
            "prediction": pred,
            "votes":      votes,
            "correct":    pred == rec["label"],
            "truncated":  sum(truncated),
        })

        if (i + 1) % 10 == 0:
            n_correct = sum(r["correct"] for r in results)
            print(f"  [{label} | rank {rank} | {i+1}/{len(records)}]  acc={n_correct/(i+1):.1%}", flush=True)

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
    n_models = len(metrics)
    colors   = (COLORS * ((n_models // len(COLORS)) + 1))[:n_models]
    names    = [m["model"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, 3 * n_models), 5))

    for ax, key, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1",       "F1 Score (+1)"),
    ]:
        values = [m[key] for m in metrics]
        bars   = ax.bar(names, values, color=colors, edgecolor="white", width=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random")
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=15)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1%}", ha="center", fontsize=9)

    fig.suptitle(f"Majority Vote @ 4  —  {metrics[0]['n']} samples", fontsize=12)
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

    checkpoints  = [c.strip() for c in args.rlvr_checkpoints.split(",")]
    labels_rlvr  = [l.strip() for l in args.model_labels.split(",")]
    model_types  = [t.strip() for t in args.model_types.split(",")]
    assert len(checkpoints) == len(labels_rlvr) == len(model_types), \
        "--rlvr_checkpoints, --model_labels and --model_types must have the same number of entries"

    if master:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Eval: {world_size} GPUs | {args.n_eval} samples | majority vote @ {args.num_votes}", flush=True)
        for lbl, ckpt, mtype in zip(labels_rlvr, checkpoints, model_types):
            print(f"  {lbl} ({mtype}) : {ckpt}", flush=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    all_records = load_test_data(args.data_path, args.data_offset, args.n_eval, args.max_prompt_chars)
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
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()

    all_metrics = []

    def eval_and_gather(model, label, csv_name, model_type):
        if master:
            print(f"\nRunning inference [{label}] (type={model_type})...", flush=True)
        dist.barrier()
        shard_results = run_eval(
            model, tokenizer, shard,
            args.num_votes, args.max_new_tokens, device, rank, label, model_type,
        )
        gathered = [None] * world_size
        dist.gather_object(shard_results, gathered if master else None, dst=0)
        if master:
            full_results = [None] * len(all_records)
            for r, shard_res in enumerate(gathered):
                for i, res in enumerate(shard_res):
                    full_results[r + i * world_size] = res
            all_metrics.append(compute_metrics(full_results, label))
            pd.DataFrame([
                {"label": res["label"], "prediction": res["prediction"], "votes": str(res["votes"])}
                for res in full_results
            ]).to_csv(os.path.join(args.output_dir, csv_name), index=False)
        dist.barrier()

    # ── 1. Baseline (binary prompt) ───────────────────────────────────────────
    eval_and_gather(base_model, "Baseline", "baseline_predictions.csv", "binary")

    # ── 2. RLVR checkpoints ───────────────────────────────────────────────────
    peft_model = None
    for i, (ckpt, lbl, mtype) in enumerate(zip(checkpoints, labels_rlvr, model_types)):
        adapter_name = f"adapter_{i}"
        if master:
            print(f"\nLoading LoRA adapter [{lbl}]: {ckpt}", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base_model, ckpt, adapter_name=adapter_name).eval()
        else:
            peft_model.load_adapter(ckpt, adapter_name=adapter_name)
            peft_model.set_adapter(adapter_name)

        eval_and_gather(peft_model, lbl, f"predictions_{adapter_name}.csv", mtype)

    # ── Summary ───────────────────────────────────────────────────────────────
    if master:
        save_plots(all_metrics, args.output_dir)
        summary_df = pd.DataFrame(all_metrics)
        summary_df.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
        print(f"\nAll results saved to {args.output_dir}/", flush=True)
        print(summary_df.to_string(index=False))

    dist.destroy_process_group()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Majority-vote @ 4 eval: baseline vs N RLVR checkpoints")
    p.add_argument("--base_model",        type=str, required=True)
    p.add_argument("--rlvr_checkpoints",  type=str, required=True,
                   help="Comma-separated LoRA checkpoint paths")
    p.add_argument("--model_labels",      type=str, required=True,
                   help="Comma-separated display labels (one per checkpoint)")
    p.add_argument("--model_types",       type=str, required=True,
                   help="Comma-separated types: 'binary' or 'gaussian' (one per checkpoint)")
    p.add_argument("--data_path",         type=str, default="data/merged_data.parquet")
    p.add_argument("--output_dir",        type=str, default="results/eval")
    p.add_argument("--data_offset",       type=int, default=2000)
    p.add_argument("--n_eval",            type=int, default=100)
    p.add_argument("--num_votes",         type=int, default=4)
    p.add_argument("--max_new_tokens",    type=int, default=700)
    p.add_argument("--max_prompt_chars",  type=int, default=6000)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
