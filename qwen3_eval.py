"""
qwen3_eval.py — Zero-shot evaluation of Qwen3-4B on earnings call return prediction.

Steps:
  1. Download Qwen/Qwen3-4B from HuggingFace to local dir
  2. Load with vLLM (tensor_parallel_size=3)
  3. Majority vote @ 4 on 100 test samples
  4. Save accuracy, F1, predictions

Thinking mode disabled (enable_thinking=False) for fast direct predictions.

Usage:
  python qwen3_eval.py --data_path data/merged_data.parquet
"""

import os
import re
import argparse
from collections import Counter

import pandas as pd
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Download ─────────────────────────────────────────────────────────────────

def download_model(model_id: str, local_dir: str) -> str:
    if os.path.isdir(local_dir) and any(f.endswith(".safetensors") for f in os.listdir(local_dir)):
        print(f"Model already at {local_dir}, skipping download.", flush=True)
        return local_dir
    print(f"Downloading {model_id} → {local_dir} ...", flush=True)
    snapshot_download(repo_id=model_id, local_dir=local_dir)
    print("Download complete.", flush=True)
    return local_dir


# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, text: str, industry: str, date: str) -> str:
    instruction = (
        f"Date: {date}\n"
        f"Industry: {industry}\n"
        f"Earnings Call Transcript:\n{text}\n\n"
        "Based on this earnings call, predict whether the stock's 1-month return "
        "will be positive (+1) or negative (-1).\n"
        "Answer with only +1 or -1:"
    )
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


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

def load_test_data(data_path: str, offset: int, n_eval: int, max_prompt_chars: int, tokenizer) -> list[dict]:
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)
    df = df.iloc[offset : offset + n_eval].reset_index(drop=True)
    print(f"Eval set: {len(df)} samples (rows {offset}–{offset + len(df) - 1})", flush=True)

    records = []
    for _, row in df.iterrows():
        text = row["text"] if pd.notna(row["text"]) else ""
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars] + "\n[truncated]"
        date_str = pd.Period(row["date"]).strftime("%B %Y")
        records.append({
            "prompt": build_prompt(tokenizer, text, row["industry"], date_str),
            "label":  "+1" if row["ret_3M_shifted"] > 0 else "-1",
        })
    return records


# ─── Inference ────────────────────────────────────────────────────────────────

def run_eval(llm: LLM, records: list[dict], num_votes: int, max_new_tokens: int) -> list[dict]:
    sampling_params = SamplingParams(
        n=num_votes,
        temperature=0.6,
        max_tokens=max_new_tokens,
    )

    print(f"Generating {len(records)} × {num_votes} completions...", flush=True)
    outputs = llm.generate([r["prompt"] for r in records], sampling_params)

    results = []
    for i, (rec, output) in enumerate(zip(records, outputs)):
        completions = [o.text for o in output.outputs]
        votes = [extract_prediction(c) for c in completions]
        pred  = majority_vote(completions)
        results.append({
            "label":      rec["label"],
            "prediction": pred,
            "votes":      votes,
            "correct":    pred == rec["label"],
        })
        if (i + 1) % 20 == 0:
            n_correct = sum(r["correct"] for r in results)
            print(f"  [{i+1}/{len(records)}]  running acc={n_correct/(i+1):.1%}", flush=True)

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

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Accuracy     : {accuracy:.1%}  ({correct}/{n})")
    print(f"  F1 (+1)      : {f1:.3f}  (on {len(valid)} non-abstained)")
    print(f"  Abstentions  : {abstentions}/{n}  ({abstentions/n:.1%})")
    label_dist = Counter(r["label"] for r in results)
    print(f"  Label dist   : +1={label_dist['+1']}  -1={label_dist['-1']}")
    print(f"{'─'*55}")

    return {"model": label, "accuracy": accuracy, "f1": f1, "abstentions": abstentions, "n": n}


# ─── Plot ─────────────────────────────────────────────────────────────────────

def save_plot(metrics: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, key, title in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "f1",       "F1 Score (+1)"),
    ]:
        bar = ax.bar([metrics["model"]], [metrics[key]], color="#2CA02C", width=0.4)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.text(bar[0].get_x() + bar[0].get_width() / 2, metrics[key] + 0.02,
                f"{metrics[key]:.1%}", ha="center", fontsize=10)

    fig.suptitle(f"Qwen3-4B Zero-shot  —  {metrics['n']} samples", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "qwen3_eval.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}", flush=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Download
    model_path = download_model(args.model_id, args.model_local_dir)

    # 2. Tokenizer (for prompt building only)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 3. Data
    records = load_test_data(args.data_path, args.data_offset, args.n_eval, args.max_prompt_chars, tokenizer)

    # 4. vLLM engine
    print(f"\nLoading vLLM ({args.tensor_parallel_size} GPUs)...", flush=True)
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
    )

    # 5. Eval
    results = run_eval(llm, records, args.num_votes, args.max_new_tokens)

    # 6. Metrics + save
    metrics = compute_metrics(results, "Qwen3-4B (zero-shot, no think)")
    save_plot(metrics, args.output_dir)

    pd.DataFrame([
        {"label": r["label"], "prediction": r["prediction"], "votes": str(r["votes"])}
        for r in results
    ]).to_csv(os.path.join(args.output_dir, "qwen3_predictions.csv"), index=False)

    pd.DataFrame([metrics]).to_csv(os.path.join(args.output_dir, "qwen3_summary.csv"), index=False)
    print(f"\nAll results saved to {args.output_dir}/", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot eval of Qwen3-4B on earnings call return prediction")
    p.add_argument("--model_id",             type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--model_local_dir",      type=str, default="/scratch/models/qwen3-4b",
                   help="Local path to download/load the model")
    p.add_argument("--data_path",            type=str, default="data/merged_data.parquet")
    p.add_argument("--output_dir",           type=str, default="results/qwen3_eval")
    p.add_argument("--data_offset",          type=int, default=2000)
    p.add_argument("--n_eval",               type=int, default=100)
    p.add_argument("--num_votes",            type=int, default=4)
    p.add_argument("--max_new_tokens",       type=int, default=200)
    p.add_argument("--max_prompt_chars",     type=int, default=6000)
    p.add_argument("--tensor_parallel_size", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
