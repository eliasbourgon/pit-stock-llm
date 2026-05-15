"""
Zero-shot baseline: run the frozen LLM on the return-prediction task with no fine-tuning.

Dataset: data/merged_data.parquet
  columns: date (period[M]), text (earnings transcript), ret_3M_shifted (float), industry (str)

Outputs per-sample predictions and aggregate metrics (accuracy, F1).

Usage:
    python baseline.py \
        --model_name Diamegs/PIT-4B-FT-201312 \
        [--data_path data/merged_data.parquet] \
        [--max_prompt_chars 6000] \
        [--max_new_tokens 256] \
        [--gpu_memory_utilization 0.9] \
        [--output_csv baseline_results.csv]
"""

import re
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(text: str, industry: str, date: str) -> str:
    return (
        f"Date: {date}\n"
        f"Industry: {industry}\n"
        f"Earnings Call Transcript:\n{text}\n\n"
        "Based on this earnings call, predict whether the stock's 1-month return "
        "will be positive (+1) or negative (-1).\n"
        "Answer (+1 or -1):"
    )


# ─── Prediction extraction ────────────────────────────────────────────────────

def extract_prediction(text: str) -> str | None:
    match = re.search(r"([+-]1)\b", text.strip())
    return match.group(1) if match else None


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data(data_path: str, max_prompt_chars: int, n_test: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(data_path)

    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)
    if n_test > 0:
        df = df.head(n_test)
        print(f"[TEST MODE] Using {len(df)} samples")

    df["label"] = df["ret_3M_shifted"].apply(lambda r: "+1" if r > 0 else "-1")

    def make_prompt(row):
        t = row["text"]
        if len(t) > max_prompt_chars:
            t = t[:max_prompt_chars] + "\n[truncated]"
        date_str = row["date"].strftime("%B %Y")  # e.g. "November 2011"
        return build_prompt(t, row["industry"], date_str)

    df["prompt"] = df.apply(make_prompt, axis=1)
    return df


# ─── Inference ────────────────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    model_name: str,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    temperature: float = 0.0,
) -> pd.DataFrame:
    print(f"Loading {model_name} with HuggingFace Transformers ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    do_sample = temperature > 0.0
    raw_outputs = []
    for prompt in df["prompt"].tolist():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    df = df.copy()
    df["raw_output"] = raw_outputs
    df["prediction"] = df["raw_output"].apply(extract_prediction)
    return df


# ─── Metrics ──────────────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame) -> None:
    print("\n─── Raw outputs (debug) ────────────────────────")
    for i, row in df.iterrows():
        print(f"  [{i}] prediction={row['prediction']} | raw: {repr(row['raw_output'][:200])}")

    parseable = df[df["prediction"].notna()]
    unparseable = len(df) - len(parseable)

    if len(parseable) == 0:
        print(f"  No parseable predictions ({unparseable} unparseable). Check raw outputs above.")
        return

    y_true = parseable["label"].tolist()
    y_pred = parseable["prediction"].tolist()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label="+1", average="binary")
    cm  = confusion_matrix(y_true, y_pred, labels=["+1", "-1"])

    print("\n─── Baseline Results ───────────────────────────")
    print(f"  Samples evaluated  : {len(parseable)} / {len(df)}")
    print(f"  Unparseable outputs: {unparseable}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  F1 (+1 class)      : {f1:.4f}")
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"    {'':6} {'pred+1':>8} {'pred-1':>8}")
    print(f"    {'true+1':6} {cm[0,0]:>8} {cm[0,1]:>8}")
    print(f"    {'true-1':6} {cm[1,0]:>8} {cm[1,1]:>8}")
    print("────────────────────────────────────────────────\n")


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, output_prefix: str) -> None:
    parseable = df[df["prediction"].notna()]
    if len(parseable) == 0:
        print("Skipping plots: no parseable predictions.")
        return
    y_true = parseable["label"].tolist()
    y_pred = parseable["prediction"].tolist()

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, pos_label="+1", average="binary")
    cm  = confusion_matrix(y_true, y_pred, labels=["+1", "-1"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Zero-shot Baseline — 3-Month Return Direction Prediction", fontsize=13, fontweight="bold")

    # 1. Confusion matrix
    ax = axes[0]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["pred +1", "pred -1"],
        yticklabels=["true +1", "true -1"],
        ax=ax, cbar=False,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # 2. Bar chart: accuracy & F1
    ax = axes[1]
    metrics = {"Accuracy": acc, "F1 (+1)": f1}
    bars = ax.bar(metrics.keys(), metrics.values(), color=["steelblue", "darkorange"], width=0.4)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Metrics")
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.1%}",
                ha="center", va="bottom", fontsize=11)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="random baseline")
    ax.legend(fontsize=8)

    # 3. Prediction distribution vs true labels
    ax = axes[2]
    unparseable_count = len(df) - len(parseable)
    counts = pd.DataFrame({
        "True":      [sum(l == "+1" for l in y_true), sum(l == "-1" for l in y_true)],
        "Predicted": [sum(p == "+1" for p in y_pred), sum(p == "-1" for p in y_pred)],
    }, index=["+1", "-1"])
    counts.plot(kind="bar", ax=ax, color=["steelblue", "darkorange"], edgecolor="white", width=0.6)
    ax.set_title(f"Label Distribution\n(unparseable: {unparseable_count})")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = f"{output_prefix}_plots.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plots saved to {out_path}")
    plt.show()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot baseline for PIT 3-month return prediction")
    p.add_argument("--model_name", required=True, help="HuggingFace model ID")
    p.add_argument("--data_path",  default="data/merged_data.parquet",
                   help=".parquet with columns [date, text, ret_3M_shifted, industry]")
    p.add_argument("--max_prompt_chars", type=int, default=6000)
    p.add_argument("--max_new_tokens",          type=int,   default=256)
    p.add_argument("--gpu_memory_utilization",  type=float, default=0.9,
                   help="Fraction of GPU memory vLLM may use (default 0.9)")
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Sampling temperature (0.0 = greedy, >0 = stochastic)")
    p.add_argument("--output_csv", type=str, default="baseline_results.csv",
                   help="Path to save per-sample predictions")
    p.add_argument("--n_test", type=int, default=0,
                   help="If > 0, only evaluate on first N samples (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = load_data(args.data_path, args.max_prompt_chars, n_test=args.n_test)
    print(f"Loaded {len(df)} samples | +1={sum(df['label']=='+1')} -1={sum(df['label']=='-1')}")

    df = run_inference(df, args.model_name, args.max_new_tokens, args.gpu_memory_utilization, args.temperature)

    evaluate(df)

    df[["date", "industry", "ret_3M_shifted", "label", "prediction", "raw_output"]].to_csv(
        args.output_csv, index=False
    )
    print(f"Per-sample results saved to {args.output_csv}")

    output_prefix = args.output_csv.rsplit(".", 1)[0]
    plot_results(df, output_prefix)
