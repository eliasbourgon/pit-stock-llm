"""
RLVR fine-tuning pipeline for PIT stock return prediction.

Input : .parquet with columns [transcript, return]
Output: +1 / -1 direction prediction (30-day return)
Method: GRPO (Group Relative Policy Optimization) via TRL
"""

import re
import argparse

import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer

# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(transcript: str) -> str:
    # Completion-style prompt for GPT-2 base architecture (no chat template)
    return (
        f"Earnings Call Transcript:\n{transcript}\n\n"
        "Financial Analysis:\n<think>"
    )


# ─── Data ─────────────────────────────────────────────────────────────────────


def load_dataset(data_path: str, tokenizer: AutoTokenizer, max_prompt_chars: int) -> Dataset:
    df = pd.read_parquet(data_path)

    required_cols = {"transcript", "return"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")

    df["label"] = df["return"].apply(lambda r: "+1" if r > 0 else "-1")

    def format_sample(row):
        transcript = row["transcript"]
        if len(transcript) > max_prompt_chars:
            transcript = transcript[:max_prompt_chars] + "\n[truncated]"
        return {"prompt": build_prompt(transcript), "label": row["label"]}

    records = [format_sample(row) for _, row in df.iterrows()]
    return Dataset.from_list(records)


# ─── Reward ───────────────────────────────────────────────────────────────────


def extract_prediction(text: str) -> str | None:
    """Extract +1 or -1 from model output, searching after </think> if present."""
    parts = re.split(r"</think>", text, maxsplit=1)
    search_in = parts[-1] if len(parts) > 1 else text
    match = re.search(r"([+-]1)\b", search_in.strip())
    return match.group(1) if match else None


def reward_fn(completions: list[str], label: list[str], **_) -> list[float]:
    """
    +1.0 if prediction matches label, -1.0 otherwise (including unparseable outputs).
    `label` is injected automatically by GRPOTrainer from the dataset column.
    """
    rewards = []
    for completion, lbl in zip(completions, label):
        pred = extract_prediction(completion)
        rewards.append(1.0 if pred == lbl else -1.0)
    return rewards


# ─── Training ─────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(args.data_path, tokenizer, max_prompt_chars=args.max_prompt_chars)
    print(f"Dataset size: {len(dataset)} samples | label distribution: "
          f"+1={sum(1 for l in dataset['label'] if l=='+1')} "
          f"-1={sum(1 for l in dataset['label'] if l=='-1')}")

    # GPT-2 architecture: combined QKV projection (c_attn), output (c_proj), MLP (c_fc)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=0.9,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model_name,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        reward_funcs=reward_fn,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLVR pipeline for PIT stock return prediction")
    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model ID, e.g. Diamegs/PIT-4B-FT-201312")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .parquet file with [transcript, return] columns")
    parser.add_argument("--output_dir", type=str, required=True)

    # Data
    parser.add_argument("--max_prompt_chars", type=int, default=6000,
                        help="Max transcript characters before truncation")
    parser.add_argument("--max_prompt_length", type=int, default=2048,
                        help="Max tokens for the prompt (GRPO tokenizer truncation)")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Max tokens for the generated completion (CoT + answer)")

    # GRPO
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Completions per prompt for group-relative advantage (higher=better estimates, more VRAM)")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--save_steps", type=int, default=100)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
