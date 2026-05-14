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
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(text: str, industry: str, date: str) -> str:
    # Completion-style prompt for GPT-2 base architecture (no chat template)
    # Matches baseline.py format exactly for fair comparison
    return (
        f"Date: {date}\n"
        f"Industry: {industry}\n"
        f"Earnings Call Transcript:\n{text}\n\n"
        "Financial Analysis:\n<think>"
    )


# ─── Data ─────────────────────────────────────────────────────────────────────


def load_dataset(data_path: str, tokenizer: AutoTokenizer, max_prompt_chars: int, n_test: int = 0) -> Dataset:
    df = pd.read_parquet(data_path)

    required_cols = {"text", "ret_3M_shifted", "industry", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}. Run pre_process.py first.")

    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)
    if n_test > 0:
        df = df.head(n_test)
        print(f"[TEST MODE] Using {len(df)} samples")

    df["label"] = df["ret_3M_shifted"].apply(lambda r: "+1" if r > 0 else "-1")

    def format_sample(row):
        text = row["text"]
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars] + "\n[truncated]"
        date_str = row["date"].strftime("%B %Y")
        return {"prompt": build_prompt(text, row["industry"], date_str), "label": row["label"]}

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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    dataset = load_dataset(
        args.data_path, tokenizer,
        max_prompt_chars=args.max_prompt_chars,
        n_test=args.n_test if args.test else 0,
    )
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
        max_steps=1 if args.test else -1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_generations=args.num_generations,
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
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
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

    # Test mode
    parser.add_argument("--test",   action="store_true", help="Smoke test: 1 training step on --n_test samples")
    parser.add_argument("--n_test", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
