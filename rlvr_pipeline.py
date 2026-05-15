"""
RLVR fine-tuning pipeline for PIT stock return prediction.

Input : .parquet with columns [transcript, return]
Output: +1 / -1 direction prediction (30-day return)
Method: GRPO (Group Relative Policy Optimization) via TRL
"""

import re
import argparse
import time

import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, TrainerState, TrainerControl

# ─── Prompt ───────────────────────────────────────────────────────────────────

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
    matches = re.findall(r"([+-]1)\b", text.strip())
    return matches[-1] if matches else None


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


# ─── Logging callback ─────────────────────────────────────────────────────────


class RewardLogger(TrainerCallback):
    def __init__(self):
        self._start = time.time()

    def on_train_begin(self, _args, state: TrainerState, _control: TrainerControl, **_kwargs):
        if state.is_local_process_zero and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved  = torch.cuda.memory_reserved(i)  / 1024**3
                print(f"[GPU {i}] {allocated:.1f} GB allocated / {reserved:.1f} GB reserved", flush=True)

    def on_log(self, _args, state: TrainerState, _control: TrainerControl, logs=None, **_kwargs):
        if logs is None or not state.is_local_process_zero:
            return
        elapsed = time.time() - self._start
        step = state.global_step
        reward     = logs.get("reward", float("nan"))
        reward_std = logs.get("reward_std", float("nan"))
        loss       = logs.get("loss", float("nan"))
        kl         = logs.get("kl", float("nan"))
        lr         = logs.get("learning_rate", float("nan"))
        print(
            f"[step {step:>5} | {elapsed:6.0f}s] "
            f"reward={reward:+.3f} ± {reward_std:.3f}  "
            f"loss={loss:.4f}  kl={kl:.4f}  lr={lr:.2e}",
            flush=True,
        )


# ─── Training ─────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    is_main = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    if is_main:
        total  = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total/1e9:.2f}B parameters", flush=True)

    dataset = load_dataset(
        args.data_path, tokenizer,
        max_prompt_chars=args.max_prompt_chars,
        n_test=args.n_test if args.test else 0,
    )
    if is_main:
        print(f"Dataset size: {len(dataset)} samples | label distribution: "
              f"+1={sum(1 for l in dataset['label'] if l=='+1')} "
              f"-1={sum(1 for l in dataset['label'] if l=='-1')}", flush=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=1 if args.test else -1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if args.test else args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_generations=2 if args.test else args.num_generations,
        generation_batch_size=2 if args.test else args.batch_size * args.num_generations,
        max_completion_length=32 if args.test else args.max_completion_length,
        temperature=0.9,
        gradient_checkpointing=False,
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
        callbacks=[RewardLogger()],
    )

    if is_main:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in trainer.model.parameters())
        print(f"LoRA trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
              f"({100*trainable/total:.2f}%)", flush=True)

    if is_main:
        print("─" * 70, flush=True)
        print("Starting GRPO training...", flush=True)
        print("─" * 70, flush=True)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if is_main:
        print(f"Model saved to {args.output_dir}", flush=True)


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
