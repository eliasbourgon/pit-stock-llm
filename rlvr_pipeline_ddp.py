"""
RLVR fine-tuning pipeline — multi-GPU DDP version.
Based on rlvr_pipeline_fast.py, DDP init pattern from post_training/sft.py.

torchrun --nproc_per_node=3 rlvr_pipeline_ddp.py --model_name ... --data_path ... --output_dir ...
"""

import os
import re
import argparse
import time

import torch
import torch.distributed as dist
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
        date_str = pd.Period(row["date"]).strftime("%B %Y")
        return {"prompt": build_prompt(text, row["industry"], date_str), "label": row["label"]}

    records = [format_sample(row) for _, row in df.iterrows()]
    return Dataset.from_list(records)


# ─── Reward ───────────────────────────────────────────────────────────────────


def extract_prediction(text: str) -> str | None:
    matches = re.findall(r"([+-]1)\b", text.strip())
    return matches[-1] if matches else None


def reward_fn(completions: list[str], label: list[str], **_) -> list[float]:
    rewards = []
    for completion, lbl in zip(completions, label):
        pred = extract_prediction(completion)
        rewards.append(1.0 if pred == lbl else -1.0)
    return rewards


# ─── Logging callback ─────────────────────────────────────────────────────────


class RewardLogger(TrainerCallback):
    def __init__(self, rank: int, world_size: int):
        self._start = time.time()
        self._rank = rank
        self._world_size = world_size

    def on_train_begin(self, _args, state: TrainerState, _control: TrainerControl, **_kwargs):
        # Each rank logs its own GPU — torch.cuda.memory_allocated(i) only sees
        # memory allocated by the current process, so rank 0 cannot report other GPUs.
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved  = torch.cuda.memory_reserved(device)  / 1024**3
            print(f"[rank {self._rank} / GPU {device}] {allocated:.1f} GB allocated / {reserved:.1f} GB reserved", flush=True)

    def on_log(self, _args, state: TrainerState, _control: TrainerControl, logs=None, **_kwargs):
        if logs is None or not state.is_local_process_zero:
            return
        elapsed = time.time() - self._start
        step       = state.global_step
        reward     = logs.get("reward", float("nan"))
        reward_std = logs.get("reward_std", float("nan"))
        loss       = logs.get("loss", float("nan"))
        kl         = logs.get("kl", float("nan"))
        lr         = logs.get("learning_rate", float("nan"))
        print(
            f"[step {step:>5} | {elapsed:6.0f}s] "
            f"reward={reward:+.3f} ± {reward_std:.3f}  "
            f"loss={loss:.4f}  kl={kl:.4f}  lr={lr:.2e}  "
            f"gpus={self._world_size}",
            flush=True,
        )


# ─── Training ─────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    # Free ~10-20% on Ampere+ GPUs via TF32 tensor cores (same as sft.py)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # ── DDP setup — same pattern as post_training/sft.py ─────────────────────
    assert torch.cuda.is_available(), "CUDA required"
    dist.init_process_group(backend="nccl")
    ddp_rank       = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device         = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)

    if master_process:
        print(f"DDP: {ddp_world_size} GPUs, this is rank {ddp_rank}", flush=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model — loaded on CPU in bf16, GRPOTrainer places it on `device` ──────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    if master_process:
        total = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total/1e9:.2f}B parameters", flush=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(
        args.data_path, tokenizer,
        max_prompt_chars=args.max_prompt_chars,
        n_test=args.n_test if args.test else 0,
    )
    if master_process:
        eff_batch = args.batch_size * ddp_world_size * (1 if args.test else args.grad_accum)
        print(
            f"Dataset: {len(dataset)} samples | "
            f"+1={sum(1 for l in dataset['label'] if l=='+1')}  "
            f"-1={sum(1 for l in dataset['label'] if l=='-1')}\n"
            f"Effective batch: {args.batch_size} × {ddp_world_size} GPUs × "
            f"{1 if args.test else args.grad_accum} accum = {eff_batch}",
            flush=True,
        )

    # ── LoRA — same target modules as sft.py ──────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["c_q", "c_k", "c_v", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── GRPO config ───────────────────────────────────────────────────────────
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
        generation_batch_size=ddp_world_size * 2 if args.test else args.batch_size * args.num_generations,
        max_completion_length=32 if args.test else args.max_completion_length,
        temperature=0.9,
        gradient_checkpointing=False,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="none",
        # LoRA only trains a subset of params — unused params would cause DDP errors
        ddp_find_unused_parameters=False,
        # Avoid NCCL timeout on long transcript generation
        ddp_timeout=1800,
    )

    # GRPOTrainer detects the already-initialized process group and wraps
    # the model with DDP internally (via accelerate). No manual DDP() needed.
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=[RewardLogger(ddp_rank, ddp_world_size)],
    )

    if master_process:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in trainer.model.parameters())
        print(f"LoRA trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({100*trainable/total:.2f}%)", flush=True)
        print("─" * 70, flush=True)
        print("Starting GRPO training (DDP)...", flush=True)
        print("─" * 70, flush=True)

    trainer.train()

    # Save from master only (weights are identical across all ranks after DDP)
    if master_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLVR pipeline — multi-GPU DDP")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Data
    parser.add_argument("--max_prompt_chars",      type=int, default=6000)
    parser.add_argument("--max_prompt_length",     type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=512)

    # GRPO
    parser.add_argument("--num_generations", type=int, default=6)

    # Training
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--save_steps", type=int,   default=100)

    # LoRA
    parser.add_argument("--lora_r",     type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Test mode
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--n_test", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
