"""
RLVR fine-tuning pipeline v2 — PnL reward.

Reward = sign(prediction) × clip(actual_return, ±reward_clip)
instead of binary +1/-1.  This removes the positive-prediction bias
by making the signal proportional to the magnitude of what was missed.

torchrun --nproc_per_node=3 rlvr_pipeline_ddp_v2.py --model_name ... --data_path ... --output_dir ...
"""

import os
import re
import argparse
import time

import torch
import torch.distributed as dist
import pandas as pd
import wandb
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


def load_dataset(data_path: str, tokenizer: AutoTokenizer, max_prompt_chars: int, n_test: int = 0, args=None) -> Dataset:
    df = pd.read_parquet(data_path)

    required_cols = {"text", "ret_3M_shifted", "industry", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}. Run pre_process.py first.")

    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)

    if n_test > 0:
        df = df.head(n_test)
        print(f"[TEST MODE] Using {len(df)} samples")
    else:
        df = df.iloc[args.data_offset:].reset_index(drop=True)
        print(f"Dataset: {len(df)} samples (offset={args.data_offset})", flush=True)

    df["label"] = df["ret_3M_shifted"].apply(lambda r: "+1" if r > 0 else "-1")

    def format_sample(row):
        text = row["text"]
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars] + "\n[truncated]"
        date_str = pd.Period(row["date"]).strftime("%B %Y")
        return {
            "prompt": build_prompt(text, row["industry"], date_str),
            "label":  row["label"],
            "ret":    float(row["ret_3M_shifted"]),  # actual return for PnL reward
        }

    records = [format_sample(row) for _, row in df.iterrows()]
    return Dataset.from_list(records)


# ─── Reward ───────────────────────────────────────────────────────────────────


def extract_prediction(text: str) -> str | None:
    matches = re.findall(r"([+-]1)\b", text.strip())
    return matches[-1] if matches else None


_REWARD_CLIP = 0.15  # set at module level so the callback can read it; overridden in train()


def pnl_reward(completions: list[str], label: list[str], ret: list[float], **_) -> list[float]:
    """
    PnL reward: sign(prediction) × clip(actual_return, ±_REWARD_CLIP).

    Correct direction on a big move → large positive reward.
    Wrong direction on a big move  → large negative reward.
    Near-zero return               → near-zero reward regardless.
    Abstention (no +1/-1 found)    → 0.0  (neutral, not penalised).
    """
    rewards = []
    for completion, r in zip(completions, ret):
        pred = extract_prediction(completion)
        if pred is None:
            rewards.append(0.0)
        else:
            sign = 1.0 if pred == "+1" else -1.0
            clipped = max(-_REWARD_CLIP, min(_REWARD_CLIP, r))
            rewards.append(sign * clipped)
    return rewards


# ─── Logging callback ─────────────────────────────────────────────────────────


class RewardLogger(TrainerCallback):
    def __init__(self, rank: int, world_size: int, use_wandb: bool):
        self._start           = time.time()
        self._rank            = rank
        self._world_size      = world_size
        self._use_wandb       = use_wandb
        self._step_times: list[float] = []
        self._last_step_time  = time.time()

    def on_train_begin(self, _args, state: TrainerState, _control: TrainerControl, **_kwargs):
        if torch.cuda.is_available():
            device    = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved  = torch.cuda.memory_reserved(device)  / 1024**3
            print(
                f"[rank {self._rank} / GPU {device}] "
                f"{allocated:.1f} GB allocated / {reserved:.1f} GB reserved",
                flush=True,
            )
            if self._use_wandb and state.is_local_process_zero:
                wandb.log({
                    f"system/gpu{device}_allocated_gb": allocated,
                    f"system/gpu{device}_reserved_gb":  reserved,
                }, step=0)

    def on_log(self, _args, state: TrainerState, _control: TrainerControl, logs=None, **_kwargs):
        if logs is None or not state.is_local_process_zero:
            return

        now      = time.time()
        elapsed  = now - self._start
        step_dur = now - self._last_step_time
        self._last_step_time = now
        self._step_times.append(step_dur)

        step       = state.global_step
        reward     = logs.get("reward",        float("nan"))
        reward_std = logs.get("reward_std",    float("nan"))
        loss       = logs.get("loss",          float("nan"))
        kl         = logs.get("kl",            float("nan"))
        lr         = logs.get("learning_rate", float("nan"))

        avg_step_time = sum(self._step_times) / len(self._step_times)

        # GPU memory (this rank's GPU only)
        gpu_allocated_gb = float("nan")
        gpu_reserved_gb  = float("nan")
        if torch.cuda.is_available():
            dev              = torch.cuda.current_device()
            gpu_allocated_gb = torch.cuda.memory_allocated(dev) / 1024**3
            gpu_reserved_gb  = torch.cuda.memory_reserved(dev)  / 1024**3

        print(
            f"[step {step:>5} | {elapsed:6.0f}s] "
            f"pnl_reward={reward:+.4f} ± {reward_std:.4f}  "
            f"loss={loss:.4f}  kl={kl:.4f}  lr={lr:.2e}  "
            f"step_time={step_dur:.1f}s  gpus={self._world_size}",
            flush=True,
        )

        if self._use_wandb:
            dev = torch.cuda.current_device() if torch.cuda.is_available() else 0
            wandb.log({
                # ── Core training metrics ──────────────────────────────────────
                "train/pnl_reward":    reward,
                "train/reward_std":    reward_std,
                "train/loss":          loss,
                "train/kl":            kl,
                "train/learning_rate": lr,
                # ── Throughput ─────────────────────────────────────────────────
                "perf/step_time_s":    step_dur,
                "perf/avg_step_time_s": avg_step_time,
                "perf/elapsed_total_s": elapsed,
                # ── GPU memory ─────────────────────────────────────────────────
                f"system/gpu{dev}_allocated_gb": gpu_allocated_gb,
                f"system/gpu{dev}_reserved_gb":  gpu_reserved_gb,
            }, step=step)

    def on_train_end(self, _args, state: TrainerState, _control: TrainerControl, **_kwargs):
        if not state.is_local_process_zero:
            return
        total_time = time.time() - self._start
        avg_step   = sum(self._step_times) / max(len(self._step_times), 1)
        print(
            f"\nTraining complete — {state.global_step} steps in {total_time:.0f}s "
            f"(avg {avg_step:.1f}s/step)",
            flush=True,
        )
        if self._use_wandb:
            wandb.log({
                "train/total_steps":    state.global_step,
                "perf/total_time_s":    total_time,
                "perf/avg_step_time_s": avg_step,
            }, step=state.global_step)


# ─── Training ─────────────────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    global _REWARD_CLIP
    _REWARD_CLIP = args.reward_clip

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

    # ── W&B init (master only) ────────────────────────────────────────────────
    use_wandb = args.wandb and not args.test
    if master_process and use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name":            args.model_name,
                "data_path":             args.data_path,
                "epochs":                args.epochs,
                "batch_size":            args.batch_size,
                "grad_accum":            args.grad_accum,
                "effective_batch_size":  args.batch_size * ddp_world_size * args.grad_accum,
                "lr":                    args.lr,
                "lora_r":                args.lora_r,
                "lora_alpha":            args.lora_alpha,
                "num_generations":       args.num_generations,
                "max_prompt_chars":      args.max_prompt_chars,
                "max_completion_length": args.max_completion_length,
                "ddp_world_size":        ddp_world_size,
                "n_training_samples":    1000,
                "reward_type":           "pnl",
                "reward_clip":           args.reward_clip,
            },
            tags=["rlvr", "grpo", "ddp", "pnl-reward", args.model_name.split("/")[-1]],
        )
        print(f"W&B run: {wandb.run.url}", flush=True)

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
        if use_wandb:
            wandb.config.update({"model_total_params": total})

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = load_dataset(
        args.data_path, tokenizer,
        max_prompt_chars=args.max_prompt_chars,
        n_test=args.n_test if args.test else 0,
        args=args,
    )
    if master_process:
        n_pos     = sum(1 for l in dataset["label"] if l == "+1")
        n_neg     = sum(1 for l in dataset["label"] if l == "-1")
        eff_batch = args.batch_size * ddp_world_size * (1 if args.test else args.grad_accum)
        print(
            f"Dataset: {len(dataset)} samples | "
            f"+1={n_pos}  -1={n_neg} (balance={n_pos/len(dataset):.1%})\n"
            f"Effective batch: {args.batch_size} × {ddp_world_size} GPUs × "
            f"{1 if args.test else args.grad_accum} accum = {eff_batch}",
            flush=True,
        )
        if use_wandb:
            wandb.config.update({
                "dataset_size":  len(dataset),
                "dataset_pos":   n_pos,
                "dataset_neg":   n_neg,
                "label_balance": n_pos / len(dataset),
            })

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
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
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
        reward_funcs=pnl_reward,
        processing_class=tokenizer,
        callbacks=[RewardLogger(ddp_rank, ddp_world_size, use_wandb)],
    )

    if master_process:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in trainer.model.parameters())
        pct       = 100 * trainable / total
        print(f"LoRA trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({pct:.2f}%)", flush=True)
        if use_wandb:
            wandb.config.update({"lora_trainable_params": trainable, "lora_pct": pct})
        print("─" * 70, flush=True)
        print("Starting GRPO training (DDP)...", flush=True)
        print("─" * 70, flush=True)

    from transformers.trainer_utils import get_last_checkpoint
    resume_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if master_process:
        print(f"Resuming from: {resume_ckpt}" if resume_ckpt else "Starting from scratch", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save from master only (weights are identical across all ranks after DDP)
    if master_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}", flush=True)
        if use_wandb:
            wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLVR pipeline — multi-GPU DDP")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path",  type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Data
    parser.add_argument("--data_offset",           type=int, default=0,
                        help="Skip the first N samples (e.g. 1000 to start after run 1)")
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
    parser.add_argument("--save_steps", type=int,   default=500)

    # LoRA
    parser.add_argument("--lora_r",     type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Reward
    parser.add_argument("--reward_clip", type=float, default=0.15,
                        help="Clip actual returns to ±this value before computing PnL reward")

    # W&B
    parser.add_argument("--wandb",          action="store_true",
                        help="Activer le logging W&B")
    parser.add_argument("--wandb_project",  type=str, default="rlvr-earnings",
                        help="Nom du projet W&B")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Nom du run W&B (auto-généré si absent)")

    # Test mode
    parser.add_argument("--test",   action="store_true")
    parser.add_argument("--n_test", type=int, default=10)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())