"""
RLVR fine-tuning of Qwen3-4B — Gaussian reward.

Reward : exp(-((pred_decimal - true_ret)² / (2σ²)))  ∈ (0, 1]
         abstention → 0.0
Thinking: OFF — direct answer, let the model develop its own reasoning.
Speed   : vLLM backend for generation (5-10× faster than transformers).
GPU     : single GPU.

python qwen3_rlvr_gaussian.py \
  --model_path /home/bourgon/models/qwen3-4b \
  --data_path  data/merged_data.parquet \
  --output_dir checkpoints/qwen3-gaussian-v1
"""

import re, os, time, argparse, math
import torch
import pandas as pd
import wandb
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import get_last_checkpoint

_SIGMA = 0.05  # ~5% return std for Gaussian kernel


# ─── Prompt ───────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, text: str, industry: str, date: str) -> str:
    instruction = (
        f"Date: {date}\n"
        f"Industry: {industry}\n"
        f"Earnings Call Transcript:\n{text}\n\n"
        "Based on this earnings call, predict the stock's 1-month return as a percentage. "
        "Positive means price increase, negative means price decrease. "
        "Typical returns range from -5% to +5%.\n"
        "Answer in percentage (e.g. +2.3 or -1.5):"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_prediction(text: str) -> float | None:
    search = text.strip()
    # Primary: number followed by %
    matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", search)
    if not matches:
        # Fallback: signed number
        matches = re.findall(r"([+-]\d+(?:\.\d+)?)", search)
    if not matches:
        return None
    try:
        return float(matches[-1]) / 100.0  # convert % to decimal
    except ValueError:
        return None


# ─── Reward ───────────────────────────────────────────────────────────────────

def reward_fn(completions: list[str], ret: list[float], **_) -> list[float]:
    rewards = []
    for completion, true_ret in zip(completions, ret):
        pred = extract_prediction(completion)
        if pred is None:
            rewards.append(0.0)
        else:
            diff = pred - true_ret
            rewards.append(math.exp(-(diff ** 2) / (2 * _SIGMA ** 2)))
    return rewards


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data(data_path: str, tokenizer, max_prompt_chars: int, data_offset: int, n_test: int = 0) -> Dataset:
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)

    if n_test > 0:
        df = df.head(n_test)
        print(f"[TEST MODE] {len(df)} samples", flush=True)
    else:
        df = df.iloc[data_offset:].reset_index(drop=True)
        print(f"Dataset: {len(df)} samples (offset={data_offset})", flush=True)

    records = []
    for _, row in df.iterrows():
        text = row["text"]
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars] + "\n[truncated]"
        date_str = pd.Period(row["date"]).strftime("%B %Y")
        records.append({
            "prompt": build_prompt(tokenizer, text, row["industry"], date_str),
            "ret":    float(row["ret_3M_shifted"]),
        })
    return Dataset.from_list(records)


# ─── Callback ─────────────────────────────────────────────────────────────────

class RewardLogger(TrainerCallback):
    def __init__(self, use_wandb: bool):
        self._start     = time.time()
        self._use_wandb = use_wandb
        self._steps     = []
        self._last      = time.time()

    def on_log(self, _args, state: TrainerState, _control, logs=None, **_kw):
        if not logs or not state.is_local_process_zero:
            return
        now  = time.time()
        dur  = now - self._last
        self._last = now
        self._steps.append(dur)

        reward = logs.get("reward", float("nan"))
        print(
            f"[step {state.global_step:>5} | {now-self._start:6.0f}s]  "
            f"reward={reward:+.4f}  "
            f"loss={logs.get('loss', float('nan')):.4f}  "
            f"kl={logs.get('kl', float('nan')):.4f}  "
            f"step={dur:.1f}s",
            flush=True,
        )
        if self._use_wandb:
            wandb.log({
                "train/reward":   reward,
                "train/loss":     logs.get("loss", float("nan")),
                "train/kl":       logs.get("kl",   float("nan")),
                "train/lr":       logs.get("learning_rate", float("nan")),
                "perf/step_time": dur,
            }, step=state.global_step)

    def on_train_end(self, _args, state: TrainerState, _control, **_kw):
        if not state.is_local_process_zero:
            return
        total = time.time() - self._start
        avg   = sum(self._steps) / max(len(self._steps), 1)
        print(f"\nDone — {state.global_step} steps in {total:.0f}s (avg {avg:.1f}s/step)", flush=True)
        if self._use_wandb:
            wandb.finish()


# ─── Train ────────────────────────────────────────────────────────────────────

def train(args):
    torch.set_float32_matmul_precision("high")

    use_wandb = args.wandb and not args.test
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or "qwen3-gaussian",
            config=vars(args),
            tags=["rlvr", "grpo", "qwen3", "gaussian", "thinking"],
        )
        print(f"W&B: {wandb.run.url}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_data(
        args.data_path, tokenizer, args.max_prompt_chars,
        args.data_offset, n_test=args.n_test if args.test else 0,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=2 if args.test else -1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        num_generations=2 if args.test else args.num_generations,
        generation_batch_size=2 if args.test else args.batch_size * args.num_generations,
        max_completion_length=64 if args.test else args.max_completion_length,
        temperature=0.9,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        use_vllm=True,
        vllm_gpu_memory_utilization=0.3,
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=[RewardLogger(use_wandb)],
    )

    resume = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    print(f"{'Resuming from: ' + resume if resume else 'Starting from scratch'}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved → {args.output_dir}", flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",            type=str, required=True)
    p.add_argument("--data_path",             type=str, required=True)
    p.add_argument("--output_dir",            type=str, required=True)
    p.add_argument("--data_offset",           type=int, default=0)
    p.add_argument("--max_prompt_chars",      type=int, default=6000)
    p.add_argument("--max_prompt_length",     type=int, default=2048)
    p.add_argument("--max_completion_length", type=int, default=512)
    p.add_argument("--num_generations",       type=int, default=6)
    p.add_argument("--epochs",                type=int, default=1)
    p.add_argument("--batch_size",            type=int, default=1)
    p.add_argument("--grad_accum",            type=int, default=8)
    p.add_argument("--lr",                    type=float, default=5e-6)
    p.add_argument("--save_steps",            type=int, default=100)
    p.add_argument("--lora_r",                type=int, default=16)
    p.add_argument("--lora_alpha",            type=int, default=32)
    p.add_argument("--wandb",                 action="store_true")
    p.add_argument("--wandb_project",         type=str, default="rlvr-earnings-qwen3")
    p.add_argument("--wandb_run_name",        type=str, default=None)
    p.add_argument("--test",                  action="store_true")
    p.add_argument("--n_test",                type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
