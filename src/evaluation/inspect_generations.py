"""
inspect_generations.py — Qualitative generation inspector.

Runs a handful of examples through the RLVR model (and optionally the baseline)
and prints the full raw text each model generates, so you can sanity-check the
quality and style of the responses.

Single GPU, no DDP needed.

Example:
  python src/evaluation/inspect_generations.py \
    --base_model microsoft/Phi-3-mini-4k-instruct \
    --rlvr_checkpoint checkpoints/.../checkpoint-600 \
    --n_samples 5 \
    --num_votes 2
"""

import argparse
import re
import textwrap
from collections import Counter

import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─── Prompt (identical to eval.py) ───────────────────────────────────────────

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


def extract_prediction(text: str):
    matches = re.findall(r"([+-]1)\b", text.strip())
    return matches[-1] if matches else None


def majority_vote(completions: list[str]):
    votes = [extract_prediction(c) for c in completions]
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    counts = Counter(valid)
    top = counts.most_common(2)
    if len(top) == 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_samples(data_path: str, offset: int, n_samples: int, max_prompt_chars: int) -> list[dict]:
    df = pd.read_parquet(data_path)
    df = df.dropna(subset=["ret_3M_shifted"]).reset_index(drop=True)
    df = df.iloc[offset : offset + n_samples].reset_index(drop=True)

    records = []
    for _, row in df.iterrows():
        text = row["text"]
        if len(text) > max_prompt_chars:
            text = text[:max_prompt_chars] + "\n[truncated]"
        date_str = pd.Period(row["date"]).strftime("%B %Y")
        records.append({
            "prompt":   build_prompt(text, row["industry"], date_str),
            "label":    "+1" if row["ret_3M_shifted"] > 0 else "-1",
            "industry": row["industry"],
            "date":     date_str,
            "text_preview": row["text"][:300].replace("\n", " "),
        })
    return records


# ─── Inference ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_completions(
    model, tokenizer, prompt: str, num_votes: int, max_new_tokens: int, device: str
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_votes,
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    return [tokenizer.decode(out[prompt_len:], skip_special_tokens=True) for out in outputs]


# ─── Display ──────────────────────────────────────────────────────────────────

DIVIDER = "═" * 72

def print_sample(idx: int, record: dict, model_label: str, completions: list[str]) -> None:
    pred = majority_vote(completions)
    correct = pred == record["label"]

    print(f"\n{DIVIDER}")
    print(f"  Sample {idx+1}  |  {model_label}")
    print(DIVIDER)
    print(f"  Industry : {record['industry']}")
    print(f"  Date     : {record['date']}")
    print(f"  Label    : {record['label']}  |  Prediction: {pred}  |  {'✓ CORRECT' if correct else '✗ WRONG'}")
    print(f"\n  Transcript preview:")
    print(textwrap.fill(record["text_preview"] + "...", width=70, initial_indent="    ", subsequent_indent="    "))

    for i, comp in enumerate(completions):
        parsed = extract_prediction(comp)
        print(f"\n  ── Generation {i+1}  (parsed: {parsed}) {'─'*40}")
        wrapped = textwrap.fill(comp.strip(), width=70, initial_indent="    ", subsequent_indent="    ")
        print(wrapped if wrapped else "    [empty]")

    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    records = load_samples(args.data_path, args.data_offset, args.n_samples, args.max_prompt_chars)
    print(f"Loaded {len(records)} samples (offset={args.data_offset})")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base model ────────────────────────────────────────────────────────────
    print(f"\nLoading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    models_to_run = []
    if not args.rlvr_only:
        models_to_run.append(("Baseline", base_model))

    if args.rlvr_checkpoint:
        print(f"Loading LoRA adapter: {args.rlvr_checkpoint}")
        rlvr_model = PeftModel.from_pretrained(base_model, args.rlvr_checkpoint).eval()
        models_to_run.append(("RLVR", rlvr_model))

    # ── Generate & display ────────────────────────────────────────────────────
    for model_label, model in models_to_run:
        print(f"\n{'#'*72}")
        print(f"#  {model_label}")
        print(f"{'#'*72}")
        for i, record in enumerate(records):
            completions = generate_completions(
                model, tokenizer, record["prompt"],
                args.num_votes, args.max_new_tokens, device
            )
            print_sample(i, record, model_label, completions)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect raw model generations for a few samples")
    p.add_argument("--base_model",       type=str, required=True)
    p.add_argument("--rlvr_checkpoint",  type=str, default=None,
                   help="Path to LoRA checkpoint dir. Omit to run baseline only.")
    p.add_argument("--data_path",        type=str, default="data/merged_data.parquet")
    p.add_argument("--data_offset",      type=int, default=2000,
                   help="Same offset used in eval.py so you look at test samples")
    p.add_argument("--n_samples",        type=int, default=5,
                   help="Number of examples to inspect")
    p.add_argument("--num_votes",        type=int, default=2,
                   help="Generations per sample (keep low for quick inspection)")
    p.add_argument("--max_new_tokens",   type=int, default=300)
    p.add_argument("--max_prompt_chars", type=int, default=6000)
    p.add_argument("--rlvr_only",        action="store_true",
                   help="Skip baseline, only run RLVR model")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
