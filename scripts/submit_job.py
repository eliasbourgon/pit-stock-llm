"""
submit_job.py
─────────────
Orchestrates the full PIT RLVR fine-tuning pipeline:

  Step 1 — preprocess_summarize : summarize earnings calls with vLLM
  Step 2 — pre_process          : merge with CRSP returns + map SIC → industry
  Step 3 — baseline             : zero-shot evaluation before fine-tuning
  Step 4 — rlvr_single_gpu      : GRPO fine-tuning (single GPU)

Full run:
  python scripts/submit_job.py \
    --model_name Diamegs/PIT-4B-FT-201912 \
    --output_dir checkpoints/pit-2019-rlvr

Smoke test (10 samples, verify plumbing end-to-end):
  python scripts/submit_job.py \
    --model_name Diamegs/PIT-4B-FT-201912 \
    --output_dir checkpoints/pit-2019-rlvr \
    --test

Skip steps already done:
  python scripts/submit_job.py ... --skip_summarize --skip_preprocess
"""

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(model_name: str, model_dir: str) -> str:
    """Download model from HuggingFace to local dir. Returns the local path."""
    local_path = Path(model_dir)
    if local_path.exists() and any(local_path.iterdir()):
        print(f"[submit_job] Model already at {local_path} — skipping download.")
        return str(local_path)
    print(f"[submit_job] Downloading {model_name} → {local_path} ...")
    snapshot_download(repo_id=model_name, local_dir=str(local_path))
    print(f"[submit_job] Download complete.")
    return str(local_path)


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'='*64}")
    print(f"  {step}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*64}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[submit_job] FAILED at '{step}' (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)
    print(f"[submit_job] {step} — OK")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIT fine-tuning pipeline orchestrator")

    # Required
    p.add_argument("--model_name", required=True,
                   help="HuggingFace model ID, e.g. Diamegs/PIT-4B-FT-201912")
    p.add_argument("--output_dir", required=True,
                   help="Where to save the fine-tuned model")

    # Test mode
    p.add_argument("--test",   action="store_true",
                   help="Smoke test: run each step on --n_test samples only")
    p.add_argument("--n_test", type=int, default=10,
                   help="Number of samples in test mode (default: 10)")

    # Skip flags
    p.add_argument("--skip_summarize",  action="store_true",
                   help="Skip step 1 — summarized parquet already exists")
    p.add_argument("--skip_preprocess", action="store_true",
                   help="Skip step 2 — merged_data.parquet already exists")
    p.add_argument("--skip_baseline",   action="store_true",
                   help="Skip step 3 — baseline evaluation")

    # Paths (overridable)
    p.add_argument("--model_dir",          default="models/pit-model",
                   help="Local dir where the HF model will be downloaded (on PVC)")
    p.add_argument("--raw_parquet",        default="data/sm-calls_with_connectors.parquet")
    p.add_argument("--summarized_parquet", default="data/sm-calls_summarized_post2018.parquet")
    p.add_argument("--returns_csv",        default="data/Targets/monthly_crsp.csv")
    p.add_argument("--merged_parquet",     default="data/merged_data.parquet")
    p.add_argument("--baseline_output",    default="baseline_results.csv")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    # ── Step 0: Download PIT model from HuggingFace ───────────────────────────
    local_model = download_model(args.model_name, args.model_dir)

    # ── Step 1: Summarize ──────────────────────────────────────────────────────
    if not args.skip_summarize:
        cmd = [
            py, "src/preprocessing/preprocess_summarize.py",
            "--input",  args.raw_parquet,
            "--output", args.summarized_parquet,
        ]
        if args.test:
            cmd += ["--test", "--n_test", str(args.n_test)]
        run(cmd, "Step 1 — preprocess_summarize")
    else:
        print("[submit_job] Step 1 skipped (--skip_summarize)")

    # ── Step 2: Merge with returns ────────────────────────────────────────────
    if not args.skip_preprocess:
        cmd = [
            py, "src/preprocessing/pre_process.py",
            "--input",   args.summarized_parquet,
            "--returns", args.returns_csv,
            "--output",  args.merged_parquet,
        ]
        run(cmd, "Step 2 — pre_process")
    else:
        print("[submit_job] Step 2 skipped (--skip_preprocess)")

    # ── Step 3: Baseline ──────────────────────────────────────────────────────
    if not args.skip_baseline:
        cmd = [
            py, "src/evaluation/baseline.py",
            "--model_name", local_model,
            "--data_path",  args.merged_parquet,
            "--output_csv", args.baseline_output,
        ]
        if args.test:
            cmd += ["--n_test", str(args.n_test)]
        run(cmd, "Step 3 — baseline")
    else:
        print("[submit_job] Step 3 skipped (--skip_baseline)")

    # ── Step 4: RLVR fine-tuning ──────────────────────────────────────────────
    cmd = [
        py, "src/training/rlvr_single_gpu.py",
        "--model_name", local_model,
        "--data_path",  args.merged_parquet,
        "--output_dir", args.output_dir,
    ]
    if args.test:
        cmd += ["--test", "--n_test", str(args.n_test)]
    run(cmd, "Step 4 — rlvr_single_gpu (GRPO fine-tuning)")

    print(f"\n[submit_job] Pipeline complete. Fine-tuned model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
