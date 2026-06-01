# pit-stock-llm

Fine-tuning LLMs for stock return prediction using Reinforcement Learning from Verifiable Rewards (RLVR / GRPO).

The model reads earnings call transcripts and predicts whether the stock's next-month return will be positive or negative. We fine-tune the [PIT](https://huggingface.co/Diamegs) family of models and Qwen3-4B using three reward functions (binary, PnL, Gaussian) and evaluate against a zero-shot baseline.

---

## Repository structure

```
pit-stock-llm/
├── submit.sh                   # Launch a Jupyter Lab session on RunAI (sfi-sm project)
├── src/
│   ├── preprocessing/
│   │   ├── pre_process.py          # Merge earnings calls with CRSP returns
│   │   └── preprocess_summarize.py # Summarize raw transcripts with vLLM
│   ├── training/
│   │   ├── rlvr_binary.py          # GRPO fine-tuning — binary ±1 reward (DDP, 3 GPUs)
│   │   ├── rlvr_pnl.py             # GRPO fine-tuning — PnL reward (DDP, 3 GPUs)
│   │   ├── rlvr_gaussian.py        # GRPO fine-tuning — Gaussian reward (DDP, 3 GPUs)
│   │   ├── rlvr_fast.py            # GRPO fine-tuning — single GPU, speed-optimized
│   │   ├── rlvr_single_gpu.py      # GRPO fine-tuning — original single-GPU prototype
│   │   ├── qwen3_rlvr_binary.py    # Qwen3-4B — binary reward
│   │   ├── qwen3_rlvr_pnl.py       # Qwen3-4B — PnL reward
│   │   └── qwen3_rlvr_gaussian.py  # Qwen3-4B — Gaussian reward
│   └── evaluation/
│       ├── baseline.py             # Zero-shot evaluation (no fine-tuning)
│       ├── eval.py                 # Majority-vote @ N evaluation (baseline vs RLVR checkpoints)
│       ├── qwen3_eval.py           # Zero-shot / fine-tuned eval for Qwen3
│       ├── plot_eval.py            # Plot comparison figures from eval outputs
│       └── inspect_generations.py  # Inspect raw model generations
├── scripts/                    # RunAI job submission scripts
│   ├── submit_rlvr_binary.sh
│   ├── submit_rlvr_pnl.sh
│   ├── submit_rlvr_gaussian.sh
│   ├── submit_rlvr_fast.sh
│   ├── submit_qwen3_rlvr.sh
│   ├── submit_eval.sh
│   ├── submit_qwen3_eval.sh
│   ├── submit_qwen3_eval_raw.sh
│   ├── submit_inspect.sh
│   ├── run_baseline.sh
│   ├── run_preprocess.sh
│   ├── run_summarize.sh
│   ├── run_rlvr.sh
│   ├── run_pipeline.sh
│   └── submit_job.py           # End-to-end pipeline orchestrator
├── notebooks/
│   ├── data_analysis.ipynb     # Exploratory analysis of merged dataset
│   └── explo_compression.ipynb # Exploration of transcript compression strategies
├── results/                    # Evaluation outputs (CSVs + plots)
└── data/                       # NOT tracked by git — see setup below
    ├── Predictors/
    │   ├── sm-calls_with_connectors.parquet   # Raw earnings call transcripts
    │   └── sm-calls_summarized_post2018.parquet
    └── Targets/
        └── monthly_crsp.csv                   # Monthly CRSP stock returns
```

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd pit-stock-llm
```

### 2. Recreate the `data/` folder

The `data/` directory is not tracked by git. You need:

| File | Description |
|------|-------------|
| `data/Predictors/sm-calls_with_connectors.parquet` | Raw earnings call transcripts |
| `data/Targets/monthly_crsp.csv` | Monthly CRSP stock returns |

### 3. Preprocess

```bash
# Summarize raw transcripts (requires a GPU + vLLM)
python src/preprocessing/preprocess_summarize.py \
    --input  data/Predictors/sm-calls_with_connectors.parquet \
    --output data/Predictors/sm-calls_summarized_post2018.parquet

# Merge transcripts with stock returns
python src/preprocessing/pre_process.py \
    --input   data/Predictors/sm-calls_summarized_post2018.parquet \
    --returns data/Targets/monthly_crsp.csv \
    --output  data/merged_data.parquet
```

---

## Training

All training scripts target the RunAI cluster (`sfi-sm-bourgon` project, 3×A100 GPUs).

### PIT model — three reward functions

| Script | Reward | Description |
|--------|--------|-------------|
| `scripts/submit_rlvr_binary.sh` | Binary ±1 | +1 if direction correct, −1 otherwise |
| `scripts/submit_rlvr_pnl.sh` | PnL | `sign(pred) × clip(actual_return)` |
| `scripts/submit_rlvr_gaussian.sh` | Gaussian | `exp(-(pred - true)² / 2σ²)` |

```bash
# Example: launch Gaussian reward training
bash scripts/submit_rlvr_gaussian.sh
bash scripts/submit_rlvr_gaussian.sh --test          # smoke test (10 samples)
bash scripts/submit_rlvr_gaussian.sh --sigma=0.01    # tighter tolerance
```

### Qwen3-4B

```bash
bash scripts/submit_qwen3_rlvr.sh binary
bash scripts/submit_qwen3_rlvr.sh gaussian
bash scripts/submit_qwen3_rlvr.sh pnl
bash scripts/submit_qwen3_rlvr.sh all     # launches all 3 in parallel
```

---

## Evaluation

```bash
# Zero-shot baseline (PIT model, no fine-tuning)
bash scripts/run_baseline.sh

# Compare baseline vs RLVR checkpoints (majority-vote @ 4)
bash scripts/submit_eval.sh

# Qwen3 zero-shot eval
bash scripts/submit_qwen3_eval.sh
```

Results are saved to `results/`.

---

## Reward functions

| Name | Formula | Intuition |
|------|---------|-----------|
| Binary | `+1 if sign(pred)==sign(true) else -1` | Only direction matters |
| PnL | `sign(pred) × clip(\|true\|, max_clip)` | Direction + magnitude |
| Gaussian | `exp(-(pred/100 - true)² / 2σ²)` | Soft continuous signal |
