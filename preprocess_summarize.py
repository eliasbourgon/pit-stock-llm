"""
preprocess_summarize.py
───────────────────────
Summarizes all earnings calls after 2018 using vLLM + Qwen2.5-14B-Instruct.
- Filters rows where mostimportantdateutc > 2018
- Resumes from checkpoint if interrupted
- Saves checkpoint every CHECKPOINT_EVERY batches
- Supports tensor_parallel across multiple GPUs (3x A100 40GB)
- Fallback: skips rows that cause errors, logs them separately
"""

import os
import sys
import time
import logging
import argparse
import traceback
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (defaults — overridable via CLI)
# ─────────────────────────────────────────────────────────────────────────────

INPUT_PATH        = "sm-calls_with_connectors.parquet"
OUTPUT_PATH       = "sm-calls_summarized_post2018.parquet"
CHECKPOINT_PATH   = "sm-calls_summarized_checkpoint.parquet"
ERROR_LOG_PATH    = "sm-calls_errors.parquet"
LOG_PATH          = "preprocess_summarize.log"

MODEL_NAME        = "Qwen/Qwen2.5-14B-Instruct"
DATE_COL          = "mostimportantdateutc"
ID_COL            = "transcriptid"
TEXT_COL          = "text"
TEXT_LENGTH_COL   = "text_length"

YEAR_FILTER       = 2018          # strictement après cette année
BATCH_SIZE        = 32            # nb de prompts envoyés à vLLM en une fois
CHECKPOINT_EVERY  = 5             # sauvegarde checkpoint tous les N batches
MAX_INPUT_CHARS   = 120_000       # ~30k tokens Qwen — tronque si dépassé
MAX_TOKENS_OUT    = 600           # tokens de sortie max (un peu de marge vs 500)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior financial analyst specializing in earnings call analysis.
Your summaries will be used as input to a machine learning model that predicts stock returns
following earnings releases. Every word must carry signal. Omit anything a quant would ignore."""


def build_prompt(transcript: str) -> str:
    # Tronque si trop long pour éviter OOM
    if len(transcript) > MAX_INPUT_CHARS:
        transcript = transcript[:MAX_INPUT_CHARS]

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Summarize this earnings call transcript in approximately 500 words.\n"
        f"Focus exclusively on what moves stock prices after earnings:\n"
        f"- Quantified performance vs expectations (revenue, EPS, margins)\n"
        f"- Forward guidance with exact figures and direction changes\n"
        f"- Key metric trends (margins, FCF, volume, pricing) with magnitudes\n"
        f"- Management tone on demand outlook, cost control, pricing power\n"
        f"- Unexpected items: restructuring, impairments, M&A, regulatory events\n"
        f"- Analyst sentiment from Q&A: pushback areas, recurring concerns, tone shifts\n"
        f"Drop everything else: operator introductions, legal disclaimers, housekeeping remarks,\n"
        f"repetitive explanations, background context already known to the market,\n"
        f"and any statement that contains no incremental information.\n"
        f"Write in dense continuous prose. No headers, no bullet points, no filler.\n"
        f"Always anchor figures to their period (e.g. 'Q4 2022', 'FY2022') to avoid confusion.\n"
        f"TRANSCRIPT:\n{transcript}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint() -> tuple[pd.DataFrame, set]:
    """Charge le checkpoint existant et renvoie (df_done, done_ids)."""
    if Path(CHECKPOINT_PATH).exists():
        df_done = pd.read_parquet(CHECKPOINT_PATH)
        done_ids = set(df_done[ID_COL].tolist())
        log.info(f"Checkpoint trouvé : {len(done_ids):,} EC déjà traités")
        return df_done, done_ids
    log.info("Pas de checkpoint — départ de zéro")
    return pd.DataFrame(), set()


def load_errors() -> pd.DataFrame:
    """Charge le log d'erreurs existant."""
    if Path(ERROR_LOG_PATH).exists():
        return pd.read_parquet(ERROR_LOG_PATH)
    return pd.DataFrame(columns=[ID_COL, "error"])


def save_checkpoint(df_done: pd.DataFrame) -> None:
    df_done.to_parquet(CHECKPOINT_PATH, index=False)
    log.info(f"  ✓ Checkpoint sauvegardé ({len(df_done):,} lignes)")


def save_errors(df_errors: pd.DataFrame) -> None:
    df_errors.to_parquet(ERROR_LOG_PATH, index=False)


def detect_gpu_count() -> int:
    n = torch.cuda.device_count()
    log.info(f"GPUs détectés : {n}")
    return n


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize earnings calls with vLLM")
    p.add_argument("--input",  default=INPUT_PATH,  help="Input .parquet path")
    p.add_argument("--output", default=OUTPUT_PATH, help="Output .parquet path")
    p.add_argument("--test",   action="store_true",  help="Smoke test: process only --n_test rows")
    p.add_argument("--n_test", type=int, default=10, help="Rows to process in test mode")
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    input_path  = args.input
    output_path = args.output

    # ── 1. Chargement & filtre post-2018 ────────────────────────────────────
    log.info(f"Chargement de {input_path} ...")
    df = pd.read_parquet(input_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df_filtered = df[df[DATE_COL].dt.year > YEAR_FILTER].reset_index(drop=True)
    log.info(f"Lignes après {YEAR_FILTER} : {len(df_filtered):,} / {len(df):,} total")

    if args.test:
        df_filtered = df_filtered.head(args.n_test).reset_index(drop=True)
        log.info(f"[TEST MODE] Limité à {len(df_filtered)} lignes")

    # ── 2. Résumé depuis checkpoint ──────────────────────────────────────────
    df_done, done_ids = load_checkpoint()
    df_errors = load_errors()
    error_ids = set(df_errors[ID_COL].tolist()) if len(df_errors) > 0 else set()

    df_todo = df_filtered[
        ~df_filtered[ID_COL].isin(done_ids | error_ids)
    ].reset_index(drop=True)
    log.info(f"À traiter : {len(df_todo):,} EC  |  déjà faits : {len(done_ids):,}  |  erreurs : {len(error_ids):,}")

    if len(df_todo) == 0:
        log.info("Rien à traiter — tous les EC sont déjà dans le checkpoint.")
        _finalize(df_done)
        return

    # ── 3. Chargement du modèle ──────────────────────────────────────────────
    n_gpus = detect_gpu_count()
    tensor_parallel = min(n_gpus, 3)   # max 3 A100 disponibles

    log.info(f"Chargement du modèle {MODEL_NAME} (tensor_parallel={tensor_parallel}) ...")
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS_OUT,
        repetition_penalty=1.1,
    )

    vram_gb = torch.cuda.memory_allocated() / 1e9
    log.info(f"✓ Modèle chargé — VRAM allouée : {vram_gb:.1f} GB")

    # ── 4. Boucle de traitement par batches ──────────────────────────────────
    new_rows = []
    n_batches = (len(df_todo) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_errors = []

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end   = min(start + BATCH_SIZE, len(df_todo))
        batch = df_todo.iloc[start:end].copy()

        log.info(f"Batch {batch_idx+1}/{n_batches}  (rows {start}–{end-1})")

        # Construire les prompts — fallback individuel si erreur
        prompts = []
        valid_indices = []
        for i, row in batch.iterrows():
            try:
                p = build_prompt(str(row[TEXT_COL]))
                prompts.append(p)
                valid_indices.append(i)
            except Exception as e:
                log.warning(f"  ✗ Erreur build_prompt idx={i} id={row[ID_COL]}: {e}")
                batch_errors.append({ID_COL: row[ID_COL], "error": str(e)})

        if not prompts:
            log.warning(f"  Batch {batch_idx+1} entièrement en erreur, skip")
            continue

        # Inférence vLLM avec fallback global sur le batch
        try:
            outputs = llm.generate(prompts, sampling_params)
        except Exception as e:
            log.error(f"  ✗ vLLM crash sur batch {batch_idx+1}: {e}")
            log.error(traceback.format_exc())
            # Fallback : marquer toutes les lignes du batch comme erreurs
            for i in valid_indices:
                row = df_todo.iloc[i]
                batch_errors.append({ID_COL: row[ID_COL], "error": f"vllm_batch_error: {e}"})
            continue

        # Récupération des résultats
        for idx_in_batch, (orig_idx, output) in enumerate(zip(valid_indices, outputs)):
            row = df_todo.iloc[orig_idx]
            try:
                summary = output.outputs[0].text.strip()
                if not summary:
                    raise ValueError("Résumé vide généré")
                result = row.to_dict()
                result["ec_summary"] = summary
                new_rows.append(result)
            except Exception as e:
                log.warning(f"  ✗ Erreur récupération idx={orig_idx} id={row[ID_COL]}: {e}")
                batch_errors.append({ID_COL: row[ID_COL], "error": str(e)})

        # Checkpoint périodique
        if (batch_idx + 1) % CHECKPOINT_EVERY == 0 and new_rows:
            df_new = pd.DataFrame(new_rows)
            df_done = pd.concat([df_done, df_new], ignore_index=True)
            save_checkpoint(df_done)
            new_rows = []  # reset buffer

        # Sauvegarde erreurs au fil de l'eau
        if batch_errors:
            df_errors = pd.concat(
                [df_errors, pd.DataFrame(batch_errors)], ignore_index=True
            ).drop_duplicates(subset=[ID_COL])
            save_errors(df_errors)
            batch_errors = []

        elapsed = time.time() - t0
        done_so_far = len(df_done) + len(new_rows)
        rate = done_so_far / elapsed if elapsed > 0 else 0
        remaining = len(df_todo) - (batch_idx + 1) * BATCH_SIZE
        eta = remaining / (rate * BATCH_SIZE) / 60 if rate > 0 else 0
        log.info(f"  → {done_so_far:,} traités | {rate:.1f} EC/s | ETA ~{eta:.0f} min")

    # ── 5. Flush final ────────────────────────────────────────────────────────
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_done = pd.concat([df_done, df_new], ignore_index=True)
        save_checkpoint(df_done)

    if batch_errors:
        df_errors = pd.concat(
            [df_errors, pd.DataFrame(batch_errors)], ignore_index=True
        ).drop_duplicates(subset=[ID_COL])
        save_errors(df_errors)

    # ── 6. Finalisation ───────────────────────────────────────────────────────
    _finalize(df_done, output_path)
    elapsed_total = (time.time() - t0) / 60
    log.info(f"✓ Terminé en {elapsed_total:.1f} min")


def _finalize(df_done: pd.DataFrame, output_path: str) -> None:
    """Sauvegarde le fichier final et affiche les stats."""
    df_done.to_parquet(output_path, index=False)
    log.info(f"✓ Fichier final sauvegardé : {output_path}  ({len(df_done):,} lignes)")

    if "ec_summary" in df_done.columns:
        lengths = df_done["ec_summary"].dropna().str.len()
        log.info(f"  Résumés : mean={lengths.mean():.0f} chars | "
                 f"min={lengths.min()} | max={lengths.max()}")

    if Path(ERROR_LOG_PATH).exists():
        df_err = pd.read_parquet(ERROR_LOG_PATH)
        log.info(f"  Erreurs : {len(df_err):,} EC skippés → {ERROR_LOG_PATH}")


if __name__ == "__main__":
    main()