"""
plot_eval.py — Regenerate eval bar plots from existing prediction CSVs.
"""

import os
from collections import Counter
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results/eval_multi"

MODELS = [
    ("Direction reward", "predictions_adapter_0.csv"),
    ("PnL reward",       "predictions_adapter_1.csv"),
]
COLORS = ["#2CA02C", "#E91E8C", "#FFD700"]  # green, pink, yellow

BASELINE = {
    "model":       "Baseline",
    "accuracy":    0.38,
    "f1":          0.298,
    "abstention":  0.40,
    "n":           100,
}


def compute_metrics(df: pd.DataFrame, label: str) -> dict:
    n           = len(df)
    valid       = df.dropna(subset=["prediction"])
    abstentions = n - len(valid)
    correct     = (valid["prediction"] == valid["label"]).sum()
    accuracy    = correct / n
    f1 = f1_score(valid["label"], valid["prediction"].astype(int),
                  pos_label=1, average="binary") if len(valid) > 0 else 0.0

    label_dist = Counter(df["label"])
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Accuracy    : {accuracy:.1%}  ({correct}/{n})")
    print(f"  F1 (+1)     : {f1:.3f}  (on {len(valid)} non-abstained)")
    print(f"  Abstentions : {abstentions}/{n}  ({abstentions/n:.1%})")
    print(f"  Label dist  : +1={label_dist[1]}  -1={label_dist[-1]}")
    return {"model": label, "accuracy": accuracy, "f1": f1,
            "abstention": abstentions / n, "n": n}


def main():
    all_metrics = [BASELINE]
    for label, fname in MODELS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"[skip] {fname}")
            continue
        df = pd.read_csv(path)
        all_metrics.append(compute_metrics(df, label))

    if len(all_metrics) == 0:
        print("No data."); return

    names  = [m["model"] for m in all_metrics]
    colors = COLORS[:len(all_metrics)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    plots = [
        (axes[0], "accuracy",   "Accuracy"),
        (axes[1], "f1",         "F1 Score (+1)"),
        (axes[2], "abstention", "Abstention Rate"),
    ]

    for ax, key, title in plots:
        values = [m[key] for m in all_metrics]
        bars   = ax.bar(names, values, color=colors, edgecolor="white", width=0.65)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, linewidth=1.5)
        ax.set_title(title, fontsize=28, fontweight="bold", pad=16)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15, labelsize=24)
        ax.tick_params(axis="y", labelsize=18)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.1%}", ha="center", fontsize=26, fontweight="bold")

    fig.suptitle(f"Majority Vote @ 4  —  {all_metrics[0]['n']} samples", fontsize=28, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(RESULTS_DIR, "eval_comparison.png")
    fig.savefig(out, dpi=200)
    plt.close()
    print(f"\nPlot saved → {out}")


if __name__ == "__main__":
    main()
