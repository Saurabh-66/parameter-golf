"""
plot_training.py
================
Read the training_log.jsonl produced by train_llm_scratch.py
and generate publication-quality plots of all key LLM training metrics.

X-AXIS OPTIONS (--xaxis flag):
  step      : gradient update step number  [default]
  time      : wall-clock elapsed minutes
  tokens    : billions of tokens seen

NOTE ON EPOCHS:
  This training never completes a full epoch. With 10B tokens in the dataset
  and ~393M tokens seen in 30 minutes, we cover only ~4% of the data.
  "Epoch" is therefore not a useful axis here. Step, time, and tokens are
  the standard axes used in modern LLM training papers (Chinchilla, LLaMA, etc.)

Usage:
    python plot_training.py training_log.jsonl
    python plot_training.py training_log.jsonl --out plots/ --xaxis time
    python plot_training.py training_log.jsonl --xaxis tokens
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on headless cluster)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Load log
# ─────────────────────────────────────────────────────────────────────────────

def load_log(path: str) -> dict:
    """Parse the JSONL log into separate lists per event type."""
    records = {"train": [], "val": [], "config": None, "final": None, "inference": []}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("type", "")
            if t == "train":
                records["train"].append(obj)
            elif t == "val":
                records["val"].append(obj)
            elif t == "config":
                records["config"] = obj
            elif t == "final":
                records["final"] = obj
            elif t == "inference":
                records["inference"].append(obj)
    return records


def to_arrays(records: list, *keys):
    """Extract multiple fields from a list of dicts as numpy arrays."""
    return tuple(np.array([r[k] for r in records if k in r]) for k in keys)


def get_x(records: list, xaxis: str, batch_tokens: int = 32768) -> tuple:
    """
    Return (x_values, x_label) for the chosen x-axis type.

    xaxis options:
      "step"   : gradient step number (default)
      "time"   : wall-clock minutes elapsed
      "tokens" : billions of tokens processed

    Why not epoch?
      With 10B tokens in FineWeb and ~393M tokens seen in 30 min, we only
      cover ~4% of one epoch. Epoch is therefore not a useful axis here.
      Modern LLM papers (Chinchilla, LLaMA, Gemma) always use tokens or
      wall-clock time as the x-axis, never epochs.
    """
    if xaxis == "time":
        vals = np.array([r.get("elapsed_min", i * 0.15) for i, r in enumerate(records)])
        return vals, "Elapsed time (minutes)"
    elif xaxis == "tokens":
        steps = np.array([r.get("step", i) for i, r in enumerate(records)])
        return steps * batch_tokens / 1e9, "Tokens seen (billions)"
    else:  # default: step
        return np.array([r.get("step", i) for i, r in enumerate(records)]), "Gradient step"


# ─────────────────────────────────────────────────────────────────────────────
# Smoothing helper
# ─────────────────────────────────────────────────────────────────────────────

def smooth(y: np.ndarray, window: int = 20) -> np.ndarray:
    """Exponential moving average for visual smoothing of noisy curves."""
    if len(y) < 2:
        return y
    alpha  = 2.0 / (window + 1)
    out    = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i - 1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "train_loss":  "#2196F3",
    "val_loss":    "#FF5722",
    "val_bpb":     "#4CAF50",
    "tok_per_sec": "#9C27B0",
    "mfu":         "#FF9800",
    "grad_norm":   "#F44336",
    "lr":          "#607D8B",
    "sota":        "#E91E63",
    "baseline":    "#795548",
    "grid":        "#E0E0E0",
    "bg":          "#FAFAFA",
}

# Reference lines for val_bpb
SOTA_BPB     = 1.081
BASELINE_BPB = 1.224


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=9, labelpad=4)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#BDBDBD")


def make_plots(records: dict, out_dir: str, xaxis: str = "step"):
    train  = records["train"]
    val    = records["val"]
    cfg    = records["config"] or {}
    final  = records["final"]

    if not train:
        print("No training records found in log.")
        return

    os.makedirs(out_dir, exist_ok=True)
    batch_tokens = cfg.get("batch_tokens", 32768)

    # ── X-axis values ────────────────────────────────────────────────────────
    # Each record already contains elapsed_min (logged by the training script).
    # val records log: step, val_loss, val_bpb, elapsed_min, avg_step_ms
    # train records log: step, train_loss, tok_per_sec, grad_norm, lr, mfu, elapsed_min
    t_x, xlabel = get_x(train, xaxis, batch_tokens)
    v_x, _      = get_x(val,   xaxis, batch_tokens)

    # ── Metric arrays ─────────────────────────────────────────────────────────
    _, t_loss, t_toks, t_mfu, t_gnorm, t_lr = to_arrays(
        train, "step", "train_loss", "tok_per_sec", "mfu", "grad_norm", "lr"
    )
    _, v_loss, v_bpb = to_arrays(val, "step", "val_loss", "val_bpb")

    # ── Figure 1: 6-panel dashboard ───────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11), facecolor="white")
    fig.patch.set_facecolor(PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                            left=0.07, right=0.97, top=0.91, bottom=0.08)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    # Panel 0 — Training loss
    ax = axes[0]
    ax.plot(t_x, t_loss, color=PALETTE["train_loss"], alpha=0.25, linewidth=0.8,
            label="train loss (raw)")
    if len(t_loss) > 5:
        ax.plot(t_x, smooth(t_loss, 30), color=PALETTE["train_loss"],
                linewidth=2.0, label="train loss (EMA-30)")
    if len(v_loss):
        ax.plot(v_x, v_loss, color=PALETTE["val_loss"], linewidth=2.0,
                marker="o", markersize=4, label="val loss")
    style_ax(ax, "Loss (nats)", xlabel, "Cross-entropy [nats]")
    ax.legend(fontsize=8, loc="upper right")

    # Panel 1 — val_bpb
    ax = axes[1]
    if len(v_bpb):
        ax.plot(v_x, v_bpb, color=PALETTE["val_bpb"], linewidth=2.5,
                marker="o", markersize=5, label="val_bpb (this run)", zorder=5)
        ax.scatter([v_x[-1]], [v_bpb[-1]], color=PALETTE["val_bpb"],
                   s=100, zorder=10, edgecolors="white", linewidths=1.5)
        ax.annotate(f"{v_bpb[-1]:.4f}", (v_x[-1], v_bpb[-1]),
                    textcoords="offset points", xytext=(8, 4), fontsize=8,
                    color=PALETTE["val_bpb"], fontweight="bold")
    ax.axhline(SOTA_BPB,     color=PALETTE["sota"],    linewidth=1.2,
               linestyle="--", label=f"SOTA ({SOTA_BPB})")
    ax.axhline(BASELINE_BPB, color=PALETTE["baseline"], linewidth=1.2,
               linestyle=":",  label=f"Baseline ({BASELINE_BPB})")
    style_ax(ax, "val_bpb — Key Compression Metric", xlabel, "Bits per byte [lower = better]")
    ax.legend(fontsize=8, loc="upper right")
    ax.annotate("← Better", xy=(0.02, 0.10), xycoords="axes fraction",
                fontsize=8, color="green", fontweight="bold")

    # Panel 2 — Throughput
    ax = axes[2]
    if len(t_toks):
        ax.plot(t_x, t_toks / 1000, color=PALETTE["tok_per_sec"],
                linewidth=1.0, alpha=0.5, label="tok/s (÷1000)")
        ax.plot(t_x, smooth(t_toks, 20) / 1000, color=PALETTE["tok_per_sec"],
                linewidth=2.5, label="smoothed")
    style_ax(ax, "Training Throughput", xlabel, "Tokens/sec (×10³)")
    ax.legend(fontsize=8)

    # Panel 3 — MFU
    ax = axes[3]
    if len(t_mfu):
        ax.plot(t_x, t_mfu * 100, color=PALETTE["mfu"], linewidth=1.0, alpha=0.4)
        ax.plot(t_x, smooth(t_mfu, 20) * 100, color=PALETTE["mfu"],
                linewidth=2.5, label="MFU (%)")
        ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="50% (good)")
        ax.axhline(30, color="gray", linewidth=0.8, linestyle=":",  label="30% (decent)")
    style_ax(ax, "Model FLOP Utilization (MFU)", xlabel, "MFU (%)")
    ax.legend(fontsize=8)

    # Panel 4 — Gradient norm
    ax = axes[4]
    if len(t_gnorm):
        ax.plot(t_x, t_gnorm, color=PALETTE["grad_norm"],
                linewidth=0.8, alpha=0.35, label="raw")
        ax.plot(t_x, smooth(t_gnorm, 30), color=PALETTE["grad_norm"],
                linewidth=2.0, label="smoothed")
        ax.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", label="clip=1.0")
    style_ax(ax, "Gradient Norm", xlabel, "L2 norm")
    ax.legend(fontsize=8)

    # Panel 5 — LR schedule
    ax = axes[5]
    if len(t_lr):
        ax.plot(t_x, t_lr, color=PALETTE["lr"], linewidth=2.5, label="learning rate")
        if xaxis == "step" and len(t_x):
            wu       = cfg.get("warmup_steps", 100)
            wd_frac  = cfg.get("warmdown_frac", 0.30)
            total    = t_x[-1]
            wd_start = total - int(total * wd_frac)
            ymax     = t_lr.max() * 1.1
            ax.axvspan(0,        wu,       alpha=0.12, color="green",  label="warmup")
            ax.axvspan(wd_start, total,    alpha=0.12, color="orange", label="warmdown")
    style_ax(ax, "Learning Rate Schedule", xlabel, "Learning rate")
    ax.legend(fontsize=8)

    # Title
    n_layers    = cfg.get("num_layers", "?")
    n_dim       = cfg.get("model_dim",  "?")
    total_m     = final.get("total_tokens_M", "?") if final else "?"
    final_bpb   = final.get("val_bpb", "?")        if final else "?"
    xaxis_label = {"step": "step", "time": "wall-clock time", "tokens": "tokens seen"}[xaxis]

    fig.suptitle(
        f"LLM Training from Scratch  |  {n_layers}L × {n_dim}d  |  "
        f"Final val_bpb = {final_bpb:.4f}  |  {total_m:.0f}M tokens  |  x-axis: {xaxis_label}",
        fontsize=12, fontweight="bold", y=0.97,
    )

    path1 = os.path.join(out_dir, f"training_dashboard_{xaxis}.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── Figure 2: Clean val_bpb plot ──────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    ax2.set_facecolor(PALETTE["bg"])

    if len(v_bpb):
        ax2.fill_between(v_x, v_bpb, alpha=0.15, color=PALETTE["val_bpb"])
        ax2.plot(v_x, v_bpb, color=PALETTE["val_bpb"], linewidth=2.5,
                 marker="o", markersize=5,
                 label=f"This run  (final = {v_bpb[-1]:.4f} bpb)", zorder=5)
        ax2.scatter([v_x[-1]], [v_bpb[-1]], color=PALETTE["val_bpb"],
                    s=120, zorder=10, edgecolors="white", linewidths=1.5)

    ax2.axhline(SOTA_BPB,     color=PALETTE["sota"],    linewidth=1.5, linestyle="--",
                label=f"Param-golf SOTA: {SOTA_BPB} bpb")
    ax2.axhline(BASELINE_BPB, color=PALETTE["baseline"], linewidth=1.5, linestyle=":",
                label=f"Naive baseline: {BASELINE_BPB} bpb")
    ax2.axhline(1.0,          color="#9E9E9E",            linewidth=1.0, linestyle="-.",
                label="Shannon limit ≈ 1.0 bpb (English)")

    if len(v_bpb):
        gap      = v_bpb[-1] - SOTA_BPB
        mid_x    = v_x[len(v_x)//2] if len(v_x) else 0
        ax2.annotate(
            f"Gap to SOTA:\n{gap:.3f} bpb",
            xy=(mid_x, (v_bpb[-1] + SOTA_BPB) / 2),
            fontsize=9, ha="center", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.8),
        )

    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel("val_bpb  [bits per byte — lower is better]", fontsize=11)
    ax2.set_title(
        "Validation Compression Quality (val_bpb)\n"
        "val_bpb = (cross-entropy / ln 2) × (tokens / bytes)   [tokenizer-agnostic]",
        fontsize=11, fontweight="bold",
    )
    ax2.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax2.legend(fontsize=9, loc="upper right")
    for sp in ax2.spines.values():
        sp.set_linewidth(0.5)

    path2 = os.path.join(out_dir, f"val_bpb_{xaxis}.png")
    fig2.tight_layout()
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Figure 3: Loss curves ─────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4.5), facecolor=PALETTE["bg"])
    fig3.patch.set_facecolor(PALETTE["bg"])

    for ax3 in axes3:
        ax3.set_facecolor(PALETTE["bg"])
        ax3.grid(True, color=PALETTE["grid"], linewidth=0.8)
        for sp in ax3.spines.values():
            sp.set_linewidth(0.5)

    ax3 = axes3[0]
    ax3.plot(t_x, t_loss, color=PALETTE["train_loss"], alpha=0.2, linewidth=0.7)
    if len(t_loss) > 5:
        ax3.plot(t_x, smooth(t_loss, 30), color=PALETTE["train_loss"],
                 linewidth=2.5, label="Train loss (EMA-30)")
    ax3.set_title("Training Loss", fontsize=11, fontweight="bold")
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Cross-entropy [nats]")
    ax3.legend(fontsize=9)

    ax3b  = axes3[1]
    ax3b2 = ax3b.twinx()
    if len(v_loss):
        ax3b.plot(v_x,  v_loss, color=PALETTE["val_loss"],  linewidth=2.5,
                  marker="s", markersize=5, label="val_loss [nats]")
        ax3b2.plot(v_x, v_bpb,  color=PALETTE["val_bpb"],   linewidth=2.5,
                   marker="^", markersize=5, linestyle="--", label="val_bpb")
    ax3b.set_title("Validation Metrics (step-wise)", fontsize=11, fontweight="bold")
    ax3b.set_xlabel(xlabel)
    ax3b.set_ylabel("val_loss [nats]",  color=PALETTE["val_loss"])
    ax3b2.set_ylabel("val_bpb",          color=PALETTE["val_bpb"])
    ax3b.tick_params(axis="y",  labelcolor=PALETTE["val_loss"])
    ax3b2.tick_params(axis="y", labelcolor=PALETTE["val_bpb"])
    lines1, labels1 = ax3b.get_legend_handles_labels()
    lines2, labels2 = ax3b2.get_legend_handles_labels()
    ax3b.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    fig3.suptitle("Training and Validation Loss Curves", fontsize=12, fontweight="bold")
    fig3.tight_layout()
    path3 = os.path.join(out_dir, f"loss_curves_{xaxis}.png")
    fig3.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Saved: {path3}")

    # ── Figure 4: Optimizer diagnostics ──────────────────────────────────────
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4), facecolor=PALETTE["bg"])
    fig4.patch.set_facecolor(PALETTE["bg"])

    for ax4 in axes4:
        ax4.set_facecolor(PALETTE["bg"])
        ax4.grid(True, color=PALETTE["grid"], linewidth=0.8)

    ax4 = axes4[0]
    if len(t_lr):
        ax4.plot(t_x, t_lr, color=PALETTE["lr"], linewidth=2.5)
        if xaxis == "step" and len(t_x):
            wu       = cfg.get("warmup_steps", 100)
            wd_frac  = cfg.get("warmdown_frac", 0.30)
            total    = t_x[-1]
            wd_start = total - int(total * wd_frac)
            ymax     = t_lr.max()
            ax4.axvspan(0,        wu,    alpha=0.15, color="green")
            ax4.axvspan(wd_start, total, alpha=0.15, color="orange")
            ax4.text(wu / 2,                     ymax * 0.5, "Warmup",   fontsize=8, ha="center", color="green")
            ax4.text((wu + wd_start) / 2,        ymax * 0.5, "Constant", fontsize=8, ha="center")
            ax4.text((wd_start + total) / 2,     ymax * 0.5, "Warmdown", fontsize=8, ha="center", color="darkorange")
    ax4.set_title("Learning Rate Schedule", fontsize=11, fontweight="bold")
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel("Learning rate")

    ax4 = axes4[1]
    if len(t_gnorm):
        ax4.plot(t_x, t_gnorm, alpha=0.25, color=PALETTE["grad_norm"], linewidth=0.8)
        ax4.plot(t_x, smooth(t_gnorm, 30), color=PALETTE["grad_norm"],
                 linewidth=2.5, label="Grad norm (smoothed)")
        ax4.axhline(1.0, color="gray", linewidth=1.2, linestyle="--", label="Clip threshold")
        spikes = t_x[t_gnorm > 1.5]
        if len(spikes):
            ax4.scatter(spikes, t_gnorm[t_gnorm > 1.5], color="red", s=20,
                        zorder=5, alpha=0.7, label=f"Spikes >1.5: {len(spikes)}")
    ax4.set_title("Gradient Norm  [training health]", fontsize=11, fontweight="bold")
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel("L2 grad norm")
    ax4.legend(fontsize=8)

    fig4.suptitle("Optimizer Diagnostics", fontsize=12, fontweight="bold")
    fig4.tight_layout()
    path4 = os.path.join(out_dir, f"optimizer_diagnostics_{xaxis}.png")
    fig4.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"  Saved: {path4}")

    print(f"\n  All plots saved to: {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot LLM training metrics from JSONL log")
    parser.add_argument("log_file",          help="Path to training_log.jsonl")
    parser.add_argument("--out",   default="plots",  help="Output directory for plots")
    parser.add_argument("--xaxis", default="step",
                        choices=["step", "time", "tokens"],
                        help="X-axis: 'step' (default), 'time' (minutes), 'tokens' (billions)")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: log file not found: {args.log_file}")
        sys.exit(1)

    print(f"Loading log: {args.log_file}")
    records = load_log(args.log_file)
    print(f"  Train events: {len(records['train'])}")
    print(f"  Val events:   {len(records['val'])}")
    print(f"  X-axis mode:  {args.xaxis}")
    print(f"\nGenerating plots...")
    make_plots(records, args.out, xaxis=args.xaxis)
