#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh
# One-time environment setup on the MLP cluster.
# Run this ONCE from the head node (hastings) before submitting any jobs.
#
# Usage:  bash setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

echo "============================================================"
echo "  LLM Training Environment Setup"
echo "  $(date)"
echo "============================================================"

# ── 1. Install Miniconda (if not already installed) ───────────────────────────
if [ ! -d "$HOME/miniconda3" ]; then
    echo ""
    echo "[1/6] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm /tmp/miniconda.sh
    echo "  Miniconda installed."
else
    echo "[1/6] Miniconda already installed — skipping."
fi

# Initialise conda for this script
source $HOME/miniconda3/etc/profile.d/conda.sh

# ── 2. Create conda environment ───────────────────────────────────────────────
echo ""
echo "[2/6] Creating conda environment 'llm_training'..."
conda create -n llm_training python=3.11 -y 2>/dev/null || true
conda activate llm_training
echo "  Python: $(python --version)"

# ── 3. Install PyTorch (CUDA 12.8 — matches saxa driver) ─────────────────────
echo ""
echo "[3/6] Installing PyTorch with CUDA 12.8 support..."
# PyTorch nightly with CUDA 12.4 support is the most recent stable option
# The H200 MIG slice runs CUDA 12.8/13.0 but PyTorch CUDA 12.4 binaries are compatible
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    -q

python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  BF16 support: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else \"N/A\"}')
"

# ── 4. Install other dependencies ─────────────────────────────────────────────
echo ""
echo "[4/6] Installing dependencies..."
pip install -q \
    sentencepiece \
    numpy \
    matplotlib \
    huggingface_hub \
    tqdm

echo "  Installed: sentencepiece, numpy, matplotlib, huggingface_hub, tqdm"

# ── 5. Download data ──────────────────────────────────────────────────────────
echo ""
echo "[5/6] Downloading FineWeb dataset (SP1024 tokenizer, 10B subset)..."
echo "  This downloads ~20GB of tokenized training data."
echo "  Using the parameter-golf cached download script."
echo ""

cd $HOME/parameter-golf

# The parameter-golf repo provides a download script
# It fetches the pre-tokenized FineWeb shards from HuggingFace
python data/cached_challenge_fineweb.py --variant sp1024

echo ""
echo "  Dataset downloaded to: ./data/datasets/fineweb10B_sp1024/"
echo "  Tokenizer at: ./data/tokenizers/fineweb_1024_bpe.model"

# Verify the download
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "  Train shards: $TRAIN_COUNT"
echo "  Val shards:   $VAL_COUNT"

if [ "$TRAIN_COUNT" -eq 0 ]; then
    echo ""
    echo "  WARNING: No training shards found!"
    echo "  Check that cached_challenge_fineweb.py ran successfully."
    echo "  You may need to set: MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf"
fi

# ── 6. Copy training scripts ──────────────────────────────────────────────────
echo ""
echo "[6/6] Placing training scripts in $HOME/parameter-golf/..."
# (scripts are already there if you cloned the repo and added them)
# Just verify they exist
for f in train_llm_scratch.py plot_training.py train_saxa.slurm; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f  (not found — copy it here)"
    fi
done

mkdir -p logs checkpoints plots

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To submit your training job:"
echo "    cd $HOME/parameter-golf"
echo "    sbatch train_saxa.slurm"
echo ""
echo "  To monitor the job:"
echo "    squeue -u $USER"
echo "    tail -f logs/train_<JOBID>.log"
echo ""
echo "  To run interactively (for testing):"
echo "    srun -p Teaching -w saxa --gres gpu:1 --mem=32G -t 00:45:00 --pty bash"
echo "    conda activate llm_training"
echo "    cd $HOME/parameter-golf"
echo "    python train_llm_scratch.py --max_train_seconds 1800"
echo "============================================================"
