#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_env.sh
# One-time environment setup on the MLP cluster (hastings head node).
#
# WHAT THIS FIXES vs the previous version:
#   - Handles the case where conda already exists at /opt/conda (system conda)
#     AND at ~/miniconda3 (freshly installed) — uses whichever is available
#   - 'conda activate' in a bash script requires sourcing the right profile.d
#     script first; this is done explicitly before every activate call
#   - 'conda create' is now idempotent: checks if env exists before creating
#   - Removed 'set -e' (exit on any error) + silenced errors (2>/dev/null)
#     and replaced with explicit error checking so failures are visible
#   - Uses 'conda run -n ENV cmd' as fallback when activate doesn't work
#     in non-interactive shells
#
# Usage:  bash setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

ENV_NAME="llm_training"
PYTHON_VERSION="3.11"

echo "============================================================"
echo "  LLM Training Environment Setup"
echo "  $(date)"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Find or install conda
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Finding conda..."

# Helper: source the conda profile.d for a given conda root
# This is what makes 'conda activate' work in a bash script
_init_conda() {
    local conda_root="$1"
    local profile="$conda_root/etc/profile.d/conda.sh"
    if [ -f "$profile" ]; then
        source "$profile"
        echo "  Sourced: $profile"
        return 0
    fi
    return 1
}

# Priority order for finding conda:
#   1. Already in PATH (e.g. user ran 'conda init' previously or /opt/conda is set up)
#   2. ~/miniconda3  (user-installed miniconda)
#   3. ~/anaconda3   (user-installed anaconda)
#   4. /opt/conda    (system conda on this cluster)

CONDA_FOUND=0

if command -v conda &> /dev/null; then
    # conda is already in PATH — figure out its root and source its profile
    CONDA_ROOT=$(conda info --base 2>/dev/null)
    echo "  conda already in PATH (root: $CONDA_ROOT)"
    _init_conda "$CONDA_ROOT" || true
    CONDA_FOUND=1

elif [ -d "$HOME/miniconda3" ]; then
    echo "  Found ~/miniconda3"
    _init_conda "$HOME/miniconda3"
    CONDA_FOUND=1

elif [ -d "$HOME/anaconda3" ]; then
    echo "  Found ~/anaconda3"
    _init_conda "$HOME/anaconda3"
    CONDA_FOUND=1

elif [ -d "/opt/conda" ]; then
    echo "  Found /opt/conda (system conda)"
    _init_conda "/opt/conda"
    CONDA_FOUND=1
fi

# If still not found, install Miniconda
if [ "$CONDA_FOUND" -eq 0 ]; then
    echo "  No conda found — installing Miniconda to ~/miniconda3..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O /tmp/miniconda_install.sh
    bash /tmp/miniconda_install.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda_install.sh
    _init_conda "$HOME/miniconda3"
    echo "  Miniconda installed."
fi

# Final check
if ! command -v conda &> /dev/null; then
    echo ""
    echo "ERROR: conda still not available after setup. Something went wrong."
    echo "Try running: source ~/miniconda3/etc/profile.d/conda.sh"
    echo "Then run this script again."
    exit 1
fi

echo "  Using conda: $(conda --version)  at $(which conda)"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Create conda environment (idempotent)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Setting up conda environment '$ENV_NAME'..."

# Check if env already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists — skipping creation."
else
    echo "  Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: conda create failed."
        echo "Check: conda env list"
        echo "Try manually: conda create -n $ENV_NAME python=$PYTHON_VERSION -y"
        exit 1
    fi
    echo "  Environment created."
fi

# Activate the environment
# Note: 'conda activate' only works after sourcing profile.d, which we did above.
# As an extra safety net we also provide the 'conda run' fallback.
conda activate "$ENV_NAME"
if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: 'conda activate $ENV_NAME' failed."
    echo "This sometimes happens in non-interactive shells."
    echo "Falling back to 'conda run' for all subsequent steps."
    echo "Everything will still install correctly."
    USE_CONDA_RUN=1
else
    USE_CONDA_RUN=0
    echo "  Activated: $ENV_NAME"
    echo "  Python: $(python --version)"
fi

# Wrapper: run a command either in the activated env or via conda run
_conda_run() {
    if [ "$USE_CONDA_RUN" -eq 1 ]; then
        conda run -n "$ENV_NAME" "$@"
    else
        "$@"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Install PyTorch + dependencies
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch (CUDA 12.4 build — compatible with saxa CUDA 12.8/13.0)..."
echo "  This may take 5-10 minutes on first install..."

_conda_run pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch installation failed."
    echo "Check your internet connection and try again."
    exit 1
fi

echo ""
echo "  Verifying PyTorch install..."
_conda_run python -c "
import torch
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA compiled:  {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()} (False on head node — expected)')
"

echo ""
echo "[3/5] Installing other dependencies..."
_conda_run pip install sentencepiece numpy matplotlib huggingface_hub tqdm --quiet

echo "  Installed: sentencepiece, numpy, matplotlib, huggingface_hub, tqdm"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Configure conda init for future sessions
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Configuring conda for future shell sessions..."

# Run 'conda init bash' so that 'conda activate' works in future interactive
# shells (like when you ssh in and type commands manually).
# This adds a block to ~/.bashrc automatically.
conda init bash

echo "  conda init bash done."
echo "  NOTE: This modified ~/.bashrc. It takes effect in your NEXT login."
echo "  For this session, the environment is already active via sourcing."
echo ""
echo "  After this script finishes, you can verify with:"
echo "    source ~/.bashrc"
echo "    conda activate $ENV_NAME"
echo "    python --version  # should show Python 3.11.x"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Download FineWeb dataset
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[5/5] Downloading FineWeb dataset (SP1024, 10B token subset)..."
echo "  This is ~20 GB. It uses the parameter-golf download script."
echo "  Tokens cached on HuggingFace — downloads only once."
echo ""

# Check we're in the right directory
if [ ! -f "data/cached_challenge_fineweb.py" ]; then
    echo "  ERROR: data/cached_challenge_fineweb.py not found."
    echo "  Make sure you're running this script from inside the parameter-golf repo:"
    echo "    cd ~/parameter-golf"
    echo "    bash setup_env.sh"
    exit 1
fi

TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
if [ "$TRAIN_COUNT" -gt 0 ]; then
    echo "  Dataset already downloaded ($TRAIN_COUNT train shards found) — skipping."
else
    echo "  Downloading..."
    _conda_run python data/cached_challenge_fineweb.py --variant sp1024

    TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
    VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)

    if [ "$TRAIN_COUNT" -eq 0 ]; then
        echo ""
        echo "  WARNING: download may have failed (0 train shards found)."
        echo "  Try manually:"
        echo "    conda activate $ENV_NAME"
        echo "    MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \\"
        echo "        python data/cached_challenge_fineweb.py --variant sp1024"
    else
        echo "  Train shards: $TRAIN_COUNT"
        echo "  Val shards:   $VAL_COUNT"
        echo "  Dataset ready."
    fi
fi

# Verify training scripts are present
echo ""
echo "  Verifying training scripts..."
ALL_GOOD=1
for f in train_llm_scratch.py plot_training.py train_saxa.slurm inference.py; do
    if [ -f "$f" ]; then
        echo "    ✓  $f"
    else
        echo "    ✗  $f  (not found — make sure it was committed to the repo)"
        ALL_GOOD=0
    fi
done

mkdir -p logs checkpoints plots

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
if [ "$ALL_GOOD" -eq 1 ]; then
    echo "  Setup complete! Everything looks good."
else
    echo "  Setup done (with warnings — see above)."
fi
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Start a new shell (or run: source ~/.bashrc)"
echo "     Then verify with: conda activate $ENV_NAME && python --version"
echo ""
echo "  2. Submit the 30-minute training job:"
echo "       cd ~/parameter-golf"
echo "       sbatch train_saxa.slurm"
echo ""
echo "  3. Monitor the job:"
echo "       squeue -u \$USER"
echo "       tail -f logs/train_<JOBID>.log"
echo ""
echo "  4. Or test interactively first (smoke test):"
echo "       srun -p Teaching -w saxa --gres gpu:1 --mem=32G -t 00:20:00 --pty bash"
echo "       source ~/.bashrc && conda activate $ENV_NAME"
echo "       cd ~/parameter-golf"
echo "       python train_llm_scratch.py --max_train_seconds 120 --val_every_steps 30"
echo "============================================================"