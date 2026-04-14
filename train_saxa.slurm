#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# train_saxa.slurm
# Slurm batch script to train LLM from scratch on saxa (H200 MIG slice)
#
# Submit:  sbatch train_saxa.slurm
# Monitor: squeue -u $USER
#          tail -f logs/train_$SLURM_JOB_ID.log
# Cancel:  scancel <JOB_ID>
# ─────────────────────────────────────────────────────────────────────────────

# ── Slurm resource requests ──────────────────────────────────────────────────
#SBATCH --job-name=llm_from_scratch          # job name shown in squeue
#SBATCH --partition=Teaching                 # partition to submit to
#SBATCH --nodelist=saxa                      # specifically request the H200 node
#SBATCH --gres=gpu:1                         # request 1 GPU (will get 1 MIG slice)
#SBATCH --cpus-per-task=4                    # CPU cores for data loading
#SBATCH --mem=32G                            # system RAM
#SBATCH --time=00:45:00                      # wall clock limit: 45 min
#SBATCH --output=logs/train_%j.log           # stdout → logs/train_<jobid>.log
#SBATCH --error=logs/train_%j.log            # stderr → same file
#SBATCH --mail-type=END,FAIL                 # email when done or failed (optional)
# #SBATCH --mail-user=s2892267@inf.ed.ac.uk  # uncomment to enable email

# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo " Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo " Start: $(date)"
echo " Partition: $SLURM_JOB_PARTITION"
echo "============================================================"

# ── Environment setup ────────────────────────────────────────────────────────
# Activate the conda environment (created by setup_env.sh)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_training

# Print environment info for the log
echo ""
echo "--- Environment ---"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "-------------------"
echo ""

# ── Change to working directory ───────────────────────────────────────────────
# This is where you cloned the parameter-golf repo
cd $HOME/parameter-golf
mkdir -p logs checkpoints plots

# ── Run training ─────────────────────────────────────────────────────────────
# 1800 seconds = 30 minutes of training
# Log file will capture everything printed by the Python script
python train_llm_scratch.py \
    --data_dir        ./data \
    --log_file        logs/training_log_${SLURM_JOB_ID}.jsonl \
    --checkpoint_dir  checkpoints \
    --max_train_seconds 1800 \
    --seed            42 \
    --num_layers      8 \
    --model_dim       384 \
    --num_heads       6 \
    --num_kv_heads    3 \
    --batch_tokens    32768 \
    --peak_lr         3e-3 \
    --val_every_steps 100

EXIT_CODE=$?
echo ""
echo "============================================================"
echo " Training finished with exit code: $EXIT_CODE"
echo " End: $(date)"
echo "============================================================"

# ── Generate plots immediately after training ─────────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Generating training plots..."
    python plot_training.py \
        logs/training_log_${SLURM_JOB_ID}.jsonl \
        --out plots/run_${SLURM_JOB_ID}
    echo "Plots saved to plots/run_${SLURM_JOB_ID}/"
fi

echo ""
echo "Log file: logs/train_${SLURM_JOB_ID}.log"
echo "JSONL metrics: logs/training_log_${SLURM_JOB_ID}.jsonl"
echo "Checkpoint: checkpoints/final_model.pt"
exit $EXIT_CODE
