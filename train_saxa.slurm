#!/bin/bash
#SBATCH --job-name=llm_scratch_saxa
#SBATCH --partition=Teaching
#SBATCH --nodelist=saxa
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

echo "============================================================"
echo " Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo " Start: $(date)"
echo "============================================================"

# ── Activate conda (works regardless of which conda is installed) ─────────────
for CONDA_INIT in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_INIT" ]; then
        source "$CONDA_INIT"
        break
    fi
done

conda activate llm_training
if [ $? -ne 0 ]; then
    echo "ERROR: could not activate llm_training env"
    exit 1
fi

echo ""
echo "--- Environment ---"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA:    {torch.version.cuda}')"
python -c "import torch; print(f'GPU:     {torch.cuda.get_device_name(0)}')"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "-------------------"
echo ""

cd $HOME/parameter-golf
mkdir -p logs checkpoints plots

# ── Training ──────────────────────────────────────────────────────────────────
# saxa H200 MIG (1g.16gb): 16GB VRAM, Hopper arch, CUDA 13.0
# batch_tokens=32768 (32 seqs × 1024) fits in 16GB
# Hopper supports torch.compile + BF16 natively
python -u train_llm_scratch.py \
    --data_dir           ./data \
    --log_file           logs/training_log_${SLURM_JOB_ID}.jsonl \
    --checkpoint_dir     checkpoints \
    --max_train_seconds  1800 \
    --seed               42 \
    --num_layers         8 \
    --model_dim          384 \
    --num_heads          6 \
    --num_kv_heads       3 \
    --batch_tokens       32768 \
    --peak_lr            3e-3 \
    --val_every_steps    500

EXIT_CODE=$?
echo ""
echo "============================================================"
echo " Training finished — exit code: $EXIT_CODE"
echo " End: $(date)"
echo "============================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Generating plots..."
    python plot_training.py \
        logs/training_log_${SLURM_JOB_ID}.jsonl \
        --out plots/run_${SLURM_JOB_ID}
fi

echo ""
echo "Slurm log:     logs/train_${SLURM_JOB_ID}.log"
echo "JSONL metrics: logs/training_log_${SLURM_JOB_ID}.jsonl"
echo "Checkpoint:    checkpoints/final_model.pt"
exit $EXIT_CODE
