#!/bin/bash
#SBATCH --job-name=sae_stage1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu          # adjust to your cluster
#SBATCH --gpus=h200:1             # 1 GPU; change if you need more
#SBATCH --cpus-per-task=8       # data-loading threads etc.
#SBATCH --mem=120G
#SBATCH --time=24:00:00        # 2 days; raise/lower as needed
#SBATCH --output=logs/%x-%j.out  # logs/sae_stage1-<jobid>.out
#SBATCH --error=logs/%x-%j.err   # logs/sae_stage1-<jobid>.err

# ----------------- User config  ---------------- #
SHARD_DIR="/gpfs/radev/home/am3833/scratch/volumes_96"
MASK_DIR="/gpfs/radev/home/am3833/scratch/volumes_96/masks"
CKPT="checkpoints/best_model_base_patch_conv_v4_85_stable_eps1e-5.pt"
SCRIPT=train_sae_v3.py             # the rewritten script filename
# ------------------------------------------------#

# Create a unique run name
RUN="sae_v3"

# Best practice: fail on error, echo commands
set -euo pipefail
echo "Starting SAE run: $RUN"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# (Optional) load modules / conda env
module --force purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.6.3

source venv/bin/activate

# Recommended for reproducibility
export PYTHONFAULTHANDLER=1          # Python will dump a traceback on segfault
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=1            # reduce glibc fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=1

python3 -u "$SCRIPT" \
  --project sae_test \
  --run "$RUN" \
  --out checkpoints/sae/${RUN}.pt \
  --shard_dir "$SHARD_DIR" \
  --mask_dir "$MASK_DIR" \
  --checkpoint "$CKPT" \
  --arch base_patch_conv \
  --img_size 96 \
  --patch_size 8 \
  --hook_layer 1 \
  --latent_mul 4 \
  --k_sparse 4 \
  --l1 3e-4 \
  --batch_size 512 \
  --lr 5e-3 \
  --steps 25000 \
  --workers 8 \
  --vis_mask_ratio 0.85 \
  --vis_n 3 \
  --vis_int 500 \
  --compile_sae \
  --mono_eval \
  --mono_thr 0.05 \
  --color_neurons
  # --project_pos        # re-enable if you wish

echo "=== Done (RUN=$RUN, Job=$SLURM_JOB_ID) ==="
