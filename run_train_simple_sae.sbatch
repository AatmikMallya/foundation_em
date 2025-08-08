#!/bin/bash
#SBATCH --job-name=sae_stage1
#SBATCH --output=logs/sae_stage1_%j.out
#SBATCH --error=logs/sae_stage1_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --signal=TERM@120

set -euo pipefail

echo "=== Stage‑1 SAE run starting ==="
mkdir -p logs checkpoints/sae

module --force purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.6.3

source venv/bin/activate

# Runtime/env tuning
export PYTHONFAULTHANDLER=1          # Python will dump a traceback on segfault
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MALLOC_ARENA_MAX=1            # reduce glibc fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export CUDA_LAUNCH_BLOCKING=0
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
export TORCHINDUCTOR_CUDAGRAPHS=0



# ---------- CONFIG (edit as needed) ----------
RUN="k4_layer3_adj"
SHARD_DIR="/gpfs/radev/home/am3833/scratch/volumes_96"
MASK_DIR="/gpfs/radev/home/am3833/scratch/volumes_96/masks"
CKPT="checkpoints/best_model_base_patch_conv_v4_85_stable_eps1e-5.pt"

python3 -u train_simple_sae.py \
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
  --lr 1e-3 \
  --steps 25000 \
  --workers 8 \
  --vis_mask_ratio 0.85 \
  --vis_n 3 \
  --vis_int 500 \
  --compile_sae \
  --mono_eval \
  --mono_thr 0.05 \
  --color_neurons \
  # --project_pos

echo "=== Done ==="
