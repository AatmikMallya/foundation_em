#!/bin/bash

#SBATCH --job-name=test_membrane_mae_v2
#SBATCH --output=logs/test_membrane_mae_v2_%j.out
#SBATCH --error=logs/test_membrane_mae_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --signal=TERM@120

mkdir -p logs
# Load necessary modules
module purge
# module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
# module load PyTorch/2.3.0a0-foss-2022b-CUDA-12.2.2
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.6.3

# Activate virtual environment
source venv/bin/activate


# Let Torch choose more pinned‑mem buffers for DALI‑like throughput
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# Avoid each worker spawning too many OpenMP threads
export OMP_NUM_THREADS=1
# Faster NCCL peer‑direct on Hopper/H200
export NCCL_NET_GDR_LEVEL=PHB
# Optimize environment for I/O performance
export CUDA_LAUNCH_BLOCKING=0

# Run Python script with optimized I/O settings
echo "--- nvidia-smi (from job script) ---"
nvidia-smi
echo "------------------------------------"
python3 vol_train.py \
    --project_name "mae-3d-vol" \
    --run_name "base_patch_conv_v4_85_stable_eps1e-5" \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --vols_per_shard 16384 \
    --num_workers 16 \
    --val_split 0.01 \
    --img_size 96 \
    --patch_size 8 \
    --model_arch base_patch_conv \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 2e-4 \
    --min_lr 1e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.975 \
    --warmup_steps 3000 \
    --weight_decay 0.05 \
    --disable_ema \
    --grad_clip_norm 0.5 \
    --initial_masking_ratio 0.85 \
    --final_masking_ratio 0.85 \
    --vis_interval 5000 \
    --val_interval 5000 \
    --lr_schedule cosine \
    --use_amp \
    --amp_dtype bf16 \
    --prefetch_factor 4 \
    --save_best_model \

echo "Python script finished."

