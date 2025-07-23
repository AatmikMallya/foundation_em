#!/bin/bash

#SBATCH --job-name=train_sae
#SBATCH --output=logs/train_sae_%j.out
#SBATCH --error=logs/train_sae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=18:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --signal=TERM@120

mkdir -p logs
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.6.3

source venv/bin/activate

# Let Torch choose more pinned‑mem buffers for DALI‑like throughput
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# Avoid each worker spawning too many OpenMP threads
export OMP_NUM_THREADS=1
# Faster NCCL peer‑direct on Hopper/H200
export NCCL_NET_GDR_LEVEL=PHB
# Optimize environment for I/O performance
export CUDA_LAUNCH_BLOCKING=0

export RUN_NAME="base_patch_conv_patchembed_easy"

python3 sae_train.py \
    --project_name "sae-3d-vol" \
    --run_name $RUN_NAME \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --checkpoint checkpoints/best_model_base_patch_conv_v4_85_stable_eps1e-5.pt \
    --output_path checkpoints/sae/$RUN_NAME.pt \
    --model_arch base_patch_conv \
    --img_size 96 \
    --patch_size 8 \
    --extract_from encoder \
    --layer 2 \
    --latent_dim_multiplier 8 \
    --l1_coeff 0.0 \
    --ortho_coeff 0.0 \
    --batch_size 128 \
    --learning_rate 1.5e-3 \
    --total_steps 10000 \
    --use_cosine_decay \
    --warmup_steps 500 \
    --activation relu \
    --num_workers 16 \
    --prefetch_factor 4 \
    --val_interval 500 \
    --val_split 0.01 \
    --save_best_model \
    --use_amp \
    --amp_dtype bf16 \
    --grad_clip_norm 1.0 \
    --train_log_interval 10 \
    --dead_threshold 12500 \
    --resample_interval 25000 \
    --sae_variant linear \
    --compile_mae \
    --compile_sae \
    --vis_interval 1000 \
    --vis_samples 6
# --k_sparse 4 \
    


