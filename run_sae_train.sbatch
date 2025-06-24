#!/bin/bash

#SBATCH --job-name=train_sae_layer6
#SBATCH --output=logs/train_sae_layer6_%j.out
#SBATCH --error=logs/train_sae_layer6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10G
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --signal=TERM@120

mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

source venv/bin/activate

# Tune for I/O throughput
export CUDA_LAUNCH_BLOCKING=0


python3 sae_train.py \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --checkpoint checkpoints/best_model_mask_75.pt \
    --model_arch base \
    --img_size 96 --patch_size 8 \
    --layer 6 \
    --latent_dim_multiplier 8 \
    --l1_coeff 5e-4 \
    --batch_size 129 \
    --token_chunk_size 8192 \
    --learning_rate 1e-3 \
    --epochs 2 \
    --total_steps 10000 \
    --num_workers 24 --prefetch_factor 4 \
    --output_path checkpoints/sae_layer6_lat8.pt
