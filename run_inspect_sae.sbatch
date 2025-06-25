#!/bin/bash

#SBATCH --job-name=inspect_sae
#SBATCH --output=logs/inspect_sae_%j.out
#SBATCH --error=logs/inspect_sae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

source venv/bin/activate

# Set CUDA debugging flags
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Run SAE inspection
python3 inspect_sae.py \
    --sae_checkpoint checkpoints/best_sae_sae_layer6_lat3_mask75_gelu.pt \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --output_dir sae_analysis \
    --num_latents 20 \
    --batch_size 8 
