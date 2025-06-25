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
    --project_name "sae-3d-vol" \
    --run_name "sae_layer6_lat3_mask75_gelu" \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --checkpoint checkpoints/best_model_mask_75.pt \
    --output_path checkpoints/sae/sae_layer6_lat3_mask75_gelu.pt \
    --model_arch base_conv \
    --img_size 96 \
    --patch_size 8 \
    --layer 6 \
    --latent_dim_multiplier 3 \
    --l1_coeff 2e-3 \
    --batch_size 256 \
    --token_chunk_size 8192 \
    --learning_rate 1e-3 \
    --epochs 10 \
    --num_workers 24 \
    --prefetch_factor 4 \
    --val_interval 500 \
    --val_split 0.02 \
    --save_best_model \
    --use_amp \
    --amp_dtype bf16 \
    --grad_clip_norm 1.0 \
    --train_log_interval 10 \
    --l1_tune_interval 1000 \
    --dead_threshold 500 \
    --resample_interval 1000 \

    
