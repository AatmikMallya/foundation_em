#!/bin/bash

#SBATCH --job-name=test_membrane_mae_v2
#SBATCH --output=logs/test_membrane_mae_v2_%j.out
#SBATCH --error=logs/test_membrane_mae_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --signal=TERM@120

mkdir -p logs
# Load necessary modules
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

# Activate virtual environment
source venv/bin/activate

# Optimize environment for I/O performance
export CUDA_LAUNCH_BLOCKING=0

# Run Python script with optimized I/O settings
echo "--- nvidia-smi (from job script) ---"
nvidia-smi
echo "------------------------------------"
python3 vol_train.py \
    --project_name "mae-3d-vol" \
    --run_name "mask_75_v2_base" \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --num_workers 24 \
    --val_split 0.02 \
    --img_size 96 \
    --patch_size 8 \
    --model_arch base \
    --epochs 15 \
    --batch_size 256 \
    --learning_rate 5e-5 \
    --min_lr 5e-6 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_steps 3000 \
    --weight_decay 0.05 \
    --disable_ema \
    --grad_clip_norm 1.0 \
    --initial_masking_ratio 0.75 \
    --final_masking_ratio 0.75 \
    --vis_interval 5000 \
    --val_interval 2000 \
    --lr_schedule cosine \
    --use_amp \
    --amp_dtype bf16 \
    --prefetch_factor 4 \
    --save_best_model \

echo "Python script finished."
