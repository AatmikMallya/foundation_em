#!/bin/bash

#SBATCH --job-name=test_membrane_mae_v2
#SBATCH --output=logs/test_membrane_mae_v2_%j.out
#SBATCH --error=logs/test_membrane_mae_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # Adjust as needed, 4 is a good starting point
#SBATCH --mem-per-cpu=2G          # Adjust as needed, 16GB total for the job
#SBATCH --time=1-00:00:00           # Updated time limit to 48 hours
#SBATCH --partition=gpu  # Or your specific GPU partition
#SBATCH --gpus=h100:1             # Request 1 A5000 GPU

mkdir -p logs
# Load necessary modules
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

# Activate virtual environment
source venv/bin/activate

# Run Python script with updated parameters
echo "Starting Python script test_membrane_mae.py with new schedule, increased data, and batch size"
echo "--- nvidia-smi (from job script) ---"
nvidia-smi
echo "------------------------------------"
python3 test_membrane_mae.py \
    --run_name "mae_fixed_64_16_base_conv" \
    --model_arch base_conv \
    --epochs 3000 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --warmup_epochs 100 \
    --min_lr 5e-6 \
    --weight_decay 0.01 \
    --noise_level 0.05 \
    --use_amp \
    --vis_interval 100 \
    --save_interval 2500000 \
    --val_interval 100 \
    --num_workers 4 \
    --skip_validation

echo "Python script finished."
