#!/bin/bash

#SBATCH --job-name=test_membrane_mae_v2
#SBATCH --output=logs/test_membrane_mae_v2_%j.out
#SBATCH --error=logs/test_membrane_mae_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8         # Adjust as needed, 4 is a good starting point
#SBATCH --mem-per-cpu=4G          # Adjust as needed, 16GB total for the job
#SBATCH --time=12:15:00           # Updated time limit to 48 hours
#SBATCH --partition=scavenge_gpu  # Or your specific GPU partition
#SBATCH --gpus=a5000:1             # Request 1 A5000 GPU

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
    --run_name "mae_hemi_optimal_norm_dynamic_data_fixed" \
    --model_arch hemibrain_optimal \
    --epochs 4000 \
    --batch_size 64 \
    --learning_rate 2e-4 \
    --warmup_epochs 200 \
    --min_lr 5e-6 \
    --weight_decay 0.05 \
    --noise_level 0.01 \
    --use_amp \
    --norm_pix_loss \
    --vis_interval 100 \
    --save_interval 250 \
    --num_workers 8

echo "Python script finished."
