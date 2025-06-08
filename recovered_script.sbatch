#!/bin/bash

#SBATCH --job-name=test_membrane_mae_v2
#SBATCH --output=test_membrane_mae_v2_%j.out
#SBATCH --error=test_membrane_mae_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # Adjust as needed, 4 is a good starting point
#SBATCH --mem-per-cpu=4G          # Adjust as needed, 16GB total for the job
#SBATCH --time=22:59:00           # Updated time limit to 23 hours 59 mins 59 secs
#SBATCH --partition=scavenge_gpu  # Or your specific GPU partition
#SBATCH --gpus=a5000:1             # Request 1 A5000 GPU

# Load necessary modules
module purge # Ensure a clean environment
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 # Or your preferred PyTorch module

# Activate your Python virtual environment
source venv/bin/activate

# Run your Python script with updated parameters
echo "Starting Python script test_membrane_mae.py with new schedule, increased data, and batch size"
echo "--- nvidia-smi (from job script) ---"
nvidia-smi
echo "------------------------------------"
python3 test_membrane_mae.py \
    --epochs 4000 \
    --batch_size 64 \
    --train_samples 4096 \
    --val_samples 512 \
    --use_amp \
    --model_arch base \
    --vis_interval 100 \
    --vis_samples 4 \
    --grad_clip_norm 1.0 \
    --num_added_spheres "4,8" \
    --added_sphere_radii "3.0,5.0" \
    --run_name "mae_base_5k_epochs_bs64_4kdata_new_schedule_v2" \
    --save_interval 250
    # Note: --initial_mask_ratio, --final_mask_ratio, --mask_ratio_epoch_split are removed

echo "Python script finished."
