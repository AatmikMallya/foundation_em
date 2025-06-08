#!/bin/bash

#SBATCH --job-name=mae_2500_epochs_ema
#SBATCH --output=mae_2500_epochs_ema_%j.out
#SBATCH --error=mae_2500_epochs_ema_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=22:59:00           # Reduced time to comply with partition limits
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=a5000:1

# Load necessary system modules first
module purge
# Note: It's important to load any system-level libraries like libffi if needed
# This was required for interactive debugging, so it might be needed here too.
# The following line is added to prevent potential runtime errors.
module load libffi/3.4.4-GCCcore-12.2.0

# Then, load the main software module
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

# Activate your Python virtual environment
source venv/bin/activate

# Run the training script with user-specified parameters and improved hyperparameters
echo "Starting training run: 2500 epochs with EMA, improved hyperparameters, and custom mask schedule"
echo "--- nvidia-smi ---"
nvidia-smi
echo "--------------------"

python3 test_membrane_mae.py \
    --run_name "mae_2500_epochs_ema_spheres_4-8_rad_4-6_bs32" \
    --epochs 2500 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --min_lr 5e-6 \
    --weight_decay 0.02 \
    --noise_level 0.01 \
    --use_amp \
    --vis_interval 100 \
    --save_interval 250 \
    --num_added_spheres "4,8" \
    --added_sphere_radii "4.0,6.0"

echo "Python script finished." 