#!/bin/bash

#SBATCH --job-name=manim_vis_v2
#SBATCH --output=logs/manim_vis_v2_%j.out
#SBATCH --error=logs/manim_vis_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0-00:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

mkdir -p logs

# Load necessary modules
echo "Loading modules..."
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
module load cairo/1.17.4-GCCcore-12.2.0
module load Pango/1.50.12-GCCcore-12.2.0
module load FFmpeg/5.1.2-GCCcore-12.2.0
echo "Modules loaded."

# Activate virtual environment
echo "Activating venv..."
source venv/bin/activate
echo "Venv activated."

# Run Manim with medium quality for better output
echo "Running Manim..."
manim -pqm create_architecture_visualization.py MAEArchitecture
echo "Manim script finished." 