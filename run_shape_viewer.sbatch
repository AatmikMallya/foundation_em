#!/bin/bash
#SBATCH --job-name=shape_viewer
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=logs/viewer_%j.out
#SBATCH --error=logs/viewer_%j.err

module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

export MPLBACKEND=Agg  # Use non-interactive matplotlib backend

source venv/bin/activate

echo "Creating interactive shape viewer..."
python3 create_shape_viewer.py

echo "Shape viewer generation complete!"
ls -lh shape_viewer.html 