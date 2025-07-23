#!/bin/bash
#SBATCH --job-name=test_checkpoint
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --output=logs/test_checkpoint_%j.out
#SBATCH --error=logs/test_checkpoint_%j.err

# Load modules (matching training environment)
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.2.2

# Activate virtual environment
source venv/bin/activate

# Create necessary directories
mkdir -p logs

echo "Testing checkpoint loading..."

# Run just the model loading test
python run_analysis_test.py --test models

echo "Test complete." 