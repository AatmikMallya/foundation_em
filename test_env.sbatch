#!/bin/bash

#SBATCH --job-name=test_env
#SBATCH --output=logs/test_env_%j.out
#SBATCH --error=logs/test_env_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=== Testing Environment Setup ==="
echo "Current working directory: $(pwd)"
echo "User: $(whoami)"
echo "Date: $(date)"

# Load necessary modules
echo "Loading modules..."
module purge
module load miniconda
module load CUDA/11.8.0

echo "Modules loaded:"
module list

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "Python path after activation:"
which python
echo "Pip list (subset):"
pip list | grep -E "(torch|triton|numpy)"

# Test the environment
echo "Running environment test..."
python test_env.py

echo "=== Environment Test Complete ===" 