#!/bin/bash
#SBATCH --job-name=membrane_test
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:10:00
#SBATCH --output=logs/membrane_test_%j.out

module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

source venv/bin/activate

echo "Debugging overlaps in generated volumes..."
python3 debug_overlaps.py 