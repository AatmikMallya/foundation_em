#!/bin/bash
#SBATCH --job-name=benchmark_volgen
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

source venv/bin/activate

echo "Starting volume generation benchmark..."
echo "Optimizations applied:"
echo "  - Distance transform for membrane collision (O(1) vs O(N))"
echo "  - Vectorized collision detection" 
echo "  - Separable 1D convolutions instead of 3D Gaussian filter"
echo "  - Reduced Python overhead"
echo ""

python3 benchmark_volgen.py

echo "Benchmark complete!" 