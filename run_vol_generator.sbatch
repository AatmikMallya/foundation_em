#!/bin/bash
#SBATCH --job-name=membrane96
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=54G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err

module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

export OMP_NUM_THREADS=1        # one BLAS thread per process
export MKL_NUM_THREADS=1        # one BLAS thread per process

source venv/bin/activate

srun python vol_generator.py \
        --output_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
        --num_volumes 1048576 \
        --shard_size 16384 \
        --queue_max 256 \
        --num_workers 8 \
        --overwrite
