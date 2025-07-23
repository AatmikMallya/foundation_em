#!/bin/bash
#SBATCH --job-name=vol_gen_equal
#SBATCH --output=logs/vol_gen_equal_%j.out
#SBATCH --error=logs/vol_gen_equal_%j.err
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --partition=day

# Create logs directory
mkdir -p logs

# Load modules
module load Python/3.10.8-GCCcore-12.2.0

# Set environment variables for optimal CPU performance
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Generate dataset with equal combinations
python vol_generator.py \
    --output_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --num_volumes 1048576 \
    --shard_size 16384 \
    --num_workers 16 \
    --queue_max 256 \
    --generate_masks \
    --equal_combinations \
    --overwrite

echo "Volume generation with equal combinations complete!"
echo "Generated $(ls volumes_96/shard_*.tar 2>/dev/null | wc -l) volume shards"
echo "Generated $(ls volumes_96/masks/shard_*.tar 2>/dev/null | wc -l) mask shards" 