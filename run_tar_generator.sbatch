#!/bin/bash
#SBATCH --job-name=voronoi64-tar
#SBATCH --partition=day
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=58G      # 256 GiB total
#SBATCH --time=24:00:00
#SBATCH --output=logs/voro_tar_%j.out
#SBATCH --error=logs/voro_tar_%j.err

# ── environment ────────────────────────────────────────────────
module --force purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1   # gives SciPy/NumPy
source venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
freeGiB=$(awk '/MemFree/ {printf "%.0f GiB\n", $2/1024/1024}' /proc/meminfo)
echo "MemFree            = $freeGiB"

# ── run generator ──────────────────────────────────────────────
python vol_generator.py \
       --num_volumes 1048576 \
       --shard_size 16384 \
       --num_workers 16 \
       --queue_max 256 \
       --overwrite \
       --output_dir /gpfs/radev/home/am3833/scratch/voronoi_volumes_64/ \

echo "Voronoi tar-shard generation completed."
