#!/bin/bash
#SBATCH --job-name=embedding_analysis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# Load modules (matching training environment)
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.2.2

# Environment setup (matching training)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# Activate virtual environment
source venv/bin/activate

# Create necessary directories
mkdir -p logs
mkdir -p analysis_results

echo "Starting embedding analysis..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "--- GPU Info ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
echo "---------------"

# Run the analysis
srun python analyze_embeddings.py \
    --output_dir analysis_results \
    --checkpoint_dir checkpoints \
    --num_volumes 100 \
    --volume_size 96 \
    --save_data \
    --test_similarity \
    --test_probes \
    --test_translation \
    --num_pairs 2000 \
    --num_translations 200

echo "Analysis complete. Results saved in analysis_results/" 