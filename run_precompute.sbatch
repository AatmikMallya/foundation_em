#!/bin/bash

#SBATCH --job-name=precompute_activations
#SBATCH --output=logs/precompute_activations_%j.out
#SBATCH --error=logs/precompute_activations_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h200:1
#SBATCH --signal=TERM@120

mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

source venv/bin/activate

# --- Configuration ---
# Define the output directory and file for the pre-computed data
export OUTPUT_DIR="precomputed_data"
export OUTPUT_FILE="$OUTPUT_DIR/activations_and_labels_layer8.pt"

# --- Run Pre-computation Script ---
echo "Starting pre-computation of activations and labels."
echo "Output will be saved to: $OUTPUT_FILE"

# Ensure output directory exists
mkdir -p $OUTPUT_DIR

python3 precompute_activations.py \
    --mae_checkpoint checkpoints/best_model_mask_75_v3_base.pt \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --output_file $OUTPUT_FILE \
    --layer 8 \
    --num_volumes 4096 \
    --batch_size 64 \
    --num_workers 16

echo "Pre-computation finished successfully." 