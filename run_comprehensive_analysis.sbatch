#!/bin/bash

#SBATCH --job-name=comprehensive_analysis
#SBATCH --output=logs/comprehensive_analysis_%j.out
#SBATCH --error=logs/comprehensive_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0:30:00
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --signal=TERM@120

mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

source venv/bin/activate

# --- Configuration ---
export RUN_NAME="base_v3_layer8_lat4_l1-1e-3_lr1e-3"
export SAE_CHECKPOINT="checkpoints/sae/${RUN_NAME}.pt"
export ACTIVATIONS_FILE="precomputed_data/activations_and_labels_layer8.pt"
export OUTPUT_DIR="comprehensive_analysis/${RUN_NAME}"

# --- Run Analysis Script ---
echo "Starting comprehensive analysis for run: $RUN_NAME"
echo "Analyzing pre-computed data from: $ACTIVATIONS_FILE"

python3 comprehensive_sae_analysis.py \
    --sae_checkpoint $SAE_CHECKPOINT \
    --activations_file $ACTIVATIONS_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 8192 \
    --compile_sae \
    --use_amp \
    --amp_dtype bf16

echo "Comprehensive analysis complete. Results are in $OUTPUT_DIR" 