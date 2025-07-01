#!/bin/bash

#SBATCH --job-name=sae_semantic_eval
#SBATCH --output=logs/sae_semantic_eval_%j.out
#SBATCH --error=logs/sae_semantic_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1

mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
module load SciPy-bundle/2023.02-gfbf-2022b

source venv/bin/activate

# Set environment variables for H100 optimization
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Increased for larger batches
export TORCH_LOGS="+dynamo,+inductor"  # Enable compilation logging if needed

# Define paths
export SAE_CHECKPOINT="best_sae_sae_layer4_lat3_mask75_v2_lambda_1e-3_lr_8e-4.pt"
export VOLUME_DIR="/gpfs/radev/home/am3833/scratch/volumes_96"
export MASK_DIR="/gpfs/radev/home/am3833/scratch/volumes_96/masks"  # Corrected path
export OUTPUT_DIR="sae_semantic_analysis"

echo "SAE Checkpoint: checkpoints/$SAE_CHECKPOINT"
echo "Volume Directory: $VOLUME_DIR"
echo "Mask Directory: $MASK_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Run semantic evaluation with H100 optimizations
python3 evaluate_sae_semantics.py \
    --sae_checkpoint checkpoints/$SAE_CHECKPOINT \
    --volume_dir $VOLUME_DIR \
    --mask_dir $MASK_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 128 \
    --max_samples 10000 \
    --use_amp \
    --amp_dtype bf16

echo "SAE semantic evaluation complete!"
echo "Check $OUTPUT_DIR/ for results:"
echo "  - semantic_analysis.png: Correlation analysis plots"
echo "  - semantic_correlations.txt: Detailed correlation results" 