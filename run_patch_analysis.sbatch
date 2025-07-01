#!/bin/bash
#SBATCH --job-name=patch_viz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --output=logs/patch_viz_%j.out
#SBATCH --error=logs/patch_viz_%j.err

# Module loading
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

# Environment
cd /gpfs/radev/home/am3833/project/foundation_em
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Configuration
SAE_CHECKPOINT="checkpoints/best_sae_patchembed_lat8_k4_lr1e-3.pt"
VOLUME_DIR="/gpfs/radev/home/am3833/scratch/volumes_96"
MASK_DIR="/gpfs/radev/home/am3833/scratch/volumes_96/masks"
SEMANTIC_RESULTS_DIR="sae_semantic_analysis_simple"

echo "SAE Checkpoint: $SAE_CHECKPOINT"
echo "Volume Directory: $VOLUME_DIR"
echo "Mask Directory: $MASK_DIR"
echo "Semantic Results Directory: $SEMANTIC_RESULTS_DIR"

# Run patch visualization analysis
python3 run_patches_analysis.py \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --volume_dir "$VOLUME_DIR" \
    --mask_dir "$MASK_DIR" \
    --semantic_results_dir "$SEMANTIC_RESULTS_DIR" \
    --num_samples 500 \
    --top_neurons 3

echo "Patch visualization analysis complete!"
echo "Check $SEMANTIC_RESULTS_DIR/patch_visualizations/ for results" 