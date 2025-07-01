#!/bin/bash
#SBATCH --job-name=simple_semantic_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/simple_semantic_eval_%j.out
#SBATCH --error=logs/simple_semantic_eval_%j.err

# Module loading
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

# Environment
cd /gpfs/radev/home/am3833/project/foundation_em
source venv/bin/activate

# Create logs directory
mkdir -p logs

# Configuration
export LAYER=8
export RUN_NAME="base_layer8_lat4_k32_lr1e-3"
export SAE_CHECKPOINT="checkpoints/best_sae_${RUN_NAME}.pt"
VOLUME_DIR="/gpfs/radev/home/am3833/scratch/volumes_96"
MASK_DIR="/gpfs/radev/home/am3833/scratch/volumes_96/masks"
OUTPUT_DIR="sae_semantic_analysis_simple"

echo "SAE Checkpoint: $SAE_CHECKPOINT"
echo "Volume Directory: $VOLUME_DIR"
echo "Mask Directory: $MASK_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Run semantic evaluation with patch visualization
python3 evaluate_sae_semantics_simple.py \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --volume_dir "$VOLUME_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples 200 \
    --extract_from encoder \
    --layer $LAYER \
    --visualize_patches \
    --top_neurons 3

echo "Simple SAE semantic evaluation complete!"
echo "Check $OUTPUT_DIR/ for results:"
echo "  - all_class_correlations.png: Correlation histograms for each class"
echo "  - top_neurons_comparison.png: Top neurons across all classes"
echo "  - all_class_results.txt: Detailed results for all classes"
echo "  - patch_visualizations/: Individual neuron activation patches" 