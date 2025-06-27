#!/bin/bash
#SBATCH --job-name=sae_semantic
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/semantic_%j.out
#SBATCH --error=logs/semantic_%j.err

module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

# Tune for I/O throughput
export CUDA_LAUNCH_BLOCKING=0

source venv/bin/activate

# Default paths - modify as needed
SAE_CHECKPOINT="checkpoints/sae_layer2_lat3_mask75_v2_lambda_1e-3_lr_8e-4/latest_checkpoint.pth"
VIT_CHECKPOINT="checkpoints/mae_vit_base_96_latest.pth"
VOLUME_DIR="/gpfs/radev/home/am3833/scratch/volumes_96"
MASK_DIR="/gpfs/radev/home/am3833/scratch/volumes_96/masks"
OUTPUT_DIR="sae_semantic_analysis"

echo "SAE checkpoint: $SAE_CHECKPOINT"
echo "ViT checkpoint: $VIT_CHECKPOINT"
echo "Volume dir: $VOLUME_DIR"
echo "Mask dir: $MASK_DIR"
echo "Output dir: $OUTPUT_DIR"

python3 evaluate_sae_semantics.py \
    --sae_checkpoint "$SAE_CHECKPOINT" \
    --vit_checkpoint "$VIT_CHECKPOINT" \
    --volume_dir "$VOLUME_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --layer 2 \
    --batch_size 16 \
    --max_samples 5000 \
    --device cuda

echo "Semantic evaluation complete!" 