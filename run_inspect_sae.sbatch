#!/bin/bash

#SBATCH --job-name=inspect_sae
#SBATCH --output=logs/inspect_sae_%j.out
#SBATCH --error=logs/inspect_sae_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0:20:00
#SBATCH --partition=gpu
#SBATCH --gpus=1

mkdir -p logs
module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1
module load SciPy-bundle/2023.02-gfbf-2022b

source venv/bin/activate

# Set CUDA debugging flags
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export CHECKPOINT=best_sae_sae_layer4_lat3_mask75_v2_lambda_1e-3_lr_8e-4.pt

echo "Checkpoint: checkpoints/$CHECKPOINT"
echo "Output directory: sae_analysis"

# Run SAE inspection
python3 inspect_sae.py \
    --sae_checkpoint checkpoints/$CHECKPOINT \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --output_dir sae_analysis/$CHECKPOINT \
    --num_latents 40 \
    --batch_size 8 \

echo "SAE inspection complete!"
echo "Check sae_analysis/ for results:"
echo "  - sae_summary_stats.png: Overall SAE health"
echo "  - latent_XXXX_enhanced.png: Enhanced analysis with metrics"
echo "  - latent_XXXX_overlay.png: Standard overlay visualizations"
 