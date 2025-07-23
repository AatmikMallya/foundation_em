#!/bin/bash

#SBATCH --job-name=sae_analysis
#SBATCH --output=logs/sae_analysis_%j.out
#SBATCH --error=logs/sae_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --signal=TERM@120

mkdir -p logs
mkdir -p sae_analysis_results

module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

source venv/bin/activate

# Optimize environment
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=/gpfs/radev/home/am3833/project/foundation_em:$PYTHONPATH

echo "--- GPU Info ---"
nvidia-smi
echo "---------------"

echo "Starting SAE Monosemanticity Analysis..."
echo "Time: $(date)"

python3 sae_analysis.py \
    --vit_checkpoint checkpoints/best_model_base_patch_conv_v4_85_stable_eps1e-5.pt \
    --sae_checkpoint checkpoints/base_patch_conv_patchembed_lat32_topk4_l1-5e-3_lr2e-3.pt \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --model_arch base_patch_conv \
    --img_size 96 \
    --patch_size 8 \
    --extract_from patchembed \
    --layer 3 \
    --sae_variant linear \
    --activation relu \
    --max_shards 5 \
    --max_volumes_per_shard 50 \
    --min_selectivity 0.7 \
    --min_dominance 0.6 \
    --num_reconstruction_samples 20 \
    --output_dir sae_analysis_results \
    --compile_models \
    --use_amp \
    --amp_dtype bf16

echo "Analysis complete at $(date)"
echo "Results saved to sae_analysis_results/"
echo "View the interactive dashboard at: sae_analysis_results/sae_dashboard.html"
echo "View the reconstruction viewer at: sae_analysis_results/reconstruction_viewer.html" 