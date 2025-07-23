#!/bin/bash
#SBATCH --job-name=vis_slices
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/vis_slices_%j.out
#SBATCH --error=logs/vis_slices_%j.err
#SBATCH --gpus=1

module purge
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1

source venv/bin/activate

srun python visualize_checkpoint_slices.py \
    --checkpoint_path checkpoints/best_model_mask_75_v3_base.pt \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --num_examples 12 \
    --output_file reconstruction_viewer.html
    