#!/bin/bash

#SBATCH --job-name=sae_id
#SBATCH --output=logs/sae_id_%j.out
#SBATCH --error=logs/sae_id_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=6G
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --signal=TERM@120

echo "Starting run"
mkdir -p logs
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.6.3
echo "Modules loaded"

echo "Activating virtual environment"
source venv/bin/activate
echo "Virtual environment activated"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

echo "Running training script"
RUN="id_layer2_whiten"
python3 -u train_simple_sae.py \
    --project sae_test \
    --run $RUN \
    --out checkpoints/sae/${RUN}.pt \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --checkpoint checkpoints/best_model_base_patch_conv_v4_85_stable_eps1e-5.pt \
    --arch base_patch_conv \
    --img_size 96 \
    --patch_size 8 \
    --vis_mask_ratio 0.85 \
    --vis_int 1000 \
    --layer 2 \
    --batch_size 512 \
    --lr 1.5e-3 \
    --steps 20000 \
