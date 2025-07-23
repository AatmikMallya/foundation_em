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

mkdir -p logs
module purge
module load Python/3.10.8-GCCcore-12.2.0-bare
module load CUDA/12.6.3

source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export OMP_NUM_THREADS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_LAUNCH_BLOCKING=0

RUN="id_layer2_patchembed"
python3 train_simple_sae.py \
    --shard_dir /gpfs/radev/home/am3833/scratch/volumes_96 \
    --checkpoint checkpoints/mae_pretrained.pt \
    --img_size 96 \
    --layer 2 \
    --batch_size 128 \
    --lr 1e-3 \
    --steps 5000 \
    --project sae_identity \
    --run $RUN \
    --out checkpoints/sae/${RUN}.pt
